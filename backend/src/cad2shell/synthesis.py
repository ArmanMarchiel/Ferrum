"""Stage 5 -- synthesis: voxel tree -> ceramic shell mesh.

    metal = part U cup U sprue U runner U ingates U risers      (one solid)
    shell = dilate(metal, t) \\ dilate(metal, 0)                (a thin skin)
    shell = shell \\ above(cup mouth)                           (open to pour)

then marching cubes back to triangles. The dilation radius is computed in
continuous voxel units, and marching cubes places the isosurface at the 0.5
level set, so the emitted wall lands on the requested thickness rather than a
voxel short of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from . import primitives as pr
from .voxel import Grid


@dataclass
class ShellResult:
    shell_mesh: trimesh.Trimesh
    fired_mesh: trimesh.Trimesh | None
    tree_mesh: trimesh.Trimesh
    metal: np.ndarray            # bool occupancy of all metal
    shell: np.ndarray            # bool occupancy of the ceramic skin
    grid: Grid
    thickness_measured: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def build_metal(grid: Grid, part_occ: np.ndarray, segments) -> tuple[np.ndarray, dict]:
    """Union the part occupancy with every gating segment on the same grid."""
    metal = part_occ.copy()
    added = pr.stamp_segments(metal, grid, segments)
    return metal, added


def _dilate_mm(mask: np.ndarray, grid: Grid, distance_mm: float) -> np.ndarray:
    """Dilate by a true Euclidean distance, not a voxel count.

    A binary-dilation iteration count would quantise the shell to whole
    voxels (and with a cubic structuring element would over-reach on the
    diagonals). Thresholding the EDT of the complement gives an isotropic,
    sub-voxel-accurate offset.
    """
    outside = ~mask
    dist = ndimage.distance_transform_edt(outside, sampling=(grid.pitch,) * 3)
    return mask | (dist <= distance_mm)


def build_shell(grid: Grid, metal: np.ndarray, cfg) -> np.ndarray:
    """The ceramic skin: an offset of the metal minus the metal itself."""
    grown = _dilate_mm(metal, grid, cfg.shell_thickness)
    return grown & ~metal


def open_pour_cup(shell: np.ndarray, grid: Grid, segments,
                  metal_top: float | None = None) -> tuple[np.ndarray, dict]:
    """Clear shell above the cup mouth so metal can be poured in.

    Without this the skin closes over the top of the cup and the mould is a
    sealed void -- watertight, but unpourable.
    """
    cups = [s for s in segments if s.meta.get("open_to_world")]
    if not cups:
        return shell, {"opened": False}
    mouth_z = max(float(max(s.p0[2], s.p1[2])) for s in cups)
    # A mouth inside the casting is not a mouth. Clearing shell above it would
    # cut the top off the mould without opening a path to the cavity, leaving a
    # sealed shell that looks watertight and cannot be poured.
    if metal_top is not None and mouth_z <= metal_top + 1e-6:
        raise ValueError(
            f"the pour cup sits at z={mouth_z:.1f} mm, inside the casting "
            f"which reaches z={metal_top:.1f} mm -- the mould would be sealed. "
            "Check the pour direction: the part may be oriented so the sprue "
            "cannot reach open air."
        )
    above = pr.half_space_above(grid, mouth_z)
    n_before = int(shell.sum())
    shell = shell & ~above
    return shell, {
        "opened": True,
        "mouth_z": mouth_z,
        "cells_cleared": n_before - int(shell.sum()),
    }


def measure_thickness(shell: np.ndarray, grid: Grid, metal: np.ndarray | None = None) -> dict:
    """Measure the realised wall thickness of the ceramic skin, in mm.

    Measured by counting shell voxels along rays cast outward from the metal
    through the skin, one ray per axis direction. That is a direct reading of
    the wall the mesher will emit.

    An EDT-of-the-skin approach was tried first and is wrong here: the medial
    axis of a discrete slab lands on whole voxels, so twice the local distance
    can only ever be an even multiple of the pitch, and a 5 mm wall at 1.5 mm
    pitch reports as 6.00. Counting cells along a ray has no such quantisation
    beyond the pitch itself.
    """
    if not shell.any():
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    if metal is None:
        return _thickness_from_edt(shell, grid)

    runs: list[float] = []
    for axis in range(3):
        # Walk each 1-D line of the grid along `axis`, and for every maximal
        # run of skin that is adjacent to metal, record its length.
        moved_shell = np.moveaxis(shell, axis, -1)
        moved_metal = np.moveaxis(metal, axis, -1)
        flat_shell = moved_shell.reshape(-1, moved_shell.shape[-1])
        flat_metal = moved_metal.reshape(-1, moved_metal.shape[-1])
        # only lines that actually contain metal are load-bearing walls
        keep = flat_metal.any(axis=1) & flat_shell.any(axis=1)
        for line_s, line_m in zip(flat_shell[keep], flat_metal[keep]):
            idx = np.flatnonzero(line_s)
            if idx.size == 0:
                continue
            splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
            for run in splits:
                lo, hi = run[0], run[-1]
                touches_metal = (lo > 0 and line_m[lo - 1]) or \
                                (hi + 1 < line_m.size and line_m[hi + 1])
                if touches_metal:
                    runs.append(run.size * grid.pitch)
    if not runs:
        return _thickness_from_edt(shell, grid)
    vals = np.asarray(runs, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n": int(vals.size),
    }


def _thickness_from_edt(shell: np.ndarray, grid: Grid) -> dict:
    """Fallback measurement when the metal mask is unavailable."""
    dist = ndimage.distance_transform_edt(shell, sampling=(grid.pitch,) * 3)
    peak = ndimage.grey_dilation(dist, size=(3, 3, 3), mode="constant", cval=0.0)
    medial = shell & (dist >= peak - 1e-9) & (dist > 0)
    vals = 2.0 * dist[medial] if medial.any() else 2.0 * dist[shell]
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)), "n": int(vals.size)}


def to_mesh(mask: np.ndarray, grid: Grid, name: str = "") -> trimesh.Trimesh:
    """Marching cubes on a binary mask -> a closed, outward-facing mesh.

    The mask is padded with one empty voxel so any part of it touching the
    array border still produces a closed surface instead of an open boundary.
    Vertices come back in index space and are mapped to world by the same
    origin/pitch the Grid uses, keeping the mesh on the pipeline's frame.
    """
    if not mask.any():
        raise ValueError(f"cannot mesh an empty mask ({name})")
    padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0.0)
    verts, faces, normals, _ = measure.marching_cubes(padded, level=0.5)
    # undo the pad, then index -> world. Marching-cubes vertices are in the
    # array's own continuous index space, where integer i is a sample centre;
    # a sample centre is grid voxel centre => origin + (i + 0.5) * pitch.
    verts = verts - 1.0
    verts = grid.origin + (verts + 0.5) * grid.pitch
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    return mesh


def fire_shell(shell_mesh: trimesh.Trimesh, cfg) -> trimesh.Trimesh:
    """The shell as it comes out of the burnout furnace.

    An investment shell is built green around the pattern, then fired (here
    ``cfg.fire_temperature``, typically 950-1300 C) to burn the pattern out and
    sinter the ceramic. The shell contracts during that firing by roughly
    0.3-0.6% linear for a typical fused-silica / zircon system.

    This is a property of the ceramic alone. The alloy poured into the mould
    afterwards has no bearing on it, so no alloy term appears here.

    The contraction is applied about the shell's own centroid, so the fired
    shell stays concentric with the green one instead of drifting toward the
    world origin.
    """
    fired = shell_mesh.copy()
    centre = shell_mesh.bounding_box.centroid
    fired.apply_translation(-centre)
    fired.apply_scale(cfg.fired_scale)
    fired.apply_translation(centre)
    return fired


def exact_cavity(outer: trimesh.Trimesh, part, plan, cfg, risers=()) -> trimesh.Trimesh | None:
    """Cut the mould cavity with the ORIGINAL part mesh, not its voxels.

    The outer surface of the shell can be a voxel offset -- it is the outside of
    a ceramic shell and nobody measures it. The inner surface is different: it
    IS the casting, so every voxel of quantisation on it becomes a defect on the
    part. At 1.5 mm pitch the carved-from-voxels cavity was out by 0.23 mm
    median and 1.3 mm at p99, which shows up as visible ridging in a bore.

    Subtracting the original mesh instead makes the cavity exact at any pitch.
    The gating is subtracted too, as analytic cones rather than stamped voxels,
    so the sprue bore is smooth as well.

    Returns None if the boolean fails, so the caller can fall back to the voxel
    cavity rather than losing the mould entirely.
    """
    cutters = [part.mesh]
    cut_segments = list(plan.segments)
    for r in risers:
        cut_segments.extend(r.segments())
    for seg in cut_segments:
        # Extend each gating cone slightly past its own endpoints. A cutter that
        # ends exactly on the part's surface produces a tangential contact, and
        # the boolean leaves a few valence-4 edges there -- the mesh does not
        # leak but no longer reads as closed. A small overlap makes the
        # intersection transversal and the result cleanly manifold.
        cone = _segment_solid(seg, overlap=cfg.voxel_pitch)
        if cone is not None:
            cutters.append(cone)
    try:
        cavity = trimesh.boolean.difference([outer] + cutters, engine="manifold")
    except Exception:
        return None
    if isinstance(cavity, trimesh.Scene):  # pragma: no cover
        cavity = trimesh.util.concatenate(list(cavity.geometry.values()))
    if not isinstance(cavity, trimesh.Trimesh) or len(cavity.faces) == 0:
        return None
    # manifold3d already returns a closed mesh. Do NOT run the usual cleanup
    # here: `unique_faces` removes coincident triangles that this solid
    # legitimately has where the sprue meets the part's flat top, and deleting
    # them punches 12 holes in an otherwise watertight shell.
    if not cavity.is_watertight:
        repaired = cavity.copy()
        repaired.merge_vertices()
        try:
            trimesh.repair.fill_holes(repaired)
        except Exception:
            pass
        if repaired.is_watertight:
            cavity = repaired
    trimesh.repair.fix_normals(cavity)
    if cavity.is_watertight and cavity.volume < 0:
        cavity.invert()
    if not cavity.is_watertight:
        return None          # fall back to the voxel cavity rather than ship a leak
    return cavity


def drop_zero_volume_debris(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Strip zero-volume sliver triangles the cavity boolean leaves behind.

    Where analytic cutters meet -- a runner joining the sprue, an ingate
    entering a rotated instance -- the boolean can emit a few triangles with no
    area alongside the real surface. In memory they are inert: they reference
    their own vertices, so the mesh reports as one watertight body and the
    volume is right.

    They stop being inert on export. STL stores bare triangles with no shared
    vertices, so re-reading welds everything by position and the slivers fuse
    into the surface, putting four faces on an edge that should carry two. The
    shell then reads as non-watertight despite having no hole in it -- and
    `is_watertight` is exactly what every downstream consumer checks before
    trusting the mould.

    The removal is deliberately conservative. It only acts when the mesh splits
    into components of which precisely one encloses any volume, and it keeps
    the result only if that component is itself watertight and holds essentially
    all the volume. Anything else -- two real solids, an open surface, a body
    that loses volume -- is returned untouched, because then the extra
    components are not debris and discarding them would be destroying geometry.

    This is much narrower than the `unique_faces` cleanup, which deletes the
    legitimately coincident triangles where the sprue meets the part's flat top
    and punches actual holes in the shell.
    """
    try:
        bodies = mesh.split(only_watertight=False)
    except Exception:
        return mesh
    if len(bodies) < 2:
        return mesh
    solid = [b for b in bodies if abs(b.volume) > 1e-9]
    if len(solid) != 1:
        return mesh                      # genuinely several solids: not debris
    kept = solid[0]
    if not kept.is_watertight:
        return mesh                      # dropping them would not help anyway
    if abs(kept.volume) < 0.999 * abs(mesh.volume):
        return mesh                      # the discards carried real volume
    return kept


def exact_tree(part, plan, cfg, risers=()) -> trimesh.Trimesh | None:
    """The metal tree as an exact union, not a voxel blob.

    The casting plus everything attached to it -- sprue, cup, runners, ingates
    and risers -- all of which are already available as exact geometry: the
    part as the uploaded mesh, the rest as analytic cones. Unioning them keeps
    the casting surface identical to the CAD and the sprue perfectly round,
    instead of re-deriving the whole thing from the voxel grid where every
    surface picks up a staircase.

    `risers` matters and is easy to forget: riser segments hang off the Riser
    objects rather than off `plan.segments`, so a version of this that unioned
    only the plan produced a tree with no risers on it -- while the shell,
    built from the voxel metal mask, had the riser cavities in it. The mould
    showed domes the casting did not, which is exactly backwards from what a
    riser is.

    Returns None if the boolean fails, so the caller can fall back to the voxel
    tree rather than losing it entirely.
    """
    solids = [part.mesh]
    tree_segments = list(plan.segments)
    for r in risers:
        tree_segments.extend(r.segments())
    for seg in tree_segments:
        cone = _segment_solid(seg, overlap=cfg.voxel_pitch * 0.5)
        if cone is not None:
            solids.append(cone)
    if len(solids) == 1:
        return part.mesh.copy()
    try:
        tree = trimesh.boolean.union(solids, engine="manifold")
    except Exception:
        return None
    if isinstance(tree, trimesh.Scene):  # pragma: no cover
        tree = trimesh.util.concatenate(list(tree.geometry.values()))
    if not isinstance(tree, trimesh.Trimesh) or len(tree.faces) == 0:
        return None
    trimesh.repair.fix_normals(tree)
    if tree.is_watertight and tree.volume < 0:
        tree.invert()
    if not tree.is_watertight:
        return None
    return tree


def _segment_solid(seg, sections: int = 64, overlap: float = 0.0) -> trimesh.Trimesh | None:
    """A gating segment as an analytic truncated cone, for exact subtraction.

    `overlap` extends the cone past both endpoints along its own axis, so it
    cuts through neighbouring solids rather than meeting them tangentially.
    """
    axis = np.asarray(seg.p1) - np.asarray(seg.p0)
    height = float(np.linalg.norm(axis))
    if height < 1e-9:
        return None
    try:
        unit = axis / height
        p0 = np.asarray(seg.p0) - unit * overlap
        height = height + 2.0 * overlap
        # a cone frustum along +Z, then rotated onto the segment's axis
        angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
        ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        bottom = np.hstack([ring * seg.r0, np.zeros((sections, 1))])
        top = np.hstack([ring * seg.r1, np.full((sections, 1), height)])
        verts = np.vstack([bottom, top, [[0, 0, 0]], [[0, 0, height]]])
        faces = []
        n = sections
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n + j])            # side
            faces.append([i, n + j, n + i])
            faces.append([2 * n, j, i])            # bottom cap
            faces.append([2 * n + 1, n + i, n + j])  # top cap
        cone = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)
        cone.merge_vertices()
        trimesh.repair.fix_normals(cone)
        transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], unit)
        transform[:3, 3] = p0
        cone.apply_transform(transform)
        return cone
    except Exception:
        return None


def build_cavity(grid: Grid, part_occ: np.ndarray, cfg) -> np.ndarray:
    """STEP 1 -- the ceramic around the casting, and nothing else.

    This is the mould proper: the negative space of the CAD, offset outward by
    the shell thickness. It knows nothing about how metal will get in.

    The step is deliberately independent of quantity. One part or forty, the
    ceramic wrapped around each casting is the same ceramic -- the instances
    are disjoint solids on a shared grid, so dilating their union offsets each
    one exactly as it would alone. `test_cavity_is_identical_per_instance`
    pins that down by comparing the volume against N times the one-up shell.

    What multi-cavity actually changes is STEP 2: one casting is fed by a sprue
    landing straight on it, while N castings need a sprue, runners out to each
    instance and an ingate into every one. That is `gating.design_multi`, and
    it is the only part of the pipeline that has to count.
    """
    grown = _dilate_mm(part_occ, grid, cfg.shell_thickness)
    return grown & ~part_occ


def build(part, grid: Grid, plan, risers, cfg, part_occ=None) -> ShellResult:
    """Stage 5 entry point: cavity (step 1) then feed system (step 2).

    The two are separable in principle -- `build_cavity` above produces the
    mould on its own -- but they are unioned before the offset here rather than
    offset separately and merged. A shell built around the casting and a shell
    built around the sprue would each carry their own wall through the region
    where the two solids meet, leaving a ridge of ceramic sticking into the
    join instead of a smooth continuous skin.
    """
    from .hotspots import voxelize_fast

    segments = list(plan.segments)
    for r in risers:
        segments.extend(r.segments())

    if part_occ is None:
        part_occ = voxelize_fast(part.mesh, grid)

    # step 2: the feed system joins the casting to make one connected solid
    metal, added = build_metal(grid, part_occ, segments)
    shell = build_shell(grid, metal, cfg)
    shell, cup_info = open_pour_cup(shell, grid, segments,
                                    metal_top=float(part.mesh.bounds[1][2]))

    # The tree is exact geometry too: the uploaded part unioned with analytic
    # gating cones. Deriving it from `metal` would voxelise the casting surface
    # all over again, which is what made the tree look faceted while the shell
    # around it was smooth.
    tree_mesh = exact_tree(part, plan, cfg, risers) if cfg.exact_cavity else None
    tree_exact = tree_mesh is not None
    if tree_mesh is None:
        tree_mesh = to_mesh(metal, grid, "tree")

    # The shell's OUTER surface comes from the voxel offset; its cavity is cut
    # with the original geometry so the casting surface is exact.
    shell_mesh = None
    cavity_exact = False
    if cfg.exact_cavity:
        grown_only = _dilate_mm(metal, grid, cfg.shell_thickness)
        if cup_info.get("opened"):
            grown_only = grown_only & ~pr.half_space_above(grid, cup_info["mouth_z"])
        outer = to_mesh(grown_only, grid, "outer")
        shell_mesh = exact_cavity(outer, part, plan, cfg, risers)
        cavity_exact = shell_mesh is not None
    if shell_mesh is None:
        shell_mesh = to_mesh(shell, grid, "shell")
    # Clean the meshes that get exported. A sliver that is inert in memory
    # becomes a non-manifold edge once the STL is re-read, so this has to
    # happen before the export, not after it.
    shell_mesh = drop_zero_volume_debris(shell_mesh)
    tree_mesh = drop_zero_volume_debris(tree_mesh)
    fired_mesh = fire_shell(shell_mesh, cfg) if cfg.show_fired_shell else None

    return ShellResult(
        shell_mesh=shell_mesh, fired_mesh=fired_mesh,
        tree_mesh=tree_mesh, metal=metal, shell=shell,
        grid=grid,
        thickness_measured=measure_thickness(shell, grid, metal=metal),
        diagnostics={
            "segments_stamped": added,
            "cup": cup_info,
            "cavity_exact": cavity_exact,
            "tree_exact": tree_exact,
            "metal_voxels": int(metal.sum()),
            "shell_voxels": int(shell.sum()),
            "metal_volume": float(metal.sum() * grid.voxel_volume),
            "shell_volume": float(shell.sum() * grid.voxel_volume),
            "fired": {
                "temperature": cfg.fire_temperature,
                "linear_shrinkage": cfg.fire_shrinkage,
                "scale": cfg.fired_scale,
                "volume": float(fired_mesh.volume) if fired_mesh is not None else None,
                # every linear dimension loses the same fraction
                "extents_delta": (
                    (fired_mesh.extents - shell_mesh.extents).tolist()
                    if fired_mesh is not None else None
                ),
            },
        },
    )
