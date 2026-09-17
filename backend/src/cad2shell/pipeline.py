"""run() -- orchestrate ingest -> hot spots -> gating -> risers -> synthesis.

Coordinate-frame contract, which is where this kind of pipeline usually goes
wrong: exactly one lattice exists per run. `grid_part` covers the casting and
`grid_full` covers the casting plus its gating tree, and `grid_full` is built
by *expanding* `grid_part`, so the two share an origin lattice and differ by a
pure integer index offset. Hot spots are detected on `grid_full` -- the same
grid the risers are stamped into -- so a riser cannot land one voxel off the
thermal centre it was sized for.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from . import gating, hotspots, ingest, layout as layout_mod, report, risers, synthesis
from .config import Config
from .voxel import Grid


@dataclass
class Stats:
    part: dict
    casting_modulus: float
    hot_spots: list[dict]
    gating: dict
    risers: list[dict]
    shell: dict
    metal_volume: float
    part_volume: float
    part_volume_voxel: float
    yield_pct: float
    grid: dict
    layout: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "part": self.part,
            "part_volume": self.part_volume,
            "part_volume_voxel": self.part_volume_voxel,
            "casting_modulus": self.casting_modulus,
            "hot_spots": self.hot_spots,
            "gating": self.gating,
            "risers": self.risers,
            "shell": self.shell,
            "metal_volume": self.metal_volume,
            "yield_pct": self.yield_pct,
            "grid": self.grid,
            "layout": self.layout,
            "files": self.files,
            "timings": self.timings,
            "warnings": self.warnings,
            "config": self.config,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


def _tree_bounds(part, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Conservative world bounds for part + gating + risers + shell.

    Sized before the tree exists, so it is deliberately generous: the grid is
    trimmed to nothing, but an undersized grid would clip the sprue.
    """
    lo, hi = np.asarray(part.bounds[0]), np.asarray(part.bounds[1])
    span = float(np.max(hi - lo))
    # riser worst case: modulus method on the thickest plausible section
    r_guess = risers.diameter_for_modulus(
        cfg.riser_modulus_factor * max(part.modulus, 1.0), cfg.riser_hd_ratio
    )
    up = cfg.cup_height + cfg.riser_hd_ratio * r_guess + r_guess
    out = max(cfg.runner_clearance * 2.0 + r_guess, 0.35 * span)
    down = cfg.runner_clearance + 0.25 * span
    pad = cfg.shell_thickness * 1.5 + 3.0 * cfg.voxel_pitch
    return (
        lo - np.array([out + pad, out * 0.5 + pad, down + pad]),
        hi + np.array([out * 0.5 + pad, out * 0.5 + pad, up + pad]),
    )


def run(part_path=None, out_prefix="out/part", cfg: Config | None = None,
        mesh=None, write_files: bool = True, progress=None) -> Stats:
    """Full pipeline. Returns Stats; writes <prefix>_shell.stl, _tree.stl, _section.png.

    Either `part_path` (a mesh file) or `mesh` (an in-memory Trimesh) must be
    given.
    """
    cfg = cfg or Config()
    t: dict[str, float] = {}
    warnings: list[str] = list(cfg.resolution_warnings())
    clock = time.perf_counter

    # `progress(step, fraction, detail)` is called as each stage starts, so a
    # caller can show what the pipeline is doing instead of an opaque spinner.
    def emit(step: str, fraction: float, detail: str = "") -> None:
        if progress is not None:
            try:
                progress(step, fraction, detail)
            except Exception:
                pass

    # -- stage 1: ingest ------------------------------------------------
    emit("reading mesh", 0.05)
    t0 = clock()
    if mesh is not None:
        part = ingest.from_mesh(mesh, cfg)
    elif part_path is not None:
        part = ingest.load_part(part_path, cfg)
    else:
        raise ValueError("provide part_path or mesh")
    t["ingest"] = clock() - t0

    # -- stage 1b: multi-cavity layout ----------------------------------
    # Replicate the part into `cfg.quantity` instances and carry on with the
    # cluster as the thing being moulded. Everything downstream -- voxels, hot
    # spots, shell offset, cavity boolean -- treats it as one multi-body
    # casting, so no later stage needs to know how many parts it holds. At
    # quantity=1 this is the identity and the single-part path is unchanged.
    # -- orient for pouring ---------------------------------------------
    # Everything downstream assumes +Z is up: the sprue rises, the cup opens at
    # the top, hot spots are fed from above. A CAD file's own +Z is not
    # reliably the direction the casting is poured, so the part is rotated to
    # put `cfg.pour_up` on +Z here and the finished meshes are rotated back at
    # the end. The build itself needs no knowledge of any of this.
    up = np.asarray(cfg.pour_up, dtype=np.float64)
    to_z = trimesh.geometry.align_vectors(up, np.array([0.0, 0.0, 1.0]))
    reorient = not np.allclose(to_z, np.eye(4), atol=1e-9)
    if reorient:
        emit("orienting for pour", 0.06, "aligning the pour direction with +Z")
        turned = part.mesh.copy()
        turned.apply_transform(to_z)
        part = ingest.from_mesh(turned, cfg)
    from_z = np.linalg.inv(to_z)

    single_part = part
    t0 = clock()
    layout = layout_mod.build(part, cfg)
    if cfg.quantity > 1:
        emit("laying out parts", 0.08,
             f"{cfg.quantity} instances, {layout.arrangement} arrangement")
        part = ingest.from_mesh(layout.mesh, cfg)
    t["layout"] = clock() - t0

    # Sanity-check the scale before doing anything expensive. STEP and IGES
    # files frequently arrive in inches or metres, and a part whose whole
    # bounding box is smaller than the shell that would wrap it produces a
    # geometrically valid but physically meaningless mould -- a 5 mm "casting"
    # inside a 6 mm ceramic skin. Say so plainly rather than emitting nonsense.
    span = float(np.max(part.extents))
    if span < 2.0 * cfg.shell_thickness:
        warnings.append(
            f"the part is only {span:.1f} mm across but the shell is "
            f"{cfg.shell_thickness} mm thick. The file may be in inches or "
            f"metres rather than millimetres -- try scaling it, or reduce the "
            f"shell thickness."
        )
    try:
        n_bodies = len(part.mesh.split(only_watertight=False))
    except Exception:
        n_bodies = 1
    # A multi-cavity cluster is legitimately N separate bodies -- that is what
    # was asked for -- so only flag bodies the layout did not create.
    if n_bodies > cfg.quantity:
        extra = n_bodies // max(cfg.quantity, 1)
        warnings.append(
            f"the file contains {extra} separate bodies; this looks like an "
            "assembly rather than a single casting. Gating is generated for all "
            "of them together, which is rarely what you want."
        )

    if not part.watertight:
        warnings.append("part mesh is not watertight even after repair; "
                        "voxel occupancy may be unreliable")

    # -- the single shared lattice --------------------------------------
    emit("building voxel grid", 0.12,
           f"{part.volume/1000:.1f} cm3 part, modulus {part.modulus:.2f} mm")
    t0 = clock()
    grid_part = Grid.from_bounds(part.bounds[0], part.bounds[1],
                                 cfg.voxel_pitch, cfg.grid_pad)
    lo_full, hi_full = _tree_bounds(part, cfg)
    pad_lo = np.maximum(np.ceil((grid_part.origin - lo_full) / cfg.voxel_pitch), 0).astype(int)
    far_part = grid_part.origin + np.asarray(grid_part.shape) * cfg.voxel_pitch
    pad_hi = np.maximum(np.ceil((hi_full - far_part) / cfg.voxel_pitch), 0).astype(int)
    grid = grid_part.expanded(pad_lo, pad_hi)
    # Assert the frames really are translatable; a failure here is the bug
    # this design exists to prevent.
    offset = grid_part.offset_in(grid)
    # Refuse, rather than warn, above the ceiling. A warning is appended to a
    # list the caller only reads once the job finishes -- which is no use when
    # the job is the thing that exhausts the machine. The distance transform
    # allocates several float64 arrays the size of the whole grid, so a 32M
    # voxel grid peaks near a gigabyte per call and several calls are made;
    # past that the machine swaps, fills its disk, and the build dies hundreds
    # of seconds in with nothing to show.
    #
    # The suggested pitch is computed rather than hand-waved, so the message
    # tells the user exactly what to type.
    if grid.n_voxels > cfg.max_voxels:
        # Aim comfortably under the ceiling and round UP to a tenth. Landing
        # exactly on the limit means the suggested pitch is itself refused,
        # which tells the user to type a number that does not work.
        need = (grid.n_voxels / (0.75 * cfg.max_voxels)) ** (1.0 / 3.0)
        suggest = math.ceil(cfg.voxel_pitch * need * 10.0) / 10.0
        raise ValueError(
            f"this part needs a {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} "
            f"grid = {grid.n_voxels/1e6:.0f}M voxels at {cfg.voxel_pitch} mm "
            f"pitch, over the {cfg.max_voxels/1e6:.0f}M limit. The distance "
            f"transform needs several arrays that size and the machine would "
            f"run out of memory. Raise the voxel pitch to about "
            f"{suggest:.1f} mm, or raise max_voxels if you know the machine "
            f"can take it."
        )
    if grid.n_voxels > cfg.max_voxels // 2:
        warnings.append(
            f"grid is {grid.shape} = {grid.n_voxels/1e6:.1f}M voxels; "
            f"the build will be slow. A pitch of "
            f"{cfg.voxel_pitch * (grid.n_voxels / (cfg.max_voxels // 4)) ** (1/3):.1f} mm "
            "would be considerably faster."
        )
    t["grid"] = clock() - t0

    # -- occupancy, once, on the full grid ------------------------------
    emit("voxelising part", 0.20,
           f"grid {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} "
           f"at {cfg.voxel_pitch} mm pitch")
    t0 = clock()
    part_occ = hotspots.voxelize_fast(part.mesh, grid)
    t["voxelize"] = clock() - t0
    vox_vol = float(part_occ.sum() * grid.voxel_volume)
    # Only meaningful when the part is big enough to voxelise at all; below a
    # few voxels the ratio explodes and reports absurd percentages.
    if part.volume > 0 and vox_vol > 0 and part.volume > 8 * grid.voxel_volume:
        err = abs(vox_vol - part.volume) / part.volume
        if err > 0.10:
            warnings.append(
                f"voxelised volume differs from mesh volume by {err*100:.0f}% "
                f"at pitch {cfg.voxel_pitch}mm; reduce voxel_pitch for accuracy"
            )
    elif part.volume <= 8 * grid.voxel_volume:
        warnings.append(
            f"the part occupies only a few voxels at {cfg.voxel_pitch}mm pitch; "
            "results are meaningless until the pitch is reduced or the part is "
            "scaled up"
        )

    # -- stage 2: hot spots (same grid that gets stamped) ---------------
    emit("finding hot spots", 0.45, "distance transform of the casting")
    t0 = clock()
    spots = hotspots.detect(part, grid, cfg, occ=part_occ,
                            instances=max(1, int(cfg.quantity)))
    t["hotspots"] = clock() - t0

    # -- stage 4 before 3: risers set the poured mass the choke must pass
    emit("sizing risers", 0.58,
           f"{len(spots)} hot spot{'s' if len(spots) != 1 else ''} found")
    t0 = clock()
    riser_list: list = []          # sized after gating, which claims a spot
    t["risers"] = clock() - t0

    # -- stage 3: gating ------------------------------------------------
    emit("sizing gating", 0.65,
         f"sprue, {cfg.quantity} runners and ingates" if cfg.quantity > 1
         else "single top-gated sprue")
    t0 = clock()
    if cfg.quantity > 1:
        plan = gating.design_multi(part, layout, cfg, hotspots=spots)
    else:
        plan = gating.design(part, cfg, risers=None, hotspots=spots)
    t["gating"] = clock() - t0

    # -- stage 4: risers, for whatever the sprue does not already feed ----
    # Gating runs first because the sprue doubles as a feeder: it is sized to
    # out-freeze the section it lands on, so that hot spot needs no riser of
    # its own. One appendage to cut off instead of two stacked on the same
    # thermal centre.
    t0 = clock()
    # Every point where metal enters the casting counts as a feeder. On a
    # multi-cavity mould that is the ingates, one per instance -- NOT the
    # central sprue, which touches no casting at all. Measuring reach from the
    # sprue made a riser depend on where an instance sat in the cluster, so
    # identical parts came out with different numbers of them.
    if cfg.use_risers:
        feeders = [(sg.p1, sg.r1) for sg in plan.segments if sg.kind == "ingate"]
        if not feeders:
            feeders = [(sg.p0, sg.r0) for sg in plan.segments if sg.kind == "sprue"]
        riser_list = risers.design(spots, part, cfg, feeders=feeders)
    else:
        riser_list = []
    t["risers"] = clock() - t0

    # -- stage 5: synthesis --------------------------------------------
    emit("building shell", 0.72,
           f"choke {2*plan.choke_radius:.1f} mm, offsetting {cfg.shell_thickness} mm skin")
    t0 = clock()
    # Risers are stamped into the metal, not merely sized. They were designed
    # and then discarded here for a while -- `max_risers` and the whole modulus
    # calculation were computing a number that never reached the mould.
    result = synthesis.build(part, grid, plan, riser_list, cfg, part_occ=part_occ)

    # Back into the file's own frame, so the STLs the user downloads line up
    # with the CAD they uploaded rather than with the pour orientation.
    if reorient:
        for name in ("shell_mesh", "fired_mesh", "tree_mesh"):
            mesh = getattr(result, name, None)
            if mesh is not None:
                mesh.apply_transform(from_z)
    t["synthesis"] = clock() - t0

    # -- outputs --------------------------------------------------------
    files: dict[str, str] = {}
    emit("exporting", 0.92,
           f"{len(result.shell_mesh.faces):,} shell triangles")
    t0 = clock()
    if write_files:
        prefix = Path(out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        shell_path = prefix.with_name(prefix.name + "_shell.stl")
        result.shell_mesh.export(shell_path)
        files["shell_stl"] = str(shell_path)
        if cfg.show_fired_shell and result.fired_mesh is not None:
            fired_path = prefix.with_name(prefix.name + "_shell_fired.stl")
            result.fired_mesh.export(fired_path)
            files["fired_stl"] = str(fired_path)
        if cfg.write_tree_stl:
            tree_path = prefix.with_name(prefix.name + "_tree.stl")
            result.tree_mesh.export(tree_path)
            files["tree_stl"] = str(tree_path)
        if cfg.write_section_png:
            png_path = prefix.with_name(prefix.name + "_section.png")
            report.section_png(result, png_path, cfg, part_occ=part_occ)
            files["section_png"] = str(png_path)
    t["export"] = clock() - t0

    metal_volume = result.diagnostics["metal_volume"]
    thick = result.thickness_measured
    # One voxel of quantisation is inherent to a rasterised wall, so the
    # tolerance is a whole pitch plus a relative allowance.
    tol = cfg.voxel_pitch + 0.15 * cfg.shell_thickness
    if abs(thick["median"] - cfg.shell_thickness) > tol:
        warnings.append(
            f"measured shell wall {thick['median']:.2f}mm is outside "
            f"{tol:.2f}mm of the {cfg.shell_thickness}mm target"
        )

    emit("done", 1.0)
    return Stats(
        part=part.as_dict(),
        part_volume=part.volume,
        casting_modulus=part.modulus,
        hot_spots=spots.as_list(),
        gating={**plan.as_dict(), "ratio": list(cfg.gating_ratio),
                "segments": [s.as_dict() for s in plan.segments]},
        risers=[r.as_dict() for r in riser_list],
        shell={
            "thickness_target": cfg.shell_thickness,
            "thickness_measured": thick,
            "watertight": bool(result.shell_mesh.is_watertight),
            "volume": result.diagnostics["shell_volume"],
            "triangles": int(len(result.shell_mesh.faces)),
            "tree_triangles": int(len(result.tree_mesh.faces)),
            "tree_watertight": bool(result.tree_mesh.is_watertight),
            "cup": result.diagnostics["cup"],
            "cavity_exact": result.diagnostics.get("cavity_exact", False),
            "tree_exact": result.diagnostics.get("tree_exact", False),
            "fired": result.diagnostics["fired"],
        },
        metal_volume=metal_volume,
        part_volume_voxel=vox_vol,
        # Yield uses the *voxelised* part volume so numerator and denominator
        # share one basis; mixing mesh volume with voxel metal volume can
        # exceed 100% on thin parts that under-voxelise.
        yield_pct=100.0 * vox_vol / metal_volume if metal_volume > 0 else 0.0,
        grid={**grid.as_dict(), "part_grid_offset": offset.tolist()},
        layout={**layout.as_dict(),
                "part_volume_each": single_part.volume,
                "part_volume_total": part.volume},
        files=files, timings=t, warnings=warnings, config=cfg.as_dict(),
    )
