"""The metal path must be one connected component, or part of it never fills."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from cad2shell import demo_parts, gating, hotspots, ingest, risers, synthesis
from cad2shell.config import Config
from cad2shell.voxel import Grid
from cad2shell.pipeline import _tree_bounds


def _rebuild(name, **overrides):
    """Re-run the stages far enough to inspect the voxel grids directly."""
    cfg = Config(**overrides)
    part = ingest.from_mesh(demo_parts.get(name))
    gp = Grid.from_bounds(part.bounds[0], part.bounds[1], cfg.voxel_pitch, cfg.grid_pad)
    lo, hi = _tree_bounds(part, cfg)
    pad_lo = np.maximum(np.ceil((gp.origin - lo) / cfg.voxel_pitch), 0).astype(int)
    far = gp.origin + np.asarray(gp.shape) * cfg.voxel_pitch
    pad_hi = np.maximum(np.ceil((hi - far) / cfg.voxel_pitch), 0).astype(int)
    grid = gp.expanded(pad_lo, pad_hi)
    occ = hotspots.voxelize_fast(part.mesh, grid)
    spots = hotspots.detect(part, grid, cfg, occ=occ)
    rl = risers.design(spots, part, cfg)
    plan = gating.design(part, cfg, risers=rl, hotspots=spots)
    targets = [[p.centre[0], p.centre[1], float(part.bounds[0][2]) + 0.5 * cfg.voxel_pitch]
               for p in rl] or [[w[0], w[1], float(part.bounds[0][2])] for w in spots.world]
    plan = gating.add_ingates(plan, np.asarray(targets), cfg)
    res = synthesis.build(part, grid, plan, rl, cfg, part_occ=occ)
    return dict(cfg=cfg, part=part, grid=grid, occ=occ, spots=spots,
                risers=rl, plan=plan, result=res)


def test_metal_path_is_single_connected_component(case):
    """Part + sprue + runner + ingates + risers must form ONE solid.

    A second component means metal poured into the cup can never reach it --
    the casting would come out short. 6-connectivity is used because metal
    flows through faces, not through a shared edge or corner.
    """
    pitch = 1.0 if case == "thin_bracket" else 1.5
    st = _rebuild(case, voxel_pitch=pitch)
    metal = st["result"].metal
    structure = ndimage.generate_binary_structure(3, 1)     # faces only
    _, n = ndimage.label(metal, structure=structure)
    assert n == 1, f"{case}: metal path has {n} disconnected components"


def test_every_riser_is_attached_to_the_casting(case):
    """Each riser must touch the part, otherwise it feeds nothing."""
    pitch = 1.0 if case == "thin_bracket" else 1.5
    st = _rebuild(case, voxel_pitch=pitch)
    grid, occ = st["grid"], st["occ"]
    structure = ndimage.generate_binary_structure(3, 1)
    for r in st["risers"]:
        riser_only = grid.empty()
        from cad2shell import primitives as pr
        for seg in r.segments():
            pr.stamp_capsule(riser_only, grid, seg.p0, seg.p1, seg.r0, seg.r1,
                             round_ends=seg.kind == "neck")
        joined = riser_only | occ
        _, n = ndimage.label(joined, structure=structure)
        assert n == 1, f"{case}: riser {r.index} is not attached to the casting"


def test_shell_fully_separates_metal_from_the_outside(case):
    """Except at the cup mouth, no metal voxel may touch open air.

    If it did, the mould would leak there. The cup mouth is the one legitimate
    opening, so the test floods from the grid boundary and checks that the only
    metal the flood reaches is at the cup.
    """
    pitch = 1.0 if case == "thin_bracket" else 1.5
    st = _rebuild(case, voxel_pitch=pitch)
    res, grid = st["result"], st["grid"]
    metal, shell = res.metal, res.shell

    free = ~(metal | shell)                     # open space
    structure = ndimage.generate_binary_structure(3, 1)
    labels, _ = ndimage.label(free, structure=structure)
    # the component touching the grid corner is the outside world
    outside_label = labels[0, 0, 0]
    assert outside_label != 0, "grid corner is not empty; padding is too small"
    outside = labels == outside_label

    # dilate the outside by one voxel and see which metal it touches
    touching = ndimage.binary_dilation(outside, structure=structure) & metal
    if not touching.any():
        return                                   # fully enclosed is also fine
    mouth_z = res.diagnostics["cup"]["mouth_z"]
    zs = grid.to_world(np.argwhere(touching))[:, 2]
    assert zs.min() >= mouth_z - 2.0 * grid.pitch, (
        f"{case}: metal is exposed at z={zs.min():.1f}, well below the cup mouth "
        f"at z={mouth_z:.1f} -- the shell has a gap"
    )
