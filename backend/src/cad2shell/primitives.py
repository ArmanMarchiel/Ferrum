"""Voxel-space stamping of gating primitives.

Everything is rasterised as a tapered capsule: the locus of points within a
linearly-interpolated radius of a line segment. That single shape covers
sprues (tapered), runners and ingates (constant), riser bodies and necks.

All world -> index conversion goes through the Grid, so a segment stamped here
lands on precisely the cells the hot-spot stage analysed.
"""
from __future__ import annotations

import numpy as np

from .voxel import Grid


def _segment_bbox_slice(grid: Grid, p0, p1, rmax: float, pad: int = 2):
    """Index-space bounding box of a capsule, clipped to the grid."""
    lo_w = np.minimum(p0, p1) - rmax
    hi_w = np.maximum(p0, p1) + rmax
    lo = grid.to_index(lo_w)[0] - pad
    hi = grid.to_index(hi_w)[0] + pad + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(grid.shape))
    if np.any(hi <= lo):
        return None
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def stamp_capsule(occ: np.ndarray, grid: Grid, p0, p1, r0: float, r1: float,
                  round_ends: bool = False) -> int:
    """OR a tapered cone/capsule into `occ`. Returns the number of cells added.

    By default the ends are **flat** (a truncated cone bounded by the two end
    planes). This matters metallurgically: a capsule's hemispherical cap would
    push the sprue's choke half a bore below the plane where it actually meets
    the runner, so the narrowest cross-section -- the thing the whole Bernoulli
    calculation sizes -- would not be where the design says it is. Pass
    round_ends=True for a true capsule (used for riser necks, where a fillet
    is desirable).

    Only the segment's bounding box is evaluated, which keeps this cheap even
    on a large grid.
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    r0, r1 = float(r0), float(r1)
    sl = _segment_bbox_slice(grid, p0, p1, max(r0, r1))
    if sl is None:
        return 0

    # World centres of every voxel in the bbox.
    ii = np.arange(sl[0].start, sl[0].stop)
    jj = np.arange(sl[1].start, sl[1].stop)
    kk = np.arange(sl[2].start, sl[2].stop)
    gi, gj, gk = np.meshgrid(ii, jj, kk, indexing="ij")
    idx = np.stack([gi, gj, gk], axis=-1)
    pts = grid.origin + (idx.astype(np.float64) + 0.5) * grid.pitch

    axis = p1 - p0
    length_sq = float(axis @ axis)
    rel = pts - p0
    if length_sq < 1e-12:
        # Degenerate segment: a sphere of the larger radius.
        dist = np.linalg.norm(rel, axis=-1)
        inside = dist <= max(r0, r1)
    else:
        t_raw = (rel @ axis) / length_sq
        t = np.clip(t_raw, 0.0, 1.0)
        closest = p0 + t[..., None] * axis
        dist = np.linalg.norm(pts - closest, axis=-1)
        radius = r0 + t * (r1 - r0)     # linear taper along the axis
        inside = dist <= radius
        if not round_ends:
            # Cut at the end planes so the solid is a truncated cone, not a
            # capsule: no hemispherical overhang past either endpoint.
            inside &= (t_raw >= 0.0) & (t_raw <= 1.0)

    before = int(occ[sl].sum())
    occ[sl] |= inside
    return int(occ[sl].sum()) - before


def stamp_segments(occ: np.ndarray, grid: Grid, segments) -> dict:
    """Stamp a list of Segments. Returns per-kind added-cell counts.

    Necks are stamped with rounded ends so the riser-to-casting junction gets
    a fillet instead of a sharp re-entrant corner; everything else is flat-
    ended so cross-sections land exactly where they were designed.
    """
    added: dict[str, int] = {}
    for seg in segments:
        round_ends = bool(seg.meta.get("round_ends", seg.kind == "neck"))
        n = stamp_capsule(occ, grid, seg.p0, seg.p1, seg.r0, seg.r1,
                          round_ends=round_ends)
        added[seg.kind] = added.get(seg.kind, 0) + n
    return added


def stamp_sphere(occ: np.ndarray, grid: Grid, centre, radius: float) -> int:
    return stamp_capsule(occ, grid, centre, centre, radius, radius, round_ends=True)


def half_space_above(grid: Grid, z: float) -> np.ndarray:
    """Boolean mask of voxels whose centres lie above world height z."""
    k = np.arange(grid.shape[2])
    zc = grid.origin[2] + (k + 0.5) * grid.pitch
    mask = np.zeros(grid.shape, dtype=bool)
    mask[:, :, zc > z] = True
    return mask
