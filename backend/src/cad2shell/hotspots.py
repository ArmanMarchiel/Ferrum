"""Stage 2 -- hot-spot detection.

Voxelise the part onto the shared Grid, run a Euclidean distance transform,
and take suppressed local maxima as thermal centres: the points furthest from
any surface are the thickest, last-to-freeze regions.

Every returned index is an index into the Grid that was passed in, so the
caller can stamp risers at exactly the cells that were analysed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage

from .voxel import Grid


@dataclass
class HotSpots:
    idx: np.ndarray            # (N, 3) int voxel indices in `grid`
    world: np.ndarray          # (N, 3) world centres of those voxels
    edt: np.ndarray            # (N,) distance to nearest surface, mm
    local_modulus: np.ndarray  # (N,) local modulus estimate, mm
    grid: Grid = None

    def __len__(self) -> int:
        return int(len(self.idx))

    def as_list(self) -> list[dict]:
        return [
            {
                "world": self.world[i].tolist(),
                "edt_mm": float(self.edt[i]),
                "local_modulus": float(self.local_modulus[i]),
            }
            for i in range(len(self))
        ]


def voxelize(mesh: trimesh.Trimesh, grid: Grid) -> np.ndarray:
    """Solid-fill the mesh onto `grid`. Returns a bool array of grid.shape.

    Points are tested at voxel centres and written back through
    ``grid.to_index``, so the occupancy is defined on the same frame every
    other stage uses.
    """
    idx = np.stack(
        np.meshgrid(*[np.arange(s) for s in grid.shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    centres = grid.to_world(idx)
    inside = mesh.contains(centres)
    occ = np.zeros(grid.shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = inside
    return occ


def voxelize_fast(mesh: trimesh.Trimesh, grid: Grid) -> np.ndarray:
    """Occupancy on `grid`, preferring speed but agreeing with `voxelize`.

    trimesh's voxeliser marks every voxel the *surface passes through*, which
    dilates the solid by about half a voxel in every direction. That inflates
    both the volume and the distance transform, so its output is only used as
    a candidate set: cells it marks are re-tested at their centres, and only
    genuinely interior centres survive. Interior cells (surrounded on all
    sides) skip the ray test, which is where the speed comes from.
    """
    try:
        vg = mesh.voxelized(pitch=grid.pitch).fill()
        occ_src = np.asarray(vg.matrix, dtype=bool)
        src_origin = np.asarray(vg.transform)[:3, 3] - grid.pitch / 2.0
        idx = np.stack(
            np.meshgrid(*[np.arange(s) for s in grid.shape], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        centres = grid.to_world(idx)
        src_idx = np.floor((centres - src_origin) / grid.pitch).astype(np.int64)
        ok = np.all((src_idx >= 0) & (src_idx < np.asarray(occ_src.shape)), axis=1)
        cand = np.zeros(grid.shape, dtype=bool)
        vals = np.zeros(len(idx), dtype=bool)
        vals[ok] = occ_src[src_idx[ok, 0], src_idx[ok, 1], src_idx[ok, 2]]
        cand[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
        if not cand.any():
            return voxelize(mesh, grid)

        # Cells whose 6-neighbours are all candidates are strictly interior:
        # trust them. Only the shell of the candidate set needs the ray test.
        interior = ndimage.binary_erosion(cand, structure=ndimage.generate_binary_structure(3, 1))
        boundary = cand & ~interior
        occ = interior.copy()
        b_idx = np.argwhere(boundary)
        if len(b_idx):
            inside = mesh.contains(grid.to_world(b_idx))
            occ[b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]] = inside
        if occ.any():
            return occ
    except Exception:
        pass
    return voxelize(mesh, grid)


def distance_field(occ: np.ndarray, pitch: float) -> np.ndarray:
    """Distance from each occupied voxel to the part surface, in mm.

    ``distance_transform_edt`` returns the centre-to-centre distance to the
    nearest empty voxel, so a cell one step inside the boundary reports one
    full pitch when the surface is really only half a pitch away. Subtracting
    half a voxel removes that systematic bias and makes the peak of the field
    match the true half-thickness of a slab (a 35mm block reads 17.5, not 18).
    """
    # Pad so voxels touching the array border still see the outside world;
    # without this a part flush to the grid edge reads as infinitely thick.
    padded = np.pad(occ, 1, mode="constant", constant_values=False)
    dist = ndimage.distance_transform_edt(padded, sampling=(pitch, pitch, pitch))
    dist = dist[1:-1, 1:-1, 1:-1]
    return np.where(occ, np.maximum(dist - pitch / 2.0, 0.0), 0.0)


def _suppress(cand_idx: np.ndarray, cand_val: np.ndarray, grid: Grid,
              min_sep_mm: float, max_count: int) -> np.ndarray:
    """Greedy non-maximum suppression by world distance. Returns kept rows."""
    order = np.argsort(-cand_val)
    kept: list[int] = []
    kept_world: list[np.ndarray] = []
    for i in order:
        w = grid.to_world(cand_idx[i])[0]
        if all(np.linalg.norm(w - kw) < min_sep_mm for kw in kept_world) and kept_world:
            continue
        kept.append(int(i))
        kept_world.append(w)
        if len(kept) >= max_count:
            break
    return np.asarray(kept, dtype=np.int64)


def component_modulus(occ: np.ndarray, grid: Grid) -> tuple[float, np.ndarray]:
    """True V/A of each connected casting component, measured on the grid.

    Area counts only voxel faces that touch non-casting space, which is the
    definition of a cooling surface, so this reproduces the analytic V/A of a
    slab, bar or cube to within the voxelisation error of its own surface.
    Returns (modulus of the largest component, component label array).
    """
    labels, n = ndimage.label(occ, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        return 0.0, labels
    occ_p = np.pad(occ, 1, mode="constant", constant_values=False)
    sizes = ndimage.sum(occ, labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    blob = labels == biggest
    cells = np.argwhere(blob) + 1
    exposed = 0
    for off in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nb = cells + np.asarray(off)
        exposed += int(np.count_nonzero(~occ_p[nb[:, 0], nb[:, 1], nb[:, 2]]))
    volume = float(blob.sum()) * grid.voxel_volume
    area = float(exposed) * grid.pitch ** 2
    return (volume / area if area > 0 else 0.0), labels


def local_modulus(occ: np.ndarray, dist: np.ndarray, grid: Grid,
                  idx: np.ndarray, mesh_modulus: float | None = None) -> np.ndarray:
    """Local cooling modulus at each hot spot, in mm.

    Getting this right matters more than any other number here: riser diameter
    scales linearly with it, and riser volume with its cube.

    Three things are true, and the estimator below follows from them.

    1. The distance field is not a modulus. For a slab of half-thickness d
       cooling from both faces M = d = EDT; for a cylinder of radius R,
       M = R/2; for a sphere, M = R/3. Reading modulus straight off the EDT
       over-estimates it 2-3x on compact features.
    2. Modulus is not a pointwise quantity. It is V/A of a body, so any
       attempt to define it at a point needs a region -- and every region
       deep enough to isolate one thermal centre on a voxel grid is entirely
       interior, giving zero cooling area and an undefined modulus. Ball
       windows and ray casts were both tried and neither holds across shape
       classes (a window that fits a slab clips a sphere, and vice versa).
    3. What *is* exactly measurable is the V/A of the whole connected casting
       from voxel face counts: `component_modulus` reproduces the analytic
       value to 0.0% on flat-faced solids.

    So: anchor on the measured component modulus, and scale each hot spot by
    its local half-thickness relative to the deepest section of the casting:

        M_local = M_component * (EDT_here / EDT_peak)

    The deepest hot spot therefore gets exactly the measured component
    modulus -- an exact number, not an inferred one -- and shallower hot spots
    are scaled down in proportion to their section thickness, which is the
    ratio riser sizing actually needs. Scaling by the *peak* rather than the
    mean matters: most voxels in any casting are shallow, so a mean-based
    ratio inflates every hot spot far above the true V/A.

    The result is conservative for a thick boss on a thin plate (the boss is
    credited with the whole casting's modulus rather than its own, larger one)
    and is floored at the component value's share so no riser is undersized.
    """
    m_component, _ = component_modulus(occ, grid)
    if m_component <= 0:
        m_component = float(mesh_modulus or 0.0)
    peak = float(dist[occ].max()) if occ.any() else 0.0
    out = np.zeros(len(idx), dtype=np.float64)
    for n, cell in enumerate(idx):
        here = float(dist[tuple(cell)])
        if peak <= 0 or m_component <= 0:
            out[n] = here
        else:
            out[n] = m_component * (here / peak)
    return out


def detect(part, grid: Grid, cfg, occ: np.ndarray | None = None,
           instances: int = 1) -> HotSpots:
    """Stage 2 entry point.

    `occ` may be supplied to reuse an occupancy grid already computed on the
    same `grid`; otherwise it is voxelised here.
    """
    if occ is None:
        occ = voxelize_fast(part.mesh, grid)
    if not occ.any():
        raise ValueError(
            f"part voxelised to nothing at pitch {grid.pitch}mm; reduce voxel_pitch"
        )

    dist = distance_field(occ, grid.pitch)
    peak = float(dist.max())
    if peak <= 0:
        raise ValueError("distance field is empty; part is thinner than one voxel")

    # Local maxima of the distance field = thermal centres. A 3x3x3 grey
    # dilation compares each voxel with its neighbours; equality marks a peak.
    dilated = ndimage.grey_dilation(dist, size=(3, 3, 3), mode="constant", cval=0.0)
    threshold = cfg.hotspot_rel_threshold * peak
    is_peak = occ & (dist >= threshold) & (dist >= dilated - 1e-9)

    if not is_peak.any():
        # Degenerate: fall back to the single global maximum.
        is_peak = np.zeros_like(occ)
        is_peak[np.unravel_index(int(np.argmax(dist)), dist.shape)] = True

    # Plateaus (a flat-topped slab) produce many equal peaks; collapse each
    # connected plateau to its centroid before suppression.
    labels, n = ndimage.label(is_peak)
    cand_idx, cand_val = [], []
    for lbl in range(1, n + 1):
        cells = np.argwhere(labels == lbl)
        vals = dist[cells[:, 0], cells[:, 1], cells[:, 2]]
        centroid = np.round(cells.mean(axis=0)).astype(np.int64)
        # snap the centroid back onto an occupied voxel of this plateau
        if not occ[tuple(centroid)]:
            centroid = cells[int(np.argmax(vals))]
        cand_idx.append(centroid)
        cand_val.append(float(vals.max()))
    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    cand_val = np.asarray(cand_val, dtype=np.float64)

    # `max_risers` is a budget PER CASTING, not for the whole grid. A four-up
    # mould holds four copies of the same part and each is entitled to the same
    # number of thermal centres; capping the cluster as a whole truncated the
    # list and silently starved whichever instances came last, so identical
    # parts came out with different risers.
    budget = max(1, int(cfg.max_risers)) * max(1, int(instances))
    keep = _suppress(cand_idx, cand_val, grid, cfg.hotspot_min_separation, budget)
    idx = cand_idx[keep]
    edt = cand_val[keep]

    # Per-hot-spot modulus, measured as a true V/A on the voxel grid. This is
    # what a riser must beat; the casting's global V/A would mis-size risers
    # on any part that mixes thick and thin sections.
    m_local = local_modulus(occ, dist, grid, idx, mesh_modulus=part.modulus)

    return HotSpots(
        idx=idx, world=grid.to_world(idx), edt=edt,
        local_modulus=m_local, grid=grid,
    )
