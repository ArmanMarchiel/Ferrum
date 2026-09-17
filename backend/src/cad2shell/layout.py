"""Stage 1b -- multi-cavity layout: N copies of one part in a single mould.

A production investment shell rarely carries one casting. The pattern wax is
assembled into a *tree*: several identical patterns clustered around one
central sprue, poured together and cut apart afterwards. Running four parts on
one tree divides the sprue metal, the shell ceramic and the furnace time by
four, which is where the economics of the process actually live.

This module owns only the *placement* question -- where the N instances sit
relative to each other. Everything downstream (voxelisation, hot spots, shell
offset, cavity boolean) then treats the cluster as one multi-body casting and
needs no changes. The gating that ties the instances together is
`gating.design_multi`.

Two arrangements, chosen by `cfg.layout`:

``grid``
    Instances on a rectangular raster, row-major, centred on the origin of the
    original part. Simple, dense, and the right answer for a flat part that
    will be gated from a runner bar.

``radial``
    Instances on a circle about a central axis, each rotated to face outward,
    which is the classic wax-tree cluster. Denser in Z for tall parts and it
    gives every instance an identical feed path from the central sprue -- so
    they fill and freeze alike, which a raster cannot promise.

The spacing between neighbours is `cfg.part_spacing` millimetres of clear gap.
It must leave room for two shell walls plus the ceramic between them, so the
default is derived from the shell thickness rather than being a bare constant:
two instances closer than 2*shell_thickness would have their shells merge into
one solid block with no ceramic between the cavities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class Instance:
    """One placed copy of the part."""
    index: int
    transform: np.ndarray        # 4x4 world transform applied to the base mesh
    centre: np.ndarray           # world centroid of the placed bounding box
    angle: float = 0.0           # radians about +Z, radial layouts only

    def __post_init__(self) -> None:
        self.transform = np.asarray(self.transform, dtype=np.float64).reshape(4, 4)
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "centre": self.centre.tolist(),
            "angle_deg": math.degrees(self.angle),
            "transform": self.transform.tolist(),
        }


@dataclass
class Layout:
    """The full arrangement: N instances plus the mesh they combine into."""
    instances: list[Instance]
    mesh: trimesh.Trimesh        # all instances concatenated, one multi-body mesh
    quantity: int
    arrangement: str
    spacing: float
    pitch_xy: np.ndarray         # centre-to-centre step actually used
    radius: float = 0.0          # radial layouts: cluster radius, mm
    columns: int = 1             # grid layouts: instances per row
    rows: int = 1

    def as_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "arrangement": self.arrangement,
            "spacing": self.spacing,
            "pitch_xy": np.asarray(self.pitch_xy).tolist(),
            "radius": self.radius,
            "columns": self.columns,
            "rows": self.rows,
            "instances": [i.as_dict() for i in self.instances],
        }

    @property
    def centres(self) -> np.ndarray:
        """(N,3) array of instance centres -- what the gating stage feeds."""
        if not self.instances:
            return np.zeros((0, 3))
        return np.array([i.centre for i in self.instances], dtype=np.float64)


def _up_axis(cfg) -> np.ndarray:
    """The axis a part stands on, as a unit vector.

    `rot` in a placement means "turned on the plate", so it is taken about
    this rather than about a fixed +Z. With no pour direction configured the
    two are identical, which is why every existing single-orientation build is
    unaffected.
    """
    up = np.asarray(getattr(cfg, "pour_up", (0.0, 0.0, 1.0)), dtype=float).reshape(3)
    n = np.linalg.norm(up)
    if n == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return up / n


def default_spacing(cfg) -> float:
    """Clear gap between neighbouring instances, in mm.

    Two adjacent cavities each carry a shell wall of `shell_thickness`, so the
    clear metal-to-metal distance must exceed TWICE that or the two walls meet
    with no parting between them. Measured directly: at 6 mm shell and a 12 mm
    gap the space between two castings is 12 mm of solid ceramic and 0 mm of
    air -- one block, and the castings come out joined. It takes just over
    2*shell before any parting appears, so a voxel of margin is added on top.

    This is measured surface to surface, not box to box. A casting fills only
    part of its bounding box, and box separation is a different, much larger
    quantity -- demanding it wastes plate area on parts that nest together
    perfectly well.
    """
    if cfg.part_spacing is not None:
        return float(cfg.part_spacing)
    return float(2.0 * cfg.shell_thickness + 2.0 * cfg.voxel_pitch)


def grid_shape(n: int, columns: int | None = None) -> tuple[int, int]:
    """Rows and columns for `n` instances on a raster.

    Defaults to the squarest arrangement, which keeps the cluster compact and
    therefore the runner bar short. A long thin row would need a runner as long
    as the row, and the metal in it is pure loss.
    """
    if n <= 0:
        raise ValueError("quantity must be >= 1")
    if columns is not None and columns > 0:
        cols = min(int(columns), n)
    else:
        cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def from_placements(part, cfg, placements) -> Layout:
    """Place instances exactly where the caller says, ignoring the auto layout.

    `placements` is a list of dicts, one per instance, each with ``x``, ``y``,
    ``z`` (mm, offsets applied to the base part) and ``rot`` (degrees about
    +Z, applied about the part's own centre). That is what the viewer's gizmo
    produces, and it is the only way a hand-made arrangement can reach the
    build -- the raster and the circle in this module answer a different
    question, which is where to put parts when nobody has said.

    The rotation is about the part's own centre rather than the world origin,
    so turning an instance in place does not also fling it across the plate.

    ``rot`` is a turn *on the plate*, so it is taken about the pour-up axis
    rather than about +Z. The two are the same thing until the part is flipped;
    once it is, spinning about +Z would tip each instance onto its side by a
    different amount -- a radial cluster gives every instance a different
    ``rot`` -- and the ring of upright parts would come out as a fan of parts
    lying at odd angles.
    """
    base = part.mesh
    lo, hi = np.asarray(base.bounds[0]), np.asarray(base.bounds[1])
    base_centre = (lo + hi) / 2.0
    up = _up_axis(cfg)

    instances: list[Instance] = []
    meshes: list[trimesh.Trimesh] = []
    for i, pl in enumerate(placements):
        dx = float(pl.get("x", 0.0))
        dy = float(pl.get("y", 0.0))
        dz = float(pl.get("z", 0.0))
        angle = math.radians(float(pl.get("rot", 0.0)))

        R = trimesh.transformations.rotation_matrix(angle, up, base_centre)
        T = np.eye(4)
        T[:3, 3] = [dx, dy, dz]
        M = T @ R
        m = base.copy()
        m.apply_transform(M)
        meshes.append(m)
        centre = (M @ np.append(base_centre, 1.0))[:3]
        instances.append(Instance(i, M, centre, angle=angle))

    if not instances:
        raise ValueError("no placements given")

    return Layout(instances=instances, mesh=_combine(meshes),
                  quantity=len(instances), arrangement="custom",
                  spacing=default_spacing(cfg),
                  pitch_xy=np.array([0.0, 0.0]))


def build(part, cfg) -> Layout:
    """Place `cfg.quantity` copies of `part` and return the combined mesh.

    The base mesh is used as-is for instance 0 when the layout is a grid and
    the quantity is 1, so a single-part run is bit-identical to the behaviour
    before multi-cavity support existed.
    """
    n = int(cfg.quantity)
    if n < 1:
        raise ValueError("quantity must be >= 1")

    # A hand-made arrangement overrides the automatic one entirely.
    if getattr(cfg, "placements", None):
        return from_placements(part, cfg, cfg.placements)

    base = part.mesh
    extents = np.asarray(base.extents, dtype=np.float64)
    lo, hi = np.asarray(base.bounds[0]), np.asarray(base.bounds[1])
    base_centre = (lo + hi) / 2.0
    gap = default_spacing(cfg)

    if n == 1:
        inst = Instance(0, np.eye(4), base_centre)
        return Layout(instances=[inst], mesh=base.copy(), quantity=1,
                      arrangement=cfg.layout, spacing=gap,
                      pitch_xy=np.array([0.0, 0.0]), columns=1, rows=1)

    if cfg.layout == "radial":
        return _radial(base, base_centre, extents, n, gap, cfg)
    return _grid(base, base_centre, extents, n, gap, cfg)


def _grid(base, base_centre, extents, n: int, gap: float, cfg) -> Layout:
    """Row-major raster, centred on the original part's centre."""
    rows, cols = grid_shape(n, cfg.layout_columns)
    step_x = float(extents[0] + gap)
    step_y = float(extents[1] + gap)

    # Centre the whole raster on the base part, so the cluster grows outward
    # symmetrically instead of marching off in +X.
    span_x = (cols - 1) * step_x
    span_y = (rows - 1) * step_y

    instances: list[Instance] = []
    meshes: list[trimesh.Trimesh] = []
    for i in range(n):
        r, c = divmod(i, cols)
        dx = c * step_x - span_x / 2.0
        dy = r * step_y - span_y / 2.0
        T = np.eye(4)
        T[:3, 3] = [dx, dy, 0.0]
        m = base.copy()
        m.apply_transform(T)
        meshes.append(m)
        instances.append(Instance(i, T, base_centre + np.array([dx, dy, 0.0])))

    return Layout(instances=instances, mesh=_combine(meshes), quantity=n,
                  arrangement="grid", spacing=gap,
                  pitch_xy=np.array([step_x, step_y]), columns=cols, rows=rows)


def _radial(base, base_centre, extents, n: int, gap: float, cfg) -> Layout:
    """Instances on a circle, each turned to face outward from the axis.

    The cluster radius is set so neighbouring instances clear each other by
    `gap` along the circle: with N instances the angular step is 2*pi/N, and
    the chord between neighbouring centres must exceed the instance's own
    in-plane footprint plus the gap.

        chord = 2 R sin(pi/N)  >=  footprint + gap
        =>  R >= (footprint + gap) / (2 sin(pi/N))

    For N = 2 this degenerates to a straight line, which is correct: two parts
    face each other across the sprue.
    """
    # The footprint that matters is the one presented tangentially. Each
    # instance is rotated to face outward, so its Y extent lies along the
    # circle -- but a rotated part sweeps its diagonal in the worst case, so
    # the in-plane diagonal is the honest bound.
    foot = float(math.hypot(extents[0], extents[1]))
    step = 2.0 * math.pi / n
    radius = (foot + gap) / (2.0 * math.sin(math.pi / n)) if n > 1 else 0.0
    # Never let the cluster sit so tight that an instance overlaps the sprue.
    radius = max(radius, 0.5 * foot + gap)

    instances: list[Instance] = []
    meshes: list[trimesh.Trimesh] = []
    for i in range(n):
        angle = i * step
        dx = radius * math.cos(angle)
        dy = radius * math.sin(angle)
        # rotate about the part's own centre, then translate out to the circle
        R = trimesh.transformations.rotation_matrix(angle, _up_axis(cfg), base_centre)
        T = np.eye(4)
        T[:3, 3] = [dx, dy, 0.0]
        M = T @ R
        m = base.copy()
        m.apply_transform(M)
        meshes.append(m)
        centre = (M @ np.append(base_centre, 1.0))[:3]
        instances.append(Instance(i, M, centre, angle=angle))

    return Layout(instances=instances, mesh=_combine(meshes), quantity=n,
                  arrangement="radial", spacing=gap,
                  pitch_xy=np.array([2.0 * radius * math.sin(math.pi / n)] * 2),
                  radius=radius)


def _combine(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Concatenate placed instances into one multi-body mesh.

    Deliberately NOT a boolean union: the instances are disjoint by
    construction, so a union would cost a great deal of time to compute exactly
    the concatenation. The result is a legitimately multi-body watertight mesh
    -- trimesh reports `is_watertight` True for it because every body is closed
    -- and both the voxeliser and the cavity boolean handle it directly.
    """
    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices()
    return combined
