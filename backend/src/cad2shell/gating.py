"""Stage 3 -- gating design.

Size the choke from continuity + Bernoulli, then apply a non-pressurised
gating ratio to get runner and ingate areas. The output is a list of Segments
in world millimetres; the synthesis stage rasterises them and knows nothing
about the metallurgy, so this whole module can be swapped out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .config import GRAVITY

# How close two thermal centres must be in modulus to count as equally heavy.
# Symmetric features voxelise to nearly, not exactly, the same value.
_TIE_TOLERANCE = 0.05


@dataclass
class Segment:
    """A tapered capsule of metal between two world points."""
    kind: str                  # 'sprue' | 'cup' | 'runner' | 'ingate' | 'riser' | 'neck'
    p0: np.ndarray
    p1: np.ndarray
    r0: float
    r1: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p0 = np.asarray(self.p0, dtype=np.float64).reshape(3)
        self.p1 = np.asarray(self.p1, dtype=np.float64).reshape(3)
        self.r0 = float(self.r0)
        self.r1 = float(self.r1)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def volume(self) -> float:
        """Truncated-cone volume; a fair estimate of the metal it holds."""
        h = self.length
        return float(math.pi * h / 3.0 * (self.r0**2 + self.r0 * self.r1 + self.r1**2))

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "p0": self.p0.tolist(), "p1": self.p1.tolist(),
            "r0": self.r0, "r1": self.r1, "length": self.length,
            "volume": self.volume, **self.meta,
        }


@dataclass
class GatingPlan:
    choke_area: float          # mm^2
    choke_radius: float        # mm
    sprue_r_bottom: float
    sprue_r_top: float
    runner_radius: float
    ingate_radius: float
    n_ingates: int
    metal_volume: float = 0.0  # mm^3 of metal the tree carries
    segments: list[Segment] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "choke_area": self.choke_area,
            "choke_radius": self.choke_radius,
            "sprue_r_bottom": self.sprue_r_bottom,
            "sprue_r_top": self.sprue_r_top,
            "runner_radius": self.runner_radius,
            "ingate_radius": self.ingate_radius,
            "n_ingates": self.n_ingates,
            "metal_volume": self.metal_volume,
            "ratio": None,
        }


"""Note on what is NOT here any more.

An earlier version sized the choke from Bernoulli and continuity:

    A_choke = W / (rho * t * Cd * sqrt(2 g H))

with a pour time, a discharge coefficient and a metallostatic head. It was
correct arithmetic answering the wrong question for this scheme. The sprue in
an investment tree doubles as the feeder, so its base is set by the modulus it
has to out-freeze -- and that requirement is always the larger of the two. The
fill-rate choke never won on any part tested, which made pour_time, Cd, the
alloy density and head_height inputs that changed nothing in the output.

They have been removed rather than left on screen implying otherwise. If a
scheme ever needs a fill-limited choke, the formula is above and belongs back
here with a test that shows it changing the geometry.
"""


def gate_point(part, hotspots, cfg) -> np.ndarray:
    """Where the sprue meets the casting.

    Directly above the heaviest hot spot: the thickest section is the last to
    freeze, so it is where shrinkage porosity forms if it is not fed. A ray is
    cast upward from that thermal centre to find the point on the part's top
    surface above it, so the sprue lands on real material rather than hovering
    over a hole or a recess.
    """
    lo, hi = part.bounds
    if hotspots is None or len(hotspots) == 0:
        return np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2]])

    # Gate on the group of equally heavy thermal centres, not on whichever one
    # happens to come first. A symmetric casting has its hot spots in mirrored
    # pairs of identical modulus, and `argmax` on a tie picks one arbitrarily --
    # which puts the sprue off to one side of a part that plainly wants feeding
    # down its centreline, and feeds one half of the casting better than the
    # other.
    #
    # Everything within a small tolerance of the peak counts as tied, since two
    # sides of a symmetric part rarely voxelise to bit-identical moduli.
    mod = np.asarray(hotspots.local_modulus, dtype=np.float64)
    peak = float(mod.max())
    tied = np.flatnonzero(mod >= peak * (1.0 - _TIE_TOLERANCE))
    spot = np.asarray(hotspots.world, dtype=np.float64)[tied].mean(axis=0)

    # Then pull the gate onto any plane the casting is symmetric about.
    #
    # Averaging the tied spots alone is not enough. Two sides of a symmetric
    # feature voxelise to slightly different moduli -- 3.94 against 3.18 for
    # the same boss on either side of a plummer block -- so only one side ties
    # for the peak and the average sits over that side. The result is a sprue
    # a few millimetres off a centreline the part obviously has.
    #
    # Symmetry is measured on the mesh, not guessed from the hot spots: mirror
    # the part about the candidate plane and see whether it lands on itself.
    spot = _snap_to_symmetry(part, spot)

    # The centroid of a mirrored pair can land in the gap between them -- over
    # the bore of a plummer block, say. `_gate_site` handles that: it requires
    # metal under the point it returns and searches outward when the column is
    # hollow, so the sprue stays on the centreline but moves along it to solid
    # material rather than jumping to one side.

    # The sprue lands on the solid surface above the thermal centre. Taking the
    # highest ray hit is not enough: on a part with a bore the column above a
    # hot spot can break out into the bore, and the sprue would then pour into
    # a hole instead of onto the casting. `_gate_site` requires metal beneath
    # the surface it picks, and searches outward when the column is hollow.
    site = _gate_site(part, spot, cfg)
    if site is not None:
        return site
    return np.array([spot[0], spot[1], float(hi[2])])


def _snap_to_symmetry(part, point, tol_frac: float = 0.02):
    """Move `point` onto the casting's plane of symmetry, where it has one.

    Checked per axis by mirroring the mesh about the plane through the part's
    centre and comparing volumes and bounds. A casting that is symmetric about
    a plane should be gated on it -- both halves then freeze and feed alike,
    which is the whole reason a founder centres a sprue by eye.

    Deliberately conservative: it only moves the gate along an axis that really
    is a mirror plane, and leaves an asymmetric part's gate exactly where the
    thermal analysis put it.
    """
    p = np.asarray(point, dtype=np.float64).copy()
    lo, hi = part.bounds
    centre = (lo + hi) / 2.0
    extent = hi - lo
    try:
        volume = float(part.mesh.volume)
    except Exception:
        return p
    if volume <= 0:
        return p

    for axis in (0, 1):          # only in plan; +Z is the pour direction
        if extent[axis] <= 1e-9:
            continue
        mirror = np.eye(4)
        mirror[axis, axis] = -1.0
        mirror[axis, 3] = 2.0 * centre[axis]
        flipped = part.mesh.copy()
        flipped.apply_transform(mirror)
        # a mirror plane maps the bounding box onto itself
        if not np.allclose(flipped.bounds, part.bounds,
                           atol=tol_frac * float(np.max(extent))):
            continue
        # and preserves the volume of the intersection, which is the real test
        try:
            shared = trimesh.boolean.intersection(
                [part.mesh, flipped], engine="manifold")
            overlap = float(shared.volume) if shared is not None else 0.0
        except Exception:
            continue
        if overlap >= (1.0 - tol_frac) * volume:
            p[axis] = centre[axis]
    return p


def design(part, cfg, risers=None, hotspots=None) -> GatingPlan:
    """Stage 3 entry point: a single top-gated feeder.

    Investment shells are poured as one assembly hanging from a single sprue,
    so the gating is one appendage attached directly to the casting:

        pour cup  ->  tapered sprue  ->  top surface of the part

    There is deliberately no runner and no ingates. The earlier layout parked
    the sprue beside the part and ran a runner the full length of the casting
    underneath it, with ingates climbing back up -- a horizontal plate scheme
    for a sand mould with a parting line. On a 165 mm part that produced a
    174 mm runner and a tree larger than the casting it fed, none of it touching
    the part.

    The sprue also serves as the feeder. Its base is widened to the diameter a
    riser would need over the section it lands on, so it out-freezes that
    section and can feed it; the taper up to the pour cup keeps the falling
    stream against the wall so the system does not aspirate air.
    """
    lo, hi = part.bounds

    metal_volume = part.volume + sum(r.volume for r in (risers or []))

    gate = gate_point(part, hotspots, cfg)
    # The sprue must clear the casting, not merely rise a fixed distance from
    # wherever it landed. On a tall part the gate site can be well below the
    # top, and a fixed 30 mm of sprue is then swallowed entirely: the cup ends
    # up buried inside the casting's own silhouette, `open_pour_cup` clears
    # shell above a mouth that is inside the part, and the mould comes out
    # sealed with no way to pour into it.
    z_clear = float(hi[2]) + cfg.sprue_height
    z_sprue_top = max(float(gate[2] + cfg.sprue_height), z_clear)

    # The sprue doubles as the feeder, so its base is set by the modulus of the
    # section it lands on: it has to stay molten until that section has frozen
    # or it cannot feed the shrinkage.
    m_feed = float(np.max(hotspots.local_modulus)) if (
        hotspots is not None and len(hotspots)) else part.modulus
    from .risers import diameter_for_modulus
    r_feeder = diameter_for_modulus(
        cfg.riser_modulus_factor * m_feed, cfg.riser_hd_ratio) / 2.0

    r_base = r_feeder
    r_top = r_base * cfg.sprue_taper
    r_cup = r_top * cfg.cup_radius_factor

    segments: list[Segment] = [
        # sprue: from the casting's top surface up to the cup, widening as it
        # rises so the choke stays at the bottom where the metal enters
        Segment("sprue",
                [gate[0], gate[1], float(gate[2])],
                [gate[0], gate[1], z_sprue_top],
                r_base, r_top,
                {"choke_at": "p0", "feeder_radius": r_feeder}),
        # pour cup: a short funnel, open to the world at its top
        Segment("cup",
                [gate[0], gate[1], z_sprue_top],
                [gate[0], gate[1], z_sprue_top + cfg.cup_height],
                r_top, r_cup,
                {"open_to_world": True}),
    ]

    plan = GatingPlan(
        choke_area=math.pi * r_base ** 2, choke_radius=r_base,
        sprue_r_bottom=r_base, sprue_r_top=r_top,
        runner_radius=0.0, ingate_radius=0.0, n_ingates=0,
        metal_volume=metal_volume, segments=segments,
    )
    plan._layout = {
        "gate": gate.tolist(), "z_sprue_top": z_sprue_top,
        "r_cup": r_cup, "r_feeder": r_feeder,
    }
    return plan


def add_ingates(plan: GatingPlan, targets: np.ndarray, cfg) -> GatingPlan:
    """Retained for API compatibility; a top-gated sprue has no ingates.

    Ingates connect a runner to the casting. This scheme has no runner -- the
    sprue meets the part directly -- so there is nothing to connect.
    """
    return plan


def design_multi(part, layout, cfg, hotspots=None) -> GatingPlan:
    """Gating for a multi-cavity mould: one sprue feeding N instances.

    The single-part scheme drops a sprue straight onto the casting, because
    with one casting the sprue and the feeder are the same appendage. That does
    not generalise: giving each of N instances its own sprue would produce N
    separate moulds that happen to share a bounding box, and nothing would tie
    them into one pourable assembly.

    So the multi-cavity tree is the classic three-level one:

        pour cup -> central sprue -> runner(s) -> one ingate per instance

    The sprue stands on the cluster's own axis, clear of every instance, and
    runs down past them. Runners leave it at the level of the instance tops and
    reach out to each casting; a short ingate drops from the runner onto the
    part. Metal therefore enters every cavity by an equal-length path, which is
    what makes N castings from one pour come out alike.

    Areas follow `cfg.gating_ratio` = (choke, runner, ingate), non-pressurised
    at 1:2:2 by default. The ratio is applied to the *total* flow: the runner
    area is shared between the branches leaving the sprue and the ingate area
    is per-instance, so each cavity sees total_ingate_area / N.
    """
    centres = layout.centres
    n = len(centres)
    if n <= 1:
        return design(part, cfg, hotspots=hotspots)

    lo, hi = part.bounds
    z_top = float(hi[2])

    # -- the metal, and therefore the choke ------------------------------
    # part.volume here is already the volume of the whole cluster, because the
    # pipeline ingests the combined mesh. The choke passes all of it.
    metal_volume = part.volume

    # -- where the sprue stands -----------------------------------------
    # On the cluster's centroid in plan, which for both layouts is the point
    # equidistant from every instance. For a grid with an even instance count
    # that lands in the gap between them, which is exactly where a sprue wants
    # to be: touching none of the castings.
    axis_xy = centres[:, :2].mean(axis=0)

    # The runner bar sits above the instances so metal falls into each cavity
    # rather than having to climb. Ingates then drop from it onto each part.
    z_runner = z_top + cfg.runner_clearance_multi
    # As in `design`: the sprue top is measured from the top of the casting so
    # the cup always clears it, never from the gate site alone.
    z_sprue_top = max(z_runner + cfg.sprue_height,
                      float(hi[2]) + cfg.sprue_height)

    # -- the sprue must out-freeze what it feeds ---------------------------
    # It is the last liquid metal in the system, so it has to stay molten until
    # the castings have solidified, exactly as in the single-part case.
    m_feed = float(np.max(hotspots.local_modulus)) if (
        hotspots is not None and len(hotspots)) else part.modulus
    from .risers import diameter_for_modulus
    r_feeder = diameter_for_modulus(
        cfg.riser_modulus_factor * m_feed, cfg.riser_hd_ratio) / 2.0

    r_base = r_feeder
    r_top = r_base * cfg.sprue_taper
    r_cup = r_top * cfg.cup_radius_factor
    a_choke_real = math.pi * r_base ** 2

    # -- ratio-driven runner and ingate areas -----------------------------
    ratio_c, ratio_r, ratio_g = cfg.gating_ratio
    total_runner_area = a_choke_real * (ratio_r / ratio_c)
    total_ingate_area = a_choke_real * (ratio_g / ratio_c)

    # Each instance is fed by its own branch, so the per-branch area is the
    # total divided between them. Radial layouts run one branch per instance;
    # a grid runs one spine per row, so its branches carry a row each.
    per_ingate_area = total_ingate_area / n
    r_ingate = math.sqrt(per_ingate_area / math.pi)

    segments: list[Segment] = []

    # sprue: from the runner level up to the cup, narrowest at the bottom so
    # the choke is where the metal enters the distribution system
    sprue_bottom = np.array([axis_xy[0], axis_xy[1], z_runner])
    segments.append(Segment(
        "sprue", sprue_bottom,
        [axis_xy[0], axis_xy[1], z_sprue_top],
        r_base, r_top,
        {"choke_at": "p0", "feeder_radius": r_feeder}))
    segments.append(Segment(
        "cup", [axis_xy[0], axis_xy[1], z_sprue_top],
        [axis_xy[0], axis_xy[1], z_sprue_top + cfg.cup_height],
        r_top, r_cup, {"open_to_world": True}))

    # -- runners and ingates ----------------------------------------------
    # One runner per instance, radiating from the sprue base out to a point
    # above the instance, then an ingate dropping onto the casting. Radiating
    # spokes rather than a single bar keep every feed path the same length,
    # which a shared bar cannot do for instances at different distances.
    n_branches = max(n, 1)
    per_runner_area = total_runner_area / n_branches
    r_runner = math.sqrt(per_runner_area / math.pi)

    # Runners leave the sprue at slightly different heights rather than all
    # meeting it at one point. Physically this is how a wax tree is actually
    # assembled -- patterns are stacked up the sprue, not welded to a single
    # ring -- and geometrically it is what keeps the mould exportable.
    #
    # Every junction in this tree is a place where two or more analytic cutters
    # meet, and a cutter that *ends* exactly on another's surface touches it
    # tangentially. The cavity boolean answers a tangential contact with
    # zero-area triangles, which survive in memory harmlessly but weld into the
    # surface when the STL is re-read -- putting four faces on an edge that
    # should carry two, so the shell reads as non-watertight despite having no
    # hole in it. Both ends of every runner therefore overlap what they join,
    # exactly as `synthesis._segment_solid` does for the same reason.
    stagger = 0.25 * cfg.voxel_pitch
    overlap = max(0.5 * cfg.voxel_pitch, 0.25 * r_ingate)

    # One instance's own footprint, which is what the gate search must scale to.
    try:
        bodies = part.mesh.split(only_watertight=False)
        instance_span = float(max(
            np.max((b.bounds[1] - b.bounds[0])[:2]) for b in bodies))
    except Exception:
        instance_span = float(np.max(part.extents[:2])) / max(n, 1)

    for i, c in enumerate(centres):
        sprue_tap = sprue_bottom + np.array([0.0, 0.0, i * stagger])
        # Land the ingate on real material. The instance centre is preferred,
        # but a bore through the middle of the casting makes that column empty
        # air, so the site is searched for rather than assumed.
        site = _gate_site(part, c, cfg,
                          instance_angle=getattr(layout.instances[i], 'angle', 0.0),
                          sprue_xy=axis_xy, instance_span=instance_span)
        if site is None:
            # Nothing solid anywhere in this instance's footprint. Feeding it
            # is not possible, so say so rather than gate into a void.
            raise ValueError(
                f"instance {i} has no solid surface to gate onto; the part may "
                "be a shell or the layout may have placed it badly")
        gate_xy = site[:2]
        z_land = float(site[2])

        runner_end = np.array([gate_xy[0], gate_xy[1], z_runner + i * stagger])

        # The runner runs from inside the sprue to past the ingate's axis, so
        # neither junction is a tangency. Extending along its own direction
        # keeps every branch the same length, which is the invariant that makes
        # the cavities fill alike.
        direction = runner_end - sprue_tap
        span = float(np.linalg.norm(direction))
        if span > 1e-9:
            unit = direction / span
            runner_p0 = sprue_tap - unit * overlap
            runner_p1 = runner_end + unit * overlap
        else:
            runner_p0, runner_p1 = sprue_tap, runner_end
        segments.append(Segment(
            "runner", runner_p0, runner_p1, r_runner, r_runner,
            {"instance": i}))
        # The ingate starts above the runner's axis so the two solids overlap,
        # and ends below the casting's surface so it cuts into real material
        # rather than stopping on it.
        segments.append(Segment(
            "ingate",
            [gate_xy[0], gate_xy[1], runner_end[2] + overlap],
            [gate_xy[0], gate_xy[1], z_land - overlap],
            r_ingate, r_ingate,
            {"instance": i, "round_ends": True}))

    plan = GatingPlan(
        choke_area=a_choke_real, choke_radius=r_base,
        sprue_r_bottom=r_base, sprue_r_top=r_top,
        runner_radius=r_runner, ingate_radius=r_ingate, n_ingates=n,
        metal_volume=metal_volume, segments=segments,
    )
    plan._layout = {
        "axis_xy": axis_xy.tolist(), "z_runner": z_runner,
        "z_sprue_top": z_sprue_top, "r_cup": r_cup,
        "r_feeder": r_feeder,
        "quantity": n, "arrangement": layout.arrangement,
        "runner_area_total": total_runner_area,
        "ingate_area_total": total_ingate_area,
        "ingate_area_each": per_ingate_area,
    }
    return plan


def _solid_top_at(part, xy, z_above: float, z_floor: float | None = None,
                  pitch: float = 1.5, span: float | None = None) -> float | None:
    """Height of the casting's top SOLID surface in the column at (x, y).

    Not simply the highest ray hit. A ray dropped down the axis of a bore hits
    the far side of the bore and the underside of the part, and both are
    surfaces of the mesh -- so "the highest hit" happily reports a face that
    has nothing but air above it. That is how an ingate ends up pouring into a
    bore instead of onto the casting.

    A hit only counts as a landing site if the material immediately below it
    is solid, which is tested directly with a containment probe. Returns None
    when the whole column is hollow, so the caller can look elsewhere rather
    than gate into a hole.
    """
    try:
        locs, _, _ = part.mesh.ray.intersects_location(
            np.array([[xy[0], xy[1], z_above]]),
            np.array([[0.0, 0.0, -1.0]]))
    except Exception:
        return None
    if not len(locs):
        return None
    # The FIRST surface the falling stream meets, not the highest solid one.
    # A ray dropped down a bore passes the bore wall and lands on the bore
    # floor, which does have metal beneath it -- so a "highest solid hit" test
    # accepts it and the ingate is run down inside the hole. Metal poured from
    # above stops at whatever it hits first, so that is the only landing point
    # that means anything.
    #
    # It still has to be solid underneath: the first hit in a column that only
    # clips a thin overhang is not somewhere to attach a feeder.
    eps = 1e-3 * max(1.0, float(np.max(part.extents)))
    z = float(np.max(locs[:, 2]))
    try:
        if not bool(part.mesh.contains(np.array([[xy[0], xy[1], z - eps]]))[0]):
            return None
    except Exception:
        return z

    # Distinguish the floor of a bore from a merely low step. Height cannot
    # tell them apart -- a stepped casting has good surfaces far below its high
    # point, and rejecting those leaves the part with nowhere to gate at all.
    #
    # What separates them is how the ingate would have to reach the site. The
    # ingate is a cylinder of finite width, so the test is whether one of that
    # width can come straight down onto this surface without cutting through
    # the casting on the way. Rays are cast down the wall of that cylinder: on
    # an open step they all reach the same surface, while in a bore they strike
    # the material ringing the hole well above the floor.
    # The test is scale-free: walk outward until the surface height changes,
    # and see whether this column sits in a pit or on a plateau. A bore of any
    # diameter is a pit -- ringed on every side by material that stands well
    # above its floor -- while a step is open on at least one side, which is
    # what lets the metal reach it.
    #
    # Fixed probe distances were tried first and cannot work: a ring smaller
    # than the bore samples only the hole's own floor and reports it clear,
    # and a ring larger than the part leaves it entirely.
    depth_tol = max(4.0 * pitch, 2.0)
    # One instance's own size. Using the cluster's extents here makes the very
    # first probe step clear of the instance entirely, so a bore is never seen.
    span = float(span) if span else float(np.max(part.extents[:2]))
    open_sides = 0
    for k in range(8):
        a = 2.0 * math.pi * k / 8.0
        cos_a, sin_a = math.cos(a), math.sin(a)
        higher = False
        for frac in (0.02, 0.05, 0.10, 0.18, 0.30):
            r = frac * span
            try:
                hits, _, _ = part.mesh.ray.intersects_location(
                    np.array([[xy[0] + r * cos_a, xy[1] + r * sin_a, z_above]]),
                    np.array([[0.0, 0.0, -1.0]]))
            except Exception:
                return z
            if not len(hits):
                break                  # ran off the part: this side is open
            if float(np.max(hits[:, 2])) > z + depth_tol:
                higher = True
                break                  # this side is walled above us
        if not higher:
            open_sides += 1
    if open_sides == 0:
        return None                    # enclosed on every side: a bore
    return z


def _gate_site(part, instance_centre, cfg, instance_angle: float = 0.0,
               sprue_xy=None, instance_span: float | None = None) -> np.ndarray | None:
    """Where on one instance the ingate should land.

    The instance's own centre is tried first, because a feed that enters over
    the middle of the casting fills it symmetrically. On a part with a bore
    through the middle -- a plummer block, a bearing housing, any ring -- that
    column is empty, so the search spirals outward over the instance's
    footprint and takes the highest genuinely solid column it finds.

    Preferring the highest such column keeps the ingate short and puts it on
    the casting's upper surface, which is where a top feed belongs.
    """
    lo, hi = part.bounds
    z_above = float(hi[2]) + 10.0

    # A landing site has to be on the casting's upper surface, not somewhere
    # down inside it. "Upper" is judged against the tallest column found in
    # this instance's own footprint, so a stepped part still gates onto its
    # lower step rather than being rejected for not reaching the highest point
    # on the whole casting.
    centre = np.asarray(instance_centre, dtype=np.float64)
    # The search radius must scale to ONE instance, not to the cluster. Sizing
    # it from the cluster's extents makes the probe ring land in the empty
    # space between widely spaced parts, and the instance is then reported as
    # having no surface to gate onto at all.
    span = float(instance_span) if instance_span else float(np.max(part.extents[:2]))
    reach = 0.5 * span
    probes = [centre[:2]]
    for frac in (0.15, 0.30, 0.45):
        for k in range(8):
            angle = 2.0 * math.pi * k / 8.0
            probes.append(centre[:2] + frac * reach
                          * np.array([math.cos(angle), math.sin(angle)]))
    tops = [t for t in (_solid_top_at(part, xy, z_above, pitch=cfg.voxel_pitch, span=span)
                              for xy in probes)
            if t is not None]
    if not tops:
        return None
    # No height floor. A stepped casting has perfectly good surfaces well below
    # its highest point -- the boss_plate's plate is 25 mm under its boss --
    # and rejecting them for being low leaves the instance with nowhere to gate
    # at all. The bore case is already excluded inside `_solid_top_at`, by
    # requiring the first surface met from above to have solid metal beneath it.
    z_floor = None

    z = _solid_top_at(part, centre[:2], z_above, z_floor,
                      pitch=cfg.voxel_pitch, span=span)
    if z is not None:
        return np.array([centre[0], centre[1], z])

    # The centre is over a hole, so search outward for solid material. The
    # angular offset is measured in the instance's OWN frame, not the world's:
    # a radial layout rotates every instance, and probing along world axes
    # would land each one on a different feature of the same part. Feeding
    # nominally identical castings through different paths is exactly what
    # multi-up gating exists to avoid.
    span = float(instance_span) if instance_span else float(np.max(part.extents[:2]))
    phase = float(instance_angle or 0.0)
    n_probe = 24
    # Collect candidates from EVERY ring before choosing, not just the first
    # ring that happens to hit metal. On a part with a bore the near edge and
    # the far edge lie at different radii, so a search that returns from the
    # first successful ring cannot compare them -- and a radial cluster ended
    # up gating every instance on its FAR lobe, each runner crossing the whole
    # tree when half the distance would do.
    ring = []
    for frac in (0.12, 0.20, 0.28, 0.36, 0.45, 0.55):
        radius = frac * span
        for k in range(n_probe):
            angle = phase + 2.0 * math.pi * k / n_probe
            xy = centre[:2] + radius * np.array([math.cos(angle), math.sin(angle)])
            z = _solid_top_at(part, xy, z_above, z_floor,
                              pitch=cfg.voxel_pitch, span=span)
            if z is not None:
                # keep the probe index so ties break the same way for every
                # instance, making the chosen site symmetric across the cluster
                ring.append((k, xy, z))
    if True:
        if ring:
            # Among sites that are all high enough to gate onto, take the one
            # NEAREST THE SPRUE.
            #
            # Height was the primary key here, which is wrong on a symmetric
            # part: both sides offer an equally good landing face, and a site a
            # fraction of a millimetre higher on the far side beat one much
            # closer on the near side. Every runner then crossed the whole
            # cluster to reach the far lobe of its instance -- 116 mm each on a
            # four-up plummer block where half that would do, all of it metal
            # to remelt.
            #
            # So height becomes a tolerance rather than a ranking: anything
            # within a shell thickness of the best face is an acceptable place
            # to land, and among those the shortest feed wins.
            best_z = max(r[2] for r in ring)
            tol = max(cfg.shell_thickness, 2.0 * cfg.voxel_pitch)
            usable = [r for r in ring if r[2] >= best_z - tol]

            def rank(r):
                _, xy, z = r
                near = float(np.linalg.norm(xy - np.asarray(sprue_xy)[:2])) \
                    if sprue_xy is not None else 0.0
                # `r[0]` is the probe index, which breaks remaining ties the
                # same way for every instance and keeps the cluster symmetric.
                return (round(near, 6), r[0])

            k, xy, z = min(usable, key=rank)
            return np.array([xy[0], xy[1], z])
    return None
