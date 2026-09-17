"""Stage 4 -- riser sizing by the modulus method.

A riser must freeze after the section it feeds, so it needs a larger cooling
modulus: M_riser >= f * M_section, with f = 1.2 by convention (Chvorinov's
rule makes freezing time proportional to M^2, so f = 1.2 buys ~44% more
freezing time).

For a cylindrical top riser of diameter D and height H, standing on the
casting so its base face is not a cooling surface:

    V = pi D^2 H / 4
    A = pi D H  (side)  +  pi D^2 / 4  (top)
    M = V / A = D H / (4 H + D)

With H = k*D this solves in closed form for D, which keeps the sizing
deterministic and makes the invariant testable exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gating import Segment


@dataclass
class Riser:
    centre: np.ndarray      # world point on the casting where the riser stands
    diameter: float
    height: float
    modulus: float          # the riser's own cooling modulus, mm
    feeds_modulus: float    # modulus of the section it feeds, mm
    neck_diameter: float
    neck_length: float
    index: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)

    @property
    def volume(self) -> float:
        return float(math.pi * self.diameter**2 / 4.0 * self.height)

    @property
    def factor(self) -> float:
        """Achieved modulus ratio -- the quantity the invariant test checks."""
        return float(self.modulus / self.feeds_modulus) if self.feeds_modulus > 0 else math.inf

    def segments(self) -> list[Segment]:
        """Neck + body, as metal to be stamped."""
        r = self.diameter / 2.0
        rn = self.neck_diameter / 2.0
        base = self.centre
        neck_top = base + np.array([0.0, 0.0, self.neck_length])
        body_top = neck_top + np.array([0.0, 0.0, self.height])
        return [
            Segment("neck", base, neck_top, rn, rn, {"riser": self.index}),
            Segment("riser", neck_top, body_top, r, r, {"riser": self.index}),
        ]

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "world": self.centre.tolist(),
            "diameter": self.diameter,
            "height": self.height,
            "modulus": self.modulus,
            "feeds_modulus": self.feeds_modulus,
            "factor": self.factor,
            "neck_diameter": self.neck_diameter,
            "neck_length": self.neck_length,
            "volume": self.volume,
        }


def cylinder_modulus(diameter: float, height: float, base_is_fed: bool = True) -> float:
    """Cooling modulus of a cylinder. If base_is_fed, the base face does not cool."""
    d, h = float(diameter), float(height)
    volume = math.pi * d**2 / 4.0 * h
    area = math.pi * d * h + math.pi * d**2 / 4.0
    if not base_is_fed:
        area += math.pi * d**2 / 4.0
    return volume / area


def diameter_for_modulus(target_modulus: float, hd_ratio: float) -> float:
    """Invert M = D H / (4H + D) with H = k D.

        M = k D^2 / (4 k D + D) = k D / (4k + 1)
        =>  D = M (4k + 1) / k
    """
    k = float(hd_ratio)
    if k <= 0:
        raise ValueError("riser_hd_ratio must be positive")
    return float(target_modulus * (4.0 * k + 1.0) / k)


def size_riser(feeds_modulus: float, cfg, index: int = 0,
               centre=(0.0, 0.0, 0.0)) -> Riser:
    """Size one riser to beat `feeds_modulus` by cfg.riser_modulus_factor."""
    target = cfg.riser_modulus_factor * float(feeds_modulus)
    d = diameter_for_modulus(target, cfg.riser_hd_ratio)
    d = max(d, cfg.min_riser_diameter)
    h = cfg.riser_hd_ratio * d
    m = cylinder_modulus(d, h, base_is_fed=True)

    # If the diameter floor kicked in, recompute height so the achieved modulus
    # still meets the target rather than silently under-feeding.
    if m < target:
        # M = D H / (4H + D)  ->  H = M D / (D - 4M), valid while D > 4M
        if d > 4.0 * target:
            h = target * d / (d - 4.0 * target)
            m = cylinder_modulus(d, h, base_is_fed=True)
        else:
            d = diameter_for_modulus(target, cfg.riser_hd_ratio)
            h = cfg.riser_hd_ratio * d
            m = cylinder_modulus(d, h, base_is_fed=True)

    neck_d = max(cfg.riser_neck_factor * d, 2.0)
    return Riser(
        centre=centre, diameter=d, height=h, modulus=m,
        feeds_modulus=float(feeds_modulus), neck_diameter=neck_d,
        neck_length=max(0.4 * neck_d, 2.0), index=index,
    )


def design(hotspots, part, cfg, feeders=()) -> list[Riser]:
    """Stage 4 entry point: a top riser per hot spot the sprue cannot feed.

    The riser stands on the part surface directly above its hot spot, so it
    feeds down into the thermal centre. Sizing uses the hot spot's *local*
    modulus, not the casting's global V/A, so a part mixing thick and thin
    sections gets each riser matched to the section it actually feeds.

    `feeders` is every point where liquid metal already enters the casting, as
    (world_point, radius) pairs: the sprue foot on a single-part mould, and one
    entry per ingate on a multi-cavity one. A hot spot within feeding distance
    of any of them gets NO riser.

    In an investment tree the feeder is deliberately sized to do a riser's job
    -- the sprue base is widened to 1.2x the modulus of the section it lands on
    -- so it already IS the riser for what it reaches. Adding another on top
    stacks two feeders on one thermal centre: more metal to remelt, a taller
    tree, and two things to cut off where one would do.

    Passing only the sprue on a multi-cavity mould is the bug this signature
    exists to prevent. Every instance is fed by its own ingate, but measuring
    reach from the central sprue instead made the answer depend on where an
    instance happened to sit in the cluster -- so four identical parts came out
    with two risers, one, one and none.
    """
    risers: list[Riser] = []
    ray = part.mesh.ray
    hi_z = float(part.bounds[1][2])

    # How far the sprue can feed, not merely how wide its foot is.
    #
    # A feeder does not only serve the metal directly under it. Liquid flows
    # sideways through the section as it solidifies, and the accepted reach for
    # a plate-like section is about 4.5x its thickness from the feeder edge
    # (roughly 2x from riser edge as the feeding distance proper, plus a 2.5x
    # end-effect where the casting's own end chills it). Expressed in modulus,
    # which is what this pipeline measures, that is about 9 x M for a plate.
    #
    # This is why a riser next to the pour cup looks wrong and IS wrong: two
    # hot spots either side of one plate are the same continuous body of metal,
    # and one feeder fed through the plate reaches both. Only a thermal centre
    # genuinely beyond that reach -- an isolated boss, a separate arm -- needs
    # its own riser.
    reach = cfg.feeding_distance_factor * float(part.modulus)
    claims = [(np.asarray(pt, dtype=np.float64)[:2], float(r) + reach)
              for pt, r in (feeders or ())]

    index = 0
    for i in range(len(hotspots)):
        hs_world = hotspots.world[i]
        m_local = float(hotspots.local_modulus[i])

        spot_xy = np.asarray(hs_world, dtype=np.float64)[:2]
        if any(float(np.linalg.norm(spot_xy - cxy)) <= cr for cxy, cr in claims):
            continue            # within reach of a feeder that already exists

        # Walk up from the hot spot to the top surface of the casting.
        origin = np.array([[hs_world[0], hs_world[1], hs_world[2]]])
        try:
            locs, _, _ = ray.intersects_location(origin, np.array([[0.0, 0.0, 1.0]]))
            z_top = float(locs[:, 2].max()) if len(locs) else hi_z
        except Exception:
            z_top = hi_z
        centre = np.array([hs_world[0], hs_world[1], z_top])

        r = size_riser(m_local, cfg, index=index, centre=centre)
        r.meta["hotspot_world"] = hs_world.tolist()
        r.meta["hotspot_edt"] = float(hotspots.edt[i])
        risers.append(r)
        index += 1

    return risers
