"""Process configuration and alloy presets.

Units throughout: millimetres, seconds, kilograms. Densities are kg/mm^3 so
that ``density * volume_mm3`` is a mass in kg.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

# density kg/mm^3, pour temperature degC, typical fluidity-driven fill time scale
ALLOYS: dict[str, dict] = {
    "A356":    {"density": 2.68e-6, "pour_temp": 700.0,  "label": "A356 aluminium"},
    "AlSi7Mg": {"density": 2.68e-6, "pour_temp": 700.0,  "label": "AlSi7Mg aluminium"},
    "bronze":  {"density": 8.80e-6, "pour_temp": 1150.0, "label": "C90300 tin bronze"},
    "brass":   {"density": 8.50e-6, "pour_temp": 1000.0, "label": "yellow brass"},
    "steel":   {"density": 7.85e-6, "pour_temp": 1600.0, "label": "carbon steel"},
    "316L":    {"density": 7.98e-6, "pour_temp": 1580.0, "label": "316L stainless"},
    "IN718":   {"density": 8.19e-6, "pour_temp": 1450.0, "label": "Inconel 718"},
    "Ti64":    {"density": 4.43e-6, "pour_temp": 1700.0, "label": "Ti-6Al-4V"},
}

GRAVITY = 9806.65  # mm/s^2

# Linear firing shrinkage of an investment shell, as a fraction, against firing
# temperature in degC. An investment shell is built green around the pattern
# then fired to burn the pattern out and sinter the ceramic; it contracts while
# it does so, and the fired mould is what the casting is actually made in.
#
# Shrinkage is a property of the CERAMIC SYSTEM and its firing schedule alone.
# The alloy poured in afterwards has no bearing on it, which is why no alloy
# term appears anywhere in this table.
#
# The curves are the published behaviour of each system: fused silica is nearly
# inert until cristobalite conversion begins above ~1200 C, zircon is stable
# and low-shrinking throughout, and alumina sinters hard and keeps densifying
# to well over 1500 C. Values between listed temperatures are interpolated
# linearly; outside the range the nearest endpoint is held, since extrapolating
# a sintering curve past its measured span is not defensible.
CERAMICS: dict[str, dict] = {
    "fused_silica": {
        "label": "fused silica",
        "curve": [(900, 0.0010), (1000, 0.0015), (1100, 0.0022),
                  (1200, 0.0030), (1300, 0.0040), (1400, 0.0058),
                  (1500, 0.0080)],
    },
    "zircon": {
        "label": "zircon / fused silica",
        "curve": [(900, 0.0008), (1000, 0.0012), (1100, 0.0018),
                  (1200, 0.0025), (1300, 0.0033), (1400, 0.0045),
                  (1500, 0.0060)],
    },
    "alumina": {
        "label": "alumino-silicate",
        "curve": [(900, 0.0015), (1000, 0.0025), (1100, 0.0040),
                  (1200, 0.0060), (1300, 0.0085), (1400, 0.0120),
                  (1500, 0.0165)],
    },
}

# Holding longer at temperature continues the sintering that drives shrinkage,
# but with rapidly diminishing return -- most of the contraction happens in the
# first hour or two. Modelled as a multiplier on the curve value, normalised so
# the standard 2 h hold is 1.0.
HOLD_REFERENCE_HOURS = 2.0


def hold_factor(hours: float) -> float:
    """Multiplier on the tabulated shrinkage for a non-standard hold time.

    Sintering shrinkage follows roughly t^(1/4) once the ceramic is at
    temperature: doubling a 2 h hold adds about 19%, not 100%. Normalised so
    the 2 h reference returns exactly 1.0.
    """
    h = max(float(hours), 0.05)
    return float((h / HOLD_REFERENCE_HOURS) ** 0.25)


def shrinkage_for(ceramic: str, temperature: float, hold_hours: float = HOLD_REFERENCE_HOURS) -> float:
    """Linear firing shrinkage as a fraction, from the ceramic's own curve.

    This is the point of taking a firing temperature at all: shrinkage is not
    something the user should have to look up and type, it is what the schedule
    produces. Temperature and hold time are the knobs a foundry actually sets.
    """
    spec = CERAMICS.get(ceramic)
    if spec is None:
        raise ValueError(
            f"unknown ceramic {ceramic!r}; use one of {sorted(CERAMICS)}")
    curve = spec["curve"]
    t = float(temperature)
    if t <= curve[0][0]:
        base = curve[0][1]
    elif t >= curve[-1][0]:
        base = curve[-1][1]
    else:
        base = curve[-1][1]
        for (t0, s0), (t1, s1) in zip(curve, curve[1:]):
            if t0 <= t <= t1:
                base = s0 + (s1 - s0) * (t - t0) / (t1 - t0)
                break
    return float(base * hold_factor(hold_hours))


@dataclass
class Config:
    # --- alloy / process ---
    # The alloy is carried for reporting -- pour mass, and what the casting is
    # made of -- not because the mould geometry depends on it. It does not: the
    # sprue is sized by the modulus it must out-freeze, which is geometry, and
    # the choke area formula divides by density so the alloy cancels out of it
    # anyway.
    alloy: str = "A356"
    density: float | None = None      # kg/mm^3; None -> from alloy preset
    gating_ratio: tuple[float, float, float] = (1.0, 2.0, 2.0)  # choke:runner:ingate
    sprue_taper: float = 1.25         # top radius / bottom radius of the sprue
    sprue_height: float = 30.0        # mm, sprue length above the casting
    cup_height: float = 18.0          # mm, pour cup depth
    cup_radius_factor: float = 2.6    # cup radius / sprue top radius
    runner_clearance: float = 6.0     # mm gap between part underside and runner axis
    ingate_count: int | None = None   # None -> one ingate per riser/hot spot

    # --- multi-cavity ---
    # How many copies of the part this mould carries. A production investment
    # shell rarely runs one casting: identical patterns are clustered around
    # one sprue and poured together, which divides the sprue metal, the ceramic
    # and the furnace time between them. quantity=1 keeps the single-part
    # behaviour exactly.
    quantity: int = 1
    layout: str = "grid"              # 'grid' raster | 'radial' wax-tree cluster
    layout_columns: int | None = None # grid: instances per row; None -> squarest
    part_spacing: float | None = None # mm clear gap; None -> 2*shell + 2*pitch
    runner_clearance_multi: float = 8.0  # mm from instance top to the runner bar
    # Hand-made arrangement from the viewer's gizmo: one dict per instance with
    # x/y/z offsets in mm and rot in degrees about +Z. When present it replaces
    # the automatic layout entirely -- the raster and the circle answer "where
    # should these go", and this answers "the user already said".
    placements: list | None = None

    # --- risers ---
    riser_modulus_factor: float = 1.2
    riser_hd_ratio: float = 1.5       # riser height / diameter
    riser_neck_factor: float = 0.55   # neck diameter / riser diameter
    # Separate risers are off by default. On most parts the sprue already
    # feeds every thermal centre within reach, so a riser adds metal to remelt
    # and a second appendage to cut off for nothing. Turn it on for a casting
    # with an isolated heavy section the sprue genuinely cannot reach.
    use_risers: bool = False
    max_risers: int = 4
    # How far a feeder reaches through the section it feeds, as a multiple of
    # the casting modulus. A feeder serves more than the metal directly beneath
    # it: liquid flows sideways through the section while it solidifies. For a
    # plate the accepted reach is ~4.5x wall thickness, which in modulus terms
    # is ~9x M. Raise it to feed further from one sprue, lower it to force more
    # risers.
    feeding_distance_factor: float = 9.0
    min_riser_diameter: float = 6.0   # mm floor so risers stay printable/voxelisable

    # --- hot spots ---
    hotspot_rel_threshold: float = 0.55   # fraction of peak EDT to qualify
    hotspot_min_separation: float = 8.0   # mm non-maximum-suppression radius

    # --- shell firing ---
    # An investment shell is built green, then fired (typically 950-1300 C) to
    # burn out the pattern and develop strength. The ceramic contracts during
    # that firing, so the fired mould is slightly smaller than the one that was
    # built. This is a property of the ceramic system alone -- the alloy that
    # will later be poured into it has no bearing on it.
    ceramic: str = "fused_silica"     # shell system; sets the shrinkage curve
    fire_temperature: float = 1300.0  # degC, shell firing temperature
    fire_hold_hours: float = 2.0      # h at temperature
    # Linear contraction on firing. Left as None it is DERIVED from the ceramic
    # system and the firing schedule above, which is the honest direction: a
    # foundry sets a temperature and a hold, and the shrinkage is what those
    # produce. Set it explicitly to override with measured data for your own
    # slurry.
    fire_shrinkage: float | None = None
    show_fired_shell: bool = True

    # --- shell / discretisation ---
    # Cut the mould cavity with the original mesh instead of its voxels. The
    # outer shell surface stays a voxel offset (nobody measures the outside of
    # a ceramic shell); the cavity is the casting, so it must be exact.
    exact_cavity: bool = True
    shell_thickness: float = 6.0      # mm ceramic skin
    voxel_pitch: float = 1.5          # mm
    grid_pad: int = 3                 # voxels of empty margin around everything
    # Which way is up when the mould is poured, as a world vector in the
    # uploaded file's own frame. Everything downstream assumes +Z is up -- the
    # sprue rises, the cup opens at the top, hot spots are fed from above -- so
    # the part is rotated to put this vector on +Z before the build and rotated
    # back afterwards.
    #
    # It exists because a CAD file's +Z is not reliably the direction the
    # casting is poured. A manifold modelled lying down is 435 mm along X and
    # 195 mm in Z, and gating it "up the file's Z" put the sprue through the
    # side of the part and buried the pour cup inside the casting.
    pour_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Hard ceiling on the lattice. The distance transform allocates several
    # float64 arrays the size of the whole grid, so the peak is roughly
    # 32 bytes per voxel: 25M voxels is about 0.8 GB, which a laptop handles.
    # Past this the build is refused with a suggested pitch rather than left to
    # exhaust the machine's memory and disk.
    max_voxels: int = 25_000_000

    # --- output ---
    write_section_png: bool = True
    write_tree_stl: bool = True
    section_axis: int = 0             # 0=x, 1=y -> plane normal for the PNG
    units: str = "mm"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.alloy not in ALLOYS and self.density is None:
            raise ValueError(
                f"unknown alloy {self.alloy!r}; pass density= explicitly or use one of {sorted(ALLOYS)}"
            )
        if self.density is None:
            self.density = ALLOYS[self.alloy]["density"]
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.voxel_pitch <= 0:
            raise ValueError("voxel_pitch must be positive")
        if self.shell_thickness <= 0:
            raise ValueError("shell_thickness must be positive")
        if self.ceramic not in CERAMICS:
            raise ValueError(
                f"unknown ceramic {self.ceramic!r}; use one of {sorted(CERAMICS)}")
        if self.fire_temperature <= 0:
            raise ValueError("fire_temperature must be positive")
        if self.fire_hold_hours <= 0:
            raise ValueError("fire_hold_hours must be positive")
        if self.fire_shrinkage is None:
            self.fire_shrinkage = shrinkage_for(
                self.ceramic, self.fire_temperature, self.fire_hold_hours)
        if not -0.05 < self.fire_shrinkage < 0.05:
            raise ValueError("fire_shrinkage must be a small fraction, e.g. 0.004 for 0.4%")
        if len(self.gating_ratio) != 3:
            raise ValueError("gating_ratio must be (choke, runner, ingate)")
        if self.riser_modulus_factor < 1.0:
            raise ValueError("riser_modulus_factor must be >= 1.0")
        if int(self.quantity) < 1:
            raise ValueError("quantity must be >= 1")
        self.quantity = int(self.quantity)
        if self.layout not in ("grid", "radial"):
            raise ValueError("layout must be 'grid' or 'radial'")
        if self.layout_columns is not None and int(self.layout_columns) < 1:
            raise ValueError("layout_columns must be >= 1 when given")
        if self.part_spacing is not None and self.part_spacing < 0:
            raise ValueError("part_spacing must be non-negative")
        up = np.asarray(self.pour_up, dtype=float).reshape(3)
        if float(np.linalg.norm(up)) < 1e-9:
            raise ValueError("pour_up must be a non-zero direction")
        self.pour_up = tuple(float(v) for v in up / np.linalg.norm(up))
        if self.placements is not None:
            if not isinstance(self.placements, (list, tuple)) or not self.placements:
                raise ValueError("placements must be a non-empty list when given")
            self.placements = [dict(p) for p in self.placements]
            # the arrangement defines how many parts there are
            self.quantity = len(self.placements)
        self.gating_ratio = tuple(float(x) for x in self.gating_ratio)

    # -- resolution helpers ---------------------------------------------
    @property
    def alloy_label(self) -> str:
        return ALLOYS.get(self.alloy, {}).get("label", self.alloy)

    @property
    def ceramic_label(self) -> str:
        return CERAMICS.get(self.ceramic, {}).get("label", self.ceramic)

    @property
    def fired_scale(self) -> float:
        """Linear scale factor from the green shell to the fired shell."""
        return 1.0 - self.fire_shrinkage

    @property
    def shell_voxels(self) -> float:
        """Shell thickness in continuous voxel units."""
        return self.shell_thickness / self.voxel_pitch

    def resolution_warnings(self) -> list[str]:
        """Discretisation problems worth telling the user about."""
        warn = []
        if self.shell_voxels < 2.0:
            warn.append(
                f"shell_thickness {self.shell_thickness}mm is only "
                f"{self.shell_voxels:.1f} voxels at pitch {self.voxel_pitch}mm; "
                "reduce voxel_pitch for an accurate wall"
            )
        return warn

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gating_ratio"] = list(self.gating_ratio)
        d["alloy_label"] = self.alloy_label
        d["ceramic_label"] = self.ceramic_label
        return d
