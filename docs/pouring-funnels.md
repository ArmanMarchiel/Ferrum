# Pouring funnels: the gating tree

How `cad2shell` decides what metal-carrying appendage to hang off a casting,
and where every millimetre of it comes from. This is stages 3 and 4 of the
pipeline — `gating.py` and `risers.py` — plus the primitives (`primitives.py`)
that turn the result into voxels.

Everything here is deterministic. No fitting, no sampling, no ML: each
dimension is a closed-form function of the part's geometry and the `Config`.
If you change a number in this document you should be able to point at the
line of Python that produced it.

---

## Anatomy

There are two schemes, and which one you get depends only on `cfg.quantity`.

### Single-part (`cfg.quantity == 1`) — `gating.design`

```
pour cup  →  tapered sprue  →  top surface of the casting
```

That is the whole tree. There is **no runner and no ingates**: `design` emits
exactly two `Segment`s, `"sprue"` and `"cup"`, and
`test_gating_is_a_single_attached_appendage` (`tests/test_metallurgy.py`) pins
that `set(kinds) == {"sprue", "cup"}`, `n_ingates == 0` and
`runner_radius == 0.0`.

The docstring on `design` says why. An earlier layout parked the sprue beside
the part and ran a runner the length of the casting underneath it — a
horizontal-plate scheme for a sand mould with a parting line. On a 165 mm part
that produced a 174 mm runner and a tree larger than the casting it fed, none of
it touching the part. An investment shell is poured as one assembly hanging from
one sprue, so the gating is one appendage attached directly to the casting.

**Pour cup.** A short truncated cone from the sprue top upward, radius
`r_top → r_top * cfg.cup_radius_factor` over `cfg.cup_height`. It carries
`meta["open_to_world"] = True`, which is the flag `synthesis.open_pour_cup`
looks for: it clears all shell above the highest cup endpoint so the mould has
a mouth. Without that the ceramic skin closes over the top and you get a
watertight but unpourable void.

**Sprue.** Rises from the gate point on the casting to `z_sprue_top`, widening
as it goes (`r_top = r_base * cfg.sprue_taper`, default 1.25). The choke — the
narrowest section — is therefore at the **bottom**, flagged in the segment meta
as `"choke_at": "p0"`. A falling stream contracts naturally; matching that
contraction keeps the metal against the sprue wall instead of letting it break
away and aspirate air into the mould. `test_sprue_tapers_downward` and
`test_sprue_widens_toward_the_cup` assert `sprue_r_top > sprue_r_bottom`.

**In this scheme the sprue is also the feeder.** That is the single most
important thing to understand about this module — see *Sizing*, below.

### Multi-cavity (`cfg.quantity > 1`) — `gating.design_multi`

```
pour cup  →  central sprue  →  one runner per instance  →  one ingate per instance
```

Dropping a sprue on each of N instances would produce N separate moulds that
happen to share a bounding box, with nothing tying them into one pourable
assembly. So the multi-cavity tree is the classic three-level one. The sprue
stands on the cluster's plan centroid (`centres[:, :2].mean(axis=0)`), clear of
every instance, and runs down to `z_runner = z_top + cfg.runner_clearance_multi`
— *above* the instance tops, so metal falls into each cavity rather than having
to climb. Runners radiate from there to a point over each part and a short
ingate drops onto it.

Radiating spokes, not a shared bar: a bar cannot give equal-length feed paths to
instances at different distances, and equal path length is what makes N castings
from one pour come out alike.

### Risers — `risers.design`

A riser is a reservoir of metal that freezes *after* the section it feeds, so
that section draws liquid from the riser instead of tearing a shrinkage void in
itself. In this codebase risers are a **top-feeding cylinder plus a neck**,
stamped as two segments (`Riser.segments()`: `"neck"` then `"riser"`).

They are **off by default** (`cfg.use_risers = False`), and that is a deliberate
metallurgical claim, not a convenience. Because the sprue is already sized as a
feeder, most parts need nothing else; an extra riser is metal to remelt and a
second appendage to saw off. `test_risers_are_off_by_default` and
`test_a_compact_part_needs_only_the_pour_appendage` hold this down.

---

## The formulas actually implemented

### Riser sizing: modulus method

Cooling modulus `M = V / A`. Chvorinov's rule makes freezing time proportional
to `M²`, so a riser needs `M_riser ≥ f · M_section`, with `f = 1.2` by
convention — which buys roughly 44% more freezing time (the `risers.py` module
docstring spells this out).

For a cylindrical top riser of diameter D and height H standing on the casting,
so its base face is not a cooling surface (`risers.cylinder_modulus`):

```
V = π D² H / 4
A = π D H  (side)  +  π D² / 4  (top)
M = V / A = D H / (4H + D)
```

With `H = k·D` (`cfg.riser_hd_ratio`, default 1.5) this inverts in closed form
(`risers.diameter_for_modulus`):

```
M = k D² / (4k D + D) = k D / (4k + 1)
⇒  D = M (4k + 1) / k
```

Closed form is the point: the sizing is exact and the invariant is exactly
testable. `test_modulus_inversion_is_exact` checks the round trip to `rel=1e-9`
for several targets and ratios, and `test_riser_modulus_formula_matches_geometry`
re-derives `D·H/(4H+D)` from the reported D and H of every riser in every run.

`risers.size_riser` then applies the diameter floor `cfg.min_riser_diameter`
(6 mm, so risers stay printable and voxelisable). If the floor binds, height is
**recomputed** rather than left short:

```
M = D H / (4H + D)  ⇒  H = M D / (D − 4M),  valid while D > 4M
```

`test_riser_diameter_floor_over_feeds_never_under_feeds` pins that the floor can
only over-feed. Neck: `neck_d = max(cfg.riser_neck_factor * d, 2.0)` (0.55 · D),
`neck_length = max(0.4 * neck_d, 2.0)`.

### Sprue base: the same formula, applied to the sprue

Both `design` and `design_multi` size the sprue base by importing
`diameter_for_modulus` from `risers`:

```python
m_feed = float(np.max(hotspots.local_modulus)) if (
    hotspots is not None and len(hotspots)) else part.modulus
r_feeder = diameter_for_modulus(
    cfg.riser_modulus_factor * m_feed, cfg.riser_hd_ratio) / 2.0
r_base = r_feeder
```

The sprue is the last liquid metal in the system. It has to stay molten until
the section it lands on has frozen, or it cannot feed that section's shrinkage.
So its base is exactly the diameter a riser would need over that section.
`test_sprue_base_can_feed_the_section_it_lands_on` recomputes `d_needed` from the
hot spot moduli and asserts the realised base meets it.

`choke_area = π · r_base²` and `choke_radius == sprue_r_bottom` follow from that,
not from any flow calculation.

### The Bernoulli choke — deliberately removed

The README's pipeline table still advertises
`A_choke = W / (ρ·t·Cd·√(2gH))`. **That formula is not in the code any more.**
There is a block comment in `gating.py`, immediately after `GatingPlan`, headed
*"Note on what is NOT here any more"*, which says: the arithmetic was correct but
it was answering the wrong question for this scheme. The sprue doubles as the
feeder, the modulus requirement is always the larger of the two, and the
fill-rate choke never won on any part tested — which made `pour_time`, `Cd`, the
alloy density and `head_height` inputs that changed nothing in the output. They
were removed rather than left on screen implying otherwise.

Consequences a new developer should not trip over:

- There is **no `pour_time`, `Cd`, `head_height` or `riser_factor`** in
  `Config`. Do not go looking. The riser factor is spelled
  `cfg.riser_modulus_factor`.
- `GRAVITY = 9806.65  # mm/s²` still exists in `config.py` and is still
  imported by `gating.py`, but **nothing reads it**. It and the archived
  formula are where a fill-limited choke would go back if a scheme ever needed
  one — and the comment is explicit that it should come back *with a test that
  shows it changing the geometry*.
- Alloy **density does not affect any dimension**. `Config.alloy` is carried for
  reporting — pour mass, what the casting is made of — and the docstring on the
  field says so. The alloy table in `config.py` is `density` (kg/mm³) and
  `pour_temp` only.
- `test_the_sprue_is_sized_by_modulus_not_fill_rate` exists specifically to
  catch a fill-rate term reappearing without changing the geometry.

### Gating ratio (multi-cavity only)

`cfg.gating_ratio = (choke, runner, ingate)`, default `(1.0, 2.0, 2.0)` — a
non-pressurised 1:2:2 system. Applied in `design_multi` to the *total* flow:

```python
total_runner_area = a_choke_real * (ratio_r / ratio_c)
total_ingate_area = a_choke_real * (ratio_g / ratio_c)
per_ingate_area   = total_ingate_area / n      # n = instances
per_runner_area   = total_runner_area / n_branches
r_ingate = sqrt(per_ingate_area / π)
r_runner = sqrt(per_runner_area / π)
```

Non-pressurised means total area increases away from the choke, keeping the
runner and ingates full but unpressurised so the metal enters the cavity slowly.
On the single-part path `cfg.gating_ratio` is unused — there is nothing to
apportion.

### Where the sprue lands: `gate_point`

Directly above the heaviest hot spot, because the thickest section freezes last
and is where shrinkage porosity forms if it is not fed. Three refinements, each
with a failure it was written for:

1. **Ties.** `argmax` over hot-spot moduli breaks ties arbitrarily, so a
   symmetric casting's mirrored pair put the gate on whichever came first.
   Everything within `_TIE_TOLERANCE = 0.05` (5%) of the peak counts as tied and
   the gate goes to their mean.
2. **Symmetry snap** (`_snap_to_symmetry`). Averaging tied spots is still not
   enough: two sides of a symmetric feature voxelise to *different* moduli
   (3.94 vs 3.18 for the same boss on a plummer block), so only one side ties
   and the average sits over it. Symmetry is measured on the mesh — mirror the
   part about the candidate plane, boolean-intersect it with itself, require
   ≥98% volume overlap — and only in X and Y, since +Z is the pour direction.
   `test_a_symmetric_casting_is_gated_on_its_centreline` and
   `test_an_asymmetric_casting_keeps_its_thermal_gate` are the pair.
3. **Solid landing** (`_gate_site` → `_solid_top_at`). The gate must land on
   metal. `_solid_top_at` takes the **first** surface a falling stream would
   meet, then requires solid material immediately beneath it, then runs an
   eight-direction probe to distinguish the floor of a bore (walled above on
   every side) from a merely low step (open on at least one side). If the column
   is hollow the search spirals outward over the instance footprint. There is no
   height floor — the boss_plate's plate is 25 mm below its boss and is a
   perfectly good landing face.

Sprue top: `z_sprue_top = max(gate_z + cfg.sprue_height, hi_z + cfg.sprue_height)`.
Measuring from the top of the casting rather than from the gate alone is what
keeps the cup outside the part's silhouette on a tall part — otherwise
`open_pour_cup` raises (see *Failure modes*).

### Which hot spots get risers: `risers.design`

A hot spot within feeding distance of an existing feeder gets **no** riser:

```python
reach = cfg.feeding_distance_factor * float(part.modulus)
claims = [(pt_xy, r + reach) for pt, r in feeders]
```

`cfg.feeding_distance_factor = 9.0`. The reasoning is in the comment: a feeder
does not only serve the metal directly beneath it — liquid flows sideways
through the section while it solidifies, and the accepted reach for a plate is
about 4.5× wall thickness from the feeder edge (≈2× as feeding distance proper
plus a 2.5× end effect). In modulus terms, ≈9 × M for a plate. Raise it to feed
further from one sprue; lower it to force more risers.

For each surviving hot spot the riser stands on the part's top surface directly
above the thermal centre (upward ray cast, falling back to the part's top Z),
and is sized from that hot spot's **local** modulus, not the casting's global
V/A, so a part mixing thick and thin sections gets each riser matched to what it
actually feeds.

---

## Parameters that drive dimensions

| parameter | default | what it moves |
|---|---|---|
| `riser_modulus_factor` | 1.2 | the feeding safety factor — scales the sprue base **and** every riser diameter |
| `riser_hd_ratio` | 1.5 | riser H/D; enters `D = M(4k+1)/k`, so it changes the sprue base too |
| `riser_neck_factor` | 0.55 | neck diameter / riser diameter |
| `min_riser_diameter` | 6.0 mm | floor; when it binds, height is recomputed so feeding is still met |
| `use_risers` | `False` | whether separate risers are placed at all |
| `max_risers` | 4 | hot-spot budget **per casting**, applied in `hotspots.detect` as `max_risers × instances` |
| `feeding_distance_factor` | 9.0 | how far an existing feeder reaches, as a multiple of `part.modulus` |
| `sprue_taper` | 1.25 | `r_top / r_base` |
| `sprue_height` | 30.0 mm | sprue length above the casting |
| `cup_height` / `cup_radius_factor` | 18.0 mm / 2.6 | cup depth and mouth radius |
| `gating_ratio` | (1, 2, 2) | runner and ingate areas — **multi-cavity only** |
| `runner_clearance_multi` | 8.0 mm | instance top → runner bar height |
| `voxel_pitch` | 1.5 mm | junction overlaps and probe tolerances scale off it |
| `density` / `alloy` | A356, 2.68e-6 kg/mm³ | **reporting only** — pour mass; no geometry depends on it |
| `GRAVITY` | 9806.65 mm/s² | currently unused; kept for a future fill-limited choke |

Note the coupling: `riser_modulus_factor` and `riser_hd_ratio` are not
riser-only knobs. Because `gating.design` calls `diameter_for_modulus` to size
the sprue base, they resize the sprue, the cup and the choke area as well.

`max_risers` is a budget on **hot spots**, not on risers — it is consumed in
`hotspots.detect`'s non-maximum suppression, and it is multiplied by the
instance count so a four-up mould does not silently starve whichever instances
were labelled last.

---

## How the tree connects to the part, and what must hold

Ordering matters and is not obvious from the stage numbers. In `pipeline.run`,
**stage 3 (gating) runs before stage 4 (risers)**, because the sprue claims a
hot spot by feeding it:

```python
plan = gating.design(part, cfg, risers=None, hotspots=spots)      # or design_multi
if cfg.use_risers:
    feeders = [(sg.p1, sg.r1) for sg in plan.segments if sg.kind == "ingate"]
    if not feeders:
        feeders = [(sg.p0, sg.r0) for sg in plan.segments if sg.kind == "sprue"]
    riser_list = risers.design(spots, part, cfg, feeders=feeders)
```

The `feeders` argument is every point where liquid metal actually **enters the
casting**. On a multi-cavity mould that is the ingates, one per instance — *not*
the central sprue, which touches no casting at all. Measuring reach from the
central sprue made a riser depend on where an instance happened to sit in the
cluster, so four identical parts came out with two risers, one, one and none.
The `design(..., feeders=...)` signature exists to make that bug hard to
reintroduce.

Rasterisation goes through `primitives.stamp_capsule`: every gating element —
tapered sprue, constant-radius runner and ingate, riser body and neck — is the
same shape, a tapered capsule. Ends are **flat** by default (the solid is cut at
both end planes, not a hemispherical capsule), because a hemispherical cap would
push the sprue's narrowest cross-section half a bore below the plane where the
design says the choke is. `round_ends=True` is used for riser necks, where a
fillet at the casting junction is desirable, and for multi-cavity ingates via
`meta["round_ends"]`.

All world↔index arithmetic goes through `Grid`, so a segment lands on precisely
the cells the hot-spot stage analysed.

### Invariants (`tests/test_connectivity.py`)

- **`test_metal_path_is_single_connected_component`** — part ∪ sprue ∪ runner ∪
  ingates ∪ risers must label as exactly one component under
  `generate_binary_structure(3, 1)`. Faces only, because metal flows through
  faces, not through a shared edge or corner. A second component means metal
  poured into the cup can never reach it and the casting comes out short.
- **`test_every_riser_is_attached_to_the_casting`** — each riser is stamped in
  isolation, OR-ed with the part occupancy, and must label as one component.
- **`test_shell_fully_separates_metal_from_the_outside`** — flood the free space
  from the grid corner, dilate by one voxel, and the only metal it may touch is
  at or above `diagnostics["cup"]["mouth_z"]` (within two voxels). Anywhere else
  is a hole the mould would leak through.

Junction geometry is what makes the first of these survive export. In
`design_multi`, runners are extended by `overlap = max(0.5 * pitch, 0.25 * r_ingate)`
past both the sprue tap and the ingate axis, and successive runners leave the
sprue at staggered heights (`stagger = 0.25 * cfg.voxel_pitch`). A cutter that
*ends* exactly on another's surface touches it tangentially, and the cavity
boolean answers tangential contact with zero-area triangles — harmless in memory,
but they weld into the surface when the STL is re-read, putting four faces on an
edge that should carry two. The shell then reads as non-watertight despite having
no actual hole.

---

## Gotchas and failure modes

**The README is stale on the choke.** It still lists
`A_choke = W / (ρ·t·Cd·√(2gH))` as what stage 3 does, and mentions a 1:2:2
non-pressurised ratio as though it always applies. Neither is true of the
single-part path. Trust `gating.py`.

**`GRAVITY` is imported but dead.** `from .config import GRAVITY` at the top of
`gating.py` has no remaining reader. Do not infer from the import that a
Bernoulli calculation is happening.

**Changing a riser knob resizes the sprue.** `riser_modulus_factor` and
`riser_hd_ratio` feed `diameter_for_modulus`, which sizes the sprue base. There
is no separate sprue-sizing parameter.

**A chunky part genuinely gets a huge sprue.** The modulus method is honest
about this: `D = M(4k+1)/k` with k=1.5 is `D ≈ 4.67 M`, and with the 1.2 factor
`D ≈ 5.6 × M_section`. Real practice would add a chill instead. There are no
chills in this codebase.

**Sealed mould.** `synthesis.open_pour_cup` raises `ValueError` when the cup
mouth is at or below the top of the metal — "the pour cup sits at z=…, inside
the casting". The usual cause is `cfg.pour_up`: a CAD file's +Z is not reliably
the pour direction, and gating "up the file's Z" on a part modelled lying down
puts the sprue through the side of the casting. Set `pour_up` to the real pour
direction.

**Nowhere to gate.** `design_multi` raises
`"instance {i} has no solid surface to gate onto"` when `_gate_site` finds no
solid column anywhere in an instance's footprint — a shell-like part, or a
layout that placed the instance badly. `_solid_top_at` returning `None` for a
bore is by design; it is what stops an ingate pouring down a hole.

**Search radii scale to one instance, not the cluster.** `_gate_site` takes
`instance_span` from `part.mesh.split(...)` bodies. Passing the cluster's
extents makes the first probe ring land in the empty space between widely spaced
parts, and the instance is then reported as having no surface at all. Same trap
inside `_solid_top_at`'s bore detection.

**Probe angles are in the instance's own frame.** `_gate_site` takes
`instance_angle` and phases its search by it, because a radial layout rotates
every instance and probing along world axes would land each one on a different
feature of the same part.

**Ingate site ranking is nearest-the-sprue, not highest.** Height is a tolerance
(`max(cfg.shell_thickness, 2 * voxel_pitch)` below the best face), and among
acceptable faces the shortest feed wins, with the probe index breaking remaining
ties identically for every instance. Ranking by height first made every runner
cross the whole cluster to reach its instance's far lobe — 116 mm each on a
four-up plummer block where half would do.

**Local modulus, not EDT.** Risers are sized from `hotspots.local_modulus`,
which measures true V/A from voxel face counts. Reading modulus off the distance
transform oversizes risers 2–3× (`M = d` for a slab, but `R/2` for a cylinder
and `R/3` for a sphere). If you add a sizing path, take `local_modulus`.

**Curved surfaces voxelise low.** `component_modulus` reads ~33% low on a
sphere, so a curved heavy section will be under-fed at a coarse pitch. Reduce
`--pitch` to converge.

## Open questions

- `add_ingates` is a no-op retained for API compatibility, but
  `tests/test_connectivity.py::_rebuild` still calls it with computed `targets`
  that go nowhere. Harmless, but the targets computation there is dead weight.
- `GatingPlan.as_dict` hardcodes `"ratio": None`; `pipeline.run` overwrites it
  with `list(cfg.gating_ratio)` when it assembles the stats. Reading the plan
  directly rather than the stats will show `None`.
