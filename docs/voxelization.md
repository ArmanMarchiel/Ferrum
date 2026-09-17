# Voxelization

How the mesh becomes a lattice, who owns the lattice, and what every other
stage is allowed to assume about it.

Source: `voxel.py` (the frame), `hotspots.py` (occupancy + distance field),
`primitives.py` (stamping), `synthesis.py` (the offset back out to a mesh), all
under `backend/src/cad2shell/`.

## Why there is a grid at all

The pipeline's job is `mesh -> grid -> shell`. The shell is an **offset
solid**: everything within `shell_thickness` of the metal, minus the metal
itself (`synthesis.build_shell`). That is a morphological operation, and the
honest way to compute it on arbitrary user CAD is on a raster, not on triangles.

Two more things want a raster for the same reason. **Hot spots** are local
maxima of the distance-to-surface field, and `distance_transform_edt` needs an
array. **Modulus** — `hotspots.component_modulus` measures true V/A by counting
voxel faces that touch non-casting space, which *is* the definition of a cooling
surface, and reproduces the analytic V/A exactly on flat-faced solids.

What does **not** go through the grid, and deliberately so: the cavity surface
and the tree. With `exact_cavity=True` (the default) the mould's inner surface
is cut with the original mesh and analytic cones
(`synthesis.exact_cavity`, `synthesis.exact_tree`), because that surface *is*
the casting and every voxel of quantisation on it becomes a defect on the part.
The voxel grid supplies the **outer** shell surface, the thermal analysis, and
the connectivity — never the finished casting surface. See
[Where the grid stops](#where-the-grid-stops).

## The Grid abstraction

`voxel.Grid` is a frozen dataclass of `origin`, `pitch`, `shape`. It owns
**all** world↔index arithmetic; no other stage does the conversion itself. That
single rule is what keeps hot-spot detection and riser stamping on one frame.

### Coordinate convention

`origin` is the world position of the **corner** of voxel `[0,0,0]`. Voxel
`[i,j,k]` spans `origin + [i,j,k]*pitch` to `origin + [i+1,j+1,k+1]*pitch`, and
its centre is half a pitch in. The lattice is axis-aligned and isotropic — one
scalar `pitch` for all three axes, no rotation.

```python
from cad2shell.voxel import Grid

g = Grid.from_bounds([-10, -5, 0], [10, 5, 20], pitch=1.5, pad_voxels=2)
g.to_index([[3.7, 3.7, 3.7]])   # -> floor((p - origin) / pitch)
g.to_world([[3, 3, 3]])         # -> voxel CENTRE, origin + (idx + 0.5)*pitch
g.contains(idx)                 # bounds check, per row
g.mm(2.0)                       # voxels -> mm
g.voxels(6.0)                   # mm -> continuous voxel units (not rounded)
```

Note the asymmetry: `to_index` takes any world point and floors it to the cell
containing it; `to_world` returns the **centre**, not the corner. The pair
round-trips exactly in one direction — `to_index(to_world(i)) == i` for every
voxel — and `test_index_world_roundtrip_is_exact` in
`backend/tests/test_frame.py` sweeps a whole grid to prove it.

### floor, never round

`to_index` floors. This is not a style preference. A point 3.7 voxels along
belongs to cell 3; rounding would send it to cell 4 — a cell it is not inside.
Two stages that disagree by one cell punch pinholes in thin walls.
`test_mapping_uses_floor_not_round` pins it.

### One lattice, integer offsets

`Grid.from_bounds` snaps the origin to a global multiple of `pitch`:

```python
origin = np.floor(lo / pitch - pad_voxels) * pitch
```

so any two grids built this way at the same pitch share one lattice. Growth is
by **whole voxels** through `expanded(pad_lo, pad_hi)`, which preserves pitch
and alignment, and `offset_in(other)` returns the integer index offset between
two aligned grids — raising if the pitches differ or the origins are not an
integer number of voxels apart.

`pipeline.run` uses exactly this. `grid_part` covers the casting;
`grid_full` (called `grid` in the code) is `grid_part.expanded(pad_lo, pad_hi)`
sized from `_tree_bounds`, so it covers the casting *plus* the gating tree and
shares the casting's lattice by construction. The pipeline then asserts it:

```python
offset = grid_part.offset_in(grid)   # raises if the frames drifted
```

The offset is reported as `stats.grid["part_grid_offset"]`, and
`test_pipeline_keeps_hotspots_and_stamping_on_one_frame` checks it is integral
and non-negative for every case. **Hot spots are detected on `grid` — the same
grid risers are stamped into** — so a riser cannot land a voxel off the thermal
centre it was sized for.

## Occupancy

Two voxelisers live in `hotspots.py` and are required to agree.

`voxelize` is the reference: build every voxel centre, ask
`mesh.contains(centres)`, write back on the Grid's frame. Correct, and slow —
one ray test per cell.

`voxelize_fast` is what the pipeline calls. It uses `mesh.voxelized(pitch).fill()`
only as a **candidate set**, because trimesh's voxeliser marks every voxel the
*surface passes through*, dilating the solid by about half a voxel in every
direction and inflating both the volume and the distance transform. So: erode
the candidates, trust cells whose 6-neighbours are all candidates as strictly
interior, and ray-test only the boundary shell. Anything that throws or produces
nothing falls back to `voxelize`.

If you change one of these, change both — the whole design rests on them
producing the same occupancy.

### The half-voxel bias in the distance field

`distance_transform_edt` returns centre-to-centre distance to the nearest empty
voxel, so a cell one step inside the boundary reports a full pitch when the
surface is really half a pitch away. `hotspots.distance_field` subtracts
`pitch/2` and clamps at zero, which makes the peak match the true half-thickness
of a slab (a 35 mm block reads 17.5, not 18). It also pads the array by one
cell before the transform, because otherwise a part flush to the grid edge reads
as infinitely thick.

## Choosing `voxel_pitch`

`Config.voxel_pitch` defaults to **1.5 mm** (`--pitch` on the CLI,
`voxel_pitch` on `POST /generate`). `grid_pad` (default 3) is the margin in
voxels around everything.

The tradeoff is the usual one, with a cubic cost: memory and time scale as
`pitch⁻³`, accuracy scales roughly linearly. What the pitch actually buys:

| affected by pitch | not affected by pitch |
|---|---|
| outer shell surface (a voxel offset) | cavity surface, when `exact_cavity=True` |
| realised wall thickness quantisation | tree surface, when `exact_cavity=True` |
| EDT / hot-spot placement | riser sizing formula (analytic, `M = D·H/(4H+D)`) |
| `component_modulus` accuracy | choke/sprue base (modulus method, analytic) |
| voxelised part volume, and so `yield_pct` | |

### Rules of thumb the code enforces

- **Shell thickness must be at least ~2 voxels.** `Config.shell_voxels` is
  `shell_thickness / voxel_pitch`, and `resolution_warnings()` complains below
  2.0. A rasterised wall can only be a whole number of cells thick, so the
  quantisation floor is one pitch; the accepted tolerance throughout is
  `voxel_pitch + 0.15 * shell_thickness` (used both as a runtime warning in
  `pipeline.run` and as the assertion in `test_wall_thickness_within_tolerance`).
- **Thin features need a finer pitch.** The test suite picks the pitch per part
  in `backend/tests/conftest.py`:

  ```python
  CASES = {
      "boss_plate":   dict(voxel_pitch=1.5, shell_thickness=6.0),
      "thin_bracket": dict(voxel_pitch=1.0, shell_thickness=4.0),
      "thick_block":  dict(voxel_pitch=1.5, shell_thickness=6.0),
  }
  ```

  The bracket's 5 mm wall needs the finer grid to voxelise faithfully — at
  1.5 mm it is barely three cells through.
- **Refining must never make things worse.** `test_finer_pitch_does_not_worsen_the_wall`
  runs a 5 mm target at pitch 2.0 and 1.0 and asserts the error does not grow.
  That is the monotonicity guarantee you can rely on when tuning.
- **Curved surfaces staircase.** `component_modulus` reads ~33% low on a sphere;
  flat-faced parts are exact. Reducing the pitch converges it.

### The memory ceiling

`Config.max_voxels` defaults to **25M**. The distance transform allocates
several `float64` arrays the size of the whole grid, so the peak is roughly
32 bytes per voxel — 25M voxels is about 0.8 GB, which a laptop handles.

`pipeline.run` **refuses** above the ceiling rather than warning, and the
refusal computes a pitch that will actually work:

```
this part needs a <nx>x<ny>x<nz> grid = 161M voxels at 1.5 mm pitch, over the
25M limit. The distance transform needs several arrays that size and the
machine would run out of memory. Raise the voxel pitch to about <p> mm, or
raise max_voxels if you know the machine can take it.
```

(161M is the real figure for the 435x308x195 mm case in
`test_an_oversized_grid_is_refused_before_it_allocates`.)

This was a warning once. A warning lands in a list the caller only reads when
the job *finishes* — no use when the job is the thing exhausting the machine.
`test_an_oversized_grid_is_refused_before_it_allocates` parses the suggested
pitch out of the message and re-runs with it, so the number in the error is
guaranteed to work. Above `max_voxels // 2` you get a "this will be slow"
warning instead.

## Where the grid feeds downstream

```
ingest -> Grid.from_bounds -> expanded to grid_full
             |
             +-> voxelize_fast(part.mesh, grid)  ==  part_occ  (computed ONCE)
                   |
                   +-> hotspots.detect      EDT, suppressed maxima, local_modulus
                   |     -> HotSpots.idx are indices into THIS grid
                   +-> synthesis.build      metal = part_occ | stamped segments
                         -> shell = dilate(metal, t) & ~metal
                         -> marching cubes -> outer surface
```

`part_occ` is voxelised once in `pipeline.run` and threaded into both
`hotspots.detect(occ=...)` and `synthesis.build(part_occ=...)`. Do not
re-voxelise in a new stage; take the array.

**Hot spots** (`hotspots.detect`) return `idx` (indices into the grid you passed
in), `world` (`grid.to_world(idx)`), `edt`, and `local_modulus`. The modulus is
not the EDT: `local_modulus` anchors on the measured `component_modulus` and
scales each spot by `EDT_here / EDT_peak`, because reading modulus straight off
the distance field over-sizes risers 2–3× on compact features.

**Stamping** (`primitives.stamp_capsule`) rasterises every gating element as a
tapered capsule, evaluating only the segment's index-space bounding box. Ends
are **flat** by default — a hemispherical cap would push the sprue's choke half
a bore below the plane where it actually meets the runner, so the narrowest
cross-section would not be where the design says it is. `round_ends=True` gives
a true capsule, used for riser necks where a fillet is wanted.

**Offsetting** (`synthesis._dilate_mm`) dilates by a true Euclidean distance —
threshold the EDT of the complement — not by a `binary_dilation` iteration
count, which would quantise the shell to whole voxels and over-reach on the
diagonals with a cubic structuring element.

**Meshing** (`synthesis.to_mesh`) pads the mask by one empty voxel so a mask
touching the array border still closes, runs marching cubes at level 0.5, and
maps vertices back with the Grid's own arithmetic:

```python
verts = verts - 1.0                              # undo the pad
verts = grid.origin + (verts + 0.5) * grid.pitch # index space -> world
```

The `+ 0.5` is the same centre convention as `to_world`. Marching-cubes
vertices live in the array's continuous index space where integer `i` is a
sample centre, and a sample centre is a voxel centre.

## Where the grid stops

With `exact_cavity=True`:

- the **outer** shell surface comes from the voxel offset (`grown_only`), meshed
  by marching cubes;
- the **cavity** is cut from that outer solid with the original part mesh and
  analytic cones via `manifold3d`;
- the **tree** is the uploaded mesh unioned with analytic cones.

At 1.5 mm pitch the carved-from-voxels cavity was out by 0.23 mm median and
1.3 mm at p99 — visible ridging in a bore. `test_cavity_accuracy_does_not_depend_on_pitch`
now pins that the cavity error no longer scales with pitch (it used to go
1.31 → 0.65 → 0.30 mm as the pitch halved).

Both boolean paths return `None` on failure so the caller falls back to the
voxel cavity/tree rather than losing the mould; `stats.shell["cavity_exact"]`
and `["tree_exact"]` tell you which path ran.

## Gotchas and failure modes

- **Never do world↔index arithmetic outside `Grid`.** If you need a new mapping,
  add a method. The one place in the codebase that computes indices by hand is
  `voxelize_fast`, mapping into *trimesh's* grid (`src_origin`,
  `src_idx = floor(...)`) — note it still floors, and note it immediately
  re-tests the result against the mesh.
- **`to_world` gives centres, `origin` is a corner.** Mixing them up is a
  half-pitch bias that survives every test that only checks watertightness.
- **Grids at different pitches are not translatable.** `offset_in` raises
  rather than guessing; `test_misaligned_grids_are_rejected` covers the
  misaligned-origin case.
- **Part voxelised to nothing.** `hotspots.detect` raises
  `part voxelised to nothing at pitch Xmm; reduce voxel_pitch`; a non-empty
  occupancy with a zero distance field raises
  `distance field is empty; part is thinner than one voxel`.
- **Silent under-voxelisation.** `pipeline.run` compares voxel volume against
  mesh volume and warns past 10%, and separately warns when the part occupies
  fewer than ~8 voxels. Watch the warnings list; these are the symptoms of a
  pitch that is too coarse for the part, or of a file that is in inches.
- **Yield mixes bases if you are not careful.** `yield_pct` deliberately uses
  the *voxelised* part volume over the voxel metal volume, so numerator and
  denominator share one basis. Substituting `part.volume` can push a thin part
  over 100%.
- **`grid_full` is generous on purpose.** `_tree_bounds` is sized before the
  tree exists, from a worst-case riser guess. An oversized grid is trimmed to
  nothing later; an undersized one clips the sprue.
- **Pour orientation happens before the grid exists.** The mesh is rotated so
  `cfg.pour_up` lands on +Z, the whole build runs in that frame, and the
  finished meshes are rotated back. Everything voxel-side may assume +Z is up.
- **Non-watertight input degrades occupancy.** `mesh.contains` is a ray test;
  if the repair in `ingest` did not close the mesh the pipeline warns
  `voxel occupancy may be unreliable` and carries on. Treat the result as
  suspect.
- **Zero-volume debris is a meshing hazard, not a grid one.** Slivers left by
  the boolean are inert in memory but fuse on STL round-trip and break
  `is_watertight`. `drop_zero_volume_debris` handles it — conservatively — and
  must run *before* export.

## Modifying this stage

Safe: anything inside `voxelize_fast`'s speed path (the fallback guarantees
correctness), the offset method in `_dilate_mm`, the thresholds in `detect`.

Needs care: anything touching `origin`, `pitch`, the floor rule, or
`to_world`'s `+ 0.5`. Run `test_frame.py` first — it is small, fast, and exists
specifically to catch this class of bug — then the full suite:

`pytest.ini` lives in `backend/` and puts `src/` on the path, so run from
there:

```bash
cd backend
../.venv/bin/python -m pytest -q tests/test_frame.py
../.venv/bin/python -m pytest -q
```

### Unused surface

`Grid.bounds`, `Grid.mm()` and `Grid.voxels()` have no callers in `src/` or
`tests/` as of this writing — they are convenience API waiting for a user, not
load-bearing. `Config.shell_voxels` does the mm→voxel conversion the pipeline
actually needs. Do not assume changing them is free for external callers, but
nothing inside the repo depends on them.
