# cad2shell

Part STL in, watertight ceramic **investment-shell mould** wrapped around a full
**gating tree** out. Deterministic — no ML: every dimension traces to a
documented foundry rule.

Open-source stack only: `trimesh`, `scipy`, `scikit-image`, `manifold3d`,
`matplotlib`, FastAPI, three.js. No proprietary CAD dependency.

## Layout

```
backend/         Python: the pipeline, the API and their tests
  cad2shell.py     CLI entry point
  src/
    cad2shell/     the pipeline package
    server/        FastAPI app and the project store
  tests/
  pytest.ini       puts src/ on sys.path
frontend/        React + TypeScript, built with Vite
  src/
    pages/         Home (the filesystem) and Editor (the CAD viewer)
    components/    toolbar, sidebar, viewport, dialogs
    hooks/         three.js scene, undo/redo, job polling
    services/      the API client
    types/         the backend's wire contract
    styles/        styles.css — one stylesheet for both pages
  public/          static assets
docs/            how the pipeline works, stage by stage
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
(cd frontend && npm install)

# CLI (demo parts remain available here for testing)
cd backend
../.venv/bin/python cad2shell.py part.stl -o ../out/part --alloy A356 --shell 6 --pitch 1.5
../.venv/bin/python cad2shell.py --demo boss_plate -o ../out/demo

# backend -> http://127.0.0.1:8000
cd backend/src && ../../.venv/bin/python -m uvicorn server.app:app --reload

# frontend dev server -> http://127.0.0.1:5173 (proxies the API to :8000)
cd frontend && npm run dev

# tests
cd backend && ../.venv/bin/python -m pytest -q
```

From the repo root, `npm run dev`, `npm run cli` and `npm test` wrap the
backend commands above.

## Documentation

| doc | what it covers |
|---|---|
| [docs/voxelization.md](docs/voxelization.md) | the mesh → grid → shell path, the `Grid` frame, choosing `voxel_pitch` |
| [docs/pouring-funnels.md](docs/pouring-funnels.md) | the gating tree: cup, sprue, runners, ingates, risers, and the foundry rules behind each dimension |
| [docs/cad-uploads.md](docs/cad-uploads.md) | accepted formats and why native CAD is not one, the two-phase job model, ingest and persistence |

## Web API

Two-phase by design, because the pipeline is pure CPU and takes seconds to
minutes on a real part:

| call | returns |
|---|---|
| `POST /generate` (multipart STL + params) | immediately: `job_id`, part stats, `urls` |
| `GET /jobs/{id}/status` | `phase` = queued / building / done / error, plus full stats when done |
| `GET /jobs/{id}/{part,shell,tree}.stl` | the meshes |

The pipeline runs on a worker thread. Running it inline blocked the event loop
for the whole build, which made the server stop answering *every* request --
the browser's upload hung forever and nothing reached the log.

## The four visuals

Each is independently toggleable in the viewer:

1. **Original CAD** — the ingested part, shown within ~0.5 s of upload
2. **Shell CAD** — the green ceramic mould, as built
3. **Shell modified** — the same mould after firing shrinkage
4. **Gating tree** — sprue, runner, ingates and risers (off by default)

Dropping a file shows only the part. The mould is built when you press
**Generate mould**, so process parameters can be set first.

Plus a matplotlib cross-section PNG per run.

### Firing shrinkage

An investment shell is built green around the pattern, then fired (default
1300 °C) to burn the pattern out and sinter the ceramic. The shell contracts
during that firing — 0.4 % linear by default, typical for a fused-silica /
zircon system. `shell_modified` is the green shell scaled by that factor about
its own centroid.

This is a property of the ceramic alone. The alloy poured in afterwards has no
bearing on it, and `test_firing_is_independent_of_alloy` pins that down.

### Viewer controls

Orbit by dragging, zoom with the wheel or the **+/−** buttons, turn the model
90° about a world axis with **X / Y / Z**, and **Reset** to clear the rotation.
**Fit view** reframes; the clip slider cuts into the mould.

## Pipeline

| stage | module | what it does |
|---|---|---|
| 1 ingest | `ingest.py` | load, repair to watertight, compute V, A, M = V/A |
| 2 hot spots | `hotspots.py` | voxelise, `distance_transform_edt`, suppressed local maxima |
| 3 gating | `gating.py` | size the sprue by modulus (`M = DH/(4H+D)`), then a 1:2:2 non-pressurised ratio |
| 4 risers | `risers.py` | cylindrical top risers sized so `M_riser ≥ 1.2 × M_section` |
| 5 synthesis | `synthesis.py` | stamp the tree, dilate by shell thickness, subtract, open the cup, marching cubes |

Stages are independently swappable. `Segment` (`kind, p0, p1, r0, r1`) is the only
currency between the metallurgy stages and synthesis, so `gating.py` can be
replaced wholesale without touching the rasteriser.

```python
from cad2shell import run, Config
stats = run("part.stl", "out/part", Config(alloy="A356", shell_thickness=6.0))
```

Writes `<prefix>_shell.stl`, `<prefix>_shell_fired.stl`, `<prefix>_tree.stl`
and `<prefix>_section.png`.

## The two classic bugs, and how they are prevented

**One coordinate frame.** `voxel.Grid` owns all world↔index arithmetic; no stage
does it itself. `grid_full` is built by *expanding* `grid_part`, so the two share
a lattice and differ by a pure integer offset — asserted at runtime and in
`tests/test_frame.py`. Hot spots are detected on the same grid the risers are
stamped into, so a riser cannot land a voxel off the thermal centre it was sized for.

**`floor`, never `round`.** `Grid.to_index` floors. A point 3.7 voxels along
belongs to cell 3; rounding would send it to cell 4 — a cell it is not inside —
punching pinholes in thin walls where two stages disagree by one cell.
`test_mapping_uses_floor_not_round` pins this.

## Notes on the metallurgy

- **Local modulus is not the EDT.** For a slab of half-thickness d cooling from
  both faces M = d = EDT, but for a cylinder M = R/2 and for a sphere M = R/3.
  Reading modulus off the distance field over-sizes risers 2–3×. `component_modulus`
  measures true V/A from voxel face counts (exact to 0.0% on flat-faced solids)
  and hot spots are scaled from it by relative section depth.
- **The choke is sized by modulus, not by fill rate.** An earlier version sized
  it from Bernoulli and continuity, with a pour time, a discharge coefficient
  and a metallostatic head. The sprue in an investment tree doubles as the
  feeder, so its base is set by the modulus it has to out-freeze — and that
  requirement always won. Those inputs changed nothing in the output, so they
  were removed rather than left on screen implying otherwise. Alloy density
  now affects only the reported pour mass. See `gating.py` and
  [docs/pouring-funnels.md](docs/pouring-funnels.md).
- **Riser sizing is analytic**, not voxel-measured: `M = D·H/(4H + D)` for a top
  riser whose base face is fed, inverted in closed form. The achieved factor is
  exactly 1.2000.
- A chunky part genuinely needs a huge riser. The modulus method is honest about
  this; real practice would add a chill.

## Tested invariants

`pytest -q` — 73 tests over a boss-plate, a thin bracket and a thick block:

- shell mesh and exported STL are watertight, positive signed volume
- wall thickness tracks the target across 3–8 mm (one voxel + 15% tolerance)
- metal path is a **single** 6-connected component
- every riser beats 1.2× the section it feeds, and its reported modulus matches
  its own geometry
- gating areas follow the configured ratio; sprue tapers toward the choke
- yield is physical (0 < y ≤ 100%)
- hot spots lie inside the part
- API contract: stats keys, valid STLs, uploads, config plumbing, 422s

## Known limitations

- Curved surfaces voxelise with staircase error; `component_modulus` reads ~33%
  low on a sphere. Flat-faced parts are exact. Reduce `--pitch` to converge.
- Riser placement is vertical top-feeding only; no side risers or chills.
- Grid memory scales as pitch⁻³; a warning fires above 40M voxels.

## Licence

Copyright © 2026 Arman Marchiel. All rights reserved.

This is **not** open-source software. The source is published for viewing
only — no right to use, copy, modify or distribute it is granted. See
[LICENSE](LICENSE) for the full terms, or get in touch for permission.

The third-party packages it depends on remain under their own licences.
