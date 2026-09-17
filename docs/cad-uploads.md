# CAD uploads and ingest

How a part gets from the user's disk into the pipeline: what formats are
accepted, what happens to the mesh on the way in, and where it ends up.

Code referenced here: `backend/src/server/app.py` (endpoints, limits,
job model), `backend/src/cad2shell/ingest.py` (stage 1), and
`backend/src/server/store.py` (persistence). The contract is pinned by
`backend/tests/test_api.py` and `backend/tests/test_projects.py`.

## Accepted formats

`ACCEPTED_SUFFIXES` in `app.py` is the single source of truth, and it is
triangle-mesh formats only:

```python
ACCEPTED_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".3mf"}
```

The set is advertised to the frontend by `GET /api/meta` as
`accepted_formats`, alongside `limits.max_upload_mb`, so the UI never has to
hardcode it.

### Why native CAD formats are not supported

STEP, IGES, DWG, DXF and PRT **were supported briefly and were then removed**.
The reasoning, verbatim from the comment above `ACCEPTED_SUFFIXES`, is that
they are not meshes, so each needs a conversion step whose results were not
trustworthy enough to ship:

- **STEP** files arrived as multi-body assemblies in the wrong units.
- **DXF** is a 2-D drawing — there is no thickness to extrude.
- **DWG** and **PRT** are proprietary formats needing external tooling.

The pipeline needs one closed solid. Exporting STL from the CAD package is the
one path that reliably produces that, so the server asks for it up front
rather than half-converting and producing a mould that looks plausible and is
meaningless. `test_cad_formats_are_rejected` parametrises over `.step`,
`.stp`, `.dxf`, `.dwg`, `.prt` and `.iges` and asserts a 415 for each;
`test_meta_advertises_the_accepted_formats` asserts they are not advertised
either. Re-adding one means re-solving units, assemblies and licensing, not
just widening the set.

The units problem has an echo downstream: `pipeline.py` warns when the part's
largest extent is under `2 × shell_thickness`, because STEP and IGES files
frequently arrive in inches or metres and a 5 mm "casting" inside a 6 mm
ceramic skin is geometrically valid and physically meaningless.

## The upload endpoints

Two endpoints take a mesh. Both enforce the same suffix and size checks.

| call | purpose |
|---|---|
| `POST /generate` (multipart) | ingest a part and (optionally) start a build |
| `POST /api/fs/projects` (multipart) | save an editor session, mesh included |

### `POST /generate`

Multipart form. `file` is the mesh; every other field is a process parameter
fed to `_build_config` (`alloy`, `shell_thickness`, `voxel_pitch`, `quantity`,
`placements`, …). One extra flag matters here:

- `build` (default `true`) — when `false`, the request ingests and stops.

```bash
curl -F file=@part.stl -F voxel_pitch=1.5 -F shell_thickness=6 \
     http://127.0.0.1:8000/generate
```

```json
{
  "job_id": "3f9a1c7b2d04",
  "phase": "queued",
  "urls": {
    "shell_stl":   "/jobs/3f9a1c7b2d04/shell.stl",
    "fired_stl":   "/jobs/3f9a1c7b2d04/shell_fired.stl",
    "tree_stl":    "/jobs/3f9a1c7b2d04/tree.stl",
    "section_png": "/jobs/3f9a1c7b2d04/section.png",
    "part_stl":    "/jobs/3f9a1c7b2d04/part.stl",
    "status":      "/jobs/3f9a1c7b2d04/status"
  },
  "part":   { "volume": 30000.0, "area": 5900.0, "modulus": 5.08,
              "watertight": true, "triangles": 12, "bounds": [], "extents": [],
              "repairs": [] },
  "config": { }
}
```

`build=false` exists because of a real bug. The viewer uses it on drop: it
wants the part on screen immediately, but the mould must not be built until
the process parameters — quantity among them — have been set. Starting a build
on drop anyway meant every upload ran a full single-part mould that was thrown
away, and worse, that build could finish *after* the real one and overwrite
the correct mould on screen with a one-up shell. With `build=false` the
response carries `phase: "ingested"` and no job is queued.

## The two-phase job model

`POST /generate` returns as soon as the part itself is on disk — typically
well under a second. Ingest runs synchronously (it is a load plus a repair
pass, and the viewer needs the part mesh immediately); everything slow happens
afterwards on a worker thread.

```
POST /generate ──► ingest (sync, fast) ──► 200 with job_id + part stats
                                     └──► EXEC worker thread: run(...)
GET /jobs/{id}/status  (poll) ──► phase: queued → building → done | error
GET /jobs/{id}/shell.stl etc.  once phase == "done"
```

### Why a worker thread rather than inline

From the comment above `EXEC`: the pipeline is pure CPU and takes seconds to
minutes on a real part. Running it inline blocks the event loop for that whole
time, which makes the server stop answering *every* request — **including the
poll the frontend uses to find out whether the job finished**. The browser's
upload appeared to hang forever and nothing reached the log. So each job runs
on a `ThreadPoolExecutor(max_workers=2)` and the request handlers only ever
touch `JOB_STATE`, a dict guarded by `JOB_LOCK`.

`test_server_stays_responsive_while_a_job_builds` is the regression test: it
hammers `/health` while a build runs and asserts it keeps answering.

### Polling

```bash
curl http://127.0.0.1:8000/jobs/3f9a1c7b2d04/status
```

`phase` is one of `queued`, `building`, `ingested`, `done` or `error`. While
building, `step`, `progress` (0–1) and `detail` are updated by the pipeline's
`progress` callback — a build that only said "building" for minutes was
indistinguishable from a hung one, which is what
`test_progress_is_reported_while_building` guards. On `done`, `stats` holds
the full stats dict plus `job_id` and `urls`. On `error`, the state carries
`error`, `error_type` and the last six traceback lines.

Job artefacts live in `tempfile.gettempdir()/cad2shell_jobs/<job_id>/`. That
directory does **not** survive a reboot — see persistence below.

## Size limits and validation

`MAX_UPLOAD_MB = 64`, checked against the raw byte length after reading the
upload, on both endpoints. The order of checks in `/generate` is:

1. Config is parsed first — a bad `voxel_pitch` or unknown `alloy` is a `422`
   before the file is even looked at.
2. Missing or unnamed `file` → `422 "no file uploaded"`.
3. Over `MAX_UPLOAD_MB` → `413`.
4. Suffix not in `ACCEPTED_SUFFIXES` → `415`, with the accepted list in the
   message. The suffix is taken from the filename, lowercased, defaulting to
   `.stl` when there is none.
5. The bytes are written to `<job>/upload<suffix>` and ingest runs.

Note that validation is by **filename suffix**, not by sniffing content. A
`.step` file renamed to `.stl` passes the suffix check and then fails in
ingest, which is a `422` rather than a `415`.

## What ingest does to the mesh

`ingest.load_part(path)` is stage 1. It calls
`trimesh.load(path, force="mesh", process=False)` — `force="mesh"` so a scene
collapses toward geometry, `process=False` so trimesh's own vertex merging
does not silently change the part before the repair log records anything — and
hands the result to `from_mesh`, which does three things:

**1. Reduce to one `Trimesh`** (`_as_single_mesh`). A `Scene` has its
`Trimesh` geometries concatenated, noting `concatenated N scene geometries`.
No triangle geometry at all raises `ValueError("file contains no triangle
geometry")`.

**2. Repair, best effort** (`repair`), escalating only as far as needed and
logging each step into `repairs`:

- Drop infinite values, degenerate faces and duplicate faces; drop
  unreferenced vertices.
- If not watertight, `merge_vertices()` — fixes the common case of an STL
  whose triangles share no vertices.
- If still not watertight, `trimesh.repair.fill_holes()`.
- If still not watertight, a **voxel remesh** at `max(extents)/128`, filled
  and run through marching cubes. This always yields a closed surface but is
  costly and lossy, so it is the last resort.
- Finally `fix_normals()`, and if the signed volume is negative the mesh is
  inverted — an inside-out winding.

**3. Characterise** (`characterise`): volume (absolute), surface area, and the
casting modulus `M = V/A`. Zero area raises; zero or negative volume after
repair raises.

There is **no unit conversion anywhere in ingest**. The pipeline treats mesh
coordinates as millimetres throughout (`volume` is mm³, `modulus` mm). A file
authored in inches is ingested as a part 25.4× too small, which is why the
scale sanity check in `pipeline.py` exists.

The result is a `PartMesh`, whose `as_dict()` is exactly the `part` object in
the `/generate` response: `volume`, `area`, `modulus`, `watertight`,
`triangles`, `bounds`, `extents` and the `repairs` log. A mesh that survived
repair without becoming watertight is *not* rejected — it is ingested with
`watertight: false`, and the pipeline warns that voxel occupancy may be
unreliable.

The ingested mesh is exported to `<job>/part.stl`, and the build reads *that*,
not the original upload. It is the normalised, repaired, single-body form.

## Persistence

Job directories are temporary. Anything meant to survive goes through
`store.py`, whose root is:

```
$CAD2SHELL_DATA   (falls back to ~/.cad2shell)
├── index.json              every row, rewritten whole under a process lock
└── projects/<project_id>/
    ├── part.stl            the uploaded mesh, verbatim bytes
    └── mould/              copies of a finished build's artefacts
```

Tests set `CAD2SHELL_DATA` to a tmpdir so they never touch the real store.

`POST /api/fs/projects` saves an editor session: `name`, a JSON `state` blob,
optional `parent_id`, optional `project_id` (to overwrite), and optional
`file`. The mesh is written by `store.save_project` as `part<suffix>` in the
project's blob directory, byte-for-byte as uploaded — the round trip is
asserted in `test_a_saved_project_round_trips_its_state_and_its_mesh`. A
replaced mesh in a different format unlinks the old blob, so a stale filename
cannot be served.

Two rules worth knowing:

- **A new project must carry a mesh** (`422` otherwise). The mesh is the half
  that cannot be regenerated from the JSON; without it the project would
  reopen empty.
- **An overwrite may omit it.** Re-saving after a parameter tweak keeps the
  mesh already on disk rather than pushing several megabytes back up the wire.

Finished builds are copied out of the job directory into
`projects/<id>/mould/` by `POST /api/fs/projects/{id}/mould`, because the job
directory lives in the OS temp dir and a project that only recorded a job id
would reopen with its results gone. `test_a_stored_mould_outlives_its_job`
deletes the job directory and asserts the project still serves its shell.

## Error cases

| condition | status | body |
|---|---|---|
| no `file` field, or empty filename | 422 | `no file uploaded` |
| body over 64 MB | 413 | `upload exceeds 64 MB` |
| suffix not accepted (incl. `.step`, `.dxf`, `.dwg`, `.prt`, `.iges`) | 415 | `unsupported format '.step'; accepted: …` |
| bad parameter (`voxel_pitch=-1`, `alloy=unobtanium`, malformed `pour_up`/`placements`) | 422 | `invalid configuration: …` |
| file accepted by suffix but unreadable as a mesh | 422 | `{"error": "could not read the mesh: …", "type": "…"}` |
| unknown job id on status or artefact | 404 | `unknown job <id>` |
| non-alphanumeric job id | 400 | `bad job id` |
| pipeline raised mid-build | 200 on `/generate`, then `phase: "error"` on status | `error`, `error_type`, `traceback` |

The unreadable-mesh case is the one shape break: it is a `JSONResponse` with
an `error` key rather than FastAPI's `detail`. The frontend relies on status
`404` from `/jobs/{id}/status` to detect a server restart and stop polling.

A build failure is **not** an upload failure. `/generate` has already returned
200 by then, so the client only learns about it by polling.

## If an upload is giving you trouble

- **"unsupported format"** — export STL from your CAD package. In most
  packages this is File → Export → STL; pick binary, millimetres, and a
  moderate chord tolerance. That is the one path that reliably produces a
  single closed solid.
- **"could not read the mesh"** — the suffix and the contents disagree, or the
  file is truncated. Re-export rather than renaming.
- **Part reported as tiny / a warning about inches** — the export was in
  inches or metres. Re-export in millimetres; nothing in the pipeline rescales
  for you.
- **`watertight: false` in the response** — repair could not close the
  surface. Check `part.repairs` to see how far it escalated, then fix the
  model in CAD; the mould will still build but voxel occupancy is unreliable.
- **A warning about separate bodies** — the export is an assembly. Gating is
  generated for every body together, which is rarely what you want. Export one
  casting.
- **Over 64 MB** — decimate before exporting. Mesh density above what
  `voxel_pitch` can resolve buys nothing: the part is voxelised anyway.
