"""FastAPI wrapper around the cad2shell pipeline.

    uvicorn server.app:app --reload

POST /generate takes an uploaded STL plus process parameters and returns the
shell STL, the tree STL and the stats JSON. GET / serves the viewer.
"""
from __future__ import annotations

import asyncio
import base64
import json
import io
import sys
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # backend/src

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cad2shell import Config, run
from cad2shell.config import ALLOYS, CERAMICS
from server import store
from server.store import StoreError

# backend/src/server/app.py -> repo root is three levels up.
ROOT = Path(__file__).resolve().parents[3]
# The built SPA, and the static files Vite copies into it verbatim.
DIST = ROOT / "frontend" / "dist"
PUBLIC = ROOT / "frontend" / "public"
JOBS = Path(tempfile.gettempdir()) / "cad2shell_jobs"
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="cad2shell", version="0.1.0",
              description="Part STL -> investment shell mould with gating tree")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MAX_UPLOAD_MB = 64

# Triangle-mesh formats only. Native and exchange CAD formats (STEP, IGES,
# DWG, DXF, PRT) were supported briefly and removed: they are not meshes, so
# each needs a conversion step whose results were not trustworthy enough to
# ship -- STEP files arrived as multi-body assemblies in the wrong units, DXF
# is a 2-D drawing with no thickness to extrude, and DWG and PRT are
# proprietary formats needing external tooling. Export to STL from your CAD
# package instead; that is the one path that reliably produces a single closed
# solid, which is what the pipeline needs.
ACCEPTED_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".3mf"}

# The pipeline is pure CPU and takes seconds to minutes on a real part. Running
# it inline would block the event loop for that whole time, which makes the
# server stop answering *every* request -- including the poll that the frontend
# uses to find out whether the job finished. So each job runs on a worker
# thread and the request handlers only ever touch the registry below.
EXEC = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cad2shell")
JOB_STATE: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


def _set(job: str, **fields) -> None:
    with JOB_LOCK:
        JOB_STATE.setdefault(job, {}).update(fields)


def _get(job: str) -> dict:
    with JOB_LOCK:
        return dict(JOB_STATE.get(job, {}))


def _index() -> HTMLResponse:
    """The built SPA's entry page.

    Both `/` and `/editor` return this: the app is a single bundle and its
    router picks the page from the path. In development the pages are served
    by Vite on :5173 instead, which proxies the API back here -- so a missing
    build is a normal state, not an error, and says which command to run.
    """
    page = DIST / "index.html"
    if not page.exists():
        raise HTTPException(
            404,
            "frontend/dist not built. Run `cd frontend && npm run build`, "
            "or use the Vite dev server on :5173 (`npm run dev`).",
        )
    return HTMLResponse(page.read_text())


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """The filesystem: saved projects and the folders holding them."""
    return _index()


@app.get("/editor", response_class=HTMLResponse)
def editor() -> HTMLResponse:
    """The CAD editor. `?project=<id>` reopens a saved project."""
    return _index()


@app.get("/ferrum-logo.png")
def logo():
    """The mark in the editor header, which doubles as the way back home."""
    path = PUBLIC / "ferrum-logo.png"
    if not path.exists():
        raise HTTPException(404, "logo not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/meta")
def meta() -> dict:
    """Alloy presets and defaults, for populating the UI."""
    d = Config()
    return {
        "alloys": {k: {"label": v["label"], "density": v["density"]} for k, v in ALLOYS.items()},
        "ceramics": {k: {"label": v["label"]} for k, v in CERAMICS.items()},
        "defaults": {
            "alloy": d.alloy,
            "shell_thickness": d.shell_thickness,
            "voxel_pitch": d.voxel_pitch,
            "riser_modulus_factor": d.riser_modulus_factor,
            "gating_ratio": list(d.gating_ratio),
            "ceramic": d.ceramic,
            "fire_temperature": d.fire_temperature,
            "fire_hold_hours": d.fire_hold_hours,
            "fire_shrinkage": d.fire_shrinkage,
            "use_risers": d.use_risers,
            "max_risers": d.max_risers,
            "quantity": d.quantity,
            "layout": d.layout,
        },
        "limits": {"max_upload_mb": MAX_UPLOAD_MB},
        "accepted_formats": sorted(ACCEPTED_SUFFIXES),
    }


def _build_config(alloy, density, shell_thickness, voxel_pitch,
                  riser_modulus_factor, max_risers, section_axis, use_risers=None,
                  pour_up=None,
                  ceramic=None, fire_temperature=None, fire_hold_hours=None,
                  fire_shrinkage=None,
                  quantity=None, layout=None, part_spacing=None,
                  layout_columns=None, placements=None) -> Config:
    kw: dict = {}
    if alloy:
        kw["alloy"] = alloy
    if density is not None and density > 0:
        kw["density"] = density
    if shell_thickness is not None:
        kw["shell_thickness"] = shell_thickness
    if voxel_pitch is not None:
        kw["voxel_pitch"] = voxel_pitch
    if riser_modulus_factor is not None:
        kw["riser_modulus_factor"] = riser_modulus_factor
    if use_risers is not None:
        kw["use_risers"] = bool(use_risers)
    if pour_up:
        try:
            vec = [float(v) for v in str(pour_up).split(",")]
        except ValueError as exc:
            raise HTTPException(422, f"pour_up must be 'x,y,z': {exc}") from exc
        if len(vec) != 3:
            raise HTTPException(422, "pour_up must have three components")
        if sum(v * v for v in vec) < 1e-12:
            raise HTTPException(422, "pour_up must be a non-zero direction")
        kw["pour_up"] = tuple(vec)
    if max_risers is not None:
        kw["max_risers"] = int(max_risers)
    if section_axis is not None:
        kw["section_axis"] = int(section_axis)
    if ceramic:
        kw["ceramic"] = ceramic
    if fire_temperature is not None:
        kw["fire_temperature"] = fire_temperature
    if fire_hold_hours is not None and fire_hold_hours > 0:
        kw["fire_hold_hours"] = fire_hold_hours
    if fire_shrinkage is not None:
        kw["fire_shrinkage"] = fire_shrinkage
    if quantity is not None:
        kw["quantity"] = int(quantity)
    if layout:
        kw["layout"] = layout
    if part_spacing is not None and part_spacing >= 0:
        kw["part_spacing"] = part_spacing
    if layout_columns is not None and layout_columns > 0:
        kw["layout_columns"] = int(layout_columns)
    if placements:
        # A hand-made arrangement from the viewer. It defines the quantity, so
        # it is applied last and wins over any quantity sent alongside it.
        try:
            parsed = json.loads(placements) if isinstance(placements, str) else placements
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, f"placements is not valid JSON: {exc}") from exc
        if not isinstance(parsed, list) or not parsed:
            raise HTTPException(422, "placements must be a non-empty list")
        if len(parsed) > 64:
            raise HTTPException(422, "at most 64 placements")
        clean = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise HTTPException(422, f"placement {i} is not an object")
            try:
                clean.append({k: float(item.get(k, 0.0))
                              for k in ("x", "y", "z", "rot")})
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    422, f"placement {i} has a non-numeric field: {exc}") from exc
        kw["placements"] = clean
    try:
        return Config(**kw)
    except ValueError as exc:
        raise HTTPException(422, f"invalid configuration: {exc}") from exc


def _run_job(job: str, work: Path, cfg: Config, path_arg, mesh_arg) -> None:
    """Execute the pipeline on a worker thread and record the outcome."""
    def report(step: str, progress: float, detail: str = "") -> None:
        _set(job, phase="building", step=step, progress=progress, detail=detail)

    try:
        _set(job, phase="building", step="starting", progress=0.02)
        stats = run(part_path=path_arg, out_prefix=str(work / "mould"),
                    cfg=cfg, mesh=mesh_arg, progress=report)
        payload = stats.as_dict()
        payload["job_id"] = job
        payload["urls"] = _urls(job)
        _set(job, phase="done", progress=1.0, stats=payload)
    except Exception as exc:
        _set(job, phase="error", progress=1.0, error=str(exc),
             error_type=type(exc).__name__,
             traceback=traceback.format_exc().splitlines()[-6:])


def _urls(job: str) -> dict:
    return {
        "shell_stl": f"/jobs/{job}/shell.stl",
        "fired_stl": f"/jobs/{job}/shell_fired.stl",
        "tree_stl": f"/jobs/{job}/tree.stl",
        "section_png": f"/jobs/{job}/section.png",
        "part_stl": f"/jobs/{job}/part.stl",
        "status": f"/jobs/{job}/status",
    }


@app.post("/generate")
async def generate(
    file: UploadFile | None = File(default=None),
    alloy: str | None = Form(default=None),
    density: float | None = Form(default=None),
    shell_thickness: float | None = Form(default=None),
    voxel_pitch: float | None = Form(default=None),
    riser_modulus_factor: float | None = Form(default=None),
    use_risers: bool | None = Form(default=None),
    pour_up: str | None = Form(default=None),
    max_risers: int | None = Form(default=None),
    section_axis: int | None = Form(default=None),
    ceramic: str | None = Form(default=None),
    fire_temperature: float | None = Form(default=None),
    fire_hold_hours: float | None = Form(default=None),
    fire_shrinkage: float | None = Form(default=None),
    quantity: int | None = Form(default=None),
    layout: str | None = Form(default=None),
    part_spacing: float | None = Form(default=None),
    layout_columns: int | None = Form(default=None),
    placements: str | None = Form(default=None),
    build: bool = Form(default=True),
):
    """Accept a part and start building its mould.

    Returns as soon as the part itself is on disk -- typically well under a
    second -- so the viewer can render the original CAD straight away. The
    mould is built on a worker thread; poll `urls.status` for progress and the
    finished stats.
    """
    cfg = _build_config(alloy, density, shell_thickness, voxel_pitch,
                        riser_modulus_factor, max_risers, section_axis, use_risers,
                        pour_up, ceramic, fire_temperature, fire_hold_hours, fire_shrinkage,
                        quantity, layout, part_spacing, layout_columns,
                        placements)

    job = uuid.uuid4().hex[:12]
    work = JOBS / job
    work.mkdir(parents=True, exist_ok=True)

    if file is None or not file.filename:
        raise HTTPException(422, "no file uploaded")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_MB} MB")
    suffix = Path(file.filename).suffix.lower() or ".stl"
    if suffix not in ACCEPTED_SUFFIXES:
        raise HTTPException(
            415,
            f"unsupported format {suffix!r}; accepted: "
            + ", ".join(sorted(ACCEPTED_SUFFIXES)),
        )
    src = work / f"upload{suffix}"
    src.write_bytes(raw)

    # Ingest synchronously: it is fast (a load plus a repair pass) and the
    # viewer needs the part mesh immediately. Anything slow happens on the
    # worker thread afterwards.
    def _ingest():
        from cad2shell import ingest
        part = ingest.load_part(src)
        part.mesh.export(work / "part.stl")
        return part

    try:
        part = await asyncio.get_running_loop().run_in_executor(EXEC, _ingest)
    except Exception as exc:
        return JSONResponse(status_code=422,
                            content={"error": f"could not read the mesh: {exc}",
                                     "type": type(exc).__name__})

    # `build=false` ingests and stops. The viewer uses it on drop: it wants the
    # part mesh on screen immediately, but the mould must not be built until
    # the process parameters -- quantity among them -- have been set. Starting
    # a build here anyway meant every upload ran a full single-part mould that
    # was thrown away, and worse, that build could finish *after* the real one
    # and overwrite the correct mould on screen with a one-up shell.
    if not build:
        _set(job, phase="ingested", step="ingested", progress=1.0,
             filename=file.filename)
        return JSONResponse({
            "job_id": job,
            "phase": "ingested",
            "urls": _urls(job),
            "part": part.as_dict(),
            "config": cfg.as_dict(),
        })

    _set(job, phase="queued", step="queued", progress=0.05, filename=file.filename)
    # Build from the ingested part.stl: for a converted CAD file the original
    # upload is not a mesh, so the pipeline must read what conversion produced.
    asyncio.get_running_loop().run_in_executor(
        EXEC, _run_job, job, work, cfg, work / "part.stl", None)

    return JSONResponse({
        "job_id": job,
        "phase": "queued",
        "urls": _urls(job),
        "part": part.as_dict(),
        "config": cfg.as_dict(),
    })


@app.get("/jobs/{job}/status")
def job_status(job: str):
    """Poll a job. Returns phase, progress and -- once done -- the full stats."""
    if not job.isalnum():
        raise HTTPException(400, "bad job id")
    st = _get(job)
    if not st:
        raise HTTPException(404, f"unknown job {job}")
    return JSONResponse(st)


def _serve(job: str, name: str, media: str):
    safe = uuid.UUID(hex=job) if len(job) == 32 else None  # noqa: F841
    if not job.isalnum():
        raise HTTPException(400, "bad job id")
    path = JOBS / job / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found for job {job}")
    return FileResponse(path, media_type=media, filename=f"{job}_{name}")


@app.get("/jobs/{job}/shell.stl")
def get_shell(job: str):
    return _serve(job, "mould_shell.stl", "model/stl")


@app.get("/jobs/{job}/shell_fired.stl")
def get_fired(job: str):
    return _serve(job, "mould_shell_fired.stl", "model/stl")


@app.get("/jobs/{job}/tree.stl")
def get_tree(job: str):
    return _serve(job, "mould_tree.stl", "model/stl")


@app.get("/jobs/{job}/part.stl")
def get_part(job: str):
    return _serve(job, "part.stl", "model/stl")


@app.get("/jobs/{job}/section.png")
def get_section(job: str):
    return _serve(job, "mould_section.png", "image/png")




# ── filesystem: saved projects ───────────────────────────────────────────────
#
# The home page is a filesystem over saved projects. A project is one editor
# session: the part mesh plus every process parameter and the hand arrangement
# on the plate. The mesh is stored alongside because it is the only part that
# cannot be reconstructed from the JSON -- a project that reopened without its
# part would not be a saved project at all.


def _store_call(fn, *args, **kwargs):
    """Run a store operation, mapping its refusals onto HTTP status codes."""
    try:
        return fn(*args, **kwargs)
    except StoreError as exc:
        msg = str(exc)
        raise HTTPException(404 if msg.startswith("no such item") else 422, msg) from exc


@app.get("/api/fs")
def fs_list(parent: str | None = None) -> dict:
    """One folder's contents plus the trail that leads to it."""
    return {
        "parent_id": parent,
        "breadcrumbs": _store_call(store.breadcrumbs, parent),
        "items": _store_call(store.list_items, parent),
    }


@app.get("/api/fs/all")
def fs_all() -> dict:
    """Every row. The search box spans the filesystem, not the open folder."""
    return {"items": _store_call(store.all_items)}


@app.post("/api/fs/folders")
def fs_create_folder(payload: dict = Body(...)) -> dict:
    return _store_call(store.create_folder,
                       str(payload.get("name", "")), payload.get("parent_id"))


@app.get("/api/fs/projects/{rid}")
def fs_get_project(rid: str) -> dict:
    """A saved project, with the state the editor needs to restore itself."""
    row = _store_call(store.get, rid)
    if row.get("kind") != "project":
        raise HTTPException(422, f"{rid} is a folder, not a project")
    return row


@app.get("/api/fs/projects/{rid}/part")
def fs_project_part(rid: str):
    """The stored part mesh, so the editor can reload it without an upload."""
    path = _store_call(store.mesh_path, rid)
    media = "model/stl" if path.suffix.lower() == ".stl" else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)


@app.post("/api/fs/projects")
async def fs_save_project(
    name: str = Form(...),
    state: str = Form(...),
    parent_id: str | None = Form(default=None),
    project_id: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    """Save the editor's session -- creating a project, or overwriting one.

    The mesh is optional on an overwrite: re-saving after a parameter change
    should not make the browser push the whole part up the wire again.
    """
    try:
        parsed = json.loads(state)
    except ValueError as exc:
        raise HTTPException(422, f"state is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(422, "state must be a JSON object")

    raw = None
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower() or ".stl"
        if suffix not in ACCEPTED_SUFFIXES:
            raise HTTPException(
                415,
                f"unsupported format {suffix!r}; accepted: "
                + ", ".join(sorted(ACCEPTED_SUFFIXES)),
            )
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"part exceeds {MAX_UPLOAD_MB} MB")

    # A new project with no mesh would reopen empty, which is not a saved
    # project. An overwrite may omit it and keep the one already on disk.
    if raw is None and project_id is None:
        raise HTTPException(422, "a new project needs its part mesh")

    return _store_call(store.save_project, name, parsed, parent_id, raw,
                       file.filename if file is not None else None, project_id)


@app.post("/api/fs/projects/{rid}/mould")
def fs_attach_mould(rid: str, payload: dict = Body(...)) -> dict:
    """Store a finished build with its project.

    The browser sends the job it just built; the meshes are copied out of the
    job directory, which lives in the OS temp dir and does not survive a
    reboot. Without this a reopened project would show its part and its
    parameters but claim the mould had never been built.
    """
    row = _store_call(store.get, rid)
    if row.get("kind") != "project":
        raise HTTPException(422, f"{rid} is a folder, not a project")

    job = str(payload.get("job_id", ""))
    if not job.isalnum():
        raise HTTPException(422, "bad job id")
    work = JOBS / job
    if not work.is_dir():
        raise HTTPException(404, f"unknown job {job}")

    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise HTTPException(422, "stats must be an object")
    return _store_call(store.save_mould, rid, work, stats)


@app.delete("/api/fs/projects/{rid}/mould")
def fs_clear_mould(rid: str) -> dict:
    """Forget a stored build -- the setup it was built for has changed."""
    _store_call(store.clear_mould, rid)
    return {"ok": True}


@app.get("/api/fs/projects/{rid}/mould/{which}")
def fs_project_mould(rid: str, which: str):
    """Serve one stored mould artefact back to the viewer."""
    path = _store_call(store.mould_path, rid, which)
    media = "image/png" if path.suffix.lower() == ".png" else "model/stl"
    return FileResponse(path, media_type=media, filename=f"{rid}_{path.name}")


@app.patch("/api/fs/{rid}")
def fs_update(rid: str, payload: dict = Body(...)) -> dict:
    """Rename and/or move. `parent_id` is only applied when it is sent."""
    row = None
    if "name" in payload:
        row = _store_call(store.rename, rid, str(payload["name"]))
    if "parent_id" in payload:
        row = _store_call(store.move, rid, payload["parent_id"])
    if row is None:
        raise HTTPException(422, "nothing to update: send name and/or parent_id")
    return row


@app.delete("/api/fs/{rid}")
def fs_delete(rid: str) -> dict:
    """Delete a row and everything under it."""
    return {"deleted": _store_call(store.delete, rid)}



@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": "0.1.0"}


# The bundle's own hashed JS and CSS. Mounted last, after every API route is
# declared, so a file in dist/ can never shadow one of them. It is mounted only
# when a build exists: in development Vite serves these instead, and mounting a
# missing directory raises at import time, which would take the API down with
# it for the sake of files nobody was going to ask this server for.
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")
