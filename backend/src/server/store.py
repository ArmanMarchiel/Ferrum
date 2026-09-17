"""Persistent project store for the filesystem home page.

A project is one saved editor session: the part mesh the user dropped, every
process parameter, and the hand arrangement on the plate. Saving one writes
the mesh next to a JSON record so the editor can be restored exactly -- the
mesh is the half that cannot be regenerated from parameters, and without it a
"saved project" would reopen empty.

Layout mirrors a filesystem: rows carry a `parent_id`, folders are rows with
`kind="folder"`, and the root is `parent_id = None`. That is the same shape
the browser walks, so navigating a folder is a filter rather than a tree walk.

Storage is a single JSON index plus a blob directory, under
`~/.cad2shell/projects` by default. The alternative -- a database -- buys
nothing here: the index is small, written whole, and read once per request.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Literal

Kind = Literal["folder", "project"]

ROOT_ENV = "CAD2SHELL_DATA"
DEFAULT_ROOT = Path.home() / ".cad2shell"

# The index is rewritten whole on every mutation, so concurrent writers would
# interleave and lose rows. One process-wide lock serialises them; the file is
# small enough that holding it across a full read-modify-write is free.
_LOCK = threading.RLock()

MAX_NAME = 120


class StoreError(Exception):
    """A rejected operation -- bad name, missing row, cycle. Maps to 4xx."""


def data_root() -> Path:
    """Where everything lives. Overridable so tests get a throwaway dir."""
    return Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def _projects_dir() -> Path:
    return data_root() / "projects"


def _index_path() -> Path:
    return data_root() / "index.json"


def _blob_dir(pid: str) -> Path:
    return _projects_dir() / pid


def _now() -> float:
    return time.time()


def _read_index() -> dict[str, dict]:
    path = _index_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (ValueError, OSError):
        # A corrupt index must not take the whole app down: the meshes are
        # still on disk and a fresh index is recoverable by re-saving.
        return {}
    rows = raw.get("rows") if isinstance(raw, dict) else None
    if not isinstance(rows, dict):
        return {}
    return rows


def _write_index(rows: dict[str, dict]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a crash mid-write leaves the previous index intact
    # rather than a truncated file that reads back as "no projects".
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"version": 1, "rows": rows}, indent=2))
    tmp.replace(path)


def _clean_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise StoreError("name cannot be empty")
    if len(name) > MAX_NAME:
        raise StoreError(f"name is longer than {MAX_NAME} characters")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise StoreError("name cannot contain a path separator")
    return name


def _require(rows: dict[str, dict], rid: str | None, kind: Kind | None = None) -> dict:
    if rid is None:
        raise StoreError("missing id")
    row = rows.get(rid)
    if row is None:
        raise StoreError(f"no such item: {rid}")
    if kind is not None and row["kind"] != kind:
        raise StoreError(f"{rid} is a {row['kind']}, not a {kind}")
    return row


def _check_parent(rows: dict[str, dict], parent_id: str | None) -> None:
    """A parent must exist and must be a folder. The root is None."""
    if parent_id is None:
        return
    _require(rows, parent_id, "folder")


def _descendants(rows: dict[str, dict], rid: str) -> set[str]:
    """Every row under `rid`, itself excluded."""
    out: set[str] = set()
    frontier = [rid]
    while frontier:
        current = frontier.pop()
        for cid, row in rows.items():
            if row.get("parent_id") == current and cid not in out:
                out.add(cid)
                frontier.append(cid)
    return out


def _public(row: dict) -> dict:
    """The row as the API returns it -- no on-disk paths leak to the client."""
    out = {k: v for k, v in row.items() if k not in ("mesh_file", "mould_files")}
    out["has_mesh"] = bool(row.get("mesh_file"))
    # Which mould artefacts came back, so the editor knows what it can reload
    # and what it can offer for export without asking for each in turn.
    out["mould"] = sorted((row.get("mould_files") or {}).keys())
    return out


# ── reads ────────────────────────────────────────────────────────────────────

def list_items(parent_id: str | None = None) -> list[dict]:
    """One folder's contents: folders first, then projects, each by name."""
    with _LOCK:
        rows = _read_index()
        if parent_id is not None:
            _require(rows, parent_id, "folder")
        here = [_public(r) for r in rows.values()
                if r.get("parent_id") == parent_id]
    here.sort(key=lambda r: (r["kind"] != "folder", r["name"].lower()))
    return here


def all_items() -> list[dict]:
    """Every row, for the search box, which spans the whole filesystem."""
    with _LOCK:
        rows = _read_index()
    out = [_public(r) for r in rows.values()]
    out.sort(key=lambda r: (r["kind"] != "folder", r["name"].lower()))
    return out


def breadcrumbs(parent_id: str | None) -> list[dict]:
    """Root-first trail of folders leading to `parent_id`, root excluded."""
    if parent_id is None:
        return []
    with _LOCK:
        rows = _read_index()
        trail: list[dict] = []
        seen: set[str] = set()
        current: str | None = parent_id
        while current is not None:
            if current in seen:      # a cycle cannot happen, but never hang
                break
            seen.add(current)
            row = rows.get(current)
            if row is None:
                break
            trail.append({"id": row["id"], "name": row["name"]})
            current = row.get("parent_id")
    trail.reverse()
    return trail


def get(rid: str) -> dict:
    with _LOCK:
        rows = _read_index()
        return _public(_require(rows, rid))


def mesh_path(rid: str) -> Path:
    """On-disk mesh for a saved project. Raises if it was never stored."""
    with _LOCK:
        rows = _read_index()
        row = _require(rows, rid, "project")
        name = row.get("mesh_file")
    if not name:
        raise StoreError(f"project {rid} has no stored mesh")
    path = _blob_dir(rid) / name
    if not path.exists():
        raise StoreError(f"the mesh for project {rid} is missing from disk")
    return path


# The mould a build produces, by the name the API serves it under. Jobs live in
# the OS temp directory and are lost on reboot, so a project that only recorded
# a job id would reopen with its results gone -- the meshes are copied into the
# project's own directory instead.
MOULD_FILES = {
    "shell": "mould_shell.stl",
    "fired": "mould_shell_fired.stl",
    "tree": "mould_tree.stl",
    "section": "mould_section.png",
}


def mould_path(rid: str, which: str) -> Path:
    """One stored mould artefact for a project."""
    if which not in MOULD_FILES:
        raise StoreError(f"unknown mould file {which!r}")
    with _LOCK:
        rows = _read_index()
        row = _require(rows, rid, "project")
        have = row.get("mould_files") or {}
    if which not in have:
        raise StoreError(f"project {rid} has no stored {which}")
    path = _blob_dir(rid) / "mould" / have[which]
    if not path.exists():
        raise StoreError(f"the {which} for project {rid} is missing from disk")
    return path


def save_mould(rid: str, job_dir: Path, stats: dict) -> dict:
    """Copy a finished build out of its job directory and into the project.

    Called after the meshes are on disk, so the build survives the job being
    cleaned up and the server restarting. Missing pieces are skipped rather
    than failing the save: a build with no gating tree is still worth keeping.
    """
    with _LOCK:
        rows = _read_index()
        row = _require(rows, rid, "project")

        dest = _blob_dir(rid) / "mould"
        # Replace wholesale: a second build must not leave the previous run's
        # tree behind to be served alongside the new shell.
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)

        stored: dict[str, str] = {}
        for key, fname in MOULD_FILES.items():
            src = Path(job_dir) / fname
            if not src.exists():
                continue
            shutil.copy2(src, dest / fname)
            stored[key] = fname

        row["mould_files"] = stored
        row["mould_stats"] = stats
        row["updated_at"] = _now()
        _write_index(rows)
        return _public(row)


def clear_mould(rid: str) -> None:
    """Drop a stored build. Used when the part or the parameters change, so a
    project never carries a mould that was built for a different setup."""
    with _LOCK:
        rows = _read_index()
        row = rows.get(rid)
        if row is None:
            return
        shutil.rmtree(_blob_dir(rid) / "mould", ignore_errors=True)
        row.pop("mould_files", None)
        row.pop("mould_stats", None)
        _write_index(rows)


# ── writes ───────────────────────────────────────────────────────────────────

def create_folder(name: str, parent_id: str | None = None) -> dict:
    name = _clean_name(name)
    with _LOCK:
        rows = _read_index()
        _check_parent(rows, parent_id)
        rid = uuid.uuid4().hex[:12]
        row = {"id": rid, "kind": "folder", "name": name,
               "parent_id": parent_id, "created_at": _now(),
               "updated_at": _now()}
        rows[rid] = row
        _write_index(rows)
        return _public(row)


def save_project(name: str, state: dict, parent_id: str | None = None,
                 mesh: bytes | None = None, mesh_name: str | None = None,
                 project_id: str | None = None) -> dict:
    """Create a project, or overwrite one when `project_id` is given.

    `mesh` is optional on an overwrite: re-saving a project whose part has not
    changed keeps the mesh already on disk rather than making the browser
    re-upload several megabytes on every parameter tweak.
    """
    name = _clean_name(name)
    with _LOCK:
        rows = _read_index()

        if project_id is not None:
            row = _require(rows, project_id, "project")
            rid = project_id
            # An overwrite keeps the row where it is unless a move was asked
            # for; `parent_id=None` means "the root", so it cannot double as
            # "leave it alone" -- the caller passes the current parent back.
            _check_parent(rows, parent_id)
            row["parent_id"] = parent_id
        else:
            _check_parent(rows, parent_id)
            rid = uuid.uuid4().hex[:12]
            row = {"id": rid, "kind": "project", "created_at": _now(),
                   "parent_id": parent_id, "mesh_file": None}
            rows[rid] = row

        row["name"] = name
        row["updated_at"] = _now()
        row["state"] = state

        if mesh is not None:
            blobs = _blob_dir(rid)
            blobs.mkdir(parents=True, exist_ok=True)
            suffix = Path(mesh_name or "part.stl").suffix.lower() or ".stl"
            old = row.get("mesh_file")
            fname = f"part{suffix}"
            (blobs / fname).write_bytes(mesh)
            row["mesh_file"] = fname
            row["mesh_name"] = mesh_name or fname
            row["mesh_bytes"] = len(mesh)
            # A part re-saved in a different format leaves the previous blob
            # behind, which would then be the one served for a stale filename.
            if old and old != fname:
                (blobs / old).unlink(missing_ok=True)

        _write_index(rows)
        return _public(row)


def rename(rid: str, name: str) -> dict:
    name = _clean_name(name)
    with _LOCK:
        rows = _read_index()
        row = _require(rows, rid)
        row["name"] = name
        row["updated_at"] = _now()
        _write_index(rows)
        return _public(row)


def move(rid: str, parent_id: str | None) -> dict:
    with _LOCK:
        rows = _read_index()
        row = _require(rows, rid)
        _check_parent(rows, parent_id)
        if parent_id == rid:
            raise StoreError("an item cannot be moved into itself")
        # Moving a folder into its own subtree would detach that subtree from
        # the root: it would still exist but no path could reach it.
        if row["kind"] == "folder" and parent_id in _descendants(rows, rid):
            raise StoreError("a folder cannot be moved into its own subtree")
        row["parent_id"] = parent_id
        row["updated_at"] = _now()
        _write_index(rows)
        return _public(row)


def delete(rid: str) -> list[str]:
    """Delete a row and everything beneath it. Returns the ids removed."""
    with _LOCK:
        rows = _read_index()
        _require(rows, rid)
        doomed = {rid} | _descendants(rows, rid)
        for victim in doomed:
            rows.pop(victim, None)
            shutil.rmtree(_blob_dir(victim), ignore_errors=True)
        _write_index(rows)
    return sorted(doomed)
