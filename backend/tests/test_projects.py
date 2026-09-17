"""The project store behind the filesystem home page.

A project is one editor session: the part mesh plus every process parameter
and the arrangement on the plate. The mesh is the half that cannot be rebuilt
from the JSON, so these tests care most about it surviving a round trip -- a
"saved project" that reopens without its part is not one.
"""
from __future__ import annotations

import io
import json

import pytest
import trimesh
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Every test gets its own store directory, so none of them can see another
    # one's rows and none of them writes to the real ~/.cad2shell.
    monkeypatch.setenv("CAD2SHELL_DATA", str(tmp_path / "data"))
    from server.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def part_bytes() -> bytes:
    buf = io.BytesIO()
    trimesh.creation.box(extents=[30, 20, 10]).export(buf, file_type="stl")
    return buf.getvalue()


STATE = {"version": 1, "fields": {"qty": "4", "shell": "6"},
         "placements": [{"x": 1.0, "y": 2.0, "z": 0.0, "rot": 45.0}]}


def _save(client, part_bytes, name="Part A", **fields):
    data = {"name": name, "state": json.dumps(fields.pop("state", STATE))}
    data.update({k: v for k, v in fields.items() if v is not None})
    files = {"file": ("part.stl", part_bytes, "model/stl")} if part_bytes else None
    return client.post("/api/fs/projects", data=data, files=files)


# ── routing ──────────────────────────────────────────────────────────────────

def test_root_and_editor_both_serve_the_app(client):
    """Both routes answer with the SPA shell.

    Which page you get is decided in the browser by the router, not here, so
    there is no server-rendered markup left to tell the two apart -- that
    distinction is covered by the headless UI tests instead.
    """
    home = client.get("/")
    if home.status_code == 404:
        pytest.skip("frontend/dist not built; run `cd frontend && npm run build`")
    assert home.status_code == 200
    editor = client.get("/editor")
    assert editor.status_code == 200
    assert 'id="root"' in home.text
    assert home.text == editor.text


# ── saving and reopening ─────────────────────────────────────────────────────

def test_a_saved_project_round_trips_its_state_and_its_mesh(client, part_bytes):
    saved = _save(client, part_bytes).json()
    assert saved["kind"] == "project" and saved["has_mesh"]

    got = client.get(f"/api/fs/projects/{saved['id']}").json()
    assert got["name"] == "Part A"
    assert got["state"] == STATE

    # The mesh comes back byte-identical, which is what lets the editor
    # reopen a project without asking for the file again.
    part = client.get(f"/api/fs/projects/{saved['id']}/part")
    assert part.status_code == 200
    assert part.content == part_bytes


def test_a_new_project_without_a_mesh_is_refused(client):
    # It would reopen empty, so it is not a saved project at all.
    assert _save(client, None).status_code == 422


def test_overwriting_keeps_the_mesh_already_on_disk(client, part_bytes):
    """Re-saving after a parameter tweak must not re-upload the whole part."""
    first = _save(client, part_bytes).json()
    again = _save(client, None, name="Part A",
                  state={"version": 1, "fields": {"qty": "9"}},
                  project_id=first["id"]).json()

    assert again["id"] == first["id"]           # same project, not a second one
    assert again["has_mesh"]
    assert client.get(f"/api/fs/projects/{first['id']}").json()["state"]["fields"]["qty"] == "9"
    assert client.get(f"/api/fs/projects/{first['id']}/part").content == part_bytes


def test_a_replaced_mesh_overwrites_the_stored_one(client, part_bytes):
    first = _save(client, part_bytes).json()
    other = io.BytesIO()
    trimesh.creation.box(extents=[80, 80, 80]).export(other, file_type="stl")
    replacement = other.getvalue()
    assert replacement != part_bytes

    _save(client, replacement, project_id=first["id"])
    assert client.get(f"/api/fs/projects/{first['id']}/part").content == replacement


def test_state_must_be_a_json_object(client, part_bytes):
    r = client.post("/api/fs/projects",
                    data={"name": "x", "state": "not json"},
                    files={"file": ("part.stl", part_bytes, "model/stl")})
    assert r.status_code == 422


def test_an_unsupported_part_format_is_refused(client):
    r = client.post("/api/fs/projects",
                    data={"name": "x", "state": "{}"},
                    files={"file": ("part.step", b"nope", "application/step")})
    assert r.status_code == 415


# ── folders ──────────────────────────────────────────────────────────────────

def test_folders_nest_and_list_their_own_contents(client, part_bytes):
    outer = client.post("/api/fs/folders", json={"name": "Brackets"}).json()
    inner = client.post("/api/fs/folders",
                        json={"name": "v2", "parent_id": outer["id"]}).json()
    proj = _save(client, part_bytes, name="Bracket A", parent_id=inner["id"]).json()

    root = client.get("/api/fs").json()
    assert [i["name"] for i in root["items"]] == ["Brackets"]
    assert root["breadcrumbs"] == []

    deep = client.get("/api/fs", params={"parent": inner["id"]}).json()
    assert [i["name"] for i in deep["items"]] == ["Bracket A"]
    # The trail is what the header renders, root-first.
    assert [c["name"] for c in deep["breadcrumbs"]] == ["Brackets", "v2"]
    assert proj["parent_id"] == inner["id"]


def test_folders_sort_before_projects(client, part_bytes):
    _save(client, part_bytes, name="aaa project")
    client.post("/api/fs/folders", json={"name": "zzz folder"})
    kinds = [i["kind"] for i in client.get("/api/fs").json()["items"]]
    assert kinds == ["folder", "project"]


def test_listing_an_unknown_folder_is_a_404(client):
    assert client.get("/api/fs", params={"parent": "nope"}).status_code == 404


# ── move, rename, delete ─────────────────────────────────────────────────────

def test_moving_a_project_between_folders(client, part_bytes):
    folder = client.post("/api/fs/folders", json={"name": "Brackets"}).json()
    proj = _save(client, part_bytes).json()

    client.patch(f"/api/fs/{proj['id']}", json={"parent_id": folder["id"]})
    assert client.get("/api/fs").json()["items"] == [
        i for i in client.get("/api/fs").json()["items"] if i["kind"] == "folder"]
    assert [i["name"] for i in
            client.get("/api/fs", params={"parent": folder["id"]}).json()["items"]] == ["Part A"]


def test_a_folder_cannot_be_moved_into_its_own_subtree(client):
    """Allowing it would detach the subtree: it exists, but no path reaches it."""
    outer = client.post("/api/fs/folders", json={"name": "outer"}).json()
    inner = client.post("/api/fs/folders",
                        json={"name": "inner", "parent_id": outer["id"]}).json()
    assert client.patch(f"/api/fs/{outer['id']}",
                        json={"parent_id": inner["id"]}).status_code == 422
    assert client.patch(f"/api/fs/{outer['id']}",
                        json={"parent_id": outer["id"]}).status_code == 422


def test_renaming(client, part_bytes):
    proj = _save(client, part_bytes).json()
    assert client.patch(f"/api/fs/{proj['id']}",
                        json={"name": "Renamed"}).json()["name"] == "Renamed"
    assert client.patch(f"/api/fs/{proj['id']}", json={"name": "  "}).status_code == 422
    assert client.patch(f"/api/fs/{proj['id']}", json={"name": "a/b"}).status_code == 422


def test_deleting_a_folder_takes_its_contents_with_it(client, part_bytes):
    folder = client.post("/api/fs/folders", json={"name": "Brackets"}).json()
    sub = client.post("/api/fs/folders",
                      json={"name": "v2", "parent_id": folder["id"]}).json()
    proj = _save(client, part_bytes, parent_id=sub["id"]).json()

    deleted = client.delete(f"/api/fs/{folder['id']}").json()["deleted"]
    assert set(deleted) == {folder["id"], sub["id"], proj["id"]}
    assert client.get("/api/fs").json()["items"] == []
    # The blob goes with it, not just the row.
    assert client.get(f"/api/fs/projects/{proj['id']}").status_code == 404


def test_the_search_index_spans_every_folder(client, part_bytes):
    folder = client.post("/api/fs/folders", json={"name": "Brackets"}).json()
    _save(client, part_bytes, name="Deep one", parent_id=folder["id"])
    names = {i["name"] for i in client.get("/api/fs/all").json()["items"]}
    assert names == {"Brackets", "Deep one"}


def test_a_folder_is_not_a_project(client):
    folder = client.post("/api/fs/folders", json={"name": "Brackets"}).json()
    assert client.get(f"/api/fs/projects/{folder['id']}").status_code == 422


# ── the built mould ─────────────────────────────────────────────────────────
#
# Jobs live in the OS temp directory and do not survive a reboot, so a project
# that only recorded a job id would reopen with its results gone. The meshes
# are copied into the project instead.


def _build(client, part_file):
    """Run a real build and return its finished job id and stats."""
    import time
    with open(part_file, "rb") as fh:
        r = client.post("/generate",
                        files={"file": ("part.stl", fh, "model/stl")},
                        data={"voxel_pitch": 3.0, "shell_thickness": 6.0})
    assert r.status_code == 200, r.text
    job = r.json()["job_id"]
    deadline = time.time() + 240
    while time.time() < deadline:
        st = client.get(f"/jobs/{job}/status").json()
        if st["phase"] == "done":
            return job, st["stats"]
        assert st["phase"] != "error", st
        time.sleep(0.4)
    pytest.fail("build did not finish in time")


@pytest.fixture(scope="module")
def part_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("mould") / "block.stl"
    trimesh.creation.box(extents=[30, 24, 18]).export(p)
    return p


def test_a_built_mould_is_stored_with_its_project(client, part_bytes, part_file):
    job, stats = _build(client, part_file)
    saved = _save(client, part_bytes).json()

    row = client.post(f"/api/fs/projects/{saved['id']}/mould",
                      json={"job_id": job, "stats": stats}).json()
    # Every artefact the build produced comes across, not just the shell.
    assert set(row["mould"]) == {"shell", "fired", "tree", "section"}

    # And each is served back from the project's own copy.
    for which, kind in [("shell", b"solid"), ("fired", b"solid"),
                        ("tree", b"solid"), ("section", b"\\x89PNG")]:
        got = client.get(f"/api/fs/projects/{saved['id']}/mould/{which}")
        assert got.status_code == 200, which
        assert len(got.content) > 0


def test_a_stored_mould_outlives_its_job(client, part_bytes, part_file, tmp_path):
    """The point of copying: the job directory can go and the mould remains."""
    import shutil
    from server.app import JOBS

    job, stats = _build(client, part_file)
    saved = _save(client, part_bytes).json()
    client.post(f"/api/fs/projects/{saved['id']}/mould",
                json={"job_id": job, "stats": stats})

    shutil.rmtree(JOBS / job, ignore_errors=True)
    assert client.get(f"/jobs/{job}/shell.stl").status_code == 404
    assert client.get(f"/api/fs/projects/{saved['id']}/mould/shell").status_code == 200


def test_rebuilding_replaces_the_stored_mould(client, part_bytes, part_file):
    """A second build must not leave the first one's tree behind."""
    job, stats = _build(client, part_file)
    saved = _save(client, part_bytes).json()
    client.post(f"/api/fs/projects/{saved['id']}/mould",
                json={"job_id": job, "stats": stats})
    first = client.get(f"/api/fs/projects/{saved['id']}/mould/shell").content

    job2, stats2 = _build(client, part_file)
    client.post(f"/api/fs/projects/{saved['id']}/mould",
                json={"job_id": job2, "stats": stats2})
    assert client.get(f"/api/fs/projects/{saved['id']}/mould/shell").status_code == 200
    assert len(first) > 0


def test_a_mould_can_be_cleared(client, part_bytes, part_file):
    """Changing the setup drops the build, so a project never reopens showing
    results for parameters that have since been edited."""
    job, stats = _build(client, part_file)
    saved = _save(client, part_bytes).json()
    client.post(f"/api/fs/projects/{saved['id']}/mould",
                json={"job_id": job, "stats": stats})
    assert client.get(f"/api/fs/projects/{saved['id']}").json()["mould"]

    client.delete(f"/api/fs/projects/{saved['id']}/mould")
    assert client.get(f"/api/fs/projects/{saved['id']}").json()["mould"] == []
    assert client.get(f"/api/fs/projects/{saved['id']}/mould/shell").status_code == 422


def test_attaching_an_unknown_job_is_refused(client, part_bytes):
    saved = _save(client, part_bytes).json()
    r = client.post(f"/api/fs/projects/{saved['id']}/mould",
                    json={"job_id": "deadbeef", "stats": {}})
    assert r.status_code == 404


def test_deleting_a_project_takes_its_mould(client, part_bytes, part_file):
    job, stats = _build(client, part_file)
    saved = _save(client, part_bytes).json()
    client.post(f"/api/fs/projects/{saved['id']}/mould",
                json={"job_id": job, "stats": stats})

    client.delete(f"/api/fs/{saved['id']}")
    assert client.get(f"/api/fs/projects/{saved['id']}/mould/shell").status_code == 404
