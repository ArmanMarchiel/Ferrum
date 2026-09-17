"""FastAPI contract: the endpoint the frontend depends on.

The API is two-phase by design. `POST /generate` ingests the mesh and returns
immediately with the part; the mould is built on a worker thread and collected
by polling `/jobs/{id}/status`. That keeps the event loop free -- running the
pipeline inline wedged the server for the whole build and made it stop
answering every other request, including the poll itself.
"""
from __future__ import annotations

import io
import time

import pytest
import trimesh
from fastapi.testclient import TestClient

from server.app import app

TIMEOUT = 180.0


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def part_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("api") / "block.stl"
    trimesh.creation.box(extents=[40, 30, 25]).export(p)
    return p


def _post(client, part_file, **fields):
    with open(part_file, "rb") as fh:
        return client.post("/generate",
                           files={"file": (part_file.name, fh, "model/stl")},
                           data=fields)


def _await_done(client, job, timeout=TIMEOUT):
    """Poll a job to completion and return its final state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/jobs/{job}/status").json()
        if st.get("phase") in ("done", "error"):
            return st
        time.sleep(0.1)
    pytest.fail(f"job {job} did not finish within {timeout}s")


def test_health(client):
    assert client.get("/health").json()["ok"] is True


def test_meta_lists_alloys_and_defaults(client):
    m = client.get("/api/meta").json()
    assert "A356" in m["alloys"]
    assert m["defaults"]["shell_thickness"] > 0
    # demo parts were removed from the product; they must not come back
    assert "demo_parts" not in m


def test_index_page_is_served(client):
    """`/` serves the built SPA, which the router then splits into pages.

    The bundle is hashed, so this asserts on the shape of the entry document --
    a root to mount into and a module script -- rather than on any markup,
    which now only exists after React has run.
    """
    r = client.get("/")
    if r.status_code == 404:
        pytest.skip("frontend/dist not built; run `cd frontend && npm run build`")
    assert r.status_code == 200
    assert 'id="root"' in r.text
    assert 'type="module"' in r.text
    # the demo-part UI is gone
    assert 'id="demos"' not in r.text


def test_editor_serves_the_same_bundle(client):
    """`/editor` is the same document: the SPA router picks the page."""
    a = client.get("/")
    b = client.get("/editor")
    if a.status_code == 404:
        pytest.skip("frontend/dist not built; run `cd frontend && npm run build`")
    assert b.status_code == 200
    assert a.text == b.text


def test_generate_returns_the_part_without_waiting(client, part_file):
    """Phase 1 must return promptly with the part, before the mould exists."""
    t0 = time.time()
    r = _post(client, part_file, voxel_pitch="1.5")
    elapsed = time.time() - t0
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["phase"] in ("queued", "building")
    assert d["part"]["triangles"] > 0
    assert d["part"]["volume"] > 0
    # the part STL is downloadable straight away
    assert client.get(d["urls"]["part_stl"]).status_code == 200
    assert elapsed < 20.0, f"phase 1 took {elapsed:.1f}s; it should not build the mould"


def test_job_completes_and_returns_full_stats(client, part_file):
    d = _post(client, part_file, voxel_pitch="2.0").json()
    st = _await_done(client, d["job_id"])
    assert st["phase"] == "done", st.get("error")
    stats = st["stats"]
    for key in ("part_volume", "casting_modulus", "hot_spots", "gating",
                "risers", "shell", "metal_volume", "yield_pct", "grid"):
        assert key in stats, f"stats JSON is missing {key!r}"
    assert stats["shell"]["watertight"] is True
    assert 0 < stats["yield_pct"] <= 100


def test_all_artifacts_are_served_after_completion(client, part_file):
    d = _post(client, part_file, voxel_pitch="2.0").json()
    _await_done(client, d["job_id"])
    for name in ("shell_stl", "tree_stl", "part_stl", "section_png"):
        assert client.get(d["urls"][name]).status_code == 200, name


def test_generated_stls_are_valid_meshes(client, part_file):
    d = _post(client, part_file, voxel_pitch="2.0").json()
    _await_done(client, d["job_id"])
    for name in ("shell_stl", "tree_stl", "part_stl"):
        raw = client.get(d["urls"][name]).content
        mesh = trimesh.load(io.BytesIO(raw), file_type="stl")
        assert len(mesh.faces) > 0, f"{name} has no triangles"
        assert mesh.is_watertight, f"{name} is not watertight"


def test_config_params_reach_the_pipeline(client, part_file):
    d = _post(client, part_file, shell_thickness="3.0", voxel_pitch="1.0",
              alloy="steel").json()
    assert d["config"]["shell_thickness"] == 3.0
    assert d["config"]["alloy"] == "steel"
    stats = _await_done(client, d["job_id"])["stats"]
    assert stats["shell"]["thickness_target"] == 3.0
    assert abs(stats["shell"]["thickness_measured"]["median"] - 3.0) <= 1.5


def test_server_stays_responsive_while_a_job_builds(client, part_file):
    """The regression that motivated the threadpool.

    A synchronous pipeline blocked the event loop for the entire build, so the
    server answered nothing -- the browser's upload appeared to hang forever
    with no request ever reaching the log.
    """
    d = _post(client, part_file, voxel_pitch="0.8").json()
    job = d["job_id"]
    served = 0
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        assert client.get("/health").json()["ok"] is True
        served += 1
        if client.get(f"/jobs/{job}/status").json().get("phase") in ("done", "error"):
            break
        time.sleep(0.05)
    assert served > 1, "the job finished too fast to prove responsiveness"


def test_unsupported_format_is_rejected(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("this is not a mesh")
    with open(bad, "rb") as fh:
        r = client.post("/generate", files={"file": ("notes.txt", fh, "text/plain")})
    assert r.status_code == 415


def test_missing_file_is_rejected(client):
    assert client.post("/generate", data={}).status_code == 422


def test_invalid_config_is_rejected(client, part_file):
    assert _post(client, part_file, voxel_pitch="-1").status_code == 422
    assert _post(client, part_file, alloy="unobtanium").status_code == 422


def test_unreadable_mesh_reports_an_error(client, tmp_path):
    junk = tmp_path / "broken.stl"
    junk.write_bytes(b"not really an stl file at all")
    with open(junk, "rb") as fh:
        r = client.post("/generate", files={"file": ("broken.stl", fh, "model/stl")})
    assert r.status_code == 422
    assert "error" in r.json()


def test_unknown_job_returns_404(client):
    assert client.get("/jobs/deadbeef1234/status").status_code == 404
    assert client.get("/jobs/deadbeef1234/shell.stl").status_code == 404


def test_meta_advertises_the_accepted_formats(client):
    """Triangle-mesh formats only; CAD formats were removed deliberately."""
    formats = client.get("/api/meta").json()["accepted_formats"]
    assert ".stl" in formats
    for gone in (".dxf", ".dwg", ".prt", ".step", ".iges"):
        assert gone not in formats, f"{gone} should no longer be advertised"


@pytest.mark.parametrize("ext", [".step", ".stp", ".dxf", ".dwg", ".prt", ".iges"])
def test_cad_formats_are_rejected(client, tmp_path, ext):
    """CAD formats must 415 rather than half-working.

    They were briefly accepted via conversion and then pulled: STEP files
    arrived as multi-body assemblies in the wrong units, DXF is a 2-D drawing
    with no thickness to extrude, and DWG and PRT need proprietary tooling.
    Rejecting them up front is honest; silently producing a meaningless mould
    is not.
    """
    src = tmp_path / f"part{ext}"
    src.write_bytes(b"\x00" * 64)
    with open(src, "rb") as fh:
        r = client.post("/generate",
                        files={"file": (src.name, fh, "application/octet-stream")})
    assert r.status_code == 415, r.text


@pytest.mark.parametrize("ext", [".obj", ".ply", ".off"])
def test_other_mesh_formats_still_work(client, tmp_path, ext):
    """These share the STL code path, so they keep working."""
    import trimesh
    src = tmp_path / f"part{ext}"
    trimesh.creation.box(extents=[40, 30, 25]).export(src)
    with open(src, "rb") as fh:
        r = client.post("/generate",
                        files={"file": (src.name, fh, "application/octet-stream")},
                        data={"voxel_pitch": "2.5"})
    assert r.status_code == 200, r.text
    stats = _await_done(client, r.json()["job_id"])
    assert stats["phase"] == "done", stats.get("error")


def test_progress_is_reported_while_building(client, part_file):
    """The status must name the stage, not just spin.

    A build that only says "building" for minutes is indistinguishable from a
    hung one -- which is exactly how a lost job presented before this.
    """
    d = _post(client, part_file, voxel_pitch="0.9").json()
    job = d["job_id"]
    steps, deadline = set(), time.time() + TIMEOUT
    while time.time() < deadline:
        st = client.get(f"/jobs/{job}/status").json()
        if st.get("step"):
            steps.add(st["step"])
        if st.get("phase") in ("done", "error"):
            break
        time.sleep(0.05)
    assert steps, "no pipeline stage was ever reported"
    assert steps - {"starting", "queued"}, f"only trivial steps reported: {steps}"


def test_status_404s_for_an_unknown_job(client):
    """The frontend relies on 404 to detect a server restart and stop polling."""
    assert client.get("/jobs/abc123def456/status").status_code == 404


def test_tiny_part_is_flagged_not_silently_moulded(client, tmp_path):
    """A part smaller than its own shell is a unit error, not a casting.

    STEP and IGES often arrive in inches or metres. Wrapping a 5 mm object in a
    6 mm ceramic skin is geometrically valid and physically meaningless, so the
    pipeline has to say so.
    """
    import trimesh
    src = tmp_path / "tiny.stl"
    trimesh.creation.box(extents=[4, 3, 2]).export(src)
    with open(src, "rb") as fh:
        r = client.post("/generate", files={"file": ("tiny.stl", fh, "model/stl")},
                        data={"shell_thickness": "6", "voxel_pitch": "1.0"})
    assert r.status_code == 200, r.text
    stats = _await_done(client, r.json()["job_id"])
    assert stats["phase"] == "done", stats.get("error")
    warnings = " ".join(stats["stats"]["warnings"]).lower()
    assert "inches" in warnings or "mm across" in warnings, \
        f"a sub-shell-sized part was not flagged: {stats['stats']['warnings']}"


def test_warning_percentages_are_sane(client, tmp_path):
    """No warning may report an absurd number.

    A near-zero part volume in the denominator once produced
    '284993413919539104%', which tells the user nothing.
    """
    import re

    import trimesh
    src = tmp_path / "small.stl"
    trimesh.creation.box(extents=[3, 3, 3]).export(src)
    with open(src, "rb") as fh:
        r = client.post("/generate", files={"file": ("small.stl", fh, "model/stl")},
                        data={"voxel_pitch": "1.5"})
    stats = _await_done(client, r.json()["job_id"])
    for w in stats["stats"]["warnings"]:
        for pct in re.findall(r"(\d+(?:\.\d+)?)%", w):
            assert float(pct) < 10000, f"nonsense percentage in warning: {w}"
