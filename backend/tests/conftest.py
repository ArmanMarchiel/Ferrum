"""Shared fixtures.

Pipeline runs are the expensive part of this suite, so each (part, config)
combination is executed once per session and shared across the tests that
assert on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent          # backend/
sys.path.insert(0, str(ROOT / "src"))

from cad2shell import Config, demo_parts, run  # noqa: E402

# name -> config overrides. The pitch is chosen per part: the thin bracket has
# a 5 mm wall that needs a finer grid to voxelise faithfully.
CASES = {
    "boss_plate":   dict(voxel_pitch=1.5, shell_thickness=6.0),
    "thin_bracket": dict(voxel_pitch=1.0, shell_thickness=4.0),
    "thick_block":  dict(voxel_pitch=1.5, shell_thickness=6.0),
}


@pytest.fixture(scope="session")
def outdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cad2shell")


@pytest.fixture(scope="session")
def results(outdir) -> dict:
    """Run the pipeline once per case; every invariant test reads from here."""
    out = {}
    for name, overrides in CASES.items():
        cfg = Config(**overrides)
        stats = run(mesh=demo_parts.get(name), out_prefix=str(outdir / name), cfg=cfg)
        out[name] = {"stats": stats, "cfg": cfg, "name": name}
    return out


@pytest.fixture(scope="session")
def artifacts(outdir, results) -> dict:
    """Meshes reloaded from the exported STLs, so the files themselves are tested."""
    import trimesh
    loaded = {}
    for name in CASES:
        loaded[name] = {
            "shell": trimesh.load(outdir / f"{name}_shell.stl"),
            "tree": trimesh.load(outdir / f"{name}_tree.stl"),
        }
    return loaded


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        metafunc.parametrize("case", sorted(CASES), scope="session")
