"""Mould cavity accuracy.

The cavity IS the casting surface, so any error on it becomes a defect on the
part. Carving it out of the voxel grid left a staircase: at 1.5 mm pitch the
cavity was out by 0.23 mm median and 1.3 mm at p99, which shows as visible
ridging in a bore. Subtracting the original mesh instead makes it exact.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from cad2shell import Config, demo_parts, run


def _cavity_error(part_mesh, shell_mesh, stats, n=20000):
    """Distance from the part's surface to the cavity wall that should mirror it.

    Points under the sprue footprint are excluded: the sprue replaces the shell
    there, so there is legitimately no cavity wall to measure against. On a
    chunky part the sprue can be wide enough to cover most of the top face --
    a 37 mm sprue on a 50x40 face -- and including those points measures the
    gate, not the cavity.
    """
    pts, _ = trimesh.sample.sample_surface(part_mesh, n)
    sprue = next(s for s in stats["gating"]["segments"] if s["kind"] == "sprue")
    base = np.asarray(sprue["p0"])
    r = float(sprue["r0"]) * 1.15          # a little margin for the cut edge
    under_gate = (np.linalg.norm(pts[:, :2] - base[:2], axis=1) <= r) & \
                 (pts[:, 2] >= base[2] - 1e-6)
    pts = pts[~under_gate]
    assert len(pts) > n // 10, "the sprue covers nearly the whole part"
    return np.abs(trimesh.proximity.signed_distance(shell_mesh, pts))


@pytest.mark.parametrize("part_name", ["thick_block", "boss_plate"])
def test_cavity_matches_the_original_geometry(part_name, tmp_path):
    """The cavity must reproduce the CAD surface, not a voxelised copy of it."""
    mesh = demo_parts.get(part_name)
    stats = run(mesh=mesh, out_prefix=str(tmp_path / part_name),
                cfg=Config(voxel_pitch=2.0)).as_dict()
    assert stats["shell"]["cavity_exact"], "fell back to the voxel cavity"

    shell = trimesh.load(stats["files"]["shell_stl"])
    err = _cavity_error(mesh, shell, stats)
    assert np.percentile(err, 99) < 0.05, (
        f"{part_name}: cavity is off by {np.percentile(err, 99):.3f} mm at p99"
    )


def test_cavity_accuracy_does_not_depend_on_pitch(tmp_path):
    """Exactness is the point: a coarser grid must not blunt the casting surface.

    With the old voxel cavity the error scaled linearly with pitch
    (1.31 -> 0.65 -> 0.30 mm as pitch halved). Now the pitch only affects the
    OUTER shell surface, which nobody measures.
    """
    mesh = demo_parts.thick_block()
    errors = {}
    for pitch in (2.5, 1.5):
        stats = run(mesh=mesh, out_prefix=str(tmp_path / f"p{pitch}"),
                    cfg=Config(voxel_pitch=pitch)).as_dict()
        shell = trimesh.load(stats["files"]["shell_stl"])
        errors[pitch] = float(np.percentile(_cavity_error(mesh, shell, stats), 99))
    assert errors[2.5] < 0.05 and errors[1.5] < 0.05, errors


@pytest.mark.parametrize("pitch", [2.5, 2.0, 1.5])
def test_exact_shell_is_still_watertight(pitch, tmp_path):
    """A boolean result that pinches is not printable.

    Cutters that ended exactly on the part's surface met it tangentially and
    left valence-4 edges; the mesh did not leak but no longer read as closed.
    The gating cutters now overlap slightly so the intersection is transversal.
    """
    stats = run(mesh=demo_parts.boss_plate(), out_prefix=str(tmp_path / "wt"),
                cfg=Config(voxel_pitch=pitch)).as_dict()
    shell = trimesh.load(stats["files"]["shell_stl"])
    assert shell.is_watertight, f"pitch {pitch}: shell is not watertight"
    assert len(trimesh.repair.broken_faces(shell)) == 0
    assert shell.volume > 0


def test_falls_back_to_the_voxel_cavity_when_disabled(tmp_path):
    """The voxel path must still work, as a fallback for meshes booleans reject."""
    stats = run(mesh=demo_parts.thick_block(), out_prefix=str(tmp_path / "vox"),
                cfg=Config(voxel_pitch=2.0, exact_cavity=False)).as_dict()
    assert stats["shell"]["cavity_exact"] is False
    shell = trimesh.load(stats["files"]["shell_stl"])
    assert shell.is_watertight


@pytest.mark.parametrize("part_name", ["thick_block", "boss_plate"])
def test_gating_tree_is_exact_geometry(part_name, tmp_path):
    """The tree must be the real casting unioned with real cones.

    It was still being re-derived from the voxel grid after the shell cavity
    was made exact, so the mould looked smooth while the tree inside it was
    visibly faceted -- staircased bores and a polygonal sprue.
    """
    mesh = demo_parts.get(part_name)
    stats = run(mesh=mesh, out_prefix=str(tmp_path / part_name),
                cfg=Config(voxel_pitch=2.0)).as_dict()
    assert stats["shell"]["tree_exact"], "fell back to the voxel tree"

    tree = trimesh.load(stats["files"]["tree_stl"])
    assert tree.is_watertight
    assert len(tree.split(only_watertight=False)) == 1, "the tree is not one solid"

    # Every point of the casting surface must lie on the tree's surface, since
    # the tree contains the part unchanged -- except under the sprue, where the
    # two are fused and the part's original face no longer exists.
    err = _cavity_error(mesh, tree, stats)
    assert np.percentile(err, 99) < 0.05, (
        f"{part_name}: tree surface is off the part by "
        f"{np.percentile(err, 99):.3f} mm at p99"
    )


def test_exact_tree_is_far_lighter_than_the_voxel_one(tmp_path):
    """Exact geometry is also cheaper: marching cubes emits a triangle per cell."""
    mesh = demo_parts.thick_block()
    exact = run(mesh=mesh, out_prefix=str(tmp_path / "e"),
                cfg=Config(voxel_pitch=1.5)).as_dict()
    voxel = run(mesh=mesh, out_prefix=str(tmp_path / "v"),
                cfg=Config(voxel_pitch=1.5, exact_cavity=False)).as_dict()
    n_exact = len(trimesh.load(exact["files"]["tree_stl"]).faces)
    n_voxel = len(trimesh.load(voxel["files"]["tree_stl"]).faces)
    assert n_exact < n_voxel / 3, f"exact {n_exact} vs voxel {n_voxel} triangles"
