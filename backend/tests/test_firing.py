"""Shell firing shrinkage.

An investment shell is built green around the pattern, then fired to burn the
pattern out and sinter the ceramic. The shell contracts during that firing.
This is a property of the ceramic system alone -- the alloy poured into the
mould afterwards has no bearing on it, which is what these tests pin down.
"""
from __future__ import annotations

import numpy as np
import pytest

from cad2shell import Config, demo_parts, run
from cad2shell.synthesis import fire_shell


def _mould(**kw):
    cfg = Config(voxel_pitch=2.0, **kw)
    return run(mesh=demo_parts.thick_block(), cfg=cfg, write_files=False), cfg


def test_fired_shell_is_uniformly_smaller():
    stats, cfg = _mould()
    fired = stats.as_dict()["shell"]["fired"]
    assert fired["scale"] == pytest.approx(1.0 - cfg.fire_shrinkage)
    # every linear dimension loses the same fraction
    delta = np.asarray(fired["extents_delta"])
    assert np.all(delta < 0), "the fired shell must be smaller than the green one"


def test_firing_is_independent_of_alloy():
    """The ceramic shrinks on firing; the metal has not been poured yet.

    Two moulds that differ only in alloy must have identical fired geometry.
    """
    a = run(mesh=demo_parts.thick_block(), write_files=False,
            cfg=Config(voxel_pitch=2.0, alloy="A356")).as_dict()
    b = run(mesh=demo_parts.thick_block(), write_files=False,
            cfg=Config(voxel_pitch=2.0, alloy="steel")).as_dict()
    assert a["shell"]["fired"]["scale"] == b["shell"]["fired"]["scale"]
    assert a["shell"]["fired"]["volume"] == pytest.approx(b["shell"]["fired"]["volume"])
    assert a["shell"]["fired"]["extents_delta"] == pytest.approx(
        b["shell"]["fired"]["extents_delta"])


def test_shrinkage_scales_with_the_configured_fraction():
    small, _ = _mould(fire_shrinkage=0.002)
    large, _ = _mould(fire_shrinkage=0.010)
    ds = abs(np.asarray(small.as_dict()["shell"]["fired"]["extents_delta"])).max()
    dl = abs(np.asarray(large.as_dict()["shell"]["fired"]["extents_delta"])).max()
    assert dl > ds * 4, "a 5x larger shrinkage must move the surface much further"


def test_fired_shell_stays_concentric():
    """Scaling must happen about the shell's own centre, not the world origin.

    Scaling about the origin would translate the mould as well as shrink it,
    which would show up as the fired shell drifting off the green one.
    """
    import trimesh
    cfg = Config()
    green = trimesh.creation.box(extents=[80, 60, 40])
    green.apply_translation([200.0, -150.0, 90.0])       # far from the origin
    fired = fire_shell(green, cfg)
    drift = np.abs(fired.bounding_box.centroid - green.bounding_box.centroid).max()
    assert drift < 1e-6, f"fired shell drifted {drift:.4f} mm off centre"


def test_fired_shell_is_watertight_and_exported():
    cfg = Config(voxel_pitch=2.0)
    stats = run(mesh=demo_parts.boss_plate(), out_prefix="out/_t_fired", cfg=cfg)
    d = stats.as_dict()
    assert "fired_stl" in d["files"], "the fired shell was not exported"
    import trimesh
    fired = trimesh.load(d["files"]["fired_stl"])
    assert fired.is_watertight
    assert fired.volume > 0


def test_firing_can_be_disabled():
    stats = run(mesh=demo_parts.thick_block(), write_files=False,
                cfg=Config(voxel_pitch=2.0, show_fired_shell=False)).as_dict()
    assert stats["shell"]["fired"]["volume"] is None


def test_absurd_shrinkage_is_rejected():
    with pytest.raises(ValueError, match="fire_shrinkage"):
        Config(fire_shrinkage=0.5)


def test_alloy_does_not_change_any_mould_geometry():
    """The alloy is not a mould parameter, so no dimension may depend on it.

    In the choke formula A = W / (rho*t*Cd*sqrt(2gH)), the poured mass W is
    itself rho*V -- so density cancels and the choke is a function of volume
    and fill time alone. Risers are sized by modulus (geometry) and the shell
    by an offset, neither of which sees the alloy either. The only thing that
    changes is the reported pour mass, which is why the alloy selector was
    removed from the process panel.
    """
    keys = []
    for alloy in ("A356", "steel", "Ti64", "IN718"):
        d = run(mesh=demo_parts.boss_plate(), write_files=False,
                cfg=Config(voxel_pitch=2.0, alloy=alloy)).as_dict()
        keys.append((
            round(d["gating"]["choke_radius"], 9),
            round(d["gating"]["runner_radius"], 9),
            round(d["gating"]["ingate_radius"], 9),
            round(d["shell"]["volume"], 6),
            round(d["metal_volume"], 6),
            tuple(round(r["diameter"], 9) for r in d["risers"]),
            round(d["shell"]["fired"]["scale"], 9),
        ))
    assert len(set(keys)) == 1, f"mould geometry varied with alloy: {keys}"


def test_the_mould_does_not_depend_on_the_alloy():
    """Shell geometry is set by the ceramic and the part, never by the metal.

    The alloy is carried for reporting, but nothing about the mould follows
    from it: the sprue is sized by the modulus it must out-freeze, which is
    geometry, and firing shrinkage is a property of the ceramic alone.
    """
    import trimesh
    from cad2shell import Config, demo_parts, run

    mesh = demo_parts.get("boss_plate")
    out = []
    for alloy in ("A356", "steel"):
        stats = run(mesh=mesh, out_prefix=f"/tmp/alloy_{alloy}",
                    cfg=Config(alloy=alloy, voxel_pitch=2.5), write_files=False)
        out.append(stats)
    assert out[0].gating["sprue_r_bottom"] == pytest.approx(
        out[1].gating["sprue_r_bottom"], rel=1e-9)
    assert out[0].shell["volume"] == pytest.approx(
        out[1].shell["volume"], rel=1e-9)

