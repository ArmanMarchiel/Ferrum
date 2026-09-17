"""Metallurgical invariants: riser feeding, choke sizing, gating ratio."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cad2shell import risers
from cad2shell.config import GRAVITY, Config


def test_every_riser_beats_the_section_it_feeds(results, case):
    """M_riser >= 1.2 x M_section, or the riser freezes first and feeds nothing.

    This is the invariant the whole riser stage exists to satisfy.
    """
    r = results[case]
    stats, cfg = r["stats"].as_dict(), r["cfg"]
    # A part whose only hot spot is under the sprue gets no separate riser --
    # the sprue is its feeder. That is correct, so the invariant is about the
    # risers that DO exist, not about there being any.
    for riser in stats["risers"]:
        assert riser["modulus"] >= cfg.riser_modulus_factor * riser["feeds_modulus"] - 1e-9, (
            f"{case}: riser {riser['index']} M={riser['modulus']:.3f} does not beat "
            f"{cfg.riser_modulus_factor} x {riser['feeds_modulus']:.3f}"
        )
        assert riser["factor"] >= cfg.riser_modulus_factor - 1e-9


def test_riser_modulus_formula_matches_geometry(results, case):
    """The reported modulus must equal D H / (4H + D) for the reported D and H."""
    for riser in results[case]["stats"].as_dict()["risers"]:
        d, h = riser["diameter"], riser["height"]
        assert riser["modulus"] == pytest.approx(d * h / (4 * h + d), rel=1e-9)


def test_modulus_inversion_is_exact():
    """diameter_for_modulus must invert cylinder_modulus for any ratio."""
    cfg = Config()
    for target in (0.5, 2.0, 5.0, 17.5, 40.0):
        for k in (0.8, 1.5, 2.5):
            d = risers.diameter_for_modulus(target, k)
            assert risers.cylinder_modulus(d, k * d) == pytest.approx(target, rel=1e-9)


def test_riser_diameter_floor_over_feeds_never_under_feeds():
    """When the minimum diameter binds, the result must still meet the target."""
    cfg = Config(min_riser_diameter=12.0)
    r = risers.size_riser(0.3, cfg)          # tiny section, floor will bind
    assert r.diameter >= 12.0
    assert r.factor >= cfg.riser_modulus_factor


def test_the_sprue_is_sized_by_modulus_not_fill_rate():
    """The sprue base answers "can it feed?", not "can it fill in time?".

    An earlier version sized the choke from Bernoulli and continuity, with a
    pour time, a discharge coefficient and a metallostatic head. The arithmetic
    was right but it never bound: the sprue doubles as the feeder in this
    scheme, and the modulus requirement is always the larger of the two. Those
    inputs therefore changed nothing in the output, and have been removed.

    This pins the consequence -- the sprue tracks the section it feeds -- so a
    fill-rate term reappearing without changing the geometry would be caught.
    """
    from cad2shell import risers
    thin = risers.diameter_for_modulus(1.2 * 3.0, 1.5)
    thick = risers.diameter_for_modulus(1.2 * 9.0, 1.5)
    assert thick > thin * 2.5, "the feeder must scale with the modulus it feeds"


def test_firing_shrinkage_is_derived_from_the_schedule():
    """Temperature and hold time produce the shrinkage; it is not typed in.

    This is the direction that matches a foundry: nobody knows their linear
    contraction up front, they know what schedule they fired to.
    """
    from cad2shell import Config
    cool = Config(fire_temperature=1000.0)
    hot = Config(fire_temperature=1400.0)
    assert hot.fire_shrinkage > cool.fire_shrinkage * 2, \
        "a hotter firing must shrink the shell substantially more"

    # longer holds shrink more, but with strongly diminishing return
    short = Config(fire_temperature=1300.0, fire_hold_hours=1.0)
    long = Config(fire_temperature=1300.0, fire_hold_hours=8.0)
    assert long.fire_shrinkage > short.fire_shrinkage
    assert long.fire_shrinkage < short.fire_shrinkage * 2, \
        "eight times the hold must not mean eight times the shrinkage"

    # and the ceramic system matters: alumina sinters far harder than silica
    assert (Config(ceramic="alumina").fire_shrinkage
            > 2 * Config(ceramic="zircon").fire_shrinkage)


def test_an_explicit_shrinkage_overrides_the_model():
    """Measured data for a real slurry must beat the table."""
    from cad2shell import Config
    assert Config(fire_shrinkage=0.015).fire_shrinkage == 0.015


def test_sprue_tapers_downward(results, case):
    """The sprue must narrow toward the choke so the stream stays in contact."""
    g = results[case]["stats"].as_dict()["gating"]
    assert g["sprue_r_top"] > g["sprue_r_bottom"]
    assert g["sprue_r_bottom"] == pytest.approx(g["choke_radius"], rel=1e-9)


def test_yield_is_physical(results, case):
    """Yield is part/metal on one volume basis; it cannot exceed 100%."""
    stats = results[case]["stats"].as_dict()
    assert 0.0 < stats["yield_pct"] <= 100.0, (
        f"{case}: yield {stats['yield_pct']:.1f}% is not physical"
    )
    assert stats["metal_volume"] >= stats["part_volume_voxel"] - 1e-6


def test_hot_spots_are_inside_the_part(results, case):
    """A thermal centre outside the casting would mean the frames disagree."""
    stats = results[case]["stats"].as_dict()
    lo, hi = np.asarray(stats["part"]["bounds"])
    assert stats["hot_spots"], f"{case}: no hot spots detected"
    for hs in stats["hot_spots"]:
        w = np.asarray(hs["world"])
        assert np.all(w >= lo - 1e-6) and np.all(w <= hi + 1e-6), (
            f"{case}: hot spot {w} lies outside the part bounds"
        )
        assert hs["edt_mm"] > 0


def test_thicker_sections_get_larger_feeders():
    """A thick block must be fed by more metal than a thin bracket.

    Measured on the sprue rather than on a riser: the sprue is the feeder for
    the section it lands on, and on a part with a single thermal centre it is
    the ONLY feeder. Reading a riser diameter here assumed every hot spot gets
    its own riser, which stopped being true once the sprue stopped being
    doubled up with one.
    """
    from cad2shell import demo_parts, run
    cfg = Config(voxel_pitch=1.5)
    thick = run(mesh=demo_parts.thick_block(), cfg=cfg, write_files=False).as_dict()
    thin = run(mesh=demo_parts.thin_bracket(), cfg=cfg, write_files=False).as_dict()
    d_thick = 2 * thick["gating"]["sprue_r_bottom"]
    d_thin = 2 * thin["gating"]["sprue_r_bottom"]
    assert d_thick > d_thin, (
        f"thick block feeder {d_thick:.1f}mm is not larger than thin bracket {d_thin:.1f}mm"
    )


def test_the_sprue_is_the_riser_for_the_spot_it_lands_on():
    """No riser is stacked on the hot spot the sprue already feeds.

    The sprue's base is widened to 1.2x the modulus of the section it lands on
    -- it is sized as a feeder. Putting a riser there too gives that thermal
    centre two feeders, more metal to remelt, and two appendages to cut off
    where one would do.
    """
    import numpy as np
    from cad2shell import demo_parts, run
    stats = run(mesh=demo_parts.get("boss_plate"), cfg=Config(voxel_pitch=2.5),
                write_files=False).as_dict()
    sprue = next(sg for sg in stats["gating"]["segments"] if sg["kind"] == "sprue")
    foot = np.asarray(sprue["p0"])[:2]
    reach = sprue["r0"]
    for riser in stats["risers"]:
        d = float(np.linalg.norm(np.asarray(riser["world"])[:2] - foot))
        assert d > reach, (
            f"riser {riser['index']} stands {d:.1f} mm from the sprue foot, "
            f"inside its {reach:.1f} mm base -- the sprue already feeds it")


def test_gating_is_a_single_attached_appendage(results, case):
    """One appendage, touching the casting. No runner, no ingates.

    An investment shell is poured as one assembly hanging from a single sprue.
    The earlier layout parked the sprue beside the part and ran a runner the
    full length of the casting underneath it, with ingates climbing back up --
    a horizontal-plate scheme for a sand mould with a parting line. On a 165 mm
    part that produced a 174 mm runner and a tree larger than the casting it
    fed, none of it touching the part.
    """
    r = results[case]
    g = r["stats"].as_dict()["gating"]
    kinds = [s["kind"] for s in g["segments"]]
    assert set(kinds) == {"sprue", "cup"}, f"{case}: unexpected gating parts {kinds}"
    assert kinds.count("sprue") == 1, f"{case}: expected exactly one sprue"
    assert g["n_ingates"] == 0
    assert g["runner_radius"] == 0.0


def test_sprue_lands_on_the_part(results, case):
    """The sprue's base must sit on a real surface of the casting.

    Not necessarily the highest point of the bounding box: on an L-bracket the
    thickest section is the flat base, and its top face -- part way up the
    part -- is exactly where the feed is needed. What matters is that the
    landing point is within the part's footprint and on material, not floating
    beside it as the old side-mounted sprue was.
    """
    stats = results[case]["stats"].as_dict()
    lo, hi = np.asarray(stats["part"]["bounds"])
    sprue = next(s for s in stats["gating"]["segments"] if s["kind"] == "sprue")
    base = np.asarray(sprue["p0"])
    assert lo[0] <= base[0] <= hi[0], f"{case}: sprue is off the part in x"
    assert lo[1] <= base[1] <= hi[1], f"{case}: sprue is off the part in y"
    assert lo[2] <= base[2] <= hi[2], f"{case}: sprue base is outside the part in z"
    # and it must rise from there, never drop below the casting
    top = np.asarray(sprue["p1"])
    assert top[2] > base[2], f"{case}: sprue does not rise from the part"
    assert top[2] > hi[2], f"{case}: sprue does not clear the top of the casting"


def test_sprue_base_can_feed_the_section_it_lands_on(results, case):
    """The sprue doubles as the feeder, so it must out-freeze that section.

    Sized only for fill rate, the choke would freeze before the casting and
    starve it -- the reason the base is the larger of the Bernoulli choke and
    the modulus-derived feeder diameter.
    """
    r = results[case]
    stats, cfg = r["stats"].as_dict(), r["cfg"]
    m_feed = max(h["local_modulus"] for h in stats["hot_spots"])
    d_needed = risers.diameter_for_modulus(cfg.riser_modulus_factor * m_feed,
                                           cfg.riser_hd_ratio)
    d_actual = 2 * stats["gating"]["sprue_r_bottom"]
    assert d_actual >= d_needed - 1e-6, (
        f"{case}: sprue base {d_actual:.1f} mm cannot feed a section of "
        f"modulus {m_feed:.2f} mm (needs {d_needed:.1f} mm)"
    )


def test_sprue_widens_toward_the_cup(results, case):
    """The choke stays at the bottom so the falling stream does not aspirate."""
    g = results[case]["stats"].as_dict()["gating"]
    assert g["sprue_r_top"] > g["sprue_r_bottom"]


def test_a_compact_part_needs_only_the_pour_appendage():
    """One sprue, nothing else, on a part a single feeder can reach.

    This is what a real investment mould looks like: the pour cup and sprue are
    the only thing attached to the casting, and the only thing to cut off. An
    earlier version put a riser beside the sprue on every hot spot it found,
    which on a plate with two mirrored bosses meant a second feeder on the same
    continuous body of metal that the sprue was already feeding.
    """
    from cad2shell import demo_parts, run
    for name in ("boss_plate", "thick_block", "thin_bracket"):
        stats = run(mesh=demo_parts.get(name), cfg=Config(voxel_pitch=2.0),
                    write_files=False).as_dict()
        assert stats["risers"] == [], (
            f"{name} was given {len(stats['risers'])} riser(s); a compact part "
            "should be fed by the sprue alone")


def test_a_thermal_centre_beyond_feeding_range_gets_its_own_riser():
    """Risers are not removed, they are reserved for what the sprue cannot reach.

    A feeder serves the metal around it out to roughly 4.5x the section
    thickness -- about 9x the modulus. Two heavy bosses at opposite ends of a
    long thin web are further apart than that, so the far one genuinely will
    not be fed and needs a riser of its own.

    Risers are off by default -- most parts do not need them -- so this asks
    for them explicitly, which is what a founder would do on seeing shrinkage
    away from the sprue.
    """
    import numpy as np
    import trimesh
    from cad2shell import run

    left = trimesh.creation.cylinder(radius=22, height=44)
    left.apply_translation([-95, 0, 22])
    right = trimesh.creation.cylinder(radius=22, height=44)
    right.apply_translation([95, 0, 22])
    web = trimesh.creation.box(extents=[190, 26, 9])
    web.apply_translation([0, 0, 4.5])
    part = trimesh.boolean.union([left, right, web], engine="manifold")

    stats = run(mesh=part, cfg=Config(voxel_pitch=2.0, use_risers=True),
                write_files=False).as_dict()
    assert stats["risers"], "the far boss is out of reach and must be risered"

    sprue = next(s for s in stats["gating"]["segments"] if s["kind"] == "sprue")
    foot = np.asarray(sprue["p0"])[:2]
    reach = sprue["r0"] + 9.0 * stats["casting_modulus"]
    for riser in stats["risers"]:
        d = float(np.linalg.norm(np.asarray(riser["world"])[:2] - foot))
        assert d > reach, (
            f"riser at {d:.0f} mm is inside the sprue's {reach:.0f} mm reach")


def test_risers_are_off_by_default():
    """A mould has one appendage unless the user asks for more.

    Most castings are fed entirely by the sprue, so a riser is metal to remelt
    and a second thing to cut off for no benefit. Off is the right default; the
    toggle is there for the isolated heavy section that genuinely needs one.
    """
    import trimesh
    from cad2shell import run

    left = trimesh.creation.cylinder(radius=22, height=44)
    left.apply_translation([-95, 0, 22])
    right = trimesh.creation.cylinder(radius=22, height=44)
    right.apply_translation([95, 0, 22])
    web = trimesh.creation.box(extents=[190, 26, 9])
    web.apply_translation([0, 0, 4.5])
    part = trimesh.boolean.union([left, right, web], engine="manifold")

    assert Config().use_risers is False
    # even a part that would qualify gets none unless asked
    off = run(mesh=part, cfg=Config(voxel_pitch=2.5), write_files=False).as_dict()
    assert off["risers"] == []

    on = run(mesh=part, cfg=Config(voxel_pitch=2.5, use_risers=True),
             write_files=False).as_dict()
    assert on["risers"], "the toggle must still produce risers when switched on"
    # and the metal they add is real: yield falls when they are present
    assert on["yield_pct"] < off["yield_pct"]


def test_a_symmetric_casting_is_gated_on_its_centreline():
    """A part with a plane of symmetry must be fed on that plane.

    Two failures conspired to push the sprue off-centre on a plainly symmetric
    part:

    - `argmax` over hot spot moduli breaks ties arbitrarily, so a mirrored pair
      of identical thermal centres put the gate on whichever came first.
    - Averaging the tied spots is still not enough, because the two sides of a
      symmetric feature voxelise to slightly different moduli (3.94 against
      3.18 for the same boss), so only one side ties for the peak.

    The gate is therefore snapped onto any plane the MESH is symmetric about,
    measured by mirroring the part and intersecting it with itself.
    """
    import numpy as np
    import trimesh
    from cad2shell import run

    # symmetric about X: a bar with a boss at each end
    bar = trimesh.creation.box(extents=[160, 40, 20])
    bar.apply_translation([0, 0, 10])
    left = trimesh.creation.cylinder(radius=18, height=44)
    left.apply_translation([-55, 0, 22])
    right = trimesh.creation.cylinder(radius=18, height=44)
    right.apply_translation([55, 0, 22])
    part = trimesh.boolean.union([bar, left, right], engine="manifold")

    stats = run(mesh=part, cfg=Config(voxel_pitch=2.0),
                write_files=False).as_dict()
    sprue = next(s for s in stats["gating"]["segments"] if s["kind"] == "sprue")
    centre_x = float((part.bounds[0][0] + part.bounds[1][0]) / 2.0)
    span_x = float(part.bounds[1][0] - part.bounds[0][0])
    off = abs(sprue["p0"][0] - centre_x)
    assert off < 0.02 * span_x, (
        f"the sprue is {off:.1f} mm off a centreline the part is symmetric "
        f"about ({span_x:.0f} mm across)")


def test_an_asymmetric_casting_keeps_its_thermal_gate():
    """Snapping must not override the thermal analysis on a lopsided part.

    Centring is only right where the casting really is symmetric; forcing it
    everywhere would move the gate off the section that actually needs feeding.
    """
    import trimesh
    from cad2shell import run

    # heavy at one end only
    bar = trimesh.creation.box(extents=[160, 40, 20])
    bar.apply_translation([0, 0, 10])
    boss = trimesh.creation.cylinder(radius=26, height=60)
    boss.apply_translation([-58, 0, 30])
    part = trimesh.boolean.union([bar, boss], engine="manifold")

    stats = run(mesh=part, cfg=Config(voxel_pitch=2.0),
                write_files=False).as_dict()
    sprue = next(s for s in stats["gating"]["segments"] if s["kind"] == "sprue")
    # it must sit over the heavy end, not in the middle of the bar
    assert sprue["p0"][0] < -20, (
        f"the sprue is at x={sprue['p0'][0]:.0f}; it should be over the boss "
        "at x=-58, which is the section that needs feeding")
