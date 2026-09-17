"""Multi-cavity moulds: N copies of one part in a single shell.

The invariants that matter here are the ones a foundry would check on the wax
tree before investing it: every instance is actually present, none of them
touch, they are all fed from one sprue by an equal-length path, and the shell
that wraps the lot is still a single watertight solid.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import trimesh

from cad2shell import Config, demo_parts, layout, run

# A coarse pitch keeps this suite quick; the geometry questions asked here are
# about placement and topology, not about sub-millimetre wall accuracy.
PITCH = 2.5


@pytest.fixture(scope="module")
def quad(tmp_path_factory):
    """One four-up grid mould, shared by the tests that only read from it."""
    cfg = Config(voxel_pitch=PITCH, shell_thickness=6.0, quantity=4)
    out = tmp_path_factory.mktemp("quad")
    return run(mesh=demo_parts.get("boss_plate"), out_prefix=str(out / "q4"), cfg=cfg), cfg


@pytest.fixture(scope="module")
def single(tmp_path_factory):
    cfg = Config(voxel_pitch=PITCH, shell_thickness=6.0)
    out = tmp_path_factory.mktemp("single")
    return run(mesh=demo_parts.get("boss_plate"), out_prefix=str(out / "q1"), cfg=cfg), cfg


# -- placement -------------------------------------------------------------

def test_quantity_one_is_the_untouched_single_part_path(single):
    """quantity=1 must not perturb the existing behaviour at all.

    The multi-cavity code is a new branch, and the cheapest way for it to do
    damage is to change the answer for everyone who never asked for it.
    """
    stats, _ = single
    assert stats.layout["quantity"] == 1
    assert len(stats.layout["instances"]) == 1
    # a single part is gated by the sprue directly: no runner, no ingates
    assert stats.gating["n_ingates"] == 0
    kinds = {s["kind"] for s in stats.gating["segments"]}
    assert kinds == {"sprue", "cup"}


def test_every_instance_is_present_in_the_casting(quad, single):
    """Four parts means four times the metal, to within voxel noise."""
    quad_stats, _ = quad
    one_stats, _ = single
    each = quad_stats.layout["part_volume_each"]
    total = quad_stats.layout["part_volume_total"]
    assert quad_stats.layout["quantity"] == 4
    assert len(quad_stats.layout["instances"]) == 4
    # exact: the instances are rigid copies, so the volume is exactly 4x
    assert total == pytest.approx(4.0 * each, rel=1e-9)
    # and `each` really is the part we started from
    assert each == pytest.approx(one_stats.part_volume, rel=1e-9)


def test_the_combined_mesh_is_four_separate_bodies(quad):
    """The cluster is N disjoint solids, not one merged blob."""
    stats, cfg = quad
    part = demo_parts.get("boss_plate")
    from cad2shell import ingest
    lay = layout.build(ingest.from_mesh(part, cfg), cfg)
    bodies = lay.mesh.split(only_watertight=False)
    assert len(bodies) == 4
    for b in bodies:
        assert b.is_watertight


def test_instances_do_not_overlap(quad):
    """Two cavities that intersect are one cavity, and the casting is scrap."""
    stats, cfg = quad
    from cad2shell import ingest
    lay = layout.build(ingest.from_mesh(demo_parts.get("boss_plate"), cfg), cfg)
    boxes = [b.bounds for b in lay.mesh.split(only_watertight=False)]
    for a, b in itertools.combinations(boxes, 2):
        # AABBs must be disjoint on at least one axis
        disjoint = any(a[1][k] <= b[0][k] or b[1][k] <= a[0][k] for k in range(3))
        assert disjoint, "two instances share space"


def test_spacing_leaves_room_for_two_shell_walls(quad):
    """Neighbours closer than two shell thicknesses would fuse their shells.

    Each cavity carries its own ceramic wall. If the gap between two instances
    is under 2*shell_thickness there is no ceramic between them at all -- the
    two shells merge and the castings come out joined.
    """
    stats, cfg = quad
    gap = stats.layout["spacing"]
    assert gap >= 2.0 * cfg.shell_thickness


def test_explicit_spacing_is_honoured(tmp_path):
    cfg = Config(voxel_pitch=PITCH, quantity=4, part_spacing=25.0)
    from cad2shell import ingest
    lay = layout.build(ingest.from_mesh(demo_parts.get("boss_plate"), cfg), cfg)
    assert lay.spacing == pytest.approx(25.0)
    bodies = sorted(lay.mesh.split(only_watertight=False),
                    key=lambda m: (m.bounds[0][1], m.bounds[0][0]))
    # the two instances in the first row are one clear gap apart in X
    a, b = bodies[0].bounds, bodies[1].bounds
    assert b[0][0] - a[1][0] == pytest.approx(25.0, abs=1e-6)


# -- gating ----------------------------------------------------------------

def test_one_sprue_feeds_every_instance(quad):
    """A single pour point, and one ingate landing on each casting."""
    stats, _ = quad
    segs = stats.gating["segments"]
    kinds = [s["kind"] for s in segs]
    assert kinds.count("sprue") == 1, "a mould has exactly one sprue"
    assert kinds.count("cup") == 1
    assert kinds.count("ingate") == 4
    assert kinds.count("runner") == 4
    assert stats.gating["n_ingates"] == 4
    # every instance is served exactly once
    fed = sorted(s["instance"] for s in segs if s["kind"] == "ingate")
    assert fed == [0, 1, 2, 3]


def test_radial_feed_paths_are_equal_length(tmp_path_factory):
    """A radial cluster gives every cavity an identical path from the choke.

    Equal paths are what make N castings from one pour come out alike, and a
    circle is the only arrangement that can promise them: every instance sits
    the same distance from the axis and is rotated to present the same face, so
    the gate site found on one is the same site on all of them.
    """
    cfg = Config(voxel_pitch=PITCH, shell_thickness=6.0, quantity=4,
                 layout="radial")
    out = tmp_path_factory.mktemp("radial_paths")
    stats = run(mesh=demo_parts.get("boss_plate"),
                out_prefix=str(out / "r4"), cfg=cfg)
    lengths = [s["length"] for s in stats.gating["segments"]
               if s["kind"] == "runner"]
    assert len(lengths) == 4
    assert max(lengths) == pytest.approx(min(lengths), rel=1e-6)


def test_grid_feed_paths_stay_within_a_sane_spread(quad):
    """A raster cannot equalise its paths, but it must not be wild either.

    Instances on a rectangle sit at genuinely different distances from a
    central sprue -- that is the arrangement's own geometry, not a defect, and
    it is the reason `radial` exists. What matters is that the spread stays
    bounded: the corner cavity should not be fed through a runner several times
    the length of the near one, or it will still be filling when the near one
    has frozen.
    """
    stats, _ = quad
    lengths = [s["length"] for s in stats.gating["segments"]
               if s["kind"] == "runner"]
    assert len(lengths) == 4
    assert max(lengths) <= 2.0 * min(lengths), \
        f"grid runner lengths vary too widely: {[round(x, 1) for x in lengths]}"


def test_ingates_land_on_the_castings(quad):
    """An ingate ending in mid-air feeds nothing.

    Each ingate's lower end must sit on or inside its own instance, so the
    metal actually enters the cavity.
    """
    stats, cfg = quad
    from cad2shell import ingest
    lay = layout.build(ingest.from_mesh(demo_parts.get("boss_plate"), cfg), cfg)
    bodies = lay.mesh.split(only_watertight=False)
    tips = [np.asarray(s["p1"]) for s in stats.gating["segments"] if s["kind"] == "ingate"]
    assert len(tips) == 4
    for tip in tips:
        # the tip lies within the XY footprint of exactly one instance, at or
        # below that instance's top surface
        hits = [b for b in bodies
                if b.bounds[0][0] - 1e-6 <= tip[0] <= b.bounds[1][0] + 1e-6
                and b.bounds[0][1] - 1e-6 <= tip[1] <= b.bounds[1][1] + 1e-6]
        assert len(hits) == 1, f"ingate at {tip} is over {len(hits)} instances"
        assert tip[2] <= hits[0].bounds[1][2] + 1e-6


def test_choke_passes_the_whole_cluster(quad, single):
    """Four castings need more metal through the choke than one does.

    A_choke scales with the poured mass, so the four-up choke must be larger
    than the single-part one -- if it is not, the multi-cavity plan is sizing
    itself on one part and the mould will not fill in the target time.
    """
    quad_stats, _ = quad
    one_stats, _ = single
    assert quad_stats.gating["metal_volume"] > 3.5 * one_stats.gating["metal_volume"]


def test_gating_ratio_is_respected_across_the_branches(quad):
    """Total runner and ingate areas follow the configured non-pressurised ratio."""
    stats, cfg = quad
    rc, rr, rg = cfg.gating_ratio
    a_choke = stats.gating["choke_area"]
    n = stats.gating["n_ingates"]
    total_runner = n * math.pi * stats.gating["runner_radius"] ** 2
    total_ingate = n * math.pi * stats.gating["ingate_radius"] ** 2
    assert total_runner == pytest.approx(a_choke * rr / rc, rel=1e-6)
    assert total_ingate == pytest.approx(a_choke * rg / rc, rel=1e-6)


# -- the shell itself ------------------------------------------------------

def test_multi_cavity_shell_is_one_watertight_solid(quad):
    """The whole point: N cavities, ONE mould."""
    stats, _ = quad
    assert stats.shell["watertight"], "multi-cavity shell leaks"
    assert stats.shell["volume"] > 0


def test_shell_wall_holds_its_target_with_four_cavities(quad):
    """Adding cavities must not thin the ceramic between them."""
    stats, cfg = quad
    measured = stats.shell["thickness_measured"]["median"]
    tol = cfg.voxel_pitch + 0.15 * cfg.shell_thickness
    assert abs(measured - cfg.shell_thickness) <= tol


def test_yield_improves_with_quantity(quad, single):
    """Sharing one sprue between four parts is why anyone runs a multi-up tree.

    The gating metal is very nearly fixed, so spreading it over four castings
    must raise the yield. If it does not, the multi-cavity tree is generating
    four times the gating too and there is no reason to use it.
    """
    quad_stats, _ = quad
    one_stats, _ = single
    assert quad_stats.yield_pct > one_stats.yield_pct
    assert 0 < quad_stats.yield_pct <= 100


def test_exported_multi_cavity_shell_is_a_valid_solid(quad, tmp_path):
    stats, _ = quad
    mesh = trimesh.load(stats.files["shell_stl"], force="mesh")
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_no_spurious_assembly_warning_for_a_multi_up_mould(quad):
    """N bodies is what was asked for, so it must not be reported as a mistake."""
    stats, _ = quad
    assert not any("assembly rather than a single casting" in w
                   for w in stats.warnings)


# -- radial layout ---------------------------------------------------------

def test_radial_layout_places_instances_on_a_circle(tmp_path):
    cfg = Config(voxel_pitch=PITCH, quantity=6, layout="radial")
    from cad2shell import ingest
    lay = layout.build(ingest.from_mesh(demo_parts.get("boss_plate"), cfg), cfg)
    assert lay.arrangement == "radial"
    centres = lay.centres
    axis = centres[:, :2].mean(axis=0)
    radii = np.linalg.norm(centres[:, :2] - axis, axis=1)
    # equidistant from the cluster axis, which is what makes the feed paths equal
    assert radii.max() == pytest.approx(radii.min(), rel=1e-6)
    angles = sorted(math.degrees(i.angle) for i in lay.instances)
    assert angles == pytest.approx([0, 60, 120, 180, 240, 300], abs=1e-6)


def test_radial_instances_clear_each_other(tmp_path):
    cfg = Config(voxel_pitch=PITCH, quantity=5, layout="radial")
    from cad2shell import ingest
    lay = layout.build(ingest.from_mesh(demo_parts.get("boss_plate"), cfg), cfg)
    bodies = lay.mesh.split(only_watertight=False)
    assert len(bodies) == 5
    for a, b in itertools.combinations(bodies, 2):
        gap = trimesh.proximity.ProximityQuery(a).signed_distance(
            b.vertices[:: max(1, len(b.vertices) // 200)])
        assert gap.max() < 0, "radial instances intersect"


# -- configuration ---------------------------------------------------------

def test_grid_shape_is_the_squarest_arrangement():
    """A compact cluster keeps the runners short; a long row wastes metal."""
    assert layout.grid_shape(4) == (2, 2)
    assert layout.grid_shape(6) == (2, 3)
    assert layout.grid_shape(9) == (3, 3)
    assert layout.grid_shape(1) == (1, 1)


def test_explicit_columns_override_the_default():
    assert layout.grid_shape(6, columns=2) == (3, 2)
    assert layout.grid_shape(6, columns=6) == (1, 6)


@pytest.mark.parametrize("bad", [0, -1])
def test_quantity_must_be_positive(bad):
    with pytest.raises(ValueError, match="quantity"):
        Config(quantity=bad)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="layout"):
        Config(layout="spiral")


# -- the two steps of shell creation ---------------------------------------
#
# Building a mould is two separable steps:
#
#   1. the ceramic around the casting -- the negative space of the CAD
#   2. the feed system -- sprue, runners, ingates, and the cup opening
#
# Step 1 is the same work whatever the quantity: N disjoint instances on one
# grid offset exactly as each would alone. Step 2 is the only thing multi-
# cavity actually changes, from "a sprue on the casting" to "a sprue, a runner
# per instance and an ingate into each".
#
# These tests pin that split, because it is what keeps quantity from having to
# thread through the whole pipeline.

def test_cavity_is_identical_per_instance(tmp_path):
    """Step 1 scales exactly with quantity and changes in no other way.

    If this fails, the cavities are interacting -- either the instances are
    close enough that their shells have merged, or the offset is being applied
    to something other than the bare casting.
    """
    import numpy as np
    from cad2shell import hotspots, ingest, synthesis
    from cad2shell.voxel import Grid

    def cavity_volume(part, cfg):
        base = Grid.from_bounds(part.bounds[0], part.bounds[1],
                                cfg.voxel_pitch, cfg.grid_pad)
        pad = int(np.ceil(2 * cfg.shell_thickness / cfg.voxel_pitch)) + 3
        grid = base.expanded(np.array([pad] * 3), np.array([pad] * 3))
        occ = hotspots.voxelize_fast(part.mesh, grid)
        cavity = synthesis.build_cavity(grid, occ, cfg)
        return float(cavity.sum()) * grid.voxel_volume

    mesh = demo_parts.get("boss_plate")
    one_cfg = Config(voxel_pitch=2.0, shell_thickness=6.0)
    one = cavity_volume(ingest.from_mesh(mesh, one_cfg), one_cfg)

    for n in (2, 4, 6):
        cfg = Config(voxel_pitch=2.0, shell_thickness=6.0, quantity=n)
        lay = layout.build(ingest.from_mesh(mesh, cfg), cfg)
        many = cavity_volume(ingest.from_mesh(lay.mesh, cfg), cfg)
        assert many == pytest.approx(n * one, rel=1e-9), (
            f"{n}-up cavity is not {n} copies of the one-up cavity; "
            "the instances' shells are interacting")


def test_only_the_feed_system_changes_with_quantity(quad, single):
    """Step 2 is where multi-cavity lives, and step 1 is untouched.

    The single-part tree is a sprue landing on the casting. The four-up tree is
    a sprue, four runners and four ingates. The shell wall itself is the same
    ceramic in both.
    """
    quad_stats, quad_cfg = quad
    one_stats, one_cfg = single

    one_kinds = {s["kind"] for s in one_stats.gating["segments"]}
    quad_kinds = {s["kind"] for s in quad_stats.gating["segments"]}
    assert one_kinds == {"sprue", "cup"}
    assert quad_kinds == {"sprue", "cup", "runner", "ingate"}

    # step 1 is unchanged: the wall target and what was achieved both hold
    assert quad_cfg.shell_thickness == one_cfg.shell_thickness
    tol = quad_cfg.voxel_pitch + 0.15 * quad_cfg.shell_thickness
    for st in (one_stats, quad_stats):
        assert abs(st.shell["thickness_measured"]["median"]
                   - st.shell["thickness_target"]) <= tol


def test_the_mould_grows_to_hold_every_instance(quad, single):
    """A four-up mould must actually be bigger than a one-up mould.

    Stats agreeing while the exported shell stayed one-up sized would mean the
    cluster never reached synthesis -- so this measures the STL on disk, not
    the numbers the pipeline reports about itself.
    """
    quad_stats, _ = quad
    one_stats, _ = single
    one = trimesh.load(one_stats.files["shell_stl"])
    many = trimesh.load(quad_stats.files["shell_stl"])

    one_xy = (one.bounds[1] - one.bounds[0])[:2]
    many_xy = (many.bounds[1] - many.bounds[0])[:2]
    assert (many_xy > one_xy * 1.5).all(), (
        f"four-up shell is {many_xy.round(1)} mm against a one-up "
        f"{one_xy.round(1)} mm -- the cluster did not reach synthesis")
    assert many.volume > 2.0 * one.volume


# -- gates must land on the casting ----------------------------------------

@pytest.fixture(scope="module")
def bored_part():
    """A plummer-block-like casting: a block with a bore through its middle.

    The demo parts are all solid at their centroids, which is why gating into a
    hole went unnoticed -- every test part happened to have metal exactly where
    the naive gate site pointed. This one does not.
    """
    block = trimesh.creation.box(extents=[80, 40, 50])
    bore = trimesh.creation.cylinder(radius=14, height=60)
    bore.apply_translation([0, 0, 10])
    return trimesh.boolean.difference([block, bore], engine="manifold")


@pytest.mark.parametrize("arrangement", ["grid", "radial"])
def test_ingates_land_on_metal_not_in_a_bore(bored_part, arrangement, tmp_path):
    """Every ingate must end on the casting's top surface.

    The failure this guards against is specific and was visible in the viewer:
    the gate site was the instance's bounding-box centroid, which for a bored
    part is empty air. The ray cast down that column hit the *inside* of the
    bore, so the check for "did we hit the part" passed while the ingate was
    being run down a hole -- feeding the mould through the bore rather than
    onto the casting.
    """
    from cad2shell import gating, hotspots, ingest, pipeline
    from cad2shell.voxel import Grid

    cfg = Config(voxel_pitch=2.0, shell_thickness=5.0, quantity=4,
                 layout=arrangement)
    lay = layout.build(ingest.from_mesh(bored_part, cfg), cfg)
    part = ingest.from_mesh(lay.mesh, cfg)

    lo, hi = pipeline._tree_bounds(part, cfg)
    base = Grid.from_bounds(part.bounds[0], part.bounds[1],
                            cfg.voxel_pitch, cfg.grid_pad)
    pad_lo = np.maximum(np.ceil((base.origin - lo) / cfg.voxel_pitch), 0).astype(int)
    far = base.origin + np.asarray(base.shape) * cfg.voxel_pitch
    pad_hi = np.maximum(np.ceil((hi - far) / cfg.voxel_pitch), 0).astype(int)
    grid = base.expanded(pad_lo, pad_hi)
    occ = hotspots.voxelize_fast(part.mesh, grid)
    spots = hotspots.detect(part, grid, cfg, occ=occ)
    plan = gating.design_multi(part, lay, cfg, hotspots=spots)

    tips = [np.asarray(s["p1"]) for s in
            (seg.as_dict() for seg in plan.segments) if s["kind"] == "ingate"]
    assert len(tips) == 4

    eps = 0.05
    top_z = float(part.bounds[1][2])
    for tip in tips:
        # metal immediately below the landing point
        probe = np.array([[tip[0], tip[1], tip[2] + eps]])
        assert bool(part.mesh.contains(probe)[0]), \
            f"ingate tip {tip.round(1)} is not on solid metal"
        # and it is the casting's upper surface, not the floor of the bore
        assert tip[2] > top_z - cfg.shell_thickness - 2 * cfg.voxel_pitch, \
            (f"ingate tip {tip.round(1)} sits {top_z - tip[2]:.1f} mm below the "
             "casting top -- it is being fed down a hole")


def test_single_part_sprue_also_avoids_a_bore(bored_part, tmp_path):
    """The one-up path has the same hazard and the same fix.

    Its sprue lands on the surface above the heaviest hot spot, and on a bored
    part that column can break out into the bore.
    """
    from cad2shell import gating, hotspots, ingest, pipeline
    from cad2shell.voxel import Grid

    cfg = Config(voxel_pitch=2.0, shell_thickness=5.0)
    part = ingest.from_mesh(bored_part, cfg)
    lo, hi = pipeline._tree_bounds(part, cfg)
    base = Grid.from_bounds(part.bounds[0], part.bounds[1],
                            cfg.voxel_pitch, cfg.grid_pad)
    pad_lo = np.maximum(np.ceil((base.origin - lo) / cfg.voxel_pitch), 0).astype(int)
    far = base.origin + np.asarray(base.shape) * cfg.voxel_pitch
    pad_hi = np.maximum(np.ceil((hi - far) / cfg.voxel_pitch), 0).astype(int)
    grid = base.expanded(pad_lo, pad_hi)
    occ = hotspots.voxelize_fast(part.mesh, grid)
    spots = hotspots.detect(part, grid, cfg, occ=occ)
    plan = gating.design(part, cfg, hotspots=spots)

    sprue = next(s for s in plan.segments if s.kind == "sprue")
    probe = np.array([[sprue.p0[0], sprue.p0[1], sprue.p0[2] - 0.05]])
    assert bool(part.mesh.contains(probe)[0]), \
        "the sprue does not land on solid metal"


def test_a_bored_part_still_produces_a_watertight_multi_mould(bored_part, tmp_path):
    """The end-to-end result on the shape that exposed the bug."""
    cfg = Config(voxel_pitch=2.0, shell_thickness=5.0, quantity=4)
    stats = run(mesh=bored_part, out_prefix=str(tmp_path / "bored"), cfg=cfg)
    assert stats.shell["watertight"]
    assert stats.gating["n_ingates"] == 4
    assert trimesh.load(stats.files["shell_stl"]).is_watertight


# -- hand-made arrangements ------------------------------------------------
#
# The viewer's gizmo produces an explicit placement per instance. These pin the
# path it feeds: the arrangement must reach the build unchanged, and the gating
# must adapt to it rather than to the raster it replaced.

def test_placements_override_the_automatic_layout():
    from cad2shell import ingest
    pl = [{"x": 0, "y": 0, "rot": 0}, {"x": 130, "y": 0, "rot": 90},
          {"x": 0, "y": 110, "rot": 180}, {"x": 130, "y": 110, "rot": 270}]
    cfg = Config(voxel_pitch=PITCH, placements=pl)
    # the arrangement defines the quantity; nothing else needs to agree
    assert cfg.quantity == 4
    lay = layout.build(ingest.from_mesh(demo_parts.get("boss_plate"), cfg), cfg)
    assert lay.arrangement == "custom"
    assert len(lay.instances) == 4
    assert len(lay.mesh.split(only_watertight=False)) == 4


def test_a_placement_rotates_about_the_parts_own_centre():
    """Turning a part in place must not also fling it across the plate.

    Rotating about the world origin instead is the classic version of this bug:
    the part spins into a different position entirely, and the arrangement the
    user made on screen is not the one that gets built.
    """
    from cad2shell import ingest
    base = ingest.from_mesh(demo_parts.get("boss_plate"), Config())
    centre0 = (base.bounds[0] + base.bounds[1]) / 2.0

    cfg = Config(voxel_pitch=PITCH, placements=[{"x": 0, "y": 0, "rot": 90}])
    lay = layout.build(base, cfg)
    centre1 = lay.instances[0].centre
    assert centre1[:2] == pytest.approx(centre0[:2], abs=1e-6), \
        "rotation moved the part instead of turning it in place"


def test_tiers_stack_clear_of_each_other():
    """A tier lifted in Z must not touch the tier below.

    Stacking a packed tier up the sprue is the one thing vertical placement is
    for, and it is only worth anything if the two tiers stay separate solids
    with ceramic between them.
    """
    from cad2shell import ingest
    base = ingest.from_mesh(demo_parts.get("boss_plate"), Config())
    height = float(base.extents[2])
    cfg0 = Config(shell_thickness=6.0, voxel_pitch=PITCH)
    step = height + layout.default_spacing(cfg0)

    pl = [{"x": 0, "y": 0, "z": 0, "rot": 0},
          {"x": 0, "y": 0, "z": step, "rot": 0}]
    cfg = Config(voxel_pitch=PITCH, shell_thickness=6.0, placements=pl)
    lay = layout.build(base, cfg)
    bodies = sorted(lay.mesh.split(only_watertight=False),
                    key=lambda m: m.bounds[0][2])
    assert len(bodies) == 2
    clear = bodies[1].bounds[0][2] - bodies[0].bounds[1][2]
    assert clear >= 2 * cfg.shell_thickness, \
        f"tiers are only {clear:.1f} mm apart; their shells would merge"


def test_gating_follows_a_hand_arrangement(tmp_path):
    """Packing parts inward must actually shorten the runners.

    If the sprue stayed where the automatic layout put it, a hand-packed
    cluster would be fed from a point that no longer has anything near it --
    and rearranging would buy nothing.
    """
    mesh = demo_parts.get("boss_plate")
    wide = [{"x": -160, "y": 0}, {"x": 160, "y": 0},
            {"x": 0, "y": -160}, {"x": 0, "y": 160}]
    tight = [{"x": -55, "y": 0}, {"x": 55, "y": 0},
             {"x": 0, "y": -75}, {"x": 0, "y": 75}]

    lengths = {}
    for name, pl in (("wide", wide), ("tight", tight)):
        cfg = Config(voxel_pitch=PITCH, shell_thickness=6.0, placements=pl)
        stats = run(mesh=mesh, out_prefix=str(tmp_path / name), cfg=cfg,
                    write_files=False)
        lengths[name] = sum(s["length"] for s in stats.gating["segments"]
                            if s["kind"] == "runner")
    assert lengths["tight"] < lengths["wide"], \
        "packing the parts closer did not shorten the feed system"


def test_a_hand_arrangement_still_builds_a_watertight_mould(tmp_path):
    pl = [{"x": -60, "y": -50, "rot": 0}, {"x": 60, "y": -50, "rot": 90},
          {"x": -60, "y": 50, "rot": 180}, {"x": 60, "y": 50, "rot": 270}]
    cfg = Config(voxel_pitch=PITCH, shell_thickness=6.0, placements=pl)
    stats = run(mesh=demo_parts.get("boss_plate"),
                out_prefix=str(tmp_path / "hand"), cfg=cfg)
    assert stats.shell["watertight"]
    assert stats.gating["n_ingates"] == 4
    assert trimesh.load(stats.files["shell_stl"]).is_watertight


def test_a_stacked_tier_raises_the_yield(tmp_path):
    """The reason to stack: one sprue feeding twice as many castings."""
    mesh = demo_parts.get("boss_plate")
    base = [{"x": -55, "y": -45}, {"x": 55, "y": -45},
            {"x": -55, "y": 45}, {"x": 55, "y": 45}]
    cfg1 = Config(voxel_pitch=PITCH, shell_thickness=6.0, placements=base)
    one = run(mesh=mesh, out_prefix=str(tmp_path / "t1"), cfg=cfg1,
              write_files=False)

    from cad2shell import ingest
    height = float(ingest.from_mesh(mesh, Config()).extents[2])
    step = height + layout.default_spacing(cfg1)
    two_tiers = [dict(p, z=0) for p in base] + [dict(p, z=step) for p in base]
    cfg2 = Config(voxel_pitch=PITCH, shell_thickness=6.0, placements=two_tiers)
    two = run(mesh=mesh, out_prefix=str(tmp_path / "t2"), cfg=cfg2,
              write_files=False)

    assert two.layout["quantity"] == 8
    assert two.yield_pct > one.yield_pct


def test_identical_instances_get_identical_risers():
    """Four copies of one part must be fed four identical ways.

    Two separate bugs produced uneven risers across a cluster, and both were
    invisible on a single part:

    - `max_risers` capped hot spots across the WHOLE grid rather than per
      casting, so a four-up mould detected the budget's worth of thermal
      centres and whichever instances came last got none.
    - Riser placement measured feeding distance from the central sprue, which
      touches no casting at all. Whether an instance got a riser then depended
      on where it happened to sit in the cluster.

    Together they gave four identical parts two risers, one, one and none.
    """
    import numpy as np
    import trimesh
    from collections import Counter

    # a dumbbell: two heavy bosses too far apart for one feeder to reach both,
    # so every instance genuinely needs risers and the count is worth checking
    left = trimesh.creation.cylinder(radius=22, height=44)
    left.apply_translation([-95, 0, 22])
    right = trimesh.creation.cylinder(radius=22, height=44)
    right.apply_translation([95, 0, 22])
    web = trimesh.creation.box(extents=[190, 26, 9])
    web.apply_translation([0, 0, 4.5])
    part = trimesh.boolean.union([left, right, web], engine="manifold")

    for n in (2, 4, 6):
        stats = run(mesh=part, cfg=Config(voxel_pitch=2.5, quantity=n,
                                          use_risers=True),
                    write_files=False).as_dict()
        centres = np.array([i["centre"][:2] for i in stats["layout"]["instances"]])
        per = Counter()
        for riser in stats["risers"]:
            d = np.linalg.norm(centres - np.asarray(riser["world"])[:2], axis=1)
            per[int(np.argmin(d))] += 1
        counts = [per.get(i, 0) for i in range(n)]
        assert len(set(counts)) == 1, (
            f"{n}-up mould gave risers {counts} across identical instances")


def test_hot_spots_are_found_on_every_instance():
    """The thermal-centre budget is per casting, not for the whole cluster."""
    import numpy as np
    from collections import Counter

    for n in (2, 4):
        stats = run(mesh=demo_parts.get("boss_plate"),
                    cfg=Config(voxel_pitch=2.5, quantity=n),
                    write_files=False).as_dict()
        centres = np.array([i["centre"][:2] for i in stats["layout"]["instances"]])
        per = Counter()
        for hs in stats["hot_spots"]:
            d = np.linalg.norm(centres - np.asarray(hs["world"])[:2], axis=1)
            per[int(np.argmin(d))] += 1
        counts = [per.get(i, 0) for i in range(n)]
        assert all(c > 0 for c in counts), (
            f"{n}-up mould found hot spots {counts}; some instances got none")
        assert len(set(counts)) == 1, (
            f"{n}-up mould found uneven hot spots {counts}")


def test_ingates_take_the_shortest_path_to_the_sprue():
    """A symmetric part offers two equally good landing faces; take the near one.

    Gate sites were ranked by height first, so a face a fraction of a
    millimetre higher on the FAR side of an instance beat one much closer on
    the near side. Every runner then crossed the whole cluster -- 116 mm each
    on a four-up plummer block where 91 mm reached an equally good face -- and
    all of that is metal to remelt.

    Compounding it, the ring search returned from the first radius that hit
    metal at all. On a bored part the near and far edges lie at different
    radii, so it could not compare them.
    """
    import numpy as np

    mesh = demo_parts.get("boss_plate")
    for arrangement in ("grid", "radial"):
        stats = run(mesh=mesh,
                    cfg=Config(voxel_pitch=3.0, quantity=4, layout=arrangement),
                    write_files=False).as_dict()
        runners = [s for s in stats["gating"]["segments"] if s["kind"] == "runner"]
        assert len(runners) == 4

        # every ingate must land on the half of its instance facing the sprue
        axis = np.array(stats["gating"]["_layout"]["axis_xy"]) \
            if "_layout" in stats["gating"] else np.zeros(2)
        centres = np.array([i["centre"][:2] for i in stats["layout"]["instances"]])
        for seg in (s for s in stats["gating"]["segments"] if s["kind"] == "ingate"):
            tip = np.asarray(seg["p1"])[:2]
            inst = centres[int(np.argmin(np.linalg.norm(centres - tip, axis=1)))]
            # the gate is no further from the sprue than the instance centre is
            assert np.linalg.norm(tip - axis) <= np.linalg.norm(inst - axis) + 1.0, (
                f"{arrangement}: the ingate landed on the far side of its "
                "instance, lengthening the runner for nothing")
