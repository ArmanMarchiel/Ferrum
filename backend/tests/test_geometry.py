"""Shell geometry invariants: watertight, positive volume, wall within tolerance."""
from __future__ import annotations

import pytest
import numpy as np


def test_shell_is_watertight(results, case):
    """A mould with a hole in it leaks. This is the headline invariant."""
    stats = results[case]["stats"].as_dict()
    assert stats["shell"]["watertight"], f"{case}: shell mesh is not watertight"


def test_exported_shell_stl_is_watertight(artifacts, case):
    """Re-loaded from disk, so the STL file itself is checked, not just memory."""
    shell = artifacts[case]["shell"]
    assert shell.is_watertight, f"{case}: exported shell STL is not watertight"
    assert shell.is_winding_consistent, f"{case}: shell winding is inconsistent"


def test_shell_volume_is_positive(results, case):
    stats = results[case]["stats"].as_dict()
    assert stats["shell"]["volume"] > 0
    # Outward-facing normals give a positive signed volume; a negative one
    # means the surface is inside-out and the STL would print as a void.
    assert results[case]["stats"].as_dict()["shell"]["volume"] > 0


def test_exported_shell_has_positive_signed_volume(artifacts, case):
    shell = artifacts[case]["shell"]
    assert shell.volume > 0, f"{case}: shell has negative signed volume (inverted normals)"


def test_wall_thickness_within_tolerance(results, case):
    """The ceramic wall must come out at the requested thickness.

    Tolerance is one voxel plus 15% of target. A rasterised wall can only be
    a whole number of cells thick, so one pitch of quantisation is inherent;
    the pitch is the lever that tightens it.
    """
    r = results[case]
    stats, cfg = r["stats"].as_dict(), r["cfg"]
    measured = stats["shell"]["thickness_measured"]["median"]
    target = cfg.shell_thickness
    tol = cfg.voxel_pitch + 0.15 * target
    assert abs(measured - target) <= tol, (
        f"{case}: wall {measured:.2f}mm vs target {target}mm (tol {tol:.2f}mm)"
    )


def test_shell_encloses_the_part(results, artifacts, case):
    """The mould's bounding box must contain the casting's, with room for the wall."""
    stats = results[case]["stats"].as_dict()
    part_lo, part_hi = np.asarray(stats["part"]["bounds"])
    shell = artifacts[case]["shell"]
    assert np.all(shell.bounds[0] <= part_lo + 1e-6), f"{case}: shell does not reach below the part"
    assert np.all(shell.bounds[1] >= part_hi - 1e-6), f"{case}: shell does not reach above the part"


def test_pour_cup_is_opened(results, case):
    """A sealed shell is watertight but unpourable; the cup must be cut open."""
    stats = results[case]["stats"].as_dict()
    cup = stats["shell"]["cup"]
    assert cup["opened"], f"{case}: pour cup was never opened"
    assert cup["cells_cleared"] > 0, f"{case}: cup opening removed no material"


def test_wall_tracks_the_requested_thickness():
    """A regression guard: the measured wall must follow the target, not snap.

    The first thickness measurement used the EDT of the skin, whose medial
    axis lands on whole voxels; twice that distance can only be an even
    multiple of the pitch, so a 5 mm request silently measured as 6.00 and
    every value quantised to 3/3/6/6/9. Sweeping several targets catches that
    class of bug, which a single-value test cannot.
    """
    from cad2shell import Config, demo_parts, run
    pitch = 1.0
    for target in (3.0, 4.0, 5.0, 6.0, 8.0):
        stats = run(mesh=demo_parts.thick_block(), write_files=False,
                    cfg=Config(shell_thickness=target, voxel_pitch=pitch)).as_dict()
        got = stats["shell"]["thickness_measured"]["median"]
        assert abs(got - target) <= pitch + 0.15 * target, (
            f"wall {got:.2f}mm does not track target {target}mm"
        )


def test_finer_pitch_does_not_worsen_the_wall():
    """Halving the pitch must not make the realised wall less accurate."""
    from cad2shell import Config, demo_parts, run
    target = 5.0
    errs = []
    for pitch in (2.0, 1.0):
        stats = run(mesh=demo_parts.thick_block(), write_files=False,
                    cfg=Config(shell_thickness=target, voxel_pitch=pitch)).as_dict()
        errs.append(abs(stats["shell"]["thickness_measured"]["median"] - target))
    assert errs[1] <= errs[0] + 1e-9, f"refining the grid worsened the wall: {errs}"


def test_an_oversized_grid_is_refused_before_it_allocates():
    """A part too big for the machine must fail immediately, not eventually.

    The distance transform allocates several float64 arrays the size of the
    whole grid. A 435x308x195 mm part at 1.5 mm pitch needs 161M voxels, so
    those arrays run to gigabytes: the machine swaps, fills its disk, and the
    build dies hundreds of seconds in having produced nothing.

    This used to be a warning appended to a list the caller only reads once the
    job finishes -- no use at all when the job is what exhausts the machine.
    """
    import re
    import pytest
    import trimesh
    from cad2shell import Config, run

    big = trimesh.creation.box(extents=[435, 308, 195])
    with pytest.raises(ValueError, match="over the .* limit") as exc:
        run(mesh=big, cfg=Config(voxel_pitch=1.5), write_files=False)

    # the message must name a pitch that actually works, not one that is itself
    # refused for landing exactly on the ceiling
    suggested = float(re.search(r"about ([\d.]+) mm", str(exc.value)).group(1))
    stats = run(mesh=big, cfg=Config(voxel_pitch=suggested), write_files=False)
    assert stats.shell["watertight"]


def test_a_normal_part_is_not_refused():
    """The ceiling must not get in the way of ordinary work."""
    from cad2shell import Config, demo_parts, run
    stats = run(mesh=demo_parts.get("boss_plate"), cfg=Config(voxel_pitch=1.5),
                write_files=False)
    assert stats.shell["watertight"]


def test_the_pour_cup_always_clears_the_casting():
    """A cup buried inside the part means a sealed mould that cannot be poured.

    The sprue used to rise a fixed distance from wherever it landed. On a tall
    part the gate site can be well below the top, so that fixed height was
    swallowed entirely: the cup ended up inside the casting's own silhouette,
    the shell was cleared above a mouth that was not open to anything, and the
    result was a watertight solid with no way in. The exported tree had exactly
    the same bounding box as the part, which is the giveaway.
    """
    import numpy as np
    import trimesh
    from cad2shell import Config, run

    # tall and awkward: a heavy base with a thin tower, so the heaviest hot
    # spot is far below the top of the part
    base = trimesh.creation.cylinder(radius=45, height=40)
    base.apply_translation([0, 0, 20])
    tower = trimesh.creation.cylinder(radius=12, height=220)
    tower.apply_translation([0, 0, 150])
    part = trimesh.boolean.union([base, tower], engine="manifold")

    stats = run(mesh=part, cfg=Config(voxel_pitch=3.0), write_files=False)
    cup = next(s for s in stats.gating["segments"] if s["kind"] == "cup")
    part_top = float(part.bounds[1][2])
    assert cup["p1"][2] > part_top, (
        f"the cup mouth is at z={cup['p1'][2]:.1f} but the casting reaches "
        f"z={part_top:.1f} -- the mould would be sealed")


def test_the_pour_direction_can_be_set_from_any_axis():
    """A CAD file's +Z is not reliably the casting's up.

    A manifold modelled lying down is 435 mm along X and 195 mm in Z; gating it
    up the file's own Z put the sprue through the side of the part. The build
    orientation is therefore an input, and the finished meshes come back in the
    file's frame so they still line up with the uploaded CAD.
    """
    import numpy as np
    from cad2shell import Config, demo_parts, run

    mesh = demo_parts.get("boss_plate")
    default = run(mesh=mesh, cfg=Config(voxel_pitch=2.5), write_files=False)
    explicit = run(mesh=mesh, cfg=Config(voxel_pitch=2.5, pour_up=(0, 0, 1)),
                   write_files=False)
    assert default.shell["volume"] == pytest.approx(
        explicit.shell["volume"], rel=1e-9), "an explicit +Z must change nothing"

    for up in ((1, 0, 0), (0, 1, 0), (0, 0, -1)):
        stats = run(mesh=mesh, cfg=Config(voxel_pitch=2.5, pour_up=up),
                    write_files=False)
        assert stats.shell["watertight"], f"pour_up={up} produced a leaking shell"
        assert stats.shell["volume"] > 0
