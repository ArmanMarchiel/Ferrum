"""The coordinate-frame invariants -- the bugs this design exists to prevent."""
from __future__ import annotations

import numpy as np
import pytest

from cad2shell.voxel import Grid


def test_index_world_roundtrip_is_exact():
    """to_index(to_world(i)) == i for every voxel. Guards the floor/round rule."""
    g = Grid.from_bounds([-10, -5, 0], [10, 5, 20], 1.5, pad_voxels=2)
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in g.shape], indexing="ij"), -1).reshape(-1, 3)
    assert np.array_equal(g.to_index(g.to_world(idx)), idx)


def test_mapping_uses_floor_not_round():
    """A point in the far half of a voxel belongs to that voxel, not the next.

    Rounding would send it to index+1 -- a cell it is not inside -- which
    punches pinholes in thin walls where two stages disagree by one cell.
    """
    g = Grid(origin=[0.0, 0.0, 0.0], pitch=1.0, shape=(8, 8, 8))
    p = np.array([[3.7, 3.7, 3.7]])
    assert np.array_equal(g.to_index(p)[0], [3, 3, 3])
    assert not np.array_equal(g.to_index(p)[0], np.round(p[0]).astype(int))


def test_expanded_grid_is_lattice_aligned():
    """A padded grid must differ from its parent by a pure integer offset."""
    g = Grid.from_bounds([-7.3, 2.1, -0.4], [11.9, 6.6, 20.2], 1.25, pad_voxels=2)
    e = g.expanded([3, 1, 4], [2, 5, 0])
    off = g.offset_in(e)
    assert np.array_equal(off, [3, 1, 4])
    # a world point maps to the same physical cell in both frames
    pt = np.array([[1.234, 3.456, 7.89]])
    assert np.array_equal(g.to_index(pt)[0] + off, e.to_index(pt)[0])


def test_misaligned_grids_are_rejected():
    a = Grid(origin=[0.0, 0.0, 0.0], pitch=1.0, shape=(4, 4, 4))
    b = Grid(origin=[0.37, 0.0, 0.0], pitch=1.0, shape=(4, 4, 4))
    with pytest.raises(ValueError, match="lattice-aligned"):
        a.offset_in(b)


def test_pipeline_keeps_hotspots_and_stamping_on_one_frame(results):
    """The part grid and the full grid must be integer-translatable.

    If they were not, a riser would be stamped at a different physical place
    than the hot spot it was sized for.
    """
    for name, r in results.items():
        off = np.asarray(r["stats"].as_dict()["grid"]["part_grid_offset"])
        assert off.dtype.kind in "iu" or np.allclose(off, np.round(off)), name
        assert np.all(off >= 0), name
