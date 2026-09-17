"""Built-in test parts, so the pipeline is runnable with no input file.

Each returns a watertight Trimesh in millimetres sitting on z=0.
"""
from __future__ import annotations

import numpy as np
import trimesh


def _drop_to_zero(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])
    return mesh


def boss_plate(plate=(80.0, 60.0, 10.0), boss_r=14.0, boss_h=26.0,
               bosses=((-20.0, 0.0), (22.0, 8.0))) -> trimesh.Trimesh:
    """A thin plate with thick cylindrical bosses -- two obvious hot spots."""
    parts = [trimesh.creation.box(extents=plate)]
    parts[0].apply_translation([0.0, 0.0, plate[2] / 2.0])
    for (x, y) in bosses:
        c = trimesh.creation.cylinder(radius=boss_r, height=boss_h, sections=48)
        c.apply_translation([x, y, plate[2] + boss_h / 2.0 - 1.0])
        parts.append(c)
    mesh = trimesh.boolean.union(parts)
    if isinstance(mesh, trimesh.Scene):  # pragma: no cover
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return _drop_to_zero(mesh)


def thin_bracket(length=90.0, width=40.0, thickness=5.0, rib_h=22.0) -> trimesh.Trimesh:
    """An L-bracket of uniform thin wall -- low modulus, stresses riser sizing."""
    base = trimesh.creation.box(extents=[length, width, thickness])
    base.apply_translation([0.0, 0.0, thickness / 2.0])
    wall = trimesh.creation.box(extents=[thickness, width, rib_h])
    wall.apply_translation([-length / 2.0 + thickness / 2.0, 0.0, rib_h / 2.0])
    mesh = trimesh.boolean.union([base, wall])
    if isinstance(mesh, trimesh.Scene):  # pragma: no cover
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return _drop_to_zero(mesh)


def thick_block(extents=(50.0, 40.0, 35.0)) -> trimesh.Trimesh:
    """A chunky block -- one central hot spot, high modulus."""
    mesh = trimesh.creation.box(extents=list(extents))
    mesh.apply_translation([0.0, 0.0, extents[2] / 2.0])
    trimesh.repair.fix_normals(mesh)
    return _drop_to_zero(mesh)


def stepped_shaft(radii=(9.0, 18.0, 11.0), heights=(24.0, 26.0, 20.0)) -> trimesh.Trimesh:
    """Stacked cylinders of differing section -- a graded hot-spot ladder."""
    parts, z = [], 0.0
    for r, h in zip(radii, heights):
        c = trimesh.creation.cylinder(radius=r, height=h, sections=48)
        c.apply_translation([0.0, 0.0, z + h / 2.0])
        parts.append(c)
        z += h
    mesh = trimesh.boolean.union(parts)
    if isinstance(mesh, trimesh.Scene):  # pragma: no cover
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    return _drop_to_zero(mesh)


DEMO_PARTS = {
    "boss_plate": boss_plate,
    "thin_bracket": thin_bracket,
    "thick_block": thick_block,
    "stepped_shaft": stepped_shaft,
}


def get(name: str) -> trimesh.Trimesh:
    if name not in DEMO_PARTS:
        raise KeyError(f"unknown demo part {name!r}; choose from {sorted(DEMO_PARTS)}")
    return DEMO_PARTS[name]()
