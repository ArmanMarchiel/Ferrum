"""Stage 1 -- ingest: load, repair, and characterise the part mesh."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class PartMesh:
    mesh: trimesh.Trimesh
    volume: float          # mm^3
    area: float            # mm^2
    modulus: float         # mm, casting modulus V/A
    repairs: list[str] = field(default_factory=list)
    watertight: bool = True

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.mesh.bounds, dtype=np.float64)

    @property
    def extents(self) -> np.ndarray:
        return self.bounds[1] - self.bounds[0]

    def mass(self, density: float) -> float:
        return float(density * self.volume)

    def as_dict(self) -> dict:
        return {
            "volume": self.volume,
            "area": self.area,
            "modulus": self.modulus,
            "watertight": self.watertight,
            "triangles": int(len(self.mesh.faces)),
            "bounds": self.bounds.tolist(),
            "extents": self.extents.tolist(),
            "repairs": list(self.repairs),
        }


def _as_single_mesh(obj) -> tuple[trimesh.Trimesh, list[str]]:
    """Reduce whatever trimesh loaded into one Trimesh."""
    notes: list[str] = []
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("file contains no triangle geometry")
        if len(geoms) > 1:
            notes.append(f"concatenated {len(geoms)} scene geometries")
        obj = trimesh.util.concatenate(geoms)
    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"unsupported geometry type {type(obj).__name__}")
    return obj, notes


def repair(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Best-effort watertighting. Returns the mesh and a log of what was done."""
    notes: list[str] = []
    mesh = mesh.copy()

    n0 = len(mesh.faces)
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) != n0:
        notes.append(f"removed {n0 - len(mesh.faces)} degenerate/duplicate faces")

    if not mesh.is_watertight:
        merged = mesh.copy()
        merged.merge_vertices()
        if merged.is_watertight:
            mesh, _ = merged, notes.append("merged coincident vertices")

    if not mesh.is_watertight:
        filled = mesh.copy()
        try:
            trimesh.repair.fill_holes(filled)
            if filled.is_watertight:
                mesh, _ = filled, notes.append("filled boundary holes")
        except Exception as exc:  # pragma: no cover - trimesh internals
            notes.append(f"fill_holes failed: {exc}")

    if not mesh.is_watertight:
        # Last resort: a voxel remesh always yields a closed surface. Costly and
        # lossy, so it is only reached when the topology is genuinely broken.
        try:
            pitch = float(np.max(mesh.extents)) / 128.0
            solid = mesh.voxelized(pitch=pitch).fill().marching_cubes
            solid.merge_vertices()
            trimesh.repair.fix_normals(solid)
            if solid.is_watertight and solid.volume > 0:
                mesh = solid
                notes.append(f"voxel-remeshed to close the surface (pitch {pitch:.3f}mm)")
        except Exception as exc:  # pragma: no cover
            notes.append(f"voxel remesh failed: {exc}")

    trimesh.repair.fix_normals(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
        notes.append("inverted inside-out winding")
    return mesh, notes


def characterise(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    """Return (volume, area, modulus=V/A). Volume is absolute."""
    volume = abs(float(mesh.volume))
    area = float(mesh.area)
    if area <= 0:
        raise ValueError("mesh has zero surface area")
    return volume, area, volume / area


def from_mesh(mesh: trimesh.Trimesh, cfg=None) -> PartMesh:
    """Ingest an in-memory mesh (used by the demo parts and the API)."""
    mesh, notes = _as_single_mesh(mesh)
    mesh, repair_notes = repair(mesh)
    notes += repair_notes
    volume, area, modulus = characterise(mesh)
    if volume <= 0:
        raise ValueError("part volume is zero or negative after repair")
    return PartMesh(
        mesh=mesh, volume=volume, area=area, modulus=modulus,
        repairs=notes, watertight=bool(mesh.is_watertight),
    )


def load_part(path, cfg=None) -> PartMesh:
    """Stage 1 entry point: load a mesh file from disk and characterise it."""
    loaded = trimesh.load(str(path), force="mesh", process=False)
    part = from_mesh(loaded, cfg)
    return part
