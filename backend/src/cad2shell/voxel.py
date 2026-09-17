"""Shared voxel coordinate frame.

Every stage that touches voxels takes a Grid and returns arrays expressed in
that Grid. This module owns the world <-> index arithmetic so no other stage
reimplements it -- that is what keeps hot-spot detection and stamping on one
frame, and it is the single place where the floor-vs-round rule lives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """An axis-aligned isotropic voxel lattice.

    origin is the world position of the *corner* of voxel [0, 0, 0]; voxel
    [i, j, k] therefore covers ``origin + [i,j,k]*pitch`` to
    ``origin + [i+1,j+1,k+1]*pitch`` and has its centre half a pitch in.
    """

    origin: np.ndarray
    pitch: float
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=np.float64).reshape(3))
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        if self.pitch <= 0:
            raise ValueError(f"pitch must be positive, got {self.pitch}")
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"shape must be positive, got {self.shape}")

    # -- construction ----------------------------------------------------
    @classmethod
    def from_bounds(cls, lo, hi, pitch: float, pad_voxels: int = 2) -> "Grid":
        """Grid covering [lo, hi] plus pad_voxels of margin on every side.

        The origin is snapped to a global multiple of ``pitch`` so that any two
        Grids built this way share one lattice and differ by an integer offset.
        """
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        if np.any(hi < lo):
            raise ValueError("hi must be >= lo componentwise")
        origin = np.floor(lo / pitch - pad_voxels) * pitch
        far = np.ceil(hi / pitch + pad_voxels) * pitch
        shape = np.maximum(np.round((far - origin) / pitch).astype(np.int64), 1)
        return cls(origin=origin, pitch=float(pitch), shape=tuple(shape))

    def expanded(self, pad_lo, pad_hi) -> "Grid":
        """Grow by whole voxels, preserving pitch and lattice alignment.

        pad_lo / pad_hi are voxel counts (scalar or per-axis). The returned
        Grid's lattice is identical, so an index in ``self`` maps into the
        result by adding ``self.offset_in(result)``.
        """
        pad_lo = np.broadcast_to(np.asarray(pad_lo, dtype=np.int64), (3,))
        pad_hi = np.broadcast_to(np.asarray(pad_hi, dtype=np.int64), (3,))
        if np.any(pad_lo < 0) or np.any(pad_hi < 0):
            raise ValueError("padding must be non-negative")
        return Grid(
            origin=self.origin - pad_lo * self.pitch,
            pitch=self.pitch,
            shape=tuple(np.asarray(self.shape) + pad_lo + pad_hi),
        )

    def offset_in(self, other: "Grid") -> np.ndarray:
        """Integer index offset to add when moving an index from self to other."""
        if abs(other.pitch - self.pitch) > 1e-9:
            raise ValueError("grids have different pitch; indices are not translatable")
        delta = (self.origin - other.origin) / self.pitch
        rounded = np.round(delta)
        if np.any(np.abs(delta - rounded) > 1e-6):
            raise ValueError("grids are not lattice-aligned; indices are not translatable")
        return rounded.astype(np.int64)

    # -- mapping ---------------------------------------------------------
    def to_index(self, pts) -> np.ndarray:
        """World points -> integer voxel indices.

        Uses floor, never round. Rounding would map the two halves of a voxel
        to different indices and collapse cells that straddle a half-boundary,
        which shows up as pinholes in thin walls.
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        return np.floor((pts - self.origin) / self.pitch).astype(np.int64)

    def to_world(self, idx) -> np.ndarray:
        """Integer voxel indices -> world position of the voxel *centre*."""
        idx = np.atleast_2d(np.asarray(idx))
        return self.origin + (idx.astype(np.float64) + 0.5) * self.pitch

    def contains(self, idx) -> np.ndarray:
        idx = np.atleast_2d(np.asarray(idx))
        shape = np.asarray(self.shape, dtype=np.int64)
        return np.all((idx >= 0) & (idx < shape), axis=1)

    # -- convenience -----------------------------------------------------
    @property
    def voxel_volume(self) -> float:
        return float(self.pitch ** 3)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.origin, self.origin + np.asarray(self.shape) * self.pitch])

    def empty(self, dtype=bool) -> np.ndarray:
        return np.zeros(self.shape, dtype=dtype)

    def mm(self, n_voxels: float) -> float:
        return float(n_voxels) * self.pitch

    def voxels(self, mm: float) -> float:
        """Millimetres -> continuous voxel units (not rounded)."""
        return float(mm) / self.pitch

    def as_dict(self) -> dict:
        return {
            "pitch": self.pitch,
            "shape": list(self.shape),
            "origin": self.origin.tolist(),
            "n_voxels": self.n_voxels,
        }
