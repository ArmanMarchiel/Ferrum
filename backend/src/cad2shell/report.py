"""Cross-section rendering.

A mid-plane slice through the voxel grid is the clearest way to confirm the
mould is right: you can see the skin is continuous, the metal path is one
connected tree, and the cup is open at the top.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def section_arrays(result, part_occ=None, axis: int = 0, index: int | None = None):
    """Extract a 2-D categorical slice: 0 empty, 1 shell, 2 gating metal, 3 part."""
    grid = result.grid
    if index is None:
        # Slice where the section is most informative: the plane containing the
        # most gating metal (risers + sprue), so the cut shows the feed path
        # rather than passing between two risers and revealing only the part.
        gating_only = result.metal & ~part_occ if part_occ is not None else result.metal
        per_plane = gating_only.sum(axis=tuple(a for a in (0, 1, 2) if a != axis))
        if per_plane.max() > 0:
            index = int(np.argmax(per_plane))
        else:
            src = part_occ if part_occ is not None and part_occ.any() else result.metal
            index = int(round(np.argwhere(src).mean(axis=0)[axis]))
        index = int(np.clip(index, 0, grid.shape[axis] - 1))

    take = lambda a: np.take(a, index, axis=axis)
    shell2 = take(result.shell)
    metal2 = take(result.metal)
    part2 = take(part_occ) if part_occ is not None else np.zeros_like(metal2)

    cat = np.zeros(shell2.shape, dtype=np.uint8)
    cat[shell2] = 1
    cat[metal2 & ~part2] = 2
    cat[part2] = 3
    return cat, index


def section_png(result, path, cfg, part_occ=None, axis: int | None = None) -> str:
    """Write the cross-section PNG. Returns the path."""
    grid = result.grid
    axis = cfg.section_axis if axis is None else axis
    cat, index = section_arrays(result, part_occ=part_occ, axis=axis)

    # world extent of the two in-plane axes
    other = [a for a in (0, 1, 2) if a != axis]
    lo = grid.origin
    hi = grid.origin + np.asarray(grid.shape) * grid.pitch
    extent = [lo[other[0]], hi[other[0]], lo[other[1]], hi[other[1]]]

    cmap = ListedColormap(["#ffffff", "#c9b8a3", "#d94f2b", "#2b6cb0"])
    fig, ax = plt.subplots(figsize=(9.0, 7.0), dpi=130)
    ax.imshow(cat.T, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=3,
              interpolation="nearest", aspect="equal")

    labels = "xyz"
    slice_world = grid.origin[axis] + (index + 0.5) * grid.pitch
    ax.set_xlabel(f"{labels[other[0]]} (mm)")
    ax.set_ylabel(f"{labels[other[1]]} (mm)")
    ax.set_title(
        f"Investment shell cross-section  |  {labels[axis]} = {slice_world:.1f} mm\n"
        f"{cfg.alloy_label}  ·  shell {cfg.shell_thickness:.1f} mm  ·  "
        f"pitch {cfg.voxel_pitch:.2f} mm",
        fontsize=11,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#2b6cb0", label="part"),
            Patch(facecolor="#d94f2b", label="gating / risers"),
            Patch(facecolor="#c9b8a3", label="ceramic shell"),
        ],
        loc="upper right", framealpha=0.9, fontsize=9,
    )
    # Crop to what is actually occupied, with a small margin: an untrimmed
    # grid can be mostly empty and squeeze the mould into a corner.
    occupied = np.argwhere(cat > 0)
    if len(occupied):
        m = 6
        u0, v0 = occupied.min(axis=0) - m
        u1, v1 = occupied.max(axis=0) + m
        w0 = grid.origin[other[0]] + np.clip(u0, 0, cat.shape[0]) * grid.pitch
        w1 = grid.origin[other[0]] + np.clip(u1, 0, cat.shape[0]) * grid.pitch
        h0 = grid.origin[other[1]] + np.clip(v0, 0, cat.shape[1]) * grid.pitch
        h1 = grid.origin[other[1]] + np.clip(v1, 0, cat.shape[1]) * grid.pitch
        ax.set_xlim(w0, w1)
        ax.set_ylim(h0, h1)
    ax.grid(alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)
