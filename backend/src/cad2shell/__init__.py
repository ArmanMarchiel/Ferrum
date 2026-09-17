"""cad2shell -- part STL to investment-shell mould with a full gating tree.

    from cad2shell import run, Config
    stats = run("part.stl", "out/part", Config(alloy="A356", shell_thickness=6.0))

Deterministic, no ML: every dimension traces to a documented foundry rule.
"""
from .config import ALLOYS, Config
from .gating import GatingPlan, Segment
from .hotspots import HotSpots
from .ingest import PartMesh
from .pipeline import Stats, run
from .risers import Riser
from .synthesis import ShellResult
from .voxel import Grid

__version__ = "0.1.0"
__all__ = [
    "run", "Config", "Stats", "Grid", "PartMesh", "HotSpots",
    "GatingPlan", "Segment", "Riser", "ShellResult", "ALLOYS",
]
