"""
Equilibration diagnostics for simulated polymer melts.

The public API re-exported here covers storage (``Trajectory``, HDF5 and
LAMMPS I/O in :mod:`polyequil.io`), data types (``Box``, ``ChainTopology``,
``ChainConfiguration``), and ensemble evaluation (``TrajectoryEnsemble``,
``ExecutionPolicy``, ``ObservableResult``).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from importlib.metadata import version

# ----- local -----
from polyequil.box import Box
from polyequil.configuration import ChainConfiguration
from polyequil.ensemble import TimeResolvedResult, TrajectoryEnsemble
from polyequil.errors import (
    EnsembleError,
    FormatError,
    PolyequilError,
    ValidationError,
)
from polyequil.execution import ExecutionPolicy
from polyequil.indices import Estimate
from polyequil.results import ObservableResult, PointSet, SeparableAxes
from polyequil.topology import ChainTopology, linear_chains
from polyequil.trajectory import FrameRef, Trajectory, TrajectoryWindow, WindowData

__version__ = version("polyequil")

__all__ = [
    "Box",
    "ChainConfiguration",
    "ChainTopology",
    "EnsembleError",
    "Estimate",
    "ExecutionPolicy",
    "FormatError",
    "FrameRef",
    "ObservableResult",
    "PointSet",
    "PolyequilError",
    "SeparableAxes",
    "TimeResolvedResult",
    "Trajectory",
    "TrajectoryEnsemble",
    "TrajectoryWindow",
    "ValidationError",
    "WindowData",
    "__version__",
    "linear_chains",
]
