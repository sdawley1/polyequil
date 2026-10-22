"""
Package-specific exceptions.

Rewritten from heteromorph's ``errors.py``. ``EnsembleError`` names flat task
indices, because a task here is a (replica, time) or (replica, window) pair
rather than a single realization.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections.abc import Sequence

# ============================================================
# EXCEPTIONS
# ============================================================


class PolyequilError(Exception):
    """Base class for errors raised by :mod:`polyequil`."""


class ValidationError(PolyequilError, ValueError):
    """Input data violate a documented package contract."""


class FormatError(PolyequilError, ValueError):
    """A trajectory, topology, or log file is malformed or unsupported."""


class EnsembleError(PolyequilError, RuntimeError):
    """An ensemble evaluation failed for one or more identified tasks.

    ``task_indices`` is definitive when one task's observable raised. When a
    worker process died it is the set of *candidate* in-flight tasks: the parent
    knows only which were submitted and unconsumed. Every message states which
    case it reports.
    """

    def __init__(self, message: str, *, task_indices: Sequence[int] = ()) -> None:
        super().__init__(message)
        self.task_indices = tuple(int(index) for index in task_indices)
