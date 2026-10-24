"""
Time averaging of a frame observable over one replica's window.

Averaging frames of one trajectory is legitimate for an *estimate* (it reduces
the per-replica noise of, e.g., S(k) at small k, ``docs/design.md`` section 3,
item 2); it is never a source of error bars, which come only from independent
replicas through :class:`~polyequil.ensemble.TrajectoryEnsemble`.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections.abc import Callable
from typing import Any

# ----- third party -----
import numpy as np

# ----- local -----
from polyequil.errors import ValidationError
from polyequil.results import ObservableResult, results_compatible
from polyequil.trajectory import WindowData

# ============================================================
# CONSTANTS
# ============================================================

TIME_AVERAGE: str = "equal-weight mean over the frames of one replica's window"

# ============================================================
# PUBLIC API
# ============================================================


def window_mean(
    window: WindowData,
    *,
    frame_observable: Callable[..., ObservableResult],
    threads: int = 1,
    **parameters: Any,
) -> ObservableResult:
    """Return the mean of a frame observable over every frame of a window.

    Use with ``TrajectoryEnsemble.evaluate_windows(window_mean, windows=...,
    frame_observable=structure_factor, k_max=...)``.

    Parameters
    ----------
    window : WindowData
        Frames of one replica.
    frame_observable : Callable[..., ObservableResult]
        Module-level frame observable ``f(config, *, threads, **parameters)``.
    threads : int
        Passed through to the observable.
    **parameters : Any
        Passed through to the observable.

    Returns
    -------
    ObservableResult
        Mean values with the frame results' coordinates; the convention gains
        ``time_average`` and the auxiliary data gain ``frame_count``.

    Raises
    ------
    ValidationError
        If the observable is not callable, returns something other than an
        ObservableResult, or returns incompatible results for two frames.
    """
    if not isinstance(window, WindowData):
        raise ValidationError(f"window must be WindowData, not {type(window).__name__}")
    if not callable(frame_observable):
        raise ValidationError("frame_observable must be callable")
    reference = _frame_result(window, 0, frame_observable, threads, parameters)
    total = np.array(reference.values, copy=True)
    for index in range(1, window.n_frames):
        result = _frame_result(window, index, frame_observable, threads, parameters)
        reason = results_compatible(reference, result)
        if reason is not None:
            raise ValidationError(
                f"frame {index} (t = {window.times[index]:g}) is incompatible with "
                f"frame 0: {reason}"
            )
        total += result.values
    return ObservableResult(
        values=total / window.n_frames,
        coordinates=reference.coordinates,
        convention={**reference.convention, "time_average": TIME_AVERAGE},
        auxiliary={**reference.auxiliary, "frame_count": window.n_frames},
        provenance=reference.provenance,
    )


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _frame_result(
    window: WindowData,
    index: int,
    frame_observable: Callable[..., ObservableResult],
    threads: int,
    parameters: dict[str, Any],
) -> ObservableResult:
    """Evaluate the observable on one frame and check its return type."""
    result = frame_observable(window.frame(index), threads=threads, **parameters)
    if not isinstance(result, ObservableResult):
        raise ValidationError("the frame observable must return an ObservableResult")
    return result
