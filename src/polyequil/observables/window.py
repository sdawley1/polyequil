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
        If ``window`` is not a ``WindowData``, the observable is not callable,
        returns something other than an ObservableResult, or returns
        incompatible results for two frames. Errors raised by
        ``frame_observable`` propagate unchanged.

    Examples
    --------
    A three-bead rod with unit bonds (``R_g^2 = 2/3``) stretched to bonds of
    length 2 (``R_g^2 = 8/3``) in the next frame averages to ``5/3``:

    >>> import numpy as np
    >>> from polyequil.observables.chain import radius_of_gyration
    >>> from polyequil.observables.window import window_mean
    >>> from polyequil.topology import linear_chains
    >>> from polyequil.trajectory import WindowData
    >>> rod = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    >>> window = WindowData(
    ...     topology=linear_chains([3]),
    ...     steps=np.array([0, 100]),
    ...     times=np.array([0.0, 1.0]),
    ...     box_lo=np.zeros((2, 3)),
    ...     box_hi=np.full((2, 3), 10.0),
    ...     positions=np.stack([rod, 2 * rod]),
    ...     images=np.zeros((2, 3, 3), dtype=np.int32),
    ...     velocities=None,
    ...     images_source="dump",
    ... )
    >>> result = window_mean(window, frame_observable=radius_of_gyration)
    >>> round(float(result.values), 12)
    1.666666666667
    >>> result.auxiliary["frame_count"]
    2
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
