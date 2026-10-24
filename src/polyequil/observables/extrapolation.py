"""
Small-k limit of a shell-averaged structure factor: S(k) = S0 + a k^2.

S(0) itself is zero at fixed N, so the density channel needs S0 from the
k -> 0 limit (``docs/design.md`` section 3, item 1). The fit is weighted least
squares with weights fixed by geometry (shell mode counts), which makes S0 a
linear function of S(k). Fitting each replica and averaging across replicas
therefore gives the same mean as fitting the ensemble mean, and the
across-replica standard error of S0 is honest: it carries the shell-to-shell
covariance and the fit error without any further modelling. The same fit
applied to S_Q(k > 0) cross-checks the direct S_Q(0).

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
from numpy.typing import ArrayLike, NDArray

# ----- local -----
from polyequil.configuration import ChainConfiguration
from polyequil.conventions import SMALL_K_FORM, SMALL_K_WEIGHTS
from polyequil.errors import ValidationError
from polyequil.observables.window import window_mean
from polyequil.results import ObservableResult, SeparableAxes
from polyequil.trajectory import WindowData

# ============================================================
# CONSTANTS
# ============================================================

PARAMETERS: tuple[str, str] = ("S0", "a")
K_FIT_RELATIVE_SLACK: float = 1e-12

# ============================================================
# PUBLIC API
# ============================================================


def fit_small_k(
    k: ArrayLike, values: ArrayLike, counts: ArrayLike, *, k_fit: float
) -> NDArray[np.float64]:
    """Return ``[S0, a]`` from a weighted fit of ``S0 + a k^2`` over 0 < k <= k_fit.

    Parameters
    ----------
    k : ArrayLike
        (S,) shell wavenumbers.
    values : ArrayLike
        (S,) shell values.
    counts : ArrayLike
        (S,) half-space mode counts per shell, used as weights.
    k_fit : float
        Largest wavenumber in the fit, in 1/sigma.

    Returns
    -------
    NDArray[np.float64]
        Intercept and curvature.

    Raises
    ------
    ValidationError
        If the shapes differ, ``k_fit`` is not positive, or fewer than two
        shells lie in the fit range.
    """
    k_array = np.asarray(k, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(counts, dtype=np.float64)
    if k_array.ndim != 1 or y.shape != k_array.shape or w.shape != k_array.shape:
        raise ValidationError(
            f"k, values, and counts need one equal 1-D shape, got "
            f"{k_array.shape}, {y.shape}, {w.shape}"
        )
    inside = _fit_mask(k_array, k_fit)
    if np.any(w[inside] <= 0):
        raise ValidationError("shell counts in the fit range must be positive")
    root_w = np.sqrt(w[inside])
    design = np.stack([np.ones(inside.sum()), k_array[inside] ** 2], axis=1)
    solution, *_ = np.linalg.lstsq(design * root_w[:, None], y[inside] * root_w)
    assert solution.shape == (2,), "two fit parameters"
    fitted: NDArray[np.float64] = solution
    return fitted


def small_k_limit(
    item: ChainConfiguration | WindowData,
    *,
    frame_observable: Callable[..., ObservableResult],
    k_fit: float,
    threads: int = 1,
    **parameters: Any,
) -> ObservableResult:
    """Return ``[S0, a]`` for one frame or one window of one replica.

    The observable (e.g. ``structure_factor``) is evaluated with
    ``k_max = k_fit``; for a window it is first averaged over the frames.
    Evaluate over replicas with ``TrajectoryEnsemble.evaluate`` or
    ``evaluate_windows`` to get S0 with its standard error.

    Parameters
    ----------
    item : ChainConfiguration | WindowData
        One frame, or one replica's window.
    frame_observable : Callable[..., ObservableResult]
        Shell-averaged observable with a single ``k`` axis and
        ``auxiliary["shell_counts"]``.
    k_fit : float
        Largest wavenumber in the fit (the q_max of the design doc).
    threads : int
        Passed through to the observable.
    **parameters : Any
        Passed through to the observable (not ``k_max``).

    Returns
    -------
    ObservableResult
        Values (2,) on axis ``parameter`` = (S0, a); the convention records
        the form, weights, k_fit, the fitted shells, and the source convention.
    """
    if "k_max" in parameters:
        raise ValidationError("k_max is set to k_fit; do not pass it")
    _fit_mask(np.zeros(1), k_fit, require_two=False)
    if isinstance(item, WindowData):
        source = window_mean(
            item,
            frame_observable=frame_observable,
            threads=threads,
            k_max=k_fit,
            **parameters,
        )
    elif isinstance(item, ChainConfiguration):
        source = frame_observable(item, threads=threads, k_max=k_fit, **parameters)
    else:
        raise ValidationError(f"cannot fit a {type(item).__name__}")
    k, counts = _shells_of(source)
    inside = _fit_mask(k, k_fit)
    return ObservableResult(
        values=fit_small_k(k, source.values, counts, k_fit=k_fit),
        coordinates=SeparableAxes((("parameter", np.arange(2.0)),)),
        convention={
            "quantity": "small-k limit",
            "parameters": PARAMETERS,
            "form": SMALL_K_FORM,
            "weights": SMALL_K_WEIGHTS,
            "k_fit": float(k_fit),
            "fit_shells": k[inside],
            "fit_counts": counts[inside],
            "source": source.convention,
        },
    )


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _fit_mask(
    k: NDArray[np.float64], k_fit: float, *, require_two: bool = True
) -> NDArray[np.bool_]:
    """Return the shells with 0 < k <= k_fit; validate k_fit and the count."""
    valid = isinstance(k_fit, (int, float)) and not isinstance(k_fit, bool)
    if not valid or not np.isfinite(k_fit) or k_fit <= 0:
        raise ValidationError(f"k_fit must be positive and finite, got {k_fit!r}")
    inside: NDArray[np.bool_] = (k > 0.0) & (k <= k_fit * (1.0 + K_FIT_RELATIVE_SLACK))
    if require_two and inside.sum() < 2:
        raise ValidationError(
            f"fewer than two shells with 0 < k <= {k_fit}; raise k_fit"
        )
    return inside


def _shells_of(
    source: ObservableResult,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the k axis and shell counts of a shell-averaged result."""
    axes = source.coordinates
    if not isinstance(axes, SeparableAxes) or axes.names() != ("k",):
        raise ValidationError("the observable must return values on a single k axis")
    if "shell_counts" not in source.auxiliary:
        raise ValidationError("the observable must report auxiliary['shell_counts']")
    counts = np.asarray(source.auxiliary["shell_counts"], dtype=np.float64)
    return axes.axes[0][1], counts
