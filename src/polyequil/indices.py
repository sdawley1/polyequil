"""
Equilibrium indices from the fluctuation and response sides of each relation.

``X_rho = S0 / (rho T kappa_T) - 1`` (Hopkins, Stillinger, and Torquato 2012)
and ``X_Q = S_Q(0) / (T chi) - 1`` (``docs/design.md`` section 1), with
k_B = 1 in LJ units. Both vanish in equilibrium. The fluctuation side comes
from :class:`~polyequil.ensemble.TrajectoryEnsemble` (via
:func:`estimate_from`); kappa_T and chi come from separate response runs and
are supplied as measured inputs. Errors are propagated to first order,
treating the two sides as independent.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import dataclass
from typing import Any

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ----- local -----
from polyequil.ensemble import TimeResolvedResult
from polyequil.errors import ValidationError

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class Estimate:
    """A value and its standard error; arrays broadcast (e.g. over time)."""

    value: FloatArray
    error: FloatArray

    def __post_init__(self) -> None:
        value = np.array(self.value, dtype=np.float64)
        error = np.array(self.error, dtype=np.float64)
        try:
            np.broadcast_shapes(value.shape, error.shape)
        except ValueError as exc:
            raise ValidationError(
                f"value {value.shape} and error {error.shape} do not broadcast"
            ) from exc
        if not np.all(np.isfinite(error)) or np.any(error < 0.0):
            raise ValidationError("an error must be finite and non-negative")
        for name, array in (("value", value), ("error", error)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


# ============================================================
# PUBLIC API
# ============================================================


def estimate_from(
    result: TimeResolvedResult, component: tuple[int, ...] = ()
) -> Estimate:
    """Return the mean and standard error of one value component at every slice.

    For a :func:`~polyequil.observables.extrapolation.small_k_limit` result,
    ``component=(0,)`` is S0; for an orientational structure factor,
    ``component=(0,)`` is S_Q(0).

    Raises
    ------
    ValidationError
        If the result has a single replica (no standard error).

    Examples
    --------
    S0 at two times from a ``small_k_limit`` evaluation over four replicas
    (values ``[S0, a]`` per time slice):

    >>> import numpy as np
    >>> from polyequil.ensemble import TimeResolvedResult
    >>> from polyequil.indices import estimate_from
    >>> from polyequil.results import SeparableAxes
    >>> result = TimeResolvedResult(
    ...     kind="frame",
    ...     times=np.array([0.0, 100.0]),
    ...     mean=np.array([[0.30, 1.2], [0.26, 1.1]]),
    ...     standard_deviation=np.array([[0.04, 0.2], [0.02, 0.2]]),
    ...     standard_error=np.array([[0.02, 0.1], [0.01, 0.1]]),
    ...     replica_count=4,
    ...     coordinates=SeparableAxes((("parameter", np.arange(2.0)),)),
    ...     metadata={},
    ... )
    >>> s0 = estimate_from(result, component=(0,))
    >>> s0.value.tolist(), s0.error.tolist()
    ([0.3, 0.26], [0.02, 0.01])
    """
    if result.standard_error is None:
        raise ValidationError("a single-replica result has no standard error")
    index: tuple[Any, ...] = (slice(None), *component)
    return Estimate(result.mean[index], result.standard_error[index])


def density_index(
    s0: Estimate, *, density: float, temperature: float, kappa_t: Estimate
) -> Estimate:
    """Return ``X_rho = S0 / (rho T kappa_T) - 1`` with its propagated error.

    Parameters
    ----------
    s0 : Estimate
        Small-k limit of the monomer structure factor.
    density : float
        Monomer number density, sigma^-3 (exact; set by the simulation).
    temperature : float
        Temperature, epsilon / k_B (exact; set by the thermostat).
    kappa_t : Estimate
        Isothermal compressibility from the response measurement.

    Returns
    -------
    Estimate
        The index; zero in equilibrium.

    Raises
    ------
    ValidationError
        If ``density`` or ``temperature`` is not positive and finite, or any
        value of ``kappa_t`` is not positive and finite.

    Examples
    --------
    S0 = 0.26 +/- 0.01 against rho T kappa_T = 0.85 x 1.0 x (0.30 +/- 0.006)
    gives an index consistent with zero, as expected in equilibrium:

    >>> from polyequil.indices import Estimate, density_index
    >>> index = density_index(
    ...     Estimate(0.26, 0.01),
    ...     density=0.85,
    ...     temperature=1.0,
    ...     kappa_t=Estimate(0.30, 0.006),
    ... )
    >>> print(f"{float(index.value):.3f} +/- {float(index.error):.3f}")
    0.020 +/- 0.044
    """
    scale = _positive(density, "density") * _positive(temperature, "temperature")
    return _ratio_index(s0, scale, kappa_t, "kappa_t")


def orientational_index(
    sq0: Estimate, *, temperature: float, chi: Estimate
) -> Estimate:
    """Return ``X_Q = S_Q(0) / (T chi) - 1`` with its propagated error.

    ``sq0`` is the direct k = 0 value of the orientational structure factor
    (or, as a cross-check, its small-k extrapolation); ``chi`` is the
    per-segment susceptibility d<P2>/dh from the aligning-field response.
    """
    return _ratio_index(sq0, _positive(temperature, "temperature"), chi, "chi")


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _positive(value: ArrayLike, name: str) -> float:
    """Return ``value`` as a float; raise unless positive and finite."""
    number = float(np.asarray(value))
    if not np.isfinite(number) or number <= 0.0:
        raise ValidationError(f"{name} must be positive and finite, got {value!r}")
    return number


def _ratio_index(
    fluctuation: Estimate, scale: float, response: Estimate, name: str
) -> Estimate:
    """Return ``fluctuation / (scale * response) - 1`` with first-order errors."""
    if not np.all(np.isfinite(response.value)) or np.any(response.value <= 0.0):
        raise ValidationError(f"{name} must be positive and finite")
    denominator = scale * response.value
    ratio = fluctuation.value / denominator
    error = np.hypot(
        fluctuation.error / denominator, ratio * response.error / response.value
    )
    return Estimate(ratio - 1.0, error)
