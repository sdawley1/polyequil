"""
Density and orientational structure factors, and the mean order tensor.

These are the fluctuation sides of the two equilibrium relations in
``docs/design.md`` section 1: S(k -> 0) = rho k_B T kappa_T for density and
S_Q(0) = k_B T chi for segment orientation. Definitions live in
:mod:`polyequil.conventions`. Every observable here is a pure function of one
frame; averaging over replicas (and over a window, via
:func:`polyequil.observables.window.window_mean`) happens elsewhere.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections.abc import Mapping

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil.configuration import ChainConfiguration
from polyequil.conventions import (
    FOURIER_KERNEL,
    Q_BASIS,
    RECIPROCAL_VECTORS,
    S_K_NORMALIZATION,
    S_Q_NORMALIZATION,
    S_Q_PREFACTOR,
    SEGMENT_DEFINITION,
    SHELLS,
)
from polyequil.errors import ValidationError
from polyequil.reciprocal import (
    ReciprocalShells,
    collective_amplitudes,
    reciprocal_shells,
    shell_mean,
)
from polyequil.results import ObservableResult, SeparableAxes

# ============================================================
# CONSTANTS
# ============================================================

SQRT2: float = float(np.sqrt(2.0))
SQRT6: float = float(np.sqrt(6.0))
TENSOR_INDEX: NDArray[np.float64] = np.arange(3, dtype=np.float64)

# ============================================================
# PUBLIC API
# ============================================================


def structure_factor(
    configuration: ChainConfiguration, *, k_max: float, threads: int = 1
) -> ObservableResult:
    """Return the monomer structure factor S(k), shell-averaged, k = 0 excluded.

    Parameters
    ----------
    configuration : ChainConfiguration
        One frame.
    k_max : float
        Largest wavenumber, in 1/sigma.
    threads : int
        Reserved by the execution layer; single-threaded.

    Returns
    -------
    ObservableResult
        Values (S,) on axis ``k``; ``auxiliary["shell_counts"]`` holds the
        number of half-space modes per shell.
    """
    _check_threads(threads)
    shells = reciprocal_shells(configuration.box, k_max, include_zero=False)
    amplitude = collective_amplitudes(
        configuration.positions, None, configuration.box, shells.modes
    )[:, 0]
    intensity = (amplitude.real**2 + amplitude.imag**2) / configuration.n_beads
    convention = {
        "quantity": "monomer structure factor",
        "normalization": S_K_NORMALIZATION,
        "fourier_kernel": FOURIER_KERNEL,
        "reciprocal_vectors": RECIPROCAL_VECTORS,
        "shells": SHELLS,
    }
    return _shell_result(shell_mean(intensity, shells), shells, convention)


def orientational_structure_factor(
    configuration: ChainConfiguration,
    *,
    k_max: float,
    segment_bonds: int = 1,
    threads: int = 1,
) -> ObservableResult:
    """Return S_Q(k) of segment orientations, shell-averaged, k = 0 first.

    ``S_Q(k) = (3/10) sum_a |sum_s q_a(u_s) exp(-i k.r_s)|^2 / N_seg``; at k = 0
    this is the per-segment fluctuation of sum_s P2(u_s.n) averaged over
    directions n, the quantity the uniform-field susceptibility chi probes.

    Parameters
    ----------
    configuration : ChainConfiguration
        One frame; chains are unwrapped with the stored image flags.
    k_max : float
        Largest wavenumber, in 1/sigma; 0 gives the k = 0 value alone.
    segment_bonds : int
        Bonds spanned by one segment (n_s).
    threads : int
        Reserved by the execution layer; single-threaded.

    Returns
    -------
    ObservableResult
        Values (S,) on axis ``k`` with ``k[0] = 0``.
    """
    _check_threads(threads)
    u, midpoints = segments(configuration, segment_bonds)
    shells = reciprocal_shells(configuration.box, k_max, include_zero=True)
    amplitude = collective_amplitudes(
        midpoints, q_components(u), configuration.box, shells.modes
    )
    intensity = np.sum(amplitude.real**2 + amplitude.imag**2, axis=1)
    per_mode = S_Q_PREFACTOR * intensity / u.shape[0]
    convention = {
        "quantity": "orientational structure factor",
        "normalization": S_Q_NORMALIZATION,
        "segment": SEGMENT_DEFINITION,
        "segment_bonds": segment_bonds,
        "q_basis": Q_BASIS,
        "fourier_kernel": FOURIER_KERNEL,
        "reciprocal_vectors": RECIPROCAL_VECTORS,
        "shells": SHELLS,
    }
    return _shell_result(shell_mean(per_mode, shells), shells, convention)


def order_tensor(
    configuration: ChainConfiguration, *, segment_bonds: int = 1, threads: int = 1
) -> ObservableResult:
    """Return the mean segment order tensor ``Q / N_seg``, shape (3, 3).

    Its ensemble mean is zero in an isotropic melt; a nonzero mean or unequal
    components is the anisotropy diagnostic reported with the orientational
    index (``docs/design.md`` section 3, item 4).
    """
    _check_threads(threads)
    u, _ = segments(configuration, segment_bonds)
    tensor = u.T @ u / u.shape[0] - np.eye(3) / 3.0
    return ObservableResult(
        values=tensor,
        coordinates=SeparableAxes((("i", TENSOR_INDEX), ("j", TENSOR_INDEX))),
        convention={
            "quantity": "mean segment order tensor Q/N_seg",
            "segment": SEGMENT_DEFINITION,
            "segment_bonds": segment_bonds,
        },
    )


def segments(
    configuration: ChainConfiguration, segment_bonds: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return unit vectors and midpoints of every n_s-bond segment, each (N_seg, 3).

    Raises
    ------
    ValidationError
        If ``segment_bonds`` is not a positive int, no chain is long enough, or
        a segment has zero length.
    """
    if isinstance(segment_bonds, bool) or not isinstance(segment_bonds, int):
        raise ValidationError(f"segment_bonds must be an int, got {segment_bonds!r}")
    if segment_bonds < 1:
        raise ValidationError(f"segment_bonds must be >= 1, got {segment_bonds}")
    first, last = configuration.topology.pairs_at_separation(segment_bonds)
    if first.size == 0:
        raise ValidationError(
            f"no chain is long enough for a segment of {segment_bonds} bonds"
        )
    x = configuration.unwrapped()
    span = x[last] - x[first]
    length = np.linalg.norm(span, axis=1, keepdims=True)
    if np.any(length == 0.0):
        raise ValidationError("a segment has zero length; its orientation is undefined")
    return span / length, 0.5 * (x[first] + x[last])


def q_components(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 5 orthonormal components of ``u u - I/3`` per row, (N, 5)."""
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    components = np.stack(
        [
            (ux * ux - uy * uy) / SQRT2,
            (2.0 * uz * uz - ux * ux - uy * uy) / SQRT6,
            SQRT2 * ux * uy,
            SQRT2 * ux * uz,
            SQRT2 * uy * uz,
        ],
        axis=1,
    )
    return components


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _check_threads(threads: int) -> None:
    """Reject a non-positive thread count."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValidationError(f"threads must be a positive int, got {threads!r}")


def _shell_result(
    values: NDArray[np.float64],
    shells: ReciprocalShells,
    convention: Mapping[str, object],
) -> ObservableResult:
    """Wrap shell values with their wavenumbers and degeneracies."""
    return ObservableResult(
        values=values,
        coordinates=SeparableAxes((("k", shells.shell_k),)),
        convention=convention,
        auxiliary={"shell_counts": shells.shell_counts},
    )
