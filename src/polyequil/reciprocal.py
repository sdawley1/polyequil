"""
Reciprocal-lattice modes, shells, and direct collective-coordinate sums.

Ported from heteromorph's ``point_patterns.py`` (``_validated_mode_limits``,
``_component_amplitude``, and the radial reduction), reduced to orthorhombic
boxes. Modes are restricted to a half-space because every weight used here is
real, so the amplitude at -k is the conjugate of that at k and carries no new
information. Conventions are in :mod:`polyequil.conventions`.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import dataclass

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ----- local -----
from polyequil._arrays import frozen_array
from polyequil.box import Box
from polyequil.conventions import SHELL_RELATIVE_TOLERANCE, SPATIAL_DIMENSION
from polyequil.errors import ValidationError

# ============================================================
# CONSTANTS
# ============================================================

MAX_DIRECT_MODES: int = 5_000_000
PARTICLE_BLOCK: int = 4096
PHASE_ENTRIES_PER_CHUNK: int = 1_000_000

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True, eq=False)
class ReciprocalShells:
    """Half-space reciprocal modes of a box, grouped into shells of equal |k|.

    ``modes`` (M, 3) are integer mode indices sorted by shell; ``shell_index``
    (M,) maps each mode to its shell; ``shell_k`` (S,) is strictly increasing;
    ``shell_counts`` (S,) counts the half-space modes of each shell. When
    ``includes_zero`` the first shell is the single mode k = 0.
    """

    modes: IntArray
    shell_index: IntArray
    shell_k: FloatArray
    shell_counts: IntArray
    includes_zero: bool

    def __post_init__(self) -> None:
        m = np.asarray(self.modes).shape[0]
        modes = frozen_array(self.modes, name="modes", dtype=np.int64, shape=(m, 3))
        index = frozen_array(
            self.shell_index, name="shell_index", dtype=np.int64, shape=(m,)
        )
        shell_k = frozen_array(
            self.shell_k, name="shell_k", dtype=np.float64, shape=(None,), finite=True
        )
        counts = frozen_array(
            self.shell_counts, name="shell_counts", dtype=np.int64, shape=shell_k.shape
        )
        if m == 0 or np.any(np.diff(shell_k) <= 0.0):
            raise ValidationError("shells need modes and strictly increasing |k|")
        if not np.array_equal(np.bincount(index, minlength=counts.size), counts):
            raise ValidationError("shell_counts disagree with shell_index")
        for name, value in (
            ("modes", modes),
            ("shell_index", index),
            ("shell_k", shell_k),
            ("shell_counts", counts),
        ):
            object.__setattr__(self, name, value)

    @property
    def mode_k(self) -> FloatArray:
        """Return |k| of every mode, shape (M,)."""
        return self.shell_k[self.shell_index]


# ============================================================
# PUBLIC API
# ============================================================


def reciprocal_shells(
    box: Box, k_max: float, *, include_zero: bool
) -> ReciprocalShells:
    """Return every half-space mode with ``|k| <= k_max``, grouped into shells.

    Parameters
    ----------
    box : Box
        Orthorhombic periodic box; ``k = 2 pi n / L``.
    k_max : float
        Largest wavenumber, in 1/sigma. With ``include_zero`` it may be 0 (or
        below the first nonzero shell), giving the k = 0 mode alone.
    include_zero : bool
        Include the k = 0 mode as its own first shell.

    Returns
    -------
    ReciprocalShells
        Modes, shell assignment, shell wavenumbers, and degeneracies.

    Raises
    ------
    ValidationError
        If ``k_max`` is negative or not finite, no mode qualifies, or the mode
        count exceeds ``MAX_DIRECT_MODES``.
    """
    valid = isinstance(k_max, (int, float)) and np.isfinite(k_max)
    if not valid or k_max < 0 or (k_max == 0 and not include_zero):
        raise ValidationError(f"k_max must be positive and finite, got {k_max!r}")
    n_max = np.floor(k_max * box.lengths / (2.0 * np.pi)).astype(np.int64)
    estimate = int(np.prod(2 * n_max + 1))
    if estimate > 2 * MAX_DIRECT_MODES:
        raise ValidationError(
            f"k_max = {k_max} needs ~{estimate // 2} modes; at most "
            f"{MAX_DIRECT_MODES} are supported by the direct sum"
        )
    modes = _half_space_modes(n_max, include_zero)
    k2 = np.sum((2.0 * np.pi * modes / box.lengths) ** 2, axis=1)
    keep = k2 <= (k_max * (1.0 + 1e-12)) ** 2
    modes, k2 = modes[keep], k2[keep]
    if modes.shape[0] == int(include_zero) and not include_zero:
        raise ValidationError(
            f"no reciprocal mode with 0 < |k| <= {k_max}; the smallest is "
            f"{2.0 * np.pi / float(np.max(box.lengths)):.4g}"
        )
    order = np.argsort(k2, kind="stable")
    modes, k2 = modes[order], k2[order]
    index, first = _group_equal(k2)
    return ReciprocalShells(
        modes=modes,
        shell_index=index,
        shell_k=np.sqrt(k2[first]),
        shell_counts=np.bincount(index),
        includes_zero=include_zero,
    )


def collective_amplitudes(
    positions: ArrayLike,
    weights: ArrayLike | None,
    box: Box,
    modes: IntArray,
) -> NDArray[np.complex128]:
    """Return ``sum_j w_j exp(-i k.r_j)`` for every mode, shape (M, c).

    Parameters
    ----------
    positions : ArrayLike
        (N, 3) positions; wrapped or not (the sum is periodic in ``r``).
    weights : ArrayLike | None
        (N, c) real weights, or None for unit weights (c = 1).
    box : Box
        The box defining ``k = 2 pi n / L``.
    modes : IntArray
        (M, 3) integer mode indices.

    Returns
    -------
    NDArray[np.complex128]
        Amplitudes, one row per mode and one column per weight component.
    """
    x = frozen_array(
        positions, name="positions", dtype=np.float64, shape=(None, 3), finite=True
    )
    n = x.shape[0]
    w = np.ones((n, 1)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != n:
        raise ValidationError(f"weights must have shape ({n}, c), got {w.shape}")
    mode_array = np.asarray(modes, dtype=np.int64)
    assert mode_array.ndim == 2 and mode_array.shape[1] == SPATIAL_DIMENSION
    # Phases are referenced to the origin, not box.lo, so the result is
    # exactly sum_j w_j exp(-i k.r_j); wrapping is exact for integer modes.
    fractional = x / box.lengths
    fractional -= np.floor(fractional)
    out = np.zeros((mode_array.shape[0], w.shape[1]), dtype=np.complex128)
    for start in range(0, n, PARTICLE_BLOCK):
        stop = min(n, start + PARTICLE_BLOCK)
        out += _block_amplitudes(fractional[start:stop], w[start:stop], mode_array)
    return out


def shell_mean(per_mode: ArrayLike, shells: ReciprocalShells) -> NDArray[np.float64]:
    """Return the mean of ``per_mode`` (M, ...) over the modes of each shell."""
    values = np.asarray(per_mode, dtype=np.float64)
    if values.shape[:1] != shells.shell_index.shape:
        raise ValidationError(
            f"per_mode has {values.shape[:1]} rows, expected {shells.shell_index.shape}"
        )
    sums = np.zeros((shells.shell_k.size, *values.shape[1:]))
    np.add.at(sums, shells.shell_index, values)
    counts = shells.shell_counts.reshape(-1, *([1] * (values.ndim - 1)))
    mean: NDArray[np.float64] = sums / counts
    return mean


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _half_space_modes(n_max: IntArray, include_zero: bool) -> IntArray:
    """Return integer modes in the box ``|n_a| <= n_max_a`` with n > 0 lexically."""
    axes = [np.arange(-m, m + 1, dtype=np.int64) for m in n_max]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    nx, ny, nz = grid.T
    positive = (nx > 0) | ((nx == 0) & (ny > 0)) | ((nx == 0) & (ny == 0) & (nz > 0))
    if include_zero:
        positive |= (nx == 0) & (ny == 0) & (nz == 0)
    modes: IntArray = grid[positive]
    return modes


def _group_equal(sorted_k2: FloatArray) -> tuple[IntArray, IntArray]:
    """Group sorted |k|^2 values equal to a relative tolerance.

    Returns each value's group index and the index of each group's first value.
    """
    scale = max(float(sorted_k2[-1]), 1.0)
    breaks = np.diff(sorted_k2) > SHELL_RELATIVE_TOLERANCE * scale
    index = np.concatenate([[0], np.cumsum(breaks)]).astype(np.int64)
    first = np.concatenate([[0], np.flatnonzero(breaks) + 1]).astype(np.int64)
    return index, first


def _block_amplitudes(
    fractional: FloatArray, weights: FloatArray, modes: IntArray
) -> NDArray[np.complex128]:
    """Sum one particle block over all modes using per-axis phase tables."""
    n_max = np.abs(modes).max(axis=0)
    tables = [
        np.exp(-2j * np.pi * np.outer(fractional[:, a], np.arange(-m, m + 1)))
        for a, m in enumerate(n_max)
    ]
    columns = modes + n_max
    out = np.empty((modes.shape[0], weights.shape[1]), dtype=np.complex128)
    chunk = max(1, PHASE_ENTRIES_PER_CHUNK // max(1, fractional.shape[0]))
    for start in range(0, modes.shape[0], chunk):
        cols = columns[start : start + chunk]
        phase = tables[0][:, cols[:, 0]] * tables[1][:, cols[:, 1]]
        phase *= tables[2][:, cols[:, 2]]
        out[start : start + chunk] = phase.T @ weights
    return out
