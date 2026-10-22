"""
Orthorhombic periodic simulation box.

Reduced from heteromorph's ``cell.py``/``domain.py``: MD boxes here are
orthorhombic, so triclinic lattice reduction is dropped and the minimum image
is closed-form. Conventions are in :mod:`polyequil.conventions`.

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
from polyequil._arrays import frozen_array
from polyequil.conventions import SPATIAL_DIMENSION
from polyequil.errors import ValidationError

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True, eq=False)
class Box:
    """A periodic orthorhombic box ``[lo, lo + lengths)`` in three dimensions."""

    lo: FloatArray
    lengths: FloatArray

    def __post_init__(self) -> None:
        dim = (SPATIAL_DIMENSION,)
        lo = frozen_array(
            self.lo, name="box lo", dtype=np.float64, shape=dim, finite=True
        )
        lengths = frozen_array(
            self.lengths, name="box lengths", dtype=np.float64, shape=dim, finite=True
        )
        if np.any(lengths <= 0.0):
            raise ValidationError(f"box lengths must be positive, got {lengths}")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "lengths", lengths)

    @classmethod
    def from_bounds(cls, lo: ArrayLike, hi: ArrayLike) -> "Box":
        """Build a box from its lower and upper corners."""
        lo_array = np.asarray(lo, dtype=np.float64)
        return cls(lo=lo_array, lengths=np.asarray(hi, dtype=np.float64) - lo_array)

    @property
    def hi(self) -> FloatArray:
        """Return the upper corner ``lo + lengths``."""
        return self.lo + self.lengths

    @property
    def volume(self) -> float:
        """Return the box volume."""
        return float(np.prod(self.lengths))

    def to_fractional(self, positions: ArrayLike) -> FloatArray:
        """Return fractional coordinates ``(x - lo) / L`` (not wrapped)."""
        return (_points(positions) - self.lo) / self.lengths

    def to_cartesian(self, fractional: ArrayLike) -> FloatArray:
        """Return Cartesian coordinates ``lo + s * L``."""
        return self.lo + _points(fractional) * self.lengths

    def image_flags(self, positions: ArrayLike) -> IntArray:
        """Return the integer image of each position: ``floor((x - lo) / L)``."""
        return np.floor(self.to_fractional(positions)).astype(np.int64)

    def wrap(self, positions: ArrayLike) -> FloatArray:
        """Return positions mapped into ``[lo, lo + L)``; idempotent."""
        fractional = self.to_fractional(positions)
        fractional -= np.floor(fractional)
        # floor() can leave exactly 1.0 when x is a hair below lo.
        fractional[fractional >= 1.0] = 0.0
        return self.to_cartesian(fractional)

    def minimum_image(self, separations: ArrayLike) -> FloatArray:
        """Return separations mapped into ``[-L/2, L/2)``; ``+L/2`` maps to ``-L/2``."""
        d = _points(separations)
        return d - self.lengths * np.floor(d / self.lengths + 0.5)

    def __eq__(self, other: object) -> bool:
        """Return whether two boxes have identical corners and lengths."""
        if not isinstance(other, Box):
            return NotImplemented
        return bool(
            np.array_equal(self.lo, other.lo)
            and np.array_equal(self.lengths, other.lengths)
        )

    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor so arrays come back read-only."""
        return (type(self), (self.lo, self.lengths))


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _points(values: ArrayLike) -> FloatArray:
    """Return a float64 array whose last axis is the spatial dimension."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != SPATIAL_DIMENSION:
        raise ValidationError(
            f"last axis must have length {SPATIAL_DIMENSION}, got shape {array.shape}"
        )
    return array
