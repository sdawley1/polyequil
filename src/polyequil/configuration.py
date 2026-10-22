"""
One frame of a linear-polymer system.

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
from numpy.typing import NDArray

# ----- local -----
from polyequil._arrays import frozen_array
from polyequil.box import Box
from polyequil.errors import ValidationError
from polyequil.topology import ChainTopology

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class ChainConfiguration:
    """Positions (as stored, usually wrapped), image flags, and topology of one frame.

    Positions are not required to lie inside the box: LAMMPS dumps atoms that
    drifted out between re-neighbourings, and the image flags are consistent
    with whatever was stored. ``unwrapped()`` is always ``x + images * L``.
    """

    box: Box
    positions: FloatArray
    images: NDArray[np.int32]
    topology: ChainTopology
    step: int
    time: float
    velocities: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box) or not isinstance(
            self.topology, ChainTopology
        ):
            raise ValidationError("box must be a Box and topology a ChainTopology")
        shape = (self.topology.n_beads, 3)
        positions = frozen_array(
            self.positions, name="positions", dtype=np.float64, shape=shape, finite=True
        )
        images = frozen_array(self.images, name="images", dtype=np.int32, shape=shape)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "images", images)
        if self.velocities is not None:
            velocities = frozen_array(
                self.velocities, name="velocities", dtype=np.float64, shape=shape
            )
            object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "step", _as_int(self.step, "step"))
        time = float(self.time)
        if not np.isfinite(time):
            raise ValidationError(f"time must be finite, got {self.time!r}")
        object.__setattr__(self, "time", time)

    @property
    def n_beads(self) -> int:
        """Return the number of beads."""
        return self.topology.n_beads

    def unwrapped(self) -> FloatArray:
        """Return unwrapped positions ``x + images * L``, shape (n, 3)."""
        return self.positions + self.images * self.box.lengths

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor so arrays come back read-only."""
        fields = (self.box, self.positions, self.images, self.topology)
        return (type(self), (*fields, self.step, self.time, self.velocities))


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _as_int(value: object, name: str) -> int:
    """Return ``value`` as a Python int; reject bools and non-integers."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be an integer, got {value!r}")
    return int(value)
