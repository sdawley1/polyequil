"""
Chain topology of a linear-polymer system.

Chains are linear, their beads are contiguous in index order, and bonds join
consecutive beads of the same chain (see :mod:`polyequil.conventions`). Bonds
and angles are stored canonically: each row increasing, rows sorted.

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
from polyequil.box import Box
from polyequil.errors import ValidationError

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True, eq=False)
class ChainTopology:
    """Bonds, angles, chain membership, and bead types of linear chains.

    Parameters
    ----------
    bonds : ArrayLike
        (n_bonds, 2) bead pairs; must be exactly the consecutive pairs of every
        chain, in any order or orientation.
    angles : ArrayLike
        (n_angles, 3) bead triples with the vertex in the middle; each must be
        three consecutive beads of one chain. May be empty.
    chain_id : ArrayLike
        (n,) chain index per bead: non-decreasing and contiguous from 0.
    bead_type : ArrayLike
        (n,) integer bead type.
    """

    bonds: IntArray
    angles: IntArray
    chain_id: IntArray
    bead_type: IntArray

    def __post_init__(self) -> None:
        chain_id = frozen_array(
            self.chain_id, name="chain_id", dtype=np.int64, shape=(None,)
        )
        n = chain_id.shape[0]
        if n == 0:
            raise ValidationError("a topology needs at least one bead")
        steps = np.diff(chain_id)
        if chain_id[0] != 0 or np.any((steps != 0) & (steps != 1)):
            raise ValidationError("chain_id must be contiguous and start at 0")
        bead_type = frozen_array(
            self.bead_type, name="bead_type", dtype=np.int64, shape=(n,)
        )
        bonds = _canonical_rows(self.bonds, "bonds", 2)
        angles = _canonical_rows(self.angles, "angles", 3)
        same = chain_id[1:] == chain_id[:-1]
        starts = np.flatnonzero(same)
        expected_bonds = np.stack([starts, starts + 1], axis=1)
        if not np.array_equal(bonds, expected_bonds):
            raise ValidationError(
                "bonds must be exactly the consecutive bead pairs of each linear chain"
            )
        if angles.size and not _angles_valid(angles, chain_id):
            raise ValidationError(
                "each angle must be three consecutive beads of a chain"
            )
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "bead_type", bead_type)
        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(self, "angles", angles)

    @property
    def n_beads(self) -> int:
        """Return the number of beads."""
        return int(self.chain_id.shape[0])

    @property
    def n_chains(self) -> int:
        """Return the number of chains."""
        return int(self.chain_id[-1]) + 1

    @property
    def chain_lengths(self) -> IntArray:
        """Return the number of beads in each chain, shape (n_chains,)."""
        return np.bincount(self.chain_id).astype(np.int64)

    @property
    def chain_starts(self) -> IntArray:
        """Return the index of the first bead of each chain, shape (n_chains,)."""
        return np.concatenate([[0], np.cumsum(self.chain_lengths)[:-1]]).astype(
            np.int64
        )

    @property
    def bead_index_in_chain(self) -> IntArray:
        """Return each bead's position along its chain (0 at the chain start)."""
        return (
            np.arange(self.n_beads, dtype=np.int64) - self.chain_starts[self.chain_id]
        )

    def pairs_at_separation(self, n: int) -> tuple[IntArray, IntArray]:
        """Return index arrays ``(i, i + n)`` of all same-chain pairs n bonds apart."""
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValidationError(f"separation must be a positive int, got {n!r}")
        remaining = self.chain_lengths[self.chain_id] - self.bead_index_in_chain
        first = np.flatnonzero(remaining > n).astype(np.int64)
        return first, first + int(n)

    def make_whole(self, positions: ArrayLike, box: Box) -> NDArray[np.int32]:
        """Return image flags that make every chain whole.

        Walks each chain's bonds with the minimum-image convention, keeping
        the first bead of each chain in its stored image. Valid only when every
        bond is shorter than half the box length. The flags are correct for
        intra-chain quantities but carry no information about diffusion across
        frames.

        Parameters
        ----------
        positions : ArrayLike
            (n, 3) positions as stored (typically wrapped).
        box : Box
            The frame's box.

        Returns
        -------
        NDArray[np.int32]
            (n, 3) image flags such that ``positions + images * L`` is whole.
        """
        x = frozen_array(
            positions, name="positions", dtype=np.float64, shape=(self.n_beads, 3)
        )
        assert x.shape[0] == self.n_beads, "positions match the topology"
        delta = np.zeros_like(x)
        same = self.chain_id[1:] == self.chain_id[:-1]
        delta[1:][same] = box.minimum_image(x[1:][same] - x[:-1][same])
        walked = np.cumsum(delta, axis=0)
        first = self.chain_starts[self.chain_id]
        unwrapped = x[first] + walked - walked[first]
        images = np.rint((unwrapped - x) / box.lengths)
        assert np.all(np.isfinite(images)), "image reconstruction stayed finite"
        flags: NDArray[np.int32] = images.astype(np.int32)
        return flags

    def __eq__(self, other: object) -> bool:
        """Return whether two topologies have identical arrays."""
        if not isinstance(other, ChainTopology):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in ("bonds", "angles", "chain_id", "bead_type")
        )

    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor so arrays come back read-only."""
        return (type(self), (self.bonds, self.angles, self.chain_id, self.bead_type))


# ============================================================
# PUBLIC API
# ============================================================


def linear_chains(chain_lengths: ArrayLike) -> ChainTopology:
    """Return the topology of linear chains with the given lengths, all type 1."""
    lengths = np.asarray(chain_lengths, dtype=np.int64)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths < 1):
        raise ValidationError(f"chain lengths must be positive, got {lengths}")
    chain_id = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    same = chain_id[1:] == chain_id[:-1]
    first = np.flatnonzero(same)
    bonds = np.stack([first, first + 1], axis=1)
    middle = np.flatnonzero(same[:-1] & same[1:]) + 1
    angles = np.stack([middle - 1, middle, middle + 1], axis=1)
    return ChainTopology(bonds, angles.reshape(-1, 3), chain_id, np.ones_like(chain_id))


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _canonical_rows(values: ArrayLike, name: str, width: int) -> IntArray:
    """Return rows oriented increasing and sorted lexicographically, read-only."""
    raw = np.asarray(values)
    if raw.size == 0:
        raw = raw.reshape(0, width)
    rows = np.array(frozen_array(raw, name=name, dtype=np.int64, shape=(None, width)))
    flip = rows[:, 0] > rows[:, -1]
    rows[flip] = rows[flip, ::-1]
    rows = rows[np.lexsort(rows.T[::-1])] if rows.size else rows
    rows.setflags(write=False)
    return rows


def _angles_valid(angles: IntArray, chain_id: IntArray) -> bool:
    """Return whether every angle is (i, i+1, i+2) within one chain."""
    n = chain_id.shape[0]
    if np.any(angles < 0) or np.any(angles >= n):
        return False
    consecutive = np.all(np.diff(angles, axis=1) == 1)
    same_chain = np.all(chain_id[angles[:, 0]] == chain_id[angles[:, 2]])
    return bool(consecutive and same_chain)
