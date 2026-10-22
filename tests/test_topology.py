import pickle

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polyequil.box import Box
from polyequil.errors import ValidationError
from polyequil.topology import ChainTopology, linear_chains
from tests.conftest import random_walk_frame
from tests.strategies import topologies


def test_linear_chains_layout():
    top = linear_chains([3, 2])
    assert top.n_beads == 5 and top.n_chains == 2
    np.testing.assert_array_equal(top.bonds, [[0, 1], [1, 2], [3, 4]])
    np.testing.assert_array_equal(top.angles, [[0, 1, 2]])
    np.testing.assert_array_equal(top.bead_index_in_chain, [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(top.chain_starts, [0, 3])


def test_bonds_and_angles_are_canonicalized():
    top = ChainTopology(
        bonds=[[2, 1], [0, 1]],
        angles=[[2, 1, 0]],
        chain_id=[0, 0, 0],
        bead_type=[1, 1, 1],
    )
    np.testing.assert_array_equal(top.bonds, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(top.angles, [[0, 1, 2]])


@pytest.mark.parametrize(
    ("bonds", "angles", "chain_id"),
    [
        ([[0, 1], [0, 2]], [], [0, 0, 0]),  # branched
        ([[0, 1]], [], [0, 0, 0]),  # missing bond
        ([[0, 1], [1, 2]], [], [0, 0, 1]),  # bond across chains
        ([[0, 1]], [], [0, 0, 2]),  # chain ids skip
        ([[0, 1], [1, 2]], [[0, 2, 1]], [0, 0, 0]),  # vertex not in the middle
    ],
)
def test_non_linear_or_inconsistent_topologies_are_rejected(bonds, angles, chain_id):
    with pytest.raises(ValidationError):
        ChainTopology(bonds, angles, chain_id, np.ones(len(chain_id)))


@given(topologies(), st.integers(1, 12))
def test_pairs_at_separation_counts_and_stay_in_chain(top, n):
    i, j = top.pairs_at_separation(n)
    assert i.size == int(np.clip(top.chain_lengths - n, 0, None).sum())
    np.testing.assert_array_equal(top.chain_id[i], top.chain_id[j])
    np.testing.assert_array_equal(j - i, n)


@given(topologies(), st.integers(0, 2**32 - 1))
def test_make_whole_recovers_bond_geometry(top, seed):
    box = Box(lo=np.full(3, -1.0), lengths=np.array([5.0, 6.0, 7.0]))
    frame = random_walk_frame(top, box, np.random.default_rng(seed))
    images = top.make_whole(frame.positions, box)
    rebuilt = frame.positions + images * box.lengths
    truth = frame.unwrapped()
    b = top.bonds
    np.testing.assert_allclose(
        rebuilt[b[:, 1]] - rebuilt[b[:, 0]], truth[b[:, 1]] - truth[b[:, 0]], atol=1e-9
    )


def test_equality_and_pickling():
    top = linear_chains([4, 4])
    copy = pickle.loads(pickle.dumps(top))
    assert copy == top and copy != linear_chains([3, 5])
    assert not copy.bonds.flags.writeable
