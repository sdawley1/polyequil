import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from polyequil.box import Box
from polyequil.observables.chain import per_chain_rg2, radius_of_gyration
from tests.conftest import random_walk_frame
from tests.strategies import topologies


@given(topologies(), st.integers(0, 2**32 - 1))
def test_rg2_matches_a_brute_force_loop(top, seed):
    box = Box(lo=np.zeros(3), lengths=np.full(3, 6.0))
    x = random_walk_frame(top, box, np.random.default_rng(seed)).unwrapped()
    expected = [
        np.mean(
            np.sum((x[top.chain_id == c] - x[top.chain_id == c].mean(0)) ** 2, axis=1)
        )
        for c in range(top.n_chains)
    ]
    np.testing.assert_allclose(
        per_chain_rg2(x, top.chain_id), expected, rtol=1e-12, atol=1e-12
    )


@given(topologies(), st.integers(0, 2**32 - 1), st.integers(-3, 3))
def test_rg_is_invariant_under_periodic_image_shifts(top, seed, shift):
    box = Box(lo=np.zeros(3), lengths=np.full(3, 6.0))
    frame = random_walk_frame(top, box, np.random.default_rng(seed))
    moved = type(frame)(box, frame.positions, frame.images + shift, top, 0, 0.0)
    np.testing.assert_allclose(
        radius_of_gyration(moved).values,
        radius_of_gyration(frame).values,
        rtol=1e-9,
        atol=1e-12,
    )
