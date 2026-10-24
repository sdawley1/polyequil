import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from polyequil.box import Box
from polyequil.errors import ValidationError
from polyequil.reciprocal import collective_amplitudes, reciprocal_shells, shell_mean

CUBE = Box(lo=np.zeros(3), lengths=np.full(3, 5.0))
BRICK = Box(lo=np.array([-1.0, 0.5, 2.0]), lengths=np.array([4.0, 5.0, 7.0]))


def _brute_force_modes(box, k_max, include_zero):
    n_max = np.floor(k_max * box.lengths / (2 * np.pi)).astype(int)
    out = []
    for n in itertools.product(*(range(-m, m + 1) for m in n_max)):
        k = np.linalg.norm(2 * np.pi * np.array(n) / box.lengths)
        positive = next((c > 0 for c in n if c != 0), include_zero)
        if positive and k <= k_max * (1 + 1e-12):
            out.append((k, n))
    return out


@pytest.mark.parametrize("box", [CUBE, BRICK])
@pytest.mark.parametrize("include_zero", [False, True])
def test_shells_match_brute_force_enumeration(box, include_zero):
    shells = reciprocal_shells(box, 4.0, include_zero=include_zero)
    expected = _brute_force_modes(box, 4.0, include_zero)
    assert shells.modes.shape == (len(expected), 3)
    assert sorted(map(tuple, shells.modes)) == sorted(n for _, n in expected)
    k_of_mode = np.linalg.norm(2 * np.pi * shells.modes / box.lengths, axis=1)
    np.testing.assert_allclose(
        shells.shell_k[shells.shell_index], k_of_mode, rtol=1e-12
    )
    assert np.all(np.diff(shells.shell_k) > 0)
    assert shells.shell_counts.sum() == len(expected)
    assert shells.includes_zero == include_zero
    assert (shells.shell_k[0] == 0.0) == include_zero


def test_half_space_holds_no_opposite_pair():
    modes = {tuple(m) for m in reciprocal_shells(BRICK, 6.0, include_zero=False).modes}
    assert all(tuple(-np.array(m)) not in modes for m in modes)


def test_cubic_shell_degeneracies():
    shells = reciprocal_shells(CUBE, 2 * np.pi / 5.0 * 1.5, include_zero=False)
    # |n|^2 = 1 (3 half-space modes) and |n|^2 = 2 (6 half-space modes)
    np.testing.assert_array_equal(shells.shell_counts, [3, 6])


def test_invalid_k_max_raises():
    with pytest.raises(ValidationError, match="k_max"):
        reciprocal_shells(CUBE, -1.0, include_zero=False)
    with pytest.raises(ValidationError, match="no reciprocal"):
        reciprocal_shells(CUBE, 0.5, include_zero=False)


def test_zero_only_when_k_max_is_below_the_first_shell():
    shells = reciprocal_shells(CUBE, 0.0, include_zero=True)
    np.testing.assert_array_equal(shells.modes, [[0, 0, 0]])
    np.testing.assert_array_equal(shells.shell_counts, [1])


def _naive(positions, weights, box, modes):
    k = 2 * np.pi * modes / box.lengths
    return np.exp(-1j * k @ positions.T) @ weights


coords = st.floats(-30, 30, allow_nan=False, allow_subnormal=False)


@given(
    hnp.arrays(np.float64, st.tuples(st.integers(1, 40), st.just(3)), elements=coords),
    st.integers(1, 3),
    st.integers(0, 2**32 - 1),
)
def test_amplitudes_match_a_naive_exponential_sum(positions, c, seed):
    weights = np.random.default_rng(seed).normal(size=(positions.shape[0], c))
    modes = reciprocal_shells(BRICK, 5.0, include_zero=True).modes
    got = collective_amplitudes(positions, weights, BRICK, modes)
    want = _naive(positions, weights, BRICK, modes)
    scale = np.abs(weights).sum()
    np.testing.assert_allclose(got, want, atol=1e-9 * scale)


@given(st.integers(0, 2**32 - 1), st.integers(-3, 3))
def test_intensity_invariant_under_translation_images_and_relabelling(seed, shift):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 10, size=(30, 3))
    modes = reciprocal_shells(BRICK, 5.0, include_zero=False).modes
    base = np.abs(collective_amplitudes(x, None, BRICK, modes)) ** 2
    moved = x + rng.normal(size=3)
    images = x + rng.integers(-2, 3, size=x.shape) * BRICK.lengths * shift
    permuted = x[rng.permutation(len(x))]
    for other in (moved, images, permuted):
        got = np.abs(collective_amplitudes(other, None, BRICK, modes)) ** 2
        np.testing.assert_allclose(got, base, atol=1e-8 * len(x) ** 2)


def test_shell_mean_averages_modes_of_each_shell():
    shells = reciprocal_shells(CUBE, 2 * np.pi / 5.0 * 1.5, include_zero=False)
    per_mode = np.arange(shells.modes.shape[0], dtype=float)
    expected = [per_mode[shells.shell_index == s].mean() for s in range(2)]
    np.testing.assert_allclose(shell_mean(per_mode, shells), expected)
    stacked = np.stack([per_mode, 2 * per_mode], axis=1)
    np.testing.assert_allclose(
        shell_mean(stacked, shells)[:, 1], 2 * np.array(expected)
    )
