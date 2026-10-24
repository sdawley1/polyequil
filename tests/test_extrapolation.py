import functools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from polyequil.ensemble import TrajectoryEnsemble
from polyequil.errors import ValidationError
from polyequil.observables.extrapolation import fit_small_k, small_k_limit
from polyequil.observables.structure import (
    orientational_structure_factor,
    structure_factor,
)
from polyequil.observables.window import window_mean
from polyequil.reciprocal import reciprocal_shells
from polyequil.trajectory import Trajectory
from tests import observable_fixtures as fx
from tests.conftest import chain_frame

CONFIG = chain_frame(0, box_length=20.0)
K_FIT = 1.2  # L = 20 -> k_min = 0.314: about a dozen shells


def test_noiseless_quadratic_is_recovered_exactly():
    result = small_k_limit(
        CONFIG, frame_observable=fx.quadratic_shells, k_fit=K_FIT, s0=0.27, a=1.3
    )
    np.testing.assert_allclose(result.values, [0.27, 1.3], rtol=1e-12)
    assert result.convention["k_fit"] == K_FIT
    assert result.convention["fit_shells"][-1] <= K_FIT


def test_intervals_have_nominal_coverage():
    """T6: per-mode exponential noise, M = 16 replicas, 2 s.e. intervals."""
    rng = np.random.default_rng(2026)
    noisy = functools.partial(fx.quadratic_shells, rng=rng)
    trials, replicas, hits = 400, 16, 0
    for _ in range(trials):
        s0 = np.array(
            [
                small_k_limit(
                    CONFIG, frame_observable=noisy, k_fit=K_FIT, s0=0.27, a=1.3
                ).values[0]
                for _ in range(replicas)
            ]
        )
        error = s0.std(ddof=1) / np.sqrt(replicas)
        hits += abs(s0.mean() - 0.27) < 2 * error
    # Student t with 15 dof: P(|t| < 2) = 0.936; binomial sd over 400 trials = 0.012
    assert 0.90 < hits / trials < 0.97


shell_values = hnp.arrays(
    np.float64, st.tuples(st.integers(2, 8), st.just(6)), elements=st.floats(0.0, 10.0)
)


@given(shell_values)
def test_fit_is_linear_so_the_mean_of_fits_is_the_fit_of_the_mean(values):
    k = np.linspace(0.3, 1.2, 6)
    counts = np.array([3, 6, 4, 12, 3, 12])
    fits = np.array([fit_small_k(k, row, counts, k_fit=1.2) for row in values])
    np.testing.assert_allclose(
        fits.mean(0), fit_small_k(k, values.mean(0), counts, k_fit=1.2), atol=1e-10
    )


def test_orientational_fit_excludes_the_direct_k0_value():
    result = small_k_limit(
        CONFIG, frame_observable=orientational_structure_factor, k_fit=K_FIT
    )
    assert result.convention["fit_shells"][0] > 0.0
    direct = orientational_structure_factor(CONFIG, k_max=0.0).values[0]
    assert np.isfinite(result.values[0]) and np.isfinite(direct)


def test_windows_are_averaged_before_fitting(ex_files):
    data = Trajectory(ex_files[0]).window(1000.0, 1100.0).load()
    got = small_k_limit(data, frame_observable=structure_factor, k_fit=1.5)
    averaged = window_mean(data, frame_observable=structure_factor, k_max=1.5)
    k = averaged.coordinates.axes[0][1]
    expected = fit_small_k(
        k, averaged.values, averaged.auxiliary["shell_counts"], k_fit=1.5
    )
    np.testing.assert_allclose(got.values, expected, rtol=1e-14)


def test_ensemble_s0_on_reference_data(ex_files):
    result = TrajectoryEnsemble(ex_files).evaluate(
        small_k_limit, times=[5000.0], frame_observable=structure_factor, k_fit=1.5
    )
    assert result.mean.shape == (1, 2) and result.has_uncertainty


def test_invalid_requests_raise():
    with pytest.raises(ValidationError, match="two shells"):
        small_k_limit(
            CONFIG, frame_observable=fx.quadratic_shells, k_fit=0.35, s0=1.0, a=0.0
        )
    with pytest.raises(ValidationError, match="k_max"):
        small_k_limit(CONFIG, frame_observable=structure_factor, k_fit=1.0, k_max=2.0)
    with pytest.raises(ValidationError, match="k_fit"):
        small_k_limit(CONFIG, frame_observable=structure_factor, k_fit=-1.0)
    shells = reciprocal_shells(CONFIG.box, 1.0, include_zero=False)
    with pytest.raises(ValidationError, match="shape"):
        fit_small_k(shells.shell_k, np.ones(3), shells.shell_counts, k_fit=1.0)
