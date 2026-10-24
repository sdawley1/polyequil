import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polyequil.ensemble import TrajectoryEnsemble
from polyequil.errors import ValidationError
from polyequil.indices import (
    Estimate,
    density_index,
    estimate_from,
    orientational_index,
)
from tests import observable_fixtures as fx

positive = st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False)


def test_indices_vanish_when_the_relations_hold():
    x_rho = density_index(
        Estimate(0.85 * 1.0 * 0.3, 0.0),
        density=0.85,
        temperature=1.0,
        kappa_t=Estimate(0.3, 0.0),
    )
    x_q = orientational_index(
        Estimate(0.2, 0.0), temperature=1.5, chi=Estimate(0.2 / 1.5, 0)
    )
    for x in (x_rho, x_q):
        np.testing.assert_allclose(x.value, 0.0, atol=1e-15)
        assert x.error == 0.0


@given(positive, positive, positive, positive, st.floats(0, 1), st.floats(0, 1))
def test_density_index_round_trips_and_propagates_errors(s0, rho, t, kappa, rs, rk):
    x = density_index(
        Estimate(s0, rs * s0),
        density=rho,
        temperature=t,
        kappa_t=Estimate(kappa, rk * kappa),
    )
    ratio = s0 / (rho * t * kappa)
    np.testing.assert_allclose(x.value, ratio - 1.0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(x.error, ratio * np.hypot(rs, rk), rtol=1e-12)


def test_propagated_error_matches_monte_carlo():
    rng = np.random.default_rng(5)
    s0 = rng.normal(0.25, 0.0025, 200_000)
    chi = rng.normal(0.12, 0.0012, 200_000)
    sampled = s0 / (1.0 * chi) - 1.0
    x = orientational_index(
        Estimate(0.25, 0.0025), temperature=1.0, chi=Estimate(0.12, 0.0012)
    )
    np.testing.assert_allclose(x.error, sampled.std(), rtol=0.03)
    np.testing.assert_allclose(
        x.value, sampled.mean(), atol=3 * x.error / np.sqrt(2e5) + 1e-4
    )


def test_time_resolved_inputs_broadcast():
    s0 = Estimate(np.array([0.2, 0.25, 0.3]), np.array([0.01, 0.01, 0.02]))
    x = density_index(s0, density=0.85, temperature=1.0, kappa_t=Estimate(0.3, 0.003))
    assert x.value.shape == x.error.shape == (3,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"density": 0.0},
        {"temperature": -1.0},
        {"kappa_t": Estimate(np.array(-0.3), np.array(0.0))},
        {"density": float("nan")},
    ],
)
def test_invalid_inputs_raise(kwargs):
    arguments = {"density": 0.85, "temperature": 1.0, "kappa_t": Estimate(0.3, 0.01)}
    arguments.update(kwargs)
    with pytest.raises(ValidationError):
        density_index(Estimate(0.25, 0.01), **arguments)
    with pytest.raises(ValidationError, match="error"):
        Estimate(1.0, -1.0)
    with pytest.raises(ValidationError, match="broadcast"):
        Estimate(np.ones(3), np.ones(2))


def test_estimate_from_an_ensemble_result(tmp_path):
    from tests.conftest import write_trajectory

    paths = [
        write_trajectory(tmp_path / f"r{k}.h5", seed=k, root_seed=k) for k in range(3)
    ]
    result = TrajectoryEnsemble(paths).evaluate(fx.mean_unwrapped)
    estimate = estimate_from(result, (2,))
    np.testing.assert_array_equal(estimate.value, result.mean[:, 2])
    np.testing.assert_array_equal(estimate.error, result.standard_error[:, 2])
    single = TrajectoryEnsemble(paths[:1]).evaluate(fx.mean_unwrapped)
    with pytest.raises(ValidationError, match="replica"):
        estimate_from(single, (0,))
