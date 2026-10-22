import pickle

import h5py
import numpy as np
import pytest

from polyequil.ensemble import TimeResolvedResult, TrajectoryEnsemble
from polyequil.errors import EnsembleError, ValidationError
from polyequil.execution import ExecutionPolicy
from polyequil.observables.chain import radius_of_gyration
from tests import observable_fixtures as fx
from tests.conftest import write_trajectory

OUTER = ExecutionPolicy("outer", total_cpus=2)


@pytest.fixture
def replicas(tmp_path):
    return [
        write_trajectory(tmp_path / f"r{k}.h5", seed=k, root_seed=k) for k in range(3)
    ]


def _direct_rg2(path, frame):
    with h5py.File(path) as h5:
        length = h5["frames/box_hi"][frame] - h5["frames/box_lo"][frame]
        x = h5["frames/positions"][frame] + h5["frames/images"][frame] * length
    chains = x.reshape(100, 20, 3)
    return np.mean(np.sum((chains - chains.mean(1, keepdims=True)) ** 2, axis=-1))


def test_rg_on_reference_data_matches_numpy_and_is_mode_independent(ex_files):
    ensemble = TrajectoryEnsemble(ex_files)
    serial = ensemble.evaluate(radius_of_gyration, times=[0.0, 5000.0])
    outer = ensemble.evaluate(radius_of_gyration, times=[0.0, 5000.0], execution=OUTER)
    np.testing.assert_array_equal(serial.mean, outer.mean)
    np.testing.assert_array_equal(serial.standard_error, outer.standard_error)
    direct = np.array([[_direct_rg2(p, f) for p in ex_files] for f in (0, -1)])
    np.testing.assert_allclose(serial.mean, direct.mean(1), rtol=1e-12)
    np.testing.assert_allclose(
        serial.standard_deviation, direct.std(1, ddof=1), rtol=1e-9
    )
    assert serial.replica_count == 3 and serial.kind == "frame"


def test_full_grid_and_coordinates(replicas):
    result = TrajectoryEnsemble(replicas).evaluate(fx.mean_unwrapped)
    assert result.mean.shape == (4, 3)
    np.testing.assert_array_equal(result.times, [0.0, 0.1, 0.2, 0.3])
    assert result.coordinates.names() == ("component",)


def test_single_replica_reports_no_uncertainty(replicas):
    result = TrajectoryEnsemble(replicas[:1]).evaluate(fx.bead_count)
    assert not result.has_uncertainty and result.standard_error is None


def test_retained_samples_do_not_change_statistics(replicas):
    ensemble = TrajectoryEnsemble(replicas)
    plain = ensemble.evaluate(fx.mean_unwrapped)
    kept = ensemble.evaluate(fx.mean_unwrapped, retain_samples=True)
    np.testing.assert_array_equal(plain.mean, kept.mean)
    assert kept.samples.shape == (4, 3, 3)
    np.testing.assert_allclose(kept.samples.mean(axis=1), kept.mean, rtol=1e-12)


def test_result_survives_pickling(replicas):
    result = TrajectoryEnsemble(replicas).evaluate(fx.mean_unwrapped)
    copy = pickle.loads(pickle.dumps(result))
    assert isinstance(copy, TimeResolvedResult) and not copy.mean.flags.writeable
    np.testing.assert_array_equal(copy.mean, result.mean)


def test_independence_checks(tmp_path, replicas):
    with pytest.raises(ValidationError, match="more than once"):
        TrajectoryEnsemble([replicas[0], replicas[0]])
    clone = write_trajectory(tmp_path / "clone.h5", seed=9, root_seed=0)
    with pytest.raises(ValidationError, match="root_seed"):
        TrajectoryEnsemble([replicas[0], clone])
    other = write_trajectory(tmp_path / "other.h5", chain_lengths=(7, 7), root_seed=5)
    with pytest.raises(ValidationError, match="topology"):
        TrajectoryEnsemble([replicas[0], other])


def test_mismatched_step_grids_need_explicit_times(tmp_path, replicas):
    sparse = write_trajectory(
        tmp_path / "s.h5", seed=4, root_seed=4, stride=20, n_frames=2
    )
    ensemble = TrajectoryEnsemble([replicas[0], sparse])
    with pytest.raises(ValidationError, match="step grid"):
        ensemble.evaluate(fx.bead_count)
    result = ensemble.evaluate(fx.bead_count, times=[0.0, 0.2])
    np.testing.assert_array_equal(result.mean, [14.0, 14.0])
    with pytest.raises(ValidationError, match="no frame"):
        ensemble.evaluate(fx.bead_count, times=[0.1])


@pytest.mark.parametrize("policy", [None, OUTER])
def test_failures_name_the_lowest_offending_task(replicas, policy):
    with pytest.raises(EnsembleError, match=r"replica 0 \(r0.h5\) at t = 0.1") as info:
        TrajectoryEnsemble(replicas).evaluate(
            fx.fails_at_step, step=10, execution=policy
        )
    assert info.value.task_indices == (3,)


def test_incompatible_results_are_attributed(replicas):
    with pytest.raises(EnsembleError, match="incompatible"):
        TrajectoryEnsemble(replicas).evaluate(fx.shape_changes)


def test_worker_death_is_reported(replicas):
    with pytest.raises(EnsembleError, match="terminated"):
        TrajectoryEnsemble(replicas).evaluate(
            fx.exits_at_step, step=10, execution=OUTER
        )


@pytest.mark.parametrize(
    ("observable", "kwargs", "fragment"),
    [
        (fx.bead_count, {"threads": 2}, "injected"),
        (fx.bead_count, {"rng": 1}, "injected"),
        (fx.bead_count, {"extra": np.random.default_rng()}, "random state"),
        (fx.missing_threads, {}, "threads"),
        (fx.positional_threads, {}, "threads"),
        (fx.bead_count, {"random_seed": 3}, "takes no rng"),
        (lambda c, *, threads=1: fx.bead_count(c), {"execution": OUTER}, "picklable"),
    ],
)
def test_caller_errors_are_raised_before_any_task(
    replicas, observable, kwargs, fragment
):
    with pytest.raises(ValidationError, match=fragment):
        TrajectoryEnsemble(replicas).evaluate(observable, **kwargs)


def test_seeded_streams_are_reproducible_distinct_and_mode_independent(replicas):
    ensemble = TrajectoryEnsemble(replicas)
    first = ensemble.evaluate(fx.gaussian, random_seed=7, retain_samples=True)
    again = ensemble.evaluate(
        fx.gaussian, random_seed=7, retain_samples=True, execution=OUTER
    )
    other = ensemble.evaluate(fx.gaussian, random_seed=8, retain_samples=True)
    np.testing.assert_array_equal(first.samples, again.samples)
    assert not np.array_equal(first.samples, other.samples)
    assert np.unique(first.samples).size == first.samples.size
    assert first.metadata["randomness"]["entropy_source"] == "user"


def test_window_evaluation(ex_files):
    result = TrajectoryEnsemble(ex_files).evaluate_windows(
        fx.window_span, windows=[(0.0, 95.0), (1000.0, 2000.0)], execution=OUTER
    )
    assert result.kind == "window" and result.times.shape == (2, 2)
    np.testing.assert_array_equal(result.mean, [[20.0, 95.0], [11.0, 1000.0]])
    np.testing.assert_array_equal(result.standard_deviation, 0.0)
