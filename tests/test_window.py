import numpy as np
import pytest

from polyequil.ensemble import TrajectoryEnsemble
from polyequil.errors import ValidationError
from polyequil.execution import ExecutionPolicy
from polyequil.observables.structure import structure_factor
from polyequil.observables.window import window_mean
from polyequil.trajectory import Trajectory
from tests import observable_fixtures as fx


def test_window_mean_equals_a_hand_loop(ex_files):
    data = Trajectory(ex_files[0]).window(1000.0, 1100.0).load()
    got = window_mean(data, frame_observable=structure_factor, k_max=1.2)
    frames = [structure_factor(data.frame(i), k_max=1.2) for i in range(data.n_frames)]
    np.testing.assert_allclose(
        got.values, np.mean([f.values for f in frames], axis=0), rtol=1e-14
    )
    assert got.auxiliary["frame_count"] == data.n_frames == 2
    np.testing.assert_array_equal(
        got.auxiliary["shell_counts"], frames[0].auxiliary["shell_counts"]
    )
    assert "window" in got.convention["time_average"]


def test_window_mean_through_the_ensemble(ex_files):
    result = TrajectoryEnsemble(ex_files).evaluate_windows(
        window_mean,
        windows=[(0.0, 45.0), (4000.0, 5000.0)],
        frame_observable=structure_factor,
        k_max=1.0,
        execution=ExecutionPolicy("outer", total_cpus=2),
    )
    assert result.mean.shape[0] == 2
    assert result.has_uncertainty


def test_incompatible_frames_raise(synthetic):
    data = Trajectory(synthetic).window(0.0, 1.0).load()
    with pytest.raises(ValidationError, match="frame 1"):
        window_mean(data, frame_observable=fx.shape_changes)
    with pytest.raises(ValidationError, match="callable"):
        window_mean(data, frame_observable="structure_factor")
