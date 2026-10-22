import pickle

import numpy as np
import pytest

from polyequil.errors import ValidationError
from polyequil.trajectory import FrameRef, Trajectory


def test_reference_trajectory_header_and_frames(ex_files):
    traj = Trajectory(ex_files[0])
    assert len(traj) == 70
    np.testing.assert_array_equal(traj.times[:3], [0.0, 5.0, 10.0])
    assert traj.times[-1] == 5000.0
    last = traj.frame(-1)
    bonds = last.topology.bonds
    x = last.unwrapped()
    assert np.linalg.norm(x[bonds[:, 0]] - x[bonds[:, 1]], axis=1).max() < 1.5


def test_frame_refs_are_picklable_and_load_in_isolation(ex_files):
    ref = Trajectory(ex_files[1]).ref(10)
    copy = pickle.loads(pickle.dumps(ref))
    assert isinstance(copy, FrameRef)
    np.testing.assert_array_equal(copy.load().positions, ref.load().positions)


def test_windows_are_inclusive_and_load_stacked_frames(ex_files):
    traj = Trajectory(ex_files[0])
    window = traj.window(1000.0, 2000.0)
    data = pickle.loads(pickle.dumps(window)).load()
    assert data.n_frames == 11
    np.testing.assert_array_equal(data.times[[0, -1]], [1000.0, 2000.0])
    np.testing.assert_array_equal(data.unwrapped()[3], data.frame(3).unwrapped())


def test_missing_times_and_bad_indices_raise(ex_files):
    traj = Trajectory(ex_files[0])
    with pytest.raises(ValidationError, match="no frame"):
        traj.index_of_time(150.0)
    with pytest.raises(ValidationError, match="no frames"):
        traj.window(5001.0, 6000.0)
    with pytest.raises(ValidationError, match="out of range"):
        traj.ref(70)
    assert traj.index_of_time(300.0) == 22
