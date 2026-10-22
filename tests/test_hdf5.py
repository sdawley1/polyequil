import h5py
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polyequil.box import Box
from polyequil.conventions import SCHEMA
from polyequil.errors import FormatError, ValidationError
from polyequil.io.hdf5 import TrajectoryWriter, read_frame, read_header, read_thermo
from polyequil.topology import linear_chains
from tests.conftest import random_walk_frame
from tests.strategies import topologies


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=25)
@given(topologies(), st.integers(1, 4), st.booleans(), st.integers(0, 2**32 - 1))
def test_write_read_round_trip_is_bit_exact(tmp_path, top, n_frames, velocities, seed):
    rng = np.random.default_rng(seed)
    box = Box(lo=np.array([-1.0, 0.0, 2.0]), lengths=np.array([4.0, 5.0, 6.0]))
    frames = [
        random_walk_frame(top, box, rng, step=3 * k, velocities=velocities)
        for k in range(n_frames)
    ]
    path = tmp_path / f"rt_{seed}_{n_frames}.h5"
    with TrajectoryWriter(path, top, {"note": "x"}, overwrite=True) as writer:
        for frame in frames:
            writer.append(frame)
    header = read_header(path)
    assert header.topology == top and header.n_frames == n_frames
    assert header.attrs["schema"] == SCHEMA and header.attrs["note"] == "x"
    for k, frame in enumerate(frames):
        back = read_frame(path, k, header.topology)
        np.testing.assert_array_equal(back.positions, frame.positions)
        np.testing.assert_array_equal(back.images, frame.images)
        assert (
            back.box == frame.box
            and back.step == frame.step
            and back.time == frame.time
        )
        if velocities:
            np.testing.assert_array_equal(back.velocities, frame.velocities)
        else:
            assert back.velocities is None


def test_reads_the_reference_files(ex_files):
    header = read_header(ex_files[0])
    assert header.attrs["schema"] == "kg_template-1"
    assert header.n_frames == 70 and header.topology.n_chains == 100
    assert read_thermo(ex_files[0])["temp"].shape == (5001,)


def test_writer_refuses_to_clobber_and_checks_frames(tmp_path, synthetic):
    with pytest.raises(ValidationError, match="exists"):
        TrajectoryWriter(synthetic, linear_chains([2]))
    top = linear_chains([3])
    box = Box(lo=np.zeros(3), lengths=np.full(3, 5.0))
    rng = np.random.default_rng(1)
    with TrajectoryWriter(tmp_path / "w.h5", top) as writer:
        writer.append(random_walk_frame(top, box, rng, step=5))
        with pytest.raises(ValidationError, match="does not follow"):
            writer.append(random_walk_frame(top, box, rng, step=5))
        with pytest.raises(ValidationError, match="topology"):
            writer.append(random_walk_frame(linear_chains([2, 1]), box, rng, step=6))
        with pytest.raises(ValidationError, match="velocities"):
            writer.append(random_walk_frame(top, box, rng, step=7, velocities=False))
    assert read_header(tmp_path / "w.h5").n_frames == 1


def test_unknown_schema_is_a_format_error(synthetic):
    with h5py.File(synthetic, "a") as h5:
        h5.attrs["schema"] = "mystery-9"
    with pytest.raises(FormatError, match="schema"):
        read_header(synthetic)
