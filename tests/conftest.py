"""Shared fixtures and Hypothesis profiles."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from polyequil.box import Box
from polyequil.configuration import ChainConfiguration
from polyequil.io.hdf5 import TrajectoryWriter
from polyequil.topology import ChainTopology, linear_chains

# ----- hypothesis profiles -----
settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile(
    "ci", max_examples=300, derandomize=True, print_blob=True, deadline=None
)
settings.register_profile(
    "thorough",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dev")

REPO = Path(__file__).resolve().parents[1]
EX_FILES = tuple(sorted((REPO / "trajectories" / "ex").glob("*.h5")))


@pytest.fixture
def ex_files() -> tuple[Path, ...]:
    """The read-only reference trajectories in trajectories/ex/."""
    assert len(EX_FILES) == 3
    return EX_FILES


def random_walk_frame(
    topology: ChainTopology,
    box: Box,
    rng: np.random.Generator,
    *,
    step: int = 0,
    dt: float = 0.01,
    bond: float = 0.97,
    velocities: bool = True,
) -> ChainConfiguration:
    """Return a frame of random-walk chains, stored wrapped with exact image flags."""
    n = topology.n_beads
    steps = rng.normal(size=(n, 3))
    steps *= bond / np.linalg.norm(steps, axis=1, keepdims=True)
    starts = topology.chain_starts[topology.chain_id] == np.arange(n)
    steps[starts] = rng.uniform(0.0, 1.0, size=(int(starts.sum()), 3)) * box.lengths
    walked = np.cumsum(steps, axis=0)
    first = topology.chain_starts[topology.chain_id]
    unwrapped = walked - walked[first] + steps[first] + box.lo
    images = box.image_flags(unwrapped).astype(np.int32)
    wrapped = unwrapped - images * box.lengths
    v = rng.normal(size=(n, 3)) if velocities else None
    return ChainConfiguration(box, wrapped, images, topology, step, step * dt, v)


def write_trajectory(
    path: Path,
    *,
    chain_lengths: tuple[int, ...] = (5, 5, 4),
    n_frames: int = 4,
    seed: int = 0,
    root_seed: int | None = None,
    stride: int = 10,
    velocities: bool = True,
) -> Path:
    """Write a small synthetic trajectory and return its path."""
    rng = np.random.default_rng(seed)
    topology = linear_chains(chain_lengths)
    box = Box(lo=np.zeros(3), lengths=np.array([6.0, 7.0, 8.0]))
    attrs = {"dt": 0.01} if root_seed is None else {"dt": 0.01, "root_seed": root_seed}
    with TrajectoryWriter(path, topology, attrs) as writer:
        for k in range(n_frames):
            writer.append(
                random_walk_frame(
                    topology, box, rng, step=k * stride, velocities=velocities
                )
            )
    return path


@pytest.fixture
def synthetic(tmp_path: Path) -> Path:
    """One small synthetic trajectory."""
    return write_trajectory(tmp_path / "synthetic.h5")


def point_configuration(positions: np.ndarray, box: Box) -> ChainConfiguration:
    """Return unbonded points (one-bead chains) stored wrapped with images."""
    images = box.image_flags(positions).astype(np.int32)
    topology = linear_chains(np.ones(len(positions), dtype=np.int64))
    wrapped = positions - images * box.lengths
    return ChainConfiguration(box, wrapped, images, topology, 0, 0.0)


def chain_frame(
    seed: int,
    *,
    chain_lengths: tuple[int, ...] = (20,) * 40,
    box_length: float = 10.0,
) -> ChainConfiguration:
    """Return random-walk chains with independent isotropic unit bonds."""
    topology = linear_chains(chain_lengths)
    box = Box(lo=np.zeros(3), lengths=np.full(3, box_length))
    return random_walk_frame(
        topology, box, np.random.default_rng(seed), bond=1.0, velocities=False
    )
