"""
Lazy trajectory handles: whole runs, single frames, and time windows.

A :class:`Trajectory` holds only a file's header (topology, steps, times,
attributes). :class:`FrameRef` and :class:`TrajectoryWindow` are small frozen,
picklable handles that worker processes load on demand, so an ensemble never
holds coordinates in memory beyond the tasks in flight.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil._arrays import frozen_array
from polyequil.box import Box
from polyequil.configuration import ChainConfiguration
from polyequil.conventions import IMAGES_SOURCES
from polyequil.errors import ValidationError
from polyequil.io.hdf5 import (
    TrajectoryHeader,
    read_frame,
    read_frames,
    read_header,
    read_thermo,
)
from polyequil.topology import ChainTopology

# ============================================================
# CONSTANTS
# ============================================================

HEADER_CACHE_SIZE: int = 16
TIME_TOLERANCE_IN_STEPS: float = 1e-6

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FrameRef:
    """A picklable handle to frame ``index`` of the trajectory file at ``path``."""

    path: Path
    index: int

    def load(self) -> ChainConfiguration:
        """Read the frame; the file header is cached per process."""
        return read_frame(self.path, self.index, header_for(self.path).topology)


@dataclass(frozen=True, slots=True)
class TrajectoryWindow:
    """A picklable handle to frames ``start:stop`` (stop exclusive) of one file."""

    path: Path
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.stop:
            raise ValidationError(
                f"need 0 <= start < stop, got {self.start}, {self.stop}"
            )

    def load(self) -> "WindowData":
        """Read every frame of the window into one :class:`WindowData`."""
        header = header_for(self.path)
        block = read_frames(self.path, self.start, self.stop)
        return WindowData(
            topology=header.topology,
            steps=block.steps,
            times=block.times,
            box_lo=block.box_lo,
            box_hi=block.box_hi,
            positions=block.positions,
            images=block.images,
            velocities=block.velocities,
            images_source=header.images_source,
        )


@dataclass(frozen=True, slots=True, eq=False)
class WindowData:
    """Coordinates of a contiguous span of one trajectory, frame axis first.

    ``images_source`` says whether image flags came from the simulation
    (``"dump"``) or were rebuilt per frame from bonds; rebuilt flags are valid
    within a frame only, so observables that track motion across frames (such
    as the MSD) must refuse them.
    """

    topology: ChainTopology
    steps: NDArray[np.int64]
    times: FloatArray
    box_lo: FloatArray
    box_hi: FloatArray
    positions: FloatArray
    images: NDArray[np.int32]
    velocities: FloatArray | None
    images_source: str

    def __post_init__(self) -> None:
        f = np.asarray(self.steps).shape[0]
        per_bead = (f, self.topology.n_beads, 3)
        arrays: dict[str, tuple[Any, tuple[int, ...]]] = {
            "steps": (np.int64, (f,)),
            "times": (np.float64, (f,)),
            "box_lo": (np.float64, (f, 3)),
            "box_hi": (np.float64, (f, 3)),
            "positions": (np.float64, per_bead),
            "images": (np.int32, per_bead),
        }
        if self.velocities is not None:
            arrays["velocities"] = (np.float64, per_bead)
        for name, (dtype, shape) in arrays.items():
            frozen = frozen_array(
                getattr(self, name), name=name, dtype=dtype, shape=shape
            )
            object.__setattr__(self, name, frozen)
        if self.images_source not in IMAGES_SOURCES:
            raise ValidationError(f"unknown images_source {self.images_source!r}")

    @property
    def n_frames(self) -> int:
        """Return the number of frames in the window."""
        return int(self.steps.shape[0])

    @property
    def box_lengths(self) -> FloatArray:
        """Return per-frame box lengths, shape (F, 3)."""
        return self.box_hi - self.box_lo

    def unwrapped(self) -> FloatArray:
        """Return unwrapped positions ``x + images * L``, shape (F, n, 3)."""
        return self.positions + self.images * self.box_lengths[:, None, :]

    def frame(self, index: int) -> ChainConfiguration:
        """Return one frame of the window as a :class:`ChainConfiguration`."""
        return ChainConfiguration(
            box=Box.from_bounds(self.box_lo[index], self.box_hi[index]),
            positions=self.positions[index],
            images=self.images[index],
            topology=self.topology,
            step=int(self.steps[index]),
            time=float(self.times[index]),
            velocities=None if self.velocities is None else self.velocities[index],
        )


class Trajectory:
    """A lazy view of one trajectory file: header in memory, frames on demand."""

    __slots__ = ("_header",)

    def __init__(self, path: str | Path) -> None:
        self._header = header_for(Path(path))

    def __len__(self) -> int:
        return self._header.n_frames

    def __repr__(self) -> str:
        return f"Trajectory({str(self.path)!r}, frames={len(self)})"

    @property
    def path(self) -> Path:
        """Return the resolved file path."""
        return self._header.path

    @property
    def header(self) -> TrajectoryHeader:
        """Return the cached file header."""
        return self._header

    @property
    def topology(self) -> ChainTopology:
        """Return the topology shared by every frame."""
        return self._header.topology

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Return the file's root attributes (frozen)."""
        return self._header.attrs

    @property
    def steps(self) -> NDArray[np.int64]:
        """Return the timestep of every frame."""
        return self._header.steps

    @property
    def times(self) -> FloatArray:
        """Return the time of every frame."""
        return self._header.times

    def ref(self, index: int) -> FrameRef:
        """Return a lazy handle to frame ``index`` (negative indices allowed)."""
        n = len(self)
        if not -n <= index < n:
            raise ValidationError(f"frame index {index} out of range for {n} frames")
        return FrameRef(self.path, index % n)

    def refs(self) -> tuple[FrameRef, ...]:
        """Return lazy handles to every frame in order."""
        return tuple(FrameRef(self.path, index) for index in range(len(self)))

    def frame(self, index: int) -> ChainConfiguration:
        """Load frame ``index`` (negative indices allowed)."""
        return self.ref(index).load()

    def index_of_time(self, time: float) -> int:
        """Return the index of the frame at ``time``; raise if there is none."""
        tolerance = TIME_TOLERANCE_IN_STEPS * _timestep(self._header)
        matches = np.flatnonzero(np.abs(self.times - time) <= tolerance)
        if matches.size != 1:
            raise ValidationError(f"{self.path.name} has no frame at t = {time:g}")
        return int(matches[0])

    def window(self, t_start: float, t_stop: float) -> TrajectoryWindow:
        """Return the frames with ``t_start <= t <= t_stop`` (both inclusive)."""
        tolerance = TIME_TOLERANCE_IN_STEPS * _timestep(self._header)
        inside = np.flatnonzero(
            (self.times >= t_start - tolerance) & (self.times <= t_stop + tolerance)
        )
        if inside.size == 0:
            raise ValidationError(
                f"{self.path.name} has no frames in [{t_start:g}, {t_stop:g}]"
            )
        return TrajectoryWindow(self.path, int(inside[0]), int(inside[-1]) + 1)

    def thermo(self) -> dict[str, NDArray[Any]]:
        """Return the thermodynamic columns stored with the trajectory."""
        return read_thermo(self.path)


# ============================================================
# PUBLIC API
# ============================================================


def header_for(path: Path) -> TrajectoryHeader:
    """Return the file's header, cached per process and invalidated on change."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _cached_header(str(resolved), stat.st_mtime_ns, stat.st_size)


# ============================================================
# PRIVATE HELPERS
# ============================================================


@functools.lru_cache(maxsize=HEADER_CACHE_SIZE)
def _cached_header(path: str, mtime_ns: int, size: int) -> TrajectoryHeader:
    """Read a header; the (mtime, size) key drops stale entries after a rewrite."""
    assert mtime_ns >= 0 and size >= 0, "stat values are non-negative"
    return read_header(path)


def _timestep(header: TrajectoryHeader) -> float:
    """Return the time per step (from attrs ``dt`` or the frame grid), else 1."""
    dt = header.attrs.get("dt")
    if isinstance(dt, (int, float)) and dt > 0:
        return float(dt)
    if header.n_frames > 1 and header.steps[1] != header.steps[0]:
        span = float(header.times[1] - header.times[0])
        return span / float(header.steps[1] - header.steps[0])
    return 1.0
