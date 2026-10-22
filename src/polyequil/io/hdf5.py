"""
Native HDF5 trajectory storage: lazy readers and a streaming writer.

Layout (schema ``polyequil-1``; ``kg_template-1`` files share it and are read
unchanged)::

    /                 attrs: schema, units, images_source, run parameters, ...
    /topology/bonds   (n_b, 2) int64      /topology/angles (n_a, 3) int64
    /topology/mol     (n,) int64          /topology/type   (n,) int64
    /frames/step      (F,) int64          /frames/time     (F,) float64
    /frames/box_lo    (F, 3) float64      /frames/box_hi   (F, 3) float64
    /frames/positions (F, n, 3) float64   /frames/images   (F, n, 3) int32
    /frames/velocities (F, n, 3) float64  (optional)
    /thermo/<name>    (T,)                (optional)

Frame datasets are chunked one frame per chunk, so any frame is read without
touching the rest of the file. Every reader opens and closes the file itself,
which keeps lazy handles picklable for worker processes.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

# ----- third party -----
import h5py
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil._metadata import freeze_mapping, freeze_metadata
from polyequil.box import Box
from polyequil.configuration import ChainConfiguration
from polyequil.conventions import (
    FRAME_CHUNK_COMPRESSION,
    FRAME_CHUNK_COMPRESSION_LEVEL,
    IMAGES_FROM_DUMP,
    IMAGES_SOURCES,
    POSITIONS_CONVENTION,
    READABLE_SCHEMAS,
    SCHEMA,
    UNITS,
)
from polyequil.errors import FormatError, ValidationError
from polyequil.topology import ChainTopology

# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_FRAME_DATASETS: tuple[str, ...] = (
    "step",
    "time",
    "box_lo",
    "box_hi",
    "positions",
    "images",
)
REQUIRED_TOPOLOGY_DATASETS: tuple[str, ...] = ("bonds", "angles", "mol", "type")

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class TrajectoryHeader:
    """Everything about a trajectory file except its per-frame coordinates."""

    path: Path
    attrs: Mapping[str, Any]
    topology: ChainTopology
    steps: NDArray[np.int64]
    times: FloatArray
    images_source: str
    has_velocities: bool

    @property
    def n_frames(self) -> int:
        """Return the number of frames."""
        return int(self.steps.shape[0])


@dataclass(frozen=True, slots=True, eq=False)
class FrameBlock:
    """Arrays for a contiguous range of frames, each with a leading frame axis."""

    steps: NDArray[np.int64]
    times: FloatArray
    box_lo: FloatArray
    box_hi: FloatArray
    positions: FloatArray
    images: NDArray[np.int32]
    velocities: FloatArray | None


# ============================================================
# READERS
# ============================================================


def read_header(path: str | Path) -> TrajectoryHeader:
    """Read attributes, topology, steps, and times without loading frames.

    Raises
    ------
    FormatError
        If the schema is unknown or a required dataset is missing or ragged.
    """
    resolved = Path(path).resolve()
    with h5py.File(resolved, "r") as h5:
        attrs = _attrs(h5)
        _require_layout(h5, resolved, attrs)
        topology = ChainTopology(
            bonds=h5["topology/bonds"][()],
            angles=h5["topology/angles"][()],
            chain_id=h5["topology/mol"][()],
            bead_type=h5["topology/type"][()],
        )
        steps = np.asarray(h5["frames/step"][()], dtype=np.int64)
        times = np.asarray(h5["frames/time"][()], dtype=np.float64)
        n_frames, n_beads = h5["frames/positions"].shape[:2]
        has_velocities = "velocities" in h5["frames"]
    if n_beads != topology.n_beads or steps.shape != (n_frames,):
        raise FormatError(f"{resolved}: frame datasets disagree with the topology")
    if n_frames > 1 and np.any(np.diff(steps) <= 0):
        raise FormatError(f"{resolved}: frame steps must strictly increase")
    source = str(attrs.get("images_source", IMAGES_FROM_DUMP))
    if source not in IMAGES_SOURCES:
        raise FormatError(f"{resolved}: unknown images_source {source!r}")
    for array in (steps, times):
        array.setflags(write=False)
    return TrajectoryHeader(
        resolved, attrs, topology, steps, times, source, has_velocities
    )


def read_frames(path: str | Path, start: int, stop: int) -> FrameBlock:
    """Read frames ``start:stop`` (stop exclusive) as one block of arrays."""
    if start < 0 or stop <= start:
        raise ValidationError(f"need 0 <= start < stop, got {start}, {stop}")
    with h5py.File(path, "r") as h5:
        frames = h5["frames"]
        if stop > frames["step"].shape[0]:
            raise ValidationError(
                f"frame range {start}:{stop} exceeds {frames['step'].shape[0]} frames"
            )
        window = slice(start, stop)
        velocities = frames["velocities"][window] if "velocities" in frames else None
        return FrameBlock(
            steps=np.asarray(frames["step"][window], dtype=np.int64),
            times=np.asarray(frames["time"][window], dtype=np.float64),
            box_lo=frames["box_lo"][window],
            box_hi=frames["box_hi"][window],
            positions=frames["positions"][window],
            images=np.asarray(frames["images"][window], dtype=np.int32),
            velocities=velocities,
        )


def read_frame(
    path: str | Path, index: int, topology: ChainTopology
) -> ChainConfiguration:
    """Read one frame as a :class:`ChainConfiguration` sharing ``topology``."""
    block = read_frames(path, index, index + 1)
    return ChainConfiguration(
        box=Box.from_bounds(block.box_lo[0], block.box_hi[0]),
        positions=block.positions[0],
        images=block.images[0],
        topology=topology,
        step=int(block.steps[0]),
        time=float(block.times[0]),
        velocities=None if block.velocities is None else block.velocities[0],
    )


def read_thermo(path: str | Path) -> dict[str, NDArray[Any]]:
    """Return every ``/thermo`` column (empty dict when the group is absent)."""
    with h5py.File(path, "r") as h5:
        if "thermo" not in h5:
            return {}
        return {name: dataset[()] for name, dataset in h5["thermo"].items()}


# ============================================================
# WRITER
# ============================================================


class TrajectoryWriter:
    """Stream frames into a new ``polyequil-1`` HDF5 file.

    Use as a context manager::

        with TrajectoryWriter(path, topology, attrs) as writer:
            for config in frames:
                writer.append(config)
            writer.write_thermo(columns)

    Parameters
    ----------
    path : str | Path
        Output file; must not exist unless ``overwrite`` is True.
    topology : ChainTopology
        Topology every appended frame must share.
    attrs : Mapping[str, Any] | None
        Extra root attributes (scalars, strings, numeric arrays).
    images_source : str
        Where image flags came from (``conventions.IMAGES_SOURCES``).
    overwrite : bool
        Replace an existing file.
    """

    def __init__(
        self,
        path: str | Path,
        topology: ChainTopology,
        attrs: Mapping[str, Any] | None = None,
        *,
        images_source: str = IMAGES_FROM_DUMP,
        overwrite: bool = False,
    ) -> None:
        self._path = Path(path)
        if self._path.exists() and not overwrite:
            raise ValidationError(f"{self._path} exists; pass overwrite=True")
        if images_source not in IMAGES_SOURCES:
            raise ValidationError(f"images_source must be one of {IMAGES_SOURCES}")
        self._topology = topology
        self._count = 0
        self._last_step: int | None = None
        self._with_velocities: bool | None = None
        self._h5 = h5py.File(self._path, "w")
        self._write_root(dict(attrs or {}), images_source)

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        """Return the number of frames written so far."""
        return self._count

    def append(self, config: ChainConfiguration) -> None:
        """Append one frame; steps must increase and the topology must match."""
        if config.topology is not self._topology and config.topology != self._topology:
            raise ValidationError("frame topology differs from the writer's topology")
        if self._last_step is not None and config.step <= self._last_step:
            raise ValidationError(
                f"step {config.step} does not follow step {self._last_step}"
            )
        if self._with_velocities is None:
            self._create_frame_datasets(config.velocities is not None)
        if (config.velocities is not None) != self._with_velocities:
            raise ValidationError("either every frame or no frame carries velocities")
        frames = self._h5["frames"]
        values: dict[str, Any] = {
            "step": config.step,
            "time": config.time,
            "box_lo": config.box.lo,
            "box_hi": config.box.hi,
            "positions": config.positions,
            "images": config.images,
        }
        if config.velocities is not None:
            values["velocities"] = config.velocities
        for name, value in values.items():
            dataset = frames[name]
            dataset.resize(self._count + 1, axis=0)
            dataset[self._count] = value
        self._count += 1
        self._last_step = config.step

    def write_thermo(self, columns: Mapping[str, NDArray[Any]]) -> None:
        """Write thermodynamic columns of equal length into ``/thermo``."""
        lengths = {np.asarray(column).shape for column in columns.values()}
        if len(lengths) > 1 or any(len(shape) != 1 for shape in lengths):
            raise ValidationError("thermo columns must be 1-D and of equal length")
        group = self._h5.require_group("thermo")
        for name, column in columns.items():
            group.create_dataset(name, data=np.asarray(column))

    def close(self) -> None:
        """Close the file; a file with no frames is still valid but empty."""
        if self._h5.id.valid:
            if self._with_velocities is None:
                self._create_frame_datasets(False)
            self._h5.close()

    def _write_root(self, attrs: dict[str, Any], images_source: str) -> None:
        """Write root attributes and the topology group."""
        attrs.update(
            {
                "schema": SCHEMA,
                "units": UNITS,
                "positions_convention": POSITIONS_CONVENTION,
                "images_source": images_source,
                "n_beads_total": self._topology.n_beads,
                "n_chains": self._topology.n_chains,
            }
        )
        frozen = freeze_mapping(attrs, role="trajectory attributes")
        for key, value in frozen.items():
            self._h5.attrs[key] = value
        group = self._h5.create_group("topology")
        group["bonds"] = self._topology.bonds
        group["angles"] = self._topology.angles
        group["mol"] = self._topology.chain_id
        group["type"] = self._topology.bead_type

    def _create_frame_datasets(self, with_velocities: bool) -> None:
        """Create the resizable frame datasets on the first append."""
        n = self._topology.n_beads
        frames = self._h5.create_group("frames")
        frames.create_dataset("step", (0,), maxshape=(None,), dtype=np.int64)
        frames.create_dataset("time", (0,), maxshape=(None,), dtype=np.float64)
        for name in ("box_lo", "box_hi"):
            frames.create_dataset(name, (0, 3), maxshape=(None, 3), dtype=np.float64)
        per_bead = [("positions", np.float64), ("images", np.int32)]
        if with_velocities:
            per_bead.append(("velocities", np.float64))
        for name, dtype in per_bead:
            frames.create_dataset(
                name,
                (0, n, 3),
                maxshape=(None, n, 3),
                dtype=dtype,
                chunks=(1, n, 3),
                compression=FRAME_CHUNK_COMPRESSION,
                compression_opts=FRAME_CHUNK_COMPRESSION_LEVEL,
                shuffle=True,
            )
        self._with_velocities = with_velocities


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _attrs(h5: h5py.File) -> Mapping[str, Any]:
    """Return root attributes as a frozen tree of plain Python values."""
    plain = {key: _plain_attr(value) for key, value in h5.attrs.items()}
    return freeze_mapping(plain, role="trajectory attributes", allow_opaque=True)


def _plain_attr(value: Any) -> Any:
    """Decode bytes and unwrap NumPy scalars from an HDF5 attribute."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return freeze_metadata(value, role="attribute", allow_opaque=True)


def _require_layout(h5: h5py.File, path: Path, attrs: Mapping[str, Any]) -> None:
    """Raise FormatError unless the file carries a readable schema and datasets."""
    schema = attrs.get("schema")
    if schema not in READABLE_SCHEMAS:
        raise FormatError(f"{path}: schema {schema!r} not in {READABLE_SCHEMAS}")
    for group, names in (
        ("topology", REQUIRED_TOPOLOGY_DATASETS),
        ("frames", REQUIRED_FRAME_DATASETS),
    ):
        missing = [name for name in names if f"{group}/{name}" not in h5]
        if missing:
            raise FormatError(f"{path}: missing datasets {group}/{missing}")
