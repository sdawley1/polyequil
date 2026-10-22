"""
Readers for LAMMPS data files, text dumps, and log files, and HDF5 conversion.

Supported: orthogonal boxes; atom styles ``bond``, ``angle``, ``molecular``,
and ``full``; ``dump custom`` text files with an ``id`` column and one of the
coordinate sets ``x y z`` (as stored), ``xu yu zu`` (unwrapped), ``xs ys zs``
(scaled), or ``xsu ysu zsu`` (scaled unwrapped), optionally ``ix iy iz`` and
``vx vy vz``. Atom ids must be ``1..n``. When a dump has stored coordinates but
no image flags, flags are rebuilt per frame from bonds (see
:meth:`~polyequil.topology.ChainTopology.make_whole`).

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil.box import Box
from polyequil.configuration import ChainConfiguration
from polyequil.conventions import IMAGES_FROM_BONDS, IMAGES_FROM_DUMP
from polyequil.errors import FormatError
from polyequil.io.hdf5 import TrajectoryWriter
from polyequil.topology import ChainTopology

# ============================================================
# CONSTANTS
# ============================================================

POSITION_COLUMNS: dict[str, tuple[tuple[str, ...], bool, bool]] = {
    # name set -> (columns, scaled, unwrapped)
    "stored": (("x", "y", "z"), False, False),
    "unwrapped": (("xu", "yu", "zu"), False, True),
    "scaled": (("xs", "ys", "zs"), True, False),
    "scaled_unwrapped": (("xsu", "ysu", "zsu"), True, True),
}
IMAGE_COLUMNS: tuple[str, ...] = ("ix", "iy", "iz")
VELOCITY_COLUMNS: tuple[str, ...] = ("vx", "vy", "vz")
ATOM_STYLE_COLUMNS: dict[str, int] = {"bond": 3, "angle": 3, "molecular": 3, "full": 4}

# ============================================================
# TYPES
# ============================================================


@dataclass(frozen=True, slots=True, eq=False)
class LammpsData:
    """Topology, box, and coordinates read from a LAMMPS data file."""

    topology: ChainTopology
    box: Box
    positions: NDArray[np.float64]
    images: NDArray[np.int32]


# ============================================================
# DATA FILES
# ============================================================


def read_data(path: str | Path) -> LammpsData:
    """Read a LAMMPS data file with linear-chain topology.

    Raises
    ------
    FormatError
        On a triclinic box, an unsupported atom style, or ids not ``1..n``.
    """
    header, sections = _data_sections(Path(path).read_text().splitlines())
    if "xy xz yz" in header:
        raise FormatError(f"{path}: triclinic boxes are not supported")
    box = Box.from_bounds(
        [header[f"{a}lo {a}hi"][0] for a in "xyz"],
        [header[f"{a}lo {a}hi"][1] for a in "xyz"],
    )
    style, rows = sections.get("Atoms", ("", []))
    if style not in ATOM_STYLE_COLUMNS:
        raise FormatError(
            f"{path}: atom style {style!r} not in {list(ATOM_STYLE_COLUMNS)}"
        )
    table = _numeric_table(rows, f"{path} Atoms")
    order = _id_order(table[:, 0], f"{path} Atoms")
    table = table[order]
    first = ATOM_STYLE_COLUMNS[style]
    positions = table[:, first : first + 3]
    stored = table.shape[1] >= first + 6
    images: NDArray[np.float64] | NDArray[np.int32] | None = (
        table[:, first + 3 : first + 6] if stored else None
    )
    _, mol_index = np.unique(table[:, 1], return_inverse=True)
    topology = ChainTopology(
        bonds=_index_rows(sections.get("Bonds", ("", []))[1], 2, f"{path} Bonds"),
        angles=_index_rows(sections.get("Angles", ("", []))[1], 3, f"{path} Angles"),
        chain_id=_first_appearance_labels(mol_index),
        bead_type=table[:, 2].astype(np.int64),
    )
    if images is None:
        images = topology.make_whole(positions, box)
    return LammpsData(topology, box, positions, np.asarray(images, dtype=np.int32))


# ============================================================
# DUMP FILES
# ============================================================


def dump_images_source(path: str | Path) -> str:
    """Return where image flags of the dump's frames will come from."""
    with open(path) as handle:
        for line in handle:
            if line.startswith("ITEM: ATOMS"):
                columns = line.split()[2:]
                kind = _position_kind(columns, str(path))
                unwrapped = POSITION_COLUMNS[kind][2]
                has_images = all(c in columns for c in IMAGE_COLUMNS)
                return (
                    IMAGES_FROM_DUMP if unwrapped or has_images else IMAGES_FROM_BONDS
                )
    raise FormatError(f"{path}: no ITEM: ATOMS line")


def iter_dump(
    path: str | Path, topology: ChainTopology, *, dt: float
) -> Iterator[ChainConfiguration]:
    """Stream the frames of a ``dump custom`` text file, sorted by atom id.

    Parameters
    ----------
    path : str | Path
        The dump file.
    topology : ChainTopology
        Topology from the matching data file.
    dt : float
        Timestep, so that ``time = step * dt``.

    Yields
    ------
    ChainConfiguration
        One frame at a time; memory use is one frame.
    """
    if not dt > 0.0:
        raise FormatError(f"dt must be positive, got {dt}")
    with open(path) as handle:
        for index in range(np.iinfo(np.int64).max):
            frame = _read_dump_frame(handle, topology, dt, f"{path} frame {index}")
            if frame is None:
                return
            yield frame


# ============================================================
# LOG FILES
# ============================================================


def read_log(path: str | Path, block: int = -1) -> dict[str, NDArray[Any]]:
    """Return one thermo block of a LAMMPS log as lower-case named columns.

    Blocks start at a header line beginning with ``Step`` and end at
    ``Loop time``; non-numeric lines inside a block (warnings) are skipped.
    ``step`` is int64; every other column is float64.
    """
    blocks = _thermo_blocks(Path(path).read_text().splitlines())
    if not blocks:
        raise FormatError(f"{path}: no thermo block found")
    try:
        header, rows = blocks[block]
    except IndexError as exc:
        raise FormatError(f"{path}: has {len(blocks)} thermo blocks") from exc
    data = np.asarray(rows, dtype=np.float64).reshape(-1, len(header))
    columns: dict[str, NDArray[Any]] = {
        name.lower(): data[:, k] for k, name in enumerate(header)
    }
    columns["step"] = columns["step"].astype(np.int64)
    return columns


# ============================================================
# CONVERSION
# ============================================================


def convert_lammps(
    data: str | Path,
    dump: str | Path,
    out: str | Path,
    *,
    log: str | Path | None = None,
    dt: float | None = None,
    attrs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert a LAMMPS data file + dump (+ log) into one ``polyequil-1`` HDF5 file.

    ``dt`` defaults to ``time / step`` from the log's thermo block and must be
    given when there is no log or the log has no ``Time`` column.
    """
    topology = read_data(data).topology
    thermo = read_log(log) if log is not None else None
    if dt is None:
        dt = _dt_from_thermo(thermo)
    record = {"dt": dt, "source_data": str(data), "source_dump": str(dump)}
    if log is not None:
        record["source_log"] = str(log)
    source = dump_images_source(dump)
    with TrajectoryWriter(
        out,
        topology,
        {**dict(attrs or {}), **record},
        images_source=source,
        overwrite=overwrite,
    ) as writer:
        for frame in iter_dump(dump, topology, dt=dt):
            writer.append(frame)
        if thermo is not None:
            writer.write_thermo(thermo)
    return Path(out)


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _is_section_header(text: str) -> bool:
    """Return whether a comment-stripped line is a data-file section keyword."""
    return bool(text) and all(word.isalpha() for word in text.split())


def _data_sections(
    lines: list[str],
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[str, list[str]]]]:
    """Split a data file into header keywords and (style comment, rows) sections."""
    header: dict[str, tuple[float, ...]] = {}
    sections: dict[str, tuple[str, list[str]]] = {}
    current: str | None = None
    for line in lines[1:]:
        text, _, comment = line.partition("#")
        text = text.strip()
        if not text:
            continue
        if _is_section_header(text):
            current = text
            sections[current] = (comment.strip(), [])
        elif current is None:
            values, keyword = _header_entry(text)
            header[keyword] = values
        else:
            sections[current][1].append(text)
    return header, sections


def _header_entry(text: str) -> tuple[tuple[float, ...], str]:
    """Split a header line into its leading numbers and trailing keyword."""
    words = text.split()
    count = 0
    for word in words:
        try:
            float(word)
        except ValueError:
            break
        count += 1
    return tuple(float(w) for w in words[:count]), " ".join(words[count:])


def _numeric_table(rows: list[str], where: str) -> NDArray[np.float64]:
    """Parse whitespace-separated rows of equal width into a float array."""
    if not rows:
        raise FormatError(f"{where}: section is empty")
    widths = {len(row.split()) for row in rows}
    if len(widths) != 1:
        raise FormatError(
            f"{where}: rows have differing column counts {sorted(widths)}"
        )
    flat = np.array(" ".join(rows).split(), dtype=np.float64)
    return flat.reshape(len(rows), widths.pop())


def _id_order(ids: NDArray[np.float64], where: str) -> NDArray[np.int64]:
    """Return the permutation sorting atoms by id; ids must be exactly 1..n."""
    order = np.argsort(ids, kind="stable")
    if not np.array_equal(ids[order], np.arange(1, ids.size + 1)):
        raise FormatError(f"{where}: atom ids must be exactly 1..n")
    return order.astype(np.int64)


def _index_rows(rows: list[str], width: int, where: str) -> NDArray[np.int64]:
    """Return zero-based atom-index rows (dropping the id and type columns)."""
    if not rows:
        return np.empty((0, width), dtype=np.int64)
    table = _numeric_table(rows, where)
    if table.shape[1] != width + 2:
        raise FormatError(f"{where}: expected {width + 2} columns")
    return table[:, 2:].astype(np.int64) - 1


def _first_appearance_labels(labels: NDArray[np.int64]) -> NDArray[np.int64]:
    """Relabel values 0, 1, ... in order of first appearance."""
    _, first = np.unique(labels, return_index=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(first.size)
    relabelled: NDArray[np.int64] = rank[labels]
    return relabelled


def _position_kind(columns: list[str], where: str) -> str:
    """Return which coordinate set a dump carries (exactly one is required)."""
    kinds = [
        k for k, (cols, _, _) in POSITION_COLUMNS.items() if set(cols) <= set(columns)
    ]
    if len(kinds) != 1 or "id" not in columns:
        raise FormatError(f"{where}: need an id column and exactly one coordinate set")
    return kinds[0]


def _read_dump_frame(
    handle: IO[str], topology: ChainTopology, dt: float, where: str
) -> ChainConfiguration | None:
    """Read one dump frame; return None at a clean end of file."""
    first = handle.readline()
    if not first:
        return None
    lines = [first] + [handle.readline() for _ in range(8)]
    if lines[0].strip() != "ITEM: TIMESTEP" or not lines[8].startswith("ITEM: ATOMS"):
        raise FormatError(f"{where}: malformed frame header")
    if len(lines[4].split()) > 6 or len(lines[5].split()) != 2:
        raise FormatError(f"{where}: triclinic boxes are not supported")
    n = int(lines[3])
    if n != topology.n_beads:
        raise FormatError(f"{where}: {n} atoms but the topology has {topology.n_beads}")
    rows = [handle.readline() for _ in range(n)]
    if not rows[-1].strip():
        raise FormatError(f"{where}: truncated frame")
    bounds = np.array([line.split() for line in lines[5:8]], dtype=np.float64)
    columns = lines[8].split()[2:]
    table = _numeric_table([row.strip() for row in rows], where)
    table = table[_id_order(table[:, columns.index("id")], where)]
    box = Box.from_bounds(bounds[:, 0], bounds[:, 1])
    positions, images = _frame_coordinates(table, columns, box, topology, where)
    velocities = (
        table[:, [columns.index(c) for c in VELOCITY_COLUMNS]]
        if all(c in columns for c in VELOCITY_COLUMNS)
        else None
    )
    step = int(lines[1])
    return ChainConfiguration(
        box, positions, images, topology, step, step * dt, velocities
    )


def _frame_coordinates(
    table: NDArray[np.float64],
    columns: list[str],
    box: Box,
    topology: ChainTopology,
    where: str,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Return stored positions and image flags for one sorted dump frame."""
    kind = _position_kind(columns, where)
    names, scaled, unwrapped = POSITION_COLUMNS[kind]
    raw = table[:, [columns.index(c) for c in names]]
    positions = box.to_cartesian(raw) if scaled else raw
    if unwrapped:
        wrapped = box.wrap(positions)
        images = np.rint((positions - wrapped) / box.lengths).astype(np.int32)
        return wrapped, images
    if all(c in columns for c in IMAGE_COLUMNS):
        flags = table[:, [columns.index(c) for c in IMAGE_COLUMNS]]
        return positions, flags.astype(np.int32)
    return positions, topology.make_whole(positions, box)


def _thermo_blocks(lines: list[str]) -> list[tuple[list[str], list[list[float]]]]:
    """Return every (header, rows) thermo block of a log file."""
    blocks: list[tuple[list[str], list[list[float]]]] = []
    current: tuple[list[str], list[list[float]]] | None = None
    for line in lines:
        words = line.split()
        if words[:1] == ["Step"]:
            current = (words, [])
            blocks.append(current)
        elif line.startswith("Loop time"):
            current = None
        elif current is not None and len(words) == len(current[0]):
            try:
                current[1].append([float(w) for w in words])
            except ValueError:
                continue
    return blocks


def _dt_from_thermo(thermo: dict[str, NDArray[Any]] | None) -> float:
    """Return ``time / step`` from a thermo block with a Time column."""
    if thermo is None or "time" not in thermo:
        raise FormatError("dt is required when no log with a Time column is given")
    steps = thermo["step"]
    nonzero = np.flatnonzero(steps != 0)
    if nonzero.size == 0:
        raise FormatError("cannot infer dt: every thermo row is at step 0")
    k = int(nonzero[0])
    return float(thermo["time"][k] / float(steps[k]))
