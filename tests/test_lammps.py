import numpy as np
import pytest

from polyequil.box import Box
from polyequil.conventions import IMAGES_FROM_BONDS, IMAGES_FROM_DUMP
from polyequil.errors import FormatError
from polyequil.io.hdf5 import read_frame, read_header, read_thermo
from polyequil.io.lammps import (
    convert_lammps,
    dump_images_source,
    iter_dump,
    read_data,
    read_log,
)
from polyequil.topology import linear_chains
from tests.conftest import random_walk_frame

TOPOLOGY = linear_chains([4, 3])
BOX = Box(lo=np.array([0.0, -1.0, 2.0]), lengths=np.array([5.0, 6.0, 7.0]))


def _frames(count=3, seed=0):
    rng = np.random.default_rng(seed)
    return [random_walk_frame(TOPOLOGY, BOX, rng, step=100 * k) for k in range(count)]


def _write_data(path, frame, style="angle", images=True):
    lines = [
        "test",
        "",
        f"{frame.n_beads} atoms",
        f"{len(TOPOLOGY.bonds)} bonds",
        f"{len(TOPOLOGY.angles)} angles",
        "1 atom types",
        "",
    ]
    for axis, name in enumerate("xyz"):
        lines.append(
            f"{float(BOX.lo[axis])!r} {float(BOX.hi[axis])!r} {name}lo {name}hi"
        )
    lines += ["", "Masses", "", "1 1.0", "", f"Atoms # {style}", ""]
    # write atoms in reverse id order, with mol ids starting at 7
    for i in reversed(range(frame.n_beads)):
        charge = " 0.0" if style == "full" else ""
        x = " ".join(repr(float(v)) for v in frame.positions[i])
        flags = " " + " ".join(str(v) for v in frame.images[i]) if images else ""
        lines.append(f"{i + 1} {TOPOLOGY.chain_id[i] + 7} 1{charge} {x}{flags}")
    lines += ["", "Bonds", ""]
    lines += [
        f"{k + 1} 1 {a + 1} {b + 1}" for k, (a, b) in enumerate(TOPOLOGY.bonds[::-1])
    ]
    lines += ["", "Angles", ""]
    lines += [
        f"{k + 1} 1 {a + 1} {b + 1} {c + 1}"
        for k, (a, b, c) in enumerate(TOPOLOGY.angles)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_dump(path, frames, kind="x"):
    out = []
    for frame in frames:
        out += [
            "ITEM: TIMESTEP",
            str(frame.step),
            "ITEM: NUMBER OF ATOMS",
            str(frame.n_beads),
            "ITEM: BOX BOUNDS pp pp pp",
        ]
        out += [
            f"{float(lo)!r} {float(hi)!r}"
            for lo, hi in zip(BOX.lo, BOX.hi, strict=True)
        ]
        if kind == "x":
            cols, data = (
                "x y z ix iy iz vx vy vz",
                np.hstack([frame.positions, frame.images, frame.velocities]),
            )
        elif kind == "xu":
            cols, data = "xu yu zu", frame.unwrapped()
        else:
            cols, data = "xs ys zs", BOX.to_fractional(frame.positions)
        out.append(f"ITEM: ATOMS {cols} id type")
        order = np.random.default_rng(frame.step).permutation(frame.n_beads)
        for i in order:
            values = " ".join(
                repr(float(v))
                if abs(v - round(v)) > 0 or "i" not in cols
                else str(int(v))
                for v in data[i]
            )
            out.append(f"{values} {i + 1} 1")
    path.write_text("\n".join(out) + "\n")
    return path


@pytest.mark.parametrize("style", ["angle", "full", "bond", "molecular"])
def test_read_data_styles_sort_ids_and_relabel_chains(tmp_path, style):
    frame = _frames(1)[0]
    data = read_data(_write_data(tmp_path / "d.data", frame, style=style))
    assert data.topology == TOPOLOGY and data.box == BOX
    np.testing.assert_array_equal(data.positions, frame.positions)
    np.testing.assert_array_equal(data.images, frame.images)


def test_read_data_without_images_rebuilds_whole_chains(tmp_path):
    frame = _frames(1)[0]
    data = read_data(_write_data(tmp_path / "d.data", frame, images=False))
    x = data.positions + data.images * BOX.lengths
    b = TOPOLOGY.bonds
    truth = frame.unwrapped()
    np.testing.assert_allclose(
        x[b[:, 1]] - x[b[:, 0]], truth[b[:, 1]] - truth[b[:, 0]], atol=1e-12
    )


@pytest.mark.parametrize("kind", ["x", "xu", "xs"])
def test_dump_coordinate_kinds(tmp_path, kind):
    frames = _frames()
    path = _write_dump(tmp_path / f"{kind}.dump", frames, kind=kind)
    read = list(iter_dump(path, TOPOLOGY, dt=0.005))
    assert [f.step for f in read] == [0, 100, 200]
    assert read[1].time == 0.5
    expected = IMAGES_FROM_BONDS if kind == "xs" else IMAGES_FROM_DUMP
    assert dump_images_source(path) == expected
    for got, want in zip(read, frames, strict=True):
        b = TOPOLOGY.bonds
        gu, wu = got.unwrapped(), want.unwrapped()
        np.testing.assert_allclose(
            gu[b[:, 1]] - gu[b[:, 0]], wu[b[:, 1]] - wu[b[:, 0]], atol=1e-9
        )
        if kind != "xs":
            np.testing.assert_allclose(gu, wu, atol=1e-9)
    if kind == "x":
        np.testing.assert_array_equal(read[2].positions, frames[2].positions)
        np.testing.assert_array_equal(read[2].velocities, frames[2].velocities)


def test_read_log_picks_blocks_and_skips_warnings(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text(
        "LAMMPS (22 Jul 2025)\n   Step   Temp\n  0  1.5\n  10  1.4\nLoop time of 1\n"
        "   Step          Time           Temp\n  0  0.0  1.0\n"
        "WARNING: something happened (src/x.cpp:1)\n  100  1.0  1.1\nLoop time of 2\n"
    )
    last = read_log(log)
    np.testing.assert_array_equal(last["step"], [0, 100])
    assert last["step"].dtype == np.int64
    np.testing.assert_array_equal(last["temp"], [1.0, 1.1])
    assert list(read_log(log, block=0)) == ["step", "temp"]


def test_convert_round_trip(tmp_path):
    frames = _frames()
    data = _write_data(tmp_path / "d.data", frames[0])
    dump = _write_dump(tmp_path / "t.dump", frames)
    log = tmp_path / "log"
    log.write_text("Step Time Temp\n0 0 1\n100 0.5 1\nLoop time\n")
    out = convert_lammps(data, dump, tmp_path / "t.h5", log=log, attrs={"tag": "x"})
    header = read_header(out)
    assert header.n_frames == 3 and header.attrs["dt"] == 0.005
    assert header.attrs["tag"] == "x" and header.images_source == IMAGES_FROM_DUMP
    np.testing.assert_array_equal(
        read_frame(out, 1, header.topology).positions, frames[1].positions
    )
    np.testing.assert_array_equal(read_thermo(out)["temp"], [1.0, 1.0])
    with pytest.raises(FormatError, match="dt is required"):
        convert_lammps(data, dump, tmp_path / "u.h5")


def test_unsupported_inputs_are_format_errors(tmp_path):
    frame = _frames(1)[0]
    data = _write_data(tmp_path / "d.data", frame)
    text = data.read_text().replace(
        "1 atom types", "1 atom types\n0.0 0.0 0.0 xy xz yz"
    )
    (tmp_path / "tri.data").write_text(text)
    with pytest.raises(FormatError, match="triclinic"):
        read_data(tmp_path / "tri.data")
    (tmp_path / "ids.data").write_text(data.read_text().replace("\n7 8 1 ", "\n9 8 1 "))
    with pytest.raises(FormatError, match=r"1\.\.n"):
        read_data(tmp_path / "ids.data")
    dump = _write_dump(tmp_path / "t.dump", [frame])
    (tmp_path / "cut.dump").write_text(
        "\n".join(dump.read_text().splitlines()[:-2]) + "\n"
    )
    with pytest.raises(FormatError, match="truncated"):
        list(iter_dump(tmp_path / "cut.dump", TOPOLOGY, dt=0.01))
