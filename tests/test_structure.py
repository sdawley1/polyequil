import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polyequil.box import Box
from polyequil.configuration import ChainConfiguration
from polyequil.ensemble import TrajectoryEnsemble
from polyequil.errors import ValidationError
from polyequil.execution import ExecutionPolicy
from polyequil.observables.structure import (
    order_tensor,
    orientational_structure_factor,
    structure_factor,
)
from polyequil.topology import linear_chains
from polyequil.trajectory import Trajectory
from tests.conftest import chain_frame, point_configuration

GOLDEN = (1 + np.sqrt(5)) / 2
ICOSAHEDRON = np.array(
    [(0, s1, s2 * GOLDEN) for s1 in (1, -1) for s2 in (1, -1)]
    + [(s1, s2 * GOLDEN, 0) for s1 in (1, -1) for s2 in (1, -1)]
    + [(s2 * GOLDEN, 0, s1) for s1 in (1, -1) for s2 in (1, -1)],
    dtype=float,
)
ICOSAHEDRON /= np.linalg.norm(ICOSAHEDRON, axis=1, keepdims=True)  # spherical 5-design


def _rebuilt(config, positions):
    """Same topology and box, new unwrapped positions (images zero)."""
    return ChainConfiguration(
        config.box,
        positions,
        np.zeros_like(config.images),
        config.topology,
        config.step,
        config.time,
    )


# ----- density structure factor -----


def test_ideal_gas_has_unit_structure_factor():
    """T1: uniform points at fixed N give <S(k)> = 1 at every k != 0."""
    box = Box(lo=np.zeros(3), lengths=np.full(3, 8.0))
    rng = np.random.default_rng(11)
    frames = [
        structure_factor(
            point_configuration(rng.uniform(0, 8, (200, 3)), box), k_max=3.0
        )
        for _ in range(200)
    ]
    mean = np.mean([f.values for f in frames], axis=0)
    counts = np.asarray(frames[0].auxiliary["shell_counts"])
    sigma = 1.0 / np.sqrt(200 * counts)
    assert np.all(np.abs(mean - 1.0) < 5 * sigma)
    assert frames[0].coordinates.names() == ("k",)


def test_perfect_lattice_has_bragg_peaks_only():
    """T2: a simple-cubic crystal scatters only at its reciprocal lattice."""
    side = np.arange(4.0)
    sites = np.stack(np.meshgrid(side, side, side, indexing="ij"), -1).reshape(-1, 3)
    box = Box(lo=np.zeros(3), lengths=np.full(3, 4.0))
    result = structure_factor(
        point_configuration(sites + 0.25, box), k_max=2 * np.pi * 1.01
    )
    k = result.coordinates.axes[0][1]
    bragg = np.isclose(k, 2 * np.pi)
    assert bragg.sum() == 1
    np.testing.assert_allclose(result.values[bragg], 64.0, rtol=1e-12)
    assert np.all(np.abs(result.values[~bragg]) < 1e-10)


def test_structure_factor_matches_a_naive_sum_on_reference_data(ex_files):
    config = Trajectory(ex_files[0]).frame(-1)
    result = structure_factor(config, k_max=1.0)
    k_shells = result.coordinates.axes[0][1]
    x = config.unwrapped()
    length = config.box.lengths[0]
    for shell, k in enumerate(k_shells[:3]):
        modes = [
            n
            for n in np.ndindex(5, 5, 5)
            if np.isclose(2 * np.pi * np.linalg.norm(np.array(n) - 2) / length, k)
            and next((c > 0 for c in np.array(n) - 2 if c != 0), False)
        ]
        per_mode = [
            np.abs(np.exp(-1j * x @ (2 * np.pi * (np.array(n) - 2) / length)).sum())
            ** 2
            / len(x)
            for n in modes
        ]
        np.testing.assert_allclose(result.values[shell], np.mean(per_mode), rtol=1e-10)


@given(st.integers(0, 2**32 - 1), st.integers(-2, 2))
def test_structure_factor_is_invariant_under_image_shifts(seed, shift):
    config = chain_frame(seed % 1000, chain_lengths=(8,) * 6, box_length=6.0)
    moved = ChainConfiguration(
        config.box, config.positions, config.images + shift, config.topology, 0, 0.0
    )
    np.testing.assert_allclose(
        structure_factor(moved, k_max=3.0).values,
        structure_factor(config, k_max=3.0).values,
        rtol=1e-9,
        atol=1e-12,
    )


# ----- orientation -----


def test_independent_isotropic_bonds_give_one_fifth_at_every_k():
    """T5: <S_Q(k)> = <P2^2> = 1/5 for independent isotropic bonds, k = 0 included."""
    results = [
        orientational_structure_factor(chain_frame(s), k_max=1.5) for s in range(200)
    ]
    k = results[0].coordinates.axes[0][1]
    assert k[0] == 0.0
    mean = np.mean([r.values for r in results], axis=0)
    # per-frame S_Q(0) = (2/15)(3/10) chi^2_5: sd 0.126; k > 0 shells are tighter
    assert np.all(np.abs(mean - 0.2) < 5 * 0.126 / np.sqrt(200))


def test_order_tensor_vanishes_on_average_and_is_exact_for_aligned_bonds():
    tensors = np.array([order_tensor(chain_frame(s)).values for s in range(200)])
    assert np.all(np.abs(tensors.mean(0)) < 0.005)
    np.testing.assert_allclose(np.trace(tensors, axis1=1, axis2=2), 0.0, atol=1e-12)
    np.testing.assert_allclose(tensors, np.swapaxes(tensors, 1, 2))
    top = linear_chains([5, 5])
    rod = np.zeros((10, 3))
    rod[:, 2] = np.tile(np.arange(5.0), 2)
    rod[5:, 0] = 3.0
    box = Box(lo=np.full(3, -1.0), lengths=np.full(3, 10.0))
    config = ChainConfiguration(box, rod, np.zeros((10, 3)), top, 0, 0.0)
    np.testing.assert_allclose(
        order_tensor(config).values, np.diag([-1 / 3, -1 / 3, 2 / 3]), atol=1e-15
    )
    assert orientational_structure_factor(config, k_max=0.0).values[0] == pytest.approx(
        0.3 * (2 / 3) * 8
    )


@given(st.integers(0, 10_000), st.sampled_from([1, 2, 5]))
def test_k0_value_equals_the_direction_averaged_p2_fluctuation(seed, n_s):
    """(3/10) Q:Q / N equals the isotropic average of (sum P2(u.n))^2 / N."""
    config = chain_frame(seed, chain_lengths=(12,) * 5)
    x = config.unwrapped()
    i, j = config.topology.pairs_at_separation(n_s)
    u = (x[j] - x[i]) / np.linalg.norm(x[j] - x[i], axis=1, keepdims=True)
    p2 = 1.5 * (u @ ICOSAHEDRON.T) ** 2 - 0.5
    expected = np.mean(p2.sum(axis=0) ** 2) / len(i)
    got = orientational_structure_factor(config, k_max=0.0, segment_bonds=n_s).values[0]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@given(st.integers(0, 10_000))
def test_k0_value_is_invariant_under_rotation_reversal_and_relabelling(seed):
    config = chain_frame(seed, chain_lengths=(10,) * 6)
    base = orientational_structure_factor(config, k_max=0.0, segment_bonds=2).values
    x = config.unwrapped()
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    reversed_beads = x.reshape(6, 10, 3)[:, ::-1].reshape(-1, 3)
    relabelled = x.reshape(6, 10, 3)[np.random.default_rng(seed).permutation(6)]
    for positions in (x @ rotation.T, reversed_beads, relabelled.reshape(-1, 3)):
        got = orientational_structure_factor(
            _rebuilt(config, positions), k_max=0.0, segment_bonds=2
        ).values
        np.testing.assert_allclose(got, base, rtol=1e-10)


def test_invalid_segment_lengths_raise():
    config = chain_frame(0, chain_lengths=(4, 4))
    with pytest.raises(ValidationError, match="segment"):
        order_tensor(config, segment_bonds=4)
    with pytest.raises(ValidationError, match="segment"):
        order_tensor(config, segment_bonds=0)


# ----- ensemble -----


def test_ensemble_runs_are_mode_independent(ex_files):
    ensemble = TrajectoryEnsemble(ex_files)
    outer = ExecutionPolicy("outer", total_cpus=2)
    for observable in (structure_factor, orientational_structure_factor):
        a = ensemble.evaluate(observable, times=[0.0, 5000.0], k_max=1.5)
        b = ensemble.evaluate(
            observable, times=[0.0, 5000.0], k_max=1.5, execution=outer
        )
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.standard_error, b.standard_error)
