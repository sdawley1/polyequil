import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from polyequil.accumulators import SampleStore, WelfordAccumulator
from polyequil.errors import ValidationError
from polyequil.results import ObservableResult, SeparableAxes

samples = hnp.arrays(
    np.float64,
    st.tuples(st.integers(1, 30), st.integers(1, 4)),
    elements=st.floats(-1e6, 1e6, allow_subnormal=False),
)


def _results(values):
    axes = SeparableAxes((("k", np.arange(values.shape[1], dtype=float)),))
    return [ObservableResult(row, axes) for row in values]


def _fold(accumulator, values):
    for result in _results(values):
        accumulator.update(result)
    return accumulator


@given(samples)
def test_matches_numpy_on_real_data(values):
    mean, variance, count = _fold(WelfordAccumulator(), values).result()
    scale = max(1.0, np.abs(values).max())
    np.testing.assert_allclose(mean, values.mean(0), rtol=1e-9, atol=1e-9 * scale)
    assert count == values.shape[0]
    if count == 1:
        assert variance is None
    else:
        np.testing.assert_allclose(
            variance, values.var(0, ddof=1), rtol=1e-7, atol=1e-7 * scale**2
        )


@given(samples, samples)
def test_complex_data_use_the_modulus_variance(re, im):
    n = min(re.shape[0], im.shape[0])
    k = min(re.shape[1], im.shape[1])
    z = re[:n, :k] + 1j * im[:n, :k]
    mean, variance, _ = _fold(WelfordAccumulator(), z).result()
    scale = max(1.0, np.abs(z).max())
    np.testing.assert_allclose(mean, z.mean(0), rtol=1e-9, atol=1e-9 * scale)
    if n > 1:
        np.testing.assert_allclose(
            variance, z.var(0, ddof=1), rtol=1e-7, atol=1e-7 * scale**2
        )


@given(samples, st.data())
def test_merge_agrees_with_the_sequential_fold(values, data):
    cut = data.draw(st.integers(0, values.shape[0]))
    left = _fold(WelfordAccumulator(), values[:cut])
    left.merge(_fold(WelfordAccumulator(), values[cut:]))
    whole = _fold(WelfordAccumulator(), values)
    scale = max(1.0, np.abs(values).max())
    np.testing.assert_allclose(left.result()[0], whole.result()[0], atol=1e-9 * scale)
    assert left.count == whole.count


def test_empty_accumulator_has_no_result():
    with pytest.raises(ValidationError):
        WelfordAccumulator().result()


def test_incompatible_update_leaves_the_state_unchanged():
    accumulator = _fold(WelfordAccumulator(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    before = accumulator.result()
    with pytest.raises(ValidationError, match="incompatible"):
        accumulator.update(
            ObservableResult(np.ones(3), SeparableAxes((("k", np.arange(3.0)),)))
        )
    after = accumulator.result()
    assert after[2] == before[2]
    np.testing.assert_array_equal(after[0], before[0])


@given(samples)
def test_sample_store_does_not_change_the_statistics(values):
    plain = _fold(WelfordAccumulator(), values).result()
    store = _fold(SampleStore(WelfordAccumulator()), values)
    stored = store.result()
    np.testing.assert_array_equal(plain[0], stored[0])
    if plain[1] is not None:
        np.testing.assert_array_equal(plain[1], stored[1])
    np.testing.assert_array_equal(store.samples(), values)


def test_store_rejects_a_prefilled_accumulator():
    filled = _fold(WelfordAccumulator(), np.ones((1, 1)))
    with pytest.raises(ValidationError, match="empty"):
        SampleStore(filled)
