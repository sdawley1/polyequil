import pickle

import numpy as np
import pytest
from hypothesis import given

from polyequil.box import Box
from polyequil.errors import ValidationError
from tests.strategies import boxes, points


@given(boxes(), points(8))
def test_wrap_lands_in_the_box_and_is_idempotent(box, x):
    wrapped = box.wrap(x)
    assert np.all(wrapped >= box.lo) and np.all(wrapped < box.hi + 1e-9 * box.lengths)
    np.testing.assert_allclose(
        box.wrap(wrapped), wrapped, atol=1e-9 * np.abs(x).max() + 1e-12
    )
    shift = np.rint((x - wrapped) / box.lengths)
    np.testing.assert_allclose(
        wrapped + shift * box.lengths, x, atol=1e-9 * max(1.0, np.abs(x).max())
    )


@given(boxes(), points(8))
def test_minimum_image_is_the_shortest_equivalent_separation(box, d):
    m = box.minimum_image(d)
    assert np.all(m >= -box.lengths / 2 - 1e-9) and np.all(m <= box.lengths / 2 + 1e-9)
    shift = (d - m) / box.lengths
    np.testing.assert_allclose(shift, np.rint(shift), atol=1e-6)


def test_half_box_ties_map_to_minus_half():
    box = Box(lo=np.zeros(3), lengths=np.array([2.0, 4.0, 6.0]))
    np.testing.assert_array_equal(
        box.minimum_image([1.0, 2.0, 3.0]), [-1.0, -2.0, -3.0]
    )
    np.testing.assert_array_equal(
        box.minimum_image([-1.0, -2.0, -3.0]), [-1.0, -2.0, -3.0]
    )


@given(boxes(), points(5))
def test_fractional_round_trip(box, x):
    np.testing.assert_allclose(
        box.to_cartesian(box.to_fractional(x)), x, atol=1e-9 * max(1.0, np.abs(x).max())
    )


def test_validation_equality_and_pickling():
    with pytest.raises(ValidationError, match="positive"):
        Box(lo=np.zeros(3), lengths=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValidationError):
        Box(lo=np.zeros(2), lengths=np.ones(2))
    box = Box.from_bounds([0, 0, 0], [1, 2, 3])
    copy = pickle.loads(pickle.dumps(box))
    assert copy == box and not copy.lengths.flags.writeable
