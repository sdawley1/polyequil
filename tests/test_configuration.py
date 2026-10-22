import numpy as np
import pytest

from polyequil.box import Box
from polyequil.configuration import ChainConfiguration
from polyequil.errors import ValidationError
from polyequil.topology import linear_chains


def test_unwrapped_adds_images_times_lengths():
    box = Box(lo=np.zeros(3), lengths=np.array([2.0, 3.0, 4.0]))
    config = ChainConfiguration(
        box,
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
        [[0, 0, 0], [1, -1, 2]],
        linear_chains([2]),
        5,
        0.05,
    )
    np.testing.assert_array_equal(config.unwrapped()[1], [3.5, -2.5, 8.5])
    assert not config.positions.flags.writeable


@pytest.mark.parametrize(
    "change",
    [
        {"positions": np.zeros((3, 3))},
        {"images": [[0.5, 0, 0], [0, 0, 0]]},
        {"step": 1.5},
        {"time": float("nan")},
        {"positions": [[np.inf, 0, 0], [0, 0, 0]]},
    ],
)
def test_invalid_frames_are_rejected(change):
    fields = {
        "box": Box(lo=np.zeros(3), lengths=np.ones(3)),
        "positions": np.zeros((2, 3)),
        "images": np.zeros((2, 3)),
        "topology": linear_chains([2]),
        "step": 0,
        "time": 0.0,
    }
    fields.update(change)
    with pytest.raises(ValidationError):
        ChainConfiguration(**fields)
