import pickle

import numpy as np
import pytest

from polyequil.errors import ValidationError
from polyequil.results import (
    ObservableResult,
    PointSet,
    SeparableAxes,
    results_compatible,
)
from tests.observable_fixtures import axes


def test_values_and_coordinates_are_frozen_copies():
    values = np.arange(3)
    result = ObservableResult(values, axes(n=[1.0, 2.0, 3.0]))
    values[0] = 7
    assert result.values.dtype == np.float64
    assert result.values[0] == 0.0
    assert not result.values.flags.writeable
    assert not result.coordinates.axes[0][1].flags.writeable


def test_result_survives_a_process_boundary_frozen():
    result = ObservableResult(np.ones(2), axes(n=[0.0, 1.0]), convention={"u": "sigma"})
    copy = pickle.loads(pickle.dumps(result, protocol=pickle.DEFAULT_PROTOCOL))
    assert not copy.values.flags.writeable
    assert results_compatible(result, copy) is None


@pytest.mark.parametrize(
    ("other", "fragment"),
    [
        (ObservableResult(np.ones(3), axes(n=[0.0, 1.0, 2.0])), "shape"),
        (ObservableResult(np.ones(2) + 0j, axes(n=[0.0, 1.0])), "dtype"),
        (
            ObservableResult(np.ones(2), axes(n=[0.0, 2.0])),
            "values differ",
        ),
        (ObservableResult(np.ones(2), axes(m=[0.0, 1.0])), "name"),
        (
            ObservableResult(np.ones(2), axes(n=[0.0, 1.0]), convention={"u": 1}),
            "convention",
        ),
        (ObservableResult(np.ones(2), PointSet(np.zeros((2, 1)), ("x",))), "kind"),
    ],
)
def test_incompatibility_reasons(other, fragment):
    reference = ObservableResult(np.ones(2), axes(n=[0.0, 1.0]))
    reason = results_compatible(reference, other)
    assert reason is not None and fragment in reason


def test_auxiliary_and_provenance_never_affect_compatibility():
    a = ObservableResult(np.ones(1), axes(n=[0.0]), auxiliary={"k": 1})
    b = ObservableResult(np.ones(1), axes(n=[0.0]), provenance={"p": 2})
    assert results_compatible(a, b) is None


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValidationError):
        ObservableResult(np.ones(2), axes(n=[0.0, 1.0, 2.0]))
    with pytest.raises(ValidationError, match="strictly increase"):
        SeparableAxes(axes(n=[0.5, 1.5]).axes, (("n", np.array([0.0, 2.0, 1.0])),))
    with pytest.raises(ValidationError, match="numeric"):
        ObservableResult(np.array(["a"]))
