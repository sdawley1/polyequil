import pickle
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from polyequil._metadata import (
    OpaqueMetadataValue,
    freeze_mapping,
    metadata_equal,
    thaw_metadata,
)
from polyequil.errors import ValidationError

trees = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda inner: (
        st.lists(inner, max_size=4)
        | st.dictionaries(st.text(min_size=1, max_size=5), inner, max_size=4)
    ),
    max_leaves=20,
)


@given(st.dictionaries(st.text(min_size=1, max_size=5), trees, max_size=5))
def test_freeze_thaw_freeze_is_stable(tree):
    frozen = freeze_mapping(tree, role="t")
    again = freeze_mapping(thaw_metadata(frozen), role="t")
    assert metadata_equal(frozen, again)
    shipped = pickle.loads(pickle.dumps(thaw_metadata(frozen)))
    assert metadata_equal(frozen, freeze_mapping(shipped, role="t"))


def test_arrays_are_copied_and_read_only():
    source = np.arange(3.0)
    frozen = freeze_mapping({"a": source, "p": Path("/x")}, role="t")
    source[0] = 99.0
    assert frozen["a"][0] == 0.0
    assert not frozen["a"].flags.writeable
    assert frozen["p"] == "/x"
    with pytest.raises(TypeError):
        frozen["b"] = 1


def test_types_compare_strictly_and_nan_equals_nan():
    assert not metadata_equal(1, 1.0)
    assert metadata_equal(float("nan"), float("nan"))
    assert metadata_equal(complex(1, float("nan")), complex(1, float("nan")))
    assert not metadata_equal(np.zeros(2), np.zeros(2, dtype=np.float32))


def test_unsupported_values_raise_or_become_opaque():
    with pytest.raises(ValidationError, match="unsupported"):
        freeze_mapping({"f": object()}, role="t")
    frozen = freeze_mapping({"f": object()}, role="t", allow_opaque=True)
    assert isinstance(frozen["f"], OpaqueMetadataValue)
    with pytest.raises(ValidationError, match="non-empty strings"):
        freeze_mapping({"": 1}, role="t")
