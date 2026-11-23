"""Docstring structure required by docs/dev/style.md section 4."""

import inspect
import re
from collections.abc import Callable

import pytest

from polyequil.indices import density_index
from polyequil.observables.chain import radius_of_gyration
from polyequil.observables.extrapolation import fit_small_k, small_k_limit
from polyequil.observables.structure import (
    orientational_structure_factor,
    structure_factor,
)
from polyequil.observables.window import window_mean
from polyequil.reciprocal import collective_amplitudes

SECTION = re.compile(r"^(\w[\w ]*)\n-{3,}$", re.MULTILINE)


def sections(obj: object) -> set[str]:
    """Return the NumPy-style section headings of an object's docstring."""
    return set(SECTION.findall(inspect.cleandoc(obj.__doc__ or "")))


# ----- D-003 -----

D003_FUNCTIONS: tuple[Callable[..., object], ...] = (
    small_k_limit,
    fit_small_k,
    structure_factor,
    orientational_structure_factor,
    window_mean,
    radius_of_gyration,
    collective_amplitudes,
    density_index,
)


@pytest.mark.parametrize("function", D003_FUNCTIONS, ids=lambda f: f.__name__)
def test_regression_D003_wp5_docstrings_document_raises(function):
    """D-003: these functions raise ValidationError, so they document it."""
    assert "Raises" in sections(function)
    raises = inspect.cleandoc(function.__doc__ or "").split("Raises\n------", 1)[1]
    assert "ValidationError" in raises
