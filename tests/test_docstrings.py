"""Docstring structure required by docs/dev/style.md section 4."""

import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable, Iterator

import pytest

import polyequil
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
FULL_SECTIONS = frozenset({"Parameters", "Returns", "Yields", "Raises"})


def sections(obj: object) -> set[str]:
    """Return the NumPy-style section headings of an object's docstring."""
    return set(SECTION.findall(inspect.cleandoc(obj.__doc__ or "")))


def public_api() -> Iterator[tuple[str, object]]:
    """Yield (qualified name, object) for every public function, class, and
    method defined in a public polyequil module."""
    for info in pkgutil.walk_packages(polyequil.__path__, "polyequil."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            defined_here = getattr(obj, "__module__", None) == module.__name__
            if name.startswith("_") or not defined_here:
                continue
            if inspect.isfunction(obj):
                yield f"{module.__name__}.{name}", obj
            elif inspect.isclass(obj):
                yield f"{module.__name__}.{name}", obj
                for attr, member in vars(obj).items():
                    if not attr.startswith("_") and inspect.isfunction(member):
                        yield f"{module.__name__}.{name}.{attr}", member


FULL_DOCSTRINGS = sorted(
    (name, obj) for name, obj in public_api() if sections(obj) & FULL_SECTIONS
)


def test_public_api_has_full_docstrings():
    """Guard the scan itself: it must find the functions it is meant to check."""
    names = {name for name, _ in FULL_DOCSTRINGS}
    assert "polyequil.observables.extrapolation.fit_small_k" in names
    assert "polyequil.topology.ChainTopology.make_whole" in names


@pytest.mark.parametrize(
    "obj", [obj for _, obj in FULL_DOCSTRINGS], ids=[n for n, _ in FULL_DOCSTRINGS]
)
def test_full_docstrings_have_examples(obj):
    """docs/dev/style.md section 4: a multi-section docstring has Examples."""
    assert "Examples" in sections(obj)
    assert ">>>" in (obj.__doc__ or "")


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
