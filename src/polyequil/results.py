"""
The per-task observable contract: values, coordinates, and conventions.

Rewritten from heteromorph's ``observables.py`` (``AffineGrid`` dropped). An
observable is a picklable callable with the signature::

    observable(item, *, threads=1, **parameters) -> ObservableResult

where ``item`` is a :class:`~polyequil.configuration.ChainConfiguration` for
frame observables or a :class:`~polyequil.trajectory.WindowData` for window
observables. ``threads`` names parallelism inside one task and is chosen by the
execution layer. A stochastic observable declares a keyword ``rng`` and draws
only from the injected generator.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ----- local -----
from polyequil._metadata import freeze_mapping, metadata_equal, thaw_metadata
from polyequil.errors import ValidationError

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]


class Coordinates(ABC):
    """Where an observable's values live: SeparableAxes or PointSet."""

    __slots__ = ()

    @abstractmethod
    def validate_against(self, shape: tuple[int, ...]) -> None:
        """Raise :class:`ValidationError` unless these coordinates fit ``shape``."""


@dataclass(frozen=True, slots=True)
class SeparableAxes(Coordinates):
    """A rectilinear grid: one bin-centre array per axis, optional bin edges.

    Zero axes describe a scalar; one axis a binned function such as MSID(n).
    """

    axes: tuple[tuple[str, FloatArray], ...] = ()
    edges: tuple[tuple[str, FloatArray], ...] = ()

    def __post_init__(self) -> None:
        entries = _pairs(self.axes, "axis")
        names = _unique_names((name for name, _ in entries), "axis")
        axes = tuple(
            (name, _frozen_coordinate(values, f"axis {name!r}", ndim=1))
            for name, (_, values) in zip(names, entries, strict=True)
        )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "edges", _validated_edges(self.edges, axes))

    def validate_against(self, shape: tuple[int, ...]) -> None:
        """Require one axis per value dimension with matching lengths."""
        if len(self.axes) != len(shape):
            raise ValidationError(
                f"values have rank {len(shape)} but {len(self.axes)} axes were supplied"
            )
        for axis, (name, coordinate) in enumerate(self.axes):
            if coordinate.shape[0] != shape[axis]:
                raise ValidationError(
                    f"axis {name!r} has length {coordinate.shape[0]} but values "
                    f"axis {axis} has length {shape[axis]}"
                )

    def names(self) -> tuple[str, ...]:
        """Return the axis names in order."""
        return tuple(name for name, _ in self.axes)

    def edge_values(self, name: str) -> FloatArray | None:
        """Return the bin edges for ``name``, or ``None`` when none were given."""
        return next((values for axis, values in self.edges if axis == name), None)

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor so arrays come back read-only."""
        return (type(self), (self.axes, self.edges))


@dataclass(frozen=True, slots=True)
class PointSet(Coordinates):
    """``n`` scattered coordinates with named components; values axis 0 is ``n``."""

    points: FloatArray
    components: tuple[str, ...]
    name: str = "point"

    def __post_init__(self) -> None:
        points = _frozen_coordinate(self.points, "points", ndim=2)
        components = _unique_names(self.components, "component")
        if len(components) != points.shape[1]:
            raise ValidationError(
                f"points have {points.shape[1]} components but "
                f"{len(components)} component names were supplied"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("a point set name must be a non-empty string")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "name", str.__str__(self.name))

    def validate_against(self, shape: tuple[int, ...]) -> None:
        """Require values whose first axis has one entry per point."""
        if len(shape) < 1:
            raise ValidationError("a point set requires values with at least one axis")
        if shape[0] != self.points.shape[0]:
            raise ValidationError(
                f"values axis 0 has length {shape[0]} but the point set has "
                f"{self.points.shape[0]} points"
            )

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor so arrays come back read-only."""
        return (type(self), (self.points, self.components, self.name))


@dataclass(frozen=True, slots=True)
class ObservableResult:
    """One task's value of an observable, with its coordinates.

    ``values`` are frozen float64 or complex128 copies; NaN and inf are allowed
    so an undefined result stays visible in the ensemble mean. ``convention``
    holds exactly what must match for averaging to be valid (normalization,
    estimator, units, binning) and is compared across tasks. ``auxiliary`` and
    ``provenance`` are task-specific, never compared, and never averaged.
    """

    values: NDArray[Any]
    coordinates: Coordinates = field(default_factory=SeparableAxes)
    convention: Mapping[str, Any] = field(default_factory=dict)
    auxiliary: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = frozen_values(self.values)
        object.__setattr__(self, "values", values)
        if type(self.coordinates) not in (SeparableAxes, PointSet):
            raise ValidationError(
                "coordinates must be SeparableAxes or PointSet, not "
                f"{type(self.coordinates).__name__}"
            )
        self.coordinates.validate_against(values.shape)
        for name in ("convention", "auxiliary", "provenance"):
            frozen = freeze_mapping(getattr(self, name), role=f"observable {name}")
            object.__setattr__(self, name, frozen)

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through the constructor when crossing a process boundary.

        ``MappingProxyType`` cannot be pickled and NumPy drops the read-only
        flag at the multiprocessing pickle protocol; reconstruction fixes both.
        """
        return (
            type(self),
            (
                self.values,
                self.coordinates,
                thaw_metadata(self.convention),
                thaw_metadata(self.auxiliary),
                thaw_metadata(self.provenance),
            ),
        )


# ============================================================
# PUBLIC API
# ============================================================


def frozen_values(values: ArrayLike) -> NDArray[Any]:
    """Return values as a read-only float64 or complex128 copy; reject non-numeric."""
    array = np.asarray(values)
    if array.dtype.kind in "OSUVMm":
        raise ValidationError(f"observable values must be numeric, not {array.dtype}")
    dtype = np.complex128 if array.dtype.kind == "c" else np.float64
    frozen = np.array(array, dtype=dtype, order="C", copy=True)
    frozen.setflags(write=False)
    return frozen


def results_compatible(
    reference: ObservableResult, candidate: ObservableResult
) -> str | None:
    """Return ``None`` when two results may be averaged, else the reason.

    Shape, dtype, coordinates (exact equality), and convention must all match.
    Never raises; the caller decides how to report.
    """
    if reference.values.shape != candidate.values.shape:
        return (
            f"values shape {candidate.values.shape} does not match the reference "
            f"shape {reference.values.shape}"
        )
    if reference.values.dtype != candidate.values.dtype:
        return (
            f"values dtype {candidate.values.dtype} does not match the reference "
            f"dtype {reference.values.dtype}"
        )
    if candidate.coordinates is not reference.coordinates:
        reason = _coordinates_reason(reference.coordinates, candidate.coordinates)
        if reason is not None:
            return reason
    if not metadata_equal(reference.convention, candidate.convention):
        return (
            f"convention {dict(candidate.convention)} does not match the "
            f"reference convention {dict(reference.convention)}"
        )
    return None


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _frozen_coordinate(values: ArrayLike, name: str, *, ndim: int) -> FloatArray:
    """Return a frozen finite float64 copy of rank ``ndim``."""
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    if array.ndim != ndim:
        raise ValidationError(f"{name} must be a rank-{ndim} array")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _equal_array(left: NDArray[Any], right: NDArray[Any]) -> bool:
    """Exact equality treating NaNs in identical positions as equal."""
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _unique_names(names: Iterable[object], kind: str) -> tuple[str, ...]:
    """Return ``names`` as a tuple of unique, non-empty, exact strings."""
    if isinstance(names, (str, bytes)):
        raise ValidationError(f"{kind} names must be supplied as a sequence")
    checked: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{kind} names must be non-empty strings")
        normalized = str.__str__(name)
        if normalized in checked:
            raise ValidationError(f"duplicate {kind} name {normalized!r}")
        checked.append(normalized)
    return tuple(checked)


def _pairs(raw: Any, kind: str) -> tuple[tuple[Any, Any], ...]:
    """Normalize a mapping or a sequence of pairs into a tuple of pairs."""
    entries = (
        tuple(raw.items())
        if isinstance(raw, Mapping)
        else tuple(tuple(entry) for entry in raw)
    )
    if any(len(entry) != 2 for entry in entries):
        raise ValidationError(f"each {kind} must be a (name, values) pair")
    return entries


def _validated_edges(
    raw: Any, axes: tuple[tuple[str, FloatArray], ...]
) -> tuple[tuple[str, FloatArray], ...]:
    """Validate bin edges against their axes and return them in axis order."""
    entries = _pairs(raw, "edge array")
    names = _unique_names((name for name, _ in entries), "edge axis")
    lengths = {name: values.shape[0] for name, values in axes}
    frozen: dict[str, FloatArray] = {}
    for name, (_, values) in zip(names, entries, strict=True):
        if name not in lengths:
            raise ValidationError(f"edge axis {name!r} has no matching axis")
        edges = _frozen_coordinate(values, f"edges for axis {name!r}", ndim=1)
        if edges.shape[0] != lengths[name] + 1:
            raise ValidationError(
                f"edges for axis {name!r} have length {edges.shape[0]} but the "
                f"axis has length {lengths[name]}"
            )
        if np.any(np.diff(edges) <= 0.0):
            raise ValidationError(f"edges for axis {name!r} must strictly increase")
        frozen[name] = edges
    return tuple((name, frozen[name]) for name, _ in axes if name in frozen)


def _named_arrays_reason(
    kind: str,
    expected: tuple[tuple[str, FloatArray], ...],
    actual: tuple[tuple[str, FloatArray], ...],
) -> str | None:
    """Compare two (name, array) sequences exactly."""
    if len(expected) != len(actual):
        return f"{kind} sets differ from the reference"
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if left[0] != right[0]:
            return f"{kind} {index} name {right[0]!r} does not match {left[0]!r}"
        if not _equal_array(left[1], right[1]):
            return f"{kind} {left[0]!r} values differ from the reference"
    return None


def _coordinates_reason(expected: Coordinates, actual: Coordinates) -> str | None:
    """Return why two coordinate objects disagree, or ``None``."""
    if type(expected) is not type(actual):
        return (
            f"coordinates of kind {type(actual).__name__} do not match the "
            f"reference kind {type(expected).__name__}"
        )
    if isinstance(expected, SeparableAxes) and isinstance(actual, SeparableAxes):
        reason = _named_arrays_reason("coordinate axis", expected.axes, actual.axes)
        return reason or _named_arrays_reason("edge axis", expected.edges, actual.edges)
    if isinstance(expected, PointSet) and isinstance(actual, PointSet):
        if expected.name != actual.name or expected.components != actual.components:
            return "point set name or components differ from the reference"
        if not _equal_array(expected.points, actual.points):
            return "point set coordinates differ from the reference"
        return None
    return f"coordinates of kind {type(expected).__name__} cannot be compared"
