"""
Equal-weight streaming aggregation of task-level observables.

Rewritten from heteromorph's ``accumulators.py``. Every contribution has equal
weight and the sample variance uses ``ddof=1``. In polyequil an accumulator
only ever aggregates one time slice (or window) across independent replicas.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from abc import ABC, abstractmethod
from typing import Any

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil.errors import ValidationError
from polyequil.results import ObservableResult, results_compatible

# ============================================================
# TYPES
# ============================================================

FloatArray = NDArray[np.float64]


class Accumulator(ABC):
    """Aggregate task-level observables into a mean and sample variance."""

    __slots__ = ()

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of results accumulated so far."""

    @abstractmethod
    def update(self, result: ObservableResult) -> None:
        """Fold one result into the running aggregate."""

    @abstractmethod
    def merge(self, other: "Accumulator") -> None:
        """Absorb another accumulator of the same type."""

    @abstractmethod
    def result(self) -> tuple[NDArray[Any], FloatArray | None, int]:
        """Return ``(mean, variance, count)``.

        ``variance`` uses ``ddof=1``, is ``None`` for fewer than two results, and
        is real: for complex values it is the modulus variance ``E|x - mean|^2``.
        """


class WelfordAccumulator(Accumulator):
    """Streaming mean and variance by Welford's recurrence.

    Memory is independent of the count. ``merge`` (Chan's formula) is exact in
    the statistical sense but not bit-identical to the sequential fold.
    """

    __slots__ = ("_count", "_m2", "_mean", "_reference")

    def __init__(self) -> None:
        self._count = 0
        self._mean: NDArray[Any] | None = None
        self._m2: FloatArray | None = None
        self._reference: ObservableResult | None = None

    @property
    def count(self) -> int:
        """Return the number of results accumulated so far."""
        return self._count

    def update(self, result: ObservableResult) -> None:
        """Fold one result in; an incompatible result leaves the state unchanged."""
        if not isinstance(result, ObservableResult):
            raise ValidationError("an accumulator update requires an ObservableResult")
        values = result.values
        if self._reference is None:
            mean = np.zeros(values.shape, dtype=values.dtype)
            m2 = np.zeros(values.shape, dtype=np.float64)
            self._mean, self._m2, self._reference = mean, m2, result
        else:
            reason = results_compatible(self._reference, result)
            if reason is not None:
                raise ValidationError(f"result is incompatible: {reason}")
        assert self._mean is not None and self._m2 is not None, "state allocated"
        self._count += 1
        delta = values - self._mean
        self._mean += delta / self._count
        residual = values - self._mean
        if values.dtype.kind == "c":
            self._m2 += (delta.conj() * residual).real
        else:
            self._m2 += delta * residual

    def merge(self, other: "Accumulator") -> None:
        """Absorb another Welford accumulator with Chan's parallel formula."""
        if not isinstance(other, WelfordAccumulator):
            raise ValidationError("a WelfordAccumulator merges only with its own type")
        if other._count == 0:
            return
        assert other._mean is not None and other._m2 is not None, "other is filled"
        if self._count == 0:
            self._count = other._count
            self._mean, self._m2 = other._mean.copy(), other._m2.copy()
            self._reference = other._reference
            return
        assert self._mean is not None and self._m2 is not None, "self is filled"
        assert self._reference is not None and other._reference is not None
        reason = results_compatible(self._reference, other._reference)
        if reason is not None:
            raise ValidationError(f"merged accumulator is incompatible: {reason}")
        delta = other._mean - self._mean
        total = self._count + other._count
        self._mean = self._mean + delta * (other._count / total)
        cross = np.real(np.conj(delta) * delta) * (self._count * other._count / total)
        self._m2 = self._m2 + other._m2 + cross
        self._count = total

    def result(self) -> tuple[NDArray[Any], FloatArray | None, int]:
        """Return read-only ``(mean, variance or None, count)``."""
        if self._count == 0 or self._mean is None or self._m2 is None:
            raise ValidationError("no results have been accumulated")
        mean = _read_only(self._mean.copy())
        if self._count < 2:
            return mean, None, self._count
        return mean, _read_only(self._m2 / (self._count - 1)), self._count


class SampleStore(Accumulator):
    """An accumulator that also retains every result's values.

    All statistics are delegated to the wrapped accumulator, so retaining
    samples cannot change the reported mean or variance by even one bit.
    """

    __slots__ = ("_inner", "_values")

    def __init__(self, inner: Accumulator) -> None:
        if not isinstance(inner, Accumulator):
            raise ValidationError("a sample store wraps an Accumulator")
        if inner.count != 0:
            raise ValidationError("a sample store must wrap an empty accumulator")
        self._inner = inner
        self._values: list[NDArray[Any]] = []

    @property
    def count(self) -> int:
        """Return the number of results accumulated so far."""
        return self._inner.count

    def update(self, result: ObservableResult) -> None:
        """Fold one result in and retain its values."""
        self._inner.update(result)
        self._values.append(result.values)

    def merge(self, other: "Accumulator") -> None:
        """Absorb another sample store."""
        if not isinstance(other, SampleStore):
            raise ValidationError("a SampleStore merges only with its own type")
        self._inner.merge(other._inner)
        self._values.extend(other._values)

    def samples(self) -> NDArray[Any]:
        """Return the read-only ``(count, *shape)`` stack of retained values."""
        if not self._values:
            raise ValidationError("no results have been accumulated")
        return _read_only(np.stack(self._values, axis=0))

    def result(self) -> tuple[NDArray[Any], FloatArray | None, int]:
        """Return the wrapped accumulator's statistics."""
        return self._inner.result()


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _read_only(array: NDArray[Any]) -> NDArray[Any]:
    """Freeze a freshly computed array in place and return it."""
    array.setflags(write=False)
    return array
