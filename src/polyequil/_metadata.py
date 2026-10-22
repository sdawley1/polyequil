"""
Detached, read-only, picklable metadata trees.

Rewritten from heteromorph's ``_metadata.py``. Mappings, sequences, scalars,
and numeric or string arrays are copied into a read-only tree. Callers that
must record an otherwise unsupported parameter may request an
:class:`OpaqueMetadataValue` snapshot, so live mutable state never leaks into a
result's provenance.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

# ----- third party -----
import numpy as np

# ----- local -----
from polyequil.errors import ValidationError

# ============================================================
# CONSTANTS
# ============================================================

MAX_REPRESENTATION_LENGTH: int = 1024

# ============================================================
# TYPES
# ============================================================


@dataclass(frozen=True, slots=True)
class OpaqueMetadataValue:
    """Immutable descriptive snapshot of an unsupported parameter value."""

    type_name: str
    representation: str


# ============================================================
# PUBLIC API
# ============================================================


def freeze_metadata(value: Any, *, role: str, allow_opaque: bool = False) -> Any:
    """Return a deeply detached, read-only copy of a metadata value.

    Parameters
    ----------
    value : Any
        Scalar, string, path, array, mapping, or sequence (nested freely).
    role : str
        Name used in error messages, extended with the path into the tree.
    allow_opaque : bool
        Replace unsupported objects with :class:`OpaqueMetadataValue` instead
        of raising.

    Returns
    -------
    Any
        Primitive copies, read-only arrays, ``MappingProxyType`` mappings, and
        tuples.

    Raises
    ------
    ValidationError
        If an unsupported value is found and ``allow_opaque`` is False.
    """
    if isinstance(value, Enum):
        return freeze_metadata(value.value, role=role, allow_opaque=allow_opaque)
    if isinstance(value, OpaqueMetadataValue):
        return OpaqueMetadataValue(str(value.type_name), str(value.representation))
    if value is None:
        return None
    scalar = _freeze_scalar(value)
    if scalar is not _UNSUPPORTED:
        return scalar
    if isinstance(value, np.generic):
        return freeze_metadata(value.item(), role=role, allow_opaque=allow_opaque)
    if isinstance(value, np.ndarray):
        return _freeze_array(value, role=role, allow_opaque=allow_opaque)
    if isinstance(value, Mapping):
        return _freeze_mapping_tree(value, role=role, allow_opaque=allow_opaque)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            freeze_metadata(item, role=f"{role}[{index}]", allow_opaque=allow_opaque)
            for index, item in enumerate(value)
        )
    if allow_opaque:
        return _opaque(value)
    raise ValidationError(
        f"{role} contains unsupported value type {type(value).__name__}; use "
        "strings, scalar values, arrays, mappings, or sequences"
    )


def freeze_mapping(
    mapping: Mapping[str, Any], *, role: str, allow_opaque: bool = False
) -> Mapping[str, Any]:
    """Freeze a string-keyed metadata mapping; raise if it is not a mapping."""
    if not isinstance(mapping, Mapping):
        raise ValidationError(f"{role} must be a mapping")
    frozen = freeze_metadata(mapping, role=role, allow_opaque=allow_opaque)
    assert isinstance(frozen, Mapping), "a frozen mapping must stay a mapping"
    return frozen


def thaw_metadata(value: Any) -> Any:
    """Return a pickle-friendly (dict, tuple, writable array) copy of a tree."""
    if isinstance(value, Mapping):
        return {key: thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(thaw_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.array(value, order="C", copy=True)
    return value


def metadata_equal(left: Any, right: Any) -> bool:
    """Compare two frozen metadata trees exactly; NaN equals NaN.

    Types must match exactly, so ``1`` and ``1.0`` differ.
    """
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return _arrays_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(metadata_equal(left[key], right[key]) for key in left)
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return False
        return len(left) == len(right) and all(
            metadata_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, (float, complex)):
        return _numbers_equal(left, right)
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


# ============================================================
# PRIVATE HELPERS
# ============================================================

_UNSUPPORTED = object()


def _plain_str(value: str) -> str:
    """Return ``value`` as an exact ``str``, dropping any subclass."""
    return value if type(value) is str else str.__str__(value)


def _freeze_scalar(value: Any) -> Any:
    """Return a primitive copy of a scalar, or the ``_UNSUPPORTED`` sentinel."""
    if isinstance(value, str):
        return _plain_str(value)
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, complex):
        return complex(value.real, value.imag)
    if isinstance(value, Path):
        return str(value)
    return _UNSUPPORTED


def _freeze_array(value: np.ndarray, *, role: str, allow_opaque: bool) -> Any:
    """Return a read-only C-order copy of a non-object, non-structured array."""
    if value.dtype.kind in "OVMm":
        if allow_opaque:
            return _opaque(value)
        raise ValidationError(
            f"{role} arrays must have a non-object, non-structured dtype, "
            f"not {value.dtype}"
        )
    frozen = np.array(value, order="C", copy=True)
    frozen.setflags(write=False)
    return frozen


def _freeze_mapping_tree(
    value: Mapping[Any, Any], *, role: str, allow_opaque: bool
) -> Mapping[str, Any]:
    """Freeze a mapping whose keys are unique non-empty strings."""
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValidationError(f"{role} mapping keys must be non-empty strings")
        normalized = _plain_str(key)
        if normalized in frozen:
            raise ValidationError(f"{role} mapping keys must stay unique")
        frozen[normalized] = freeze_metadata(
            item, role=f"{role}[{normalized!r}]", allow_opaque=allow_opaque
        )
    return MappingProxyType(frozen)


def _opaque(value: object) -> OpaqueMetadataValue:
    """Return a truncated ``repr`` snapshot of an unsupported value."""
    value_type = type(value)
    try:
        representation = repr(value)
    except Exception:
        representation = "<representation unavailable>"
    if len(representation) > MAX_REPRESENTATION_LENGTH:
        representation = representation[: MAX_REPRESENTATION_LENGTH - 3] + "..."
    return OpaqueMetadataValue(
        type_name=f"{value_type.__module__}.{value_type.__qualname__}",
        representation=representation,
    )


def _arrays_equal(left: Any, right: Any) -> bool:
    """Exact array equality; dtype and shape must match; NaN equals NaN."""
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        return False
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _numbers_equal(left: float | complex, right: float | complex) -> bool:
    """Compare floats or complexes component-wise with NaN equal to NaN."""
    pairs = [(complex(left).real, complex(right).real)]
    pairs.append((complex(left).imag, complex(right).imag))
    return all(a == b or (np.isnan(a) and np.isnan(b)) for a, b in pairs)
