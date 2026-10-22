"""
Shared validation helpers for frozen NumPy arrays.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from typing import Any

# ----- third party -----
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

# ----- local -----
from polyequil.errors import ValidationError

# ============================================================
# PUBLIC API
# ============================================================


def frozen_array(
    values: ArrayLike,
    *,
    name: str,
    dtype: DTypeLike,
    shape: tuple[int | None, ...],
    finite: bool = False,
) -> NDArray[Any]:
    """Return a read-only C-order copy of ``values`` with a checked shape.

    Parameters
    ----------
    values : ArrayLike
        Input data; integer targets reject non-integral floating input.
    name : str
        Name used in error messages.
    dtype : DTypeLike
        Target dtype.
    shape : tuple[int | None, ...]
        Required shape; ``None`` entries accept any length.
    finite : bool
        Require every value to be finite.

    Returns
    -------
    NDArray[Any]
        The frozen copy.

    Raises
    ------
    ValidationError
        On a shape mismatch, a lossy integer conversion, or non-finite values.
    """
    raw = np.asarray(values)
    target = np.dtype(dtype)
    if raw.dtype.kind in "OSUVMm":
        raise ValidationError(f"{name} must be numeric, not dtype {raw.dtype}")
    lossy = target.kind in "iu" and raw.dtype.kind == "f" and raw.size > 0
    if lossy and (not np.all(np.isfinite(raw)) or np.any(raw != np.round(raw))):
        raise ValidationError(f"{name} must hold integer values")
    array = np.array(raw, dtype=target, order="C", copy=True)
    if array.ndim != len(shape) or any(
        want is not None and have != want
        for have, want in zip(array.shape, shape, strict=True)
    ):
        raise ValidationError(f"{name} has shape {array.shape}, expected {shape}")
    if finite and not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
