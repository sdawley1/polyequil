"""
Random-stream injection and random-state detection for ensemble evaluation.

Rewritten from heteromorph's ``ensemble.py`` helpers. Each task receives its own
PCG64 stream derived from one root ``SeedSequence`` by ``spawn_key``; callers
may never smuggle generator objects through parameters.

Sam Dawley
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import functools
import inspect
import random
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

# ----- third party -----
import numpy as np

# ----- local -----
from polyequil._metadata import freeze_mapping
from polyequil.errors import ValidationError

# ============================================================
# CONSTANTS
# ============================================================

RANDOM_STATE_TYPES: tuple[type, ...] = (
    np.random.Generator,
    np.random.BitGenerator,
    np.random.RandomState,
    np.random.SeedSequence,
    random.Random,
)
ATOMIC_TYPES: tuple[type, ...] = (
    str,
    bytes,
    bytearray,
    bool,
    int,
    float,
    complex,
    Enum,
    Path,
    np.generic,
    np.ndarray,
)
MAX_VISITED_OBJECTS: int = 100_000

# ============================================================
# TYPES
# ============================================================

StreamRoot = tuple[Any, tuple[int, ...], int]

# ============================================================
# PUBLIC API
# ============================================================


def find_random_state(value: object, path: str) -> str | None:
    """Return the path of the first random-state object reachable from ``value``.

    Inspects containers, ``functools.partial`` bindings, bound methods, function
    defaults, dataclass fields, instance dictionaries, and slots. Never calls
    properties or consumes iterators. Returns ``None`` when nothing is found.
    """
    seen: set[int] = set()
    pending: list[tuple[object, str]] = [(value, path)]
    for _ in range(MAX_VISITED_OBJECTS):
        if not pending:
            return None
        item, location = pending.pop()
        if isinstance(item, RANDOM_STATE_TYPES):
            return location
        if item is None or isinstance(item, (*ATOMIC_TYPES, ModuleType, type)):
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        pending.extend(reversed(_children(item, location)))
    raise ValidationError(f"{path} is too deeply nested to scan for random state")


def root_stream(random_seed: object) -> tuple[StreamRoot, Mapping[str, Any]]:
    """Normalize a seed into a stream root and a frozen provenance record.

    ``random_seed`` may be None (system entropy, recorded so the run is
    replayable), a non-negative int or int sequence, or an unspawned
    ``SeedSequence``; live generators are refused.
    """
    if isinstance(random_seed, (bool, np.bool_)):
        raise ValidationError("random_seed cannot be a Boolean")
    live = (np.random.Generator, np.random.BitGenerator, np.random.RandomState)
    if isinstance(random_seed, (*live, random.Random)):
        raise ValidationError(
            "random_seed must be an int, int sequence, SeedSequence, or None, "
            "not live random-generator state"
        )
    root = _seed_sequence(random_seed)
    entropy = root.entropy
    assert entropy is not None, "SeedSequence always resolves its entropy"
    entropy_record: object = (
        int(entropy) if isinstance(entropy, (int, np.integer)) else tuple(entropy)
    )
    stream = (entropy_record, tuple(root.spawn_key), int(root.pool_size))
    provenance = freeze_mapping(
        {
            "scheme": "seedsequence-spawn-key-(replica, slice)-v1",
            "bit_generator": "numpy.random.PCG64",
            "root_entropy": entropy_record,
            "root_spawn_key": tuple(root.spawn_key),
            "pool_size": int(root.pool_size),
            "entropy_source": "system" if random_seed is None else "user",
            "numpy_version": np.__version__,
        },
        role="randomness provenance",
    )
    return stream, provenance


def child_generator(root: StreamRoot, key: tuple[int, ...]) -> np.random.Generator:
    """Return the PCG64 generator for one task, keyed by ``spawn_key`` suffix."""
    entropy, spawn_key, pool_size = root
    sequence = np.random.SeedSequence(
        entropy, spawn_key=(*spawn_key, *key), pool_size=pool_size
    )
    return np.random.Generator(np.random.PCG64(sequence))


# ============================================================
# PRIVATE HELPERS
# ============================================================


def _seed_sequence(random_seed: object) -> np.random.SeedSequence:
    """Return a fresh SeedSequence equivalent to ``random_seed``."""
    if isinstance(random_seed, np.random.SeedSequence):
        if random_seed.n_children_spawned != 0:
            raise ValidationError("random_seed SeedSequence has already spawned")
        return np.random.SeedSequence(
            random_seed.entropy,
            spawn_key=tuple(random_seed.spawn_key),
            pool_size=int(random_seed.pool_size),
        )
    try:
        return np.random.SeedSequence(random_seed)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "random_seed must be None, a non-negative int or int sequence, or an "
            "unspawned numpy.random.SeedSequence"
        ) from exc


def _children(item: object, location: str) -> list[tuple[object, str]]:
    """Return the inert sub-objects of ``item`` that may hold random state."""
    if isinstance(item, Mapping):
        out: list[tuple[object, str]] = []
        for key, nested in item.items():
            out += [(key, f"{location}.key[{key!r}]"), (nested, f"{location}[{key!r}]")]
        return out
    if isinstance(item, (tuple, list, set, frozenset)):
        return [(nested, f"{location}[{i}]") for i, nested in enumerate(item)]
    if isinstance(item, functools.partial):
        return [
            (item.func, f"{location}.func"),
            (item.args, f"{location}.args"),
            (item.keywords or {}, f"{location}.keywords"),
        ]
    if inspect.ismethod(item):
        return [
            (item.__func__, f"{location}.__func__"),
            (item.__self__, f"{location}.__self__"),
        ]
    if inspect.isfunction(item):
        return [
            (item.__defaults__ or (), f"{location}.__defaults__"),
            (item.__kwdefaults__ or {}, f"{location}.__kwdefaults__"),
        ]
    if inspect.isbuiltin(item):
        return []
    return _attribute_children(item, location)


def _attribute_children(item: object, location: str) -> list[tuple[object, str]]:
    """Return dataclass fields, instance attributes, and slots of an object."""
    if is_dataclass(item) and not isinstance(item, type):
        return [(getattr(item, f.name), f"{location}.{f.name}") for f in fields(item)]
    try:
        out = [(nested, f"{location}.{name}") for name, nested in vars(item).items()]
    except TypeError:
        out = []
    for owner in type(item).__mro__:
        declared = owner.__dict__.get("__slots__", ())
        for name in (declared,) if isinstance(declared, str) else declared:
            if name not in {"__dict__", "__weakref__"} and hasattr(item, name):
                out.append((getattr(item, name), f"{location}.{name}"))
    return out
