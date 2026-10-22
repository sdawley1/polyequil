# Coding style

Binding for all Python in `src/polyequil/`, `tests/`, and `scripts/`.
Mechanical rules are enforced by `ruff` and `mypy` (§6); the rest is enforced
by review. Code adapted from heteromorph names its source file in the module
docstring.

## 1. File header

Importable modules start with a module docstring. Executable scripts (files in
`scripts/` run as `python scripts/foo.py`) add `#!/usr/bin/env python3` as the
first line and are marked executable; importable modules never carry a
shebang.

```python
"""
One-line summary.

Optional longer description: what the module owns, key assumptions, and the
conventions it uses (with a pointer to `polyequil.conventions`).

Author Name(s)
MM/YYYY
"""
```

The date records when the module was created; do not update it on edits.

## 2. Section banners

Top-level sections:

```python
# ============================================================
# SECTION TITLE
# ============================================================
```

Inside a `__main__` block (shorter, to distinguish):

```python
    # =========================================================
    # SECTION TITLE
    # =========================================================
```

Sub-groups inside sections, functions, argument parsers, and imports:

```python
    # ----- label -----
```

Typical module order: `IMPORTS`, `CONSTANTS`, `TYPES` (dataclasses, ABCs),
`PUBLIC API`, `PRIVATE HELPERS`, `__main__` (scripts only).

## 3. Imports

Under an `IMPORTS` banner, in the three groups `ruff`'s isort rules produce,
each labelled. Within a group, bare `import` lines come before `from` lines,
each sorted alphabetically (`ruff check --fix` does this).

```python
# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections.abc import Sequence
from dataclasses import dataclass

# ----- third party -----
import numpy as np
from numpy.typing import NDArray

# ----- local -----
from polyequil.conventions import SCHEMA
from polyequil.results import ObservableResult
```

- Built-in generics and `|` unions: `list[float]`, `dict[str, int]`,
  `tuple[int, ...]`, `NDArray[np.float64] | None`. Abstract containers come
  from `collections.abc`, not `typing`.
- Absolute imports (`from polyequil.box import Box`).
- No wildcard imports. Public names are listed in each package's `__all__`.

## 4. Names, types, docstrings

- `UPPER_SNAKE_CASE` constants, `lower_snake_case` functions and variables,
  `CapWords` classes, a leading `_` for private helpers and modules.
- Physics names follow `docs/design.md` (`msid`, `radius_of_gyration`,
  `structure_factor`, `kappa_t`, `chi`); do not invent synonyms.
- Type hints on every parameter and return value. Arrays are
  `NDArray[np.float64]` (or the exact dtype); document shapes in the
  docstring, e.g. `(n_frames, n_beads, 3)`.
- Data types are `@dataclass(frozen=True, slots=True)`, validated in
  `__post_init__` (use `polyequil._arrays.frozen_array` for array fields),
  with `eq=False` plus an explicit `__eq__` when fields are arrays, and a
  `__reduce__` that rebuilds through the constructor so arrays stay
  read-only across process boundaries.
- NumPy-style docstrings. One line when the function has at most two
  parameters and an obvious return; otherwise a summary line, a blank line,
  and `Parameters`, `Returns`, and `Raises` sections. Give units (LJ reduced
  units unless stated) and conventions.

```python
def msid(
    configuration: ChainConfiguration,
    *,
    separations: Sequence[int],
    threads: int = 1,
) -> ObservableResult:
    """Return the mean-squared internal distance of every chain.

    Parameters
    ----------
    configuration : ChainConfiguration
        One frame; positions are unwrapped internally.
    separations : Sequence[int]
        Bond separations n, each in [1, N_chain - 1].
    threads : int
        Reserved by the ensemble execution layer.

    Returns
    -------
    ObservableResult
        Values of shape (len(separations),) holding <R^2(n)> in sigma^2,
        averaged over all chains and all pairs at each separation.

    Raises
    ------
    ValidationError
        If a separation is out of range for the shortest chain.
    """
```

## 5. Safety rules (NASA Power-of-10, adapted)

1. Simple control flow: no recursion.
2. Bounded loops: prefer `for`; every `while` has an evident bound.
3. Pre-allocate arrays (`np.empty`, `np.zeros`) instead of growing lists in
   hot loops; prefer vectorized NumPy over Python loops.
4. At most 60 lines per function, excluding the docstring and blank lines.
5. Check inputs and invariants. Public functions validate their arguments
   and raise `ValidationError` (or another package exception) with a message
   naming the offending value. `assert` is reserved for internal invariants
   that cannot fail if the code is correct, since `python -O` strips it.
   Non-trivial functions carry at least two such checks.
6. Minimal scope: no mutable module-level state beyond constants; no global
   random state. Randomness comes from an injected
   `numpy.random.Generator` (`TrajectoryEnsemble` supplies one per task).
   The one sanctioned cache is the bounded per-process header cache in
   `polyequil.trajectory` (`functools.lru_cache`, keyed on file mtime).
7. Check return values and `None` before use.
8. No `exec`, `eval`, metaclasses, or monkey-patching.
9. Zero warnings under `ruff` and `mypy --strict` from the first commit.

## 6. Tooling contract (implemented in `pyproject.toml`)

- Build with `uv` (hatchling backend); Python `>=3.11`; runtime
  dependencies NumPy `>=2`, `h5py`, `threadpoolctl`; SciPy and a `cuda`
  extra (CuPy) are added when first used; dev group: `pytest`,
  `hypothesis`, `ruff`, `mypy`. `uv.lock` is committed; CI uses
  `uv sync --locked`.
- `ruff`: `line-length = 88`, `target-version = "py311"`, rule sets
  `E, W, F, I, UP, B, SIM, N, D, ANN, NPY, PT, RUF, EXE`, pydocstyle
  convention `numpy` (D105, D203, D213, ANN401 ignored); `D`, `ANN`, and
  `PT018` relaxed under `tests/`; `PT` off under `src/`; Markdown excluded.
- `mypy`: `strict = true` over `src/` and `tests/`; test bodies of untyped
  test functions are not checked (`check_untyped_defs = false` for
  `tests.*`).
- `pytest`: markers `slow`, `gpu`, `lammps`; default
  `addopts = "-m 'not slow and not gpu and not lammps'"`; `.hypothesis/` in
  `.gitignore`.
- CI (GitHub Actions): `ruff format --check`, `ruff check`, `mypy`, the
  default `pytest` run with `--hypothesis-profile=ci`, on Ubuntu (Python
  3.11–3.13) and macOS (spawn start method).

## 7. Scripts and `__main__`

Scripts in `scripts/` (simulation drivers, SLURM helpers, figure generators)
use `if __name__ == "__main__":` directly with `argparse`, arguments grouped
by `# ----- label -----` comments, and `__main__` banners between steps.
Scripts call library functions; they do not contain analysis logic. If a
console entry point is needed, it is a `main()` function in the package
registered under `[project.scripts]`.

## 8. Template

`src/polyequil/observables/chain.py` (radius of gyration) is the reference
example of an observable module: header, banners, grouped imports, a public
observable with the `(item, *, threads=1, **params) -> ObservableResult`
contract and a `convention`, and a validated private-style helper.
`src/polyequil/box.py` is the reference example of a frozen data type.
