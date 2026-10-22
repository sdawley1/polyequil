# AGENTS.md — instructions for every coding agent in `polyequil`

This file is loaded automatically into your context. It is binding for every
task in this repository, however small. Detail lives in the linked files; read
them when the task touches their subject.

| Read | When |
|---|---|
| `docs/design.md` | Before your first task: the science, architecture (§5), and work packages (§6) |
| `docs/dev/style.md` | Before writing or editing any Python |
| `docs/dev/testing.md` | Before writing or changing any test |

## 1. What this project is for

`polyequil` is a tested Python package for diagnosing whether a simulated
polymer melt has reached equilibrium. It computes local diagnostics (density,
pressure, $g(r)$, MSID, $R_g$) and two global ones that test the equilibrium
**fluctuation–response relation**: a density index from $S(q\to0)$ and
$\kappa_T$, and an orientational index from bond-orientation fluctuations and
the susceptibility $\chi$. Results are evaluated over ensembles of independent
trajectories, with honest error bars, on CPU or GPU.

Every change should serve that goal: a physicist must be able to trust a
number this package prints. Correct and verified beats fast and clever, and
both beat feature-rich.

## 2. Repository map and commands

```
src/polyequil/                 the package
src/polyequil/_vendor/heteromorph/   vendored upstream code — READ-ONLY (§3)
tests/                         pytest suite; mirrors src/ layout
docs/                          design doc, dev guides, MkDocs site
DEFECTS.md                     defect log (§6)
```

```bash
uv sync --all-extras                       # set up the environment
uv run pytest                              # fast suite (default markers skip slow/gpu/lammps)
uv run pytest -m "slow or gpu or lammps"   # extended suite, on a machine that has them
uv run ruff format . && uv run ruff check . && uv run mypy   # must be clean
```

Work package 0 (WP0) creates the `pyproject.toml` configuration these commands
rely on (`docs/dev/style.md` §6). Until it lands, only WP0 work is valid.

## 3. Hard rules

Breaking any of these is a defect, whoever wrote the code.

1. **Never edit `src/polyequil/_vendor/`.** Adapt by subclassing or wrapping
   in `polyequil` modules. A defect in vendored code is logged in
   `DEFECTS.md` and reported to the maintainer; it is fixed upstream and
   re-synced, never patched in place.
2. **Never build a `heteromorph` `Ensemble` from frames of one trajectory.**
   Its error bars assume independent realizations. Cross-replica statistics
   go through `TrajectoryEnsemble`; within-trajectory errors use block
   averaging.
3. **One definition per physical convention.** Normalizations, units, sign
   conventions, and estimator choices (e.g. per-bond normalization of
   $S_Q$, the $S(q)$ extrapolation form) are defined once, in
   `src/polyequil/conventions.py`, and imported everywhere else.
4. **GPU results must match CPU results** to a documented tolerance, tested.
   A requested GPU backend that is unavailable raises; it never falls back
   to CPU silently.
5. **No silent failure.** Validate public inputs and raise the package's
   exception types with a message naming the bad value; never return a
   placeholder (zero, empty array, `None`) for an undefined result. NaN may
   propagate only where the docstring says so.
6. **Never commit data**: trajectories, HDF5 files, LAMMPS dumps, or
   generated figures. Tests build small inputs in code.
7. **Never push to `main`** or rewrite shared history.

## 4. Coding style (summary — full rules in `docs/dev/style.md`)

- Python ≥ 3.11; built-in generics (`list[int]`, `dict[str, float]`,
  `X | None`), never `typing.List`/`Dict`/`Optional`.
- Full type hints; zero warnings under `ruff` and `mypy --strict`.
- Every module starts with a docstring (summary, description, author,
  `MM/YYYY`) and uses section banners. Shebang lines only on executable
  scripts, never on importable modules.
- NumPy-style docstrings: one line for simple functions; `Parameters` /
  `Returns` / `Raises` sections otherwise. State units and conventions.
- Functions ≤ 60 lines; no recursion; bounded loops; pre-allocated arrays in
  hot paths; no `exec`/`eval`/metaclasses.
- Library code never calls `print`; scripts pass `flush=True`.
- Comments explain *why*, not *what*.

## 5. Write DRY code

- **Search before you write.** Before adding a function, look for one that
  already does it: `grep` the package, the vendored `heteromorph` (it already
  has `structure_factor`, `pair_correlation`, `Ensemble`, `PeriodicCell`
  minimum-image and wrapping), and NumPy/SciPy. Reuse it; extend it if it
  falls short.
- **The second copy is the signal.** When you would copy logic a second time,
  factor it into one function in the module that owns that concept, and
  replace the first copy too.
- **Respect the two layers** (`docs/design.md` §5): data types own storage
  and validation; observables are pure functions of a realization returning
  `ObservableResult`. Do not reimplement aggregation, error bars, or
  parallel execution inside an observable — `Ensemble` and
  `TrajectoryEnsemble` do that once.
- **Shared test inputs** live in `tests/conftest.py` and
  `tests/strategies.py`, not in individual test files.
- **DRY is not speculation.** Do not build abstractions, options, or
  plug-in points for a second use that does not exist yet.
- **Change surgically.** Touch only what your task needs; match the
  surrounding style; do not refactor, reformat, or "improve" unrelated
  code. Remove only the dead code your own change created.

## 6. Tests are mandatory

No change is complete without tests that pass. Details, strategies, and the
analytic references are in `docs/dev/testing.md`.

1. **Test first.** Write the test, run it, see it fail for the expected
   reason, then write the code.
2. **Every numerical function** gets at least one Hypothesis `@given`
   property test (invariance, bounds, round trip, oracle) **and** at least
   one example test against an exact or independent reference: an analytic
   result (ideal gas, Gaussian chain, isotropic unit vectors, …), a
   brute-force loop, or a library routine.
3. **Log every defect the moment you find it** — including ones outside
   your task that you will not fix — as a new entry in `DEFECTS.md`.
   Mentioning it in your reply does not count.
4. **Every fix gets a regression test** named `test_regression_D###_<slug>`,
   seen to fail before the fix, plus a property test generalizing it with
   the repro pinned by `@example(...)`.
5. Mark tests that need a GPU, LAMMPS, or more than ~10 s with
   `@pytest.mark.gpu`, `@pytest.mark.lammps`, or `@pytest.mark.slow`.
6. Never weaken, skip, or delete a failing test to make a change pass. If a
   test is wrong, say so and log it as a defect.

## 7. How to work

- **Clarify before coding.** If the task, a physics convention, or an
  interface is ambiguous, stop and ask the person who dispatched you. Do
  not guess. State the assumptions you did make in your report and PR.
- **Code against the interfaces** in `docs/design.md` §5 and the WP0
  stubs. Changing a public signature needs the maintainer's approval.
- **Set verifiable goals.** For multi-step tasks, write a short plan where
  every step has a check (`→ verify: <test or command>`), and loop until
  each check passes.
- **One work package per branch** (`wp<N>-<slug>`), merged by pull request
  with CI green. Commit messages: imperative summary line; body says why.
  If an AI tool wrote the code, end the commit with that tool's
  `Co-Authored-By:` trailer.
- **Finish with a report** containing: what changed; the exact commands you
  ran and their results (tests, `ruff`, `mypy`); assumptions made; and every
  defect ID you logged, with its status, under **Defects logged**. Never
  claim a test passes without having run it.
