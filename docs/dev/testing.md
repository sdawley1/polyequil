# Testing

Every change ships with tests that were run and pass. This guide says which
tests are required, how to write property tests, which exact references to
check against, and how defects are logged.

## 1. Required coverage

| Code under test | Required tests |
|---|---|
| New or changed numerical function or data type | ≥ 1 Hypothesis `@given` property test **and** ≥ 1 example test against an exact or independent reference |
| Defect fix | `test_regression_D###_<slug>`, seen failing before the fix, **plus** a property test generalizing it with the repro pinned by `@example(...)` |
| GPU kernel | CPU/GPU agreement test (`@pytest.mark.gpu`) on the same inputs, with a stated tolerance |
| I/O, LAMMPS input generation, SLURM glue | Example tests (round trips, golden snippets) |

Tests mirror the source layout: `src/polyequil/topology.py` →
`tests/test_topology.py`. Upstream `heteromorph` tests live under
`tests/vendor/` and must keep passing; do not edit them except to rewrite
imports during a re-sync.

## 2. Workflow

1. Write the test and run it: it must fail, for the reason you expect.
2. Write the minimum code that makes it pass.
3. Run the whole default suite, `ruff`, and `mypy`; all clean.
4. Report the commands and their output. Never report a test as passing
   without running it.

Never weaken a tolerance, add `skip`/`xfail`, or delete a test to get a green
run. A test that is itself wrong is a defect: log it (§6) and fix it in its
own commit.

## 3. Exact references

Prefer these over "looks right". Shared builders for each live in
`tests/conftest.py` and `tests/strategies.py` (§5); reuse them.

| ID | System | Exact result |
|---|---|---|
| T1 | Ideal gas (uniform random points) | $g(r)=1$, $S(q)=1$ within sampling error |
| T2 | Perfect lattice | Bragg peaks at reciprocal-lattice vectors, zero elsewhere |
| T3 | Ideal Gaussian chains | MSID $\langle R^2(n)\rangle = n b^2$; Debye form factor |
| T4 | Freely rotating chains | Closed-form MSID in $\langle\cos\theta\rangle$; correct $C_\infty$ |
| T5 | Independent isotropic unit vectors | $\langle\mathsf Q{:}\mathsf Q\rangle/N_b = 2/3$; $\langle P_2^2\rangle = 1/5$; $\langle\mathsf Q\rangle = 0$ |
| T6 | Synthetic $S(q)=S_0+aq^2$ with exponential noise | Extrapolation recovers $S_0$ within its reported interval at nominal coverage |
| T7 | AR(1) time series | Block averaging recovers the known standard error |
| T8 | Non-interacting rods in the aligning field | $\langle P_2\rangle = \beta h/5 + O(h^2)$, so the orientational index is 0 |

Other independent references: a brute-force double loop for anything
vectorized, the vendored `heteromorph` routines for $S(q)$ and $g(r)$, and
the CPU backend for any GPU kernel. Statistical tests (T1, T5, T6, T7) use a
fixed seed and a tolerance of several standard errors computed from the
sample size, never a hand-tuned constant.

## 4. Property tests with Hypothesis

A property test states a rule that holds for **every** valid input;
Hypothesis generates many inputs and shrinks failures to a minimal
counterexample. Pick every property that applies:

| Property | Example in this package |
|---|---|
| Oracle | vectorized pair histogram == double loop; GPU == CPU |
| Invariance | $S(q)$, $g(r)$, MSID, $R_g$ unchanged by translation, periodic image shifts, rigid rotation (scalar outputs), relabelling of chains |
| Conservation | bead count, bond count, density preserved by wrapping, unwrapping, artifact operators at $\lambda = 0$ |
| Bounds | $0 \le g(r)$; $S(q) \ge 0$; $-1/2 \le P_2 \le 1$; $R_g^2 \le$ max squared intra-chain distance |
| Round trip | wrap → unwrap; write → read (bit-exact); fractional → Cartesian → fractional |
| Idempotence | wrapping; making chains whole |
| Error contract | invalid input raises the documented exception |

Assert the property itself, never a value recomputed with the code under
test.

Rules:

- Constrain inputs with strategy arguments (`min_value`, `max_side`,
  `st.integers(2, 64)`), not `assume()` or `.filter()`.
- Bound magnitudes to the physical range and disable NaN, infinity, and
  subnormals in tests of the mathematics; test extreme and invalid inputs in
  separate error-contract tests.
- Compare floats with a scale-aware tolerance:
  `np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9 * scale)`.
- No hidden randomness inside a `@given` test: draw everything from
  strategies so failures replay. No `np.random.*` without a drawn seed.
- Do not commit `@reproduce_failure`; convert the shrunk input into an
  `@example(...)`.
- A `Flaky` error means the code is nondeterministic for a fixed input:
  that is a defect.
- For JIT or GPU code, set `@settings(deadline=None)`.

Profiles are registered once in `tests/conftest.py`:

```python
from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=100)
settings.register_profile("ci", max_examples=300, derandomize=True, print_blob=True)
settings.register_profile(
    "thorough", max_examples=5000, deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dev")
```

## 5. Shared strategies and fixtures

`tests/strategies.py` owns the reusable Hypothesis strategies (periodic
cells, bead positions, chain topologies of random lengths, unit vectors,
chain conformations); `tests/conftest.py` owns fixtures and the analytic
builders T1–T8. Add to these rather than defining a strategy inside a test
file. Example:

```python
import numpy as np
from hypothesis import example, given

from polyequil.observables import msid
from tests.conftest import TWO_BEAD_CHAIN_ACROSS_BOUNDARY
from tests.strategies import chain_configurations, lattice_shifts


@given(chain_configurations(), lattice_shifts())
@example(*TWO_BEAD_CHAIN_ACROSS_BOUNDARY)  # D-004 repro
def test_msid_invariant_under_image_shifts(config, shift):
    """MSID does not depend on which periodic image a bead is stored in."""
    reference = msid(config, separations=[1]).values
    shifted = msid(config.with_image_shift(shift), separations=[1]).values
    np.testing.assert_allclose(shifted, reference, rtol=1e-12)
```

## 6. Defect log

A defect is any behaviour that contradicts a docstring, a comment, the design
doc, the stated intent, or the physics: wrong results, crashes, silent NaN,
off-by-one errors, unit or convention mismatches, unchecked edge cases.

Log it in `DEFECTS.md` **when you find it**, before fixing it or continuing
your task, including defects outside your task that you will leave alone.
IDs are sequential and never reused; entries are updated, never deleted.

```markdown
## D-004 — msid double-counts pairs across the periodic boundary
- **Found:** YYYY-MM-DD, while working on <task / WP>
- **Status:** open | fixed | wontfix
- **Where:** src/polyequil/observables/chain.py:57 (`msid`)
- **Symptom:** <what is wrong, quantitatively>
- **Repro:** <smallest call and its wrong vs expected output>
- **Impact:** <which results or functions are affected>
- **Regression test:** tests/test_chain.py::test_regression_D004_boundary_pairs
- **Fix:** <commit hash or one-line description>
```

Regression tests assert on the exact repro input and at least one nearby
edge of the same failure, compare to an independent reference with an
explicit tolerance, and cite the defect ID in the docstring. A defect in
`src/polyequil/_vendor/` is logged here and reported to the maintainer; it is
not fixed in place.

List every defect ID you logged, with its status, under **Defects logged** in
your final report.
