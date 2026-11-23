# Defect log

Entry format and rules: `docs/dev/testing.md` §6. IDs are sequential (D-001, D-002, ...) and never reused; entries are updated, never deleted.

## D-001 — serial execution does not limit native threads, so serial and outer results can differ
- **Found:** 2026-10-24, while working on WP5 (structure factors)
- **Status:** fixed
- **Where:** src/polyequil/execution.py (`_execute_serially`)
- **Symptom:** `ExecutionPolicy` resolves `threads_per_worker = 1` in both modes, but only pooled workers apply it (threadpoolctl initializer). Serial mode runs observables with the parent's full BLAS pool (64 OpenBLAS threads on a Della login node), so any observable that calls BLAS (e.g. a complex matrix product) sums in a different order: `orientational_structure_factor` means differ by 5.6e-17 between serial and outer, breaking the documented bit-identity. It also oversubscribes the CPU budget the policy claims to enforce.
- **Repro:** `TrajectoryEnsemble(trajectories/ex/*.h5).evaluate(orientational_structure_factor, times=[0, 5000], k_max=1.5)` in serial vs `ExecutionPolicy("outer", total_cpus=2)`: `np.array_equal(a.mean, b.mean)` is False (expected True).
- **Impact:** every serial evaluation of a BLAS-using observable; reproducibility claims of WP3. Results are correct to rounding; no physics affected. heteromorph has the same gap (its `_execute_serially` also applies no thread limit).
- **Regression test:** tests/test_execution.py::test_regression_D001_serial_mode_limits_native_threads
- **Fix:** run the serial loop inside `threadpool_limits(policy.threads_per_worker)`.

## D-002 — density-index property test draws subnormal relative errors
- **Found:** 2026-11-16, while working on CI/test hardening (strict pytest config, coverage)
- **Status:** fixed
- **Where:** tests/test_indices.py (`test_density_index_round_trips_and_propagates_errors`)
- **Symptom:** the relative-error strategy `st.floats(0, 1)` allows subnormals, contrary to `docs/dev/testing.md` §4. At subnormal magnitudes the propagated error loses precision, so the `rtol=1e-12` check fails with a relative difference of 4.6e-12 (absolute difference 5e-324).
- **Repro:** `density_index(Estimate(1.0, 2.225073858507e-311), density=1.0, temperature=1.0, kappa_t=Estimate(29.0, 29.0 * 2.225073858507e-311))`: error 1.085079e-312 vs expected 1.085079e-312 to rtol 1e-12 (fails by 4.6e-12 relative).
- **Impact:** test only; flaky CI failure whenever Hypothesis reaches the subnormal range. `density_index` is correct; no physics affected.
- **Regression test:** tests/test_indices.py::test_density_index_round_trips_and_propagates_errors (strategy fixed; nearest in-domain edge pinned with `@example`)
- **Fix:** relative errors drawn from `{0} ∪ [1e-100, 1]`, which keeps every intermediate product normal.

## D-003 — WP5 observable docstrings omit errors their callers can trigger
- **Found:** 2026-11-23, while adding Examples sections to docstrings (WP11)
- **Status:** fixed
- **Where:** src/polyequil/observables/extrapolation.py (`small_k_limit`, `fit_small_k`), src/polyequil/observables/structure.py (`structure_factor`, `orientational_structure_factor`), src/polyequil/observables/window.py (`window_mean`), src/polyequil/observables/chain.py (`radius_of_gyration`), src/polyequil/reciprocal.py (`collective_amplitudes`), src/polyequil/indices.py (`density_index`)
- **Symptom:** `docs/dev/style.md` §4 requires a `Raises` section in every multi-section docstring. These functions raise `ValidationError` (directly or through `reciprocal_shells`, `segments`, `_check_threads`, `_ratio_index`, `frozen_array`) but have no `Raises` section, or one that omits a condition (`fit_small_k`: non-positive shell counts; `window_mean`: a `window` that is not `WindowData`).
- **Repro:** `structure_factor(config, k_max=-1.0)` raises `ValidationError`, yet `structure_factor.__doc__` has no `Raises` section.
- **Impact:** documentation only; behaviour is correct. Readers of the API reference cannot tell which inputs are rejected.
- **Regression test:** tests/test_docstrings.py::test_regression_D003_wp5_docstrings_document_raises; tests/test_structure.py::test_structure_factor_rejects_every_non_positive_or_non_finite_k_max (property, repro pinned by `@example`); each documented condition is exercised by the `test_documented_errors_are_raised` tests in test_structure.py, test_extrapolation.py, test_window.py, and test_reciprocal.py
- **Fix:** `Raises` sections added or completed, listing every condition a caller can trigger, including those raised by callees.

## D-004 — other multi-section docstrings omit a Raises section
- **Found:** 2026-11-23, while adding Examples sections to docstrings (WP11)
- **Status:** open
- **Where:** src/polyequil/topology.py (`ChainTopology.make_whole`), src/polyequil/io/lammps.py (`iter_dump`), src/polyequil/ensemble.py (`TrajectoryEnsemble`), src/polyequil/io/hdf5.py (`TrajectoryWriter`)
- **Symptom:** same as D-003, outside the WP5 observable modules: `make_whole` raises `ValidationError` for misshapen positions (via `frozen_array`), `iter_dump` raises `FormatError` for `dt <= 0` or a malformed frame, and the two class constructors raise `ValidationError`; none of these docstrings has a `Raises` section.
- **Repro:** `iter_dump(path, topology, dt=0.0)` raises `FormatError` on first iteration; `iter_dump.__doc__` has no `Raises` section.
- **Impact:** documentation only; behaviour is correct.
- **Regression test:** (none yet)
- **Fix:** (none yet; out of scope for the D-003 change)

## D-005 — collective_amplitudes validates `modes` with an assert
- **Found:** 2026-11-23, while documenting the errors of `collective_amplitudes` (WP11)
- **Status:** open
- **Where:** src/polyequil/reciprocal.py (`collective_amplitudes`, the `assert mode_array.ndim == 2 ...` line)
- **Symptom:** a public input is checked with `assert`, contrary to AGENTS.md §3 rule 5. A misshapen `modes` raises a bare `AssertionError` instead of `ValidationError`; under `python -O` the check disappears and the call fails later with an unrelated error.
- **Repro:** `collective_amplitudes(np.zeros((2, 3)), None, Box(np.zeros(3), np.full(3, 5.0)), np.array([[1, 0]]))` → `AssertionError` (with `-O`: `IndexError: list index out of range`); expected `ValidationError` naming the shape.
- **Impact:** error reporting only; well-formed calls (every call inside the package passes `ReciprocalShells.modes`) are unaffected.
- **Regression test:** (none yet)
- **Fix:** (none yet)

## D-006 — radius_of_gyration accepts non-integer `threads`
- **Found:** 2026-11-23, while documenting the errors of `radius_of_gyration` (WP11)
- **Status:** open
- **Where:** src/polyequil/observables/chain.py (`radius_of_gyration`)
- **Symptom:** `threads` is checked with `threads < 1` only, unlike the structure observables' `_check_threads`: `threads=1.5` and `threads=True` are accepted, and `threads="2"` raises a bare `TypeError` instead of `ValidationError`.
- **Repro:** `radius_of_gyration(config, threads=1.5)` returns a result; `radius_of_gyration(config, threads="2")` → `TypeError: '<' not supported between instances of 'str' and 'int'`. Expected `ValidationError` in all three cases.
- **Impact:** error reporting only; the value of `threads` does not affect the result (single-threaded).
- **Regression test:** (none yet)
- **Fix:** (none yet)
