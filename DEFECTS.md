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
