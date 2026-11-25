# Changelog

All notable changes to `polyequil` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Every pull request
adds a line under **Unreleased**; a release renames that section to the new
version and date.

## [Unreleased]

## [0.1.0] - 2026-11-25

First tagged version: the data layer, the structure-factor diagnostics and the
equilibrium indices, with documentation and CI.

### Added

- Trajectory storage and loading (WP0–WP3, #1): orthorhombic `Box`,
  `ChainTopology` and `ChainConfiguration`, an HDF5 store (schema
  `polyequil-1`) with a streaming writer, LAMMPS data/dump/log readers and
  `convert_lammps`, a lazy `Trajectory` with `FrameRef` and `TrajectoryWindow`.
- `TrajectoryEnsemble`, which reduces each time slice across independent
  replicas only, with serial and process-pool execution that give bit-identical
  results (#1).
- `radius_of_gyration`, the first observable (#1).
- Density and orientational structure factors, the small-k extrapolation of
  S(k→0), `window_mean`, and the density and orientational equilibrium indices
  with first-order error propagation (WP5 core, #2).
- Sphinx documentation site with an API reference generated from the
  docstrings, built in CI with warnings as errors (WP11, #3).
- Coverage report in CI (fails below 85% branch coverage); README examples run
  as doctests (#4).
- Runnable `Examples` sections in 19 public docstrings, executed as doctests in
  the default suite and in CI, and a test that fails if a public docstring with
  `Parameters`/`Returns`/`Raises` sections has no example (WP11, #6).
- Pull request and issue templates, and this changelog.

### Changed

- Stricter pytest configuration: `-ra`, `--strict-config`, `--strict-markers`,
  `xfail_strict` and warnings as errors (#4).
- Docstring standard: multi-section docstrings need an `Examples` section, and
  `Raises` lists every error a caller can trigger, including errors from
  callees (#6).

### Fixed

- D-001: serial execution now caps native BLAS threads, so serial and parallel
  results agree bit for bit (#2).
- D-002: a property test drew subnormal relative errors and failed spuriously
  (#4).
- D-003: the WP5 observables' docstrings now list every error they raise (#6).

[Unreleased]: https://github.com/sdawley1/polyequil/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sdawley1/polyequil/releases/tag/v0.1.0
