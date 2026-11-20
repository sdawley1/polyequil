<!-- One work package per PR. Keep the description short; link the WP or issue. -->

## What and why

<!-- What does this PR change, and why? Name the work package (e.g. WP5) or issue (#N). -->

## How it was verified

<!-- Paste the commands you ran and their results. Never claim a check passes without running it. -->

- [ ] `uv run ruff format --check . && uv run ruff check .`
- [ ] `uv run mypy`
- [ ] `uv run pytest`
- [ ] `uv run sphinx-build -W docs docs/_build/html` (if docs or docstrings changed)
- [ ] New numerical code has a Hypothesis property test and an exact-reference test (`docs/dev/testing.md`)

## Assumptions and physics choices to review

<!-- Conventions, estimators, or tolerances a reviewer should check. Write "none" if there are none. -->

## Defects logged

<!-- Every DEFECTS.md ID found or fixed in this PR, with its status (open / fixed / wontfix). Write "none" if there are none. -->

## Changelog

- [ ] Added an entry under `## [Unreleased]` in `CHANGELOG.md`
