# polyequil

Equilibration diagnostics for simulated polymer melts. `polyequil` computes
local diagnostics (density, pressure, $g(r)$, MSID, $R_g$) and two global
indices that test the equilibrium fluctuation–response relation, evaluated
over ensembles of independent trajectories with honest error bars.

## Installation

`polyequil` needs Python 3.11 or newer and is managed with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/sdawley1/polyequil.git
cd polyequil
uv sync                 # create the environment
uv run pytest           # run the fast test suite
```

## Quick start

```python
from pathlib import Path

from polyequil import ExecutionPolicy, TrajectoryEnsemble
from polyequil.observables.chain import radius_of_gyration

ensemble = TrajectoryEnsemble(sorted(Path("trajectories/ex").glob("*.h5")))
result = ensemble.evaluate(radius_of_gyration, execution=ExecutionPolicy("outer"))
result.times, result.mean, result.standard_error   # one row per time slice
```

## Building these docs

```bash
uv sync --group docs
uv run sphinx-build -W docs docs/_build/html
```

```{toctree}
:maxdepth: 2
:caption: Contents

api
design
dev/style
dev/testing
changelog
```
