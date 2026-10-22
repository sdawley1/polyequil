# polyequil
Polymer equilibration

Equilibration diagnostics for simulated polymer melts. See `docs/design.md`
for the science and architecture, and `AGENTS.md` for contributor rules.

```bash
uv sync && uv run pytest        # set up and run the test suite
```

```python
from pathlib import Path

from polyequil import ExecutionPolicy, TrajectoryEnsemble
from polyequil.observables.chain import radius_of_gyration

ensemble = TrajectoryEnsemble(sorted(Path("trajectories/ex").glob("*.h5")))
result = ensemble.evaluate(radius_of_gyration, execution=ExecutionPolicy("outer"))
result.times, result.mean, result.standard_error   # one row per time slice
```

LAMMPS output converts with
`polyequil.io.lammps.convert_lammps(data, dump, out, log=log)`.
