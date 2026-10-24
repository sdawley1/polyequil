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

Structure factors and the equilibrium indices (κ_T and χ come from response
runs):

```python
from polyequil.indices import Estimate, density_index, estimate_from
from polyequil.observables.extrapolation import small_k_limit
from polyequil.observables.structure import structure_factor

s0 = ensemble.evaluate_windows(
    small_k_limit, windows=[(4000.0, 5000.0)],
    frame_observable=structure_factor, k_fit=1.5,
)
x_rho = density_index(estimate_from(s0, (0,)), density=0.85, temperature=1.0,
                      kappa_t=Estimate(kappa, kappa_error))
```

LAMMPS output converts with
`polyequil.io.lammps.convert_lammps(data, dump, out, log=log)`.
