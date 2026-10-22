# Example Kremer–Grest trajectories

Three independent replicas of a flexible Kremer–Grest bead–spring melt,
frame-subsampled from a larger ensemble. They are the reference data for
designing and testing the `polyequil` I/O, trajectory, and observable code.
**Read-only:** do not modify, regenerate, or add files here; ask the
maintainer.

| File | Replica | Root seed |
|---|---|---|
| `kg_N20_k0_rep000.h5` | 0 | 20261022 |
| `kg_N20_k0_rep001.h5` | 1 | 20261023 |
| `kg_N20_k0_rep002.h5` | 2 | 20261024 |

## System

- 100 chains × 20 beads = 2000 beads; flexible ($k_\theta = 0$).
- $\rho = 0.85\,\sigma^{-3}$, $T = 1$, cubic periodic box $L = 13.3006\,\sigma$, NVT
  (Langevin, damping time 2), LJ units.
- FENE ($k = 30$, $R_0 = 1.5$) + WCA, `special_bonds fene`; production
  timestep 0.01 τ.
- Started from random walks with a crude soft push-off; t = 0 is the end of
  the push-off, so early frames are **out of equilibrium**.
- LAMMPS 22 Jul 2025 (Update 4). The exact input script is stored in each
  file's `input_script` attribute.

## Frames

70 frames per replica, on a common, **non-uniform** time grid:
t = 0, 5, …, 95 τ (the relaxation) and t = 100, 200, …, 5000 τ. Thermo data
are complete: every 1 τ from 0 to 5000 τ (5001 rows).

What the data show (from the full 8-replica ensemble): pressure is
stationary from t = 0, while the end-to-end MSID $\langle R^2(19)\rangle/19$
relaxes from about 1.66 to a plateau near 1.52 within roughly 1000 τ. Frames
with $t \gtrsim 1000\,\tau$ may be treated as equilibrated for N = 20.

## Schema (`schema = "kg_template-1"`)

Indices are zero-based; n = 2000 beads, F = 70 frames, T = 5001 rows.

| Path | Shape, dtype | Meaning |
|---|---|---|
| `/` attrs | — | `n_beads_per_chain`, `n_chains`, `density`, `temperature`, `k_theta`, `dt`, `npush_steps`, `dump_every_steps`, `thermo_every_steps`, seeds, `lammps_version`, `units`, conventions, `input_script`, `subsampled_from`, `frame_selection` |
| `/topology/bonds` | (1900, 2) int64 | bonded bead pairs |
| `/topology/angles` | (1800, 3) int64 | bead triples; the middle bead is the vertex |
| `/topology/mol` | (n,) int64 | chain index per bead; chains are contiguous |
| `/topology/type` | (n,) int64 | atom type (all 1) |
| `/frames/step`, `/frames/time` | (F,) int64, float64 | timestep; time in τ |
| `/frames/box_lo`, `/frames/box_hi` | (F, 3) float64 | orthogonal box bounds |
| `/frames/positions` | (F, n, 3) float64 | **wrapped** positions (12 significant digits) |
| `/frames/images` | (F, n, 3) int32 | image flags: unwrapped = positions + images × (box_hi − box_lo) |
| `/frames/velocities` | (F, n, 3) float64 | velocities |
| `/thermo/<name>` | (T,) | step, time, temp, press, pxx, pyy, pzz, pxy, pxz, pyz, poteng, kineng, e_bond, e_angle, density; energies per atom |

```python
import h5py

with h5py.File("trajectories/ex/kg_N20_k0_rep000.h5", "r") as f:
    L = f["frames/box_hi"][-1] - f["frames/box_lo"][-1]
    x = f["frames/positions"][-1] + f["frames/images"][-1] * L   # (2000, 3), unwrapped
    chains = x.reshape(f.attrs["n_chains"], -1, 3)                # (100, 20, 3)
```

## Provenance

Generated outside this repository in
`/scratch/gpfs/TORQUATO/sd2402/polymer_equilibration/kg_template/`
(`in.kg`, `build_melt.py`, `run_ensemble.slurm`, `to_hdf5.py`; SLURM array
16005792, 2026-10-22). These three files were produced from the full
1001-frame replicas by `make_examples.py` there, and verified equal to the
source on every kept frame and on all topology and thermo data.
