# Global Structural Correlations for Diagnosing Equilibration of Simulated Polymer Melts

*APC 524 Final Project — revised proposal*
Andrea Aude, Sam Dawley, Hong-Ye Wang — Fall 2026

Items marked **[PILOT]** are numbers the first simulations must fix. Settled decisions are recorded in Section 7.3. Section 9 lists what changed relative to `proposal.md` and why.

---

## 1. Core idea

Equilibration of a polymer melt is normally declared when thermodynamic observables ($T$, $P$, $\rho$) are stationary and single-chain statistics (MSID, $R_g$) and $g(r)$ have converged. These are *stationarity* tests of mostly *local* quantities. They cannot distinguish an equilibrium state from a long-lived nonequilibrium one.

Equilibrium imposes a stronger, falsifiable condition: the **static fluctuation–response relation**. For an observable $A$ with conjugate field $h$,

$$\langle \delta A^2 \rangle_{h=0} = k_B T \,\frac{\partial \langle A \rangle}{\partial h}\Big|_{h=0}.$$

The two sides are measured independently (one from spontaneous fluctuations, one from a weak applied perturbation). A system for which they disagree is out of equilibrium; agreement is necessary but not sufficient. We apply this to two collective variables:

| Channel | Fluctuation side | Response side | Index |
|---|---|---|---|
| Density | $S(q\to0)$, monomer structure factor | $\kappa_T = \rho^{-1}(\partial\rho/\partial P)_T$ | $X_\rho = S(0)/(\rho k_B T \kappa_T) - 1$ |
| Bond orientation | $S_Q(0) = \langle (\sum_b P_2(\hat u_b\cdot\hat n))^2\rangle / N_b$ | $\chi = \partial\langle P_2\rangle/\partial h$ under $H' = -h\sum_b P_2(\hat u_b\cdot\hat n)$ | $X_Q = S_Q(0)/(k_B T\chi) - 1$ |

$X_\rho$ is the nonequilibrium index of Hopkins, Stillinger, and Torquato (2012). $X_Q$ is its orientational analogue and is, to our knowledge, new as an equilibration diagnostic for melts. Both vanish in equilibrium.

**Deliverable.** A tested, documented Python package (working name `polyequil`) that computes local diagnostics (MSID, $R_g$, $g(r)$, $P$, $\rho$) and the two global indices on single trajectories and on ensembles of independent trajectories, with chain-aware topology and CPU/GPU kernels; plus a small reference dataset and a study answering the hypotheses below.

## 2. Hypotheses

- **H1 (ordering of convergence).** After preparation, diagnostics converge in the order $\rho, P \to g(r) \to X_\rho \to$ short-$n$ MSID $\to X_Q \to$ long-$n$ MSID. In particular, "cheap" diagnostics are satisfied while chain-scale structure is not.
- **H2 (stiffness).** The sensitivity of $X_Q$ to a preparation artifact of fixed amplitude increases with chain stiffness; that of $X_\rho$ does not.
- **H3 (density channel is nearly blind in the melt).** Well above $T_g$, long-wavelength density relaxes far faster than chain conformations, so $X_\rho \approx 0$ within error almost immediately. $X_\rho$ becomes informative only near $T_g$.

H3 may well hold. That is a reportable result, not a failure, and is why the orientational channel carries the project.

**Sensitivity** is defined operationally: the minimum artifact amplitude $\lambda^*$ at which a diagnostic differs from the equilibrium reference by more than $z=3$ standard errors, as a function of the number of replicas $M$ and the time $t$ since preparation. A $\lambda=0$ control fixes the false-positive rate.

## 3. Definitions and estimators

These are the points the draft left undefined; each is a place an implementation could silently go wrong.

1. **$S(q\to0)$ cannot be read off at $q=0$.** At fixed $N$, $S(0)\equiv0$. Compute $S(\mathbf q)$ on reciprocal-lattice vectors, then extrapolate with $S(q)=S_0+a q^2$ over $q<q_{\max}$ **[PILOT]**. Report $S_0$ with the fit uncertainty and the sensitivity to $q_{\max}$.
2. **Statistical cost.** A single mode's $|\rho_{\mathbf q}|^2/N$ has relative standard deviation $\approx 1$. The lowest shell has 3 independent modes, so one frame from $M=16$ replicas gives $\sim 14\%$ at $q_{\min}$. Time-resolved $X_\rho(t)$ therefore needs a window average; the window length versus time resolution trade-off must be reported, not hidden.
3. **$\kappa_T$ is measured as a response**: finite difference of $P(\rho)$ from short runs at $\rho(1\pm\epsilon)$ at fixed $T$. NPT volume fluctuations, $\kappa_T=\langle\delta V^2\rangle/(k_BT\langle V\rangle)$, are a cross-check only (they are themselves a fluctuation estimate). Thermostat and barostat must sample the correct ensemble (Nosé–Hoover/MTK or Langevin; never Berendsen).
4. **Orientation is not conserved**, so $S_Q$ is evaluated directly at $q=0$ with no extrapolation. In the isotropic phase all five independent components of $\mathsf Q=\sum_b(\hat u_b\hat u_b-\mathsf I/3)$ are equivalent and $\langle(\sum_b P_2)^2\rangle = \tfrac{3}{10}\langle\mathsf Q{:}\mathsf Q\rangle$; use the full contraction for a fivefold gain in statistics. A nonzero mean $\langle\mathsf Q\rangle$ or unequal components is itself a (cheaper) anisotropy diagnostic and is always reported alongside $X_Q$.
5. **Segment scale.** $\hat u_b$ is the unit vector spanning $n_s$ consecutive bonds; $n_s=1$ and $n_s\approx$ one Kuhn length are both computed.
6. **The draft's orientational relation is inconsistent** (one version carries a factor $\rho$, the struck-out version does not). With per-bond normalization, as above, there is no $\rho$.
7. **$\chi$ must be in the linear regime**: at least three field strengths, with the fitted slope and a nonlinearity check.
8. **$\chi$ by response is only measurable where the response can equilibrate.** Orientational relaxation of an entangled chain is as slow as the equilibration we are trying to diagnose. So $\chi$ is measured by response for brute-force-equilibrated chain lengths only; for entangled melts the comparison is against $S_Q(0)$ of the reference melt and against $\chi(N)$ extrapolated from the short-chain series.
9. **Stay in the isotropic phase.** Stiff bead–spring chains at melt density order nematically, where $\chi$ diverges and the isotropic relation fails. The largest stiffness studied must sit below that threshold **[PILOT]**.
10. **Unwrapped coordinates.** MSID and $R_g$ need chains made whole. Store wrapped positions plus integer image flags; where flags are missing, rebuild along bonds by minimum image (valid when every bond is shorter than $L/2$).

## 4. Study design

**Model.** Kremer–Grest bead–spring melt (FENE + WCA) with a bending potential $U=k_\theta(1+\cos\theta)$ in LJ units, at the standard state point $\rho=0.85$, $T=1$. An additional near-$T_g$ state point with attractive LJ tests H3 **[PILOT]**.

| Factor | Levels (defaults) |
|---|---|
| Stiffness $k_\theta$ | 3 values from flexible to the stiffest isotropic melt **[PILOT]** |
| Chain length $N$ | short (brute-force, a few Kuhn lengths), intermediate (still brute-force; anchors the $\chi(N)$ extrapolation), entangled ($N\gg N_e$) |
| System size | $L\gtrsim3R_g$ of the longest chain; $\sim5\times10^4$ beads **[PILOT]** |
| Replicas $M$ | 16 independent preparations per cell |
| Preparation | (a) brute-force MD (short chains); (b) GAMERS (Metcalfe et al.); (c) 21-step compression/decompression (Abbott et al.); (d) deliberately poor fast push-off as a known-bad control (Auhl et al.) |
| Artifact amplitude $\lambda$ | 0 (control) plus 3–4 levels |

**Artifact operators** (applied to an equilibrated reference, followed by a short local relaxation to remove overlaps):

- *Alignment:* volume-preserving uniaxial affine deformation of chain conformations by stretch $\lambda$ at fixed density.
- *Local stretching:* regenerate chains with a biased bond-angle distribution (wrong $C_\infty$), reproducing the classic MSID distortion.

**Reference data.** The entangled reference trajectories do not exist yet; Sam will generate them later. Nothing before WP10 may depend on them: development and testing use synthetic fixtures (Section 7.1) and short brute-force melts produced by WP7. The literature provides MSID and $R_g$ targets but **no** published $X_Q$; the equilibrium baseline for $X_Q$ is generated in this work.

**Out of scope.** Chemically specific DVOCB($n$) systems are excluded. Viscoelastic/Green–Kubo properties and concurrent CPU-simulation/GPU-analysis scheduling are stretch goals (Section 8).

## 5. Software architecture

### 5.1 What `heteromorph.Ensemble` provides and where it falls short

Audited at `heteromorph` v0.9.0 (commit `3abe3ca`).

| Property of `Ensemble` | Consequence here |
|---|---|
| Holds `Microstructure` realizations; the only abstract member is `domain` | A polymer frame type is admissible with no change to `Ensemble` |
| `evaluate(observable, execution=..., retain_samples=..., random_seed=...)` returns mean, sample standard deviation, and standard error; checks per-realization compatibility; records provenance | Reuse as is |
| Uncertainty assumes **independent realizations** | Correct across replicas, **wrong across frames of one trajectory**. An `Ensemble` must never be built from frames of the same run |
| No time axis | Need a trajectory type and a time-resolved evaluation |
| Every realization held in memory; no lazy source | $16\times10^3$ frames $\times5\times10^4$ beads $\approx 19$ GB in float64. Need lazy, picklable frame handles |
| Observable contract `f(realization, *, threads, **params) -> ObservableResult` | Fits single-frame observables; window observables (fluctuations, MSD) need a window realization |
| `ExecutionPolicy`: only `serial` and `outer` (process pool) are implemented | Fine for CPU. GPU observables must run `serial` to avoid device contention |
| `PointConfiguration(cell, positions, radii, species)`; no topology, images, or velocities | Subclass it |
| `structure_factor`, `pair_correlation` exist for `PointConfiguration` | Reuse for monomer $S(q)$ and $g(r)$; they double as CPU oracles for GPU kernels |
| No license selected upstream | Sam maintains both repositories; this repo is MIT-licensed |

**Vendoring (decided).** `heteromorph` is not a dependency. The needed modules are copied **verbatim** into `src/polyequil/_vendor/heteromorph/`; all their internal imports are relative, so they work at that path unmodified.

| Tier | Modules | Lines | Adds |
|---|---|---|---|
| Core (ensemble machinery) | `ensemble`, `accumulators`, `execution`, `observables`, `_metadata`, `errors`, `microstructure`, `domain`, `cell`, `_version` | ≈3.1k | NumPy |
| Points (base class, $S(q)$, $g(r)$) | `configuration`, `point_patterns`, `_radial`, `_number_windows` | ≈2.6k | SciPy |

Rules: (i) vendored files are never edited — every adaptation lives in `polyequil` modules that import from `_vendor`, so a re-sync is a file copy; (ii) `_vendor/heteromorph/VENDORED.md` records the upstream commit (`3abe3ca`) and file list; (iii) the upstream tests for these modules are copied alongside with imports rewritten and must pass in CI; (iv) lint and type checks exclude `_vendor`; (v) a defect found in vendored code is fixed upstream first, then re-synced.

### 5.2 New types (object-oriented layer)

```
ChainTopology            immutable: bonds, angles, chain_id per bead, bead index within chain
                         pairs_at_separation(n) -> (i, j) index arrays
ChainConfiguration       PointConfiguration + topology, images, velocities?, step, time
                         unwrapped() -> positions with chains made whole
FrameRef                 lazy, picklable handle (file, frame index, cell metadata); .load()
Trajectory               time-ordered frames of ONE run sharing one topology; lazy; .window(t0, t1)
TrajectoryWindow         a Microstructure wrapping a contiguous span of one trajectory
TrajectoryEnsemble       M independent trajectories on a common time grid
    .at(t)        -> heteromorph.Ensemble of the M frames at time t
    .evaluate(observable, times=..., window=...) -> TimeResolvedResult
TimeResolvedResult       EnsembleResult per time, stacked along a leading time axis
```

Notes for implementers: `PointConfiguration` is a frozen `slots=True` dataclass whose trailing fields have defaults, so new fields on the subclass must also default and be validated in `__post_init__`. Every ABC in the hierarchy declares `__slots__ = ()`. `TrajectoryEnsemble.evaluate` builds one `heteromorph.Ensemble` per time slice, so cross-replica error bars inherit heteromorph's validation unchanged.

### 5.3 Observables (functional layer)

Pure functions with the heteromorph contract, returning `ObservableResult` with a `convention` that records normalization and estimator.

- Frame: `density`, `pair_correlation`, `structure_factor` (wrappers), `msid`, `radius_of_gyration`, `q_tensor`, `bond_angle_distribution`.
- Window: `s0_extrapolated`, `q_fluctuation`, `block_average`, `msd` (stretch).
- Derived (act on results, not frames): `kappa_T_from_response`, `chi_from_response`, `nonequilibrium_index`, `orientational_index`, `detection_threshold`.

### 5.4 Backends, I/O, simulation

- **Backends.** One array-namespace switch (`numpy` | `cupy`) for three kernels: direct $S(\mathbf q)$ sum, pair histogram, $\mathsf Q$ accumulation. GPU results must match CPU to a stated tolerance. GitHub Actions has no GPU: GPU tests carry a marker and run on Della by a documented manual command.
- **I/O.** Readers for LAMMPS data files (topology) and dump files (frames); canonical on-disk store is chunked HDF5, one file per trajectory, with frame-level random access.
- **Simulation.** LAMMPS input generators for the Section 4 matrix and SLURM submission for Della. The aligning field has no stock LAMMPS equivalent: implement a small C++ `fix` (the project's compiled-language component), prototyped and cross-checked against a Python `fix external` callback.
- **Tooling.** `uv`, `src/` layout, Python $\geq3.11$, NumPy $\geq2$, SciPy $\geq1.13$ (the vendored code's floors); `ruff`, `mypy --strict`, `pytest` + Hypothesis; MkDocs; GitHub Actions for lint, types, CPU tests, docs build.

## 6. Work packages

Written to be dispatched to agents. Each lists dependencies and an acceptance test. WP0 fixes the interfaces all others code against, so it lands first and alone.

| WP | Scope | Depends on | Acceptance |
|---|---|---|---|
| **0 Scaffold + interfaces** | Repo, `uv` project, CI, vendored heteromorph (§5.1) with its upstream tests, type stubs for every class and function in 5.2–5.3 | — | Vendored tests and CI green on empty stubs; `_vendor` files byte-identical to upstream `3abe3ca`; interface doc reviewed by all three authors |
| **1 Topology + configuration** | `ChainTopology`, `ChainConfiguration`, unwrapping | 0 | Property tests: unwrap∘wrap identity; `pairs_at_separation` count $=\sum_c (N_c-n)$; chains whole across boundaries |
| **2 I/O + storage** | LAMMPS readers, HDF5 store, `FrameRef`, `Trajectory` | 1 | Round-trip is bit-exact; a 10 GB trajectory is opened and one frame read in bounded memory |
| **3 Trajectory ensemble** | `TrajectoryWindow`, `TrajectoryEnsemble`, `TimeResolvedResult` | 1, 2 | Matches a hand-rolled NumPy mean/SE on synthetic data; refuses mismatched time grids; refuses same-run frames in one ensemble |
| **4 Local observables** | MSID, $R_g$, $g(r)$, density, angle distribution | 1 | Analytic tests T1–T4 (Section 7.1) |
| **5 Global observables** | $S(q)$ + extrapolation, $\mathsf Q$ fluctuation, indices, block averaging | 1, 3 | Analytic tests T5–T8 |
| **6 GPU backend** | CuPy kernels behind the namespace switch | 4, 5 | CPU/GPU agreement to tolerance; benchmark table (speedup versus $N$, $N_q$) |
| **7 Simulation layer** | LAMMPS generators, C++ aligning-field `fix`, SLURM templates, $\kappa_T$ and $\chi$ protocols | 0 | Test T8 end-to-end in LAMMPS; C++ fix matches Python callback forces to round-off; energy conserved in NVE with field on |
| **8 Artifacts + detection** | Artifact operators, $\lambda^*$ estimator | 1, 5 | $\lambda=0$ leaves configuration unchanged; density preserved to round-off; false-positive rate at $\lambda=0$ consistent with $z$ |
| **9 Pilot** | Fix all **[PILOT]** values; measure decorrelation times; power analysis for $M$ and window length | 4, 5, 7 | Short report with chosen parameters and predicted error bars |
| **10 Production + analysis** | Run the matrix; evaluate H1–H3 | 9, 8, reference trajectories | Figures: diagnostics versus $t$; $\lambda^*$ versus stiffness; packaged reference dataset |
| **11 Docs + release** | MkDocs site, tutorial notebook, reference-data loader | all | Fresh-clone install and tutorial run from the docs alone |

WPs 1–2, 4, and 7 can start in parallel once WP0 merges; 3 and 5 follow; 6 and 8 follow those. Suggested ownership: Sam — 0, 3, 5, 8, reference trajectories; Andrea — 7, 9, 10; Hong-Ye — 2, 6, CI/docs; 1 and 4 shared. WPs 0–9 and 11 do not need the reference trajectories.

## 7. Validation

### 7.1 Analytic tests

| ID | System | Exact result |
|---|---|---|
| T1 | Ideal gas (Poisson points) | $g(r)=1$, $S(q)=1$ |
| T2 | Perfect lattice | Bragg peaks at reciprocal-lattice vectors, zero elsewhere |
| T3 | Ideal Gaussian chains | MSID $\langle R^2(n)\rangle = n b^2$; Debye form factor |
| T4 | Freely rotating chains | Closed-form MSID in $\langle\cos\theta\rangle$; correct $C_\infty$ |
| T5 | Independent isotropic unit vectors | $\langle\mathsf Q{:}\mathsf Q\rangle/N_b = 2/3$; $\langle P_2^2\rangle = 1/5$; $\langle\mathsf Q\rangle=0$ |
| T6 | Synthetic $S(q)=S_0+aq^2$ data with exponential noise | Extrapolation recovers $S_0$ within the reported interval at nominal coverage |
| T7 | Correlated AR(1) series | Block averaging recovers the known standard error |
| T8 | Non-interacting rods in the aligning field | $\langle P_2\rangle = \beta h/5 + O(h^2)$, hence $X_Q=0$ |

T8 is the end-to-end proof that the fluctuation and response pipelines agree before either is trusted on a melt. The analogous density check is a dilute WCA fluid against its virial equation of state.

### 7.2 Process

Every defect is logged in `DEFECTS.md` when found, and every fix lands with a regression test that failed first. Observables carry Hypothesis property tests for the invariances they must respect (translation, periodic image shifts, bead relabelling within the topology, rotation for scalar outputs).

### 7.3 Decisions

| # | Decision | Status |
|---|---|---|
| D1 | How to consume heteromorph | **Settled:** vendor verbatim (§5.1); Sam maintains both repositories |
| D2 | DVOCB($n$) systems | **Settled:** out of scope |
| D3 | Reference trajectories | **Settled:** do not exist yet; Sam generates them later; only WP10 depends on them |
| D4 | Schedule | **Settled:** work packages are ordered by dependency, not dated |
| D5 | Package and repo name | **Settled:** `polyequil` |
| D6 | Trajectory store | Default HDF5 (`h5py`) |
| D7 | License for this repo | **Settled:** MIT (`LICENSE`) |

## 8. Risks and stretch goals

| Risk | Mitigation |
|---|---|
| $X_\rho$ is null everywhere in the melt (H3) | Stated as a hypothesis; near-$T_g$ state point included; $X_Q$ is the primary channel |
| Low-$q$ noise swamps $X_\rho(t)$ | Power analysis in WP9 sets $M$ and window before production |
| $\chi$ unobtainable for entangled chains | Short-chain series + extrapolation; reference-melt $S_Q(0)$ as baseline (§3 item 8) |
| Stiff melts order nematically | Stiffness ceiling fixed in the pilot |
| Vendored copy drifts from upstream | Verbatim-only rule, recorded commit, upstream tests in CI (§5.1) |
| Reference trajectories arrive late | Only WP10 depends on them; everything else runs on synthetic and short brute-force data |
| Scope | Out-of-scope list in §4; WP6 and artifact type 2 are the first to cut |

**Stretch goals**, in priority order: (1) on-the-fly evaluation by driving LAMMPS in chunks from Python and analysing each chunk as it lands; (2) MSD and time-correlation observables; (3) spatially resolved $S_Q(q)$ at $q>0$.

## 9. Changes from the draft

| Draft | Problem | Revision |
|---|---|---|
| "Understand the temporal evolution of diagnostics" | No falsifiable claim; the actual questions were struck out | H1–H3 and an operational sensitivity $\lambda^*$ (§2) |
| $S(q\to0)=\rho k_BT\kappa_T$ stated bare | $S(0)\equiv0$ at fixed $N$; no extrapolation, error model, or independent $\kappa_T$ | §3 items 1–3 |
| $S_\chi(q\to0)=\rho k_BT\chi$ | Order parameter, field, and normalization undefined; two inconsistent forms | §1 table, §3 items 4–7 |
| Response measurement assumed available | Not for entangled chains; not in a nematic phase | §3 items 8–9 |
| "Compare our results to the literature" | No literature values exist for the orientational index | Reference baseline generated here (§4) |
| Artifacts "of controlled amplitude" | Operators and control undefined | §4 |
| No system sizes, state points, replica counts | Cannot cost or schedule | §4 defaults + pilot WP9 |
| "21-step [...]" | Missing citation | Abbott et al. (already in the reference list) |
| "Nonequilibrium index" used, Hopkins et al. struck | Uncited central concept | Reference restored |
| Content items 3–4 (viscoelasticity; CPU/GPU concurrency) | Scope creep; item 4 self-described as too hard; duplicate numbering; empty "P:" placeholders | Stretch goals |
| DVOCB($n$) systems | Undefined; atomistic scope | Excluded |
| Use `Ensemble` from heteromorph | No time axis, in-memory only, independence assumption, no topology | Vendored verbatim and extended by new types (§5.1–5.2) |
| "Interchangeable CPU and GPU backends" | No kernels named; CI cannot test GPU | §5.4 |
| Embedded-Python LAMMPS; C/C++ "if needed" | Aligning field has no stock implementation | C++ `fix` in WP7 |
| "Develop a unit testing framework" | Reinvents `pytest` | `pytest` + Hypothesis; analytic tests T1–T8 |
| No milestones or ownership | Not dispatchable | §6 |
| "repetition time scales as N3" | Typo | Reptation time $\sim N^{3}$ (experimentally $\approx N^{3.4}$) |

## References

Carried over from the draft:

- R. Auhl, R. Everaers, G. S. Grest, K. Kremer, and S. J. Plimpton, J. Chem. Phys. **119**, 12718 (2003) (the draft gave only the issue number, 24; page number from memory — verify).
- L. J. Abbott, K. E. Hart, and C. M. Colina, Theor. Chem. Acc. **132**, 1334 (2013).
- A. B. Hopkins, F. H. Stillinger, and S. Torquato, Phys. Rev. E **86**, 021505 (2012).
- J. K. Metcalfe, R. J. Szukalo, and M. A. Webb, J. Chem. Theory Comput. **22**, 1 (2026); <https://github.com/webbtheosim/GAMERS>.
- M. J. Abraham *et al.*, SoftwareX **1–2**, 19 (2015).
- D. Chandler, *Introduction to Modern Statistical Mechanics* (Oxford, 1987).
- R. Zwanzig, *Nonequilibrium Statistical Mechanics* (Oxford, 2001).

Added here from memory — **verify each before submission**:

- K. Kremer and G. S. Grest, J. Chem. Phys. **92**, 5057 (1990) — the model.
- R. Everaers *et al.*, Macromolecules **53**, 1901 (2020) — Kuhn-scale mapping of Kremer–Grest stiffness to commodity polymers.
- P. G. de Gennes and J. Prost, *The Physics of Liquid Crystals* (Oxford, 1993) — orientational susceptibility.
- A reference for the isotropic–nematic transition of semiflexible bead–spring melts (e.g. Faller, Kolb, and Müller-Plathe, 1999) — to be located.
