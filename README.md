# Severity-Aware Out-of-Distribution Detection for Graph Learning in Power Systems

Code and result files for the paper

> K. Berahmand, S. Forouzandeh, N. Al Khafaf, L. Tamang, A. Kamoona, and M. Jalili,
> "Severity-Aware Out-of-Distribution Detection for Graph Learning in Power Systems,"
> submitted to *IEEE Transactions on Smart Grid*.

The framework extends a post-hoc graph OOD detector with (i) a severity score that fuses
representation and physics deviations, (ii) counterfactual cause attribution by reverting
each candidate factor and re-solving the AC power flow, and (iii) evaluation against
base-case constraint violations and full N-1 screening. Experiments use the IEEE 39-bus,
IEEE 118-bus and PEGASE 1354-bus systems with generator reactive-power limits enforced.

Every number in the paper's tables is written by `make_tables.py` from the JSON files in
`results/`; nothing is typed by hand.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11, CPU only. There is no deep-learning framework dependency: the graph
convolutional autoencoder is implemented in NumPy with manual backpropagation (`gnn.py`).

## Quick start: regenerate the tables and figures from the shipped results

`results/` already contains every file the paper reads, so the tables and figures can be
rebuilt in under a minute without simulating anything:

```bash
python triage.py                 # Table VI (screening prioritization)
python make_tables.py            # tables/*.tex
python fig1_build.py --plot-only # Fig. 1
python fig_tradeoff.py           # Fig. 2
python fig_n1.py                 # Fig. 3
python fig_sens.py               # Fig. 4
```

## Full reproduction

```bash
./run_all.sh                       # all three systems
./run_all.sh case39 case118        # IEEE systems only (about 1-2 h on two cores)
```

`run_all.sh` simulates the datasets into `data/` (about 1.6 GB for the three systems, not
shipped), runs every experiment, and rebuilds `tables/` and `figs/`. PEGASE-1354 takes
about a day on a two-core machine, mostly for data generation and the 1430-outage N-1
screening. All scripts are seeded. Set `OMP_NUM_THREADS=1`: the per-sample linear algebra
is small and multithreaded BLAS only adds contention.

As a check, `python run2.py case39` on the generated IEEE-39 data reproduces the reported
detection AUROC (0.9965 ± 0.0009) and rho(S_sev, CVI) = 0.765 exactly.

## Files

| File | Role | Paper |
|---|---|---|
| `gridgen.py` | Shift taxonomy, AC power flow with Q limits, convergence diagnosis (`solve_diag`), reference ratings, loadability (`nose_point`), counterfactual reversion samples | Sec. III-A, IV-D, V-A, V-B; Tables I, II, VII |
| `gnn.py` | NumPy GCN autoencoder; sparse propagation for large systems | Sec. IV-A |
| `framework.py` | Structural descriptor, Ledoit-Wolf Mahalanobis scores, severity score with phi(d)=d/(d+m), attribution, baselines | Sec. IV-A to IV-C |
| `run2.py` | Detection (5 seeds), severity vs. CVI, lambda_p sweep, bootstrap intervals, attribution | Table III, Sec. V-D, V-E, V-H, V-J |
| `dual_role.py` | Detection AUROC of S_OOD, S_sev and D_p over 5 seeds | Table III, Fig. 2(a) |
| `sensitivity.py` | Fusion weights, CVI composition w, reference-rating level, all detectors vs. CVI | Fig. 4, Sec. V-I |
| `misspec.py` | Attribution with an inaccurate reversion operator (model error delta) | Fig. 2(b), Sec. V-H |
| `n1_full.py` | Full N-1 screening over all non-islanding single-branch outages | Tables IV, V, Fig. 3, Sec. V-F |
| `triage.py` | Screening prioritization (R@20, E90) | Table VI, Sec. V-G |
| `gnnsafe.py`, `gnnsafe_rho.py` | GNNSafe comparator and its rank correlation with the CVI | Table III, Appendix B |
| `timing.py` | Fit, scoring and power-flow times | Sec. V-C |
| `make_tables.py` | Writes `tables/*.tex` from `results/` | all result tables |
| `fig*.py` | Figures 1 to 4 | Figs. 1-4 |

## Results files

| File | Contents |
|---|---|
| `all_results_v2.json` | Detection per shift type, severity-consequence correlations, attribution, convergence classes (`solve_diag`) |
| `case*_v2.json` | The same, one file per system |
| `dual_role.json` | Five-seed AUROC of S_OOD, S_sev, D_p |
| `sensitivity.json` | Sensitivity grids and every detector against the CVI |
| `misspec.json` | Attribution accuracy against model error delta |
| `n1_full.json` | Per-point N-1 indices (`rows`), per-shift-type means (`by_cell`), Spearman rho values |
| `triage.json` | Screening prioritization metrics |
| `gnnsafe.json`, `gnnsafe_rho.json` | GNNSafe AUROC and rho |
| `timing.json` | Computational cost |
| `fig1_real.npz` | Data behind Fig. 1 |

## Notes on the implementation

* **Constraint-violation index.** In the code the CVI is stored as `Rphys["R_phys"]`
  (an earlier name); it is `0.5 r_V + 0.5 r_F` over buses and over in-service lines and
  transformers, as in Definition 4 of the paper.
* **Convergence diagnosis.** `gridgen.solve_diag` classifies every power flow, in the base
  case and after each contingency, as `ok`, `ok_retry` (numerical, kept),
  `qlim_infeasible` or `collapse`. A flat start is never used.
* **Ratings.** Placeholder rating classes (all IEEE-118 branches, PEGASE transformers) are
  replaced by reference ratings that put the nominal point at 60% loading. No power-flow
  parameter is changed.
* **PEGASE load bands** are scaled by `gridgen.LOAD_MARGIN_SCALE` (0.605) so that they sit
  at the same fractions of the loadability margin as on IEEE-39.
* **Large systems.** Above 500 buses the adjacency and the GCN propagation operator are
  sparse, and the spectral descriptor is cached by topology.

## License

MIT (see `LICENSE`).
