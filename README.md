# Beyond Detection: Severity, Attribution, and Physical Risk of Graph Distribution Shift in Power Grids

Anonymous code release for the AI4PowerGrids @ NeurIPS 2026 submission of the same name.

This repository contains the data generator, the model, every experiment script, and the
result files that each table and figure in the paper is read from. No number in the paper
is typed by hand: each one is traceable to a file in `results/`.

```
code/        data generator, model, experiments, figure scripts
results/     JSON / NPZ outputs that the paper and the figures read
```

## Install

```
pip install -r requirements.txt
```

Python 3.11 with numpy 2.4, scipy 1.16, pandas 2.3, pandapower 3.5, matplotlib 3.10.
There is no GPU or deep-learning framework dependency: the GNN encoder is implemented
directly in NumPy (`code/gnn.py`) with manual backpropagation, so everything runs on a
plain CPU install in a few minutes per system.

## Reproduce

```
cd code
python gridgen.py case39 case118   # simulate the dataset -> ../data/*.pkl  (~660 MB)
python run2.py                     # Table 1, Table A2, attribution -> ../results/all_results_v2.json
python dual_role.py                # detection AUROC of S_OOD / S_sev / D_p -> ../results/dual_role.json
python gnnsafe.py                  # GNNSafe comparator -> ../results/gnnsafe.json
python misspec.py                  # attribution under a misspecified operator -> ../results/misspec.json
python n1.py                       # N-1 post-contingency screening -> ../results/n1.json
python fig1_build.py               # figures; add --plot-only to redraw from cached arrays
python fig2_build.py
python fig3_build.py
```

`data/` and `results/` are expected as siblings of `code/`. The simulated graphs
(~660 MB) are not shipped, but `results/` already holds every file the figures and the
paper need, so `fig*_build.py --plot-only` reproduces all three figures without
regenerating them. Every script is seeded; `run2.py` and `dual_role.py` average encoder
seeds 0–4 and `run2.py` also reports paired-bootstrap intervals.

## What each file does

`gridgen.py` builds the shift taxonomy on the IEEE 39- and 118-bus MATPOWER cases through
pandapower. Every sample is a converged AC power-flow solution with `enforce_q_lims=True`.
Voltage limits are the per-bus limits shipped with the case; IEEE-118's placeholder branch
ratings are replaced by reference ratings that place the nominal point at 60 % loading
without touching any power-flow parameter, since the shipped values make the overload rate
identically zero. Physical risk is `R_phys = 0.5 r_V + 0.5 r_F` over lines and
transformers. The module also generates the counterfactual reversion samples used for
attribution, and records the fraction of draws discarded because the power flow diverged.

`framework.py` is the detect / quantify / attribute model: the split
structural-operational representation, the Ledoit–Wolf-shrunk Mahalanobis scores, the
saturating severity transform `phi(d) = d/(d+m)`, and the counterfactual attribution
weights. `gnn.py` is the NumPy GNN autoencoder underneath it.

`run2.py` produces the detection tables, the severity–risk correlations, the `lambda_p`
sweep and the idealized attribution numbers. `dual_role.py` measures how well `S_OOD`,
`S_sev` and `D_p` each detect, which is the detection half of the trade-off plotted in
Figure 2(a) and tabulated in Table A3.

`gnnsafe.py` is the GNNSafe comparator: a bus-type node classifier plus energy belief
propagation (`K=2`, `alpha=0.5`), scored in the canonical orientation only — the sign is
never chosen by looking at test AUROC.

`misspec.py` degrades the reversion operator's nominal model by `delta` and re-measures
top-1 attribution; this is the experiment that removes the circularity of an idealized
reversion, where reverting an inactive factor is a literal no-op. `n1.py` screens
single-line outages and relates base-state severity to post-contingency risk.

## Where each reported number lives

| Paper location | File | Key |
|---|---|---|
| Table 1 (per-cell AUROC, IEEE-118) | `results/all_results_v2.json` | `detection_by_cell` |
| Table A2 (overall detection, both systems) | `results/all_results_v2.json`, `results/gnnsafe.json` | `detection_overall`, `overall` |
| Table A3 (detect / rank trade-off) | `results/dual_role.json`, `results/all_results_v2.json` | `*_AUROC`, `severity_vs_risk` |
| Bootstrap CI on `rho(D_p) - rho(S_sev)` | `results/all_results_v2.json` | `Dp_minus_Ssev_rho_CI95` |
| `lambda_p` sweep (App. A.4) | `results/all_results_v2.json` | `lambda_p_sweep` |
| Attribution, idealized and compound | `results/all_results_v2.json` | `attribution` |
| Table A4 (misspecified operator) | `results/misspec.json` | per-`delta` accuracy |
| Table A5 and the N-1 correlations | `results/n1.json` | `by_cell`, `rho_*` |
| Table A1 (divergence rates) | `results/all_results_v2.json` | `reject_rate` |
| GNNSafe per-cell (App. C.1) | `results/gnnsafe.json` | `by_cell` |
