"""
Generate the LaTeX bodies of the results tables directly from the result files, so that
no number in the manuscript is typed by hand. Writes tables/*.tex, which paper.tex
\\input{}s.
"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)
SYS = ["case39", "case118", "case1354pegase"]
NAMES = {"case39": "IEEE-39", "case118": "IEEE-118", "case1354pegase": "PEGASE-1354"}

R = json.load(open(os.path.join(HERE, "results/all_results_v2.json")))
S = json.load(open(os.path.join(HERE, "results/sensitivity.json")))
D = json.load(open(os.path.join(HERE, "results/dual_role.json")))
G = json.load(open(os.path.join(HERE, "results/gnnsafe.json")))
GR = json.load(open(os.path.join(HERE, "results/gnnsafe_rho.json")))
N1 = json.load(open(os.path.join(HERE, "results/n1_full.json")))
MS = json.load(open(os.path.join(HERE, "results/misspec.json")))
avail = [s for s in SYS if s in R]


def f3(x): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def pm(m, sd): return f"${m:.3f}{{\\pm}}{sd:.3f}$" if sd >= 0.0005 else f"{m:.3f}"


# ------------------------------------------------------------------ Table: detection + rho
rows = []
def detrow(label, auroc_fn, rho_fn, bold_auroc=False, bold_rho=False):
    cells = []
    for s in SYS:
        if s not in avail:
            cells += ["--", "--"]; continue
        a = auroc_fn(s); r = rho_fn(s)
        a = "--" if a is None else a
        r = "--" if r is None else r
        cells += [a, r]
    return label + " & " + " & ".join(cells) + r" \\"

def ours_auroc(s): return pm(*R[s]["detection_overall"]["Ours"]["AUROC"])
def oper_auroc(s): return pm(*R[s]["detection_overall"]["OperOnly"]["AUROC"])
def knn_auroc(s): return pm(*R[s]["detection_overall"]["KNN"]["AUROC"])
def msp_auroc(s): return pm(*R[s]["detection_overall"]["MSP"]["AUROC"])
def en_auroc(s): return pm(*R[s]["detection_overall"]["Energy"]["AUROC"])
def rec_auroc(s): return pm(*R[s]["detection_overall"]["Recon"]["AUROC"])
def ed_auroc(s): return f3(R[s]["detection_overall"]["EdgeDiff"]["AUROC"][0])
def gs_auroc(s): return pm(*G[s]["overall"]["AUROC"]) if s in G else None
def dp_auroc(s): return f3(R[s]["detection_overall"]["PhysOnly"]["AUROC"][0])
def ds_auroc(s): return f3(S[s]["detectors_vs_cvi"]["D_s"]["AUROC"])
def sev_auroc(s): return pm(D[s]["S_sev_AUROC"], D[s]["S_sev_AUROC_std"]) if s in D else None
def rho5(s, k): return f3(R[s]["severity_vs_risk"][k][0])
def rho1(s, k): return f3(S[s]["detectors_vs_cvi"][k]["rho"]) if k in S[s]["detectors_vs_cvi"] else None
def gs_rho(s): return f3(GR[s]["rho"]) if s in GR else None

rows.append(detrow(r"$\Sood$ (split representation)", ours_auroc, lambda s: rho5(s, "S_OOD")))
rows.append(detrow(r"$D_o$ only (operational ablation)", oper_auroc, lambda s: rho5(s, "D_o")))
rows.append(detrow(r"$k$NN \cite{sun2022knn}", knn_auroc, lambda s: rho1(s, "KNN")))
rows.append(detrow(r"MSP (adapted) \cite{hendrycks2017msp}", msp_auroc, lambda s: rho1(s, "MSP")))
rows.append(detrow(r"Energy (adapted) \cite{liu2020energy}", en_auroc, lambda s: rho1(s, "Energy")))
rows.append(detrow(r"Reconstruction error", rec_auroc, lambda s: rho1(s, "Recon")))
rows.append(detrow(r"Edge difference (no learning)", ed_auroc, lambda s: rho1(s, "EdgeDiff")))
rows.append(detrow(r"GNNSafe \cite{wu2023gnnsafe}", gs_auroc, gs_rho))
rows.append(r"\midrule")
rows.append(detrow(r"$D_s$ (structural deviation)", ds_auroc, lambda s: rho5(s, "D_s")))
rows.append(detrow(r"$D_p$ (physics deviation)", dp_auroc, lambda s: rho5(s, "D_p")))
rows.append(detrow(r"$\Ssev$ (fused severity)", sev_auroc, lambda s: rho5(s, "S_sev")))
open(os.path.join(OUT, "tab_detect.tex"), "w").write("\n".join(rows) + "\n\\bottomrule\n")

# ------------------------------------------------------------------ Table: N-1 per cell
CELLS = [("ID-id", "ID"), ("topo-near", "Topology--near"), ("topo-far", "Topology--far"),
         ("load-near", "Load--near"), ("load-far", "Load--far"), ("gen-near", "Generation--near"),
         ("gen-far", "Generation--far"), ("sensor-near", "Sensor--near"), ("sensor-far", "Sensor--far"),
         ("compound-far", "Compound")]
rows = []
for key, lab in CELLS:
    cells = []
    for s in SYS:
        if s not in N1 or key not in N1[s]["by_cell"]:
            cells += ["--", "--", "--"]; continue
        c = N1[s]["by_cell"][key]
        cells += [f3(c["base"]), f3(c["N1_mean"]), f3(c["N1_max"])]
    rows.append(lab + " & " + " & ".join(cells) + r" \\")
open(os.path.join(OUT, "tab_n1.tex"), "w").write("\n".join(rows) + "\n\\bottomrule\n")
# Table: rho with the N-1 indices, transposed to fit one column
SC = [("S_sev", r"$\Ssev$"), ("S_OOD", r"$\Sood$"), ("D_p", r"$D_p$")]
TG = [("N1_mean", "avg"), ("N1_mean_x", "excl."), ("N1_max", "max")]
rows = []
for s in SYS:
    if s not in N1: continue
    tg = [t for t in TG if not (s == "case1354pegase" and t[0] == "N1_max")]
    for i, (t, tl) in enumerate(tg):
        vals = [N1[s][f"rho_{sc}_{t}"] for sc, _ in SC]
        best = max(vals)
        cells = [(r"\textbf{%.2f}" % v) if abs(round(v, 2) - round(best, 2)) < 1e-9 else "%.2f" % v for v in vals]
        lab = (r"\multirow{%d}{*}{%s}" % (len(tg), NAMES[s])) if i == 0 else ""
        rows.append(f"{lab} & {tl} & " + " & ".join(cells) + r" \\")
    if s != SYS[-1]: rows.append(r"\midrule")
open(os.path.join(OUT, "tab_n1rho.tex"), "w").write("\n".join(rows) + "\n\\bottomrule\n")

# Table: screening prioritization
T = json.load(open(os.path.join(HERE, "results/triage.json")))
def rnd_effort(n, m, q=0.9):
    import math
    j = math.ceil(q * m - 1e-12); return j * (n + 1) / (m + 1) / n
rows = []
cells = []
for s in SYS:
    cells += ["0.20", "%.2f" % rnd_effort(T[s]["n"], T[s]["n_crit"])]
rows.append("Random order & " + " & ".join(cells) + r" \\")
for item in [("S_OOD", r"$\Sood$"), ("D_p", r"$D_p$"), ("S_sev", r"$\Ssev$"), None, ("base", r"Base-case $\CVI$ (ref.)")]:
    if item is None: rows.append(r"\midrule"); continue
    sc, lab = item
    cells = []
    for s in SYS:
        e = T[s][sc]; cells += ["%.2f" % e["R20"], "%.2f" % e["E90"]]
    rows.append(lab + " & " + " & ".join(cells) + r" \\")
open(os.path.join(OUT, "tab_triage.tex"), "w").write("\n".join(rows) + "\n\\bottomrule\n")

# ------------------------------------------------------------------ misc numbers for the text
misc = {}
for s in avail:
    misc[s] = dict(n_cont=N1[s]["n_cont"] if s in N1 else None,
                   n_isl=N1[s]["n_islanding"] if s in N1 else None,
                   n_points=N1[s]["n_points"] if s in N1 else None,
                   misspec=MS.get(s), diag=R[s].get("solve_diag") if "solve_diag" in R[s] else None)
json.dump(misc, open(os.path.join(OUT, "misc.json"), "w"), indent=1)
print("tables written:", os.listdir(OUT))
