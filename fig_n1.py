"""
Figure: full N-1 security screening. One row per score (S_sev, S_OOD), one column per
system; each point is an operating point, x = base-state score, y = average post-
contingency constraint-violation index over every single-branch outage. Spearman rho is
annotated. Colors follow the shift factor (fixed order), markers are all the same.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
FAC = {"ID": "#9A9A9A", "topo": "#E69F00", "load": "#0072B2", "gen": "#009E73",
       "sensor": "#CC79A7", "compound": "#D55E00"}
NAMES = {"case39": "IEEE-39", "case118": "IEEE-118", "case1354pegase": "PEGASE-1354"}


def main(target="N1_mean"):
    R = json.load(open(os.path.join(HERE, "results/n1_full.json")))
    systems = [s for s in ["case39", "case118", "case1354pegase"] if s in R]
    fig, axes = plt.subplots(2, len(systems), figsize=(7.16, 2.25),
                             gridspec_kw=dict(wspace=0.32, hspace=0.48, left=0.07, right=0.99,
                                              top=0.93, bottom=0.2))
    axes = np.atleast_2d(axes)
    if len(systems) == 1: axes = axes.reshape(2, 1)
    ylab = {"N1_mean": r"$\mathrm{CVI}^{N-1}_{\mathrm{avg}}$", "N1_max": r"$\mathrm{CVI}^{N-1}_{\max}$"}[target]
    for j, sysn in enumerate(systems):
        rows = R[sysn]["rows"]
        y = np.array([r[target] for r in rows]); cells = [r["cell"].split("-")[0] for r in rows]
        for i, (score, lab, logx) in enumerate([("S_sev", r"severity $S_{\mathrm{sev}}$", False),
                                                ("S_OOD", r"detection $S_{\mathrm{OOD}}$", True)]):
            ax = axes[i, j]; x = np.array([r[score] for r in rows])
            for fac, col in FAC.items():
                m = np.array([c == fac for c in cells])
                if m.any():
                    ax.scatter(x[m], y[m], s=9, c=col, edgecolors="white", linewidths=0.3,
                               label=fac if (i == 0 and j == 0) else None, zorder=3)
            if logx: ax.set_xscale("log")
            rho = spearmanr(x, y).correlation
            ax.text(0.03, 0.95, rf"$\rho={rho:.2f}$", transform=ax.transAxes, va="top", fontsize=7.5)
            ax.set_xlabel(lab, labelpad=1.5)
            if j == 0: ax.set_ylabel(ylab, labelpad=2)
            if i == 0: ax.set_title(NAMES[sysn], fontsize=8, pad=3)
            ax.grid(alpha=0.2, lw=0.4); ax.set_axisbelow(True); ax.tick_params(length=2, pad=1)
            ax.set_ylim(bottom=-0.02)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.53, 0.0), handletextpad=0.1, columnspacing=0.9)
    suf = "" if target == "N1_mean" else "_max"
    fig.savefig(os.path.join(FIG, f"fig_n1{suf}.pdf")); fig.savefig(os.path.join(FIG, f"fig_n1{suf}.png"), dpi=250)
    print("wrote", f"fig_n1{suf}.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    main("N1_mean"); main("N1_max")
