"""
Figure: (a) detection ability versus consequence ranking for every score on every system
(sensitivity.json, seed 0), (b) top-1 attribution accuracy versus the reversion
operator's nominal-model error (misspec.json).
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 6.5,
                     "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
NAMES = {"case39": "IEEE-39", "case118": "IEEE-118", "case1354pegase": "PEGASE-1354"}
MK = {"case39": "o", "case118": "s", "case1354pegase": "^"}
COL = {"S_OOD": "#0072B2", "D_p": "#D55E00", "S_sev": "#009E73"}
LBL = {"S_OOD": r"$S_{\mathrm{OOD}}$", "D_p": r"$D_p$", "S_sev": r"$S_{\mathrm{sev}}$"}
OTHER = ["Mahalanobis", "KNN", "Energy", "MSP", "Recon", "EdgeDiff", "D_s", "D_o"]


def main():
    S = json.load(open(os.path.join(HERE, "results/sensitivity.json")))
    M = json.load(open(os.path.join(HERE, "results/misspec.json")))
    systems = [s for s in ["case39", "case118", "case1354pegase"] if s in S]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.5, 1.55),
                                 gridspec_kw=dict(wspace=0.42, left=0.11, right=0.99, top=0.9, bottom=0.2))
    # (a)
    for sysn in systems:
        d = S[sysn]["detectors_vs_cvi"]
        for nm in OTHER:
            if nm in d:
                ax.scatter(d[nm]["AUROC"], d[nm]["rho"], s=7, marker=MK[sysn], c="0.72",
                           edgecolors="white", linewidths=0.3, zorder=2)
        for nm in ["S_OOD", "D_p", "S_sev"]:
            ax.scatter(d[nm]["AUROC"], d[nm]["rho"], s=16, marker=MK[sysn], c=COL[nm],
                       edgecolors="white", linewidths=0.3, zorder=4)
    for nm in ["S_OOD", "D_p", "S_sev"]:
        ax.scatter([], [], c=COL[nm], s=12, label=LBL[nm])
    ax.scatter([], [], c="0.72", s=8, label="others")
    ax.legend(fontsize=5.3, loc="upper left", ncol=1, frameon=False, handletextpad=0.05, borderpad=0.1,
              labelspacing=0.2, borderaxespad=0.1)
    ax.set_xlabel("detection AUROC", labelpad=1)
    ax.set_ylabel(r"$\rho$ with CVI", labelpad=1)
    ax.set_title("(a) detection vs. ranking", fontsize=6.8, pad=2)
    ax.grid(alpha=0.2, lw=0.4); ax.set_axisbelow(True); ax.tick_params(length=2, pad=1)
    ax.set_xlim(0.4, 1.03); ax.set_yticks([-0.2, 0.2, 0.6, 1.0]); ax.set_ylim(-0.3, 1.0)
    # (b)
    for sysn in systems:
        if sysn not in M: continue
        ks = sorted([k for k in M[sysn] if k != "n"], key=float)
        xs = np.array([float(k) for k in ks]) * 100; ys = np.array([M[sysn][k] for k in ks]) * 100
        bx.plot(xs, ys, "-", marker=MK[sysn], ms=2.5, lw=0.9, label=NAMES[sysn],
                color={"case39": "#0072B2", "case118": "#E69F00", "case1354pegase": "#009E73"}[sysn])
    bx.axhline(25, color="0.5", lw=0.7, ls=(0, (2.5, 1.8)))
    bx.text(20, 28, "chance", fontsize=5.3, color="0.4", ha="right")
    bx.set_xlabel(r"model error $\delta$ (%)", labelpad=1)
    bx.set_ylabel("top-1 accuracy (%)", labelpad=1)
    bx.set_title("(b) attribution", fontsize=6.8, pad=2)
    bx.set_xticks([0, 5, 10, 15, 20])
    bx.set_ylim(0, 108); bx.grid(alpha=0.2, lw=0.4); bx.set_axisbelow(True); bx.tick_params(length=2, pad=1)
    bx.legend(fontsize=5.3, loc="center left", bbox_to_anchor=(0.0, 0.52), frameon=False, handlelength=1.6, labelspacing=0.2,
              borderpad=0.1, borderaxespad=0.1, handletextpad=0.3)
    fig.savefig(os.path.join(FIG, "fig_tradeoff.pdf")); fig.savefig(os.path.join(FIG, "fig_tradeoff.png"), dpi=250)
    print("wrote fig_tradeoff.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True); main()
