"""
Figure 2 of the paper (headline result), from the corrected experiments:
(a) the detection / risk-ranking trade-off: the detection score ranks physical risk
    poorly, the physics deviation ranks risk well but detects poorly, and only the fused
    severity is competent at both;
(b) attribution accuracy as the reversion operator's nominal model degrades, which is
    the experiment that removes the circularity of the idealized reversion.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})
C = {"S_OOD": "#2F6DB5", "D_p": "#C0392B", "S_sev": "#2E8B57"}


def main():
    R = json.load(open(os.path.join(HERE, "results/all_results_v2.json")))
    dual = json.load(open(os.path.join(HERE, "results/dual_role.json")))
    mis = json.load(open(os.path.join(HERE, "results/misspec.json")))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.5, 1.68),
                                   gridspec_kw=dict(wspace=0.34, left=0.095, right=0.985,
                                                    top=0.86, bottom=0.21))

    # ---------- (a) detection vs risk-ranking trade-off ----------
    mk = {"case39": "o", "case118": "s"}
    for sysn in ["case39", "case118"]:
        sv = R[sysn]["severity_vs_risk"]; du = dual[sysn]
        pts = {"S_OOD": (du["S_OOD_AUROC"], sv["S_OOD"][0]),
               "D_p":   (du["PhysOnly_AUROC"], sv["D_p"][0]),
               "S_sev": (du["S_sev_AUROC"], sv["S_sev"][0])}
        for k, (x, y) in pts.items():
            axA.scatter([x], [y], s=44, marker=mk[sysn], c=C[k],
                        edgecolors="white", linewidths=0.6, zorder=4)
        # connect the three scores of one system
        order = ["S_OOD", "S_sev", "D_p"]
        axA.plot([pts[k][0] for k in order], [pts[k][1] for k in order],
                 color="0.65", lw=0.7, ls=(0, (2.5, 1.8)), zorder=2)
    axA.set_xlabel("detection AUROC", fontsize=7.2, labelpad=2)
    axA.set_ylabel("Spearman $\\rho$ with $R_{\\mathrm{phys}}$", fontsize=7.2, labelpad=2)
    axA.set_title("(a) no single score does both", fontsize=8.0, pad=3)
    axA.tick_params(labelsize=6.4, length=2, pad=1)
    axA.grid(alpha=0.22, lw=0.4); axA.set_axisbelow(True)
    axA.set_xlim(0.70, 1.035); axA.set_ylim(0.25, 0.95)
    lbl = {"S_OOD": ("$S_{\\mathrm{OOD}}$", -0.012, -0.10), "D_p": ("$D_p$ only", 0.0, 0.045),
           "S_sev": ("$S_{\\mathrm{sev}}$", -0.005, 0.045)}
    du = dual["case118"]; sv = R["case118"]["severity_vs_risk"]
    for k, (x, y) in {"S_OOD": (du["S_OOD_AUROC"], sv["S_OOD"][0]),
                      "D_p": (du["PhysOnly_AUROC"], sv["D_p"][0]),
                      "S_sev": (du["S_sev_AUROC"], sv["S_sev"][0])}.items():
        t, dx, dy = lbl[k]
        axA.text(x + dx, y + dy, t, fontsize=7.0, color=C[k],
                 ha="right" if k == "S_OOD" else "center", va="bottom")
    axA.scatter([], [], marker="o", c="0.45", s=24, label="IEEE-39")
    axA.scatter([], [], marker="s", c="0.45", s=24, label="IEEE-118")
    axA.legend(fontsize=5.8, loc="lower left", handletextpad=0.2, borderpad=0.3,
               framealpha=0.9, edgecolor="0.85")

    # ---------- (b) attribution under a misspecified reversion operator ----------
    ds = [k for k in mis["case118"] if k != "n"]
    xs = np.array([float(k) for k in ds])
    o = np.argsort(xs); xs = xs[o]
    for sysn, col, m in [("case39", "#0072B2", "o"), ("case118", "#E69F00", "s")]:
        ys = np.array([mis[sysn][k] for k in ds])[o]
        axB.plot(xs * 100, ys * 100, "-", color=col, lw=1.2, marker=m, ms=3.4,
                 label=f"IEEE-{sysn[4:]}", zorder=4)
    axB.axhline(25, color="0.5", lw=0.7, ls=(0, (2.5, 1.8)), zorder=2)
    axB.text(19.5, 28, "chance (4 factors)", fontsize=5.6, color="0.4",
             ha="right", va="bottom")
    axB.set_xlabel("operator nominal-model error $\\delta$ (%)", fontsize=7.2, labelpad=2)
    axB.set_ylabel("top-1 attribution (%)", fontsize=7.2, labelpad=2)
    axB.set_title("(b) attribution vs. reversion fidelity", fontsize=8.0, pad=3)
    axB.set_ylim(0, 112); axB.tick_params(labelsize=6.4, length=2, pad=1)
    axB.grid(alpha=0.22, lw=0.4); axB.set_axisbelow(True)
    axB.legend(fontsize=6.0, loc="lower left", handlelength=1.3, borderpad=0.3,
               framealpha=0.95, edgecolor="0.85", bbox_to_anchor=(0.0, 0.02))

    fig.savefig(os.path.join(FIG, "fig3_severity_risk.pdf"))
    fig.savefig(os.path.join(FIG, "fig3_severity_risk.png"), dpi=300)
    print("wrote fig3_severity_risk.pdf")


if __name__ == "__main__":
    main()
