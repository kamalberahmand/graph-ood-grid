"""
Figure: sensitivity of the consequence ranking to (a) the severity weights, (b) the CVI
composition weight w, and (c) the reference-rating level. Reads results/sensitivity.json.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
NAMES = {"case39": "IEEE-39", "case118": "IEEE-118", "case1354pegase": "PEGASE-1354"}
SYSCOL = {"case39": "#0072B2", "case118": "#E69F00", "case1354pegase": "#009E73"}
MK = {"case39": "o", "case118": "s", "case1354pegase": "^"}
LAMS = ["(1, 1, 2)", "(1, 1, 1)", "(1, 1, 4)", "(1, 1, 8)", "(1, 2, 2)", "(2, 1, 2)",
        "(0, 1, 2)", "(1, 0, 2)", "(1, 1, 0)", "(0, 0, 1)", "(0, 1, 0)", "(1, 0, 0)"]
LAMLBL = ["1,1,2", "1,1,1", "1,1,4", "1,1,8", "1,2,2", "2,1,2", "0,1,2", "1,0,2", "1,1,0",
          "0,0,1", "0,1,0", "1,0,0"]


def main():
    S = json.load(open(os.path.join(HERE, "results/sensitivity.json")))
    systems = [s for s in ["case39", "case118", "case1354pegase"] if s in S]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.16, 1.65),
                                     gridspec_kw=dict(wspace=0.38, left=0.06, right=0.99, top=0.88, bottom=0.3, width_ratios=[1.7, 1, 1]))
    # (a) weights: rho (filled) and AUROC (hollow) per setting
    xs = np.arange(len(LAMS)); w = 0.26
    for k, sysn in enumerate(systems):
        g = S[sysn]["severity_weights"]
        rho = [g[l]["rho"] for l in LAMS]; au = [g[l]["AUROC"] for l in LAMS]
        ax.bar(xs + (k - 1) * w, rho, w, color=SYSCOL[sysn], label=NAMES[sysn], zorder=3)
        ax.plot(xs + (k - 1) * w, au, ls="none", marker=MK[sysn], ms=3, mfc="white", mec=SYSCOL[sysn], mew=0.8, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(LAMLBL, rotation=90, fontsize=5.8)
    ax.set_xlabel(r"$(\lambda_s,\lambda_o,\lambda_p)$", labelpad=1)
    ax.set_ylabel(r"$\rho(S_{\mathrm{sev}},\mathrm{CVI})$ (bars)", labelpad=2)
    ax.set_title("(a) severity weights; hollow markers: AUROC", fontsize=7.8, pad=3)
    ax.axvline(0.5 + 8, color="0.6", lw=0.5, ls=":")
    ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=0.2, lw=0.4); ax.set_axisbelow(True); ax.tick_params(length=2, pad=1)
    ax.legend(fontsize=6, loc="upper center", frameon=False, ncol=3, handletextpad=0.3, columnspacing=0.8,
              bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 1.33); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # (b) CVI composition
    for sysn in systems:
        g = S[sysn]["cvi_weight"]; ws = sorted(g, key=float)
        bx.plot([float(x) for x in ws], [g[x]["rho_Ssev"] for x in ws], "-", marker=MK[sysn], ms=3, lw=1.0, color=SYSCOL[sysn])
        bx.plot([float(x) for x in ws], [g[x]["rho_Sood"] for x in ws], "--", marker=MK[sysn], ms=3, lw=0.8, color=SYSCOL[sysn], mfc="white", mew=0.7)
    bx.plot([], [], "-", color="0.3", label=r"$S_{\mathrm{sev}}$"); bx.plot([], [], "--", color="0.3", label=r"$S_{\mathrm{OOD}}$")
    bx.legend(fontsize=6, frameon=False, loc="lower center", ncol=2, columnspacing=0.8)
    bx.set_xlabel(r"voltage weight $w$ in $\mathrm{CVI}=w\,r_V+(1-w)\,r_F$", labelpad=1, fontsize=7)
    bx.set_ylabel(r"$\rho$ with $\mathrm{CVI}_w$", labelpad=2)
    bx.set_title("(b) index composition", fontsize=7.8, pad=3)
    bx.set_ylim(0, 1.02); bx.grid(alpha=0.2, lw=0.4); bx.set_axisbelow(True); bx.tick_params(length=2, pad=1)
    # (c) rating level
    for sysn in systems:
        g = S[sysn]["rating_level"]
        if not isinstance(g, dict): continue
        ls_ = sorted(g, key=float)
        cx.plot([float(x) for x in ls_], [g[x]["rho_Ssev"] for x in ls_], "-", marker=MK[sysn], ms=3, lw=1.0, color=SYSCOL[sysn], label=NAMES[sysn])
        cx.plot([float(x) for x in ls_], [g[x]["rho_Sood"] for x in ls_], "--", marker=MK[sysn], ms=3, lw=0.8, color=SYSCOL[sysn], mfc="white", mew=0.7)
    cx.set_xlabel("reference-rating level (%)", labelpad=1, fontsize=7)
    cx.set_ylabel(r"$\rho$ with CVI", labelpad=2)
    cx.set_title("(c) reference-rating level", fontsize=7.8, pad=3)
    cx.set_ylim(0, 1.02); cx.set_xticks([50, 60, 70, 80]); cx.grid(alpha=0.2, lw=0.4); cx.set_axisbelow(True); cx.tick_params(length=2, pad=1)
    cx.legend(fontsize=6, frameon=False, loc="lower center", ncol=2, columnspacing=0.8)
    fig.savefig(os.path.join(FIG, "fig_sens.pdf")); fig.savefig(os.path.join(FIG, "fig_sens.png"), dpi=250)
    print("wrote fig_sens.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True); main()
