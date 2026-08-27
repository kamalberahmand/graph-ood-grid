"""
Figure A1 (empirical OOD taxonomy) on REAL data, IEEE-118:
(a) distribution of severity S_sev per taxonomy cell,
(b) distribution of physical risk R_phys per taxonomy cell.
Replaces the decorative taxonomy diagram with measured distributions, and makes the
sensor exception (high severity signal but low physical risk) visible.
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from framework import Model

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
OI = {"ID": "#9A9A9A", "topo": "#E69F00", "load": "#0072B2", "gen": "#009E73",
      "sensor": "#CC79A7", "compound": "#D55E00"}
CELLS = [("ID", "id"), ("topo", "near"), ("topo", "far"), ("load", "near"), ("load", "far"),
         ("gen", "near"), ("gen", "far"), ("sensor", "near"), ("sensor", "far"),
         ("compound", "far")]
TICK = ["ID", "topo\nnear", "topo\nfar", "load\nnear", "load\nfar", "gen\nnear", "gen\nfar",
        "sens\nnear", "sens\nfar", "comp"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def build(system="case118"):
    data, _ = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    M = Model(kind="gcn", seed=0).fit(data["train"])
    groups = {("ID", "id"): data["test_id"]}
    for s in data["ood"]:
        groups.setdefault((s["meta"]["label"], s["meta"]["band"]), []).append(s)
    sev, risk = {}, {}
    for key in CELLS:
        g = groups.get(key, [])
        sev[key] = np.array([M.s_sev(s)[0] for s in g])
        risk[key] = np.array([s["Rphys"]["R_phys"] for s in g])
    np.savez(os.path.join(HERE, "results/fig2_real.npz"),
             **{f"sev_{a}_{b}": sev[(a, b)] for a, b in CELLS},
             **{f"risk_{a}_{b}": risk[(a, b)] for a, b in CELLS})
    for key in CELLS:
        print(f"{key[0]:9s}{key[1]:5s} n={len(sev[key]):3d} "
              f"S_sev med={np.median(sev[key]):.2f}  R_phys med={np.median(risk[key]):.4f}")


def _box(ax, data, ylabel, title, logy=False):
    pos = np.arange(len(CELLS))
    bp = ax.boxplot([data[c] for c in CELLS], positions=pos, widths=0.62,
                    patch_artist=True, showfliers=False, whis=(5, 95),
                    medianprops=dict(color="black", lw=0.9),
                    whiskerprops=dict(lw=0.6, color="0.35"),
                    capprops=dict(lw=0.6, color="0.35"),
                    boxprops=dict(lw=0.6))
    for patch, (cls, _) in zip(bp["boxes"], CELLS):
        patch.set_facecolor(OI[cls]); patch.set_alpha(0.75); patch.set_edgecolor("0.3")
    ax.set_xticks(pos); ax.set_xticklabels(TICK, fontsize=5.9)
    ax.set_ylabel(ylabel, fontsize=7.2, labelpad=2)
    ax.set_title(title, fontsize=7.8, pad=3)
    ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.grid(axis="y", alpha=0.22, lw=0.4); ax.set_axisbelow(True)
    if logy:
        ax.set_yscale("log")
    ax.axvline(0.5, color="0.6", lw=0.6, ls=(0, (2, 2)))


def plot():
    d = np.load(os.path.join(HERE, "results/fig2_real.npz"))
    sev = {c: d[f"sev_{c[0]}_{c[1]}"] for c in CELLS}
    risk = {c: d[f"risk_{c[0]}_{c[1]}"] for c in CELLS}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85),
                             gridspec_kw=dict(wspace=0.26, left=0.075, right=0.995,
                                              top=0.86, bottom=0.20))
    _box(axes[0], sev, "severity $S_{\\mathrm{sev}}$", "(a) severity by taxonomy cell")
    _box(axes[1], risk, "physical risk $R_{\\mathrm{phys}}$",
         "(b) physical risk by taxonomy cell")
    # The sensor exception is stated in the caption rather than annotated on the axes.
    ymax = max(np.percentile(risk[c], 95) for c in CELLS)
    axes[1].set_ylim(bottom=0, top=ymax * 1.06)
    axes[0].set_ylim(top=max(np.percentile(sev[c], 95) for c in CELLS) * 1.03)
    fig.savefig(os.path.join(FIG, "fig2_taxonomy.pdf"))
    fig.savefig(os.path.join(FIG, "fig2_taxonomy.png"), dpi=300)
    print("wrote fig2_taxonomy.pdf")


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        build()
    plot()
