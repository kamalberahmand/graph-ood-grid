"""
Figure 1 (method illustration on REAL data), 3 panels:
(a) Detection in representation space: ID manifold + OOD graphs; example G' with S_OOD.
(b) Counterfactual attribution: revert each factor, re-simulate; true factor returns G' to the manifold.
(c) Reverting the identified cause restores physical feasibility (R_phys before/after).
All quantities are computed from the trained model + pandapower simulator.
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from framework import Model

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
FAC = ["topo", "load", "gen", "sensor"]
OI = {"ID": "#B8B8B8", "topo": "#E69F00", "load": "#0072B2", "gen": "#009E73",
      "sensor": "#CC79A7", "compound": "#D55E00"}
LAB = {"topo": "topology", "load": "load", "gen": "generation", "sensor": "sensor"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.linewidth": 0.7, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


def zo(M, s):
    return M.rep(s)[0]          # operational embedding (well-scaled for 2D display)


def build():
    system = "case118"
    data, info = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    attrib, _ = pickle.load(open(os.path.join(HERE, f"data/{system}_attrib.pkl"), "rb"))
    M = Model(kind="gcn", seed=0).fit(data["train"])

    # 2D map from the operational embedding, standardized then PCA
    IDzo = np.array([zo(M, s) for s in data["train"]])
    mu, sd = IDzo.mean(0), IDzo.std(0) + 1e-9
    IDs = (IDzo - mu) / sd
    c = IDs.mean(0)
    U, S, Vt = np.linalg.svd(IDs - c, full_matrices=False)
    P2 = Vt[:2].T
    proj = lambda G: (((zo(M, G) - mu) / sd) - c) @ P2
    projX = lambda X: (((X - mu) / sd) - c) @ P2
    IDp = ((IDs - c) @ P2)

    # OOD background by class
    ood_by = {}
    for s in data["ood"]:
        ood_by.setdefault(s["meta"]["label"], []).append(s)
    bg = {}
    rng = np.random.default_rng(0)
    for lab, lst in ood_by.items():
        idx = rng.choice(len(lst), size=min(40, len(lst)), replace=False)
        bg[lab] = projX(np.array([zo(M, lst[i]) for i in idx]))

    # reference level of S_OOD on ID data (for the threshold line in panel b)
    id_scores = np.array([M.s_ood(s) for s in data["test_id"]])
    id_q95 = float(np.percentile(id_scores, 95))

    # choose a legible SINGLE-FACTOR example: correct top-1, clear separation, and
    # a physical-risk drop when the cause is reverted (prefer both r_V and r_F active).
    idc = IDp.mean(0); idr = np.linalg.norm(IDp - idc, axis=1).mean() + 1e-9
    best = None
    for a in attrib:
        if a["scenario"] not in ("load", "gen", "topo"):
            continue
        so = M.s_ood(a["obs"])
        dl = {k: so - M.s_ood(a["cf"][k]) for k in FAC}
        if max(dl, key=dl.get) != a["active"][0]:
            continue
        sep = np.linalg.norm(proj(a["obs"]) - idc) / idr
        R = a["obs"]["Rphys"]
        drop = R["R_phys"] - a["cf"][a["active"][0]]["Rphys"]["R_phys"]
        if drop <= 0 or sep < 1.3 or R["r_V"] <= 0:
            continue
        score = 3.0 * drop + 0.5 * R["R_phys"] + 0.2 * sep + (1.5 if R["r_F"] > 0 else 0.0)
        if best is None or score > best[0]:
            best = (score, a, so, dl)
    _, ex, so_obs, deltas = best
    cause = ex["active"][0]

    obs_p = proj(ex["obs"])
    cf_p = {k: proj(ex["cf"][k]) for k in FAC}
    so_cf = {k: M.s_ood(ex["cf"][k]) for k in FAC}
    pos = {k: max(0.0, deltas[k]) for k in FAC}
    tot = sum(pos.values()) + 1e-12
    A = {k: pos[k] / tot for k in FAC}

    Robs = ex["obs"]["Rphys"]; Rcf = ex["cf"][cause]["Rphys"]
    ssev = M.s_sev(ex["obs"])[0]
    print("cause=%s" % cause)

    np.savez(os.path.join(HERE, "results/fig1_real.npz"),
             IDp=IDp, obs_p=obs_p,
             cf_p=np.array([cf_p[k] for k in FAC]),
             A=np.array([A[k] for k in FAC]),
             so_obs=so_obs, so_cf=np.array([so_cf[k] for k in FAC]),
             Robs=[Robs["r_bal"], Robs["r_V"], Robs["r_F"], Robs["R_phys"]],
             Rcf=[Rcf["r_bal"], Rcf["r_V"], Rcf["r_F"], Rcf["R_phys"]],
             ssev=ssev, cause=cause, id_q95=id_q95,
             bg_topo=bg.get("topo"), bg_load=bg.get("load"), bg_gen=bg.get("gen"),
             bg_sensor=bg.get("sensor"), bg_compound=bg.get("compound"))
    print("example S_OOD obs=%.1f  A=%s  R_phys obs=%.3f -> reverted=%.3f  S_sev=%.2f"
          % (so_obs, {k: round(A[k], 2) for k in FAC}, Robs["R_phys"], Rcf["R_phys"], ssev))


def ellipse_from(points, nstd=2.2, **kw):
    c = points.mean(0); cov = np.cov(points.T)
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    width, height = 2 * nstd * np.sqrt(w[::-1])
    return Ellipse(c, width, height, angle=ang, **kw), c


def plot():
    d = np.load(os.path.join(HERE, "results/fig1_real.npz"), allow_pickle=True)
    IDp, obs_p, cf_p, A = d["IDp"], d["obs_p"], d["cf_p"], d["A"]
    so_obs, so_cf = float(d["so_obs"]), np.asarray(d["so_cf"], float)
    Robs, Rcf, ssev = d["Robs"], d["Rcf"], float(d["ssev"])
    cause = str(d["cause"]); q95 = float(d["id_q95"])
    ci = FAC.index(cause)

    # width MUST match the NeurIPS text width (5.5in) so fonts are not rescaled
    # width = IEEE double-column text width (7.16in) so fonts are not rescaled
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95),
                             gridspec_kw=dict(width_ratios=[0.92, 1.20, 0.80],
                                              wspace=0.5, left=0.04, right=0.995,
                                              top=0.82, bottom=0.19))
    axA, axB, axC = axes

    # ================= (a) detection in representation space =================
    ell, cID = ellipse_from(IDp, fc="#dce7f2", ec="#7f9ec2", lw=0.9, alpha=0.75, zorder=1)
    axA.add_patch(ell)
    axA.scatter(IDp[:, 0], IDp[:, 1], s=3.5, c=OI["ID"], alpha=0.55, zorder=2, linewidths=0)
    for lab in ["load", "gen", "sensor", "compound"]:
        b = d[f"bg_{lab}"]
        if b is not None and b.size:
            axA.scatter(b[:, 0], b[:, 1], s=4.5, c=OI[lab], alpha=0.65, zorder=2, linewidths=0)
    axA.annotate("", xy=(obs_p[0], obs_p[1]), xytext=(cID[0], cID[1]),
                 arrowprops=dict(arrowstyle="-", ls=(0, (2.5, 1.8)), color="0.25", lw=0.8), zorder=3)
    axA.text(cID[0] + 0.55 * (obs_p[0] - cID[0]), cID[1] + 0.55 * (obs_p[1] - cID[1]) + 0.7,
             "$S_{\\mathrm{OOD}}$", fontsize=7.4, color="0.15", ha="center", va="bottom", zorder=7,
             bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
    axA.scatter([obs_p[0]], [obs_p[1]], marker="*", s=95, c="#D55E00",
                edgecolors="white", linewidths=0.5, zorder=6)
    axA.text(obs_p[0], obs_p[1] + 1.4, "$G'$", fontsize=7.8, ha="right", va="bottom",
             color="#D55E00", zorder=6)
    axA.text(cID[0] - 0.40 * ell.width, cID[1] + 0.42 * ell.height, "ID", fontsize=7.2,
             ha="center", va="bottom", color="#2f4d6b", style="italic", zorder=7)
    # zoom so the data fills the panel
    allx = np.concatenate([IDp[:, 0], [obs_p[0]]] + [d[f"bg_{l}"][:, 0] for l in ["load","gen","sensor","compound"]])
    ally = np.concatenate([IDp[:, 1], [obs_p[1]]] + [d[f"bg_{l}"][:, 1] for l in ["load","gen","sensor","compound"]])
    x0, x1 = np.percentile(allx, [1, 99]); y0, y1 = np.percentile(ally, [1, 99])
    x0, x1 = min(x0, obs_p[0]) , max(x1, obs_p[0]); y0, y1 = min(y0, obs_p[1]), max(y1, obs_p[1])
    axA.set_xlim(x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))
    axA.set_ylim(y0 - 0.08 * (y1 - y0), y1 + 0.18 * (y1 - y0))
    axA.set_title("(a) detect", fontsize=8.2, pad=3)
    axA.set_xlabel("representation space (PCA)", fontsize=7.0, labelpad=1.5)
    axA.set_xticks([]); axA.set_yticks([])
    handles = [Line2D([0], [0], marker='o', ls='', mfc=OI[k], mec='none', ms=3.2, label=v)
               for k, v in [("ID", "ID"), ("load", "load"), ("gen", "gen."),
                            ("sensor", "sensor"), ("compound", "comp.")]]
    axA.legend(handles=handles, fontsize=6.0, loc="lower left", ncol=2, handletextpad=0.15,
               columnspacing=0.5, borderpad=0.22, labelspacing=0.18,
               framealpha=0.92, edgecolor="0.85")

    # ================= (b) counterfactual attribution =================
    names = ["$G'$ (observed)"] + [f"revert {LAB[k]}" for k in FAC]
    vals = np.concatenate([[so_obs], so_cf])
    cols = ["#D55E00"] + [OI[k] for k in FAC]
    ypos = np.arange(len(vals))[::-1]
    for y, v, c, k in zip(ypos, vals, cols, [None] + FAC):
        hl = (k == cause)
        axB.barh(y, v, height=0.66, color=c, alpha=1.0 if (hl or k is None) else 0.42,
                 edgecolor="black" if hl else "none", linewidth=0.6 if hl else 0, zorder=3)
    axB.set_yticks(ypos); axB.set_yticklabels(names, fontsize=6.8)
    axB.set_xscale("log"); axB.set_xlim(5, max(vals) * 6)
    axB.axvline(q95, color="0.25", ls=(0, (2.5, 1.8)), lw=0.8, zorder=4)
    axB.text(q95 * 1.25, ypos[0] + 0.42, "ID 95th pct.", fontsize=6.2, color="0.3",
             ha="left", va="center")
    axB.annotate(f"$A_{{\\mathrm{{{cause}}}}}={A[ci]:.2f}$\nback to ID level",
                 xy=(so_cf[ci], ypos[ci + 1]), xytext=(so_cf[ci] * 2.6, ypos[ci + 1] + 0.02),
                 fontsize=6.4, color=OI[cause], ha="left", va="center", zorder=6,
                 arrowprops=dict(arrowstyle="-", color=OI[cause], lw=0.5))
    axB.set_title("(b) attribute by counterfactual reversion", fontsize=8.2, pad=3)
    axB.set_xlabel("$S_{\\mathrm{OOD}}$ (log scale)", fontsize=7.0, labelpad=1.5)
    axB.tick_params(axis="x", labelsize=6.4, length=2, pad=1)
    axB.tick_params(axis="y", length=0, pad=1.5)
    axB.grid(axis="x", alpha=0.22, lw=0.4, zorder=0)
    axB.set_axisbelow(True)

    # ================= (c) physical consequence =================
    labels = ["$r_V$", "$r_F$", "CVI"]
    ro = np.array([Robs[1], Robs[2], Robs[3]]) * 100.0     # percent
    rc = np.array([Rcf[1], Rcf[2], Rcf[3]]) * 100.0
    x = np.arange(3); w = 0.36
    axC.bar(x - w / 2, ro, w, color="#D55E00", label="$G'$", zorder=3)
    axC.bar(x + w / 2, rc, w, color="#2E8B57", label=f"{LAB[cause]}\nreverted", zorder=3)
    top = max(ro.max(), rc.max())
    for xi, a, b in zip(x, ro, rc):
        axC.text(xi - w / 2, a + top * 0.03, f"{a:.1f}", ha="center", fontsize=6.0, color="0.25")
        axC.text(xi + w / 2, b + top * 0.03, f"{b:.1f}", ha="center", fontsize=6.0, color="0.25")
    axC.set_xticks(x); axC.set_xticklabels(labels, fontsize=7.6)
    axC.set_title("(c) constraint-violation index", fontsize=8.2, pad=3)
    axC.set_ylabel("violation rate (%)", fontsize=7.0, labelpad=2)
    axC.set_ylim(0, top * 1.52)
    axC.tick_params(labelsize=6.6, length=2, pad=1)
    axC.legend(fontsize=6.0, loc="upper center", ncol=2, handlelength=0.7, borderpad=0.2,
               labelspacing=0.15, handletextpad=0.25, columnspacing=0.6,
               framealpha=0.0, edgecolor="none", bbox_to_anchor=(0.5, 1.02))
    axC.grid(axis="y", alpha=0.22, lw=0.4, zorder=0); axC.set_axisbelow(True)

    fig.savefig(os.path.join(FIG, "fig1_framework.pdf"))
    fig.savefig(os.path.join(FIG, "fig1_framework.png"), dpi=300)
    print("wrote fig1_framework.pdf")


if __name__ == "__main__":
    import sys
    if "--plot-only" not in sys.argv:
        build()
    plot()
