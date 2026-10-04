"""
Sensitivity and completeness analyses requested by review.

(a) Severity weights: AUROC and rho(S_sev, CVI) over a grid of (lambda_s, lambda_o, lambda_p),
    including single-term and leave-one-out settings.
(b) CVI composition: rho(score, CVI_w) for CVI_w = w r_V + (1-w) r_F, w in {0.25, 0.5, 0.75},
    and for the two terms alone.
(c) Reference-rating level: for systems with placeholder ratings, r_F recomputed with the
    reference loading at 50 / 60 / 70 / 80 % from the stored per-branch loadings, and
    rho(score, CVI) re-evaluated.
(d) Every detector's correlation with the CVI (not only S_OOD, D_p, S_sev), and the
    standalone detection AUROC of the severity components D_s and D_o.

All quantities use encoder seed 0 (the detection numbers in the main tables are five-seed
means; the sensitivity study is about the ordering of settings, not about the third
decimal) and are evaluated on the ID-test set plus every OOD cell.
"""
import os, sys, json, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from framework import Model, det_metrics
import gridgen as G

HERE = os.path.dirname(os.path.abspath(__file__))
LAM_GRID = [(1, 1, 2), (1, 1, 1), (1, 1, 4), (1, 1, 8), (1, 2, 2), (2, 1, 2),
            (0, 1, 2), (1, 0, 2), (1, 1, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
RATING_LEVELS = [50.0, 60.0, 70.0, 80.0]


def cvi_at_level(s, info, level):
    """Recompute r_F with the placeholder-class reference ratings placed at `level` %."""
    R = s["Rphys"]; lp = R["lp_rel"].astype(float)
    nl = info["n_lines"]; rc = info["rescaled_classes"]
    scale = np.ones_like(lp)
    # the reference rating at level T is L0/T, so loading relative to it is lp_rel_60 * T/60
    if rc["line"]:
        scale[:nl] = level / 60.0
    if rc["trafo"]:
        scale[nl:] = level / 60.0
    rF = float(np.mean(lp * scale > G.LOAD_MAX))
    return 0.5 * R["r_V"] + 0.5 * rF, rF


def run(system):
    data, info = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    M = Model(kind="gcn", seed=0).fit(data["train"])
    test_id, ood = data["test_id"], data["ood"]
    samp = test_id + ood
    y = np.array([0] * len(test_id) + [1] * len(ood))
    cvi = np.array([s["Rphys"]["R_phys"] for s in samp])
    rV = np.array([s["Rphys"]["r_V"] for s in samp]); rF = np.array([s["Rphys"]["r_F"] for s in samp])
    comps = np.array([M.s_sev(s)[1] for s in samp])        # (phi(Ds), phi(Do), phi(Dp))
    so = np.array([M.s_ood(s) for s in samp])
    dev = np.array([M.deviations(s) for s in samp])        # raw Ds, Do, Dp
    base = [M.baselines(s) for s in samp]
    out = {}

    # (a) severity weights
    a = {}
    for lam in LAM_GRID:
        l = np.array(lam, float)
        sev = (comps * l).sum(1) / l.sum()
        a[str(lam)] = dict(AUROC=round(det_metrics(sev[y == 0], sev[y == 1])["AUROC"], 4),
                           rho=round(float(spearmanr(sev, cvi).correlation), 4))
    out["severity_weights"] = a

    # (b) CVI composition
    sev = (comps * np.array([1, 1, 2.0])).sum(1) / 4.0
    b = {}
    for w in W_GRID:
        c = w * rV + (1 - w) * rF
        b[str(w)] = dict(rho_Ssev=round(float(spearmanr(sev, c).correlation), 4),
                         rho_Dp=round(float(spearmanr(dev[:, 2], c).correlation), 4),
                         rho_Sood=round(float(spearmanr(so, c).correlation), 4))
    out["cvi_weight"] = b

    # (c) rating level (only meaningful where a rating class is a placeholder)
    rc = info["rescaled_classes"]
    if rc["line"] or rc["trafo"]:
        c = {}
        for lev in RATING_LEVELS:
            cv = np.array([cvi_at_level(s, info, lev)[0] for s in samp])
            rf = np.array([cvi_at_level(s, info, lev)[1] for s in samp])
            c[str(lev)] = dict(rho_Ssev=round(float(spearmanr(sev, cv).correlation), 4),
                               rho_Dp=round(float(spearmanr(dev[:, 2], cv).correlation), 4),
                               rho_Sood=round(float(spearmanr(so, cv).correlation), 4),
                               mean_rF_ood=round(float(rf[y == 1].mean()), 4),
                               frac_ood_with_overload=round(float((rf[y == 1] > 0).mean()), 4))
        out["rating_level"] = c
    else:
        out["rating_level"] = "real ratings; not applicable"

    # (d) all detectors vs CVI, and standalone AUROC of the components
    d = {}
    names = ["Mahalanobis", "KNN", "Energy", "MSP", "Recon"]
    for nm in names:
        v = np.array([bb[nm] for bb in base])
        d[nm] = dict(AUROC=round(det_metrics(v[y == 0], v[y == 1])["AUROC"], 4),
                     rho=round(float(spearmanr(v, cvi).correlation), 4))
    for nm, col in (("D_s", 0), ("D_o", 1), ("D_p", 2)):
        v = dev[:, col]
        d[nm] = dict(AUROC=round(det_metrics(v[y == 0], v[y == 1])["AUROC"], 4),
                     rho=round(float(spearmanr(v, cvi).correlation), 4))
    d["S_OOD"] = dict(AUROC=round(det_metrics(so[y == 0], so[y == 1])["AUROC"], 4),
                      rho=round(float(spearmanr(so, cvi).correlation), 4))
    d["S_sev"] = dict(AUROC=round(det_metrics(sev[y == 0], sev[y == 1])["AUROC"], 4),
                      rho=round(float(spearmanr(sev, cvi).correlation), 4))
    try:
        A_nom = data["train"][0]["A"]
        ed = np.array([float(abs(s["A"] - A_nom).sum() / 2.0) for s in samp])
        d["EdgeDiff"] = dict(AUROC=round(det_metrics(ed[y == 0], ed[y == 1])["AUROC"], 4),
                             rho=round(float(spearmanr(ed, cvi).correlation), 4))
    except Exception:
        pass
    out["detectors_vs_cvi"] = d
    out["solve_diag"] = data.get("solve_diag", {})
    out["rescaled_classes"] = rc
    return out


if __name__ == "__main__":
    res = {}
    for system in (sys.argv[1:] or ["case39", "case118", "case1354pegase"]):
        res[system] = run(system)
        print(system, json.dumps(res[system]["detectors_vs_cvi"]), flush=True)
    p = os.path.join(HERE, "results/sensitivity.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev.update(res); json.dump(prev, open(p, "w"), indent=1)
    print("saved", p)
