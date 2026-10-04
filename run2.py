"""
Corrected experiment driver.

Fixes relative to the first version, all of which changed reported conclusions:
  * compound attribution is reported per factor (the previous "1/3 each" statistic was
    an arithmetic identity of the metric, not a measurement);
  * a parameter-free edge-difference detector is included, because a topology change is
    trivially visible in the adjacency matrix;
  * the operational-only Mahalanobis score is labelled as an ablation of our own model,
    not as an external baseline;
  * MSP/Energy are labelled as adapted (there is no classifier; pseudo-logits come from
    ID prototypes) rather than as the original methods;
  * attribution is evaluated under MISSPECIFIED reversion (partial undo), not only under
    the idealized reversion that makes Proposition 1 hold by construction;
  * detection and severity-risk numbers are averaged over 5 encoder seeds.
"""
import os, sys, json, pickle, time, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from framework import Model, det_metrics, struct_descriptor
import gridgen as G

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
FAC = ["topo", "load", "gen", "sensor"]
CELLS = [("topo", "near"), ("topo", "far"), ("load", "near"), ("load", "far"),
         ("gen", "near"), ("gen", "far"), ("sensor", "near"), ("sensor", "far"),
         ("compound", "far")]
SEEDS = [0, 1, 2, 3, 4]


def load(system):
    with open(os.path.join(HERE, f"data/{system}.pkl"), "rb") as f:
        data, info = pickle.load(f)
    with open(os.path.join(HERE, f"data/{system}_attrib.pkl"), "rb") as f:
        attrib, _ = pickle.load(f)
    return data, info, attrib


def edge_diff_score(s, A_nom):
    """Parameter-free detector: number of adjacency entries differing from nominal."""
    return float(abs(s["A"] - A_nom).sum() / 2.0)


def bootstrap_diff(x, y, z, n=2000, seed=0):
    """Bootstrap CI on rho(x,z) - rho(y,z) (paired)."""
    rng = np.random.default_rng(seed); m = len(z); out = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        out[i] = (spearmanr(x[idx], z[idx]).correlation
                  - spearmanr(y[idx], z[idx]).correlation)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run_system(system):
    t0 = time.time()
    data, info, attrib = load(system)
    train, test_id, ood = data["train"], data["test_id"], data["ood"]
    groups = defaultdict(list)
    for s in ood:
        groups[(s["meta"]["label"], s["meta"]["band"])].append(s)
    A_nom = train[0]["A"]

    res = {"system": system, "n_train": len(train), "n_id": len(test_id),
           "reject_rate": info.get("reject_rate", {})}

    # ---------------- detection, averaged over encoder seeds ----------------
    per_seed = defaultdict(lambda: defaultdict(list))     # method -> cell -> [auroc]
    overall = defaultdict(list)
    sev_risk = defaultdict(list)
    lam_sweep = defaultdict(list)
    sev_mag = defaultdict(list)
    for seed in SEEDS:
        M = Model(kind="gcn", seed=seed).fit(train)
        sc_id = {"Ours": np.array([M.s_ood(s) for s in test_id]),
                 "OperOnly": np.array([M.baselines(s)["Mahalanobis"] for s in test_id]),
                 "KNN": np.array([M.baselines(s)["KNN"] for s in test_id]),
                 "Energy": np.array([M.baselines(s)["Energy"] for s in test_id]),
                 "MSP": np.array([M.baselines(s)["MSP"] for s in test_id]),
                 "Recon": np.array([M.baselines(s)["Recon"] for s in test_id]),
                 "PhysOnly": np.array([M.s_phys_only(s) for s in test_id]),
                 "EdgeDiff": np.array([edge_diff_score(s, A_nom) for s in test_id])}
        sc_ood = {}
        for cell, g in groups.items():
            b = [M.baselines(s) for s in g]
            sc_ood[cell] = {
                "Ours": np.array([M.s_ood(s) for s in g]),
                "OperOnly": np.array([x["Mahalanobis"] for x in b]),
                "KNN": np.array([x["KNN"] for x in b]),
                "Energy": np.array([x["Energy"] for x in b]),
                "MSP": np.array([x["MSP"] for x in b]),
                "Recon": np.array([x["Recon"] for x in b]),
                "PhysOnly": np.array([M.s_phys_only(s) for s in g]),
                "EdgeDiff": np.array([edge_diff_score(s, A_nom) for s in g])}
        for meth in sc_id:
            for cell in CELLS:
                if cell in sc_ood:
                    per_seed[meth][f"{cell[0]}-{cell[1]}"].append(
                        det_metrics(sc_id[meth], sc_ood[cell][meth])["AUROC"])
            allo = np.concatenate([sc_ood[c][meth] for c in CELLS if c in sc_ood])
            m = det_metrics(sc_id[meth], allo)
            overall[meth].append((m["AUROC"], m["AUPR"], m["FPR95"]))

        # severity vs physical risk
        samp = test_id + ood
        Rp = np.array([s["Rphys"]["R_phys"] for s in samp])
        comps = np.array([M.s_sev(s)[1] for s in samp])
        Sv = np.array([M.s_sev(s)[0] for s in samp])
        So = np.array([M.s_ood(s) for s in samp])
        sev_risk["S_sev"].append(spearmanr(Sv, Rp).correlation)
        sev_risk["S_OOD"].append(spearmanr(So, Rp).correlation)
        for j, nm in enumerate(["D_s", "D_o", "D_p"]):
            sev_risk[nm].append(spearmanr(comps[:, j], Rp).correlation)
        for lp in [0.5, 1.0, 2.0, 4.0, 8.0]:
            v = (comps * np.array([1.0, 1.0, lp])).sum(1) / (2.0 + lp)
            lam_sweep[f"{lp}"].append(spearmanr(v, Rp).correlation)
        # severity vs injected magnitude, per family
        for fam in ["topo", "load", "gen", "sensor", "compound"]:
            g = [s for s in ood if s["meta"]["label"] == fam]
            if len(g) > 5 and np.std([s["meta"]["mag"] for s in g]) > 0:
                sev_mag[fam].append(spearmanr([M.s_sev(s)[0] for s in g],
                                              [s["meta"]["mag"] for s in g]).correlation)
        if seed == SEEDS[0]:
            M0, comps0, Rp0, Sv0, So0 = M, comps, Rp, Sv, So
            np.savez(os.path.join(HERE, f"results/{system}_scatter.npz"),
                     s_sev=Sv, s_ood=So, rphys=Rp,
                     fam=np.array(["ID"] * len(test_id) + [s["meta"]["label"] for s in ood]),
                     band=np.array(["id"] * len(test_id) + [s["meta"]["band"] for s in ood]))

    ms = lambda v: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
    res["detection_by_cell"] = {m: {c: ms(v) for c, v in d.items()} for m, d in per_seed.items()}
    res["detection_overall"] = {m: {"AUROC": ms([x[0] for x in v]),
                                    "AUPR": ms([x[1] for x in v]),
                                    "FPR95": ms([x[2] for x in v])} for m, v in overall.items()}
    res["severity_vs_risk"] = {k: ms(v) for k, v in sev_risk.items()}
    res["lambda_p_sweep"] = {k: ms(v) for k, v in lam_sweep.items()}
    res["severity_tracks_magnitude"] = {k: ms(v) for k, v in sev_mag.items()}
    lo, hi = bootstrap_diff(comps0[:, 2], Sv0, Rp0)
    res["Dp_minus_Ssev_rho_CI95"] = [round(lo, 4), round(hi, 4)]

    # ---------------- attribution, incl. misspecified reversion ----------------
    res["attribution"] = attribution(M0, attrib)
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def attribution(M, attrib):
    fracs = attrib[0].get("fracs", [1.0])
    out = {"fracs": fracs, "single_top1": {}, "compound_A": {}, "n": {}}
    for fr in fracs:
        ok = tot = 0
        for a in attrib:
            if a["scenario"] == "compound":
                continue
            so = M.s_ood(a["obs"])
            d = {k: so - M.s_ood(a["cf_frac"][(k, fr)]) for k in FAC}
            tot += 1
            ok += int(max(d, key=d.get) == a["active"][0])
        out["single_top1"][f"{fr}"] = round(ok / max(tot, 1), 4)
        out["n"][f"{fr}"] = tot
    # compound: real per-factor attribution mass (idealized reversion)
    rows = []
    for a in attrib:
        if a["scenario"] != "compound":
            continue
        so = M.s_ood(a["obs"])
        d = {k: max(0.0, so - M.s_ood(a["cf_frac"][(k, 1.0)])) for k in FAC}
        T = sum(d.values()) + 1e-12
        rows.append([d[k] / T for k in FAC])
    rows = np.array(rows)
    out["compound_A"] = {k: round(float(rows[:, i].mean()), 4) for i, k in enumerate(FAC)}
    out["compound_dominated_frac"] = round(float(np.mean(rows.max(1) > 0.9)), 4)
    out["compound_n"] = int(len(rows))
    # how often the counterfactual for an inactive factor is bit-identical to G'
    ident = tot = 0
    for a in attrib:
        for k in FAC:
            if k in a["active"]:
                continue
            tot += 1
            ident += int(np.array_equal(a["obs"]["Xo_obs"], a["cf_frac"][(k, 1.0)]["Xo_obs"])
                         and G.adj_equal(a["obs"]["A"], a["cf_frac"][(k, 1.0)]["A"]))
    out["inactive_bit_identical_frac"] = round(ident / max(tot, 1), 4)
    return out


if __name__ == "__main__":
    systems = sys.argv[1:] or ["case39", "case118"]
    allres = {}
    for system in systems:
        r = run_system(system)
        allres[system] = r
        with open(os.path.join(HERE, f"results/{system}_v2.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"[{system}] {r['runtime_s']}s")
        print("  overall AUROC:", {m: v["AUROC"] for m, v in r["detection_overall"].items()})
        print("  sev vs risk:", r["severity_vs_risk"])
        print("  attribution:", r["attribution"]["single_top1"],
              "compound_A:", r["attribution"]["compound_A"],
              "bit-identical:", r["attribution"]["inactive_bit_identical_frac"])
    p = os.path.join(HERE, "results/all_results_v2.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(allres)                       # merge: keep systems from earlier runs
    with open(p, "w") as f:
        json.dump(old, f, indent=2)
    print("saved", p)
