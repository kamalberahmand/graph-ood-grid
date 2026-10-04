"""
Full N-1 security screening (every single-branch outage, lines and transformers).

For each taxonomy cell we draw operating points with the same generator as the dataset,
compute the base-state scores (S_sev, S_OOD, D_p) and the base-state constraint-violation
index (CVI), and then screen EVERY single-branch outage of the network:

    CVI_c       constraint-violation index of the post-contingency AC solution
    secure_c    1 if the post-contingency state has no violation

Per operating point we report
    N1_mean     mean_c CVI_c  with a non-convergent contingency scored CVI_c = 1
    N1_mean_x   mean_c CVI_c  over convergent contingencies only
    N1_max      max_c  CVI_c  (worst single contingency, non-convergent counted as 1)
    N1_insec    fraction of contingencies with any violation (non-convergent counted)
    n_div       number of non-convergent contingencies, by diagnostic class

Non-convergence is diagnosed with the same retry ladder as the base-case generator
(gridgen.solve_diag): numerical non-convergence is recovered and kept, reactive-limit
infeasibility and collapse are scored as the worst outcome. Contingencies that island
the network are excluded from the set (they are reported separately), as is standard.
"""
import os, sys, json, pickle, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from framework import Model
import gridgen as G

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("ID", "id"), ("topo", "near"), ("topo", "far"), ("load", "near"), ("load", "far"),
         ("gen", "near"), ("gen", "far"), ("sensor", "near"), ("sensor", "far"),
         ("compound", "far")]


def contingency_set(net, N):
    """All single-branch outages that do not island the network."""
    cont, islanding = [], []
    for kind, tab in (("line", net.line), ("trafo", net.trafo)):
        for idx in tab.index:
            tab.at[idx, "in_service"] = False
            A, _ = G._adj_from_net(net, N)
            (cont if G._connected(A) else islanding).append((kind, int(idx)))
            tab.at[idx, "in_service"] = True
    return cont, islanding


def screen(net, cont, N, base_drop):
    risks, secure, classes = [], [], {c: 0 for c in G.SOLVE_CLASSES}
    for kind, idx in cont:
        tab = net.line if kind == "line" else net.trafo
        if not bool(tab.at[idx, "in_service"]):
            continue                                   # already out in this sample
        tab.at[idx, "in_service"] = False
        c = G.solve_diag(net)
        classes[c] += 1
        if c in ("ok", "ok_retry"):
            A, bi = G._adj_from_net(net, N)
            if G._connected(A):
                R = G._physics_risk(net, N, bi)
                r = 0.5 * R["r_V"] + 0.5 * R["r_F"]
                risks.append(r); secure.append(r == 0.0)
            else:                                      # islanded by the combined outage
                risks.append(1.0); secure.append(False)
        else:
            risks.append(1.0); secure.append(False)
        tab.at[idx, "in_service"] = True
        if c not in ("ok", "ok_retry"):
            G.solve_diag(net)                          # refresh the warm start after a failure
    risks = np.array(risks); conv = risks < 1.0
    n_div = int(classes["qlim_infeasible"] + classes["collapse"])
    return dict(N1_mean=float(risks.mean()),
                N1_mean_x=float(risks[conv].mean()) if conv.any() else 1.0,
                N1_max=float(risks.max()),
                N1_insec=float(1.0 - np.mean(secure)),
                n_cont=int(len(risks)), n_div=n_div,
                n_retry=int(classes["ok_retry"]), n_qlim=int(classes["qlim_infeasible"]),
                n_collapse=int(classes["collapse"]))


def run(system, n_per=25, seed=3, cells=None):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    data, info = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    M = Model(kind="gcn", seed=0).fit(data["train"])

    net = G.base_net(system); G._prep_ratings(net)
    net["_p0"] = net.load["p_mw"].values.copy(); net["_q0"] = net.load["q_mvar"].values.copy()
    g0 = net.gen["p_mw"].values.copy() if len(net.gen) else np.array([])
    N = len(net.bus); lines = list(net.line.index); ngen = len(net.gen)
    cont, islanding = contingency_set(net, N)
    print(f"[{system}] contingencies: {len(cont)} screened, {len(islanding)} islanding (excluded)",
          flush=True)

    def base_spec():
        return dict(gscale=G._lf(system, rng.uniform(0.9, 1.1)),
                    load_noise=rng.normal(0, 0.03, len(net.load)),
                    drop_lines=[], gen_pert=np.zeros(ngen), gen_off=[],
                    sensor_std=0.0, sensor_drop=0.0,
                    sensor_noise=np.zeros((N, 6)), sensor_mask=np.zeros(N, bool),
                    base_noise=rng.normal(0, G.BASE_OBS_NOISE, (N, 6)))

    def spec_for(cell):
        lab, band = cell
        s = base_spec()
        if lab == "topo":
            k = 1 if band == "near" else int(rng.integers(2, 4))
            s["drop_lines"] = list(rng.choice(lines, size=k, replace=False))
        elif lab == "load":
            s["gscale"] = G._lf(system, rng.uniform(1.10, 1.18) if band == "near"
                                else rng.uniform(1.20, 1.32))
            s["load_noise"] = rng.normal(0, 0.08, len(net.load))
        elif lab == "gen":
            amp = rng.uniform(0.15, 0.30) if band == "near" else rng.uniform(0.35, 0.60)
            s["gen_pert"] = rng.normal(0, amp, ngen)
        elif lab == "sensor":
            std = rng.uniform(0.02, 0.05) if band == "near" else rng.uniform(0.08, 0.15)
            s["sensor_std"] = std
            s["sensor_drop"] = 0.0 if band == "near" else rng.uniform(0.1, 0.25)
            s["sensor_noise"] = rng.normal(0, std, (N, 6))
            s["sensor_mask"] = rng.random(N) < s["sensor_drop"]
        elif lab == "compound":
            s["gscale"] = G._lf(system, rng.uniform(1.10, 1.25))
            s["load_noise"] = rng.normal(0, 0.08, len(net.load))
            s["drop_lines"] = list(rng.choice(lines, size=int(rng.integers(1, 3)), replace=False))
            s["gen_pert"] = rng.normal(0, 0.30, ngen)
        return s

    rows, per_cell = [], {}
    for cell in (cells or CELLS):
        got, tries, cell_rows = 0, 0, []
        while got < n_per and tries < n_per * 10:
            tries += 1
            sp_ = spec_for(cell)
            obs = G._build_from_spec(net, sp_, g0)
            if obs is None:
                continue
            base = obs["Rphys"]["R_phys"]
            sev = M.s_sev(obs)[0]; so = M.s_ood(obs); dp = M.s_phys_only(obs)
            G._apply_spec(net, sp_, g0)
            if not G._solve(net):
                continue
            r = screen(net, cont, N, sp_["drop_lines"])
            r.update(cell=f"{cell[0]}-{cell[1]}", base=base, S_sev=sev, S_OOD=so, D_p=dp)
            cell_rows.append(r); rows.append(r); got += 1
        if cell_rows:
            keys = ["base", "N1_mean", "N1_mean_x", "N1_max", "N1_insec", "n_div", "n_retry"]
            per_cell[f"{cell[0]}-{cell[1]}"] = {k: round(float(np.mean([x[k] for x in cell_rows])), 4)
                                               for k in keys}
            per_cell[f"{cell[0]}-{cell[1]}"]["n"] = len(cell_rows)
            c = per_cell[f"{cell[0]}-{cell[1]}"]
            print(f"  {cell[0]:9s}{cell[1]:5s} n={len(cell_rows):3d} base={c['base']:.4f} "
                  f"N1mean={c['N1_mean']:.4f} N1max={c['N1_max']:.4f} insec={c['N1_insec']:.3f} "
                  f"div/pt={c['n_div']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    def rho(a, b):
        return round(float(spearmanr([x[a] for x in rows], [x[b] for x in rows]).correlation), 4)
    out = dict(n_cont=len(cont), n_islanding=len(islanding), n_per=n_per,
               n_points=len(rows), by_cell=per_cell, runtime_s=round(time.time() - t0, 1))
    for score in ("S_sev", "S_OOD", "D_p", "base"):
        for tgt in ("N1_mean", "N1_mean_x", "N1_max", "N1_insec"):
            out[f"rho_{score}_{tgt}"] = rho(score, tgt)
    out["rows"] = [{k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
                    for k, v in r.items()} for r in rows]
    return out


def merge(system, parts):
    """Merge per-cell partial runs (same seed stream is NOT shared across parts, which is
    fine: parts are disjoint cells) into one record with correlations over all rows."""
    rows, by_cell = [], {}
    meta = None
    for part in parts:
        r = json.load(open(part))[system]
        rows += r["rows"]; by_cell.update(r["by_cell"]); meta = r
    out = dict(n_cont=meta["n_cont"], n_islanding=meta["n_islanding"], n_per=meta["n_per"],
               n_points=len(rows), by_cell=by_cell, rows=rows)
    for score in ("S_sev", "S_OOD", "D_p", "base"):
        for tgt in ("N1_mean", "N1_mean_x", "N1_max", "N1_insec"):
            out[f"rho_{score}_{tgt}"] = round(float(spearmanr([x[score] for x in rows],
                                                              [x[tgt] for x in rows]).correlation), 4)
    return out


if __name__ == "__main__":
    system = sys.argv[1]
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    if len(sys.argv) > 3 and sys.argv[3] == "merge":
        res = merge(system, sys.argv[4:])
        p = os.path.join(HERE, "results/n1_full.json")
    else:
        cells = None; tag = ""
        if len(sys.argv) > 3:                      # e.g. "0-4" -> cells 0..4
            a, b = map(int, sys.argv[3].split("-")); cells = CELLS[a:b + 1]; tag = f"_{a}-{b}"
        res = run(system, n_per=n_per, cells=cells)
        p = os.path.join(HERE, f"results/n1_full{tag}.json")
    print({k: v for k, v in res.items() if k.startswith("rho")}, flush=True)
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev[system] = res
    json.dump(prev, open(p, "w"), indent=1)
    print("saved", p)
