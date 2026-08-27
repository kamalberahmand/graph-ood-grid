"""
N-1 security screening.

Base-case violations are not the criterion transmission operators actually use: an
operating point is acceptable only if it survives the loss of any single element. This
script evaluates, for a sample of graphs from each taxonomy cell, a post-contingency risk

    R_N1(G) = mean over screened single-line outages c of  R_phys(G | c),

with a diverged post-contingency case scored as R_phys = 1 (voltage collapse is the
worst outcome, not a missing value). We also record the N-1 insecurity rate, i.e. the
fraction of screened contingencies that produce any violation.

The question this answers is whether the severity score, which sees only the base-case
state, is informative about post-contingency security -- a strictly harder and more
operationally meaningful target than the base-case risk used in the main text.
"""
import os, sys, json, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from framework import Model
import gridgen as G

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("ID", "id"), ("topo", "near"), ("topo", "far"), ("load", "near"), ("load", "far"),
         ("gen", "near"), ("gen", "far"), ("sensor", "near"), ("sensor", "far"),
         ("compound", "far")]


def screen(net, cont_lines):
    """Return (mean post-contingency R_phys, insecurity rate, divergence rate)."""
    N = len(net.bus)
    risks, insec, div = [], 0, 0
    for l in cont_lines:
        if not bool(net.line.at[l, "in_service"]):
            continue                                  # already out in this sample
        net.line.at[l, "in_service"] = False
        ok = G._solve(net)
        if ok:
            A, bi = G._adj_from_net(net, N)
            R = G._physics_risk(net, N, bi)
            r = 0.5 * R["r_V"] + 0.5 * R["r_F"]
            risks.append(r); insec += int(r > 0)
        else:
            risks.append(1.0); insec += 1; div += 1
        net.line.at[l, "in_service"] = True
    n = max(len(risks), 1)
    return float(np.mean(risks)), insec / n, div / n


def rebuild(net, s, g0):
    """Restore the network to the operating point of sample s (loads/gen/topology)."""
    net.load["p_mw"] = s["_spec"]["p"]; net.load["q_mvar"] = s["_spec"]["q"]
    if len(net.gen):
        net.gen["p_mw"] = s["_spec"]["g"]
    net.line["in_service"] = True
    if s["_spec"]["drop"]:
        net.line.loc[s["_spec"]["drop"], "in_service"] = False


def run(system, n_per=25, n_cont=30, seed=3):
    """Re-derive operating points by re-simulating from the stored adjacency + features is
    not possible, so we regenerate matched samples with the same generator and screen those."""
    rng = np.random.default_rng(seed)
    data, _ = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    M = Model(kind="gcn", seed=0).fit(data["train"])

    net = G.base_net(system); G._prep_ratings(net)
    net["_p0"] = net.load["p_mw"].values.copy(); net["_q0"] = net.load["q_mvar"].values.copy()
    g0 = net.gen["p_mw"].values.copy() if len(net.gen) else np.array([])
    N = len(net.bus); lines = list(net.line.index); ngen = len(net.gen)
    cont = list(rng.choice(lines, size=min(n_cont, len(lines)), replace=False))

    def base_spec():
        return dict(gscale=rng.uniform(0.9, 1.1),
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
            s["gscale"] = rng.uniform(1.10, 1.18) if band == "near" else rng.uniform(1.20, 1.32)
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
            s["gscale"] = rng.uniform(1.10, 1.25)
            s["load_noise"] = rng.normal(0, 0.08, len(net.load))
            s["drop_lines"] = list(rng.choice(lines, size=int(rng.integers(1, 3)), replace=False))
            s["gen_pert"] = rng.normal(0, 0.30, ngen)
        return s

    per_cell, sev_all, rn1_all, base_all, sood_all = {}, [], [], [], []
    for cell in CELLS:
        got, tries, rows = 0, 0, []
        while got < n_per and tries < n_per * 10:
            tries += 1
            sp = spec_for(cell)
            obs = G._build_from_spec(net, sp, g0)
            if obs is None:
                continue
            base_r = obs["Rphys"]["R_phys"]
            sev = M.s_sev(obs)[0]; so = M.s_ood(obs)
            G._apply_spec(net, sp, g0)                # restore this operating point
            if not G._solve(net):
                continue
            rn1, insec, dv = screen(net, cont)
            rows.append((sev, so, base_r, rn1, insec, dv)); got += 1
        if not rows:
            continue
        a = np.array(rows)
        per_cell[f"{cell[0]}-{cell[1]}"] = dict(
            n=len(rows), R_N1=round(float(a[:, 3].mean()), 4),
            insecure=round(float(a[:, 4].mean()), 4),
            cont_div=round(float(a[:, 5].mean()), 4),
            R_base=round(float(a[:, 2].mean()), 4))
        sev_all += list(a[:, 0]); sood_all += list(a[:, 1])
        base_all += list(a[:, 2]); rn1_all += list(a[:, 3])
        print(f"  {cell[0]:9s}{cell[1]:5s} n={len(rows):3d} R_base={a[:,2].mean():.4f} "
              f"R_N1={a[:,3].mean():.4f} insecure={a[:,4].mean():.3f}", flush=True)

    sev_all = np.array(sev_all); rn1_all = np.array(rn1_all)
    sood_all = np.array(sood_all); base_all = np.array(base_all)
    out = dict(n_cont=len(cont), n_per=n_per, by_cell=per_cell,
               rho_Ssev_RN1=round(float(spearmanr(sev_all, rn1_all).correlation), 4),
               rho_Sood_RN1=round(float(spearmanr(sood_all, rn1_all).correlation), 4),
               rho_Rbase_RN1=round(float(spearmanr(base_all, rn1_all).correlation), 4),
               rho_Ssev_Rbase=round(float(spearmanr(sev_all, base_all).correlation), 4))
    return out


if __name__ == "__main__":
    res = {}
    for system in (sys.argv[1:] or ["case39", "case118"]):
        print(system, flush=True)
        res[system] = run(system)
        print(" ", {k: v for k, v in res[system].items() if k.startswith("rho")}, flush=True)
    p = os.path.join(HERE, "results/n1.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev.update(res); json.dump(prev, open(p, "w"), indent=2)
    print("saved", p)
