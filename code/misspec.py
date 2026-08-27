"""
Attribution under a MISSPECIFIED reversion operator.

Proposition 1 shows top-1 attribution is exact when the reversion operator knows the
true intervention (assumption A1). In the simulator A1 holds by construction: reverting
an inactive factor is a literal no-op, so Delta_k = 0 exactly and any scoring function
would score 100%. That makes the idealized number uninformative about deployment.

Here the operator instead holds a *wrong nominal model*, controlled by delta:
  load   : believes nominal scale is 1+delta*u          (u ~ N(0,1), fixed per operator)
  gen    : believes nominal dispatch is g0*(1+delta*u)
  topo   : with prob. delta believes one extra line is out of service
  sensor : denoises imperfectly, residual noise delta*sigma
Every reversion therefore perturbs the graph, Delta_k for inactive factors is no longer
identically zero, and top-1 accuracy becomes a real measurement.
"""
import os, sys, json, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
from framework import Model
import gridgen as G

HERE = os.path.dirname(os.path.abspath(__file__))
FAC = ["topo", "load", "gen", "sensor"]
DELTAS = [0.0, 0.02, 0.05, 0.10, 0.20]


def revert_misspec(spec, k, N, net, rng, delta):
    """Revert factor k toward a nominal the operator believes in, which is wrong by delta."""
    s = dict(spec)
    if k == "topo":
        s["drop_lines"] = []
        if delta > 0 and rng.random() < min(1.0, delta * 3):
            cand = [l for l in net.line.index if l not in spec["drop_lines"]]
            if cand:
                s["drop_lines"] = [int(rng.choice(cand))]      # wrong switching status
    elif k == "load":
        s["gscale"] = 1.0 + delta * rng.normal()
        s["load_noise"] = delta * rng.normal(0, 1, len(spec["load_noise"])) * 0.3
    elif k == "gen":
        s["gen_pert"] = delta * rng.normal(0, 1, len(spec["gen_pert"])) * 0.5
        s["gen_off"] = []
    elif k == "sensor":
        s["sensor_std"] = spec["sensor_std"] * delta
        s["sensor_drop"] = spec["sensor_drop"] * delta
        s["sensor_noise"] = spec["sensor_noise"] * delta
        s["sensor_mask"] = spec["sensor_mask"] & (rng.random(N) < delta)
    return s


def run(system, n_per=25, seed=11):
    data, _ = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    M = Model(kind="gcn", seed=0).fit(data["train"])

    rng = np.random.default_rng(seed)
    net = G.base_net(system); G._prep_ratings(net)
    net["_p0"] = net.load["p_mw"].values.copy()
    net["_q0"] = net.load["q_mvar"].values.copy()
    g0 = net.gen["p_mw"].values.copy() if len(net.gen) else np.array([])
    N = len(net.bus); lines = list(net.line.index); ngen = len(net.gen)

    def base_spec():
        return dict(gscale=rng.uniform(0.95, 1.05),
                    load_noise=rng.normal(0, 0.03, len(net.load)),
                    drop_lines=[], gen_pert=np.zeros(ngen), gen_off=[],
                    sensor_std=0.0, sensor_drop=0.0,
                    sensor_noise=np.zeros((N, 6)), sensor_mask=np.zeros(N, bool),
                    base_noise=rng.normal(0, G.BASE_OBS_NOISE, (N, 6)))

    def mk(scn):
        s = base_spec()
        if scn == "topo":
            s["drop_lines"] = list(rng.choice(lines, size=int(rng.integers(1, 3)), replace=False))
        elif scn == "load":
            s["gscale"] = rng.uniform(1.10, 1.25); s["load_noise"] = rng.normal(0, 0.08, len(net.load))
        elif scn == "gen":
            s["gen_pert"] = rng.normal(0, rng.uniform(0.20, 0.40), ngen)
        elif scn == "sensor":
            std = rng.uniform(0.05, 0.12); s["sensor_std"] = std
            s["sensor_drop"] = rng.uniform(0.05, 0.2)
            s["sensor_noise"] = rng.normal(0, std, (N, 6))
            s["sensor_mask"] = rng.random(N) < s["sensor_drop"]
        return s

    acc = {f"{d}": [0, 0] for d in DELTAS}
    for scn in ["topo", "load", "gen", "sensor"]:
        made = tries = 0
        while made < n_per and tries < n_per * 12:
            tries += 1
            spec = mk(scn)
            obs = G._build_from_spec(net, spec, g0)
            if obs is None:
                continue
            so = M.s_ood(obs)
            ok_all = True
            votes = {}
            for d in DELTAS:
                r2 = np.random.default_rng(1000 + int(d * 1000) + tries)
                dl = {}
                for k in FAC:
                    cf = G._build_from_spec(net, revert_misspec(spec, k, N, net, r2, d), g0)
                    if cf is None:
                        ok_all = False; break
                    dl[k] = so - M.s_ood(cf)
                if not ok_all:
                    break
                votes[d] = max(dl, key=dl.get)
            if not ok_all:
                continue
            for d in DELTAS:
                acc[f"{d}"][1] += 1
                acc[f"{d}"][0] += int(votes[d] == scn)
            made += 1
    out = {f"{d}": round(acc[f"{d}"][0] / max(acc[f"{d}"][1], 1), 4) for d in DELTAS}
    out["n"] = acc[f"{DELTAS[0]}"][1]
    return out


if __name__ == "__main__":
    res = {}
    for system in (sys.argv[1:] or ["case39", "case118"]):
        res[system] = run(system)
        print(system, res[system], flush=True)
    p = os.path.join(HERE, "results/misspec.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev.update(res)
    json.dump(prev, open(p, "w"), indent=2)
    print("saved", p)
