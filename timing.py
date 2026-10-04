"""
Computational cost (Section V-C): encoder fit time, per-graph scoring time (S_OOD and
S_sev), and the time of one Q-limited AC power flow. Writes results/timing.json.
Run with OMP_NUM_THREADS=1. Absolute numbers depend on the machine; the paper reports a
two-core CPU without GPU.
"""
import os, sys, json, pickle, time, numpy as np
from framework import Model
import gridgen as G

HERE = os.path.dirname(os.path.abspath(__file__))


def time_system(system, n_pf=20):
    data, _ = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    t0 = time.perf_counter(); M = Model(kind="gcn", seed=0).fit(data["train"]); fit = time.perf_counter() - t0
    ts = data["test_id"]
    t0 = time.perf_counter()
    for s in ts:
        M.s_ood(s); M.s_sev(s)
    score_ms = (time.perf_counter() - t0) / len(ts) * 1e3
    net = G.base_net(system); G.solve_diag(net)          # warm-up (numba compilation)
    t0 = time.perf_counter()
    for _ in range(n_pf):
        G.solve_diag(net)
    pf_ms = (time.perf_counter() - t0) / n_pf * 1e3
    return dict(fit_s=round(fit, 1), score_ms=round(score_ms, 1), pf_ms=round(pf_ms, 1))


if __name__ == "__main__":
    systems = sys.argv[1:] or ["case39", "case118", "case1354pegase"]
    p = os.path.join(HERE, "results/timing.json")
    out = json.load(open(p)) if os.path.exists(p) else {}
    for s in systems:
        out[s] = time_system(s); print(s, out[s])
    json.dump(out, open(p, "w"), indent=1)
