"""
Screening prioritization: if only a fraction of the operating points can be sent to a full
N-1 screening, how many of the critical points does each base-case score find?
Critical = top 20% of the average post-contingency index (CVI^{N-1}_avg) on each system.
Also: states with at least one contingency without an AC solution (IEEE systems only;
on PEGASE every point has one). Reads results/n1_full.json, writes results/triage.json.
"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results/n1_full.json")))
SCORES = ["S_sev", "S_OOD", "D_p", "base"]


def recall_at(o, crit, b):
    k = int(round(b * len(o))); return float(crit[o[:k]].sum() / crit.sum())


def effort(o, crit, q=0.9):
    c = np.cumsum(crit[o]) / crit.sum(); return float((np.argmax(c >= q - 1e-12) + 1) / len(o))


out = {}
for s, D in R.items():
    rows = D["rows"]; n = len(rows)
    y = np.array([r["N1_mean"] for r in rows])
    crit = y >= np.quantile(y, 0.8)
    inf = np.array([r["n_div"] > 0 for r in rows])
    res = dict(n=n, n_crit=int(crit.sum()), n_inf=int(inf.sum()))
    for sc in SCORES:
        x = np.array([r[sc] for r in rows]); o = np.argsort(-x, kind="stable")
        e = dict(R20=recall_at(o, crit, 0.2), R10=recall_at(o, crit, 0.1), E90=effort(o, crit),
                 mass20=float(y[o[:int(round(0.2 * n))]].sum() / y.sum()))
        if 0 < inf.sum() < n:
            e["E90_inf"] = effort(o, inf)
        res[sc] = e
    out[s] = res
json.dump(out, open(os.path.join(HERE, "results/triage.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
