"""
Detection ability of the three scores that Table A3 and Figure 2(a) compare.

run2.py evaluates the detection score and the comparators; it does not evaluate the
severity as a detector, because severity is not proposed as one. The trade-off claim of
the paper, however, needs exactly that quantity: how well each of S_OOD, S_sev and D_p
separates ID from OOD, paired with how well the same quantity ranks physical risk. This
script computes the detection half, over the same five encoder seeds as run2.py, so the
two halves of the trade-off are measured on the same footing.

Output: results/dual_role.json, read by fig3_build.py and quoted in Table A3.
"""
import os, sys, json, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
from framework import Model, det_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]


def run(system):
    data, _ = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    test_id, ood = data["test_id"], data["ood"]
    acc = {"S_OOD": [], "S_sev": [], "PhysOnly": []}
    for sd in SEEDS:
        M = Model(kind="gcn", seed=sd).fit(data["train"])
        sid = {"S_OOD": np.array([M.s_ood(s) for s in test_id]),
               "S_sev": np.array([M.s_sev(s)[0] for s in test_id]),
               "PhysOnly": np.array([M.s_phys_only(s) for s in test_id])}
        sod = {"S_OOD": np.array([M.s_ood(s) for s in ood]),
               "S_sev": np.array([M.s_sev(s)[0] for s in ood]),
               "PhysOnly": np.array([M.s_phys_only(s) for s in ood])}
        for k in acc:
            acc[k].append(det_metrics(sid[k], sod[k])["AUROC"])
    out = {}
    for k, v in acc.items():
        out[f"{k}_AUROC"] = round(float(np.mean(v)), 4)
        out[f"{k}_AUROC_std"] = round(float(np.std(v)), 4)
    return out


if __name__ == "__main__":
    res = {}
    for system in (sys.argv[1:] or ["case39", "case118"]):
        res[system] = run(system)
        print(system, json.dumps(res[system]), flush=True)
    p = os.path.join(HERE, "results/dual_role.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev.update(res)
    json.dump(prev, open(p, "w"), indent=2)
    print("saved", p)
