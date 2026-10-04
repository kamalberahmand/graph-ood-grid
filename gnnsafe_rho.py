"""
rho(GNNSafe score, CVI) at seed 0 with the pooling selected in gnnsafe.json, for the
'all detectors vs consequence' table. Also records the standalone AUROC at seed 0.
"""
import os, sys, json, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from gnnsafe import GCNClassifier, bus_labels, gnnsafe_score, build_prop
from framework import det_metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def run(system):
    data, _ = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    g = json.load(open(os.path.join(HERE, "results/gnnsafe.json")))[system]
    pool = g["pool"]
    y = bus_labels(system)
    tr = [(s["Xo_obs"], build_prop(s["A"], "gcn", s["Xo_obs"])) for s in data["train"]]
    clf = GCNClassifier(data["train"][0]["Xo_obs"].shape[1], 3, seed=0).train(tr, y)
    samp = data["test_id"] + data["ood"]
    sc = np.array([gnnsafe_score(clf, s["Xo_obs"], s["A"], pool=pool) for s in samp])
    cvi = np.array([s["Rphys"]["R_phys"] for s in samp])
    lab = np.array([0] * len(data["test_id"]) + [1] * len(data["ood"]))
    return dict(pool=pool, rho=round(float(spearmanr(sc, cvi).correlation), 4),
                AUROC_seed0=round(det_metrics(sc[lab == 0], sc[lab == 1])["AUROC"], 4))


if __name__ == "__main__":
    p = os.path.join(HERE, "results/gnnsafe_rho.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    for system in (sys.argv[1:] or ["case39", "case118"]):
        prev[system] = run(system); print(system, prev[system], flush=True)
    json.dump(prev, open(p, "w"), indent=1); print("saved", p)
