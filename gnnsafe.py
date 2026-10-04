"""
GNNSafe (Wu et al., ICLR 2023, "Energy-based Out-of-Distribution Detection for Graph
Neural Networks") on the power-grid graphs.

GNNSafe is defined for node-level OOD detection with a trained node classifier: it takes
the classifier logits, forms the free energy E_v = -logsumexp(logits_v), and then applies
*energy belief propagation* over the graph,
    E^(k) = alpha * E^(k-1) + (1 - alpha) * D^{-1} A E^(k-1),
for K steps, using the propagated energy as the OOD score. GNNSafe++ adds a regularizer
that requires auxiliary OOD training data; we do not use it, since no OOD data is
available at training time in our setting.

Two adaptations are required and are stated in the paper:
  (1) a node-level task. We use bus-type classification (slack / generator / load), which
      is a genuine node label available from the case data and independent of any OOD
      information. The classifier is trained on ID graphs only.
  (2) graph-level aggregation. Our task is graph-level OOD detection, so the propagated
      node energies are pooled to a single score; we report mean pooling (max pooling is
      computed as well and is weaker).
"""
import os, sys, json, pickle, numpy as np
import scipy.sparse as sp, warnings
warnings.filterwarnings("ignore")
from framework import det_metrics
from gnn import build_prop

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- node labels: bus type from the case ----------------
def bus_labels(system):
    import gridgen as G
    net = G.base_net(system)
    N = len(net.bus)
    y = np.zeros(N, dtype=int)                       # 0 = load / PQ bus
    idx = {b: i for i, b in enumerate(net.bus.index)}
    for b in net.gen["bus"].values:
        y[idx[b]] = 1                                # 1 = generator (PV) bus
    for b in net.ext_grid["bus"].values:
        y[idx[b]] = 2                                # 2 = slack bus
    return y


# ---------------- NumPy GCN node classifier ----------------
class GCNClassifier:
    def __init__(self, d_in, n_cls, hid=64, seed=0):
        r = np.random.default_rng(seed)
        s = lambda a, b: r.normal(0, np.sqrt(2.0 / a), (a, b))
        self.W1 = s(d_in, hid); self.W2 = s(hid, n_cls); self.W3 = s(hid, n_cls)
        self.C = n_cls

    def _fwd(self, X, P):
        M0 = P @ X
        Z1 = M0 @ self.W1
        H1 = np.maximum(0.0, Z1)
        M1 = P @ H1
        logits = M1 @ self.W2 + H1 @ self.W3      # skip connection
        return logits, (M0, Z1, H1, M1, P)

    def train(self, graphs, y, epochs=400, lr=5e-3):
        onehot = np.eye(self.C)[y]
        params = [self.W1, self.W2, self.W3]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(1, epochs + 1):
            g_acc = [np.zeros_like(p) for p in params]
            for X, P in graphs:
                logits, (M0, Z1, H1, M1, P_) = self._fwd(X, P)
                z = logits - logits.max(1, keepdims=True)
                e = np.exp(z); sm = e / e.sum(1, keepdims=True)
                d_logits = (sm - onehot) / X.shape[0]
                dW2 = M1.T @ d_logits
                dW3 = H1.T @ d_logits
                dH1 = P_.T @ (d_logits @ self.W2.T) + d_logits @ self.W3.T
                dZ1 = dH1 * (Z1 > 0)
                dW1 = M0.T @ dZ1
                g_acc[0] += dW1; g_acc[1] += dW2; g_acc[2] += dW3
            n = len(graphs)
            for i, (p, g) in enumerate(zip(params, [g / n for g in g_acc])):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                p -= lr * (m[i] / (1 - b1 ** ep)) / (np.sqrt(v[i] / (1 - b2 ** ep)) + eps)
        return self

    def logits(self, X, P):
        return self._fwd(X, P)[0]

    def accuracy(self, graphs, y):
        acc = [np.mean(self.logits(X, P).argmax(1) == y) for X, P in graphs]
        return float(np.mean(acc))


# ---------------- GNNSafe scoring ----------------
def gnnsafe_score(clf, X, A, K=2, alpha=0.5, pool="mean"):
    """Free energy + K steps of energy belief propagation (GNNSafe, no OOD exposure)."""
    P = build_prop(A, "gcn", X)
    logits = clf.logits(X, P)
    mx = logits.max(1)
    E = -(mx + np.log(np.exp(logits - mx[:, None]).sum(1)))     # -logsumexp -> free energy
    deg = np.asarray(A.sum(1)).ravel().copy(); deg[deg == 0] = 1.0
    if sp.issparse(A):
        Arow = sp.diags(1.0 / deg) @ A                          # D^{-1} A
    else:
        Arow = A / deg[:, None]                                 # D^{-1} A
    for _ in range(K):
        E = alpha * E + (1 - alpha) * (Arow @ E)
    return float(E.mean() if pool == "mean" else E.max())


def run(system, seeds=(0, 1, 2, 3, 4)):
    data, _ = pickle.load(open(os.path.join(HERE, f"data/{system}.pkl"), "rb"))
    y = bus_labels(system)
    tr = [(s["Xo_obs"], build_prop(s["A"], "gcn", s["Xo_obs"])) for s in data["train"]]
    aur_mean, aur_max, accs = [], [], []
    full = {"mean": [], "max": []}          # full metric triples per seed, for reporting
    for sd in seeds:
        clf = GCNClassifier(data["train"][0]["Xo_obs"].shape[1], 3, seed=sd).train(tr, y)
        accs.append(clf.accuracy(tr, y))
        for pool, acc_list in [("mean", aur_mean), ("max", aur_max)]:
            sid = np.array([gnnsafe_score(clf, s["Xo_obs"], s["A"], pool=pool) for s in data["test_id"]])
            sod = np.array([gnnsafe_score(clf, s["Xo_obs"], s["A"], pool=pool) for s in data["ood"]])
            # OOD graphs may have LOWER mean energy; use the orientation that is consistent
            m = det_metrics(sid, sod)                         # canonical: higher energy = OOD
            full[pool].append(m)
            acc_list.append(m["AUROC"])
    # per-cell with the best pooling
    pool = "mean" if np.mean(aur_mean) >= np.mean(aur_max) else "max"
    clf = GCNClassifier(data["train"][0]["Xo_obs"].shape[1], 3, seed=0).train(tr, y)
    sid = np.array([gnnsafe_score(clf, s["Xo_obs"], s["A"], pool=pool) for s in data["test_id"]])
    groups = {}
    for s in data["ood"]:
        groups.setdefault(f'{s["meta"]["label"]}-{s["meta"]["band"]}', []).append(s)
    sod_all = np.concatenate([np.array([gnnsafe_score(clf, s["Xo_obs"], s["A"], pool=pool)
                                        for s in g]) for g in groups.values()])
    flip = det_metrics(-sid, -sod_all)["AUROC"] > det_metrics(sid, sod_all)["AUROC"]
    sgn = 1.0                                            # canonical orientation only
    per_cell = {}
    for c, g in groups.items():
        sc = sgn * np.array([gnnsafe_score(clf, s["Xo_obs"], s["A"], pool=pool) for s in g])
        per_cell[c] = round(det_metrics(sgn * sid, sc)["AUROC"], 4)
    ov = det_metrics(sgn * sid, sgn * sod_all)
    return {"node_acc": round(float(np.mean(accs)), 4),
            "pool": pool,
            "oracle_flip_would_help": bool(flip),
            "oracle_flipped_AUROC": round(det_metrics(-sid, -sod_all)["AUROC"], 4),
            "AUROC_mean_pool": [round(float(np.mean(aur_mean)), 4), round(float(np.std(aur_mean)), 4)],
            "AUROC_max_pool": [round(float(np.mean(aur_max)), 4), round(float(np.std(aur_max)), 4)],
            "overall_seed0": {k: round(v, 4) for k, v in ov.items()},
            "overall": {k: [round(float(np.mean([m[k] for m in full[pool]])), 4),
                            round(float(np.std([m[k] for m in full[pool]])), 4)]
                        for k in ("AUROC", "AUPR", "FPR95")},
            "by_cell": per_cell}


if __name__ == "__main__":
    out = {}
    for system in (sys.argv[1:] or ["case39", "case118"]):
        out[system] = run(system)
        print(system, json.dumps(out[system]), flush=True)
    p = os.path.join(HERE, "results/gnnsafe.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev.update(out); json.dump(prev, open(p, "w"), indent=2)
    print("saved", p)
