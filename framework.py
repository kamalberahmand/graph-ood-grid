"""
Framework: Detect -> Quantify -> Attribute -> Physics-risk, plus baselines & metrics.

Representation:  Z = [Zs, Zo]
  Zo  operational graph embedding from the trained NumPy GNN autoencoder (on Xo_obs)
  Zs  parameter-free structural graph descriptor (Laplacian spectrum + degree stats)
Detection:   S_OOD = Mahalanobis of [Zs,Zo] to ID (shrinkage covariance)
Severity:    S_sev = squash( lam_s*Ds + lam_o*Do + lam_p*Dp )   (adds physics term Dp)
Attribution: A_k from counterfactual factor reversion, Delta_k = S_OOD(G')-S_OOD(G'_-k)
Physics:     R_phys precomputed per sample; test corr(S_sev, R_phys)
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
from gnn import GNNAutoencoder, build_prop
from gridgen import adj_dense


# ---------------- structural descriptor ----------------
_SD_CACHE = {}
def _akey(A):
    import scipy.sparse as _sp
    return (A.indices.tobytes() + A.indptr.tobytes()) if _sp.issparse(A) else A.tobytes()

def struct_descriptor(A):
    """Spectral + degree descriptor of the topology; cached by adjacency, since the
    eigendecomposition dominates the cost and most samples share a topology."""
    k = _akey(A)
    v = _SD_CACHE.get(k)
    if v is not None:
        return v
    v = _struct_descriptor(A)
    if len(_SD_CACHE) < 20000:
        _SD_CACHE[k] = v
    return v

def _struct_descriptor(A):
    A = adj_dense(A)
    N = A.shape[0]
    deg = A.sum(1)
    d = deg.copy(); d[d == 0] = 1.0
    Dm = np.diag(1.0 / np.sqrt(d))
    L = np.eye(N) - Dm @ A @ Dm
    w = np.linalg.eigvalsh(L)
    lo = w[:10] if N >= 10 else np.pad(w, (0, 10 - N))
    hi = w[-5:]
    stats = np.array([deg.mean(), deg.std(), deg.max(), deg.min(), A.sum() / 2.0 / N])
    return np.concatenate([lo, hi, stats])


# ---------------- Mahalanobis with shrinkage ----------------
class Maha:
    def __init__(self, X):
        self.mu = X.mean(0)
        lw = LedoitWolf().fit(X)
        self.prec = lw.precision_
    def score(self, X):
        d = X - self.mu
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


def _phi(x, med):
    """Strictly increasing squashing phi(d) = d/(d+m) mapping [0,inf) -> [0,1).

    m is the ID median of the same deviation, so phi(m)=1/2. Unlike a min-max clip
    this never saturates exactly at 0 or 1, so it introduces no ties and preserves
    the ordering of deviations (required by Proposition 4)."""
    return x / (x + med + 1e-12)


# ---------------- physics deviation ----------------
def physics_vec(sample):
    R = sample["Rphys"]
    return np.array([R["r_bal"], R["v_margin"], R["f_margin"]])


# ---------------- full model ----------------
class Model:
    def __init__(self, kind="gcn", seed=0):
        self.kind = kind; self.seed = seed

    def fit(self, train, do_feats="Xo_obs"):
        self.do_feats = do_feats
        d_in = train[0][do_feats].shape[1]
        self.enc = GNNAutoencoder(d_in, hid=32, emb=16, kind=self.kind, seed=self.seed)
        graphs = [(s[do_feats], build_prop(s["A"], self.kind, s[do_feats])) for s in train]
        self.enc.train(graphs, epochs=250, lr=3e-3)
        # ID representations
        Zo = np.array([self._zo(s) for s in train])
        Zs = np.array([struct_descriptor(s["A"]) for s in train])
        Zj = np.concatenate([self._sscale(Zs), Zo], 1)
        Ph = np.array([physics_vec(s) for s in train])
        self.maha_o = Maha(Zo); self.maha_s = Maha(Zs); self.maha_j = Maha(Zj)
        # physics deviation reference (z-score to ID mean/std)
        self.pmu, self.pstd = Ph.mean(0), Ph.std(0) + 1e-9
        # baselines fit
        self.Zo_id = Zo
        self.km = KMeans(n_clusters=5, n_init=5, random_state=self.seed).fit(Zo)
        self.T = 1.0
        # calibration refs for severity squash (fit on ID)
        self.ref_ds = self.maha_s.score(Zs)
        self.ref_do = self.maha_o.score(Zo)
        self.ref_dp = np.linalg.norm((Ph - self.pmu) / self.pstd, axis=1)
        # ID medians used by the squashing phi(.) (no clipping, hence no ties)
        self.med_ds = float(np.median(self.ref_ds))
        self.med_do = float(np.median(self.ref_do))
        self.med_dp = float(np.median(self.ref_dp))
        return self

    def _sscale(self, Zs):
        if not hasattr(self, "_smu"):
            self._smu = Zs.mean(0); self._sstd = Zs.std(0) + 1e-9
        return (Zs - self._smu) / self._sstd

    def _zo(self, s):
        return self.enc.embed(s[self.do_feats], build_prop(s["A"], self.kind, s[self.do_feats]))

    # ---- representations for a sample ----
    def rep(self, s):
        Zo = self._zo(s); Zs = struct_descriptor(s["A"])
        return Zo, Zs

    def s_ood(self, s):
        Zo, Zs = self.rep(s)
        Zj = np.concatenate([self._sscale(Zs[None])[0], Zo])
        return float(self.maha_j.score(Zj[None])[0])

    def deviations(self, s):
        Zo, Zs = self.rep(s)
        Ds = float(self.maha_s.score(Zs[None])[0])
        Do = float(self.maha_o.score(Zo[None])[0])
        Ph = physics_vec(s)
        Dp = float(np.linalg.norm((Ph - self.pmu) / self.pstd))
        return Ds, Do, Dp

    def s_sev(self, s, lam=(1.0, 1.0, 2.0)):   # physics weighted higher: severity reflects consequence
        Ds, Do, Dp = self.deviations(s)
        ds = _phi(Ds, self.med_ds)
        do = _phi(Do, self.med_do)
        dp = _phi(Dp, self.med_dp)
        raw = lam[0] * ds + lam[1] * do + lam[2] * dp
        return float(raw / sum(lam)), (ds, do, dp)

    # ---- baselines (operate on Zo) ----
    def s_phys_only(self, s):
        """Physics deviation used on its own as a detector (fusion ablation)."""
        return self.deviations(s)[2]

    def baselines(self, s):
        Zo, _ = self.rep(s)
        maha = float(self.maha_o.score(Zo[None])[0])
        dists = np.linalg.norm(self.Zo_id - Zo, axis=1)
        knn = float(np.mean(np.sort(dists)[:10]))
        # prototype logits from ID clusters
        cd = np.linalg.norm(self.km.cluster_centers_ - Zo, axis=1) ** 2
        logits = -cd / (np.median(cd) + 1e-9)
        m = logits.max(); lse = m + np.log(np.exp(logits - m).sum())
        energy = float(-lse)                          # higher => more OOD
        sm = np.exp(logits - m); sm /= sm.sum()
        msp = float(1.0 - sm.max())                   # higher => more OOD
        recon = self.enc.recon_error(s[self.do_feats], build_prop(s["A"], self.kind, s[self.do_feats]))
        return dict(Mahalanobis=maha, KNN=knn, Energy=energy, MSP=msp, Recon=float(recon))


# ---------------- metrics ----------------
def fpr_at_tpr(y, score, tpr_target=0.95):
    pos = np.sort(score[y == 1])[::-1]
    thr = pos[min(len(pos) - 1, int(np.ceil(tpr_target * len(pos))) - 1)]
    neg = score[y == 0]
    return float(np.mean(neg >= thr))

def det_metrics(id_scores, ood_scores):
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return dict(AUROC=float(roc_auc_score(y, s)),
                AUPR=float(average_precision_score(y, s)),
                FPR95=fpr_at_tpr(y, s))
