"""
Lightweight NumPy graph neural network encoder (no torch/jax available).
A 2-layer GNN autoencoder trained self-supervised (feature reconstruction).
Propagation operator P is pluggable -> GCN / GraphSAGE(mean) / GIN(sum) / GAT(attention).
Graphs are tiny (39/118 nodes) so dense linear algebra is fast and exact.
"""
import numpy as np
import scipy.sparse as sp


_P_CACHE = {}

def build_prop(A, kind="gcn", Xfeat=None):
    """Return an (N,N) propagation operator from adjacency A (dense, or CSR for large
    systems; the sparse path implements the same symmetric-normalized GCN operator).
    Operators that do not depend on the features are cached by adjacency."""
    if kind in ("gcn", "sage", "gin"):
        key = (kind, (A.indices.tobytes() + A.indptr.tobytes()) if sp.issparse(A) else A.tobytes())
        P = _P_CACHE.get(key)
        if P is None:
            P = _build_prop(A, kind, Xfeat)
            if len(_P_CACHE) < 5000:
                _P_CACHE[key] = P
        return P
    return _build_prop(A, kind, Xfeat)

def _build_prop(A, kind="gcn", Xfeat=None):
    N = A.shape[0]
    if sp.issparse(A):
        if kind != "gcn":
            raise ValueError("sparse propagation implemented for the GCN operator only")
        Ah = (A + sp.eye(N, format="csr")).tocsr()
        d = np.asarray(Ah.sum(1)).ravel(); d[d == 0] = 1.0
        Dm = sp.diags(1.0 / np.sqrt(d))
        return (Dm @ Ah @ Dm).tocsr()
    I = np.eye(N)
    if kind == "gcn":                       # symmetric-normalized, self loops
        Ah = A + I
        d = Ah.sum(1); d[d == 0] = 1.0
        Dm = np.diag(1.0 / np.sqrt(d))
        return Dm @ Ah @ Dm
    if kind == "sage":                      # mean aggregation incl. self (row-normalized)
        Ah = A + I
        d = Ah.sum(1); d[d == 0] = 1.0
        return Ah / d[:, None]
    if kind == "gin":                       # sum aggregation, scaled to control magnitude
        Ah = A + I
        return Ah / max(1.0, np.sqrt(A.sum(1).mean() + 1.0))
    if kind == "gat":                       # attention over neighbours (cosine-softmax on input feats)
        H = Xfeat if Xfeat is not None else np.ones((N, 1))
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
        S = Hn @ Hn.T
        M = (A + I) > 0
        S = np.where(M, S, -1e9)
        S = S - S.max(1, keepdims=True)
        E = np.exp(S) * M
        return E / (E.sum(1, keepdims=True) + 1e-8)
    raise ValueError(kind)


def _relu(x): return np.maximum(0.0, x)


class GNNAutoencoder:
    """2-layer GNN encoder + 1-layer decoder, trained by feature reconstruction.
    Forward (per graph): H1=relu(P X W1); Z=P H1 W2; Xr=P Z W3.  Graph emb = [mean(Z),std(Z)]."""

    def __init__(self, d_in, hid=32, emb=16, kind="gcn", seed=0):
        r = np.random.default_rng(seed)
        s = lambda a, b: r.normal(0, np.sqrt(2.0 / a), (a, b))
        self.W1 = s(d_in, hid); self.W2 = s(hid, emb); self.W3 = s(emb, d_in)
        self.kind = kind; self.d_in = d_in; self.emb = emb

    def _fwd(self, X, P):
        M0 = P @ X
        Z1 = M0 @ self.W1; H1 = _relu(Z1)
        M1 = P @ H1
        Z = M1 @ self.W2
        M2 = P @ Z
        Xr = M2 @ self.W3
        cache = (X, P, M0, Z1, H1, M1, Z, M2)
        return Xr, Z, cache

    def _grads(self, cache, Xr):
        X, P, M0, Z1, H1, M1, Z, M2 = cache
        N = X.shape[0]
        G = 2.0 * (Xr - X) / N
        dW3 = M2.T @ G
        dZ = P.T @ (G @ self.W3.T)
        dW2 = M1.T @ dZ
        dH1 = P.T @ (dZ @ self.W2.T)
        dZ1 = dH1 * (Z1 > 0)
        dW1 = M0.T @ dZ1
        return dW1, dW2, dW3

    def train(self, graphs, epochs=250, lr=3e-3, verbose=False):
        """graphs: list of (X, P). Full-batch Adam over reconstruction MSE."""
        params = [self.W1, self.W2, self.W3]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for ep in range(1, epochs + 1):
            g_acc = [np.zeros_like(p) for p in params]; loss = 0.0
            for X, P in graphs:
                Xr, _, cache = self._fwd(X, P)
                loss += float(np.mean((Xr - X) ** 2))
                for a, g in zip(g_acc, self._grads(cache, Xr)):
                    a += g
            n = len(graphs)
            g_acc = [g / n for g in g_acc]
            for i, (p, g) in enumerate(zip(params, g_acc)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                mh = m[i] / (1 - b1 ** ep); vh = v[i] / (1 - b2 ** ep)
                p -= lr * mh / (np.sqrt(vh) + eps)
            if verbose and ep % 50 == 0:
                print(f"    epoch {ep:3d}  recon MSE {loss/n:.5f}", flush=True)
        return self

    def embed(self, X, P):
        _, Z, _ = self._fwd(X, P)
        return np.concatenate([Z.mean(0), Z.std(0)])

    def recon_error(self, X, P):
        Xr, _, _ = self._fwd(X, P)
        return float(np.mean((Xr - X) ** 2))
