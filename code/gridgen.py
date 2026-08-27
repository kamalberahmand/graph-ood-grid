"""
Data + shift-taxonomy generator for Graph-OOD-in-power-grids.
Uses pandapower AC power flow on IEEE benchmark systems.

Each sample is a dict with:
  A        : (N,N) adjacency (0/1) of the *actual* topology used
  Xs       : (N,ds) structural node features
  Xo_true  : (N,do) TRUE operational node features from AC PF
  Xo_obs   : (N,do) OBSERVED operational features (Xo_true + sensor corruption)
  Rphys    : dict of physics-risk raw quantities and R_phys scalar
  meta     : label ('ID'/'topo'/'load'/'gen'/'sensor'/'compound'),
             band ('id'/'near'/'far'), injected magnitude, factor mask
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
import pandapower as pp
import pandapower.networks as pn

# ---- physics thresholds ----
# Voltage limits are taken per bus from the case data (min_vm_pu / max_vm_pu); the
# fallback band is only used if the case does not ship limits.
VMIN_FALLBACK, VMAX_FALLBACK = 0.94, 1.06
LOAD_MAX = 100.0                 # % branch thermal loading
BASE_OBS_NOISE = 0.01            # baseline measurement noise on all observed features (~PMU)
TARGET_BASE_LOADING = 60.0       # rescale placeholder ratings to this base-case max loading

FACTORS = ["topo", "load", "gen", "sensor"]

def base_net(system):
    if system == "case39":
        return pn.case39()
    if system == "case118":
        return pn.case118()
    raise ValueError(system)

def _prep_ratings(net):
    """IEEE-118 ships placeholder branch ratings (two values, ~9900 MVA), so its
    thermal-overload rate is identically zero at any plausible loading. We therefore
    define per-branch *reference* ratings such that the nominal operating point sits at
    TARGET_BASE_LOADING, and report loading relative to them. No power-flow parameter is
    modified (changing trafo sn_mva would change its impedance), so the AC solution is
    untouched. IEEE-39 ships real MATPOWER ratings and is left at scale 1."""
    pp.runpp(net, numba=True)
    ls = np.ones(len(net.line)); ts = np.ones(len(net.trafo))
    rescaled = False
    if len(net.line) and net.res_line.loading_percent.max() < 20.0:
        ls = np.clip(net.res_line["loading_percent"].values / TARGET_BASE_LOADING, 1e-4, None)
        rescaled = True
    if len(net.trafo) and net.res_trafo.loading_percent.max() < 20.0:
        ts = np.clip(net.res_trafo["loading_percent"].values / TARGET_BASE_LOADING, 1e-4, None)
        rescaled = True
    net["_line_lscale"] = ls
    net["_trafo_lscale"] = ts
    net["_ratings_rescaled"] = rescaled
    return rescaled


def _adj_from_net(net, N):
    A = np.zeros((N, N))
    bus_idx = {b: i for i, b in enumerate(net.bus.index)}
    bmap = np.vectorize(bus_idx.get)
    if len(net.line):
        on = net.line["in_service"].values.astype(bool)
        fb = bmap(net.line["from_bus"].values[on]); tb = bmap(net.line["to_bus"].values[on])
        A[fb, tb] = 1.0; A[tb, fb] = 1.0
    if len(net.trafo):
        on = net.trafo["in_service"].values.astype(bool)
        hv = bmap(net.trafo["hv_bus"].values[on]); lv = bmap(net.trafo["lv_bus"].values[on])
        A[hv, lv] = 1.0; A[lv, hv] = 1.0
    return A, bus_idx

def _connected(A):
    N = A.shape[0]
    seen = {0}; stack = [0]
    while stack:
        u = stack.pop()
        for v in np.where(A[u] > 0)[0]:
            if v not in seen:
                seen.add(v); stack.append(int(v))
    return len(seen) == N

def _structural_feats(A):
    """Degree + normalized-Laplacian spectral positional encodings (topology only)."""
    N = A.shape[0]
    deg = A.sum(1)
    d = deg.copy(); d[d == 0] = 1.0
    Dinv = np.diag(1.0 / np.sqrt(d))
    L = np.eye(N) - Dinv @ A @ Dinv
    w, V = np.linalg.eigh(L)
    k = min(8, N - 1)
    pe = V[:, 1:1 + k]                       # skip trivial eigenvector
    pe = pe * np.sign(pe.sum(0) + 1e-9)      # sign convention
    feats = np.concatenate([deg[:, None] / N, w[1:1 + k][None, :].repeat(N, 0), pe], axis=1)
    return feats

def _operational_feats(net, N, bus_idx):
    """True operational state from AC PF: |V|, angle(sin,cos), P_inj, Q_inj, mean incident loading."""
    # res_bus shares net.bus index order -> direct vectorized read
    vm = net.res_bus["vm_pu"].values.astype(float)
    va = np.deg2rad(net.res_bus["va_degree"].values.astype(float))
    p = net.res_bus["p_mw"].values.astype(float)
    q = net.res_bus["q_mvar"].values.astype(float)
    vm = np.nan_to_num(vm, nan=1.0); va = np.nan_to_num(va)
    p = np.nan_to_num(p); q = np.nan_to_num(q)
    load_on_bus = np.zeros(N); cnt = np.zeros(N)
    bmap = np.vectorize(bus_idx.get)
    if len(net.line):
        on = net.line["in_service"].values.astype(bool)
        lp = np.nan_to_num(net.res_line["loading_percent"].values.astype(float))[on]
        fb = bmap(net.line["from_bus"].values[on]); tb = bmap(net.line["to_bus"].values[on])
        np.add.at(load_on_bus, fb, lp); np.add.at(load_on_bus, tb, lp)
        np.add.at(cnt, fb, 1.0); np.add.at(cnt, tb, 1.0)
    cnt[cnt == 0] = 1
    load_on_bus /= cnt
    ps = p / (np.abs(p).max() + 1e-6)
    qs = q / (np.abs(q).max() + 1e-6)
    Xo = np.stack([vm, np.sin(va), np.cos(va), ps, qs, load_on_bus / 100.0], axis=1)
    return Xo, vm, load_on_bus

def _physics_risk(net, N, bus_idx):
    vm = net.res_bus["vm_pu"].values
    lo = net.bus["min_vm_pu"].values if "min_vm_pu" in net.bus else np.full(N, VMIN_FALLBACK)
    hi = net.bus["max_vm_pu"].values if "max_vm_pu" in net.bus else np.full(N, VMAX_FALLBACK)
    lo = np.nan_to_num(lo, nan=VMIN_FALLBACK); hi = np.nan_to_num(hi, nan=VMAX_FALLBACK)
    v_viol = np.mean((vm < lo) | (vm > hi))
    # branch loading includes transformers, not only lines, and is expressed relative
    # to the reference ratings established by _prep_ratings (see that function).
    ls = net.get("_line_lscale"); ts = net.get("_trafo_lscale")
    lp = net.res_line["loading_percent"].values
    if ls is not None and len(ls) == len(lp):
        lp = lp / ls
    if len(net.trafo):
        tl = net.res_trafo["loading_percent"].values
        if ts is not None and len(ts) == len(tl):
            tl = tl / ts
        lp = np.concatenate([lp, tl])
    lp = lp[~np.isnan(lp)]
    f_viol = np.mean(lp > LOAD_MAX) if len(lp) else 0.0
    # power balance mismatch (MW): gen - load - losses, normalized by total load
    pgen = net.res_gen["p_mw"].sum() + net.res_ext_grid["p_mw"].sum() if len(net.res_ext_grid) else net.res_gen["p_mw"].sum()
    pload = net.res_load["p_mw"].sum()
    ploss = net.res_line["pl_mw"].sum() + (net.res_trafo["pl_mw"].sum() if len(net.res_trafo) else 0.0)
    r_bal = abs(pgen - pload - ploss) / (abs(pload) + 1e-6)
    # continuous soft margins (for severity's physics term; smooth, not thresholded)
    v_margin = np.mean(np.maximum(0.0, np.maximum(lo - vm, vm - hi)))
    f_margin = np.mean(np.maximum(0.0, (lp - LOAD_MAX) / 100.0)) if len(lp) else 0.0
    return dict(r_bal=float(r_bal), r_V=float(v_viol), r_F=float(f_viol),
                v_margin=float(v_margin), f_margin=float(f_margin))

def _solve(net):
    """AC power flow with generator reactive-power limits enforced. A flat start is
    NOT used: it interacts badly with Q-limit enforcement and causes spurious
    divergence. Divergence is returned as such and counted, not silently retried."""
    try:
        pp.runpp(net, numba=True, enforce_q_lims=True, calculate_voltage_angles=True,
                 max_iteration=50)
        return bool(net.converged)
    except Exception:
        return False

# cache structural features by topology signature (eigh is the main cost)
_SFEAT_CACHE = {}
def _structural_feats_cached(A):
    key = hash(A.tobytes())
    v = _SFEAT_CACHE.get(key)
    if v is None:
        v = _structural_feats(A)
        _SFEAT_CACHE[key] = v
    return v

def _sample_id_operating(net, rng, gscale=None, per_load=0.03):
    """Apply an in-distribution operating point: global load scale ~U(0.9,1.1) + per-load noise."""
    if gscale is None:
        gscale = rng.uniform(0.9, 1.1)
    net.load["p_mw"] = net["_p0"] * gscale * (1 + rng.normal(0, per_load, len(net.load)))
    net.load["q_mvar"] = net["_q0"] * gscale * (1 + rng.normal(0, per_load, len(net.load)))
    return gscale

def _make(net, label, band, mag, factor_mask, rng, sensor_std=0.0, sensor_drop=0.0):
    N = len(net.bus)
    if not _solve(net):
        return None
    A, bus_idx = _adj_from_net(net, N)
    if not _connected(A):
        return None
    Xs = _structural_feats_cached(A)
    Xo_true, vm, _ = _operational_feats(net, N, bus_idx)
    Xo_obs = Xo_true.copy() + rng.normal(0, BASE_OBS_NOISE, Xo_true.shape)  # baseline PMU noise
    if sensor_std > 0 or sensor_drop > 0:
        Xo_obs = Xo_obs + rng.normal(0, sensor_std, Xo_obs.shape)
        if sensor_drop > 0:
            mask = rng.random(Xo_obs.shape[0]) < sensor_drop
            # dropped sensors imputed with ID nominal (vm=1, angle=0, others 0)
            Xo_obs[mask, 0] = 1.0; Xo_obs[mask, 1] = 0.0; Xo_obs[mask, 2] = 1.0
            Xo_obs[mask, 3:] = 0.0
    R = _physics_risk(net, N, bus_idx)
    R["R_phys"] = 0.5 * R["r_V"] + 0.5 * R["r_F"]
    return dict(A=A, Xs=Xs, Xo_true=Xo_true, Xo_obs=Xo_obs, Rphys=R,
                meta=dict(label=label, band=band, mag=float(mag), factor_mask=factor_mask))

def generate(system, n_id_train=300, n_id_test=150, n_each=120, seed=0):
    rng = np.random.default_rng(seed)
    net = base_net(system)                      # single persistent network, mutated in place
    _prep_ratings(net)
    net["_p0"] = net.load["p_mw"].values.copy()
    net["_q0"] = net.load["q_mvar"].values.copy()
    _g0 = net.gen["p_mw"].values.copy() if len(net.gen) else np.array([])
    N = len(net.bus)
    lines = list(net.line.index)
    ngen = len(net.gen)

    def fresh():
        # reset only the fields the perturbations touch -> back to nominal
        net.load["p_mw"] = net["_p0"]; net.load["q_mvar"] = net["_q0"]
        if len(net.gen): net.gen["p_mw"] = _g0
        net.line["in_service"] = True
        return net

    data = {"train": [], "test_id": [], "ood": []}

    # ---------- ID ----------
    made = 0
    while made < n_id_train:
        net = fresh(); _sample_id_operating(net, rng)
        s = _make(net, "ID", "id", 0.0, [0, 0, 0, 0], rng)
        if s: data["train"].append(s); made += 1
    made = 0
    while made < n_id_test:
        net = fresh(); _sample_id_operating(net, rng)
        s = _make(net, "ID", "id", 0.0, [0, 0, 0, 0], rng)
        if s: data["test_id"].append(s); made += 1

    reject = {}
    def add(gen_fn, label, band, target):
        cnt, tries = 0, 0
        while cnt < target and tries < target * 20:
            tries += 1
            s = gen_fn()
            if s: data["ood"].append(s); cnt += 1
        reject[f"{label}-{band}"] = round(1.0 - cnt / max(tries, 1), 4)
        return cnt

    # ---------- Topology OOD ----------
    def topo(band):
        net = fresh(); _sample_id_operating(net, rng)
        ndrop = 1 if band == "near" else int(rng.integers(2, 4))
        drop = rng.choice(lines, size=min(ndrop, len(lines)), replace=False)
        net.line.loc[drop, "in_service"] = False
        return _make(net, "topo", band, ndrop, [1, 0, 0, 0], rng)
    add(lambda: topo("near"), "topo", "near", n_each)
    add(lambda: topo("far"), "topo", "far", n_each)

    # ---------- Load OOD ----------
    def load(band):
        net = fresh()
        gs = rng.uniform(1.10, 1.18) if band == "near" else rng.uniform(1.20, 1.32)
        _sample_id_operating(net, rng, gscale=gs, per_load=0.08)
        return _make(net, "load", band, abs(gs - 1.0), [0, 1, 0, 0], rng)
    add(lambda: load("near"), "load", "near", n_each)
    add(lambda: load("far"), "load", "far", n_each)

    # ---------- Generation OOD ----------
    def gen(band):
        net = fresh(); _sample_id_operating(net, rng)
        # redispatch: perturb generator setpoints away from nominal (mimic altered mix / high renewable)
        p0 = net.gen["p_mw"].values.copy()
        amp = rng.uniform(0.15, 0.30) if band == "near" else rng.uniform(0.35, 0.60)
        pert = rng.normal(0, amp, len(p0))
        net.gen["p_mw"] = p0 * (1 + pert)
        if band == "far":  # zero-out a subset of generators (high-renewable displacement)
            k = max(1, int(0.25 * len(p0)))
            off = rng.choice(len(p0), size=k, replace=False)
            vals = net.gen["p_mw"].values.copy(); vals[off] = 0.0
            net.gen["p_mw"] = vals
        return _make(net, "gen", band, float(np.linalg.norm(pert) / np.sqrt(len(p0))), [0, 0, 1, 0], rng)
    add(lambda: gen("near"), "gen", "near", n_each)
    add(lambda: gen("far"), "gen", "far", n_each)

    # ---------- Sensor OOD ----------
    def sensor(band):
        net = fresh(); _sample_id_operating(net, rng)
        std = rng.uniform(0.02, 0.05) if band == "near" else rng.uniform(0.08, 0.15)
        drop = 0.0 if band == "near" else rng.uniform(0.1, 0.25)
        return _make(net, "sensor", band, std, [0, 0, 0, 1], rng, sensor_std=std, sensor_drop=drop)
    add(lambda: sensor("near"), "sensor", "near", n_each)
    add(lambda: sensor("far"), "sensor", "far", n_each)

    # ---------- Compound OOD (topology + load + generation) ----------
    def compound():
        net = fresh()
        gs = rng.uniform(1.10, 1.25)
        _sample_id_operating(net, rng, gscale=gs, per_load=0.08)
        drop = rng.choice(lines, size=rng.integers(1, 3), replace=False)
        net.line.loc[drop, "in_service"] = False
        p0 = net.gen["p_mw"].values.copy()
        net.gen["p_mw"] = p0 * (1 + rng.normal(0, 0.3, len(p0)))
        return _make(net, "compound", "far", abs(gs - 1.0) + 0.5, [1, 1, 1, 0], rng)
    add(compound, "compound", "far", n_each)

    data["reject_rate"] = reject
    return data, dict(N=N, n_lines=len(lines), n_gen=ngen, reject_rate=reject)

# ============================================================================
# Spec-based sample builder + counterfactual generation (for shift attribution)
# ============================================================================
def _apply_spec(net, spec, g0):
    """Configure the persistent net from a perturbation spec (loads, topology, gen)."""
    net.load["p_mw"] = net["_p0"]; net.load["q_mvar"] = net["_q0"]
    if len(net.gen): net.gen["p_mw"] = g0
    net.line["in_service"] = True
    # load
    gs = spec["gscale"]; ln = spec["load_noise"]
    net.load["p_mw"] = net["_p0"] * gs * (1 + ln)
    net.load["q_mvar"] = net["_q0"] * gs * (1 + ln)
    # topology
    if spec["drop_lines"]:
        net.line.loc[spec["drop_lines"], "in_service"] = False
    # generation
    if len(net.gen):
        gp = g0 * (1 + spec["gen_pert"])
        if len(spec["gen_off"]):
            gp = gp.copy(); gp[spec["gen_off"]] = 0.0
        net.gen["p_mw"] = gp

def _build_from_spec(net, spec, g0, rng_unused=None):
    """Solve and extract a graph dict from a spec; sensor corruption applied at observation."""
    _apply_spec(net, spec, g0)
    if not _solve(net):
        return None
    N = len(net.bus)
    A, bus_idx = _adj_from_net(net, N)
    if not _connected(A):
        return None
    Xs = _structural_feats_cached(A)
    Xo_true, vm, _ = _operational_feats(net, N, bus_idx)
    Xo_obs = Xo_true.copy() + spec.get("base_noise", np.zeros_like(Xo_true))[:Xo_true.shape[0]]
    if spec["sensor_std"] > 0 or spec["sensor_drop"] > 0:
        Xo_obs = Xo_obs + spec["sensor_noise"][:Xo_obs.shape[0]]
        m = spec["sensor_mask"][:Xo_obs.shape[0]]
        if m.any():
            Xo_obs[m, 0] = 1.0; Xo_obs[m, 1] = 0.0; Xo_obs[m, 2] = 1.0; Xo_obs[m, 3:] = 0.0
    R = _physics_risk(net, N, bus_idx)
    R["R_phys"] = 0.5 * R["r_V"] + 0.5 * R["r_F"]
    return dict(A=A, Xs=Xs, Xo_true=Xo_true, Xo_obs=Xo_obs, Rphys=R)

def _revert(spec, k, N, nominal_gscale=1.0, frac=1.0):
    """Return a copy of spec with factor k reverted toward the ID condition.

    frac=1 is the idealized reversion assumed by Proposition 1 (a perfectly known
    intervention). frac<1 models a MISSPECIFIED reversion operator, i.e. an operator
    that only partially knows or can only partially undo the intervention: a fraction
    (1-frac) of the intervention is left in place.
    """
    s = dict(spec)
    f = float(np.clip(frac, 0.0, 1.0))
    if k == "topo":
        drop = list(spec["drop_lines"])
        keep = int(round((1.0 - f) * len(drop)))       # leave this many lines still out
        s["drop_lines"] = drop[:keep]
    elif k == "load":
        s["gscale"] = spec["gscale"] + f * (nominal_gscale - spec["gscale"])
        s["load_noise"] = spec["load_noise"] * (1.0 - f)
    elif k == "gen":
        s["gen_pert"] = spec["gen_pert"] * (1.0 - f)
        s["gen_off"] = [] if f >= 1.0 else spec["gen_off"]
    elif k == "sensor":
        s["sensor_std"] = spec["sensor_std"] * (1.0 - f)
        s["sensor_drop"] = spec["sensor_drop"] * (1.0 - f)
        s["sensor_noise"] = spec["sensor_noise"] * (1.0 - f)
        s["sensor_mask"] = spec["sensor_mask"] & (np.random.default_rng(0).random(N) < (1.0 - f))
    return s

def generate_attrib(system, n_per=80, seed=7, fracs=(1.0, 0.75, 0.5, 0.25)):
    """Generate attribution test samples, each with its 4 single-factor counterfactuals.
    Scenarios: single topo / load / gen / sensor, plus compound(topo+load+gen)."""
    rng = np.random.default_rng(seed)
    net = base_net(system)
    _prep_ratings(net)
    net["_p0"] = net.load["p_mw"].values.copy(); net["_q0"] = net.load["q_mvar"].values.copy()
    g0 = net.gen["p_mw"].values.copy() if len(net.gen) else np.array([])
    N = len(net.bus); lines = list(net.line.index); ngen = len(net.gen)

    def base_spec():
        return dict(gscale=rng.uniform(0.95, 1.05), load_noise=rng.normal(0, 0.03, len(net.load)),
                    drop_lines=[], gen_pert=np.zeros(ngen), gen_off=[],
                    sensor_std=0.0, sensor_drop=0.0,
                    sensor_noise=np.zeros((N, 6)), sensor_mask=np.zeros(N, bool),
                    base_noise=rng.normal(0, BASE_OBS_NOISE, (N, 6)))

    def spec_topo(s):
        s["drop_lines"] = list(rng.choice(lines, size=int(rng.integers(1, 3)), replace=False))
    def spec_load(s):
        s["gscale"] = rng.uniform(1.10, 1.25); s["load_noise"] = rng.normal(0, 0.08, len(net.load))
    def spec_gen(s):
        s["gen_pert"] = rng.normal(0, rng.uniform(0.20, 0.40), ngen)
    def spec_sensor(s):
        std = rng.uniform(0.05, 0.12); s["sensor_std"] = std
        s["sensor_drop"] = rng.uniform(0.05, 0.2)
        s["sensor_noise"] = rng.normal(0, std, (N, 6))
        s["sensor_mask"] = rng.random(N) < s["sensor_drop"]

    scenarios = {"topo": [spec_topo], "load": [spec_load], "gen": [spec_gen],
                 "sensor": [spec_sensor], "compound": [spec_topo, spec_load, spec_gen]}
    out = []
    for name, fns in scenarios.items():
        made, tries = 0, 0
        while made < n_per and tries < n_per * 30:
            tries += 1
            s = base_spec()
            for fn in fns:
                fn(s)
            obs = _build_from_spec(net, s, g0)
            if obs is None:
                continue
            cfs = {}
            ok = True
            for fr in fracs:
                for k in FACTORS:
                    cf = _build_from_spec(net, _revert(s, k, N, frac=fr), g0)
                    if cf is None:
                        ok = False; break
                    cfs[(k, fr)] = cf
                if not ok:
                    break
            if not ok:
                continue
            active = [k for k in FACTORS if (k in name) or (name == "compound" and k in ("topo", "load", "gen"))]
            out.append(dict(scenario=name, active=active, obs=obs,
                            cf={k: cfs[(k, 1.0)] for k in FACTORS},   # backwards-compatible view
                            cf_frac=cfs, fracs=list(fracs)))
            made += 1
    return out, dict(N=N)


if __name__ == "__main__":
    import pickle, time, sys, os
    from collections import Counter
    HERE = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    systems = sys.argv[1:] or ["case39", "case118"]
    for sysname in systems:
        t = time.time()
        data, info = generate(sysname, n_id_train=250, n_id_test=150, n_each=100, seed=1)
        n_ood = len(data["ood"])
        c = Counter((s["meta"]["label"], s["meta"]["band"]) for s in data["ood"])
        print(f"{sysname}: N={info['N']} lines={info['n_lines']} gen={info['n_gen']} | "
              f"train={len(data['train'])} test_id={len(data['test_id'])} ood={n_ood} "
              f"({time.time()-t:.1f}s)", flush=True)
        print("   ood breakdown:", dict(c), flush=True)
        with open(os.path.join(HERE, f"data/{sysname}.pkl"), "wb") as f:
            pickle.dump((data, info), f)
        print(f"   saved data/{sysname}.pkl", flush=True)
        ta = time.time()
        attrib, ainfo = generate_attrib(sysname, n_per=40, seed=7)
        ca = Counter(a["scenario"] for a in attrib)
        print(f"   attrib samples: {len(attrib)} {dict(ca)} ({time.time()-ta:.1f}s)", flush=True)
        with open(os.path.join(HERE, f"data/{sysname}_attrib.pkl"), "wb") as f:
            pickle.dump((attrib, ainfo), f)
        print(f"   saved data/{sysname}_attrib.pkl", flush=True)
