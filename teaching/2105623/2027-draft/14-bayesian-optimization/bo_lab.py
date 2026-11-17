"""2105623 Session 14: Bayesian optimization and AI-assisted modeling (2027 draft).

Part A  Gaussian-process surrogates: the Matern 5/2 kernel, the posterior by hand and in code, hyperparameters by
        the marginal likelihood (a grid search, so every run gives the same answer)
Part B  Bayesian optimization: expected improvement and the upper confidence bound on a 1-D test function
        (Forrester et al., 2008, sign changed); BO, random search and a grid on Session 5's six-hump camel
Part C  AI-assisted modeling: a shipping LP written by a language model with three planted errors, and the tests
        that find them
Lab 14  (a) A hidden, noisy CSTR yield (synthetic kinetics, A -> B -> C with the side reaction A -> D) by a grid,
        random search and BO with 20 experiments; (b) a batch-plant schedule (MILP) written by a language model
        with three planted errors, and eight tests

Runs in desktop Python and in the browser (NumPy and SciPy only).
Notation: m(x) is the posterior mean, s(x) the posterior standard deviation, ell a length-scale, g the noise
variance as a fraction of the signal variance, kappa the weight of s(x) in the upper confidence bound.
All inputs are scaled to [0, 1] before the GP sees them; outputs are standardized.
"""
import json, math, sys
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.stats import norm


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


# ---------------------------------------------------------------------------
# Part A. Gaussian-process regression
# ---------------------------------------------------------------------------
SQ5 = math.sqrt(5.0)
ELL = np.geomspace(0.05, 2.0, 12)                        # length-scales tried, per input (scaled units)
GN = np.array([1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3])  # noise variance / signal variance tried


def k52(A, B, ell):
    """Matern 5/2 correlation between the rows of A and B (inputs scaled to [0, 1])."""
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    D = np.sqrt((((A[:, None, :] - B[None, :, :]) / ell) ** 2).sum(-1))
    return (1 + SQ5 * D + 5.0 / 3.0 * D * D) * np.exp(-SQ5 * D)


def log_ml(X, z, ell, g):
    """Log marginal likelihood of standardized data z, with the signal variance profiled out."""
    n = len(z)
    L = np.linalg.cholesky(k52(X, X, ell) + (g + 1e-10) * np.eye(n))
    a = np.linalg.solve(L, z)
    sf2 = a @ a / n
    return -0.5 * n * math.log(sf2) - np.log(np.diag(L)).sum() - 0.5 * n * (1 + math.log(2 * math.pi))


def gp_fit(X, y, ell=None, g=None):
    """Fit a GP. Without ell and g, pick them by the marginal likelihood over the grids ELL (per input) and GN."""
    X = np.atleast_2d(np.asarray(X, float)); y = np.asarray(y, float); n, d = X.shape
    yb = y.mean(); ys = y.std() or 1.0; z = (y - yb) / ys
    if ell is None:
        E = np.array(np.meshgrid(*([ELL] * d), indexing="ij")).reshape(d, -1).T
        G = GN if g is None else np.array([g])
        D = np.sqrt((((X[:, None, :] - X[None, :, :])[None] / E[:, None, None, :]) ** 2).sum(-1))
        K = (1 + SQ5 * D + 5.0 / 3.0 * D * D) * np.exp(-SQ5 * D)
        Kg = K[None] + (G[:, None, None, None] + 1e-10) * np.eye(n)
        L = np.linalg.cholesky(Kg)
        a = np.linalg.solve(L, np.broadcast_to(z[:, None], Kg.shape[:-1] + (1,)))[..., 0]
        sf2 = (a * a).sum(-1) / n
        ll = -0.5 * n * np.log(sf2) - np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1) - 0.5 * n * (1 + math.log(2 * math.pi))
        ig, ie = np.unravel_index(int(np.argmax(ll)), ll.shape)
        ell, g = E[ie], float(G[ig])
    ell = np.broadcast_to(np.asarray(ell, float), (d,)).copy()
    L = np.linalg.cholesky(k52(X, X, ell) + (g + 1e-10) * np.eye(n))
    a = np.linalg.solve(L, z); alpha = np.linalg.solve(L.T, a)
    return {"X": X, "y": y, "yb": yb, "ys": ys, "ell": ell, "g": g, "sf2": float(a @ a / n), "L": L, "alpha": alpha,
            "ll": log_ml(X, z, ell, g)}


def gp_pred(M, Xs):
    """Posterior mean m and standard deviation s of the latent function at the rows of Xs."""
    Ks = k52(np.atleast_2d(np.asarray(Xs, float)), M["X"], M["ell"])
    m = M["yb"] + M["ys"] * (Ks @ M["alpha"])
    V = np.linalg.solve(M["L"], Ks.T)
    v = M["ys"] ** 2 * M["sf2"] * np.maximum(1.0 - (V * V).sum(0), 0.0)
    return m, np.sqrt(v)


def ei(m, s, best):
    """Expected improvement over best, for maximization."""
    m, s = np.asarray(m, float), np.asarray(s, float); imp = m - best
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(s > 1e-12, imp / s, 0.0)
        return np.where(s > 1e-12, imp * norm.cdf(z) + s * norm.pdf(z), np.maximum(imp, 0.0))


def lhs(n, d, rng):
    """Latin hypercube: one point in each of n slices of every input."""
    return np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(d)])


def cand_grid(d, n):
    g1 = np.linspace(0, 1, n)
    return np.array(np.meshgrid(*([g1] * d), indexing="ij")).reshape(d, -1).T


def bo(f, d, budget, seed, n0=6, sigma=0.0, acq="ei", kappa=2.0, X0=None, ncand=41, record=False):
    """Bayesian optimization (maximize): n0 Latin-hypercube points (or X0), then the best point of the
    acquisition on a grid of candidates. Noise of standard deviation sigma is added to every experiment."""
    rng = np.random.default_rng(seed)
    X = [np.asarray(x, float) for x in X0] if X0 is not None else list(lhs(n0, d, rng))
    y = [float(f(x) + sigma * rng.standard_normal()) for x in X]
    C = cand_grid(d, ncand); steps = []
    while len(y) < budget:
        M = gp_fit(X, y); m, s = gp_pred(M, C)
        best = float(gp_pred(M, np.array(X))[0].max())
        a = ei(m, s, best) if acq == "ei" else m + kappa * s
        k = int(np.argmax(a)); x = C[k]
        if record:
            steps.append({"n": len(y), "ell": M["ell"].tolist(), "g": M["g"], "best": best, "next": x.tolist(),
                          "m": float(m[k]), "s": float(s[k]), "acq": float(a[k])})
        X.append(x); y.append(float(f(x) + sigma * rng.standard_normal()))
    out = {"X": np.array(X), "y": np.array(y)}
    if record:
        out["steps"] = steps
    return out


# ---------------------------------------------------------------------------
# Lecture examples for Parts A and B
# ---------------------------------------------------------------------------
def forrester(x):
    """Forrester et al. (2008) 1-D test function, sign changed so that it is maximized; x in [0, 1]."""
    x = float(np.asarray(x).ravel()[0])
    return -(6 * x - 2) ** 2 * math.sin(12 * x - 4)


def by_hand():
    """Posterior at one new point from two observations, with the correlations given (prior mean 0, variance 1)."""
    K = np.array([[1.0, 0.5], [0.5, 1.0]]); y = np.array([1.0, 2.0]); ks = np.array([0.8, 0.3])
    w = np.linalg.solve(K, y); v = np.linalg.solve(K, ks)
    return {"Kinv_y": w.tolist(), "m": float(ks @ w), "var": float(1 - ks @ v), "s": float(math.sqrt(1 - ks @ v))}


HYP_X = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def hyper_demo():
    """Log marginal likelihood of ten noise-free Forrester points against the length-scale (g fixed at 1e-6)."""
    X = np.array(HYP_X)[:, None]; y = np.array([forrester(x) for x in HYP_X]); z = (y - y.mean()) / y.std()
    ells = np.geomspace(0.02, 2.0, 61)
    ll = [log_ml(X, z, np.array([e]), 1e-6) for e in ells]
    k = int(np.argmax(ll))
    return {"x": HYP_X, "y": y.tolist(), "ells": ells.tolist(), "ll": ll, "best_ell": float(ells[k]), "best_ll": float(ll[k])}


BO1_X0 = [[0.15], [0.5], [0.85]]


def bo_1d():
    """EI and UCB (kappa = 2) on the Forrester function from the same three points, eight experiments each."""
    out = {}
    for acq in ("ei", "ucb"):
        r = bo(forrester, 1, 11, 0, acq=acq, X0=BO1_X0, ncand=201, record=True)
        out[acq] = {"x": r["X"][:, 0].tolist(), "y": r["y"].tolist(), "steps": r["steps"]}
    xs = np.linspace(0, 1, 1000001); fv = -(6 * xs - 2) ** 2 * np.sin(12 * xs - 4); k = int(np.argmax(fv))
    out["xmax"], out["fmax"] = float(xs[k]), float(fv[k])
    return out


CAMEL_BOX = [(-2.5, 2.5), (-1.5, 1.5)]   # Session 5's box


def camel(v):
    """Session 5's six-hump camel, sign changed (maximize); v in [0, 1]^2 is scaled to the box."""
    x = CAMEL_BOX[0][0] + (CAMEL_BOX[0][1] - CAMEL_BOX[0][0]) * v[0]
    y = CAMEL_BOX[1][0] + (CAMEL_BOX[1][1] - CAMEL_BOX[1][0]) * v[1]
    return -((4 - 2.1 * x * x + x ** 4 / 3) * x * x + x * y + (-4 + 4 * y * y) * y * y)


CAMEL_MAX = 1.031628453489877


def best_so_far(y):
    return np.maximum.accumulate(np.asarray(y, float))


def compare_camel(budget=30, seeds=20):
    """Median best-so-far value of random search and BO (6 Latin-hypercube points, then EI) over 20 seeds,
    and full grids of 20 (5 x 4) and 30 (6 x 5) points."""
    ran, bos = [], []
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        ran.append(best_so_far([camel(p) for p in rng.random((budget, 2))]))
        bos.append(best_so_far(bo(camel, 2, budget, 200 + s)["y"]))
    ran, bos = np.array(ran), np.array(bos)
    grids = {}
    for nx, ny in ((5, 4), (6, 5)):
        pts = [(a, b) for a in np.linspace(0, 1, nx) for b in np.linspace(0, 1, ny)]
        grids[str(nx * ny)] = max(camel(p) for p in pts)
    one = bo(camel, 2, 20, 200)
    return {"n": list(range(1, budget + 1)), "random_med": np.median(ran, 0).tolist(), "bo_med": np.median(bos, 0).tolist(),
            "random_q": [np.percentile(ran, 25, 0).tolist(), np.percentile(ran, 75, 0).tolist()],
            "bo_q": [np.percentile(bos, 25, 0).tolist(), np.percentile(bos, 75, 0).tolist()],
            "grid": grids, "fmax": CAMEL_MAX, "seed0": {"X": one["X"].tolist(), "y": one["y"].tolist()},
            "bo20": bos[:, 19].tolist(), "random20": ran[:, 19].tolist()}


# ---------------------------------------------------------------------------
# Part C. A shipping LP "written by a language model", with three planted errors
# ---------------------------------------------------------------------------
SHIP = {"plants": ["A", "B", "C"], "customers": ["C1", "C2", "C3"],
        "cap_per_day": [20.0, 15.0, 10.0],      # t/d; plants run 7 days a week
        "days": 7, "demand": [160.0, 90.0, 60.0],  # t/week
        "cost": [[12.0, 20.0, 30.0], [25.0, 10.0, 18.0], [28.0, 22.0, 9.0]]}   # USD/t, row = plant, column = customer


def ship_model(data, fix_sign=False, fix_units=False, fix_index=False):
    """The model as the language model wrote it; each switch repairs one planted error.
    Error 1 (sign): demand rows written as 'at most' the demand.  Error 2 (units): capacity in t/d against demand
    in t/week.  Error 3 (index): cost[j][i] instead of cost[i][j]."""
    P, Cn = len(data["plants"]), len(data["customers"])
    c = [data["cost"][i][j] if fix_index else data["cost"][j][i] for i in range(P) for j in range(Cn)]
    cap = [v * data["days"] if fix_units else v for v in data["cap_per_day"]]
    A, b = [], []
    for i in range(P):
        A.append([1.0 if k // Cn == i else 0.0 for k in range(P * Cn)]); b.append(cap[i])
    for j in range(Cn):
        row = [1.0 if k % Cn == j else 0.0 for k in range(P * Cn)]
        if fix_sign:
            A.append([-v for v in row]); b.append(-data["demand"][j])
        else:
            A.append(row); b.append(data["demand"][j])
    r = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    x = None if r.x is None else np.round(r.x, 9).reshape(P, Cn)
    return {"ok": r.status == 0, "status": int(r.status), "x": None if x is None else (x + 0.0).tolist(),
            "cost": None if r.x is None else float(r.fun)}


def ship_tests(model, data=SHIP):
    """Six tests written from the specification only; each returns True (pass) or False (fail)."""
    sol = model(data); x = np.array(sol["x"]) if sol["x"] is not None else None
    d, cst = np.array(data["demand"]), np.array(data["cost"])
    weekly = np.array(data["cap_per_day"]) * data["days"]
    t = {}
    t["solver reports success"] = sol["ok"]
    t["every customer gets its demand"] = bool(x is not None and np.all(x.sum(0) >= d - 1e-6))
    t["weekly capacity respected"] = bool(x is not None and np.all(x.sum(1) <= weekly + 1e-6))
    t["cost recomputed from the plan"] = bool(x is not None and abs((cst * x).sum() - sol["cost"]) < 1e-6)
    lb = float((d * cst.min(0)).sum())
    t["cost at least the lower bound"] = bool(x is not None and sol["cost"] >= lb - 1e-6)
    one = dict(data, demand=[10.0, 0.0, 0.0]); s1 = model(one)
    t["known answer: 10 t to C1 costs 120"] = bool(s1["ok"] and abs(s1["cost"] - 120.0) < 1e-6)
    return t, sol


def ship_results():
    versions = [("as written", {}), ("sign fixed", {"fix_sign": True}),
                ("sign and units fixed", {"fix_sign": True, "fix_units": True}),
                ("all three fixed", {"fix_sign": True, "fix_units": True, "fix_index": True})]
    out = []
    for name, kw in versions:
        t, sol = ship_tests(lambda dd, kw=kw: ship_model(dd, **kw))
        out.append({"name": name, "tests": t, "ok": sol["ok"], "cost": sol["cost"], "x": sol["x"]})
    d, cst = np.array(SHIP["demand"]), np.array(SHIP["cost"])
    return {"data": SHIP, "versions": out, "lower_bound": float((d * cst.min(0)).sum())}


# ---------------------------------------------------------------------------
# Lab 14A. A hidden, noisy CSTR yield (synthetic kinetics; students see only the experiments)
# ---------------------------------------------------------------------------
CSTR = {"T": (60.0, 120.0), "tau": (1.0, 60.0), "Tref": 90.0, "k_ref": (0.4, 0.012, 0.04),   # 1/min at Tref
        "E_R": (10000.0, 2000.0, 20000.0), "sigma": 1.5}                                       # K; yield points


def to_units(v):
    """Scaled inputs in [0, 1]^2 to temperature (deg C) and residence time (min, log scale)."""
    T = CSTR["T"][0] + (CSTR["T"][1] - CSTR["T"][0]) * v[0]
    tau = CSTR["tau"][0] * (CSTR["tau"][1] / CSTR["tau"][0]) ** v[1]
    return T, tau


def cstr_yield(v):
    """Yield of B (% of A fed) in a CSTR: A -> B (k1), B -> C (k2), A -> D (k3)."""
    T, tau = to_units(v); TK = T + 273.15; TR = CSTR["Tref"] + 273.15
    k1, k2, k3 = (kr * math.exp(-E * (1 / TK - 1 / TR)) for kr, E in zip(CSTR["k_ref"], CSTR["E_R"]))
    return 100 * k1 * tau / ((1 + (k1 + k3) * tau) * (1 + k2 * tau))


def cstr_optimum():
    """The best yield, by the closed form: at each T, tau* = 1/sqrt((k1 + k3) k2)."""
    best = None
    for T in np.linspace(*CSTR["T"], 60001):
        TK = T + 273.15; TR = CSTR["Tref"] + 273.15
        k1, k2, k3 = (kr * math.exp(-E * (1 / TK - 1 / TR)) for kr, E in zip(CSTR["k_ref"], CSTR["E_R"]))
        tau = min(max(1 / math.sqrt((k1 + k3) * k2), CSTR["tau"][0]), CSTR["tau"][1])
        Y = 100 * k1 * tau / ((1 + (k1 + k3) * tau) * (1 + k2 * tau))
        if best is None or Y > best[0]:
            best = (Y, float(T), tau)
    return {"Y": best[0], "T": best[1], "tau": best[2]}


YMAX = None


def lab_methods(seed, budget=20, sigma=1.5, acq="ei", record=False):
    """One run of the three methods with the same budget; each recommends its best observed experiment."""
    rng = np.random.default_rng(1000 + seed)
    nT = {12: 3, 20: 4, 30: 5}[budget]; nL = budget // nT
    G = [(a, b) for a in np.linspace(0, 1, nT) for b in np.linspace(0, 1, nL)]
    yg = [cstr_yield(p) + sigma * rng.standard_normal() for p in G]
    Rn = rng.random((budget, 2)); yr = [cstr_yield(p) + sigma * rng.standard_normal() for p in Rn]
    b = bo(cstr_yield, 2, budget, 2000 + seed, sigma=sigma, acq=acq, record=record)
    out = {}
    for name, X, y in (("grid", np.array(G), np.array(yg)), ("random", Rn, np.array(yr)), ("bo", b["X"], b["y"])):
        k = int(np.argmax(y)); true = cstr_yield(X[k])
        out[name] = {"best_obs": float(y[k]), "true_at_best": true, "shortfall": YMAX - true, "k": k}
        if record:
            out[name]["X"] = X.tolist(); out[name]["y"] = np.asarray(y).tolist()
    if record:
        out["bo"]["steps"] = b["steps"]
    return out


LAB_SEED = 3


def lab_a():
    global YMAX
    opt = cstr_optimum(); YMAX = opt["Y"]
    run = lab_methods(LAB_SEED, record=True)
    s0 = run["bo"]["steps"][0]
    # EI by hand at the first point BO chose, from the GP fitted to the six Latin-hypercube experiments
    hand = {"m": s0["m"], "s": s0["s"], "best": s0["best"], "z": (s0["m"] - s0["best"]) / s0["s"]}
    hand["Phi"] = float(norm.cdf(hand["z"])); hand["phi"] = float(norm.pdf(hand["z"]))
    hand["ei"] = (s0["m"] - s0["best"]) * hand["Phi"] + s0["s"] * hand["phi"]
    hand["T"], hand["tau"] = to_units(s0["next"])
    conds = {"base": {"budget": 20, "sigma": 1.5, "acq": "ei"}, "noisy": {"budget": 20, "sigma": 4.0, "acq": "ei"},
             "small": {"budget": 12, "sigma": 1.5, "acq": "ei"}, "ucb": {"budget": 20, "sigma": 1.5, "acq": "ucb"}}
    summ = {}
    for key, c in conds.items():
        rows = [lab_methods(s, **c) for s in range(20)]
        summ[key] = dict(c)
        for mth in ("grid", "random", "bo"):
            sh = np.array([r[mth]["shortfall"] for r in rows]); ob = np.array([r[mth]["best_obs"] - r[mth]["true_at_best"] for r in rows])
            summ[key][mth] = {"median": float(np.median(sh)), "mean": float(sh.mean()), "worst": float(sh.max()),
                              "within1": int((sh <= 1.0).sum()), "optimism": float(ob.mean())}
        summ[key]["bo_beats_grid"] = int(sum(r["bo"]["shortfall"] < r["grid"]["shortfall"] for r in rows))
    return {"data": CSTR, "opt": opt, "seed": LAB_SEED, "run": run, "hand": hand, "conditions": summ}


# ---------------------------------------------------------------------------
# Lab 14B. A batch-plant schedule "written by a language model", with three planted errors
# ---------------------------------------------------------------------------
PLANT = {"products": ["P1", "P2", "P3"], "reactors": ["R1", "R2"], "days": 5,
         # batch size (t), batch time (h) and cost (USD per batch); P3 cannot be made in R1
         "size": {"P1/R1": 8, "P2/R1": 6, "P1/R2": 10, "P2/R2": 8, "P3/R2": 5},
         "time": {"P1/R1": 4, "P2/R1": 5, "P1/R2": 6, "P2/R2": 6, "P3/R2": 8},
         "cost": {"P1/R1": 900, "P2/R1": 800, "P1/R2": 1100, "P2/R2": 1000, "P3/R2": 1200},
         "hours": 20, "hold": 20,                                   # h per reactor per day; USD per t per day
         "demand": {"P1": [0, 30, 0, 30, 10], "P2": [20, 0, 25, 0, 15], "P3": [0, 10, 0, 15, 5]},   # t due by day
         "stock0": {"P1": 5, "P2": 4, "P3": 3}}


def schedule_llm(data, fix_units=False, fix_index=False, fix_data=False):
    """The schedule as the language model wrote it; each switch repairs one planted error.
    Error 1 (units): hours available converted to minutes, batch times left in hours.
    Error 2 (index): demand[p][t - 1] in the balance of day t, which wraps to the last day when t = 0.
    Error 3 (data): a guessed entry for P3 in R1 (5 t, 6 h, 800 USD), which the plant cannot make."""
    P, Rr, T = data["products"], data["reactors"], data["days"]
    size, tim, cost = dict(data["size"]), dict(data["time"]), dict(data["cost"])
    if not fix_data:
        size["P3/R1"], tim["P3/R1"], cost["P3/R1"] = 5, 6, 800
    pairs = [f"{p}/{r}" for p in P for r in Rr if f"{p}/{r}" in size]
    nb = len(pairs) * T; N = nb + len(P) * T
    ib = lambda k, t: k * T + t
    isk = lambda j, t: nb + j * T + t
    c = np.zeros(N)
    for k, pr in enumerate(pairs):
        c[[ib(k, t) for t in range(T)]] = cost[pr]
    c[nb:] = data["hold"]
    A, lo, hi = [], [], []
    avail = data["hours"] if fix_units else data["hours"] * 60          # "convert hours to minutes"
    for r in Rr:
        for t in range(T):
            row = np.zeros(N)
            for k, pr in enumerate(pairs):
                if pr.endswith("/" + r):
                    row[ib(k, t)] = tim[pr]
            A.append(row); lo.append(-np.inf); hi.append(avail)
    for j, p in enumerate(P):
        for t in range(T):
            row = np.zeros(N); row[isk(j, t)] = 1.0
            for k, pr in enumerate(pairs):
                if pr.startswith(p + "/"):
                    row[ib(k, t)] = -size[pr]
            prev = data["stock0"][p]
            if t > 0:
                row[isk(j, t - 1)] = -1.0; prev = 0.0
            dem = data["demand"][p][t] if fix_index else data["demand"][p][t - 1]
            A.append(row); lo.append(prev - dem); hi.append(prev - dem)
    r = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.r_[np.ones(nb), np.zeros(N - nb)],
             bounds=Bounds(0, np.inf))
    if r.x is None:
        return {"ok": False, "status": int(r.status)}
    x = np.round(r.x).astype(int)
    return {"ok": r.status == 0, "status": int(r.status), "cost": float(r.fun),
            "batches": {pr: [int(x[ib(k, t)]) for t in range(T)] for k, pr in enumerate(pairs)},
            "stock": {p: [r6(r.x[isk(j, t)]) for t in range(T)] for j, p in enumerate(P)}}


TESTS_B = ["T1 solver reports optimal", "T2 reactor hours per day at most 20", "T3 stock balance over the week",
           "T4 stock balance day by day", "T5 no P3 batches in R1", "T6 zero demand gives no batches",
           "T7 known answer: 60 t of P1 on day 1 costs 6,700 USD", "T8 doubling sizes, demands and stocks, halving holding cost, keeps the cost"]


def sched_tests(model, data=PLANT):
    """Eight tests written from the specification only (data = the specification)."""
    sol = model(data); res = []
    res.append(bool(sol["ok"]))
    hours_ok = True
    for r in data["reactors"]:
        for t in range(data["days"]):
            h = sum(n[t] * data["time"][pr] for pr, n in sol["batches"].items() if pr.endswith("/" + r) and pr in data["time"])
            hours_ok &= h <= data["hours"] + 1e-9
    res.append(hours_ok)
    made = lambda p, t: sum(n[t] * data["size"].get(pr, 5) for pr, n in sol["batches"].items() if pr.startswith(p + "/"))
    week = all(abs(data["stock0"][p] + sum(made(p, t) for t in range(data["days"])) - sum(data["demand"][p])
                   - sol["stock"][p][-1]) < 1e-6 for p in data["products"])
    res.append(week)
    daily = True
    for p in data["products"]:
        prev = data["stock0"][p]
        for t in range(data["days"]):
            daily &= abs(prev + made(p, t) - data["demand"][p][t] - sol["stock"][p][t]) < 1e-6; prev = sol["stock"][p][t]
    res.append(daily)
    res.append(all(v == 0 for pr, n in sol["batches"].items() if pr == "P3/R1" for v in n))
    zero = model(dict(data, demand={p: [0] * data["days"] for p in data["products"]}))
    res.append(bool(zero["ok"]) and all(v == 0 for n in zero["batches"].values() for v in n))
    one = model(dict(data, demand={"P1": [60, 0, 0, 0, 0], "P2": [0] * 5, "P3": [0] * 5}, stock0={"P1": 0, "P2": 0, "P3": 0}))
    res.append(bool(one["ok"]) and abs(one["cost"] - 6700.0) < 1e-6)
    big = dict(data, size={k: 2 * v for k, v in data["size"].items()}, hold=data["hold"] / 2,
               demand={p: [2 * v for v in data["demand"][p]] for p in data["products"]},
               stock0={p: 2 * v for p, v in data["stock0"].items()})
    sb = model(big)
    res.append(bool(sb["ok"]) and abs(sb["cost"] - sol["cost"]) < 1e-6)
    return res, sol


def lab_b():
    combos = {}
    for u in (False, True):
        for i in (False, True):
            for d in (False, True):
                res, sol = sched_tests(lambda dd, u=u, i=i, d=d: schedule_llm(dd, fix_units=u, fix_index=i, fix_data=d))
                combos[f"{int(u)}{int(i)}{int(d)}"] = {"tests": res, "cost": sol.get("cost"), "batches": sol.get("batches"),
                                                      "stock": sol.get("stock")}
    return {"data": PLANT, "tests": TESTS_B, "combos": combos}


# ---------------------------------------------------------------------------
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if not math.isfinite(v) else round(v, 10) + 0.0
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def classroom_results():
    import scipy
    R = {"meta": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}}
    R["lecture"] = {"by_hand": by_hand(), "hyper": hyper_demo(), "bo1": bo_1d(), "camel": compare_camel(),
                    "ship": ship_results(), "ELL": ELL.tolist(), "GN": GN.tolist()}
    if sys.platform != "emscripten":      # the lab results are recorded; the browser's live run recomputes the lecture only
        R["lab"] = {"a": lab_a(), "b": lab_b()}
    return clean(R)


if __name__ == "__main__":
    if "--json" in sys.argv:
        R = classroom_results()
        with open("bo-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote bo-results.json", R["meta"]); sys.exit()
    import time; t0 = time.time()
    print("A by hand", by_hand())
    H = hyper_demo(); print("A hyper best ell", round(H["best_ell"], 4), "ll", round(H["best_ll"], 3))
    B = bo_1d(); print("B 1-D max", round(B["xmax"], 5), round(B["fmax"], 4))
    for a in ("ei", "ucb"):
        print("  ", a, [(round(x, 3), round(y, 3)) for x, y in zip(B[a]["x"], B[a]["y"])])
    Cm = compare_camel(); print("B camel median at 20: random", round(Cm["random_med"][19], 4), "BO", round(Cm["bo_med"][19], 4),
                                "grid", {k: round(v, 4) for k, v in Cm["grid"].items()}, "at 30:", round(Cm["random_med"][29], 4), round(Cm["bo_med"][29], 4))
    S = ship_results()
    for v in S["versions"]:
        print("C ship", v["name"], v["ok"], v["cost"], [k for k, ok in v["tests"].items() if not ok])
    A = lab_a(); print("Lab A optimum", {k: round(v, 3) for k, v in A["opt"].items()})
    print("  hand", {k: round(v, 4) for k, v in A["hand"].items()})
    for m in ("grid", "random", "bo"):
        r = A["run"][m]; print("  run", m, round(r["best_obs"], 2), round(r["true_at_best"], 2), round(r["shortfall"], 2), to_units(r["X"][r["k"]]))
    for k, c in A["conditions"].items():
        print("  cond", k, {m: (round(c[m]["median"], 2), round(c[m]["mean"], 2), c[m]["within1"], round(c[m]["optimism"], 2)) for m in ("grid", "random", "bo")}, "bo beats grid", c["bo_beats_grid"])
    Lb = lab_b()
    for k, v in Lb["combos"].items():
        print("Lab B fixes", k, v["cost"], "".join("." if ok else "F" for ok in v["tests"]))
    print("time", round(time.time() - t0, 1))
