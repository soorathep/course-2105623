"""2105623 Session 10: parameter estimation (2027 draft).

Part A  least squares by hand and Gauss-Newton against Levenberg-Marquardt (A -> B -> C, B measured)
Part B  statistics of the estimate: covariance, intervals, Arrhenius reparameterization, residual patterns, outliers
Part C  design of sampling times
Lab 10  alpha-pinene isomerization (Fuguitt and Hawkins, 1947; Box, Hunter, MacGregor and Erjavec, 1973)

Runs in desktop Python and in the browser (NumPy and SciPy). Course notation: lambda and mu are multipliers,
so the Levenberg-Marquardt damping is written nu.
"""
import json, math, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy import stats


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, d=3):
    return f"{float(v):.{d}g}"


# ---------------------------------------------------------------------------
# Part A1. One Gauss-Newton step by hand: C = 100 exp(-k t)
# ---------------------------------------------------------------------------
DECAY = {"t": [1.0, 2.0, 4.0], "C": [74.0, 55.0, 30.0]}       # h, mol/m3 (synthetic)


def decay_gn(k0=0.2, steps=5):
    t = np.array(DECAY["t"]); C = np.array(DECAY["C"]); k = k0; rows = []
    for i in range(steps + 1):
        m = 100 * np.exp(-k * t); r = m - C; J = -100 * t * np.exp(-k * t)
        rows.append({"i": i, "k": k, "model": m.tolist(), "r": r.tolist(), "J": J.tolist(), "SSE": float(r @ r),
                     "Jtr": float(J @ r), "JtJ": float(J @ J)})
        k = k - float(J @ r) / float(J @ J)
    return rows


# ---------------------------------------------------------------------------
# Part A2. A -> B -> C in a batch reactor, C_A0 = 1 mol/L, only B measured (synthetic, sigma = 0.01 mol/L)
# ---------------------------------------------------------------------------
ABC = {"t": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0], "true": [0.6, 0.25], "sigma": 0.03, "seed": 9}


def abc_B(k, t):
    k1, k2 = k; t = np.asarray(t, float); D = k2 - k1
    if D == 0:                                           # limit k2 -> k1
        return k1 * t * np.exp(-k1 * t)
    return k1 * (np.exp(-k1 * t) - np.exp(-k2 * t)) / D


def abc_J(k, t):
    k1, k2 = k; t = np.asarray(t, float); D = k2 - k1; E1 = np.exp(-k1 * t); E2 = np.exp(-k2 * t)
    return np.column_stack([(E1 - E2) * k2 / D**2 - k1 * t * E1 / D, -k1 * (E1 - E2) / D**2 + k1 * t * E2 / D])


def abc_data():
    rng = np.random.default_rng(ABC["seed"])
    return (abc_B(ABC["true"], ABC["t"]) + ABC["sigma"] * rng.standard_normal(len(ABC["t"]))).round(4).tolist()


def abc_fit(k0, method="lm", iters=60):
    """Gauss-Newton (full steps) or Levenberg-Marquardt (Marquardt scaling, nu divided by 3 on success,
    multiplied by 2 on failure). Same algorithm as the slide's JavaScript."""
    t = np.array(ABC["t"]); y = np.array(abc_data()); k = np.array(k0, float)
    sse = lambda k: float(np.sum((abc_B(k, t) - y)**2))
    path = [k.tolist()]; nu = None; status = "iteration limit"; nf = 1
    for it in range(iters):
        if k[0] <= 0 or k[1] <= 0 or abs(k[1] - k[0]) < 1e-9:
            status = "left the physical region (k <= 0 or k1 = k2)"; break
        r = abc_B(k, t) - y; J = abc_J(k, t); g = J.T @ r; A = J.T @ J
        if np.max(np.abs(g)) < 1e-12:
            status = "converged"; break
        if method == "gn":
            p = np.linalg.solve(A, -g); k = k + p; nf += 1
        else:
            if nu is None:
                nu = 1e-3
            f0 = sse(k)
            for _ in range(60):
                p = np.linalg.solve(A + nu * np.diag(np.diag(A)), -g); kn = k + p; nf += 1
                if kn[0] > 0 and kn[1] > 0 and abs(kn[1] - kn[0]) > 1e-9 and sse(kn) < f0:
                    k = kn; nu = max(nu / 3, 1e-12); break
                nu *= 2
                if nu > 1e12:
                    break
            else:
                status = "converged"; break           # no step of any length lowers the SSE: a minimum
        if method == "lm" and nu > 1e12:
            status = "converged"; break           # no step of any length lowers the SSE: a minimum
        path.append(k.tolist())
        if np.max(np.abs(p)) < 1e-10 * (1 + np.max(np.abs(k))):
            status = "converged"; break
    return {"method": method, "k": k.tolist(), "SSE": sse(k) if k.min() > 0 else None, "nit": len(path) - 1, "nfev": nf, "status": status, "path": path}


def abc_region(idx=None, k0=(0.6, 0.25)):
    """Fit k on the samples idx (default: all), then the covariance, standard errors, correlation, 95% intervals,
    and the two 95% joint regions: the linearized ellipse and the exact region SSE <= level (F test)."""
    t = np.array(ABC["t"]); y = np.array(abc_data()); idx = list(range(len(t))) if idx is None else list(idx)
    t, y = t[idx], y[idx]; n, p = len(t), 2
    k = least_squares(lambda k: abc_B(k, t) - y, k0, jac=lambda k: abc_J(k, t), method="lm").x
    r = abc_B(k, t) - y; J = abc_J(k, t); sse = float(r @ r); s2 = sse / (n - p)
    C = s2 * np.linalg.inv(J.T @ J); se = np.sqrt(np.diag(C)); tq = stats.t.ppf(0.975, n - p); Fq = stats.f.ppf(0.95, p, n - p)
    return {"idx": idx, "k": k.tolist(), "SSE": sse, "s2": s2, "cov": C.tolist(), "se": se.tolist(), "corr": float(C[0, 1] / (se[0] * se[1])),
            "ci": [[k[i] - tq * se[i], k[i] + tq * se[i]] for i in range(2)], "t": float(tq), "F": float(Fq),
            "ci_log": [[k[i] * math.exp(-tq * se[i] / k[i]), k[i] * math.exp(tq * se[i] / k[i])] for i in range(2)],
            "level": sse * (1 + p / (n - p) * Fq), "exact_k1": abc_extent(t, y, sse * (1 + p / (n - p) * Fq))}


def abc_extent(t, y, level):
    """Range of k1 inside the exact region SSE <= level: k1 on a log grid, SSE minimized over k2 on a grid.
    Returns [lowest k1, highest k1, open], where open means the region reaches the end of the grid (k1 = 50)."""
    k1 = np.geomspace(0.05, 50, 700); k2 = np.linspace(0.02, 1.0, 491)
    A, B = np.meshgrid(k1, k2, indexing="ij"); D = np.where(np.abs(B - A) < 1e-9, 1e-9, B - A)
    E = A[..., None] * (np.exp(-A[..., None] * t) - np.exp(-B[..., None] * t)) / D[..., None]
    prof = ((E - y)**2).sum(-1).min(1); inside = k1[prof <= level]
    return [float(inside.min()), float(inside.max()), bool(prof[-1] <= level)]


def abc_all(k, t):
    """C_A, C_B, C_C and their derivatives with respect to (k1, k2)."""
    t = np.asarray(t, float); A = np.exp(-k[0] * t); B = abc_B(k, t); JB = abc_J(k, t)
    JA = np.column_stack([-t * A, 0 * t]); return {"A": (A, JA), "B": (B, JB), "C": (1 - A - B, -JA - JB)}


def abc_ident():
    """Local identifiability of (k1, k2) for different measurements, at the true k with sigma = 0.03 mol/L."""
    k = ABC["true"]; t = np.array(ABC["t"]); out = {}
    for name, sp, mask in (("A only", "A", t > 0), ("B only", "B", t > 0), ("C only", "C", t > 0), ("B, t <= 1 min", "B", t <= 1), ("A and C", "AC", t > 0)):
        J = np.vstack([abc_all(k, t[mask])[c][1] for c in sp]); sv = np.linalg.svd(J, compute_uv=False)
        M = J.T @ J / ABC["sigma"]**2
        se = np.sqrt(np.diag(np.linalg.inv(M))).tolist() if sv[-1] > 1e-10 * sv[0] else [None, None]
        ratio = float(sv[-1] / sv[0]); ratio = 0.0 if ratio < 1e-12 else ratio
        out[name] = {"sv": sv.tolist(), "ratio": ratio, "se": se}
    return out


def abc_scipy(k0):
    t = np.array(ABC["t"]); y = np.array(abc_data())
    r = least_squares(lambda k: abc_B(k, t) - y, k0, jac=lambda k: abc_J(k, t), method="lm")
    return {"k": r.x.tolist(), "cost": float(r.cost), "nfev": int(r.nfev), "status": int(r.status)}


# ---------------------------------------------------------------------------
# Part B1. Arrhenius with synthetic noise: k = k0 exp(-E/RT), 5% multiplicative error
# ---------------------------------------------------------------------------
ARR = {"T": [300.0, 310.0, 320.0, 330.0, 340.0, 350.0], "lnk0": math.log(5e9), "E": 75000.0, "R": 8.314, "noise": 0.05, "seed": 4}


def arr_data():
    T = np.array(ARR["T"]); rng = np.random.default_rng(ARR["seed"])
    k = np.exp(ARR["lnk0"] - ARR["E"] / (ARR["R"] * T)) * np.exp(ARR["noise"] * rng.standard_normal(len(T)))
    return [float(f"{v:.4g}") for v in k]


def arr_fit(Tref=None):
    """Linear least squares on ln k. Without Tref the parameters are (ln k0, E); with Tref they are (ln k_ref, E),
    ln k = ln k_ref - (E/R)(1/T - 1/Tref)."""
    T = np.array(ARR["T"]); y = np.log(np.array(arr_data())); x = 1 / T - (0 if Tref is None else 1 / Tref)
    X = np.column_stack([np.ones_like(x), -x / ARR["R"]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; n, p = len(y), 2
    s2 = float(r @ r) / (n - p); C = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(C))
    return {"a": float(b[0]), "E": float(b[1]), "se_a": float(se[0]), "se_E": float(se[1]), "corr": float(C[0, 1] / (se[0] * se[1])), "s2": s2}


# ---------------------------------------------------------------------------
# Part B2. Residual patterns: second-order data fitted with a first-order model
# ---------------------------------------------------------------------------
RES = {"t": [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12], "C0": 2.0, "k2": 0.15, "sigma": 0.01, "seed": 4}


def res_data():
    t = np.array(RES["t"], float); rng = np.random.default_rng(RES["seed"])
    return (RES["C0"] / (1 + RES["k2"] * RES["C0"] * t) + RES["sigma"] * rng.standard_normal(len(t))).round(4).tolist()


def res_fits():
    t = np.array(RES["t"], float); y = np.array(res_data()); out = {}
    for name, f in (("first order", lambda p: p[0] * np.exp(-p[1] * t)), ("second order", lambda p: p[0] / (1 + p[1] * p[0] * t))):
        r = least_squares(lambda p: f(p) - y, [2.0, 0.2], method="lm")
        res = f(r.x) - y; runs = 1 + int(np.sum(np.sign(res[1:]) != np.sign(res[:-1])))
        out[name] = {"p": r.x.tolist(), "SSE": float(res @ res), "res": res.tolist(), "runs": runs}
    return out


# ---------------------------------------------------------------------------
# Part C. Design: where to sample B in A -> B -> C (D-optimal: largest det J'J)
# ---------------------------------------------------------------------------
def abc_design(times, k=None):
    k = ABC["true"] if k is None else k
    J = abc_J(k, times); M = J.T @ J / ABC["sigma"]**2; C = np.linalg.inv(M)
    return {"times": list(times), "logdet": float(np.linalg.slogdet(M)[1]), "se": np.sqrt(np.diag(C)).tolist()}


def abc_dopt(grid=None):
    """Best four sampling times on a 0.25 min grid (repeats allowed), by exhaustive search over pairs replicated twice,
    then one-at-a-time improvement."""
    grid = np.arange(0.25, 12.01, 0.25) if grid is None else np.asarray(grid)
    J = abc_J(ABC["true"], grid); best = None
    for i in range(len(grid)):
        for j in range(i, len(grid)):
            M = 2 * (np.outer(J[i], J[i]) + np.outer(J[j], J[j])); d = np.linalg.det(M)
            if best is None or d > best[0]:
                best = (d, [i, i, j, j])
    pick = best[1]
    for _ in range(5):
        for m in range(4):
            dets = [np.linalg.det(sum(np.outer(J[q], J[q]) for q in pick[:m] + [g] + pick[m + 1:])) for g in range(len(grid))]
            pick[m] = int(np.argmax(dets))
    return abc_design(sorted(grid[pick].tolist()))


# ---------------------------------------------------------------------------
# Part B3. An outlier in a calibration line: least squares against the Huber loss
# ---------------------------------------------------------------------------
OUT = {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "a": 2.0, "b": 1.5, "sigma": 0.2, "seed": 5, "bad": 6, "shift": 3.0, "f_scale": 0.3}


def out_data():
    x = np.array(OUT["x"]); rng = np.random.default_rng(OUT["seed"])
    y = OUT["a"] + OUT["b"] * x + OUT["sigma"] * rng.standard_normal(len(x)); y[OUT["bad"]] += OUT["shift"]
    return y.round(3).tolist()


def out_fits(y=None):
    x = np.array(OUT["x"]); y = np.array(out_data() if y is None else y); res = {}
    for loss in ("linear", "huber", "soft_l1"):
        r = least_squares(lambda p: p[0] + p[1] * x - y, [0.0, 1.0], loss=loss, f_scale=OUT["f_scale"])
        res[loss] = {"p": r.x.tolist(), "nfev": int(r.nfev)}
    return res


# ---------------------------------------------------------------------------
# Lab 10. alpha-pinene: y1 alpha-pinene, y2 dipentene, y3 allo-ocimene, y4 pyronene, y5 dimer (weight %)
#   dy1/dt = -(t1 + t2) y1, dy2/dt = t1 y1, dy3/dt = t2 y1 - (t3 + t4) y3 + t5 y5, dy4/dt = t3 y3, dy5/dt = t4 y3 - t5 y5
# ---------------------------------------------------------------------------
PIN = {"t": [1230.0, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0],
       "Y": [[88.35, 7.3, 2.3, 0.4, 1.75], [76.4, 15.6, 4.5, 0.7, 2.8], [65.1, 23.1, 5.3, 1.1, 5.8], [50.4, 32.9, 6.0, 1.5, 9.3],
             [37.5, 42.7, 6.0, 1.9, 12.0], [25.9, 49.1, 5.9, 2.2, 17.0], [14.0, 57.4, 5.1, 2.6, 21.0], [4.5, 63.1, 3.8, 2.9, 25.7]],
       "names": ["alpha-pinene", "dipentene", "allo-ocimene", "pyronene", "dimer"], "scale": 1e-5}


def pin_K(th):
    t1, t2, t3, t4, t5 = th
    return np.array([[-(t1 + t2), 0, 0, 0, 0], [t1, 0, 0, 0, 0], [t2, 0, -(t3 + t4), 0, t5], [0, 0, t3, 0, 0], [0, 0, t4, 0, -t5]])


def pin_model(th, t=None):
    """Weight % of each species at the times t (min), for rate constants th (1/min). The ODEs are linear,
    so y(t) = expm(K t) y(0) with y(0) = (100, 0, 0, 0, 0)."""
    t = PIN["t"] if t is None else t
    K = pin_K(th); y0 = np.array([100.0, 0, 0, 0, 0])
    return np.array([expm(K * ti) @ y0 for ti in t])


def pin_fit(cols=(0, 1, 2, 3, 4), x0=(1, 1, 1, 1, 1)):
    """Ordinary least squares on the chosen responses; parameters in units of 1e-5 1/min."""
    Y = np.array(PIN["Y"]); c = list(cols); s = PIN["scale"]
    res = lambda p: (pin_model(np.array(p) * s) - Y)[:, c].ravel()
    r = least_squares(res, x0, method="lm", xtol=1e-14, ftol=1e-14)
    n, p = len(r.fun), 5; sse = float(r.fun @ r.fun); s2 = sse / (n - p)
    C = s2 * np.linalg.inv(r.jac.T @ r.jac); se = np.sqrt(np.diag(C)); tq = stats.t.ppf(0.975, n - p)
    Z = (pin_model(r.x * s) - Y)
    return {"cols": c, "theta": r.x.tolist(), "se": se.tolist(), "ci": [[r.x[i] - tq * se[i], r.x[i] + tq * se[i]] for i in range(5)],
            "corr": (C / np.outer(se, se)).tolist(), "SSE": sse, "s2": s2, "n": n, "nfev": int(r.nfev), "status": int(r.status),
            "resid": Z.tolist(), "ss_by_response": (Z**2).sum(0).tolist(), "t": float(tq)}


def pin_jac(th, t=None, h=1e-6):
    """d(model)/d(theta) by central differences, theta in units of 1e-5 1/min; shape (len(t), 5 responses, 5 parameters)."""
    th = np.array(th, float); s = PIN["scale"]; out = []
    for i in range(5):
        e = np.zeros(5); e[i] = h * max(1.0, abs(th[i]))
        out.append((pin_model((th + e) * s, t) - pin_model((th - e) * s, t)) / (2 * e[i]))
    return np.stack(out, axis=-1)


def pin_subsets(th, s2):
    """Local identifiability for every set of measured responses: singular values of J and standard errors."""
    J = pin_jac(th); out = []
    for m in range(1, 32):
        cols = [j for j in range(5) if m >> j & 1]
        Js = J[:, cols, :].reshape(-1, 5); sv = np.linalg.svd(Js, compute_uv=False)
        rank = int(np.sum(sv > 1e-8 * sv[0]))
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(Js.T @ Js))) if rank == 5 else [None] * 5
        out.append({"cols": cols, "sv": (sv / sv[0]).tolist(), "rank": rank, "se": [None if v is None else float(v) for v in se]})
    return out


def pin_design(th, s2, extra):
    """Standard errors if one more sample of all five responses were taken at time t_new (min)."""
    J0 = pin_jac(th).reshape(-1, 5); M0 = J0.T @ J0; Jx = pin_jac(th, extra); out = []
    for i, tn in enumerate(extra):
        Jn = Jx[i]; out.append({"t": tn, "se": np.sqrt(np.diag(s2 * np.linalg.inv(M0 + Jn.T @ Jn))).tolist()})
    return out


def pin_eigen():
    Y = np.array(PIN["Y"]); D = Y - Y.mean(0); w, V = np.linalg.eigh(D.T @ D)
    V = V * np.sign(V[np.abs(V).argmax(0), range(5)])        # sign convention: largest component positive
    return {"eig": w.tolist(), "vec": V.T.tolist(), "rowsum": Y.sum(1).tolist(), "y4_rule": (0.03 * (100 - Y[:, 0])).tolist()}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def lab_results():
    full = pin_fit(); no4 = pin_fit((0, 1, 2, 4), x0=tuple(full["theta"]))
    pred = pin_model(np.array(full["theta"]) * PIN["scale"], np.linspace(0, 40000, 81).tolist())
    extra = [1000 * i for i in range(1, 81)]
    return {"data": PIN, "full": full, "no_y4": no4, "eigen": pin_eigen(), "curve_t": np.linspace(0, 40000, 81).tolist(), "curve": pred.tolist(),
            "subsets": pin_subsets(full["theta"], full["s2"]), "design": pin_design(full["theta"], full["s2"], extra)}


def classroom_results():
    import scipy
    R = {"meta": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}}
    starts = {"near": [0.8, 0.2], "far": [2.0, 0.05], "swapped": [0.2, 0.7]}
    fits = {k: {m: abc_fit(v, m) for m in ("gn", "lm")} for k, v in starts.items()}
    R["lecture"] = {"decay": decay_gn(), "abc": {"t": ABC["t"], "y": abc_data(), "true": ABC["true"], "sigma": ABC["sigma"], "fits": fits,
                                                 "scipy": {k: abc_scipy(v) for k, v in starts.items()},
                                                 "region": {"all": abc_region(), "late": abc_region([4, 5, 6, 7])}, "ident": abc_ident()},
                    "arr": {"T": ARR["T"], "k": arr_data(), "R": ARR["R"], "plain": arr_fit(), "centered": arr_fit(1 / np.mean(1 / np.array(ARR["T"])))},
                    "res": {"t": RES["t"], "y": res_data(), "fits": res_fits()},
                    "out": {"x": OUT["x"], "y": out_data(), "f_scale": OUT["f_scale"], "bad": OUT["bad"], "fits": out_fits()},
                    "design": {"uniform": abc_design([2.5, 5.0, 7.5, 10.0]), "early": abc_design([0.5, 1.0, 1.5, 2.0]), "dopt": abc_dopt()}}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    if "--json" in sys.argv:
        R = classroom_results()
        with open("est-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote est-results.json", R["meta"]); sys.exit()
    print("A decay", [(q["i"], r6(q["k"]), r6(q["SSE"])) for q in decay_gn()])
    for k0 in ([0.8, 0.2], [2.0, 0.05], [0.2, 0.7]):
        a, b = abc_fit(k0, "gn"), abc_fit(k0, "lm")
        print("A ABC", k0, "GN", a["status"], a["nit"], r6(a["k"]), "| LM", b["status"], b["nit"], r6(b["k"]), b["SSE"] and sig(b["SSE"]))
    for nm, ix in (("all", None), ("late", [4, 5, 6, 7])):
        st = abc_region(ix); print("B region", nm, r6(st["k"]), r6(st["se"]), r6(st["corr"]), r6(st["ci"]), r6(st["level"]))
    print("B ident", {k: (sig(v["ratio"]), v["se"] and [None if q is None else sig(q) for q in v["se"]]) for k, v in abc_ident().items()})
    print("B outlier", out_data(), {k: r6(v["p"]) for k, v in out_fits().items()})
    print("B Arrhenius", {k: r6(v) for k, v in arr_fit().items()}, {k: r6(v) for k, v in arr_fit(1 / np.mean(1 / np.array(ARR["T"]))).items()})
    print("B residuals", {k: (r6(v["p"]), sig(v["SSE"]), v["runs"]) for k, v in res_fits().items()})
    print("C design", abc_design([2.5, 5, 7.5, 10]), abc_design([0.5, 1, 1.5, 2]), abc_dopt())
    f = pin_fit(); print("D pinene", r6(f["theta"]), r6(f["se"]), sig(f["SSE"], 5))
