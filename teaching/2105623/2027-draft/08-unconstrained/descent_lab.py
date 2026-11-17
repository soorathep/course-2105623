"""Session 8 (2027 draft): algorithms for unconstrained minimization.

Steepest descent, Newton's method and BFGS, written out in a few lines each so
that every iteration, function evaluation and gradient evaluation can be
counted. All three use the same Armijo backtracking line search.

Run `python descent_lab.py` for a printed summary, or `python descent_lab.py --json`
to write descent-results.json, which the slides and the lab page read.
"""
import json
import math
import sys
import warnings

import numpy as np


def r6(v):
    """Round for printing, so recorded and live runs print the same text."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, d=4):
    """A number with d significant digits, as text (for values spanning many decades)."""
    return f"{float(v):.{d}g}"


# ---------------------------------------------------------------------------
# Part A. One line search and three methods
# ---------------------------------------------------------------------------
class Counter:
    """Wrap f and its derivatives so every call is counted."""
    def __init__(self, f, g=None, h=None):
        self.f0, self.g0, self.h0 = f, g, h
        self.nf = self.ng = self.nh = 0

    def f(self, x):
        self.nf += 1
        return self.f0(x)

    def g(self, x):
        self.ng += 1
        return self.g0(x)

    def h(self, x):
        self.nh += 1
        return self.h0(x)


def armijo(F, x, fx, gx, p, alpha0=1.0, c1=1e-4, rho=0.5, max_back=60):
    """Backtracking: halve alpha until f(x + alpha p) <= f(x) + c1 alpha g'p."""
    slope = float(gx @ p); a = alpha0
    for _ in range(max_back):
        xn = x + a * p; fn = F.f(xn)
        if np.isfinite(fn) and fn <= fx + c1 * a * slope:
            return a, xn, fn
        a *= rho
    return 0.0, x, fx


def run(method, f, g, x0, h=None, gtol=1e-6, max_iter=5000, keep=400, gscale=None):
    """method: 'gd' (steepest descent), 'newton' (modified Newton), 'pure_newton' (full steps, no
    line search, no modification) or 'bfgs'. Stops when max|gscale * grad f| <= gtol (gscale = 1 unless given)."""
    F = Counter(f, g, h); x = np.array(x0, float); fx = F.f(x); gx = F.g(x)
    n = len(x); Hinv = np.eye(n); path = [x.tolist()]; gs = np.ones(n) if gscale is None else np.array(gscale, float)
    gn = [float(np.max(np.abs(gs * gx)))]; fv = [float(fx)]
    status = "max_iter"
    for k in range(max_iter):
        if np.max(np.abs(gs * gx)) <= gtol:
            status = "converged"; break
        if method == "gd":
            p = -gx
        elif method in ("newton", "pure_newton"):
            H = F.h(x)
            if method == "newton":
                lam, V = np.linalg.eigh(H)                       # make the model convex: flip and floor eigenvalues
                lam = np.maximum(np.abs(lam), 1e-12 * max(1e-12, float(np.max(np.abs(lam)))))
                p = -V @ ((V.T @ gx) / lam)
            else:
                p = -np.linalg.solve(H, gx)
        else:
            p = -Hinv @ gx
        if method == "pure_newton":
            xn = x + p; fn = F.f(xn); a = 1.0
        else:
            a, xn, fn = armijo(F, x, fx, gx, p)
            if a == 0.0:
                status = "line search failed"; break
        gn_new = F.g(xn)
        if method == "bfgs":
            s = xn - x; y = gn_new - gx; sy = float(s @ y)
            if sy > 1e-12 * float(np.linalg.norm(s) * np.linalg.norm(y)):
                rho_ = 1.0 / sy; I = np.eye(n)
                Hinv = (I - rho_ * np.outer(s, y)) @ Hinv @ (I - rho_ * np.outer(y, s)) + rho_ * np.outer(s, s)
        x, fx, gx = xn, fn, gn_new
        if len(path) < keep:
            path.append(x.tolist())
        gn.append(float(np.max(np.abs(gs * gx)))); fv.append(float(fx))
    return {"method": method, "x": x.tolist(), "f": float(fx), "nit": len(gn) - 1, "nfev": F.nf, "ngev": F.ng, "nhev": F.nh,
            "gnorm": gn[-1], "status": status, "path": path, "gnorms": gn[:keep], "fvals": fv[:keep]}


def summary(r):
    return f'{r["method"]:11s} nit {r["nit"]:5d} nfev {r["nfev"]:5d} ngev {r["ngev"]:5d} |g| {sig(r["gnorm"], 2):>8s} {r["status"]}'


# ---------------------------------------------------------------------------
# Part B. Test functions
# ---------------------------------------------------------------------------
def quadratic(kappa, theta_deg=30.0):
    """f = 1/2 x'Ax with eigenvalues 1 and kappa, axes rotated by theta."""
    t = math.radians(theta_deg); Q = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    A = Q @ np.diag([1.0, kappa]) @ Q.T
    return (lambda x: 0.5 * x @ A @ x), (lambda x: A @ x), (lambda x: A), A


def worst_start(kappa, theta_deg=30.0, radius=4.0):
    """The start from which steepest descent zigzags most: gradient at 45 degrees to both axes."""
    t = math.radians(theta_deg); Q = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    u = np.array([kappa, 1.0]); u = radius * u / np.linalg.norm(u)
    return Q @ u


def gd_exact_quadratic(kappa, x0=None, theta_deg=30.0, tol=1e-6, max_iter=100000):
    """Steepest descent with the exact step on a quadratic: iterations to cut |grad| by tol."""
    f, g, h, A = quadratic(kappa, theta_deg)
    x = worst_start(kappa, theta_deg) if x0 is None else np.array(x0, float); g0 = np.linalg.norm(A @ x)
    for k in range(max_iter):
        gk = A @ x
        if np.linalg.norm(gk) <= tol * g0:
            return k
        x = x - (gk @ gk) / (gk @ A @ gk) * gk
    return max_iter


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def himmelblau_g(x):
    A = x[0]**2 + x[1] - 11; B = x[0] + x[1]**2 - 7
    return np.array([4 * x[0] * A + 2 * B, 2 * A + 4 * x[1] * B])


def himmelblau_h(x):
    A = x[0]**2 + x[1] - 11; B = x[0] + x[1]**2 - 7
    return np.array([[4 * A + 8 * x[0]**2 + 2, 4 * x[0] + 4 * x[1]], [4 * x[0] + 4 * x[1], 2 + 4 * B + 8 * x[1]**2]])


def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def rosenbrock_g(x):
    return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])


def rosenbrock_h(x):
    return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200.0]])


def double_well(x):
    """1-D: f = x^4 - 4x^2 + x, for the Newton-model demo."""
    return x**4 - 4 * x**2 + x


def newton_1d(x0, modified=False, n=8):
    """Newton on the double well f = x^4 - 4x^2 + x. Pure: x - f'/f''. Modified: curvature max(|f''|, 1)
    and Armijo backtracking, so every step goes downhill."""
    xs = [float(x0)]; x = float(x0)
    for _ in range(n):
        d1 = 4 * x**3 - 8 * x + 1; d2 = 12 * x**2 - 8
        if not modified:
            x = x - d1 / d2
        else:
            p = -d1 / max(abs(d2), 1.0); a = 1.0
            while double_well(x + a * p) > double_well(x) + 1e-4 * a * d1 * p and a > 1e-10:
                a /= 2
            x = x + a * p
        xs.append(x)
    return xs


def fd_errors(x0=1.5):
    """Forward, central and complex-step derivative errors for f = e^x / sqrt(sin^3 x + cos^3 x)."""
    f = lambda z: np.exp(z) / np.sqrt(np.sin(z)**3 + np.cos(z)**3)
    s, c = math.sin(x0), math.cos(x0); den = s**3 + c**3
    exact = f(x0) * (1 - 0.5 * (3 * s * s * c - 3 * c * c * s) / den)
    out = []
    for k in range(1, 21):
        hh = 10.0**-k
        fwd = (f(x0 + hh) - f(x0)) / hh; cen = (f(x0 + hh) - f(x0 - hh)) / (2 * hh); cs = (f(x0 + 1j * hh)).imag / hh
        out.append({"h": hh, "forward": abs(fwd - exact) / abs(exact), "central": abs(cen - exact) / abs(exact),
                    "complex": max(abs(cs - exact) / abs(exact), 1e-17)})
    return {"x0": x0, "exact": float(exact), "rows": out}


def backtracking_by_hand():
    """f = x1^2 + 10 x2^2 at (1, 1), p = -grad f: the trials of Armijo backtracking (c1 = 1e-4, halving)."""
    A = np.diag([2.0, 20.0]); f = lambda x: 0.5 * x @ A @ x; x = np.array([1.0, 1.0]); g = A @ x; p = -g
    fx = f(x); slope = float(g @ p); a = 1.0; rows = []
    while True:
        fn = f(x + a * p); ok = fn <= fx + 1e-4 * a * slope
        rows.append({"alpha": a, "x": (x + a * p).tolist(), "f": float(fn), "bound": float(fx + 1e-4 * a * slope), "ok": bool(ok)})
        if ok:
            break
        a /= 2
    return {"x": x.tolist(), "f": float(fx), "g": g.tolist(), "slope": slope, "trials": rows, "exact": float((g @ g) / (g @ A @ g))}


def scaling_sweep(exps=None):
    """A quadratic written in badly scaled units: x2 = s z2. Iterations of each method versus s = 10^e."""
    A0 = np.array([[2.0, 0.6], [0.6, 1.0]]); out = []
    for e in (np.arange(0, 4.01, 0.25) if exps is None else exps):
        s = 10.0**e; Di = np.diag([1.0, 1.0 / s]); A = Di @ A0 @ Di; xs = np.array([1.0, 2.0 * s])
        f = lambda x, A=A, xs=xs: 0.5 * (x - xs) @ A @ (x - xs); g = lambda x, A=A, xs=xs: A @ (x - xs); h = lambda x, A=A: A
        row = {"log10s": float(e), "cond": float(np.linalg.cond(A))}
        for m in ("gd", "bfgs", "newton"):
            r = run(m, f, g, [0.0, 0.0], h=h, gtol=1e-6, max_iter=20000, keep=2, gscale=[1.0, s])
            row[m] = r["nit"] if r["status"] == "converged" else None
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Lab 8. Two-stage compressor with an intercooler (synthetic data)
# ---------------------------------------------------------------------------
COMP = {"n": 50.0, "gamma": 1.4, "eta": 0.8, "p1": 1e5, "p3": 1e6, "T1": 300.0,       # mol/s, -, -, Pa, Pa, K
        "Tw_in": 293.0, "Tw_out": 303.0, "U": 40.0,                                    # cooling water K, gas-side U W/m2K
        "c_power": 0.8, "c_area": 300.0, "c_water": 0.03,                               # USD/yr per W, per m2, per W removed
        "x0": [5e5, 340.0], "scale": [1e5, 10.0],                                       # start (Pa, K); scaled units: bar and 10 K
        "units": "p in Pa and T in K (SI); cost in USD/yr"}


def comp_cost(x, D=COMP):
    """Annual cost: power for both stages + intercooler area + cooling water. Works for complex x (complex step)."""
    p, T = x[0], x[1]
    if np.real(p) <= 0:
        return np.inf
    a = (D["gamma"] - 1) / D["gamma"]; R = 8.314; cp = D["gamma"] * R / (D["gamma"] - 1)
    r1 = (p / D["p1"])**a; r2 = (D["p3"] / p)**a
    Tout = D["T1"] + D["T1"] * (r1 - 1) / D["eta"]
    d1 = Tout - D["Tw_out"]; d2 = T - D["Tw_in"]
    if np.real(d1) <= 0 or np.real(d2) <= 0:
        return np.inf
    lm = (d1 - d2) / np.log(d1 / d2) if abs(np.real(d1 - d2)) > 1e-9 else d1
    W = D["n"] * cp * (D["T1"] * (r1 - 1) + T * (r2 - 1)) / D["eta"]
    Q = D["n"] * cp * (Tout - T); A = Q / (D["U"] * lm)
    return D["c_power"] * W + D["c_area"] * A + D["c_water"] * Q


def comp_parts(x, D=COMP):
    p, T = x; a = (D["gamma"] - 1) / D["gamma"]; R = 8.314; cp = D["gamma"] * R / (D["gamma"] - 1)
    r1 = (p / D["p1"])**a; r2 = (D["p3"] / p)**a; Tout = D["T1"] + D["T1"] * (r1 - 1) / D["eta"]
    d1 = Tout - D["Tw_out"]; d2 = T - D["Tw_in"]; lm = (d1 - d2) / math.log(d1 / d2)
    W = D["n"] * cp * (D["T1"] * (r1 - 1) + T * (r2 - 1)) / D["eta"]; Q = D["n"] * cp * (Tout - T)
    return {"power_kW": W / 1e3, "duty_kW": Q / 1e3, "area_m2": Q / (D["U"] * lm), "T_out1": Tout}


def comp_grad(x, D=COMP):
    """Complex-step gradient: exact to rounding, with no subtraction error."""
    x = np.array(x, float); g = np.zeros(2); hh = 1e-20 * np.maximum(1.0, np.abs(x))
    for i in range(2):
        xc = x.astype(complex); xc[i] += 1j * hh[i]
        g[i] = np.imag(comp_cost(xc, D)) / hh[i]
    return g


def comp_hess(x, D=COMP):
    """Central differences of the complex-step gradient."""
    x = np.array(x, float); H = np.zeros((2, 2)); hh = 1e-5 * np.maximum(1.0, np.abs(x))
    for j in range(2):
        e = np.zeros(2); e[j] = hh[j]
        H[:, j] = (comp_grad(x + e, D) - comp_grad(x - e, D)) / (2 * hh[j])
    return 0.5 * (H + H.T)


def scaled(D=COMP):
    """The same cost in scaled variables z = x / s (p in bar, T in units of 10 K)."""
    s = np.array(D["scale"])
    return (lambda z: comp_cost(z * s, D)), (lambda z: comp_grad(z * s, D) * s), (lambda z: s[:, None] * comp_hess(z * s, D) * s[None, :])


def lab_results():
    D = COMP; s = np.array(D["scale"]); x0 = np.array(D["x0"])
    fz, gz, hz = scaled(D)
    si = {m: run(m, comp_cost, comp_grad, x0, h=comp_hess, gtol=1e-3, max_iter=5000, keep=300) for m in ("gd", "newton", "bfgs")}
    sc = {m: run(m, fz, gz, x0 / s, h=hz, gtol=1e-3, max_iter=5000, keep=300) for m in ("gd", "newton", "bfgs")}
    xs = np.array(sc["newton"]["x"]) * s
    Hsi = comp_hess(xs); Hsc = s[:, None] * Hsi * s[None, :]
    sweep = []
    for e in np.arange(0, 5.01, 0.5):                       # scale of p only: 10^e Pa per unit; T in units of 10 K
        sp = np.array([10.0**e, 10.0])
        fz2 = lambda z, sp=sp: comp_cost(z * sp, D); gz2 = lambda z, sp=sp: comp_grad(z * sp, D) * sp
        r = run("gd", fz2, gz2, x0 / sp, gtol=1e-3, max_iter=5000, keep=2); rb = run("bfgs", fz2, gz2, x0 / sp, gtol=1e-3, max_iter=5000, keep=2)
        sweep.append({"log10s": float(e), "gd": r["nit"], "gd_status": r["status"], "gd_f": r["f"], "bfgs": rb["nit"], "bfgs_status": rb["status"]})
    ros = {m: run(m, rosenbrock, rosenbrock_g, [-1.2, 1.0], h=rosenbrock_h, gtol=1e-6, max_iter=20000, keep=400)
           for m in ("gd", "newton", "bfgs")}
    for r in ros.values():                                              # error |x - x*| by iteration, for the rate plot
        r["err"] = [float(np.linalg.norm(np.array(p) - 1.0)) for p in r["path"]]
    return {"data": D, "opt": {"x": xs.tolist(), "f": float(comp_cost(xs)), **comp_parts(xs),
                               "geometric_mean_bar": math.sqrt(D["p1"] * D["p3"]) / 1e5},
            "hess_si": {"H": Hsi.tolist(), "eig": np.linalg.eigvalsh(Hsi).tolist(), "cond": float(np.linalg.cond(Hsi))},
            "hess_scaled": {"H": Hsc.tolist(), "eig": np.linalg.eigvalsh(Hsc).tolist(), "cond": float(np.linalg.cond(Hsc))},
            "si": si, "scaled": {m: dict(r, x_si=(np.array(r["x"]) * s).tolist()) for m, r in sc.items()},
            "rosenbrock": ros, "p_scale_sweep": sweep}


# ---------------------------------------------------------------------------
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


def classroom_results():
    R = {"meta": {"numpy": np.__version__, "python": sys.version.split()[0]}}
    kap = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    hstarts = {"origin": [0.0, 0.0], "left": [-1.0, 3.5], "low": [1.0, -4.5]}
    race = {}
    for name, x0 in hstarts.items():
        race[name] = {m: run(m, himmelblau, himmelblau_g, x0, h=himmelblau_h, gtol=1e-8, max_iter=2000, keep=300)
                      for m in ("gd", "pure_newton", "newton", "bfgs")}
    R["lecture"] = {"kappa": [{"kappa": k, "gd_exact": gd_exact_quadratic(k), "factor": ((k - 1) / (k + 1))**2} for k in kap],
                    "backtrack": backtracking_by_hand(), "scaling": scaling_sweep(),
                    "newton1d": {"pure": newton_1d(0.3), "modified": newton_1d(0.3, True)},
                    "race": race, "fd": fd_errors()}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("descent-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote descent-results.json"); sys.exit()
    L = R["lecture"]
    print("A kappa", [(q["kappa"], q["gd_exact"]) for q in L["kappa"]])
    for name, rr in L["race"].items():
        for m, r in rr.items():
            print("B race", name, summary(r), "x", r6(r["x"]))
    print("A backtrack", [(q["alpha"], r6(q["f"]), q["ok"]) for q in L["backtrack"]["trials"]], "exact", r6(L["backtrack"]["exact"]))
    print("C scaling", [(q["log10s"], sig(q["cond"], 3), q["gd"], q["bfgs"], q["newton"]) for q in L["scaling"]])
    print("C fd", [(sig(q["h"], 1), sig(q["forward"], 2), sig(q["central"], 2), sig(q["complex"], 2)) for q in L["fd"]["rows"]])
    lab = R["lab"]
    print("E opt", r6(lab["opt"]["x"]), r6(lab["opt"]["f"]), {k: r6(v) for k, v in lab["opt"].items() if k not in ("x", "f")})
    print("E cond SI", sig(lab["hess_si"]["cond"]), r6(lab["hess_si"]["eig"]), "scaled", sig(lab["hess_scaled"]["cond"]), r6(lab["hess_scaled"]["eig"]))
    for m, r in lab["si"].items():
        print("E SI    ", summary(r), "x", r6(r["x"]), "f", r6(r["f"]))
    for m, r in lab["scaled"].items():
        print("E scaled", summary(r), "x", r6(r["x_si"]), "f", r6(r["f"]))
    print("E p-scale sweep", [(q["log10s"], q["gd"], q["gd_status"], round(q["gd_f"]), q["bfgs"]) for q in lab["p_scale_sweep"]])
    for m, r in lab["rosenbrock"].items():
        print("E rosen ", summary(r), "x", r6(r["x"]))
