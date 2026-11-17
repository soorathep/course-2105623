"""2105623 Session 5: convexity and optimality.

Part A  Convex sets and functions: a Hessian worked by hand, quadratic forms,
        chemical equilibrium as a convex Gibbs minimization, and a Margules
        liquid whose Gibbs energy of mixing stops being convex.
Part B  Economies of scale: concave equipment cost, two parallel units, and a
        geometric-programming vessel.
Part C  Optimality and certificates: basins of attraction on the six-hump camel
        function, how many starts are enough, a solver's report, and an alpha-BB
        spatial branch and bound in one variable.
Part D  Lab 5: Rosenbrock, Himmelblau, and a heat-recovery exchanger whose
        installed cost follows the six-tenths rule.

All data are synthetic or classic test functions. Runs in the browser (Pyodide:
NumPy, SciPy) and in desktop Python.
    python convexity_lab.py          print every part
    python convexity_lab.py --json   write convexity-results.json for the slides and the lab page
"""
import json
import sys
import warnings

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def r6(a):
    return (np.round(np.asarray(a, float), 6) + 0.0).tolist()


# ---------------------------------------------------------------------------
# Part A. Convex sets and functions
# ---------------------------------------------------------------------------
def quartic(v):
    x, y = v
    return x**4 + y**4 - 4 * x * y + 1


def quartic_grad(v):
    x, y = v
    return np.array([4 * x**3 - 4 * y, 4 * y**3 - 4 * x])


def quartic_hess(v):
    x, y = v
    return np.array([[12 * x**2, -4.0], [-4.0, 12 * y**2]])


def quartic_worked():
    """Stationary points of x^4 + y^4 - 4xy + 1 and the Hessian test at each."""
    pts = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]
    rows = []
    for p in pts:
        H = quartic_hess(p); lam = np.linalg.eigvalsh(H)
        kind = "minimum" if lam.min() > 0 else ("maximum" if lam.max() < 0 else "saddle")
        rows.append({"point": list(p), "f": quartic(p), "grad": r6(quartic_grad(p)), "hess": H.tolist(),
                     "eig": r6(lam), "kind": kind})
    # where is the Hessian positive semidefinite? 144 x^2 y^2 >= 16 (and the diagonal is >= 0 anyway)
    return {"points": rows, "psd_region": "x^2 y^2 >= 1/9"}


def quadratic_forms():
    """f = 1/2 (a x^2 + 2 b x y + c y^2) for a = c = 2 and b from 0 to 3: eigenvalues 2 - b and 2 + b."""
    out = []
    for b in np.round(np.arange(0.0, 3.01, 0.5), 2):
        H = np.array([[2.0, b], [b, 2.0]]); lam = np.linalg.eigvalsh(H)
        out.append({"b": float(b), "eig": r6(lam), "kind": "positive definite" if lam.min() > 1e-12 else
                    ("positive semidefinite" if lam.min() > -1e-12 else "indefinite")})
    return out


G_ISO = np.array([0.0, -0.4, -0.9])            # g_i = G_i°/RT for three isomers A, B, C (synthetic)


def gibbs_equilibrium(starts=25, seed=3):
    """Minimize G/RT = sum n_i (g_i + ln(n_i / N)) subject to n_A + n_B + n_C = 1, n >= 0.
    The function is convex (sum of perspectives of x ln x), so every start must give the same answer.
    Closed form: x_i = exp(-g_i) / sum exp(-g)."""
    f = lambda n: float(np.sum(n * (G_ISO + np.log(np.maximum(n, 1e-300) / np.sum(n)))))
    exact = np.exp(-G_ISO) / np.exp(-G_ISO).sum()
    rng = np.random.default_rng(seed); runs = []
    for k in range(starts):
        x0 = rng.dirichlet(np.ones(3))
        with warnings.catch_warnings():          # newer SciPy warns when a trial step leaves the bounds
            warnings.simplefilter("ignore", RuntimeWarning)
            r = minimize(f, x0, method="SLSQP", bounds=[(1e-10, 1)] * 3,
                         constraints=[{"type": "eq", "fun": lambda n: np.sum(n) - 1}], options={"ftol": 1e-12, "maxiter": 300})
        runs.append({"start": r6(x0), "x": r6(r.x), "G": float(r.fun), "ok": bool(r.success)})
    spread = float(max(np.abs(np.array(q["x"]) - exact).max() for q in runs))
    return {"g": G_ISO.tolist(), "exact": r6(exact), "G_exact": f(exact), "runs": runs, "max_deviation": spread}


def margules(A_values=(0.0, 1.0, 2.0, 2.5, 3.0)):
    """Gibbs energy of mixing of a binary liquid, g = x ln x + (1-x) ln(1-x) + A x (1-x) (per RT).
    g'' = 1/x + 1/(1-x) - 2A, smallest at x = 1/2: convex for A <= 2. For A > 2 the curve has two wells:
    a common tangent gives two liquid phases (symmetric case: the two minima of g)."""
    xs = np.linspace(0.001, 0.999, 199); out = []
    for A in A_values:
        g = xs * np.log(xs) + (1 - xs) * np.log(1 - xs) + A * xs * (1 - xs)
        split = None
        if A > 2:
            r = minimize_scalar(lambda x: x * np.log(x) + (1 - x) * np.log(1 - x) + A * x * (1 - x),
                                bounds=(1e-9, 0.5 - 1e-9), method="bounded", options={"xatol": 1e-12})
            split = [float(r.x), float(1 - r.x)]
        out.append({"A": A, "g": r6(g), "min_second_derivative": 4 - 2 * A, "split": split})
    return {"x": r6(xs), "cases": out}


# ---------------------------------------------------------------------------
# Part B. Economies of scale
# ---------------------------------------------------------------------------
UNIT = {"C0": 100.0, "A0": 100.0, "n": 0.6, "max": 1000.0}   # kUSD for a 100 m2 unit, six-tenths rule, max size m2


def unit_cost(A):
    A = np.asarray(A, float)
    return np.where(A > 0, UNIT["C0"] * (np.maximum(A, 0) / UNIT["A0"]) ** UNIT["n"], 0.0)


def parallel_units(total=1500.0):
    """Two units share a required area; each at most 1,000 m2. Cost c(A1) + c(total - A1) along the
    feasible segment A1 in [total - 1000, 1000]: concave, so the minimum is at an end and the equal split
    is the maximum. A local method started at the equal split stops there: the gradient is zero."""
    lo, hi = max(0.0, total - UNIT["max"]), min(UNIT["max"], total)
    a1 = np.linspace(lo, hi, 101)
    cost = unit_cost(a1) + unit_cost(total - a1)
    f = lambda a: float(unit_cost(a[0]) + unit_cost(total - a[0]))
    starts = [lo + 1e-3 * (hi - lo), 0.25 * lo + 0.75 * hi, 0.5 * (lo + hi), 0.75 * lo + 0.25 * hi]
    runs = []
    for s0 in starts:
        r = minimize(f, [s0], method="L-BFGS-B", bounds=[(lo, hi)])
        runs.append({"start": float(s0), "A1": float(r.x[0]), "cost": float(r.fun), "nit": int(r.nit)})
    mid = 0.5 * total; h = 1.0
    d2 = (f([mid + h]) - 2 * f([mid]) + f([mid - h])) / h**2
    return {"total": total, "lo": lo, "hi": hi, "a1": r6(a1), "cost": r6(cost), "runs": runs,
            "equal_split_cost": f([mid]), "end_cost": f([hi]), "second_derivative_at_equal_split": d2}


def parallel_sweep():
    return [parallel_units(t) for t in (1100.0, 1300.0, 1500.0, 1700.0, 1900.0)]


def vessel_gp(V=10.0):
    """Closed cylindrical vessel: minimize surface area pi D L + pi D^2/2 subject to pi D^2 L / 4 >= V.
    With D = e^u, L = e^w both are posynomials: convex in (u, w). Optimum L = D = (4V/pi)^(1/3)."""
    f = lambda z: np.log(np.pi * np.exp(z[0] + z[1]) + np.pi * np.exp(2 * z[0]) / 2)
    con = {"type": "ineq", "fun": lambda z: np.log(np.pi / 4) + 2 * z[0] + z[1] - np.log(V)}   # log of the volume row
    runs = []
    for z0 in ([0.0, 0.0], [2.0, -1.0], [-1.0, 3.0], [1.5, 1.5]):
        r = minimize(f, z0, method="SLSQP", constraints=[con], options={"ftol": 1e-12})
        D, L = np.exp(r.x)
        runs.append({"start": z0, "D": float(D), "L": float(L), "area": float(np.exp(r.fun))})
    D = (4 * V / np.pi) ** (1 / 3)
    return {"V": V, "D_exact": D, "area_exact": float(1.5 * np.pi * D**2), "runs": runs}


# ---------------------------------------------------------------------------
# Part C. Optimality and certificates
# ---------------------------------------------------------------------------
def camel(v):
    x, y = v
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2


def camel_grad(v):
    x, y = v
    return np.array([8 * x - 8.4 * x**3 + 2 * x**5 + y, x - 8 * y + 16 * y**3])


def fd_hessian(f, x, h=1e-4):
    x = np.asarray(x, float); n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    return 0.5 * (H + H.T)


def camel_basins(nx=25, ny=17):
    """BFGS from a grid of starts on [-2.5, 2.5] x [-1.5, 1.5]; label each start by the minimum it reaches."""
    xs, ys = np.linspace(-2.5, 2.5, nx), np.linspace(-1.5, 1.5, ny)
    mins, grid = [], []
    for y in ys:
        row = []
        for x in xs:
            r = minimize(camel, [x, y], jac=camel_grad, method="BFGS", options={"gtol": 1e-9})
            k = next((i for i, m in enumerate(mins) if np.hypot(*(np.array(m["x"]) - r.x)) < 1e-3), None)
            if k is None:
                lam = np.linalg.eigvalsh(fd_hessian(camel, r.x))
                mins.append({"x": r6(r.x), "f": float(r.fun), "eig": r6(lam), "count": 0,
                             "kind": "minimum" if lam.min() > 0 else "saddle"}); k = len(mins) - 1
            mins[k]["count"] += 1; row.append(k)
        grid.append(row)
    order = sorted(range(len(mins)), key=lambda i: mins[i]["f"])
    remap = {old: new for new, old in enumerate(order)}
    mins = [mins[i] for i in order]; grid = [[remap[k] for k in row] for row in grid]
    total = nx * ny; fglob = mins[0]["f"]
    p_glob = sum(m["count"] for m in mins if abs(m["f"] - fglob) < 1e-6) / total
    report_origin = solver_report(x0=(0.0, 0.0))
    return {"xs": r6(xs), "ys": r6(ys), "grid": grid, "minima": mins, "starts": total, "p_global": p_glob, "origin": report_origin}


def starts_needed(p, delta=0.01):
    """Smallest N with (1 - p)^N <= delta."""
    return int(np.ceil(np.log(delta) / np.log(1 - p)))


def solver_report(x0=(-1.5, 1.0)):
    """What a local solver returns, and the second-order check it does not do for you."""
    r = minimize(camel, x0, jac=camel_grad, method="BFGS", options={"gtol": 1e-8})
    lam = np.linalg.eigvalsh(fd_hessian(camel, r.x))
    return {"x0": list(x0), "success": bool(r.success), "message": str(r.message), "x": r6(r.x), "fun": float(r.fun),
            "nit": int(r.nit), "nfev": int(r.nfev), "grad_norm": float(np.linalg.norm(camel_grad(r.x))), "eig": r6(lam)}


def poly(x):
    """A one-variable test function with two minima: (x - 1.5)^2 (x - 6)^2 / 10 + 0.8 x on [0, 8]."""
    return (x - 1.5)**2 * (x - 6)**2 / 10 + 0.8 * x


def poly_d2(x):
    # (x-1.5)^2 (x-6)^2 = q^2 with q = (x-1.5)(x-6) = x^2 - 7.5x + 9; d2(q^2) = 2 q'^2 + 2 q q''
    q = x**2 - 7.5 * x + 9; qp = 2 * x - 7.5
    return (2 * qp**2 + 4 * q) / 10


def min_d2(l, u):
    """Exact minimum of the quadratic poly_d2 over [l, u] (vertex at x = 3.75)."""
    cand = [l, u] + ([3.75] if l < 3.75 < u else [])
    return min(poly_d2(c) for c in cand)


def alpha_bb(lo=0.0, hi=8.0, tol=1e-3, max_nodes=100):
    """Spatial branch and bound in one variable with the alpha-BB underestimator
    L(x) = f(x) + alpha (l - x)(u - x), alpha = max(0, -min f''/2) on the node's interval.
    Each node: minimize L (convex) for a lower bound, then a local search on f inside the node for an
    upper bound. Breadth first; prune a node whose lower bound is within tol of the incumbent."""
    nodes, inc, incx = [], np.inf, None
    queue = [(lo, hi, 0, None)]
    while queue and len(nodes) < max_nodes:
        l, u, depth, parent = queue.pop(0)
        alpha = max(0.0, -min_d2(l, u) / 2)
        L = lambda x: poly(x) + alpha * (l - x) * (u - x)
        r = minimize_scalar(L, bounds=(l, u), method="bounded", options={"xatol": 1e-10})
        lb = float(r.fun)
        loc = minimize_scalar(poly, bounds=(l, u), method="bounded", options={"xatol": 1e-10})
        if loc.fun < inc:
            inc, incx = float(loc.fun), float(loc.x)
        node = {"id": len(nodes), "parent": parent, "l": l, "u": u, "depth": depth, "alpha": alpha,
                "lb": lb, "x_lb": float(r.x), "ub": float(loc.fun), "x_ub": float(loc.x),
                "max_gap": alpha * (u - l) ** 2 / 4}
        if lb >= inc - tol:
            node["status"] = "pruned"
        else:
            node["status"] = "branched"; m = 0.5 * (l + u)
            queue += [(l, m, depth + 1, node["id"]), (m, u, depth + 1, node["id"])]
        node["incumbent"] = inc
        # lower bound over everything still open after this node: the queue's parents' bounds and the incumbent
        open_lbs = [nodes[q[3]]["lb"] if q[3] is not None and q[3] < len(nodes) else lb for q in queue]
        node["global_lb"] = min(open_lbs + [inc])
        nodes.append(node)
    # fix global_lb for children queued by this node (parents are known now)
    xs = np.linspace(lo, hi, 161)
    return {"nodes": nodes, "x": r6(xs), "f": r6(poly(xs)), "incumbent": inc, "x_best": incx,
            "complete": not queue, "tol": tol}


# ---------------------------------------------------------------------------
# Part D. Lab 5
# ---------------------------------------------------------------------------
def rosen(v):
    x, y = v
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosen_hess(v):
    x, y = v
    return np.array([[2 - 400 * (y - x**2) + 800 * x**2, -400 * x], [-400 * x, 200.0]])


def himmel(v):
    x, y = v
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def himmel_grad(v):
    x, y = v
    return np.array([4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7), 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)])


def himmel_hess(v):
    x, y = v
    return np.array([[12 * x**2 + 4 * y - 42, 4 * x + 4 * y], [4 * x + 4 * y, 4 * x + 12 * y**2 - 26]])


HX = {"Qmax": 3000.0, "C": 50.0, "dTmax": 60.0, "U": 0.5, "value": 40.0, "inst": 17.0, "crf": 0.2, "Amax": 300.0}
# Two streams of 50 kW/K each, 60 K apart: at most 3,000 kW can be recovered (balanced counterflow,
# effectiveness = NTU / (1 + NTU), NTU = U A / C). Heat not recovered is bought as low-pressure steam:
# value = 40 USD per kW-year (about 2,000 h/yr at 5.5 USD/GJ). Installed cost 17 kUSD x A^0.6 (A in m2),
# annualized with a capital recovery factor of 0.2. TAC in kUSD/yr.


def hx_tac(A, value=None):
    v = (HX["value"] if value is None else value) / 1000.0          # kUSD per kW-year
    A = np.asarray(A, float)
    ntu = HX["U"] * np.maximum(A, 0) / HX["C"]
    return HX["crf"] * HX["inst"] * np.maximum(A, 0) ** 0.6 + v * HX["Qmax"] / (1 + ntu)


def hx_stationary(value=None):
    """All stationary points of TAC(A) on (0, Amax] from a fine grid, refined; plus the end A = 0."""
    A = np.linspace(1e-6, HX["Amax"], 300001); T = hx_tac(A, value); d = np.diff(T)
    pts = []
    for k in np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0]:
        kind = "local min" if d[k] < 0 else "local max"
        f = (lambda a: hx_tac(a, value)) if kind == "local min" else (lambda a: -hx_tac(a, value))
        r = minimize_scalar(f, bounds=(A[k], A[k + 2]), method="bounded", options={"xatol": 1e-9})
        pts.append({"A": float(r.x), "tac": float(hx_tac(r.x, value)), "kind": kind})
    return pts


def hx_study(value=None, starts=(1.0, 3.0, 6.0, 20.0, 100.0, 250.0)):
    f = lambda a: float(hx_tac(a[0], value))
    runs = []
    for a0 in starts:
        r = minimize(f, [a0], method="L-BFGS-B", bounds=[(0.0, HX["Amax"])])
        runs.append({"start": a0, "A": float(r.x[0]), "tac": float(r.fun), "nit": int(r.nit)})
    st = hx_stationary(value); tac0 = float(hx_tac(0.0, value))
    inner = [p for p in st if p["kind"] == "local min"]
    best = min([{"A": 0.0, "tac": tac0}] + [{"A": p["A"], "tac": p["tac"]} for p in inner], key=lambda q: q["tac"])
    hump = next((p for p in st if p["kind"] == "local max"), None)
    return {"value": HX["value"] if value is None else value, "runs": runs, "stationary": st, "tac_zero": tac0,
            "best": best, "hump": hump, "interior": inner[0] if inner else None}


def hx_breakpoint(lo=20.0, hi=60.0):
    """Value of heat (USD/kW-yr) at which the interior minimum ties with A = 0."""
    g = lambda v: (hx_study(v, starts=())["interior"] or {"tac": np.inf})["tac"] - hx_tac(0.0, v)
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if g(m) > 0 else (lo, m)
    return 0.5 * (lo + hi)


def lab_results():
    pts = {"(1, 1)": (1.0, 1.0), "(0, 1)": (0.0, 1.0), "(-1, 1)": (-1.0, 1.0), "(0.5, 0.25)": (0.5, 0.25)}
    rosen_pts = {k: {"hess": rosen_hess(p).tolist(), "eig": r6(np.linalg.eigvalsh(rosen_hess(p)))} for k, p in pts.items()}
    rs = minimize(rosen, [-1.2, 1.0], method="BFGS", options={"gtol": 1e-9})
    # Himmelblau: multistart from a 10 x 10 grid on [-4.5, 4.5]^2 (row by row, y outer)
    g = np.linspace(-4.5, 4.5, 10); mins = []; assign = []
    for y in g:
        for x in g:
            r = minimize(himmel, [x, y], jac=himmel_grad, method="BFGS", options={"gtol": 1e-10})
            k = next((i for i, m in enumerate(mins) if np.hypot(*(np.array(m["x"]) - r.x)) < 1e-4), None)
            if k is None:
                mins.append({"x": r6(r.x), "f": float(r.fun), "count": 0,
                             "eig": r6(np.linalg.eigvalsh(himmel_hess(r.x)))}); k = len(mins) - 1
            mins[k]["count"] += 1; assign.append(k)
    order = sorted(range(len(mins)), key=lambda i: (-mins[i]["x"][0], -mins[i]["x"][1]))
    remap = {old: new for new, old in enumerate(order)}
    mins = [mins[i] for i in order]; assign = [remap[k] for k in assign]
    from scipy.optimize import root
    s = root(himmel_grad, [-0.27, -0.92], tol=1e-14)
    mx = {"x": r6(s.x), "f": float(himmel(s.x)), "eig": r6(np.linalg.eigvalsh(himmel_hess(s.x)))}
    values = list(range(20, 61, 1))
    sweep = [hx_study(v, starts=()) for v in values]
    As = np.linspace(0, HX["Amax"], 301)
    return {"rosen": {"points": rosen_pts, "boundary_c": 0.005,
                      "bfgs": {"x": r6(rs.x), "nit": int(rs.nit), "fun": float(rs.fun)}},
            "himmel": {"minima": mins, "starts": 100, "grid": r6(g), "assign": assign, "max": mx},
            "hx": {"data": HX, "study": hx_study(), "values": values,
                   "sweep": [{"value": q["value"], "best": q["best"], "hump": q["hump"], "interior": q["interior"]} for q in sweep],
                   "breakpoint": hx_breakpoint(), "A": r6(As), "tac": r6(hx_tac(As))}}


def classroom_results():
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "python": sys.version.split()[0]}}
    cb = camel_basins()
    R["lecture"] = {"quartic": quartic_worked(), "quad": quadratic_forms(), "gibbs": gibbs_equilibrium(),
                    "margules": margules(), "unit": UNIT, "parallel": parallel_sweep(), "vessel": vessel_gp(),
                    "camel": cb, "starts_needed": {"p": cb["p_global"], "N_99": starts_needed(cb["p_global"], 0.01),
                                                   "table": [{"p": p, "N": starts_needed(p, 0.01)} for p in (0.5, 0.2, 0.1, 0.05, 0.01)]},
                    "report": solver_report(), "abb": alpha_bb()}
    R["lab"] = lab_results()
    return R


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("convexity-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote convexity-results.json")
        sys.exit()
    L = R["lecture"]
    print("A quartic", [(p["point"], p["kind"], p["eig"]) for p in L["quartic"]["points"]])
    print("A gibbs exact", L["gibbs"]["exact"], "max deviation over starts", L["gibbs"]["max_deviation"])
    print("A margules", [(c["A"], c["min_second_derivative"], c["split"]) for c in L["margules"]["cases"]])
    for p in L["parallel"]:
        print("B parallel", p["total"], "ends", round(p["end_cost"], 2), "equal", round(p["equal_split_cost"], 2),
              "d2", round(p["second_derivative_at_equal_split"], 6), [(round(q["start"]), round(q["A1"], 1), round(q["cost"], 2)) for q in p["runs"]])
    print("B vessel", L["vessel"]["D_exact"], [(round(q["D"], 4), round(q["L"], 4)) for q in L["vessel"]["runs"]])
    cb = L["camel"]; print("C camel minima", [(m["x"], round(m["f"], 4), m["count"]) for m in cb["minima"]], "p_global", cb["p_global"], L["starts_needed"])
    print("C report", L["report"])
    ab = L["abb"]; print("C abb nodes", len(ab["nodes"]), "incumbent", ab["incumbent"], ab["x_best"], "complete", ab["complete"])
    lab = R["lab"]
    print("D rosen", lab["rosen"]["points"], lab["rosen"]["bfgs"])
    print("D himmel", lab["himmel"]["minima"], lab["himmel"]["max"])
    print("D hx", lab["hx"]["study"])
    print("D hx breakpoint", lab["hx"]["breakpoint"])
    print("D hx sweep", [(q["value"], round(q["best"]["A"], 1), round(q["best"]["tac"], 2)) for q in lab["hx"]["sweep"][::5]])
    for n in ab["nodes"]:
        print("  node", n["id"], n["parent"], round(n["l"], 3), round(n["u"], 3), "alpha", round(n["alpha"], 4), "lb", round(n["lb"], 4),
              "ub", round(n["ub"], 4), n["status"], "inc", round(n["incumbent"], 4), "glb", round(n["global_lb"], 4))
