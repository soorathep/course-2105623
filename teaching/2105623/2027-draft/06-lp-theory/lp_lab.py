"""2105623 Session 6: LP from the inside. Vertices and bases, the simplex method, duality and sensitivity.

Part A  Principles: the Session 1 plant in standard form, every basic solution, the fundamental theorem.
Part B  The simplex method by hand (exact fractions), Phase I, and the Klee-Minty cube.
Part C  Duality: the dual of the plant, strong duality from the final basis, complementary slackness.
Part D  Sensitivity: right-hand-side and cost ranging from the basis inverse, the value function v*(b),
        and a degenerate vertex with one-sided prices.
Part E  Lab 6: weekly production planning with limited reactor hours (synthetic, textbook style).

Runs in the browser (Pyodide: NumPy, SciPy with HiGHS) and in desktop Python.
    python lp_lab.py          print every part
    python lp_lab.py --json   write lp-results.json for the slides and the lab page
"""
import itertools
import json
import sys
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PLANT = {"products": ["Additive", "Binder"],
         "resources": ["Reactor train", "Separation and drying", "Finishing and packaging"],
         "A": [[3, 2], [1, 2], [1, 1]], "b": [24, 16, 9], "c": [400, 300],
         "units": "x in t/week; rows in h/week; margins in USD/t"}

LAB = {"products": ["Solvent", "Resin", "Plasticizer", "Additive"],
       "resources": ["Reactor hours", "Distillation hours", "Feedstock", "Plasticizer market"],
       "units": ["h/week", "h/week", "t/week", "t/week"],
       "A": [[3.0, 2.0, 5.0, 1.5], [1.0, 2.0, 1.0, 1.0], [1.1, 0.9, 1.3, 1.0], [0.0, 0.0, 1.0, 0.0]],
       "b": [400.0, 220.0, 160.0, 50.0], "c": [180.0, 140.0, 230.0, 90.0]}


def f2s(q):
    """A Fraction as a short string: 8, -400/3."""
    q = Fr(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def ffloat(q):
    return float(Fr(q))


# ---------------------------------------------------------------------------
# Part A. Standard form and basic solutions
# ---------------------------------------------------------------------------
def standard_form(P):
    """Add one slack per row: A x + s = b, x, s >= 0."""
    A = [[Fr(v) for v in row] for row in P["A"]]
    m, n = len(A), len(A[0])
    As = [row + [Fr(1 if i == k else 0) for k in range(m)] for i, row in enumerate(A)]
    names = [f"x{j+1}" for j in range(n)] + [f"s{i+1}" for i in range(m)]
    c = [Fr(v) for v in P["c"]] + [Fr(0)] * m
    return As, [Fr(v) for v in P["b"]], c, names


def solve_fr(B, rhs):
    """Gauss-Jordan elimination with Fractions; None if singular."""
    m = len(B); M = [list(B[i]) + [rhs[i]] for i in range(m)]
    for col in range(m):
        piv = next((r for r in range(col, m) if M[r][col] != 0), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]; M[col] = [v / p for v in M[col]]
        for r in range(m):
            if r != col and M[r][col] != 0:
                f = M[r][col]; M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [M[i][m] for i in range(m)]


def basic_solutions(P=PLANT):
    """Every choice of m basic columns: x_B = B^-1 b, the rest zero."""
    As, b, c, names = standard_form(P)
    m, N = len(As), len(As[0])
    out = []
    for basis in itertools.combinations(range(N), m):
        B = [[As[i][j] for j in basis] for i in range(m)]
        xb = solve_fr(B, b)
        if xb is None:
            out.append({"basis": [names[j] for j in basis], "singular": True}); continue
        x = [Fr(0)] * N
        for k, j in enumerate(basis):
            x[j] = xb[k]
        feas = all(v >= 0 for v in x)
        z = sum(ci * xi for ci, xi in zip(c, x))
        nonbasic = [names[j] for j in range(N) if j not in basis]
        out.append({"basis": [names[j] for j in basis], "nonbasic": nonbasic, "x": [f2s(v) for v in x],
                    "xf": [ffloat(v) for v in x], "feasible": feas, "z": ffloat(z),
                    "degenerate": feas and any(x[j] == 0 for j in basis)})
    return {"names": names, "bases": out}


# ---------------------------------------------------------------------------
# Part B. The simplex method
# ---------------------------------------------------------------------------
def simplex_trace(P=PLANT, rule="dantzig", max_iter=500, record=True):
    """Tableau simplex for max c x s.t. A x <= b, x >= 0 with b >= 0, starting from the slack basis.
    rule: 'dantzig' (largest reduced cost) or 'bland' (smallest index with positive reduced cost)."""
    As, b, c, names = standard_form(P)
    m, N = len(As), len(As[0])
    T = [row[:] + [b[i]] for i, row in enumerate(As)]          # m rows of [A | b]
    basis = list(range(N - m, N))
    its = []
    for it in range(max_iter):
        cb = [c[j] for j in basis]
        d = [c[j] - sum(cb[i] * T[i][j] for i in range(m)) for j in range(N)]   # reduced costs
        x = [Fr(0)] * N
        for i, j in enumerate(basis):
            x[j] = T[i][N]
        z = sum(c[j] * x[j] for j in range(N))
        cand = [j for j in range(N) if d[j] > 0]
        rec = {"iter": it, "basis": [names[j] for j in basis], "x": [f2s(v) for v in x], "z": f2s(z), "zf": ffloat(z),
               "d": [f2s(v) for v in d]}
        if record:
            rec["tableau"] = [[f2s(v) for v in row] for row in T]
        if not cand:
            rec["status"] = "optimal"
            y = solve_fr([[As[i][basis[k]] for i in range(m)] for k in range(m)], cb)   # B^T y = c_B
            rec["y"] = [f2s(v) for v in y]
            its.append(rec); break
        e = max(cand, key=lambda j: (d[j], -j)) if rule == "dantzig" else min(cand)
        ratios = [(T[i][N] / T[i][e]) if T[i][e] > 0 else None for i in range(m)]
        if all(r is None for r in ratios):
            rec["status"] = "unbounded"; rec["entering"] = names[e]; its.append(rec); break
        best = min(r for r in ratios if r is not None)
        rows = [i for i in range(m) if ratios[i] == best]
        r = min(rows, key=lambda i: basis[i]) if rule == "bland" else rows[0]
        rec.update({"status": "pivot", "entering": names[e], "ratios": [None if q is None else f2s(q) for q in ratios],
                    "leaving": names[basis[r]], "pivot": f2s(T[r][e]), "step": f2s(best)})
        its.append(rec)
        p = T[r][e]; T[r] = [v / p for v in T[r]]
        for i in range(m):
            if i != r and T[i][e] != 0:
                f = T[i][e]; T[i] = [a - f * bb for a, bb in zip(T[i], T[r])]
        basis[r] = e
    return {"names": names, "iterations": its, "pivots": sum(1 for q in its if q["status"] == "pivot")}


def phase_one_example():
    """Session 1's infeasible supply contract: packaging x1 + x2 <= 9 and a contract x1 + x2 >= 12.
    Phase I minimizes the artificial a in x1 + x2 - e + a = 12: its optimum is a = 3 > 0, so no plan exists."""
    # variables: x1, x2, s (packaging slack), e (surplus), a (artificial)
    A_eq = [[1, 1, 1, 0, 0], [1, 1, 0, -1, 1]]
    r = linprog([0, 0, 0, 0, 1], A_eq=A_eq, b_eq=[9, 12], bounds=[(0, None)] * 5, method="highs")
    return {"phase1_value": float(r.fun), "x": [float(v) for v in r.x], "farkas_y": [1, 1],
            "message": "Phase I optimum 3 > 0: no feasible plan"}


def klee_minty(n):
    """max sum 2^(n-j) x_j  s.t.  sum_{j<i} 2^(i-j+1) x_j + x_i <= 5^i,  x >= 0."""
    A = [[(2 ** (i - j + 1) if j < i else (1 if j == i else 0)) for j in range(1, n + 1)] for i in range(1, n + 1)]
    b = [5 ** i for i in range(1, n + 1)]
    c = [2 ** (n - j) for j in range(1, n + 1)]
    return {"A": A, "b": b, "c": c}


def klee_minty_table(nmax=10):
    rows = []
    for n in range(2, nmax + 1):
        P = klee_minty(n)
        tr = simplex_trace(P, rule="dantzig", record=False, max_iter=5000)
        ds = linprog([-v for v in P["c"]], A_ub=P["A"], b_ub=P["b"], method="highs-ds", options={"presolve": False})
        ip = linprog([-v for v in P["c"]], A_ub=P["A"], b_ub=P["b"], method="highs-ipm", options={"presolve": False})
        rows.append({"n": n, "dantzig_pivots": tr["pivots"], "two_pow": 2 ** n - 1,
                     "highs_simplex": int(ds.nit), "highs_ipm": int(ip.nit), "z": float(-ds.fun)})
    return rows


# ---------------------------------------------------------------------------
# Part C and D. Duality and sensitivity
# ---------------------------------------------------------------------------
def solve(P, b=None, c=None):
    """max c x s.t. A x <= b, x >= 0 with HiGHS; returns x, value, duals (USD per unit of each row)."""
    b = P["b"] if b is None else b; c = P["c"] if c is None else c
    r = linprog([-v for v in c], A_ub=P["A"], b_ub=b, bounds=[(0, None)] * len(c), method="highs")
    if r.status != 0:
        return {"status": int(r.status), "message": r.message}
    y = [-float(v) for v in r.ineqlin.marginals]
    return {"status": 0, "x": [float(v) for v in r.x], "v": float(-r.fun), "y": y,
            "slack": [float(v) for v in r.ineqlin.residual]}


def dual_lp(P):
    """min b y s.t. A^T y >= c, y >= 0."""
    A = np.array(P["A"], float)
    r = linprog(P["b"], A_ub=-A.T, b_ub=[-v for v in P["c"]], bounds=[(0, None)] * len(P["b"]), method="highs")
    return {"y": [float(v) for v in r.x], "w": float(r.fun)}


def ranging(P, tol=1e-9):
    """From the optimal basis: the interval of each b_i over which the basis (and so every price) stays,
    and the interval of each c_j over which the plan x stays."""
    sol = solve(P)
    A = np.array(P["A"], float); m, n = A.shape
    As = np.hstack([A, np.eye(m)]); c = np.array(list(P["c"]) + [0.0] * m)
    x_full = np.array(sol["x"] + sol["slack"])
    basis = [j for j in range(n + m) if x_full[j] > tol]
    # complete a degenerate basis with slacks of tight rows if needed
    while len(basis) < m:
        for j in range(n + m):
            if j not in basis and np.linalg.matrix_rank(As[:, basis + [j]]) == len(basis) + 1:
                basis.append(j); break
    basis = sorted(basis)
    B = As[:, basis]; Binv = np.linalg.inv(B); xb = Binv @ np.array(P["b"], float)
    rhs = []
    for i in range(m):
        col = Binv[:, i]
        lo = max([-xb[k] / col[k] for k in range(m) if col[k] > tol], default=-np.inf)
        hi = min([-xb[k] / col[k] for k in range(m) if col[k] < -tol], default=np.inf)
        rhs.append({"row": i, "b": P["b"][i], "lo": P["b"][i] + lo, "hi": P["b"][i] + hi, "price": sol["y"][i]})
    y = c[basis] @ Binv
    d = c - y @ As                                               # reduced costs (<= 0 at a max optimum)
    nonbasic = [j for j in range(n + m) if j not in basis]
    cost = []
    BinvA = Binv @ As
    for j in range(n):
        if j in basis:
            # c_j -> c_j + D changes the reduced costs to d_q - D * row_q; all must stay <= 0
            row = BinvA[basis.index(j)]
            lo_D = max([d[q] / row[q] for q in nonbasic if row[q] > tol], default=-np.inf)
            hi_D = min([d[q] / row[q] for q in nonbasic if row[q] < -tol], default=np.inf)
            cost.append({"j": j, "c": P["c"][j], "basic": True, "lo": P["c"][j] + lo_D, "hi": P["c"][j] + hi_D})
        else:
            cost.append({"j": j, "c": P["c"][j], "basic": False, "reduced_cost": float(d[j]),
                         "lo": -np.inf, "hi": P["c"][j] - float(d[j])})
    return {"basis": basis, "Binv": Binv.round(10).tolist(), "xb": xb.tolist(), "y": y.tolist(), "d": d.tolist(),
            "rhs": rhs, "cost": cost, "solution": sol}


def value_function(P, row, grid):
    """v*(b) and the price of row `row` as its right-hand side moves."""
    out = []
    for v in grid:
        b = list(P["b"]); b[row] = float(v)
        s = solve(P, b=b)
        out.append({"b": float(v), "v": s.get("v"), "price": s["y"][row] if s["status"] == 0 else None,
                    "x": s.get("x")})
    return out


def breakpoints(vf, tol=1e-6):
    """Where the price changes along a value-function sweep."""
    bps = []
    for a, b in zip(vf, vf[1:]):
        if a["price"] is not None and b["price"] is not None and abs(a["price"] - b["price"]) > tol:
            bps.append({"between": [a["b"], b["b"]], "price_before": a["price"], "price_after": b["price"]})
    return bps


def degenerate_example():
    """Add a feedstock row 2 x1 + x2 <= 15 that also passes through (6, 3): three rows tight, two variables.
    The prices are no longer unique: one more hour and one fewer hour have different values."""
    P = {"A": PLANT["A"] + [[2, 1]], "b": PLANT["b"] + [15], "c": PLANT["c"]}
    base = solve(P)
    one = []
    for i in range(4):
        up = list(P["b"]); up[i] += 1; dn = list(P["b"]); dn[i] -= 1
        one.append({"row": i, "solver_price": base["y"][i],
                    "gain_plus_one": solve(P, b=up)["v"] - base["v"], "loss_minus_one": base["v"] - solve(P, b=dn)["v"]})
    # the set of optimal duals: min and max of each y_i over {A^T y >= c, y >= 0, b y = v*}
    A = np.array(P["A"], float); rng = []
    for i in range(4):
        e = np.zeros(4); e[i] = 1
        lo = linprog(e, A_ub=-A.T, b_ub=[-v for v in P["c"]], A_eq=[P["b"]], b_eq=[base["v"]], bounds=[(0, None)] * 4, method="highs")
        hi = linprog(-e, A_ub=-A.T, b_ub=[-v for v in P["c"]], A_eq=[P["b"]], b_eq=[base["v"]], bounds=[(0, None)] * 4, method="highs")
        rng.append([float(lo.fun), float(-hi.fun)])
    return {"P": P, "solution": base, "one_sided": one, "dual_ranges": rng}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def lab_results():
    P = LAB
    base = solve(P); rg = ranging(P); du = dual_lp(P)
    reactor = value_function(P, 0, np.arange(200, 701, 5))
    plus1 = solve(P, b=[P["b"][0] + 1] + P["b"][1:])
    plus100 = solve(P, b=[P["b"][0] + 100] + P["b"][1:])
    # Additive (index 3) is not made: sweep its margin
    j_out = 3
    sweep = []
    for cm in np.arange(70.0, 150.01, 0.5):
        c = list(P["c"]); c[j_out] = float(cm); s = solve(P, c=c)
        sweep.append({"c": float(cm), "v": s["v"], "x": s["x"]})
    return {"data": P, "base": base, "ranging": rg, "dual": du, "reactor": reactor, "breakpoints": breakpoints(reactor),
            "plus1": plus1, "plus100": plus100, "not_made": j_out, "margin_sweep": sweep,
            "bty": float(np.dot(P["b"], base["y"]))}


def clean(o):
    """JSON-safe: infinite bounds become None (no limit), NumPy numbers become Python numbers."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def classroom_results():
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "python": sys.version.split()[0]}}
    tr = simplex_trace(PLANT, "dantzig"); trb = simplex_trace(PLANT, "bland")
    pv = value_function(PLANT, 0, np.arange(0, 40.01, 0.5))
    R["lecture"] = {"plant": PLANT, "bases": basic_solutions(PLANT), "simplex": tr, "simplex_bland": trb,
                    "phase1": phase_one_example(), "klee_minty": klee_minty_table(10),
                    "dual": dual_lp(PLANT), "ranging": ranging(PLANT), "reactor": pv, "breakpoints": breakpoints(pv),
                    "degenerate": degenerate_example()}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("lp-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote lp-results.json"); sys.exit()
    L = R["lecture"]
    for q in L["bases"]["bases"]:
        print("A basis", q["basis"], q.get("x"), q.get("feasible"), q.get("z"), "degenerate" if q.get("degenerate") else "")
    for it in L["simplex"]["iterations"]:
        print("B", it["iter"], it["basis"], it["x"], it["z"], "d", it["d"], it.get("entering"), it.get("ratios"), it.get("leaving"), it.get("y"))
    print("B bland pivots", L["simplex_bland"]["pivots"], [q["basis"] for q in L["simplex_bland"]["iterations"]])
    print("B phase1", L["phase1"])
    print("B klee-minty", [(q["n"], q["dantzig_pivots"], q["two_pow"], q["highs_simplex"], q["highs_ipm"]) for q in L["klee_minty"]])
    print("C dual", L["dual"])
    rg = L["ranging"]; print("D rhs", [(q["row"], q["lo"], q["hi"], round(q["price"], 3)) for q in rg["rhs"]])
    print("D cost", rg["cost"]); print("D breakpoints reactor", L["breakpoints"])
    dg = L["degenerate"]; print("D degenerate", dg["solution"]["x"], dg["solution"]["y"], dg["one_sided"], dg["dual_ranges"])
    lab = R["lab"]; print("E base", lab["base"]); print("E ranging rhs", [(q["row"], q["lo"], q["hi"], round(q["price"], 4)) for q in lab["ranging"]["rhs"]])
    print("E cost", lab["ranging"]["cost"]); print("E bp", lab["breakpoints"]); print("E plus1", lab["plus1"]["v"] - lab["base"]["v"], "plus100", lab["plus100"]["v"] - lab["base"]["v"])
    print("E dual", lab["dual"], "bty", lab["bty"])
