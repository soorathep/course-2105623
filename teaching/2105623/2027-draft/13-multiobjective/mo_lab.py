"""2105623 Session 13: multi-objective and nonconvex optimization (2027 draft).

Part A  Two objectives: a three-product plan, profit against wastewater (LP), by the epsilon-constraint method and
        by weighted sums; five heat-recovery projects (a discrete front with unsupported points); the batch reactor
        of Session 11 as a front of yield against batch time
Part B  A nonconvex problem: the pooling problem of Haverly (1978), its value as a function of the pool quality,
        local solvers from many starts, the McCormick relaxation and spatial branch and bound on the pool quality
Lab 13  A utility system (MILP): cost against CO2 by the epsilon-constraint method and by a carbon price;
        Haverly's second instance by multistart, by a scan of the pool quality, and by SCIP in Colab

Runs in desktop Python and in the browser (NumPy and SciPy: HiGHS through linprog and milp, SLSQP).
The global solver part needs PySCIPOpt (pip install pyscipopt), used in the Colab notebook only.
Notation: epsilon is the bound on the second objective; w is a weight; mu is the multiplier of an inequality row.
"""
import json, math, sys
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds, minimize


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


# ---------------------------------------------------------------------------
# Part A1. Three products, profit (USD/d) against wastewater (m3/d); a contract takes at least 10 t/d of P3
# ---------------------------------------------------------------------------
WW = {"names": ["P1", "P2", "P3"], "margin": [150.0, 100.0, 70.0], "water": [6.0, 2.5, 1.0],
      "reactor": [0.5, 0.4, 0.25], "sep": [0.3, 0.5, 0.2], "hours": [24.0, 20.0], "min": [0.0, 0.0, 10.0], "max": [30.0, 40.0, 50.0]}


def ww_lp(eps=None):
    """Maximize profit subject to reactor and separation hours, market limits and, if given, wastewater <= eps.
    Returns the plan, profit, wastewater and the multiplier mu of the wastewater row (USD per m3)."""
    W = WW; A = [W["reactor"], W["sep"]]; b = list(W["hours"])
    if eps is not None:
        A = A + [W["water"]]; b = b + [eps]
    r = linprog(-np.array(W["margin"]), A_ub=A, b_ub=b, bounds=list(zip(W["min"], W["max"])), method="highs")
    mu = -r.ineqlin.marginals
    return {"x": r.x.tolist(), "profit": float(-r.fun), "water": float(np.dot(W["water"], r.x)),
            "mu": float(mu[2]) if eps is not None else 0.0, "mu_hours": mu[:2].tolist()}


def ww_weighted(w, scale=None):
    """Weighted sum: maximize (1 - w) profit / P - w water / V, with P and V the ranges of the payoff table."""
    W = WW; P, V = scale or ww_ranges()
    c = -(1 - w) * np.array(W["margin"]) / P + w * np.array(W["water"]) / V
    r = linprog(c, A_ub=[W["reactor"], W["sep"]], b_ub=W["hours"], bounds=list(zip(W["min"], W["max"])), method="highs")
    return {"w": w, "x": r.x.tolist(), "profit": float(np.dot(W["margin"], r.x)), "water": float(np.dot(W["water"], r.x))}


def ww_ranges():
    """Ranges of the payoff table: the most profitable plan and the plan with the least wastewater."""
    top = ww_lp(); low = ww_lp(float(np.dot(WW["water"], WW["min"])))
    return top["profit"] - low["profit"], top["water"] - low["water"]


def ww_front(step=2.0):
    top = ww_lp(); pts = []; e0 = float(np.dot(WW["water"], WW["min"]))
    for e in np.arange(e0, top["water"] + 1e-9, step):
        q = ww_lp(float(e)); q["eps"] = float(e); pts.append(q)
    q = ww_lp(top["water"]); q["eps"] = top["water"]; pts.append(q)
    ws = {}
    for w in np.round(np.linspace(0, 1, 101), 2):
        s = ww_weighted(float(w)); ws.setdefault((round(s["water"], 6), round(s["profit"], 6)), []).append(float(w))
    return {"top": top, "pts": pts, "weighted": [[k[0], k[1], v[0], v[-1]] for k, v in sorted(ws.items())]}


# ---------------------------------------------------------------------------
# Part A2. Five heat-recovery projects: choose any subset; investment (kUSD) against energy saved (MWh/yr)
# ---------------------------------------------------------------------------
PROJ = {"names": ["E-101 preheat", "flash steam", "economizer", "pinch retrofit", "condensate return"],
        "cost": [120.0, 200.0, 80.0, 300.0, 150.0], "save": [900.0, 1300.0, 400.0, 2300.0, 700.0]}


def proj_front():
    """All 32 subsets; the nondominated ones (less cost, more saving); which of them a weighted sum (a price on
    energy) can select: the supported points, on the upper convex hull."""
    c = np.array(PROJ["cost"]); s = np.array(PROJ["save"]); allp = []
    for k in range(32):
        z = [(k >> i) & 1 for i in range(5)]; allp.append({"z": z, "cost": float(c @ z), "save": float(s @ z)})
    nd = [p for p in allp if not any((q["cost"] <= p["cost"] and q["save"] >= p["save"]) and (q["cost"] < p["cost"] or q["save"] > p["save"]) for q in allp)]
    nd.sort(key=lambda p: p["cost"])
    hull = []
    for p in nd:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (b["cost"] - a["cost"]) * (p["save"] - a["save"]) - (b["save"] - a["save"]) * (p["cost"] - a["cost"]) >= -1e-9:
                hull.pop()
            else:
                break
        hull.append(p)
    hs = {(p["cost"], p["save"]) for p in hull}
    for p in nd:
        p["supported"] = (p["cost"], p["save"]) in hs
    # the price of energy (kUSD per MWh/yr) at which each hull edge is chosen
    prices = [(hull[i + 1]["cost"] - hull[i]["cost"]) / (hull[i + 1]["save"] - hull[i]["save"]) for i in range(len(hull) - 1)]
    return {"all": allp, "nd": nd, "prices": prices}


def proj_cap(budget):
    """Most energy saved within an investment budget (the epsilon-constraint on cost): a 0-1 knapsack by MILP."""
    c = np.array(PROJ["cost"]); s = np.array(PROJ["save"])
    r = milp(-s, constraints=LinearConstraint(c.reshape(1, -1), -np.inf, budget), integrality=np.ones(5), bounds=Bounds(0, 1))
    z = np.round(r.x).astype(int)
    return {"budget": budget, "z": z.tolist(), "cost": float(c @ z), "save": float(s @ z)}


# ---------------------------------------------------------------------------
# Part A3. The batch reactor of Session 11: the best x2 for each batch time, then profit per hour of cycle
# ---------------------------------------------------------------------------
BR = {"h": 0.005, "k10": 4000.0, "E1": 2500.0, "k20": 620000.0, "E2": 5000.0}


def br_sim(Ts, tf):
    N = len(Ts); n = max(1, int(round(tf / N / BR["h"]))); h = tf / N / n; a, b = 1.0, 0.0
    for T in Ts:
        k1 = BR["k10"] * math.exp(-BR["E1"] / T); k2 = BR["k20"] * math.exp(-BR["E2"] / T)
        for _ in range(n):
            f1a = -k1 * a * a; f1b = k1 * a * a - k2 * b
            a2 = a + h / 2 * f1a; b2 = b + h / 2 * f1b; f2a = -k1 * a2 * a2; f2b = k1 * a2 * a2 - k2 * b2
            a3 = a + h / 2 * f2a; b3 = b + h / 2 * f2b; f3a = -k1 * a3 * a3; f3b = k1 * a3 * a3 - k2 * b3
            a4 = a + h * f3a; b4 = b + h * f3b; f4a = -k1 * a4 * a4; f4b = k1 * a4 * a4 - k2 * b4
            a += h / 6 * (f1a + 2 * f2a + 2 * f3a + f4a); b += h / 6 * (f1b + 2 * f2b + 2 * f3b + f4b)
    return b


def br_best(tf, N=10):
    """Epsilon-constraint with the batch time fixed: maximize x2(tf) over 10 temperatures, T = 298 + 100 u."""
    r = minimize(lambda u: -br_sim(298 + 100 * u, tf), np.full(N, 0.4), method="SLSQP", bounds=[(0, 1)] * N,
                 options={"ftol": 1e-12, "maxiter": 300})
    return {"tf": tf, "x2": float(-r.fun), "T": (298 + 100 * r.x).tolist(), "nit": int(r.nit)}


def br_front():
    return [br_best(float(t)) for t in np.round(np.arange(0.25, 3.0001, 0.125), 3)]


# ---------------------------------------------------------------------------
# Part B. Haverly's pooling problem (P-formulation). Feeds A (3% S) and B (1% S) go to one pool; C (2% S) goes
#   to the products directly. Products X (at most 2.5% S) and Y (at most 1.5% S).
#   Instance 1: costs A 6, B 16, C 10; prices X 9, Y 15; demands X 100, Y 200 (Haverly, 1978).
# ---------------------------------------------------------------------------
HAV1 = {"cA": 6.0, "cB": 16.0, "cC": 10.0, "pX": 9.0, "pY": 15.0, "dX": 100.0, "dY": 200.0}


def hav_lp(q, H=None):
    """With the pool quality q fixed, the problem is an LP in the flows A, B, Px, Py (pool to X, Y), Cx, Cy."""
    H = HAV1 if H is None else H
    c = np.array([H["cA"], H["cB"], -H["pX"], -H["pY"], H["cC"] - H["pX"], H["cC"] - H["pY"]])
    Aeq = [[1, 1, -1, -1, 0, 0], [3, 1, -q, -q, 0, 0]]
    Aub = [[0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, q - 2.5, 0, -0.5, 0], [0, 0, 0, q - 1.5, 0, 0.5]]
    r = linprog(c, A_ub=Aub, b_ub=[H["dX"], H["dY"], 0, 0], A_eq=Aeq, b_eq=[0, 0], bounds=[(0, None)] * 6, method="highs")
    return {"q": q, "profit": float(-r.fun), "x": r.x.tolist()}


def hav_scan(H=None, n=201):
    return [hav_lp(float(q), H) for q in np.linspace(1, 3, n)]


def hav_nlp(H=None):
    """The full nonconvex problem for SLSQP. z = [A, B, Cx, Cy, Px, Py, q]; maximize profit."""
    H = HAV1 if H is None else H
    f = lambda z: -(H["pX"] * (z[4] + z[2]) + H["pY"] * (z[5] + z[3]) - H["cA"] * z[0] - H["cB"] * z[1] - H["cC"] * (z[2] + z[3]))
    cons = [{"type": "eq", "fun": lambda z: z[0] + z[1] - z[4] - z[5]},
            {"type": "eq", "fun": lambda z: 3 * z[0] + z[1] - z[6] * (z[4] + z[5])},
            {"type": "ineq", "fun": lambda z: 2.5 * (z[4] + z[2]) - z[6] * z[4] - 2 * z[2]},
            {"type": "ineq", "fun": lambda z: 1.5 * (z[5] + z[3]) - z[6] * z[5] - 2 * z[3]},
            {"type": "ineq", "fun": lambda z: H["dX"] - z[4] - z[2]},
            {"type": "ineq", "fun": lambda z: H["dY"] - z[5] - z[3]}]
    bounds = [(0, H["dX"] + H["dY"])] * 2 + [(0, H["dX"]), (0, H["dY"]), (0, H["dX"]), (0, H["dY"]), (1, 3)]
    return f, cons, bounds


def hav_multistart(H=None, n=100, seed=13):
    """SLSQP from n random starts (uniform in the bounds). Returns, per start, the starting q and the final profit,
    or None when SLSQP fails or the answer breaks a row by more than 1e-5."""
    f, cons, b = hav_nlp(H); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        z0 = np.array([rng.uniform(lo, hi) for lo, hi in b])
        r = minimize(f, z0, method="SLSQP", bounds=b, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        viol = max(abs(c["fun"](r.x)) if c["type"] == "eq" else max(0.0, -c["fun"](r.x)) for c in cons)
        ok = bool(r.success) and viol < 1e-5
        out.append({"q0": float(z0[6]), "profit": round(float(-r.fun), 1) + 0.0 if ok else None, "q": float(r.x[6]), "nit": int(r.nit)})
    return out


def tally(runs):
    t = {}
    for r in runs:
        k = "fail" if r["profit"] is None else f"{r['profit']:.1f}"
        t[k] = t.get(k, 0) + 1
    return dict(sorted(t.items(), key=lambda kv: -kv[1]))


def hav_relax(qL, qU, H=None):
    """McCormick relaxation of the P-formulation for q in [qL, qU]: w = q P replaced by its four envelope rows,
    with 0 <= Px <= dX and 0 <= Py <= dY. Returns the upper bound on profit and the q of the relaxed answer."""
    H = HAV1 if H is None else H; n = 9                      # A B Cx Cy Px Py q wx wy
    c = np.zeros(n); c[0] = H["cA"]; c[1] = H["cB"]; c[2] = H["cC"] - H["pX"]; c[3] = H["cC"] - H["pY"]; c[4] = -H["pX"]; c[5] = -H["pY"]
    Aeq = [[1, 1, 0, 0, -1, -1, 0, 0, 0], [3, 1, 0, 0, 0, 0, 0, -1, -1]]
    A = [[0, 0, -0.5, 0, -2.5, 0, 0, 1, 0], [0, 0, 0, 0.5, 0, -1.5, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0]]
    b = [0, 0, H["dX"], H["dY"]]
    for iP, iw, PU in ((4, 7, H["dX"]), (5, 8, H["dY"])):
        r = np.zeros(n); r[iw] = -1; r[iP] = qL; A.append(r); b.append(0)
        r = np.zeros(n); r[iw] = -1; r[iP] = qU; r[6] = PU; A.append(r); b.append(qU * PU)
        r = np.zeros(n); r[iw] = 1; r[iP] = -qU; A.append(r); b.append(0)
        r = np.zeros(n); r[iw] = 1; r[iP] = -qL; r[6] = -PU; A.append(r); b.append(-qL * PU)
    bnd = [(0, None)] * 6 + [(qL, qU), (None, None), (None, None)]
    r = linprog(c, A_ub=np.array(A, float), b_ub=b, A_eq=Aeq, b_eq=[0, 0], bounds=bnd, method="highs")
    return {"qL": qL, "qU": qU, "ub": float(-r.fun), "q": float(r.x[6]), "x": r.x.tolist()}


def hav_bb(H=None, tol=1e-6):
    """Spatial branch and bound on q: the McCormick LP gives an upper bound on each interval; the LP with q fixed at
    the relaxed q and at the interval ends gives feasible plans (lower bounds). Branch at the relaxed q."""
    inc, incq, nodes, stack, k = -1e18, None, [], [(1.0, 3.0, 0)], 0
    while stack and k < 50:
        lo, hi, d = stack.pop(0); k += 1; R = hav_relax(lo, hi, H)
        cand = max((hav_lp(qq, H)["profit"], qq) for qq in (R["q"], lo, hi))
        if cand[0] > inc + tol:
            inc, incq = cand
        status = "pruned" if R["ub"] <= inc + tol else "branched"
        nodes.append({"k": k, "lo": lo, "hi": hi, "depth": d, "ub": R["ub"], "qrel": R["q"], "inc": inc, "status": status})
        if status == "branched":
            mid = R["q"] if lo + 1e-3 < R["q"] < hi - 1e-3 else (lo + hi) / 2
            stack += [(lo, mid, d + 1), (mid, hi, d + 1)]
    return {"nodes": nodes, "best": inc, "q": incq}


def hav_scip(H=None, verbose=False, nodelimit=None, logfile=None):
    """Global solution by SCIP (PySCIPOpt), which handles the bilinear rows by spatial branch and bound."""
    try:
        from pyscipopt import Model
    except ImportError:
        return None
    H = HAV1 if H is None else H; m = Model("haverly")
    if not verbose:
        m.hideOutput()
    if logfile:
        m.setLogfile(logfile)
    if nodelimit:
        m.setParam("limits/nodes", nodelimit)
    A = m.addVar("A", lb=0); B = m.addVar("B", lb=0); Cx = m.addVar("Cx", lb=0); Cy = m.addVar("Cy", lb=0)
    Px = m.addVar("Px", lb=0, ub=H["dX"]); Py = m.addVar("Py", lb=0, ub=H["dY"]); q = m.addVar("q", lb=1, ub=3)
    m.addCons(A + B == Px + Py, "pool"); m.addCons(3 * A + B == q * (Px + Py), "quality")
    m.addCons(q * Px + 2 * Cx <= 2.5 * (Px + Cx), "specX"); m.addCons(q * Py + 2 * Cy <= 1.5 * (Py + Cy), "specY")
    m.addCons(Px + Cx <= H["dX"], "demandX"); m.addCons(Py + Cy <= H["dY"], "demandY")
    m.setObjective(H["pX"] * (Px + Cx) + H["pY"] * (Py + Cy) - H["cA"] * A - H["cB"] * B - H["cC"] * (Cx + Cy), "maximize")
    m.optimize()
    return {"status": m.getStatus(), "profit": m.getObjVal(), "dual_bound": m.getDualbound(), "gap": m.getGap(),
            "nodes": m.getNTotalNodes(), "q": m.getVal(q), "flows": {v.name: m.getVal(v) for v in (A, B, Cx, Cy, Px, Py)},
            "scip": m.version()}


# ---------------------------------------------------------------------------
# Lab 13A. A utility system for 60 MW of steam and 20 MW of electricity (synthetic, illustrative prices).
#   Coal, gas and biomass boilers, a heat pump on grid power, a gas-turbine CHP, and grid import.
#   Each unit has an on/off binary, a minimum load and a fixed cost per hour when on. Cost in USD/h, CO2 in t/h.
#   CO2: coal 94.6 and natural gas 56.1 t/TJ (IPCC 2006 defaults), biomass counted as zero, grid 0.45 t/MWh.
# ---------------------------------------------------------------------------
UT = {"S": 60.0, "E": 20.0,
      "coal": {"cap": 60.0, "mn": 24.0, "eff": 0.85, "fuel": 10.0, "ef": 0.3406, "fix": 600.0},
      "gas": {"cap": 60.0, "mn": 12.0, "eff": 0.92, "fuel": 32.0, "ef": 0.2020, "fix": 150.0},
      "bio": {"cap": 30.0, "mn": 15.0, "eff": 0.80, "fuel": 22.0, "ef": 0.0, "fix": 500.0},
      "hp": {"cap": 20.0, "mn": 5.0, "cop": 2.5, "fix": 300.0},
      "chp": {"Fmax": 60.0, "Fmin": 30.0, "ee": 0.35, "es": 0.45, "fuel": 32.0, "ef": 0.2020, "fix": 400.0},
      "grid": {"cap": 40.0, "price": 95.0, "ef": 0.45}}
UNITS = ["coal boiler", "gas boiler", "biomass boiler", "heat pump", "gas-turbine CHP"]


def ut_model():
    """Variables: steam from coal, gas, biomass, heat pump (MW); CHP fuel F (MW); grid import g (MW);
    binaries for the five units. Rows: steam >= 60 (CHP gives 0.45 F); electricity = 20 (CHP 0.35 F, grid, minus
    the heat pump's power); minimum and maximum loads."""
    P = UT; n = 11; C = np.zeros(n); EM = np.zeros(n)
    for i, k in enumerate(["coal", "gas", "bio"]):
        u = P[k]; C[i] = u["fuel"] / u["eff"]; EM[i] = u["ef"] / u["eff"]; C[6 + i] = u["fix"]
    C[9] = P["hp"]["fix"]; C[4] = P["chp"]["fuel"]; EM[4] = P["chp"]["ef"]; C[10] = P["chp"]["fix"]
    C[5] = P["grid"]["price"]; EM[5] = P["grid"]["ef"]
    A = []; lo = []; hi = []
    r = np.zeros(n); r[:4] = 1; r[4] = P["chp"]["es"]; A.append(r); lo.append(P["S"]); hi.append(np.inf)
    r = np.zeros(n); r[4] = P["chp"]["ee"]; r[5] = 1; r[3] = -1 / P["hp"]["cop"]; A.append(r); lo.append(P["E"]); hi.append(P["E"])
    for i, k in enumerate(["coal", "gas", "bio", "hp"]):
        u = P[k]; r = np.zeros(n); r[i] = 1; r[6 + i] = -u["cap"]; A.append(r); lo.append(-np.inf); hi.append(0)
        r = np.zeros(n); r[i] = 1; r[6 + i] = -u["mn"]; A.append(r); lo.append(0); hi.append(np.inf)
    r = np.zeros(n); r[4] = 1; r[10] = -P["chp"]["Fmax"]; A.append(r); lo.append(-np.inf); hi.append(0)
    r = np.zeros(n); r[4] = 1; r[10] = -P["chp"]["Fmin"]; A.append(r); lo.append(0); hi.append(np.inf)
    integ = np.array([0] * 6 + [1] * 5); bnd = Bounds([0] * n, [np.inf] * 5 + [P["grid"]["cap"]] + [1] * 5)
    return C, EM, np.array(A), np.array(lo), np.array(hi), integ, bnd


def ut_solve(obj, eps=None):
    C, EM, A, lo, hi, integ, bnd = ut_model()
    if eps is not None:
        A = np.vstack([A, EM]); lo = np.r_[lo, -np.inf]; hi = np.r_[hi, eps]
    r = milp(obj, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=bnd, options={"mip_rel_gap": 1e-10})
    x = r.x; on = [int(round(v)) for v in x[6:]]
    return {"cost": float(C @ x), "co2": float(EM @ x), "on": on, "steam": x[:4].tolist(), "F": float(x[4]), "grid": float(x[5])}


def ut_results():
    C, EM = ut_model()[:2]
    cmin = ut_solve(C); emin = ut_solve(EM); emin_lex = ut_solve(C, eps=emin["co2"] + 1e-7)
    eps = np.linspace(emin_lex["co2"], cmin["co2"], 21); front = []
    for e in eps:
        q = ut_solve(C, eps=float(e) + 1e-9); q["eps"] = float(e); front.append(q)
    prices = []
    for p in range(0, 201, 1):
        q = ut_solve(C + p * EM); q["price"] = p; prices.append(q)
    return {"data": UT, "units": UNITS, "cost_min": cmin, "co2_min": emin_lex, "front": front, "prices": prices}


def lab_results():
    H2 = dict(HAV1, dX=600.0)
    ms = hav_multistart(H2)
    return {"utility": ut_results(), "hav2": {"data": H2, "scan": hav_scan(H2), "multistart": ms, "tally": tally(ms),
                                              "root_bound": hav_relax(1.0, 3.0, H2)["ub"]}}


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
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def classroom_results():
    import scipy
    R = {"meta": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}}
    ms = hav_multistart()
    R["lecture"] = {"ww": ww_front(), "ww_data": WW, "proj": proj_front(), "proj_data": PROJ,
                    "proj_caps": [proj_cap(float(b)) for b in range(0, 851, 10)], "batch": br_front(),
                    "hav1": {"data": HAV1, "scan": hav_scan(), "multistart": ms, "tally": tally(ms), "bb": hav_bb(),
                             "root": hav_relax(1.0, 3.0)}}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    if "--json" in sys.argv:
        R = classroom_results()
        with open("mo-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote mo-results.json", R["meta"]); sys.exit()
    F = ww_front(); print("A1 top", {k: r6(v) for k, v in F["top"].items()})
    print("   weighted-sum vertices", [(r6(a), r6(b)) for a, b, _, _ in F["weighted"]])
    print("   mu along the front", sorted({round(p["mu"], 4) for p in F["pts"]}))
    P = proj_front(); print("A2 nondominated", [(p["cost"], p["save"], p["supported"]) for p in P["nd"]], "prices", r6(P["prices"]))
    B = br_front(); print("A3 batch", [(b["tf"], r6(b["x2"])) for b in B])
    print("B scan H1 best", max((s["profit"], s["q"]) for s in hav_scan()))
    print("B multistart H1", tally(hav_multistart()))
    print("B root", r6(hav_relax(1, 3)["ub"]), "bb", [(n["k"], r6(n["lo"]), r6(n["hi"]), r6(n["ub"]), n["status"]) for n in hav_bb()["nodes"]])
    print("B SCIP", hav_scip())
    L = lab_results(); U = L["utility"]
    print("Lab cost-min", r6(U["cost_min"]["cost"]), r6(U["cost_min"]["co2"]), U["cost_min"]["on"], "co2-min", r6(U["co2_min"]["cost"]), r6(U["co2_min"]["co2"]), U["co2_min"]["on"])
    for q in U["front"]:
        print("   eps", r6(q["eps"]), "cost", r6(q["cost"]), "co2", r6(q["co2"]), q["on"])
    seen = []
    for q in U["prices"]:
        key = tuple(q["on"]) + (round(q["co2"], 3),)
        if not seen or seen[-1][1] != key:
            seen.append((q["price"], key, round(q["cost"], 2)))
    print("   by price", seen)
    print("Lab H2 tally", L["hav2"]["tally"], "best scan", max((s["profit"], s["q"]) for s in L["hav2"]["scan"]), "root", L["hav2"]["root_bound"])
    print("Lab H2 SCIP", hav_scip(L["hav2"]["data"]))
