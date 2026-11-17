"""2105623 Session 2: LP models for blending, planning and flowsheets.

Part A  Lecture: blending two gasoline grades from five components. Ratio
        specifications written as linear rows; Reid vapor pressure blended on
        the right basis (an index) or the wrong one (linear); the vertex count;
        convex and concave price tiers; a linear-fractional objective.
Part B  Lab 2: a small refinery written as a stream table (one balance row per
        stream, one column per activity), with the gasoline pool on top.

All data are synthetic but of realistic size. Volumes in m3/d, prices in USD/m3.
Runs in the browser (Pyodide: NumPy, SciPy/HiGHS) and in desktop Python.
    python blending_lab.py          print every part
    python blending_lab.py --json   write blending-results.json for the slides and the lab page
"""
import json
import sys

import numpy as np
from scipy.optimize import linprog

EXPO = 1.25                      # RVP blending index: I(R) = R ** 1.25 (blends linearly by volume)


def rvp_index(r):
    return np.asarray(r, float) ** EXPO


def rvp_from_index(i):
    return float(np.asarray(i, float) ** (1 / EXPO))


class LP:
    """A tiny named-row, named-column LP builder on top of scipy.optimize.linprog (HiGHS)."""

    def __init__(self):
        self.cols, self.obj, self.ub = [], [], []
        self.rows = []                                   # (name, {col: coef}, sense, rhs)

    def var(self, name, obj=0.0, ub=None):
        self.cols.append(name); self.obj.append(obj); self.ub.append(ub)

    def row(self, name, terms, sense, rhs=0.0):
        self.rows.append((name, dict(terms), sense, float(rhs)))

    def solve(self, maximize=True):
        n, ix = len(self.cols), {c: k for k, c in enumerate(self.cols)}
        def mat(sel):
            R = [r for r in self.rows if r[2] in sel]
            A = np.zeros((len(R), n)); b = np.zeros(len(R))
            for k, (_, t, s, rhs) in enumerate(R):
                sg = -1.0 if s == ">=" else 1.0
                for c, a in t.items():
                    A[k, ix[c]] += sg * a
                b[k] = sg * rhs
            return R, A, b
        Ru, Au, bu = mat(("<=", ">=")); Re, Ae, be = mat(("==",))
        c = -np.array(self.obj) if maximize else np.array(self.obj)
        res = linprog(c, A_ub=Au if len(Ru) else None, b_ub=bu if len(Ru) else None,
                      A_eq=Ae if len(Re) else None, b_eq=be if len(Re) else None,
                      bounds=[(0, u) for u in self.ub], method="highs")
        if res.status != 0:
            return {"status": {2: "infeasible", 3: "unbounded"}.get(res.status, res.message)}
        x = {k: float(v) for k, v in zip(self.cols, res.x.round(9) + 0.0)}
        sg = -1.0 if maximize else 1.0
        duals, slack = {}, {}
        for (name, t, s, rhs), y in zip(Ru, res.ineqlin.marginals if len(Ru) else []):
            duals[name] = float(sg * y * (-1.0 if s == ">=" else 1.0))
            slack[name] = float(abs(rhs - sum(a * x[c] for c, a in t.items())))
        for (name, t, s, rhs), y in zip(Re, res.eqlin.marginals if len(Re) else []):
            duals[name] = float(sg * y); slack[name] = 0.0
        scale = {r[0]: max(1.0, abs(r[3]), sum(abs(a * x[c]) for c, a in r[1].items())) for r in self.rows}
        tight_rows = [r[0] for r in self.rows if slack[r[0]] <= 1e-7 * scale[r[0]]]
        tight_ub = [c for c, u in zip(self.cols, self.ub) if u is not None and abs(x[c] - u) <= 1e-7 * max(1, u)]
        return {"status": "optimal", "obj": float(sg * res.fun), "x": x, "duals": duals, "slack": slack,
                "tight_rows": tight_rows, "tight_ub": tight_ub,
                "positive": [c for c in self.cols if x[c] > 1e-7]}


# ---------------------------------------------------------------------------
# Part A. Lecture: a two-grade gasoline blend
# ---------------------------------------------------------------------------
COMP = ["Butane", "LSR naphtha", "Reformate", "FCC gasoline", "Alkylate"]
RON = {"Butane": 94.0, "LSR naphtha": 70.0, "Reformate": 98.0, "FCC gasoline": 92.0, "Alkylate": 96.0}
RVP = {"Butane": 360.0, "LSR naphtha": 80.0, "Reformate": 25.0, "FCC gasoline": 50.0, "Alkylate": 30.0}   # kPa
COST = {"Butane": 380.0, "LSR naphtha": 520.0, "Reformate": 640.0, "FCC gasoline": 600.0, "Alkylate": 700.0}
AVAIL = {"Butane": 300.0, "LSR naphtha": 900.0, "Reformate": 1200.0, "FCC gasoline": 1500.0, "Alkylate": 600.0}
GRADES = ["Regular", "Premium"]
RON_MIN = {"Regular": 91.0, "Premium": 95.0}
RVP_MAX = 60.0
PRICE = {"Regular": 690.0, "Premium": 750.0}
DEMAND = {"Regular": 2500.0, "Premium": 1500.0}


def blend(ron_min=None, rvp_max=RVP_MAX, rvp_basis="index", avail=None, price=None, demand=None):
    """Maximize the daily margin of the two-grade blend.
    rvp_basis = 'index' blends R**1.25 by volume (correct); 'linear' blends R itself (the common error)."""
    ron_min = dict(RON_MIN, **(ron_min or {})); avail = dict(AVAIL, **(avail or {}))
    price = dict(PRICE, **(price or {})); demand = dict(DEMAND, **(demand or {}))
    f = rvp_index if rvp_basis == "index" else (lambda r: np.asarray(r, float))
    m = LP()
    for i in COMP:
        for g in GRADES:
            m.var(f"{i}>{g}", obj=price[g] - COST[i])
    for i in COMP:
        m.row(f"avail {i}", {f"{i}>{g}": 1 for g in GRADES}, "<=", avail[i])
    for g in GRADES:
        m.row(f"demand {g}", {f"{i}>{g}": 1 for i in COMP}, "<=", demand[g])
        # sum_i (RON_i - RON_min) x_ig >= 0   <=>   blend RON >= RON_min whenever the grade is made
        m.row(f"RON {g}", {f"{i}>{g}": RON[i] - ron_min[g] for i in COMP}, ">=")
        m.row(f"RVP {g}", {f"{i}>{g}": float(f(RVP[i]) - f(rvp_max)) for i in COMP}, "<=")
    r = m.solve()
    if r["status"] != "optimal":
        return r
    out = {"status": "optimal", "margin": r["obj"], "recipe": {}, "props": {}, "tight": r["tight_rows"],
           "positive": r["positive"], "duals": r["duals"]}
    for g in GRADES:
        v = {i: r["x"][f"{i}>{g}"] for i in COMP}; V = sum(v.values())
        out["recipe"][g] = v
        if V > 1e-9:
            w = {i: v[i] / V for i in COMP}
            out["props"][g] = {"volume": V, "RON": sum(w[i] * RON[i] for i in COMP),
                               "RVP_true": rvp_from_index(sum(w[i] * rvp_index(RVP[i]) for i in COMP)),
                               "RVP_linear": sum(w[i] * RVP[i] for i in COMP)}
        else:
            out["props"][g] = {"volume": 0.0}
    out["n_positive"], out["n_tight"] = len(r["positive"]), len(r["tight_rows"])
    return out


def butane_curve(base=("Reformate", "FCC gasoline"), fractions=np.linspace(0, 0.2, 21)):
    """True (index) and linear RVP of a 50/50 reformate + FCC gasoline base with added butane."""
    out = []
    for fb in fractions:
        w = {base[0]: (1 - fb) / 2, base[1]: (1 - fb) / 2, "Butane": fb}
        out.append({"butane": float(fb), "linear": sum(w[i] * RVP[i] for i in w),
                    "true": rvp_from_index(sum(w[i] * rvp_index(RVP[i]) for i in w))})
    return out


TIER = 300.0                     # m3/d in each alkylate price tier


def _tier_lp(p1, p2, tier1_lb=0.0, tier2_ub=TIER, tier1_ub=TIER):
    m = LP()
    for i in COMP:
        for g in GRADES:
            m.var(f"{i}>{g}", obj=PRICE[g] - (0.0 if i == "Alkylate" else COST[i]))
    m.var("tier1", obj=-p1, ub=tier1_ub); m.var("tier2", obj=-p2, ub=tier2_ub)
    if tier1_lb:
        m.row("tier 1 used first", {"tier1": 1}, ">=", tier1_lb)
    m.row("alkylate bought", {**{f"Alkylate>{g}": 1 for g in GRADES}, "tier1": -1, "tier2": -1}, "==")
    for i in COMP:
        if i != "Alkylate":
            m.row(f"avail {i}", {f"{i}>{g}": 1 for g in GRADES}, "<=", AVAIL[i])
    for g in GRADES:
        m.row(f"demand {g}", {f"{i}>{g}": 1 for i in COMP}, "<=", DEMAND[g])
        m.row(f"RON {g}", {f"{i}>{g}": RON[i] - RON_MIN[g] for i in COMP}, ">=")
        m.row(f"RVP {g}", {f"{i}>{g}": float(rvp_index(RVP[i]) - rvp_index(RVP_MAX)) for i in COMP}, "<=")
    return m.solve()


def price_tiers(rising=(720.0, 780.0), discount=(780.0, 700.0)):
    """Alkylate bought in two tiers of 300 m3/d. Rising prices (a convex cost) are an LP.
    A volume discount (a concave cost) is not: the LP buys the cheap second tier without the first.
    The correct answer for the discount enumerates the two segments, which is what a binary does."""
    out = {}
    for name, (p1, p2) in (("rising", rising), ("discount", discount)):
        r = _tier_lp(p1, p2); t1, t2 = r["x"]["tier1"], r["x"]["tier2"]; q = t1 + t2
        true_cost = p1 * min(q, TIER) + p2 * max(q - TIER, 0.0)
        out[name] = {"p1": p1, "p2": p2, "tier1": t1, "tier2": t2, "alkylate": q,
                     "lp_cost": p1 * t1 + p2 * t2, "true_cost": true_cost, "lp_margin": r["obj"],
                     "true_margin": r["obj"] + (p1 * t1 + p2 * t2) - true_cost}
    p1, p2 = discount
    seg1 = _tier_lp(p1, p2, tier2_ub=0.0)                        # stay in tier 1
    seg2 = _tier_lp(p1, p2, tier1_lb=TIER)                       # fill tier 1, then tier 2
    best = max((seg1, seg2), key=lambda r: r["obj"])
    out["discount"]["correct"] = {"alkylate": best["x"]["tier1"] + best["x"]["tier2"], "margin": best["obj"],
                                  "segments": [seg1["obj"], seg2["obj"]]}
    return out


def charnes_cooper(fixed=40000.0, vmax=3000.0):
    """Cheapest premium blend per m3 when the batch carries a fixed daily cost F:
       min (sum c_i x_i + F) / sum x_i   s.t. specs, availability, sum x_i <= vmax.
    Charnes-Cooper: t = 1 / sum x_i, y = t x gives an LP in (y, t)."""
    m = LP()
    for i in COMP:
        m.var(f"y {i}", obj=COST[i])
    m.var("t", obj=fixed)
    m.row("normalize", {f"y {i}": 1 for i in COMP}, "==", 1.0)
    for i in COMP:
        m.row(f"avail {i}", {f"y {i}": 1, "t": -AVAIL[i]}, "<=")
    m.row("tank", {"t": vmax}, ">=", 1.0)                      # sum x = 1/t <= vmax
    m.row("RON", {f"y {i}": RON[i] - RON_MIN["Premium"] for i in COMP}, ">=")
    m.row("RVP", {f"y {i}": float(rvp_index(RVP[i]) - rvp_index(RVP_MAX)) for i in COMP}, "<=")
    r = m.solve(maximize=False); t = r["x"]["t"]
    V = 1 / t
    # check by brute force: for a fixed volume V the problem is an ordinary LP
    def cost_at(V):
        q = LP()
        for i in COMP:
            q.var(i, obj=COST[i], ub=AVAIL[i])
        q.row("volume", {i: 1 for i in COMP}, "==", V)
        q.row("RON", {i: RON[i] - RON_MIN["Premium"] for i in COMP}, ">=")
        q.row("RVP", {i: float(rvp_index(RVP[i]) - rvp_index(RVP_MAX)) for i in COMP}, "<=")
        s = q.solve(maximize=False)
        return (s["obj"] + fixed) / V if s["status"] == "optimal" else None
    scan = [{"V": float(v), "cost_per_m3": cost_at(v)} for v in np.linspace(500, vmax, 26)]
    return {"fixed": fixed, "vmax": vmax, "volume": V, "cost_per_m3": r["obj"],
            "recipe": {i: r["x"][f"y {i}"] * V for i in COMP}, "scan": scan}


def spec_rows():
    """The eight specification rows of the two-grade blend, coefficient by coefficient."""
    out = []
    for i in COMP:
        out.append({"comp": i, "RON": RON[i], "RVP": RVP[i], "index": float(rvp_index(RVP[i])),
                    "ron_reg": RON[i] - RON_MIN["Regular"], "ron_pre": RON[i] - RON_MIN["Premium"],
                    "rvp_row": float(rvp_index(RVP[i]) - rvp_index(RVP_MAX))})
    return {"rows": out, "index_spec": float(rvp_index(RVP_MAX))}


def recompute_premium():
    """Recompute Premium's octane and true RVP from the optimal recipe, term by term."""
    b = blend(); v = b["recipe"]["Premium"]; V = sum(v.values())
    terms = [{"comp": i, "v": v[i], "v_ron": v[i] * RON[i], "index": float(rvp_index(RVP[i])),
              "v_index": float(v[i] * rvp_index(RVP[i]))} for i in COMP if v[i] > 1e-9]
    avg_index = sum(t["v_index"] for t in terms) / V
    return {"terms": terms, "volume": V, "ron": sum(t["v_ron"] for t in terms) / V,
            "avg_index": avg_index, "rvp_true": rvp_from_index(avg_index)}


def spec_dual():
    """The dual of the Premium octane row, converted into USD/d per octane number:
    raising Q by dQ is the same as asking sum_i (RON_i - Q) x_i >= dQ * V, so d(margin)/dQ = dual * V."""
    b = blend(); y = b["duals"]["RON Premium"]; V = b["props"]["Premium"]["volume"]
    up = blend(ron_min={"Premium": RON_MIN["Premium"] + 0.01})
    return {"dual": y, "volume": V, "predicted_per_ron": y * V,
            "finite_difference_per_ron": (up["margin"] - b["margin"]) / 0.01}


def line_fits(outlier=False):
    """Fit y = a + b x to plant data three ways. L1 and L-infinity fits are LPs (epigraph);
    least squares is a QP solved here by lstsq. An outlier shows how each one responds."""
    x = np.array([10, 15, 20, 25, 30, 35, 40, 45], float)
    y = np.array([7.1, 9.4, 12.2, 14.4, 17.3, 19.4, 22.2, 24.6])
    if outlier:
        y = y.copy(); y[6] = 30.0
    n = len(x)
    ls = np.linalg.lstsq(np.vstack([np.ones(n), x]).T, y, rcond=None)[0]
    # L1: min sum t_i, t_i >= +/-(y_i - a - b x_i); variables a, b, t
    c = np.concatenate([[0, 0], np.ones(n)])
    A = np.vstack([np.hstack([-np.ones((n, 1)), -x[:, None], -np.eye(n)]), np.hstack([np.ones((n, 1)), x[:, None], -np.eye(n)])])
    r1 = linprog(c, A_ub=A, b_ub=np.concatenate([-y, y]), bounds=[(None, None)] * 2 + [(0, None)] * n, method="highs")
    # L-infinity: min t, t >= +/-(y_i - a - b x_i)
    A2 = np.vstack([np.hstack([-np.ones((n, 1)), -x[:, None], -np.ones((n, 1))]), np.hstack([np.ones((n, 1)), x[:, None], -np.ones((n, 1))])])
    r2 = linprog([0, 0, 1], A_ub=A2, b_ub=np.concatenate([-y, y]), bounds=[(None, None)] * 2 + [(0, None)], method="highs")
    fit = lambda ab: {"a": float(ab[0]), "b": float(ab[1]), "max_abs": float(np.max(np.abs(y - ab[0] - ab[1] * x))),
                      "sum_abs": float(np.sum(np.abs(y - ab[0] - ab[1] * x))), "sum_sq": float(np.sum((y - ab[0] - ab[1] * x) ** 2))}
    return {"x": x.tolist(), "y": y.tolist(), "ls": fit(ls), "l1": fit(r1.x[:2]), "linf": fit(r2.x[:2])}


# A small pooling problem: two crudes share one tank before the two grades; a third goes direct.
POOL_DATA = {"S": {"A": 3.2, "B": 0.8, "C": 2.3}, "cost": {"A": 5.2, "B": 15.7, "C": 11.6},   # wt% sulfur, USD/t
             "price": {"X": 8.7, "Y": 16.7}, "demand": {"X": 100.0, "Y": 200.0}, "smax": {"X": 2.7, "Y": 1.3}}


def pooling(starts=40, seed=1):
    """Sulfur balance in the shared tank: q (a + b) = S_A a + S_B b, with q a variable: bilinear.
    A local NLP method (SLSQP) from random starts; and the LP if every crude had its own tank."""
    from scipy.optimize import minimize
    d = POOL_DATA; S, cst, P, D, Q = d["S"], d["cost"], d["price"], d["demand"], d["smax"]
    def neg_profit(v):
        a, b, px, py, cx, cy, q = v
        return -(P["X"] * (px + cx) + P["Y"] * (py + cy) - cst["A"] * a - cst["B"] * b - cst["C"] * (cx + cy))
    cons = [{"type": "eq", "fun": lambda v: v[0] + v[1] - v[2] - v[3]},
            {"type": "eq", "fun": lambda v: v[6] * (v[2] + v[3]) - S["A"] * v[0] - S["B"] * v[1]},
            {"type": "ineq", "fun": lambda v: Q["X"] * (v[2] + v[4]) - v[6] * v[2] - S["C"] * v[4]},
            {"type": "ineq", "fun": lambda v: Q["Y"] * (v[3] + v[5]) - v[6] * v[3] - S["C"] * v[5]},
            {"type": "ineq", "fun": lambda v: D["X"] - v[2] - v[4]}, {"type": "ineq", "fun": lambda v: D["Y"] - v[3] - v[5]}]
    bnds = [(0, 300)] * 6 + [(S["B"], S["A"])]
    rng = np.random.default_rng(seed); runs = []
    for k in range(starts):
        x0 = np.concatenate([rng.uniform(0, 200, 6), [rng.uniform(S["B"], S["A"])]])
        r = minimize(neg_profit, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        runs.append({"start": k + 1, "ok": bool(r.success), "profit": float(-r.fun), "x": [round(float(v), 3) for v in r.x]})
    # each crude in its own tank: blend A, B, C straight into X and Y, which is an LP
    m = LP()
    for c_ in "ABC":
        for g in "XY":
            m.var(f"{c_}>{g}", obj=P[g] - cst[c_])
    for g in "XY":
        m.row(f"demand {g}", {f"{c_}>{g}": 1 for c_ in "ABC"}, "<=", D[g])
        m.row(f"S {g}", {f"{c_}>{g}": S[c_] - Q[g] for c_ in "ABC"}, "<=")
    lp = m.solve()
    vals = sorted({round(r["profit"], 1) for r in runs if r["ok"]})
    return {"data": d, "runs": runs, "distinct": [{"profit": v, "count": sum(1 for r in runs if r["ok"] and round(r["profit"], 1) == v)} for v in vals],
            "separate_tanks_lp": lp["obj"]}


# ---------------------------------------------------------------------------
# Part B. Lab 2: a small refinery as a stream table
# ---------------------------------------------------------------------------
CDU_YIELD = {"LPG": 0.02, "LSR": 0.10, "HN": 0.18, "Distillate": 0.38, "Residue": 0.32}   # m3 per m3 crude
REF_YIELD = {"Reformate": 0.85, "LPG": 0.15}                                             # m3 per m3 heavy naphtha
LAB = {"crude_price": 480.0, "cdu_cap": 8000.0, "ref_cap": 1100.0, "ref_opex": 25.0,
       "butane_price": 380.0, "butane_max": 300.0, "alkylate_price": 700.0, "alkylate_max": 600.0,
       "price": {"LPG": 420.0, "Naphtha": 500.0, "Distillate": 620.0, "Residue": 430.0,
                 "Regular": 690.0, "Premium": 750.0},
       "demand": {"Regular": 1500.0, "Premium": 900.0}, "rvp_max": 60.0,
       "ron_min": {"Regular": 91.0, "Premium": 95.0}}
POOL = ["LSR", "Reformate", "Butane", "Alkylate"]
LAB_RON = {"LSR": 70.0, "Reformate": 98.0, "Butane": 94.0, "Alkylate": 96.0}
LAB_RVP = {"LSR": 80.0, "Reformate": 25.0, "Butane": 360.0, "Alkylate": 30.0}


def refinery(rvp_basis="index", ref_cap=None, premium_price=None, rvp_max=None, ron_premium=None,
             details=True):
    """Stream-table LP. Columns are activities; each intermediate stream has one balance row:
       (made by activities) - (used by activities) - (sold) = 0."""
    d = LAB; P = dict(d["price"])
    if premium_price is not None:
        P["Premium"] = premium_price
    ref_cap = d["ref_cap"] if ref_cap is None else ref_cap
    rvp_max = d["rvp_max"] if rvp_max is None else rvp_max
    ron_min = dict(d["ron_min"]); ron_min["Premium"] = ron_min["Premium"] if ron_premium is None else ron_premium
    f = rvp_index if rvp_basis == "index" else (lambda r: np.asarray(r, float))
    m = LP()
    m.var("CDU", obj=-d["crude_price"])                                 # crude bought and distilled
    m.var("REF", obj=-d["ref_opex"])                                    # heavy naphtha reformed
    m.var("buy Butane", obj=-d["butane_price"])
    m.var("buy Alkylate", obj=-d["alkylate_price"])
    for s in ("LPG", "Distillate", "Residue"):
        m.var(f"sell {s}", obj=P[s])
    m.var("sell LSR as naphtha", obj=P["Naphtha"]); m.var("sell HN as naphtha", obj=P["Naphtha"])
    for g in GRADES:
        for i in POOL:
            m.var(f"{i}>{g}", obj=P[g])
    bl = lambda i: {f"{i}>{g}": -1 for g in GRADES}
    m.row("LPG", {"CDU": CDU_YIELD["LPG"], "REF": REF_YIELD["LPG"], "sell LPG": -1}, "==")
    m.row("LSR", {"CDU": CDU_YIELD["LSR"], "sell LSR as naphtha": -1, **bl("LSR")}, "==")
    m.row("HN", {"CDU": CDU_YIELD["HN"], "REF": -1, "sell HN as naphtha": -1}, "==")
    m.row("Reformate", {"REF": REF_YIELD["Reformate"], **bl("Reformate")}, "==")
    m.row("Distillate", {"CDU": CDU_YIELD["Distillate"], "sell Distillate": -1}, "==")
    m.row("Residue", {"CDU": CDU_YIELD["Residue"], "sell Residue": -1}, "==")
    m.row("Butane", {"buy Butane": 1, **bl("Butane")}, "==")
    m.row("Alkylate", {"buy Alkylate": 1, **bl("Alkylate")}, "==")
    m.row("capacity CDU", {"CDU": 1}, "<=", d["cdu_cap"])
    m.row("capacity reformer", {"REF": 1}, "<=", ref_cap)
    m.row("supply Butane", {"buy Butane": 1}, "<=", d["butane_max"])
    m.row("supply Alkylate", {"buy Alkylate": 1}, "<=", d["alkylate_max"])
    for g in GRADES:
        m.row(f"demand {g}", {f"{i}>{g}": 1 for i in POOL}, "<=", d["demand"][g])
        m.row(f"RON {g}", {f"{i}>{g}": LAB_RON[i] - ron_min[g] for i in POOL}, ">=")
        m.row(f"RVP {g}", {f"{i}>{g}": float(f(LAB_RVP[i]) - f(rvp_max)) for i in POOL}, "<=")
    r = m.solve()
    if r["status"] != "optimal":
        return r
    x = r["x"]
    out = {"status": "optimal", "profit": r["obj"], "x": x}
    if not details:
        return out
    grades = {}
    for g in GRADES:
        v = {i: x[f"{i}>{g}"] for i in POOL}; V = sum(v.values())
        w = {i: v[i] / V for i in POOL} if V > 0 else None
        grades[g] = {"recipe": v, "volume": V,
                     "RON": sum(w[i] * LAB_RON[i] for i in POOL) if w else None,
                     "RVP_true": rvp_from_index(sum(w[i] * rvp_index(LAB_RVP[i]) for i in POOL)) if w else None,
                     "RVP_linear": sum(w[i] * LAB_RVP[i] for i in POOL) if w else None}
    inflow = x["CDU"] + x["buy Butane"] + x["buy Alkylate"]
    outflow = (x["sell LPG"] + x["sell Distillate"] + x["sell Residue"] + x["sell LSR as naphtha"]
               + x["sell HN as naphtha"] + grades["Regular"]["volume"] + grades["Premium"]["volume"])
    out.update({"grades": grades, "closure": {"in": inflow, "out": outflow, "difference": inflow - outflow},
                "tight": r["tight_rows"], "n_positive": len(r["positive"]), "positive": r["positive"],
                "n_tight": len(r["tight_rows"]), "n_cols": len(x), "n_rows": len(m.rows),
                "duals": r["duals"], "slack": r["slack"],
                "rows": [{"name": nm, "sense": se, "rhs": rh} for nm, _, se, rh in m.rows]})
    return out


def stream_dof():
    """Degrees of freedom of the refinery stream table: columns minus the rank of the balance rows."""
    names = ["LPG", "LSR", "HN", "Reformate", "Distillate", "Residue", "Butane", "Alkylate"]
    cols = ["CDU", "REF", "buy Butane", "buy Alkylate", "sell LPG", "sell Distillate", "sell Residue",
            "sell LSR as naphtha", "sell HN as naphtha"] + [f"{i}>{g}" for g in GRADES for i in POOL]
    ix = {c: k for k, c in enumerate(cols)}
    A = np.zeros((len(names), len(cols)))
    def put(r, terms):
        for c, a in terms.items():
            A[names.index(r), ix[c]] += a
    bl = lambda i: {f"{i}>{g}": -1 for g in GRADES}
    put("LPG", {"CDU": CDU_YIELD["LPG"], "REF": REF_YIELD["LPG"], "sell LPG": -1})
    put("LSR", {"CDU": CDU_YIELD["LSR"], "sell LSR as naphtha": -1, **bl("LSR")})
    put("HN", {"CDU": CDU_YIELD["HN"], "REF": -1, "sell HN as naphtha": -1})
    put("Reformate", {"REF": REF_YIELD["Reformate"], **bl("Reformate")})
    put("Distillate", {"CDU": CDU_YIELD["Distillate"], "sell Distillate": -1})
    put("Residue", {"CDU": CDU_YIELD["Residue"], "sell Residue": -1})
    put("Butane", {"buy Butane": 1, **bl("Butane")}); put("Alkylate", {"buy Alkylate": 1, **bl("Alkylate")})
    rank = int(np.linalg.matrix_rank(A))
    return {"columns": len(cols), "balance_rows": len(names), "rank": rank, "dof": len(cols) - rank,
            "column_sums": {c: round(float(A[:, ix[c]].sum()), 9) for c in ("CDU", "REF")}}


def lab_results():
    base = refinery()
    lin = refinery(rvp_basis="linear")
    up = refinery(ref_cap=LAB["ref_cap"] + 1.0, details=False)
    # the dual of the reformer row, checked by re-solving with one more m3/d
    shadow_ref = {"dual": base["duals"]["capacity reformer"], "finite_difference": up["profit"] - base["profit"]}
    lo, hi = 690.0, 810.0                       # premium price at which the plan switches (bisection)
    plan = lambda p: round(refinery(premium_price=p, details=False)["x"]["Reformate>Premium"], 6)
    hi_plan = plan(hi)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if plan(mid) == hi_plan else (mid, hi)
    sweeps = {
        "premium_price": [{"value": p, **_brief(refinery(premium_price=p))} for p in np.arange(690.0, 811.0, 5.0)],
        "ref_cap": [{"value": c, **_brief(refinery(ref_cap=c))} for c in np.arange(700.0, 1651.0, 50.0)],
        "rvp_max": [{"value": v, **_brief(refinery(rvp_max=v))} for v in np.arange(50.0, 72.5, 2.5)],
    }
    return {"base": base, "linear_rvp": lin, "shadow_reformer": shadow_ref, "premium_breakpoint": (lo + hi) / 2,
            "linear_rvp_true_check": {g: lin["grades"][g]["RVP_true"] for g in GRADES},
            "sweeps": sweeps,
            "data": {"cdu_yield": CDU_YIELD, "ref_yield": REF_YIELD, "lab": LAB, "ron": LAB_RON, "rvp": LAB_RVP,
                     "rvp_index_spec": float(rvp_index(LAB["rvp_max"]))}}


def _brief(r):
    g = r["grades"]
    return {"profit": r["profit"], "regular": g["Regular"]["volume"], "premium": g["Premium"]["volume"],
            "butane": r["x"]["buy Butane"], "alkylate": r["x"]["buy Alkylate"], "REF": r["x"]["REF"],
            "lsr_naphtha": r["x"]["sell LSR as naphtha"]}


def classroom_results():
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                  "python": sys.version.split()[0]}}
    base = blend()
    R["lecture"] = {
        "data": {"comp": COMP, "ron": RON, "rvp": RVP, "cost": COST, "avail": AVAIL, "grades": GRADES,
                 "ron_min": RON_MIN, "rvp_max": RVP_MAX, "price": PRICE, "demand": DEMAND},
        "base": base,
        "linear_rvp": blend(rvp_basis="linear"),
        "ron_sweep": [dict(spec=float(s), **_lec(blend(ron_min={"Premium": float(s)})))
                      for s in np.arange(93.0, 99.01, 0.25)],
        "rvp_sweep": [dict(spec=float(s), **_lec(blend(rvp_max=float(s)))) for s in np.arange(45.0, 75.01, 2.5)],
        "butane_curve": butane_curve(),
        "tiers": price_tiers(),
        "charnes_cooper": charnes_cooper(),
        "spec_rows": spec_rows(),
        "recompute": recompute_premium(),
        "spec_dual": spec_dual(),
        "fits": {"clean": line_fits(), "outlier": line_fits(outlier=True)},
        "pooling": pooling(),
        "dof": stream_dof(),
    }
    R["lab"] = lab_results()
    return R


def _lec(r):
    if r["status"] != "optimal":
        return {"status": r["status"]}
    return {"status": "optimal", "margin": r["margin"], "premium": r["recipe"]["Premium"],
            "regular": r["recipe"]["Regular"], "props": r["props"], "n_positive": r["n_positive"],
            "n_tight": r["n_tight"], "tight": r["tight"]}


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("blending-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote blending-results.json")
        sys.exit()
    L = R["lecture"]; b = L["base"]
    print("PART A  Two-grade blend")
    print(f"  margin {b['margin']:.1f} USD/d; positive flows {b['n_positive']}, tight rows {b['n_tight']}: {b['tight']}")
    for g in GRADES:
        print(f"  {g}: " + ", ".join(f"{i} {v:.1f}" for i, v in b["recipe"][g].items() if v > 1e-6), "|",
              {k: round(v, 2) for k, v in b["props"][g].items()})
    l = L["linear_rvp"]
    print(f"  linear RVP model: margin {l['margin']:.1f}; true RVP " +
          ", ".join(f"{g} {l['props'][g].get('RVP_true', 0):.2f}" for g in GRADES))
    print("  tiers:", L["tiers"])
    cc = L["charnes_cooper"]; print(f"  Charnes-Cooper: V {cc['volume']:.1f}, cost/m3 {cc['cost_per_m3']:.2f}; scan",
                                    [(round(s['V']), round(s['cost_per_m3'], 2) if s['cost_per_m3'] else None) for s in cc["scan"]])
    for s in L["ron_sweep"]:
        print("   RON", s["spec"], s.get("status"), round(s.get("margin", 0), 1), s.get("n_positive"), s.get("n_tight"))
    print("PART B  Refinery")
    lb = R["lab"]["base"]
    print(f"  profit {lb['profit']:.1f}; closure {lb['closure']}; positive {lb['n_positive']}, tight {lb['n_tight']}, cols {lb['n_cols']}, rows {lb['n_rows']}")
    print("  activities:", {k: round(v, 1) for k, v in lb["x"].items() if v > 1e-6})
    for g in GRADES:
        print(f"  {g}:", {k: round(v, 3) if isinstance(v, float) else v for k, v in lb["grades"][g].items() if k != "recipe"},
              {i: round(v, 1) for i, v in lb["grades"][g]["recipe"].items()})
    print("  tight:", lb["tight"])
    print("  shadow reformer", R["lab"]["shadow_reformer"], "premium breakpoint", R["lab"]["premium_breakpoint"])
    li = R["lab"]["linear_rvp"]
    print(f"  linear RVP: profit {li['profit']:.1f}; true RVP", R["lab"]["linear_rvp_true_check"],
          {g: {i: round(v, 1) for i, v in li['grades'][g]['recipe'].items()} for g in GRADES})
    for k, sw in R["lab"]["sweeps"].items():
        print(" ", k)
        for s in sw:
            print("    ", {a: round(b, 1) for a, b in s.items()})
