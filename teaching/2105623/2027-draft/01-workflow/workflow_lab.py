"""2105623 Session 1: the optimization workflow.

Part A  The lecture example: a two-product plan on shared equipment
        (Additive and Binder), solved and read the way the slides read it.
Part B  Lab 1: a colleague's model with three planted errors. Each error can
        be fixed separately, so every combination of fixes is solved and kept.

Runs in the browser (Pyodide: NumPy, SciPy/HiGHS) and in desktop Python.
    python workflow_lab.py          print every part
    python workflow_lab.py --json   write workflow-results.json for the slides and the lab page
"""
import itertools
import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# ---------------------------------------------------------------------------
# Part A. Lecture example (same data as Week 1 of 2026)
# ---------------------------------------------------------------------------
PRODUCTS = ["Additive", "Binder"]
RESOURCES = ["Reactor train", "Separation and drying", "Finishing and packaging"]
MARGIN = np.array([400.0, 300.0])                 # USD per tonne
USE = np.array([[3.0, 2.0], [1.0, 2.0], [1.0, 1.0]])   # hours per tonne
AVAIL = np.array([24.0, 16.0, 9.0])               # hours per week


def product_mix(avail=AVAIL, margin=MARGIN):
    """Maximize weekly margin. Returns plan, value, slack and shadow prices."""
    res = linprog(-np.asarray(margin), A_ub=USE, b_ub=avail, bounds=[(0, None)] * 2, method="highs")
    if res.status != 0:
        return {"status": {2: "infeasible", 3: "unbounded"}.get(res.status, res.message)}
    x = res.x
    used = USE @ x
    shadow = -res.ineqlin.marginals                  # USD per extra hour
    return {"status": "optimal", "x": (x.round(9) + 0.0).tolist(), "z": float(margin @ x),
            "used": used.tolist(), "slack": (np.asarray(avail) - used).round(9).tolist(),
            "shadow": (shadow.round(9) + 0.0).tolist(),
            "dual_value": float(shadow @ np.asarray(avail))}


def lp_certificate(y=(100.0, 0.0, 100.0), x=(6.0, 3.0)):
    """Check a claimed optimum without trusting any solver: primal feasibility,
    dual feasibility (y >= 0 and A^T y >= c) and equal objectives (no gap)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return {"Ax": (USE @ x).tolist(), "b": AVAIL.tolist(), "primal_feasible": bool(np.all(USE @ x <= AVAIL + 1e-9) and np.all(x >= 0)),
            "ATy": (USE.T @ y).tolist(), "c": MARGIN.tolist(), "dual_feasible": bool(np.all(y >= 0) and np.all(USE.T @ y >= MARGIN - 1e-9)),
            "primal_value": float(MARGIN @ x), "dual_value": float(AVAIL @ y), "gap": float(AVAIL @ y - MARGIN @ x)}


def integer_batches(batch=4.0):
    """Products made only in whole batches of `batch` tonnes: x = batch * n, n integer."""
    c = -MARGIN * batch
    res = milp(c, constraints=LinearConstraint(USE * batch, -np.inf, AVAIL),
               integrality=np.ones(2), bounds=Bounds(0, np.inf))
    n = res.x
    return {"status": "optimal", "x": ((batch * n).round(9) + 0.0).tolist(), "z": float(MARGIN @ (batch * n))}


def broken_models():
    """Two ways to break the lecture model, solved so students see the status."""
    out = {}
    # infeasible: add a supply contract x1 + x2 >= 12
    A = np.vstack([USE, [-1.0, -1.0]]); b = np.concatenate([AVAIL, [-12.0]])
    r = linprog(-MARGIN, A_ub=A, b_ub=b, bounds=[(0, None)] * 2, method="highs")
    out["supply_contract"] = {"status": {0: "optimal", 2: "infeasible", 3: "unbounded"}[r.status]}
    # unbounded: keep only x2 - x1 <= 5
    r = linprog(-MARGIN, A_ub=[[-1.0, 1.0]], b_ub=[5.0], bounds=[(0, None)] * 2, method="highs")
    out["resources_deleted"] = {"status": {0: "optimal", 2: "infeasible", 3: "unbounded"}[r.status]}
    return out


# ---------------------------------------------------------------------------
# Part B. Lab 1: debug a colleague's flowsheet model
# ---------------------------------------------------------------------------
# Isomerization A -> B with recycle and purge. Two fresh feeds: a cheap one with
# 10 mol% inert I and an expensive one with 0.5 mol% inert. Prices are quoted per kg.
X, PHI = 0.60, 0.10                     # per-pass conversion; purged fraction of the recycle loop
Y_A = {"F1": 0.90, "F2": 0.995}         # mole fraction of A in each fresh feed
MW = {"A": 60.0, "B": 60.0, "I": 28.0}  # kg/kmol
PRICE_KG = {"F1": 0.80, "F2": 1.10, "B": 2.00}   # USD/kg
PURGE_COST = 4.0                        # USD per kmol purged
CAP, DEMAND = 100.0, 40.0               # kmol/h at the reactor inlet; kmol/h of B the market takes
VARS = ["F1", "F2", "FA", "FI", "AM", "IM", "Aout", "B", "PA", "PI", "RA", "RI"]
ERRORS = {
    "units": "Feed prices in USD/kg are multiplied by flows in kmol/h",
    "inert": "The inert recycle equation RI = (1 - phi) IM is missing",
    "sign": "The mixer balance for A is written AM = FA - RA",
}


def feed_mw(f, yA=None):
    yA = Y_A[f] if yA is None else yA
    return yA * MW["A"] + (1 - yA) * MW["I"]


def flowsheet(fix_units=False, fix_inert=False, fix_sign=False, phi=PHI, y1=None,
              cap=CAP, demand=DEMAND, price_b=None):
    """The colleague's model with any subset of the three errors fixed (all True = correct)."""
    y = dict(Y_A); y["F1"] = Y_A["F1"] if y1 is None else y1
    pb = PRICE_KG["B"] if price_b is None else price_b
    n, ix = len(VARS), {v: k for k, v in enumerate(VARS)}
    rows, rhs, names = [], [], []

    def eq(name, terms, b=0.0):
        r = np.zeros(n)
        for v, a in terms.items():
            r[ix[v]] += a
        rows.append(r); rhs.append(b); names.append(name)

    eq("feed_A", {"FA": 1, "F1": -y["F1"], "F2": -y["F2"]})
    eq("feed_I", {"FI": 1, "F1": -(1 - y["F1"]), "F2": -(1 - y["F2"])})
    eq("mixer_A", {"AM": 1, "FA": -1, "RA": -1 if fix_sign else 1})
    eq("mixer_I", {"IM": 1, "FI": -1, "RI": -1})
    eq("reactor_A", {"Aout": 1, "AM": -(1 - X)})
    eq("reactor_B", {"B": 1, "AM": -X})
    eq("purge_A", {"PA": 1, "Aout": -phi})
    eq("recycle_A", {"RA": 1, "Aout": -(1 - phi)})
    eq("purge_I", {"PI": 1, "IM": -phi})
    if fix_inert:
        eq("recycle_I", {"RI": 1, "IM": -(1 - phi)})
    cost = {f: PRICE_KG[f] * (feed_mw(f, y[f]) if fix_units else 1.0) for f in ("F1", "F2")}
    c = np.zeros(n)
    c[ix["F1"]], c[ix["F2"]] = cost["F1"], cost["F2"]
    c[ix["B"]] = -pb * MW["B"]
    c[ix["PA"]] = c[ix["PI"]] = PURGE_COST
    A_ub = np.zeros((2, n)); A_ub[0, ix["AM"]] = A_ub[0, ix["IM"]] = 1; A_ub[1, ix["B"]] = 1
    res = linprog(c, A_ub=A_ub, b_ub=[cap, demand], A_eq=np.array(rows), b_eq=rhs,
                  bounds=[(0, None)] * n, method="highs")
    if res.status != 0:
        return {"status": {2: "infeasible", 3: "unbounded"}.get(res.status, res.message)}
    x = dict(zip(VARS, res.x))
    true_cost = {f: PRICE_KG[f] * feed_mw(f, y[f]) for f in ("F1", "F2")}
    true_profit = (pb * MW["B"] * x["B"] - true_cost["F1"] * x["F1"] - true_cost["F2"] * x["F2"]
                   - PURGE_COST * (x["PA"] + x["PI"]))
    out = {"status": "optimal", "flows": {k: round(v, 6) + 0.0 for k, v in x.items()},
           "model_objective": float(-res.fun), "true_profit": float(true_profit),
           "throughput": float(x["AM"] + x["IM"]),
           # plant-wide conservation, computed from the solution only
           "closure": {"A_and_B": float(x["FA"] - x["PA"] - x["B"]), "inert": float(x["FI"] - x["PI"])}}
    if fix_units and fix_inert and fix_sign:
        y_eq, y_ub = res.eqlin.marginals, res.ineqlin.marginals
        reduced = c - np.array(rows).T @ y_eq - A_ub.T @ y_ub
        dual_obj = float(np.array(rhs) @ y_eq + np.array([cap, demand]) @ y_ub)
        out["certificate"] = {
            "shadow_capacity": float(-y_ub[0]), "shadow_demand": float(-y_ub[1]),
            "min_reduced_cost": float(reduced.min()),
            "complementarity": float(np.abs(reduced * res.x).max()),
            "primal_objective": float(c @ res.x), "dual_objective": dual_obj,
            "gap": float(abs(c @ res.x - dual_obj))}
    return out


def lab_all_combinations():
    out = {}
    for fu, fi, fs in itertools.product([False, True], repeat=3):
        key = f"{int(fu)}{int(fi)}{int(fs)}"            # units, inert, sign
        out[key] = flowsheet(fu, fi, fs)
        out[key]["zero_price"] = flowsheet(fu, fi, fs, price_b=0.0)["flows"]
    return out


def lab_sweeps():
    def run(**kw):
        r = flowsheet(True, True, True, **kw)
        return {"F1": r["flows"]["F1"], "F2": r["flows"]["F2"], "B": r["flows"]["B"],
                "profit": r["true_profit"], "throughput": r["throughput"],
                "shadow_capacity": r["certificate"]["shadow_capacity"],
                "shadow_demand": r["certificate"]["shadow_demand"]}
    return {"phi": [dict(value=v, **run(phi=v)) for v in (0.05, 0.075, 0.10, 0.15, 0.20)],
            "inert_cheap": [dict(value=v, **run(y1=1 - v)) for v in (0.05, 0.10, 0.15, 0.20)],
            "capacity": [dict(value=v, **run(cap=v)) for v in (80.0, 90.0, 100.0, 110.0, 120.0)],
            "demand": [dict(value=v, **run(demand=v)) for v in (30.0, 35.0, 40.0, 45.0, 50.0)]}


def classroom_results():
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                  "python": sys.version.split()[0]}}
    base = product_mix()
    R["lecture"] = {
        "base": base,
        "packaging_sweep": [dict(hours=h, **product_mix(avail=np.array([24.0, 16.0, h])))
                            for h in np.arange(5.0, 15.5, 0.5)],
        "integer_batches": integer_batches(),
        "certificate": lp_certificate(),
        "rounded_lp": {"x": [4.0, 0.0], "z": float(MARGIN @ np.array([4.0, 0.0]))},
        "broken": broken_models(),
    }
    R["lab"] = {"combinations": lab_all_combinations(), "sweeps": lab_sweeps(),
                "data": {"X": X, "phi": PHI, "y_A": Y_A, "MW": MW, "price_kg": PRICE_KG,
                         "purge_cost": PURGE_COST, "cap": CAP, "demand": DEMAND,
                         "feed_mw": {f: feed_mw(f) for f in ("F1", "F2")}},
                "errors": ERRORS}
    return R


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("workflow-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote workflow-results.json")
        sys.exit()
    b = R["lecture"]["base"]
    print("PART 1  Lecture example")
    print(f"  plan {b['x']} t/week, margin {b['z']:.2f} USD/week, status {b['status']}")
    print(f"  slack {b['slack']} h, shadow prices {b['shadow']} USD/h")
    print(f"  shadow prices x availability = {b['dual_value']:.2f} (equals the margin)")
    p14 = [r for r in R["lecture"]["packaging_sweep"] if r["hours"] == 14.0][0]
    print(f"  packaging 14 h: plan {p14['x']}, margin {p14['z']:.2f}")
    print(f"  whole 4 t batches: {R['lecture']['integer_batches']}; LP rounded down: {R['lecture']['rounded_lp']}")
    print(f"  broken models: {R['lecture']['broken']}")
    print("\nPART 2  Lab 1, every combination of fixes (units, inert, sign)")
    for k, v in R["lab"]["combinations"].items():
        f = v["flows"]
        print(f"  {k}: F1 {f['F1']:7.2f}  F2 {f['F2']:7.2f}  B {f['B']:6.2f}  model {v['model_objective']:8.1f}"
              f"  true {v['true_profit']:8.1f}  closure A+B {v['closure']['A_and_B']:6.2f}  inert {v['closure']['inert']:6.2f}")
    print("  certificate of the repaired model:", R["lab"]["combinations"]["111"]["certificate"])
