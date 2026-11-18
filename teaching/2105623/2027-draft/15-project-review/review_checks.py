"""2105623 Session 15: checks for any optimization result, and the practice review of Lab 15 (2027 draft).

Part A  Reusable checks: an LP certificate from HiGHS (primal and dual residuals, duality gap), recomputing an
        objective from the specification's data, closing a balance, and converting a multiplier to the units of
        the decision maker
Lab 15  Report D: a biodiesel feedstock blend (LP; synthetic data). The report's numbers are checked
        against its own specification.

Runs in desktop Python and in the browser (NumPy and SciPy only). Copy the functions into your own code and call
them on your own model.
"""
import json, math, sys
import numpy as np
from scipy.optimize import linprog


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


# ---------------------------------------------------------------------------
# Part A. Reusable checks
# ---------------------------------------------------------------------------
def lp_certificate(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """Solve min c'x with HiGHS and return the primal value, the dual value built from the marginals, the gap,
    and the largest primal and dual residuals. A gap near zero with small residuals is a certificate of
    optimality; the solver's status message is not."""
    c = np.asarray(c, float); n = len(c)
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if r.status != 0:
        return {"status": int(r.status), "message": r.message}
    x = r.x
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in (bounds or [(0, None)] * n)], float) if bounds is not None else np.zeros(n)
    hi = np.array([b[1] if b[1] is not None else np.inf for b in (bounds or [(0, None)] * n)], float) if bounds is not None else np.full(n, np.inf)
    if lo.shape == ():
        lo, hi = np.full(n, lo), np.full(n, hi)
    dual = 0.0; red = c.copy(); prim = 0.0
    if A_ub is not None:
        A, b, y = np.asarray(A_ub, float), np.asarray(b_ub, float), r.ineqlin.marginals
        dual += b @ y; red -= A.T @ y; prim = max(prim, float(np.max(A @ x - b, initial=0.0)))
    if A_eq is not None:
        A, b, y = np.asarray(A_eq, float), np.asarray(b_eq, float), r.eqlin.marginals
        dual += b @ y; red -= A.T @ y; prim = max(prim, float(np.max(np.abs(A @ x - b), initial=0.0)))
    ml, mu = r.lower.marginals, r.upper.marginals
    dual += sum(l * m for l, m in zip(lo, ml) if np.isfinite(l)) + sum(u * m for u, m in zip(hi, mu) if np.isfinite(u))
    red -= ml + mu
    return {"status": 0, "x": x.tolist(), "primal": float(r.fun), "dual": float(dual), "gap": float(r.fun - dual),
            "primal_residual": prim, "dual_residual": float(np.max(np.abs(red))),
            "ineq_marginals": [] if A_ub is None else r.ineqlin.marginals.tolist(),
            "eq_marginals": [] if A_eq is None else r.eqlin.marginals.tolist(), "upper_marginals": mu.tolist()}


def recompute(x, prices):
    """The objective of a reported plan, recomputed from the specification's price table."""
    return float(np.dot(x, prices))


def closure(inputs, outputs):
    """Balance closure: total in minus total out (zero when the balance closes)."""
    return float(np.sum(inputs) - np.sum(outputs))


def per_unit_of_limit(marginal, scale):
    """A multiplier of a row written as sum((p_i - limit) x_i) <= 0 is per unit of (property x mass).
    Multiplying by the total mass gives the value per unit of the limit itself."""
    return float(marginal * scale)


# a small LP for the lecture: max 30 x1 + 20 x2 with two shared resources
LECTURE_LP = {"c": [-30.0, -20.0], "A_ub": [[1.0, 1.0], [2.0, 1.0]], "b_ub": [40.0, 60.0]}


def lecture_certificate():
    return lp_certificate(LECTURE_LP["c"], A_ub=LECTURE_LP["A_ub"], b_ub=LECTURE_LP["b_ub"])


# ---------------------------------------------------------------------------
# Lab 15. Report D: a biodiesel feedstock blend (synthetic data)
# ---------------------------------------------------------------------------
SPEC_D = {"oils": ["palm oil", "used cooking oil", "soybean oil"],
      "price": [900.0, 650.0, 1050.0],          # USD/t, specification (supplier quote)
      "cloud": [13.0, 9.0, -1.0],               # cloud point, deg C
      "acid": [0.3, 1.6, 0.2],                  # acid value after pretreatment, mg KOH/g
      "iodine": [52.0, 95.0, 130.0],            # iodine value, g I2/100 g
      "limits": {"cloud": 10.0, "acid": 0.9, "iodine": 110.0},
      "blend": 100.0,                           # t/d of blend
      "uco_max": 40.0}                          # t/d of used cooking oil available

REPORT = {"x": [50.0, 40.0, 12.0],              # the stream table in the report (t/d)
          "cost": 80300.0,                      # USD/d, as reported
          "price_used": [900.0, 620.0, 1050.0], # what the code used (an old quote for used cooking oil)
          "cloud_shadow_claim": 10.7}           # "relaxing the cloud point by 1 deg C saves 10.7 USD/d"


def report_d_model(price=None):
    """The blend LP: minimize cost; blend limits written as sum((p_i - limit) x_i) <= 0."""
    G = SPEC_D; p = np.array(price or G["price"], float)
    A = [np.array(G[k]) - G["limits"][k] for k in ("cloud", "acid", "iodine")]
    return lp_certificate(p, A_ub=A, b_ub=[0.0, 0.0, 0.0], A_eq=[[1.0, 1.0, 1.0]], b_eq=[G["blend"]],
                          bounds=[(0, None), (0, G["uco_max"]), (0, None)])


def report_d_review():
    sol = report_d_model(); rep = REPORT
    cert = sol
    out = {"solution": sol, "report_run": report_d_model(rep["price_used"]),
           "recomputed_cost": recompute(sol["x"], SPEC_D["price"]),
           "report_cost_recomputed": recompute([50.0, 40.0, 10.0], SPEC_D["price"]),
           "report_balance": closure(rep["x"], [SPEC_D["blend"]]),
           "cloud_value_per_degC": -per_unit_of_limit(sol["ineq_marginals"][0], SPEC_D["blend"]),
           "uco_value_per_t": -sol["upper_marginals"][1],
           "properties": {k: float(np.dot(SPEC_D[k], sol["x"]) / SPEC_D["blend"]) for k in ("cloud", "acid", "iodine")},
           "gap": cert["gap"]}
    # sensitivity to the price of used cooking oil, plus or minus 20%
    sens = []
    for f in (0.8, 0.9, 1.0, 1.1, 1.2):
        p = list(SPEC_D["price"]); p[1] *= f; s = report_d_model(p)
        sens.append({"factor": f, "cost": s["primal"], "x": s["x"]})
    out["sensitivity"] = sens
    return out


# the twelve checklist items and the key for Report D (P pass, F fail, NS not shown, NA not applicable)
CHECKLIST = ["Specification: sets, data and constraints, with units",
             "Data: sources and dates",
             "Model class stated and justified",
             "Balances close in the stream table",
             "Objective recomputed from the plan and the specification's data",
             "A certificate, not only the solver status",
             "Multipliers interpreted with units and sign",
             "Global or local stated correctly",
             "Uncertain data: sensitivity or scenarios",
             "Several objectives: a front and a stated rule",
             "Tests written from the specification",
             "Reporting: code, versions and AI disclosure"]
KEY_D = ["P", "P", "P", "F", "F", "F", "F", "P", "NS", "NA", "F", "F"]


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
    return o


def classroom_results():
    import scipy
    R = {"meta": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]},
         "lecture": {"cert": lecture_certificate(), "checklist": CHECKLIST},
         "lab": {"spec": SPEC_D, "report": REPORT, "review": report_d_review(), "checklist": CHECKLIST, "key": KEY_D}}
    return clean(R)


if __name__ == "__main__":
    if "--json" in sys.argv:
        R = classroom_results()
        with open("checks-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote checks-results.json", R["meta"]); sys.exit()
    L = lecture_certificate(); print("lecture LP", r6(L["x"]), "primal", r6(L["primal"]), "dual", r6(L["dual"]), "gap", r6(L["gap"]), "marginals", r6(L["ineq_marginals"]))
    G = report_d_review(); s = G["solution"]
    print("Report D optimum", r6(s["x"]), r6(s["primal"]), "gap", r6(s["gap"]), "residuals", s["primal_residual"], s["dual_residual"])
    print("  marginals", r6(s["ineq_marginals"]), r6(s["eq_marginals"]), r6(s["upper_marginals"]))
    print("  report: balance", G["report_balance"], "cost of its plan at spec prices", G["report_cost_recomputed"], "vs reported", REPORT["cost"])
    print("  value of 1 degC of cloud point", r6(G["cloud_value_per_degC"]), "USD/d; UCO", r6(G["uco_value_per_t"]), "USD/t", "properties", {k: r6(v) for k, v in G["properties"].items()})
    print("  sensitivity", [(q["factor"], r6(q["cost"]), r6(q["x"])) for q in G["sensitivity"]])
