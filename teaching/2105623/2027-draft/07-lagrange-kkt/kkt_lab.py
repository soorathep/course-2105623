"""Session 7 (2027 draft): Lagrange multipliers and the KKT conditions.

Course convention, used in every function below:
    minimize f(x)  subject to  g_i(x) <= 0  (multiplier mu_i >= 0)
                               h_j(x)  = 0  (multiplier lambda_j, any sign)
    L(x, mu, lambda) = f(x) + sum_i mu_i g_i(x) + sum_j lambda_j h_j(x)
A maximization is written as "minimize -f". Writing a constraint as g(x) - b <= 0
gives dv*/db = -mu: relaxing it by one unit lowers the minimum by mu.

Run `python kkt_lab.py` for a printed summary, or `python kkt_lab.py --json`
to write kkt-results.json, which the slides and the lab page read.
"""
import itertools
import json
import math
import sys
import warnings

import numpy as np
from scipy.optimize import brentq, minimize


def r6(v):
    """Round for printing, so recorded and live runs print the same text."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


# ---------------------------------------------------------------------------
# Part A. Lagrange: one equality constraint
# ---------------------------------------------------------------------------
ELLIPSE = {"a": 2.0, "b": 1.0, "target": [0.6, 0.35]}   # h(x) = x1^2/a^2 + x2^2/b^2 - 1 = 0


def ellipse_point(t, E=ELLIPSE):
    """Point, f, gradients, lambda and curvature at angle t on the ellipse."""
    a, b, (p, q) = E["a"], E["b"], E["target"]
    x = np.array([a * math.cos(t), b * math.sin(t)])
    gf = 2 * (x - np.array([p, q]))                    # f = (x1 - p)^2 + (x2 - q)^2
    gh = np.array([2 * x[0] / a**2, 2 * x[1] / b**2])
    lam = -gf @ gh / (gh @ gh)                         # best fit of grad f + lambda grad h = 0
    tau = np.array([-a * math.sin(t), b * math.cos(t)]); tau /= np.linalg.norm(tau)
    HL = 2 * np.eye(2) + lam * np.diag([2 / a**2, 2 / b**2])   # Hessian of the Lagrangian
    return {"t": t, "x": x, "f": float((x - [p, q]) @ (x - [p, q])), "grad_f": gf, "grad_h": gh, "lam": lam,
            "residual": float(np.linalg.norm(gf + lam * gh)), "slope": float(gf @ tau), "curv": float(tau @ HL @ tau)}


def ellipse_stationary(E=ELLIPSE, n=3600):
    """Every point where grad f is parallel to grad h, classified by the second-order test."""
    ts = np.linspace(0, 2 * math.pi, n + 1)
    s = [ellipse_point(t, E)["slope"] for t in ts]
    out = []
    for k in range(n):
        if s[k] == 0 or s[k] * s[k + 1] < 0:
            t = brentq(lambda u: ellipse_point(u, E)["slope"], ts[k], ts[k + 1], xtol=1e-14)
            P = ellipse_point(t, E)
            out.append({"t": t, "x": P["x"].tolist(), "f": P["f"], "lam": P["lam"], "curv": P["curv"],
                        "kind": "local min" if P["curv"] > 0 else "local max"})
    fmin = min(q["f"] for q in out); fmax = max(q["f"] for q in out)
    for q in out:
        q["kind"] = ("global min" if abs(q["f"] - fmin) < 1e-12 else "global max" if abs(q["f"] - fmax) < 1e-12 else q["kind"])
    return out


def tank(V=10.0):
    """Closed cylinder of volume V with the least surface: min 2 pi r^2 + 2 pi r h  s.t.  pi r^2 h - V = 0."""
    r = (V / (2 * math.pi)) ** (1 / 3); h = 2 * r
    lam = -2 / r                                         # from dL/dh = 2 pi r + lambda pi r^2 = 0
    A = 2 * math.pi * r**2 + 2 * math.pi * r * h
    Ah = lambda v: 6 * math.pi * (v / (2 * math.pi)) ** (2 / 3)
    return {"V": V, "r": r, "h": h, "D": 2 * r, "lam": lam, "A": A, "dA_dV": -lam, "fd": (Ah(V + 1e-4) - Ah(V - 1e-4)) / 2e-4}


def line_sensitivity(b):
    """min (x1-3)^2 + (x2-2)^2  s.t.  x1 + x2 - b = 0: closed form, so dv*/db = -lambda can be checked."""
    lam = 5 - b
    x = [3 - lam / 2, 2 - lam / 2]
    return {"b": b, "x": x, "lam": lam, "v": lam**2 / 2, "dv_db": -lam}


# ---------------------------------------------------------------------------
# Part A2. Equal marginal value: three electrolyzer stacks share the power
# ---------------------------------------------------------------------------
STACKS = {"names": ["Stack A", "Stack B", "Stack C"],
          "a": [20.0, 19.5, 19.0],      # kg H2 per MWh at low load
          "b": [0.40, 0.25, 0.15],      # loss of marginal yield, kg H2 per MW^2 h
          "Pmax": [5.0, 6.0, 8.0],      # MW
          "units": "P in MW, H2 in kg/h, marginal yield in kg/MWh"}


def h2(P, S=STACKS):
    return [a * p - b * p * p for a, b, p in zip(S["a"], S["b"], P)]


def allocate_power(Ptot, S=STACKS):
    """Maximize total H2 = sum(a P - b P^2) s.t. sum P = Ptot, 0 <= P <= Pmax.
    KKT (written as min -H2): a - 2 b P = lambda for a stack between its bounds,
    mu_up = (a - 2 b Pmax) - lambda at full load, mu_lo = lambda - a when off."""
    a, b, Pm = map(np.array, (S["a"], S["b"], S["Pmax"]))
    Ptot = min(max(Ptot, 0.0), float(Pm.sum()))
    P_of = lambda lam: np.clip((a - lam) / (2 * b), 0, Pm)
    lo, hi = float((a - 2 * b * Pm).min()) - 1, float(a.max()) + 1
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if P_of(m).sum() > Ptot else (lo, m)
    lam = (lo + hi) / 2; P = P_of(lam); marg = a - 2 * b * P
    mu_up = np.where(P >= Pm - 1e-9, marg - lam, 0.0); mu_lo = np.where(P <= 1e-9, lam - marg, 0.0)
    return {"Ptot": Ptot, "P": P.tolist(), "H2": h2(P, S), "total": float(sum(h2(P, S))), "lam": lam,
            "marginal": marg.tolist(), "mu_up": np.maximum(mu_up, 0).tolist(), "mu_lo": np.maximum(mu_lo, 0).tolist()}


# ---------------------------------------------------------------------------
# Part B. KKT with inequalities
# ---------------------------------------------------------------------------
# Explorer: f = (x1 - T1)^2 + (x2 - T2)^2 over the region
#   g1 = x1^2 + x2^2 - 9 <= 0,   g2 = 0.5 - x2 <= 0,   g3 = x1 - 2 <= 0
EXPLORER = {"R": 3.0, "ymin": 0.5, "xmax": 2.0}


def g_explorer(x, E=EXPLORER):
    return np.array([x[0]**2 + x[1]**2 - E["R"]**2, E["ymin"] - x[1], x[0] - E["xmax"]])


def grad_g_explorer(x):
    return np.array([[2 * x[0], 2 * x[1]], [0.0, -1.0], [1.0, 0.0]])


def kkt_explorer(T, E=EXPLORER, tol=1e-9):
    """Enumerate active sets; return the point that passes all four KKT conditions."""
    T = np.array(T, float); R, y0, x0 = E["R"], E["ymin"], E["xmax"]
    cands = {(): [T]}
    cands[(0,)] = [R * T / np.linalg.norm(T)] if np.linalg.norm(T) > 0 else []
    cands[(1,)] = [np.array([T[0], y0])]
    cands[(2,)] = [np.array([x0, T[1]])]
    s1 = math.sqrt(R**2 - y0**2); s2 = math.sqrt(R**2 - x0**2)
    cands[(0, 1)] = [np.array([s1, y0]), np.array([-s1, y0])]
    cands[(0, 2)] = [np.array([x0, s2]), np.array([x0, -s2])]
    cands[(1, 2)] = [np.array([x0, y0])]
    for act, pts in cands.items():
        for x in pts:
            gf = 2 * (x - T); G = grad_g_explorer(x)[list(act)]
            mu = np.zeros(3)
            if act:
                sol = np.linalg.lstsq(G.T, -gf, rcond=None)[0]; mu[list(act)] = sol
            res = kkt_residuals(gf, grad_g_explorer(x), g_explorer(x, E), mu)
            if max(res.values()) < 1e-7:
                return {"T": T.tolist(), "x": x.tolist(), "active": list(act), "mu": mu.tolist(), "g": g_explorer(x, E).tolist(),
                        "grad_f": gf.tolist(), "residuals": res}
    raise RuntimeError("no KKT point found")


def kkt_residuals(grad_f, G, g, mu, H=None, h=None, lam=None):
    """The four KKT residuals used in this course (all zero at a KKT point)."""
    st = np.array(grad_f, float) + np.array(G, float).T @ np.array(mu, float)
    if H is not None:
        st = st + np.array(H, float).T @ np.array(lam, float)
    primal = max([0.0] + [float(v) for v in g] + ([abs(float(v)) for v in h] if h is not None else []))
    return {"stationarity": float(np.max(np.abs(st))), "primal": primal,
            "dual": max(0.0, float(-np.min(mu))) if len(mu) else 0.0,
            "complementarity": float(np.max(np.abs(np.array(mu) * np.array(g)))) if len(mu) else 0.0}


# Guess the active set: min (x1-3)^2 + (x2-2)^2  s.t.  g1 = x1 + x2 - 3 <= 0, g2 = x1 - 1.5 <= 0, g3 = -x2 <= 0
GUESS = {"T": [3.0, 2.0], "A": [[1.0, 1.0], [1.0, 0.0], [0.0, -1.0]], "b": [3.0, 1.5, 0.0],
         "names": ["g1: x1 + x2 <= 3", "g2: x1 <= 1.5", "g3: x2 >= 0"]}


def test_guess(active, P=GUESS):
    """Treat the guessed constraints as equalities, solve the KKT system, then test the rest."""
    A, b, T = np.array(P["A"]), np.array(P["b"]), np.array(P["T"])
    act = list(active); m = len(act)
    if m > 2 or (m == 2 and abs(np.linalg.det(A[act])) < 1e-12):
        return {"active": act, "ok": False, "reason": "inconsistent: more independent equalities than variables"}
    K = np.zeros((2 + m, 2 + m)); K[:2, :2] = 2 * np.eye(2); K[:2, 2:] = A[act].T; K[2:, :2] = A[act]
    rhs = np.concatenate([2 * T, b[act]])
    z = np.linalg.solve(K, rhs); x = z[:2]; mu = np.zeros(3); mu[act] = z[2:]
    g = A @ x - b
    viol = [i for i in range(3) if g[i] > 1e-9]; neg = [i for i in act if mu[i] < -1e-9]
    reason = ("passes all four conditions: the optimum" if not viol and not neg else
              ("primal: violates " + ", ".join(f"g{i+1}" for i in viol)) if viol else
              ("dual: mu" + ", mu".join(str(i + 1) for i in neg) + " < 0, so releasing that constraint lowers f"))
    return {"active": act, "x": x.tolist(), "mu": mu.tolist(), "g": g.tolist(), "f": float((x - T) @ (x - T)),
            "ok": not viol and not neg, "reason": reason}


def all_guesses(P=GUESS):
    return [test_guess(list(c), P) for k in range(4) for c in itertools.combinations(range(3), k)]


def simple_case():
    """By hand: min x1^2 + x2^2  s.t.  g = 2 - x1 - x2 <= 0. Two cases."""
    return {"inactive": {"x": [0.0, 0.0], "g": 2.0, "verdict": "infeasible"},
            "active": {"x": [1.0, 1.0], "mu": 2.0, "f": 2.0, "verdict": "KKT point; convex problem, so the global minimum"}}


def lp_as_kkt():
    """Session 1 plant as min -c'x s.t. Ax - b <= 0 (mu) and -x <= 0 (nu): the prices are KKT multipliers."""
    A = np.array([[3.0, 2.0], [1.0, 2.0], [1.0, 1.0]]); b = np.array([24.0, 16.0, 9.0]); c = np.array([400.0, 300.0])
    rows = []
    for x, mu in (([6.0, 3.0], [100.0, 0.0, 100.0]), ([8.0, 0.0], [400 / 3, 0.0, 0.0])):
        x = np.array(x); mu = np.array(mu); nu = A.T @ mu - c      # stationarity: -c + A'mu - nu = 0
        G = np.vstack([A, -np.eye(2)]); g = np.concatenate([A @ x - b, -x]); m = np.concatenate([mu, nu])
        res = kkt_residuals(-c, G, g, m)
        rows.append({"x": x.tolist(), "mu": mu.tolist(), "nu": nu.tolist(), "residuals": res})
    return rows


# ---------------------------------------------------------------------------
# Part C. When KKT works: qualifications, second order, the Lagrangian dual
# ---------------------------------------------------------------------------
def cusp():
    """min -x1  s.t.  g1 = x2 - (1 - x1)^3 <= 0,  g2 = -x2 <= 0. Optimum (1, 0); LICQ fails there."""
    grad_f = [-1.0, 0.0]; G = [[0.0, 1.0], [0.0, -1.0]]          # at (1, 0)
    return {"x": [1.0, 0.0], "grad_f": grad_f, "grad_g": G, "rank": int(np.linalg.matrix_rank(np.array(G))),
            "note": "no mu solves -1 + 0 mu1 + 0 mu2 = 0: no multipliers exist, though (1, 0) is the minimum"}


def dual_convex(mu):
    """min x^2  s.t.  1 - x <= 0.  q(mu) = min_x x^2 + mu (1 - x) = mu - mu^2/4 at x = mu/2."""
    return {"mu": mu, "x": mu / 2, "q": mu - mu**2 / 4}


def dual_nonconvex(mu):
    """min -x^2 over X = [-1, 2]  s.t.  x - 1 <= 0. f* = -1; q(mu) = min(-1 - 2 mu, mu - 4)."""
    a, b = -1 - 2 * mu, mu - 4
    return {"mu": mu, "x": -1.0 if a <= b else 2.0, "q": min(a, b)}


def solver_check(T=(3.0, 2.0)):
    """SLSQP on the guess problem; multipliers from the KKT stationarity at its answer."""
    A, b = np.array(GUESS["A"]), np.array(GUESS["b"])
    cons = [{"type": "ineq", "fun": (lambda x, i=i: b[i] - A[i] @ x), "jac": (lambda x, i=i: -A[i])} for i in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimize(lambda x: (x[0] - T[0])**2 + (x[1] - T[1])**2, [0.0, 0.0], jac=lambda x: 2 * (x - np.array(T)),
                     method="SLSQP", constraints=cons, options={"ftol": 1e-12})
    g = A @ r.x - b; act = [i for i in range(3) if g[i] > -1e-7]
    mu = np.zeros(3); mu[act] = np.linalg.lstsq(A[act].T, -2 * (r.x - np.array(T)), rcond=None)[0]
    return {"x": r.x.tolist(), "active": act, "mu": mu.tolist(), "residuals": kkt_residuals(2 * (r.x - np.array(T)), A, g, mu)}


# ---------------------------------------------------------------------------
# Lab 7. Feed allocation between three parallel CSTRs (synthetic)
# ---------------------------------------------------------------------------
LAB = {"names": ["R1", "R2", "R3"], "V": [6.0, 5.0, 4.0], "k": [1.5, 1.0, 0.2],   # m^3, 1/h (first order A -> B)
       "Fmax": [5.0, 11.0, 6.0], "Fmin": [1.0, 1.0, 2.0],                          # pump limits and minimum circulation, m^3/h
       "CA0": 2.0, "Ftot": 15.0,                                                   # kmol A/m^3, m^3/h of feed to share
       "dH": 80000.0, "Qmax1": 125.0,                                              # kJ released per kmol B; cooling limit of R1, kW
       "units": "F in m^3/h, production in kmol B/h, cooling in kW"}


def production(F, D=LAB):
    """CSTR, first order: B made = CA0 F k tau / (1 + k tau) = CA0 F kV / (F + kV), kmol/h."""
    return [D["CA0"] * f * k * v / (f + k * v) for f, k, v in zip(F, D["k"], D["V"])]


def marginal(F, D=LAB):
    return [D["CA0"] * (k * v)**2 / (f + k * v)**2 for f, k, v in zip(F, D["k"], D["V"])]


def cooling(F, D=LAB):
    """Heat released in R1, kW."""
    return production(F, D)[0] * D["dH"] / 3600


def allocate_feed(D=LAB):
    """Exact optimum by the equal-marginal rule, with the KKT multipliers of every row.
    Problem (as min): -sum P_i(F_i)  s.t.  h = sum F_i - Ftot = 0 (lambda),
    g_cool = Q1(F1) - Qmax1 <= 0, g_pump,i = F_i - Fmax_i <= 0, g_min,i = Fmin_i - F_i <= 0."""
    kV = np.array(D["k"]) * np.array(D["V"]); CA0 = D["CA0"]
    Pcap = D["Qmax1"] * 3600 / D["dH"]                                   # kmol/h that R1 can cool
    Fcool = Pcap * kV[0] / (CA0 * kV[0] - Pcap) if CA0 * kV[0] > Pcap else math.inf
    hi = np.array(D["Fmax"], float); hi[0] = min(hi[0], Fcool); lo = np.array(D["Fmin"], float)
    if not lo.sum() - 1e-12 <= D["Ftot"] <= hi.sum() + 1e-12:
        return {"feasible": False}
    F_of = lambda lam: np.clip(kV * (np.sqrt(CA0 / lam) - 1), lo, hi)
    a, b = 1e-9, CA0 * 4
    for _ in range(300):
        m = (a + b) / 2
        a, b = (m, b) if F_of(m).sum() > D["Ftot"] else (a, m)
    lam = (a + b) / 2; F = F_of(lam); dP = np.array(marginal(F, D))
    free = [i for i in range(3) if lo[i] + 1e-9 < F[i] < hi[i] - 1e-9]
    if free:
        lam = float(dP[free[0]])
    mu_cool = 0.0; mu_pump = np.zeros(3); mu_min = np.zeros(3)
    kappa = D["dH"] / 3600                                               # kW per kmol/h
    for i in range(3):
        if F[i] >= hi[i] - 1e-9 and dP[i] > lam:
            if i == 0 and Fcool < D["Fmax"][0] - 1e-9:
                mu_cool = (dP[0] - lam) / (kappa * dP[0])
            else:
                mu_pump[i] = dP[i] - lam
        if F[i] <= lo[i] + 1e-9 and dP[i] < lam:
            mu_min[i] = lam - dP[i]
    P = production(F, D)
    # residuals of the four KKT conditions, rows in the order cool, pump 1-3, min 1-3
    grad_f = -dP
    G = np.vstack([[kappa * dP[0], 0, 0], np.eye(3), -np.eye(3)])
    g = np.concatenate([[cooling(F, D) - D["Qmax1"]], F - np.array(D["Fmax"]), np.array(D["Fmin"]) - F])
    mu = np.concatenate([[mu_cool], mu_pump, mu_min])
    res = kkt_residuals(grad_f, G, g, mu, H=[[1.0, 1.0, 1.0]], h=[F.sum() - D["Ftot"]], lam=[lam])
    act = [n for n, gi in zip(["cooling R1", "pump R1", "pump R2", "pump R3", "minimum R1", "minimum R2", "minimum R3"], g) if gi > -1e-7]
    return {"feasible": True, "F": F.tolist(), "P": P, "total": float(sum(P)), "conversion": [p / (CA0 * f) for p, f in zip(P, F)],
            "marginal": dP.tolist(), "lam": lam, "mu_cool": mu_cool, "mu_pump": mu_pump.tolist(), "mu_min": mu_min.tolist(),
            "cooling_kW": cooling(F, D), "Fcool": Fcool, "g": g.tolist(), "active": act, "residuals": res}


def slsqp_feed(D=LAB):
    """The same problem with a general NLP solver, as a cross-check."""
    cons = [{"type": "eq", "fun": lambda F: F.sum() - D["Ftot"]},
            {"type": "ineq", "fun": lambda F: D["Qmax1"] - cooling(F, D)}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = minimize(lambda F: -sum(production(F, D)), [D["Ftot"] / 3] * 3, method="SLSQP", constraints=cons,
                     bounds=list(zip(D["Fmin"], D["Fmax"])), options={"ftol": 1e-12, "maxiter": 500})
    F = r.x; dP = np.array(marginal(F, D)); kappa = D["dH"] / 3600
    G = np.vstack([[kappa * dP[0], 0, 0], np.eye(3), -np.eye(3)])
    g = np.concatenate([[cooling(F, D) - D["Qmax1"]], F - np.array(D["Fmax"]), np.array(D["Fmin"]) - F])
    act = [i for i in range(7) if g[i] > -1e-6]
    M = np.vstack([np.ones(3), G[act]]).T                      # columns: equality row, then the active rows
    z = np.linalg.lstsq(M, dP, rcond=None)[0]                  # stationarity: -dP + lambda*1 + G_act' mu_act = 0
    mu = np.zeros(7); mu[act] = z[1:]
    res = kkt_residuals(-dP, G, g, mu, H=[[1.0, 1.0, 1.0]], h=[F.sum() - D["Ftot"]], lam=[z[0]])
    return {"F": F.tolist(), "total": float(-r.fun), "nit": int(r.nit), "success": bool(r.success),
            "lam": float(z[0]), "mu": mu.tolist(), "residuals": res}


def with_change(**kw):
    D = dict(LAB); D.update(kw); return allocate_feed(D)


def lab_results():
    base = allocate_feed()
    sweep = []
    for Ft in np.round(np.arange(6.0, 20.01, 0.05), 2):
        q = with_change(Ftot=float(Ft))
        if q["feasible"]:
            sweep.append({"Ftot": float(Ft), "F": q["F"], "total": q["total"], "lam": q["lam"], "active": q["active"],
                          "mu_cool": q["mu_cool"], "mu_min3": q["mu_min"][2], "mu_pump2": q["mu_pump"][1]})
    changes = [i for i in range(1, len(sweep)) if sweep[i]["active"] != sweep[i - 1]["active"]]
    # exact feeds where the active set changes: R1 reaches its cooling limit; R2 reaches its pump limit
    kV = np.array(LAB["k"]) * np.array(LAB["V"]); Fc = base["Fcool"]
    lam_c = LAB["CA0"] * kV[0]**2 / (Fc + kV[0])**2
    F2_c = kV[1] * (math.sqrt(LAB["CA0"] / lam_c) - 1)
    bounds = {"cooling_binds_above": Fc + F2_c + LAB["Fmin"][2], "pump2_binds_above": Fc + LAB["Fmax"][1] + LAB["Fmin"][2]}
    return {"data": LAB, "base": base, "slsqp": slsqp_feed(),
            "feed_plus1": with_change(Ftot=LAB["Ftot"] + 1), "cool_plus10": with_change(Qmax1=LAB["Qmax1"] + 10),
            "min3_minus1": with_change(Fmin=[1.0, 1.0, 1.0]),
            "sweep": sweep, "boundaries": bounds, "set_changes": [{"between": [sweep[i - 1]["Ftot"], sweep[i]["Ftot"]],
                                             "before": sweep[i - 1]["active"], "after": sweep[i]["active"]} for i in changes]}


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
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "python": sys.version.split()[0]}}
    st = ellipse_stationary()
    R["lecture"] = {
        "ellipse": {"E": ELLIPSE, "stationary": st},
        "tank": tank(10.0),
        "line": [line_sensitivity(b) for b in (3.0, 4.0, 5.0, 6.0)],
        "stacks": {"data": STACKS, "at12": allocate_power(12.0),
                   "sweep": [allocate_power(float(p)) for p in np.round(np.arange(0, 19.001, 0.1), 1)]},
        "explorer": {"E": EXPLORER, "cases": [kkt_explorer(T) for T in ([0.0, 2.0], [0.0, 5.0], [1.0, -1.0], [3.5, 1.5], [3.5, -1.0], [3.5, 4.0], [-4.0, -1.0])]},
        "guess": {"P": GUESS, "all": all_guesses()},
        "simple": simple_case(), "lp": lp_as_kkt(), "cusp": cusp(),
        "dual_convex": [dual_convex(float(m)) for m in np.round(np.arange(0, 4.001, 0.05), 2)],
        "dual_nonconvex": [dual_nonconvex(float(m)) for m in np.round(np.arange(0, 3.001, 0.05), 2)],
        "solver": solver_check()}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("kkt-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote kkt-results.json"); sys.exit()
    L = R["lecture"]
    for q in L["ellipse"]["stationary"]:
        print("A ellipse", r6(q["x"]), "f", r6(q["f"]), "lambda", r6(q["lam"]), "curvature", r6(q["curv"]), q["kind"])
    print("A tank", {k: r6(v) for k, v in L["tank"].items()})
    print("A line", [(q["b"], r6(q["lam"]), r6(q["v"])) for q in L["line"]])
    s = L["stacks"]["at12"]; print("A stacks 12 MW", r6(s["P"]), "H2", r6(s["total"]), "lambda", r6(s["lam"]), "mu_up", r6(s["mu_up"]), "mu_lo", r6(s["mu_lo"]))
    for q in L["explorer"]["cases"]:
        print("B explorer T", q["T"], "x", r6(q["x"]), "active", q["active"], "mu", r6(q["mu"]))
    for q in L["guess"]["all"]:
        print("B guess", q["active"], r6(q.get("x", [])), r6(q.get("mu", [])), q["reason"])
    for q in L["lp"]:
        print("B LP x", q["x"], "mu", r6(q["mu"]), "nu", r6(q["nu"]), {k: r6(v) for k, v in q["residuals"].items()})
    print("C cusp", L["cusp"])
    print("C solver", r6(L["solver"]["x"]), L["solver"]["active"], r6(L["solver"]["mu"]))
    lab = R["lab"]; b = lab["base"]
    print("E base F", r6(b["F"]), "total", r6(b["total"]), "lambda", r6(b["lam"]), "mu_cool", r6(b["mu_cool"]), "mu_pump", r6(b["mu_pump"]),
          "mu_min", r6(b["mu_min"]), "active", b["active"], {k: r6(v) for k, v in b["residuals"].items()})
    print("E slsqp", r6(lab["slsqp"]["F"]), r6(lab["slsqp"]["total"]))
    print("E +1 feed", r6(lab["feed_plus1"]["total"] - b["total"]), "+10 kW", r6(lab["cool_plus10"]["total"] - b["total"]),
          "Fmin3 2 -> 1", r6(lab["min3_minus1"]["total"] - b["total"]))
    print("E set changes", lab["set_changes"], {k: r6(v) for k, v in lab["boundaries"].items()})
    print("E +10 kW plan", r6(lab["cool_plus10"]["F"]), "16 m3/h plan", r6(lab["feed_plus1"]["F"]), "Fmin3 = 1 plan", r6(lab["min3_minus1"]["F"]))
    print("E conversions", r6(b["conversion"]), "marginals", r6(b["marginal"]), "cooling kW", r6(b["cooling_kW"]), "Fcool", r6(b["Fcool"]))
