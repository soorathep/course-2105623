"""2105623 Session 9: algorithms for constrained NLP (2027 draft).

Course form: minimize f(x) subject to g(x) <= 0 (multipliers mu >= 0) and h(x) = 0 (multipliers lambda),
L = f + sum(mu g) + sum(lambda h). The barrier parameter is written tau (Ipopt calls it mu).

Part A  penalty, augmented Lagrangian and Newton on the KKT conditions (x1 + x2 on a circle)
Part B  a teaching SQP with damped BFGS and an l1 merit function, and a log barrier (x1 x2 in an ellipse)
Part C  starting points: SLSQP from a grid of starts on a nonconvex feasible set
Lab 9   the Williams-Otto plant (Ray and Szekely, 1973), with SciPy SLSQP here and Ipopt through CasADi in Colab

Runs in desktop Python and in the browser (NumPy and SciPy). Ipopt needs CasADi: pip install casadi.
"""
import json, math, sys
import numpy as np
from scipy.optimize import minimize


def r6(v):
    """Round for printing, so recorded and live runs print the same text."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, d=3):
    return f"{float(v):.{d}g}"


# ---------------------------------------------------------------------------
# Part A. min x1 + x2  s.t.  h = x1^2 + x2^2 - 2 = 0   (minimum (-1, -1), lambda = 0.5)
# ---------------------------------------------------------------------------
def circ_f(x): return x[0] + x[1]
def circ_h(x): return x[0]**2 + x[1]**2 - 2.0


def penalty_min(rho, x0=(-1.2, -1.2), tol=1e-12):
    """Minimize Q = f + (rho/2) h^2 by Newton with backtracking. Returns x, h, lambda estimate rho*h, cond(Hessian of Q)."""
    x = np.array(x0, float)
    Q = lambda x: circ_f(x) + 0.5 * rho * circ_h(x)**2
    for _ in range(200):
        h = circ_h(x); gh = 2 * x
        g = np.ones(2) + rho * h * gh
        H = rho * (np.outer(gh, gh) + h * 2 * np.eye(2))
        lam, V = np.linalg.eigh(H); lam = np.maximum(np.abs(lam), 1e-8)
        p = -V @ ((V.T @ g) / lam)
        a = 1.0
        while Q(x + a * p) > Q(x) + 1e-4 * a * (g @ p) and a > 1e-12:
            a /= 2
        x = x + a * p
        if np.max(np.abs(g)) < tol:
            break
    h = circ_h(x); gh = 2 * x
    H = rho * (np.outer(gh, gh) + h * 2 * np.eye(2))
    return {"rho": rho, "x": x.tolist(), "h": h, "lam_est": rho * h, "cond": float(np.linalg.cond(H))}


def penalty_path(rhos=None):
    rhos = np.logspace(-0.5, 3, 15) if rhos is None else rhos
    out, x0 = [], (-1.2, -1.2)
    for r in rhos:
        q = penalty_min(float(r), x0); out.append(q); x0 = q["x"]
    return out


def aug_lagrangian(rho=2.0, lam0=0.0, iters=8):
    """Minimize L_A = f + lambda h + (rho/2) h^2 for fixed lambda, then lambda <- lambda + rho h. rho stays fixed."""
    lam, x, rows = lam0, np.array([-1.2, -1.2]), []
    for k in range(iters):
        LA = lambda x: circ_f(x) + lam * circ_h(x) + 0.5 * rho * circ_h(x)**2
        for _ in range(200):
            h = circ_h(x); gh = 2 * x
            g = np.ones(2) + (lam + rho * h) * gh
            H = rho * np.outer(gh, gh) + (lam + rho * h) * 2 * np.eye(2)
            e, V = np.linalg.eigh(H); e = np.maximum(np.abs(e), 1e-8)
            p = -V @ ((V.T @ g) / e); a = 1.0
            while LA(x + a * p) > LA(x) + 1e-4 * a * (g @ p) and a > 1e-12:
                a /= 2
            x = x + a * p
            if np.max(np.abs(g)) < 1e-12:
                break
        h = circ_h(x)
        rows.append({"k": k, "lam": lam, "x": x.tolist(), "h": h})
        lam = lam + rho * h
    return rows


def newton_kkt(x0=(-1.5, -0.5), lam0=1.0, iters=8):
    """Newton's method on grad f + lambda grad h = 0, h = 0. Each step solves [[H_L, a],[a', 0]] [p; d] = -[grad L; h]."""
    x, lam, rows = np.array(x0, float), float(lam0), []
    for k in range(iters + 1):
        a = 2 * x; gL = np.ones(2) + lam * a; h = circ_h(x)
        res = float(np.max(np.abs(np.r_[gL, h])))
        rows.append({"k": k, "x": x.tolist(), "lam": lam, "res": res})
        if res < 1e-14:
            break
        K = np.zeros((3, 3)); K[:2, :2] = 2 * lam * np.eye(2); K[:2, 2] = a; K[2, :2] = a
        d = np.linalg.solve(K, -np.r_[gL, h])
        x = x + d[:2]; lam = lam + d[2]
    return rows


# ---------------------------------------------------------------------------
# Part B. min -x1 x2  s.t.  g1 = x1^2 + 4 x2^2 - 4 <= 0,  g2 = -x1 <= 0,  g3 = -x2 <= 0
#         minimum (sqrt 2, sqrt 2 / 2), f* = -1, mu1 = 1/4
# ---------------------------------------------------------------------------
def ell_f(x): return -x[0] * x[1]
def ell_df(x): return np.array([-x[1], -x[0]])
def ell_g(x): return np.array([x[0]**2 + 4 * x[1]**2 - 4, -x[0], -x[1]])
def ell_dg(x): return np.array([[2 * x[0], 8 * x[1]], [-1.0, 0.0], [0.0, -1.0]])


def qp_enum(c, B, gval, A, tol=1e-10):
    """min c'p + p'Bp/2 s.t. gval + A p <= 0, two variables, by trying every active set of size 0, 1 or 2."""
    m, best = len(gval), None
    sets = [()] + [(i,) for i in range(m)] + [(i, j) for i in range(m) for j in range(i + 1, m)]
    for S in sets:
        k = len(S); K = np.zeros((2 + k, 2 + k)); K[:2, :2] = B
        for t, i in enumerate(S):
            K[:2, 2 + t] = A[i]; K[2 + t, :2] = A[i]
        rhs = np.r_[-c, [-gval[i] for i in S]]
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            continue
        p, mu = sol[:2], sol[2:]
        if np.any(mu < -tol) or np.any(gval + A @ p > 1e-9):
            continue
        val = c @ p + 0.5 * p @ B @ p
        if best is None or val < best[0] - 1e-14:
            full = np.zeros(m); full[list(S)] = mu
            best = (val, p, full, S)
    return best


def sqp_teach(x0, iters=40, tol=1e-8):
    """SQP with a damped BFGS Hessian of the Lagrangian (Powell), an l1 merit function and backtracking."""
    x, B, rho = np.array(x0, float), np.eye(2), 0.0
    rows, status = [], "iteration limit"
    for k in range(iters):
        f, df, g, A = ell_f(x), ell_df(x), ell_g(x), ell_dg(x)
        qp = qp_enum(df, B, g, A)
        if qp is None:
            rows.append({"k": k, "x": x.tolist(), "f": f, "viol": float(np.sum(np.maximum(g, 0)))})
            status = "QP subproblem infeasible"; break
        _, p, mu, S = qp
        viol = float(np.sum(np.maximum(g, 0)))
        rows.append({"k": k, "x": x.tolist(), "f": f, "viol": viol, "p": p.tolist(), "mu": mu.tolist(), "active": list(S)})
        if np.linalg.norm(p) < tol:
            status = "converged"; break
        rho = max(rho, 1.5 * float(np.max(mu)) + 0.1)
        phi = lambda z: ell_f(z) + rho * float(np.sum(np.maximum(ell_g(z), 0)))
        D = df @ p - rho * viol; a = 1.0
        while phi(x + a * p) > phi(x) + 1e-4 * a * D and a > 1e-10:
            a /= 2
        rows[-1]["alpha"] = a
        xn = x + a * p; s = xn - x
        y = (ell_df(xn) + ell_dg(xn).T @ mu) - (df + A.T @ mu)
        sBs, sy = s @ B @ s, s @ y
        if sy < 0.2 * sBs:
            th = 0.8 * sBs / (sBs - sy); y = th * y + (1 - th) * (B @ s); sy = s @ y
        Bs = B @ s
        B = B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy
        x = xn
    return {"status": status, "nit": len(rows) - 1, "x": x.tolist(), "f": ell_f(x), "rows": rows}


def slsqp_ell(x0):
    path = [list(x0)]
    r = minimize(ell_f, x0, jac=ell_df, method="SLSQP", constraints=[{"type": "ineq", "fun": lambda x: -ell_g(x), "jac": lambda x: -ell_dg(x)}],
                 options={"ftol": 1e-10, "maxiter": 100}, callback=lambda xk: path.append(list(xk)))
    return {"status": int(r.status), "message": r.message, "nit": int(r.nit), "nfev": int(r.nfev), "x": r.x.tolist(), "f": float(r.fun), "path": path}


def ell_audit(x, scale=1.0):
    """The four KKT residuals of min -x1 x2 on the ellipse at x (objective multiplied by scale):
    mu by least squares on the rows within 1e-6 of active."""
    x = np.array(x, float); g = ell_g(x); A = ell_dg(x); df = scale * ell_df(x)
    act = [i for i in range(3) if g[i] > -1e-6]
    mu = np.zeros(3)
    if act:
        m, *_ = np.linalg.lstsq(A[act].T, -df, rcond=None); mu[act] = m
    return {"stationarity": float(np.max(np.abs(df + A.T @ mu))), "primal": float(np.max(np.maximum(g, 0))),
            "dual": float(max(0.0, -np.min(mu))), "complementarity": float(np.max(np.abs(mu * g))), "mu": mu.tolist(), "active": act}


def slsqp_messages():
    """Five SLSQP runs on the ellipse problem, and what the residuals of the problem as posed say about each."""
    con = [{"type": "ineq", "fun": lambda x: -ell_g(x)}]
    runs = [("as posed", 1.0, con, 100), ("maxiter = 2", 1.0, con, 2), ("f times 1e-8", 1e-8, con, 100), ("f times 1e6", 1e6, con, 100)]
    out = []
    for name, sc, cs, mi in runs:
        r = minimize(lambda x: sc * ell_f(x), [0.5, 0.5], method="SLSQP", constraints=cs, options={"maxiter": mi, "ftol": 1e-6 if mi == 100 else 1e-10})
        out.append({"name": name, "status": int(r.status), "message": r.message, "nit": int(r.nit), "x": r.x.tolist(), "f": float(ell_f(r.x)), "audit": ell_audit(r.x)})
    con2 = [{"type": "ineq", "fun": lambda x: np.array([1 - x[0]**2 - x[1]**2, x[0] + x[1] - 3])}]
    r = minimize(lambda x: x[0]**2 + x[1]**2, [0.2, 0.2], method="SLSQP", constraints=con2)
    g = np.array([r.x[0]**2 + r.x[1]**2 - 1, 3 - r.x[0] - r.x[1]])
    out.append({"name": "infeasible rows", "status": int(r.status), "message": r.message, "nit": int(r.nit), "x": r.x.tolist(), "f": float(r.fun),
                "audit": {"primal": float(np.max(np.maximum(g, 0)))}})
    return out


def ell_ipopt(x0=(0.5, 0.5), infeasible=False, print_level=0):
    """Ipopt through CasADi on the ellipse problem; infeasible=True adds the row x1 + x2 >= 3, which cannot hold."""
    import casadi as ca
    x = ca.SX.sym("x", 2)
    g = [x[0]**2 + 4 * x[1]**2 - 4] + ([3 - x[0] - x[1]] if infeasible else [])
    S = ca.nlpsol("ell", "ipopt", {"x": x, "f": -x[0] * x[1], "g": ca.vertcat(*g)}, {"ipopt.print_level": print_level, "print_time": 0, "ipopt.sb": "yes"})
    r = S(x0=list(x0), lbx=[0, 0], ubx=[np.inf, np.inf], lbg=-np.inf, ubg=0)
    st = S.stats()
    return {"status": st["return_status"], "iter": int(st["iter_count"]), "x": np.array(r["x"]).ravel().tolist(), "mu": np.array(r["lam_g"]).ravel().tolist()}


def ipopt_log(code):
    """Run code that prints an Ipopt log in a subprocess and return the log without the banner."""
    import subprocess
    out = subprocess.run([sys.executable, "-c", "import nlp_lab as m; " + code], capture_output=True, text=True).stdout
    keep = [ln for ln in out.splitlines() if not ln.startswith("*") and "Ipopt is released" not in ln and "visit https" not in ln
            and "This program contains Ipopt" not in ln and not ln.startswith("CasADi")]
    while keep and not keep[0].strip():
        keep.pop(0)
    return "\n".join(keep).strip()


def barrier_min(tau, x0=(1.0, 0.5)):
    """Minimize f - tau * sum(log(-g_i)) by Newton with backtracking, from a strictly feasible start."""
    x = np.array(x0, float)
    def phi(z):
        gz = ell_g(z)
        return np.inf if np.any(gz >= 0) else ell_f(z) - tau * np.sum(np.log(-gz))
    for _ in range(200):
        g, A = ell_g(x), ell_dg(x)
        grad = ell_df(x) + A.T @ (tau / -g)
        H = np.array([[0.0, -1.0], [-1.0, 0.0]]) + tau / -g[0] * np.diag([2.0, 8.0]) + sum(np.outer(A[i], A[i]) * tau / g[i]**2 for i in range(3))
        e, V = np.linalg.eigh(H); e = np.maximum(np.abs(e), 1e-10)
        p = -V @ ((V.T @ grad) / e); a = 1.0
        while phi(x + a * p) > phi(x) + 1e-4 * a * (grad @ p) and a > 1e-14:
            a /= 2
        x = x + a * p
        if np.max(np.abs(grad)) < 1e-12:
            break
    g = ell_g(x)
    return {"tau": tau, "x": x.tolist(), "f": ell_f(x), "mu": (tau / -g).tolist(), "gap": float(3 * tau)}


def barrier_path(taus=None):
    taus = np.logspace(0, -4, 17) if taus is None else taus
    out, x0 = [], (1.0, 0.5)
    for t in taus:
        q = barrier_min(float(t), x0); out.append(q); x0 = q["x"]
    return out


# ---------------------------------------------------------------------------
# Part C. min -x1 - 2 x2  s.t.  x1 x2 - 1 <= 0,  0 <= x1 <= 4,  0 <= x2 <= 3   (nonconvex feasible set)
#         local minimum (4, 0.25), f = -4.5; global minimum (1/3, 3), f = -6.333
# ---------------------------------------------------------------------------
def hyp_slsqp(x0, keep_path=True):
    path = [list(map(float, x0))]
    r = minimize(lambda x: -x[0] - 2 * x[1], x0, jac=lambda x: np.array([-1.0, -2.0]), method="SLSQP",
                 bounds=[(0, 4), (0, 3)], constraints=[{"type": "ineq", "fun": lambda x: np.array([1 - x[0] * x[1]]),
                                                        "jac": lambda x: np.array([[-x[1], -x[0]]])}],
                 options={"ftol": 1e-10, "maxiter": 100}, callback=(lambda xk: path.append(xk.tolist())) if keep_path else None)
    return {"status": int(r.status), "nit": int(r.nit), "x": r.x.tolist(), "f": float(r.fun), "path": path}


def hyp_label(r):
    if r["status"] != 0:
        return "failed"
    x = r["x"]
    if abs(x[0] - 1 / 3) < 1e-4 and abs(x[1] - 3) < 1e-4:
        return "global"
    if abs(x[0] - 4) < 1e-4 and abs(x[1] - 0.25) < 1e-4:
        return "local"
    return "other"


def hyp_grid(n=21):
    xs = np.linspace(0.05, 3.95, n); ys = np.linspace(0.05, 2.95, n); cells = []
    for y in ys:
        for x in xs:
            r = hyp_slsqp([x, y])
            cells.append({"x0": [round(float(x), 4), round(float(y), 4)], "label": hyp_label(r), "nit": r["nit"], "end": [round(v, 4) for v in r["x"]],
                          "path": [[round(a, 3), round(b, 3)] for a, b in r["path"]]})
    return {"n": n, "cells": cells}


# ---------------------------------------------------------------------------
# Lab 9. Williams-Otto plant (Ray and Szekely, 1973; ROI form as in Biegler's texts)
#   A + B -> C (k1), B + C -> P + E (k2), C + P -> G (k3); CSTR, decanter removes G, column takes P overhead
#   (P equal to 10% of E stays in the bottoms), bottoms split into purge (fraction eta) and recycle.
#   Units: lb/h, ft^3, degrees Rankine; density 50 lb/ft^3; P production fixed at 4763 lb/h.
# ---------------------------------------------------------------------------
WO = {"rho": 50.0, "FP": 4763.0, "k0": [5.9755e9, 2.5962e12, 9.6283e15], "E": [12000.0, 15000.0, 20000.0],
      "price": {"P": 2207.0, "purge": 50.0, "A": 168.0, "B": 252.0, "eff": 2.22, "G": 84.0, "V": 60.0, "capital": 600.0},
      "names": ["FA", "FB", "V", "T", "eta", "A", "B", "C", "E", "P", "G"],
      "units": ["lb/h", "lb/h", "ft3", "degR", "-", "lb/h", "lb/h", "lb/h", "lb/h", "lb/h", "lb/h"],
      "scale": [1e4, 1e4, 10.0, 100.0, 1.0, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4],
      "lo": [0, 0, 30, 580, 0, 0, 0, 0, 0, 0, 0], "hi": [1e5, 1e5, 100, 680, 1, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5]}


def wo_model(x, exp=np.exp, D=WO, FP=None):
    """x = [FA, FB, V, T, eta, A, B, C, E, P, G] in plant units; A..G are reactor effluent flows.
    Returns ROI in % and the 7 equality rows h (lb/h): 6 component balances with the recycle closed, and the P spec."""
    FA, FB, V, T, eta, a, b, c, e, p, g = [x[i] for i in range(11)]
    S = a + b + c + e + p + g
    k1, k2, k3 = [D["k0"][i] * exp(-D["E"][i] / T) for i in range(3)]
    r1 = k1 * (a / S) * (b / S) * V * D["rho"]; r2 = k2 * (b / S) * (c / S) * V * D["rho"]; r3 = k3 * (c / S) * (p / S) * V * D["rho"]
    Pout = p - 0.1 * e                                   # P taken overhead
    bott = [a, b, c, e, 0.1 * e]                         # column bottoms: A, B, C, E, and P
    R = [(1 - eta) * v for v in bott]                    # recycle
    purge = eta * (a + b + c + e + 0.1 * e)
    h = [FA + R[0] - r1 - a, FB + R[1] - r1 - r2 - b, R[2] + 2 * r1 - 2 * r2 - r3 - c, R[3] + 2 * r2 - e,
         R[4] + r2 - 0.5 * r3 - p, 1.5 * r3 - g, Pout - (D["FP"] if FP is None else FP)]
    c_ = D["price"]
    roi = 100 * (c_["P"] * Pout + c_["purge"] * purge - c_["A"] * FA - c_["B"] * FB - c_["eff"] * S - c_["G"] * g
                 - c_["V"] * V * D["rho"]) / (c_["capital"] * V * D["rho"])
    return roi, h


def wo_parts(x, D=WO):
    FA, FB, V, T, eta, a, b, c, e, p, g = x
    S = a + b + c + e + p + g
    return {"ROI": wo_model(x)[0], "FP": p - 0.1 * e, "purge": eta * (a + b + c + e + 0.1 * e), "FG": g, "Feff": S,
            "recycle": (1 - eta) * (a + b + c + e + 0.1 * e), "T_C": (T - 491.67) / 1.8, "V_m3": V * 0.0283168}


WO_STARTS = {  # scaled units: flows in 10^4 lb/h, V in 10 ft3, T in 100 degR, eta as is
    "base":      [1.0, 2.0, 6.0, 6.5, 0.10, 1.0, 3.0, 0.5, 4.0, 1.0, 0.10],
    "hot":       [1.5, 3.0, 4.0, 6.8, 0.20, 5.0, 15.0, 1.0, 15.0, 2.0, 0.30],
    "cold":      [1.0, 2.0, 9.0, 5.8, 0.10, 3.0, 8.0, 1.0, 8.0, 1.5, 0.05],
    "big":       [3.0, 6.0, 10.0, 6.0, 0.50, 20.0, 40.0, 5.0, 30.0, 10.0, 2.0],
    "lean":      [0.5, 0.5, 3.0, 6.2, 0.05, 0.2, 0.2, 0.1, 0.2, 0.5, 0.01],
    "no purge":  [1.3, 3.0, 3.1, 6.7, 0.00, 4.7, 14.7, 0.8, 14.6, 1.9, 0.30],
    "empty":     [1.0, 2.0, 5.0, 6.5, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def wo_scaled(D=WO, FP=None):
    s = np.array(D["scale"])
    f = lambda z: -wo_model(z * s, FP=FP)[0] / 100
    hz = lambda z: np.array(wo_model(z * s, FP=FP)[1]) / 1e4
    return s, f, hz


def wo_jac(fun, z, hstep=1e-7):
    z = np.array(z, float); f0 = np.atleast_1d(fun(z)); J = np.zeros((len(f0), len(z)))
    for i in range(len(z)):
        e = np.zeros(len(z)); e[i] = hstep * max(1.0, abs(z[i]))
        J[:, i] = (np.atleast_1d(fun(z + e)) - np.atleast_1d(fun(z - e))) / (2 * e[i])
    return J


def wo_slsqp(z0, D=WO, FP=None, Tmax=None, scaled=True, maxiter=200):
    """SciPy SLSQP on the Williams-Otto plant from a start z0 in scaled units. Bounds as in WO; Tmax tightens T."""
    s = np.array(D["scale"]) if scaled else np.ones(11)
    lo = np.array(D["lo"], float) / s; hi = np.array(D["hi"], float) / s
    if Tmax is not None:
        hi[3] = Tmax / s[3]
    f = lambda z: -wo_model(z * s, FP=FP)[0] / 100
    hz = lambda z: np.array(wo_model(z * s, FP=FP)[1]) / 1e4
    z0 = np.array(z0, float) * (np.array(D["scale"]) / s)
    with np.errstate(all="ignore"):
        r = minimize(f, z0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=[{"type": "eq", "fun": hz}],
                     options={"ftol": 1e-10, "maxiter": maxiter})
    x = r.x * s
    with np.errstate(all="ignore"):
        roi, h = wo_model(x, FP=FP)
    return {"status": int(r.status), "message": r.message, "nit": int(r.nit), "nfev": int(r.nfev), "x": x.tolist(),
            "roi": float(roi) if np.isfinite(roi) else None, "maxh": float(np.max(np.abs(h))) if np.all(np.isfinite(h)) else None}


def wo_audit(x, D=WO, FP=None, Tmax=None):
    """Residual audit at a candidate x (plant units): multipliers by least squares on the stationarity rows,
    using the equality rows and the bounds that are active. Returns the four KKT residuals (scaled problem)."""
    s = np.array(D["scale"]); z = np.array(x) / s
    lo = np.array(D["lo"], float) / s; hi = np.array(D["hi"], float) / s
    if Tmax is not None:
        hi[3] = Tmax / s[3]
    f = lambda z: -wo_model(z * s, FP=FP)[0] / 100
    hz = lambda z: np.array(wo_model(z * s, FP=FP)[1]) / 1e4
    gf = wo_jac(f, z)[0]; Jh = wo_jac(hz, z)
    act_lo = np.where(z - lo < 1e-7)[0]; act_hi = np.where(hi - z < 1e-7)[0]
    cols = [Jh.T] + [-np.eye(11)[:, [i]] for i in act_lo] + [np.eye(11)[:, [i]] for i in act_hi]
    M = np.hstack(cols)
    m, *_ = np.linalg.lstsq(M, -gf, rcond=None)
    lam = m[:7]; mu_lo = m[7:7 + len(act_lo)]; mu_hi = m[7 + len(act_lo):]
    stat = float(np.max(np.abs(gf + M @ m)))
    prim = float(max(np.max(np.abs(hz(z))), np.max(np.maximum(lo - z, 0)), np.max(np.maximum(z - hi, 0))))
    dual = float(max([0.0] + [-v for v in mu_lo] + [-v for v in mu_hi]))
    return {"stationarity": stat, "primal": prim, "dual": dual, "complementarity": 0.0, "lambda": lam.tolist(),
            "active_lo": [D["names"][i] for i in act_lo], "active_hi": [D["names"][i] for i in act_hi],
            "mu_hi": mu_hi.tolist(), "mu_lo": mu_lo.tolist()}


def wo_ipopt(z0, D=WO, FP=None, Tmax=None, print_level=0, scaled=True):
    """Ipopt through CasADi (exact first and second derivatives by automatic differentiation).
    Same scaled variables and rows as wo_slsqp. Needs: pip install casadi."""
    import casadi as ca
    s = np.array(D["scale"]) if scaled else np.ones(11)
    z = ca.SX.sym("z", 11)
    roi, h = wo_model([z[i] * s[i] for i in range(11)], exp=ca.exp, FP=FP)
    fsc, hsc = (100.0, 1e4) if scaled else (1.0, 1.0)
    nlp = {"x": z, "f": -roi / fsc, "g": ca.vertcat(*h) / hsc}
    opts = {"ipopt.print_level": print_level, "print_time": 0, "ipopt.max_iter": 500, "ipopt.sb": "yes"}
    if print_level == 0:
        opts["ipopt.print_level"] = 0
    S = ca.nlpsol("wo", "ipopt", nlp, opts)
    lo = np.array(D["lo"], float) / s; hi = np.array(D["hi"], float) / s
    if Tmax is not None:
        hi[3] = Tmax / s[3]
    z0 = np.array(z0, float) * (np.array(D["scale"]) / s)
    r = S(x0=z0, lbx=lo, ubx=hi, lbg=0, ubg=0)
    st = S.stats(); x = np.array(r["x"]).ravel() * s
    lam = np.array(r["lam_g"]).ravel() * (100.0 / hsc * fsc / 100.0)    # multipliers of the scaled rows
    mux = np.array(r["lam_x"]).ravel()
    return {"status": st["return_status"], "success": bool(st["success"]), "iter": int(st["iter_count"]), "x": x.tolist(),
            "roi": float(-r["f"]) * fsc, "lam_g_scaled": lam.tolist(), "lam_x_scaled": mux.tolist()}


def wo_ipopt_log(start="base"):
    """The full Ipopt iteration log from one start (print_level 5), as text."""
    return ipopt_log(f"m.wo_ipopt(m.WO_STARTS[{start!r}], print_level=5)")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def lab_results():
    starts = WO_STARTS
    sl = {k: wo_slsqp(z) for k, z in starts.items()}
    sl_un = {k: wo_slsqp(z, scaled=False) for k, z in starts.items()}
    ip = {k: wo_ipopt(z) for k, z in starts.items()}
    base = ip["base"]; xo = base["x"]
    audit = {"slsqp_base": wo_audit(sl["base"]["x"]), "ipopt_base": wo_audit(xo),
             "slsqp_unscaled_hot": wo_audit(sl_un["hot"]["x"]), "slsqp_unscaled_base": wo_audit(sl_un["base"]["x"])}
    fp_plus = wo_ipopt(starts["base"], FP=4863.0)
    tsweep = []
    for Tm in [680, 678, 676, 674, 672, 670, 668, 666, 664, 662, 660, 655, 650, 645, 640, 635, 630, 625, 620]:
        q = wo_ipopt(starts["base"], Tmax=float(Tm))
        tsweep.append({"Tmax": Tm, "status": q["status"], "roi": q["roi"], "T": q["x"][3], "V": q["x"][2], "FA": q["x"][0], "FB": q["x"][1],
                       "eta": q["x"][4], "muT": q["lam_x_scaled"][3]})
    return {"data": {k: WO[k] for k in ("rho", "FP", "k0", "E", "price", "names", "units", "scale", "lo", "hi")}, "starts": starts,
            "opt": {"x": xo, "roi": base["roi"], **wo_parts(xo), "lam": base["lam_g_scaled"]},
            "slsqp": sl, "slsqp_unscaled": sl_un, "ipopt": ip, "audit": audit, "fp_plus": {"roi": fp_plus["roi"], "x": fp_plus["x"]},
            "tsweep": tsweep, "log": wo_ipopt_log("base")}


def classroom_results():
    import scipy
    R = {"meta": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}}
    try:
        import casadi
        R["meta"]["casadi"] = casadi.__version__
    except ImportError:
        pass
    sq_starts = {"inside": [0.5, 0.5], "outside": [2.5, 1.5], "corner": [0.2, 1.8]}
    R["lecture"] = {"penalty": penalty_path(), "penalty_check": penalty_path([1, 10, 100, 1000]), "al": aug_lagrangian(), "newton_kkt": newton_kkt(), "newton_kkt_far": newton_kkt((0.8, 1.4), -0.3),
                    "sqp": {k: sqp_teach(v) for k, v in sq_starts.items()}, "slsqp_ell": {k: slsqp_ell(v) for k, v in sq_starts.items()},
                    "barrier": barrier_path(), "barrier_check": barrier_path([1, 0.1, 0.01, 0.001]), "hyp": hyp_grid(), "hyp_check": [dict({k: v for k, v in hyp_slsqp(x0).items() if k != "path"}, x0=x0, label=hyp_label(hyp_slsqp(x0))) for x0 in ([0.5, 0.5], [3.5, 0.5], [2.0, 2.5], [0.5, 2.5])],
                    "messages": slsqp_messages(), "ell_ipopt": ell_ipopt(), "ell_ipopt_inf": ell_ipopt(infeasible=True),
                    "log_ell": ipopt_log("m.ell_ipopt(print_level=5)"), "log_ell_inf": ipopt_log("m.ell_ipopt(infeasible=True, print_level=5)")}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    if "--json" in sys.argv:
        R = classroom_results()
        with open("nlp-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote nlp-results.json", R["meta"]); sys.exit()
    print("A penalty", [(sig(q["rho"]), r6(q["x"]), sig(q["h"]), r6(q["lam_est"]), sig(q["cond"])) for q in penalty_path([1, 10, 100, 1000])])
    print("A AL", [(q["k"], r6(q["lam"]), r6(q["x"]), sig(q["h"])) for q in aug_lagrangian()])
    print("A Newton-KKT", [(q["k"], r6(q["x"]), r6(q["lam"]), sig(q["res"])) for q in newton_kkt()])
    for k, v in {"inside": [0.5, 0.5], "outside": [2.5, 1.5], "corner": [0.2, 1.8]}.items():
        r = sqp_teach(v); s = slsqp_ell(v)
        print("B SQP", k, r["status"], r["nit"], r6(r["x"]), r6(r["f"]), "| SLSQP", s["status"], s["nit"], s["nfev"], r6(s["x"]), r6(s["f"]))
    print("B barrier", [(sig(q["tau"]), r6(q["x"]), r6(q["f"]), r6(q["mu"][0])) for q in barrier_path([1, 0.1, 0.01, 0.001])])
    for x0 in ([0.5, 0.5], [3.5, 0.5], [2.0, 2.5], [0.5, 2.5]):
        r = hyp_slsqp(x0); print("C hyp", x0, hyp_label(r), r["status"], r["nit"], r6(r["x"]), r6(r["f"]))
    for k, z in WO_STARTS.items():
        r = wo_slsqp(z); print("D WO SLSQP", k, r["status"], r["nit"], r["nfev"], r["roi"] and r6(r["roi"]), r["message"])
