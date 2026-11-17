"""2105623 Session 11: dynamic optimization and model predictive control (2027 draft).

Part A  Batch reactor A -> B -> C, temperature profile that maximizes B at t = 1 h (Logsdon and Biegler, 1989):
        isothermal operation, single shooting with a piecewise-constant temperature, and (in Colab, with CasADi)
        orthogonal collocation solved by Ipopt
Part B  Collocation by hand: Radau collocation on dx/dt = -k x, and the sparsity of the collocation NLP
Part C  Level control of two tanks in series by MPC, solved at every step as a bounded least-squares problem

Runs in desktop Python and in the browser (NumPy and SciPy). Collocation with Ipopt needs CasADi: pip install casadi.
Course notation: lambda and mu are multipliers; the barrier parameter of Session 9 was tau.
"""
import json, math, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar, lsq_linear


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, d=4):
    return f"{float(v):.{d}g}"


# ---------------------------------------------------------------------------
# Part A. Batch reactor: dx1/dt = -k1 x1^2, dx2/dt = k1 x1^2 - k2 x2, x(0) = (1, 0), 0 <= t <= 1 h,
#         k1 = 4000 exp(-2500/T), k2 = 620000 exp(-5000/T) (1/h), 298 <= T <= 398 K; maximize x2(1)
# ---------------------------------------------------------------------------
BR = {"tf": 1.0, "h": 0.005, "Tlo": 298.0, "Thi": 398.0, "k10": 4000.0, "E1": 2500.0, "k20": 620000.0, "E2": 5000.0}


def br_rates(T, k2f=1.0):
    return BR["k10"] * math.exp(-BR["E1"] / T), k2f * BR["k20"] * math.exp(-BR["E2"] / T)


def br_sim(Ts, x0=(1.0, 0.0), k2f=1.0, t_int=None, traj=False):
    """Integrate with the temperature Ts[i] held on interval i (piecewise constant) by classical RK4 with a fixed step
    of 0.005 h. Each interval lasts tf/len(Ts) unless t_int is given. Returns x at the end, or (t, x1, x2) if traj."""
    t_int = BR["tf"] / len(Ts) if t_int is None else t_int
    n = int(round(t_int / BR["h"])); h = t_int / n
    a, b = float(x0[0]), float(x0[1]); t = 0.0; out = [(t, a, b)]
    for T in Ts:
        k1, k2 = br_rates(float(T), k2f)
        for _ in range(n):
            f1a = -k1 * a * a; f1b = k1 * a * a - k2 * b
            a2 = a + h / 2 * f1a; b2 = b + h / 2 * f1b; f2a = -k1 * a2 * a2; f2b = k1 * a2 * a2 - k2 * b2
            a3 = a + h / 2 * f2a; b3 = b + h / 2 * f2b; f3a = -k1 * a3 * a3; f3b = k1 * a3 * a3 - k2 * b3
            a4 = a + h * f3a; b4 = b + h * f3b; f4a = -k1 * a4 * a4; f4b = k1 * a4 * a4 - k2 * b4
            a += h / 6 * (f1a + 2 * f2a + 2 * f3a + f4a); b += h / 6 * (f1b + 2 * f2b + 2 * f3b + f4b); t += h
            if traj:
                out.append((t, a, b))
    if traj:
        return [list(c) for c in zip(*out)]
    return [a, b]


def br_iso():
    """Isothermal operation: x2(1) against a constant T, and the best constant T."""
    Ts = np.arange(298.0, 398.01, 2.0)
    r = minimize_scalar(lambda T: -br_sim([T])[1], bounds=(BR["Tlo"], BR["Thi"]), method="bounded", options={"xatol": 1e-6})
    return {"T": Ts.tolist(), "x2": [br_sim([T])[1] for T in Ts], "Tbest": float(r.x), "x2best": float(-r.fun)}


def br_shoot(N, k2f=1.0, x0=(1.0, 0.0), t_int=None, u0=None):
    """Single shooting: N piecewise-constant temperatures, T = 298 + 100 u with 0 <= u <= 1, maximize x2(tf) by SLSQP
    with finite-difference gradients. Each objective evaluation integrates the model over the whole horizon."""
    u0 = np.full(N, 0.5) if u0 is None else np.asarray(u0, float)
    r = minimize(lambda u: -br_sim(298 + 100 * u, x0, k2f, t_int)[1], u0, method="SLSQP", bounds=[(0, 1)] * N,
                 options={"ftol": 1e-12, "maxiter": 300})
    return {"N": N, "T": (298 + 100 * r.x).tolist(), "x2": float(-r.fun), "nit": int(r.nit), "nfev": int(r.nfev), "status": int(r.status)}


def br_robust():
    """The nominal 20-interval profile when k2 is really 10% higher: predicted, obtained, best possible, and a
    shrinking-horizon re-optimization that uses the measured state but keeps the nominal (wrong) k2."""
    nom = br_shoot(20); tru = br_shoot(20, k2f=1.1)
    applied = br_sim(nom["T"], k2f=1.1)[1]
    x = [1.0, 0.0]; used = []; u_prev = (np.array(nom["T"]) - 298) / 100
    for i in range(20):
        T = nom["T"][0] if i == 0 else br_shoot(20 - i, x0=x, t_int=0.05, u0=u_prev[i:])["T"][0]
        used.append(T); x = br_sim([T], x0=x, k2f=1.1, t_int=0.05)
    return {"predicted": nom["x2"], "applied": applied, "best": tru["x2"], "T_best": tru["T"], "shrinking": x[1], "T_shrinking": used}


def br_colloc(NE=50, K=3, Thi=398.0, k2f=1.0, print_level=0):
    """Direct collocation: NE finite elements, K Radau points each, T constant on each element; Ipopt through CasADi.
    Variables: element-start states, collocation-point states and one temperature per element (in 100 K)."""
    import casadi as ca
    tau, Cm, Dv = colloc_matrices(K)
    h = BR["tf"] / NE; opti = ca.Opti(); Xk = opti.variable(2); opti.subject_to(Xk == ca.DM([1, 0])); us = []
    for e in range(NE):
        u = opti.variable(); opti.subject_to(opti.bounded(BR["Tlo"] / 100, u, Thi / 100)); opti.set_initial(u, 3.5); us.append(u)
        k1 = BR["k10"] * ca.exp(-BR["E1"] / (100 * u)); k2 = k2f * BR["k20"] * ca.exp(-BR["E2"] / (100 * u))
        Xc = [opti.variable(2) for _ in range(K)]
        for xc in Xc:
            opti.set_initial(xc, [0.5, 0.3])
        Xend = Dv[0] * Xk
        for j in range(1, K + 1):
            xp = Cm[0][j] * Xk
            for r in range(K):
                xp = xp + Cm[r + 1][j] * Xc[r]
            x = Xc[j - 1]
            opti.subject_to(xp == h * ca.vertcat(-k1 * x[0]**2, k1 * x[0]**2 - k2 * x[1]))
            Xend = Xend + Dv[j] * Xc[j - 1]
        Xk = opti.variable(2); opti.set_initial(Xk, [0.5, 0.3]); opti.subject_to(Xk == Xend)
    opti.minimize(-Xk[1])
    opti.solver("ipopt", {"print_time": False}, {"print_level": print_level, "tol": 1e-10, "sb": "yes"})
    s = opti.solve(); st = s.stats()
    return {"NE": NE, "K": K, "x2": float(s.value(Xk[1])), "iter": int(st["iter_count"]), "status": st["return_status"],
            "nvar": int(opti.nx), "ncon": int(opti.ng), "T": [100 * float(s.value(u)) for u in us]}


# ---------------------------------------------------------------------------
# Part B. Collocation by hand
# ---------------------------------------------------------------------------
RADAU = {1: [1.0], 2: [1 / 3, 1.0], 3: [(4 - math.sqrt(6)) / 10, (4 + math.sqrt(6)) / 10, 1.0]}


def colloc_matrices(K):
    """Lagrange polynomials through tau = (0, Radau points): C[r][j] = derivative of polynomial r at point j,
    D[r] = polynomial r at tau = 1 (the end of the element)."""
    tau = [0.0] + RADAU[K]; Cm = [[0.0] * (K + 1) for _ in range(K + 1)]; Dv = [0.0] * (K + 1)
    for j in range(K + 1):
        p = np.poly1d([1.0])
        for r in range(K + 1):
            if r != j:
                p = p * np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        Dv[j] = float(p(1.0)); dp = np.polyder(p)
        for r in range(K + 1):
            Cm[j][r] = float(dp(tau[r]))
    return tau, Cm, Dv


def colloc_decay(k=5.0, K=3, NE=4):
    """Radau collocation of dx/dt = -k x, x(0) = 1 on [0, 1]: returns the element-end values and the error at t = 1."""
    tau, Cm, Dv = colloc_matrices(K); h = 1.0 / NE; x = 1.0; ends = [1.0]; pts = []
    for e in range(NE):
        A = np.array([[Cm[r + 1][j] + (k * h if r + 1 == j else 0.0) for r in range(K)] for j in range(1, K + 1)])
        b = np.array([-Cm[0][j] * x for j in range(1, K + 1)]); xc = np.linalg.solve(A, b)
        pts += [((e + tau[j + 1]) * h, float(xc[j])) for j in range(K)]
        x = Dv[0] * x + sum(Dv[j + 1] * xc[j] for j in range(K)); ends.append(float(x))
    return {"K": K, "NE": NE, "ends": ends, "points": pts, "x1": x, "err": abs(x - math.exp(-k))}


def colloc_pattern(NE=4, K=3):
    """Sparsity of the equality-row Jacobian of the batch collocation NLP. Variables, element by element:
    [X_e (2), Xc_e,1..K (2K), u_e (1)], then the final state. Rows: 2K collocation rows and 2 continuity rows per
    element (the fixed initial state and the bounds on u are left out). Returns the nonzero (row, column) pairs."""
    tau, Cm, Dv = colloc_matrices(K); nv_e = 2 + 2 * K + 1; nz = []; row = 0
    for e in range(NE):
        c0 = e * nv_e; xc = lambda r, s: c0 + 2 + 2 * r + s
        for j in range(K):
            for s in range(2):
                cols = [c0 + s] + [xc(r, s) for r in range(K)] + [xc(j, 0), c0 + 2 + 2 * K] + ([xc(j, 1)] if s == 1 else [])
                nz += [(row, c) for c in sorted(set(cols))]; row += 1
        for s in range(2):
            cols = ([c0 + s] if abs(Dv[0]) > 1e-12 else []) + [xc(r, s) for r in range(K) if abs(Dv[r + 1]) > 1e-12] + [c0 + nv_e + s]
            nz += [(row, c) for c in sorted(set(cols))]; row += 1
    return {"NE": NE, "K": K, "rows": row, "cols": NE * nv_e + 2, "nz": nz, "density": len(nz) / (row * (NE * nv_e + 2))}


# ---------------------------------------------------------------------------
# Part C. Two tanks in series; MPC on the level of tank 2
#   A1 dh1/dt = u - h1/R1,  A2 dh2/dt = h1/R1 - h2/R2 + d;  u: valve flow 0 to 0.5 m3/min; d: unmeasured flow into tank 2
# ---------------------------------------------------------------------------
TANK = {"A1": 1.0, "A2": 1.0, "R1": 2.0, "R2": 4.0, "dt": 0.5, "umax": 0.5, "steps": 120, "x0": [0.6, 1.2], "u0": 0.3,
        "r0": 1.2, "r1": 1.6, "k_r": 10, "d1": -0.05, "k_d": 60}


def tank_model():
    """Exact discretization with the inputs held over each 0.5 min step (zero-order hold)."""
    A1, A2, R1, R2 = TANK["A1"], TANK["A2"], TANK["R1"], TANK["R2"]
    Ac = np.array([[-1 / (A1 * R1), 0], [1 / (A2 * R1), -1 / (A2 * R2)]]); Bc = np.array([[1 / A1, 0], [0, 1 / A2]])
    M = expm(np.block([[Ac, Bc], [np.zeros((2, 4))]]) * TANK["dt"])
    return M[:2, :2], M[:2, 2], M[:2, 3]


def tank_pred(N):
    """Level of tank 2 over the next N steps: y = Phi x + G U + Gd d."""
    Ad, Bd, Ed = tank_model(); P = [np.linalg.matrix_power(Ad, k) for k in range(N + 1)]
    Phi = np.array([P[i + 1][1] for i in range(N)]); G = np.zeros((N, N)); Gd = np.zeros(N)
    for i in range(N):
        for j in range(i + 1):
            G[i, j] = (P[i - j] @ Bd)[1]
        Gd[i] = sum((P[i - j] @ Ed)[1] for j in range(i + 1))
    return Phi, G, Gd


def mpc_solve(x, u_prev, r, dhat, N, w, P=None):
    """min sum (y_i - r_i)^2 + w sum (u_i - u_{i-1})^2 subject to 0 <= u_i <= umax: a bounded linear least-squares
    problem in U, solved by lsq_linear (bounded-variable least squares)."""
    Phi, G, Gd = tank_pred(N) if P is None else P
    D = np.eye(N) - np.eye(N, k=-1); e = np.zeros(N); e[0] = u_prev
    A = np.vstack([G, math.sqrt(w) * D]); b = np.concatenate([np.asarray(r) - Phi @ x - Gd * dhat, math.sqrt(w) * e])
    return lsq_linear(A, b, bounds=(0, TANK["umax"]), method="bvls").x


def tank_run(N=20, w=1.0, est=True, openloop=False):
    """Closed loop for 60 min: setpoint 1.2 -> 1.6 m at 5 min; an unmeasured withdrawal of 0.05 m3/min from tank 2
    starts at 30 min. est: update a constant-disturbance estimate from the one-step prediction error of h2.
    openloop: plan the whole hour at t = 0 and apply it without feedback."""
    Ad, Bd, Ed = tank_model(); P = tank_pred(N); S = TANK["steps"]
    rr = lambda k: TANK["r0"] if k < TANK["k_r"] else TANK["r1"]
    x = np.array(TANK["x0"]); u = TANK["u0"]; dh = 0.0; H1 = [x[0]]; H2 = [x[1]]; U = []; Dh = []
    plan = mpc_solve(x, u, [rr(i + 1) for i in range(S)], 0.0, S, w) if openloop else None
    for k in range(S):
        if openloop:
            u = plan[k]
        else:
            u = mpc_solve(x, u, [rr(k + i + 1) for i in range(N)], dh, N, w, P)[0]
        d = TANK["d1"] if k >= TANK["k_d"] else 0.0
        xp = Ad @ x + Bd * u + Ed * dh; x = Ad @ x + Bd * u + Ed * d
        if est and not openloop:
            dh = dh + (x[1] - xp[1]) / Ed[1]
        H1.append(x[0]); H2.append(x[1]); U.append(u); Dh.append(dh)
    H2 = np.array(H2); seg = H2[TANK["k_r"]:TANK["k_d"]]; step = TANK["r1"] - TANK["r0"]
    out = np.where(np.abs(seg - TANK["r1"]) > 0.02 * step)[0]
    after = H2[TANK["k_d"]:]
    return {"N": N, "w": w, "est": est, "openloop": openloop, "h1": r6(H1), "h2": r6(H2), "u": r6(U), "dhat": r6(Dh),
            "overshoot": float(max(0.0, (seg.max() - TANK["r1"]) / step * 100)), "settle": float((out[-1] + 1 - 0) * TANK["dt"]) if len(out) else 0.0,
            "dist_dev": float(np.max(np.abs(after - TANK["r1"]))), "offset": float(H2[-1] - TANK["r1"]),
            "sat": int(np.sum(np.array(U) > TANK["umax"] - 1e-6))}


def tank_grid():
    return [tank_run(N, w) for N in (5, 10, 20, 40) for w in (0.01, 0.1, 1.0, 10.0)]


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
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def lab_results():
    iso = br_iso(); shoot = {N: br_shoot(N) for N in (1, 2, 4, 5, 10, 20)}
    tr = {N: br_sim(shoot[N]["T"], traj=True) for N in (1, 20)}
    return {"br": BR, "iso": iso, "shoot": {str(k): v for k, v in shoot.items()}, "traj": {str(k): v for k, v in tr.items()},
            "robust": br_robust(), "tank": TANK, "mpc_grid": tank_grid(), "mpc_noest": tank_run(20, 1.0, est=False), "mpc_open": tank_run(20, 1.0, openloop=True)}


def colloc_results():
    """Collocation runs for the slides and the lab (CasADi); None where CasADi is not installed (the browser)."""
    try:
        import casadi
    except ImportError:
        return None
    runs = {str(n): br_colloc(n) for n in (10, 25, 50, 100)}
    return {"casadi": casadi.__version__, "runs": runs, "Tmax370": br_colloc(50, Thi=370.0), "k2f11": br_colloc(50, k2f=1.1), "log": br_colloc_log()}


def br_colloc_log():
    """Ipopt's printed log for the 50-element collocation run (Lab 11, notebook Step 3), without the banner."""
    import os, subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, "-c", "import dyn_lab as m; m.br_colloc(50, 3, print_level=5)"], cwd=here, capture_output=True, text=True)
    lines = r.stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("This is Ipopt")), 0)
    return "\n".join(lines[start:]).strip()


def classroom_results():
    import scipy
    R = {"meta": {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}}
    R["lecture"] = {"decay": {f"{K}-{n}": colloc_decay(K=K, NE=n) for K in (1, 2, 3) for n in (1, 2, 4, 8, 16)},
                    "pattern": colloc_pattern(4, 3), "colloc": colloc_results(),
                    "mpc_first": mpc_solve([0.6, 1.2], 0.3, [1.6] * 20, 0.0, 20, 1.0).tolist()}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    if "--json" in sys.argv:
        R = classroom_results()
        with open("dyn-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote dyn-results.json", R["meta"]); sys.exit()
    iso = br_iso(); print("A iso", r6(iso["Tbest"]), r6(iso["x2best"]))
    for N in (1, 5, 20):
        s = br_shoot(N); print("A shoot", N, r6(s["x2"]), s["nit"], s["nfev"], r6(s["T"][:3]))
    rb = br_robust(); print("A robust", {k: r6(v) for k, v in rb.items() if not k.startswith("T")})
    for K in (1, 2, 3):
        print("B decay K", K, [sig(colloc_decay(K=K, NE=n)["err"], 3) for n in (1, 2, 4, 8, 16)])
    p = colloc_pattern(); print("B pattern", p["rows"], p["cols"], len(p["nz"]), r6(p["density"]))
    for q in (tank_run(20, 1.0, est=False), tank_run(20, 1.0), tank_run(20, 1.0, openloop=True), tank_run(5, 10.0)):
        print("C tank", q["N"], q["w"], q["est"], q["openloop"], r6(q["overshoot"]), q["settle"], r6(q["offset"]), r6(q["dist_dev"]), q["sat"])
