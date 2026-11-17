"""2105623 Session 12: optimization under uncertainty (2027 draft).

Part A  Capacity under uncertain demand: a newsvendor-type two-stage problem, EV, RP, EEV, WS, EVPI and VSS
Part B  Sample average approximation, risk (CVaR), a chance constraint on a tank level (the two tanks of Session 11),
        and a batch temperature profile for uncertain k2 (the reactor of Session 11)
Part C  Robust optimization of a fuel blend with uncertain sulfur contents (budget of uncertainty)
Lab 12  Dedicated and flexible capacity under correlated demand; risk; a blend with one uncertain coefficient

Runs in desktop Python and in the browser (NumPy and SciPy; the LPs use HiGHS through scipy.optimize.linprog).
Notation: scenarios s with probabilities p_s; risk weight beta; chance level epsilon; budget of uncertainty Gamma.
"""
import json, math, sys
import numpy as np
from scipy.optimize import linprog, minimize, lsq_linear
from scipy.linalg import expm
from scipy import stats


def r6(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(u) for u in v]
    v = round(float(v), 6)
    return 0.0 if v == 0 else v


def sig(v, d=4):
    return f"{float(v):.{d}g}"


# ---------------------------------------------------------------------------
# Part A. One product, capacity K built now (USD 225 per t/d per day), demand d known later.
#   Own production y <= min(K, d) earns 400 USD/t; demand above K is met by a toll manufacturer at 100 USD/t.
#   profit(K, d) = 100 d + 300 min(K, d) - 225 K
# ---------------------------------------------------------------------------
NV = {"m": 400.0, "mt": 100.0, "c": 225.0, "d": [40.0, 100.0, 180.0], "p": [0.3, 0.5, 0.2]}


def nv_profit(K, d):
    return NV["mt"] * d + (NV["m"] - NV["mt"]) * min(K, d) - NV["c"] * K


def nv_lp(d, p, fixK=None):
    """Extensive form: variables K, then y_s (own production) and z_s (toll) for each scenario; maximize
    -c K + sum_s p_s (m y_s + mt z_s), s.t. y_s <= K, y_s + z_s <= d_s."""
    n = len(d); nv = 1 + 2 * n; c = np.zeros(nv); c[0] = NV["c"]; A = []; b = []
    for s in range(n):
        c[1 + 2 * s] = -p[s] * NV["m"]; c[2 + 2 * s] = -p[s] * NV["mt"]
        r = np.zeros(nv); r[1 + 2 * s] = 1; r[0] = -1; A.append(r); b.append(0.0)
        r = np.zeros(nv); r[1 + 2 * s] = 1; r[2 + 2 * s] = 1; A.append(r); b.append(d[s])
    bounds = [(0, None)] * nv
    if fixK is not None:
        bounds[0] = (fixK, fixK)
    r = linprog(c, A_ub=np.array(A), b_ub=b, bounds=bounds, method="highs")
    return float(r.x[0]), float(-r.fun)


def nv_measures():
    d, p = NV["d"], NV["p"]; dm = float(np.dot(d, p))
    Kev, EVf = nv_lp([dm], [1.0]); K, RP = nv_lp(d, p); _, EEV = nv_lp(d, p, fixK=Kev)
    WS = float(sum(p[s] * nv_lp([d[s]], [1.0])[1] for s in range(len(d))))
    curve = [[k, float(sum(p[s] * nv_profit(k, d[s]) for s in range(len(d))))] for k in np.arange(0, 201, 2.0)]
    return {"dmean": dm, "K_EV": Kev, "EV": EVf, "K_RP": K, "RP": RP, "EEV": EEV, "WS": WS, "EVPI": WS - RP, "VSS": RP - EEV,
            "curve": curve, "by_scenario": [[nv_profit(Kev, x) for x in d], [nv_profit(K, x) for x in d]]}


# ---------------------------------------------------------------------------
# Part B1. Sample average approximation with a lognormal demand (median 90 t/d, sigma of ln d = 0.5)
# ---------------------------------------------------------------------------
LOGN = {"median": 90.0, "s": 0.5}


def true_K():
    """The exact optimum: the newsvendor quantile, P(d <= K*) = 1 - c/(m - mt) = 0.25."""
    q = 1 - NV["c"] / (NV["m"] - NV["mt"])
    return float(LOGN["median"] * math.exp(LOGN["s"] * stats.norm.ppf(q)))


def true_profit(K, n=200001):
    """Expected profit for capacity K under the lognormal demand, by numerical integration on a fine quantile grid."""
    u = (np.arange(n) + 0.5) / n; d = LOGN["median"] * np.exp(LOGN["s"] * stats.norm.ppf(u))
    return float(np.mean(NV["mt"] * d + (NV["m"] - NV["mt"]) * np.minimum(K, d) - NV["c"] * K))


def saa(N, reps=20, seed=12):
    """Solve the SAA extensive form for reps independent samples of size N; return K, the SAA objective
    (in-sample) and the true expected profit of each K (out of sample)."""
    rng = np.random.default_rng(seed + N); out = []
    for _ in range(reps):
        d = LOGN["median"] * np.exp(LOGN["s"] * rng.standard_normal(N))
        K, obj = nv_lp(d.tolist(), [1.0 / N] * N)
        out.append({"K": K, "saa": obj, "true": true_profit(K, 20001)})
    return out


def saa_bias(N, reps=20000, seed=1):
    """Mean SAA optimal value over many samples, with the newsvendor shortcut: the SAA optimum is at a sample point,
    so the best of the N sample points is the SAA optimal value (the same value the LP gives). In expectation this
    value is above the true optimum for a maximization (Mak, Morton and Wood, 1999)."""
    rng = np.random.default_rng(seed + N); out = np.empty(reps); m, mt, c = NV["m"], NV["mt"], NV["c"]
    for r0 in range(0, reps, 500):
        d = np.sort(LOGN["median"] * np.exp(LOGN["s"] * rng.standard_normal((min(500, reps - r0), N))), axis=1)
        cs = np.cumsum(d, axis=1); j = np.arange(1, N + 1)
        val = (mt * cs[:, -1:] + (m - mt) * (cs + (N - j) * d)) / N - c * d
        out[r0:r0 + d.shape[0]] = val.max(axis=1)
    return {"N": N, "mean": float(out.mean()), "se": float(out.std(ddof=1) / math.sqrt(reps))}


def profit_quantile(K, q=0.1):
    """The q-quantile of profit for capacity K under the lognormal demand (profit increases with d)."""
    d = LOGN["median"] * math.exp(LOGN["s"] * stats.norm.ppf(q))
    return float(NV["mt"] * d + (NV["m"] - NV["mt"]) * min(K, d) - NV["c"] * K)


def saa_study():
    Ks = true_K(); v = true_profit(Ks); Kev = LOGN["median"] * math.exp(LOGN["s"] ** 2 / 2)
    return {"K_true": Ks, "obj_true": v, "runs": {str(N): saa(N) for N in (5, 20, 100, 500)},
            "bias": {str(N): saa_bias(N, 20000 if N < 500 else 5000) for N in (5, 20, 100, 500)},
            "report": {"K_mean": Kev, "E_mean": true_profit(Kev), "q10_opt": profit_quantile(Ks), "q10_mean": profit_quantile(Kev)}}


# ---------------------------------------------------------------------------
# Part B2. Risk: maximize (1 - beta) E[profit] + beta CVaR (the mean of the worst 10% of outcomes)
#   CVaR_10% = max_eta { eta - (1/0.1) sum_s p_s max(0, eta - profit_s) }   (Rockafellar and Uryasev, 2000)
# ---------------------------------------------------------------------------
def nv_cvar(beta, d, alpha=0.1):
    """Variables: K, y_s, z_s, eta, u_s >= eta - profit_s. Returns K, E[profit], CVaR."""
    n = len(d); p = 1.0 / n; nv = 1 + 2 * n + 1 + n; c = np.zeros(nv); A = []; b = []
    ie = 1 + 2 * n; iu = ie + 1
    c[0] += (1 - beta) * NV["c"]; c[ie] = -beta
    for s in range(n):
        c[1 + 2 * s] -= (1 - beta) * p * NV["m"]; c[2 + 2 * s] -= (1 - beta) * p * NV["mt"]; c[iu + s] = beta * p / alpha
        r = np.zeros(nv); r[1 + 2 * s] = 1; r[0] = -1; A.append(r); b.append(0.0)
        r = np.zeros(nv); r[1 + 2 * s] = 1; r[2 + 2 * s] = 1; A.append(r); b.append(d[s])
        r = np.zeros(nv); r[ie] = 1; r[iu + s] = -1; r[1 + 2 * s] = -NV["m"]; r[2 + 2 * s] = -NV["mt"]; r[0] = NV["c"]; A.append(r); b.append(0.0)
    bounds = [(0, None)] * nv; bounds[ie] = (None, None)
    r = linprog(c, A_ub=np.array(A), b_ub=b, bounds=bounds, method="highs")
    K = float(r.x[0]); prof = np.array([nv_profit(K, x) for x in d])
    worst = np.sort(prof)[:int(round(alpha * n))]
    return {"beta": beta, "K": K, "E": float(prof.mean()), "CVaR": float(worst.mean())}


def risk_study(n=200, seed=7):
    rng = np.random.default_rng(seed); d = (LOGN["median"] * np.exp(LOGN["s"] * rng.standard_normal(n))).tolist()
    return {"betas": [nv_cvar(b, d) for b in np.round(np.arange(0, 1.0001, 0.1), 2)], "d": d}


# ---------------------------------------------------------------------------
# Part B3. A chance constraint: the two tanks of Session 11 under MPC (N = 20, w = 1, disturbance estimate on).
#   A withdrawal of unknown size d ~ N(0.05, 0.015^2), truncated to [0, 0.1] m3/min, starts without warning.
#   Choose the setpoint r as low as possible such that P(h2 stays above 1.45 m) >= 1 - epsilon.
# ---------------------------------------------------------------------------
TANK = {"A1": 1.0, "A2": 1.0, "R1": 2.0, "R2": 4.0, "dt": 0.5, "umax": 0.5, "N": 20, "w": 1.0, "hmin": 1.45,
        "d_mean": 0.05, "d_sd": 0.015, "steps": 20}


def tank_model():
    A1, A2, R1, R2 = TANK["A1"], TANK["A2"], TANK["R1"], TANK["R2"]
    Ac = np.array([[-1 / (A1 * R1), 0], [1 / (A2 * R1), -1 / (A2 * R2)]]); Bc = np.array([[1 / A1, 0], [0, 1 / A2]])
    M = expm(np.block([[Ac, Bc], [np.zeros((2, 4))]]) * TANK["dt"])
    return M[:2, :2], M[:2, 2], M[:2, 3]


def tank_pred(N):
    Ad, Bd, Ed = tank_model(); P = [np.linalg.matrix_power(Ad, k) for k in range(N + 1)]
    Phi = np.array([P[i + 1][1] for i in range(N)]); G = np.zeros((N, N)); Gd = np.zeros(N)
    for i in range(N):
        for j in range(i + 1):
            G[i, j] = (P[i - j] @ Bd)[1]
        Gd[i] = sum((P[i - j] @ Ed)[1] for j in range(i + 1))
    return Phi, G, Gd


def tank_dip(r, d, P=None, model=None):
    """Closed loop at steady state with setpoint r when a withdrawal d starts; MPC with the disturbance estimate.
    Returns the lowest h2 over the next 10 min (the lowest point comes 1 to 2 min after the start)."""
    Ad, Bd, Ed = tank_model() if model is None else model; N, w = TANK["N"], TANK["w"]
    Phi, G, Gd = tank_pred(N) if P is None else P
    D = np.eye(N) - np.eye(N, k=-1); A = np.vstack([G, math.sqrt(w) * D])
    u = r / TANK["R2"]; x = np.array([r * TANK["R1"] / TANK["R2"], r]); dh = 0.0; low = r
    for k in range(TANK["steps"]):
        e = np.zeros(N); e[0] = u
        u = lsq_linear(A, np.concatenate([r - Phi @ x - Gd * dh, math.sqrt(w) * e]), bounds=(0, TANK["umax"]), method="bvls").x[0]
        xp = Ad @ x + Bd * u + Ed * dh; x = Ad @ x + Bd * u - Ed * d; dh += (x[1] - xp[1]) / Ed[1]; low = min(low, x[1])
    return float(low)


def tank_chance(n=200, seed=3, r_ref=1.55):
    """Dip below the setpoint for n sampled withdrawals (simulated at r_ref), then the lowest setpoint for each
    chance level, and checks of the chosen setpoints by simulating again at those setpoints."""
    rng = np.random.default_rng(seed); P = tank_pred(TANK["N"]); M = tank_model()
    d = np.clip(TANK["d_mean"] + TANK["d_sd"] * rng.standard_normal(n), 0.0, 0.1)
    dips = np.array([r_ref - tank_dip(r_ref, x, P, M) for x in d])
    eps = [0.2, 0.1, 0.05, 0.01]
    r_eps = {str(e): float(TANK["hmin"] + np.quantile(dips, 1 - e, method="higher")) for e in eps}
    r_worst = float(TANK["hmin"] + (r_ref - tank_dip(r_ref, 0.1, P, M)))
    d_new = np.clip(TANK["d_mean"] + TANK["d_sd"] * np.random.default_rng(seed + 100).standard_normal(n), 0.0, 0.1)
    r_mean = float(TANK["hmin"] + (r_ref - tank_dip(r_ref, TANK["d_mean"], P, M))); check = {}
    for key, r in list(r_eps.items()) + [("worst", r_worst), ("mean", r_mean)]:
        lows = np.array([tank_dip(r, x, P, M) for x in d_new]); check[key] = float(np.mean(lows < TANK["hmin"] - 1e-9))
    return {"d": d.tolist(), "dips": dips.tolist(), "r": r_eps, "r_worst": r_worst, "r_mean": r_mean, "violation_new": check,
            "dip_of_d": [[x, r_ref - tank_dip(r_ref, x, P, M)] for x in np.round(np.arange(0, 0.1001, 0.01), 2)]}


# ---------------------------------------------------------------------------
# Part B4. The batch reactor of Session 11 when k2 is uncertain: ln(k2 factor) ~ N(0, 0.1^2), 7 equally likely
#   scenarios at the midpoints of 7 probability bins. Maximize E[x2(1 h)] with 10 temperatures.
# ---------------------------------------------------------------------------
BR = {"h": 0.005, "k10": 4000.0, "E1": 2500.0, "k20": 620000.0, "E2": 5000.0}
K2F = [float(math.exp(0.1 * stats.norm.ppf((i + 0.5) / 7))) for i in range(7)]


def br_sim(Ts, k2f=1.0):
    n = int(round(1 / len(Ts) / BR["h"])); h = 1 / len(Ts) / n; a, b = 1.0, 0.0
    for T in Ts:
        k1 = BR["k10"] * math.exp(-BR["E1"] / T); k2 = k2f * BR["k20"] * math.exp(-BR["E2"] / T)
        for _ in range(n):
            f1a = -k1 * a * a; f1b = k1 * a * a - k2 * b
            a2 = a + h / 2 * f1a; b2 = b + h / 2 * f1b; f2a = -k1 * a2 * a2; f2b = k1 * a2 * a2 - k2 * b2
            a3 = a + h / 2 * f2a; b3 = b + h / 2 * f2b; f3a = -k1 * a3 * a3; f3b = k1 * a3 * a3 - k2 * b3
            a4 = a + h * f3a; b4 = b + h * f3b; f4a = -k1 * a4 * a4; f4b = k1 * a4 * a4 - k2 * b4
            a += h / 6 * (f1a + 2 * f2a + 2 * f3a + f4a); b += h / 6 * (f1b + 2 * f2b + 2 * f3b + f4b)
    return b


def br_opt(k2fs, N=10, worst=False):
    """Maximize the mean (or, with worst=True, the smallest) x2(1 h) over the k2 scenarios; T = 298 + 100 u."""
    if not worst:
        r = minimize(lambda u: -np.mean([br_sim(298 + 100 * u, f) for f in k2fs]), np.full(N, 0.4), method="SLSQP",
                     bounds=[(0, 1)] * N, options={"ftol": 1e-12, "maxiter": 300})
        u = r.x
    else:
        z0 = np.concatenate([np.full(N, 0.4), [0.5]])
        cons = [{"type": "ineq", "fun": (lambda z, f=f: br_sim(298 + 100 * z[:N], f) - z[N])} for f in k2fs]
        r = minimize(lambda z: -z[N], z0, method="SLSQP", bounds=[(0, 1)] * N + [(0, 1)], constraints=cons, options={"ftol": 1e-12, "maxiter": 300})
        u = r.x[:N]
    T = 298 + 100 * u; x = [br_sim(T, f) for f in k2fs]
    return {"T": T.tolist(), "x2": x, "mean": float(np.mean(x)), "min": float(np.min(x)), "nit": int(r.nit)}


def br_study():
    nom = br_opt([1.0]); sto = br_opt(K2F); rob = br_opt(K2F, worst=True)
    xn = [br_sim(nom["T"], f) for f in K2F]
    return {"k2f": K2F, "nominal": {"T": nom["T"], "x2": xn, "mean": float(np.mean(xn)), "min": float(np.min(xn)), "predicted": nom["x2"][0]},
            "stochastic": sto, "robust": rob}


# ---------------------------------------------------------------------------
# Part C. A fuel blend with uncertain sulfur (lecture): 100 t of blend, sulfur <= 0.50 wt%, viscosity index <= 30,
#   5 components with nominal sulfur s_i and a possible deviation delta_i. Budget of uncertainty Gamma
#   (Bertsimas and Sim, 2004): at most Gamma of the sulfur contents reach their worst values.
# ---------------------------------------------------------------------------
BLEND = {"names": ["high-S residue", "low-S residue", "cracked cycle oil", "gasoil", "hydrotreated cutter"],
         "cost": [360.0, 455.0, 420.0, 610.0, 690.0], "s": [2.40, 0.40, 1.10, 0.10, 0.02], "dev": [0.25, 0.15, 0.20, 0.03, 0.01],
         "vbi": [42.0, 38.0, 22.0, 15.0, 12.0], "avail": [40.0, 45.0, 30.0, 100.0, 100.0], "total": 100.0, "smax": 0.50, "vmax": 30.0}


def blend_robust(Gamma, B=None):
    """min cost s.t. sum x = total, sum vbi x <= vmax total, sum s x + Gamma z + sum q <= smax total,
    z + q_i >= dev_i x_i, z, q >= 0 (the robust counterpart of the sulfur row for budget Gamma)."""
    B = BLEND if B is None else B; n = len(B["cost"]); nv = 2 * n + 1; c = np.zeros(nv); c[:n] = B["cost"]
    A = []; b = []
    A.append(np.r_[B["vbi"], np.zeros(n + 1)] - np.r_[np.full(n, B["vmax"]), np.zeros(n + 1)]); b.append(0.0)
    A.append(np.r_[np.array(B["s"]) - B["smax"], Gamma, np.ones(n)]); b.append(0.0)
    for i in range(n):
        r = np.zeros(nv); r[i] = B["dev"][i]; r[n] = -1; r[n + 1 + i] = -1; A.append(r); b.append(0.0)
    Aeq = [np.r_[np.ones(n), np.zeros(n + 1)]]
    bounds = [(0, B["avail"][i]) for i in range(n)] + [(0, None)] * (n + 1)
    r = linprog(c, A_ub=np.array(A), b_ub=b, A_eq=np.array(Aeq), b_eq=[B["total"]], bounds=bounds, method="highs")
    return {"Gamma": Gamma, "x": r.x[:n].tolist(), "cost": float(r.fun), "status": int(r.status)}


def blend_mc(x, n=20000, seed=5, B=None):
    """Probability that the blend breaks the sulfur limit when each s_i is uniform on s_i +- dev_i, independently."""
    B = BLEND if B is None else B; rng = np.random.default_rng(seed); x = np.asarray(x)
    S = np.array(B["s"]) + np.array(B["dev"]) * rng.uniform(-1, 1, (n, len(x)))
    return float(np.mean(S @ x > B["smax"] * B["total"] + 1e-9))


def blend_study():
    out = []
    for G in np.round(np.arange(0, 5.0001, 0.25), 2):
        q = blend_robust(float(G)); q["viol"] = blend_mc(q["x"]); out.append(q)
    return out


# ---------------------------------------------------------------------------
# Lab 12A. Dedicated line for product A (USD 170 per t/d per day) and a flexible line for A or B (USD 200);
#   B takes 1.25 t/d of flexible capacity per t/d; margins 300 (A) and 450 (B) USD/t; 9 demand scenarios with
#   negatively correlated demands (a market that shifts between A and B).
# ---------------------------------------------------------------------------
LAB = {"cA": 170.0, "cF": 200.0, "mA": 300.0, "mB": 450.0, "rB": 1.25,
       "scen": [[50, 70, 0.15], [50, 40, 0.10], [50, 10, 0.05], [90, 70, 0.10], [90, 40, 0.20], [90, 10, 0.10],
                [130, 70, 0.05], [130, 40, 0.10], [130, 10, 0.15]]}


def lab_lp(S, fixK=None, beta=0.0, alpha=0.1):
    """Extensive form with an optional CVaR term: maximize (1 - beta) E[profit] + beta CVaR_alpha.
    Variables: KA, KF, per scenario aD, aF, b, u_s; and eta."""
    L = LAB; n = len(S); nv = 2 + 4 * n + 1; ie = nv - 1; c = np.zeros(nv); A = []; b = []
    c[0] = (1 - beta) * L["cA"]; c[1] = (1 - beta) * L["cF"]; c[ie] = -beta
    for s, (dA, dB, p) in enumerate(S):
        j = 2 + 4 * s
        c[j] -= (1 - beta) * p * L["mA"]; c[j + 1] -= (1 - beta) * p * L["mA"]; c[j + 2] -= (1 - beta) * p * L["mB"]; c[j + 3] = beta * p / alpha
        for row, rhs in (({j: 1, 0: -1}, 0.0), ({j + 1: 1, j + 2: L["rB"], 1: -1}, 0.0), ({j: 1, j + 1: 1}, dA), ({j + 2: 1}, dB),
                         ({ie: 1, j + 3: -1, j: -L["mA"], j + 1: -L["mA"], j + 2: -L["mB"], 0: L["cA"], 1: L["cF"]}, 0.0)):
            r = np.zeros(nv)
            for k, v in row.items():
                r[k] = v
            A.append(r); b.append(rhs)
    bounds = [(0, None)] * nv; bounds[ie] = (None, None)
    if fixK is not None:
        bounds[0] = (fixK[0], fixK[0]); bounds[1] = (fixK[1], fixK[1])
    r = linprog(c, A_ub=np.array(A), b_ub=b, bounds=bounds, method="highs")
    K = r.x[:2]; prof = []
    for s, (dA, dB, p) in enumerate(S):
        j = 2 + 4 * s; prof.append(L["mA"] * (r.x[j] + r.x[j + 1]) + L["mB"] * r.x[j + 2] - L["cA"] * K[0] - L["cF"] * K[1])
    return {"K": K.tolist(), "profit": prof, "E": float(np.dot(prof, [p for _, _, p in S]))}


def lab_profit(K, dA, dB):
    """Second stage of Lab 12A in closed form: the dedicated line makes A; the flexible line makes B first
    (450 / 1.25 = 360 USD per t/d of flexible capacity, more than the 300 of A), then A."""
    L = LAB; aD = min(K[0], dA); b = min(dB, K[1] / L["rB"]); aF = min(dA - aD, K[1] - L["rB"] * b)
    return L["mA"] * (aD + aF) + L["mB"] * b - L["cA"] * K[0] - L["cF"] * K[1]


def cvar_of(prof, p, alpha=0.1):
    """Mean of the worst alpha probability mass of a discrete distribution."""
    order = np.argsort(prof); left = alpha; tot = 0.0
    for i in order:
        take = min(p[i], left); tot += take * prof[i]; left -= take
        if left <= 1e-12:
            break
    return tot / alpha


def lab_results():
    S = LAB["scen"]; p = [q[2] for q in S]
    rp = lab_lp(S); dA = float(np.dot([q[0] for q in S], p)); dB = float(np.dot([q[1] for q in S], p))
    ev = lab_lp([[dA, dB, 1.0]]); eev = lab_lp(S, fixK=ev["K"])
    ws = [lab_lp([[a, b_, 1.0]]) for a, b_, _ in S]; WS = float(np.dot([w["E"] for w in ws], p))
    risk = []
    for beta in np.round(np.arange(0, 1.0001, 0.1), 2):
        q = lab_lp(S, beta=float(beta)); q["beta"] = float(beta); q["CVaR"] = cvar_of(q["profit"], p); risk.append(q)
    for q in (rp, ev, eev):
        q["CVaR"] = cvar_of(q["profit"], p) if len(q["profit"]) == len(p) else None
    return {"data": LAB, "mean": [dA, dB], "RP": rp, "EV": ev, "EEV": eev, "WS": WS, "WS_K": [w["K"] for w in ws],
            "VSS": rp["E"] - eev["E"], "EVPI": WS - rp["E"], "risk": risk, "blend": lab_blend()}


# ---------------------------------------------------------------------------
# Lab 12B. A low-sulfur fuel oil blend: 4 components, sulfur <= 0.50 wt% (the IMO 2020 global limit for marine
#   fuel), viscosity index <= 34. The sulfur of the low-S residue cargo is uncertain: nominal 0.30 wt%,
#   assay standard deviation 0.04 wt%; at most 0.42 wt% (3 standard deviations).
# ---------------------------------------------------------------------------
LBL = {"names": ["high-S residue", "low-S residue", "marine gasoil", "hydrotreated cutter"], "cost": [380.0, 470.0, 650.0, 700.0],
       "s": [2.50, 0.30, 0.08, 0.01], "vbi": [40.0, 38.0, 20.0, 18.0], "avail": [100.0, 70.0, 100.0, 100.0], "total": 100.0,
       "smax": 0.50, "vmax": 34.0, "k": 1, "sd": 0.04, "smax_k": 0.42}


def lbl_lp(s_k):
    """min cost s.t. sum x = 100, sum vbi x <= 34 * 100, sum s x <= 0.5 * 100, with the sulfur of component k = s_k."""
    L = LBL; s = list(L["s"]); s[L["k"]] = s_k; n = len(s)
    A = [np.array(L["vbi"]) - L["vmax"], np.array(s) - L["smax"]]
    r = linprog(L["cost"], A_ub=np.array(A), b_ub=[0.0, 0.0], A_eq=[np.ones(n)], b_eq=[L["total"]],
                bounds=[(0, a) for a in L["avail"]], method="highs")
    return {"s_k": s_k, "x": r.x.tolist(), "cost": float(r.fun)}


def lbl_mc(x, n=20000, seed=8):
    L = LBL; rng = np.random.default_rng(seed); s = np.array(L["s"]); x = np.asarray(x)
    sk = np.clip(s[L["k"]] + L["sd"] * rng.standard_normal(n), 0, L["smax_k"])
    S = s @ x - s[L["k"]] * x[L["k"]] + sk * x[L["k"]]
    return float(np.mean(S > L["smax"] * L["total"] + 1e-9))


def lab_blend():
    L = LBL; out = {}
    for name, sk in (("nominal", L["s"][L["k"]]), ("chance95", L["s"][L["k"]] + stats.norm.ppf(0.95) * L["sd"]),
                     ("chance99", L["s"][L["k"]] + stats.norm.ppf(0.99) * L["sd"]), ("robust", L["smax_k"])):
        q = lbl_lp(sk); q["viol"] = lbl_mc(q["x"]); out[name] = q
    sweep = []
    for sk in np.round(np.arange(0.30, 0.4201, 0.01), 2):
        q = lbl_lp(float(sk)); q["viol"] = lbl_mc(q["x"]); sweep.append(q)
    out["sweep"] = sweep
    return out


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
    R["lecture"] = {"nv": nv_measures(), "saa": saa_study(), "risk": risk_study(), "tank": tank_chance(), "batch": br_study(),
                    "blend": blend_study(), "blend_data": BLEND}
    R["lab"] = lab_results()
    return clean(R)


if __name__ == "__main__":
    if "--json" in sys.argv:
        R = classroom_results()
        with open("unc-results.json", "w") as fh:
            json.dump(R, fh)
        print("wrote unc-results.json", R["meta"]); sys.exit()
    m = nv_measures(); print("A", {k: r6(v) for k, v in m.items() if k not in ("curve", "by_scenario")})
    s = saa_study(); print("B saa K*", r6(s["K_true"]), r6(s["obj_true"]))
    for N, runs in s["runs"].items():
        K = [q["K"] for q in runs]; print("  N", N, "K mean sd", r6(np.mean(K)), r6(np.std(K)), "saa mean", r6(np.mean([q["saa"] for q in runs])), "true mean", r6(np.mean([q["true"] for q in runs])))
    rk = risk_study(); print("B risk", [(q["beta"], r6(q["K"]), round(q["E"]), round(q["CVaR"])) for q in rk["betas"]])
    t = tank_chance(); print("B tank", {k: r6(v) for k, v in t["r"].items()}, "worst", r6(t["r_worst"]), "mean", r6(t["r_mean"]), "viol", t["violation_new"])
    print("  dip(d)", [(q[0], r6(q[1])) for q in t["dip_of_d"]])
    print("B saa bias", {k: (round(v["mean"], 1), round(v["se"], 1)) for k, v in s["bias"].items()})
    b = br_study(); print("B batch nominal", r6(b["nominal"]["mean"]), r6(b["nominal"]["min"]), "stoch", r6(b["stochastic"]["mean"]), r6(b["stochastic"]["min"]), "robust", r6(b["robust"]["mean"]), r6(b["robust"]["min"]))
    for q in blend_study()[::4]:
        print("C blend", q["Gamma"], r6(q["x"]), round(q["cost"], 1), q["viol"])
    L = lab_results(); print("Lab RP", r6(L["RP"]["K"]), r6(L["RP"]["E"]), "EV", r6(L["EV"]["K"]), r6(L["EV"]["E"]), "EEV", r6(L["EEV"]["E"]), "WS", r6(L["WS"]), "VSS", r6(L["VSS"]), "EVPI", r6(L["EVPI"]))
    print("Lab risk", [(q["beta"], r6(q["K"]), round(q["E"]), round(q["CVaR"])) for q in L["risk"]])
    for k in ("nominal", "chance95", "chance99", "robust"):
        q = L["blend"][k]; print("Lab blend", k, r6(q["s_k"]), r6(q["x"]), round(q["cost"], 2), q["viol"])
