"""2105623 Session 4: multiperiod models (inventory, storage, the end of the horizon).

Part A  Food Manufacture: six-month oil blending with storage
        (data from H. P. Williams, Model Building in Mathematical Programming,
        5th ed., Section 12.1).
Part B  A battery beside rooftop PV under the PEA time-of-use tariff for
        medium general service, Schedule 3.2.3 (voltage below 22 kV): energy
        4.3297 THB/kWh on-peak, 2.6369 off-peak, demand charge 210 THB/kW per
        month on the highest on-peak demand; Ft and VAT not included.
        Load and PV profiles are synthetic teaching data.
          Lecture: a small factory (300 kWp PV, 300 kWh / 150 kW battery).
          Lab 4:   an office building (200 kWp PV, 200 kWh / 100 kW battery).
Part C  Forecast error and re-planning (rolling horizon) for the factory.

Runs in the browser (Pyodide: NumPy, SciPy/HiGHS) and in desktop Python.
    python multiperiod_lab.py          print every part
    python multiperiod_lab.py --json   write multiperiod-results.json for the slides and the lab page
Every result is checked by recomputing balances and the objective from the
solution itself. A solver status of "optimal" is not treated as evidence.
"""
import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# ---------------------------------------------------------------------------
# Part A. Food Manufacture (Williams 12.1)
# ---------------------------------------------------------------------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
OILS = ["VEG1", "VEG2", "OIL1", "OIL2", "OIL3"]
IS_VEG = np.array([1, 1, 0, 0, 0], dtype=bool)
HARDNESS = np.array([8.8, 6.1, 2.0, 4.2, 5.0])
PRICE = np.array([          # GBP per tonne, rows = months, columns = oils
    [110, 120, 130, 110, 115],
    [130, 130, 110, 90, 115],
    [110, 140, 130, 100, 95],
    [120, 110, 120, 120, 125],
    [100, 120, 150, 110, 105],
    [90, 100, 140, 80, 135],
], dtype=float)
SELL = 150.0                # GBP per tonne of product
INITIAL_STOCK = 500.0       # tonnes of each oil at the start
MAX_STOCK = 1000.0          # tonnes of each oil


def food_manufacture(storage_cost=5.0, veg_capacity=200.0, nonveg_capacity=250.0,
                     terminal=True, carryover=True, initial=INITIAL_STOCK):
    """Maximize profit over six months. Returns the plan, profit, checks and the value of oil
    (dual of each stock balance: GBP per extra tonne of oil i available in month t)."""
    nI, nT = len(OILS), len(MONTHS)
    n = 3 * nI * nT                                   # buy, use, stock

    def b(i, t): return i * nT + t
    def u(i, t): return nI * nT + i * nT + t
    def s(i, t): return 2 * nI * nT + i * nT + t

    c = np.zeros(n)                                   # minimize -profit
    for i in range(nI):
        for t in range(nT):
            c[b(i, t)] = PRICE[t, i]
            c[u(i, t)] = -SELL
            c[s(i, t)] = storage_cost

    A_eq, b_eq = [], []                               # stock balance: s(t-1) + buy - use - s(t) = 0
    for i in range(nI):
        for t in range(nT):
            row = np.zeros(n)
            row[b(i, t)], row[u(i, t)], row[s(i, t)] = 1, -1, -1
            if t == 0:
                b_eq.append(-initial)
            else:
                row[s(i, t - 1)] = 1
                b_eq.append(0.0)
            A_eq.append(row)

    A_ub, b_ub, ub_names = [], [], []
    for t in range(nT):
        veg, nonveg, hmax, hmin = (np.zeros(n) for _ in range(4))
        for i in range(nI):
            (veg if IS_VEG[i] else nonveg)[u(i, t)] = 1
            hmax[u(i, t)] = HARDNESS[i] - 6.0         # hardness <= 6, a ratio row as in Session 2
            hmin[u(i, t)] = 3.0 - HARDNESS[i]         # hardness >= 3
        A_ub += [veg, nonveg, hmax, hmin]
        b_ub += [veg_capacity, nonveg_capacity, 0.0, 0.0]
        ub_names += [f"veg_cap_{t}", f"nonveg_cap_{t}", f"hard_max_{t}", f"hard_min_{t}"]

    bounds = [(0, None)] * (2 * nI * nT)
    for i in range(nI):
        for t in range(nT):
            lo, hi = 0.0, MAX_STOCK
            if not carryover:
                lo = hi = initial
            if terminal and t == nT - 1:
                lo = hi = INITIAL_STOCK
            bounds.append((lo, hi))

    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        return {"status": res.message, "profit": None}
    x = res.x
    buy = np.array([[x[b(i, t)] for i in range(nI)] for t in range(nT)])
    use = np.array([[x[u(i, t)] for i in range(nI)] for t in range(nT)])
    stock = np.array([[x[s(i, t)] for i in range(nI)] for t in range(nT)])
    product = use.sum(axis=1)

    prev = np.full(nI, initial)                       # independent audit from the solution only
    balance = []
    for t in range(nT):
        balance.append(np.abs(prev + buy[t] - use[t] - stock[t]).max())
        prev = stock[t]
    quality = (use @ HARDNESS) / np.where(product > 0, product, 1)
    profit = SELL * product.sum() - (PRICE * buy).sum() - storage_cost * stock.sum()
    duals = {name: -m for name, m in zip(ub_names, res.ineqlin.marginals)}
    value = np.array(res.eqlin.marginals).reshape(nI, nT).T   # GBP per extra tonne available, [t, i]
    r6 = lambda a: (np.round(a, 6) + 0.0).tolist()
    return {
        "status": "optimal", "profit": float(profit), "solver_objective": float(-res.fun),
        "buy": r6(buy), "use": r6(use), "stock": r6(stock), "product": r6(product),
        "hardness": np.round(quality, 6).tolist(), "value": r6(value),
        "revenue": float(SELL * product.sum()), "purchase": float((PRICE * buy).sum()),
        "storage": float(storage_cost * stock.sum()), "stock_months": float(stock.sum()),
        "checks": {
            "max_stock_balance_residual_t": float(max(balance)),
            "hardness_within_3_to_6": bool(np.all((quality > 3 - 1e-7) & (quality < 6 + 1e-7))),
            "veg_capacity_ok": bool(np.all(use[:, IS_VEG].sum(axis=1) <= veg_capacity + 1e-7)),
            "nonveg_capacity_ok": bool(np.all(use[:, ~IS_VEG].sum(axis=1) <= nonveg_capacity + 1e-7)),
            "objective_residual_gbp": float(abs(profit - (-res.fun))),
        },
        "capacity_value_gbp_per_t": {
            "veg": [round(duals[f"veg_cap_{t}"], 4) for t in range(nT)],
            "nonveg": [round(duals[f"nonveg_cap_{t}"], 4) for t in range(nT)],
        },
    }


def food_size():
    nI, nT = len(OILS), len(MONTHS)
    return {"variables": 3 * nI * nT, "balance_rows": nI * nT, "monthly_rows": 4 * nT,
            "dof": 3 * nI * nT - nI * nT}


# ---------------------------------------------------------------------------
# Part B. A battery beside rooftop PV under PEA TOU, Schedule 3.2.3
# ---------------------------------------------------------------------------
HOURS = np.arange(24)                                 # hour t covers t:00 to t+1:00
PEAK, OFFPEAK = 4.3297, 2.6369                        # THB/kWh, PEA 3.2.3, without Ft and VAT
DEMAND_MONTH = 210.0                                  # THB per kW per month, highest on-peak demand
WEEKDAYS = 22                                         # the monthly demand charge spread over 22 weekdays
DEMAND_DAY = DEMAND_MONTH / WEEKDAYS                  # THB per kW per weekday
ON_PEAK = (HOURS >= 9) & (HOURS < 22)                 # Monday to Friday, 09:00 to 22:00
TOU_WEEKDAY = np.where(ON_PEAK, PEAK, OFFPEAK)
TOU_WEEKEND = np.full(24, OFFPEAK)                    # Saturday, Sunday and holidays are off-peak all day
CLOUD = np.ones(24)
CLOUD[12:17] = [0.7, 0.35, 0.3, 0.5, 0.8]             # a cloudy afternoon

# lecture site: a small factory, one day shift
LOAD_FACTORY = np.array([70, 70, 70, 70, 70, 80, 150, 190, 210, 215, 220, 220,
                         180, 215, 220, 220, 210, 190, 130, 95, 85, 80, 75, 70], float)
BAT_FACTORY = {"E": 300.0, "P": 150.0, "eta_c": 0.95, "eta_d": 0.95, "soc_min": 30.0, "soc0": 150.0, "wear": 0.05}
KWP_FACTORY = 300.0
# lab site: an office building
LOAD_OFFICE = np.array([55, 55, 55, 55, 55, 58, 65, 85, 120, 130, 135, 140,
                        110, 135, 140, 135, 130, 115, 100, 95, 90, 80, 65, 60], float)
LOAD_OFFICE_SAT = np.array([55, 55, 55, 55, 55, 55, 60, 75, 99, 104, 109, 109,
                            99, 104, 109, 104, 99, 85, 75, 70, 65, 60, 58, 55], float)
BAT_OFFICE = {"E": 200.0, "P": 100.0, "eta_c": 0.95, "eta_d": 0.95, "soc_min": 20.0, "soc0": 100.0, "wear": 0.05}
KWP_OFFICE = 200.0


def pv_profile(kwp, cloud=None):
    """Clear-sky PV output in kW for each hour (synthetic, sunrise 6:00, sunset 18:00)."""
    mid = HOURS + 0.5
    shape = np.clip(np.sin(np.pi * (mid - 6.0) / 12.0), 0, None) ** 1.3
    pv = 0.8 * kwp * shape
    return pv * (CLOUD if cloud is True else (cloud if cloud is not None else 1.0))


def dispatch(load, pv, price, bat, demand=0.0, peak_mask=ON_PEAK, curtail=True, terminal=True,
             binary=False, soc_start=None, soc_end=None, d_floor=0.0):
    """Least-cost hourly dispatch. Variables per hour: import, charge, discharge, curtailment,
    state of charge (kWh at the end of the hour), plus one variable d for the highest on-peak
    import (an epigraph, as in Session 2) when demand > 0, and binaries y when binary=True."""
    load, pv, price = (np.asarray(a, float) for a in (load, pv, price))
    n = len(load)
    mask = np.asarray(peak_mask[-n:] if len(peak_mask) >= n else peak_mask, bool)
    soc_start = bat["soc0"] if soc_start is None else soc_start
    soc_end = bat["soc0"] if soc_end is None else soc_end
    IMP, CH, DIS, CUR, SOC = (k * n for k in range(5))
    D = 5 * n
    Y = D + 1
    nv = Y + (n if binary else 0)
    c = np.zeros(nv)
    c[IMP:IMP + n] = price
    c[DIS:DIS + n] = bat["wear"]
    c[D] = demand
    rows, lo, hi = [], [], []
    for t in range(n):                                # power balance (kW over one hour = kWh)
        r = np.zeros(nv); r[IMP + t], r[DIS + t], r[CH + t], r[CUR + t] = 1, 1, -1, -1
        rows.append(r); lo.append(load[t] - pv[t]); hi.append(load[t] - pv[t])
    for t in range(n):                                # state-of-charge balance
        r = np.zeros(nv); r[SOC + t], r[CH + t], r[DIS + t] = 1, -bat["eta_c"], 1 / bat["eta_d"]
        if t > 0:
            r[SOC + t - 1] = -1
        rhs = soc_start if t == 0 else 0.0
        rows.append(r); lo.append(rhs); hi.append(rhs)
    n_eq = len(rows)
    if demand > 0:                                    # d >= import(t) in every on-peak hour
        for t in range(n):
            if mask[t]:
                r = np.zeros(nv); r[IMP + t], r[D] = 1, -1
                rows.append(r); lo.append(-np.inf); hi.append(0.0)
    if binary:                                        # y = 1 charging allowed, y = 0 discharging allowed
        for t in range(n):
            r = np.zeros(nv); r[CH + t], r[Y + t] = 1, -bat["P"]; rows.append(r); lo.append(-np.inf); hi.append(0.0)
            r = np.zeros(nv); r[DIS + t], r[Y + t] = 1, bat["P"]; rows.append(r); lo.append(-np.inf); hi.append(bat["P"])
    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    ub[CH:CH + n] = bat["P"]; ub[DIS:DIS + n] = bat["P"]
    ub[CUR:CUR + n] = pv if curtail else 0.0
    lb[SOC:SOC + n], ub[SOC:SOC + n] = bat["soc_min"], bat["E"]
    if terminal:
        lb[SOC + n - 1] = max(soc_end, bat["soc_min"])
    lb[D] = d_floor if demand > 0 else 0.0
    ub[D] = np.inf if demand > 0 else 0.0
    if binary:
        ub[Y:Y + n] = 1.0
    A = np.array(rows)
    if binary:
        integ = np.zeros(nv); integ[Y:Y + n] = 1
        res = milp(c, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lb, ub), integrality=integ)
        ok, x, marg = res.status == 0, res.x, None
    else:
        A_ub = A[n_eq:] if len(rows) > n_eq else None
        res = linprog(c, A_eq=A[:n_eq], b_eq=np.array(lo[:n_eq]), A_ub=A_ub,
                      b_ub=np.array(hi[n_eq:]) if A_ub is not None else None,
                      bounds=list(zip(lb, ub)), method="highs")
        ok, x = res.status == 0, res.x
        marg = res.eqlin.marginals[:n] if ok else None
    if not ok:
        return {"status": "infeasible" if res.status == 2 else str(res.message), "cost": None}
    g = {k: x[o:o + n] for k, o in (("imp", IMP), ("ch", CH), ("dis", DIS), ("cur", CUR), ("soc", SOC))}
    soc_prev = np.concatenate([[soc_start], g["soc"][:-1]])
    power_res = g["imp"] + pv - g["cur"] + g["dis"] - load - g["ch"]
    soc_res = soc_prev + bat["eta_c"] * g["ch"] - g["dis"] / bat["eta_d"] - g["soc"]
    peak_imp = float(g["imp"][mask].max()) if mask.any() else 0.0
    energy = float(price @ g["imp"]); wear = float(bat["wear"] * g["dis"].sum())
    dcharge = float(demand * max(peak_imp, d_floor)) if demand > 0 else 0.0
    cost = energy + wear + dcharge
    out = {k: (np.round(v, 6) + 0.0).tolist() for k, v in g.items()}
    out.update({"status": "optimal", "cost": cost, "energy_cost": energy, "wear_cost": wear, "demand_cost": dcharge,
                "peak_import": peak_imp, "simultaneous_kw": float(np.minimum(g["ch"], g["dis"]).max()),
                "hour_of_max_simultaneous": int(np.argmax(np.minimum(g["ch"], g["dis"]))),
                "curtailed_kwh": float(g["cur"].sum()),
                "checks": {"max_power_balance_residual_kw": float(np.abs(power_res).max()),
                           "max_soc_balance_residual_kwh": float(np.abs(soc_res).max()),
                           "objective_residual_thb": float(abs(cost - res.fun)), "soc_end_kwh": float(g["soc"][-1])},
                "marginal_thb_per_kwh": None if marg is None else np.round(marg, 4).tolist()})
    return out


def no_battery(load, pv, price, demand=0.0, peak_mask=ON_PEAK):
    """Cost with PV but no battery; surplus PV is curtailed."""
    imp = np.clip(np.asarray(load) - np.asarray(pv), 0, None)
    peak = float(imp[peak_mask].max()) if demand > 0 else 0.0
    energy = float(np.asarray(price) @ imp)
    return {"cost": energy + demand * peak, "energy_cost": energy, "demand_cost": demand * peak,
            "peak_import": float(imp[peak_mask].max()), "imp": (np.round(imp, 6) + 0.0).tolist()}


def soc_by_hand():
    """The four-hour example on the slides, worked from the SoC balance."""
    b = BAT_FACTORY; soc = 150.0; rows = []
    for ch, dis in ((100, 0), (0, 0), (0, 150), (0, 50)):
        new = soc + b["eta_c"] * ch - dis / b["eta_d"]
        rows.append({"start": soc, "charge": ch, "discharge": dis, "end": new}); soc = new
    return rows


def arbitrage():
    b = BAT_FACTORY
    rt = b["eta_c"] * b["eta_d"]
    return {"round_trip": rt, "cost_per_kwh_out": OFFPEAK / rt + b["wear"], "peak": PEAK, "offpeak": OFFPEAK,
            "margin_per_kwh_out": PEAK - (OFFPEAK / rt + b["wear"]),
            "breakeven_rt": (OFFPEAK) / (PEAK - b["wear"])}


# ---------------------------------------------------------------------------
# Part C. Forecast error and re-planning (factory, cloudy afternoon)
# ---------------------------------------------------------------------------
def rolling_horizon(load, pv_actual, pv_forecast, price, bat, demand=0.0):
    """Re-plan every hour to the end of the day. The current hour uses the measured PV,
    later hours use the forecast. Only the first hour is applied. The highest on-peak
    import so far carries over as a floor on d."""
    soc, d_floor, applied, plans = bat["soc0"], 0.0, {"ch": [], "dis": [], "imp": [], "soc": [], "cur": []}, []
    for h in range(len(load)):
        pv_view = np.concatenate([[pv_actual[h]], pv_forecast[h + 1:]])
        plan = dispatch(load[h:], pv_view, price[h:], bat, demand=demand, peak_mask=ON_PEAK[h:],
                        soc_start=soc, soc_end=bat["soc0"], d_floor=d_floor)
        if h in (0, 11, 13, 15):
            plans.append({"hour": h, "soc": [None] * h + plan["soc"], "ch": [None] * h + plan["ch"], "dis": [None] * h + plan["dis"]})
        ch, dis, imp, cur = plan["ch"][0], plan["dis"][0], plan["imp"][0], plan["cur"][0]
        soc = soc + bat["eta_c"] * ch - dis / bat["eta_d"]
        if ON_PEAK[h]:
            d_floor = max(d_floor, imp)
        for k, v in (("ch", ch), ("dis", dis), ("imp", imp), ("soc", soc), ("cur", cur)):
            applied[k].append(round(float(v), 6) + 0.0)
    imp = np.array(applied["imp"])
    energy = float(price @ imp); wear = float(bat["wear"] * sum(applied["dis"]))
    peak = float(imp[ON_PEAK].max())
    applied.update(cost=energy + wear + demand * peak, energy_cost=energy, demand_cost=demand * peak, peak_import=peak,
                   plans=plans)
    return applied


def open_loop(load, pv_actual, pv_forecast, price, bat, demand=0.0):
    """Fix the battery schedule from the morning forecast; the grid absorbs the error."""
    plan = dispatch(load, pv_forecast, price, bat, demand=demand)
    ch, dis = np.array(plan["ch"]), np.array(plan["dis"])
    net = np.asarray(load) - np.asarray(pv_actual) + ch - dis
    imp = np.clip(net, 0, None)
    peak = float(imp[ON_PEAK].max())
    energy = float(np.asarray(price) @ imp)
    return {"cost": energy + bat["wear"] * dis.sum() + demand * peak, "energy_cost": energy,
            "demand_cost": demand * peak, "peak_import": peak, "imp": (np.round(imp, 6) + 0.0).tolist(),
            "ch": plan["ch"], "dis": plan["dis"], "soc": plan["soc"]}


# ---------------------------------------------------------------------------
# Everything the slides and the lab show
# ---------------------------------------------------------------------------
def lecture_results():
    F = {"reference": food_manufacture(), "no_terminal": food_manufacture(terminal=False),
         "no_carryover": food_manufacture(carryover=False), "size": food_size()}
    ref = F["reference"]
    # finite-difference check of one stock-balance dual: one more tonne of VEG1 at the start
    plus = food_manufacture(initial=INITIAL_STOCK + 1.0, terminal=True)
    F["dual_check"] = {"note": "all five initial stocks + 1 t", "profit_plus": plus["profit"] if plus["profit"] else None,
                       "sum_values_month0": float(sum(ref["value"][0]))}
    pv = pv_profile(KWP_FACTORY)
    st = {"pv": np.round(pv, 4).tolist(), "load": LOAD_FACTORY.tolist(), "price": TOU_WEEKDAY.tolist(),
          "battery": BAT_FACTORY, "kwp": KWP_FACTORY}
    st["no_battery_energy"] = no_battery(LOAD_FACTORY, pv, TOU_WEEKDAY)
    st["no_battery_demand"] = no_battery(LOAD_FACTORY, pv, TOU_WEEKDAY, demand=DEMAND_DAY)
    st["energy_only"] = dispatch(LOAD_FACTORY, pv, TOU_WEEKDAY, BAT_FACTORY)
    st["with_demand"] = dispatch(LOAD_FACTORY, pv, TOU_WEEKDAY, BAT_FACTORY, demand=DEMAND_DAY)
    st["end_sweep"] = []
    for e in (30, 75, 150, 225, 300):
        r = dispatch(LOAD_FACTORY, pv, TOU_WEEKDAY, BAT_FACTORY, demand=DEMAND_DAY, soc_end=float(e))
        st["end_sweep"].append({"end": e, "cost": r["cost"], "soc": r["soc"], "ch": r["ch"], "dis": r["dis"]})
    nt = dispatch(LOAD_FACTORY, pv, TOU_WEEKDAY, BAT_FACTORY, demand=DEMAND_DAY, terminal=False)
    st["no_terminal"] = {"cost": nt["cost"], "soc_end": nt["checks"]["soc_end_kwh"]}
    st["soc_by_hand"] = soc_by_hand(); st["arbitrage"] = arbitrage()
    pv_c = pv_profile(KWP_FACTORY, cloud=True)
    U = {"pv_forecast": st["pv"], "pv_actual": np.round(pv_c, 4).tolist(),
         "perfect": dispatch(LOAD_FACTORY, pv_c, TOU_WEEKDAY, BAT_FACTORY, demand=DEMAND_DAY),
         "open_loop": open_loop(LOAD_FACTORY, pv_c, pv, TOU_WEEKDAY, BAT_FACTORY, demand=DEMAND_DAY),
         "rolling": rolling_horizon(LOAD_FACTORY, pv_c, pv, TOU_WEEKDAY, BAT_FACTORY, demand=DEMAND_DAY),
         "no_battery": no_battery(LOAD_FACTORY, pv_c, TOU_WEEKDAY, demand=DEMAND_DAY)}
    return {"food": F, "storage": st, "uncertainty": U,
            "tariff": {"peak": PEAK, "offpeak": OFFPEAK, "demand_month": DEMAND_MONTH, "weekdays": WEEKDAYS,
                       "demand_day": DEMAND_DAY}}


def lab_results():
    """Results behind Lab 4 (office building). Not shown on the slides."""
    hold = []
    for h in range(0, 11):
        r = food_manufacture(storage_cost=float(h))
        hold.append({"h": h, "profit": r["profit"], "stock_months": r["stock_months"], "stock": r["stock"]})
    pv = pv_profile(KWP_OFFICE)
    base = dispatch(LOAD_OFFICE, pv, TOU_WEEKDAY, BAT_OFFICE, demand=DEMAND_DAY)
    nb = no_battery(LOAD_OFFICE, pv, TOU_WEEKDAY, demand=DEMAND_DAY)
    nt = dispatch(LOAD_OFFICE, pv, TOU_WEEKDAY, BAT_OFFICE, demand=DEMAND_DAY, terminal=False)
    size = []
    for E in (100, 150, 200, 250, 300, 350, 400):
        b = dict(BAT_OFFICE, E=float(E), soc_min=0.1 * E, soc0=0.5 * E)
        r = dispatch(LOAD_OFFICE, pv, TOU_WEEKDAY, b, demand=DEMAND_DAY)
        size.append({"E": E, "cost": r["cost"], "saving": nb["cost"] - r["cost"], "peak_import": r["peak_import"],
                     "soc": r["soc"], "ch": r["ch"], "dis": r["dis"],
                     "max_residual": max(r["checks"]["max_power_balance_residual_kw"], r["checks"]["max_soc_balance_residual_kwh"])})
    sat = {}
    for name, kw in (("lp", {"curtail": False}), ("milp", {"curtail": False, "binary": True}),
                     ("lp_curtail", {"curtail": True}), ("milp_curtail", {"curtail": True, "binary": True})):
        r = dispatch(LOAD_OFFICE_SAT, pv, TOU_WEEKEND, BAT_OFFICE, **kw)
        sat[name] = {"status": r["status"]} if r["cost"] is None else {
            "status": r["status"], "cost": r["cost"], "ch": r["ch"], "dis": r["dis"], "soc": r["soc"], "cur": r["cur"],
            "curtailed_kwh": r["curtailed_kwh"], "max_simultaneous_kw": r["simultaneous_kw"],
            "hour_of_max": r["hour_of_max_simultaneous"]}
    return {"holding": hold, "pv": np.round(pv, 4).tolist(), "load": LOAD_OFFICE.tolist(), "load_sat": LOAD_OFFICE_SAT.tolist(),
            "price": TOU_WEEKDAY.tolist(), "battery": BAT_OFFICE, "kwp": KWP_OFFICE,
            "base": base, "no_battery": nb, "no_terminal": {"cost": nt["cost"], "soc_end": nt["checks"]["soc_end_kwh"],
                                                            "soc": nt["soc"], "ch": nt["ch"], "dis": nt["dis"]},
            "size": size, "saturday": sat, "soc_task": {"start": 100.0, "charge": 80.0, "discharge": 60.0,
                                                        "after_charge": 100 + 0.95 * 80, "after_discharge": 100 + 0.95 * 80 - 60 / 0.95}}


def classroom_results():
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "python": sys.version.split()[0]},
         "data": {"months": MONTHS, "oils": OILS, "price": PRICE.tolist(), "hardness": HARDNESS.tolist(),
                  "is_veg": IS_VEG.tolist(), "sell": SELL, "initial": INITIAL_STOCK, "max_stock": MAX_STOCK,
                  "hours": HOURS.tolist(), "on_peak": ON_PEAK.tolist()}}
    R["lecture"] = lecture_results()
    R["lab"] = lab_results()
    return R


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("multiperiod-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote multiperiod-results.json")
        sys.exit()
    F = R["lecture"]["food"]
    print("A reference profit", round(F["reference"]["profit"], 2), "| no terminal", round(F["no_terminal"]["profit"], 2),
          "| no carryover", round(F["no_carryover"]["profit"], 2), "| checks", F["reference"]["checks"])
    S = R["lecture"]["storage"]
    for k in ("no_battery_energy", "no_battery_demand"):
        print("B", k, {q: round(v, 1) for q, v in S[k].items() if q != "imp"})
    for k in ("energy_only", "with_demand"):
        r = S[k]; print("B", k, round(r["cost"], 1), "energy", round(r["energy_cost"], 1), "demand", round(r["demand_cost"], 1),
                        "peak", round(r["peak_import"], 1), r["checks"])
    U = R["lecture"]["uncertainty"]
    print("C", {k: round(U[k]["cost"], 1) for k in ("no_battery", "open_loop", "rolling", "perfect")})
    L = R["lab"]
    print("Lab base", round(L["base"]["cost"], 1), "no battery", round(L["no_battery"]["cost"], 1), "no terminal", L["no_terminal"])
    print("Lab saturday", {k: (v["status"], round(v.get("cost") or 0, 1), round(v.get("max_simultaneous_kw") or 0, 2), round(v.get("curtailed_kwh") or 0, 1)) for k, v in L["saturday"].items()})
    print("Lab size", [(s["E"], round(s["saving"], 1)) for s in L["size"]])
