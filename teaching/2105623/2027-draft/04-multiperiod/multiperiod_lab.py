"""2105623 Lab 4: multiperiod optimization.

Two time-indexed models, each solved with SciPy's HiGHS interface so that the
same file runs in the browser (Pyodide) and in desktop Python:

  Part A  Food Manufacture: six-month oil blending with storage
          (data from H. P. Williams, Model Building in Mathematical
          Programming, 5th ed., Section 12.1).
  Part B  24-hour dispatch of a battery beside rooftop PV under a
          time-of-use tariff (PEA Tariff 2.2.2, energy charge only;
          load and PV profiles are synthetic teaching data).

Run   python multiperiod_lab.py          print every part
      python multiperiod_lab.py --step   pause after each part
      python multiperiod_lab.py --json   write the recorded results file

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
                     terminal=True, carryover=True):
    """Maximize profit over six months. Returns plan, profit, checks and duals."""
    nI, nT = len(OILS), len(MONTHS)
    n = 3 * nI * nT                                   # buy, use, stock

    def b(i, t): return i * nT + t                    # buy[i, t]
    def u(i, t): return nI * nT + i * nT + t          # use[i, t]
    def s(i, t): return 2 * nI * nT + i * nT + t      # stock[i, t] at month end

    c = np.zeros(n)                                   # minimize -profit
    for i in range(nI):
        for t in range(nT):
            c[b(i, t)] = PRICE[t, i]
            c[u(i, t)] = -SELL
            c[s(i, t)] = storage_cost

    A_eq, b_eq = [], []                               # stock balance
    for i in range(nI):
        for t in range(nT):
            row = np.zeros(n)
            row[b(i, t)], row[u(i, t)], row[s(i, t)] = 1, -1, -1
            if t == 0:
                b_eq.append(-INITIAL_STOCK)
            else:
                row[s(i, t - 1)] = 1
                b_eq.append(0.0)
            A_eq.append(row)

    A_ub, b_ub, ub_names = [], [], []
    for t in range(nT):
        veg, nonveg, hmax, hmin = (np.zeros(n) for _ in range(4))
        for i in range(nI):
            (veg if IS_VEG[i] else nonveg)[u(i, t)] = 1
            hmax[u(i, t)] = HARDNESS[i] - 6.0         # hardness <= 6
            hmin[u(i, t)] = 3.0 - HARDNESS[i]         # hardness >= 3
        A_ub += [veg, nonveg, hmax, hmin]
        b_ub += [veg_capacity, nonveg_capacity, 0.0, 0.0]
        ub_names += [f"veg_cap_{t}", f"nonveg_cap_{t}", f"hard_max_{t}", f"hard_min_{t}"]

    bounds = [(0, None)] * (2 * nI * nT)
    for i in range(nI):
        for t in range(nT):
            lo, hi = 0.0, MAX_STOCK
            if not carryover:
                lo = hi = INITIAL_STOCK
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

    # Independent audit from the solution values only
    prev = np.full(nI, INITIAL_STOCK)
    balance = []
    for t in range(nT):
        balance.append(np.abs(prev + buy[t] - use[t] - stock[t]).max())
        prev = stock[t]
    quality = (use @ HARDNESS) / np.where(product > 0, product, 1)
    profit = SELL * product.sum() - (PRICE * buy).sum() - storage_cost * stock.sum()
    marg = res.ineqlin.marginals                      # d(-profit)/d(rhs)
    duals = {name: -m for name, m in zip(ub_names, marg)}
    return {
        "status": "optimal",
        "profit": profit,
        "solver_objective": -res.fun,
        "buy": (buy.round(6) + 0.0).tolist(), "use": (use.round(6) + 0.0).tolist(),
        "stock": (stock.round(6) + 0.0).tolist(), "product": (product.round(6) + 0.0).tolist(),
        "hardness": quality.round(6).tolist(),
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


# ---------------------------------------------------------------------------
# Part B. Battery beside rooftop PV under a time-of-use tariff
# ---------------------------------------------------------------------------
HOURS = np.arange(24)                                 # hour t covers t:00 to t+1:00
PEAK, OFFPEAK = 5.7982, 2.6369                        # THB/kWh, PEA 2.2.2, no Ft or VAT
TOU_WEEKDAY = np.where((HOURS >= 9) & (HOURS < 22), PEAK, OFFPEAK)
TOU_SATURDAY = np.full(24, OFFPEAK)                   # weekends are off-peak all day
LOAD_WEEKDAY = np.array([55, 55, 55, 55, 55, 58, 65, 85, 120, 130, 135, 140,
                         110, 135, 140, 135, 130, 115, 100, 95, 90, 80, 65, 60], float)
LOAD_SATURDAY = np.array([55, 55, 55, 55, 55, 55, 60, 75, 99, 104, 109, 109,
                          99, 104, 109, 104, 99, 85, 75, 70, 65, 60, 58, 55], float)
CLOUD = np.ones(24)
CLOUD[12:17] = [0.7, 0.35, 0.3, 0.5, 0.8]             # cloudy afternoon
BATTERY = {"E": 200.0, "P": 100.0, "eta_c": 0.95, "eta_d": 0.95,
           "soc_min": 20.0, "soc0": 100.0, "wear": 0.05}   # wear: THB per kWh discharged


def pv_profile(kwp=200.0, cloud=None):
    """Clear-sky PV output in kW for each hour (synthetic, sunrise 6:00, sunset 18:00)."""
    mid = HOURS + 0.5
    shape = np.clip(np.sin(np.pi * (mid - 6.0) / 12.0), 0, None) ** 1.3
    pv = 0.8 * kwp * shape
    return pv * (CLOUD if cloud is True else (cloud if cloud is not None else 1.0))


def dispatch(load, pv, price, export=False, export_price=0.0, curtail=True,
             terminal=True, binary=False, soc_start=None, soc_end=None, battery=None):
    """Least-cost 1-hour dispatch. Variables per hour: import, export, charge,
    discharge, curtailment, state of charge (kWh at the end of the hour)."""
    bat = dict(BATTERY, **(battery or {}))
    load, pv, price = (np.asarray(a, float) for a in (load, pv, price))
    n = len(load)
    soc_start = bat["soc0"] if soc_start is None else soc_start
    soc_end = bat["soc0"] if soc_end is None else soc_end
    IMP, EXP, CH, DIS, CUR, SOC, Y = (k * n for k in range(7))
    nv = 7 * n if binary else 6 * n

    c = np.zeros(nv)
    c[IMP:IMP + n] = price
    c[EXP:EXP + n] = -export_price
    c[DIS:DIS + n] = bat["wear"]

    rows, lo, hi, names = [], [], [], []
    for t in range(n):                                # power balance (kW, 1 h)
        r = np.zeros(nv)
        r[IMP + t], r[DIS + t], r[CH + t], r[EXP + t], r[CUR + t] = 1, 1, -1, -1, -1
        rows.append(r); lo.append(load[t] - pv[t]); hi.append(load[t] - pv[t]); names.append(("power", t))
    for t in range(n):                                # state-of-charge balance
        r = np.zeros(nv)
        r[SOC + t], r[CH + t], r[DIS + t] = 1, -bat["eta_c"], 1 / bat["eta_d"]
        rhs = soc_start if t == 0 else 0.0
        if t > 0:
            r[SOC + t - 1] = -1
        rows.append(r); lo.append(rhs); hi.append(rhs); names.append(("soc", t))
    if binary:                                        # y = 1 charge allowed, y = 0 discharge allowed
        for t in range(n):
            r = np.zeros(nv); r[CH + t], r[Y + t] = 1, -bat["P"]
            rows.append(r); lo.append(-np.inf); hi.append(0.0); names.append(("ychg", t))
            r = np.zeros(nv); r[DIS + t], r[Y + t] = 1, bat["P"]
            rows.append(r); lo.append(-np.inf); hi.append(bat["P"]); names.append(("ydis", t))

    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    ub[EXP:EXP + n] = np.inf if export else 0.0
    ub[CH:CH + n] = bat["P"]
    ub[DIS:DIS + n] = bat["P"]
    ub[CUR:CUR + n] = pv if curtail else 0.0
    lb[SOC:SOC + n], ub[SOC:SOC + n] = bat["soc_min"], bat["E"]
    if terminal:
        lb[SOC + n - 1] = max(soc_end, bat["soc_min"])
    if binary:
        ub[Y:Y + n] = 1.0

    A = np.array(rows)
    if binary:
        integ = np.zeros(nv); integ[Y:Y + n] = 1
        res = milp(c, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lb, ub),
                   integrality=integ)
        ok, x, marg = res.status == 0, res.x, None
    else:
        eq = [i for i, (a, z) in enumerate(zip(lo, hi)) if a == z]
        res = linprog(c, A_eq=A[eq], b_eq=np.array(lo)[eq], bounds=list(zip(lb, ub)),
                      method="highs")
        ok, x = res.status == 0, res.x
        marg = res.eqlin.marginals[:n] if ok else None
    if not ok:
        return {"status": "infeasible" if res.status == 2 else str(res.message), "cost": None}

    g = {k: x[o:o + n] for k, o in (("imp", IMP), ("exp", EXP), ("ch", CH), ("dis", DIS),
                                     ("cur", CUR), ("soc", SOC))}
    soc_prev = np.concatenate([[soc_start], g["soc"][:-1]])
    power_res = g["imp"] + pv - g["cur"] + g["dis"] - load - g["ch"] - g["exp"]
    soc_res = soc_prev + bat["eta_c"] * g["ch"] - g["dis"] / bat["eta_d"] - g["soc"]
    cost = float(price @ g["imp"] - export_price * g["exp"].sum() + bat["wear"] * g["dis"].sum())
    out = {k: (np.round(v, 6) + 0.0).tolist() for k, v in g.items()}
    out.update({
        "status": "optimal",
        "cost": cost,
        "energy_cost": float(price @ g["imp"]),
        "simultaneous_kw": float(np.minimum(g["ch"], g["dis"]).max()),
        "checks": {
            "max_power_balance_residual_kw": float(np.abs(power_res).max()),
            "max_soc_balance_residual_kwh": float(np.abs(soc_res).max()),
            "objective_residual_thb": float(abs(cost - res.fun)),
            "soc_end_kwh": float(g["soc"][-1]),
        },
        "marginal_thb_per_kwh": None if marg is None else np.round(marg, 4).tolist(),
    })
    return out


def no_battery_cost(load, pv, price):
    """Cost when the site has PV but no battery; surplus PV is curtailed."""
    imp = np.clip(np.asarray(load) - np.asarray(pv), 0, None)
    return float(np.asarray(price) @ imp)


def rolling_horizon(load, pv_actual, pv_forecast, price):
    """Re-plan every hour to the end of the day. The current hour uses the
    measured PV, later hours use the forecast. Only the first hour is applied."""
    bat = BATTERY
    soc, cost, applied = bat["soc0"], 0.0, {"ch": [], "dis": [], "imp": [], "soc": []}
    for h in range(len(load)):
        pv_view = np.concatenate([[pv_actual[h]], pv_forecast[h + 1:]])
        plan = dispatch(load[h:], pv_view, price[h:], soc_start=soc, soc_end=bat["soc0"])
        ch, dis, imp = plan["ch"][0], plan["dis"][0], plan["imp"][0]
        soc = soc + bat["eta_c"] * ch - dis / bat["eta_d"]
        cost += price[h] * imp + bat["wear"] * dis
        for k, v in (("ch", ch), ("dis", dis), ("imp", imp), ("soc", soc)):
            applied[k].append(round(float(v), 6) + 0.0)
    applied["cost"] = float(cost)
    return applied


def open_loop(load, pv_actual, pv_forecast, price):
    """Fix the battery schedule from the morning forecast; the grid absorbs the error."""
    plan = dispatch(load, pv_forecast, price)
    ch, dis = np.array(plan["ch"]), np.array(plan["dis"])
    net = np.asarray(load) - np.asarray(pv_actual) + ch - dis
    imp = np.clip(net, 0, None)
    return {"cost": float(np.asarray(price) @ imp + BATTERY["wear"] * dis.sum()),
            "imp": (np.round(imp, 6) + 0.0).tolist()}


# ---------------------------------------------------------------------------
# Everything the slides show, in one dictionary
# ---------------------------------------------------------------------------
def classroom_results():
    pv_clear, pv_cloudy = pv_profile(), pv_profile(cloud=True)
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                  "python": sys.version.split()[0]},
         "data": {"hours": HOURS.tolist(), "tou_weekday": TOU_WEEKDAY.tolist(),
                  "load_weekday": LOAD_WEEKDAY.tolist(), "load_saturday": LOAD_SATURDAY.tolist(),
                  "pv_clear": np.round(pv_clear, 4).tolist(),
                  "pv_cloudy": np.round(pv_cloudy, 4).tolist(), "battery": BATTERY,
                  "oils": OILS, "months": MONTHS, "price": PRICE.tolist(),
                  "hardness": HARDNESS.tolist()}}
    # Part A
    R["food"] = {"reference": food_manufacture()}
    R["food"]["holding_sweep"] = []
    for h in range(0, 11):
        run = food_manufacture(storage_cost=h)
        R["food"]["holding_sweep"].append(
            {"h": h, "profit": run["profit"], "stock_total": float(np.sum(run["stock"])),
             "buy": run["buy"], "stock": run["stock"]})
    R["food"]["no_carryover"] = food_manufacture(carryover=False)
    R["food"]["no_terminal"] = food_manufacture(terminal=False)
    # Part B
    base = dispatch(LOAD_WEEKDAY, pv_clear, TOU_WEEKDAY)
    R["storage"] = {
        "base": base,
        "no_battery_cost": no_battery_cost(LOAD_WEEKDAY, pv_clear, TOU_WEEKDAY),
        "no_terminal": dispatch(LOAD_WEEKDAY, pv_clear, TOU_WEEKDAY, terminal=False),
        "base_binary": dispatch(LOAD_WEEKDAY, pv_clear, TOU_WEEKDAY, binary=True),
        "saturday_lp_no_curtail": dispatch(LOAD_SATURDAY, pv_clear, TOU_SATURDAY, curtail=False),
        "saturday_milp_no_curtail": dispatch(LOAD_SATURDAY, pv_clear, TOU_SATURDAY, curtail=False, binary=True),
        "saturday_lp_curtail": dispatch(LOAD_SATURDAY, pv_clear, TOU_SATURDAY, curtail=True),
    }
    R["uncertainty"] = {
        "perfect": dispatch(LOAD_WEEKDAY, pv_cloudy, TOU_WEEKDAY),
        "open_loop": open_loop(LOAD_WEEKDAY, pv_cloudy, pv_clear, TOU_WEEKDAY),
        "rolling": rolling_horizon(LOAD_WEEKDAY, pv_cloudy, pv_clear, TOU_WEEKDAY),
        "no_battery_cost": no_battery_cost(LOAD_WEEKDAY, pv_cloudy, TOU_WEEKDAY),
    }
    return R


def _print_parts(R, step=False):
    def pause():
        if step:
            input("\nPress Enter for the next part ...")
    f = R["food"]["reference"]
    print("PART 1  Food Manufacture, reference plan")
    print(f"  profit = {f['profit']:,.2f} GBP (solver {f['solver_objective']:,.2f})")
    for m, p, h in zip(MONTHS, f["product"], f["hardness"]):
        print(f"  {m}: product {p:7.2f} t  hardness {h:5.3f}")
    print("  checks:", f["checks"]); pause()
    print("\nPART 2  What storage is worth")
    print(f"  no carryover : {R['food']['no_carryover']['profit']:,.2f} GBP")
    print(f"  no terminal  : {R['food']['no_terminal']['profit']:,.2f} GBP  (end effect)")
    for row in R["food"]["holding_sweep"]:
        print(f"  holding {row['h']:2d} GBP/t-month: profit {row['profit']:11,.2f}  "
              f"total month-end stock {row['stock_total']:8.1f} t")
    pause()
    s = R["storage"]
    print("\nPART 3  Weekday dispatch with TOU")
    print(f"  no battery {s['no_battery_cost']:.2f} THB, with battery {s['base']['cost']:.2f} THB")
    print("  checks:", s["base"]["checks"]); pause()
    print("\nPART 4  Terminal condition")
    print(f"  without SoC(24) >= SoC(0): {s['no_terminal']['cost']:.2f} THB, "
          f"ends at {s['no_terminal']['checks']['soc_end_kwh']:.1f} kWh"); pause()
    print("\nPART 5  Charge and discharge in the same hour")
    print(f"  base LP simultaneous {s['base']['simultaneous_kw']:.3f} kW; MILP cost {s['base_binary']['cost']:.2f}")
    print(f"  Saturday, no curtailment, LP  : simultaneous {s['saturday_lp_no_curtail']['simultaneous_kw']:.2f} kW")
    print(f"  Saturday, no curtailment, MILP: {s['saturday_milp_no_curtail']['status']}")
    hc = s["saturday_lp_curtail"]
    print(f"  Saturday with curtailment, LP: simultaneous {hc['simultaneous_kw']:.3f} kW, "
          f"curtailed {sum(hc['cur']):.1f} kWh"); pause()
    u = R["uncertainty"]
    print("\nPART 6  Forecast error and re-planning (cloudy afternoon)")
    print(f"  no battery {u['no_battery_cost']:.2f} | fixed morning plan {u['open_loop']['cost']:.2f} | "
          f"re-plan hourly {u['rolling']['cost']:.2f} | perfect forecast {u['perfect']['cost']:.2f} THB")


if __name__ == "__main__":
    results = classroom_results()
    if "--json" in sys.argv:
        with open("lab4-results.json", "w") as fh:
            json.dump(results, fh, indent=1)
        print("wrote lab4-results.json")
    else:
        _print_parts(results, step="--step" in sys.argv)
