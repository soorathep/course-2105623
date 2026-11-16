"""2105623 Session 3: MILP for logical and discrete decisions.

Part A  Lecture: the volume discount left open in Session 2, closed with one
        binary; the big-M picture for a fixed charge; a plain branch-and-bound
        tree; aggregated and disaggregated facility location.
Part B  Lab 3: commit four boilers over 24 hours with minimum loads, no-load
        costs, start-up costs and a reserve margin.

All data are synthetic. Runs in the browser (Pyodide: NumPy, SciPy/HiGHS) and
in desktop Python.
    python milp_lab.py          print every part
    python milp_lab.py --json   write milp-results.json for the slides and the lab page
"""
import itertools
import json
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


class MILP:
    """A small named-variable MILP builder on scipy.optimize.milp (HiGHS)."""

    def __init__(self):
        self.cols, self.obj, self.lb, self.ub, self.int = [], [], [], [], []
        self.rows = []

    def var(self, name, obj=0.0, lb=0.0, ub=np.inf, integer=False):
        self.cols.append(name); self.obj.append(obj); self.lb.append(lb); self.ub.append(ub); self.int.append(integer)

    def row(self, name, terms, lo=-np.inf, hi=np.inf):
        self.rows.append((name, dict(terms), lo, hi))

    def solve(self, maximize=False, relax=False, options=None):
        ix = {c: k for k, c in enumerate(self.cols)}
        A = np.zeros((len(self.rows), len(self.cols)))
        for k, (_, t, _, _) in enumerate(self.rows):
            for c, a in t.items():
                A[k, ix[c]] += a
        c = np.array(self.obj) * (-1.0 if maximize else 1.0)
        integ = np.zeros(len(self.cols)) if relax else np.array(self.int, float)
        t0 = time.perf_counter()
        r = milp(c, constraints=LinearConstraint(A, [x[2] for x in self.rows], [x[3] for x in self.rows]),
                 integrality=integ, bounds=Bounds(self.lb, self.ub), options=options or {})
        dt = time.perf_counter() - t0
        if r.status != 0 or r.x is None:
            return {"status": {2: "infeasible", 3: "unbounded"}.get(r.status, r.message)}
        sg = -1.0 if maximize else 1.0
        out = {"status": "optimal", "obj": float(sg * r.fun), "x": dict(zip(self.cols, (r.x.round(9) + 0.0).tolist())),
               "seconds": dt}
        if not relax and any(self.int):
            out.update(nodes=int(getattr(r, "mip_node_count", 0) or 0), gap=float(getattr(r, "mip_gap", 0.0) or 0.0),
                       bound=float(sg * getattr(r, "mip_dual_bound", r.fun)))
        return out


def branch_and_bound(c, A_ub, b_ub, lb, ub, ints, maximize=False, max_nodes=5000):
    """Textbook LP-based branch and bound: depth first, most fractional variable,
    no presolve and no cuts. Every node is recorded so the tree can be drawn."""
    sg = -1.0 if maximize else 1.0
    nodes, inc, incx, nid = [], None, None, 0
    stack = [(0, None, "root", np.array(lb, float), np.array(ub, float), None, 0)]
    while stack and len(nodes) < max_nodes:
        idn, parent, desc, lo, hi, plp, depth = stack.pop()
        r = linprog(sg * np.asarray(c, float), A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lo, hi)), method="highs")
        node = {"id": idn, "parent": parent, "branch": desc, "depth": depth}
        if r.status != 0:
            node.update(status="infeasible", lp=None)
        else:
            z = float(sg * r.fun); node["lp"] = z
            worse = inc is not None and ((z <= inc + 1e-9) if maximize else (z >= inc - 1e-9))
            frac = [(j, r.x[j]) for j in ints if abs(r.x[j] - round(r.x[j])) > 1e-6]
            if worse:
                node["status"] = "pruned by bound"
            elif not frac:
                node["status"] = "integer"; inc, incx = z, r.x.copy()
            else:
                j, v = max(frac, key=lambda t: -abs(t[1] - np.floor(t[1]) - 0.5))
                node.update(status="branched", var=int(j), value=float(v))
                l1, u1 = lo.copy(), hi.copy(); u1[j] = np.floor(v)
                l2, u2 = lo.copy(), hi.copy(); l2[j] = np.ceil(v)
                for (ll, uu, d) in ((l1, u1, f"{j}<={int(np.floor(v))}"), (l2, u2, f"{j}>={int(np.ceil(v))}")):
                    nid += 1; stack.append((nid, idn, d, ll, uu, z, depth + 1))
        opn = [s[5] for s in stack if s[5] is not None]
        cand = opn + ([inc] if inc is not None else [])
        node["incumbent"] = inc
        node["bound"] = (max(cand) if maximize else min(cand)) if cand else inc
        nodes.append(node)
    return {"nodes": nodes, "best": inc, "x": None if incx is None else incx.round(9).tolist(), "complete": not stack}


# ---------------------------------------------------------------------------
# Part A1. Session 2's volume discount, closed with one binary
# ---------------------------------------------------------------------------
COMP = ["Butane", "LSR naphtha", "Reformate", "FCC gasoline", "Alkylate"]
RON = {"Butane": 94.0, "LSR naphtha": 70.0, "Reformate": 98.0, "FCC gasoline": 92.0, "Alkylate": 96.0}
RVP = {"Butane": 360.0, "LSR naphtha": 80.0, "Reformate": 25.0, "FCC gasoline": 50.0, "Alkylate": 30.0}
COST = {"Butane": 380.0, "LSR naphtha": 520.0, "Reformate": 640.0, "FCC gasoline": 600.0}
AVAIL = {"Butane": 300.0, "LSR naphtha": 900.0, "Reformate": 1200.0, "FCC gasoline": 1500.0}
GRADES, RON_MIN, RVP_MAX = ["Regular", "Premium"], {"Regular": 91.0, "Premium": 95.0}, 60.0
PRICE, DEMAND = {"Regular": 690.0, "Premium": 750.0}, {"Regular": 2500.0, "Premium": 1500.0}
TIER, P1, P2 = 300.0, 780.0, 700.0          # alkylate: first 300 m3/d at 780, beyond that 700 USD/m3


def discount_blend(relax=False):
    """Tier 2 may be used only when tier 1 is full: q1 >= 300 z, q2 <= 300 z, z binary."""
    m = MILP()
    for i in COMP:
        for g in GRADES:
            m.var(f"{i}>{g}", obj=PRICE[g] - COST.get(i, 0.0))
    m.var("q1", obj=-P1, ub=TIER); m.var("q2", obj=-P2, ub=TIER); m.var("z", ub=1.0, integer=True)
    m.row("tier 1 full if tier 2 used", {"q1": 1, "z": -TIER}, lo=0.0)
    m.row("tier 2 only if z", {"q2": 1, "z": -TIER}, hi=0.0)
    m.row("alkylate bought", {**{f"Alkylate>{g}": 1 for g in GRADES}, "q1": -1, "q2": -1}, lo=0.0, hi=0.0)
    for i in COMP[:-1]:
        m.row(f"avail {i}", {f"{i}>{g}": 1 for g in GRADES}, hi=AVAIL[i])
    for g in GRADES:
        m.row(f"demand {g}", {f"{i}>{g}": 1 for i in COMP}, hi=DEMAND[g])
        m.row(f"RON {g}", {f"{i}>{g}": RON[i] - RON_MIN[g] for i in COMP}, lo=0.0)
        m.row(f"RVP {g}", {f"{i}>{g}": RVP[i] ** 1.25 - RVP_MAX ** 1.25 for i in COMP}, hi=0.0)
    r = m.solve(maximize=True, relax=relax)
    x = r["x"]
    return {"margin": r["obj"], "q1": x["q1"], "q2": x["q2"], "z": x["z"], "alkylate": x["q1"] + x["q2"],
            "nodes": r.get("nodes"), "gap": r.get("gap")}


# ---------------------------------------------------------------------------
# Part A2. The big-M picture: one boiler or bought steam
# ---------------------------------------------------------------------------
FC = {"demand": 30.0, "cap": 50.0, "fuel": 20.0, "fixed": 600.0, "buy": 45.0}   # t/h, t/h, USD/t, USD/h, USD/t


def fixed_charge(M):
    """min fuel*x + fixed*y + buy*w  s.t.  x + w >= demand, x <= M y, x <= cap, y in {0,1}."""
    out = {"M": M}
    for relax in (True, False):
        m = MILP()
        m.var("x", obj=FC["fuel"], ub=FC["cap"]); m.var("w", obj=FC["buy"]); m.var("y", obj=FC["fixed"], ub=1.0, integer=True)
        m.row("steam", {"x": 1, "w": 1}, lo=FC["demand"]); m.row("big-M", {"x": 1, "y": -M}, hi=0.0)
        r = m.solve(relax=relax)
        out["lp" if relax else "milp"] = {"cost": r["obj"], "x": r["x"]["x"], "y": r["x"]["y"], "w": r["x"]["w"]}
    true_best = min(FC["fuel"] * FC["demand"] + FC["fixed"], FC["buy"] * FC["demand"])
    out["valid"] = abs(out["milp"]["cost"] - true_best) < 1e-6
    out["gap"] = (out["milp"]["cost"] - out["lp"]["cost"]) / out["milp"]["cost"]
    return out


# ---------------------------------------------------------------------------
# Part A3. A branch-and-bound tree: heat-recovery projects under a budget
# ---------------------------------------------------------------------------
PROJ = ["P1", "P2", "P3", "P4", "P5"]
CAPEX = [120.0, 90.0, 75.0, 60.0, 150.0]       # kUSD
SAVE = [54.0, 38.0, 30.0, 27.0, 60.0]          # kUSD per year
BUDGET = 300.0


def project_tree():
    r = branch_and_bound(SAVE, [CAPEX], [BUDGET], [0] * 5, [1] * 5, range(5), maximize=True)
    # layout: x by in-order position of the final tree, y by depth
    kids = {}
    for n in r["nodes"]:
        kids.setdefault(n["parent"], []).append(n["id"])
    order, pos = [], {}
    def walk(i):
        ch = sorted(kids.get(i, []))
        if ch:
            walk(ch[0])
        order.append(i)
        for c in ch[1:]:
            walk(c)
    walk(0)
    for k, i in enumerate(order):
        pos[i] = k
    for n in r["nodes"]:
        n["xpos"] = pos[n["id"]]
        if "branch" in n and n["branch"] != "root":
            j, rest = (n["branch"].split("<=") + [None])[:2] if "<=" in n["branch"] else n["branch"].split(">=")
            n["label"] = f"{PROJ[int(j)]} = {'0' if '<=' in n['branch'] else '1'}"
    greedy, spent = [], 0.0
    for j in sorted(range(5), key=lambda j: -SAVE[j] / CAPEX[j]):
        if spent + CAPEX[j] <= BUDGET:
            greedy.append(PROJ[j]); spent += CAPEX[j]
    return {"nodes": r["nodes"], "best": r["best"], "x": r["x"], "chosen": [PROJ[j] for j in range(5) if r["x"][j] > 0.5],
            "root_lp": r["nodes"][0]["lp"], "greedy": greedy, "greedy_saving": sum(SAVE[PROJ.index(p)] for p in greedy),
            "width": len(order)}


# ---------------------------------------------------------------------------
# Part A4. Formulation strength: aggregated vs disaggregated facility location
# ---------------------------------------------------------------------------
def location(n_sites=8, n_plants=20, seed=7):
    """Open solvent-recovery sites (fixed cost) and assign each plant to one site.
    Aggregated:    sum_j x_ij <= n_plants * y_i     (one row per site)
    Disaggregated: x_ij <= y_i                      (one row per pair; the same integer set)"""
    rng = np.random.default_rng(seed)
    F = rng.uniform(2000, 4000, n_sites); d = rng.uniform(5, 20, n_plants)
    ps = rng.uniform(0, 100, (n_sites, 2)); pp = rng.uniform(0, 100, (n_plants, 2))
    tc = 2.0 * np.linalg.norm(ps[:, None] - pp[None], axis=2) * d[None, :]
    n = n_sites + n_sites * n_plants
    c = np.concatenate([F, tc.ravel()])
    out = {"sites": ps.round(2).tolist(), "plants": pp.round(2).tolist(), "fixed": F.round(1).tolist(), "demand": d.round(2).tolist()}
    for name in ("aggregated", "disaggregated"):
        A, b = [], []
        for j in range(n_plants):
            r = np.zeros(n); r[n_sites + np.arange(n_sites) * n_plants + j] = -1; A.append(r); b.append(-1.0)
        if name == "aggregated":
            for i in range(n_sites):
                r = np.zeros(n); r[n_sites + i * n_plants:n_sites + (i + 1) * n_plants] = 1; r[i] = -n_plants; A.append(r); b.append(0.0)
        else:
            for i in range(n_sites):
                for j in range(n_plants):
                    r = np.zeros(n); r[n_sites + i * n_plants + j] = 1; r[i] = -1; A.append(r); b.append(0.0)
        A, b = np.array(A), np.array(b)
        bb = branch_and_bound(c, A, b, [0] * n, [1] * n, range(n_sites), max_nodes=3000)
        integ = np.zeros(n); integ[:n_sites] = 1
        t0 = time.perf_counter()
        h = milp(c, constraints=LinearConstraint(A, -np.inf, b), integrality=integ, bounds=Bounds(0, 1))
        dt = time.perf_counter() - t0
        lp = linprog(c, A_ub=A, b_ub=b, bounds=[(0, 1)] * n, method="highs")
        y = h.x[:n_sites]; x = h.x[n_sites:].reshape(n_sites, n_plants)
        out[name] = {"rows": int(A.shape[0]), "lp_bound": float(lp.fun), "optimum": float(h.fun),
                     "root_gap": float((h.fun - lp.fun) / h.fun), "bb_nodes": len(bb["nodes"]), "bb_best": bb["best"],
                     "highs_nodes": int(getattr(h, "mip_node_count", 0) or 0), "highs_seconds": dt,
                     "open": [int(i) for i in range(n_sites) if y[i] > 0.5],
                     "assign": [int(np.argmax(x[:, j])) for j in range(n_plants)],
                     "lp_y": (lp.x[:n_sites].round(4) + 0.0).tolist()}
    return out


# ---------------------------------------------------------------------------
# Part B. Lab 3: commit four boilers over 24 hours
# ---------------------------------------------------------------------------
BOILERS = ["B1", "B2", "B3", "B4"]
UMAX = {"B1": 120.0, "B2": 100.0, "B3": 60.0, "B4": 40.0}       # t/h steam
LMIN = {"B1": 50.0, "B2": 40.0, "B3": 20.0, "B4": 10.0}         # t/h minimum stable load
FUEL = {"B1": 18.0, "B2": 20.0, "B3": 24.0, "B4": 30.0}         # USD per t steam
NOLOAD = {"B1": 900.0, "B2": 700.0, "B3": 300.0, "B4": 120.0}   # USD per hour while on
START = {"B1": 6000.0, "B2": 4500.0, "B3": 1200.0, "B4": 300.0}  # USD per start
U0 = {"B1": 1, "B2": 0, "B3": 0, "B4": 0}                        # on at midnight
STEAM = [120, 115, 110, 110, 115, 130, 165, 205, 235, 250, 255, 250,
         240, 245, 250, 240, 225, 215, 205, 190, 170, 150, 135, 125]  # t/h, hour 0 to 23
RESERVE = 0.10                                                     # committed capacity >= 1.1 x demand


def boilers(M=None, M_b=None, relax=False, start_scale=1.0, hours=None, reserve=RESERVE, fix_u=None):
    """x[b,t] load, u[b,t] on/off (binary), s[b,t] start-up (continuous: s >= u_t - u_{t-1} and a
    positive start cost make it 0 or 1 whenever u is integer).
    Load rows: L u <= x <= M u, with M = UMAX (tight) unless M or M_b says otherwise; x <= UMAX always."""
    hours = list(range(24)) if hours is None else list(hours)
    m = MILP()
    for b in BOILERS:
        for t in hours:
            m.var(f"x {b} {t}", obj=FUEL[b], ub=UMAX[b])
            m.var(f"u {b} {t}", obj=NOLOAD[b], ub=1.0, integer=True)
            m.var(f"s {b} {t}", obj=START[b] * start_scale, ub=1.0)
            mm = UMAX[b] if M is None else M
            if M_b and b in M_b:
                mm = M_b[b]
            m.row(f"max load {b} {t}", {f"x {b} {t}": 1, f"u {b} {t}": -mm}, hi=0.0)
            m.row(f"min load {b} {t}", {f"x {b} {t}": 1, f"u {b} {t}": -LMIN[b]}, lo=0.0)
            if fix_u is not None:
                m.row(f"fix {b} {t}", {f"u {b} {t}": 1}, lo=fix_u[b], hi=fix_u[b])
            if len(hours) > 1 or start_scale:
                prev = hours.index(t)
                if prev == 0:
                    m.row(f"start {b} {t}", {f"s {b} {t}": 1, f"u {b} {t}": -1}, lo=-U0[b] if hours[0] == 0 else -np.inf)
                else:
                    m.row(f"start {b} {t}", {f"s {b} {t}": 1, f"u {b} {t}": -1, f"u {b} {hours[prev - 1]}": 1}, lo=0.0)
    for t in hours:
        m.row(f"steam {t}", {f"x {b} {t}": 1 for b in BOILERS}, lo=STEAM[t])
        m.row(f"reserve {t}", {f"u {b} {t}": UMAX[b] for b in BOILERS}, lo=(1 + reserve) * STEAM[t])
    r = m.solve(relax=relax)
    if r["status"] != "optimal":
        return r
    x = r["x"]
    out = {"status": "optimal", "cost": r["obj"], "seconds": r["seconds"],
           "u": {b: [x[f"u {b} {t}"] for t in hours] for b in BOILERS},
           "x": {b: [x[f"x {b} {t}"] for t in hours] for b in BOILERS},
           "starts": {b: [x[f"s {b} {t}"] for t in hours] for b in BOILERS}}
    # count start-ups from the on/off pattern itself (s is free when its cost is zero)
    out["n_starts"] = sum(1 for b in BOILERS for k, t in enumerate(hours)
                          if out["u"][b][k] > 0.5 and (out["u"][b][k - 1] if k else (U0[b] if t == 0 else 0)) < 0.5) \
        if not relax else round(sum(sum(v) for v in out["starts"].values()), 6)
    out["n_fractional"] = sum(1 for b in BOILERS for v in out["u"][b] if 1e-6 < v < 1 - 1e-6)
    out["parts"] = {"fuel": sum(FUEL[b] * x[f"x {b} {t}"] for b in BOILERS for t in hours),
                    "no_load": sum(NOLOAD[b] * x[f"u {b} {t}"] for b in BOILERS for t in hours),
                    "start": sum(START[b] * start_scale * x[f"s {b} {t}"] for b in BOILERS for t in hours)}
    if not relax:
        out.update(nodes=r.get("nodes"), gap=r.get("gap"), bound=r.get("bound"))
    return out


def enumerate_hour(t=19):
    """Every on/off pattern for one hour (no start-up costs): each fixed pattern is an LP."""
    rows = []
    for pat in itertools.product([0, 1], repeat=4):
        fu = dict(zip(BOILERS, pat))
        cap = sum(UMAX[b] * fu[b] for b in BOILERS)
        r = boilers(hours=[t], start_scale=0.0, fix_u=fu, relax=True)
        rows.append({"pattern": "".join(map(str, pat)), "on": [b for b in BOILERS if fu[b]], "capacity": cap,
                     "status": r["status"], "cost": r.get("cost"),
                     "load": {b: r["x"][b][0] for b in BOILERS} if r["status"] == "optimal" else None})
    best = min((r for r in rows if r["status"] == "optimal"), key=lambda r: r["cost"])
    one = boilers(hours=[t], start_scale=0.0)
    return {"hour": t, "steam": STEAM[t], "rows": rows, "feasible": sum(r["status"] == "optimal" for r in rows),
            "best": best, "milp_cost": one["cost"], "milp_on": [b for b in BOILERS if one["u"][b][0] > 0.5]}


def lab_results():
    tight, tight_lp = boilers(), boilers(relax=True)
    loose, loose_lp = boilers(M=10000.0), boilers(M=10000.0, relax=True)
    small = boilers(M_b={"B1": 100.0})
    sweep = []
    for k in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        r = boilers(start_scale=k)
        sweep.append({"scale": k, "cost": r["cost"], "starts": r["n_starts"], "nodes": r["nodes"],
                      "u": r["u"], "parts": r["parts"]})
    return {"tight": tight, "tight_lp": tight_lp, "loose": loose, "loose_lp": loose_lp, "small_M": small,
            "enum": enumerate_hour(19), "start_sweep": sweep,
            "data": {"boilers": BOILERS, "umax": UMAX, "lmin": LMIN, "fuel": FUEL, "noload": NOLOAD, "start": START,
                     "u0": U0, "steam": STEAM, "reserve": RESERVE}}


def classroom_results():
    R = {"meta": {"numpy": np.__version__, "scipy": __import__("scipy").__version__, "python": sys.version.split()[0]}}
    R["lecture"] = {
        "discount": {"milp": discount_blend(), "lp": discount_blend(relax=True), "tier": TIER, "p1": P1, "p2": P2},
        "fixed_charge": {"data": FC, "cases": [fixed_charge(M) for M in (20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 200.0, 500.0, 1000.0)]},
        "tree": project_tree(),
        "tree_data": {"projects": PROJ, "capex": CAPEX, "save": SAVE, "budget": BUDGET},
        "location": location(),
    }
    R["lab"] = lab_results()
    return R


def _sched(u):
    return {b: "".join("#" if v > 0.999 else ("." if v < 1e-3 else "~") for v in u[b]) for b in u}


if __name__ == "__main__":
    R = classroom_results()
    if "--json" in sys.argv:
        with open("milp-results.json", "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote milp-results.json")
        sys.exit()
    L = R["lecture"]
    print("A1 discount:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in L["discount"]["milp"].items()},
          "| LP relaxation:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in L["discount"]["lp"].items()})
    print("A2 fixed charge:")
    for c in L["fixed_charge"]["cases"]:
        print(f"   M {c['M']:7.0f}: LP {c['lp']['cost']:8.1f} (y {c['lp']['y']:.3f})  MILP {c['milp']['cost']:7.1f}  valid {c['valid']}  gap {100*c['gap']:5.1f}%")
    t = L["tree"]; print(f"A3 tree: {len(t['nodes'])} nodes, best {t['best']} {t['chosen']}, root LP {t['root_lp']:.2f}, greedy {t['greedy']} {t['greedy_saving']}")
    for k in ("aggregated", "disaggregated"):
        v = L["location"][k]
        print(f"A4 {k}: rows {v['rows']}, LP {v['lp_bound']:.0f}, opt {v['optimum']:.0f}, root gap {100*v['root_gap']:.1f}%, plain B&B {v['bb_nodes']} nodes, HiGHS {v['highs_nodes']} nodes, open {v['open']}")
    lab = R["lab"]
    for k in ("tight", "tight_lp", "loose", "loose_lp", "small_M"):
        v = lab[k]
        print(f"B {k}: cost {v['cost']:.1f}, starts {v['n_starts']}, fractional u {v['n_fractional']}, nodes {v.get('nodes')}, gap {v.get('gap')}")
    for b, s in _sched(lab["tight"]["u"]).items():
        print("   ", b, s)
    e = lab["enum"]; print(f"B enum hour {e['hour']}: {e['feasible']} feasible, best {e['best']['on']} {e['best']['cost']:.1f}; MILP {e['milp_on']} {e['milp_cost']:.1f}")
    for s in lab["start_sweep"]:
        print(f"   start x{s['scale']}: cost {s['cost']:.1f}, starts {s['starts']}, nodes {s['nodes']}")
