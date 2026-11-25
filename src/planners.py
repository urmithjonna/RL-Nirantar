"""MILP planner (perfect foresight, MPC, MPC-CC). Same objective and physics as src/sim.py. Written from notebooks/03_simulator.ipynb.
Identical to the archived v1-v3 planner; the only addition is the optional reserve (chance constraint) used by MPC-CC."""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from sim import SPD, DT


def split_pv(pv, ess, oth, load_on):
    pv_e = np.minimum(pv, ess); left = pv - pv_e
    pv_o = np.minimum(left, oth); left = left - pv_o
    pv_n = np.minimum(left, load_on)
    return ess - pv_e, oth - pv_o, load_on - pv_n, left - pv_n, pv_o


def peak_mask(t, cfg):
    hour = (np.asarray(t) % SPD) / 4
    return ((hour >= cfg.peak_hours[0]) & (hour < cfg.peak_hours[1])).astype(float)


def solve_milp(nE, nO, lon, sur, peak, s0, cfg, terminal_value=0.0, force_no_grid_charge_first=False, n_binary=None,
               mip_rel_gap=1e-4, time_limit=None, reserve=None):
    """Per step: be, bo, bn (battery to essentials / other cut load / connected homes), cp, cg (charge from solar / grid),
    s (stored energy after the step), z (1 = grid charging allowed, then no discharge), x (import above cap in peak hours),
    q (reserve shortfall, kWh; only with `reserve`).
    reserve = (R_kwh, n_steps, w): s_h + q_h >= R_kwh for h < n_steps, each kWh of shortfall costing w x DT per step."""
    H = len(nE); c = cfg; K = 9; nv = K * H
    iv = lambda k, h: k * H + h
    cost = np.zeros(nv); w = c.w_wear * DT
    cost[iv(0, 0):iv(0, 0) + H] = -c.w_ess * DT + w
    cost[iv(1, 0):iv(1, 0) + H] = -c.w_other * DT + w
    cost[iv(2, 0):iv(2, 0) + H] = -c.w_peak * DT * peak + w
    cost[iv(3, 0):iv(3, 0) + H] = -1e-6
    cost[iv(4, 0):iv(4, 0) + H] = c.w_peak * DT * peak
    cost[iv(5, H - 1)] = -terminal_value
    cost[iv(7, 0):iv(7, 0) + H] = c.w_peak_excess * DT * peak
    r, cc, v, lo, hi = [], [], [], [], []; row = 0
    for h in range(H):
        for k in range(5):
            r.append(row); cc.append(iv(k, h)); v.append(1.0)
        lo.append(-np.inf); hi.append(c.batt_kw); row += 1
        r += [row, row]; cc += [iv(4, h), iv(6, h)]; v += [1.0, -c.batt_kw]; lo.append(-np.inf); hi.append(0.0); row += 1
        for k in range(3):
            r.append(row); cc.append(iv(k, h)); v.append(1.0)
        r.append(row); cc.append(iv(6, h)); v.append(c.batt_kw); lo.append(-np.inf); hi.append(c.batt_kw); row += 1
        for k, coef in [(0, DT / c.eta), (1, DT / c.eta), (2, DT / c.eta), (3, -DT * c.eta), (4, -DT * c.eta), (5, 1.0)]:
            r.append(row); cc.append(iv(k, h)); v.append(coef)
        if h > 0:
            r.append(row); cc.append(iv(5, h - 1)); v.append(-1.0)
        b = s0 if h == 0 else 0.0; lo.append(b); hi.append(b); row += 1
        r += [row, row, row]; cc += [iv(7, h), iv(2, h), iv(4, h)]; v += [-1.0, -1.0, 1.0]
        lo.append(-np.inf); hi.append(c.cap_kw - lon[h]); row += 1
    q_ub = np.zeros(H)
    if reserve is not None:
        R, n_res, w_q = reserve
        for h in range(min(n_res, H)):
            r += [row, row]; cc += [iv(5, h), iv(8, h)]; v += [1.0, 1.0]; lo.append(R); hi.append(np.inf); row += 1
            q_ub[h] = np.inf
        cost[iv(8, 0):iv(8, 0) + H] = w_q * DT
    A = sp.csr_matrix((v, (r, cc)), shape=(row, nv))
    zub = np.ones(H)
    if force_no_grid_charge_first:
        zub[0] = 0.0
    ub = np.concatenate([nE, nO, lon, sur, np.full(H, c.batt_kw), np.full(H, c.batt_kwh), zub, np.full(H, np.inf), q_ub])
    lb = np.concatenate([np.zeros(5 * H), np.full(H, c.soc_min * c.batt_kwh), np.zeros(3 * H)])
    nb = H if n_binary is None else min(n_binary, H)        # exact on/off only for the first n_binary steps
    integ = np.r_[np.zeros(6 * H), np.ones(nb), np.zeros(H - nb), np.zeros(2 * H)]
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"mip_rel_gap": mip_rel_gap, **({"time_limit": time_limit} if time_limit else {})})
    if res.x is None:
        raise RuntimeError(res.message)
    out = {k: res.x[iv(i, 0):iv(i, 0) + H] for i, k in enumerate(["be", "bo", "bn", "cp", "cg", "s", "z", "x", "q"])}
    out["obj"] = res.fun
    out["mip_gap"] = float(getattr(res, "mip_gap", 0.0) or 0.0); out["status"] = int(res.status)   # 1 = time limit hit
    return out


def to_action(plan, h, oth_need, pv_o, cfg):
    u = (plan["bn"][h] - plan["cg"][h]) / cfg.batt_kw
    return u, ((pv_o + plan["bo"][h]) / oth_need if oth_need > 1e-9 else 1.0)


def true_inputs(env, t0, H, cfg):
    t = np.arange(t0, t0 + H); d, off = env.demand[t], env.off[t]
    return (cfg.essential_share * (d * off).sum(1), (1 - cfg.essential_share) * (d * off).sum(1), (d * ~off).sum(1), env.pv[t])
