"""Baseline controllers and the expert. Written from notebooks/03_simulator.ipynb.
Rule policies, MPC v2 and perfect foresight are the archived v1-v3 controllers (baselines.py, baselines_v2.py; duration model off);
MPC-CC (chance-constrained MPC) is new (results/FINAL_SPEC.md section 6)."""
import numpy as np
from scipy.stats import norm
from sim import StreetEnv, BATT_LEVELS, NL, SPD
from planners import split_pv, peak_mask, solve_milp, to_action, true_inputs


# ---------- rules ----------
def act(level, full):
    return int(np.where(BATT_LEVELS == level)[0][0]) + (NL if full else 0)


def policy_idle_full(env, obs):
    return act(0.0, True)


def policy_backup_only(env, obs):
    return act(-1.0, True) if obs[0] < 0.999 and not obs[5] > 0 else act(0.0, True)


def policy_rule(env, obs):
    soc, hour = obs[0], (env.t % SPD) / 4
    full = soc > 0.25
    if 18 <= hour < 22 and soc > 0.40:
        return act(1.0, full)
    if hour < 6 and soc < 0.90 and obs[5] == 0:
        return act(-1.0, full)
    return act(0.0, full)


POLICIES = {"idle_full": policy_idle_full, "backup_only": policy_backup_only, "rule": policy_rule}


def run_policy(bank, eps, cfg, name, i, shield=False, **kw):
    env = StreetEnv(bank, eps, cfg, train=False, shield=shield, **kw)
    obs, _ = env.reset(options={"episode": i}); done, tot = False, 0.0
    while not done:
        obs, r, done, _, _ = env.step(POLICIES[name](env, obs)); tot += r
    return {**env.metrics(), "reward_per_week": tot, "episode": i}


# ---------- perfect foresight (bound) ----------
def run_perfect(bank, eps, cfg, i, time_limit=600):
    env = StreetEnv(bank, eps, cfg, train=False, shield=False); env.reset(options={"episode": i})
    ess, oth, lon, pv = true_inputs(env, 0, env.T, cfg); nE, nO, nL, sur, pv_o = split_pv(pv, ess, oth, lon)
    plan = solve_milp(nE, nO, nL, sur, peak_mask(np.arange(env.T), cfg), env.soc, cfg, time_limit=time_limit)
    done, tot = False, 0.0
    while not done:
        t = env.t; _, r, done, _, _ = env.step_cont(*to_action(plan, t, oth[t], pv_o[t], cfg)); tot += r
    return {**env.metrics(), "reward_per_week": tot, "episode": i, "pf_mip_gap": plan["mip_gap"], "pf_time_limited": plan["status"] == 1}


# ---------- MPC ----------
def _interp(h, pts, vals):
    return np.array([np.interp(h, pts, vals[:, j]) for j in range(vals.shape[1])])


def mpc_inputs(env, t, H, cfg):
    """Expected street inputs for steps t .. t+H-1 from information available at t (cut and demand forecasts up to +4 h,
    then the 7-day same-time cut rate and yesterday's demand; solar = yesterday, now = last completed hour)."""
    idx = t + np.arange(H); idx_c = np.minimum(idx, env.T - 1)
    past = idx_c - SPD; have_y = past >= 0
    d_now, off_now = env.demand[t], env.off[t]
    d = np.where(have_y[:, None], env.demand[np.maximum(past, 0)], d_now[None, :])
    pv = np.where(have_y, env.pv[np.maximum(past, 0)], env.pv_prev[idx_c] * cfg.pv_kwp)
    p = np.repeat(env.cut_rate[(t // SPD) * SPD + idx % SPD][:, None], len(d_now), 1)
    pts = np.array([0, 4, 8, 16])
    P = np.vstack([off_now.astype(float), env.cut_p[t].T]); L = np.vstack([d_now, env.load_fc[t].T])
    for h in range(1, min(17, H)):
        p[h] = _interp(h, pts, P); d[h] = _interp(h, pts, L)
    p[0] = off_now.astype(float); d[0] = d_now
    pv[0] = env.pv_obs[t] * cfg.pv_kwp
    ess = cfg.essential_share * (p * d).sum(1); oth = (1 - cfg.essential_share) * (p * d).sum(1); lon = ((1 - p) * d).sum(1)
    return ess, oth, lon, pv, idx


def reserve_kwh(env, z):
    """MPC-CC reserve on stored energy: R = need_mean + z * need_sd (in usable energy), as stored kWh above the floor."""
    c, t = env.cfg, env.t
    need, _ = env.need_avail()
    off = env.off[t]; p = env.cut_p[t]; p1, p2, p4 = p[:, 0], p[:, 1], p[:, 2]
    hours = np.where(off, 0.5 * (1 + p1) + 0.5 * (p1 + p2) + (p2 + p4), 0.5 * p1 + 0.5 * (p1 + p2) + (p2 + p4))
    load = env.load_fc[t].mean(1)
    sd = float(np.sqrt(((c.essential_share * load * hours) ** 2 * p1 * (1 - p1)).sum()))
    usable = c.batt_kwh * (1 - c.soc_min) * c.eta
    R = min(need + z * sd, usable)
    return c.soc_min * c.batt_kwh + R / c.eta


def run_mpc(bank, eps, cfg, i, kind="v2", eps_cc=None, H=96):
    """kind "v2": MPC v2 (forecasts, shield on). kind "cc": MPC-CC (forecasts, reserve with risk eps_cc, no shield)."""
    env = StreetEnv(bank, eps, cfg, train=False, shield=(kind == "v2")); env.reset(options={"episode": i})
    z = norm.ppf(1 - eps_cc) if kind == "cc" else None
    done, tot = False, 0.0
    while not done:
        t = env.t
        ess, oth, lon, pv, idx = mpc_inputs(env, t, H, cfg)
        nE, nO, nL, sur, pv_o = split_pv(pv, ess, oth, lon)
        cut_now = bool(env.off[t].any())
        res = (reserve_kwh(env, z), 16, cfg.w_ess) if kind == "cc" else None
        plan = solve_milp(nE, nO, nL, sur, peak_mask(idx, cfg), env.soc, cfg, terminal_value=cfg.w_peak,
                          force_no_grid_charge_first=cut_now, mip_rel_gap=1e-2, reserve=res)
        u, frac = to_action(plan, 0, oth[0], pv_o[0], cfg)
        if cut_now and u < 0:
            u = 0.0
        _, r, done, _, _ = env.step_cont(u, frac); tot += r
    return {**env.metrics(), "reward_per_week": tot, "episode": i}


# ---------- the expert (perfect foresight re-planned from the state reached) ----------
def expert_plan(env, t, soc, cfg):
    ess, oth, lon, pv = true_inputs(env, t, env.T - t, cfg)
    nE, nO, nL, sur, pv_o = split_pv(pv, ess, oth, lon)
    plan = solve_milp(nE, nO, nL, sur, peak_mask(np.arange(t, env.T), cfg), soc, cfg, n_binary=32, mip_rel_gap=1e-3)
    return plan, oth, pv_o


def discrete(u, frac):
    return int(np.argmin(np.abs(BATT_LEVELS - u))) + NL * int(frac >= 0.5)


def demonstrate(bank, eps, cfg, i, replan=16):
    """Expert on one episode (no shield), re-planning every `replan` steps. Returns metrics, the 23-input states and actions."""
    env = StreetEnv(bank, eps, cfg, train=False, shield=False)
    obs, _ = env.reset(options={"episode": i}); done, tot, O, A = False, 0.0, [], []
    while not done:
        t = env.t
        if t % replan == 0:
            plan, oth, pv_o = expert_plan(env, t, env.soc, cfg); t0 = t
        h = t - t0
        a = discrete(*to_action(plan, h, oth[h], pv_o[h], cfg))
        O.append(obs); A.append(a)
        obs, r, done, _, _ = env.step(a); tot += r
    return {**env.metrics(), "reward_per_week": tot, "episode": i}, np.asarray(O, np.float32), np.asarray(A, np.int64)


def expert_label(env, t, soc, cfg):
    plan, oth, pv_o = expert_plan(env, t, soc, cfg)
    return discrete(*to_action(plan, 0, oth[0], pv_o[0], cfg))
