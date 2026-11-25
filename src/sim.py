"""Nirantar street simulator with the safety shield. Written from notebooks/03_simulator.ipynb.

Physics and reward are those of versions 1-3 (archived src/street_env.py), unchanged; the shield is the version 2 shield with a
4 h reserve (archived src/v2.py, duration model off). Only the data access differs: a "bank" holds one array per region, and an
episode is (region, first row, homes), so overlapping training windows share memory.

Physics per 15-minute step:
  1. Homes in a cut are fed from the shared backup line: solar first, then the battery. Essentials (30% of demand)
     first, shared in proportion; the rest of their load only as far as the backup share allows.
  2. Leftover solar feeds homes with grid supply; surplus charges the battery; the rest is curtailed (no export).
  3. The controller's battery action: discharge to grid-connected homes, or charge from the grid (never both in a step).
Shield (before every action): need = expected essential energy of the street over the next 4 h; avail = usable battery energy.
  1. discharge to grid-connected homes is reduced (1 -> 0.5 -> 0) until avail - discharge >= need
  2. non-essential load of homes in a cut -> essentials only if avail - that energy < need
  3. if avail < need, no home is in a cut and it is not 18-22 h: charge from the grid at full power
"""
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import json, pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DT, SPD, WEEK = 0.25, 96, 672
BATT_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
NL = len(BATT_LEVELS)
N_OBS = 23
UNSERVED_KWH = 1e-3          # a home-step counts as "essentials off" if short by more than 1 Wh
WEAR_INR_PER_KWH = 1.18      # BNEF 2025 pack price x LFP cycle life (results/config.json)
HORIZON_H = 4.0


@dataclass
class StreetConfig:
    n_homes: int = 10
    pv_kwp_per_home: float = 0.5
    batt_kwh_per_home: float = 0.75
    batt_kw_per_home: float = 0.25
    eta: float = 0.95
    soc_min: float = 0.10
    essential_share: float = 0.30
    peak_hours: tuple = (18, 22)
    peak_cap_kw_per_home: float = 0.5
    w_ess: float = 10.0
    w_other: float = 3.0
    w_peak: float = 1.0
    w_peak_excess: float = 3.0
    w_wear: float = 0.20
    w_fair: float = 5.0
    reward_scale: float = 0.1
    use_forecasts: bool = True

    @property
    def pv_kwp(self): return self.pv_kwp_per_home * self.n_homes
    @property
    def batt_kwh(self): return self.batt_kwh_per_home * self.n_homes
    @property
    def batt_kw(self): return self.batt_kw_per_home * self.n_homes
    @property
    def cap_kw(self): return self.peak_cap_kw_per_home * self.n_homes


def load_cfg(root):
    c = dict(json.loads((Path(root) / "results" / "config.json").read_text())["config"])
    c["peak_hours"] = tuple(c["peak_hours"])
    return StreetConfig(**c)


def jain(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 1.0
    if (x ** 2).sum() == 0:
        return float("nan")
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))


_BANKS = {}


def load_bank(path):
    """Region arrays (cached per process). bank[region] = {"start": first timestamp, "homes": [...], "demand": [T, N], "cut": [T, N] bool,
    "pv", "pv_obs", "pv_prev", "cut_rate": [T], "cut_p": [T, N, 3], "load_fc": [T, N, 3], optional "cut_p_ref", "load_fc_ref"}."""
    path = str(path)
    if path not in _BANKS:
        _BANKS[path] = pickle.loads(Path(path).read_bytes())
    return _BANKS[path]


def episode_arrays(bank, ep, cut_key="cut_p", load_key="load_fc"):
    """Arrays of one week for the episode's homes. ep = {"region", "t0", "pool_idx", optional "homes_idx"}."""
    B = bank[ep["region"]]; a = ep["t0"]; b = a + WEEK
    return {"demand": B["demand"][a:b], "cut": B["cut"][a:b], "pv": B["pv"][a:b], "pv_obs": B["pv_obs"][a:b], "pv_prev": B["pv_prev"][a:b],
            "cut_rate": B["cut_rate"][a:b], "cut_p": B[cut_key][a:b], "load_fc": B[load_key][a:b], "start": B["start"] + np.timedelta64(15 * a, "m")}


class StreetEnv(gym.Env):
    """obs: 23 inputs (v2 state). stack > 1 returns the last `stack` states concatenated (oldest first), for the GRU policy."""
    metadata = {"render_modes": []}

    def __init__(self, bank, episodes, cfg, train=True, seed=0, shield=True, stack=1, cut_key="cut_p", load_key="load_fc"):
        super().__init__()
        self.bank, self.episodes, self.cfg, self.train = bank, episodes, cfg, train
        self.use_shield, self.stack, self.cut_key, self.load_key = shield, stack, cut_key, load_key
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(NL * 2)
        self.observation_space = spaces.Box(-1.0, 5.0, shape=(N_OBS * stack,), dtype=np.float32)
        self._i = -1
        self.shield_steps = 0; self.steps = 0

    # ---------- episodes ----------
    def _pick(self, options):
        if self.train:
            ep = self.episodes[self.rng.integers(len(self.episodes))]
            cols = np.asarray(ep["pool_idx"])[np.sort(self.rng.choice(len(ep["pool_idx"]), self.cfg.n_homes, replace=False))]
        else:
            self._i = options["episode"] if options and "episode" in options else (self._i + 1) % len(self.episodes)
            ep = self.episodes[self._i]; cols = np.asarray(ep["homes_idx"])
        return ep, cols

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ep, cols = self._pick(options)
        A = episode_arrays(self.bank, ep, self.cut_key, self.load_key); c = self.cfg
        self.ep, self.cols = ep, cols
        self.demand = A["demand"][:, cols].astype(np.float64)
        self.off = A["cut"][:, cols].astype(bool)
        self.pv = A["pv"].astype(np.float64) * c.pv_kwp
        self.pv_obs = A["pv_obs"].astype(np.float64); self.pv_prev = A["pv_prev"].astype(np.float64)
        self.cut_rate = A["cut_rate"].astype(np.float64)
        self.cut_p = A["cut_p"][:, cols, :].astype(np.float64); self.load_fc = A["load_fc"][:, cols, :].astype(np.float64)
        self.start = A["start"]
        self.T = self.demand.shape[0]; self.t = 0
        self.soc = (self.rng.uniform(0.3, 0.9) if self.train else 0.5) * c.batt_kwh
        if options and "soc" in options:
            self.soc = float(options["soc"]) * c.batt_kwh
        n = c.n_homes
        self.cut_demand, self.cut_served = np.zeros(n), np.zeros(n)
        self.log = {k: [] for k in ["ess_uns", "oth_uns", "import", "pv_used", "pv_avail", "dis", "ch", "soc", "ess_uns_homes"]}
        self.shield_steps = 0; self.steps = 0
        o = self._obs_one(); self.hist = deque([o] * self.stack, maxlen=self.stack)
        return self._stacked(), {}

    def restart_history(self):
        """After changing the state by hand (e.g. the battery charge), rebuild the observation history from the current state."""
        o = self._obs_one(); self.hist = deque([o] * self.stack, maxlen=self.stack)
        return self._stacked()

    # ---------- observation (identical to v1-v3, 23 inputs) ----------
    def _obs_one(self):
        c, t = self.cfg, min(self.t, self.T - 1)
        d, off = self.demand[t], self.off[t]
        k = t % SPD
        dow = (int((self.start - np.datetime64("1970-01-05")) / np.timedelta64(1, "D")) + t // SPD) % 7
        tod = 2 * np.pi * k / SPD
        day0 = (t // SPD) * SPD
        cr = lambda h: self.cut_rate[day0 + (t + h) % SPD]
        served = self.cut_served / np.maximum(self.cut_demand, 1e-9)
        worst = served[self.cut_demand > 1e-6].min() if (self.cut_demand > 1e-6).any() else 1.0
        o = [self.soc / c.batt_kwh if c.batt_kwh > 0 else 0.0, self.pv_obs[t], self.pv_prev[t],
             d[~off].sum() / c.n_homes, c.essential_share * d[off].sum() / c.n_homes, d[off].sum() / c.n_homes, off.mean(),
             np.sin(tod), np.cos(tod), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
             cr(0), cr(4), cr(8), worst]
        p, lf = self.cut_p[t], self.load_fc[t]                   # [homes, 3] for +1 h, +2 h, +4 h
        o += list(p.mean(0)) + list((p[:, :2] * c.essential_share * lf[:, :2]).mean(0)) + list(lf.mean(0))
        return np.asarray(o, dtype=np.float32)

    def _stacked(self):
        return np.concatenate(list(self.hist)).astype(np.float32) if self.stack > 1 else self.hist[-1]

    # ---------- shield (v2, 4 h, no duration model) ----------
    def need_avail(self):
        c, t = self.cfg, self.t
        off = self.off[t]; p = self.cut_p[t]
        p1, p2, p4 = p[:, 0], p[:, 1], p[:, 2]
        hours = 0.5 * p1 + 0.5 * (p1 + p2) + (p2 + p4)                     # expected cut hours in the next 4 h
        if off.any():
            hours = np.where(off, 0.5 * (1 + p1) + 0.5 * (p1 + p2) + (p2 + p4), hours)
        load = self.load_fc[t].mean(1)
        usable = c.batt_kwh * (1 - c.soc_min) * c.eta
        need = min(float((c.essential_share * load * hours).sum()), usable)
        avail = max(self.soc - c.soc_min * c.batt_kwh, 0.0) * c.eta
        return need, avail

    def shield_cont(self, u, frac):
        c, t = self.cfg, self.t
        need, avail = self.need_avail()
        off = self.off[t]; P = c.batt_kw
        if u > 0:
            u = float(np.clip((avail - need) / (P * DT), 0.0, u))
        if frac > 0 and off.any():
            draw = (1 - c.essential_share) * self.demand[t][off].sum() * DT
            if draw > 1e-12:
                frac = float(np.clip((avail - need) / draw, 0.0, frac))
        hour = (t % SPD) / 4
        if avail < need and not off.any() and not (c.peak_hours[0] <= hour < c.peak_hours[1]):
            u = -1.0
        return u, frac

    def shield_action(self, a):
        c, t = self.cfg, self.t
        u, full = BATT_LEVELS[a % NL], a // NL
        need, avail = self.need_avail()
        off = self.off[t]; P = c.batt_kw
        while u > 0 and avail - u * P * DT < need:
            u -= 0.5
        if full and off.any():
            draw = (1 - c.essential_share) * self.demand[t][off].sum() * DT
            if avail - draw < need:
                full = 0
        hour = (t % SPD) / 4
        if avail < need and not off.any() and not (c.peak_hours[0] <= hour < c.peak_hours[1]):
            u = -1.0
        return int(np.where(np.isclose(BATT_LEVELS, u))[0][0]) + NL * int(full)

    # ---------- step ----------
    def step(self, action):
        a = int(action)
        if self.use_shield:
            b = self.shield_action(a); self.shield_steps += int(b != a); a = b
        return self._physics(BATT_LEVELS[a % NL], 1.0 if a // NL == 1 else 0.0)

    def step_cont(self, u, oth_frac, shield=None):
        if (self.use_shield if shield is None else shield):
            u2, f2 = self.shield_cont(u, oth_frac)
            self.shield_steps += int(abs(u2 - u) > 1e-9 or abs(f2 - oth_frac) > 1e-9); u, oth_frac = u2, f2
        return self._physics(u, oth_frac)

    def _physics(self, u, oth_frac):
        c, t = self.cfg, self.t
        u = float(np.clip(u, -1, 1))
        if abs(u) < 1e-9:
            u = 0.0
        d, off, pv = self.demand[t], self.off[t], self.pv[t]
        ess = c.essential_share * d * off
        oth = (1 - c.essential_share) * d * off * float(np.clip(oth_frac, 0, 1))
        e_room = max(self.soc - c.soc_min * c.batt_kwh, 0.0) * c.eta / DT
        P = c.batt_kw

        def serve(need, avail):
            tot = need.sum()
            if tot <= 0:
                return np.zeros_like(need), avail
            got = min(tot, avail)
            return need * (got / tot), avail - got
        ess_pv, pv_left = serve(ess, pv)
        oth_pv, pv_left = serve(oth, pv_left)
        b_avail = min(e_room, P) if u >= 0 else 0.0
        ess_b, b_left = serve(ess - ess_pv, b_avail)
        oth_b, b_left = serve(oth - oth_pv, b_left)
        b_out = b_avail - b_left
        ess_got, oth_got = ess_pv + ess_b, oth_pv + oth_b
        load_on = d[~off].sum()
        pv_on = min(pv_left, load_on); pv_surplus = pv_left - pv_on
        b_on = min(u * P, P - b_out, max(e_room - b_out, 0.0), load_on - pv_on) if u > 0 else 0.0
        room = c.batt_kwh - self.soc
        ch_pv = max(min(pv_surplus, P - b_out - b_on, room / (c.eta * DT)), 0.0)
        ch_grid = max(min(-u * P, P - ch_pv, room / (c.eta * DT) - ch_pv), 0.0) if u < 0 else 0.0
        dis, ch = b_out + b_on, ch_pv + ch_grid
        self.soc = float(np.clip(self.soc + (ch * c.eta - dis / c.eta) * DT, 0.0, c.batt_kwh))
        grid_import = load_on - pv_on - b_on + ch_grid
        ess_uns = (ess - ess_got) * DT
        oth_uns = ((1 - c.essential_share) * d * off - oth_got) * DT
        self.cut_demand += d * off * DT
        self.cut_served += (ess_got + oth_got) * DT
        hour = (t % SPD) / 4
        in_peak = c.peak_hours[0] <= hour < c.peak_hours[1]
        excess = max(grid_import - c.cap_kw, 0.0) if in_peak else 0.0
        r = -(c.w_ess * ess_uns.sum() + c.w_other * oth_uns.sum() + c.w_peak * grid_import * DT * in_peak
              + c.w_peak_excess * excess * DT + c.w_wear * dis * DT)
        L = self.log
        L["ess_uns"].append(ess_uns.sum()); L["oth_uns"].append(oth_uns.sum()); L["import"].append(grid_import)
        L["pv_used"].append((pv - max(pv_surplus - ch_pv, 0)) * DT); L["pv_avail"].append(pv * DT)
        L["dis"].append(dis * DT); L["ch"].append(ch * DT); L["soc"].append(self.soc); L["ess_uns_homes"].append(ess_uns > UNSERVED_KWH)
        self.t += 1; self.steps += 1
        done = self.t >= self.T
        if done:
            mask = self.cut_demand > 1e-6
            if mask.any():
                j = np.nan_to_num(jain(self.cut_served[mask] / self.cut_demand[mask]), nan=1.0)
                r += c.w_fair * (j - 1.0) * c.n_homes
        self.hist.append(self._obs_one())
        return self._stacked(), r * c.reward_scale, done, False, {}

    def metrics(self):
        c, L = self.cfg, self.log
        days = self.T / SPD
        imp = np.array(L["import"]).reshape(-1, SPD)[:, c.peak_hours[0] * 4:c.peak_hours[1] * 4]
        mask = self.cut_demand > 1e-6
        per = lambda x: float(x / c.n_homes / days * 30)
        return {
            "LOLH_ess_h_per_home_month": per(np.array(L["ess_uns_homes"]).sum() * DT),
            "ENS_ess_kwh_per_home_month": per(np.sum(L["ess_uns"])),
            "cut_demand_served_pct": float(100 * self.cut_served.sum() / max(self.cut_demand.sum(), 1e-9)),
            "jain_fairness": jain(self.cut_served[mask] / self.cut_demand[mask]) if mask.any() else 1.0,
            "evening_peak_import_kw": float(imp.max(axis=1).mean()),
            "evening_import_kwh_per_day": float(imp.sum() * DT / days),
            "evening_excess_kwh_per_day": float(np.clip(imp - c.cap_kw, 0, None).sum() * DT / days),
            "solar_used_pct": float(100 * np.sum(L["pv_used"]) / max(np.sum(L["pv_avail"]), 1e-9)),
            "battery_cycles_per_day": float(np.sum(L["dis"]) / max(c.batt_kwh, 1e-9) / days),
            "wear_cost_inr_per_home_month": per(np.sum(L["dis"]) * WEAR_INR_PER_KWH),
            "shield_rate_pct": 100.0 * self.shield_steps / max(self.steps, 1),
        }


def evaluate(policy, bank, episodes, cfg, env_kw, sb3=True):
    """Every episode at once (batched). policy = an SB3 model (deterministic) or a function (env, obs) -> action.
    Returns one row per episode (metrics and the reward per week)."""
    envs = [StreetEnv(bank, episodes, cfg, train=False, **env_kw) for _ in episodes]
    obs = np.stack([e.reset(options={"episode": i})[0] for i, e in enumerate(envs)])
    tot, active = np.zeros(len(envs)), np.ones(len(envs), bool)
    while active.any():
        act = policy.predict(obs, deterministic=True)[0] if sb3 else [policy(e, obs[i]) if active[i] else 0 for i, e in enumerate(envs)]
        for i in np.where(active)[0]:
            o, r, done, _, _ = envs[i].step(int(act[i])); tot[i] += r; obs[i] = o
            if done:
                active[i] = False
    return [{**e.metrics(), "reward_per_week": tot[i], "episode": i} for i, e in enumerate(envs)]
