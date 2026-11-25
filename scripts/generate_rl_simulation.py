"""Nirantar - animated RL simulation video (50 s, 1920x1080, 30 fps) built from a real rollout of the frozen GRU-PPO controller.

Stage 1 (rollout): the final controller (GRU, seed 2) drives the street simulator (src/sim.py, safety shield on) on every validation week.
  Every energy flow, the policy's action probabilities, the critic value, the shield decision and the forecasts are recorded per
  15-minute step. One 24-hour day is chosen automatically (cuts + midday solar charging + evening discharge) and saved to
  results/rl_simulation_rollout.csv. The recorded physics are checked against the unmodified simulator (same reward, bit for bit).
Stage 2 (render): 1,500 frames are drawn with matplotlib in parallel and encoded with ffmpeg to Report/yuva_yodha_rl_simulation.mp4.

Validation data only: the Pune test set is never touched.   Run:  .venv\\Scripts\\python.exe scripts\\generate_rl_simulation.py
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT_MP4 = ROOT / "Report" / "yuva_yodha_rl_simulation.mp4"
ROLLOUT_CSV = ROOT / "results" / "rl_simulation_rollout.csv"
NET, SEED = "GRU", 2
W, H, FPS, SECONDS = 1920, 1080, 30, 50
N_FRAMES = FPS * SECONDS
INTRO, OUTRO = 6 * FPS, 6 * FPS                     # 0-6 s intro, 44-50 s outro
SIM_FRAMES = N_FRAMES - INTRO - OUTRO               # 38 s of simulation
SPD = 96


# ============================================================ stage 1: rollout
def rollout():
    import torch
    from stable_baselines3 import PPO
    from sim import StreetEnv, load_cfg, load_bank, BATT_LEVELS, NL, DT
    from learn import env_kw

    class RecEnv(StreetEnv):
        """StreetEnv with every intermediate flow of _physics recorded (physics copied unchanged from src/sim.py)."""
        def step(self, action):
            a = int(action); self.proposed = a
            if self.use_shield:
                b = self.shield_action(a); self.shield_steps += int(b != a); a = b
            self.applied = a
            return self._physics(BATT_LEVELS[a % NL], 1.0 if a // NL == 1 else 0.0)

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
            self.rec = dict(u=u, full=oth_frac, pv=pv, load_on=load_on, load_cut=float((d * off).sum()), ess_need=float(ess.sum()),
                            pv_cut=float(ess_pv.sum() + oth_pv.sum()), pv_on=pv_on, b_cut=float(b_out), b_on=float(b_on), ch_pv=ch_pv,
                            ch_grid=ch_grid, curtailed=max(pv_surplus - ch_pv, 0.0),
                            ess_short=(ess - ess_got).tolist(), full_short=(d * off - ess_got - oth_got).tolist(), off=off.astype(int).tolist())
            # ---- original physics continues unchanged
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
            self.rec.update(grid=grid_import, soc_after=self.soc, ess_uns=float(ess_uns.sum()))
            return self._finish(r, ess_uns, oth_uns, grid_import, pv, pv_surplus, ch_pv, dis, ch)

        def _finish(self, r, ess_uns, oth_uns, grid_import, pv, pv_surplus, ch_pv, dis, ch):
            from sim import UNSERVED_KWH, jain
            c, L = self.cfg, self.log
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

    cfg = load_cfg(ROOT); bank = load_bank(ROOT / "data" / "processed" / "bank_trainval.pkl")
    val = json.loads((ROOT / "data" / "processed" / "episodes.json").read_text())["val"]
    model = PPO.load(ROOT / "models" / "rl" / f"{NET}_s{SEED}" / "best_model.zip", device="cpu")
    rows = []
    for i, ep in enumerate(val):
        env = RecEnv(bank, val, cfg, train=False, **env_kw(NET))
        obs, _ = env.reset(options={"episode": i}); done = False
        while not done:
            t = env.t
            with torch.no_grad():
                ot = torch.as_tensor(obs[None], dtype=torch.float32)
                probs = model.policy.get_distribution(ot).distribution.probs[0].numpy()
                value = float(model.policy.predict_values(ot)[0, 0])
            a = int(np.argmax(probs))                      # deterministic = most likely action (same as predict(deterministic=True))
            need, avail = env.need_avail()
            s = env._obs_one()
            cp, lf = env.cut_p[t], env.load_fc[t]
            obs, r, done, _, _ = env.step(a)
            R = env.rec
            rows.append({"week": i, "region": ep["region"], "start": str(env.start), "t": t, "hour": (t % SPD) / 4, "day": t // SPD,
                         "proposed": env.proposed, "applied": env.applied, "shield": int(env.proposed != env.applied), "value": value,
                         "reward": r, "need_kwh": need, "avail_kwh": avail, "soc_before": float(s[0]),
                         **{f"p{k}": float(probs[k]) for k in range(10)}, **{f"s{k}": float(s[k]) for k in range(23)},
                         "cutp_1": float(cp[:, 0].mean()), "cutp_2": float(cp[:, 1].mean()), "cutp_4": float(cp[:, 2].mean()),
                         "cutp_1_hi": float(cp[:, 0].max()), "cutp_2_hi": float(cp[:, 1].max()), "cutp_4_hi": float(cp[:, 2].max()),
                         "lf_1": float(lf[:, 0].sum()), "lf_2": float(lf[:, 1].sum()), "lf_4": float(lf[:, 2].sum()),
                         **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in R.items()}})
        # --- the recorded physics must equal the unmodified simulator for the same actions
        acts = [x["proposed"] for x in rows if x["week"] == i]
        ref = StreetEnv(bank, val, cfg, train=False, **env_kw(NET)); ref.reset(options={"episode": i})
        tot_ref = sum(ref.step(a)[1] for a in acts); tot_rec = sum(x["reward"] for x in rows if x["week"] == i)
        assert abs(tot_ref - tot_rec) < 1e-9, (i, tot_ref, tot_rec)
    df = pd.DataFrame(rows)
    # --- choose one day: evening discharge + midday solar charging + cut homes backed by the battery
    g = df.groupby(["week", "day"])
    score = pd.DataFrame({
        "eve_dis": g.apply(lambda x: x.loc[(x.hour >= 18) & (x.hour < 22), "b_on"].sum() * 0.25, include_groups=False),
        "mid_ch": g.apply(lambda x: x.loc[(x.hour >= 10) & (x.hour < 15), "ch_pv"].sum() * 0.25, include_groups=False),
        "cut_backup": g.apply(lambda x: (x.b_cut + x.pv_cut).sum() * 0.25, include_groups=False),
        "cut_steps": g.apply(lambda x: (x.load_cut > 0).sum(), include_groups=False),
        "uns": g.apply(lambda x: x.ess_uns.sum(), include_groups=False),
        "n": g.size()})
    score = score[(score.n == SPD) & (score.cut_steps >= 12)]
    score["score"] = (score.eve_dis / score.eve_dis.max() + score.mid_ch / score.mid_ch.max() + score.cut_backup / score.cut_backup.max()
                      - 0.5 * (score.uns > 0.05))
    wk, dy = score.score.idxmax()
    day = df[(df.week == wk) & (df.day == dy)].reset_index(drop=True)
    day.to_csv(ROLLOUT_CSV, index=False)
    print(f"rollout: {len(val)} validation weeks checked (recorded physics == simulator); chosen week {wk} ({day.region[0]}), day {dy} -> {ROLLOUT_CSV}")
    return day


# ============================================================ stage 2: render
BG, PANEL, EDGE, TXT, MUTED = "#0b1020", "#121a2e", "#26314d", "#e6ebf5", "#8a96b3"
SOLAR, BATT, GRID, HOMEC, CUT, BAD = "#ffc83d", "#3ddc97", "#ff6b6b", "#5aa9ff", "#ffa94d", "#ff4d6d"
ACT_LABEL = ["−1", "−½", "0", "+½", "+1"]
FEATS = [("Battery state of charge", 0, 0, 1), ("Solar, last completed hour", 1, 0, 0.8), ("Solar 24 h ago", 2, 0, 0.6),
         ("Load of supplied homes (kW/home)", 3, 0, 1.0), ("Essential need of cut homes", 4, 0, 0.3), ("Share of homes in a cut", 6, 0, 1),
         ("Past-7-day cut rate (now)", 11, 0, 0.6), ("Worst-served home (share)", 14, 0, 1), ("Cut probability +1 h (mean)", 15, 0, 0.6),
         ("Demand forecast +1 h (kW/home)", 20, 0, 1.0)]


def timeline(day):
    """Continuous step index tau for every simulation frame, slowing down near three key decision moments."""
    h = day.hour.values
    morning = int(np.argmax(np.where((h >= 5) & (h < 10), day.load_on + day.load_cut, -1)))
    midday = int(np.argmax(np.where((h >= 10) & (h < 15), day.ch_pv, -1)))
    evening = int(np.argmax(np.where((h >= 18) & (h < 22), day.b_on + day.b_cut, -1)))
    keys = [(morning, "Morning ramp"), (midday, "Midday solar surplus"), (evening, "Evening peak")]
    w = np.ones(SPD)
    for k, _ in keys:
        w[max(k - 3, 0):min(k + 4, SPD)] = 3.0
    cum = np.r_[0, np.cumsum(w)]; cum = cum / cum[-1] * SIM_FRAMES
    tau = np.interp(np.arange(SIM_FRAMES), cum, np.arange(SPD + 1))
    return np.clip(tau, 0, SPD - 1e-6), keys


def reason(r):
    lvl, full = ACT_LABEL[int(r.applied) % 5], int(r.applied) // 5
    n_cut = int(sum(json.loads(r.off)))
    peak = 18 <= r.hour < 22
    msgs = []
    if r.shield:
        msgs.append("Shield overrode the policy: keep ≥ 4 h of essential reserve")
    if n_cut:
        src = "solar + battery" if r.b_cut > 1e-3 and r.pv_cut > 1e-3 else ("battery" if r.b_cut > 1e-3 else "solar")
        msgs.append(f"{n_cut} home{'s' if n_cut > 1 else ''} cut → {src} backs up {'their full load' if full else 'essentials only'}")
    if r.ch_grid > 1e-3:
        msgs.append("Off-peak grid charge → building reserve before the evening")
    elif r.ch_pv > 0.05:
        msgs.append("Midday solar surplus → stored in the battery")
    if peak and r.b_on > 1e-3:
        msgs.append("Evening peak 18–22 h → battery discharges to cut grid import")
    elif peak:
        msgs.append("Evening peak → holding charge for possible cuts")
    if r.cutp_1 > 0.15:
        msgs.append(f"Forecast: cut risk {100 * r.cutp_1:.0f}% in the next hour")
    if not msgs:
        msgs.append("No surplus, no peak, no cut → hold the battery")
    return f"action: battery {lvl} · {'full backup' if full else 'essentials only'}", msgs[:3]


def render_chunk(args):
    frames, day_csv, tau, keys, outdir = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "text.color": TXT, "axes.edgecolor": EDGE,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    day = pd.read_csv(day_csv)
    hrs = day.hour.values + 0.125
    solar = day.pv.values; demand = (day.load_on + day.load_cut).values; grid = day.grid.values
    socp = 100 * day.soc_after.values / 7.5; ncut = np.array([sum(json.loads(o)) for o in day.off])
    key_steps = {k: name for k, name in keys}
    EDGES = {"solar_batt": [(22, 46), (32, 46), (32, 34), (38, 34)], "solar_homes": [(22, 53), (62, 53), (62, 42), (66, 42)],
             "batt_homes": [(50, 30), (58, 30), (58, 28), (66, 28)], "grid_homes": [(22, 9.5), (30, 9.5), (30, 6.5), (63, 6.5), (63, 26), (66, 26)],
             "grid_batt": [(22, 16), (32, 16), (32, 26), (38, 26)]}

    def along(pts, ph):
        pts = np.asarray(pts, float); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)] / seg.sum()
        out = []
        for q in ph:
            k = min(np.searchsorted(cum, q, side="right") - 1, len(seg) - 1); a = (q - cum[k]) / max(cum[k + 1] - cum[k], 1e-9)
            out.append(pts[k] + a * (pts[k + 1] - pts[k]))
        return np.array(out)

    def frame(f):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        intro, outro = f < INTRO, f >= INTRO + SIM_FRAMES
        x = tau[min(max(f - INTRO, 0), SIM_FRAMES - 1)] if not intro else 0.0
        s = min(int(x), SPD - 1); r = day.iloc[s]; frac = x - s
        clock = f"{int(r.hour):02d}:{int(round((r.hour % 1) * 60)):02d}"
        # ---------------- title bar
        fig.text(0.015, 0.955, "Nirantar · reinforcement-learning control of a shared street battery", fontsize=26, weight="bold", color=TXT)
        fig.text(0.015, 0.925, f"10 homes · 5 kWp solar · 7.5 kWh / 2.5 kW battery · frozen GRU-PPO policy (seed {SEED}) + safety shield · "
                               f"real validation day, {day.region[0]} · 1 step = 15 min", fontsize=13, color=MUTED)
        fig.text(0.985, 0.94, clock if not intro else "00:00", fontsize=40, weight="bold", color=SOLAR, ha="right", va="center", family="DejaVu Sans Mono")
        # ---------------- flow diagram
        ax = fig.add_axes([0.012, 0.30, 0.56, 0.60]); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.5, 0.5), 99, 61, boxstyle="round,pad=0,rounding_size=2", fc=PANEL, ec=EDGE, lw=1.5))
        ax.text(3, 58, "PHYSICAL ENVIRONMENT · live power flow (kW)", fontsize=13, weight="bold", color=MUTED)
        flows = {"solar_batt": r.ch_pv, "solar_homes": r.pv_on + r.pv_cut, "batt_homes": r.b_on + r.b_cut,
                 "grid_homes": max(r.load_on - r.pv_on - r.b_on, 0.0), "grid_batt": r.ch_grid}
        cols = {"solar_batt": SOLAR, "solar_homes": SOLAR, "batt_homes": BATT, "grid_homes": GRID, "grid_batt": GRID}
        tglob = f / FPS
        for k, pts in EDGES.items():
            v = flows[k]; P_ = np.array(pts, float)
            on = v > 0.02 and not intro
            ax.plot(P_[:, 0], P_[:, 1], color=cols[k] if on else EDGE, lw=1.5 + 3.5 * min(v / 2.5, 1.5) if on else 1.2, alpha=0.55 if on else 0.6,
                    ls="-" if on else (0, (3, 3)), solid_capstyle="round", solid_joinstyle="round", zorder=1)
            if on:
                n = 9; ph = (np.arange(n) / n + tglob * (0.12 + 0.18 * min(v, 3))) % 1
                q = along(P_, ph)
                ax.scatter(q[:, 0], q[:, 1], s=30 + 60 * min(v, 3), color=cols[k], edgecolors="white", linewidths=0.6, zorder=3, alpha=0.95)
                seg = np.linalg.norm(np.diff(P_, axis=0), axis=1); j = int(np.argmax(seg)); m = (P_[j] + P_[j + 1]) / 2
                ax.text(m[0], m[1] + 1.9, f"{v:.2f} kW", fontsize=12, color=cols[k], ha="center", weight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="none", alpha=0.85))
        # nodes
        def node(xy, title, sub, col):
            ax.add_patch(FancyBboxPatch((xy[0] - 8, xy[1] - 6), 16, 12, boxstyle="round,pad=0,rounding_size=1.5", fc=BG, ec=col, lw=2.5, zorder=5))
            ax.text(xy[0], xy[1] + 2, title, ha="center", fontsize=14, weight="bold", color=col, zorder=6)
            ax.text(xy[0], xy[1] - 2.5, sub, ha="center", fontsize=11.5, color=TXT, zorder=6)
        node((14, 50), "SOLAR PV", f"{r.pv:.2f} kW" + (f"\ncurtailed {r.curtailed:.2f}" if r.curtailed > 0.05 else ""), SOLAR)
        peak = 18 <= r.hour < 22
        node((14, 14), "UTILITY GRID", f"import {r.grid:+.2f} kW", GRID)
        ax.text(14, 4.2, "EVENING PEAK 18–22 h · imports penalised" if peak else "off-peak · no peak penalty", ha="center", fontsize=10.5,
                weight="bold", color=BG if peak else MUTED, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc=GRID if peak else PANEL, ec=GRID if peak else EDGE))
        # battery gauge
        soc = (r.soc_before if intro else r.soc_before + frac * (r.soc_after / 7.5 - r.soc_before))
        ax.add_patch(FancyBboxPatch((38, 20), 12, 20, boxstyle="round,pad=0,rounding_size=1", fc=BG, ec=BATT, lw=2.5, zorder=5))
        ax.add_patch(Rectangle((42, 40), 4, 1.4, fc=BATT, ec="none", zorder=5))
        ax.add_patch(Rectangle((39, 21), 10, 18 * soc, fc=BATT, ec="none", alpha=0.85, zorder=6))
        ax.plot([38.5, 49.5], [21 + 18 * 0.1] * 2, color=BAD, lw=1.5, ls=":", zorder=7)
        ax.text(44, 44, "BATTERY", ha="center", fontsize=14, weight="bold", color=BATT)
        ax.text(44, 28.5, f"{100 * soc:.0f}%", ha="center", fontsize=18, weight="bold", color=TXT, zorder=8)
        ax.text(44, 16.2, f"{7.5 * soc:.2f} kWh", ha="center", fontsize=11.5, color=TXT)
        state = "charging" if r.ch_pv + r.ch_grid > 1e-3 else ("discharging" if r.b_on + r.b_cut > 1e-3 else "idle")
        ax.text(44, 13.2, state, ha="center", fontsize=11.5, color=BATT if state != "idle" else MUTED, style="italic")
        # homes
        off = json.loads(r.off); ess_short = json.loads(r.ess_short); full_short = json.loads(r.full_short)
        ax.text(82, 50.5, "STREET · 10 HOMES", ha="center", fontsize=14, weight="bold", color=HOMEC)
        for i in range(10):
            cx, cy = 70 + (i % 5) * 6, 42 - (i // 5) * 14
            if not off[i] or intro: col, lab = HOMEC, "grid"
            elif ess_short[i] > 1e-4: col, lab = BAD, "no power"
            elif full_short[i] > 1e-4: col, lab = CUT, "essentials"
            else: col, lab = BATT, "full backup"
            ax.add_patch(Rectangle((cx - 2.2, cy - 3), 4.4, 4.2, fc=col, ec="white", lw=0.8, zorder=5, alpha=0.95))
            ax.add_patch(Polygon([[cx - 2.8, cy + 1.2], [cx, cy + 4], [cx + 2.8, cy + 1.2]], fc=col, ec="white", lw=0.8, zorder=5))
            ax.text(cx, cy - 5.2, lab, ha="center", fontsize=8.5, color=col)
        ax.text(82, 17.5, f"supplied by grid: load {r.load_on:.2f} kW", ha="center", fontsize=11, color=HOMEC)
        ax.text(82, 14.6, f"in a cut: {int(sum(off))} home(s), demand {r.load_cut:.2f} kW", ha="center", fontsize=11, color=CUT)
        for k2, (lab, col) in enumerate([("on grid", HOMEC), ("cut · full backup", BATT), ("cut · essentials only", CUT), ("cut · unserved", BAD)]):
            lx = [36, 50, 66, 84][k2]
            ax.add_patch(Rectangle((lx, 2.6), 1.6, 1.6, fc=col, ec="none"))
            ax.text(lx + 2.2, 2.7, lab, fontsize=10, color=MUTED, va="bottom")
        # ---------------- RL brain panel
        bx = fig.add_axes([0.585, 0.30, 0.405, 0.60]); bx.set_xlim(0, 100); bx.set_ylim(0, 100); bx.axis("off")
        bx.add_patch(FancyBboxPatch((0.5, 0.5), 99, 99, boxstyle="round,pad=0,rounding_size=2", fc=PANEL, ec=EDGE, lw=1.5))
        bx.text(3, 95, "RL BRAIN · s_t → forecasts → GRU-PPO π(a|s) → shield → a_t", fontsize=13, weight="bold", color=MUTED)
        bx.text(3, 89.5, "observation s_t  (10 of 23 inputs · GRU reads last 8 = 2 h)", fontsize=11, color=TXT)
        for j, (name, k, lo, hi) in enumerate(FEATS):
            y = 85 - j * 3.6; v = float(np.clip((r[f"s{k}"] - lo) / (hi - lo), 0, 1))
            bx.text(3, y, name, fontsize=9.6, color=MUTED, va="center")
            bx.add_patch(Rectangle((36, y - 1.1), 14, 2.2, fc=BG, ec=EDGE, lw=0.6))
            bx.add_patch(Rectangle((36, y - 1.1), 14 * v, 2.2, fc=HOMEC, ec="none", alpha=0.9))
            bx.text(51, y, f"{r[f's{k}']:.2f}", fontsize=9.6, color=TXT, va="center")
        # forecast cone
        fx = fig.add_axes([0.84, 0.655, 0.12, 0.165], facecolor=BG)
        hz = np.array([0, 1, 2, 4]); cut_now = float(r.s6)
        mean = np.array([cut_now, r.cutp_1, r.cutp_2, r.cutp_4]); hi = np.array([cut_now, r.cutp_1_hi, r.cutp_2_hi, r.cutp_4_hi])
        fx.fill_between(hz, mean * 0, hi, color=CUT, alpha=0.25, lw=0); fx.plot(hz, mean, color=CUT, lw=2, marker="o", ms=4)
        fx.set_ylim(0, 1); fx.set_xticks(hz); fx.set_xticklabels(["now", "+1h", "+2h", "+4h"], fontsize=9); fx.tick_params(labelsize=8.5)
        fx.set_title("forecast look-ahead\ncut risk (orange) · demand kW (blue)", fontsize=9.5, color=TXT, pad=3)
        fx2 = fx.twinx(); fx2.plot(hz, [r.load_on + r.load_cut, r.lf_1, r.lf_2, r.lf_4], color=HOMEC, lw=1.5, ls="--", marker="s", ms=3)
        fx2.set_ylim(0, max(6, (day.load_on + day.load_cut).max() * 1.1)); fx2.tick_params(labelsize=8.5, colors=HOMEC)
        for a_ in (fx, fx2):
            for sp in a_.spines.values(): sp.set_color(EDGE)
        # policy distribution
        px = fig.add_axes([0.61, 0.475, 0.23, 0.105], facecolor=BG)
        probs = np.array([r[f"p{k}"] for k in range(10)])
        colors = [BATT if k == int(r.proposed) else ("#4b5675" if k < 5 else "#5d4b75") for k in range(10)]
        px.bar(np.arange(10), probs, color=colors, width=0.75)
        px.set_xticks(np.arange(10)); px.set_xticklabels([f"{l}\n{'E' if k < 5 else 'F'}" for k, l in enumerate(ACT_LABEL * 2)], fontsize=8.5)
        px.set_ylim(0, 1); px.tick_params(labelsize=8.5)
        px.set_title("PPO actor π(a|s) · level (−charge … +discharge) × backup (E/F)", fontsize=9.5, color=TXT, pad=3, loc="left")
        for sp in px.spines.values(): sp.set_color(EDGE)
        bx.text(64.5, 47, f"critic V(s) = {r.value:+.2f}", fontsize=11, color=MUTED)
        # shield + reserve
        bx.text(64.5, 42, "SAFETY SHIELD", fontsize=11, weight="bold", color=GRID if r.shield else BATT)
        bx.text(64.5, 38.2, ("OVERRIDE  " if r.shield else "pass  ") + f"a = {ACT_LABEL[int(r.proposed) % 5]}{'F' if r.proposed >= 5 else 'E'}"
                f" → {ACT_LABEL[int(r.applied) % 5]}{'F' if r.applied >= 5 else 'E'}", fontsize=10.5, color=TXT)
        need, avail = r.need_kwh, r.avail_kwh
        bx.text(64.5, 34, f"4 h essential need {need:.2f} kWh", fontsize=9.8, color=MUTED)
        bx.add_patch(Rectangle((64.5, 29.5), 32, 2.4, fc=BG, ec=EDGE)); bx.add_patch(Rectangle((64.5, 29.5), 32 * min(avail / 6.5, 1), 2.4, fc=BATT, alpha=0.85))
        bx.plot([64.5 + 32 * min(need / 6.5, 1)] * 2, [28.9, 32.5], color=BAD, lw=2)
        bx.text(64.5, 25.6, f"usable stored {avail:.2f} kWh", fontsize=9.8, color=MUTED)
        # decision callout
        head, msgs = reason(r)
        bx.add_patch(FancyBboxPatch((3, 2.5), 94, 16, boxstyle="round,pad=0,rounding_size=1.5", fc=BG, ec=SOLAR, lw=1.8))
        bx.text(5, 14.6, head if not intro else "waiting for 00:00 …", fontsize=11.5, weight="bold", color=SOLAR)
        for k2, m in enumerate(msgs if not intro else []):
            bx.text(5, 10.6 - k2 * 3.3, "• " + m, fontsize=10.8, color=TXT)
        # ---------------- bottom time series
        tx = fig.add_axes([0.045, 0.055, 0.91, 0.2], facecolor=PANEL)
        upto = s + 1 if not intro else 0
        tx.axvspan(18, 22, color=GRID, alpha=0.08, lw=0)
        for k3 in range(upto):
            if ncut[k3]: tx.axvspan(day.hour[k3], day.hour[k3] + 0.25, ymin=0, ymax=0.06 * ncut[k3], color=CUT, alpha=0.5, lw=0)
        tx.plot(hrs[:upto], solar[:upto], color=SOLAR, lw=2.2, label="solar (kW)")
        tx.plot(hrs[:upto], demand[:upto], color=HOMEC, lw=2.2, label="street demand (kW)")
        tx.plot(hrs[:upto], grid[:upto], color=GRID, lw=1.8, label="grid import (kW)")
        tx.set_xlim(0, 24); tx.set_ylim(min(-0.5, grid.min() - 0.3), max(solar.max(), demand.max(), grid.max()) * 1.12)
        tx.set_xticks(range(0, 25, 3)); tx.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=10)
        tx.set_ylabel("kW", fontsize=11); tx.grid(alpha=0.15)
        t2 = tx.twinx(); t2.plot(hrs[:upto], socp[:upto], color=BATT, lw=2.4, label="battery SoC (%)"); t2.set_ylim(0, 105)
        t2.set_ylabel("SoC %", fontsize=11, color=BATT); t2.tick_params(colors=BATT, labelsize=10)
        if not intro: tx.axvline(r.hour + frac * 0.25, color="white", lw=1.5, alpha=0.9)
        for sp in list(tx.spines.values()) + list(t2.spines.values()): sp.set_color(EDGE)
        h1, l1 = tx.get_legend_handles_labels(); h2, l2 = t2.get_legend_handles_labels()
        if h1: tx.legend(h1 + h2, l1 + l2, loc="upper left", ncol=4, fontsize=10, frameon=False, labelcolor=TXT)
        fig.text(0.955, 0.265, "orange ticks = homes in a cut · red band = evening peak", fontsize=10, color=MUTED, ha="right")
        # ---------------- key-moment badge
        near = [name for k, name in keys if abs(x - (k + 0.5)) < 3.5] if not (intro or outro) else []
        if near:
            fig.text(0.42, 0.885, f"KEY MOMENT · {near[0]}", fontsize=17, weight="bold", color=BG, ha="center",
                     bbox=dict(boxstyle="round,pad=0.45", fc=SOLAR, ec="none"))
        # ---------------- intro overlay
        if intro:
            a_ = min(1, f / 20)
            ov = fig.add_axes([0.02, 0.04, 0.96, 0.85]); ov.set_xlim(0, 100); ov.set_ylim(0, 60); ov.axis("off")
            ov.add_patch(FancyBboxPatch((0, 0), 100, 60, boxstyle="round,pad=0,rounding_size=2", fc=BG, ec=SOLAR, lw=2, alpha=0.96 * a_))
            ov.text(50, 52, "How the RL control loop works", ha="center", fontsize=30, weight="bold", color=TXT, alpha=a_)
            steps = [("Street state s_t", "battery, solar, demand,\ncut pattern, fairness", HOMEC), ("Forecasts", "cut risk + demand\n+1 / +2 / +4 h", CUT),
                     ("GRU-PPO policy", "last 8 states → GRU\n→ π(a|s), 10 actions", BATT), ("Safety shield", "keeps ≥ 4 h of\nessential reserve", GRID),
                     ("Simulator", "energy flows, cuts\n→ reward, s_t+1", SOLAR)]
            shown = int(np.clip((f - 15) / 25, 0, 5))
            for k2, (t1, t2s, col) in enumerate(steps[:shown]):
                cx = 10 + k2 * 20
                ov.add_patch(FancyBboxPatch((cx - 8.5, 24), 17, 16, boxstyle="round,pad=0,rounding_size=1.5", fc=PANEL, ec=col, lw=2.5))
                ov.text(cx, 35, t1, ha="center", fontsize=15, weight="bold", color=col)
                ov.text(cx, 28, t2s, ha="center", fontsize=12, color=TXT, linespacing=1.4)
                if k2 < 4 and k2 + 1 < shown: ov.annotate("", (cx + 11.5, 32), (cx + 8.5, 32), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
            if shown == 5:
                ov.plot([90, 90, 10, 10], [23.5, 18, 18, 22.5], color=MUTED, lw=1.8)
                ov.annotate("", (10, 23.8), (10, 21.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8))
                ov.text(50, 14.5, "reward r_t + next state s_t+1 feed back every 15 minutes", ha="center", fontsize=13, color=MUTED)
            ov.text(50, 4.5, "Behaviour cloning + DAgger from a perfect-foresight MILP expert, then PPO for 1M steps · validation data only",
                    ha="center", fontsize=12, color=MUTED, alpha=a_)
        # ---------------- outro overlay
        if outro:
            a_ = min(1, (f - INTRO - SIM_FRAMES) / 20)
            ov = fig.add_axes([0.6, 0.315, 0.38, 0.57]); ov.set_xlim(0, 100); ov.set_ylim(0, 100); ov.axis("off")
            ov.add_patch(FancyBboxPatch((0, 0), 100, 100, boxstyle="round,pad=0,rounding_size=2", fc=BG, ec=SOLAR, lw=2, alpha=0.97 * a_))
            dt = 0.25; cutE = (day.load_cut * dt).sum(); served = cutE - (np.array([sum(json.loads(v)) for v in day.full_short]) * dt).sum()
            ev = day[(day.hour >= 18) & (day.hour < 22)]
            lines = [("24-hour dispatch complete", TXT, 22, "bold"),
                     (f"Solar produced {(day.pv * dt).sum():.1f} kWh · stored {(day.ch_pv * dt).sum():.1f} kWh · curtailed {(day.curtailed * dt).sum():.1f} kWh", SOLAR, 12.5, "normal"),
                     (f"Battery discharged {((day.b_on + day.b_cut) * dt).sum():.1f} kWh (to cut homes {(day.b_cut * dt).sum():.1f})", BATT, 12.5, "normal"),
                     (f"Homes in a cut: {int((ncut > 0).sum() * 15)} min with ≥1 home cut · {100 * served / max(cutE, 1e-9):.0f}% of their demand served", CUT, 12.5, "normal"),
                     (f"Essential energy not served: {day.ess_uns.sum() * 1000:.0f} Wh", BAD if day.ess_uns.sum() > 0.01 else BATT, 12.5, "normal"),
                     (f"Evening (18–22 h) grid import {(ev.grid * dt).sum():.1f} kWh · cap 5.76 kW", GRID, 12.5, "normal"),
                     (f"Shield overrides: {int(day.shield.sum())} of 96 steps · day reward {day.reward.sum():+.2f}", MUTED, 12.5, "normal")]
            for k2, (txt, col, fs, wt) in enumerate(lines):
                ov.text(5, 88 - k2 * 11.5 - (4 if k2 else 0), txt, fontsize=fs, color=col, weight=wt, alpha=a_)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return img

    from PIL import Image
    for f in frames:
        Image.fromarray(frame(f)).save(Path(outdir) / f"f{f:05d}.png", optimize=False, compress_level=1)
    return len(frames)


def main():
    day = pd.read_csv(ROLLOUT_CSV) if ROLLOUT_CSV.exists() and "--rerun" not in sys.argv else rollout()
    tau, keys = timeline(day)
    print("key moments:", [(name, f"{day.hour[k]:05.2f} h") for k, name in keys])
    tmp = Path(tempfile.mkdtemp(prefix="nirantar_frames_"))
    try:
        from joblib import Parallel, delayed
        nj = max(1, min(8, os.cpu_count() or 1))
        chunks = [list(range(i, N_FRAMES, nj)) for i in range(nj)]
        Parallel(n_jobs=nj)(delayed(render_chunk)((c, str(ROLLOUT_CSV), tau, keys, str(tmp))) for c in chunks)
        OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT_MP4)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", OUT_MP4)


if __name__ == "__main__":
    main()
