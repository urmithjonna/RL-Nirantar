"""Causal forecaster inputs and the simulator's input arrays. Written from notebooks/02_forecasters.ipynb.
Feature definitions are those of versions 1-3 (archived src/features.py); every input at time t uses only data up to t."""
import numpy as np
import pandas as pd

SPD = 96
H = {"1h": 4, "2h": 8, "4h": 16}
CUT_F = ["into_min", "since_min", "share_1h", "share_24h", "share_7d", "home_slot7", "city_now", "city_1h",
         "city_slot7", "tod_sin", "tod_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos", "weekend"]       # v2: without cut_now
LOAD_F = ["d_now", "d_l1", "d_l4", "d_l8", "d_mean24", "d_yday_tgt", "d_week_tgt", "cut_now", "temp",
          "tod_sin", "tod_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos", "weekend"]


def runs_into_since(cut):
    x = np.nan_to_num(cut.values, nan=0.0); into = np.zeros_like(x); since = np.zeros_like(x)
    a = np.zeros(x.shape[1]); b = np.zeros(x.shape[1])
    for t in range(x.shape[0]):
        on = x[t] == 1; a = np.where(on, a + 1, 0); b = np.where(on, 0, np.minimum(b + 1, 30 * SPD))
        into[t], since[t] = a, b
    return pd.DataFrame(into * 15, cut.index, cut.columns), pd.DataFrame(since * 15, cut.index, cut.columns)


def target_time(idx, h):
    tt = idx + pd.Timedelta(minutes=15 * h); k = tt.hour * 4 + tt.minute // 15
    return pd.DataFrame({"tod_sin": np.sin(2 * np.pi * k / SPD), "tod_cos": np.cos(2 * np.pi * k / SPD),
                         "dow_sin": np.sin(2 * np.pi * tt.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * tt.dayofweek / 7),
                         "mon_sin": np.sin(2 * np.pi * (tt.month - 1) / 12), "mon_cos": np.cos(2 * np.pi * (tt.month - 1) / 12),
                         "weekend": (tt.dayofweek >= 5).astype(float)}, index=idx)


def _long(frames):
    return pd.DataFrame({k: v.stack(future_stack=True) for k, v in frames.items()})


def cut_table(cut, h, into, since, ok_slot=None):
    idx = cut.index
    city_now = cut.mean(axis=1)
    home_slot7 = sum(cut.shift(SPD * k - h) for k in range(1, 8)) / 7
    city_slot7 = sum(city_now.shift(SPD * k - h) for k in range(1, 8)) / 7
    frames = {"cut_now": cut, "into_min": into, "since_min": since,
              "share_1h": cut.rolling(4, min_periods=2).mean(), "share_24h": cut.rolling(96, min_periods=48).mean(),
              "share_7d": cut.rolling(672, min_periods=300).mean(), "home_slot7": home_slot7, "y": cut.shift(-h)}
    if ok_slot is not None:
        frames.update({"ok_t": ok_slot, "ok_tt": ok_slot.shift(-h)})
    X = _long(frames)
    t = X.index.get_level_values(0)
    for name, s in {"city_now": city_now, "city_1h": city_now.rolling(4).mean(), "city_slot7": city_slot7}.items():
        X[name] = s.reindex(t).values
    tf = target_time(idx, h).reindex(t); X[tf.columns] = tf.values
    X["hour_now"] = t.hour
    return X.astype(np.float32)


def load_table(demand, sup, cut, temp_std, scale, h, ok_slot=None):
    frames = {"d_now": demand / scale, "d_l1": demand.shift(1) / scale, "d_l4": demand.shift(4) / scale,
              "d_l8": demand.shift(8) / scale, "d_mean24": demand.rolling(96, min_periods=48).mean() / scale,
              "d_yday_tgt": demand.shift(SPD - h) / scale, "d_week_tgt": demand.shift(672 - h) / scale,
              "cut_now": cut, "scale": scale, "y_kw": sup.shift(-h)}
    if ok_slot is not None:
        frames.update({"ok_t": ok_slot, "ok_tt": ok_slot.shift(-h)})
    X = _long(frames)
    t = X.index.get_level_values(0)
    X["temp"] = temp_std.reindex(t).values
    tf = target_time(demand.index, h).reindex(t); X[tf.columns] = tf.values
    X["y"] = X.y_kw / X.scale
    return X.astype(np.float32)


def home_scale(sup, fallback):
    return sup.rolling(672, min_periods=24).mean().fillna(fallback).clip(lower=0.02)


def predict_cut(F, hn, X, cut_f=None):
    c = F["cut"][hn]
    return c["iso"].predict(c["model"].predict_proba(X[cut_f or F["cut_features"]])[:, 1])


def forecasts(tab, F, scalers, cut_f=None, load_f=None):
    """Per-slot, per-home forecasts at +1/+2/+4 h for one region: cut probability and demand (kW). Returns two arrays [T, N, 3].
    F = a frozen forecaster bundle {"cut": {h: {model, iso}}, "load": {h: {0.5: model}}, "cut_features", "load_features"}."""
    cut, dem, sup, solar = tab["cut"], tab["demand"], tab["supplied_load"], tab["solar"]
    into, since = runs_into_since(cut)
    temp_std = (solar.t2m.reindex(cut.index) - scalers["temperature_mean"]) / scalers["temperature_std"]
    scale = home_scale(sup, scalers["home_load_fallback_kw"])
    P, L = [], []
    for hn, h in H.items():
        Xc = cut_table(cut, h, into, since)
        P.append(pd.Series(predict_cut(F, hn, Xc, cut_f), index=Xc.index).unstack().reindex(index=cut.index, columns=cut.columns)); del Xc
        Xl = load_table(dem, sup, cut, temp_std, scale, h)
        lm = F["load"][hn][0.5]
        L.append(pd.Series(np.clip(lm.predict(Xl[load_f or F["load_features"]]), 0, None) * Xl.scale.values, index=Xl.index).unstack()
                 .reindex(index=cut.index, columns=cut.columns)); del Xl
    return (np.nan_to_num(np.stack([p.values for p in P], -1), nan=0.0).astype(np.float32),
            np.nan_to_num(np.stack([l.values for l in L], -1), nan=0.0).astype(np.float32))


def obs_inputs(tab):
    """Simulator inputs per slot: true solar (kW/kWp), last completed hour's solar, solar 24 h ago, city 7-day same-time cut rate."""
    idx = tab["cut"].index
    pv = tab["solar"].pv_kw_per_kwp.reindex(idx)
    pv_obs = (tab["hourly"].ghi / 1000 * 0.75).reindex(idx, method="ffill")
    pv_prev = pv.shift(SPD)
    share = tab["cut"].mean(axis=1)
    day = share.groupby([idx.normalize(), idx.hour * 4 + idx.minute // 15]).mean().unstack()
    rate = day.shift(1).rolling(7, min_periods=3).mean()
    cut_rate = pd.Series(rate.reindex(idx.normalize()).values[np.arange(len(idx)), (idx.hour * 4 + idx.minute // 15)], idx)
    return pv, pv_obs, pv_prev, cut_rate


def make_bank_region(tab, cut_p, load_fc, extra=None):
    """One region of the simulator bank (see sim.load_bank)."""
    idx = tab["cut"].index
    pv, pv_obs, pv_prev, cut_rate = obs_inputs(tab)
    B = {"start": np.datetime64(idx[0], "m"), "homes": list(tab["cut"].columns),
         "demand": tab["demand"].values.astype(np.float32), "cut": np.nan_to_num(tab["cut"].values, nan=0.0).astype(bool),
         "pv": pv.fillna(0).values.astype(np.float32), "pv_obs": pv_obs.fillna(0).values.astype(np.float32),
         "pv_prev": pv_prev.fillna(0).values.astype(np.float32),
         "cut_rate": cut_rate.fillna(cut_rate.mean()).values.astype(np.float32), "cut_p": cut_p, "load_fc": load_fc}
    B.update(extra or {})
    return B
