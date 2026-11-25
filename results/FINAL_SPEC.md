# Nirantar — final study plan

**Written 2026-10-25 and locked by notebook 00 (SHA-256 in `results/FINAL_SPEC_lock.json`) before any Pune data table is built and before any model in this study is trained.** Earlier versions (v1–v4) are archived in `_archive/` and are not re-run. This is the project's final, self-contained pipeline.

## 1. Goal

Train one controller for a shared street solar + community battery (10 homes) on **all the outage data we have**, and test it **once** on a new state it has never seen.

| Part | Data | Use |
|---|---|---|
| **Train** | Mathura (CEEW) 2019-05-01 – 2020-08-31 · Bareilly (CEEW) 2019-07-16 – 2020-12-31 · Uttar Pradesh (eMARC load + ESMI cuts) 2019-03-15 – 2019-09-30 | forecasters, demonstrations, DAgger, PPO |
| **Validation** | Mathura 2020-09-01 – 2021-02-20 · Bareilly 2021-01-01 – 2021-10-31 · Uttar Pradesh 2019-10-01 – 2019-12-31 | every choice (forecaster gate, DAgger round, checkpoint, MPC ε, final network) |
| **Test (once)** | **Pune 2018** (eMARC load + ESMI cuts + NASA POWER solar) | the final comparison only |

- **Windows:** train windows are 7-day windows starting every day (sliding); each training episode draws 10 random complete homes.
- **Validation and test weeks:** non-overlapping 7-day weeks with a fixed 10 homes (seed 0).
- **Boundaries:** no window crosses a period boundary.
- **Complete home-day:** at least 90 of 96 15-minute slots present.

## 2. Data rules (as in v1–v3)

**CEEW meters (Mathura, Bareilly)**
- Cut: at least half of a 15-minute slot's 3-minute readings are below 50 V.
- Demand during a cut: the home's mean supplied load at the same slot in the same month (fallback: same slot, all months).
- Homes with < 20% coverage are excluded.

**eMARC + ESMI (Uttar Pradesh, Pune)**
- Load: main-line meters. Blocks < 1 W are unknown and filled with home × month × slot mean, then home × slot mean, then home mean.
- Cuts: ESMI minute voltage. A 15-minute slot is cut if at least half of its present minutes are < 50 V; a slot with fewer than 8 minutes present is supply-on.
- Uttar Pradesh: ESMI 2019 hour repair, and pairing (scenario A) exactly as v3 (`_archive/results_v1-v4/v3`):
  - Kanpur homes → Kanpur site;
  - Gonda homes → Bahraich/Barabanki sites, round-robin, seed 0.
- Pune:
  - Homes: the 44 "Pune city" main-line homes.
  - Sites: Pune-district **domestic** ESMI sites with ≥ 300 days of 2018 data **and** ≥ 0.5 mean cut hours per day.
  - Pairing: each home gets one site, round-robin in random order (seed 0), fixed for the year.
  - The ESMI Hour column is checked with the v3 continuity rule; broken site-days are dropped.
- Test weeks: non-overlapping from 2018-01-15, kept if ≥ 10 homes are complete on all 7 days.

**Solar and temperature:** NASA POWER hourly, shifted +30 min, converted UTC→IST, performance ratio 0.75, and a causal hourly series. Points:
- Mathura 27.49 N 77.67 E;
- Bareilly 28.37 N 79.43 E;
- Lucknow 26.85 N 80.95 E (Uttar Pradesh);
- Pune 18.52 N 73.86 E.

**Scalers:** temperature mean and standard deviation over the pooled TRAIN periods. Home load is scaled by its own previous-7-day mean supplied load; the fallback is the median pooled-TRAIN home mean.

**System and reward:** unchanged from v1–v3 (`step5_config`):
- System per home: 0.5 kWp, 0.75 kWh, 0.25 kW.
- Reward weights: w_ess 10, w_other 3, w_peak 1, w_peak_excess 3, w_wear 0.2, w_fair 5, scale 0.1.
- Peak cap: 0.576 kW/home at 18–22 h.

This keeps every controller, including the earlier PPO-BC, scored on the same reward.

## 3. Forecasters (fit on pooled TRAIN, gate on pooled VALIDATION)

- **Cut risk (+1/+2/+4 h, per home):** the v2 recipe.
  - Features: CUT_F without `cut_now`.
  - Models: CatBoost at +1 h and +2 h, XGBoost at +4 h, with the v1/v2 hyperparameters, fit on a 50% sample of train-core.
  - Hold-out: the last 15% (by time) of each region's TRAIN period, used for early stopping and isotonic calibration.
- **Demand (+1/+2/+4 h, per home):** LightGBM quantile regression (α = 0.5), v1 hyperparameters and features, same core / hold-out.
- **Gate (unchanged rule):** ≥ 5% better than the best simple baseline at +1 h and +2 h on VALIDATION.
  - Cut baselines (Brier): city 7-day same time, home 7-day same time, persistence table.
  - Demand baselines (street MAE): yesterday, persistence.
  - A forecaster that fails is replaced by its best baseline in the state and the shield, and this is reported.

## 4. Simulator, state and safety shield

- **Simulator:** the v1–v3 street simulator, unchanged:
  - 15-minute steps, 7-day episodes;
  - 10 actions = 5 battery levels × backup mode.
- **State:** the 23-input v2 state.
- **Safety shield:** the v2 shield with a 4 h reserve (duration model off), on for every learned controller.

## 5. Learned controller

1. **Expert:** the perfect-foresight demonstrator.
   - A MILP re-planned every 16 steps from the state reached.
   - On/off decisions exact for 32 steps, `mip_rel_gap` 1e-3, nearest discrete action, no shield.
   - Demonstrations are recorded on every TRAIN window.
2. **Behaviour cloning (π₀):**
   - Cross-entropy, Adam 1e-3, batch 1024, ≤ 30 epochs.
   - Epoch chosen on the last 15% of TRAIN windows (by time, per region).
   - Runs on the GPU when available.
3. **DAgger, 2 rounds:** in round k:
   - roll out π₍k−1₎ (sampled actions, shield off, random start charge 30–90%) on 300 TRAIN windows drawn at random (seed 1000·s + k);
   - label every 8th visited step with the expert;
   - add the labels to the data and re-clone from the same initialisation.
   - The round (0–2) with the best mean VALIDATION reward is kept.
4. **PPO fine-tuning** from that policy:
   - Settings: learning rate 1e-4, n_steps 2048, batch 256, 10 epochs, clip 0.2, entropy 0.01, γ 0.995, shield on.
   - Length: 1M steps.
   - Checkpoint: the best mean VALIDATION reward, evaluated at step 0 and every 100k.
5. **Two policy networks, both through steps 2–4, 5 seeds each (0–4):**
   - **MLP:** current state (23) → 128-128.
   - **GRU (deep sequence model):** the last 8 states (2 h × 23 inputs) → GRU (64 hidden) → 128-128.
6. **Final controller:** the network with the better mean VALIDATION reward over the 5 seeds. Everything is frozen (hashes) before the Pune data is processed into episodes.

## 6. Baselines (all on the same Pune weeks)

| Group | Methods |
|---|---|
| Simple | no system, solar only, backup-only inverter, rule-based, rule-based + shield |
| Planning | MPC v2 (risk-aware, forecasts, shield); **MPC-CC** (chance-constrained); perfect foresight (bound) |
| Earlier best | **PPO-BC from v3**, frozen, trained on Mathura only (5 seeds), run with its own frozen forecasters |
| This study | final controller (5 seeds), and the other network (5 seeds) |

**MPC-CC details:**
- The MPC v2 MILP (forecast inputs, 24 h horizon, re-planned every step, `mip_rel_gap` 1e-2), plus a reserve on the planned battery energy for each of the next 16 steps:
  - s_h ≥ R − slack_h, with R = need_mean + z₁₋ε · need_sd;
  - need_mean is the 4 h shield need;
  - need_sd = √(Σ_homes (0.3 · load_i · hours_i)² · p_i(1 − p_i)), with p_i the +1 h cut probability.
- slack is penalised at w_ess. No shield.
- ε ∈ {0.20, 0.10, 0.05}, chosen on VALIDATION.

**Perfect foresight:** exact week MILP, 600 s cap per week, with the optimality gap reported.

## 7. Test (notebook 07, run once, refuses a second run)

**Comparison method:** paired per Pune week; 95% bootstrap CI (10,000 resamples, seed 0); learned methods are the mean over seeds 0–4.

**Main questions:**

| | Comparison | Pass rule |
|---|---|---|
| **Q1** | Final − Rule-based | reward CI > 0 **and** essentials-off CI upper end < +0.5 h/home·month |
| **Q2** | Final − MPC-CC (ε*) | reward CI > 0 |
| **Q3** | Final − PPO-BC (v3, Mathura only) | reward CI > 0 |

**Also reported:**
- GRU − MLP.
- Every individual outcome for every method:
  - essentials off (h/home·month);
  - cut demand served (%);
  - evening grid import (kWh/day) and energy above the cap;
  - battery cycles/day and wear (₹/home·month);
  - fairness (Jain);
  - shield rate;
  - reward.
- The derived score (no system = 0, perfect foresight = 100) and the share of the rule → perfect-foresight gap closed.
- Per-seed results.

## 8. Rules

- No setting changes after this lock; anything that fails is reported as failed.
- The Pune data is only turned into episodes after the final controller is frozen, and Pune results are computed once.
- **Limitations stated in advance:**
  - Uttar Pradesh and Pune load and cuts come from different devices in the same district and year.
  - The expert knows the future of the training week; the controller never does.
  - Simulation only (no hardware).
