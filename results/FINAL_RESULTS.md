# Nirantar — Final Results

Plan: `results/FINAL_SPEC.md` (locked before training; sha 3000ccea…). Every choice was made on validation. The Pune 2018 test was run **once** (`results/test_done.json`).

## Setup in one paragraph

A 10-home street shares a solar + battery system. The controller learns from real outage and load data from three regions: Mathura and Bareilly (CEEW smart meters) and Uttar Pradesh (Prayas eMARC + ESMI).
- Train: 980 week windows. Validation: the last months of each region, 73 weeks. Test: Pune 2018, an unseen city, 50 weeks.
- Pipeline: a perfect-foresight MILP expert → behaviour cloning → DAgger (2 rounds) → PPO (1M steps) behind a 4-hour safety shield.
- Two policy networks, 5 seeds each: an MLP and a GRU over the last 8 states.

## 1. Validation (73 weeks) — model selection

| Method | Reward/week ↑ | Essentials off (h/home·month) ↓ | Cut demand served (%) ↑ | Evening grid (kWh/day) ↓ |
|---|---|---|---|---|
| Rule-based | −15.11 | 3.32 | 88.5 | 12.26 |
| Rule-based + shield | −14.79 | 2.27 | 88.5 | 12.27 |
| MPC v2 (risk-aware) | −16.73 | 4.46 | 72.8 | 11.42 |
| MPC-CC (ε = 0.10, chosen from 0.20/0.10/0.05) | −15.47 | 4.46 | 79.6 | 11.45 |
| Imitation only (MLP) | −13.95 ± 0.03 | 2.50 ± 0.58 | 75.5 | 9.90 |
| Imitation only (GRU) | −13.27 ± 0.23 | 2.85 ± 0.15 | 81.5 | 10.00 |
| PPO-DAgger (MLP) | −11.84 ± 0.11 | 2.53 ± 0.98 | 90.4 | 9.87 |
| **PPO-DAgger (GRU) — chosen** | **−11.58 ± 0.10** | **2.11 ± 0.17** | **91.3** | **9.88** |
| Perfect foresight (bound) | −9.66 | 0.77 | 95.6 | 9.52 |

- DAgger round kept per seed: MLP 0, 1, 1, 0, 1; GRU 0, 0, 0, 0, 0.
- PPO checkpoints chosen at steps of 200k–1M (`results/rl_val_per_run.csv`).
- Learning curves for every seed: `results/figures/06_rl_validation_curves.png`.
- Final network = GRU (better mean validation reward), frozen in `results/freeze.json`.

## 2. Pune 2018 test (50 weeks, run once)

| Method | Essentials off ↓ | Served % ↑ | Evening grid ↓ | Cycles/day | Jain ↑ | Reward/week ↑ | Score (0–100) |
|---|---|---|---|---|---|---|---|
| No system | 30.70 | 0.0 | 9.43 | 0.00 | – | −14.27 | 0 |
| Solar only | 7.80 | 65.4 | 9.29 | 0.00 | 0.892 | −9.64 | 42 |
| Backup-only inverter | 0.43 | 98.0 | 9.60 | 0.10 | 0.999 | −7.15 | 65 |
| Rule-based | 0.89 | 93.3 | 5.20 | 0.63 | 0.988 | −4.97 | 85 |
| Rule-based + shield | 0.78 | 93.3 | 5.20 | 0.63 | 0.988 | −4.96 | 85 |
| MPC v2 | 3.04 | 80.0 | 3.73 | 0.80 | 0.952 | −5.00 | 85 |
| MPC-CC (ε = 0.10) | 3.25 | 82.4 | 3.75 | 0.79 | 0.948 | −4.91 | 86 |
| PPO-BC (v3, Mathura only) | 1.43 | 90.5 | 3.49 | 0.85 | 0.980 | −4.14 | 93 |
| Nirantar PPO-DAgger (MLP) | 1.05 | 93.5 | 3.55 | 0.85 | 0.989 | −3.95 | 95 |
| **Nirantar PPO-DAgger (GRU) ★** | **1.05** | **93.5** | **3.62** | **0.90** | **0.988** | **−4.04** | **94** |
| Perfect foresight (bound) | 0.19 | 98.5 | 3.27 | 0.91 | 0.999 | −3.36 | 100 |

The learned rows are means over 5 seeds. The derived share of the rule-based → perfect-foresight reward gap that each method closes:
- GRU: 57.6%
- MLP: 63.1%
- PPO-BC v3: 51.3%

## 3. Answers to the pre-registered questions

Each difference is final − other, paired by Pune week, with a 95% bootstrap CI (10,000 resamples).

| Question | Reward/week | Essentials off | Evening grid | Cut demand served | Verdict |
|---|---|---|---|---|---|
| Q1 vs rule-based | +0.93 [+0.74, +1.19] | +0.17 [−0.28, +0.49] | −1.58 [−1.72, −1.47] | +0.21 [−1.18, +1.78] | **YES** |
| Q2 vs MPC-CC | +0.87 [+0.54, +1.26] | −2.19 [−3.31, −1.25] | −0.13 [−0.30, +0.03] | +11.12 [+7.69, +14.76] | **YES** |
| Q3 vs PPO-BC (v3) | +0.10 [−0.01, +0.22] | −0.38 [−0.77, −0.07] | +0.13 [+0.10, +0.15] | +3.00 [+1.85, +4.23] | **NO** |
| GRU − MLP | −0.09 [−0.15, −0.03] | +0.01 [−0.21, +0.28] | +0.06 [+0.04, +0.08] | +0.03 [−0.46, +0.56] | MLP better |

- **Q1 YES.** Higher reward and 1.6 kWh/day less evening grid import than rule-based control. Essentials-off is not worse: the CI upper bound is +0.495, just inside the +0.5 h limit.
- **Q2 YES.** Higher reward, 2.2 h less essentials-off and 11 points more cut demand served than chance-constrained MPC.
- **Q3 NO, narrowly.** The reward CI touches zero. The final controller does have significantly fewer essentials-off hours (−0.38 h) and serves 3 points more cut demand than PPO-BC v3, but uses slightly more evening grid.
- **GRU vs MLP.** The GRU was chosen on validation, but on Pune the MLP is slightly and significantly better on reward. The deep sequence model did not pay off.
- **Per seed** (`results/test_per_seed.csv`): all 10 Nirantar policies and all 5 PPO-BC policies beat rule-based on reward.

## 4. Deviations from the plan (disclosed)

- **Notebook 04 crash.** It crashed once on a label-formatting bug (`NAMES.get` evaluated its default eagerly). After the fix it was re-run with identical computations; no results changed.
- **DAgger usually not kept.** The rule "keep the round with the best validation reward" kept round 0 (behaviour cloning only) for every GRU seed and for MLP seeds 0 and 3.

## 5. Limitations

- Simulation only, on a 10-home street. Cuts are inferred from meter voltage (< 50 V), which is a proxy.
- One test city (Pune 2018), 50 weeks, 5 seeds.
- The learned controllers cycle the battery more than rule-based control: 0.85–0.90 vs 0.63 cycles/day, about ₹7/home·month more wear under the linear wear model.
