"""Builds the data and assets of the project website (docs/, served by GitHub Pages) from the saved results.
Run:  .venv\\Scripts\\python.exe scripts\\build_site_data.py"""
import json, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, DOCS = ROOT / "results", ROOT / "docs"
(DOCS / "assets").mkdir(parents=True, exist_ok=True)

day = pd.read_csv(RES / "rl_simulation_rollout.csv")
keep = ["hour", "pv", "load_on", "load_cut", "pv_on", "pv_cut", "b_on", "b_cut", "ch_pv", "ch_grid", "curtailed", "grid", "soc_before", "soc_after",
        "proposed", "applied", "shield", "value", "reward", "need_kwh", "avail_kwh", "cutp_1", "cutp_2", "cutp_4", "cutp_1_hi", "cutp_2_hi", "cutp_4_hi",
        "lf_1", "lf_2", "lf_4", "ess_uns"]
steps = []
for _, r in day.iterrows():
    d = {k: round(float(r[k]), 4) for k in keep}
    d["probs"] = [round(float(r[f"p{k}"]), 4) for k in range(10)]
    d["s"] = [round(float(r[f"s{k}"]), 4) for k in range(23)]
    d["off"] = json.loads(r.off); d["ess_short"] = [round(x, 5) for x in json.loads(r.ess_short)]
    d["full_short"] = [round(x, 5) for x in json.loads(r.full_short)]
    steps.append(d)

test = pd.read_csv(RES / "test_results.csv", index_col=0)
order = [("3 Rule-based", "Rule-based"), ("3s Rule-based + shield", "Rule-based + shield"), ("4a MPC v2 (risk-aware)", "MPC v2"), ("4b MPC-CC", "MPC-CC (ε = 0.10)"),
         ("6 PPO-BC (v3, Mathura only)", "PPO-BC (v3)"), ("7 Nirantar PPO-DAgger (MLP)", "Nirantar MLP"), ("7 Nirantar PPO-DAgger (GRU)", "Nirantar GRU ★"),
         ("5 Perfect foresight", "Perfect foresight (bound)")]
results = [{"method": lab, "ess": round(test.loc[k, "Essentials off (h/home·month)"], 2), "served": round(test.loc[k, "Cut demand served (%)"], 1),
            "evening": round(test.loc[k, "Evening grid (kWh/day)"], 2), "cycles": round(test.loc[k, "Battery cycles/day"], 2),
            "reward": round(test.loc[k, "Reward/week"], 2), "score": round(test.loc[k, "Score: no system 0 → perfect 100"])} for k, lab in order]
q = pd.read_csv(RES / "test_questions.csv", index_col=0)
verdict = list(json.loads((RES / "test_verdict.json").read_text(encoding="utf-8")).values())
names = {"Q1": "vs rule-based", "Q2": "vs MPC-CC", "Q3": "vs PPO-BC (v3)", "GRU − MLP": "GRU − MLP"}
questions = []
for i, (k, r) in enumerate(q.iterrows()):
    questions.append({"q": k, "name": names[k], "reward": r["Reward: mean"], "reward_ci": r["Reward: 95% CI"], "ess": r["Essentials off: mean"],
                      "ess_ci": r["Essentials off: 95% CI"], "evening": r["Evening grid: mean"], "evening_ci": r["Evening grid: 95% CI"],
                      "verdict": ("YES" if verdict[i] else "NO") if i < 3 else "MLP better"})
data = {"region": day.region[0], "start": day.start[0], "steps": steps, "results": results, "questions": questions,
        "data": json.loads((RES / "data_summary.json").read_text(encoding="utf-8"))["regions"]}
(DOCS / "data.js").write_text("window.NIRANTAR = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
for f in ["fig_architecture.png", "fig_eda.png", "fig_imitation.png", "06_rl_validation_curves.png", "07_test_pune.png"]:
    shutil.copy2(RES / "figures" / f, DOCS / "assets" / f)
shutil.copy2(ROOT / "Report" / "yuva_yodha_rl_simulation.mp4", DOCS / "assets" / "yuva_yodha_rl_simulation.mp4")
print("docs/data.js:", len(steps), "steps;", (DOCS / "data.js").stat().st_size // 1024, "KB")
