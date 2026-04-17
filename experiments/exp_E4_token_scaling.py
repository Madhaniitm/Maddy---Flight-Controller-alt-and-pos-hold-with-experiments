"""
EXP-E4: Token Usage vs Mission Complexity + Cost Scaling
=========================================================
Goal:
    Run 5 missions of increasing complexity (1-step to 5-step) and measure
    how token usage and USD cost scale with task complexity. N=3 runs per
    complexity level to capture stochastic variation.

    Missions:
        Level 1 (1-step) : "Arm the drone"
        Level 2 (2-step) : "Arm and take off to 1.0 m"
        Level 3 (3-step) : "Arm, take off to 1.0 m, hold for 5 seconds"
        Level 4 (4-step) : "Arm, take off to 1.0 m, hold 5s, then land"
        Level 5 (5-step) : "Arm, take off to 1.0 m, hold 5s, rotate 90°, then land and disarm"

Metrics:
    - tokens_in     : input tokens (bootstrap CI per complexity level)
    - tokens_out    : output tokens (bootstrap CI)
    - api_calls     : number of API calls (bootstrap CI)
    - cost_usd      : USD per mission (bootstrap CI)
    - linearity_r2  : R² of linear fit (tokens vs n_steps) — tests near-linear growth claim

Paper References:
    - ReAct (Yao et al. 2022): multi-step chaining drives token growth
    - Vemprala2023: task complexity vs API cost for robotics LLM supervision
    - InnerMonologue (Huang et al. 2022): accumulated context causes token growth
"""

import os, sys, json, time, csv, math, pathlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS   = 5
OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
}

MISSIONS = [
    {
        "level": 1,
        "n_steps": 1,
        "goal": "Arm the drone. Confirm when armed.",
        "label": "Arm only",
    },
    {
        "level": 2,
        "n_steps": 2,
        "goal": "Arm the drone and take off to 1.0 m altitude. Confirm when hovering.",
        "label": "Arm + takeoff",
    },
    {
        "level": 3,
        "n_steps": 3,
        "goal": (
            "Arm the drone, take off to 1.0 m, then enable altitude hold "
            "and hover for 5 seconds. Confirm each step."
        ),
        "label": "Arm + takeoff + hold",
    },
    {
        "level": 4,
        "n_steps": 4,
        "goal": (
            "Arm the drone, take off to 1.0 m, hold altitude for 5 seconds, "
            "then land safely. Confirm each step. Announce 'DONE' when landed."
        ),
        "label": "Arm + takeoff + hold + land",
    },
    {
        "level": 5,
        "n_steps": 5,
        "goal": (
            "Arm the drone, take off to 1.0 m, hold altitude for 5 seconds, "
            "rotate 90 degrees clockwise, then land and disarm. "
            "Confirm each step. Announce 'MISSION COMPLETE' when done."
        ),
        "label": "Arm + TO + hold + rotate + land",
    },
]

# ── Statistics helpers ─────────────────────────────────────────────────────────
def bootstrap_ci(data: list, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr   = np.array(data, dtype=float)
    boots = [stat(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)), 4), round(float(lo), 4), round(float(hi), 4)

def linear_r2(xs: list, ys: list) -> float:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if len(x) < 2:
        return float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0

# ── Single run ─────────────────────────────────────────────────────────────────
def run_once(run_idx: int, mission: dict) -> dict:
    agent = DAgent(session_id=f"E4_r{run_idx}_L{mission['level']}")
    t0 = time.time()
    reply, stats, trace = agent.run_agent_loop(mission["goal"])
    wall_time = time.time() - t0

    mission_done = any(kw in reply.upper()
                       for kw in ("DONE","COMPLETE","LANDED","ARMED","HOVERING"))

    return {
        "run":       run_idx,
        "level":     mission["level"],
        "n_steps":   mission["n_steps"],
        "label":     mission["label"],
        "done":      int(mission_done),
        "tokens_in": stats.get("tokens_in", 0),
        "tokens_out":stats.get("tokens_out", 0),
        "api_calls": stats.get("api_calls", 0),
        "cost_usd":  round(stats.get("cost_usd", 0.0), 7),
        "time_s":    round(wall_time, 3),
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-E4: Token Usage vs Mission Complexity")
    print(f"N_RUNS={N_RUNS} per level, {len(MISSIONS)} complexity levels")
    print("=" * 60)

    all_rows = []
    for mission in MISSIONS:
        print(f"\n=== Level {mission['level']}: {mission['label']} ===")
        for run in range(1, N_RUNS + 1):
            row = run_once(run, mission)
            all_rows.append(row)
            print(f"  run={run} tok_in={row['tokens_in']} tok_out={row['tokens_out']} "
                  f"calls={row['api_calls']} ${row['cost_usd']:.6f}")

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "E4_runs.csv"
    fields   = ["run","level","n_steps","label","done","tokens_in","tokens_out",
                "api_calls","cost_usd","time_s"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Per-level stats ────────────────────────────────────────────────────────
    level_stats = {}
    for mission in MISSIONS:
        lv  = mission["level"]
        lr  = [r for r in all_rows if r["level"] == lv]
        ti_m, ti_lo, ti_hi = bootstrap_ci([r["tokens_in"]  for r in lr])
        to_m, to_lo, to_hi = bootstrap_ci([r["tokens_out"] for r in lr])
        ca_m, ca_lo, ca_hi = bootstrap_ci([r["api_calls"]  for r in lr])
        co_m, co_lo, co_hi = bootstrap_ci([r["cost_usd"]   for r in lr])
        level_stats[lv] = {
            "n_steps": mission["n_steps"],
            "label":   mission["label"],
            "ti": ti_m, "ti_lo": ti_lo, "ti_hi": ti_hi,
            "to": to_m, "to_lo": to_lo, "to_hi": to_hi,
            "ca": ca_m, "ca_lo": ca_lo, "ca_hi": ca_hi,
            "co": co_m, "co_lo": co_lo, "co_hi": co_hi,
        }

    # Linearity R²
    steps = [level_stats[lv]["n_steps"]  for lv in sorted(level_stats)]
    tis   = [level_stats[lv]["ti"]       for lv in sorted(level_stats)]
    costs = [level_stats[lv]["co"]       for lv in sorted(level_stats)]
    r2_tokens = linear_r2(steps, tis)
    r2_cost   = linear_r2(steps, costs)

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "E4_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["level","n_steps","label","metric","value","ci_lo","ci_hi","note"])
        for lv, s in sorted(level_stats.items()):
            cw.writerow([lv, s["n_steps"], s["label"], "tokens_in",  s["ti"], s["ti_lo"], s["ti_hi"], "Bootstrap 95%"])
            cw.writerow([lv, s["n_steps"], s["label"], "tokens_out", s["to"], s["to_lo"], s["to_hi"], "Bootstrap 95%"])
            cw.writerow([lv, s["n_steps"], s["label"], "api_calls",  s["ca"], s["ca_lo"], s["ca_hi"], "Bootstrap 95%"])
            cw.writerow([lv, s["n_steps"], s["label"], "cost_usd",   s["co"], s["co_lo"], s["co_hi"], "Bootstrap 95%"])
        cw.writerow(["ALL","","","r2_tokens_vs_steps", round(r2_tokens,4),"","",
                     "R² of linear fit: tokens_in vs n_steps"])
        cw.writerow(["ALL","","","r2_cost_vs_steps",   round(r2_cost,4),"","",
                     "R² of linear fit: cost_usd vs n_steps"])
        for key, ref in PAPER_REFS.items():
            cw.writerow(["","","",f"ref_{key}", ref,"","",""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ns    = [level_stats[lv]["n_steps"] for lv in sorted(level_stats)]
        lbls  = [f"L{lv}" for lv in sorted(level_stats)]

        # Tokens in vs steps
        ax = axes[0]
        ti_vals = [level_stats[lv]["ti"] for lv in sorted(level_stats)]
        ti_errs = [[level_stats[lv]["ti"] - level_stats[lv]["ti_lo"] for lv in sorted(level_stats)],
                   [level_stats[lv]["ti_hi"] - level_stats[lv]["ti"]  for lv in sorted(level_stats)]]
        ax.errorbar(ns, ti_vals, yerr=ti_errs, fmt="o-", capsize=6, color="#3498db", linewidth=2)
        # Linear fit
        slope, intercept = np.polyfit(ns, ti_vals, 1)
        fit_x = np.linspace(min(ns)-0.2, max(ns)+0.2, 100)
        ax.plot(fit_x, slope*fit_x + intercept, "r--", alpha=0.7,
                label=f"Linear fit R²={r2_tokens:.3f}")
        ax.set_xticks(ns)
        ax.set_xticklabels(lbls)
        ax.set_xlabel("Complexity level (n_steps)")
        ax.set_ylabel("Input tokens")
        ax.set_title("E4: Token Usage vs Complexity")
        ax.legend(fontsize=8)

        # API calls vs steps
        ax2 = axes[1]
        ca_vals = [level_stats[lv]["ca"] for lv in sorted(level_stats)]
        ca_errs = [[level_stats[lv]["ca"] - level_stats[lv]["ca_lo"] for lv in sorted(level_stats)],
                   [level_stats[lv]["ca_hi"] - level_stats[lv]["ca"]  for lv in sorted(level_stats)]]
        ax2.errorbar(ns, ca_vals, yerr=ca_errs, fmt="s-", capsize=6, color="#e67e22", linewidth=2)
        ax2.set_xticks(ns)
        ax2.set_xticklabels(lbls)
        ax2.set_xlabel("Complexity level (n_steps)")
        ax2.set_ylabel("API calls")
        ax2.set_title("E4: API Calls vs Complexity")

        # Cost vs steps
        ax3 = axes[2]
        co_vals = [level_stats[lv]["co"] for lv in sorted(level_stats)]
        co_errs = [[level_stats[lv]["co"] - level_stats[lv]["co_lo"] for lv in sorted(level_stats)],
                   [level_stats[lv]["co_hi"] - level_stats[lv]["co"]  for lv in sorted(level_stats)]]
        ax3.errorbar(ns, co_vals, yerr=co_errs, fmt="^-", capsize=6, color="#2ecc71", linewidth=2)
        s2, i2 = np.polyfit(ns, co_vals, 1)
        ax3.plot(fit_x, s2*fit_x + i2, "r--", alpha=0.7,
                 label=f"Linear fit R²={r2_cost:.3f}")
        ax3.set_xticks(ns)
        ax3.set_xticklabels(lbls)
        ax3.set_xlabel("Complexity level (n_steps)")
        ax3.set_ylabel("Cost per mission (USD)")
        ax3.set_title("E4: Cost vs Complexity")
        ax3.legend(fontsize=8)

        fig.suptitle(
            "EXP-E4 Token Usage + Cost vs Mission Complexity\n"
            "ReAct (Yao 2022), Inner Monologue (Huang 2022), Vemprala 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "E4_token_scaling.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── E4 Summary ──────────────────────────────────────────────────────")
    print(f"{'Level':6s} {'Steps':6s} {'Tok_in':8s} {'API_calls':10s} {'Cost':10s} {'Label'}")
    for lv, s in sorted(level_stats.items()):
        print(f"  L{lv}    {s['n_steps']}      {s['ti']:.0f}     {s['ca']:.1f}         "
              f"${s['co']:.6f}  {s['label']}")
    print(f"\nToken linearity R²  = {r2_tokens:.4f}")
    print(f"Cost   linearity R² = {r2_cost:.4f}")


if __name__ == "__main__":
    main()
