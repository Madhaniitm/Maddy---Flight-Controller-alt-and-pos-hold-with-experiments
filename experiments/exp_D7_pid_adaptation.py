"""
EXP-D7: Multi-LLM Iterative PID Adaptation
==========================================
Goal:
    Inject bad PID gains (roll_angle_kp × 5) and have each of 4 LLM backends
    iteratively tune them over up to 3 iterations until altitude RMSE improves.
    N=3 independent runs per (LLM × iteration_budget) combination = 36 total.

    Each iteration:
        1. Observe telemetry (detect_anomaly / analyze_flight)
        2. Suggest PID changes (suggest_pid_tuning)
        3. Apply gains (set_pid_gains / adjust_pid_live)
        4. Re-measure RMSE (evaluate 2 s of hover)

Metrics:
    - rmse_before_cm       : altitude RMSE before any tuning (bootstrap CI)
    - rmse_after_cm        : altitude RMSE after 3 iterations (bootstrap CI)
    - rmse_reduction_pct   : (before−after)/before × 100 (bootstrap CI)
    - iterations_to_stable : iterations until RMSE < 5 cm (bootstrap CI)
    - correction_accuracy  : did gains move in the correct direction? (Wilson CI)
    - cost_usd             : per-run (bootstrap CI)

Paper References:
    - ReAct (Yao et al. 2022): reason-act-observe loop for iterative gain tuning
    - Vemprala2023: ChatGPT for robotics, PID adaptation via LLM
    - InnerMonologue (Huang et al. 2022): telemetry observations drive replanning
"""

import os, sys, json, time, csv, math, pathlib, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent, MultiLLMRunner, MULTI_LLM_MODELS
from c_series_agent  import SIM_HZ, DT

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS_PER_CELL = 5
MAX_ITERATIONS  = 3
RMSE_TARGET_CM  = 5.0
BAD_KP_MULT     = 5.0      # multiply nominal roll_angle_kp by this
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
}

LLM_BACKENDS = list(MULTI_LLM_MODELS.keys())

TUNING_PROMPT = (
    "The drone has bad PID gains and is oscillating. "
    "Use analyze_flight and detect_anomaly to assess the current performance, "
    "then use suggest_pid_tuning to recommend improved gains, "
    "and finally apply the suggested gains. "
    "Aim to reduce altitude RMSE below 5 cm. "
    "After applying, call analyze_flight again to verify improvement."
)

# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(p, 4), round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)

def bootstrap_ci(data: list, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr   = np.array(data, dtype=float)
    boots = [stat(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)), 4), round(float(lo), 4), round(float(hi), 4)

def measure_rmse(agent: DAgent, n_steps: int = 200) -> float:
    """Run physics for n_steps and return altitude RMSE in cm."""
    errors = []
    for _ in range(n_steps):
        agent.physics_step()
        errors.append((agent.state.z - agent.state.target_z) * 100.0)
    return float(np.sqrt(np.mean(np.array(errors)**2)))

# ── Single run ─────────────────────────────────────────────────────────────────
def run_once(run_idx: int, llm_key: str) -> dict:
    agent = DAgent(session_id=f"D7_r{run_idx}_{llm_key}")
    agent.execute_tool("arm", {})
    agent.execute_tool("find_hover_throttle", {})
    agent.execute_tool("enable_altitude_hold", {})
    agent.execute_tool("set_altitude_target", {"target_m": 1.0})

    # Inject bad gains
    agent.state.roll_angle_kp *= BAD_KP_MULT

    rmse_before = measure_rmse(agent)

    # Determine initial gain direction for correction_accuracy check
    initial_kp = agent.state.roll_angle_kp

    t0 = time.time()
    iterations_to_stable = MAX_ITERATIONS  # pessimistic

    for iteration in range(1, MAX_ITERATIONS + 1):
        if llm_key == "claude":
            reply, stats, trace = agent.run_agent_loop(TUNING_PROMPT)
        else:
            runner = MultiLLMRunner(llm_key=llm_key)
            reply, stats = runner.run(
                goal       = TUNING_PROMPT,
                tools      = agent.all_tools(),
                execute_fn = agent.execute_tool,
            )

        current_rmse = measure_rmse(agent)
        if current_rmse < RMSE_TARGET_CM:
            iterations_to_stable = iteration
            break

    rmse_after = measure_rmse(agent)
    wall_time  = time.time() - t0

    reduction_pct = ((rmse_before - rmse_after) / rmse_before * 100.0
                     if rmse_before > 0 else 0.0)

    # Correction accuracy: did the gain decrease (correct direction for over-gain)?
    final_kp   = agent.state.roll_angle_kp
    gain_decreased = int(final_kp < initial_kp)

    row = {
        "run":                   run_idx,
        "llm":                   llm_key,
        "rmse_before_cm":        round(rmse_before, 3),
        "rmse_after_cm":         round(rmse_after, 3),
        "rmse_reduction_pct":    round(reduction_pct, 2),
        "iterations_to_stable":  iterations_to_stable,
        "correction_accurate":   gain_decreased,
        "api_calls":             stats.get("api_calls", 0),
        "tokens_in":             stats.get("tokens_in", 0),
        "tokens_out":            stats.get("tokens_out", 0),
        "cost_usd":              round(stats.get("cost_usd", 0.0), 6),
        "time_s":                round(wall_time, 3),
    }
    print(f"  [D7 run={run_idx} {llm_key:8s}] "
          f"RMSE {rmse_before:.1f}→{rmse_after:.1f}cm "
          f"({reduction_pct:.1f}%) iters={iterations_to_stable}")
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D7: Multi-LLM Iterative PID Adaptation")
    print(f"N={N_RUNS_PER_CELL} runs/LLM, {len(LLM_BACKENDS)} backends, MAX_ITER={MAX_ITERATIONS}")
    print("=" * 60)

    all_rows = []
    for llm in LLM_BACKENDS:
        print(f"\n=== LLM: {llm} ===")
        for run in range(1, N_RUNS_PER_CELL + 1):
            all_rows.append(run_once(run, llm))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D7_runs.csv"
    fields   = ["run","llm","rmse_before_cm","rmse_after_cm","rmse_reduction_pct",
                "iterations_to_stable","correction_accurate",
                "api_calls","tokens_in","tokens_out","cost_usd","time_s"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Stats per LLM ─────────────────────────────────────────────────────────
    llm_stats   = {}
    summary_csv = OUT_DIR / "D7_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["llm","metric","value","ci_lo","ci_hi","note"])
        for llm in LLM_BACKENDS:
            lr = [r for r in all_rows if r["llm"] == llm]

            rmse_b, rb_lo, rb_hi = bootstrap_ci([r["rmse_before_cm"]      for r in lr])
            rmse_a, ra_lo, ra_hi = bootstrap_ci([r["rmse_after_cm"]       for r in lr])
            red_m, red_lo, red_hi= bootstrap_ci([r["rmse_reduction_pct"]  for r in lr])
            iters, i_lo, i_hi    = bootstrap_ci([r["iterations_to_stable"] for r in lr])
            cost_m, c_lo, c_hi   = bootstrap_ci([r["cost_usd"]            for r in lr])

            k_corr = sum(r["correction_accurate"] for r in lr)
            ca, ca_lo, ca_hi     = wilson_ci(k_corr, len(lr))

            llm_stats[llm] = {
                "rmse_a": rmse_a, "ra_lo": ra_lo, "ra_hi": ra_hi,
                "red": red_m, "red_lo": red_lo, "red_hi": red_hi,
                "ca": ca, "ca_lo": ca_lo, "ca_hi": ca_hi,
            }

            cw.writerow([llm, "rmse_before_cm",      rmse_b, rb_lo, rb_hi, "Bootstrap 95%"])
            cw.writerow([llm, "rmse_after_cm",        rmse_a, ra_lo, ra_hi, "Bootstrap 95%"])
            cw.writerow([llm, "rmse_reduction_pct",   red_m,  red_lo,red_hi,"Bootstrap 95%"])
            cw.writerow([llm, "iterations_to_stable", iters,  i_lo,  i_hi,  "Bootstrap 95%"])
            cw.writerow([llm, "correction_accuracy",  ca,     ca_lo, ca_hi, "Wilson 95%"])
            cw.writerow([llm, "cost_usd",             cost_m, c_lo,  c_hi,  "Bootstrap 95%"])

        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["#3498db","#e67e22","#2ecc71","#9b59b6"]

        ax = axes[0]
        for i, llm in enumerate(LLM_BACKENDS):
            s = llm_stats[llm]
            ax.bar(llm, s["rmse_a"],
                   yerr=[[s["rmse_a"]-s["ra_lo"]], [s["ra_hi"]-s["rmse_a"]]],
                   capsize=6, color=colors[i])
        ax.axhline(RMSE_TARGET_CM, linestyle="--", color="red",
                   label=f"Target {RMSE_TARGET_CM}cm")
        ax.set_ylabel("RMSE after tuning (cm)")
        ax.set_title("D7: Post-Tuning Altitude RMSE")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        for i, llm in enumerate(LLM_BACKENDS):
            s = llm_stats[llm]
            ax2.bar(llm, s["red"],
                    yerr=[[s["red"]-s["red_lo"]], [s["red_hi"]-s["red"]]],
                    capsize=6, color=colors[i])
        ax2.set_ylabel("RMSE reduction (%)")
        ax2.set_title("D7: RMSE Reduction per LLM")

        ax3 = axes[2]
        for i, llm in enumerate(LLM_BACKENDS):
            s = llm_stats[llm]
            ax3.bar(llm, s["ca"],
                    yerr=[[s["ca"]-s["ca_lo"]], [s["ca_hi"]-s["ca"]]],
                    capsize=6, color=colors[i])
        ax3.set_ylim(0, 1.1)
        ax3.set_ylabel("Correction accuracy (Wilson 95% CI)")
        ax3.set_title("D7: Gain Correction Direction Accuracy")

        fig.suptitle(
            "EXP-D7 Multi-LLM Iterative PID Adaptation\n"
            "ReAct (Yao 2022), Vemprala 2023, Inner Monologue (Huang 2022)",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D7_pid_adaptation.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D7 Summary ──────────────────────────────────────────────────────")
    for llm in LLM_BACKENDS:
        s = llm_stats[llm]
        print(f"  {llm:10s}: RMSE_after={s['rmse_a']:.1f}cm "
              f"red={s['red']:.1f}% [{s['red_lo']:.1f},{s['red_hi']:.1f}%] "
              f"correct_dir={s['ca']:.2f}")


if __name__ == "__main__":
    main()
