"""
EXP-C5: Human Describes Problem → LLM Diagnoses and Fixes  (N=5 runs)
=======================================================================
Inject roll_angle_kp × 5. Human reports oscillation.
LLM must: analyze_flight → suggest_pid_tuning → set_tuning_params → apply_tuning.

N=5 runs. Reports RMSE reduction mean±std and diagnostic success rate.

Outputs:
  results/C5_runs.csv
  results/C5_summary.csv
  results/C5_human_describes_problem.png — RMSE before/after per run + aggregate
"""

import sys, os, csv, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent
from drone_sim import Kp_roll_angle

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", "C5_runs.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", "C5_summary.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", "C5_human_describes_problem.png")

KP_DEFAULT      = Kp_roll_angle
KP_INJECT_MULT  = 5.0
KP_INJECTED     = KP_DEFAULT * KP_INJECT_MULT
N_RUNS          = 5

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Diagnostic cycle (analyze→suggest→apply) is a ReAct loop: reason from telemetry, act on gains."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "analyze_flight() result embedded in context triggers LLM to self-diagnose and select fix."
    ),
    "Vemprala2023": (
        "Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). "
        "ChatGPT for Robotics: Design Principles and Model Abilities. MSR-TR-2023-8. arXiv:2306.17582. "
        "Establishes LLM-based gain tuning as a novel capability beyond prior UAV LLM work."
    ),
}

HUMAN_PROBLEM_MSG = (
    "The drone is oscillating badly on roll — it keeps swinging left and right "
    "rapidly and cannot stabilise. Please diagnose the problem and fix it."
)

EXPECTED_SEQUENCE = ["analyze_flight", "suggest_pid_tuning", "set_tuning_params", "apply_tuning"]

# ── Statistics helpers ─────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

# ── Single-run function ────────────────────────────────────────────────────────

def run_once(run_idx):
    print(f"\n[C5] ── Run {run_idx+1}/{N_RUNS} ─────────────────────────────────")
    agent = SimAgent(session_id=f"C5_run{run_idx}")

    # Inject bad kp
    agent.physics.pid_roll_angle.kp = KP_INJECTED

    # Arm and hover at 1.0 m (direct sim)
    with agent.state.lock:
        agent.state.armed = True
        agent.state.ch5   = 1000
    hover_pwm = agent._find_hover()
    hover_thr  = (hover_pwm - 1000) / 1000.0
    with agent.state.lock:
        s = agent.state
        s.hover_thr_locked = hover_thr
        s.althold  = True
        s.alt_sp   = s.z
        s.alt_sp_mm = s.z * 1000
    agent.physics.pid_alt_pos.reset()
    agent.physics.pid_alt_vel.reset()
    with agent.state.lock:
        agent.state.alt_sp    = 1.0
        agent.state.alt_sp_mm = 1000.0

    # Fly 20 s with bad gain
    agent.wait_sim(20.0)

    # RMSE before
    tel_before = agent.tel_buf.copy()
    roll_errs_before = [s["er"] for s in tel_before] if tel_before else []
    rmse_before = math.sqrt(sum(e**2 for e in roll_errs_before) / len(roll_errs_before)) \
        if roll_errs_before else float("nan")
    n_tel_before = len(tel_before)
    t_before_end = agent.sim_time

    # LLM diagnosis
    history = [
        {"role": "user",
         "content": "The drone is currently armed and hovering at 1.0 m with altitude hold active. "
                    "I need you to diagnose and fix a flight problem."},
        {"role": "assistant",
         "content": [{"type": "text",
                      "text": "Understood. Drone is at 1.0 m with altitude hold. Ready to diagnose."}]},
    ]
    text, api_stats, tool_trace = agent.run_agent_loop(
        HUMAN_PROBLEM_MSG, history=list(history), max_turns=15,
    )

    # Fly 20 s after fix
    t_after_start = agent.sim_time
    agent.wait_sim(20.0)

    tel_after = agent.tel_buf[n_tel_before:]
    roll_errs_after = [s["er"] for s in tel_after] if tel_after else []
    rmse_after = math.sqrt(sum(e**2 for e in roll_errs_after) / len(roll_errs_after)) \
        if roll_errs_after else float("nan")

    # Metrics
    tools_used    = [t["name"] for t in tool_trace]
    tools_set     = set(tools_used)
    found_sequence= [t for t in EXPECTED_SEQUENCE if t in tools_set]
    kp_final      = agent.physics.pid_roll_angle.kp
    kp_reduced    = kp_final < KP_INJECTED
    kp_reduction  = (KP_INJECTED - kp_final) / KP_INJECTED * 100 if kp_reduced else 0
    rmse_reduction= ((rmse_before - rmse_after) / rmse_before * 100
                     if rmse_before > 0 and not math.isnan(rmse_after) else 0)
    roll_identified = any(
        "roll_angle_kp" in str(t.get("args", {})) or
        "roll" in str(t.get("result", "")).lower()
        for t in tool_trace if t["name"] in ("set_tuning_params", "suggest_pid_tuning")
    )
    sequence_ok = len(found_sequence) >= 3
    passed      = sequence_ok and kp_reduced and roll_identified

    n_api  = len(api_stats)
    in_tok = sum(s["input_tokens"]  for s in api_stats)
    out_tok= sum(s["output_tokens"] for s in api_stats)
    cost   = sum(s["cost_usd"]      for s in api_stats)

    print(f"  RMSE: {rmse_before:.4f}→{rmse_after:.4f}  "
          f"reduction={rmse_reduction:.0f}%  kp: {KP_INJECTED:.4f}→{kp_final:.4f}  pass={passed}")

    return {
        "run":                run_idx + 1,
        "kp_injected":        round(KP_INJECTED, 5),
        "kp_after_fix":       round(kp_final, 5),
        "kp_reduced":         int(kp_reduced),
        "kp_reduction_pct":   round(kp_reduction, 1),
        "rmse_before_deg":    round(rmse_before, 5),
        "rmse_after_deg":     round(rmse_after, 5),
        "rmse_reduction_pct": round(rmse_reduction, 1),
        "roll_identified":    int(roll_identified),
        "sequence_ok":        int(sequence_ok),
        "found_sequence":     ";".join(found_sequence),
        "passed":             int(passed),
        "api_calls":          n_api,
        "input_tokens":       in_tok,
        "output_tokens":      out_tok,
        "cost_usd":           round(cost, 6),
    }

# ── Run N times ────────────────────────────────────────────────────────────────

all_results = [run_once(i) for i in range(N_RUNS)]

# ── Aggregate ─────────────────────────────────────────────────────────────────

def col(key):
    return [r[key] for r in all_results]

n_pass       = sum(col("passed"))
n_seq_ok     = sum(col("sequence_ok"))
n_kp_reduced = sum(col("kp_reduced"))
n_roll_id    = sum(col("roll_identified"))

pass_lo, pass_hi  = wilson_ci(n_pass,       N_RUNS)
seq_lo,  seq_hi   = wilson_ci(n_seq_ok,     N_RUNS)
kp_lo,   kp_hi    = wilson_ci(n_kp_reduced, N_RUNS)

rmse_bf  = col("rmse_before_deg")
rmse_af  = col("rmse_after_deg")
rmse_red = col("rmse_reduction_pct")
kp_red   = col("kp_reduction_pct")

rmse_bf_ci  = bootstrap_ci(rmse_bf)
rmse_af_ci  = bootstrap_ci(rmse_af)
rmse_red_ci = bootstrap_ci(rmse_red)

print(f"\n[C5] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
print(f"  Success rate:       {n_pass}/{N_RUNS}  CI=[{pass_lo:.2f},{pass_hi:.2f}]")
print(f"  Correct sequence:   {n_seq_ok}/{N_RUNS}  CI=[{seq_lo:.2f},{seq_hi:.2f}]")
print(f"  kp reduced:         {n_kp_reduced}/{N_RUNS}")
print(f"  RMSE before (deg):  {np.mean(rmse_bf):.4f}±{np.std(rmse_bf):.4f}  "
      f"CI=[{rmse_bf_ci[0]:.4f},{rmse_bf_ci[1]:.4f}]")
print(f"  RMSE after  (deg):  {np.mean(rmse_af):.4f}±{np.std(rmse_af):.4f}  "
      f"CI=[{rmse_af_ci[0]:.4f},{rmse_af_ci[1]:.4f}]")
print(f"  RMSE reduction (%): {np.mean(rmse_red):.1f}±{np.std(rmse_red):.1f}  "
      f"CI=[{rmse_red_ci[0]:.1f},{rmse_red_ci[1]:.1f}]")
print(f"  kp reduction (%):   {np.mean(kp_red):.1f}±{np.std(kp_red):.1f}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_results[0].keys())
    w.writeheader()
    w.writerows(all_results)
print(f"[C5] Per-run CSV: {OUT_RUNS}")

summary_rows = [
    ("n_runs",                  N_RUNS),
    ("kp_default",              round(KP_DEFAULT, 5)),
    ("kp_injected",             round(KP_INJECTED, 5)),
    ("n_pass",                  n_pass),
    ("success_rate",            round(n_pass / N_RUNS, 3)),
    ("success_rate_ci_lo",      round(pass_lo, 3)),
    ("success_rate_ci_hi",      round(pass_hi, 3)),
    ("sequence_ok_rate",        round(n_seq_ok / N_RUNS, 3)),
    ("kp_reduced_rate",         round(n_kp_reduced / N_RUNS, 3)),
    ("roll_identified_rate",    round(n_roll_id / N_RUNS, 3)),
    ("rmse_before_mean_deg",    round(float(np.mean(rmse_bf)), 5)),
    ("rmse_before_std_deg",     round(float(np.std(rmse_bf)), 5)),
    ("rmse_before_ci_lo",       round(rmse_bf_ci[0], 5)),
    ("rmse_before_ci_hi",       round(rmse_bf_ci[1], 5)),
    ("rmse_after_mean_deg",     round(float(np.mean(rmse_af)), 5)),
    ("rmse_after_std_deg",      round(float(np.std(rmse_af)), 5)),
    ("rmse_after_ci_lo",        round(rmse_af_ci[0], 5)),
    ("rmse_after_ci_hi",        round(rmse_af_ci[1], 5)),
    ("rmse_reduction_mean_pct", round(float(np.mean(rmse_red)), 1)),
    ("rmse_reduction_std_pct",  round(float(np.std(rmse_red)), 1)),
    ("rmse_reduction_ci_lo",    round(rmse_red_ci[0], 1)),
    ("rmse_reduction_ci_hi",    round(rmse_red_ci[1], 1)),
    ("kp_reduction_mean_pct",   round(float(np.mean(kp_red)), 1)),
    ("kp_reduction_std_pct",    round(float(np.std(kp_red)), 1)),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow([f"ref_{ref_key}", ref_val])
print(f"[C5] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

run_idx  = np.arange(1, N_RUNS + 1)
bar_cols = ["green" if r["passed"] else "red" for r in all_results]

# Left: RMSE before vs after per run
ax1 = axes[0]
w   = 0.35
ax1.bar(run_idx - w/2, rmse_bf, w, label="Before fix", color="tomato", alpha=0.75, edgecolor="black")
ax1.bar(run_idx + w/2, rmse_af, w, label="After fix",  color="mediumseagreen", alpha=0.75, edgecolor="black")
ax1.set_xticks(run_idx)
ax1.set_xticklabels([f"R{i}\n({'✓' if r['passed'] else '✗'})"
                     for i, r in enumerate(all_results, 1)], fontsize=8)
ax1.set_ylabel("Roll error RMSE (deg)")
ax1.set_title("RMSE before vs after LLM fix\n(per run)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# Middle: RMSE reduction % with CI
ax2 = axes[1]
ax2.bar(run_idx, rmse_red, color=bar_cols, alpha=0.75, edgecolor="black")
ax2.axhline(np.mean(rmse_red), color="navy", ls="--", lw=1.5,
            label=f"Mean={np.mean(rmse_red):.1f}%")
ax2.fill_between([0.5, N_RUNS + 0.5],
                 rmse_red_ci[0], rmse_red_ci[1],
                 alpha=0.12, color="navy", label=f"95% CI [{rmse_red_ci[0]:.1f},{rmse_red_ci[1]:.1f}]%")
ax2.axhline(40, color="red", ls=":", lw=1, label="40% threshold")
ax2.set_xticks(run_idx)
ax2.set_xticklabels([f"Run {i}" for i in run_idx])
ax2.set_ylabel("RMSE reduction (%)")
ax2.set_title(f"RMSE reduction per run\n(green=pass, red=fail)")
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3, axis="y")

# Right: aggregate mean ± CI (before vs after)
ax3 = axes[2]
categories = ["Before fix", "After fix"]
means  = [float(np.mean(rmse_bf)), float(np.mean(rmse_af))]
stds   = [float(np.std(rmse_bf)),  float(np.std(rmse_af))]
colors = ["tomato", "mediumseagreen"]
bars   = ax3.bar(categories, means, color=colors, alpha=0.75, edgecolor="black")
ax3.errorbar(categories, means, yerr=stds, fmt="none", ecolor="black", capsize=8, lw=2)
for bar, m, s in zip(bars, means, stds):
    ax3.text(bar.get_x() + bar.get_width()/2, m + s + 0.001,
             f"{m:.4f}±{s:.4f}", ha="center", fontsize=9)
ax3.set_ylabel("Roll error RMSE (deg)")
ax3.set_title(f"Aggregate RMSE (mean±std, N={N_RUNS})\n"
              f"Reduction: {np.mean(rmse_red):.1f}±{np.std(rmse_red):.1f}%  "
              f"CI=[{rmse_red_ci[0]:.1f},{rmse_red_ci[1]:.1f}%]")
ax3.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    f"EXP-C5: LLM Fault Diagnosis  (N={N_RUNS} runs, temperature=0.2)\n"
    f"Success: {n_pass}/{N_RUNS}  (95% CI: {pass_lo:.2f}–{pass_hi:.2f})  |  "
    f"kp_inject={KP_INJECTED:.4f} ({KP_INJECT_MULT}×default)",
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C5] Plot: {OUT_PNG}")

print(f"\n[C5] RESULT: {n_pass}/{N_RUNS} passed  (95% CI: {pass_lo:.2f}–{pass_hi:.2f})")
print(f"       RMSE reduction: {np.mean(rmse_red):.1f}±{np.std(rmse_red):.1f}%")
