"""
EXP-C1: Natural Language → Tool Chain Execution  (N=5 runs)
============================================================
Human types: "take off and hover at 1 metre"
LLM must interpret and execute the correct tool sequence.

N=5 independent runs at temperature=0.2.
Reports mean ± std and 95% bootstrap CI for all metrics.
Success rate uses Wilson confidence interval.

Outputs:
  results/C1_runs.csv        — per-run metrics
  results/C1_summary.csv     — aggregate statistics
  results/C1_nl_to_toolchain.png — 5 altitude curves + mean ± 1σ band
"""

import sys, os, csv, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", "C1_runs.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", "C1_summary.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", "C1_nl_to_toolchain.png")

COMMAND    = "take off and hover at 1 metre"
TARGET_ALT = 1.0
TOLERANCE  = 0.10
N_RUNS     = 5
EXPECTED_SEQUENCE = ["arm", "find_hover_throttle", "enable_altitude_hold", "set_altitude_target"]

# ── Paper references that justify this experiment's design ─────────────────────
PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Justifies the reason→act→observe loop used in run_agent_loop."
    ),
    "Vemprala2023": (
        "Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). "
        "ChatGPT for Robotics: Design Principles and Model Abilities. "
        "MSR-TR-2023-8. arXiv:2306.17582. "
        "Closest prior work: GPT-4 on UAV tasks via structured function-call API. "
        "Provides API call overhead and success-rate benchmarks for comparison."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "Mechanism behind LLM autonomously inserting wait/stability-check from tool results."
    ),
}

# ── Statistics helpers ─────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

# ── Single-run function ────────────────────────────────────────────────────────

def run_once(run_idx):
    print(f"\n[C1] ── Run {run_idx+1}/{N_RUNS} ─────────────────────────────────")
    agent = SimAgent(session_id=f"C1_run{run_idx}")
    t_wall_start = time.time()

    final_text, api_stats, tool_trace = agent.run_agent_loop(COMMAND)
    t_wall_total = time.time() - t_wall_start

    agent.wait_sim(8.0)

    tel = agent.get_telem_arrays()
    z_arr = tel.get("z_true", np.array([]))
    z_final_samples = z_arr[-30:] if len(z_arr) >= 30 else z_arr
    z_ss   = float(np.mean(z_final_samples))   if len(z_final_samples) > 0 else float("nan")
    z_rmse = float(np.sqrt(np.mean((z_final_samples - TARGET_ALT)**2))) if len(z_final_samples) > 0 else float("nan")
    alt_error_cm = abs(z_ss - TARGET_ALT) * 100

    tool_names = [t["name"] for t in tool_trace]
    found_order = []
    for expected in EXPECTED_SEQUENCE:
        if expected in tool_names:
            found_order.append(expected)
    sequence_score = len(found_order)
    alt_pass = alt_error_cm <= TOLERANCE * 100
    seq_pass = sequence_score >= 3
    passed   = alt_pass and seq_pass

    total_api_calls = len(api_stats)
    total_in_tok    = sum(s["input_tokens"]  for s in api_stats)
    total_out_tok   = sum(s["output_tokens"] for s in api_stats)
    total_cost      = sum(s["cost_usd"]      for s in api_stats)
    mean_latency    = float(np.mean([s["latency_s"] for s in api_stats])) if api_stats else 0.0

    print(f"  z_ss={z_ss:.3f}m  err={alt_error_cm:.1f}cm  seq={sequence_score}/4  "
          f"api={total_api_calls}  pass={passed}")

    return {
        "run":            run_idx + 1,
        "z_ss_m":         round(z_ss, 4),
        "alt_error_cm":   round(alt_error_cm, 2),
        "z_rmse_cm":      round(z_rmse * 100, 3),
        "sequence_score": sequence_score,
        "alt_pass":       int(alt_pass),
        "seq_pass":       int(seq_pass),
        "passed":         int(passed),
        "api_calls":      total_api_calls,
        "input_tokens":   total_in_tok,
        "output_tokens":  total_out_tok,
        "cost_usd":       round(total_cost, 6),
        "mean_latency_s": round(mean_latency, 3),
        "wall_time_s":    round(t_wall_total, 1),
        "sim_time_s":     round(agent.sim_time, 1),
        "_tel":           tel,   # kept in memory for plotting, not written to CSV
    }

# ── Run N times ────────────────────────────────────────────────────────────────

all_results = []
for i in range(N_RUNS):
    all_results.append(run_once(i))

# ── Aggregate statistics ───────────────────────────────────────────────────────

def col(key):
    return [r[key] for r in all_results]

n_pass  = sum(col("passed"))
n_altok = sum(col("alt_pass"))
n_seqok = sum(col("seq_pass"))

pass_lo,  pass_hi  = wilson_ci(n_pass,  N_RUNS)
alt_lo,   alt_hi   = wilson_ci(n_altok, N_RUNS)
seq_lo,   seq_hi   = wilson_ci(n_seqok, N_RUNS)

z_ss_vals    = col("z_ss_m")
rmse_vals    = col("z_rmse_cm")
err_vals     = col("alt_error_cm")
api_vals     = col("api_calls")
lat_vals     = col("mean_latency_s")
cost_vals    = col("cost_usd")

z_ci  = bootstrap_ci(z_ss_vals)
rm_ci = bootstrap_ci(rmse_vals)
la_ci = bootstrap_ci(lat_vals)

print(f"\n[C1] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
print(f"  Success rate:    {n_pass}/{N_RUNS}  (95% CI: {pass_lo:.2f}–{pass_hi:.2f})")
print(f"  Alt error (cm):  {np.mean(err_vals):.2f} ± {np.std(err_vals):.2f}")
print(f"  z_ss (m):        {np.mean(z_ss_vals):.4f} ± {np.std(z_ss_vals):.4f}  "
      f"(CI: {z_ci[0]:.4f}–{z_ci[1]:.4f})")
print(f"  RMSE (cm):       {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}  "
      f"(CI: {rm_ci[0]:.3f}–{rm_ci[1]:.3f})")
print(f"  API calls:       {np.mean(api_vals):.1f} ± {np.std(api_vals):.1f}")
print(f"  Mean latency(s): {np.mean(lat_vals):.2f} ± {np.std(lat_vals):.2f}  "
      f"(CI: {la_ci[0]:.2f}–{la_ci[1]:.2f})")
print(f"  Total cost est.: ${sum(cost_vals):.4f}")

# ── Save per-run CSV ───────────────────────────────────────────────────────────
csv_keys = [k for k in all_results[0].keys() if not k.startswith("_")]
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_keys)
    w.writeheader()
    for r in all_results:
        w.writerow({k: r[k] for k in csv_keys})
print(f"[C1] Per-run CSV: {OUT_RUNS}")

# ── Save summary CSV ───────────────────────────────────────────────────────────
summary_rows = [
    ("n_runs",              N_RUNS),
    ("ref_ReAct",           PAPER_REFS["ReAct"]),
    ("ref_Vemprala2023",    PAPER_REFS["Vemprala2023"]),
    ("ref_InnerMonologue",  PAPER_REFS["InnerMonologue"]),
    ("n_pass",              n_pass),
    ("success_rate",        round(n_pass / N_RUNS, 3)),
    ("success_rate_ci_lo",  round(pass_lo, 3)),
    ("success_rate_ci_hi",  round(pass_hi, 3)),
    ("alt_error_cm_mean",   round(float(np.mean(err_vals)), 3)),
    ("alt_error_cm_std",    round(float(np.std(err_vals)), 3)),
    ("z_ss_mean_m",         round(float(np.mean(z_ss_vals)), 4)),
    ("z_ss_std_m",          round(float(np.std(z_ss_vals)), 4)),
    ("z_ss_ci_lo",          round(z_ci[0], 4)),
    ("z_ss_ci_hi",          round(z_ci[1], 4)),
    ("rmse_cm_mean",        round(float(np.mean(rmse_vals)), 3)),
    ("rmse_cm_std",         round(float(np.std(rmse_vals)), 3)),
    ("rmse_ci_lo_cm",       round(rm_ci[0], 3)),
    ("rmse_ci_hi_cm",       round(rm_ci[1], 3)),
    ("api_calls_mean",      round(float(np.mean(api_vals)), 1)),
    ("api_calls_std",       round(float(np.std(api_vals)), 1)),
    ("mean_latency_mean_s", round(float(np.mean(lat_vals)), 3)),
    ("mean_latency_std_s",  round(float(np.std(lat_vals)), 3)),
    ("mean_latency_ci_lo",  round(la_ci[0], 3)),
    ("mean_latency_ci_hi",  round(la_ci[1], 3)),
    ("total_cost_usd",      round(sum(cost_vals), 6)),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow([f"ref_{ref_key}", ref_val])
print(f"[C1] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

# — Top: altitude curves for each run —
run_colors = plt.cm.Blues(np.linspace(0.35, 0.85, N_RUNS))
max_len = 0
z_matrix = []
for i, r in enumerate(all_results):
    tel = r.get("_tel", {})
    t_arr = tel.get("t", np.array([]))
    z_arr = tel.get("z_true", np.array([]))
    if len(t_arr) > 0:
        t_s = t_arr / 1000.0
        ax1.plot(t_s, z_arr, color=run_colors[i], lw=1.0, alpha=0.55,
                 label=f"Run {i+1} (z_ss={r['z_ss_m']:.3f}m)")
        max_len = max(max_len, len(z_arr))
        z_matrix.append(z_arr)

# Mean ± 1σ band (align to shortest run)
if z_matrix:
    min_len = min(len(z) for z in z_matrix)
    z_stack = np.array([z[:min_len] for z in z_matrix])
    z_mean  = z_stack.mean(axis=0)
    z_std   = z_stack.std(axis=0)
    tel0 = all_results[0].get("_tel", {})
    t_common = tel0.get("t", np.array([]))[:min_len] / 1000.0
    ax1.plot(t_common, z_mean, color="navy", lw=2.0, label=f"Mean (n={N_RUNS})", zorder=5)
    ax1.fill_between(t_common, z_mean - z_std, z_mean + z_std,
                     alpha=0.15, color="navy", label="±1σ band")

ax1.axhline(TARGET_ALT, color="red", ls="--", lw=1.2, alpha=0.7,
            label=f"Target {TARGET_ALT}m")
ax1.axhspan(TARGET_ALT - TOLERANCE, TARGET_ALT + TOLERANCE,
            alpha=0.07, color="green", label=f"±{TOLERANCE*100:.0f}cm tolerance")
ax1.set_ylabel("Altitude (m)")
ax1.set_xlabel("Simulated time (s)")
ax1.set_title(
    f'EXP-C1: Natural Language → Tool Chain  (N={N_RUNS} runs, temperature=0.2)\n'
    f'Command: "{COMMAND}"\n'
    f'Success: {n_pass}/{N_RUNS}  '
    f'(95% CI: {pass_lo:.2f}–{pass_hi:.2f})  |  '
    f'z_ss = {np.mean(z_ss_vals):.3f} ± {np.std(z_ss_vals):.3f} m  |  '
    f'RMSE = {np.mean(rmse_vals):.2f} ± {np.std(rmse_vals):.2f} cm'
)
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)

# — Bottom: bar chart of per-run metrics —
x = np.arange(N_RUNS)
width = 0.3
ax2.bar(x - width/2, col("alt_error_cm"), width, label="Alt error (cm)",
        color="steelblue", alpha=0.75, edgecolor="black")
ax2.bar(x + width/2, col("z_rmse_cm"),    width, label="RMSE (cm)",
        color="darkorange", alpha=0.75, edgecolor="black")
ax2.axhline(TOLERANCE * 100, color="red", ls="--", lw=1, label=f"±{TOLERANCE*100:.0f}cm target")
ax2.set_xticks(x)
ax2.set_xticklabels([f"Run {i+1}\n({'✓' if r['passed'] else '✗'})" for i, r in enumerate(all_results)])
ax2.set_ylabel("Error (cm)")
ax2.set_title(f"Per-run altitude error & RMSE  |  API calls: {col('api_calls')}  "
              f"|  Mean latency: {np.mean(lat_vals):.2f}±{np.std(lat_vals):.2f}s")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C1] Plot: {OUT_PNG}")

print(f"\n[C1] RESULT: {n_pass}/{N_RUNS} runs passed  "
      f"(95% CI: {pass_lo:.2f}–{pass_hi:.2f})")
print(f"       z_ss = {np.mean(z_ss_vals):.4f} ± {np.std(z_ss_vals):.4f} m")
print(f"       RMSE = {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f} cm")
