"""
EXP-C4: Human Correction Mid-Mission  (N=5 runs)
=================================================
"hover at 0.5m" → executing → interrupt: "actually go to 1.2m"
N=5 independent runs. Reports success rate with Wilson 95% CI.

Outputs:
  results/C4_runs.csv        — per-run metrics
  results/C4_summary.csv     — aggregate statistics
  results/C4_mid_mission_correction.png
"""

import sys, os, csv, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", "C4_runs.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", "C4_summary.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", "C4_mid_mission_correction.png")

INITIAL_CMD    = "hover at 0.5 metres"
CORRECTION_CMD = "actually go to 1.2 metres instead"
INITIAL_TARGET = 0.5
CORRECT_TARGET = 1.2
TOLERANCE      = 0.12
N_RUNS         = 5

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Correction is processed as a new observation in the running ReAct loop."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "LLM infers current drone state from conversation history to avoid unnecessary re-arm."
    ),
}

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
    print(f"\n[C4] ── Run {run_idx+1}/{N_RUNS} ─────────────────────────────────")
    agent   = SimAgent(session_id=f"C4_run{run_idx}")
    history = []

    # Phase 1: initial command
    text1, stats1, trace1 = agent.run_agent_loop(
        INITIAL_CMD, history=list(history), max_turns=15,
    )
    history.append({"role": "user",      "content": INITIAL_CMD})
    history.append({"role": "assistant", "content": [{"type": "text", "text": text1}]})

    agent.wait_sim(4.0)
    with agent.state.lock:
        z_phase1   = round(agent.state.ekf_z, 3)
        armed_ph1  = agent.state.armed
        althold_ph1= agent.state.althold

    # Phase 2: mid-mission correction
    text2, stats2, trace2 = agent.run_agent_loop(
        CORRECTION_CMD, history=list(history), max_turns=8,
    )
    agent.wait_sim(6.0)
    with agent.state.lock:
        z_final     = round(agent.state.ekf_z, 3)
        armed_final = agent.state.armed

    tools_phase2 = [t["name"] for t in trace2]

    set_alt_calls     = [t for t in trace2 if t["name"] == "set_altitude_target"]
    correct_target_set= any(abs(t["args"].get("meters", 0) - CORRECT_TARGET) < 0.15
                            for t in set_alt_calls)
    re_armed     = "arm" in tools_phase2
    re_took_off  = "find_hover_throttle" in tools_phase2
    alt_reached  = abs(z_final - CORRECT_TARGET) <= TOLERANCE
    passed       = correct_target_set and not re_armed and not re_took_off

    first_set_idx   = next((i for i, t in enumerate(tools_phase2)
                            if t == "set_altitude_target"), None)
    meta_tools      = {"plan_workflow", "report_progress"}
    non_meta_before = [t for t in (tools_phase2[:first_set_idx] if first_set_idx is not None else tools_phase2)
                       if t not in meta_tools]

    n_api1 = len(stats1); n_api2 = len(stats2)
    in_tok = sum(s["input_tokens"]  for s in stats1 + stats2)
    out_tok= sum(s["output_tokens"] for s in stats1 + stats2)
    cost   = sum(s["cost_usd"]      for s in stats1 + stats2)

    print(f"  z_phase1={z_phase1:.3f}m  z_final={z_final:.3f}m  "
          f"correct_target={correct_target_set}  re_armed={re_armed}  pass={passed}")

    return {
        "run":                run_idx + 1,
        "z_phase1_m":         z_phase1,
        "z_final_m":          z_final,
        "alt_error_cm":       round(abs(z_final - CORRECT_TARGET) * 100, 1),
        "correct_target_set": int(correct_target_set),
        "re_armed":           int(re_armed),
        "re_took_off":        int(re_took_off),
        "alt_reached":        int(alt_reached),
        "tools_before_set":   len(non_meta_before),
        "passed":             int(passed),
        "api_calls_ph1":      n_api1,
        "api_calls_ph2":      n_api2,
        "input_tokens":       in_tok,
        "output_tokens":      out_tok,
        "cost_usd":           round(cost, 6),
        "tools_ph2":          ";".join(tools_phase2[:10]),
    }

# ── Run N times ────────────────────────────────────────────────────────────────

all_results = [run_once(i) for i in range(N_RUNS)]

# ── Aggregate ─────────────────────────────────────────────────────────────────

def col(key):
    return [r[key] for r in all_results]

n_pass    = sum(col("passed"))
n_correct = sum(col("correct_target_set"))
n_no_rearm= sum(1 for r in all_results if not r["re_armed"])
n_alt_ok  = sum(col("alt_reached"))

pass_lo,  pass_hi  = wilson_ci(n_pass,    N_RUNS)
corr_lo,  corr_hi  = wilson_ci(n_correct, N_RUNS)
nore_lo,  nore_hi  = wilson_ci(n_no_rearm,N_RUNS)

alt_err_vals = col("alt_error_cm")
tools_before = col("tools_before_set")
err_ci = bootstrap_ci(alt_err_vals)

print(f"\n[C4] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
print(f"  Success rate:            {n_pass}/{N_RUNS}  CI=[{pass_lo:.2f},{pass_hi:.2f}]")
print(f"  Correct target set:      {n_correct}/{N_RUNS}  CI=[{corr_lo:.2f},{corr_hi:.2f}]")
print(f"  No re-arm:               {n_no_rearm}/{N_RUNS}  CI=[{nore_lo:.2f},{nore_hi:.2f}]")
print(f"  Alt error (cm):          {np.mean(alt_err_vals):.2f}±{np.std(alt_err_vals):.2f}  "
      f"CI=[{err_ci[0]:.2f},{err_ci[1]:.2f}]")
print(f"  Non-meta tools before set:{np.mean(tools_before):.1f}±{np.std(tools_before):.1f}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_results[0].keys())
    w.writeheader()
    w.writerows(all_results)
print(f"[C4] Per-run CSV: {OUT_RUNS}")

summary_rows = [
    ("n_runs",                    N_RUNS),
    ("n_pass",                    n_pass),
    ("success_rate",              round(n_pass / N_RUNS, 3)),
    ("success_rate_ci_lo",        round(pass_lo, 3)),
    ("success_rate_ci_hi",        round(pass_hi, 3)),
    ("correct_target_rate",       round(n_correct / N_RUNS, 3)),
    ("correct_target_ci_lo",      round(corr_lo, 3)),
    ("correct_target_ci_hi",      round(corr_hi, 3)),
    ("no_rearm_rate",             round(n_no_rearm / N_RUNS, 3)),
    ("no_rearm_ci_lo",            round(nore_lo, 3)),
    ("no_rearm_ci_hi",            round(nore_hi, 3)),
    ("alt_error_mean_cm",         round(float(np.mean(alt_err_vals)), 2)),
    ("alt_error_std_cm",          round(float(np.std(alt_err_vals)), 2)),
    ("alt_error_ci_lo_cm",        round(err_ci[0], 2)),
    ("alt_error_ci_hi_cm",        round(err_ci[1], 2)),
    ("tools_before_set_mean",     round(float(np.mean(tools_before)), 2)),
    ("tools_before_set_std",      round(float(np.std(tools_before)), 2)),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow([f"ref_{ref_key}", ref_val])
print(f"[C4] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: per-run altitude error
ax1 = axes[0]
bar_colors = ["green" if r["passed"] else "red" for r in all_results]
ax1.bar(range(1, N_RUNS + 1), alt_err_vals, color=bar_colors, alpha=0.75, edgecolor="black")
ax1.axhline(TOLERANCE * 100, color="red", ls="--", lw=1.5,
            label=f"Tolerance {TOLERANCE*100:.0f}cm")
ax1.axhline(np.mean(alt_err_vals), color="navy", ls="-", lw=1.5,
            label=f"Mean={np.mean(alt_err_vals):.2f}cm")
ax1.set_xlabel("Run")
ax1.set_ylabel("Alt error at 1.2m target (cm)")
ax1.set_title(f"Altitude error per run\n(green=pass, red=fail)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# Middle: binary metric rates
ax2 = axes[1]
metrics  = ["Success", "Correct\ntarget", "No\nre-arm", "Alt\nreached"]
rates    = [n_pass/N_RUNS, n_correct/N_RUNS, n_no_rearm/N_RUNS, n_alt_ok/N_RUNS]
ci_los   = [pass_lo, corr_lo, nore_lo, wilson_ci(n_alt_ok, N_RUNS)[0]]
ci_his   = [pass_hi, corr_hi, nore_hi, wilson_ci(n_alt_ok, N_RUNS)[1]]
err_lo2  = [r - l for r, l in zip(rates, ci_los)]
err_hi2  = [h - r for r, h in zip(rates, ci_his)]
ax2.bar(range(4), rates, color=["green","steelblue","steelblue","steelblue"],
        alpha=0.75, edgecolor="black")
ax2.errorbar(range(4), rates, yerr=[err_lo2, err_hi2],
             fmt="none", ecolor="black", capsize=6, lw=1.5)
ax2.set_xticks(range(4))
ax2.set_xticklabels(metrics, fontsize=9)
ax2.set_ylim(0, 1.2)
ax2.set_ylabel("Rate")
ax2.set_title(f"Binary metric rates (N={N_RUNS})\nError bars = Wilson 95% CI")
ax2.grid(True, alpha=0.3, axis="y")
for i, (r, lo, hi) in enumerate(zip(rates, ci_los, ci_his)):
    ax2.text(i, r + 0.05, f"{r:.2f}", ha="center", fontsize=9)

# Right: z trajectory per run (phase1 and correction zones)
ax3 = axes[2]
ax3.axhline(INITIAL_TARGET, color="orange", ls=":", lw=1.2, alpha=0.7,
            label=f"Initial target {INITIAL_TARGET}m")
ax3.axhline(CORRECT_TARGET, color="purple", ls=":", lw=1.2, alpha=0.7,
            label=f"Corrected target {CORRECT_TARGET}m")
ax3.axhspan(CORRECT_TARGET - TOLERANCE, CORRECT_TARGET + TOLERANCE,
            alpha=0.08, color="purple")
run_colors = plt.cm.Blues(np.linspace(0.35, 0.85, N_RUNS))
for i, r in enumerate(all_results):
    label = f"Run {i+1} ({'✓' if r['passed'] else '✗'})"
    ax3.scatter([1, 2], [r["z_phase1_m"], r["z_final_m"]],
                color=run_colors[i], s=80, zorder=5)
    ax3.plot([1, 2], [r["z_phase1_m"], r["z_final_m"]],
             color=run_colors[i], lw=1.5, alpha=0.7, label=label)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(["After\nphase 1\n(0.5m target)", "After\ncorrection\n(1.2m target)"])
ax3.set_ylabel("EKF altitude (m)")
ax3.set_title("Altitude before/after correction\n(per run)")
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

fig.suptitle(
    f"EXP-C4: Mid-Mission Correction  (N={N_RUNS} runs, temperature=0.2)\n"
    f"Success: {n_pass}/{N_RUNS}  (95% CI: {pass_lo:.2f}–{pass_hi:.2f})  |  "
    f"Alt error: {np.mean(alt_err_vals):.2f}±{np.std(alt_err_vals):.2f} cm",
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C4] Plot: {OUT_PNG}")

print(f"\n[C4] RESULT: {n_pass}/{N_RUNS} passed  "
      f"(95% CI: {pass_lo:.2f}–{pass_hi:.2f})")
