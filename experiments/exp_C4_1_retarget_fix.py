"""
EXP-C4.1: Mid-Mission Correction — Re-Targeting Protocol Fix  (N=5 runs)
=========================================================================
Identical to EXP-C4 in every respect (same two-phase structure, same commands,
same drone setup, same N=5, same scoring) EXCEPT one change: the system prompt
is augmented with a MID-MISSION RE-TARGETING PROTOCOL.

WHY C4 FAILED:
--------------
C4 (2/5, 40%) produced two distinct failure modes:

  Failure A — Silent freeze (Runs 1 & 2, api_calls_ph2 = 0):
    The LLM completed Phase 1 (drone at 0.5 m, mission "done") and when the
    correction arrived, it made ZERO tool calls. It treated the correction as a
    conversational remark, not an action trigger. The LLM had no frame for
    "I am hovering at target AND a new target just arrived → act now."

  Failure B — Absolute vs relative confusion (Run 4, z_final = 0.803 m):
    The LLM called set_altitude_target but with the wrong value (~0.8 m instead
    of 1.2 m). It likely interpreted "take it to 1.2 m" as a relative increment
    from the current state rather than an absolute target from the ground.

WHAT THIS FIX IS — AND ISN'T:
------------------------------
WRONG fix: hardcode specific commands in the prompt:
  "if you see 'actually go to 1.2m', call set_altitude_target(1.2)"
  That pre-answers the test case. Not a fix — a cheat.

RIGHT fix: ONE general structural rule that closes BOTH failure modes without
naming any specific altitude value or command phrasing.

The rule describes the PATTERN, not the examples:
  • Any message received while the drone is airborne + hovering that names
    a new altitude = a RE-TARGETING command, not a new mission start.
  • All altitude values in such commands are absolute metres from the ground.
  • Act on it immediately — never treat it as purely informational.

This protocol:
  • Is general (applies to "go to 2.0m", "new target 0.8m", "climb to 1.5m", etc.)
  • Fixes Failure A (explicit "act immediately" clause)
  • Fixes Failure B (explicit "absolute, not relative" clause)
  • Does not restrict what targets can be given — any valid altitude works
  • Is testable (we check whether set_altitude_target(1.2) is called correctly)

EXPECTED C4.1 OUTCOME:
-----------------------
  • Freeze runs (1, 2): Fixed — "act immediately" eliminates silent non-response.
  • Wrong-target run (4): Fixed — "absolute metres from ground" eliminates
    relative-increment misinterpretation.
  • Passing runs (3, 5): Still pass — consistent with what they already did.
  Expected result: 4/5 or 5/5 (vs C4 baseline of 2/5).

Outputs:
  results/C4_1_runs.csv        — per-run metrics
  results/C4_1_summary.csv     — aggregate statistics
  results/C4_1_retarget_fix.png
"""

import sys, os, csv, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import c_series_agent    # import module so we can patch SYSTEM_PROMPT

# ── Patch the system prompt BEFORE creating any SimAgent ──────────────────────
#
# ONE general structural rule — no specific altitudes named, no specific
# phrases named. Describes the pattern by drone state + command structure.
# Covers both failure modes: re-tasking trigger AND absolute interpretation.
#
RETARGETING_PROTOCOL = """
━━ MID-MISSION RE-TARGETING PROTOCOL ━━

When the drone is AIRBORNE and HOVERING at a target altitude, and a message
specifies a new altitude — regardless of how it is phrased:

  1. CLASSIFY it as a RE-TARGETING command, not a new mission start.
     Do NOT disarm, re-arm, or restart the takeoff sequence.
     The drone state is valid — only the altitude target changes.

  2. The altitude value is ALWAYS absolute metres from the ground.
     Never interpret it as a relative increment from the current altitude.
     "take it to X m" means set_altitude_target(X), not current_z + X.

  3. Act IMMEDIATELY — never treat a re-targeting command as informational:
       set_altitude_target(X)
       wait(4.0)
       check_altitude_reached(X, 0.10)

  4. Altitude hold mode:
       If already active  → keep it active, just call set_altitude_target(X).
       If not active      → call enable_altitude_hold() first, then set target.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

c_series_agent.SYSTEM_PROMPT = c_series_agent.SYSTEM_PROMPT + RETARGETING_PROTOCOL

from c_series_agent import SimAgent    # import class AFTER patching the module constant

# ── Output paths ──────────────────────────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

# ── Guardrail toggle (--guardrail on|off) ──────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--guardrail", choices=["on", "off"], default="on")
_args, _ = _parser.parse_known_args()
GUARDRAIL_ENABLED = _args.guardrail == "on"
GUARDRAIL_SUFFIX  = "guardrail_on" if GUARDRAIL_ENABLED else "guardrail_off"

OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", f"C4_1_runs_{GUARDRAIL_SUFFIX}.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", f"C4_1_summary_{GUARDRAIL_SUFFIX}.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", f"C4_1_retarget_fix_{GUARDRAIL_SUFFIX}.png")

# ── Identical experimental parameters to C4 ───────────────────────────────────
INITIAL_CMD    = "hover at 0.5 metres"
CORRECTION_CMD = "actually go to 1.2 metres instead"
INITIAL_TARGET = 0.5
CORRECT_TARGET = 1.2
TOLERANCE      = 0.12
N_RUNS         = 5

# C4 baseline results for comparison plots
C4_BASELINE = {
    "n_pass":          2,
    "success_rate":    0.40,
    "ci_lo":           0.12,
    "ci_hi":           0.77,
    "alt_error_mean":  36.3,
    "alt_error_std":   31.2,
    # per-run z_final for comparison
    "z_finals":        [0.497, 0.497, 1.205, 0.803, 1.207],
    "passed":          [0, 0, 1, 0, 1],
    "api_calls_ph2":   [0, 0, 8, 8, 8],
}

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Correction is processed as a new observation in the running ReAct loop."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "LLM infers current drone state from conversation history to avoid re-arm."
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

# ── Single-run function  (identical structure to C4) ──────────────────────────

def run_once(run_idx):
    print(f"\n[C4.1] ── Run {run_idx+1}/{N_RUNS} ───────────────────────────────")
    agent   = SimAgent(session_id=f"C4_1_run{run_idx}", guardrail_enabled=GUARDRAIL_ENABLED)
    history = []

    # ── Phase 1: initial hover command ────────────────────────────────────────
    print(f"  [Phase 1] '{INITIAL_CMD}'")
    text1, stats1, trace1, _ = agent.run_agent_loop(
        INITIAL_CMD, history=list(history), max_turns=15,
    )
    history.append({"role": "user",      "content": INITIAL_CMD})
    history.append({"role": "assistant", "content": [{"type": "text",
                                                       "text": text1 or "Done."}]})

    agent.wait_sim(4.0)
    with agent.state.lock:
        z_phase1    = round(agent.state.ekf_z, 3)
        armed_ph1   = agent.state.armed
        althold_ph1 = agent.state.althold

    print(f"  [Phase 1 complete] z={z_phase1:.3f}m  armed={armed_ph1}  althold={althold_ph1}")

    # ── Phase 2: mid-mission correction ───────────────────────────────────────
    print(f"  [Phase 2] '{CORRECTION_CMD}'")
    text2, stats2, trace2, _ = agent.run_agent_loop(
        CORRECTION_CMD, history=list(history), max_turns=8,
    )

    agent.wait_sim(6.0)
    with agent.state.lock:
        z_final     = round(agent.state.ekf_z, 3)
        armed_final = agent.state.armed

    tools_phase2      = [t["name"] for t in trace2]
    set_alt_calls     = [t for t in trace2 if t["name"] == "set_altitude_target"]
    correct_target_set= any(abs(t["args"].get("meters", 0) - CORRECT_TARGET) < 0.15
                            for t in set_alt_calls)
    re_armed          = "arm" in tools_phase2
    re_took_off       = "find_hover_throttle" in tools_phase2
    alt_reached       = abs(z_final - CORRECT_TARGET) <= TOLERANCE
    passed            = correct_target_set and not re_armed and not re_took_off

    first_set_idx = next((i for i, t in enumerate(tools_phase2)
                          if t == "set_altitude_target"), None)
    meta_tools    = {"plan_workflow", "report_progress"}
    non_meta_before = [t for t in
                       (tools_phase2[:first_set_idx] if first_set_idx is not None
                        else tools_phase2)
                       if t not in meta_tools]

    n_api1 = len(stats1); n_api2 = len(stats2)
    in_tok = sum(s["input_tokens"]  for s in stats1 + stats2)
    out_tok= sum(s["output_tokens"] for s in stats1 + stats2)
    cost   = sum(s["cost_usd"]      for s in stats1 + stats2)

    # Classify failure mode if failed
    if not passed:
        if n_api2 == 0:
            failure_mode = "freeze"          # no tool calls in Ph2
        elif not correct_target_set:
            failure_mode = "wrong_target"    # called tool but wrong value
        else:
            failure_mode = "other"
    else:
        failure_mode = "none"

    print(f"  z_phase1={z_phase1:.3f}m  z_final={z_final:.3f}m  "
          f"correct={correct_target_set}  re_armed={re_armed}  "
          f"pass={passed}  failure={failure_mode}")

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
        "failure_mode":       failure_mode,
        "api_calls_ph1":      n_api1,
        "api_calls_ph2":      n_api2,
        "input_tokens":       in_tok,
        "output_tokens":      out_tok,
        "cost_usd":           round(cost, 6),
        "tools_ph2":          ";".join(tools_phase2[:12]),
    }

# ── Run N times ────────────────────────────────────────────────────────────────

all_results = [run_once(i) for i in range(N_RUNS)]

# ── Aggregate statistics ───────────────────────────────────────────────────────

def col(key):
    return [r[key] for r in all_results]

n_pass     = sum(col("passed"))
n_correct  = sum(col("correct_target_set"))
n_no_rearm = sum(1 for r in all_results if not r["re_armed"])
n_alt_ok   = sum(col("alt_reached"))
n_freeze   = sum(1 for r in all_results if r["failure_mode"] == "freeze")
n_wrong_tgt= sum(1 for r in all_results if r["failure_mode"] == "wrong_target")

pass_lo,  pass_hi  = wilson_ci(n_pass,    N_RUNS)
corr_lo,  corr_hi  = wilson_ci(n_correct, N_RUNS)
nore_lo,  nore_hi  = wilson_ci(n_no_rearm,N_RUNS)
alt_lo,   alt_hi   = wilson_ci(n_alt_ok,  N_RUNS)

alt_err_vals = col("alt_error_cm")
err_ci = bootstrap_ci(alt_err_vals)

delta_pp = round((n_pass / N_RUNS - C4_BASELINE["success_rate"]) * 100, 1)

print(f"\n[C4.1] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
print(f"  Success rate:       {n_pass}/{N_RUNS}  ({n_pass/N_RUNS:.0%})  "
      f"CI=[{pass_lo:.2f},{pass_hi:.2f}]")
print(f"  vs C4 baseline:     2/5 (40%)  → Δ = {delta_pp:+.1f} pp")
print(f"  Correct target set: {n_correct}/{N_RUNS}  CI=[{corr_lo:.2f},{corr_hi:.2f}]")
print(f"  No re-arm:          {n_no_rearm}/{N_RUNS}  CI=[{nore_lo:.2f},{nore_hi:.2f}]")
print(f"  Freeze runs:        {n_freeze}/{N_RUNS}  (was 2/5 in C4)")
print(f"  Wrong-target runs:  {n_wrong_tgt}/{N_RUNS}  (was 1/5 in C4)")
print(f"  Alt error (cm):     {np.mean(alt_err_vals):.2f}±{np.std(alt_err_vals):.2f}  "
      f"CI=[{err_ci[0]:.2f},{err_ci[1]:.2f}]")

# ── Save CSVs ──────────────────────────────────────────────────────────────────

with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_results[0].keys())
    w.writeheader()
    w.writerows(all_results)
print(f"\n[C4.1] Per-run CSV : {OUT_RUNS}")

summary_rows = [
    ("experiment",              "C4.1"),
    ("n_runs",                  N_RUNS),
    ("n_pass",                  n_pass),
    ("success_rate",            round(n_pass / N_RUNS, 3)),
    ("success_rate_ci_lo",      round(pass_lo, 3)),
    ("success_rate_ci_hi",      round(pass_hi, 3)),
    ("vs_c4_baseline_rate",     C4_BASELINE["success_rate"]),
    ("delta_percentage_points", delta_pp),
    ("correct_target_rate",     round(n_correct / N_RUNS, 3)),
    ("correct_target_ci_lo",    round(corr_lo, 3)),
    ("correct_target_ci_hi",    round(corr_hi, 3)),
    ("no_rearm_rate",           round(n_no_rearm / N_RUNS, 3)),
    ("no_rearm_ci_lo",          round(nore_lo, 3)),
    ("no_rearm_ci_hi",          round(nore_hi, 3)),
    ("alt_reached_rate",        round(n_alt_ok / N_RUNS, 3)),
    ("n_freeze_failures",       n_freeze),
    ("n_wrong_target_failures", n_wrong_tgt),
    ("alt_error_mean_cm",       round(float(np.mean(alt_err_vals)), 2)),
    ("alt_error_std_cm",        round(float(np.std(alt_err_vals)), 2)),
    ("alt_error_ci_lo_cm",      round(err_ci[0], 2)),
    ("alt_error_ci_hi_cm",      round(err_ci[1], 2)),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow([f"ref_{ref_key}", ref_val])
print(f"[C4.1] Summary CSV : {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f"EXP-C4.1: Mid-Mission Correction — Re-Targeting Protocol  "
    f"(N={N_RUNS}, temperature=0.2)\n"
    f"C4 baseline: 2/5 (40%)  →  C4.1: {n_pass}/{N_RUNS} "
    f"({n_pass/N_RUNS:.0%}, CI: {pass_lo:.2f}–{pass_hi:.2f})  "
    f"  Δ = {delta_pp:+.1f} pp",
    fontsize=12, fontweight="bold"
)

run_labels = [f"Run {r['run']}" for r in all_results]
pass_colors = ["#2ecc71" if r["passed"] else "#e74c3c" for r in all_results]

# ── Fig A: C4 vs C4.1 success rate comparison ─────────────────────────────────
ax = axes[0, 0]
exps    = ["C4\n(baseline)", "C4.1\n(fix)"]
rates   = [C4_BASELINE["success_rate"], n_pass / N_RUNS]
ci_los_ = [C4_BASELINE["ci_lo"], pass_lo]
ci_his_ = [C4_BASELINE["ci_hi"], pass_hi]
bar_cols = ["#e74c3c", "#2ecc71" if n_pass > C4_BASELINE["n_pass"] else "#e67e22"]
bars = ax.bar(exps, rates, color=bar_cols, alpha=0.85, edgecolor="black", width=0.45)
ax.errorbar([0, 1], rates,
            yerr=[[r - l for r, l in zip(rates, ci_los_)],
                  [h - r for r, h in zip(rates, ci_his_)]],
            fmt="none", ecolor="black", capsize=8, lw=2)
for xi, (r, lo, hi) in enumerate(zip(rates, ci_los_, ci_his_)):
    n = int(round(r * N_RUNS))
    ax.text(xi, r + 0.05, f"{n}/{N_RUNS}\n({r:.0%})", ha="center",
            fontsize=11, fontweight="bold")
    ax.text(xi, 0.04, f"CI\n[{lo:.2f}–{hi:.2f}]", ha="center",
            fontsize=8, color="#555")
ax.set_ylim(0, 1.3)
ax.set_ylabel("Success rate", fontsize=11)
ax.set_title("C4 vs C4.1: Overall Success Rate", fontsize=11, fontweight="bold")
ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.4, label="50% line")
ax.grid(True, axis="y", alpha=0.3)
if delta_pp > 0:
    ax.annotate(f"Δ = {delta_pp:+.1f} pp", xy=(0.5, max(rates) + 0.12),
                ha="center", fontsize=12, fontweight="bold",
                color="#27ae60" if delta_pp > 0 else "#e74c3c")

# ── Fig B: Per-run altitude error C4 vs C4.1 ─────────────────────────────────
ax = axes[0, 1]
x = np.arange(N_RUNS)
w = 0.35
c4_errs  = [abs(z - CORRECT_TARGET) * 100 for z in C4_BASELINE["z_finals"]]
c41_errs = col("alt_error_cm")
c4_cols  = ["#27ae60" if p else "#e74c3c" for p in C4_BASELINE["passed"]]
c41_cols = ["#27ae60" if r["passed"] else "#e74c3c" for r in all_results]

ax.bar(x - w/2, c4_errs,  w, color=c4_cols,  alpha=0.6, edgecolor="black",
       label="C4 (baseline)")
ax.bar(x + w/2, c41_errs, w, color=c41_cols, alpha=0.85, edgecolor="black",
       label="C4.1 (fix)")
ax.axhline(TOLERANCE * 100, color="purple", lw=1.5, ls="--",
           label=f"Pass threshold {TOLERANCE*100:.0f} cm")
ax.set_xticks(x)
ax.set_xticklabels(run_labels, fontsize=9)
ax.set_ylabel("Altitude error from 1.2 m (cm)", fontsize=10)
ax.set_title("Per-Run Altitude Error\nC4 vs C4.1  (green=pass, red=fail)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# ── Fig C: Phase 2 altitude trajectory comparison ─────────────────────────────
ax = axes[0, 2]
ax.axhline(INITIAL_TARGET, color="orange", ls=":", lw=1.5, alpha=0.7,
           label=f"Initial target {INITIAL_TARGET} m")
ax.axhline(CORRECT_TARGET, color="purple", ls="-", lw=1.5, alpha=0.7,
           label=f"Corrected target {CORRECT_TARGET} m")
ax.axhspan(CORRECT_TARGET - TOLERANCE, CORRECT_TARGET + TOLERANCE,
           alpha=0.08, color="purple", label="±12 cm pass band")

cmap_r = plt.cm.Blues(np.linspace(0.35, 0.85, N_RUNS))
for i, (r, c4_z) in enumerate(zip(all_results, C4_BASELINE["z_finals"])):
    # C4 as dashed
    ax.plot([1, 2], [r["z_phase1_m"], c4_z], "--",
            color=cmap_r[i], lw=1.2, alpha=0.5)
    # C4.1 as solid
    ax.plot([1, 2], [r["z_phase1_m"], r["z_final_m"]], "-o",
            color=cmap_r[i], lw=2, ms=7,
            label=f"R{i+1} ({'✓' if r['passed'] else '✗'})")

ax.set_xticks([1, 2])
ax.set_xticklabels(["After Phase 1\n(target 0.5 m)",
                    "After Correction\n(target 1.2 m)"])
ax.set_ylabel("EKF altitude (m)", fontsize=10)
ax.set_title("Altitude Trajectory Per Run\n(solid=C4.1, dashed=C4 baseline)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ── Fig D: Phase 2 API calls  (freeze detection) ──────────────────────────────
ax = axes[1, 0]
c4_api2  = C4_BASELINE["api_calls_ph2"]
c41_api2 = col("api_calls_ph2")
bar_cols_c4  = ["#27ae60" if p else "#e74c3c" for p in C4_BASELINE["passed"]]
bar_cols_c41 = ["#27ae60" if r["passed"] else "#e74c3c" for r in all_results]

ax.bar(x - w/2, c4_api2,  w, color=bar_cols_c4,  alpha=0.6, edgecolor="black",
       label="C4 (baseline)")
ax.bar(x + w/2, c41_api2, w, color=bar_cols_c41, alpha=0.85, edgecolor="black",
       label="C4.1 (fix)")
ax.axhline(0, color="red", lw=1.5, ls="--", alpha=0.5)
ax.annotate("← C4 freeze runs\n(0 tool calls)", xy=(0.5, 0.3),
            fontsize=8, color="#e74c3c", style="italic")
ax.set_xticks(x)
ax.set_xticklabels(run_labels, fontsize=9)
ax.set_ylabel("Phase 2 API calls", fontsize=10)
ax.set_title("Phase 2 LLM Activity (API calls)\n"
             "C4 Runs 1&2: 0 calls = freeze failure",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# ── Fig E: Binary metric comparison C4 vs C4.1 ────────────────────────────────
ax = axes[1, 1]
metrics_labels = ["Success\nrate", "Correct\ntarget", "No\nre-arm", "Alt\nreached"]
c4_rates   = [C4_BASELINE["success_rate"], 0.40, 1.00, 0.40]
c41_rates  = [n_pass/N_RUNS, n_correct/N_RUNS, n_no_rearm/N_RUNS, n_alt_ok/N_RUNS]
x2 = np.arange(len(metrics_labels))
ax.bar(x2 - w/2, c4_rates,  w, color="#e74c3c", alpha=0.7, edgecolor="black",
       label="C4 baseline")
ax.bar(x2 + w/2, c41_rates, w, color="#2ecc71", alpha=0.85, edgecolor="black",
       label="C4.1 fix")
for xi, (c4r, c41r) in enumerate(zip(c4_rates, c41_rates)):
    dp = (c41r - c4r) * 100
    color = "#27ae60" if dp > 0 else ("#e74c3c" if dp < 0 else "grey")
    ax.text(xi, max(c4r, c41r) + 0.05, f"{dp:+.0f} pp",
            ha="center", fontsize=9, fontweight="bold", color=color)
ax.set_xticks(x2)
ax.set_xticklabels(metrics_labels, fontsize=9)
ax.set_ylim(0, 1.35)
ax.set_ylabel("Rate", fontsize=10)
ax.set_title("Binary Metrics: C4 vs C4.1\n(+pp = improvement)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

# ── Fig F: Failure mode breakdown ─────────────────────────────────────────────
ax = axes[1, 2]
n_c4_freeze    = 2; n_c4_wrong = 1; n_c4_pass = 2
n_c41_freeze   = n_freeze
n_c41_wrong    = n_wrong_tgt
n_c41_pass     = n_pass
n_c41_other    = N_RUNS - n_c41_freeze - n_c41_wrong - n_c41_pass

categories = ["Pass", "Freeze\n(no tool calls)", "Wrong\ntarget", "Other\nfailure"]
c4_counts  = [n_c4_pass,  n_c4_freeze,  n_c4_wrong,  0]
c41_counts = [n_c41_pass, n_c41_freeze, n_c41_wrong, n_c41_other]
cat_colors = ["#2ecc71", "#e74c3c", "#e67e22", "#95a5a6"]

x3 = np.arange(len(categories))
bars_c4  = ax.bar(x3 - w/2, c4_counts,  w, color=cat_colors, alpha=0.5,
                  edgecolor="black", label="C4 baseline")
bars_c41 = ax.bar(x3 + w/2, c41_counts, w, color=cat_colors, alpha=0.9,
                  edgecolor="black", label="C4.1 fix")
for xi, (c4v, c41v) in enumerate(zip(c4_counts, c41_counts)):
    if c4v > 0:
        ax.text(xi - w/2, c4v + 0.05, str(c4v), ha="center", fontsize=10,
                fontweight="bold", alpha=0.7)
    if c41v > 0:
        ax.text(xi + w/2, c41v + 0.05, str(c41v), ha="center", fontsize=11,
                fontweight="bold")
ax.set_xticks(x3)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, N_RUNS + 1)
ax.set_ylabel("Number of runs", fontsize=10)
ax.set_title("Failure Mode Breakdown\n(which failures did the protocol fix?)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
ax.set_yticks(range(N_RUNS + 1))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"[C4.1] Plot: {OUT_PNG}")

print(f"\n[C4.1] RESULT: {n_pass}/{N_RUNS} passed  "
      f"(95% CI: {pass_lo:.2f}–{pass_hi:.2f})  "
      f"vs C4 baseline 2/5 (40%)  Δ = {delta_pp:+.1f} pp")
print(f"[C4.1] Freeze failures:      {n_freeze}/5  (was 2/5 in C4)")
print(f"[C4.1] Wrong-target failures:{n_wrong_tgt}/5  (was 1/5 in C4)")
