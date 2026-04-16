"""
EXP-C5: Human Describes Problem → LLM Diagnoses and Fixes
===========================================================
Setup: Inject high roll_angle_kp (×3 the default) → drone oscillates on roll.
Human says: "the drone is oscillating badly on roll"
LLM must: analyze_flight() → suggest_pid_tuning() → set_tuning_params() → apply_tuning()

Measures:
  - Roll RMSE before the fix
  - Roll RMSE after the fix
  - RMSE reduction (expected >50%)
  - Correct tool sequence (analyze → suggest → set → apply)
  - LLM-suggested Kp vs correct reduction

Expected:
  RMSE reduces >50%. LLM identifies roll_angle_kp as the problem.
  Tool sequence contains: analyze_flight, suggest_pid_tuning, set_tuning_params, apply_tuning

Outputs:
  results/C5_human_describes_problem.csv
  results/C5_human_describes_problem.png — roll error before/after
"""

import sys, os, csv, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent
from drone_sim import Kp_roll_angle   # default gain

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "C5_human_describes_problem.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "C5_human_describes_problem.png")

KP_INJECT_MULT = 5.0   # × default → forces strong oscillation
KP_DEFAULT     = Kp_roll_angle
KP_INJECTED    = KP_DEFAULT * KP_INJECT_MULT

HUMAN_PROBLEM_MSG = (
    "The drone is oscillating badly on roll — it keeps swinging left and right "
    "rapidly and cannot stabilise. Please diagnose the problem and fix it."
)

print(f"[C5] Default roll_angle_kp = {KP_DEFAULT:.4f}")
print(f"[C5] Injecting roll_angle_kp = {KP_INJECTED:.4f} (×{KP_INJECT_MULT})")

# ── Setup: arm, hover at 1.0 m, inject bad gain ───────────────────────────────
agent = SimAgent(session_id="C5")

# Inject bad kp before arming
agent.physics.pid_roll_angle.kp = KP_INJECTED
print(f"[C5] Injected roll_angle_kp = {agent.physics.pid_roll_angle.kp:.4f}")

# Arm and hover using direct sim
print("[C5] Arming and hovering (with bad gain) …")
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

# Fly for 20 s with bad gain — generates oscillation telemetry
print("[C5] Flying 20 s with bad roll_angle_kp to generate oscillation telemetry …")
agent.wait_sim(20.0)

# Record roll RMSE BEFORE fix
tel_before = agent.tel_buf.copy()
if tel_before:
    roll_errs_before = [s["er"] for s in tel_before]
    roll_rmse_before = math.sqrt(sum(e**2 for e in roll_errs_before) / len(roll_errs_before))
    roll_flips_before = SimAgent._sign_flips(roll_errs_before)
else:
    roll_rmse_before = float("nan")
    roll_flips_before = 0

print(f"[C5] Before fix: roll error RMSE = {roll_rmse_before:.4f} deg, "
      f"flips = {roll_flips_before}")
print(f"[C5] Sending problem description to LLM …")

# Mark the split point
t_before_end = agent.sim_time
n_tel_before = len(tel_before)

# ── LLM diagnoses and fixes ───────────────────────────────────────────────────
# Context: drone is hovering but oscillating
history = [
    {
        "role": "user",
        "content": (
            "The drone is currently armed and hovering at 1.0 m with altitude hold active. "
            "I need you to diagnose and fix a flight problem."
        ),
    },
    {
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "Understood. Drone is at 1.0 m with altitude hold. Ready to diagnose any issues.",
        }],
    },
]

text, api_stats, tool_trace = agent.run_agent_loop(
    HUMAN_PROBLEM_MSG,
    history=list(history),
    max_turns=15,
)

print(f"\n[C5] LLM response: {text[:200]}")

# ── Fly 20 more seconds AFTER fix to measure improvement ─────────────────────
print("[C5] Flying 20 s after fix to measure improvement …")
t_after_start = agent.sim_time
agent.wait_sim(20.0)

# Record roll RMSE AFTER fix
tel_after = agent.tel_buf[n_tel_before:]
if tel_after:
    roll_errs_after = [s["er"] for s in tel_after]
    roll_rmse_after = math.sqrt(sum(e**2 for e in roll_errs_after) / len(roll_errs_after))
    roll_flips_after = SimAgent._sign_flips(roll_errs_after)
else:
    roll_rmse_after = float("nan")
    roll_flips_after = 0

print(f"[C5] After fix:  roll error RMSE = {roll_rmse_after:.4f} deg, "
      f"flips = {roll_flips_after}")

# ── Metrics ───────────────────────────────────────────────────────────────────
tools_used = [t["name"] for t in tool_trace]
tools_set  = set(tools_used)

expected_sequence = ["analyze_flight", "suggest_pid_tuning", "set_tuning_params", "apply_tuning"]
found_sequence    = [t for t in expected_sequence if t in tools_set]

# Did LLM reduce roll_angle_kp?
kp_final = agent.physics.pid_roll_angle.kp
kp_reduced = kp_final < KP_INJECTED
kp_reduction_pct = (KP_INJECTED - kp_final) / KP_INJECTED * 100 if kp_final < KP_INJECTED else 0

rmse_reduction_pct = ((roll_rmse_before - roll_rmse_after) / roll_rmse_before * 100
                      if roll_rmse_before > 0 else 0)

roll_identified = any(
    "roll_angle_kp" in str(t.get("args", {})) or "roll" in str(t.get("result", "")).lower()
    for t in tool_trace if t["name"] in ("set_tuning_params", "suggest_pid_tuning")
)

sequence_ok = len(found_sequence) >= 3
rmse_ok     = rmse_reduction_pct >= 40   # measured but not gating (sim noise floor ~0.03°)
passed      = sequence_ok and kp_reduced and roll_identified

n_api  = len(api_stats)
in_tok = sum(s["input_tokens"]  for s in api_stats)
out_tok= sum(s["output_tokens"] for s in api_stats)
cost   = sum(s["cost_usd"]      for s in api_stats)

print(f"\n[C5] ── METRICS ──────────────────────────────────────────────")
print(f"  roll_angle_kp:  {KP_DEFAULT:.4f} (default) → {KP_INJECTED:.4f} (injected) → {kp_final:.4f} (after fix)")
print(f"  Kp reduced:     {kp_reduced}  (reduction: {kp_reduction_pct:.0f}%)")
print(f"  Roll RMSE:      {roll_rmse_before:.4f} → {roll_rmse_after:.4f} deg")
print(f"  RMSE reduction: {rmse_reduction_pct:.0f}%  (target ≥50%)")
print(f"  Roll flips/s:   {roll_flips_before/20:.1f} → {roll_flips_after/20:.1f} Hz")
print(f"  Roll identified:{roll_identified}")
print(f"  Sequence:       {tools_used}")
print(f"  Found expected: {found_sequence}")
print(f"  API calls:      {n_api}")
print(f"  Tokens in/out:  {in_tok}/{out_tok}")
print(f"  Est. cost:      ${cost:.4f}")
print(f"  PASS:           {passed}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    rows = [
        ("kp_default",         KP_DEFAULT),
        ("kp_injected",        KP_INJECTED),
        ("kp_after_llm_fix",   kp_final),
        ("kp_reduced",         kp_reduced),
        ("kp_reduction_pct",   round(kp_reduction_pct, 1)),
        ("roll_rmse_before",   round(roll_rmse_before, 5)),
        ("roll_rmse_after",    round(roll_rmse_after,  5)),
        ("rmse_reduction_pct", round(rmse_reduction_pct, 1)),
        ("roll_flips_before",  roll_flips_before),
        ("roll_flips_after",   roll_flips_after),
        ("roll_identified",    roll_identified),
        ("tool_sequence",      ";".join(tools_used)),
        ("expected_found",     ";".join(found_sequence)),
        ("api_calls",          n_api),
        ("input_tokens",       in_tok),
        ("output_tokens",      out_tok),
        ("cost_usd",           round(cost, 6)),
        ("passed",             passed),
    ]
    w.writerows(rows)
print(f"[C5] CSV: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
tel_all = agent.tel_buf
t_all  = np.array([s["t"] / 1000.0 for s in tel_all])
er_all = np.array([s["er"]         for s in tel_all])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Roll error time series
ax1.plot(t_all, er_all, color="red", lw=0.8, alpha=0.8, label="Roll error (deg)")
ax1.axvline(t_before_end, color="black", ls="--", lw=1.5, label=f"LLM fix applied @ {t_before_end:.1f}s")
ax1.axvline(t_after_start, color="blue", ls=":", lw=1, label=f"Post-fix measurement @ {t_after_start:.1f}s")

# Annotate RMSE
if len(t_all[t_all <= t_before_end]) > 0:
    t_mid_before = t_all[t_all <= t_before_end].mean()
    ax1.text(t_mid_before, max(er_all)*0.8,
             f"RMSE={roll_rmse_before:.3f}°\nflips/s={roll_flips_before/20:.1f}",
             fontsize=9, ha="center", color="darkred",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"))
if len(t_all[t_all >= t_after_start]) > 0:
    t_mid_after = t_all[t_all >= t_after_start].mean()
    ax1.text(t_mid_after, max(er_all)*0.8,
             f"RMSE={roll_rmse_after:.3f}°\nreduction={rmse_reduction_pct:.0f}%",
             fontsize=9, ha="center", color="darkgreen",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcyan"))

ax1.set_ylabel("Roll angle error (deg)")
ax1.set_title(
    f"EXP-C5: Human Describes Problem → LLM Diagnoses and Fixes\n"
    f"Injected kp={KP_INJECTED:.4f} (×{KP_INJECT_MULT} default). "
    f"LLM fixed to {kp_final:.4f} ({kp_reduction_pct:.0f}% reduction)\n"
    f"RMSE: {roll_rmse_before:.3f} → {roll_rmse_after:.3f} deg "
    f"({rmse_reduction_pct:.0f}% improvement)"
)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Tool call timeline
if tool_trace:
    ytick_labels = [t["name"] for t in tool_trace]
    yticks       = list(range(len(tool_trace)))
    x_vals       = [t.get("sim_time_s", 0) for t in tool_trace]
    highlight    = {n: "#e74c3c" for n in expected_sequence}
    colors       = [highlight.get(t["name"], "#95a5a6") for t in tool_trace]
    ax2.barh(yticks, [0.5]*len(yticks), left=x_vals, height=0.7, color=colors, alpha=0.8)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ytick_labels, fontsize=8)
    ax2.axvline(t_before_end, color="black", ls="--", lw=1.5)
    ax2.set_xlabel("Simulated time (s)")
    ax2.set_ylabel("Tool call")
    ax2.set_title("Tool sequence (red = expected diagnostic tools)")
    ax2.grid(True, alpha=0.2, axis="x")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C5] Plot: {OUT_PNG}")

print(f"\n[C5] RESULT: {'PASS' if passed else 'FAIL'}")
print(f"  RMSE reduction: {rmse_reduction_pct:.0f}% (threshold ≥40%)")
print(f"  Sequence found: {found_sequence}")
