"""
EXP-C4: Human Correction Mid-Mission
======================================
Scenario: "hover at 0.5m" → executing → interrupt: "actually go to 1.2m"

Measures:
  - Does LLM call set_altitude_target(1.2) without re-arming?
  - Time from correction to new target command (tool calls between interruption and new set)
  - No unnecessary steps (no re-arm, no re-takeoff)

Expected: LLM directly calls set_altitude_target(1.2) without full re-sequence.

Outputs:
  results/C4_mid_mission_correction.csv
  results/C4_mid_mission_correction.png
"""

import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "C4_mid_mission_correction.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "C4_mid_mission_correction.png")

INITIAL_CMD    = "hover at 0.5 metres"
CORRECTION_CMD = "actually go to 1.2 metres instead"
INITIAL_TARGET = 0.5
CORRECT_TARGET = 1.2
TOLERANCE      = 0.12  # m

print(f"[C4] Phase 1: \"{INITIAL_CMD}\"")

agent   = SimAgent(session_id="C4")
history = []

# ── Phase 1: Initial command ──────────────────────────────────────────────────
t_phase1_start = agent.sim_time

text1, stats1, trace1 = agent.run_agent_loop(
    INITIAL_CMD,
    history=list(history),
    max_turns=15,
)

# Accumulate history
history.append({"role": "user", "content": INITIAL_CMD})
history.append({"role": "assistant", "content": [{"type": "text", "text": text1}]})

# Let drone settle briefly at 0.5 m
agent.wait_sim(4.0)

with agent.state.lock:
    z_phase1 = round(agent.state.ekf_z, 3)
    armed_phase1 = agent.state.armed
    althold_phase1 = agent.state.althold

t_phase1_end = agent.sim_time
print(f"[C4] After phase 1: z={z_phase1:.3f}m, armed={armed_phase1}, althold={althold_phase1}")
print(f"[C4] Phase 1 tools: {[t['name'] for t in trace1]}")

# ── Phase 2: Mid-mission correction ──────────────────────────────────────────
print(f"\n[C4] Phase 2 (correction): \"{CORRECTION_CMD}\"")
t_correction = agent.sim_time

text2, stats2, trace2 = agent.run_agent_loop(
    CORRECTION_CMD,
    history=list(history),
    max_turns=8,
)

# Wait for drone to reach new target
agent.wait_sim(6.0)

with agent.state.lock:
    z_final     = round(agent.state.ekf_z, 3)
    armed_final = agent.state.armed
    alt_sp_final= round(agent.state.alt_sp, 3)

t_correction_end = agent.sim_time
tools_phase2 = [t["name"] for t in trace2]

# ── Metrics ───────────────────────────────────────────────────────────────────

# 1. Did LLM set the correct target?
set_alt_calls = [t for t in trace2 if t["name"] == "set_altitude_target"]
correct_target_set = any(
    abs(t["args"].get("meters", 0) - CORRECT_TARGET) < 0.15
    for t in set_alt_calls
)

# 2. Did LLM avoid re-arming?
re_armed = "arm" in tools_phase2

# 3. Did LLM avoid re-takeoff (find_hover_throttle)?
re_took_off = "find_hover_throttle" in tools_phase2

# 4. How many tool calls between correction and set_altitude_target?
first_set_idx = next((i for i, t in enumerate(trace2)
                      if t["name"] == "set_altitude_target"), None)
tools_before_set = tools_phase2[:first_set_idx] if first_set_idx is not None else tools_phase2
# Remove meta tools (plan_workflow, report_progress, wait)
meta_tools = {"plan_workflow", "report_progress"}
non_meta_before = [t for t in tools_before_set if t not in meta_tools]
tools_to_correction = len(non_meta_before)

# 5. Altitude reached?
alt_reached = abs(z_final - CORRECT_TARGET) <= TOLERANCE

# Combined pass
passed = correct_target_set and not re_armed and not re_took_off

# API stats
n_api1 = len(stats1);   in1 = sum(s["input_tokens"] for s in stats1)
n_api2 = len(stats2);   in2 = sum(s["input_tokens"] for s in stats2)
out1 = sum(s["output_tokens"] for s in stats1)
out2 = sum(s["output_tokens"] for s in stats2)
cost  = sum(s["cost_usd"] for s in stats1 + stats2)

print(f"\n[C4] ── METRICS ──────────────────────────────────────────────")
print(f"  Phase 1 target ({INITIAL_TARGET}m):")
print(f"    z after phase 1:    {z_phase1:.3f} m")
print(f"    Phase 1 tools:      {[t['name'] for t in trace1][:8]}")
print(f"  Correction to {CORRECT_TARGET}m:")
print(f"    Correct target set: {correct_target_set}  (expected set_altitude_target(~1.2))")
print(f"    Re-armed:           {re_armed}            (expected: False)")
print(f"    Re-took-off:        {re_took_off}         (expected: False)")
print(f"    Tools before target: {tools_to_correction} non-meta calls")
print(f"    Phase 2 tools:      {tools_phase2[:8]}")
print(f"    z_final:            {z_final:.3f} m  (target {CORRECT_TARGET}m)")
print(f"    Alt reached:        {alt_reached}  (tolerance ±{TOLERANCE*100:.0f}cm)")
print(f"  API calls ph1/ph2:  {n_api1}/{n_api2}")
print(f"  Tokens in:          {in1+in2}  out: {out1+out2}")
print(f"  PASS:               {passed}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = [
    {
        "phase":           "initial",
        "command":         INITIAL_CMD,
        "target_m":        INITIAL_TARGET,
        "z_achieved_m":    z_phase1,
        "tools_used":      ";".join([t["name"] for t in trace1]),
        "api_calls":       n_api1,
        "input_tokens":    in1,
        "output_tokens":   out1,
    },
    {
        "phase":           "correction",
        "command":         CORRECTION_CMD,
        "target_m":        CORRECT_TARGET,
        "z_achieved_m":    z_final,
        "tools_used":      ";".join(tools_phase2),
        "api_calls":       n_api2,
        "input_tokens":    in2,
        "output_tokens":   out2,
    },
]
metrics_row = {
    "correct_target_set":     correct_target_set,
    "re_armed":               re_armed,
    "re_took_off":            re_took_off,
    "tools_before_correction":tools_to_correction,
    "alt_reached":            alt_reached,
    "alt_error_cm":           round(abs(z_final - CORRECT_TARGET) * 100, 1),
    "passed":                 passed,
    "total_cost_usd":         round(cost, 6),
}
with open(OUT_CSV, "w", newline="") as f:
    all_keys = list(rows[0].keys()) + list(metrics_row.keys())
    w = csv.DictWriter(f, fieldnames=["phase","command","target_m","z_achieved_m",
                                       "tools_used","api_calls","input_tokens","output_tokens"])
    w.writeheader()
    w.writerows(rows)
    # Append metrics as extra row
    w.writerow({"phase": "METRICS", "command": str(metrics_row), "target_m":"",
                "z_achieved_m":"","tools_used":"","api_calls":"","input_tokens":"","output_tokens":""})
print(f"[C4] CSV: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
tel = agent.get_telem_arrays()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

if len(tel.get("t", [])) > 0:
    t_s  = tel["t"] / 1000.0
    ax1.plot(t_s, tel["z_true"],          color="blue",  lw=1.5, label="True altitude")
    ax1.plot(t_s, tel["lw_z"] / 1000.0,  color="green", lw=1.2, label="EKF altitude", alpha=0.8)
    ax1.step(t_s, tel["altsp"] / 1000.0, color="red",   lw=1.5, ls="--", label="Setpoint", where="post")

# Mark phase boundary
ax1.axvline(t_phase1_end, color="black", ls="--", lw=1.5, label=f"Correction @ {t_phase1_end:.1f}s")
ax1.axhline(INITIAL_TARGET, color="orange", ls=":", lw=1, alpha=0.6, label=f"Initial target {INITIAL_TARGET}m")
ax1.axhline(CORRECT_TARGET, color="purple", ls=":", lw=1, alpha=0.6, label=f"Corrected target {CORRECT_TARGET}m")
ax1.axhspan(CORRECT_TARGET - TOLERANCE, CORRECT_TARGET + TOLERANCE, alpha=0.08, color="purple")

ax1.set_ylabel("Altitude (m)")
ax1.set_title(
    f"EXP-C4: Mid-Mission Correction\n"
    f"Phase 1: \"{INITIAL_CMD}\" → Phase 2: \"{CORRECTION_CMD}\"\n"
    f"Correct target set: {correct_target_set} | "
    f"Re-armed: {re_armed} | "
    f"Tools before set: {tools_to_correction} | "
    f"Alt reached: {alt_reached}"
)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Tool call trace
all_trace = [(t, "phase1", "#3498db") for t in trace1] + [(t, "phase2", "#e74c3c") for t in trace2]
if all_trace:
    ytick_labels = [f'{phase}\n{t["name"]}' for t, phase, _ in all_trace]
    yticks = list(range(len(all_trace)))
    x_vals = [t.get("sim_time_s", 0) for t, _, _ in all_trace]
    colors = [c for _, _, c in all_trace]
    ax2.barh(yticks, [0.5]*len(yticks), left=x_vals, height=0.7,
             color=colors, alpha=0.7)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ytick_labels, fontsize=7)
    ax2.axvline(t_phase1_end, color="black", ls="--", lw=1.5)
    ax2.set_xlabel("Simulated time (s)")
    ax2.set_ylabel("Tool call")
    ax2.set_title("Tool sequence: blue=phase1 initial, red=phase2 correction")
    ax2.grid(True, alpha=0.2, axis="x")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C4] Plot: {OUT_PNG}")

print(f"\n[C4] RESULT: {'PASS' if passed else 'FAIL'}")
print(f"  correct_target_set={correct_target_set}, re_armed={re_armed}, re_took_off={re_took_off}")
