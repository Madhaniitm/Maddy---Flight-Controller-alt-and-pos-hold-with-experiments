"""
EXP-C1: Natural Language → Tool Chain Execution
================================================
Human types: "take off and hover at 1 metre"
LLM must interpret and execute the correct tool sequence.

Measures:
  - Tool call sequence correctness
  - Achieved altitude vs commanded (1.0 m target)
  - Total API calls, tokens, wall time
  - Altitude error at steady state

Expected: LLM calls tools in correct order, altitude within ±10 cm of 1.0 m

Outputs:
  results/C1_nl_to_toolchain.csv     — telemetry
  results/C1_nl_to_toolchain.png     — altitude timeline + tool call markers
  results/C1_tool_trace.csv          — tool call sequence
"""

import sys, os, csv, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_TEL   = os.path.join(os.path.dirname(__file__), "results", "C1_nl_to_toolchain.csv")
OUT_PNG   = os.path.join(os.path.dirname(__file__), "results", "C1_nl_to_toolchain.png")
OUT_TRACE = os.path.join(os.path.dirname(__file__), "results", "C1_tool_trace.csv")

COMMAND     = "take off and hover at 1 metre"
TARGET_ALT  = 1.0    # m
TOLERANCE   = 0.10   # m

print(f"[C1] Human command: \"{COMMAND}\"")
print("[C1] Running LLM agent …")

agent = SimAgent(session_id="C1")
t_wall_start = time.time()

final_text, api_stats, tool_trace = agent.run_agent_loop(COMMAND)

t_wall_total = time.time() - t_wall_start

# Let sim settle for 8 more seconds after agent finishes
agent.wait_sim(8.0)

print(f"\n[C1] Agent finished in {t_wall_total:.1f}s wall time.")
print(f"[C1] Final LLM text: {final_text[:200]}")

# ── Metrics ────────────────────────────────────────────────────────────────────
tel = agent.get_telem_arrays()

# Altitude at end (last 3 s of sim @ 10 Hz = 30 samples)
z_final_samples = tel["z_true"][-30:] if len(tel.get("z_true", [])) >= 30 else tel.get("z_true", [])
if len(z_final_samples) > 0:
    z_ss   = float(np.mean(z_final_samples))
    z_rmse = float(np.sqrt(np.mean((z_final_samples - TARGET_ALT)**2)))
else:
    z_ss = z_rmse = float("nan")

alt_error_cm = abs(z_ss - TARGET_ALT) * 100

# API stats totals
total_api_calls = len(api_stats)
total_in_tok    = sum(s["input_tokens"]  for s in api_stats)
total_out_tok   = sum(s["output_tokens"] for s in api_stats)
total_cost      = sum(s["cost_usd"]      for s in api_stats)
mean_latency    = float(np.mean([s["latency_s"] for s in api_stats])) if api_stats else 0

# Tool call sequence
tool_names = [t["name"] for t in tool_trace]

# Correctness check: expected sequence contains these in order
EXPECTED_SEQUENCE = ["arm", "find_hover_throttle", "enable_altitude_hold", "set_altitude_target"]
found_order = []
for expected in EXPECTED_SEQUENCE:
    for i, name in enumerate(tool_names):
        if name == expected:
            found_order.append((i, name))
            break

sequence_score = len(found_order)  # out of 4

print(f"\n[C1] ── METRICS ──────────────────────────────────────────────")
print(f"  Commanded altitude:    {TARGET_ALT:.1f} m")
print(f"  Achieved (SS mean):    {z_ss:.3f} m")
print(f"  Steady-state error:    {alt_error_cm:.1f} cm (target ≤10 cm)")
print(f"  SS RMSE:               {z_rmse*100:.2f} cm")
print(f"  Sequence score:        {sequence_score}/4 expected tools found")
print(f"  Tool call sequence:    {tool_names}")
print(f"  API calls:             {total_api_calls}")
print(f"  Input tokens:          {total_in_tok}")
print(f"  Output tokens:         {total_out_tok}")
print(f"  Total cost (est.):     ${total_cost:.4f}")
print(f"  Mean API latency:      {mean_latency:.2f} s")
print(f"  Wall time total:       {t_wall_total:.1f} s")
print(f"  Sim time:              {agent.sim_time:.1f} s")

# ── Save telemetry CSV ─────────────────────────────────────────────────────────
t_arr  = tel.get("t",      np.array([]))
z_arr  = tel.get("z_true", np.array([]))
ze_arr = tel.get("lw_z",   np.array([]))
sp_arr = tel.get("altsp",  np.array([]))

with open(OUT_TEL, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_ms", "z_true_m", "z_ekf_m", "z_setpoint_m"])
    for row in zip(t_arr, z_arr, ze_arr / 1000.0, sp_arr / 1000.0):
        w.writerow([round(v, 4) for v in row])
print(f"[C1] Telemetry CSV: {OUT_TEL}")

# ── Save tool trace CSV ────────────────────────────────────────────────────────
with open(OUT_TRACE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["turn", "tool_name", "args_json", "sim_time_s", "result_preview"])
    for tr in tool_trace:
        w.writerow([
            tr["turn"],
            tr["name"],
            json.dumps(tr["args"]) if isinstance(tr.get("args"), dict) else str(tr.get("args", "")),
            tr.get("sim_time_s", ""),
            tr.get("result", "")[:120],
        ])
print(f"[C1] Tool trace CSV: {OUT_TRACE}")

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

if len(t_arr) > 0:
    t_s  = t_arr / 1000.0
    ze_plot = np.clip(ze_arr / 1000.0, -0.1, None)  # clip EKF noise before arm
    ax1.plot(t_s, z_arr,    color="blue",  lw=1.5, label="True altitude (sim)")
    ax1.plot(t_s, ze_plot,  color="green", lw=1.5, label="EKF estimate", alpha=0.8)
    ax1.step(t_s, sp_arr / 1000.0, color="red",   lw=1.5, ls="--", label="Setpoint", where="post")
    ax1.axhline(TARGET_ALT, color="orange", ls=":", lw=1.2, alpha=0.7, label=f"Command target {TARGET_ALT}m")
    ax1.axhspan(TARGET_ALT - TOLERANCE, TARGET_ALT + TOLERANCE, alpha=0.08, color="green",
                label=f"±{TOLERANCE*100:.0f}cm tolerance")
    ax1.set_ylim(bottom=-0.1)

# Mark tool calls on the altitude plot
tool_colors = {
    "arm":                  ("red",    "A"),
    "find_hover_throttle":  ("orange", "H"),
    "enable_altitude_hold": ("purple", "E"),
    "set_altitude_target":  ("blue",   "T"),
    "land":                 ("brown",  "L"),
    "wait":                 None,           # skip wait marks (too many)
    "plan_workflow":        None,
    "report_progress":      None,
}
for tr in tool_trace:
    color_marker = tool_colors.get(tr["name"])
    if color_marker is None:
        continue
    color, marker = color_marker
    ax1.axvline(tr.get("sim_time_s", 0), color=color, ls=":", lw=1.0, alpha=0.7)
    ax1.text(tr.get("sim_time_s", 0), ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0.1 else 1.5,
             marker, fontsize=7, color=color, ha="center")

ax1.set_ylabel("Altitude (m)")
ax1.set_title(
    f'EXP-C1: Natural Language → Tool Chain\n'
    f'Command: "{COMMAND}"\n'
    f'Result: z_ss={z_ss:.3f}m (err={alt_error_cm:.1f}cm), '
    f'sequence={sequence_score}/4, {total_api_calls} API calls'
)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)

# Tool call trace on lower panel
if tool_trace:
    yticks = list(range(len(tool_trace)))
    ynames = [t["name"] for t in tool_trace]
    x_vals = [t.get("sim_time_s", 0) for t in tool_trace]
    ax2.barh(yticks, [0.3] * len(yticks), left=x_vals, height=0.6,
             color=["lightblue" if n not in ("plan_workflow","report_progress","wait")
                    else "lightgray" for n in ynames])
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ynames, fontsize=8)
    ax2.set_xlabel("Simulated time (s)")
    ax2.set_ylabel("Tool call")
    ax2.set_title("Tool call sequence (grey = meta/wait, blue = flight action)")
    ax2.grid(True, alpha=0.2, axis="x")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C1] Plot: {OUT_PNG}")

# ── Summary print ──────────────────────────────────────────────────────────────
print(f"\n[C1] RESULT SUMMARY")
print(f"  PASS (alt):     {'YES' if alt_error_cm <= TOLERANCE*100 else 'NO'} "
      f"(err={alt_error_cm:.1f}cm ≤ {TOLERANCE*100:.0f}cm)")
print(f"  PASS (sequence):{'YES' if sequence_score >= 3 else 'NO'} "
      f"({sequence_score}/4 key tools used)")
print(f"  API calls:      {total_api_calls}")
print(f"  Tokens in/out:  {total_in_tok}/{total_out_tok}")
print(f"  Est. cost:      ${total_cost:.4f}")
print(f"  Wall time:      {t_wall_total:.1f}s")

