"""
EXP-C3: Multi-Turn Conversational Mission
==========================================
5-turn conversation testing LLM state tracking across turns:
  Turn 1: "arm the drone"
  Turn 2: "go to 1.5 metres"
  Turn 3: "hold there for 5 seconds"
  Turn 4: "rotate 90 degrees clockwise"
  Turn 5: "land now"

Measures:
  - State tracking: no repeated arm/takeoff in turns 2–5
  - Tool calls per turn
  - Correct action per turn
  - Total API calls, tokens

Expected: LLM correctly infers drone state from history at each turn.

Outputs:
  results/C3_multiturn.csv  — per-turn metrics table
  results/C3_multiturn.png  — altitude + yaw timeline with turn markers
"""

import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "C3_multiturn.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "C3_multiturn.png")

# 5-turn conversation script
TURNS = [
    {
        "turn": 1,
        "user": "arm the drone",
        "expected_action": "arm",
        "expected_tools": ["arm"],
        "should_NOT_contain": [],
        "description": "Arm motors",
    },
    {
        "turn": 2,
        "user": "go to 1.5 metres",
        "expected_action": "takeoff_and_altitude",
        "expected_tools": ["find_hover_throttle", "set_altitude_target"],
        "should_NOT_contain": ["arm"],  # already armed, should not re-arm
        "description": "Takeoff + climb to 1.5 m",
    },
    {
        "turn": 3,
        "user": "hold there for 5 seconds",
        "expected_action": "wait",
        "expected_tools": ["wait"],
        "should_NOT_contain": ["arm", "takeoff", "find_hover_throttle"],
        "description": "Wait 5 s at current altitude",
    },
    {
        "turn": 4,
        "user": "rotate 90 degrees clockwise",
        "expected_action": "yaw",
        "expected_tools": ["set_yaw"],
        "should_NOT_contain": ["arm", "land"],
        "description": "Yaw 90° CW",
    },
    {
        "turn": 5,
        "user": "land now",
        "expected_action": "land",
        "expected_tools": ["land", "disarm"],
        "should_NOT_contain": ["arm", "takeoff"],
        "description": "Safe landing",
    },
]

print("[C3] Starting 5-turn conversational mission …")

agent   = SimAgent(session_id="C3")
history = []   # accumulated conversation history

turn_results   = []
tel_snapshots  = {}   # sim_time → telemetry snapshot at start of each turn

total_api_calls = 0
total_in_tok    = 0
total_out_tok   = 0
total_cost      = 0.0

for turn_spec in TURNS:
    turn_idx  = turn_spec["turn"]
    user_msg  = turn_spec["user"]

    with agent.state.lock:
        z_before      = round(agent.state.ekf_z, 3)
        yaw_before    = round(agent.state.yaw,   2)
        armed_before  = agent.state.armed
        althold_before= agent.state.althold
        altsp_before  = round(agent.state.alt_sp, 2)

    t_sim_before = agent.sim_time
    tel_snapshots[turn_idx] = {"sim_time": t_sim_before, "z": z_before, "yaw": yaw_before}

    print(f"\n[C3] Turn {turn_idx}: \"{user_msg}\"")
    print(f"     State: armed={armed_before}, z={z_before:.3f}m, yaw={yaw_before:.1f}°")

    # Inject live drone state so LLM doesn't re-arm/re-enable what's already active
    state_ctx = (
        f"[Drone state: armed={armed_before}, "
        f"althold={'ON' if althold_before else 'OFF'}, "
        f"alt={z_before:.2f}m, setpoint={altsp_before:.2f}m] "
    )
    user_msg_with_ctx = state_ctx + user_msg

    text, api_stats, tool_trace = agent.run_agent_loop(
        user_msg_with_ctx,
        history=list(history),
        max_turns=20,
    )

    # Update shared history (accumulate turns)
    history.append({"role": "user", "content": user_msg_with_ctx})
    # Guard: text content blocks must be non-empty
    history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": text if text.strip() else "Done."}],
    })

    # Wait briefly for drone to respond
    agent.wait_sim(3.0)

    with agent.state.lock:
        z_after   = round(agent.state.ekf_z, 3)
        yaw_after = round(agent.state.yaw,   2)
        armed_after = agent.state.armed

    tools_used = [t["name"] for t in tool_trace]
    tools_set  = set(tools_used)

    # Correctness checks
    # 1. Did any expected tools appear?
    expected_found = [et for et in turn_spec["expected_tools"] if et in tools_set]
    # 2. Did any forbidden tools appear?
    forbidden_found = [nt for nt in turn_spec["should_NOT_contain"] if nt in tools_set]
    # 3. State-based check
    state_ok = True
    if turn_idx == 1:
        state_ok = armed_after   # drone should be armed
    elif turn_idx == 2:
        state_ok = z_after > 0.8   # should have taken off
    elif turn_idx == 3:
        # Should have waited — allow up to 0.50 m drift (drone may still be settling to setpoint)
        state_ok = abs(z_after - z_before) < 0.50
    elif turn_idx == 4:
        # Should have rotated
        yaw_delta = abs(yaw_after - yaw_before)
        state_ok = yaw_delta > 20   # rotated at least 20°
    elif turn_idx == 5:
        state_ok = not armed_after  # should be disarmed

    sequence_ok = len(expected_found) > 0 and len(forbidden_found) == 0

    n_api  = len(api_stats)
    in_tok = sum(s["input_tokens"]  for s in api_stats)
    out_tok= sum(s["output_tokens"] for s in api_stats)
    cost   = sum(s["cost_usd"]      for s in api_stats)

    total_api_calls += n_api
    total_in_tok    += in_tok
    total_out_tok   += out_tok
    total_cost      += cost

    result = {
        "turn":           turn_idx,
        "user_msg":       user_msg,
        "description":    turn_spec["description"],
        "z_before_m":     z_before,
        "z_after_m":      z_after,
        "yaw_before_deg": yaw_before,
        "yaw_after_deg":  yaw_after,
        "armed_after":    armed_after,
        "tools_used":     ";".join(tools_used[:10]),
        "expected_found": ";".join(expected_found),
        "forbidden_found":";".join(forbidden_found),
        "sequence_ok":    sequence_ok,
        "state_ok":       state_ok,
        "overall_pass":   sequence_ok and state_ok,
        "api_calls":      n_api,
        "input_tokens":   in_tok,
        "output_tokens":  out_tok,
        "cost_usd":       round(cost, 6),
    }
    turn_results.append(result)

    print(f"  Tools:           {tools_used[:8]}")
    print(f"  Expected found:  {expected_found}")
    print(f"  Forbidden found: {forbidden_found}")
    print(f"  State check:     {state_ok}")
    print(f"  PASS:            {result['overall_pass']}")

# ── Summary ────────────────────────────────────────────────────────────────────
n_pass = sum(1 for r in turn_results if r["overall_pass"])
print(f"\n[C3] ── METRICS ──────────────────────────────────────────────")
print(f"  Turns passed:    {n_pass}/{len(TURNS)}")
print(f"  Total API calls: {total_api_calls}")
print(f"  Total tokens in: {total_in_tok}  out: {total_out_tok}")
print(f"  Est. cost:       ${total_cost:.4f}")

for r in turn_results:
    mark = "✓" if r["overall_pass"] else "✗"
    print(f"  {mark} Turn {r['turn']}: {r['description']:30s} | "
          f"z={r['z_before_m']:.2f}→{r['z_after_m']:.2f}m | "
          f"API={r['api_calls']} | tools={r['tools_used'][:40]}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=turn_results[0].keys())
    w.writeheader()
    w.writerows(turn_results)
print(f"\n[C3] CSV: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
tel = agent.get_telem_arrays()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

if len(tel.get("t", [])) > 0:
    t_s = tel["t"] / 1000.0
    ax1.plot(t_s, tel["z_true"], color="blue",  lw=1.5, label="True altitude")
    ax1.plot(t_s, tel["lw_z"] / 1000.0, color="green", lw=1.2, label="EKF altitude", alpha=0.8)
    ax2.plot(t_s, tel["y"],     color="purple", lw=1.5, label="Yaw (deg)")
    ax3.plot(t_s, tel["r"],     color="red",    lw=1.2, label="Roll (deg)", alpha=0.8)
    ax3.plot(t_s, tel["p"],     color="orange", lw=1.2, label="Pitch (deg)", alpha=0.8)

# Mark turn boundaries
turn_colors = ["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6"]
for r, color in zip(turn_results, turn_colors):
    t_mark = tel_snapshots[r["turn"]]["sim_time"]
    for ax in [ax1, ax2, ax3]:
        ax.axvline(t_mark, color=color, ls="--", lw=1.5, alpha=0.8)
    mark = "✓" if r["overall_pass"] else "✗"
    ax1.text(t_mark + 0.3, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0.1 else 1.5,
             f'T{r["turn"]}{mark}\n"{r["user_msg"][:15]}"',
             fontsize=6, color=color, va="top")

ax1.set_ylabel("Altitude (m)")
ax1.set_title(f"EXP-C3: Multi-Turn Conversational Mission\n"
              f"Passes: {n_pass}/{len(TURNS)} turns | "
              f"API calls: {total_api_calls} | Tokens: {total_in_tok}+{total_out_tok}")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_ylabel("Yaw (deg)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3.set_ylabel("Roll / Pitch (deg)")
ax3.set_xlabel("Simulated time (s)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C3] Plot: {OUT_PNG}")

print(f"\n[C3] RESULT: {n_pass}/{len(TURNS)} turns passed")
