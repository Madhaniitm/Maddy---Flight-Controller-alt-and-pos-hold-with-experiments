"""
EXP-C2: Command Ambiguity Resolution
=====================================
Send 6 progressively ambiguous commands. Measure how the LLM
interprets each and whether it correctly infers the target altitude.

Commands (in order of increasing ambiguity):
  1. "go to 2 metres"              — explicit
  2. "climb to 2m"                 — paraphrase
  3. "go higher"                   — relative, no number
  4. "go up a bit"                 — vague relative
  5. "ascend slowly to a safe height" — abstract
  6. "I want it higher"            — indirect

Expected:
  Cmd 1–2: exact 2.0 m interpretation
  Cmd 3–4: reasonable increment (0.1–0.8 m above current)
  Cmd 5–6: LLM picks a safe default (~1.0–1.5 m) OR asks for clarification

Measures: correct interpretation rate, clarification requests, actual target used.

Outputs:
  results/C2_ambiguity.csv   — per-command results table
  results/C2_ambiguity.png   — interpretation bar chart
"""

import sys, os, csv, json, re, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "C2_ambiguity.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "C2_ambiguity.png")

# Commands with ground-truth expected target altitude (None = "reasonable" range)
COMMANDS = [
    ("go to 2 metres",                    2.0,  (1.90, 2.10), "explicit"),
    ("climb to 2m",                       2.0,  (1.90, 2.10), "paraphrase"),
    ("go higher",                         None, (0.10, 1.50), "relative_no_num"),
    ("go up a bit",                       None, (0.05, 0.60), "vague_relative"),
    ("ascend slowly to a safe height",    None, (0.05, 2.00), "abstract"),
    ("I want it higher",                  None, (0.05, 2.00), "indirect"),
]

def extract_altitude_target(tool_trace, text_response):
    """
    Infer what altitude the LLM targeted from its tool calls.
    Returns (target_m, source) where source is 'set_altitude_target',
    'takeoff', or 'text_inference'.
    Uses the LAST set_altitude_target call (final target in multi-step climbs).
    """
    # Look for explicit set_altitude_target calls — use last one (final target)
    last_target = None
    for tr in tool_trace:
        if tr["name"] == "set_altitude_target":
            meters = tr["args"].get("meters")
            if meters is not None:
                last_target = float(meters)
    if last_target is not None:
        return last_target, "set_altitude_target"

    # Look for takeoff hover_power — not a target altitude
    # Try to parse a number from the text response
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:m(?:etre?s?)?|meter?s?)\b',
                         text_response, re.IGNORECASE)
    if numbers:
        return float(numbers[0]), "text_inference"

    return None, "unknown"

def asked_for_clarification(text_response):
    """True if the LLM asked a clarifying question."""
    clarify_words = ["clarif", "specific", "how high", "what height",
                     "please specify", "did you mean", "could you clarify"]
    lower = text_response.lower()
    return any(w in lower for w in clarify_words)

# ── Pre-flight: arm and hover at 1.0 m so commands have context ───────────────
print("[C2] Pre-arming drone to 1.0 m as starting state …")
agent_base = SimAgent(session_id="C2_base")
# Use one SimAgent for all 6 commands to share conversation context (drone is already in air)
# We'll run the takeoff first, then each ambiguous command as a follow-up turn.

# Arm and hover using direct sim control (bypass LLM for setup)
with agent_base.state.lock:
    agent_base.state.armed = True
    agent_base.state.ch5   = 1000

hover_pwm = agent_base._find_hover()
hover_thr  = (hover_pwm - 1000) / 1000.0
with agent_base.state.lock:
    s = agent_base.state
    s.hover_thr_locked = hover_thr
    s.althold  = True
    s.alt_sp   = s.z
    s.alt_sp_mm = s.z * 1000
agent_base.physics.pid_alt_pos.reset()
agent_base.physics.pid_alt_vel.reset()
with agent_base.state.lock:
    agent_base.state.alt_sp    = 1.0
    agent_base.state.alt_sp_mm = 1000.0
agent_base.wait_sim(8.0)   # settle at 1.0 m

with agent_base.state.lock:
    z_base = round(agent_base.state.ekf_z, 3)
print(f"[C2] Drone at {z_base:.3f} m, ready for ambiguous commands.")

# Shared conversation history — drone is already flying at ~1.0 m
shared_history = [
    {
        "role": "user",
        "content": (
            "The drone is currently armed and hovering at approximately 1.0 m with "
            "altitude hold active. I will give you a series of flight commands."
        ),
    },
    {
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": (
                "Understood. The drone is airborne at ~1.0 m with altitude hold "
                "active. Ready to receive your commands."
            ),
        }],
    },
]

results = []

for idx, (command, gt_exact, gt_range, cmd_type) in enumerate(COMMANDS, start=1):
    print(f"\n[C2] Command {idx}/6: \"{command}\"  (type: {cmd_type})")

    with agent_base.state.lock:
        z_before = round(agent_base.state.ekf_z, 3)

    # Snapshot history so each command uses the same base context
    # but accumulates previous turns (simulates multi-turn session)
    turn_text, api_stats, tool_trace = agent_base.run_agent_loop(
        command,
        history=list(shared_history),
        max_turns=10,
    )

    # Update shared history with this exchange
    shared_history.append({"role": "user", "content": command})
    shared_history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": turn_text}],
    })

    # Wait for drone to settle after any commands
    agent_base.wait_sim(6.0)
    with agent_base.state.lock:
        z_after = round(agent_base.state.ekf_z, 3)

    increment = z_after - z_before

    target_m, target_src = extract_altitude_target(tool_trace, turn_text)
    clarified = asked_for_clarification(turn_text)

    # Correctness judgement
    if gt_exact is not None:
        correct = (target_m is not None and abs(target_m - gt_exact) <= 0.15)
        expected_str = f"{gt_exact:.1f} m"
    else:
        lo, hi = gt_range
        if clarified:
            correct = True   # asking for clarification is acceptable for vague commands
        elif target_m is not None:
            correct = lo <= increment <= hi
        else:
            correct = lo <= increment <= hi
        expected_str = f"increment {gt_range[0]}–{gt_range[1]} m"

    total_api  = len(api_stats)
    total_toks = (sum(s["input_tokens"] + s["output_tokens"] for s in api_stats))
    tools_used = [t["name"] for t in tool_trace]

    print(f"  Target extracted: {target_m} ({target_src})")
    print(f"  Altitude: {z_before:.3f} → {z_after:.3f} m (Δ={increment:+.3f} m)")
    print(f"  Clarification asked: {clarified}")
    print(f"  Correct: {correct}  (expected: {expected_str})")
    print(f"  Tools: {tools_used[:6]}")

    results.append({
        "cmd_idx":       idx,
        "command":       command,
        "cmd_type":      cmd_type,
        "z_before_m":    z_before,
        "z_after_m":     z_after,
        "increment_m":   round(increment, 3),
        "target_m":      target_m,
        "target_source": target_src,
        "asked_clarify": clarified,
        "correct":       correct,
        "api_calls":     total_api,
        "tokens":        total_toks,
        "tools":         ";".join(tools_used[:6]),
    })

# ── Metrics summary ────────────────────────────────────────────────────────────
n_correct   = sum(1 for r in results if r["correct"])
n_clarified = sum(1 for r in results if r["asked_clarify"])
print(f"\n[C2] ── METRICS ──────────────────────────────────────────────")
print(f"  Correct interpretations: {n_correct}/{len(results)} "
      f"({n_correct/len(results)*100:.0f}%)")
print(f"  Clarifications requested: {n_clarified} "
      f"(on vague commands 4–6)")
for r in results:
    mark = "✓" if r["correct"] else "✗"
    print(f"  {mark} [{r['cmd_type']:18s}] \"{r['command'][:40]}\" "
          f"→ Δ{r['increment_m']:+.2f}m  target={r['target_m']}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\n[C2] CSV: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels = [f'C{r["cmd_idx"]}: {r["cmd_type"]}\n"{r["command"][:25]}…"'
          if len(r["command"]) > 25 else f'C{r["cmd_idx"]}: {r["cmd_type"]}\n"{r["command"]}"'
          for r in results]
increments = [r["increment_m"] for r in results]
colors     = ["green" if r["correct"] else "red" for r in results]

bars = ax1.barh(labels, increments, color=colors, alpha=0.75, edgecolor="black", lw=0.5)
ax1.axvline(0, color="black", lw=0.8)
for bar, r in zip(bars, results):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{r["increment_m"]:+.2f}m', va="center", fontsize=8)
    if r["asked_clarify"]:
        ax1.text(bar.get_width() + 0.10, bar.get_y() + bar.get_height() / 2,
                 "?", va="center", fontsize=10, color="blue")
ax1.set_xlabel("Altitude increment (m)")
ax1.set_title(f"C2: Altitude increment per command\n"
              f"Green=correct, Red=incorrect, ?=asked clarification\n"
              f"Correct: {n_correct}/{len(results)}")
ax1.grid(True, alpha=0.3, axis="x")

# Target extracted vs expected
targets = [r["target_m"] if r["target_m"] is not None else float("nan") for r in results]
expected_exact = [r["command"].endswith("2 metres") or r["command"].endswith("2m") for r in results]
ax2.scatter(range(1, 7), targets, s=80, zorder=5,
            c=["green" if r["correct"] else "red" for r in results])
ax2.axhline(2.0, color="blue", ls="--", lw=1, label="Explicit target 2.0 m")
ax2.set_xticks(range(1, 7))
ax2.set_xticklabels([f"C{r['cmd_idx']}\n({r['cmd_type'][:8]})" for r in results], fontsize=8)
ax2.set_ylabel("Extracted target altitude (m)")
ax2.set_title(f"Extracted altitude target per command\n"
              f"({n_clarified} clarification requests shown separately)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C2] Plot: {OUT_PNG}")

print(f"\n[C2] RESULT: {n_correct}/{len(results)} correct interpretations "
      f"({n_correct/len(results)*100:.0f}%)")
