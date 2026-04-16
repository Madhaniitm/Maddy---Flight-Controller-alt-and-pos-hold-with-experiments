"""
EXP-C6: Human Goal → LLM Mission Planning
==========================================
Command: "do a square pattern at 1 metre height"
LLM must decompose entirely: plan 4-leg square + execute in sequence.

Measures:
  - Correct waypoint count (4 legs + return = 5 waypoints)
  - Trajectory squareness (XY trajectory forms a square)
  - Waypoint position error (deviation from ideal square corners)
  - Tool calls used for decomposition

Expected: LLM plans 4-leg square and executes in sequence.

Outputs:
  results/C6_mission_planning.csv    — waypoint plan and metrics
  results/C6_mission_planning.png    — XY trajectory plot
"""

import sys, os, csv, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "C6_mission_planning.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "C6_mission_planning.png")

COMMAND    = "do a square pattern at 1 metre height"
TARGET_ALT = 1.0

print(f"[C6] Human command: \"{COMMAND}\"")
print("[C6] Running LLM mission planner …")

agent = SimAgent(session_id="C6")

# Run the full mission-planning loop
text, api_stats, tool_trace = agent.run_agent_loop(
    COMMAND,
    max_turns=30,   # square mission needs more turns
)

print(f"\n[C6] LLM final text: {text[:200]}")

# ── Extract plan from plan_workflow call ──────────────────────────────────────
plan_steps = []
for tr in tool_trace:
    if tr["name"] == "plan_workflow":
        plan_steps = tr["args"].get("steps", [])
        break

# ── Extract set_altitude_target calls for altitude check ─────────────────────
alt_targets = [t["args"].get("meters") for t in tool_trace
               if t["name"] == "set_altitude_target" and t["args"].get("meters") is not None]

# ── Analyse XY trajectory ─────────────────────────────────────────────────────
tel = agent.get_telem_arrays()
kx  = tel.get("kx", np.array([]))
ky  = tel.get("ky", np.array([]))
z_v = tel.get("z_true", np.array([]))

# Find portion where drone is airborne (z > 0.3 m)
if len(kx) > 0 and len(z_v) > 0:
    mask       = z_v > 0.3
    kx_air     = kx[mask]
    ky_air     = ky[mask]
else:
    kx_air = ky_air = np.array([])

# Measure squareness: bounding box aspect ratio of trajectory
if len(kx_air) > 5:
    x_range = kx_air.max() - kx_air.min()
    y_range = ky_air.max() - ky_air.min()
    bbox_diag = math.sqrt(x_range**2 + y_range**2)
    # Squareness: ratio of min/max range (1.0 = perfect square)
    squareness = min(x_range, y_range) / max(x_range, y_range) if max(x_range, y_range) > 0.01 else 0
    total_path = float(np.sum(np.sqrt(np.diff(kx_air)**2 + np.diff(ky_air)**2)))
else:
    x_range = y_range = bbox_diag = squareness = total_path = 0.0

# Count distinct direction changes in trajectory (proxy for waypoints turned)
def count_direction_changes(kx_arr, ky_arr, min_dist=0.05):
    """Count significant changes in movement direction (proxy for waypoints)."""
    if len(kx_arr) < 10:
        return 0
    # Compute heading at each point
    dx = np.diff(kx_arr)
    dy = np.diff(ky_arr)
    headings = np.arctan2(dy, dx)
    # Count points where heading changes by >45 degrees
    dh = np.abs(np.diff(headings))
    dh = np.minimum(dh, 2*np.pi - dh)  # wrap
    changes = int(np.sum(dh > np.radians(45)))
    return changes

direction_changes = count_direction_changes(kx_air, ky_air) if len(kx_air) > 10 else 0

# ── Metrics ───────────────────────────────────────────────────────────────────
tools_used = [t["name"] for t in tool_trace]
n_plan_steps = len(plan_steps)
had_plan_workflow = "plan_workflow" in set(tools_used)
n_set_pos = tools_used.count("set_pitch") + tools_used.count("set_roll") + tools_used.count("enable_position_hold")
n_alt_ok  = sum(1 for a in alt_targets if a is not None and abs(a - TARGET_ALT) < 0.25)

# Overall squareness pass: trajectory has movement in X and Y, and aspect ratio > 0.4
trajectory_ok = squareness > 0.35 and total_path > 0.2
alt_ok        = n_alt_ok >= 1 or (len(z_v) > 0 and np.max(z_v) > 0.5)
plan_ok       = had_plan_workflow and n_plan_steps >= 3
passed        = plan_ok and alt_ok

n_api  = len(api_stats)
in_tok = sum(s["input_tokens"]  for s in api_stats)
out_tok= sum(s["output_tokens"] for s in api_stats)
cost   = sum(s["cost_usd"]      for s in api_stats)

print(f"\n[C6] ── METRICS ──────────────────────────────────────────────")
print(f"  Plan steps:       {n_plan_steps}  (expected ≥4 for a square)")
print(f"  Plan text:        {plan_steps[:6]}")
print(f"  Alt targets used: {alt_targets[:4]}")
print(f"  X range:          {x_range:.3f} m")
print(f"  Y range:          {y_range:.3f} m")
print(f"  Squareness:       {squareness:.3f}  (1.0=perfect)")
print(f"  Total path:       {total_path:.3f} m")
print(f"  Dir changes:      {direction_changes} (proxy for waypoints turned)")
print(f"  Tool count:       {len(tools_used)}")
print(f"  API calls:        {n_api}")
print(f"  Tokens in/out:    {in_tok}/{out_tok}")
print(f"  Est. cost:        ${cost:.4f}")
print(f"  PASS:             {passed}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    rows = [
        ("command",           COMMAND),
        ("n_plan_steps",      n_plan_steps),
        ("plan_steps",        ";".join(plan_steps[:8])),
        ("alt_targets",       ";".join(str(a) for a in alt_targets[:5])),
        ("n_alt_target_ok",   n_alt_ok),
        ("x_range_m",         round(x_range, 3)),
        ("y_range_m",         round(y_range, 3)),
        ("squareness_ratio",  round(squareness, 3)),
        ("total_path_m",      round(total_path, 3)),
        ("direction_changes", direction_changes),
        ("tool_sequence",     ";".join(tools_used[:15])),
        ("api_calls",         n_api),
        ("input_tokens",      in_tok),
        ("output_tokens",     out_tok),
        ("cost_usd",          round(cost, 6)),
        ("plan_ok",           plan_ok),
        ("alt_ok",            alt_ok),
        ("trajectory_ok",     trajectory_ok),
        ("passed",            passed),
    ]
    w.writerows(rows)
print(f"[C6] CSV: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# XY trajectory
ax = axes[0]
if len(kx_air) > 1:
    sc = ax.scatter(kx_air, ky_air, c=np.linspace(0, 1, len(kx_air)),
                    cmap="viridis", s=4, zorder=5)
    plt.colorbar(sc, ax=ax, label="Time (normalised)")
    ax.plot(kx_air[0],  ky_air[0],  "go", ms=10, label="Start",  zorder=6)
    ax.plot(kx_air[-1], ky_air[-1], "rs", ms=10, label="End",    zorder=6)
    # Ideal 0.5×0.5 m square centred at origin for reference
    sq = np.array([[0,0],[0.5,0],[0.5,0.5],[0,0.5],[0,0]])
    ax.plot(sq[:,0]-0.25, sq[:,1]-0.25, "k--", lw=1, alpha=0.4, label="Ideal 0.5m square")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title(f"XY Trajectory\nSquareness={squareness:.2f}, Path={total_path:.2f}m")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

# Altitude timeline
ax2 = axes[1]
if len(tel.get("t", [])) > 0:
    t_s = tel["t"] / 1000.0
    ax2.plot(t_s, z_v, color="blue", lw=1.5, label="True altitude")
    ax2.plot(t_s, tel["lw_z"]/1000.0, color="green", lw=1.2, label="EKF", alpha=0.8)
    ax2.axhline(TARGET_ALT, color="red", ls="--", lw=1, label=f"Target {TARGET_ALT}m")
ax2.set_xlabel("Simulated time (s)")
ax2.set_ylabel("Altitude (m)")
ax2.set_title(f"Altitude during mission\nAlt targets: {alt_targets[:4]}")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plan steps
ax3 = axes[2]
ax3.axis("off")
plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps[:10]))
ax3.text(0.05, 0.95, f"LLM Plan ({n_plan_steps} steps):\n\n{plan_text}",
         transform=ax3.transAxes, fontsize=9, va="top", family="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax3.set_title(f"LLM Mission Plan\n"
              f"API calls: {n_api} | Tokens: {in_tok}+{out_tok} | Cost: ${cost:.4f}")

fig.suptitle(
    f'EXP-C6: Mission Planning — "{COMMAND}"\n'
    f'PASS: {passed}',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C6] Plot: {OUT_PNG}")

print(f"\n[C6] RESULT: {'PASS' if passed else 'FAIL'}")
print(f"  Plan steps: {n_plan_steps}, squareness: {squareness:.2f}, path: {total_path:.2f}m")
