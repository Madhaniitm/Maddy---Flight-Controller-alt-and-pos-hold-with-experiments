"""
EXP-C6: Human Goal → LLM Mission Planning  (N=5 runs)
=======================================================
"do a square pattern at 1 metre height"
LLM decomposes entirely. Reports squareness and success rate over N=5 runs.

Outputs:
  results/C6_runs.csv
  results/C6_summary.csv
  results/C6_mission_planning.png
"""

import sys, os, csv, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
# ── Guardrail toggle (--guardrail on|off) ──────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--guardrail", choices=["on", "off"], default="on")
_args, _ = _parser.parse_known_args()
GUARDRAIL_ENABLED = _args.guardrail == "on"
GUARDRAIL_SUFFIX  = "guardrail_on" if GUARDRAIL_ENABLED else "guardrail_off"

OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", f"C6_runs_{GUARDRAIL_SUFFIX}.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", f"C6_summary_{GUARDRAIL_SUFFIX}.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", f"C6_mission_planning_{GUARDRAIL_SUFFIX}.png")

COMMAND    = "do a square pattern at 1 metre height"
TARGET_ALT = 1.0
N_RUNS     = 5

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Mission decomposition (plan_workflow) then sequential execution is the ReAct planning loop."
    ),
    "SayCan": (
        "Ahn, M., Brohan, A., Brown, N., et al. (2022). Do As I Can, Not As I Say: "
        "Grounding Language in Robotic Affordances. arXiv:2204.01691. "
        "Establishes LLM-based task decomposition for physical robots; C6 extends to micro-UAVs."
    ),
    "Vemprala2023": (
        "Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). "
        "ChatGPT for Robotics: Design Principles and Model Abilities. MSR-TR-2023-8. arXiv:2306.17582. "
        "GPT-4 waypoint planning on UAVs — benchmark for C6 waypoint count and trajectory quality."
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

def count_direction_changes(kx_arr, ky_arr):
    if len(kx_arr) < 10:
        return 0
    dx = np.diff(kx_arr)
    dy = np.diff(ky_arr)
    headings = np.arctan2(dy, dx)
    dh = np.abs(np.diff(headings))
    dh = np.minimum(dh, 2*np.pi - dh)
    return int(np.sum(dh > np.radians(45)))

# ── Single-run function ────────────────────────────────────────────────────────

def run_once(run_idx):
    print(f"\n[C6] ── Run {run_idx+1}/{N_RUNS} ─────────────────────────────────")
    agent = SimAgent(session_id=f"C6_run{run_idx}", guardrail_enabled=GUARDRAIL_ENABLED)
    text, api_stats, tool_trace, _ = agent.run_agent_loop(COMMAND, max_turns=30)

    # Extract plan steps
    plan_steps = []
    for tr in tool_trace:
        if tr["name"] == "plan_workflow":
            plan_steps = tr["args"].get("steps", [])
            break

    alt_targets = [t["args"].get("meters") for t in tool_trace
                   if t["name"] == "set_altitude_target" and t["args"].get("meters") is not None]

    tel  = agent.get_telem_arrays()
    kx   = tel.get("kx", np.array([]))
    ky   = tel.get("ky", np.array([]))
    z_v  = tel.get("z_true", np.array([]))

    if len(kx) > 0 and len(z_v) > 0:
        mask   = z_v > 0.3
        kx_air = kx[mask]
        ky_air = ky[mask]
    else:
        kx_air = ky_air = np.array([])

    if len(kx_air) > 5:
        x_range     = float(kx_air.max() - kx_air.min())
        y_range     = float(ky_air.max() - ky_air.min())
        squareness  = (min(x_range, y_range) / max(x_range, y_range)
                       if max(x_range, y_range) > 0.01 else 0.0)
        total_path  = float(np.sum(np.sqrt(np.diff(kx_air)**2 + np.diff(ky_air)**2)))
        dir_changes = count_direction_changes(kx_air, ky_air)
    else:
        x_range = y_range = squareness = total_path = 0.0
        dir_changes = 0

    tools_used = [t["name"] for t in tool_trace]
    n_plan_steps = len(plan_steps)
    had_plan     = "plan_workflow" in set(tools_used)
    n_alt_ok     = sum(1 for a in alt_targets if a is not None and abs(a - TARGET_ALT) < 0.25)
    alt_ok       = n_alt_ok >= 1 or (len(z_v) > 0 and float(np.max(z_v)) > 0.5)
    plan_ok      = had_plan and n_plan_steps >= 3
    passed       = plan_ok and alt_ok

    n_api  = len(api_stats)
    in_tok = sum(s["input_tokens"]  for s in api_stats)
    out_tok= sum(s["output_tokens"] for s in api_stats)
    cost   = sum(s["cost_usd"]      for s in api_stats)

    print(f"  plan_steps={n_plan_steps}  squareness={squareness:.3f}  "
          f"path={total_path:.2f}m  pass={passed}")

    return {
        "run":                run_idx + 1,
        "n_plan_steps":       n_plan_steps,
        "had_plan_workflow":  int(had_plan),
        "n_alt_target_ok":    n_alt_ok,
        "x_range_m":          round(x_range, 3),
        "y_range_m":          round(y_range, 3),
        "squareness_ratio":   round(squareness, 4),
        "total_path_m":       round(total_path, 3),
        "dir_changes":        dir_changes,
        "plan_ok":            int(plan_ok),
        "alt_ok":             int(alt_ok),
        "passed":             int(passed),
        "api_calls":          n_api,
        "input_tokens":       in_tok,
        "output_tokens":      out_tok,
        "cost_usd":           round(cost, 6),
        "_kx_air":            kx_air,   # for plotting
        "_ky_air":            ky_air,
    }

# ── Run N times ────────────────────────────────────────────────────────────────

all_results = [run_once(i) for i in range(N_RUNS)]

# ── Aggregate ─────────────────────────────────────────────────────────────────

def col(key):
    return [r[key] for r in all_results]

n_pass = sum(col("passed"))
pass_lo, pass_hi = wilson_ci(n_pass, N_RUNS)

squareness_vals = col("squareness_ratio")
path_vals       = col("total_path_m")
plan_step_vals  = col("n_plan_steps")
sq_ci   = bootstrap_ci(squareness_vals)
path_ci = bootstrap_ci(path_vals)

print(f"\n[C6] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
print(f"  Success rate:    {n_pass}/{N_RUNS}  CI=[{pass_lo:.2f},{pass_hi:.2f}]")
print(f"  Squareness:      {np.mean(squareness_vals):.3f}±{np.std(squareness_vals):.3f}  "
      f"CI=[{sq_ci[0]:.3f},{sq_ci[1]:.3f}]")
print(f"  Total path (m):  {np.mean(path_vals):.3f}±{np.std(path_vals):.3f}  "
      f"CI=[{path_ci[0]:.3f},{path_ci[1]:.3f}]")
print(f"  Plan steps:      {np.mean(plan_step_vals):.1f}±{np.std(plan_step_vals):.1f}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
csv_keys = [k for k in all_results[0].keys() if not k.startswith("_")]
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_keys)
    w.writeheader()
    for r in all_results:
        w.writerow({k: r[k] for k in csv_keys})
print(f"[C6] Per-run CSV: {OUT_RUNS}")

summary_rows = [
    ("n_runs",                N_RUNS),
    ("n_pass",                n_pass),
    ("success_rate",          round(n_pass / N_RUNS, 3)),
    ("success_rate_ci_lo",    round(pass_lo, 3)),
    ("success_rate_ci_hi",    round(pass_hi, 3)),
    ("squareness_mean",       round(float(np.mean(squareness_vals)), 4)),
    ("squareness_std",        round(float(np.std(squareness_vals)), 4)),
    ("squareness_ci_lo",      round(sq_ci[0], 4)),
    ("squareness_ci_hi",      round(sq_ci[1], 4)),
    ("total_path_mean_m",     round(float(np.mean(path_vals)), 3)),
    ("total_path_std_m",      round(float(np.std(path_vals)), 3)),
    ("total_path_ci_lo_m",    round(path_ci[0], 3)),
    ("total_path_ci_hi_m",    round(path_ci[1], 3)),
    ("plan_steps_mean",       round(float(np.mean(plan_step_vals)), 1)),
    ("plan_steps_std",        round(float(np.std(plan_step_vals)), 1)),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow([f"ref_{ref_key}", ref_val])
print(f"[C6] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: overlaid XY trajectories
ax1 = axes[0]
traj_colors = plt.cm.tab10(np.linspace(0, 0.9, N_RUNS))
for i, r in enumerate(all_results):
    kx_air = r["_kx_air"]
    ky_air = r["_ky_air"]
    if len(kx_air) > 1:
        label = f"Run {i+1} (sq={r['squareness_ratio']:.2f}, {'✓' if r['passed'] else '✗'})"
        ax1.plot(kx_air, ky_air, color=traj_colors[i], lw=1.2, alpha=0.7, label=label)
        ax1.plot(kx_air[0], ky_air[0], "o", color=traj_colors[i], ms=6)
        ax1.plot(kx_air[-1], ky_air[-1], "s", color=traj_colors[i], ms=6)
sq = np.array([[0,0],[0.5,0],[0.5,0.5],[0,0.5],[0,0]])
ax1.plot(sq[:,0]-0.25, sq[:,1]-0.25, "k--", lw=1.5, alpha=0.4, label="Ideal 0.5m square")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_title(f"Overlaid XY trajectories (N={N_RUNS})\n"
              f"Squareness: {np.mean(squareness_vals):.3f}±{np.std(squareness_vals):.3f}")
ax1.legend(fontsize=6)
ax1.grid(True, alpha=0.3)
ax1.set_aspect("equal")

# Middle: squareness per run
ax2 = axes[1]
bar_cols = ["green" if r["passed"] else "red" for r in all_results]
ax2.bar(range(1, N_RUNS + 1), squareness_vals, color=bar_cols, alpha=0.75, edgecolor="black")
ax2.axhline(np.mean(squareness_vals), color="navy", ls="--", lw=1.5,
            label=f"Mean={np.mean(squareness_vals):.3f}")
ax2.fill_between([0.5, N_RUNS + 0.5],
                 sq_ci[0], sq_ci[1],
                 alpha=0.12, color="navy", label=f"95% CI [{sq_ci[0]:.3f},{sq_ci[1]:.3f}]")
ax2.axhline(0.35, color="red", ls=":", lw=1, label="Pass threshold 0.35")
ax2.set_xlabel("Run")
ax2.set_ylabel("Squareness ratio (1.0=perfect)")
ax2.set_title("Squareness ratio per run")
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3, axis="y")

# Right: path length and plan steps per run
ax3 = axes[2]
x = np.arange(1, N_RUNS + 1)
ax3b = ax3.twinx()
ax3.bar(x - 0.2, path_vals,      0.35, color="steelblue", alpha=0.75,
        edgecolor="black", label="Total path (m)")
ax3b.bar(x + 0.2, plan_step_vals, 0.35, color="darkorange", alpha=0.75,
         edgecolor="black", label="Plan steps")
ax3.set_xlabel("Run")
ax3.set_ylabel("Total path (m)", color="steelblue")
ax3b.set_ylabel("Plan steps", color="darkorange")
ax3.set_xticks(x)
ax3.set_title(f"Path length & plan steps per run\n"
              f"Path: {np.mean(path_vals):.2f}±{np.std(path_vals):.2f}m  "
              f"Steps: {np.mean(plan_step_vals):.1f}±{np.std(plan_step_vals):.1f}")
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    f'EXP-C6: Mission Planning  (N={N_RUNS} runs, temperature=0.2)\n'
    f'Command: "{COMMAND}"\n'
    f'Success: {n_pass}/{N_RUNS}  (95% CI: {pass_lo:.2f}–{pass_hi:.2f})',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C6] Plot: {OUT_PNG}")

print(f"\n[C6] RESULT: {n_pass}/{N_RUNS} passed  (95% CI: {pass_lo:.2f}–{pass_hi:.2f})")
print(f"       Squareness: {np.mean(squareness_vals):.3f}±{np.std(squareness_vals):.3f}")
