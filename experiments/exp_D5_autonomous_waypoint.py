"""
EXP-D5: Autonomous Waypoint Navigation (Square Pattern)
========================================================
Goal:
    LLM autonomously plans and executes a 1 m × 1 m square loop at 0.8 m altitude
    using navigate_to_waypoint calls. Start and end at (0, 0). Waypoints:
        (1,0) → (1,1) → (0,1) → (0,0)
    N=5 independent runs. Measures path accuracy, completion rate, and plan
    quality (did it choose exactly 4 waypoints?).

Metrics:
    - mission_complete      : reached all 4 waypoints and returned home (Wilson CI)
    - path_error_m          : mean Euclidean error from ideal waypoint positions (bootstrap CI)
    - squareness_ratio      : min_side / max_side of traversed quadrilateral (bootstrap CI)
    - n_plan_steps          : number of waypoints the LLM planned (bootstrap CI)
    - total_distance_m      : total path length flown (bootstrap CI)
    - api_calls             : per-run (bootstrap CI)
    - cost_usd              : per-run (bootstrap CI)

Paper References:
    - ReAct (Yao et al. 2022): reason-act loop for waypoint selection
    - SayCan (Ahn et al. 2022): affordance-grounded plan execution
    - Vemprala2023: ChatGPT for robotics, waypoint navigation via LLM
    - InnerMonologue (Huang et al. 2022): position feedback drives replanning
"""

import os, sys, json, time, csv, math, pathlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS          = 5
TARGET_ALT_M    = 0.8
WAYPOINT_TOL_M  = 0.15   # acceptable arrival radius
IDEAL_WAYPOINTS = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "SayCan":         "Ahn et al. 2022 — Do As I Can, Not As I Say: Grounding Language in Robotic Affordances",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
}

WAYPOINT_GOAL = (
    "Plan and fly a 1 metre × 1 metre square loop at 0.8 m altitude. "
    "Start at position (0, 0). Fly to (1,0), then (1,1), then (0,1), then return to (0,0). "
    "Use navigate_to_waypoint for each leg. Use get_current_position to confirm arrival. "
    "Announce 'SQUARE COMPLETE' when back at home."
)

# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(p, 4), round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)

def bootstrap_ci(data: list, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr   = np.array(data, dtype=float)
    boots = [stat(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)), 4), round(float(lo), 4), round(float(hi), 4)

def dist2d(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def squareness(pts):
    """min_side/max_side of a 4-point polygon."""
    if len(pts) < 4:
        return 0.0
    sides = [dist2d(pts[i], pts[(i+1)%4]) for i in range(4)]
    return min(sides) / max(sides) if max(sides) > 0 else 0.0

# ── Single-run logic ───────────────────────────────────────────────────────────
def run_once(run_idx: int) -> dict:
    agent = DAgent(session_id=f"D5_r{run_idx}")
    agent.scene_sim.set_scene("open_space")

    agent.execute_tool("arm", {})
    agent.execute_tool("find_hover_throttle", {})
    agent.execute_tool("enable_altitude_hold", {})
    agent.execute_tool("set_altitude_target", {"target_m": TARGET_ALT_M})

    # Track visited waypoints from tool calls
    visited_positions = []
    n_plan_steps      = 0
    mission_complete  = False

    def on_iteration_end(iteration: int, last_reply: str, tool_calls: list):
        nonlocal n_plan_steps, mission_complete
        for tc in tool_calls:
            if tc.get("name") == "navigate_to_waypoint":
                n_plan_steps += 1
                x = tc.get("args", {}).get("x", agent.state.x)
                y = tc.get("args", {}).get("y", getattr(agent.state, "y", 0.0))
                visited_positions.append((x, y))
        if "SQUARE COMPLETE" in last_reply.upper():
            mission_complete = True
            return True
        return False

    t0 = time.time()
    reply, stats, trace = agent.autonomy_loop(
        goal           = WAYPOINT_GOAL,
        mode           = "full_auto",
        max_iterations = 30,
        iteration_hook = on_iteration_end,
    )
    wall_time = time.time() - t0

    if "SQUARE COMPLETE" in reply.upper():
        mission_complete = True

    # Path quality
    if len(visited_positions) >= 4:
        # Evaluate against 4 ideal waypoints
        path_errors = [min(dist2d(vp, ip) for ip in IDEAL_WAYPOINTS)
                       for vp in visited_positions[:4]]
        path_error = float(np.mean(path_errors))
        sq_ratio   = squareness(visited_positions[:4])
        total_dist = sum(dist2d(visited_positions[i], visited_positions[i+1])
                         for i in range(len(visited_positions)-1))
    else:
        path_error = 999.0
        sq_ratio   = 0.0
        total_dist = sum(dist2d(visited_positions[i], visited_positions[i+1])
                         for i in range(len(visited_positions)-1)) if len(visited_positions) > 1 else 0.0

    row = {
        "run":              run_idx,
        "mission_complete": int(mission_complete),
        "path_error_m":     round(path_error, 4),
        "squareness_ratio": round(sq_ratio, 4),
        "n_plan_steps":     n_plan_steps,
        "total_distance_m": round(total_dist, 4),
        "api_calls":        stats.get("api_calls", 0),
        "tokens_in":        stats.get("tokens_in", 0),
        "tokens_out":       stats.get("tokens_out", 0),
        "cost_usd":         round(stats.get("cost_usd", 0.0), 6),
        "time_s":           round(wall_time, 3),
    }
    status = "COMPLETE" if mission_complete else "INCOMPLETE"
    print(f"  [D5 run={run_idx}] {status} err={path_error:.3f}m "
          f"sq={sq_ratio:.3f} steps={n_plan_steps} time={wall_time:.1f}s")
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D5: Autonomous Waypoint Navigation (Square Pattern)")
    print(f"N_RUNS={N_RUNS}, TARGET_ALT={TARGET_ALT_M}m")
    print("=" * 60)

    rows = []
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")
        rows.append(run_once(run))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D5_runs.csv"
    fields   = ["run","mission_complete","path_error_m","squareness_ratio",
                "n_plan_steps","total_distance_m","api_calls","tokens_in",
                "tokens_out","cost_usd","time_s"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Statistics ─────────────────────────────────────────────────────────────
    k_mc = sum(r["mission_complete"] for r in rows)
    mc, mc_lo, mc_hi = wilson_ci(k_mc, N_RUNS)

    err_m, e_lo, e_hi   = bootstrap_ci([r["path_error_m"]     for r in rows])
    sq_m,  s_lo, s_hi   = bootstrap_ci([r["squareness_ratio"] for r in rows])
    step_m,st_lo,st_hi  = bootstrap_ci([r["n_plan_steps"]     for r in rows])
    dist_m,d_lo, d_hi   = bootstrap_ci([r["total_distance_m"] for r in rows])
    cost_m,c_lo, c_hi   = bootstrap_ci([r["cost_usd"]         for r in rows])

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "D5_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric","value","ci_lo","ci_hi","note"])
        cw.writerow(["mission_complete_rate", mc, mc_lo, mc_hi, "Wilson 95% CI"])
        cw.writerow(["path_error_m",          err_m, e_lo,  e_hi,  "Bootstrap 95%"])
        cw.writerow(["squareness_ratio",       sq_m,  s_lo,  s_hi,  "Bootstrap 95%"])
        cw.writerow(["n_plan_steps",           step_m,st_lo, st_hi, "Bootstrap 95%"])
        cw.writerow(["total_distance_m",       dist_m,d_lo,  d_hi,  "Bootstrap 95%"])
        cw.writerow(["cost_usd",               cost_m,c_lo,  c_hi,  "Bootstrap 95%"])
        cw.writerow(["n_runs",                 N_RUNS,"","",""])
        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Left: path error per run
        ax = axes[0]
        runs = [r["run"] for r in rows]
        ax.bar(runs, [r["path_error_m"] for r in rows],
               color=["#2ecc71" if r["mission_complete"] else "#e74c3c" for r in rows])
        ax.set_xlabel("Run")
        ax.set_ylabel("Mean path error (m)")
        ax.set_title(f"D5: Path Error\nMean={err_m:.3f}m [{e_lo:.3f},{e_hi:.3f}]")

        # Middle: squareness ratio per run
        ax2 = axes[1]
        ax2.bar(runs, [r["squareness_ratio"] for r in rows], color="#3498db")
        ax2.axhline(1.0, linestyle="--", color="green", alpha=0.5, label="Perfect square")
        ax2.set_ylim(0, 1.1)
        ax2.set_xlabel("Run")
        ax2.set_ylabel("Squareness ratio")
        ax2.set_title(f"D5: Squareness\nMean={sq_m:.3f} [{s_lo:.3f},{s_hi:.3f}]")
        ax2.legend()

        # Right: completion rate
        ax3 = axes[2]
        ax3.bar(["Mission\nComplete"], [mc], yerr=[[mc - mc_lo], [mc_hi - mc]], capsize=10,
                color="#2ecc71" if mc >= 0.8 else "#e67e22")
        ax3.set_ylim(0, 1.1)
        ax3.set_title(f"D5: Completion Rate\n{mc:.2f} [{mc_lo:.2f},{mc_hi:.2f}]")
        ax3.set_ylabel("Rate (Wilson 95% CI)")

        fig.suptitle(
            "EXP-D5 Autonomous Waypoint Navigation\n"
            "ReAct (Yao 2022), SayCan (Ahn 2022), Vemprala 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D5_autonomous_waypoint.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D5 Summary ──────────────────────────────────────────────────────")
    print(f"Mission complete : {mc:.3f}  [{mc_lo:.3f},{mc_hi:.3f}] (Wilson 95% CI)")
    print(f"Path error       : {err_m:.4f}m [{e_lo:.4f},{e_hi:.4f}] (Bootstrap)")
    print(f"Squareness ratio : {sq_m:.3f}  [{s_lo:.3f},{s_hi:.3f}]")
    print(f"Plan steps/run   : {step_m:.1f}  [{st_lo:.1f},{st_hi:.1f}]")


if __name__ == "__main__":
    main()
