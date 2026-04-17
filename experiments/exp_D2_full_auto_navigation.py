"""
EXP-D2: Full-Autonomous Vision-Guided Wall Approach
=====================================================
Goal:
    Test the full_auto autonomy loop: drone moves forward toward a virtual wall
    at 1.5 m, using analyze_frame observations to decide when to stop.
    Agent must stop (issue stop_movement) before reaching < 20 cm from wall.
    N=5 independent runs.

Metrics:
    - stopped_safely       : agent called stop before wall_close threshold (binary)
    - stopping_distance_m  : drone X when stop was issued (bootstrap CI)
    - n_iterations         : autonomy loop iterations used (bootstrap CI)
    - api_calls            : total API calls per run (bootstrap CI)
    - cost_usd             : API cost per run (bootstrap CI)
    - time_to_stop_s       : wall-clock from start to stop (bootstrap CI)

Paper References:
    - ReAct (Yao et al. 2022): reason-act-observe loop in autonomy_loop()
    - InnerMonologue (Huang et al. 2022): visual observations drive motion decisions
    - SayCan (Ahn et al. 2022): LLM-selected affordance-grounded actions
    - Vemprala2023: ChatGPT for robotics, tool API paradigm
"""

import os, sys, json, time, csv, math, pathlib, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent, SCENE_TYPES

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS         = 5
WALL_POS_M     = 1.5        # virtual wall X position
SAFE_STOP_M    = 0.20       # must stop before this distance from wall
MAX_ITERATIONS = 25
OUT_DIR        = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
    "SayCan":         "Ahn et al. 2022 — Do As I Can, Not As I Say: Grounding Language in Robotic Affordances",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

NAVIGATION_GOAL = (
    "You are controlling a drone in a corridor. Your goal: move forward toward the wall "
    "ahead. Use analyze_frame each step to check proximity. Stop immediately if the wall "
    "is very close (< 25 cm). Move in small increments (0.1–0.2 m). "
    "Report 'MISSION COMPLETE' once safely stopped."
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
    arr    = np.array(data, dtype=float)
    boots  = [stat(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)), 4), round(float(lo), 4), round(float(hi), 4)

# ── Single-run logic ───────────────────────────────────────────────────────────
def run_once(run_idx: int) -> dict:
    agent = DAgent(session_id=f"D2_r{run_idx}")
    agent.scene_sim.set_wall_position(WALL_POS_M)
    agent.scene_sim.set_scene("open_space")

    # Arm and hover before nav test
    agent.execute_tool("arm", {})
    agent.execute_tool("find_hover_throttle", {})
    agent.execute_tool("enable_altitude_hold", {})
    agent.execute_tool("set_altitude_target", {"target_m": 0.8})
    time.sleep(0.05)

    drone_x_at_stop = WALL_POS_M   # pessimistic default
    stopped_safely  = False
    n_iter          = 0
    stop_detected   = False

    t0 = time.time()

    # Wrap autonomy loop with monitoring
    def on_iteration_end(iteration: int, last_reply: str, tool_calls: list):
        nonlocal drone_x_at_stop, stopped_safely, n_iter, stop_detected
        n_iter = iteration
        # Check if stop_movement was called
        for tc in tool_calls:
            if tc.get("name") == "stop_movement":
                stop_detected = True
                drone_x_at_stop = agent.state.x
                dist_from_wall  = WALL_POS_M - drone_x_at_stop
                stopped_safely  = dist_from_wall >= SAFE_STOP_M
                return True   # signal to end loop
        # Safety: if drone has been pushed to wall
        if agent.state.x >= WALL_POS_M - 0.05:
            drone_x_at_stop = agent.state.x
            return True
        return False

    # Run autonomy loop
    reply, stats, trace = agent.autonomy_loop(
        goal           = NAVIGATION_GOAL,
        mode           = "full_auto",
        max_iterations = MAX_ITERATIONS,
        iteration_hook = on_iteration_end,
    )

    wall_time = time.time() - t0
    if not stop_detected:
        # Check final drone position
        drone_x_at_stop = agent.state.x
        dist_from_wall  = WALL_POS_M - drone_x_at_stop
        stopped_safely  = dist_from_wall >= SAFE_STOP_M

    stopping_dist = WALL_POS_M - drone_x_at_stop

    row = {
        "run":               run_idx,
        "stopped_safely":    int(stopped_safely),
        "stopping_dist_m":   round(stopping_dist, 3),
        "n_iterations":      n_iter,
        "api_calls":         stats.get("api_calls", 0),
        "tokens_in":         stats.get("tokens_in", 0),
        "tokens_out":        stats.get("tokens_out", 0),
        "cost_usd":          round(stats.get("cost_usd", 0.0), 6),
        "time_to_stop_s":    round(wall_time, 3),
    }
    status = "SAFE" if stopped_safely else "CRASH"
    print(f"  [D2 run={run_idx}] {status} dist={stopping_dist:.3f}m "
          f"iters={n_iter} time={wall_time:.1f}s")
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D2: Full-Autonomous Vision-Guided Wall Approach")
    print(f"N_RUNS={N_RUNS}, WALL={WALL_POS_M}m, SAFE_STOP>{SAFE_STOP_M}m")
    print("=" * 60)

    rows = []
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")
        rows.append(run_once(run))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D2_runs.csv"
    fields   = ["run","stopped_safely","stopping_dist_m","n_iterations",
                "api_calls","tokens_in","tokens_out","cost_usd","time_to_stop_s"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Statistics ─────────────────────────────────────────────────────────────
    k_safe = sum(r["stopped_safely"] for r in rows)
    sr, sr_lo, sr_hi = wilson_ci(k_safe, N_RUNS)

    dist_m, d_lo, d_hi = bootstrap_ci([r["stopping_dist_m"] for r in rows])
    iters,  i_lo, i_hi = bootstrap_ci([r["n_iterations"]    for r in rows])
    calls,  c_lo, c_hi = bootstrap_ci([r["api_calls"]       for r in rows])
    cost_m, co_lo,co_hi= bootstrap_ci([r["cost_usd"]        for r in rows])
    ttime,  t_lo, t_hi = bootstrap_ci([r["time_to_stop_s"]  for r in rows])

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "D2_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_lo", "ci_hi", "note"])
        w.writerow(["safe_stop_rate",    sr,     sr_lo,  sr_hi,  "Wilson 95% CI"])
        w.writerow(["stopping_dist_m",   dist_m, d_lo,   d_hi,   "Bootstrap 95% CI"])
        w.writerow(["n_iterations",      iters,  i_lo,   i_hi,   "Bootstrap 95% CI"])
        w.writerow(["api_calls",         calls,  c_lo,   c_hi,   "Bootstrap 95% CI"])
        w.writerow(["cost_usd",          cost_m, co_lo,  co_hi,  "Bootstrap 95% CI"])
        w.writerow(["time_to_stop_s",    ttime,  t_lo,   t_hi,   "Bootstrap 95% CI"])
        w.writerow(["n_runs",            N_RUNS, "",     "",      ""])
        for key, ref in PAPER_REFS.items():
            w.writerow([f"ref_{key}", ref, "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Left: stopping distance per run
        ax = axes[0]
        runs = [r["run"] for r in rows]
        dists = [r["stopping_dist_m"] for r in rows]
        colors = ["#2ecc71" if r["stopped_safely"] else "#e74c3c" for r in rows]
        ax.bar(runs, dists, color=colors)
        ax.axhline(SAFE_STOP_M, color="red", linestyle="--", label=f"Safe threshold {SAFE_STOP_M}m")
        ax.set_xlabel("Run")
        ax.set_ylabel("Distance from wall (m)")
        ax.set_title("D2: Stopping Distance per Run")
        ax.legend()

        # Middle: iteration count per run
        ax2 = axes[1]
        ax2.bar(runs, [r["n_iterations"] for r in rows], color="#3498db")
        ax2.set_xlabel("Run")
        ax2.set_ylabel("Iterations")
        ax2.set_title("D2: Autonomy Loop Iterations")

        # Right: safe stop rate with CI
        ax3 = axes[2]
        ax3.bar(["Safe\nStop Rate"], [sr], yerr=[[sr - sr_lo], [sr_hi - sr]], capsize=10,
                color="#2ecc71" if sr >= 0.8 else "#e67e22")
        ax3.set_ylim(0, 1.1)
        ax3.axhline(1.0, color="green", linestyle=":", alpha=0.5)
        ax3.set_title(f"D2: Safe Stop Rate\n{sr:.2f} [{sr_lo:.2f}, {sr_hi:.2f}]")
        ax3.set_ylabel("Rate (Wilson 95% CI)")

        fig.suptitle(
            "EXP-D2 Full-Autonomous Wall Approach\n"
            "ReAct (Yao 2022), Inner Monologue (Huang 2022), SayCan (Ahn 2022)",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D2_full_auto_navigation.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D2 Summary ──────────────────────────────────────────────────────")
    print(f"Safe stop rate   : {sr:.3f}  [{sr_lo:.3f}, {sr_hi:.3f}] (Wilson 95% CI)")
    print(f"Stopping dist    : {dist_m:.3f}m [{d_lo:.3f}, {d_hi:.3f}] (Bootstrap 95% CI)")
    print(f"Iterations/run   : {iters:.1f}  [{i_lo:.1f}, {i_hi:.1f}]")
    print(f"Cost/run         : ${cost_m:.5f}")


if __name__ == "__main__":
    main()
