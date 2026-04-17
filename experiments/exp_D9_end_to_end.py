"""
EXP-D9: End-to-End Autonomous Room Exploration (Flagship)
==========================================================
Goal:
    "Explore the room for 15 seconds" — the most ambitious D-series test.
    The agent receives a single high-level goal and must autonomously:
        1. Arm and take off to 0.8 m
        2. Use analyze_frame in a loop to navigate around obstacles
        3. Visit as many distinct grid cells as possible in 15 s of sim time
        4. Safely land and disarm on its own

    Room: 3 m × 3 m virtual space with 3 fixed obstacle positions.
    Grid cells: 0.5 m × 0.5 m = 36 cells.
    N=5 independent runs.

Metrics:
    - mission_complete      : agent landed safely within budget (Wilson CI)
    - cells_visited         : distinct 0.5 m grid cells entered (bootstrap CI)
    - coverage_pct          : cells_visited / 36 × 100 (bootstrap CI)
    - collisions            : number of wall_close events without stop (bootstrap CI)
    - n_vision_calls        : total analyze_frame calls (bootstrap CI)
    - total_distance_m      : total path length (bootstrap CI)
    - api_calls             : total API calls per run (bootstrap CI)
    - cost_usd              : per-run (bootstrap CI)

Paper References:
    - ReAct (Yao et al. 2022): end-to-end reason-act-observe loop
    - InnerMonologue (Huang et al. 2022): visual observations drive replanning
    - SayCan (Ahn et al. 2022): affordance-grounded action selection at each step
    - Vemprala2023: LLMs for full autonomous mission execution
"""

import os, sys, json, time, csv, math, pathlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent, SCENE_DESCRIPTIONS

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS          = 5
SIM_DURATION_S  = 15.0
TARGET_ALT_M    = 0.8
ROOM_SIZE_M     = 3.0
CELL_SIZE_M     = 0.5
N_CELLS_SIDE    = int(ROOM_SIZE_M / CELL_SIZE_M)   # 6
N_CELLS_TOTAL   = N_CELLS_SIDE ** 2               # 36
MAX_ITERATIONS  = 40
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
    "SayCan":         "Ahn et al. 2022 — Do As I Can, Not As I Say: Grounding Language in Robotic Affordances",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

# Fixed obstacles (x, y) in the 3×3 room
OBSTACLES = [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)]
OBSTACLE_RADIUS_M = 0.20

EXPLORATION_GOAL = (
    "Explore as much of the room as possible in 15 seconds. "
    "Start by arming and taking off to 0.8 m. "
    "Use analyze_frame before every movement to check for obstacles. "
    "Move in varied directions to maximise area covered. "
    "Avoid obstacles. After 15 seconds of exploration, land safely and disarm. "
    "Announce 'EXPLORATION COMPLETE' when disarmed."
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

def cell_id(x: float, y: float) -> tuple:
    cx = min(int(x / CELL_SIZE_M), N_CELLS_SIDE - 1)
    cy = min(int(y / CELL_SIZE_M), N_CELLS_SIDE - 1)
    return (max(cx, 0), max(cy, 0))

def near_obstacle(x: float, y: float) -> bool:
    return any(math.sqrt((x-ox)**2 + (y-oy)**2) < OBSTACLE_RADIUS_M
               for ox, oy in OBSTACLES)

# ── Single run ─────────────────────────────────────────────────────────────────
def run_once(run_idx: int) -> dict:
    agent = DAgent(session_id=f"D9_r{run_idx}")
    # Start in open area
    agent.scene_sim.set_scene("open_space")
    agent.scene_sim.set_wall_position(ROOM_SIZE_M)

    visited_cells = set()
    positions     = []
    collisions    = 0
    vision_calls  = 0
    mission_complete = False

    # Track position history for distance calculation
    prev_pos = (agent.state.x, getattr(agent.state, "y", 0.0))
    total_distance = 0.0
    start_time = time.time()

    def on_iteration_end(iteration: int, last_reply: str, tool_calls: list):
        nonlocal collisions, vision_calls, mission_complete, total_distance, prev_pos

        x  = agent.state.x
        y  = getattr(agent.state, "y", 0.0)

        # Update coverage
        visited_cells.add(cell_id(x, y))
        positions.append((x, y))

        # Accumulate distance
        dist_step = math.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
        total_distance += dist_step
        prev_pos = (x, y)

        for tc in tool_calls:
            tname = tc.get("name", "")
            if tname == "analyze_frame":
                vision_calls += 1
            elif tname in ("move_forward", "move_backward", "move_left", "move_right"):
                # Check obstacle proximity post-move
                if near_obstacle(x, y):
                    collisions += 1

        # Check elapsed sim time or completion signal
        elapsed = time.time() - start_time
        if elapsed >= SIM_DURATION_S:
            return True
        if "EXPLORATION COMPLETE" in last_reply.upper():
            mission_complete = True
            return True
        return False

    t0 = time.time()
    reply, stats, trace = agent.autonomy_loop(
        goal           = EXPLORATION_GOAL,
        mode           = "full_auto",
        max_iterations = MAX_ITERATIONS,
        iteration_hook = on_iteration_end,
    )
    wall_time = time.time() - t0

    if "EXPLORATION COMPLETE" in reply.upper():
        mission_complete = True

    # Check landed/disarmed state
    if agent.state.armed == False:
        mission_complete = True

    n_cells    = len(visited_cells)
    coverage   = n_cells / N_CELLS_TOTAL * 100.0

    row = {
        "run":              run_idx,
        "mission_complete": int(mission_complete),
        "cells_visited":    n_cells,
        "coverage_pct":     round(coverage, 2),
        "collisions":       collisions,
        "n_vision_calls":   vision_calls,
        "total_distance_m": round(total_distance, 3),
        "api_calls":        stats.get("api_calls", 0),
        "tokens_in":        stats.get("tokens_in", 0),
        "tokens_out":       stats.get("tokens_out", 0),
        "cost_usd":         round(stats.get("cost_usd", 0.0), 6),
        "time_s":           round(wall_time, 3),
    }
    status = "COMPLETE" if mission_complete else "TIMEOUT"
    print(f"  [D9 run={run_idx}] {status} cells={n_cells}/{N_CELLS_TOTAL} "
          f"cov={coverage:.1f}% col={collisions} vis={vision_calls} "
          f"dist={total_distance:.2f}m time={wall_time:.1f}s")
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D9: End-to-End Autonomous Room Exploration")
    print(f"N_RUNS={N_RUNS}, ROOM={ROOM_SIZE_M}m×{ROOM_SIZE_M}m, "
          f"CELLS={N_CELLS_TOTAL}, SIM_T={SIM_DURATION_S}s")
    print("=" * 60)

    rows = []
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")
        rows.append(run_once(run))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D9_runs.csv"
    fields   = ["run","mission_complete","cells_visited","coverage_pct","collisions",
                "n_vision_calls","total_distance_m","api_calls","tokens_in",
                "tokens_out","cost_usd","time_s"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Statistics ─────────────────────────────────────────────────────────────
    k_mc = sum(r["mission_complete"] for r in rows)
    mc, mc_lo, mc_hi   = wilson_ci(k_mc, N_RUNS)

    cov_m, cv_lo, cv_hi  = bootstrap_ci([r["coverage_pct"]     for r in rows])
    cell_m,cl_lo, cl_hi  = bootstrap_ci([r["cells_visited"]    for r in rows])
    col_m, co_lo, co_hi  = bootstrap_ci([r["collisions"]       for r in rows])
    vis_m, vi_lo, vi_hi  = bootstrap_ci([r["n_vision_calls"]   for r in rows])
    dist_m,d_lo,  d_hi   = bootstrap_ci([r["total_distance_m"] for r in rows])
    cost_m,c_lo,  c_hi   = bootstrap_ci([r["cost_usd"]         for r in rows])
    call_m,ca_lo, ca_hi  = bootstrap_ci([r["api_calls"]        for r in rows])

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "D9_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric","value","ci_lo","ci_hi","note"])
        cw.writerow(["mission_complete_rate",  mc,     mc_lo,  mc_hi,  "Wilson 95% CI"])
        cw.writerow(["coverage_pct",           cov_m,  cv_lo,  cv_hi,  "Bootstrap 95%"])
        cw.writerow(["cells_visited",          cell_m, cl_lo,  cl_hi,  "Bootstrap 95%"])
        cw.writerow(["collisions",             col_m,  co_lo,  co_hi,  "Bootstrap 95%"])
        cw.writerow(["n_vision_calls",         vis_m,  vi_lo,  vi_hi,  "Bootstrap 95%"])
        cw.writerow(["total_distance_m",       dist_m, d_lo,   d_hi,   "Bootstrap 95%"])
        cw.writerow(["api_calls",              call_m, ca_lo,  ca_hi,  "Bootstrap 95%"])
        cw.writerow(["cost_usd",               cost_m, c_lo,   c_hi,   "Bootstrap 95%"])
        cw.writerow(["n_runs",                 N_RUNS, "",     "",      ""])
        cw.writerow(["n_cells_total",          N_CELLS_TOTAL, "", "",  "36 cells in 3×3 m room"])
        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # Coverage per run
        ax = axes[0, 0]
        runs = [r["run"] for r in rows]
        clrs = ["#2ecc71" if r["mission_complete"] else "#e74c3c" for r in rows]
        ax.bar(runs, [r["coverage_pct"] for r in rows], color=clrs)
        ax.axhline(cov_m, linestyle="--", color="navy",
                   label=f"Mean {cov_m:.1f}% [{cv_lo:.1f},{cv_hi:.1f}]")
        ax.set_xlabel("Run")
        ax.set_ylabel("Coverage (%)")
        ax.set_title("D9: Room Coverage per Run")
        ax.legend(fontsize=8)

        # Collisions per run
        ax2 = axes[0, 1]
        ax2.bar(runs, [r["collisions"] for r in rows], color="#e74c3c")
        ax2.set_xlabel("Run")
        ax2.set_ylabel("Collisions")
        ax2.set_title(f"D9: Collisions per Run\nMean={col_m:.1f} [{co_lo:.1f},{co_hi:.1f}]")

        # Vision calls per run
        ax3 = axes[1, 0]
        ax3.bar(runs, [r["n_vision_calls"] for r in rows], color="#3498db")
        ax3.set_xlabel("Run")
        ax3.set_ylabel("analyze_frame calls")
        ax3.set_title(f"D9: Vision Calls per Run\nMean={vis_m:.1f}")

        # Summary bar: key metrics normalised
        ax4 = axes[1, 1]
        metrics  = ["Mission\nComplete", "Coverage\n(%/100)", "Zero\nCollisions"]
        vals     = [mc, cov_m/100.0, 1.0 - col_m/max(1.0, max(r["collisions"] for r in rows))]
        err_lo   = [mc - mc_lo, (cov_m - cv_lo)/100.0, 0]
        err_hi   = [mc_hi - mc, (cv_hi - cov_m)/100.0, 0]
        ax4.bar(metrics, vals, yerr=[err_lo, err_hi], capsize=8,
                color=["#2ecc71","#3498db","#e67e22"])
        ax4.set_ylim(0, 1.2)
        ax4.set_title("D9: Normalised Key Metrics")
        ax4.set_ylabel("Score (CI shown where applicable)")

        fig.suptitle(
            "EXP-D9 End-to-End Autonomous Room Exploration\n"
            "ReAct (Yao 2022), Inner Monologue (Huang 2022), "
            "SayCan (Ahn 2022), Vemprala 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D9_end_to_end.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D9 Summary ──────────────────────────────────────────────────────")
    print(f"Mission complete : {mc:.3f}  [{mc_lo:.3f},{mc_hi:.3f}] (Wilson 95% CI)")
    print(f"Coverage         : {cov_m:.1f}% [{cv_lo:.1f},{cv_hi:.1f}%] (Bootstrap)")
    print(f"Cells visited    : {cell_m:.1f}  [{cl_lo:.1f},{cl_hi:.1f}]")
    print(f"Collisions/run   : {col_m:.1f}  [{co_lo:.1f},{co_hi:.1f}]")
    print(f"Vision calls/run : {vis_m:.1f}  [{vi_lo:.1f},{vi_hi:.1f}]")
    print(f"Distance/run     : {dist_m:.2f}m [{d_lo:.2f},{d_hi:.2f}m]")
    print(f"API calls/run    : {call_m:.1f}  [{ca_lo:.1f},{ca_hi:.1f}]")
    print(f"Cost/run         : ${cost_m:.5f}")


if __name__ == "__main__":
    main()
