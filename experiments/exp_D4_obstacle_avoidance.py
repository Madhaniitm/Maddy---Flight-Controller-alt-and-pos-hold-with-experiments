"""
EXP-D4: Vision-Based Obstacle Avoidance at Multiple Distances
=============================================================
Goal:
    Present the agent with 4 obstacle distances (0.15 m, 0.30 m, 0.55 m, 0.90 m)
    and verify it issues the correct stop/avoid action. 10 trials per distance
    per run; N=5 independent runs → 200 total trials.

    Correct actions:
        dist ≤ 0.25 m → must call stop_movement (wall_close scenario)
        dist ≤ 0.55 m → may call stop_movement OR move_backward (wall_far caution)
        dist > 0.55 m → may call move_forward (safe to proceed)

Metrics:
    - per_distance_accuracy : fraction correct actions per distance (Wilson CI)
    - overall_accuracy      : fraction correct over all 200 trials (Wilson CI)
    - response_latency_s    : per-trial wall-clock (bootstrap CI)
    - api_calls             : per-trial (bootstrap CI)
    - cost_usd              : per-trial (bootstrap CI)

Paper References:
    - ReAct (Yao et al. 2022): reason-act-observe loop per trial
    - InnerMonologue (Huang et al. 2022): scene text drives motion decision
    - SayCan (Ahn et al. 2022): affordance-grounded LLM action selection
    - Vemprala2023: LLMs for robotics tool API
"""

import os, sys, json, time, csv, math, pathlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS          = 5
TRIALS_PER_DIST = 10
DISTANCES_M     = [0.15, 0.30, 0.55, 0.90]   # obstacle distances to test
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
    "SayCan":         "Ahn et al. 2022 — Do As I Can, Not As I Say: Grounding Language in Robotic Affordances",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

# What counts as a correct action for each distance bracket
def is_correct(tool_name: str, dist_m: float) -> bool:
    if dist_m <= 0.25:
        return tool_name == "stop_movement"
    elif dist_m <= 0.55:
        return tool_name in ("stop_movement", "move_backward")
    else:   # > 0.55 m
        return tool_name in ("move_forward", "analyze_frame")

AVOIDANCE_PROMPT = (
    "Analyze the camera frame, identify the obstacle distance, and decide whether to "
    "stop, move backward, or move forward. Issue exactly ONE movement command. "
    "Use analyze_frame first."
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

# ── Single trial ───────────────────────────────────────────────────────────────
def run_trial(run_idx: int, dist_m: float, trial_idx: int) -> dict:
    agent = DAgent(session_id=f"D4_r{run_idx}_d{int(dist_m*100)}_t{trial_idx}")
    agent.scene_sim.set_obstacle_distance(dist_m)

    t0 = time.time()
    reply, stats, trace = agent.run_agent_loop(AVOIDANCE_PROMPT)
    latency = time.time() - t0

    # Extract first movement command issued
    action_taken = "none"
    for step in trace:
        if step.get("role") == "tool_use":
            name = step.get("name", "")
            if name in ("stop_movement", "move_forward", "move_backward",
                        "move_left", "move_right"):
                action_taken = name
                break

    correct = is_correct(action_taken, dist_m)
    return {
        "run":           run_idx,
        "dist_m":        dist_m,
        "trial":         trial_idx,
        "action_taken":  action_taken,
        "correct":       int(correct),
        "latency_s":     round(latency, 3),
        "api_calls":     stats.get("api_calls", 0),
        "tokens_in":     stats.get("tokens_in", 0),
        "tokens_out":    stats.get("tokens_out", 0),
        "cost_usd":      round(stats.get("cost_usd", 0.0), 6),
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D4: Vision-Based Obstacle Avoidance")
    print(f"N_RUNS={N_RUNS}, DISTANCES={DISTANCES_M}, TRIALS_PER_DIST={TRIALS_PER_DIST}")
    print("=" * 60)

    all_rows = []
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")
        for dist in DISTANCES_M:
            for trial in range(1, TRIALS_PER_DIST + 1):
                row = run_trial(run, dist, trial)
                all_rows.append(row)
                status = "OK" if row["correct"] else "FAIL"
                print(f"  dist={dist:.2f}m trial={trial} action={row['action_taken']:16s} "
                      f"{status} lat={row['latency_s']:.2f}s")

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D4_runs.csv"
    fields   = ["run","dist_m","trial","action_taken","correct",
                "latency_s","api_calls","tokens_in","tokens_out","cost_usd"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Per-distance accuracy with Wilson CI ───────────────────────────────────
    dist_stats = {}
    for dist in DISTANCES_M:
        dr = [r for r in all_rows if r["dist_m"] == dist]
        k  = sum(r["correct"] for r in dr)
        n  = len(dr)
        acc, lo, hi = wilson_ci(k, n)
        dist_stats[dist] = {"k": k, "n": n, "acc": acc, "lo": lo, "hi": hi}

    total_k = sum(r["correct"] for r in all_rows)
    total_n = len(all_rows)
    ov_acc, ov_lo, ov_hi = wilson_ci(total_k, total_n)

    lat_m, lat_lo, lat_hi  = bootstrap_ci([r["latency_s"] for r in all_rows])
    cost_m, c_lo, c_hi     = bootstrap_ci([r["cost_usd"]  for r in all_rows])

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "D4_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric","value","ci_lo","ci_hi","note"])
        cw.writerow(["overall_accuracy", ov_acc, ov_lo, ov_hi,
                     f"Wilson 95% CI, N={total_n}"])
        for dist, s in dist_stats.items():
            cw.writerow([f"accuracy_{dist}m", s["acc"], s["lo"], s["hi"],
                         f"k={s['k']}/{s['n']}"])
        cw.writerow(["latency_s",   lat_m,  lat_lo, lat_hi, "Bootstrap 95%"])
        cw.writerow(["cost_usd",    cost_m, c_lo,   c_hi,   "Bootstrap 95%"])
        cw.writerow(["n_runs",      N_RUNS, "", "", ""])
        cw.writerow(["total_trials",total_n,"","",""])
        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        xs   = [f"{d:.2f}m" for d in DISTANCES_M]
        accs = [dist_stats[d]["acc"] for d in DISTANCES_M]
        errs_lo = [dist_stats[d]["acc"] - dist_stats[d]["lo"] for d in DISTANCES_M]
        errs_hi = [dist_stats[d]["hi"] - dist_stats[d]["acc"] for d in DISTANCES_M]
        clrs = ["#2ecc71" if a >= 0.8 else "#e67e22" if a >= 0.5 else "#e74c3c"
                for a in accs]
        ax.bar(xs, accs, yerr=[errs_lo, errs_hi], capsize=8, color=clrs)
        ax.axhline(ov_acc, linestyle="--", color="navy",
                   label=f"Overall {ov_acc:.2f} [{ov_lo:.2f},{ov_hi:.2f}]")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Obstacle distance")
        ax.set_ylabel("Accuracy (Wilson 95% CI)")
        ax.set_title("D4: Avoidance Accuracy by Distance")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        for dist in DISTANCES_M:
            lats = [r["latency_s"] for r in all_rows if r["dist_m"] == dist]
            ax2.scatter([f"{dist:.2f}m"]*len(lats), lats, s=30, alpha=0.5)
        ax2.set_xlabel("Obstacle distance")
        ax2.set_ylabel("Latency (s)")
        ax2.set_title("D4: Per-Trial Latency")

        fig.suptitle(
            "EXP-D4 Vision-Based Obstacle Avoidance\n"
            "ReAct (Yao 2022), Inner Monologue (Huang 2022), SayCan (Ahn 2022)",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D4_obstacle_avoidance.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D4 Summary ──────────────────────────────────────────────────────")
    print(f"Overall accuracy : {ov_acc:.3f} [{ov_lo:.3f},{ov_hi:.3f}] (Wilson 95% CI)")
    for dist, s in dist_stats.items():
        print(f"  {dist:.2f}m : {s['acc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]  "
              f"({s['k']}/{s['n']})")


if __name__ == "__main__":
    main()
