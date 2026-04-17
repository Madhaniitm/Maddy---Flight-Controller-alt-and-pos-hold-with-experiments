"""
EXP-D1: Vision-Only Scene Classification Accuracy
===================================================
Goal:
    Present 10 synthetic scene descriptions (via SceneSimulator) to the LLM via
    the analyze_frame tool and measure how accurately it classifies each scene
    type. N=5 independent runs; per-scene Wilson CI on accuracy.

Metrics:
    - per_scene_accuracy    : fraction correct across N runs for each scene
    - overall_accuracy      : fraction of all (scene × run) pairs correct
    - latency_s             : wall-clock time for analyze_frame call
    - api_calls             : number of API calls per run
    - tokens_in / tokens_out: token usage per run
    - cost_usd              : API cost per run

Paper References (saved in PAPER_REFS and summary CSV):
    - ReAct (Yao et al. 2022): reason+act loop — LLM uses analyze_frame as a tool
    - InnerMonologue (Huang et al. 2022): tool observations drive replanning
    - Vemprala2023: LLMs for robotics via tool APIs

Statistical validity:
    Wilson CI for per-scene and overall accuracy (binary outcomes, N=5).
    N=5 runs recommended by Bouthillier et al. 2021 for stochastic LLM evals.
"""

import os, sys, json, time, csv, math, pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import (
    DAgent, SceneSimulator, SCENE_TYPES, SCENE_LABELS,
    SCENE_DESCRIPTIONS,
)

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS   = 5
OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

# Classification keyword → canonical label (LLM free-text → discrete class)
KEYWORD_TO_LABEL = {
    "open_space":       "open_space",
    "open space":       "open_space",
    "wall_close":       "wall_close",
    "wall close":       "wall_close",
    "very close":       "wall_close",
    "immediate stop":   "wall_close",
    "wall_far":         "wall_far",
    "wall far":         "wall_far",
    "proceed with caution": "wall_far",
    "floor_or_downward":"floor_or_downward",
    "floor pattern":    "floor_or_downward",
    "tiled floor":      "floor_or_downward",
    "pointing downward":"floor_or_downward",
    "ceiling":          "ceiling_or_upward",
    "ceiling_or_upward":"ceiling_or_upward",
    "pointing upward":  "ceiling_or_upward",
    "obstacle_left":    "obstacle_left",
    "obstacle left":    "obstacle_left",
    "on left":          "obstacle_left",
    "obstacle_right":   "obstacle_right",
    "obstacle right":   "obstacle_right",
    "on right":         "obstacle_right",
    "low_visibility":   "low_visibility",
    "low visibility":   "low_visibility",
    "very dark":        "low_visibility",
    "overexposed":      "low_visibility",
    "white-out":        "low_visibility",
    "cannot identify":  "low_visibility",
}

def classify_response(text: str) -> str:
    """Extract classification label from LLM free-text response."""
    t = text.lower()
    for kw, label in KEYWORD_TO_LABEL.items():
        if kw in t:
            return label
    return "unknown"

# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(p, 4), round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)

# ── Single-run logic ───────────────────────────────────────────────────────────
CLASSIFICATION_PROMPT = (
    "Look at the camera frame and classify the scene into exactly one of: "
    "open_space, wall_close, wall_far, floor_or_downward, ceiling_or_upward, "
    "obstacle_left, obstacle_right, low_visibility. "
    "Use analyze_frame to view the frame first, then state your classification."
)

def run_once(run_idx: int) -> list[dict]:
    """
    Run all 10 scenes through the agent once.
    Returns list of per-scene result dicts.
    """
    results = []
    for scene in SCENE_TYPES:
        agent = DAgent(session_id=f"D1_r{run_idx}_{scene}")
        agent.scene_sim.set_scene(scene)
        expected = SCENE_LABELS[scene]

        t0 = time.time()
        reply, stats, trace = agent.run_agent_loop(CLASSIFICATION_PROMPT)
        latency_s = time.time() - t0

        predicted = classify_response(reply)
        correct   = int(predicted == expected)

        results.append({
            "run":       run_idx,
            "scene":     scene,
            "expected":  expected,
            "predicted": predicted,
            "correct":   correct,
            "latency_s": round(latency_s, 3),
            "api_calls": stats.get("api_calls", 0),
            "tokens_in": stats.get("tokens_in", 0),
            "tokens_out":stats.get("tokens_out", 0),
            "cost_usd":  round(stats.get("cost_usd", 0.0), 6),
        })
        print(f"  [D1 run={run_idx} scene={scene}] pred={predicted} "
              f"expect={expected} {'OK' if correct else 'FAIL'} "
              f"lat={latency_s:.2f}s")
    return results

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D1: Vision-Only Scene Classification Accuracy")
    print(f"N_RUNS={N_RUNS}, SCENES={len(SCENE_TYPES)}")
    print("=" * 60)

    all_rows = []
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")
        rows = run_once(run)
        all_rows.extend(rows)

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D1_runs.csv"
    fieldnames = ["run","scene","expected","predicted","correct",
                  "latency_s","api_calls","tokens_in","tokens_out","cost_usd"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Per-scene accuracy with Wilson CI ─────────────────────────────────────
    scene_stats = {}
    for scene in SCENE_TYPES:
        scene_rows = [r for r in all_rows if r["scene"] == scene]
        k = sum(r["correct"] for r in scene_rows)
        n = len(scene_rows)
        acc, lo, hi = wilson_ci(k, n)
        scene_stats[scene] = {
            "correct": k, "trials": n,
            "accuracy": acc, "wilson_lo": lo, "wilson_hi": hi,
        }

    # ── Overall accuracy ──────────────────────────────────────────────────────
    total_k = sum(r["correct"] for r in all_rows)
    total_n = len(all_rows)
    ov_acc, ov_lo, ov_hi = wilson_ci(total_k, total_n)

    # ── Average latency / cost ────────────────────────────────────────────────
    avg_lat  = round(sum(r["latency_s"] for r in all_rows) / len(all_rows), 3)
    avg_cost = round(sum(r["cost_usd"]  for r in all_rows) / len(all_rows), 6)

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "D1_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_lo", "ci_hi", "note"])
        w.writerow(["overall_accuracy", ov_acc, ov_lo, ov_hi,
                    f"Wilson 95% CI, N={total_n} trials"])
        for scene, s in scene_stats.items():
            w.writerow([f"accuracy_{scene}", s["accuracy"],
                        s["wilson_lo"], s["wilson_hi"],
                        f"k={s['correct']}/{s['trials']}"])
        w.writerow(["avg_latency_s",  avg_lat,  "", "", "mean over all scene×run"])
        w.writerow(["avg_cost_usd",   avg_cost, "", "", "mean per scene call"])
        w.writerow(["n_runs",         N_RUNS,   "", "", ""])
        w.writerow(["n_scenes",       len(SCENE_TYPES), "", "", ""])
        # Paper references
        for key, ref in PAPER_REFS.items():
            w.writerow([f"ref_{key}", ref, "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: per-scene accuracy bar with CI
        ax = axes[0]
        scenes = list(scene_stats.keys())
        accs   = [scene_stats[s]["accuracy"]  for s in scenes]
        errs_lo= [scene_stats[s]["accuracy"] - scene_stats[s]["wilson_lo"] for s in scenes]
        errs_hi= [scene_stats[s]["wilson_hi"] - scene_stats[s]["accuracy"] for s in scenes]
        colors = ["#2ecc71" if a >= 0.8 else "#e67e22" if a >= 0.5 else "#e74c3c"
                  for a in accs]
        ax.barh(scenes, accs, xerr=[errs_lo, errs_hi], color=colors, capsize=4)
        ax.axvline(ov_acc, color="navy", linestyle="--", label=f"Overall {ov_acc:.2f}")
        ax.set_xlabel("Accuracy (Wilson 95% CI)")
        ax.set_title("D1: Per-Scene Classification Accuracy")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

        # Right: latency distribution per scene
        ax2 = axes[1]
        for scene in SCENE_TYPES:
            lats = [r["latency_s"] for r in all_rows if r["scene"] == scene]
            ax2.scatter([scene]*len(lats), lats, s=30, alpha=0.6)
        ax2.set_xticks(range(len(SCENE_TYPES)))
        ax2.set_xticklabels(SCENE_TYPES, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Latency (s)")
        ax2.set_title("D1: Latency per Scene (N=5 runs)")

        fig.suptitle(
            "EXP-D1 Vision Classification — ReAct + analyze_frame\n"
            "Yao et al. 2022 (ReAct), Huang et al. 2022 (Inner Monologue), "
            "Vemprala et al. 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D1_vision_classification.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D1 Summary ──────────────────────────────────────────────────────")
    print(f"Overall accuracy : {ov_acc:.3f}  [{ov_lo:.3f}, {ov_hi:.3f}] (Wilson 95% CI)")
    print(f"Avg latency/call : {avg_lat:.3f} s")
    print(f"Avg cost/call    : ${avg_cost:.5f}")
    print("\nPer-scene:")
    for s, v in scene_stats.items():
        print(f"  {s:22s}: {v['accuracy']:.2f}  [{v['wilson_lo']:.2f}, {v['wilson_hi']:.2f}]"
              f"  ({v['correct']}/{v['trials']})")


if __name__ == "__main__":
    main()
