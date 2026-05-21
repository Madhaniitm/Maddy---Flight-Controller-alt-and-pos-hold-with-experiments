"""
EXP-G1v2: Enhanced YOLO Tier Validation
========================================
Goal:
    Rerun of G1 with research-paper improvements to the YOLO tier.
    Tests whether the enhanced pipeline closes the gap identified in G1:
      - wall_close:    yolo_only=0%,  combined=20%  (YOLO COCO blind spot)
      - blocked_lens:  yolo_only=0%,  combined=80%  (YOLO COCO blind spot)
      - dim_light:     yolo_only=0%,  combined=60%  (low-light failure)
      - person_far:    yolo_only=0%,  combined=0%   (distance heuristic error)

Enhancements (see enhanced_yolo_pipeline.py for paper citations):
    1. YOLO-World (CVPR 2024)  — custom hazard classes beyond COCO-80
    2. CLAHE enhancement       — low-light preprocessing before YOLO
    3. CLIP scene screening    — category-agnostic hazard detection fallback
    4. YOLO11 fallback         — better architecture than YOLOv8n
    5. LLM distance adjudication — prompt instructs LLM to validate est_dist

Three conditions (same as G1 for direct comparison):
    yolo_enhanced_only : YOLO-World + CLAHE + CLIP + rule-based risk
    llm_only           : LLM image-only (baseline, same prompt as G1)
    combined_enhanced  : enhanced metadata + distance-adjudication prompt → LLM

Models (LLM conditions): claude, gpt4o, gpt4o_mini, gemini
Scenes: all 8 (run03 real hardware frames)
N runs: 5

Output: Image verbalization experiments/results/G1v2_runs_<ts>.csv
        Image verbalization experiments/results/G1v2_summary_<ts>.csv
"""

import sys, pathlib, datetime, time
import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
EXP_DIR   = pathlib.Path(__file__).parent
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (  # noqa: E402
    get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR, SCENES
)
from enhanced_yolo_pipeline import (  # noqa: E402
    load_enhanced_yolo, load_clip, enhanced_yolo_infer,
    enhanced_rule_risk, LLM_ONLY_PROMPT, COMBINED_PROMPT_TEMPLATE
)

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]


def main():
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_total = 2 * len(MODELS) * len(SCENES) * N_RUNS
    yolo_tot  = len(SCENES) * N_RUNS

    print("=" * 65)
    print("EXP-G1v2: Enhanced YOLO Tier Validation")
    print("Enhancements: YOLO-World + CLAHE + CLIP + LLM distance adjudication")
    print(f"Conditions: yolo_enhanced_only | llm_only | combined_enhanced")
    print(f"Scenes={len(SCENES)}  Models={MODELS}  N={N_RUNS}")
    print(f"LLM trials: {llm_total}  YOLO trials: {yolo_tot}")
    print(f"Estimated time: ~{llm_total * 5 // 60} min")
    print("=" * 65)

    print("\nLoading enhanced YOLO tier…")
    yolo_model, yolo_type = load_enhanced_yolo()

    print("\nLoading CLIP scene screener…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows = []

    for scene in SCENES:
        label = scene["label"]
        truth = scene["truth"]
        print(f"\n── Scene: {label}  (truth={truth}) ──")

        jpeg = get_saved_frame(label)

        # ── yolo_enhanced_only (N_RUNS timing + rule-based risk, no LLM)
        tier2_result = enhanced_yolo_infer(
            yolo_model, yolo_type, clip_model, clip_preprocess, clip_tokenizer, jpeg
        )
        yolo_risk = enhanced_rule_risk(tier2_result["yolo_meta"], tier2_result["clip_risk"])
        correct   = int(yolo_risk == truth)

        for run in range(1, N_RUNS + 1):
            # Re-time for each run (CLAHE + YOLO timing variance)
            t2 = enhanced_yolo_infer(
                yolo_model, yolo_type, clip_model, clip_preprocess, clip_tokenizer, jpeg
            )
            all_rows.append({
                "condition":    "yolo_enhanced_only",
                "model":        "yolo_enhanced",
                "scene_label":  label, "truth": truth, "run": run,
                "yolo_type":    t2["yolo_type"],
                "yolo_meta":    t2["yolo_meta"],
                "clip_label":   t2["clip_label"],
                "clip_risk":    t2["clip_risk"],
                "clip_conf":    t2["clip_conf"],
                "detected_risk":  yolo_risk,
                "risk_correct":   correct,
                "quality_score":  correct,
                "s1_scene": 0, "s2_proximity": 0, "s3_risk": correct,
                "s4_length": 0, "s5_pilot_action": 0,
                "word_count": 0,
                "latency_ms":    t2["yolo_ms"],
                "input_tokens":  0, "output_tokens": 0, "cost_usd": 0.0,
                "reply":         "",
                "error":         "",
            })

        print(f"   yolo_enhanced_only: det={yolo_risk:8s}  correct={correct}  "
              f"yolo={tier2_result['yolo_ms']:.1f}ms  "
              f"clip={tier2_result['clip_label'][:30]}({tier2_result['clip_risk']})")

        # ── llm_only and combined_enhanced
        for condition in ("llm_only", "combined_enhanced"):

            if condition == "llm_only":
                prompt = LLM_ONLY_PROMPT
            else:
                prompt = COMBINED_PROMPT_TEMPLATE.format(
                    yolo_meta  = tier2_result["yolo_meta"],
                    clip_label = tier2_result["clip_label"],
                    clip_conf  = tier2_result["clip_conf"],
                    clip_risk  = tier2_result["clip_risk"],
                )

            for run in range(1, N_RUNS + 1):
                for model in MODELS:
                    res    = call_vision_llm(jpeg, prompt, model=model,
                                             max_tokens=300, temperature=0.0)
                    scores = score_verbalization(res["reply"], truth)

                    all_rows.append({
                        "condition":      condition,
                        "model":          model,
                        "scene_label":    label,
                        "truth":          truth,
                        "run":            run,
                        "yolo_type":      tier2_result["yolo_type"] if condition == "combined_enhanced" else "",
                        "yolo_meta":      tier2_result["yolo_meta"] if condition == "combined_enhanced" else "",
                        "clip_label":     tier2_result["clip_label"] if condition == "combined_enhanced" else "",
                        "clip_risk":      tier2_result["clip_risk"] if condition == "combined_enhanced" else "",
                        "clip_conf":      tier2_result["clip_conf"] if condition == "combined_enhanced" else 0.0,
                        "detected_risk":  scores["detected_risk"] or "",
                        "risk_correct":   scores["s3_risk"],
                        "quality_score":  scores["quality_score"],
                        "s1_scene":       scores["s1_scene"],
                        "s2_proximity":   scores["s2_proximity"],
                        "s3_risk":        scores["s3_risk"],
                        "s4_length":      scores["s4_length"],
                        "s5_pilot_action": scores["s5_pilot_action"],
                        "word_count":     scores["word_count"],
                        "latency_ms":     res["latency_ms"],
                        "input_tokens":   res["input_tokens"],
                        "output_tokens":  res["output_tokens"],
                        "cost_usd":       res["cost_usd"],
                        "reply":          res["reply"],
                        "error":          res["error"][:80] if res["error"] else "",
                    })

                    status = "✓" if scores["s3_risk"] else "✗"
                    print(f"   {condition:20s}  {model:12s}  run={run}  "
                          f"det={scores['detected_risk'] or '?':8s}  "
                          f"{status}  lat={res['latency_ms']:.0f}ms")
                time.sleep(0.3)

    # ── Save runs CSV
    fields = [
        "condition", "model", "scene_label", "truth", "run",
        "yolo_type", "yolo_meta", "clip_label", "clip_risk", "clip_conf",
        "detected_risk", "risk_correct", "quality_score",
        "s1_scene", "s2_proximity", "s3_risk", "s4_length", "s5_pilot_action",
        "word_count", "latency_ms", "input_tokens", "output_tokens",
        "cost_usd", "reply", "error",
    ]
    runs_csv = RESULTS_DIR / f"G1v2_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary
    CONDITIONS = ["yolo_enhanced_only", "llm_only", "combined_enhanced"]
    print(f"\n── G1v2 Summary ─────────────────────────────────────────────────")
    print(f"  {'condition':22s}  {'model':14s}  {'risk_acc':>8s}  "
          f"{'quality':>7s}  {'lat_ms':>8s}  {'cost':>9s}")
    print("-" * 78)

    summary_rows = []
    for cond in CONDITIONS:
        models_for = ["yolo_enhanced"] if cond == "yolo_enhanced_only" else MODELS
        for model in models_for:
            hr = [r for r in all_rows
                  if r["condition"] == cond and r["model"] == model
                  and not r["error"]]
            if not hr:
                continue
            acc, alo, ahi = wilson_ci(sum(r["risk_correct"] for r in hr), len(hr))
            qm, _, _      = bootstrap_ci([r["quality_score"] for r in hr])
            lm, _, _      = bootstrap_ci([r["latency_ms"]    for r in hr])
            cm, _, _      = bootstrap_ci([r["cost_usd"]      for r in hr])
            print(f"  {cond:22s}  {model:14s}  "
                  f"{acc:.3f}[{alo:.3f},{ahi:.3f}]  "
                  f"{qm:.2f}/5    {lm:.0f}ms  ${cm:.5f}")
            summary_rows.append({
                "condition": cond, "model": model, "n_trials": len(hr),
                "risk_accuracy": round(acc, 4),
                "acc_lo": round(alo, 4), "acc_hi": round(ahi, 4),
                "mean_quality": round(qm, 4),
                "latency_ms": round(lm, 2), "cost_usd": round(cm, 6),
            })

    # ── Per-scene breakdown for yolo_enhanced_only
    print(f"\n── G1v2 YOLO-Enhanced Per-Scene ─────────────────────────────────")
    scenes_seen = sorted(set(r["scene_label"] for r in all_rows))
    for s in scenes_seen:
        sr = [r for r in all_rows
              if r["condition"] == "yolo_enhanced_only" and r["scene_label"] == s]
        if not sr:
            continue
        acc = sum(r["risk_correct"] for r in sr) / len(sr)
        det = sr[0]["detected_risk"]
        truth = sr[0]["truth"]
        clip_l = sr[0]["clip_label"][:35]
        print(f"  {s:20s}  truth={truth:7s}  det={det:8s}  "
              f"acc={acc:.0%}  clip='{clip_l}'")

    summary_csv = RESULTS_DIR / f"G1v2_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["condition", "model", "n_trials", "risk_accuracy",
               "acc_lo", "acc_hi", "mean_quality", "latency_ms", "cost_usd"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[G1v2] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
