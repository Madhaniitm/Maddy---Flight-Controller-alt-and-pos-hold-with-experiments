"""
EXP-G1v2 Patch — Fill missing/wrong scenes (blocked_lens, person_far, object_table)
=====================================================================================
Issues found in 093832 run:
  - blocked_lens : 40/40 LLM calls errored (network failure, last scene in batch)
  - person_far   : 31/40 LLM calls errored + truth was wrong (safe → caution)
  - object_table : truth was wrong (safe → caution; laptop close-up fills frame)

This script:
  1. Loads the 5 clean scenes from the existing 093832 runs CSV
  2. Drops all 3 patch scenes (stale truth labels / network errors)
  3. Runs fresh trials for only those 3 scenes with corrected truth labels
  4. Merges and saves a clean combined runs + summary CSV

Truth corrections:
  object_table : safe   → caution (laptop close-up at ~0.5m, not safe)
  person_far   : safe   → caution (cluttered industrial lab in all runs)
  blocked_lens : hazard → hazard  (correct, just re-running after network fix)

Run: /opt/homebrew/bin/python3.11 experiments/exp_G1v2_patch_missing.py
"""

import sys, csv, pathlib, datetime, time
import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
EXP_DIR   = pathlib.Path(__file__).parent
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR, SCENES
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_clip, enhanced_yolo_infer,
    enhanced_rule_risk, LLM_ONLY_PROMPT, COMBINED_PROMPT_TEMPLATE
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_RUNS_CSV  = RESULTS_DIR / "G1v2_runs_20260521_093832.csv"
PATCH_SCENES   = {"blocked_lens", "person_far", "object_table"}  # 3 scenes re-run
N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

FIELDS = [
    "condition", "model", "scene_label", "truth", "run",
    "yolo_type", "yolo_meta", "clip_label", "clip_risk", "clip_conf",
    "detected_risk", "risk_correct", "quality_score",
    "s1_scene", "s2_proximity", "s3_risk", "s4_length", "s5_pilot_action",
    "word_count", "latency_ms", "input_tokens", "output_tokens",
    "cost_usd", "reply", "error",
]


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load good rows from existing CSV (exclude patch scenes entirely)
    print(f"Loading base data from {BASE_RUNS_CSV.name} …")
    good_rows = []
    with open(BASE_RUNS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scene_label"] not in PATCH_SCENES:
                good_rows.append(row)
    print(f"  Kept {len(good_rows)} rows from 6 good scenes.")

    # ── Load models
    print("\nLoading enhanced YOLO tier…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading CLIP scene screener…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    patch_rows = []
    patch_scenes = [s for s in SCENES if s["label"] in PATCH_SCENES]

    for scene in patch_scenes:
        label = scene["label"]
        truth = scene["truth"]
        print(f"\n── Scene: {label}  (truth={truth}) ──")

        jpeg = get_saved_frame(label)

        # yolo_enhanced_only
        tier2     = enhanced_yolo_infer(yolo_model, yolo_type,
                                        clip_model, clip_preprocess, clip_tokenizer, jpeg)
        yolo_risk = enhanced_rule_risk(tier2["yolo_meta"], tier2["clip_risk"])
        correct   = int(yolo_risk == truth)

        for run in range(1, N_RUNS + 1):
            t2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                     clip_model, clip_preprocess, clip_tokenizer, jpeg)
            patch_rows.append({
                "condition": "yolo_enhanced_only", "model": "yolo_enhanced",
                "scene_label": label, "truth": truth, "run": run,
                "yolo_type": t2["yolo_type"], "yolo_meta": t2["yolo_meta"],
                "clip_label": t2["clip_label"], "clip_risk": t2["clip_risk"],
                "clip_conf": t2["clip_conf"],
                "detected_risk": yolo_risk, "risk_correct": correct,
                "quality_score": correct,
                "s1_scene": 0, "s2_proximity": 0, "s3_risk": correct,
                "s4_length": 0, "s5_pilot_action": 0, "word_count": 0,
                "latency_ms": t2["yolo_ms"],
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "reply": "", "error": "",
            })

        print(f"   yolo_enhanced_only: {yolo_risk}  {'✓' if correct else '✗'}  "
              f"clip={tier2['clip_label'][:30]}({tier2['clip_risk']})")

        # llm_only and combined_enhanced
        for condition in ("llm_only", "combined_enhanced"):
            prompt = LLM_ONLY_PROMPT if condition == "llm_only" else (
                COMBINED_PROMPT_TEMPLATE.format(
                    yolo_meta  = tier2["yolo_meta"],
                    clip_label = tier2["clip_label"],
                    clip_conf  = tier2["clip_conf"],
                    clip_risk  = tier2["clip_risk"],
                )
            )

            for run in range(1, N_RUNS + 1):
                for model in MODELS:
                    res    = call_vision_llm(jpeg, prompt, model=model,
                                             max_tokens=300, temperature=0.0)
                    scores = score_verbalization(res["reply"], truth)

                    patch_rows.append({
                        "condition": condition, "model": model,
                        "scene_label": label, "truth": truth, "run": run,
                        "yolo_type":  tier2["yolo_type"] if condition == "combined_enhanced" else "",
                        "yolo_meta":  tier2["yolo_meta"] if condition == "combined_enhanced" else "",
                        "clip_label": tier2["clip_label"] if condition == "combined_enhanced" else "",
                        "clip_risk":  tier2["clip_risk"] if condition == "combined_enhanced" else "",
                        "clip_conf":  tier2["clip_conf"] if condition == "combined_enhanced" else 0.0,
                        "detected_risk": scores["detected_risk"] or "",
                        "risk_correct": scores["s3_risk"],
                        "quality_score": scores["quality_score"],
                        "s1_scene": scores["s1_scene"], "s2_proximity": scores["s2_proximity"],
                        "s3_risk": scores["s3_risk"], "s4_length": scores["s4_length"],
                        "s5_pilot_action": scores["s5_pilot_action"],
                        "word_count": scores["word_count"],
                        "latency_ms": res["latency_ms"],
                        "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                        "cost_usd": res["cost_usd"], "reply": res["reply"],
                        "error": res["error"][:80] if res["error"] else "",
                    })

                    status = "✓" if scores["s3_risk"] else "✗"
                    print(f"   {condition:20s}  {model:12s}  run={run}  "
                          f"det={scores['detected_risk'] or '?':8s}  "
                          f"{status}  {res['latency_ms']:.0f}ms")
                time.sleep(0.3)

    # ── Merge and save
    # Convert good_rows (dicts with string values) to match types
    all_rows = good_rows + patch_rows
    runs_csv = RESULTS_DIR / f"G1v2_runs_patched_{ts}.csv"
    write_csv(runs_csv, all_rows, FIELDS)
    print(f"\nMerged runs → {runs_csv}  ({len(all_rows)} total rows)")

    # ── Recompute summary
    CONDITIONS = ["yolo_enhanced_only", "llm_only", "combined_enhanced"]
    print(f"\n── G1v2 Patched Summary ──────────────────────────────────────────")
    print(f"  {'condition':22s}  {'model':14s}  {'n':>3s}  {'risk_acc':>8s}  "
          f"{'quality':>7s}  {'lat_ms':>8s}  {'cost':>9s}")
    print("-" * 82)

    summary_rows = []
    for cond in CONDITIONS:
        models_for = ["yolo_enhanced"] if cond == "yolo_enhanced_only" else MODELS
        for model in models_for:
            hr = [r for r in all_rows
                  if r["condition"] == cond and r["model"] == model
                  and not r.get("error", "")]
            if not hr:
                continue
            risk_vals = [int(r["risk_correct"]) for r in hr]
            qual_vals = [float(r["quality_score"]) for r in hr]
            lat_vals  = [float(r["latency_ms"]) for r in hr]
            cost_vals = [float(r["cost_usd"]) for r in hr]
            acc, alo, ahi = wilson_ci(sum(risk_vals), len(risk_vals))
            qm, _, _  = bootstrap_ci(qual_vals)
            lm, _, _  = bootstrap_ci(lat_vals)
            cm, _, _  = bootstrap_ci(cost_vals)
            print(f"  {cond:22s}  {model:14s}  {len(hr):>3d}  "
                  f"{acc:.3f}[{alo:.3f},{ahi:.3f}]  "
                  f"{qm:.2f}/5    {lm:.0f}ms  ${cm:.5f}")
            summary_rows.append({
                "condition": cond, "model": model, "n_trials": len(hr),
                "risk_accuracy": round(acc, 4),
                "acc_lo": round(alo, 4), "acc_hi": round(ahi, 4),
                "mean_quality": round(qm, 4),
                "latency_ms": round(lm, 2), "cost_usd": round(cm, 6),
            })

    # Per-scene breakdown
    print(f"\n── Per-scene (yolo_enhanced_only) ───────────────────────────────")
    for scene in SCENES:
        s = scene["label"]
        sr = [r for r in all_rows if r["condition"] == "yolo_enhanced_only"
              and r["scene_label"] == s]
        if not sr:
            continue
        acc = sum(int(r["risk_correct"]) for r in sr) / len(sr)
        print(f"  {s:20s}  truth={sr[0]['truth']:7s}  "
              f"det={sr[0]['detected_risk']:8s}  acc={acc:.0%}")

    summary_csv = RESULTS_DIR / f"G1v2_summary_patched_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["condition", "model", "n_trials", "risk_accuracy",
               "acc_lo", "acc_hi", "mean_quality", "latency_ms", "cost_usd"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[G1v2-patch] Done — {len(patch_rows)} new trials added.")


if __name__ == "__main__":
    main()
