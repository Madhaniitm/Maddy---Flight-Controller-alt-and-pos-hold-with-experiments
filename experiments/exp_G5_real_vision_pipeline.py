"""
EXP-G5: End-to-End Vision Pipeline Validation
===============================================
Goal:
    Validate the full pipeline on saved real hardware frames:

        frame load → YOLO inference → LLM call → pilot action suggestion

    Each stage is timed independently (Bootstrap CI). The pipeline uses
    the same saved run03 frames from real ESP32-S3-Sense hardware captures.
    No live camera required — frames are representative of real hardware output.

    Stages measured per frame:
        load_ms   : time to read JPEG from disk (simulates camera capture)
        yolo_ms   : YOLOv8n inference time (model loaded once)
        llm_ms    : LLM API round-trip (YOLO metadata + image → reply)
        total_ms  : load + yolo + llm end-to-end

    Models: claude, gpt4o, gpt4o_mini, gemini
    Scenes: all 8 (run03 saved frames)
    N runs: 5

    Total LLM calls: 4 models × 8 scenes × 5 runs = 160 (~20 min)

Output: Image verbalization experiments/results/G5_runs_<ts>.csv
        Image verbalization experiments/results/G5_summary_<ts>.csv
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
    COMBINED_PROMPT_TEMPLATE
)

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]


def main():
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(MODELS) * len(SCENES) * N_RUNS
    print("=" * 65)
    print("EXP-G5: End-to-End Vision Pipeline Validation")
    print(f"Stages: load → YOLO → LLM  |  Models={MODELS}")
    print(f"Scenes={len(SCENES)}  N={N_RUNS}  Total LLM calls={total}")
    print(f"Estimated time: ~{total * 5 // 60} min")
    print("=" * 65)

    print("\nLoading enhanced YOLO tier…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading CLIP scene screener…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows = []

    for scene in SCENES:
        label = scene["label"]
        truth = scene["truth"]
        print(f"\n── Scene: {label}  (truth={truth}) ──")

        for run in range(1, N_RUNS + 1):
            # Stage 1: Load frame (simulates camera capture)
            t0   = time.perf_counter()
            jpeg = get_saved_frame(label)
            load_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # Stage 2: Enhanced YOLO full pipeline (CLAHE + YOLO-World + CLIP)
            t_tier2 = time.perf_counter()
            tier2   = enhanced_yolo_infer(yolo_model, yolo_type,
                                          clip_model, clip_preprocess,
                                          clip_tokenizer, jpeg)
            yolo_ms   = round((time.perf_counter() - t_tier2) * 1000.0, 2)
            yolo_meta = tier2["yolo_meta"]

            # Stage 3: LLM call (all models)
            prompt = COMBINED_PROMPT_TEMPLATE.format(
                yolo_meta  = yolo_meta,
                clip_label = tier2["clip_label"],
                clip_conf  = tier2["clip_conf"],
                clip_risk  = tier2["clip_risk"],
            )

            for model in MODELS:
                res    = call_vision_llm(jpeg, prompt, model=model,
                                         max_tokens=300, temperature=0.0)
                scores = score_verbalization(res["reply"], truth)
                llm_ms = res["latency_ms"]

                all_rows.append({
                    "model":         model,
                    "scene_label":   label,
                    "truth":         truth,
                    "run":           run,
                    "load_ms":       load_ms,
                    "yolo_ms":       yolo_ms,
                    "llm_ms":        llm_ms,
                    "total_ms":      round(load_ms + yolo_ms + llm_ms, 1),
                    "yolo_objects":  0 if "none" in yolo_meta else yolo_meta.count(";") + 1,
                    "yolo_type":     tier2["yolo_type"],
                    "clip_label":    tier2["clip_label"],
                    "clip_risk":     tier2["clip_risk"],
                    "clip_conf":     tier2["clip_conf"],
                    "detected_risk": scores["detected_risk"] or "",
                    "risk_correct":  scores["s3_risk"],
                    "quality_score": scores["quality_score"],
                    "input_tokens":  res["input_tokens"],
                    "output_tokens": res["output_tokens"],
                    "cost_usd":      res["cost_usd"],
                    "reply":         res["reply"],
                    "error":         res["error"][:80] if res["error"] else "",
                })

                status = "✓" if scores["s3_risk"] else "✗"
                print(f"   run={run}  {model:12s}  "
                      f"load={load_ms:.0f}ms  yolo={yolo_ms:.0f}ms  "
                      f"llm={llm_ms:.0f}ms  total={load_ms+yolo_ms+llm_ms:.0f}ms  {status}")

            time.sleep(0.3)

    # ── Save runs
    fields = ["model","scene_label","truth","run",
              "load_ms","yolo_ms","llm_ms","total_ms","yolo_objects",
              "yolo_type","clip_label","clip_risk","clip_conf",
              "detected_risk","risk_correct","quality_score",
              "input_tokens","output_tokens","cost_usd","reply","error"]
    runs_csv = RESULTS_DIR / f"G5_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per model
    print(f"\n── G5 Summary ──────────────────────────────────────────────────")
    print(f"  {'model':12s}  {'load_ms':>8s}  {'yolo_ms':>8s}  "
          f"{'llm_ms':>8s}  {'total_ms':>9s}  {'risk_acc':>8s}")
    print("-" * 68)

    summary_rows = []
    for model in MODELS:
        hr = [r for r in all_rows if r["model"] == model and not r["error"]]
        if not hr:
            continue
        lom, _, _  = bootstrap_ci([r["load_ms"]  for r in hr])
        yom, _, _  = bootstrap_ci([r["yolo_ms"]  for r in hr])
        lmm, lm_lo, lm_hi = bootstrap_ci([r["llm_ms"]   for r in hr])
        tom, _, _  = bootstrap_ci([r["total_ms"] for r in hr])
        acc, alo, ahi = wilson_ci(sum(r["risk_correct"] for r in hr), len(hr))

        print(f"  {model:12s}  {lom:>8.1f}  {yom:>8.1f}  "
              f"{lmm:>8.0f}  {tom:>9.0f}  "
              f"{acc:.3f}[{alo:.3f},{ahi:.3f}]")

        summary_rows.append({
            "model": model, "n_trials": len(hr),
            "load_ms": round(lom, 2), "yolo_ms": round(yom, 2),
            "llm_ms": round(lmm, 1), "llm_lo": round(lm_lo, 1),
            "llm_hi": round(lm_hi, 1), "total_ms": round(tom, 1),
            "risk_accuracy": round(acc, 4), "acc_lo": round(alo, 4),
            "acc_hi": round(ahi, 4),
        })

    # YOLO timing (shared across all models)
    yolo_all = list({r["yolo_ms"] for r in all_rows})  # unique per scene×run
    yom_all, yo_lo, yo_hi = bootstrap_ci([r["yolo_ms"] for r in all_rows])
    print(f"\n  YOLO inference (all scenes, N={len(SCENES)*N_RUNS}): "
          f"{yom_all:.1f}ms [{yo_lo:.1f},{yo_hi:.1f}]")

    summary_csv = RESULTS_DIR / f"G5_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["model","n_trials","load_ms","yolo_ms",
               "llm_ms","llm_lo","llm_hi","total_ms",
               "risk_accuracy","acc_lo","acc_hi"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[G5] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
