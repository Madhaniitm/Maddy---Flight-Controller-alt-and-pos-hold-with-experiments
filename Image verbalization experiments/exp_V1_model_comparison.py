"""
EXP-V1: Multi-Model Vision Comparison (Enhanced YOLO-World Pipeline)
=====================================================================
Goal:
    Compare Claude, GPT-4o, GPT-4o-mini, and Gemini on the same
    8 canonical scenes × N=5 runs = 160 trials total.

    Each model receives an identical JPEG + YOLO-World + CLIP metadata.
    Pipeline: CLAHE → YOLO-World (17 hazard classes) → CLIP scene screen
              → COMBINED_PROMPT_TEMPLATE → LLM

    Metrics per model:
        - classification_accuracy : {safe/caution/hazard} vs ground truth (Wilson CI)
        - quality_score           : 0-5 rubric (Bootstrap CI)
        - latency_ms              : end-to-end API call time (Bootstrap CI)
        - cost_usd                : per-call API cost (Bootstrap CI)
        - word_count              : verbalization length (Bootstrap CI)

Output: results/V1_runs_<ts>.csv, results/V1_summary_<ts>.csv
"""

import sys, time, pathlib, datetime
import numpy as np
import cv2

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = pathlib.Path(__file__).parent
EXP_DIR   = REPO_ROOT / "experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    SCENES, get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_clip, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer, COMBINED_PROMPT_TEMPLATE_CLIP
)
from robust_local_detector import load_mediapipe_detector, detect_hazard

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 60)
    print("EXP-V1: Multi-Model Vision Comparison (YOLO-World + CLIP)")
    print(f"Models={MODELS}  Scenes={len(SCENES)}  N={N_RUNS}")
    print(f"Total LLM calls: {len(MODELS) * len(SCENES) * N_RUNS}")
    print("=" * 60)

    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5 — local emergency detector)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print("Loading CLIP scene screener (V-series — retained for thesis evidence)…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")

        for run in range(1, N_RUNS + 1):
            jpeg     = get_saved_frame(scene["label"])
            img_bgr  = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            t_local  = time.perf_counter()
            local_r  = detect_hazard(img_bgr, depth_map=None,
                                     mp_detector=mp_detector, mp_type=mp_type)
            local_ms = round((time.perf_counter() - t_local) * 1000.0, 2)

            tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                        clip_model, clip_preprocess,
                                        clip_tokenizer, jpeg,
                                        coco_model=coco_model,
                                        depth_pipe=depth_pipe)
            # Append local detector result so LLM sees MediaPipe person detection
            # even when YOLO-World misclassifies person as wall/structural object
            tier2["yolo_meta"] += (
                f"\n  Local detector (Tier 1.5 — MediaPipe EfficientDet-Lite0, advisory — image overrides): "
                f"{local_r['metadata']}"
            )
            prompt = COMBINED_PROMPT_TEMPLATE_CLIP.format(
                yolo_meta  = tier2["yolo_meta"],
                clip_label = tier2["clip_label"],
                clip_conf  = tier2["clip_conf"],
                clip_risk  = tier2["clip_risk"],
            )
            print(f"   run={run}  local={local_ms:.0f}ms({local_r['risk']})  yolo={tier2['yolo_meta'][:60]}")

            for model in MODELS:
                res    = call_vision_llm(jpeg, prompt, model=model,
                                         max_tokens=300, temperature=0.0)
                scores = score_verbalization(res["reply"], scene["truth"])

                row = {
                    "scene_id":        scene["id"],
                    "scene_label":     scene["label"],
                    "truth":           scene["truth"],
                    "model":           model,
                    "run":             run,
                    "local_ms":        local_ms,
                    "local_risk":      local_r["risk"],
                    "quality_score":   scores["quality_score"],
                    "s1_scene":        scores["s1_scene"],
                    "s2_proximity":    scores["s2_proximity"],
                    "s3_risk":         scores["s3_risk"],
                    "s4_length":       scores["s4_length"],
                    "s5_pilot_action": scores["s5_pilot_action"],
                    "detected_risk":   scores["detected_risk"] or "",
                    "detected_action": scores["detected_action"] or "",
                    "word_count":      scores["word_count"],
                    "yolo_meta":       tier2["yolo_meta"],
                    "clip_label":      tier2["clip_label"],
                    "clip_risk":       tier2["clip_risk"],
                    "latency_ms":      res["latency_ms"],
                    "input_tokens":    res["input_tokens"],
                    "output_tokens":   res["output_tokens"],
                    "cost_usd":        res["cost_usd"],
                    "error":           res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)
                print(f"     {model:12s}  quality={scores['quality_score']}/5  "
                      f"risk={scores['detected_risk'] or '?':8s}  "
                      f"lat={res['latency_ms']:.0f}ms  ${res['cost_usd']:.5f}")

            time.sleep(0.3)

    # ── Save runs CSV
    fields = ["scene_id","scene_label","truth","model","run",
              "local_ms","local_risk",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "s5_pilot_action","detected_risk","detected_action","word_count",
              "yolo_meta","clip_label","clip_risk",
              "latency_ms","input_tokens","output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / f"V1_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per model
    print(f"\n── V1 Summary ──────────────────────────────────────────────")
    print(f"  {'model':12s}  {'acc':>6s}  [lo,  hi ]  {'quality':>7s}  {'lat_ms':>7s}  {'$/call':>8s}")
    print("-" * 68)
    summary_rows = []
    for model in MODELS:
        mr = [r for r in all_rows if r["model"] == model and not r["error"]]
        if not mr:
            print(f"  {model:12s} — no data"); continue
        acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in mr), len(mr))
        qm,  qlo, qhi = bootstrap_ci([r["quality_score"] for r in mr])
        lm,  llo, lhi = bootstrap_ci([r["latency_ms"]    for r in mr])
        cm,  clo, chi = bootstrap_ci([r["cost_usd"]      for r in mr])
        wm,  _,   _   = bootstrap_ci([r["word_count"]    for r in mr])
        print(f"  {model:12s}  {acc:.3f}  [{alo:.3f},{ahi:.3f}]  "
              f"{qm:.2f}/5  {lm:.0f}ms  ${cm:.5f}  words={wm:.0f}")
        summary_rows.append({
            "model": model, "n_trials": len(mr),
            "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
            "quality": qm, "q_lo": qlo, "q_hi": qhi,
            "latency_ms": lm, "lat_lo": llo, "lat_hi": lhi,
            "cost_usd": cm, "cost_lo": clo, "cost_hi": chi,
            "word_count": wm,
        })

    summary_csv = RESULTS_DIR / f"V1_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["model","n_trials","accuracy","acc_lo","acc_hi",
               "quality","q_lo","q_hi","latency_ms","lat_lo","lat_hi",
               "cost_usd","cost_lo","cost_hi","word_count"])

    # ── Per-scene breakdown
    print(f"\n── V1 Per-scene accuracy ───────────────────────────────────")
    print(f"  {'scene':20s}  truth   " + "".join(f"  {m:>12s}" for m in MODELS))
    for scene in SCENES:
        row_str = f"  {scene['label']:20s}  {scene['truth']:7s}"
        for model in MODELS:
            mr = [r for r in all_rows
                  if r["scene_label"] == scene["label"] and r["model"] == model]
            acc = round(sum(r["s3_risk"] for r in mr) / len(mr), 2) if mr else float("nan")
            row_str += f"  {acc:.2f}        "
        print(row_str)

    print(f"\nData    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V1] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
