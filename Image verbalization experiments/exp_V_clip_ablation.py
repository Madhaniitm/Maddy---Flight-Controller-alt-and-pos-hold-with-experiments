"""
EXP-V_CLIP_ABLATION: CLIP Necessity Ablation
==============================================
Goal:
    Prove that CLIP metadata in the LLM prompt adds no measurable
    improvement to risk classification accuracy.

    Two conditions on IDENTICAL frames, IDENTICAL YOLO+DA v2 metadata:
        with_clip : COMBINED_PROMPT_TEMPLATE_CLIP  (CLIP label + conf + risk in prompt)
        no_clip   : COMBINED_PROMPT_TEMPLATE        (YOLO only, no CLIP mention)

    CLIP is still computed for both conditions — only whether it appears
    in the LLM prompt differs. This isolates the effect of CLIP *information*
    on LLM performance, not the effect of the YOLO pipeline.

    If accuracy(no_clip) ≈ accuracy(with_clip) → CLIP adds no value.
    If accuracy(no_clip) > accuracy(with_clip) → CLIP actually hurts (noise/distraction).

    Additionally records CLIP's standalone accuracy and latency overhead.

    CLIP latency: ~73 ms overhead (23% of Tier 2 pipeline).
    Skipping CLIP in no_clip condition saves that overhead entirely.

Models : claude, gpt4o, gpt4o_mini, gemini
Scenes : all 8 canonical scenes
N runs : 5
Trials : 2 conditions × 4 models × 8 scenes × 5 runs = 320 LLM calls (~40 min)

Output : results/V_clip_ablation_runs_<ts>.csv
         results/V_clip_ablation_summary_<ts>.csv
"""

import sys, pathlib, datetime, time
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
    enhanced_yolo_infer,
    COMBINED_PROMPT_TEMPLATE,       # no CLIP — production prompt
    COMBINED_PROMPT_TEMPLATE_CLIP,  # with CLIP — V-series prompt
)
from robust_local_detector import load_mediapipe_detector, detect_hazard

N_RUNS     = 5
MODELS     = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
CONDITIONS = ["with_clip", "no_clip"]


def main():
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(CONDITIONS) * len(MODELS) * len(SCENES) * N_RUNS
    print("=" * 65)
    print("EXP-V_CLIP_ABLATION: CLIP Necessity Ablation")
    print(f"Conditions: {CONDITIONS}")
    print(f"Models={MODELS}  Scenes={len(SCENES)}  N={N_RUNS}")
    print(f"Total LLM calls: {total}  (~40 min)")
    print("=" * 65)

    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print("Loading CLIP (runs every frame — only prompt inclusion varies)…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows  = []
    clip_rows = []   # CLIP standalone accuracy (one row per scene × run)

    for scene in SCENES:
        label = scene["label"]
        truth = scene["truth"]
        print(f"\n── Scene: {label}  (truth={truth}) ──")

        for run in range(1, N_RUNS + 1):
            # ── Tier 1.5: local emergency detector
            jpeg     = get_saved_frame(label)
            img_bgr  = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            t_local  = time.perf_counter()
            local_r  = detect_hazard(img_bgr, depth_map=None,
                                     mp_detector=mp_detector, mp_type=mp_type)
            local_ms = round((time.perf_counter() - t_local) * 1000.0, 2)

            # ── Condition A (with_clip): Tier 2 WITH CLIP — run and time it
            t_with = time.perf_counter()
            tier2_clip = enhanced_yolo_infer(yolo_model, yolo_type,
                                             clip_model, clip_preprocess,
                                             clip_tokenizer, jpeg,
                                             coco_model=coco_model,
                                             depth_pipe=depth_pipe,
                                             use_clip=True)
            yolo_ms_with = round((time.perf_counter() - t_with) * 1000.0, 1)

            # Append local detector so LLM sees MediaPipe person detection
            # even when YOLO-World misclassifies person as wall/structural object
            local_meta_str = (
                f"\n  Local detector (Tier 1.5 — MediaPipe EfficientDet-Lite0, advisory — image overrides): "
                f"{local_r['metadata']}"
            )
            yolo_meta  = tier2_clip["yolo_meta"] + local_meta_str
            clip_label = tier2_clip["clip_label"]
            clip_conf  = tier2_clip["clip_conf"]
            clip_risk  = tier2_clip["clip_risk"]

            # ── Condition B (no_clip): Tier 2 WITHOUT CLIP — truly skip CLIP inference
            t_no = time.perf_counter()
            tier2_noclip = enhanced_yolo_infer(yolo_model, yolo_type,
                                               None, None, None, jpeg,
                                               coco_model=coco_model,
                                               depth_pipe=depth_pipe,
                                               use_clip=False)
            yolo_ms_no = round((time.perf_counter() - t_no) * 1000.0, 1)
            # yolo_meta is identical — only CLIP part differs

            # ── CLIP standalone accuracy (no LLM)
            clip_correct = int(clip_risk == truth)
            clip_rows.append({
                "scene_label":  label,
                "truth":        truth,
                "run":          run,
                "clip_label":   clip_label,
                "clip_conf":    clip_conf,
                "clip_risk":    clip_risk,
                "clip_correct": clip_correct,
                "yolo_ms_with_clip": yolo_ms_with,
                "yolo_ms_no_clip":   yolo_ms_no,
                "clip_overhead_ms":  round(yolo_ms_with - yolo_ms_no, 1),
            })

            # ── Build both prompts (same YOLO metadata, differ only in CLIP section)
            prompt_with = COMBINED_PROMPT_TEMPLATE_CLIP.format(
                yolo_meta  = yolo_meta,
                clip_label = clip_label,
                clip_conf  = clip_conf,
                clip_risk  = clip_risk,
            )
            prompt_no = COMBINED_PROMPT_TEMPLATE.format(
                yolo_meta  = yolo_meta,
            )

            clip_overhead = yolo_ms_with - yolo_ms_no
            print(f"   run={run}  local={local_ms:.0f}ms  "
                  f"clip={clip_label}→{clip_risk}({'✓' if clip_correct else '✗'})  "
                  f"clip_overhead={clip_overhead:.0f}ms")

            # ── Tier 3: LLM — both conditions, all models
            for condition, prompt, yolo_ms in [
                    ("with_clip", prompt_with, yolo_ms_with),
                    ("no_clip",   prompt_no,   yolo_ms_no)]:
                for model in MODELS:
                    res    = call_vision_llm(jpeg, prompt, model=model,
                                             max_tokens=300, temperature=0.0)
                    scores = score_verbalization(res["reply"], truth)

                    all_rows.append({
                        "condition":     condition,
                        "model":         model,
                        "scene_label":   label,
                        "truth":         truth,
                        "run":           run,
                        "local_ms":      local_ms,
                        "local_risk":    local_r["risk"],
                        "yolo_ms":       yolo_ms,
                        "clip_label":    clip_label,
                        "clip_conf":     clip_conf,
                        "clip_risk":     clip_risk,
                        "clip_correct":  clip_correct,
                        "yolo_meta":     yolo_meta,
                        "risk_correct":  scores["s3_risk"],
                        "detected_risk": scores["detected_risk"] or "",
                        "quality_score": scores["quality_score"],
                        "reply":         res["reply"],
                        "latency_ms":    res["latency_ms"],
                        "input_tokens":  res["input_tokens"],
                        "output_tokens": res["output_tokens"],
                        "cost_usd":      res["cost_usd"],
                        "error":         res["error"][:80] if res["error"] else "",
                    })

                    status = "✓" if scores["s3_risk"] else "✗"
                    print(f"     [{condition:9s}]  {model:12s}  "
                          f"risk={scores['detected_risk'] or '?':8s}  "
                          f"lat={res['latency_ms']:.0f}ms  {status}")

                time.sleep(0.2)   # between conditions

        time.sleep(0.3)   # between scenes

    # ── Save runs CSV
    run_fields = ["condition","model","scene_label","truth","run",
                  "local_ms","local_risk","yolo_ms",
                  "clip_label","clip_conf","clip_risk","clip_correct",
                  "yolo_meta","risk_correct","detected_risk","quality_score","reply",
                  "latency_ms","input_tokens","output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / f"V_clip_ablation_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, run_fields)

    # ── CLIP standalone summary
    clip_acc = sum(r["clip_correct"] for r in clip_rows) / len(clip_rows)
    mean_overhead = np.mean([r["clip_overhead_ms"] for r in clip_rows])
    print(f"\n── CLIP Standalone Accuracy & Latency ──────────────────────────")
    print(f"  Overall accuracy: {sum(r['clip_correct'] for r in clip_rows)}/{len(clip_rows)} "
          f"= {clip_acc:.3f}  (random baseline 3-class = 0.333)")
    print(f"  CLIP latency overhead: {mean_overhead:.1f} ms per frame "
          f"({mean_overhead / np.mean([r['yolo_ms_with_clip'] for r in clip_rows]) * 100:.1f}% "
          f"of Tier 2 pipeline)")
    print(f"\n  {'scene':22s}  {'truth':8s}  {'clip_risk':12s}  {'acc':>6s}")
    print("  " + "-" * 56)
    for scene in SCENES:
        sr = [r for r in clip_rows if r["scene_label"] == scene["label"]]
        if not sr: continue
        mode_risk = max(set(r["clip_risk"] for r in sr),
                        key=lambda x: sum(1 for r in sr if r["clip_risk"] == x))
        acc = sum(r["clip_correct"] for r in sr) / len(sr)
        flag = "← SAFETY MISS" if scene["truth"] == "hazard" and acc < 1.0 else ""
        print(f"  {scene['label']:22s}  {scene['truth']:8s}  {mode_risk:12s}  "
              f"{acc:.2f}  {flag}")

    # ── Main ablation summary
    print(f"\n── Ablation: with_clip vs no_clip ──────────────────────────────")
    print(f"  {'condition':12s}  {'model':12s}  {'accuracy':>8s}  {'ci':>18s}  "
          f"{'quality':>8s}  {'delta_vs_clip':>14s}")
    print("  " + "-" * 80)

    summary_rows = []
    acc_with = {}
    for condition in CONDITIONS:
        for model in MODELS:
            hr = [r for r in all_rows
                  if r["condition"] == condition and r["model"] == model
                  and not r["error"]]
            if not hr:
                continue
            acc, alo, ahi = wilson_ci(sum(r["risk_correct"] for r in hr), len(hr))
            qm,  _,   _   = bootstrap_ci([r["quality_score"] for r in hr])

            if condition == "with_clip":
                acc_with[model] = acc
                delta_str = "baseline"
            else:
                delta = acc - acc_with.get(model, acc)
                sign  = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta:+.3f} {'↑ CLIP hurts' if delta > 0.02 else '↓ CLIP helps' if delta < -0.02 else '≈ no effect'}"

            print(f"  {condition:12s}  {model:12s}  {acc:.3f}     "
                  f"[{alo:.3f}, {ahi:.3f}]  {qm:.2f}/5  {delta_str}")

            summary_rows.append({
                "condition": condition, "model": model, "n_trials": len(hr),
                "accuracy": round(acc, 4), "acc_lo": round(alo, 4),
                "acc_hi": round(ahi, 4), "quality": round(qm, 3),
            })

    # ── Per-scene breakdown: with vs without
    print(f"\n── Per-scene accuracy: with_clip vs no_clip (all models avg) ──")
    print(f"  {'scene':22s}  {'truth':8s}  {'clip':>6s}  {'with_clip':>9s}  "
          f"{'no_clip':>7s}  {'delta':>7s}")
    print("  " + "-" * 72)
    for scene in SCENES:
        label = scene["label"]
        truth = scene["truth"]
        cr = [r for r in clip_rows if r["scene_label"] == label]
        clip_a = sum(r["clip_correct"] for r in cr) / len(cr) if cr else float("nan")
        wa = [r for r in all_rows if r["scene_label"] == label
              and r["condition"] == "with_clip" and not r["error"]]
        na = [r for r in all_rows if r["scene_label"] == label
              and r["condition"] == "no_clip"  and not r["error"]]
        w_acc = sum(r["risk_correct"] for r in wa) / len(wa) if wa else float("nan")
        n_acc = sum(r["risk_correct"] for r in na) / len(na) if na else float("nan")
        delta = n_acc - w_acc
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:22s}  {truth:8s}  {clip_a:.2f}  {w_acc:9.2f}  "
              f"{n_acc:7.2f}  {sign}{delta:.2f}")

    summary_csv = RESULTS_DIR / f"V_clip_ablation_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["condition","model","n_trials","accuracy","acc_lo","acc_hi","quality"])

    # Save CLIP standalone too
    clip_csv = RESULTS_DIR / f"V_clip_ablation_clip_standalone_{ts}.csv"
    write_csv(clip_csv, clip_rows,
              ["scene_label","truth","run","clip_label","clip_conf","clip_risk",
               "clip_correct","yolo_ms_with_clip","yolo_ms_no_clip","clip_overhead_ms"])

    print(f"\nRuns     → {runs_csv}")
    print(f"Summary  → {summary_csv}")
    print(f"CLIP     → {clip_csv}")
    print(f"[V_CLIP_ABLATION] Done — {len(all_rows)} LLM trials + "
          f"{len(clip_rows)} CLIP-only observations.")


if __name__ == "__main__":
    main()
