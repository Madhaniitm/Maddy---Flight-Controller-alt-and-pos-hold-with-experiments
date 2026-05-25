"""
EXP-V6: Verbosity vs Quality Tradeoff
=======================================
Goal:
    How much does increasing max_tokens improve verbalization quality?
    Does it plateau, and at what cost?

    max_tokens levels : [64, 128, 256, 512]
    All 4 models × 8 scenes × N=5 runs = 160 trials per token level
    Total: 4 × 4 × 8 × 5 = 640 trials

    One saved frame per scene (run03, real ESP32-S3 Sense hardware).
    All model × token-level combos see the identical JPEG per scene.

    Hypothesis: Quality improves up to 256 tokens then plateaus, while
    cost increases linearly. The sweet spot balances quality and cost.

Metrics:
    - quality_score /5     Bootstrap CI per level
    - truncation_rate      reply ends mid-sentence (Wilson CI)
    - word_count           Bootstrap CI
    - efficiency           quality_score / cost_usd (quality per dollar)
    - latency_ms, cost_usd Bootstrap CI

Models:  claude, gpt4o, gpt4o_mini, gemini
N runs:  5   → 4 × 4 × 8 × 5 = 640 trials total

Output:  results/V6_runs_<timestamp>.csv
         results/V6_summary_<timestamp>.csv
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
    enhanced_yolo_infer, COMBINED_PROMPT_TEMPLATE_CLIP, COMBINED_PROMPT_TEMPLATE
)

# ── Toggle: set to False to run WITHOUT CLIP (ablation) ──────────────────────
USE_CLIP = False   # True → with CLIP (results: V6_runs_20260525_185304.csv)
                   # False → without CLIP (current run)
from robust_local_detector import load_mediapipe_detector, detect_hazard

N_RUNS     = 5
MODELS     = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MAX_TOKENS = [64, 128, 256, 512]


def is_truncated(reply: str) -> int:
    """1 if reply likely cut off mid-sentence (no terminal punctuation)."""
    s = reply.strip()
    if not s:
        return 1
    return int(s[-1] not in ".!?:\"'")


def main():
    clip_tag = "clip" if USE_CLIP else "noclip"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(MAX_TOKENS) * len(MODELS) * len(SCENES) * N_RUNS
    print("=" * 65)
    print(f"EXP-V6: Verbosity vs Quality Tradeoff  [{clip_tag}]")
    print(f"max_tokens={MAX_TOKENS}")
    print(f"Models={MODELS}  Scenes={len(SCENES)}  N={N_RUNS}")
    print(f"Total trials: {total}")
    print("=" * 65)

    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5 — local emergency detector)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    if USE_CLIP:
        print("Loading CLIP scene screener…")
        clip_model, clip_preprocess, clip_tokenizer = load_clip()
    else:
        clip_model = clip_preprocess = clip_tokenizer = None
        print("CLIP disabled (USE_CLIP=False)")

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")
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
        tier2["yolo_meta"] += (
            f"\n  Local detector (Tier 1.5 — MediaPipe EfficientDet-Lite0, advisory — image overrides): "
            f"{local_r['metadata']}"
        )
        if USE_CLIP:
            prompt = COMBINED_PROMPT_TEMPLATE_CLIP.format(
                yolo_meta  = tier2["yolo_meta"],
                clip_label = tier2["clip_label"],
                clip_conf  = tier2["clip_conf"],
                clip_risk  = tier2["clip_risk"],
            )
            print(f"   local={local_ms:.0f}ms({local_r['risk']})  yolo={tier2['yolo_meta'][:50]}  clip={tier2['clip_label']}")
        else:
            prompt = COMBINED_PROMPT_TEMPLATE.format(yolo_meta=tier2["yolo_meta"])
            print(f"   local={local_ms:.0f}ms({local_r['risk']})  yolo={tier2['yolo_meta'][:50]}  (no CLIP)")

        for run in range(1, N_RUNS + 1):
            for model in MODELS:
                for max_tok in MAX_TOKENS:
                    res    = call_vision_llm(jpeg, prompt, model=model,
                                             max_tokens=max_tok, temperature=0.0)
                    scores = score_verbalization(res["reply"], scene["truth"])
                    trunc  = is_truncated(res["reply"])
                    eff    = round(scores["quality_score"] / max(res["cost_usd"], 1e-8), 2)

                    row = {
                        "scene_id":        scene["id"],
                        "scene_label":     scene["label"],
                        "truth":           scene["truth"],
                        "model":           model,
                        "max_tokens":      max_tok,
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
                        "word_count":      scores["word_count"],
                        "truncated":       trunc,
                        "latency_ms":      res["latency_ms"],
                        "input_tokens":    res["input_tokens"],
                        "output_tokens":   res["output_tokens"],
                        "cost_usd":        res["cost_usd"],
                        "efficiency":      eff,
                        "error":           res["error"][:80] if res["error"] else "",
                    }
                    all_rows.append(row)
                    print(f"     {model:12s}  tok={max_tok:3d}  "
                          f"q={scores['quality_score']}/5  "
                          f"words={scores['word_count']:3d}  "
                          f"trunc={trunc}  lat={res['latency_ms']:.0f}ms")

            time.sleep(1)

    # ── Save runs CSV
    fields = ["scene_id","scene_label","truth","model","max_tokens","run",
              "local_ms","local_risk",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "s5_pilot_action","detected_risk","word_count","truncated",
              "latency_ms","input_tokens","output_tokens","cost_usd","efficiency","error"]
    runs_csv = RESULTS_DIR / f"V6_runs_{clip_tag}_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary by max_tokens (all models avg)
    print(f"\n── V6 Summary by Token Budget (all models) ────────────────────")
    print(f"  {'max_tok':7s}  {'acc':>6s}  {'quality':>7s}  {'words':>5s}  "
          f"{'trunc%':>6s}  {'lat_ms':>7s}  {'$/call':>8s}  {'q/USD':>7s}")
    print("-" * 75)

    summary_rows = []
    prev_qm = None
    for max_tok in MAX_TOKENS:
        tr = [r for r in all_rows if r["max_tokens"] == max_tok and not r["error"]]
        if not tr:
            continue
        acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
        qm,  qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
        wm,  _,   _   = bootstrap_ci([r["word_count"]    for r in tr])
        lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
        cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])
        em,  _,   _   = bootstrap_ci([r["efficiency"]    for r in tr if r["efficiency"] > 0])
        tr_r, _,  _   = wilson_ci(sum(r["truncated"] for r in tr), len(tr))

        delta = f"  Δq={qm-prev_qm:+.2f}" if prev_qm is not None else ""
        print(f"  {max_tok:7d}  {acc:.3f}  {qm:.2f}/5  {wm:.0f}w  "
              f"{tr_r*100:5.1f}%  {lm:.0f}ms  ${cm:.5f}  {em:.0f}{delta}")
        prev_qm = qm

        summary_rows.append({
            "model": "all", "max_tokens": max_tok, "n_trials": len(tr),
            "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
            "quality": qm, "q_lo": qlo, "q_hi": qhi,
            "word_count": wm, "truncation_rate": tr_r,
            "latency_ms": lm, "cost_usd": cm, "efficiency": em,
        })

    # ── Summary by model × max_tokens
    print(f"\n── V6 Summary by Model × Token Budget ─────────────────────────")
    print(f"  {'model':12s}  {'tok':>5s}  {'acc':>6s}  {'quality':>7s}  "
          f"{'words':>5s}  {'trunc%':>6s}  {'lat_ms':>7s}")
    for model in MODELS:
        for max_tok in MAX_TOKENS:
            tr = [r for r in all_rows
                  if r["model"] == model and r["max_tokens"] == max_tok and not r["error"]]
            if not tr:
                continue
            acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
            qm,  _,   _   = bootstrap_ci([r["quality_score"] for r in tr])
            wm,  _,   _   = bootstrap_ci([r["word_count"]    for r in tr])
            lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
            cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])
            tr_r, _,  _   = wilson_ci(sum(r["truncated"] for r in tr), len(tr))

            print(f"  {model:12s}  {max_tok:5d}  {acc:.3f}  {qm:.2f}/5  "
                  f"{wm:.0f}w  {tr_r*100:5.1f}%  {lm:.0f}ms")
            summary_rows.append({
                "model": model, "max_tokens": max_tok, "n_trials": len(tr),
                "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                "quality": qm, "q_lo": 0, "q_hi": 0,
                "word_count": wm, "truncation_rate": tr_r,
                "latency_ms": lm, "cost_usd": cm, "efficiency": 0,
            })

    # Sweet spot: best efficiency (all-model avg rows only)
    all_avg = [r for r in summary_rows if r["model"] == "all"]
    if all_avg:
        sweet = max(all_avg, key=lambda r: r["efficiency"])
        print(f"\n  Efficiency sweet spot: max_tokens={sweet['max_tokens']}  "
              f"(quality={sweet['quality']:.2f}/5, q/USD={sweet['efficiency']:.0f})")

    summary_csv = RESULTS_DIR / f"V6_summary_{clip_tag}_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["model","max_tokens","n_trials","accuracy","acc_lo","acc_hi",
               "quality","q_lo","q_hi","word_count","truncation_rate",
               "latency_ms","cost_usd","efficiency"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V6] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
