"""
EXP-V10: Resolution Ablation — 320×240 vs 640×480
===================================================
Goal:
    Test whether upgrading from the native nano-drone resolution
    (320×240, 85–102 tokens/call) to 640×480 improves action
    correctness, narration quality, or scene accuracy.

    The experiment addresses the RAL reviewer question:
        "Is 320×240 genuinely sufficient, or does higher resolution help?"

Method:
    - Take the same run03 real hardware frames used in G5
    - Condition A: native 320×240 (unchanged JPEG)
    - Condition B: bicubic upscale to 640×480, re-encode as JPEG
    - Run the full G5 pipeline (YOLO+DA v2 + LLM) on both
    - Compare: action correctness, scene accuracy, token count, latency

    Models: all four (claude, gpt4o, gpt4o_mini, gemini)
    Runs:   5 per scene per resolution
    Total LLM calls: 4 models × 8 scenes × 5 runs × 2 resolutions = 320

Output:
    results/V10_runs_<ts>.csv
    results/V10_summary_<ts>.csv
"""

import sys, pathlib, datetime, time, io
import numpy as np
import cv2

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
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer, COMBINED_PROMPT_TEMPLATE
)
from robust_local_detector import (  # noqa: E402
    load_mediapipe_detector, detect_hazard
)

N_RUNS     = 5
MODELS     = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
RESOLUTION = {
    "320x240": None,          # no resize — load native JPEG
    "640x480": (640, 480),    # bicubic upscale before API call
}


def upscale_jpeg(jpeg: bytes, target_wh: tuple) -> bytes:
    """Bicubic upscale + re-encode to JPEG, quality=90."""
    arr = np.frombuffer(jpeg, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_up = cv2.resize(img, target_wh, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img_up, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def main():
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(MODELS) * len(SCENES) * N_RUNS * len(RESOLUTION)
    print("=" * 65)
    print("EXP-V10: Resolution Ablation  320×240 vs 640×480")
    print(f"Models={MODELS}  Scenes={len(SCENES)}  N={N_RUNS}")
    print(f"Total LLM calls: {total}  (~{total*5//60} min)")
    print("=" * 65)

    print("\nLoading models (shared across both resolutions)…")
    mp_detector, mp_type = load_mediapipe_detector()
    yolo_model,  yolo_type = load_enhanced_yolo()
    coco_model,  _         = load_coco_yolo()
    depth_pipe,  _         = load_depth_anything()

    all_rows = []

    for res_label, upscale_wh in RESOLUTION.items():
        print(f"\n{'='*20}  Resolution: {res_label}  {'='*20}")

        for scene in SCENES:
            label = scene["label"]
            truth = scene["truth"]
            print(f"\n── Scene: {label}  truth={truth}  res={res_label} ──")

            for run in range(1, N_RUNS + 1):
                # Load native 320×240 frame
                jpeg_native = get_saved_frame(label)

                # Upscale if needed
                if upscale_wh is not None:
                    jpeg = upscale_jpeg(jpeg_native, upscale_wh)
                else:
                    jpeg = jpeg_native

                # Tier 1.5 — MediaPipe on native (resolution-independent: same drone)
                img_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                t_local = time.perf_counter()
                local_result = detect_hazard(img_bgr, depth_map=None,
                                             mp_detector=mp_detector, mp_type=mp_type)
                local_ms = round((time.perf_counter() - t_local) * 1000.0, 2)

                # Tier 2 — YOLO + DAv2 on the (possibly upscaled) image
                t_tier2 = time.perf_counter()
                tier2   = enhanced_yolo_infer(yolo_model, yolo_type,
                                              None, None, None, jpeg,
                                              coco_model=coco_model,
                                              depth_pipe=depth_pipe)
                yolo_ms = round((time.perf_counter() - t_tier2) * 1000.0, 2)

                yolo_meta = (
                    f"{tier2['yolo_meta']}\n"
                    f"  Local detector (advisory): {local_result['metadata']}"
                )
                prompt = COMBINED_PROMPT_TEMPLATE.format(yolo_meta=yolo_meta)

                for model in MODELS:
                    res = call_vision_llm(jpeg, prompt, model=model,
                                         max_tokens=256, temperature=0.0)
                    scores  = score_verbalization(res["reply"], truth)
                    llm_ms  = res["latency_ms"]

                    all_rows.append({
                        "resolution":    res_label,
                        "model":         model,
                        "scene_label":   label,
                        "truth":         truth,
                        "run":           run,
                        "local_ms":      local_ms,
                        "yolo_ms":       yolo_ms,
                        "llm_ms":        llm_ms,
                        "total_ms":      round(local_ms + yolo_ms + llm_ms, 1),
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
                    print(f"  run={run}  {model:12s}  {res_label}  "
                          f"yolo={yolo_ms:.0f}ms  llm={llm_ms:.0f}ms  "
                          f"tok={res['input_tokens']}  {status}")

                time.sleep(0.3)

    # ── Save runs CSV
    fields = ["resolution","model","scene_label","truth","run",
              "local_ms","yolo_ms","llm_ms","total_ms",
              "detected_risk","risk_correct","quality_score",
              "input_tokens","output_tokens","cost_usd","reply","error"]
    runs_csv = RESULTS_DIR / f"V10_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary: per resolution × model
    print(f"\n{'='*65}")
    print("EXP-V10 SUMMARY")
    print(f"{'resolution':12s}  {'model':12s}  {'N':>4s}  "
          f"{'ActCorr':>8s}  {'SceneAcc':>9s}  "
          f"{'Tokens':>7s}  {'LLM_ms':>8s}")
    print("-" * 70)

    summary_rows = []
    for res_label in RESOLUTION:
        for model in MODELS:
            hr = [r for r in all_rows
                  if r["resolution"] == res_label
                  and r["model"] == model
                  and not r["error"]]
            if not hr:
                continue

            n    = len(hr)
            acc,  a_lo, a_hi = wilson_ci(sum(r["risk_correct"] for r in hr), n)
            ltm, lt_lo, lt_hi = bootstrap_ci([r["llm_ms"] for r in hr])
            tok  = np.mean([r["input_tokens"] for r in hr])
            qual = np.mean([r["quality_score"] for r in hr])

            print(f"  {res_label:12s}  {model:12s}  {n:>4d}  "
                  f"{acc:.3f}[{a_lo:.3f},{a_hi:.3f}]  "
                  f"{qual:.2f}/5  {tok:.0f}tok  {ltm:.0f}ms")

            summary_rows.append({
                "resolution":  res_label,
                "model":       model,
                "n_trials":    n,
                "act_corr":    round(acc, 4),
                "acc_lo":      round(a_lo, 4),
                "acc_hi":      round(a_hi, 4),
                "quality":     round(qual, 3),
                "mean_tokens": round(float(tok), 1),
                "llm_ms":      round(ltm, 1),
                "llm_lo":      round(lt_lo, 1),
                "llm_hi":      round(lt_hi, 1),
            })

    summary_csv = RESULTS_DIR / f"V10_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["resolution","model","n_trials",
               "act_corr","acc_lo","acc_hi","quality",
               "mean_tokens","llm_ms","llm_lo","llm_hi"])

    # ── Print delta table (640×480 minus 320×240)
    print(f"\n── Delta table (640×480 − 320×240) ──────────────────────────")
    print(f"{'model':12s}  {'ΔActCorr':>10s}  {'ΔTokens':>9s}  {'ΔLLM_ms':>9s}")
    print("-" * 48)
    for model in MODELS:
        r320 = next((r for r in summary_rows if r["resolution"]=="320x240" and r["model"]==model), None)
        r640 = next((r for r in summary_rows if r["resolution"]=="640x480" and r["model"]==model), None)
        if r320 and r640:
            dAcc = round((r640["act_corr"] - r320["act_corr"]) * 100, 1)
            dTok = round(r640["mean_tokens"] - r320["mean_tokens"], 1)
            dMs  = round(r640["llm_ms"] - r320["llm_ms"], 0)
            print(f"  {model:12s}  {dAcc:+.1f} pp  {dTok:+.0f} tok  {dMs:+.0f} ms")

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V10] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
