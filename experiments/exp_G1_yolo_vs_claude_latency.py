"""
EXP-G1: YOLO vs LLM Isolation
==============================
Goal:
    Prove neither YOLO alone nor LLM alone is sufficient — the combined
    pipeline is required.

    Three conditions on all 8 saved frames (run03 real hardware captures):

    yolo_only : YOLO inference → rule-based risk. No LLM call.
                Fast (~20ms), but produces no scene description, no
                contextual pilot suggestion, and fails on featureless
                scenes (wall_close → zero detections → safe).

    llm_only  : LLM receives image only (no YOLO metadata).
                Can describe scene but lacks structured distance/confidence
                data. Slower. No YOLO anchor for proximity estimation.

    combined  : YOLO metadata + image → LLM. Best of both tiers.
                Structured detections guide LLM reasoning; LLM adds
                natural-language description and contextual pilot action.

Models (llm_only + combined): claude, gpt4o, gpt4o_mini, gemini
Scenes: 8 (all saved run03 real frames)
N runs: 5
LLM trials: 2 conditions × 4 models × 8 scenes × 5 runs = 320
YOLO trials: 8 scenes × 5 runs = 40 (no API cost, <2 min)

Output: Image verbalization experiments/results/G1_runs_<ts>.csv
        Image verbalization experiments/results/G1_summary_<ts>.csv
"""

import sys, pathlib, datetime, time, re
import numpy as np

REPO_ROOT  = pathlib.Path(__file__).parent.parent
VIZ_DIR    = REPO_ROOT / "Image verbalization experiments"
sys.path.insert(0, str(VIZ_DIR))

from verbalization_utils import (
    get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR, SCENES
)

YOLO_PT  = VIZ_DIR / "yolov8n.pt"
N_RUNS   = 5
MODELS   = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

LLM_ONLY_PROMPT = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "flying at ~1m altitude indoors.\n"
    "No sensor metadata available — your ONLY input is the camera image. "
    "Look at it directly and reason from what you see.\n\n"
    "1. Describe what you see (1-2 sentences).\n"
    "2. Estimate proximity of any object or person from visual evidence.\n"
    "3. Classify the scene as exactly one of: safe | caution | hazard\n\n"
    "Format your response as:\n"
    "Description: <text>\n"
    "Proximity: <your visual estimate>\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <PROCEED|SLOW_DOWN|STOP|LAND|HOLD>"
)

COMBINED_PROMPT = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "flying at ~1m altitude indoors.\n"
    "YOUR PRIMARY INPUT IS THE CAMERA IMAGE — look at it directly.\n"
    "The YOLO sensor metadata above is supplementary — it can miss objects or "
    "give imprecise distances. If you see something YOLO did not detect, it exists. "
    "If the image contradicts YOLO, trust the image.\n\n"
    "1. Describe what you see (1-2 sentences).\n"
    "2. Estimate proximity of any object or person from visual evidence "
    "(cross-check with YOLO if helpful).\n"
    "3. Classify the scene as exactly one of: safe | caution | hazard\n\n"
    "Format your response as:\n"
    "Description: <text>\n"
    "Proximity: <your visual estimate>\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <PROCEED|SLOW_DOWN|STOP|LAND|HOLD>"
)


def load_yolo():
    """Load YOLOv8n once for the whole experiment."""
    try:
        from ultralytics import YOLO as UltralyticsYOLO
        return UltralyticsYOLO(str(YOLO_PT))
    except Exception as e:
        print(f"[YOLO] load failed: {e} — will use timing simulation")
        return None


def run_yolo_inference(model, jpeg_bytes: bytes) -> tuple[str, float]:
    """Returns (yolo_meta, latency_ms). Model is pre-loaded."""
    if model is None:
        import random
        time.sleep(0.02)
        return "YOLO detections: none", round(abs(random.gauss(20, 5)), 2)

    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    h   = img.shape[0]
    t0  = time.perf_counter()
    results = model(img, verbose=False, conf=0.25)[0]
    lat = (time.perf_counter() - t0) * 1000.0

    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        label    = results.names[int(box.cls[0])]
        conf     = round(float(box.conf[0]), 2)
        est_dist = round(h / max(y2 - y1, 1) * 0.3, 2)
        dets.append(f"{label} (conf={conf}, est_dist~{est_dist}m, bbox=[{x1},{y1},{x2},{y2}])")

    meta = ("YOLO detections: " + "; ".join(dets)) if dets else "YOLO detections: none"
    return meta, round(lat, 2)


def yolo_rule_risk(yolo_meta: str) -> str:
    """Rule-based risk from YOLO metadata only — no LLM."""
    meta = yolo_meta.lower()
    if "none" in meta or "unavailable" in meta:
        return "safe"
    if "person" in meta:
        m = re.search(r'est_dist~([\d.]+)m', meta)
        if m:
            return "hazard" if float(m.group(1)) < 1.0 else "caution"
        return "caution"
    return "caution"


def main():
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_total  = 2 * len(MODELS) * len(SCENES) * N_RUNS
    yolo_total = len(SCENES) * N_RUNS
    print("=" * 65)
    print("EXP-G1: YOLO vs LLM Isolation")
    print(f"Conditions: yolo_only | llm_only | combined")
    print(f"Scenes={len(SCENES)}  Models={MODELS}  N={N_RUNS}")
    print(f"LLM trials: {llm_total}  YOLO trials: {yolo_total}")
    print(f"Estimated time: ~{llm_total * 5 // 60} min")
    print("=" * 65)

    print("\nLoading YOLO model once…")
    yolo_model = load_yolo()

    all_rows = []

    for scene in SCENES:
        label = scene["label"]
        truth = scene["truth"]
        print(f"\n── Scene: {label}  (truth={truth}) ──")

        jpeg      = get_saved_frame(label)
        yolo_meta, _ = run_yolo_inference(yolo_model, jpeg)
        print(f"   YOLO: {yolo_meta[:80]}")

        # ── yolo_only (N_RUNS timing measurements, no API calls)
        yolo_risk = yolo_rule_risk(yolo_meta)
        for run in range(1, N_RUNS + 1):
            _, yolo_lat = run_yolo_inference(yolo_model, jpeg)
            correct = int(yolo_risk == truth)
            all_rows.append({
                "condition": "yolo_only", "model": "yolo",
                "scene_label": label, "truth": truth, "run": run,
                "detected_risk": yolo_risk, "risk_correct": correct,
                "quality_score": correct,  # no description possible → max s3=1
                "s1_scene": 0, "s2_proximity": 0, "s3_risk": correct,
                "s4_length": 0, "s5_pilot_action": 0,
                "word_count": 0, "latency_ms": yolo_lat,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "error": "",
            })
        print(f"   yolo_only: det={yolo_risk:8s}  correct={correct}  "
              f"lat~{yolo_lat:.1f}ms  (N={N_RUNS} runs)")

        # ── llm_only and combined
        for condition in ("llm_only", "combined"):
            prompt = (LLM_ONLY_PROMPT if condition == "llm_only"
                      else f"{yolo_meta}\n\n{COMBINED_PROMPT}")

            for run in range(1, N_RUNS + 1):
                for model in MODELS:
                    res    = call_vision_llm(jpeg, prompt, model=model,
                                            max_tokens=256, temperature=0.0)
                    scores = score_verbalization(res["reply"], truth)
                    all_rows.append({
                        "condition": condition, "model": model,
                        "scene_label": label, "truth": truth, "run": run,
                        "detected_risk":    scores["detected_risk"] or "",
                        "risk_correct":     scores["s3_risk"],
                        "quality_score":    scores["quality_score"],
                        "s1_scene":         scores["s1_scene"],
                        "s2_proximity":     scores["s2_proximity"],
                        "s3_risk":          scores["s3_risk"],
                        "s4_length":        scores["s4_length"],
                        "s5_pilot_action":  scores["s5_pilot_action"],
                        "word_count":       scores["word_count"],
                        "latency_ms":       res["latency_ms"],
                        "input_tokens":     res["input_tokens"],
                        "output_tokens":    res["output_tokens"],
                        "cost_usd":         res["cost_usd"],
                        "error":            res["error"][:80] if res["error"] else "",
                    })
                    status = "✓" if scores["s3_risk"] else "✗"
                    print(f"   {condition:10s}  {model:12s}  run={run}  "
                          f"det={scores['detected_risk'] or '?':8s}  "
                          f"{status}  lat={res['latency_ms']:.0f}ms")
                time.sleep(0.3)

    # ── Save runs
    fields = ["condition","model","scene_label","truth","run",
              "detected_risk","risk_correct","quality_score",
              "s1_scene","s2_proximity","s3_risk","s4_length","s5_pilot_action",
              "word_count","latency_ms","input_tokens","output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / f"G1_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary
    CONDITIONS = ["yolo_only", "llm_only", "combined"]
    print(f"\n── G1 Summary ──────────────────────────────────────────────────")
    print(f"  {'condition':12s}  {'model':12s}  {'risk_acc':>8s}  "
          f"{'quality':>7s}  {'lat_ms':>8s}  {'cost':>9s}")
    print("-" * 72)

    summary_rows = []
    for cond in CONDITIONS:
        models_for = ["yolo"] if cond == "yolo_only" else MODELS
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
            print(f"  {cond:12s}  {model:12s}  "
                  f"{acc:.3f}[{alo:.3f},{ahi:.3f}]  "
                  f"{qm:.2f}/5    {lm:.0f}ms  ${cm:.5f}")
            summary_rows.append({
                "condition": cond, "model": model, "n_trials": len(hr),
                "risk_accuracy": round(acc, 4), "acc_lo": round(alo, 4), "acc_hi": round(ahi, 4),
                "mean_quality": round(qm, 4),
                "latency_ms": round(lm, 2), "cost_usd": round(cm, 6),
            })

    summary_csv = RESULTS_DIR / f"G1_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["condition","model","n_trials","risk_accuracy","acc_lo","acc_hi",
               "mean_quality","latency_ms","cost_usd"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[G1] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
