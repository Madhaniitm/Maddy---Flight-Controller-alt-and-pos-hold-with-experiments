"""
EXP-V1: Multi-Model Vision Comparison
=======================================
Goal:
    Compare Claude, GPT-4o, Gemini-Flash, and LLaVA (Ollama) on the same
    10 canonical scenes × N=5 frames each = 200 trials total.

    Each model receives an identical JPEG and identical prompt.
    Metrics per model:
        - classification_accuracy : {safe/caution/hazard} vs ground truth (Wilson CI)
        - quality_score           : 0-4 rubric (Bootstrap CI)
        - latency_ms              : end-to-end API call time (Bootstrap CI)
        - cost_usd                : per-call API cost (Bootstrap CI)
        - word_count              : verbalization length (Bootstrap CI)

Output: results/V1_runs.csv, results/V1_summary.csv
"""

import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, fetch_jpeg, get_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)
import numpy as np

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gemini", "llava"]

PROMPT = (
    "You are an AI safety monitor for a drone camera. "
    "Look at the image and:\n"
    "1. Describe what you see (1-2 sentences).\n"
    "2. Estimate proximity of any object or person to the camera.\n"
    "3. Classify the scene as exactly one of: safe | caution | hazard\n\n"
    "Format your response as:\n"
    "Description: <text>\n"
    "Proximity: <estimate>\n"
    "Risk: <safe|caution|hazard>"
)

def main():
    print("="*60)
    print("EXP-V1: Multi-Model Vision Comparison")
    print(f"Models={MODELS}  Scenes={len(SCENES)}  N={N_RUNS}")
    print("="*60)
    if not preflight():
        ans = input("ESP32 not reachable. Use synthetic frames? [y/N]: ")
        if ans.strip().lower() != "y":
            return

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")
        print(f"   Setup: {scene['setup']}")
        input("   [READY] Press Enter when scene is set up…")

        for run in range(1, N_RUNS+1):
            # Capture once, all models see identical frame
            jpeg = get_frame(scene["label"])
            print(f"   run={run}  frame captured ({len(jpeg)} bytes)")

            for model in MODELS:
                res    = call_vision_llm(jpeg, PROMPT, model=model, max_tokens=200)
                scores = score_verbalization(res["reply"], scene["truth"])

                row = {
                    "scene_id":    scene["id"],
                    "scene_label": scene["label"],
                    "truth":       scene["truth"],
                    "model":       model,
                    "run":         run,
                    "quality_score": scores["quality_score"],
                    "s1_scene":      scores["s1_scene"],
                    "s2_proximity":  scores["s2_proximity"],
                    "s3_risk":       scores["s3_risk"],
                    "s4_length":     scores["s4_length"],
                    "detected_risk": scores["detected_risk"] or "",
                    "word_count":    scores["word_count"],
                    "latency_ms":    res["latency_ms"],
                    "input_tokens":  res["input_tokens"],
                    "output_tokens": res["output_tokens"],
                    "cost_usd":      res["cost_usd"],
                    "error":         res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)
                print(f"     {model:8s}  quality={scores['quality_score']}/4  "
                      f"risk={scores['detected_risk'] or '?':8s}  "
                      f"lat={res['latency_ms']:.0f}ms  "
                      f"${res['cost_usd']:.5f}")

            time.sleep(2)

    # ── Save runs CSV
    fields = ["scene_id","scene_label","truth","model","run",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "detected_risk","word_count","latency_ms","input_tokens",
              "output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / "V1_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per model
    print(f"\n── V1 Summary ──────────────────────────────────────────────")
    summary_rows = []
    for model in MODELS:
        mr = [r for r in all_rows if r["model"] == model and not r["error"]]
        if not mr:
            print(f"  {model:8s} — no data"); continue

        # Accuracy = s3_risk correct
        acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in mr), len(mr))
        qm,  qlo, qhi = bootstrap_ci([r["quality_score"] for r in mr])
        lm,  llo, lhi = bootstrap_ci([r["latency_ms"]    for r in mr])
        cm,  clo, chi = bootstrap_ci([r["cost_usd"]      for r in mr])
        wm,  _,   _   = bootstrap_ci([r["word_count"]    for r in mr])

        print(f"  {model:8s}  acc={acc:.3f}[{alo:.3f},{ahi:.3f}]  "
              f"quality={qm:.2f}  lat={lm:.0f}ms  ${cm:.5f}  words={wm:.0f}")

        summary_rows.append({
            "model": model,
            "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
            "quality": qm,   "q_lo":   qlo, "q_hi":   qhi,
            "latency_ms": lm,"lat_lo": llo, "lat_hi": lhi,
            "cost_usd": cm,  "cost_lo": clo,"cost_hi": chi,
            "word_count": wm,
        })

    summary_csv = RESULTS_DIR / "V1_summary.csv"
    write_csv(summary_csv, summary_rows,
              ["model","accuracy","acc_lo","acc_hi","quality","q_lo","q_hi",
               "latency_ms","lat_lo","lat_hi","cost_usd","cost_lo","cost_hi","word_count"])

    # ── Per-scene accuracy heatmap data
    print(f"\n── V1 Per-scene breakdown ──────────────────────────────────")
    print(f"  {'scene':20s}" + "".join(f"  {m:8s}" for m in MODELS))
    for scene in SCENES:
        row_str = f"  {scene['label']:20s}"
        for model in MODELS:
            mr = [r for r in all_rows
                  if r["scene_label"]==scene["label"] and r["model"]==model]
            acc = round(sum(r["s3_risk"] for r in mr)/len(mr), 2) if mr else float("nan")
            row_str += f"  {acc:.2f}    "
        print(row_str)

    print(f"\nData   → {runs_csv}")
    print(f"Summary→ {summary_csv}")

if __name__ == "__main__":
    main()
