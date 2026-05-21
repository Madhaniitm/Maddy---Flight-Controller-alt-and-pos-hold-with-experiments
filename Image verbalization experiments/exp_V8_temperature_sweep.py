"""
EXP-V8: Temperature Sweep
==========================
Goal:
    Measure how LLM temperature affects classification accuracy and
    output consistency in the drone vision pipeline. Justifies the
    temperature=0.2 setting used across all production pipeline calls.

    Temperature [0.0, 0.2, 0.5, 0.8, 1.0] × 4 models × 8 scenes × N=3 runs.
    One frame captured per run — all model × temperature combos see
    the identical JPEG, isolating temperature/model effects from frame variance.

    Prompt: V1 production prompt (structured drone copilot role + YOLO ref).

    Hypothesis:
      Low temperature (≤0.2): highest accuracy, lowest label-flip rate.
      High temperature (0.8–1.0): lower accuracy, higher variance, more
      hallucination of objects not present.

Metrics:
    - classification_accuracy  (Wilson CI) vs temperature
    - label_flip_rate          fraction of run-pairs where risk label changes
    - quality_score /5         (Bootstrap CI) vs temperature
    - latency_ms, cost_usd     (Bootstrap CI)

Models:  claude, gpt4o, gpt4o_mini, gemini
N runs:  3   → 5 × 4 × 8 × 3 = 480 trials total (~30 minutes)

Output:  results/V8_runs_<timestamp>.csv
         results/V8_summary_<timestamp>.csv
"""

import sys, os, time, pathlib, datetime, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_saved_frame, call_vision_llm, score_verbalization,
    run_yolo_on_frame, build_llm_prompt,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR
)

N_RUNS       = 5
MODELS       = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0]
MAX_TOKENS   = 256

TASK_PROMPT = (
    "You are an AI copilot for a drone. "
    "The YOLO metadata above shows what the onboard detector found in this frame.\n"
    "Using both the metadata and the image:\n"
    "1. Describe what you see (1-2 sentences).\n"
    "2. Estimate proximity of any object or person to the camera.\n"
    "3. Classify the scene as exactly one of: safe | caution | hazard\n\n"
    "Format your response as:\n"
    "Description: <text>\n"
    "Proximity: <estimate>\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <PROCEED|SLOW_DOWN|STOP|LAND|HOLD>"
)


def label_flip_rate(risk_labels: list[str]) -> float:
    """Fraction of consecutive run-pairs where risk label changes."""
    if len(risk_labels) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(risk_labels, risk_labels[1:]) if a != b)
    return round(changes / (len(risk_labels) - 1), 4)


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(TEMPERATURES) * len(MODELS) * len(SCENES) * N_RUNS
    print("=" * 65)
    print("EXP-V8: Temperature Sweep")
    print(f"Temperatures={TEMPERATURES}")
    print(f"Models={MODELS}  Scenes={len(SCENES)}  N={N_RUNS}")
    print(f"Total trials: {total}")
    print("=" * 65)

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")
        jpeg      = get_saved_frame(scene["label"])
        yolo_meta = run_yolo_on_frame(jpeg)
        prompt    = build_llm_prompt(yolo_meta, TASK_PROMPT)
        print(f"   frame={len(jpeg)}B  yolo={yolo_meta[:60]}")

        for run in range(1, N_RUNS + 1):

            for model in MODELS:
                for temp in TEMPERATURES:
                    res    = call_vision_llm(jpeg, prompt, model=model,
                                             max_tokens=MAX_TOKENS, temperature=temp)
                    scores = score_verbalization(res["reply"], scene["truth"])
                    s3     = scores["s3_risk"]

                    row = {
                        "scene_id":        scene["id"],
                        "scene_label":     scene["label"],
                        "truth":           scene["truth"],
                        "model":           model,
                        "temperature":     temp,
                        "run":             run,
                        "quality_score":   scores["quality_score"],
                        "s1_scene":        scores["s1_scene"],
                        "s2_proximity":    scores["s2_proximity"],
                        "s3_risk":         s3,
                        "s4_length":       scores["s4_length"],
                        "s5_pilot_action": scores["s5_pilot_action"],
                        "detected_risk":   scores["detected_risk"] or "",
                        "word_count":      scores["word_count"],
                        "latency_ms":      res["latency_ms"],
                        "input_tokens":    res["input_tokens"],
                        "output_tokens":   res["output_tokens"],
                        "cost_usd":        res["cost_usd"],
                        "error":           res["error"][:80] if res["error"] else "",
                    }
                    all_rows.append(row)
                    print(f"     {model:12s}  t={temp:.1f}  "
                          f"q={scores['quality_score']}/5  "
                          f"risk={scores['detected_risk'] or '?':8s}  "
                          f"lat={res['latency_ms']:.0f}ms")

    # ── Save runs CSV
    fields = ["scene_id","scene_label","truth","model","temperature","run",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "s5_pilot_action","detected_risk","word_count",
              "latency_ms","input_tokens","output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / f"V8_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary by temperature (all models avg)
    print(f"\n── V8 Summary by Temperature (all models) ─────────────────────────")
    print(f"  {'temp':5s}  {'acc':>6s}  [lo,  hi ]  {'quality':>7s}  "
          f"{'flip%':>6s}  {'lat_ms':>7s}  {'$/call':>8s}")
    print("-" * 72)

    summary_rows = []
    for temp in TEMPERATURES:
        tr = [r for r in all_rows if r["temperature"] == temp and not r["error"]]
        if not tr:
            continue

        acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
        qm,  _,   _   = bootstrap_ci([r["quality_score"] for r in tr])
        lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
        cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])

        # Label flip rate: per scene × model group across runs
        flip_rates = []
        for scene in SCENES:
            for model in MODELS:
                labels = [r["detected_risk"] for r in tr
                          if r["scene_label"] == scene["label"]
                          and r["model"] == model and r["detected_risk"]]
                if labels:
                    flip_rates.append(label_flip_rate(labels))
        flip_mean = float(np.mean(flip_rates)) if flip_rates else 0.0

        print(f"  {temp:5.1f}  {acc:.3f}  [{alo:.3f},{ahi:.3f}]  "
              f"{qm:.2f}/5  {flip_mean*100:5.1f}%  {lm:.0f}ms  ${cm:.5f}")

        summary_rows.append({
            "model": "all", "temperature": temp, "n_trials": len(tr),
            "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
            "quality": qm, "label_flip_rate": round(flip_mean, 4),
            "latency_ms": lm, "cost_usd": cm,
        })

    # ── Summary by model × temperature
    print(f"\n── V8 Summary by Model × Temperature ──────────────────────────────")
    print(f"  {'model':12s}  {'temp':5s}  {'acc':>6s}  {'quality':>7s}  "
          f"{'flip%':>6s}  {'lat_ms':>7s}")
    for model in MODELS:
        for temp in TEMPERATURES:
            tr = [r for r in all_rows
                  if r["model"] == model and r["temperature"] == temp and not r["error"]]
            if not tr:
                continue
            acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
            qm,  _,   _   = bootstrap_ci([r["quality_score"] for r in tr])
            lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
            cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])

            flip_rates = []
            for scene in SCENES:
                labels = [r["detected_risk"] for r in tr
                          if r["scene_label"] == scene["label"] and r["detected_risk"]]
                if labels:
                    flip_rates.append(label_flip_rate(labels))
            flip_mean = float(np.mean(flip_rates)) if flip_rates else 0.0

            print(f"  {model:12s}  {temp:5.1f}  {acc:.3f}  {qm:.2f}/5  "
                  f"{flip_mean*100:5.1f}%  {lm:.0f}ms")
            summary_rows.append({
                "model": model, "temperature": temp, "n_trials": len(tr),
                "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                "quality": qm, "label_flip_rate": round(flip_mean, 4),
                "latency_ms": lm, "cost_usd": cm,
            })

    # Recommended temperature: highest accuracy − lowest flip rate (all-model avg)
    all_avg = [r for r in summary_rows if r["model"] == "all"]
    if all_avg:
        best = max(all_avg, key=lambda r: r["accuracy"] - r["label_flip_rate"])
        print(f"\n  Recommended temperature: {best['temperature']:.1f}  "
              f"(acc={best['accuracy']:.3f}, flip={best['label_flip_rate']*100:.1f}%)")

    summary_csv = RESULTS_DIR / f"V8_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["model","temperature","n_trials","accuracy","acc_lo","acc_hi",
               "quality","label_flip_rate","latency_ms","cost_usd"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V8] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
