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
    load_enhanced_yolo, load_clip, enhanced_yolo_infer,
    COMBINED_PROMPT_TEMPLATE
)

N_RUNS       = 5
MODELS       = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0]
MAX_TOKENS   = 300


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

    print("\nLoading enhanced YOLO tier…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading CLIP scene screener…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")
        jpeg  = get_saved_frame(scene["label"])
        tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                    clip_model, clip_preprocess,
                                    clip_tokenizer, jpeg)
        prompt = COMBINED_PROMPT_TEMPLATE.format(
            yolo_meta  = tier2["yolo_meta"],
            clip_label = tier2["clip_label"],
            clip_conf  = tier2["clip_conf"],
            clip_risk  = tier2["clip_risk"],
        )
        print(f"   yolo={tier2['yolo_meta'][:60]}  clip={tier2['clip_label']}")

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
