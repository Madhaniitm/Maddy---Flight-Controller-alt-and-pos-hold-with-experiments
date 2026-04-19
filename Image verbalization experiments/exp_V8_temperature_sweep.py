"""
EXP-V8: Temperature Sweep — Consistency vs Creativity
=======================================================
Goal:
    Measure how LLM temperature affects verbalization quality, classification
    accuracy, and run-to-run consistency on drone camera scenes.

    temperatures : [0.0, 0.2, 0.5, 0.8, 1.0]
    Fixed        : Claude, zero_shot prompt, max_tokens=200
    Scenes       : 10 canonical scenes × N=5 runs = 250 trials per temperature
                   Total: 1250 trials

    Hypothesis:
      - Low temperature (0.0–0.2): highest classification accuracy, lowest
        run-to-run variance — deterministic/near-deterministic outputs.
      - High temperature (0.8–1.0): lower accuracy, higher creative variance,
        occasional hallucination of objects not present.
      - Quality (0-4 rubric) may plateau or degrade above 0.5.

    The "sweet spot" temperature balances accuracy and quality with acceptable
    variance — relevant for deployment where reproducibility matters.

Metrics:
    - classification_accuracy  : risk level correct (Wilson CI)
    - quality_score (0-4)      : Bootstrap CI per temperature
    - consistency_score        : std dev of quality across N runs per scene
                                 (lower = more consistent) — Bootstrap CI
    - variance_risk            : fraction of scenes where risk label changes
                                 across N runs (higher temp → more label flipping)
    - latency_ms               : Bootstrap CI
    - output_tokens            : Bootstrap CI
    - cost_usd                 : Bootstrap CI

Paper context:
    Extends V6 (max_tokens sweep) to the temperature axis.
    Combined with V9 (model × params), establishes the full parameter
    sensitivity map for VLM-based drone scene verbalization.

References:
    - Yao et al. 2022 (ReAct): temperature=0.2 chosen for agent reliability
    - Vemprala et al. 2023: temperature sensitivity in UAV task completion
    - Brown et al. 2020 (GPT-3): temperature as creativity vs consistency knob
"""

import sys, os, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_RUNS       = 5
MODEL        = "claude"
TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0]
MAX_TOKENS   = 200

PROMPT = (
    "You are a drone camera monitor. "
    "Look at the image and describe what you see. "
    "Estimate proximity of objects. "
    "Classify the scene as exactly one of: safe | caution | hazard"
)


def label_flips(risk_labels: list[str]) -> float:
    """Fraction of consecutive pairs where label changes — measures instability."""
    if len(risk_labels) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(risk_labels, risk_labels[1:]) if a != b)
    return round(changes / (len(risk_labels) - 1), 4)


def main():
    print("=" * 60)
    print("EXP-V8: Temperature Sweep")
    print(f"temperatures={TEMPERATURES}  model={MODEL}  N={N_RUNS}")
    print(f"Total trials: {len(TEMPERATURES)} × {len(SCENES)} × {N_RUNS} = "
          f"{len(TEMPERATURES)*len(SCENES)*N_RUNS}")
    print("=" * 60)

    if not preflight():
        ans = input("ESP32 not reachable. Use synthetic frames? [y/N]: ")
        if ans.strip().lower() != "y":
            return

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")
        print(f"   Setup: {scene['setup']}")
        input("   [READY] Press Enter when scene is set up…")

        for run in range(1, N_RUNS + 1):
            jpeg = get_frame(scene["label"])
            print(f"   run={run}")

            for temp in TEMPERATURES:
                res    = call_vision_llm(jpeg, PROMPT, model=MODEL,
                                        max_tokens=MAX_TOKENS, temperature=temp)
                scores = score_verbalization(res["reply"], scene["truth"])

                row = {
                    "scene_id":      scene["id"],
                    "scene_label":   scene["label"],
                    "truth":         scene["truth"],
                    "temperature":   temp,
                    "run":           run,
                    "quality_score": scores["quality_score"],
                    "s1_scene":      scores["s1_scene"],
                    "s2_proximity":  scores["s2_proximity"],
                    "s3_risk":       scores["s3_risk"],
                    "s4_length":     scores["s4_length"],
                    "detected_risk": scores.get("detected_risk", ""),
                    "word_count":    scores["word_count"],
                    "latency_ms":    res["latency_ms"],
                    "input_tokens":  res["input_tokens"],
                    "output_tokens": res["output_tokens"],
                    "cost_usd":      res["cost_usd"],
                    "error":         res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)
                print(f"     temp={temp:.1f}  quality={scores['quality_score']}/4  "
                      f"risk={scores.get('detected_risk','?'):8s}  "
                      f"lat={res['latency_ms']:.0f}ms  ${res['cost_usd']:.6f}")

            time.sleep(2)

    # ── Save runs
    fields = ["scene_id", "scene_label", "truth", "temperature", "run",
              "quality_score", "s1_scene", "s2_proximity", "s3_risk", "s4_length",
              "detected_risk", "word_count", "latency_ms", "input_tokens",
              "output_tokens", "cost_usd", "error"]
    runs_csv = RESULTS_DIR / "V8_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per temperature
    print(f"\n── V8 Summary ──────────────────────────────────────────────────────")
    print(f"  {'temp':6s}  quality            accuracy  consistency  variance  latency   cost")
    print(f"  {'':6s}  (mean[lo,hi])       (Wilson)  (std↓good)   (flip%)   (ms)      (USD)")
    print(f"  {'-'*85}")

    summary_rows = []
    prev_qm = None
    for temp in TEMPERATURES:
        tr = [r for r in all_rows if r["temperature"] == temp and not r["error"]]
        if not tr:
            continue

        qm, qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
        lm, _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
        cm, clo, chi = bootstrap_ci([r["cost_usd"]       for r in tr])
        ot, _,   _   = bootstrap_ci([r["output_tokens"]  for r in tr])

        # Accuracy
        n_correct = sum(r["s3_risk"] for r in tr)
        acc, alo, ahi = wilson_ci(n_correct, len(tr))

        # Consistency: mean std of quality_score across runs, per scene
        scene_stds = []
        for scene in SCENES:
            sc_rows = [r["quality_score"] for r in tr if r["scene_label"] == scene["label"]]
            if len(sc_rows) >= 2:
                scene_stds.append(float(np.std(sc_rows)))
        cons_mean = float(np.mean(scene_stds)) if scene_stds else 0.0

        # Label flip rate: per scene, how often does risk label change across runs?
        flip_rates = []
        for scene in SCENES:
            labels = [r["detected_risk"] for r in tr
                      if r["scene_label"] == scene["label"] and r["detected_risk"]]
            if labels:
                flip_rates.append(label_flips(labels))
        flip_mean = float(np.mean(flip_rates)) if flip_rates else 0.0

        delta = f"Δq={qm-prev_qm:+.2f}" if prev_qm is not None else "      "
        print(f"  {temp:6.1f}  {qm:.2f}[{qlo:.2f},{qhi:.2f}]  "
              f"  {acc:.3f}[{alo:.3f},{ahi:.3f}]  "
              f"  σ={cons_mean:.3f}    {flip_mean*100:.1f}%    "
              f"  {lm:.0f}ms    ${cm:.6f}  {delta}")
        prev_qm = qm

        summary_rows.append({
            "temperature":      temp,
            "quality_mean":     round(qm, 4),
            "quality_lo":       round(qlo, 4),
            "quality_hi":       round(qhi, 4),
            "accuracy":         round(acc, 4),
            "accuracy_lo":      round(alo, 4),
            "accuracy_hi":      round(ahi, 4),
            "consistency_std":  round(cons_mean, 4),
            "label_flip_rate":  round(flip_mean, 4),
            "latency_ms":       round(lm, 2),
            "output_tokens":    round(ot, 1),
            "cost_usd_mean":    round(cm, 6),
            "cost_usd_lo":      round(clo, 6),
            "cost_usd_hi":      round(chi, 6),
        })

    # Sweet spot: highest accuracy × lowest flip rate
    if summary_rows:
        best = max(summary_rows,
                   key=lambda r: r["accuracy"] - r["label_flip_rate"])
        print(f"\n  Recommended temperature: {best['temperature']:.1f}  "
              f"(acc={best['accuracy']:.3f}, flip={best['label_flip_rate']*100:.1f}%)")

    summary_csv = RESULTS_DIR / "V8_summary.csv"
    write_csv(summary_csv, summary_rows,
              ["temperature", "quality_mean", "quality_lo", "quality_hi",
               "accuracy", "accuracy_lo", "accuracy_hi",
               "consistency_std", "label_flip_rate",
               "latency_ms", "output_tokens",
               "cost_usd_mean", "cost_usd_lo", "cost_usd_hi"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V8] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
