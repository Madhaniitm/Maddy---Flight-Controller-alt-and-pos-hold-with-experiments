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
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_saved_frame, call_vision_llm, score_verbalization,
    run_yolo_on_frame, build_llm_prompt,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR
)

N_RUNS     = 5
MODELS     = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MAX_TOKENS = [64, 128, 256, 512]

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


def is_truncated(reply: str) -> int:
    """1 if reply likely cut off mid-sentence (no terminal punctuation)."""
    s = reply.strip()
    if not s:
        return 1
    return int(s[-1] not in ".!?:\"'")


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(MAX_TOKENS) * len(MODELS) * len(SCENES) * N_RUNS
    print("=" * 65)
    print("EXP-V6: Verbosity vs Quality Tradeoff")
    print(f"max_tokens={MAX_TOKENS}")
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
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "s5_pilot_action","detected_risk","word_count","truncated",
              "latency_ms","input_tokens","output_tokens","cost_usd","efficiency","error"]
    runs_csv = RESULTS_DIR / f"V6_runs_{ts}.csv"
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

    summary_csv = RESULTS_DIR / f"V6_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["model","max_tokens","n_trials","accuracy","acc_lo","acc_hi",
               "quality","q_lo","q_hi","word_count","truncation_rate",
               "latency_ms","cost_usd","efficiency"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V6] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
