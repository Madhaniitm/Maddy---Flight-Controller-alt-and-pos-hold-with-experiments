"""
EXP-FAILURE: Failure Case Reasoning Analysis
=============================================
Goal:
    Capture the FULL LLM response for the 3 systematic failure scenes
    to understand WHY models misclassify — model prior vs prompt issue
    vs visual perception failure.

    Failure scenes selected from V1/V2 results:
        wall_close  — truth=hazard, all models say safe   (0–40% accuracy)
        door_open   — truth=safe,   all models say hazard  (5% accuracy)
        person_far  — truth=safe,   all models say caution (0–25% accuracy)

    Frames: run03 real hardware captures (best frame per scene).
    Prompt: V1 production prompt (same as V1/V2 baseline).
    N runs: 3 per scene per model (to check consistency of reasoning).

    Full reply saved — not truncated. Enables qualitative analysis of
    what the model observed, what it reasoned, and why it classified wrong.

Models:  claude, gpt4o, gpt4o_mini, gemini
Trials:  3 scenes × 4 models × 3 runs = 36 trials (~5 minutes)

Output:  results/failure_analysis_<timestamp>.csv
"""

import sys, pathlib, datetime, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    get_saved_frame, call_vision_llm, score_verbalization,
    run_yolo_on_frame, build_llm_prompt, write_csv, RESULTS_DIR
)

N_RUNS = 3
MODELS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

FAILURE_SCENES = [
    {"label": "wall_close",  "truth": "hazard",  "failure": "all models say safe — blank wall misread as empty room"},
    {"label": "door_open",   "truth": "safe",    "failure": "all models say hazard/caution — open doorway misread as obstacle"},
    {"label": "person_far",  "truth": "safe",    "failure": "all models say caution — distant person escalated to risk"},
]

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


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(FAILURE_SCENES) * len(MODELS) * N_RUNS
    print("=" * 65)
    print("EXP-FAILURE: Failure Case Reasoning Analysis")
    print(f"Scenes={[s['label'] for s in FAILURE_SCENES]}")
    print(f"Models={MODELS}  N={N_RUNS}")
    print(f"Total trials: {total}  (~5 minutes)")
    print("=" * 65)

    all_rows = []

    for scene in FAILURE_SCENES:
        print(f"\n── Scene: {scene['label']}  (truth={scene['truth']}) ──")
        print(f"   Known failure: {scene['failure']}")

        jpeg      = get_saved_frame(scene["label"])
        yolo_meta = run_yolo_on_frame(jpeg)
        prompt    = build_llm_prompt(yolo_meta, TASK_PROMPT)
        print(f"   frame={len(jpeg)}B  yolo={yolo_meta[:80]}")

        for run in range(1, N_RUNS + 1):
            for model in MODELS:
                res    = call_vision_llm(jpeg, prompt, model=model,
                                         max_tokens=256, temperature=0.0)
                scores = score_verbalization(res["reply"], scene["truth"])
                correct = scores["s3_risk"]

                row = {
                    "scene_label":    scene["label"],
                    "truth":          scene["truth"],
                    "known_failure":  scene["failure"],
                    "model":          model,
                    "run":            run,
                    "detected_risk":  scores["detected_risk"] or "",
                    "correct":        correct,
                    "quality_score":  scores["quality_score"],
                    "s1_scene":       scores["s1_scene"],
                    "s2_proximity":   scores["s2_proximity"],
                    "s3_risk":        scores["s3_risk"],
                    "word_count":     scores["word_count"],
                    "yolo_meta":      yolo_meta,
                    "full_reply":     res["reply"],
                    "latency_ms":     res["latency_ms"],
                    "input_tokens":   res["input_tokens"],
                    "output_tokens":  res["output_tokens"],
                    "cost_usd":       res["cost_usd"],
                    "error":          res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)

                status = "✓ CORRECT" if correct else "✗ WRONG"
                print(f"   {model:12s}  run={run}  "
                      f"detected={scores['detected_risk'] or '?':8s}  "
                      f"{status}  lat={res['latency_ms']:.0f}ms")
                if not correct:
                    # Print first 200 chars of reply to show reasoning live
                    print(f"   >>> {res['reply'][:200].replace(chr(10), ' ')}")

        time.sleep(0.5)

    # ── Save
    fields = ["scene_label","truth","known_failure","model","run",
              "detected_risk","correct","quality_score","s1_scene","s2_proximity",
              "s3_risk","word_count","yolo_meta","full_reply",
              "latency_ms","input_tokens","output_tokens","cost_usd","error"]
    out_csv = RESULTS_DIR / f"failure_analysis_{ts}.csv"
    write_csv(out_csv, all_rows, fields)

    # ── Summary
    print(f"\n── Failure Analysis Summary ────────────────────────────────────")
    print(f"  {'scene':20s}  {'model':12s}  {'acc':>5s}  detected_risks")
    for scene in FAILURE_SCENES:
        for model in MODELS:
            tr = [r for r in all_rows
                  if r["scene_label"]==scene["label"] and r["model"]==model]
            acc  = sum(r["correct"] for r in tr) / len(tr) if tr else 0
            dets = [r["detected_risk"] for r in tr]
            print(f"  {scene['label']:20s}  {model:12s}  {acc:.0%}    {dets}")

    print(f"\n── Full replies saved to: {out_csv}")
    print(f"[FAILURE] Done — {len(all_rows)} trials recorded.")
    print(f"\nOpen the CSV to read the full model reasoning for each failure case.")


if __name__ == "__main__":
    main()
