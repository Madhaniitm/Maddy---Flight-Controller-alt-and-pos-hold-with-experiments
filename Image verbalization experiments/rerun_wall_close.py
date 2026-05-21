"""
Rerun wall_close scene for V1 — appends 20 fresh rows to V1_runs.csv.
Run this with ESP32 connected and wall_close scene set up.
"""
import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    get_frame, call_vision_llm, score_verbalization,
    run_yolo_on_frame, build_llm_prompt, write_csv, preflight, RESULTS_DIR
)
import csv

MODELS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
N_RUNS = 5
SCENE  = {"id": 2, "label": "wall_close", "truth": "hazard",
           "setup": "Point camera at wall from ~25cm away."}

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

FIELDS = ["scene_id","scene_label","truth","model","run",
          "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
          "s5_pilot_action","detected_risk","detected_action","word_count",
          "latency_ms","input_tokens","output_tokens","cost_usd","error"]

def main():
    print("="*55)
    print("Re-run: wall_close scene (V1)")
    print(f"Models={MODELS}  N={N_RUNS}  → 20 new rows")
    print("="*55)

    if not preflight():
        print("ESP32 not reachable — connect and retry.")
        return

    print(f"\nScene: {SCENE['label']}  (truth={SCENE['truth']})")
    print(f"Setup: {SCENE['setup']}")
    input("[READY] Point camera at wall ~25cm away, then press Enter…")

    all_rows = []
    for run in range(1, N_RUNS + 1):
        jpeg = get_frame(SCENE["label"])
        if len(jpeg) < 3000:
            print(f"  run={run}  WARNING: small frame ({len(jpeg)}B) — may be synthetic")
        else:
            print(f"  run={run}  frame={len(jpeg)}B")

        yolo_meta = run_yolo_on_frame(jpeg)
        prompt    = build_llm_prompt(yolo_meta, TASK_PROMPT)
        print(f"  yolo: {yolo_meta[:70]}")

        for model in MODELS:
            res    = call_vision_llm(jpeg, prompt, model=model, max_tokens=256)
            scores = score_verbalization(res["reply"], SCENE["truth"])
            row = {
                "scene_id":        SCENE["id"],
                "scene_label":     SCENE["label"],
                "truth":           SCENE["truth"],
                "model":           model,
                "run":             run,
                "quality_score":   scores["quality_score"],
                "s1_scene":        scores["s1_scene"],
                "s2_proximity":    scores["s2_proximity"],
                "s3_risk":         scores["s3_risk"],
                "s4_length":       scores["s4_length"],
                "s5_pilot_action": scores["s5_pilot_action"],
                "detected_risk":   scores["detected_risk"] or "",
                "detected_action": scores["detected_action"] or "",
                "word_count":      scores["word_count"],
                "latency_ms":      res["latency_ms"],
                "input_tokens":    res["input_tokens"],
                "output_tokens":   res["output_tokens"],
                "cost_usd":        res["cost_usd"],
                "error":           res["error"][:80] if res["error"] else "",
            }
            all_rows.append(row)
            print(f"    {model:12s}  quality={scores['quality_score']}/5  "
                  f"risk={scores['detected_risk'] or '?':8s}  "
                  f"lat={res['latency_ms']:.0f}ms")
        time.sleep(1)

    # Append to the most recent V1_runs_*.csv
    import glob
    existing = sorted(glob.glob(str(RESULTS_DIR / "V1_runs_*.csv")))
    if existing:
        runs_csv = pathlib.Path(existing[-1])
    else:
        runs_csv = RESULTS_DIR / "V1_runs.csv"
    with open(runs_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writerows(all_rows)

    print(f"\nAppended {len(all_rows)} rows → {runs_csv}")

    # Quick summary
    print("\n── wall_close accuracy ──")
    for model in MODELS:
        mr  = [r for r in all_rows if r["model"] == model]
        acc = sum(r["s3_risk"] for r in mr) / len(mr)
        q   = sum(r["quality_score"] for r in mr) / len(mr)
        print(f"  {model:12s}  acc={acc:.1f}  quality={q:.1f}/5")

if __name__ == "__main__":
    main()
