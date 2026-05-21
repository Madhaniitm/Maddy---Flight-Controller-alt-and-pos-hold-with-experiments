"""
EXP-V2: Prompt Technique Comparison
=====================================
Goal:
    Compare 5 prompting strategies across all 4 models on the same
    8 canonical scenes × N=5 runs = 800 trials total.

    Techniques:
        zero_shot   : bare instruction, no examples, no structure
        few_shot_3  : 3 worked examples prepended before the question
        cot         : explicit step-by-step reasoning chain
        structured  : request JSON output with defined fields
        react       : Reason → Observe → Act loop

    Each scene is captured once per run; all models × techniques
    see the identical JPEG and YOLO metadata.

Metrics:
    - classification_accuracy   Wilson CI
    - quality_score (0-5 rubric) Bootstrap CI
    - input_tokens / latency_ms  Bootstrap CI
    - cost_usd                   Bootstrap CI

Output: results/V2_runs_<timestamp>.csv, results/V2_summary_<timestamp>.csv
"""

import sys, os, time, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization, extract_json_risk,
    run_yolo_on_frame, build_llm_prompt,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

# ── Prompt definitions ────────────────────────────────────────────────────────
PROMPTS = {

"zero_shot": (
    "Describe what you see in this drone camera image. "
    "Is there anything dangerous or blocking the path?\n"
    "End with: Risk: <safe|caution|hazard>"
),

"few_shot_3": (
    "You are a drone camera safety monitor. Here are 3 examples of good responses:\n\n"
    "Example 1:\n"
    "Image: empty corridor\n"
    "Response: The camera shows a clear, empty corridor with no obstacles. "
    "Plenty of space to proceed. Risk: safe\n\n"
    "Example 2:\n"
    "Image: wall 20cm away\n"
    "Response: A concrete wall is very close to the camera, approximately 20cm away. "
    "Immediate collision risk if the drone moves forward. Risk: hazard\n\n"
    "Example 3:\n"
    "Image: dim room with boxes\n"
    "Response: Low-light environment with scattered boxes on the floor. "
    "Reduced visibility increases collision risk. Risk: caution\n\n"
    "Now describe this new image:\n"
    "What do you see? Estimate proximity of objects. Classify as safe | caution | hazard.\n"
    "End with: Risk: <safe|caution|hazard>"
),

"cot": (
    "Look at this drone camera image and think step by step:\n"
    "Step 1: What objects or people are visible in the image?\n"
    "Step 2: How close are they to the camera? Estimate in metres if possible.\n"
    "Step 3: Is the lighting adequate for safe navigation?\n"
    "Step 4: Based on steps 1-3, what is the overall risk level?\n\n"
    "Answer each step, then conclude with: Risk: <safe|caution|hazard>"
),

"structured": (
    "Analyse this drone camera image and respond ONLY with valid JSON in this exact format:\n"
    "{\n"
    '  "objects_visible": ["list", "of", "objects"],\n'
    '  "proximity_estimate": "distance description",\n'
    '  "lighting_quality": "good|dim|dark",\n'
    '  "risk_level": "safe|caution|hazard",\n'
    '  "description": "one sentence summary",\n'
    '  "recommended_action": "PROCEED|SLOW_DOWN|STOP|LAND|HOLD"\n'
    "}\n"
    "Respond with JSON only, no other text."
),

"react": (
    "You are a drone vision agent. Use the Reason-Observe-Act framework:\n\n"
    "REASON: What question do I need to answer? "
    "(Is this scene safe for drone navigation?)\n\n"
    "OBSERVE: Look carefully at the image. Describe exactly what you see — "
    "objects, people, proximity, lighting, obstructions.\n\n"
    "ACT: Based on your observation, classify the scene and state what the drone should do.\n"
    "Final answer must include: Risk: <safe|caution|hazard>"
),
}

TECHNIQUES = list(PROMPTS.keys())

def parse_risk(reply: str, technique: str) -> str | None:
    """Extract risk level from reply, technique-aware."""
    if technique == "structured":
        r = extract_json_risk(reply)
        if r: return r
    low = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if f"risk: {lvl}" in low or f'"risk_level": "{lvl}"' in low:
            return lvl
    for lvl in ("hazard", "caution", "safe"):
        if lvl in low:
            return lvl
    return None

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("="*65)
    print("EXP-V2: Prompt Technique Comparison")
    print(f"Models={MODELS}")
    print(f"Techniques={TECHNIQUES}")
    print(f"Scenes={len(SCENES)}  N={N_RUNS}  → {len(SCENES)*N_RUNS*len(TECHNIQUES)*len(MODELS)} trials")
    print("="*65)

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
            # Capture once — all models × techniques see identical frame
            jpeg     = get_frame(scene["label"])
            yolo_meta = run_yolo_on_frame(jpeg)
            print(f"   run={run}  frame={len(jpeg)}B  yolo={yolo_meta[:60]}")

            for tech, task_prompt in PROMPTS.items():
                prompt = build_llm_prompt(yolo_meta, task_prompt)

                for model in MODELS:
                    res      = call_vision_llm(jpeg, prompt, model=model,
                                               max_tokens=320, temperature=0.2)
                    detected = parse_risk(res["reply"], tech)
                    scores   = score_verbalization(res["reply"], scene["truth"])
                    s3       = int(detected == scene["truth"]) if detected else 0

                    row = {
                        "scene_id":        scene["id"],
                        "scene_label":     scene["label"],
                        "truth":           scene["truth"],
                        "model":           model,
                        "technique":       tech,
                        "run":             run,
                        "quality_score":   scores["s1_scene"] + scores["s2_proximity"] + s3 + scores["s4_length"] + scores["s5_pilot_action"],
                        "s1_scene":        scores["s1_scene"],
                        "s2_proximity":    scores["s2_proximity"],
                        "s3_risk":         s3,
                        "s4_length":       scores["s4_length"],
                        "s5_pilot_action": scores["s5_pilot_action"],
                        "detected_risk":   detected or "",
                        "detected_action": scores["detected_action"] or "",
                        "word_count":      scores["word_count"],
                        "latency_ms":      res["latency_ms"],
                        "input_tokens":    res["input_tokens"],
                        "output_tokens":   res["output_tokens"],
                        "cost_usd":        res["cost_usd"],
                        "error":           res["error"][:80] if res["error"] else "",
                    }
                    all_rows.append(row)
                    print(f"     {tech:12s}  {model:12s}  q={row['quality_score']}/5  "
                          f"risk={detected or '?':8s}  lat={res['latency_ms']:.0f}ms")

            time.sleep(1)

    # ── Save runs CSV
    fields = ["scene_id","scene_label","truth","model","technique","run",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "s5_pilot_action","detected_risk","detected_action","word_count",
              "latency_ms","input_tokens","output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / f"V2_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per technique (averaged across models)
    print(f"\n── V2 Summary by Technique (all models) ───────────────────────")
    print(f"  {'technique':14s}  {'acc':>6s}  [lo, hi ]  {'quality':>7s}  {'lat_ms':>7s}  {'$/call':>8s}")
    summary_rows = []
    for tech in TECHNIQUES:
        tr = [r for r in all_rows if r["technique"] == tech and not r["error"]]
        if not tr: continue
        acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
        qm,  qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
        lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
        cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])
        print(f"  {tech:14s}  {acc:.3f}  [{alo:.3f},{ahi:.3f}]  {qm:.2f}/5  {lm:.0f}ms  ${cm:.5f}")
        summary_rows.append({
            "model": "all", "technique": tech,
            "n_trials": len(tr),
            "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
            "quality":  qm,  "q_lo":   qlo, "q_hi":   qhi,
            "latency_ms": lm, "cost_usd": cm,
        })

    # ── Summary per model × technique
    print(f"\n── V2 Summary by Model × Technique ────────────────────────────")
    print(f"  {'model':12s}  {'technique':14s}  {'acc':>6s}  {'quality':>7s}  {'lat_ms':>7s}")
    for model in MODELS:
        for tech in TECHNIQUES:
            tr = [r for r in all_rows if r["model"]==model and r["technique"]==tech and not r["error"]]
            if not tr: continue
            acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
            qm,  qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
            lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
            cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])
            print(f"  {model:12s}  {tech:14s}  {acc:.3f}  {qm:.2f}/5  {lm:.0f}ms")
            summary_rows.append({
                "model": model, "technique": tech,
                "n_trials": len(tr),
                "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                "quality":  qm,  "q_lo":   qlo, "q_hi":   qhi,
                "latency_ms": lm, "cost_usd": cm,
            })

    summary_csv = RESULTS_DIR / f"V2_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["model","technique","n_trials","accuracy","acc_lo","acc_hi",
               "quality","q_lo","q_hi","latency_ms","cost_usd"])

    print(f"\nData   → {runs_csv}")
    print(f"Summary→ {summary_csv}")

if __name__ == "__main__":
    main()
