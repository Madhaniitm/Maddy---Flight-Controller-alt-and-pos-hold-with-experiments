"""
EXP-V2: Prompt Technique Comparison
=====================================
Goal:
    Compare 5 prompting strategies for drone camera verbalization.
    Fixed model: Claude. Same 10 scenes × N=5 runs = 250 trials.

    Techniques:
        zero_shot   : bare instruction, no examples, no structure
        few_shot_3  : 3 worked examples prepended before the question
        cot         : explicit step-by-step reasoning chain
        structured  : request JSON output with defined fields
        react       : Reason → Observe → Act loop

Metrics:
    - quality_score (0-4 rubric)    Bootstrap CI
    - classification_accuracy       Wilson CI
    - input_tokens / output_tokens  Bootstrap CI
    - latency_ms                    Bootstrap CI
    - cost_usd                      Bootstrap CI
"""

import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization, extract_json_risk,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_RUNS = 5
MODEL  = "claude"

# ── Prompt definitions ────────────────────────────────────────────────────────
PROMPTS = {

"zero_shot": (
    "Describe what you see in this drone camera image. "
    "Is there anything dangerous or blocking the path?"
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
    "What do you see? Estimate proximity of objects. Classify as safe | caution | hazard."
),

"cot": (
    "Look at this drone camera image and think step by step:\n"
    "Step 1: What objects or people are visible in the image?\n"
    "Step 2: How close are they to the camera? Estimate in centimetres if possible.\n"
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
    '  "recommended_action": "proceed|slow_down|stop"\n'
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

def parse_risk_from_technique(reply: str, technique: str) -> str | None:
    """Extract risk level from reply based on prompt technique."""
    if technique == "structured":
        r = extract_json_risk(reply)
        if r: return r
    r = reply.lower()
    for lvl in ("hazard","caution","safe"):
        if f"risk: {lvl}" in r or f'"risk_level": "{lvl}"' in r:
            return lvl
    for lvl in ("hazard","caution","safe"):
        if lvl in r:
            return lvl
    return None

def main():
    print("="*60)
    print("EXP-V2: Prompt Technique Comparison")
    print(f"Model={MODEL}  Techniques={list(PROMPTS)}  N={N_RUNS}")
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
            jpeg = get_frame(scene["label"])
            print(f"   run={run}")

            for tech, prompt in PROMPTS.items():
                res = call_vision_llm(jpeg, prompt, model=MODEL,
                                      max_tokens=320, temperature=0.2)

                # Use technique-aware risk extractor for scoring
                detected = parse_risk_from_technique(res["reply"], tech)
                scores   = score_verbalization(res["reply"], scene["truth"])
                # Override s3 with technique-aware detection
                s3 = int(detected == scene["truth"]) if detected else 0

                row = {
                    "scene_id":    scene["id"],
                    "scene_label": scene["label"],
                    "truth":       scene["truth"],
                    "technique":   tech,
                    "run":         run,
                    "quality_score":  scores["s1_scene"] + scores["s2_proximity"] + s3 + scores["s4_length"],
                    "s1_scene":       scores["s1_scene"],
                    "s2_proximity":   scores["s2_proximity"],
                    "s3_risk":        s3,
                    "s4_length":      scores["s4_length"],
                    "detected_risk":  detected or "",
                    "word_count":     scores["word_count"],
                    "latency_ms":     res["latency_ms"],
                    "input_tokens":   res["input_tokens"],
                    "output_tokens":  res["output_tokens"],
                    "cost_usd":       res["cost_usd"],
                    "error":          res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)
                print(f"     {tech:12s}  quality={row['quality_score']}/4  "
                      f"risk={detected or '?':8s}  "
                      f"in_tok={res['input_tokens']}  "
                      f"lat={res['latency_ms']:.0f}ms")

            time.sleep(2)

    # ── Save
    fields = ["scene_id","scene_label","truth","technique","run",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "detected_risk","word_count","latency_ms","input_tokens",
              "output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / "V2_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per technique
    print(f"\n── V2 Summary ──────────────────────────────────────────────")
    print(f"  {'technique':14s}  acc    quality  lat_ms  in_tok  cost")
    summary_rows = []
    for tech in PROMPTS:
        tr = [r for r in all_rows if r["technique"]==tech and not r["error"]]
        if not tr: continue
        acc,  alo, ahi = wilson_ci(sum(r["s3_risk"]       for r in tr), len(tr))
        qm,   qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
        lm,   _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
        im_,  _,   _   = bootstrap_ci([r["input_tokens"]  for r in tr])
        cm,   _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])
        print(f"  {tech:14s}  {acc:.3f}  {qm:.2f}/4   {lm:.0f}ms  {im_:.0f}   ${cm:.5f}")
        summary_rows.append({
            "technique":   tech,
            "accuracy":    acc,  "acc_lo": alo, "acc_hi": ahi,
            "quality":     qm,   "q_lo":   qlo, "q_hi":   qhi,
            "latency_ms":  lm,
            "input_tokens": im_,
            "cost_usd":    cm,
        })

    summary_csv = RESULTS_DIR / "V2_summary.csv"
    write_csv(summary_csv, summary_rows,
              ["technique","accuracy","acc_lo","acc_hi","quality","q_lo","q_hi",
               "latency_ms","input_tokens","cost_usd"])

    print(f"\nData   → {runs_csv}")
    print(f"Summary→ {summary_csv}")

if __name__ == "__main__":
    main()
