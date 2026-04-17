"""
EXP-V4: Model × Prompt Technique Interaction
===============================================
Goal:
    Test whether structured/CoT prompting benefits all models equally,
    or whether the gain is model-dependent (interaction effect).

    3 × 3 factorial design:
        Models  : claude, gpt4o, gemini
        Prompts : zero_shot, cot, structured

    N=3 per cell per scene (10 scenes) = 270 trials.

    Each model×prompt combination is compared on quality_score and accuracy.
    Interaction is shown as:  marginal means and delta(structured - zero_shot)
    per model — if delta differs significantly across models, interaction exists.

Metrics:
    - quality_score (0-4) Bootstrap CI per cell
    - accuracy (Wilson CI) per cell
    - delta_cot       = quality(cot)        - quality(zero_shot) per model
    - delta_structured= quality(structured) - quality(zero_shot) per model
    - latency_ms, cost_usd Bootstrap CI
"""

import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization, extract_json_risk,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_RUNS = 3
MODELS = ["claude", "gpt4o", "gemini"]

PROMPTS = {
    "zero_shot": (
        "Describe what you see in this drone camera image. "
        "Is there anything dangerous? Classify: safe | caution | hazard"
    ),
    "cot": (
        "Analyse this drone camera image step by step:\n"
        "Step 1: List all visible objects.\n"
        "Step 2: Estimate how close each object is to the camera.\n"
        "Step 3: Assess the lighting quality.\n"
        "Step 4: Determine the overall risk.\n"
        "Conclude with: Risk: <safe|caution|hazard>"
    ),
    "structured": (
        "Analyse this drone camera image. "
        "Respond ONLY in this JSON format:\n"
        '{"objects": [], "closest_object_cm": 0, '
        '"lighting": "good|dim|dark", "risk_level": "safe|caution|hazard", '
        '"description": "", "action": "proceed|slow|stop"}'
    ),
}

def parse_risk(reply: str, technique: str) -> str | None:
    if technique == "structured":
        r = extract_json_risk(reply)
        if r: return r
    lo = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if f"risk: {lvl}" in lo or f'"risk_level": "{lvl}"' in lo or f"risk_level: {lvl}" in lo:
            return lvl
    for lvl in ("hazard", "caution", "safe"):
        if lvl in lo:
            return lvl
    return None

def main():
    print("="*60)
    print("EXP-V4: Model × Prompt Interaction (3×3 factorial)")
    print(f"Models={MODELS}  Prompts={list(PROMPTS)}  N={N_RUNS}")
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

            for model in MODELS:
                for tech, prompt in PROMPTS.items():
                    res    = call_vision_llm(jpeg, prompt, model=model,
                                            max_tokens=280, temperature=0.2)
                    det    = parse_risk(res["reply"], tech)
                    scores = score_verbalization(res["reply"], scene["truth"])
                    s3     = int(det == scene["truth"]) if det else 0
                    q      = scores["s1_scene"] + scores["s2_proximity"] + s3 + scores["s4_length"]

                    row = {
                        "scene_id":    scene["id"],
                        "scene_label": scene["label"],
                        "truth":       scene["truth"],
                        "model":       model,
                        "technique":   tech,
                        "run":         run,
                        "quality_score": q,
                        "s1_scene":      scores["s1_scene"],
                        "s2_proximity":  scores["s2_proximity"],
                        "s3_risk":       s3,
                        "s4_length":     scores["s4_length"],
                        "detected_risk": det or "",
                        "word_count":    scores["word_count"],
                        "latency_ms":    res["latency_ms"],
                        "input_tokens":  res["input_tokens"],
                        "output_tokens": res["output_tokens"],
                        "cost_usd":      res["cost_usd"],
                        "error":         res["error"][:80] if res["error"] else "",
                    }
                    all_rows.append(row)
                    print(f"     {model:8s} × {tech:12s}  "
                          f"q={q}/4  risk={det or '?':8s}  {res['latency_ms']:.0f}ms")

            time.sleep(2)

    # ── Save
    fields = ["scene_id","scene_label","truth","model","technique","run",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "detected_risk","word_count","latency_ms","input_tokens",
              "output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / "V4_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── 3×3 cell means table
    print(f"\n── V4 Quality Matrix (mean quality_score) ──────────────────")
    header = f"  {'model':8s}" + "".join(f"  {t:14s}" for t in PROMPTS)
    print(header)
    matrix_rows = []
    for model in MODELS:
        row_str = f"  {model:8s}"
        for tech in PROMPTS:
            cell = [r["quality_score"] for r in all_rows
                    if r["model"]==model and r["technique"]==tech and not r["error"]]
            if cell:
                m, lo, hi = bootstrap_ci(cell)
                row_str += f"  {m:.2f}[{lo:.2f},{hi:.2f}]"
            else:
                row_str += "  N/A           "
            matrix_rows.append({
                "model": model, "technique": tech,
                "mean_quality": round(sum(cell)/len(cell),3) if cell else None,
                "n": len(cell),
            })
        print(row_str)

    # ── Interaction: delta(structured) - delta(zero_shot) per model
    print(f"\n── V4 Gain over zero_shot ──────────────────────────────────")
    print(f"  {'model':8s}  Δ(cot-zero)  Δ(struct-zero)")
    for model in MODELS:
        z_scores = [r["quality_score"] for r in all_rows
                    if r["model"]==model and r["technique"]=="zero_shot" and not r["error"]]
        c_scores = [r["quality_score"] for r in all_rows
                    if r["model"]==model and r["technique"]=="cot" and not r["error"]]
        s_scores = [r["quality_score"] for r in all_rows
                    if r["model"]==model and r["technique"]=="structured" and not r["error"]]
        if z_scores and c_scores and s_scores:
            zm = sum(z_scores)/len(z_scores)
            cm = sum(c_scores)/len(c_scores)
            sm = sum(s_scores)/len(s_scores)
            print(f"  {model:8s}  Δcot={cm-zm:+.2f}    Δstruct={sm-zm:+.2f}")

    # ── Accuracy matrix
    print(f"\n── V4 Accuracy Matrix ──────────────────────────────────────")
    print(header)
    for model in MODELS:
        row_str = f"  {model:8s}"
        for tech in PROMPTS:
            cell = [r for r in all_rows
                    if r["model"]==model and r["technique"]==tech and not r["error"]]
            if cell:
                acc,_,_ = wilson_ci(sum(r["s3_risk"] for r in cell), len(cell))
                row_str += f"  acc={acc:.3f}        "
            else:
                row_str += "  N/A           "
        print(row_str)

    matrix_csv = RESULTS_DIR / "V4_matrix.csv"
    write_csv(matrix_csv, matrix_rows, ["model","technique","mean_quality","n"])
    print(f"\nData   → {runs_csv}")
    print(f"Matrix → {matrix_csv}")

if __name__ == "__main__":
    main()
