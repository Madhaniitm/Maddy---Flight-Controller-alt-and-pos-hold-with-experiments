"""
EXP-V9: Model × Temperature × Max-Tokens Full Factorial
=========================================================
Goal:
    Quantify the joint effect of model identity, inference temperature, and
    token budget on verbalization quality, classification accuracy, and cost
    for drone camera scenes.

    3 × 3 × 3 full factorial design:
        Models      : claude, gpt4o, gemini
        Temperatures: [0.0, 0.5, 1.0]
        Max-tokens  : [128, 256, 512]

    N=3 per cell per scene × 10 scenes = 810 trials total.

    Hypotheses:
      H1 (main effect — model):  Claude > GPT-4o > Gemini for structured scene
           description quality, replicating V2 main-effect result.
      H2 (main effect — temp):   Temperature 0.0 maximises accuracy;
           1.0 maximises variance and risk of hallucination.
      H3 (main effect — tokens): Quality plateaus above 256 tokens (V6 finding);
           cost scales linearly with output tokens.
      H4 (interaction temp×model): Temperature sensitivity may differ across
           model families — frontier models more robust to high temp than
           smaller / open-weights models.
      H5 (interaction tokens×model): Larger models may extract more benefit
           from a larger token budget (better at self-editing / density).

Metrics per cell (model × temp × max_tokens):
    - quality_score (0-4)      : Bootstrap CI
    - accuracy                 : Wilson CI
    - latency_ms               : Bootstrap CI
    - output_tokens            : Bootstrap CI
    - cost_usd                 : Bootstrap CI
    - efficiency               : quality_score / cost_usd  (Bootstrap CI)

Interaction terms reported:
    - Δ_temp(0.0→1.0)  per model        (quality delta across temperature range)
    - Δ_tokens(128→512) per model       (quality delta across token range)
    - Δ_model(claude−gemini) per temp   (model gap as a function of temperature)

Paper context:
    Extends V8 (temperature sweep, Claude only) to the full 3-model space.
    Provides the parameter-sensitivity map that V4 (model×prompt) and V6
    (max_tokens sweep) introduced separately but never combined.
    Combined result: "model choice dominates temperature and token budget as
    a quality driver", or its negation.

References:
    - Brown et al. 2020 (GPT-3): temperature as creativity vs consistency knob
    - Yao et al. 2022 (ReAct): temperature=0.2 chosen for agent reliability
    - Vemprala et al. 2023: multi-model UAV task-completion comparison
    - Ouyang et al. 2022 (InstructGPT): response length vs quality in RLHF
"""

import sys, os, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_RUNS       = 3
MODELS       = ["claude", "gpt4o", "gemini"]
TEMPERATURES = [0.0, 0.5, 1.0]
MAX_TOKENS_L = [128, 256, 512]

PROMPT = (
    "You are a drone camera monitor. "
    "Look at the image and describe what you see. "
    "Estimate proximity of objects. "
    "Classify the scene as exactly one of: safe | caution | hazard"
)

TOTAL_TRIALS = len(MODELS) * len(TEMPERATURES) * len(MAX_TOKENS_L) * len(SCENES) * N_RUNS


def cell_key(model: str, temp: float, max_tok: int) -> str:
    return f"{model}|{temp:.1f}|{max_tok}"


def main():
    print("=" * 70)
    print("EXP-V9: Model × Temperature × Max-Tokens Full Factorial (3×3×3)")
    print(f"Models={MODELS}")
    print(f"Temperatures={TEMPERATURES}")
    print(f"Max-tokens={MAX_TOKENS_L}")
    print(f"N={N_RUNS} per cell per scene  |  Total trials: {TOTAL_TRIALS}")
    print("=" * 70)

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

            for model in MODELS:
                for temp in TEMPERATURES:
                    for max_tok in MAX_TOKENS_L:
                        res    = call_vision_llm(jpeg, PROMPT, model=model,
                                                 max_tokens=max_tok,
                                                 temperature=temp)
                        scores = score_verbalization(res["reply"], scene["truth"])

                        row = {
                            "scene_id":      scene["id"],
                            "scene_label":   scene["label"],
                            "truth":         scene["truth"],
                            "model":         model,
                            "temperature":   temp,
                            "max_tokens":    max_tok,
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
                        print(
                            f"     {model:8s} t={temp:.1f} tok={max_tok:3d}  "
                            f"q={scores['quality_score']}/4  "
                            f"risk={scores.get('detected_risk','?'):8s}  "
                            f"{res['latency_ms']:.0f}ms  ${res['cost_usd']:.6f}"
                        )

            time.sleep(2)

    # ── Save raw runs
    fields = [
        "scene_id", "scene_label", "truth", "model", "temperature", "max_tokens", "run",
        "quality_score", "s1_scene", "s2_proximity", "s3_risk", "s4_length",
        "detected_risk", "word_count", "latency_ms", "input_tokens",
        "output_tokens", "cost_usd", "error",
    ]
    runs_csv = RESULTS_DIR / "V9_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Per-cell summary (27 cells)
    print(f"\n── V9 Cell Summary (quality mean [lo, hi]) ─────────────────────────────────────")
    header = f"  {'model':8s}  {'temp':5s}  {'tok':4s}  quality           accuracy          "
    header += "latency(ms)  cost_usd        efficiency(q/$)"
    print(header)
    print(f"  {'-'*100}")

    summary_rows = []
    for model in MODELS:
        for temp in TEMPERATURES:
            for max_tok in MAX_TOKENS_L:
                cell = [r for r in all_rows
                        if r["model"] == model
                        and r["temperature"] == temp
                        and r["max_tokens"] == max_tok
                        and not r["error"]]
                if not cell:
                    continue

                qm, qlo, qhi = bootstrap_ci([r["quality_score"] for r in cell])
                lm, llo, lhi = bootstrap_ci([r["latency_ms"]    for r in cell])
                cm, clo, chi = bootstrap_ci([r["cost_usd"]       for r in cell])
                ot, _,   _   = bootstrap_ci([r["output_tokens"]  for r in cell])

                n_correct = sum(r["s3_risk"] for r in cell)
                acc, alo, ahi = wilson_ci(n_correct, len(cell))

                # Efficiency: quality per dollar (use per-trial ratio to allow CI)
                eff_vals = [r["quality_score"] / r["cost_usd"]
                            for r in cell if r["cost_usd"] > 0]
                em, elo, ehi = bootstrap_ci(eff_vals) if eff_vals else (0, 0, 0)

                print(
                    f"  {model:8s}  {temp:5.1f}  {max_tok:4d}  "
                    f"{qm:.2f}[{qlo:.2f},{qhi:.2f}]  "
                    f"acc={acc:.3f}[{alo:.3f},{ahi:.3f}]  "
                    f"{lm:.0f}ms  "
                    f"${cm:.6f}[{clo:.6f},{chi:.6f}]  "
                    f"{em:.1f}[{elo:.1f},{ehi:.1f}]"
                )

                summary_rows.append({
                    "model":          model,
                    "temperature":    temp,
                    "max_tokens":     max_tok,
                    "quality_mean":   round(qm,  4),
                    "quality_lo":     round(qlo, 4),
                    "quality_hi":     round(qhi, 4),
                    "accuracy":       round(acc, 4),
                    "accuracy_lo":    round(alo, 4),
                    "accuracy_hi":    round(ahi, 4),
                    "latency_ms":     round(lm,  2),
                    "latency_lo":     round(llo, 2),
                    "latency_hi":     round(lhi, 2),
                    "output_tokens":  round(ot,  1),
                    "cost_usd_mean":  round(cm,  6),
                    "cost_usd_lo":    round(clo, 6),
                    "cost_usd_hi":    round(chi, 6),
                    "efficiency_mean":round(em,  3),
                    "efficiency_lo":  round(elo, 3),
                    "efficiency_hi":  round(ehi, 3),
                    "n":              len(cell),
                })

    summary_csv = RESULTS_DIR / "V9_summary.csv"
    write_csv(summary_csv, summary_rows, [
        "model", "temperature", "max_tokens",
        "quality_mean", "quality_lo", "quality_hi",
        "accuracy", "accuracy_lo", "accuracy_hi",
        "latency_ms", "latency_lo", "latency_hi",
        "output_tokens",
        "cost_usd_mean", "cost_usd_lo", "cost_usd_hi",
        "efficiency_mean", "efficiency_lo", "efficiency_hi",
        "n",
    ])

    # ── Interaction 1: Δ_temp per model (at median max_tokens=256)
    print(f"\n── V9 Interaction 1: Temperature Effect per Model (max_tokens=256) ──────────────")
    print(f"  {'model':8s}  q(t=0.0)  q(t=0.5)  q(t=1.0)  Δ(1.0-0.0)  acc(t=0.0)  acc(t=1.0)")
    matrix_rows = []
    for model in MODELS:
        row = {"model": model, "axis": "temperature", "fixed": "max_tokens=256"}
        row_str = f"  {model:8s}"
        q_by_temp = {}
        acc_by_temp = {}
        for temp in TEMPERATURES:
            cell = [r for r in all_rows
                    if r["model"] == model and r["temperature"] == temp
                    and r["max_tokens"] == 256 and not r["error"]]
            if cell:
                qm, _, _ = bootstrap_ci([r["quality_score"] for r in cell])
                acc, _, _ = wilson_ci(sum(r["s3_risk"] for r in cell), len(cell))
                q_by_temp[temp]   = qm
                acc_by_temp[temp] = acc
                row_str += f"  {qm:.2f}    "
                row[f"q_t{str(temp).replace('.','_')}"] = round(qm, 4)
                row[f"acc_t{str(temp).replace('.','_')}"] = round(acc, 4)
            else:
                row_str += "  N/A     "
        if 0.0 in q_by_temp and 1.0 in q_by_temp:
            delta = q_by_temp[1.0] - q_by_temp[0.0]
            row_str += f"  Δ={delta:+.2f}    "
            row["delta_temp_1_0_minus_0_0"] = round(delta, 4)
            a0 = acc_by_temp.get(0.0, float("nan"))
            a1 = acc_by_temp.get(1.0, float("nan"))
            row_str += f"  {a0:.3f}     {a1:.3f}"
        print(row_str)
        matrix_rows.append(row)

    # ── Interaction 2: Δ_tokens per model (at median temp=0.5)
    print(f"\n── V9 Interaction 2: Token Budget Effect per Model (temperature=0.5) ───────────")
    print(f"  {'model':8s}  q(128)    q(256)    q(512)    Δ(512-128)  efficiency(128)  efficiency(512)")
    for model in MODELS:
        row = {"model": model, "axis": "max_tokens", "fixed": "temperature=0.5"}
        row_str = f"  {model:8s}"
        q_by_tok = {}
        eff_by_tok = {}
        for max_tok in MAX_TOKENS_L:
            cell = [r for r in all_rows
                    if r["model"] == model and r["temperature"] == 0.5
                    and r["max_tokens"] == max_tok and not r["error"]]
            if cell:
                qm, _, _ = bootstrap_ci([r["quality_score"] for r in cell])
                eff_vals  = [r["quality_score"] / r["cost_usd"]
                             for r in cell if r["cost_usd"] > 0]
                em, _, _  = bootstrap_ci(eff_vals) if eff_vals else (0, 0, 0)
                q_by_tok[max_tok]   = qm
                eff_by_tok[max_tok] = em
                row_str += f"  {qm:.2f}    "
                row[f"q_tok{max_tok}"] = round(qm, 4)
                row[f"eff_tok{max_tok}"] = round(em, 3)
            else:
                row_str += "  N/A     "
        if 128 in q_by_tok and 512 in q_by_tok:
            delta = q_by_tok[512] - q_by_tok[128]
            row_str += f"  Δ={delta:+.2f}    "
            row["delta_tok_512_minus_128"] = round(delta, 4)
            row_str += f"  {eff_by_tok.get(128, float('nan')):.1f}          {eff_by_tok.get(512, float('nan')):.1f}"
        print(row_str)
        matrix_rows.append(row)

    # ── Interaction 3: Model gap as a function of temperature (max_tokens=256)
    print(f"\n── V9 Interaction 3: Model Gap vs Temperature (max_tokens=256) ────────────────")
    print(f"  {'temp':5s}  q(claude)  q(gpt4o)  q(gemini)  Δ(claude−gemini)")
    for temp in TEMPERATURES:
        q_by_model = {}
        for model in MODELS:
            cell = [r for r in all_rows
                    if r["model"] == model and r["temperature"] == temp
                    and r["max_tokens"] == 256 and not r["error"]]
            if cell:
                qm, _, _ = bootstrap_ci([r["quality_score"] for r in cell])
                q_by_model[model] = qm
        if len(q_by_model) == 3:
            delta = q_by_model.get("claude", 0) - q_by_model.get("gemini", 0)
            print(
                f"  {temp:5.1f}  "
                f"{q_by_model.get('claude',float('nan')):.2f}       "
                f"{q_by_model.get('gpt4o',float('nan')):.2f}      "
                f"{q_by_model.get('gemini',float('nan')):.2f}       "
                f"Δ={delta:+.2f}"
            )
            matrix_rows.append({
                "model":   "claude-gemini-gap",
                "axis":    "model_gap",
                "fixed":   f"temperature={temp}",
                "delta_model_claude_minus_gemini": round(delta, 4),
            })

    # ── Best cell
    if summary_rows:
        best_q   = max(summary_rows, key=lambda r: r["quality_mean"])
        best_acc = max(summary_rows, key=lambda r: r["accuracy"])
        best_eff = max(summary_rows, key=lambda r: r["efficiency_mean"])
        print(f"\n── V9 Best Cells ───────────────────────────────────────────────────────────────")
        print(f"  Highest quality  : model={best_q['model']:8s}  "
              f"t={best_q['temperature']:.1f}  tok={best_q['max_tokens']:3d}  "
              f"q={best_q['quality_mean']:.3f}")
        print(f"  Highest accuracy : model={best_acc['model']:8s}  "
              f"t={best_acc['temperature']:.1f}  tok={best_acc['max_tokens']:3d}  "
              f"acc={best_acc['accuracy']:.3f}")
        print(f"  Best efficiency  : model={best_eff['model']:8s}  "
              f"t={best_eff['temperature']:.1f}  tok={best_eff['max_tokens']:3d}  "
              f"eff={best_eff['efficiency_mean']:.1f} q/$")

    matrix_csv = RESULTS_DIR / "V9_matrix.csv"
    write_csv(matrix_csv, matrix_rows, list({k for r in matrix_rows for k in r}))

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"Matrix  → {matrix_csv}")
    print(f"[V9] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
