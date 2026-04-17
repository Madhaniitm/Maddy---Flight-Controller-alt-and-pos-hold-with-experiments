"""
EXP-V6: Verbosity vs Quality Tradeoff
=======================================
Goal:
    How much does increasing max_tokens improve verbalization quality?
    Does it plateau, and at what cost?

    max_tokens levels : [64, 128, 256, 512]
    Fixed             : Claude, zero_shot prompt, 10 scenes × N=5 = 200 trials.

    Hypothesis: Quality improves up to 256 tokens then plateaus, while cost
    increases linearly. The "sweet spot" balances quality and cost.

Metrics:
    - quality_score (0-4 rubric)       Bootstrap CI per level
    - word_count                        Bootstrap CI
    - truncated_reply : reply ends mid-sentence (bool) — Wilson CI
    - latency_ms                        Bootstrap CI
    - output_tokens                     Bootstrap CI
    - cost_usd                          Bootstrap CI
    - efficiency : quality_score / cost_usd   (quality per dollar)
"""

import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_RUNS      = 5
MODEL       = "claude"
MAX_TOKENS  = [64, 128, 256, 512]

PROMPT = (
    "You are a drone camera monitor. "
    "Look at the image and describe what you see. "
    "Estimate proximity of objects. "
    "Classify the scene as: safe | caution | hazard. "
    "Be as thorough as your response length allows."
)

def is_truncated(reply: str) -> int:
    """Heuristic: reply is likely truncated if it doesn't end with sentence punctuation."""
    s = reply.strip()
    if not s:
        return 1
    return int(s[-1] not in ".!?:\"'")

def main():
    print("="*60)
    print("EXP-V6: Verbosity vs Quality Tradeoff")
    print(f"max_tokens={MAX_TOKENS}  Model={MODEL}  N={N_RUNS}")
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

            for max_tok in MAX_TOKENS:
                res    = call_vision_llm(jpeg, PROMPT, model=MODEL,
                                        max_tokens=max_tok, temperature=0.2)
                scores = score_verbalization(res["reply"], scene["truth"])
                trunc  = is_truncated(res["reply"])
                eff    = round(scores["quality_score"] / max(res["cost_usd"], 1e-8), 2)

                row = {
                    "scene_id":      scene["id"],
                    "scene_label":   scene["label"],
                    "truth":         scene["truth"],
                    "max_tokens":    max_tok,
                    "run":           run,
                    "quality_score": scores["quality_score"],
                    "s1_scene":      scores["s1_scene"],
                    "s2_proximity":  scores["s2_proximity"],
                    "s3_risk":       scores["s3_risk"],
                    "s4_length":     scores["s4_length"],
                    "word_count":    scores["word_count"],
                    "truncated":     trunc,
                    "latency_ms":    res["latency_ms"],
                    "input_tokens":  res["input_tokens"],
                    "output_tokens": res["output_tokens"],
                    "cost_usd":      res["cost_usd"],
                    "efficiency":    eff,
                    "error":         res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)
                print(f"     max_tok={max_tok:3d}  quality={scores['quality_score']}/4  "
                      f"words={scores['word_count']:3d}  trunc={trunc}  "
                      f"lat={res['latency_ms']:.0f}ms  ${res['cost_usd']:.6f}")

            time.sleep(2)

    # ── Save
    fields = ["scene_id","scene_label","truth","max_tokens","run",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "word_count","truncated","latency_ms","input_tokens","output_tokens",
              "cost_usd","efficiency","error"]
    runs_csv = RESULTS_DIR / "V6_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per max_tokens level
    print(f"\n── V6 Summary ──────────────────────────────────────────────")
    print(f"  {'max_tok':8s}  quality  words  trunc%  latency  cost      efficiency")
    summary_rows = []
    prev_qm = None
    for max_tok in MAX_TOKENS:
        tr = [r for r in all_rows if r["max_tokens"]==max_tok and not r["error"]]
        if not tr: continue
        qm, qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
        wm, _,   _   = bootstrap_ci([r["word_count"]    for r in tr])
        lm, _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
        cm, clo, chi = bootstrap_ci([r["cost_usd"]      for r in tr])
        em, _,   _   = bootstrap_ci([r["efficiency"]    for r in tr if r["efficiency"]>0])
        tr_r, _,  _  = wilson_ci(sum(r["truncated"] for r in tr), len(tr))

        delta = f"  Δq={qm-prev_qm:+.2f}" if prev_qm is not None else ""
        print(f"  {max_tok:8d}  {qm:.2f}[{qlo:.2f},{qhi:.2f}]  "
              f"{wm:.0f}    {tr_r*100:.0f}%     {lm:.0f}ms    "
              f"${cm:.6f}  {em:.0f}{delta}")
        prev_qm = qm
        summary_rows.append({
            "max_tokens": max_tok,
            "quality": qm, "q_lo": qlo, "q_hi": qhi,
            "word_count": wm,
            "truncation_rate": tr_r,
            "latency_ms": lm,
            "cost_usd": cm, "cost_lo": clo, "cost_hi": chi,
            "efficiency": em,
        })

    # Sweet spot: best efficiency
    if summary_rows:
        sweet = max(summary_rows, key=lambda r: r["efficiency"])
        print(f"\n  Efficiency sweet spot: max_tokens={sweet['max_tokens']} "
              f"(quality={sweet['quality']:.2f}, efficiency={sweet['efficiency']:.0f} q/USD)")

    summary_csv = RESULTS_DIR / "V6_summary.csv"
    write_csv(summary_csv, summary_rows,
              ["max_tokens","quality","q_lo","q_hi","word_count","truncation_rate",
               "latency_ms","cost_usd","cost_lo","cost_hi","efficiency"])

    print(f"\nData   → {runs_csv}")
    print(f"Summary→ {summary_csv}")

if __name__ == "__main__":
    main()
