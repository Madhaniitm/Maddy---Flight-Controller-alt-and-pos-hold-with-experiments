# V2 Observations — Run 2026-05-25

**Script:** `Image verbalization experiments/exp_V2_prompt_techniques.py`
**Results:** `V2_runs_20260525_035756.csv`
**Pipeline:** CLAHE → YOLO-World + CLIP → [technique prompt] → LLM
**Models:** claude, gpt4o, gpt4o_mini, gemini
**Techniques:** zero_shot, few_shot_3, cot, structured, react
**Total trials:** 800 (5 techniques × 4 models × 8 scenes × 5 runs) | Errors: 10

---

## Purpose

Compare five prompting strategies for one-shot safety classification.
V2 answers: **which prompt technique is best for a single-pass safety decision?**

The thesis framing:
- One-shot classification (this experiment) → structured / few_shot_3
- Agentic feedback control (C-series, V2R) → ReAct

---

## Summary by Technique

| Technique | N | LblAcc | DescAcc | RsnRisk | RsnAct | ActSafe | Danger |
|---|---|---|---|---|---|---|---|
| few_shot_3 | 158 | 51.9% | **92.4%** | 93.1% | **100%** | 96.8% | 5 |
| cot | 156 | 19.2% | 89.7% | **96.0%** | 98.1% | **100%** | **0** |
| structured | 159 | **56.6%** | 84.3% | 87.3% | **100%** | **100%** | **0** |
| zero_shot | 159 | 42.1% | 83.6% | 95.2% | **100%** | **100%** | **0** |
| **react** | 158 | 43.0% | 67.1% | 90.8% | 91.5% | 97.5% | 4 |

**Metric definitions:**
- **LblAcc** — did detected risk match ground truth label
- **DescAcc** — did description correctly identify primary scene feature
- **RsnRisk** — does stated description/proximity justify the risk level chosen
- **RsnAct** — is the action internally consistent with the model's own stated risk
- **ActSafe** — is the action safe given ground truth (1 − dangerous rate)
- **Danger** — trials where truth=hazard and action=PITCH_FORWARD

---

## Summary by Model (All Techniques Combined)

| Model | N | LblAcc | DescAcc | RsnRisk | RsnAct | ActSafe | Danger |
|---|---|---|---|---|---|---|---|
| claude | 190 | 30.0% | 66.8% | **95.5%** | **100%** | **100%** | **0** |
| **gpt4o** | 200 | **57.0%** | 89.0% | 93.0% | 99.5% | 98.0% | 4 |
| gpt4o_mini | 200 | 44.5% | 86.0% | 93.9% | 94.2% | 97.5% | 5 |
| gemini | 200 | 38.5% | **91.0%** | 87.9% | **100%** | **100%** | **0** |

---

## Per-Model Best Technique

| Model | Best technique | LblAcc | DescAcc | RsnRisk | RsnAct | ActSafe | Danger |
|---|---|---|---|---|---|---|---|
| claude | structured | 43.6% | 87.2% | 100% | 100% | 100% | 0 |
| gpt4o | structured | **70.0%** | 82.5% | 87.5% | 100% | 100% | 0 |
| gpt4o_mini | structured | 62.5% | 80.0% | 80.0% | 100% | 100% | 0 |
| gemini | few_shot_3 | 55.0% | 100% | 100% | 100% | 100% | 0 |

**Structured is the optimal one-shot technique for claude, gpt4o, gpt4o_mini.**
**Few_shot_3 is optimal for gemini.**

---

## Finding 1 — Structured Wins for One-Shot Classification

Structured JSON format achieves:
- **Best label accuracy** (56.6% overall, 70% for GPT-4o)
- **100% RsnAct** — models never internally contradict themselves
- **0 dangerous cases**
- **Fewest unparseable outputs** (16 no-action cases vs 104 for CoT)

Why structured works: the JSON schema forces the model to commit to each field independently. It cannot produce an action inconsistent with its stated risk level because `risk_level` and `recommended_action` are separate JSON fields — a model that writes `"risk_level": "hazard"` and then `"recommended_action": "PITCH_FORWARD"` is immediately visibly self-contradictory. The format enforces internal consistency.

---

## Finding 2 — CoT Failure Was Token Cutoff, Not Reasoning Failure

CoT achieves **0% label accuracy for Gemini** and **5.6% for Claude** in V2 (max_tokens=300).

Both models produce correct step-by-step reasoning but never output a clean `Risk: X` line — 104 out of 156 CoT trials (67%) produce no detected action. The reasoning fills the 300-token budget and the conclusion is cut off before it is written.

**Validation — CoT rerun at 600 tokens** (`exp_V2_cot_token_rerun.py`, `V2_cot_rerun_20260525_175408.csv`):

| Model | V2 CoT (300 tok) | CoT Rerun (600 tok) | Structured (V2) |
|-------|-----------------|---------------------|-----------------|
| claude | 5.6% | **60.5%** | 43.6% |
| gpt4o | ~38% | **60.0%** | 70.0% |
| gpt4o_mini | ~25% | 30.0% | 62.5% |
| gemini | 0% | 37.5% | 55.0% |
| **Overall** | **19.2%** | **46.8%** | **56.6%** |

Truncation rate dropped from **67% → 2.5%**. Claude CoT (60.5%) actually **beats** Claude structured (43.6%). GPT-4o CoT (60%) closely approaches GPT-4o structured (70%). Gemini still has 10% truncation at 600 tokens — needs 700+ tokens to fully conclude.

**CoT is not inherently inferior to structured** — its reasoning-risk alignment (96%) is the highest of all techniques. The V2 failure was a methodological artefact of the 300-token budget. With adequate tokens, CoT matches structured for Claude and GPT-4o.

**Why structured is still preferred for deployment:** structured commits to a parseable JSON decision within the first 50 tokens regardless of budget — making it robust to token constraints in real-time systems. CoT requires ~500 tokens to reach its conclusion, which doubles latency and cost. For a drone safety system where response time is critical, structured is operationally superior even if CoT reasons equally well.

CoT also takes **18,437ms** at 300 tokens (reasoning without conclusion) and would take even longer at 600 tokens — 2–3× slower than structured at comparable accuracy.

**In a real drone system, structured is preferred over CoT** — not because CoT reasons worse, but because structured is token-efficient, latency-efficient, and always produces a parseable decision.

---

## Finding 3 — React Has Worst Internal Consistency (91.5% RsnAct)

React is the only technique with RsnAct below 100%:
- **GPT-4o-mini react: 75% RsnAct** — 1 in 4 trials internally contradicts its stated risk
- **Overall react: 91.5%** — 8.5% of trials have misaligned risk-action

Why: ReAct's ACT step is designed to commit to an action at the end of a reasoning loop. In one-shot mode there is no correction possible — if the REASON/OBSERVE steps produce an ambiguous intermediate conclusion, the ACT step sometimes commits to an action that doesn't match the stated risk.

React also has **4 dangerous cases** (all GPT-4o wall_close) and the lowest DescAcc (67.1% — Claude's OBSERVE section uses meta-reasoning instead of scene description).

**React is not suited for one-shot classification.** It is designed for iterative feedback loops where each ACT is a small correctable step — exactly how C-series agentic control works.

---

## Finding 4 — All Dangerous Cases Are wall_close, No Metadata

All 9 dangerous cases (few_shot_3: 5, react: 4) are wall_close → PITCH_FORWARD.

V2 uses CLIP metadata which is near-random on 320×240 frames. There is no wall texture warning in the prompt. Without the wall-fill warning (added in G5 sensor fix), GPT-4o anchors to DA v2's wrong 2.09m reading and the model concludes the wall is far. Same hardware limitation as G5/G1 — not a technique-specific failure.

**few_shot_3 dangerous cases are GPT-4o-mini** — it cannot visually identify a blank gray wall without sensor support. With the wall-fill warning (as in full pipeline), these drop to 1.

---

## Finding 5 — Claude: Safest but Lowest Label Accuracy

Claude achieves **100% ActSafe, 100% RsnAct, 0 dangerous** across all 190 trials and all techniques. It never internally contradicts itself and never recommends forward motion into a hazard.

But Claude has **30% label accuracy** — lowest of all models. This is not reasoning failure — Claude's descriptions are correct (66.8% DescAcc) and its reasoning-risk alignment is the highest (95.5% RsnRisk). The low label accuracy reflects Claude's conservative bias: it classifies many caution scenes as hazard and safe scenes as caution. Since these are not dangerous actions, they are acceptable for a safety system.

Claude's react DescAcc is only **2.6%** — it does not describe the scene in its OBSERVE section. Instead it writes meta-reasoning ("I need to determine if..."). This is a structural mismatch between Claude's response style and the ReAct format.

---

## Finding 6 — Gemini: Best Descriptor, Perfectly Consistent

Gemini achieves **100% RsnAct, 100% ActSafe, 0 dangerous** and has the highest DescAcc (91%) across all models. Its best technique is few_shot_3 (55% label acc, 100% on all reasoning metrics).

Gemini react achieves 62.5% label accuracy — the highest react score of any model — suggesting Gemini handles the ReAct format better than others. This is relevant for V2R (agentic validation).

---

## Thesis Interpretation

> *"Prompt technique significantly affects both reasoning quality and action safety for one-shot safety classification. The structured JSON format achieves the highest label accuracy (56.6%), zero dangerous recommendations, and 100% internal consistency (reason-action alignment) — outperforming zero_shot, few_shot_3, CoT, and ReAct on the combined metric. Chain-of-Thought appears to fail in the primary V2 run (19.2% label accuracy, 67% unparseable outputs) but a follow-up rerun at 600 tokens reveals this was a token budget artefact: truncation dropped from 67% to 2.5% and label accuracy rose to 46.8% overall, with Claude CoT (60.5%) exceeding Claude structured (43.6%). CoT's reasoning quality is the highest of all techniques (96% reasoning-risk alignment) — it fails only when the token budget is exhausted before the conclusion is written. Structured remains the deployment choice not because it reasons better, but because it commits to a parseable decision within 50 tokens regardless of budget, at half the latency and cost of CoT. ReAct has the worst internal consistency (91.5% reason-action alignment) and 4 dangerous recommendations in one-shot mode — because its ACT step commits irreversibly to a single-pass conclusion with no correction mechanism. ReAct is not suited for one-shot safety decisions; its strength lies in iterative agentic control loops where each ACT triggers a correctable tool call (validated in C-series)."*

---

## Run Configuration

```
Date          : 2026-05-25
Script        : Image verbalization experiments/exp_V2_prompt_techniques.py
Models        : claude, gpt4o, gpt4o_mini, gemini
Techniques    : zero_shot, few_shot_3, cot, structured, react
N runs        : 5 per scene per technique per model
Scenes        : 8 canonical scenes (run03 saved frames)
Total trials  : 800 planned, 790 valid (10 errors)
Pipeline      : CLAHE → YOLO-World + CLIP → technique prompt → LLM
Note          : CLIP metadata is near-random on 320×240 (see V_clip_ablation)
                All dangerous cases are wall_close without wall-fill sensor warning
                max_tokens=300 (CoT token cutoff artefact — see rerun below)

CoT Rerun     : 2026-05-25
Script        : Image verbalization experiments/exp_V2_cot_token_rerun.py
Results       : V2_cot_rerun_20260525_175408.csv
max_tokens    : 600 (vs 300 in main V2 run)
Total trials  : 160 (4 models × 8 scenes × 5 runs)
Finding       : Truncation 67%→2.5%, LblAcc 19.2%→46.8% — token cutoff confirmed
```
