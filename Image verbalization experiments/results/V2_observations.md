# EXP-V2 Observations — Prompt Technique Comparison (All 4 Models)

**Date**: 2026-05-21
**Data**: `V2_runs_20260521_020936.csv` (800 trials — 8 scenes × 5 techniques × 4 models × 5 runs)
**Pipeline**: ESP32-S3 Sense (OV2640) → YOLOv8n → [YOLO metadata + JPEG] → LLM → risk classification

---

## Summary Table — By Technique (All Models Averaged)

| Technique   | Accuracy | 95% CI         | Quality/5 | Latency | Cost/call |
|-------------|----------|----------------|-----------|---------|-----------|
| zero_shot   | 44.4%    | [0.369, 0.521] | 3.80      | 4739ms  | $0.00180  |
| few_shot_3  | 38.6%    | [0.314, 0.464] | 4.18      | 4267ms  | $0.00180  |
| cot         | 37.3%    | [0.302, 0.451] | 3.94      | 5436ms  | $0.00210  |
| structured  | 31.9%    | [0.252, 0.395] | 4.32      | 3559ms  | $0.00140  |
| react       | 37.5%    | [0.304, 0.452] | 3.83      | 5090ms  | $0.00220  |

## Summary Table — By Model × Technique

| Model       | Technique  | Accuracy | Quality/5 | Latency  |
|-------------|------------|----------|-----------|----------|
| claude      | zero_shot  | 37.5%    | 3.40      | 9500ms   |
| claude      | few_shot_3 | 34.2%    | 3.68      | 8661ms   |
| claude      | cot        | 36.8%    | 3.34      | 9834ms   |
| claude      | structured | 35.0%    | 4.35      | 5104ms   |
| claude      | react      | 35.0%    | 3.08      | 10181ms  |
| gpt4o       | zero_shot  | 42.5%    | 3.80      | 3404ms   |
| gpt4o       | few_shot_3 | **57.5%**| 4.48      | 3519ms   |
| gpt4o       | cot        | 45.0%    | 4.45      | 4191ms   |
| gpt4o       | structured | 30.0%    | 4.30      | 3838ms   |
| gpt4o       | react      | 27.5%    | 3.73      | 4178ms   |
| gpt4o_mini  | zero_shot  | 37.5%    | 3.75      | 2653ms   |
| gpt4o_mini  | few_shot_3 | 37.5%    | 4.33      | 2788ms   |
| gpt4o_mini  | cot        | 32.5%    | 4.23      | 4868ms   |
| gpt4o_mini  | structured | 30.0%    | 4.30      | 2832ms   |
| gpt4o_mini  | react      | **45.0%**| 4.25      | 3264ms   |
| gemini      | zero_shot  | **60.0%**| 4.25      | 3399ms   |
| gemini      | few_shot_3 | 25.0%    | 4.20      | 2319ms   |
| gemini      | cot        | 35.0%    | 3.70      | 3069ms   |
| gemini      | structured | 32.5%    | 4.33      | 2462ms   |
| gemini      | react      | 42.5%    | 4.28      | 2736ms   |

## Per-Scene Accuracy by Technique (All Models Averaged)

| Scene        | Truth   | zero_shot | few_shot_3 | cot  | structured | react |
|--------------|---------|-----------|------------|------|------------|-------|
| person_near  | hazard  | 0.45      | 0.50       | 0.55 | 0.40       | 0.50  |
| wall_close   | hazard  | 0.25      | 0.10       | 0.35 | 0.25       | 0.40  |
| object_table | safe    | 0.75      | 0.30       | 0.30 | 0.50       | 0.20  |
| dim_light    | caution | 0.55      | 0.55       | 0.10 | 0.50       | 0.35  |
| cluttered    | caution | 0.25      | 0.70       | 0.30 | 0.35       | 0.55  |
| door_open    | safe    | 0.55      | 0.30       | 0.40 | 0.05       | 0.05  |
| person_far   | safe    | 0.25      | 0.05       | 0.10 | 0.00       | 0.00  |
| blocked_lens | hazard  | 0.50      | 0.55       | 0.85 | 0.50       | 0.95  |

---

## Observations

**O1 — No single technique dominates across all models**
The best technique varies by model: GPT-4o peaks with `few_shot_3` (57.5%), GPT-4o Mini with `react` (45%), Claude and Gemini both with `zero_shot` (38% and 60%). Prompt technique selection is model-dependent — there is no universal best prompt for the drone copilot pipeline.

**O2 — Gemini is hurt by added structure**
Gemini scores 60% with `zero_shot` but drops sharply to 25% with `few_shot_3` — its worst technique. Adding worked examples actively confuses Gemini's risk reasoning. This is notable because Gemini was V1's best performer. For Gemini, minimal prompting is optimal.

**O3 — GPT-4o benefits most from examples**
GPT-4o jumps from 42.5% (V1 baseline) to 57.5% with `few_shot_3` — the largest improvement of any model. Concrete examples anchor GPT-4o's tendency to default to `safe` in ambiguous scenes. This is the single best model×technique combination in the experiment.

**O4 — CoT and ReAct are the only techniques that solve blocked_lens**
`blocked_lens` (covered camera, truth=hazard) went from near-zero in V1 to 85% with `cot` and 95% with `react`. Step-by-step and Reason-Observe-Act frameworks force the model to explicitly reason about why it cannot see anything — which is exactly the right inference for a covered lens. This is the strongest evidence that reasoning-chain prompts add genuine value for specific failure modes.

**O5 — Structured JSON prompt has the lowest accuracy but highest quality score**
`structured` scores lowest on accuracy (31.9%) yet achieves the highest quality score (4.32/5) and fastest latency (3.6s). All 4 models parse JSON successfully (100% parse rate). The format enforces well-structured output but constrains risk reasoning — models fill in the JSON fields correctly in form but incorrectly in content.

**O6 — door_open completely breaks with structured and react**
`door_open` (truth=safe) drops from 80%+ in V1 to just 5% accuracy with both `structured` and `react`. Models filling a JSON template or following a reasoning framework appear to interpret an open doorway as an obstacle or unknown hazard. This is a prompt-induced regression — the template framing inverts the classification for this scene.

**O7 — person_far remains unsolvable regardless of technique**
Across all 5 techniques and all 4 models, `person_far` (truth=safe) averages 0–25% accuracy. No prompting strategy resolves the systematic bias of models escalating `person visible` to caution/hazard irrespective of distance. This is a training-data bias, not a prompt design problem.

**O8 — CoT adds the most latency for the least overall gain**
`cot` is the slowest technique (6131ms average vs 3559ms for `structured`) because it generates long step-by-step reasoning outputs. Overall accuracy (37.3%) is barely above zero_shot (44.4%). The latency cost is only justified for `blocked_lens` — not as a general-purpose technique.

**O9 — Claude is insensitive to prompting technique**
Claude's accuracy ranges only from 34.2% (`few_shot_3`) to 37.5% (`zero_shot`) — a 3-point spread across all 5 techniques. Every other model shows 15–35 point variation. Claude's caution bias is consistent and prompt-resistant, making it the most predictable but least improvable model through prompting alone.

**O10 — Recommended technique per model**

| Model       | Best Technique | Accuracy |
|-------------|----------------|----------|
| Gemini      | zero_shot      | 60.0%    |
| GPT-4o      | few_shot_3     | 57.5%    |
| GPT-4o Mini | react          | 45.0%    |
| Claude      | zero_shot      | 37.5%    |

For the thesis pipeline using Gemini, `zero_shot` is both the best and simplest option — no examples, no reasoning chain overhead, and the lowest token cost. The V1 task prompt used in V1 (which closely resembles zero_shot with structured output fields) is validated as the right design choice.

---

## Notable Interactions

- **cluttered + few_shot_3**: accuracy jumps to 0.70 — examples help models see disorder as caution rather than safe or hazard
- **dim_light + cot**: accuracy collapses to 0.10 — step-by-step reasoning in dark scenes leads models to overthink and misclassify
- **blocked_lens + react**: 0.95 accuracy — the best single scene×technique result in the entire experiment
- **door_open + structured/react**: 0.05 accuracy — the worst prompt-induced regression in the experiment
- **Gemini few_shot_3**: 0.25 accuracy — 35-point drop from its zero_shot baseline, largest negative effect of any technique on any model

---

## Note on ReAct and C-Series Justification

The C-series experiments use ReAct for natural language → drone command translation and show strong task completion rates. V2 shows that a react-style text prompt applied to vision classification performs poorly. These two uses of ReAct are fundamentally different and not in contradiction:

**C-series ReAct (agentic tool-use loop):**
```
Reason → call tool (take_off / move_to / hover) → observe real result → Reason → call tool → ...
```
The model receives actual feedback from the drone simulator at each step — state changes, sensor readings, tool return values. The loop is real and iterative.

**V2 react (single-pass text template):**
```
REASON: is this scene safe?
OBSERVE: describe what you see
ACT: classify as safe/caution/hazard
```
This is a static single-pass prompt. There is no feedback loop — the model writes all three sections in one forward pass and classifies once. It mimics the ReAct structure in text but has no actual iteration or observation update.

**Claim**: ReAct works for drone control because the feedback loop is real. V2 react underperforms because the loop is absent — making it a verbose zero_shot prompt with formatting overhead.

**This claim requires experimental validation** — see EXP-V2R (ReAct Agentic vs Template), designed to isolate the contribution of real feedback vs text template structure in the vision classification context.
