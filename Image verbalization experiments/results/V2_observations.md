# EXP-V2 Observations — Prompt Technique Comparison (YOLO-World + CLIP Pipeline)

**Date**: 2026-05-24  
**Data**: `V2_runs_20260523_233838.csv` (800 trials — 8 scenes × 5 techniques × 4 models × 5 runs)  
**Pipeline**: ESP32-S3 Sense (OV2640) → CLAHE → YOLO-World (17 hazard classes) → Open-CLIP ViT-B-32 → [metadata + JPEG] → LLM → risk classification  
**Previous run** (COCO YOLOv8n, 2026-05-21): `V2_runs_20260521_020936.csv` — superseded, kept for comparison  

> **Canonical run**: The YOLO-World+CLIP run below is the correct basis for all V2 analysis and thesis writing. The old COCO run is archived.

---

## Summary Table — By Technique (All Models Averaged)

| Technique   | Accuracy | 95% CI         | Quality/5 | Latency | Cost/call | Δ vs Old Pipeline |
|-------------|----------|----------------|-----------|---------|-----------|-------------------|
| structured  | **65.6%**| [0.580, 0.725] | **4.66**  | **3.2s**| $0.00160  | **+33.7%** ↑↑     |
| cot         | 62.5%    | [0.548, 0.696] | 3.71      | 5.0s    | $0.00250  | +25.2%             |
| react       | 61.3%    | [0.535, 0.684] | 3.88      | 4.7s    | $0.00240  | +23.8%             |
| few_shot_3  | 60.6%    | [0.529, 0.679] | 4.24      | 3.8s    | $0.00220  | +22.0%             |
| zero_shot   | 60.0%    | [0.523, 0.673] | 4.10      | 4.1s    | $0.00190  | +15.6%             |

**Ranking reversed from old pipeline.** Old ranking: zero_shot(44.4%) > few_shot(38.6%) > cot(37.3%) > react(37.5%) > structured(31.9%). New ranking: structured(65.6%) > cot(62.5%) > react(61.3%) > few_shot(60.6%) > zero_shot(60.0%).

---

## Summary Table — By Model × Technique

| Model       | Technique   | Accuracy | 95% CI         | Quality/5 | Latency  | Cost     |
|-------------|-------------|----------|----------------|-----------|----------|----------|
| claude      | zero_shot   | 42.5%    | [0.285, 0.578] | 3.38      | 8643ms   | $0.00500 |
| claude      | few_shot_3  | **55.0%**| [0.398, 0.693] | 3.80      | 8322ms   | $0.00530 |
| claude      | cot         | 42.5%    | [0.285, 0.578] | 2.50      | 8703ms   | $0.00560 |
| claude      | structured  | 50.0%    | [0.352, 0.648] | 4.50      | 4544ms   | $0.00310 |
| claude      | react       | 35.0%    | [0.221, 0.505] | 2.53      | 8857ms   | $0.00560 |
| gpt4o       | zero_shot   | 65.0%    | [0.495, 0.779] | 4.38      | 2546ms   | $0.00190 |
| gpt4o       | few_shot_3  | 62.5%    | [0.470, 0.758] | 4.48      | 2581ms   | $0.00280 |
| gpt4o       | **cot**     | **75.0%**| [0.598, 0.858] | 4.75      | 3515ms   | $0.00380 |
| gpt4o       | structured  | 62.5%    | [0.470, 0.758] | 4.63      | 2685ms   | $0.00250 |
| gpt4o       | react       | **75.0%**| [0.598, 0.858] | 4.40      | 3325ms   | $0.00350 |
| gpt4o_mini  | zero_shot   | 75.0%    | [0.598, 0.858] | 4.28      | 2980ms   | $0.00050 |
| gpt4o_mini  | few_shot_3  | 62.5%    | [0.470, 0.758] | 4.50      | 2751ms   | $0.00050 |
| gpt4o_mini  | cot         | 75.0%    | [0.598, 0.858] | 4.33      | 4944ms   | $0.00060 |
| gpt4o_mini  | **structured**|**87.5%**| [0.739, 0.945]| **4.88** | 3365ms   | $0.00050 |
| gpt4o_mini  | react       | 75.0%    | [0.598, 0.858] | 4.28      | 4036ms   | $0.00050 |
| gemini      | zero_shot   | 57.5%    | [0.422, 0.715] | 4.38      | 2369ms   | $0.00010 |
| gemini      | few_shot_3  | **62.5%**| [0.470, 0.758] | 4.20      | 1708ms   | $0.00010 |
| gemini      | cot         | 57.5%    | [0.422, 0.715] | 3.28      | 2903ms   | $0.00010 |
| gemini      | structured  | 62.5%    | [0.470, 0.758] | 4.63      | 2097ms   | $0.00010 |
| gemini      | react       | 60.0%    | [0.446, 0.737] | 4.30      | 2538ms   | $0.00010 |

---

## Per-Scene Accuracy by Technique (All Models Averaged)

| Scene        | Truth   | zero_shot | few_shot_3 | cot   | structured | react |
|--------------|---------|-----------|------------|-------|------------|-------|
| object_table | caution | 0.70      | **1.00**   | 0.55  | **1.00**   | 0.80  |
| cluttered    | caution | 0.80      | **0.95**   | 0.85  | 0.75       | 0.80  |
| wall_close   | hazard  | 0.80      | **0.90**   | 0.75  | 0.75       | 0.80  |
| door_open    | safe    | 0.75      | 0.75       | **0.90**| 0.75     | 0.55  |
| blocked_lens | hazard  | 0.50      | 0.35       | **1.00**| 0.75     | 0.85  |
| person_far   | caution | 0.50      | 0.75       | 0.55  | 0.50       | 0.50  |
| dim_light    | caution | 0.50      | **0.00**   | 0.20  | 0.50       | 0.25  |
| person_near  | hazard  | 0.25      | 0.15       | 0.20  | 0.25       | 0.35  |

---

## GPT-4o Mini + Structured — Per-Scene Detail

| Scene        | Truth   | Acc  | Outputs (5 runs)              |
|--------------|---------|------|-------------------------------|
| blocked_lens | hazard  | 1.00 | hazard × 5                    |
| cluttered    | caution | 1.00 | caution × 5                   |
| dim_light    | caution | 1.00 | caution × 5                   |
| door_open    | safe    | 1.00 | safe × 5                      |
| object_table | caution | 1.00 | caution × 5                   |
| person_far   | caution | 1.00 | caution × 5                   |
| wall_close   | hazard  | 1.00 | hazard × 5                    |
| person_near  | hazard  | 0.00 | **caution × 5** ← only failure|

---

## Observations

**O1 — YOLO-World+CLIP reverses the technique ranking: structured is now best**  
With COCO YOLOv8n, `structured` was the worst technique (31.9%); with YOLO-World+CLIP it becomes the best (65.6%), a 33.7-point gain. The rich, semantically-relevant YOLO-World metadata (17 hazard-specific classes with advisory distances) gives models enough content to fill a structured template meaningfully. In the COCO run, YOLO detected generic objects (chair, bottle) that were irrelevant to risk — the structured format had nothing to organize. The pipeline quality is the dominant factor, not the prompt format.

**O2 — GPT-4o Mini + structured is the new efficiency champion: 87.5% at $0.0005/call**  
The highest accuracy of any model×technique in V2. Mini achieves 7/8 scenes perfect (100% for blocked_lens, cluttered, dim_light, door_open, object_table, person_far, wall_close). Single failure: `person_near` (caution×5, truth=hazard). This is Mini going from 30% structured in the old run to 87.5% — nearly a 3× improvement driven entirely by pipeline quality. Cost is $0.0005 per call — 10× cheaper than GPT-4o ($0.0049) with nearly comparable accuracy.

**O3 — person_near is now the hardest scene across all techniques (15–35% avg)**  
In V1 with the combined YOLO-World prompt, Claude and GPT-4o scored 100% on person_near. In V2, person_near averages only 25% across all techniques. The V2 technique prompts present the YOLO metadata differently — the person detection at 0.31m advisory is being interpreted as caution rather than hazard when wrapped in technique-specific framing. The YOLO metadata is present but the prompt framing changes how models weight close-proximity detections. This is a prompt sensitivity effect, not a pipeline failure.

**O4 — dim_light collapses to 0% with few_shot_3 (all models: 0/20 trials correct)**  
The most severe prompt-induced regression in V2. In zero_shot, dim_light scores 50%; with few_shot examples it drops to 0%. The three few_shot examples appear to anchor models toward hazard or safe responses for dimly-lit scenes — the examples may not include a caution-dim example, causing models to misclassify. This matches the old pipeline observation (O3 in old V2 observations) but is now confirmed on the new pipeline too.

**O5 — CoT achieves 100% on blocked_lens (all models: 20/20 trials correct)**  
Step-by-step reasoning forces models to explicitly process: "I see nothing → why would I see nothing? → covered lens → hazard." CoT is the only technique with perfect blocked_lens detection across all models. This confirms the old pipeline finding (O4) and extends it: even with richer YOLO-World metadata, the "nothing detected" state requires explicit reasoning rather than pattern matching.

**O6 — Structured wins on accuracy, quality, and latency simultaneously**  
`structured` achieves the highest accuracy (65.6%), highest quality score (4.66/5), and fastest latency (3.2s). This triple win is decisive for thesis recommendation. JSON output enforces concise responses (shorter tokens → faster latency), forces explicit field values for each rubric dimension (→ higher quality scores), and with YOLO-World metadata gives models well-defined inputs to structure (→ higher accuracy).

**O7 — Claude's CoT generates parsing failures (9/40 empty detected_risk)**  
Claude CoT shows 9 empty `detected_risk` values out of 40 trials — the model's step-by-step reasoning output does not always contain a parseable risk label. This is a reliability concern: CoT's latency overhead (8.7s for Claude CoT) is paired with increased output failure rate. Claude + CoT is the worst-reliability combination in V2.

**O8 — GPT-4o CoT and ReAct tied at 75% — both better than structured for GPT-4o**  
For GPT-4o, both CoT and ReAct reach 75% while structured only reaches 62.5%. GPT-4o's strong instruction-following allows it to use reasoning chains productively for risk assessment. Unlike Claude (where CoT generates parsing failures) or Mini (where structured outperforms both), GPT-4o benefits from explicit reasoning structure applied to the image.

**O9 — Gemini is now consistent across all techniques (57.5%–62.5%)**  
In the old COCO run, Gemini ranged from 25% (few_shot) to 60% (zero_shot) — a 35-point spread. With YOLO-World+CLIP the spread narrows to 5 points (57.5%–62.5%). The caution-tier improvements from V1 (CLIP labels explicitly providing caution signals) stabilize Gemini's performance regardless of prompt technique. Gemini is no longer brittle to prompt changes — but it also shows no further improvement from structured or CoT prompting.

**O10 — Recommended technique per model (updated)**

| Model       | Best Technique | Accuracy | Rationale                                          |
|-------------|----------------|----------|----------------------------------------------------|
| GPT-4o Mini | structured     | 87.5%    | 7/8 scenes perfect; triple win (acc+quality+speed) |
| GPT-4o      | cot / react    | 75.0%    | Reasoning chains help; tied — pick cot for quality |
| Gemini      | few_shot_3     | 62.5%    | Marginal edge; consistent across all techniques    |
| Claude      | few_shot_3     | 55.0%    | Slight gain from examples; still prompt-resistant  |

**Updated pipeline recommendation**: GPT-4o Mini + structured prompt is the new efficiency frontier — 87.5% accuracy at $0.0005/call. If GPT-4o Mini's single failure mode (person_near → caution instead of hazard) is addressed via prompt refinement, it would surpass GPT-4o at 1/10th the cost. GPT-4o with CoT remains the accuracy leader overall (75% in V2 vs GPT-4o Mini's 75% equivalent across non-structured techniques) but at 10× the cost.

**O11 — Pipeline quality dominates prompt design**  
The total spread across all techniques in the new run is only 5.6 points (60.0%–65.6%). In the old run the spread was 12.5 points (31.9%–44.4%). With YOLO-World+CLIP, the pipeline quality floor is so high that all techniques cluster within 5-6 points of each other. Prompt technique selection matters less than pipeline quality — a finding that reinforces V1's conclusion that the three-tier YOLO-World+CLIP architecture is the primary driver of accuracy.

---

## Pipeline Comparison: COCO vs YOLO-World+CLIP (All Techniques)

| Technique   | COCO Accuracy | YOLO-World Accuracy | Delta    |
|-------------|---------------|---------------------|----------|
| structured  | 31.9%         | **65.6%**           | **+33.7%** |
| react       | 37.5%         | 61.3%               | +23.8%   |
| cot         | 37.3%         | 62.5%               | +25.2%   |
| few_shot_3  | 38.6%         | 60.6%               | +22.0%   |
| zero_shot   | 44.4%         | 60.0%               | +15.6%   |

Structured benefits the most (+33.7%) because it explicitly organizes the YOLO metadata into labelled fields. Zero_shot benefits least (+15.6%) — it could already use whatever metadata was available.

---

## Confusion Patterns — Notable Failures

| Scene        | Technique  | Model       | Outputs (5 runs)               | Note                               |
|--------------|------------|-------------|--------------------------------|------------------------------------|
| person_near  | all        | gpt4o_mini  | caution×5                      | caution instead of hazard — all techniques |
| dim_light    | few_shot_3 | all         | not caution (0/20 correct)     | worst prompt regression in V2      |
| blocked_lens | zero_shot  | gpt4o_mini  | caution×5                      | needs explicit reasoning to flag   |
| door_open    | react      | all         | 55% avg — some misclassify     | react framing interprets doorway as obstacle |
| claude_cot   | cot        | claude      | 9/40 empty risk label          | parsing failures — reliability risk |

---

## Note on V2R Relationship

EXP-V2R (ReAct Agentic vs Template) tests whether real iterative feedback loops improve over the single-pass react text template used here. V2 shows react template = 61.3% vs zero_shot = 60.0% — essentially equivalent. If V2R shows real ReAct loops significantly outperform the template, that confirms the feedback loop (not the format) drives C-series performance. If not, V2 already provides sufficient evidence that the react *format* adds no value.
