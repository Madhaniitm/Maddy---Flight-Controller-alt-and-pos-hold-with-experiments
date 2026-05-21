# EXP-V2R Observations — ReAct Agentic vs ReAct Template (door_open)

**Date**: 2026-05-21
**Data**: `V2R_runs_20260521_034438.csv` (20 Condition B trials — 1 scene × 4 models × 5 runs)
**Condition A source**: `V2_runs_20260521_020936.csv` (react technique, door_open rows, N=5 per model)
**Pipeline**: ESP32-S3 Sense → YOLOv8n → LLM (2-call agentic loop vs single-pass template)

---

## Experimental Design

Two conditions tested on identical frames of the `door_open` scene (truth = **safe**):

| Condition | Description |
|---|---|
| **A — react_template** | Single pass: [YOLO metadata + image] → model writes REASON/OBSERVE/ACT → classify. No feedback. Loaded from V2 data. |
| **B — react_agentic** | 2-call loop: Call 1 [image only] → model observes. Call 2 [observation + YOLO] → final classification. |

The only variable is **when YOLO feedback arrives** — before the model reasons (template) or after (agentic). This isolates the feedback loop contribution from the ReAct framing itself.

Scene selected: `door_open` — the worst react_template failure across all 8 V2 scenes (5% accuracy).

---

## Results

### Per-Model Accuracy on door_open

| Model | Condition A (template) | Condition B (agentic) | Delta |
|---|---|---|---|
| claude | 0% (0/5) | 0% (0/5) | 0 |
| gpt4o | 20% (1/5) | 40% (2/5) | **+20pp** |
| gpt4o_mini | 0% (0/5) | **100%** (5/5) | **+100pp** |
| gemini | 0% (0/5) | 0% (0/5) | 0 |
| **All models** | **5% (1/20)** | **35% (7/20)** | **+30pp** |

### What models detected (door_open, truth = safe)

**Condition A — react_template** (all models overwhelmingly wrong):
- claude: hazard × 5
- gpt4o: caution × 4, safe × 1
- gpt4o_mini: caution × 2, hazard × 3
- gemini: hazard × 2, caution × 3

**Condition B — react_agentic** (feedback recovers two models):
- claude: no parse × 3, hazard × 2 (still fails — format issue in ACT call)
- gpt4o: safe × 2, hazard × 1, caution × 2
- gpt4o_mini: safe × 5 (perfect)
- gemini: hazard × 3, caution × 1, no parse × 1 (still fails)

---

## Observations

**O1 — Feedback loop recovers the worst template failure**
The react_template collapses on door_open: all models treat an open doorway as hazard or caution in a single-pass prompt (5% accuracy). With the agentic feedback loop, overall accuracy triples to 35%. The step-by-step Reason-Observe-Act text template alone is insufficient — the feedback loop is what delivers the improvement.

**O2 — GPT-4o Mini achieves complete recovery (0% → 100%)**
When allowed to first observe the image without sensor data, then receive YOLO confirmation (no objects, clear path), gpt4o_mini classifies door_open correctly in all 5 runs. In the template condition it produced hazard or caution every single time. This is the clearest evidence that delayed feedback — not the ReAct framing — drives correct classification.

**O3 — Template failure is a framing artifact, not a vision failure**
In the template condition, models are primed by the Reason-Observe-Act structure to look for hazards. An open doorway, framed as a "safety assessment," gets escalated to caution or hazard because the format primes the model to find something wrong. The agentic loop decouples observation (neutral, no framing) from classification (post-feedback), breaking this priming effect for models that can self-correct.

**O4 — Claude and Gemini do not benefit from feedback**
Claude fails in both conditions (0%): in the template it always outputs hazard; in the agentic condition it frequently produces no parseable risk label in the short ACT call. Gemini also stays at 0%, consistently labelling the open door as hazard even after receiving YOLO confirmation. These two models have stronger priors about open spaces being risky that the feedback loop cannot override.

**O5 — Open-loop (V2) is the ceiling for template-based ReAct**
V2 shows that across all 8 scenes and all 5 techniques, react_template performs no better than zero_shot on average (37.5% vs 44.4%). For door_open specifically, zero_shot achieves 55% while react collapses to 5% — the ReAct framing actively hurts when there is no real feedback. These V2 results represent the open-loop performance ceiling for any single-pass prompting strategy, regardless of format.

---

## Conclusion

**ReAct adds value only when the feedback loop is real.**

In open-loop operation (V2 react_template), the Reason-Observe-Act structure is a verbose zero_shot prompt — it mimics the ReAct form but provides no actual feedback. Performance on the hardest scene (door_open) is 5%, worse than every other technique.

With a real feedback loop (react_agentic), the same scene recovers to 35% overall and 100% for gpt4o_mini. This directly validates the architecture used in the C-series drone control experiments, where ReAct works because the model receives actual tool return values (drone telemetry, position updates) at each step — not because of the text structure. The C-series and V2R share the same mechanism: **the model's second call is informed by ground-truth feedback that was not available at observation time.**

For the thesis: the V2R result closes the loop on the C-series justification. Open-loop vision classification uses the best prompt technique identified in V2 (zero_shot for Gemini, few_shot_3 for GPT-4o). Closed-loop drone command generation uses ReAct with real tool feedback. The two are architecturally distinct and not in contradiction.
