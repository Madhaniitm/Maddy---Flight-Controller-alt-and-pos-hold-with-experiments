# Image Verbalization Experiments — Plan
## V-Series: Vision LLM on ESP32-S3 Sense Camera

**Hardware**: XIAO ESP32-S3 Sense (OV2640)  
**Backend** : `server.py` Flask server + YOLO + Vision LLM  
**Target**  : IEEE RA-L / IROS 2026 (camera AI section)

---

## Experiment Summary

| ID  | Name                          | Conditions          | Trials  | Key Metric              |
|-----|-------------------------------|---------------------|---------|-------------------------|
| V1  | Multi-Model Comparison        | 4 models × 10 scenes × 5 | 200 | accuracy, latency, cost |
| V2  | Prompt Technique Comparison   | 5 techniques × 10 scenes × 5 | 250 | quality_score (0-4) |
| V3  | Multilingual Input            | 5 languages × 10 scenes × 5 | 250 | relevance, lang_match |
| V4  | Model × Prompt Matrix         | 3 models × 3 prompts × 10 × 3 | 270 | interaction effect |
| V5  | YOLO Threshold Sweep          | 5 conf thresholds × 20 frames | 100 | precision, recall, F1 |
| V6  | Verbosity vs Quality          | 4 max_token levels × 10 scenes × 5 | 200 | quality vs latency |
| V7  | Scene Context History         | 3 history modes × 5 sequences × 5 | 75 | change_detected |

**Total**: ~1345 trials — all real ESP32 camera frames

---

## Shared Scene Protocol

All experiments use the same **10 canonical scenes** (operator sets up each):

| # | Scene Label     | Setup                                        | Ground Truth |
|---|-----------------|----------------------------------------------|--------------|
| 1 | clear_open      | Empty floor, good lighting                   | safe         |
| 2 | person_near     | Operator stands 1m in front                  | hazard       |
| 3 | wall_close      | Camera faces wall at ~25cm                   | hazard       |
| 4 | object_table    | Laptop on table, clear surroundings          | safe         |
| 5 | dim_light       | Room lights off, single lamp                 | caution      |
| 6 | cluttered       | Multiple objects scattered on floor          | caution      |
| 7 | door_open       | Open doorway visible                         | safe         |
| 8 | person_far      | Operator stands 3m away                      | safe         |
| 9 | blocked_lens    | Camera lens partially covered                | hazard       |
|10 | outdoor_bright  | Bright outdoor / window scene                | safe         |

---

## V1 — Multi-Model Accuracy & Efficiency Comparison

**Goal**: Find best model for real-time camera verbalization on Maddy's system.

**Conditions**: Claude (claude-opus-4-5), GPT-4o, Gemini-1.5-Flash, LLaVA (Ollama local)  
**Task**: Classify each scene → {safe, caution, hazard} + 1–3 sentence description  
**N**: 5 per model per scene = 200 trials

**Metrics**:
- `classification_accuracy` : matches ground truth (Wilson CI)
- `latency_ms`              : end-to-end API call (Bootstrap CI)
- `cost_usd`                : per call cost (Bootstrap CI)
- `quality_score`           : 0-4 rubric (Bootstrap CI)
- `word_count`              : verbalization length (Bootstrap CI)

---

## V2 — Prompt Technique Comparison

**Goal**: Which prompting style yields the best verbalization quality for drone camera?

**Conditions** (all with Claude):
- `zero_shot`   : plain "Describe this scene."
- `few_shot_3`  : 3 examples of good descriptions prepended
- `cot`         : "Think step by step: 1) What objects? 2) Proximity? 3) Risk?"
- `structured`  : "Output JSON: {objects, proximity_cm, risk_level, description}"
- `react`       : Reason→Observe→Act loop embedded in prompt

**N**: 5 per technique per scene = 250 trials  
**Metrics**: quality_score, accuracy, latency_ms, input_tokens, output_tokens

---

## V3 — Multilingual Input Comparison

**Goal**: Can the system understand commands in multiple languages and respond correctly?

**Languages**: English, Hindi (हिंदी), Tamil (தமிழ்), Spanish, French  
**Fixed model**: Claude  
**Question** (same meaning, different language):
  "What do you see? Is it safe?"

**Metrics**:
- `answer_relevance`  : reply mentions scene content (Wilson CI)
- `language_match`    : reply is in same language as question (Wilson CI)
- `classification_accuracy` : risk level correct (Wilson CI)
- `latency_ms`        : per language (Bootstrap CI)

---

## V4 — Model × Prompt Technique Interaction

**Goal**: Does structured prompting benefit all models equally, or only larger ones?

**Design**: 3 × 3 factorial  
  Models  : Claude, GPT-4o, Gemini  
  Prompts : zero_shot, cot, structured  
**N**: 3 per cell per scene (10 scenes) = 270 trials

**Metrics**: quality_score, accuracy (2-way marginal means + interaction term)

---

## V5 — YOLO Confidence Threshold Sweep

**Goal**: Find optimal YOLO confidence for anomaly detection on ESP32 frames.

**Thresholds**: [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]  
**Labeled frames**: 20 (10 hazard, 10 clear) × 5 repeats each threshold  
**Metrics**: precision, recall, F1, false_alarm_rate, miss_rate (Wilson CI per threshold)

---

## V6 — Verbosity vs Quality Tradeoff

**Goal**: How much does increasing max_tokens improve verbalization quality? At what cost?

**max_tokens levels**: [64, 128, 256, 512]  
**Fixed**: Claude, zero_shot prompt, 10 scenes × 5 = 200 trials  
**Metrics**: quality_score, word_count, latency_ms, cost_usd

---

## V7 — Scene Context History Effect

**Goal**: Does giving the LLM previous frame descriptions help it detect scene changes?

**History modes**:
- `stateless` : no prior context
- `short`     : last 2 frame descriptions
- `full`      : all prior descriptions in sequence

**Scenario**: Operator creates 5-frame sequences with a mid-sequence change  
  (e.g., clear → person enters → object placed → person leaves → clear)

**Metrics**: change_detected, description_continuity_score, input_tokens

---

## Scoring Rubric (shared across V1/V2/V3/V4/V6/V7)

| Point | Criterion |
|-------|-----------|
| +1    | Scene content described (objects, colours, layout) |
| +1    | Spatial/proximity information mentioned |
| +1    | Correct risk classification (safe/caution/hazard) |
| +1    | Response length 10–100 words (concise but complete) |

**Max = 4**

---

## Output Files

All results saved to `results/` folder:

```
results/
├── V1_runs.csv / V1_summary.csv
├── V2_runs.csv / V2_summary.csv
├── V3_runs.csv / V3_summary.csv
├── V4_runs.csv / V4_matrix.csv
├── V5_runs.csv / V5_roc.csv
├── V6_runs.csv / V6_summary.csv
└── V7_runs.csv / V7_summary.csv
```
