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

---

## System Architecture (Tier 2 — updated)

The image verbalization experiments use a three-tier pipeline:

```
Tier 1  →  PID Controller              (reflexes, 4 kHz, motor corrections)
Tier 2  →  Perception stack            (~30 fps, passes metadata to LLM)
            ├─ robust_local_detector   — EMERGENCY trigger (14 ms, no API)
            │   ├─ MediaPipe           — EfficientDet-Lite0 person detection [44,45]
            │   ├─ Texture uniformity  — Sobel gradient → wall fills frame
            │   └─ Brightness gate     — blocked lens / total darkness
            ├─ YOLOv11n (COCO)         — person + 80-class trained detection [35,39]
            ├─ YOLO-World              — open-vocab hazards: structural + threat vocab [37]
            │   ├─ Structural classes  — wall, wire, pillar, steps, barrier, ceiling…
            │   └─ Threat classes      — gun, pistol, rifle, explosive, knife, bomb…
            │       (zero-shot, ~30–50% recall on 320×240; demo-validated, no experiment)
            │       Fine-tuned YOLOv11n on Open Images V7 weapons = future extension
            ├─ DepthAnything v2        — real metric depth per object (metres) [36,42,43]
            │  Metric Indoor
            └─ CLIP [experimental]     — scene-level label (5 categories)
                NOTE: CLIP removed from production pipeline. Retained for V-series
                thesis experiments only. Proved unreliable on 320×240 ESP32 frames
                (scores within ±0.013 of 0.200 uniform baseline = effectively random).
                This failure result motivates the LLM cognitive authority claim.
Tier 3  →  LLM (cognitive layer)       (every 1–10 s scheduled + emergency trigger)
            PRIMARY threat classifier — identifies weapons, gestures, semantic threats
            that have no COCO label. Only layer that understands intent and context.
```

**Authority rule**: LLM is the cognitive authority. Tier 2 metadata is advisory.
If the image contradicts YOLO detections, the LLM's visual judgment takes precedence.

**Emergency trigger**: `robust_local_detector` runs at 14 ms (no API call) and triggers
the LLM immediately when a local hazard is detected (person close, wall fills frame,
or darkness/blocked lens). Scheduled LLM calls handle caution scenes.

**CLIP status**: Disabled in production (`use_clip=False` default in `enhanced_yolo_infer()`).
The CLIP code and experimental results are retained in `enhanced_yolo_pipeline.py`
and used in V-series experiments to demonstrate *why* a vision-capable LLM is needed
as the cognitive authority — CLIP's failure on real hardware is the evidence.

---

## Observation — Threat Detection Capability (Demo-Validated)

> **No formal experiment run. Validated by demo video. Suitable for thesis discussion section.**

### YOLO-World vocabulary extension for threats

`YOLO-World` supports open-vocabulary detection via `model.set_classes()`. The
`THREAT_CLASSES` list (`enhanced_yolo_pipeline.py`) extends the structural vocabulary
to include security threats:

```
gun, pistol, rifle, firearm, weapon,
knife, machete, axe,
explosive, bomb, grenade, suspicious package
```

**Observed behaviour on ESP32 frames (demo video):**
- YOLO-World zero-shot recalls the weapon shape ~30–50% of attempts at 320×240
- False positives on dark cylindrical objects (water bottle ↔ pistol)
- Useful as supplementary metadata but NOT a reliable hard trigger
- `enhanced_rule_risk()` maps any THREAT_RISK_CLASS detection → immediate hazard

### Fine-tuned YOLOv11n on weapons (future extension)

Fine-tuning YOLOv11n on [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
(which has `Firearm`, `Gun`, `Handgun`, `Rifle`, `Knife` labels, ~2000 images) would
push recall to ~75–85% on indoor frames — comparable to COCO-trained person detection.
Stub function `load_threat_yolo(weights_path)` is ready in `enhanced_yolo_pipeline.py`.
Not done in this thesis work — left as future work.

### Why the LLM is still the primary threat layer

Demo observation: even when YOLO-World **misses** the weapon entirely
(`YOLO detections: none`), the LLM correctly describes the scene as:

> *"A person is standing with their arm extended, holding what appears to be a handgun
> pointed toward the camera. Risk: hazard. Pilot suggested action: LAND"*

This confirms the architectural principle: **Tier 3 LLM is the only layer capable of
open-ended semantic threat classification.** YOLO-World threat vocabulary is a
best-effort early signal that reduces LLM reasoning time; it does not replace the LLM.

---

## References — Image Verbalization Chapter

Papers directly supporting the V-series and G-series experiment design,
Tier 2 pipeline choices, and LLM evaluation methodology.

**Three-tier architecture & LLM cognitive layer**

- [35] Ahmmad et al. (2025). *Autonomous Navigation of Cloud-Controlled Quadcopters
  in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic
  Reasoning.* arXiv:2508.07885.
  → Peer-reviewed validation of YOLOv11 + Depth Anything V2 + cloud LLM pipeline.
    Depth MAE = 7.2 cm; end-to-end latency < 1 s; 42 indoor trials.

- [7] Vemprala et al. (2023). *ChatGPT for Robotics: Design Principles and Model
  Abilities.* Microsoft Autonomous Systems. (Already in paper draft — CLAIM 2 basis.)

**Tier 2 — Object detection**

- [37] Cheng et al. (2024). *YOLO-World: Real-Time Open-Vocabulary Object Detection.*
  CVPR 2024. arXiv:2401.17270.
  → Justifies YOLO-World for structural hazard detection (wall, wire, barrier).
    Used in V2, V5, V6, V7, G1, G2 experiments as part of Tier 2.

- [39] Kim et al. (2024). *YOLO-IHD: Improved Real-Time Human Detection System for
  Indoor Drones.* Sensors / PMC10857234.
  → Confirms COCO-trained YOLO (80% precision) outperforms zero-shot YOLO-World for
    person detection indoors. Justifies dual-YOLO Tier 2 architecture.

- [38] Wang et al. (2024). *YOLOv10: Real-Time End-to-End Object Detection.*
  arXiv:2405.14458.
  → Contextual: YOLO model family progression. YOLOv11 (used here) follows this work.

**Tier 2 — Emergency local detector (robust_local_detector)**

- [44] Tan, M., Pang, R., Le, Q.V. (2020). *EfficientDet: Scalable and Efficient
  Object Detection.* CVPR 2020. arXiv:1911.09070.
  → EfficientDet-Lite0 (the smallest EfficientDet variant, ~13 MB) is used in
    `robust_local_detector` for person detection on 320×240 ESP32 frames.
    Achieves ~14 ms CPU inference with substantially higher recall than zero-shot
    YOLO-World on low-resolution indoor frames. No GPU required.

- [45] Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception
  Pipelines.* CVPR Workshop on CV for the Real World. arXiv:1906.08172.
  → MediaPipe provides the inference runtime for EfficientDet-Lite0 in the
    emergency local detector. The framework handles model loading, pre/post-
    processing, and CPU execution, enabling 14 ms end-to-end person detection
    latency on the companion computer without a GPU.
    Ref: `robust_local_detector.py` — `load_mediapipe_detector()`.

**Tier 2 — Monocular depth estimation**

- [36] Yang et al. (2024). *Depth Anything V2.* NeurIPS 2024. arXiv:2406.09414.
  → Primary depth model for Tier 2. Metric Indoor variant gives real metre-scale
    distances per object. Replaces broken geometric heuristic. Validated in G3.

- [42] Bui et al. (2024). *Monocular Depth Estimation for Drone Obstacle Avoidance
  in Indoor Environments.* IEEE CAI 2024. DOI:10.1109/CAI59483.2024.10802577.
  → Confirms monocular depth is viable for indoor nano-drone obstacle avoidance at
    the 0.5–3 m range used in this system's operating zone.

- [40] Piccinelli et al. (2024). *UniDepth: Universal Monocular Metric Depth
  Estimation.* CVPR 2024.
  → Alternative metric depth model compared against DA v2 in G3; DA v2 selected
    for faster CPU inference on the companion computer.

- [41] Ranftl et al. (2022). *Towards Robust Monocular Depth Estimation: Mixing
  Datasets for Zero-Shot Cross-Dataset Transfer.* IEEE TPAMI, 44(3). (MiDaS)
  → Baseline monocular depth model; foundational work motivating DA v2's approach.

- [43] Hu et al. (2025). *Survey on Monocular Metric Depth Estimation.*
  Computers (MDPI) 14(11), 502. arXiv:2501.11841.
  → Survey establishing why metric depth (absolute metres) is required for LLM
    distance reporting; benchmarks DA v2 Metric Indoor as top indoor model.

**LLM evaluation methodology**

- [21] Liang et al. (2023). *Holistic Evaluation of Language Models (HELM).* TMLR.
  (Already in draft — justifies N ≥ 5 per condition across V/G series.)

- [10] Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language
  Models.* arXiv:2210.03629. (Already in draft — basis for V2R ReAct experiments.)
