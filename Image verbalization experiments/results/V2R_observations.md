# EXP-V2R Observations — ReAct Agentic vs ReAct Template

**Date**: 2026-05-24  
**New data**: `V2R_runs_20260524_014813.csv` (60 Condition B trials — 3 scenes × 4 models × 5 runs)  
**Condition A source**: `V2_runs_20260523_233838.csv` (react technique, 3 scenes × 4 models × 5 runs)  
**Pipeline**: YOLO-World (17 hazard classes) + Open-CLIP ViT-B-32 → LLM (2-call agentic vs single-pass template)  
**Previous run** (COCO YOLOv8n, 2026-05-21): `V2R_runs_20260521_034438.csv` — superseded, kept for comparison  

---

## Experimental Design

| Condition | Description |
|-----------|-------------|
| **A — react_template** | Single pass: [YOLO-World+CLIP metadata + image] → model writes REASON/OBSERVE/ACT → classify |
| **B — react_agentic** | 2-call loop: Call 1 [image only] → model observes. Call 2 [observation + YOLO-World+CLIP] → final classification |

Scenes selected: the 3 worst react_template failures in V2 (YOLO-World run):
- `person_near` (template avg 35%) — truth: hazard
- `dim_light` (template avg 25%) — truth: caution
- `door_open` (template avg 55%) — truth: safe (kept for continuity with old V2R)

---

## Summary: Template vs Agentic per Model (3 scenes, N=15 each)

| Condition       | Model       | Accuracy | 95% CI         | Latency  | Cost/call | vs Template |
|-----------------|-------------|----------|----------------|----------|-----------|-------------|
| react_template  | gpt4o_mini  | **66.7%**| [0.417, 0.848] | 3946ms   | $0.00054  | —           |
| react_template  | gpt4o       | 33.3%    | [0.152, 0.583] | 3203ms   | $0.00363  | —           |
| react_template  | claude      | 26.7%    | [0.109, 0.520] | 9054ms   | $0.00558  | —           |
| react_template  | gemini      | 26.7%    | [0.109, 0.520] | 2557ms   | $0.00010  | —           |
| react_agentic   | gpt4o_mini  | 53.3%    | [0.301, 0.752] | 8025ms   | $0.00064  | **−13.4pp** |
| react_agentic   | gpt4o       | 20.0%    | [0.071, 0.452] | 6059ms   | $0.00741  | **−13.3pp** |
| react_agentic   | claude      | 26.7%    | [0.109, 0.520] | 14473ms  | $0.00869  | 0           |
| react_agentic   | gemini      | 26.7%    | [0.109, 0.520] | 4676ms   | $0.00015  | 0           |

**Overall: template avg 38% → agentic avg 32% (−6pp).** Agentic loop is net negative with YOLO-World+CLIP pipeline.

---

## Per-Scene Breakdown

### person_near (truth = hazard)

| Model       | Template          | Agentic            | Delta    |
|-------------|-------------------|--------------------|----------|
| claude      | 60% (hazard×3, caution×2) | 40% (hazard×2, caution×3) | −20pp |
| gpt4o       | 0% (caution×5)   | 0% (caution×5)     | 0        |
| gpt4o_mini  | 0% (caution×5)   | 0% (caution×5)     | 0        |
| gemini      | **80%** (hazard×4, caution×1) | 0% (caution×5) | **−80pp** |
| **All avg** | **35%**           | **10%**            | **−25pp** |

Gemini's 80pp regression is the most dramatic result in V2R. With YOLO-World template, Gemini correctly escalates person_near to hazard 4/5 times. With the agentic loop, Gemini observes the image first and commits to "caution" in Call 1 — Call 2 YOLO-World confirmation of person at 0.31m advisory cannot override this prior.

### dim_light (truth = caution)

| Model       | Template          | Agentic             | Delta    |
|-------------|-------------------|---------------------|----------|
| claude      | 0% (safe×5)      | 0% (hazard×2, safe×1, empty×2) | 0 |
| gpt4o       | 0% (hazard×5)    | 0% (hazard×5)       | 0        |
| gpt4o_mini  | **100%** (caution×5) | 60% (caution×3, hazard×2) | **−40pp** |
| gemini      | 0% (hazard×5)    | **60%** (caution×3, hazard×2) | **+60pp** |
| **All avg** | **25%**           | **30%**             | **+5pp** |

Mini and Gemini swap: Mini's perfect template performance degrades with the agentic loop; Gemini recovers from 0% to 60% with delayed feedback. Net effect is marginal (+5pp). Claude and GPT-4o fail in both conditions — dim_light's caution tier is not recoverable via prompting structure for these models.

### door_open (truth = safe)

| Model       | Template          | Agentic             | Delta    |
|-------------|-------------------|---------------------|----------|
| claude      | 20% (safe×1, hazard×4) | 40% (safe×2, hazard×3) | +20pp |
| gpt4o       | **100%** (safe×5) | 60% (safe×3, hazard×1, caution×1) | **−40pp** |
| gpt4o_mini  | **100%** (safe×5) | **100%** (safe×5)   | 0        |
| gemini      | 0% (caution×5)   | 20% (safe×1, caution×4) | +20pp |
| **All avg** | **55%**           | **55%**             | **0pp** |

GPT-4o's template on door_open is now perfect (100%) with YOLO-World — the agentic loop breaks it (→ 60%). GPT-4o Mini maintains 100% in both conditions (consistent with old V2R). Claude and Gemini slightly improve with agentic. Net effect: zero.

---

## Comparison: Old V2R (COCO) vs New V2R (YOLO-World)

| Metric | Old V2R (COCO, door_open only) | New V2R (YOLO-World, 3 scenes) |
|--------|-------------------------------|-------------------------------|
| Template avg | 5% (door_open) | 38% (3 scenes) |
| Agentic avg  | 35% (door_open) | 32% (3 scenes) |
| Delta        | **+30pp (feedback helps)** | **−6pp (feedback hurts)** |
| GPT-4o Mini  | 0% → 100% on door_open | 100% → 100% on door_open (maintained) |
| GPT-4o       | 20% → 40% on door_open | 100% → 60% on door_open (degraded) |
| Gemini       | 0% → 0% (no effect) | person_near 80% → 0% (massive regression) |

---

## Observations

**O1 — Agentic loop does not help with YOLO-World+CLIP pipeline (net −6pp)**  
In the old COCO run, react_template collapsed to 5% on door_open; the agentic loop recovered it to 35% (+30pp). With YOLO-World+CLIP, the template already achieves 38% across the three worst react scenes — and the agentic loop reduces this to 32%. The improvement seen in old V2R was the feedback loop compensating for weak COCO metadata. With rich YOLO-World metadata in the template, models are already well-informed; the 2-call structure adds latency and cost with no accuracy benefit.

**O2 — Agentic loop causes "prior lock-in" — models commit to wrong beliefs in Call 1**  
The core failure mode with YOLO-World: when models observe the image WITHOUT sensor data (Call 1), they form a belief about risk level. Call 2 provides YOLO-World+CLIP confirmation, but the model has already anchored. For person_near: Call 1 observation says "person in frame, caution" — Call 2 YOLO confirms person at 0.31m advisory — model still outputs caution (not hazard). The close-proximity threshold requires explicit hazard metadata at Call 1 time, not as a correction in Call 2.  
**Implication**: delayed feedback is inferior to synchronised metadata when the sensor data is hazard-specific and reliable.

**O3 — Gemini: person_near regression 80% → 0% is the sharpest prior lock-in example**  
Gemini is the clearest demonstration of lock-in. With template (Call 1 has YOLO data), Gemini escalates person at 0.31m to hazard 4/5 times. With agentic (Call 1 image-only), Gemini observes the cluttered lab scene, concludes "caution" (person present but not immediately dangerous from visual appearance), then receives YOLO confirmation — but the caution prior is fixed. 5/5 runs output caution in the agentic condition. Gemini's caution-tier over-specification from V1 now manifests as *under*-escalation when hazard data arrives late.

**O4 — GPT-4o template on door_open is now perfect (100%) — agentic breaks it**  
In old V2R, GPT-4o template scored 20% on door_open (the entire point was that it needed a feedback loop). With YOLO-World template, GPT-4o correctly classifies door_open 5/5 (safe). The agentic loop degrades this to 60% — GPT-4o observes the doorway, notes "opening — potentially hazard" in Call 1, then cannot fully override this despite YOLO confirming no obstacles. This is the inverse of the old V2R finding: pipeline quality has made the feedback loop redundant for this model+scene.

**O5 — GPT-4o Mini + door_open: 100% in both conditions (replicates old V2R)**  
The one consistent result across both V2R runs: Mini maintains 100% on door_open regardless of condition. In the agentic condition, Mini observes neutrally, receives YOLO (no detections, clear path), and correctly outputs safe every time. Mini's door_open classification is robust to prompting structure. However, Mini's dim_light performance degrades from 100% template to 60% agentic — the benefit on door_open does not generalise.

**O6 — Agentic costs 1.5×–2.0× more per call with no accuracy improvement**  
Two-call latency is 2× for all models (template: 2557–9054ms vs agentic: 4676–14473ms). Cost increase is 1.2×–2.0×. GPT-4o cost doubles ($0.0036 → $0.0074) — the most expensive agentic model at lower accuracy. For a real-time drone copilot (target <2s Tier 3 response), agentic loop is architecturally infeasible regardless of accuracy trade-offs.

---

## Conclusion (Updated)

**YOLO-World+CLIP pipeline makes the agentic feedback loop unnecessary for open-loop vision classification.**

Old conclusion (COCO pipeline): *"ReAct adds value only when the feedback loop is real — without it, react_template is a verbose zero_shot."*

**New conclusion (YOLO-World pipeline)**: The feedback loop added value with COCO because the metadata was weak — the loop compensated by letting models form a prior from visual observation alone. With YOLO-World+CLIP providing hazard-specific, distance-aware metadata, the template already contains everything the model needs. The agentic loop is net negative (−6pp) because it delays metadata delivery, causes prior lock-in in Call 1, and doubles latency/cost.

**For the thesis**: The V2R result with YOLO-World+CLIP is the stronger finding. It shows that when sensor metadata quality is high, prompting strategy converges — template, agentic, CoT all approach similar performance, and the pipeline quality is the dominant factor. This directly reinforces V1 and V2's main conclusion: *build a better sensor tier, not a more complex prompting strategy.*

**C-series justification remains unchanged**: C-series ReAct works because the model receives actual tool return values (drone position, PID state, motor commands) at each step — real feedback from the physical world. V2R tests prompting structure on static frames. The two are architecturally distinct; V2R's negative result does not contradict C-series.
