# Failure Analysis Observations — exp_failure_analysis

**Experiment**: EXP-FAILURE  
**Date run**: 2026-05-21  
**Result file**: failure_analysis_20260521_065913.csv  
**Trials**: 36 (3 scenes × 4 models × 3 runs)  
**Purpose**: Capture full LLM reasoning for systematic failure scenes to determine whether wrong classifications are model behaviour, prompt design failure, or upstream data (YOLO) failure.

---

## Scene 1: wall_close — Truth = hazard

**YOLO metadata**: `none` (zero detections — featureless blank wall returns nothing)

### What each model said (full reply evidence)

**Claude** (3/3 runs say caution — real accuracy = 0%):
> "The near-featureless frame and apparent proximity to a flat surface suggest the drone is in a tight or enclosed space. Slowing down and reassessing position is advisable to avoid collision."
> "The frame is very blurry and low contrast, suggesting the drone may be very close to a flat surface. The flat surface appears to be very close to the camera, potentially within 0.5–1 meter, given how the image fills the entire frame with no depth perspective."

Claude correctly identifies a flat surface is close. It reasons visually about proximity from frame fill and blur. It reaches **caution** but not **hazard**.

> **Note — scoring artifact**: Run 3 Claude is scored as `detected_risk=hazard, correct=1` in the CSV. This is wrong — the model wrote "Risk: caution". The word "hazards" appears in the phrase "no specific hazards are detected", and the keyword parser finds "hazard" as a substring. Real Claude accuracy for wall_close = 0/3.

**GPT-4o** (3/3 runs say safe — 0% accuracy):
> "The image shows a plain, featureless surface, likely a wall or ceiling, with no objects or people detected. Proximity: No objects or people are present to estimate proximity. Risk: Safe. Pilot suggested action: PROCEED"

GPT-4o gives an identical reply across all 3 runs (word for word). It sees the blank wall, finds no YOLO detections, concludes "no objects = safe", and says PROCEED. No attempt to reason about proximity from visual cues.

**GPT-4o-mini** (3/3 runs say safe — 0% accuracy):
> "The image appears to be a blank or empty space with no visible objects or people detected. The surface is smooth and uniformly lit. Proximity: No objects or people are present, so proximity cannot be estimated. Risk: safe. Pilot suggested action: PROCEED"

Same reasoning as GPT-4o. No proximity reasoning from visual frame characteristics.

**Gemini** (3/3 runs say safe — 0% accuracy):
> "The image shows a plain, light gray surface, possibly a wall or ceiling, with no discernible objects or features. There are no YOLO detections. Proximity: No objects are visible, so proximity cannot be estimated. Risk: Safe. PROCEED"

Identical across all 3 runs. Acknowledges it is a wall/ceiling but still says safe.

### Root cause verdict: YOLO BLIND SPOT + MODEL PRIOR — NOT a prompt failure

The prompt correctly instructs: *"Using both the metadata and the image"* and *"Estimate proximity of any object or person to the camera."* Claude follows this correctly and reasons from visual cues (blurry frame, fills the view, no depth perspective). GPT-4o, mini, and Gemini do not.

The failure has two layers:
1. **YOLO is blind to featureless surfaces** — a plain wall produces zero detections, so there is no metadata signal of danger
2. **Model prior**: Without YOLO input, three of four models reason "blank image + no detections = empty space = safe". This is a reasonable prior for open-air scenarios but wrong for indoor close-range wall approach.

Claude partially overcomes this via visual reasoning but only reaches caution. The conceptual gap is: models are not trained to interpret "entire frame filled with featureless surface" as a crash-imminent signal.

**Classification: YOLO limitation + model prior. The prompt is not at fault. The system architecture lacks a proximity sensor signal to resolve this.**

---

## Scene 2: door_open — Truth = safe

**YOLO metadata**: `none` (open doorway — no solid object detected in frame centre)

### What each model said

**Claude** (3/3 runs say caution — 0% accuracy):
> "The drone is inside a large industrial or commercial building, with metal shelving/racking structures, glass partitions, and fluorescent lighting. Structural elements (shelving, glass panels) are in close proximity — estimated within 1–3 meters. While no people or objects were flagged by YOLO, the drone is operating in a confined indoor space with glass partitions, metal racking, and low ceiling clearance. **Glass may not be reliably detected by standard YOLO models.** The risk of collision with static infrastructure is elevated."

Claude explicitly flags YOLO's known weakness with glass surfaces and uses this as justification for upgrading to caution. This is reasoned conservatism — not hallucination.

**GPT-4o** (3/3 runs say safe — **100% accuracy** ✓):
> "The image shows an indoor space with a row of tables or workstations and a glass door in the foreground. The area appears to be empty. The glass door is very close to the camera, likely within a few feet. Risk: Safe. Pilot suggested action: PROCEED"

GPT-4o sees the glass door as close, notes the area is empty, and correctly classifies as safe with PROCEED.

**GPT-4o-mini** (3/3 runs say safe — **100% accuracy** ✓):
> "The image shows an indoor space with a series of tables or workstations and glass doors leading to another area. The lighting is bright and the environment appears organized and clean. Risk: safe. Pilot suggested action: PROCEED"

Consistent safe classification across all 3 runs. Sees a clean, organised indoor environment.

**Gemini** (3/3 runs say caution — 0% accuracy):
> "The image shows an indoor environment with multiple glass partitions and doors with white frames, suggesting a structured interior. The closest object is a white-framed glass door on the right, appearing to be within a few feet of the camera. Risk: Caution. Pilot suggested action: SLOW_DOWN"

Gemini flags the nearby glass door as a proximity risk and applies caution. Identical reply across all 3 runs.

### Root cause verdict: MODEL-SPECIFIC VISUAL BIAS — NOT a prompt failure

All four models used the same structured V1 prompt and the same image. The prompt is not the issue — GPT-4o and GPT-4o-mini answer correctly 100% of the time.

The failure is model-specific:
- **Claude** over-weights the risk of glass surfaces specifically because it correctly knows YOLO cannot detect glass. This is a known-limitation over-correction — Claude's reasoning is technically sound ("glass = YOLO blind spot") but leads to a false positive in this case.
- **Gemini** treats visible glass door frames as proximity obstacles and applies caution conservatively. It does not reason about navigability.
- **GPT-4o / GPT-4o-mini** look at the same frame and read it as "clear indoor space with visible path ahead" → safe.

**Classification: Model-specific conservative visual bias. Claude and Gemini have an indoor navigation prior that treats visible structural elements as obstacles. GPT-4o/mini correctly interpret a navigable open doorway. This is not a prompt failure — it is a difference in model training priors.**

**Implication**: For this system, GPT-4o's classification is correct and more useful. Claude/Gemini's conservatism creates false positives that would cause unnecessary SLOW_DOWN commands in safe open environments.

---

## Scene 3: person_far — Truth = safe

**YOLO metadata**: `person (conf=0.59, est_dist~0.57m, bbox=[154.3,110.4,193.5,236.3])`

> The YOLO distance estimate (0.57m) is **wrong** — the person is visually several meters away. This is a YOLO calibration error for this specific frame.

### What each model said

**Claude** (0/3 accuracy — caution/hazard/hazard):

Run 1 (caution):
> "The YOLO detector estimates the person is approximately **0.57 meters** from the camera, which is extremely close — **though this may reflect a calibration artifact given the image perspective; visually the person appears to be several meters away in the mid-ground.** Regardless of actual distance, the presence of a human in an enclosed industrial environment warrants reduced speed."

Claude explicitly identifies the discrepancy between YOLO's distance estimate and the visual evidence. It correctly reasons that 0.57m is inconsistent with what the image shows. But it still rates caution — erring on the side of YOLO metadata.

Runs 2 and 3 (hazard):
> "The YOLO detector estimates the person is approximately **0.57 meters** from the camera — this is extremely close and likely an indoor confined space with limited maneuver room. Risk: hazard. STOP."

By runs 2 and 3, Claude drops the visual discrepancy caveat and accepts the YOLO distance, escalating to hazard/STOP. The reasoning collapses to trusting metadata fully.

**GPT-4o** (0/3 accuracy — all caution):
> "There is a person detected with a confidence of 0.5 at an estimated distance of 0.57 meters. Risk: Caution. Pilot suggested action: HOLD / SLOW_DOWN"

GPT-4o repeats the YOLO distance verbatim across all 3 runs without visual cross-checking. No mention of any discrepancy between the metadata and the image.

**GPT-4o-mini** (0/3 accuracy — all caution):
> "There is a detected person in the frame with a confidence of 0.5. The estimated distance to the person is approximately 0.57 meters. Risk: caution. SLOW_DOWN"

Identical reasoning to GPT-4o. Accepts YOLO distance as ground truth with no visual verification.

**Gemini** (0/3 accuracy — all hazard):
> "A person is standing in what appears to be an indoor setting. The person is approximately 0.57 meters from the camera. Risk: Hazard. Pilot suggested action: STOP"

Gemini is the most aggressive — accepts 0.57m without question, immediately classifies hazard/STOP across all 3 runs. No acknowledgment of visual evidence. Gives STOP for what is actually a safe scene with a distant person.

### Root cause verdict: YOLO METADATA CORRUPTION — NOT a model behaviour or prompt failure

The YOLO distance estimate (0.57m) is incorrect for this frame. All four models follow the prompt instruction to use YOLO metadata, trust the 0.57m figure, and respond accordingly. This is correct behaviour given the prompt — the models are doing what they are instructed to do.

**The failure is upstream — in the YOLO distance calibration.**

The most important finding here is the difference in model response to the conflict:
- **Claude (run 1)** is the only model that cross-validates YOLO metadata against visual evidence and explicitly identifies the discrepancy ("calibration artifact", "visually several meters away"). This is the correct behaviour for a safety-critical system.
- **Claude (runs 2–3)** abandons this reasoning and trusts YOLO — showing inconsistency even at temperature=0.0.
- **GPT-4o, GPT-4o-mini** do not attempt any visual cross-validation — they accept the YOLO distance as ground truth.
- **Gemini** is the most blindly trusting — hazard/STOP every time with no nuance.

**Classification: YOLO calibration error. All models are following their instructions correctly. The pipeline design vulnerability is that bad YOLO distance metadata overrides correct visual perception — even when the model (Claude, run 1) explicitly identifies the conflict.**

---

## Summary Table

| Scene | Truth | Failure type | Claude | GPT-4o | GPT-4o-mini | Gemini |
|-------|-------|-------------|--------|--------|-------------|--------|
| wall_close | hazard | YOLO blind spot + model prior | caution (0%) | safe (0%) | safe (0%) | safe (0%) |
| door_open | safe | Model visual bias | caution (0%) | **safe (100%)** | **safe (100%)** | caution (0%) |
| person_far | safe | YOLO metadata error | caution/hazard (0%) | caution (0%) | caution (0%) | hazard (0%) |

---

## Three Distinct Failure Types

### Type A — YOLO Blind Spot (wall_close)
YOLO cannot detect featureless surfaces. Zero detections + blank image → models default to "no obstacles = safe". Claude partially overcomes this via visual frame reasoning (blurry fill = proximity) but only reaches caution. The system needs either a dedicated proximity/distance sensor, or a model instruction to treat "featureless blank frame + no YOLO detections" as a proximity warning.

### Type B — Model Conservative Visual Bias (door_open)
Claude and Gemini over-classify indoor structural elements (glass, frames, racking) as collision risks even when a clear path exists. Claude's reasoning is explicit: "glass is a YOLO blind spot, so I over-correct." GPT-4o and mini correctly read the navigable scene. This is a model-prior difference, not a prompt issue — the same prompt produces correct results for GPT-4o/mini.

### Type C — YOLO Metadata Corruption (person_far)
Wrong YOLO distance estimate (0.57m for a far person) poisons all model responses. All models trust the metadata and escalate risk. Claude identifies the visual-metadata conflict in run 1 but abandons this reasoning in runs 2–3. This exposes a critical pipeline design gap: when YOLO distance is wrong, the LLM has no mechanism to override it reliably. Gemini is the most dangerous here — instant hazard/STOP for a safe scene.

---

## Thesis Implications

1. **The prompt is not the primary failure cause** for any of the three scenes. All models receive the same structured prompt; the failures arise from YOLO limitations (Type A, C) or model-specific visual priors (Type B).

2. **YOLO distance calibration is safety-critical.** Type C shows that a wrong distance estimate in YOLO metadata will cause ALL models to misclassify, even when the model's own visual perception is correct. This is the most dangerous failure mode for real deployment.

3. **Claude has the best cross-validation reasoning** — it is the only model that explicitly identifies the YOLO-visual conflict in person_far run 1. However, this reasoning is inconsistent (lost by runs 2–3 at the same temperature). For safety-critical systems, inconsistency at t=0.0 is a liability.

4. **GPT-4o/mini are best calibrated for open indoor spaces** — 100% accuracy on door_open, no false positives. They are less reactive to visual complexity.

5. **Gemini is the most aggressive** — hazard/STOP for a safe scene with a distant person (person_far). Conservative bias is safer than dangerous overconfidence, but STOP commands in a safe environment are operationally disruptive.

6. **Architectural recommendation**: For the LLM copilot layer, Type C failures suggest adding a prompt instruction for visual-metadata cross-validation: *"If the YOLO distance estimate appears inconsistent with the visual size of the object in the image, note the discrepancy and base your risk classification on visual evidence."*
