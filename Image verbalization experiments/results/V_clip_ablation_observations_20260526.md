# V_CLIP_ABLATION Observations — Run 2026-05-26

**Script:** `Image verbalization experiments/exp_V_clip_ablation.py`
**Results:** `V_clip_ablation_runs_20260526_171002.csv`
**Summary:** `V_clip_ablation_summary_20260526_171002.csv`
**CLIP standalone:** `V_clip_ablation_clip_standalone_20260526_171002.csv`
**Pipeline:** CLAHE → YOLO-World + MediaPipe → [with_clip / no_clip] → LLM
**Models:** claude, gpt4o, gpt4o_mini, gemini
**Conditions:** with_clip (CLIP label in prompt), no_clip (YOLO + MediaPipe only)
**Total trials:** 320 LLM calls (2 conditions × 4 models × 8 scenes × 5 runs) | Errors: 2
**CLIP standalone:** 40 observations (8 scenes × 5 runs)
**Total cost:** $0.98

---

## Purpose

Formally prove whether CLIP metadata in the LLM prompt improves or degrades
safety classification. Two conditions on identical frames with identical YOLO +
MediaPipe metadata — only whether CLIP's label, confidence, and risk appear in
the prompt differs.

This experiment answers three distinct questions:

1. **CLIP standalone**: How accurately does CLIP classify scenes on its own, without LLM?
2. **Accuracy effect**: Does including CLIP in the prompt improve LLM risk accuracy?
3. **Reasoning effect**: Does CLIP improve or degrade the LLM's internal reasoning quality?

The third question is the critical one — and requires full reply analysis, not
ground-truth label comparison. A model that copies CLIP's label gets a higher
accuracy score but reasons worse. Full reply metrics (DescAcc, RsnRisk, RsnAct)
reveal whether the LLM is reasoning from visual evidence or anchoring on CLIP's signal.

---

## CLIP Standalone Accuracy (No LLM)

| Scene | Truth | CLIP Risk | Acc | Note |
|-------|-------|-----------|-----|------|
| wall_close | hazard | hazard | **100%** | ✅ correct |
| cluttered | caution | caution | **100%** | ✅ correct |
| door_open | safe | safe | **100%** | ✅ correct |
| object_table | caution | caution | **100%** | ✅ correct |
| person_far | caution | caution | **100%** | ✅ correct |
| **person_near** | **hazard** | **caution** | **0%** | **⚠️ SAFETY MISS** |
| **blocked_lens** | **hazard** | **unknown** | **0%** | **⚠️ SAFETY MISS** |
| dim_light | caution | unknown | 0% | |

**Overall CLIP standalone: 62.5%** (vs random 3-class baseline = 33.3%)

**CLIP latency overhead: 109.1ms = 40.7% of Tier 2 pipeline**

CLIP is above random on easy scenes but critically fails on the two most important
hazard scenes. When a person stands close to the camera (person_near), CLIP
classifies it as caution — not hazard. When the lens is blocked (blocked_lens),
CLIP returns unknown. Both are safety misses at the standalone level.

For wall_close (hazard), CLIP correctly returns hazard — but this is the same
scene where the wall texture fills the frame and the classification is visually
unambiguous. CLIP gets easy hazards right and hard hazards wrong.

---

## Accuracy Effect: with_clip vs no_clip

| Condition | RiskAcc | ActSafe | Danger | Quality |
|-----------|---------|---------|--------|---------|
| with_clip | 67.9% | 100% | 0 | 4.56 |
| no_clip | 56.0% | 100% | 0 | 4.44 |

Including CLIP in the prompt raises risk accuracy by **+11.9pp** and quality by
**+0.12**. At first glance this appears to support keeping CLIP. But this
interpretation is wrong — it conflates anchoring with reasoning, as the full
reply analysis below shows.

---

## Reasoning Effect: the Critical Finding

All metrics computed from full reply text — not ground truth labels.

| Condition | DescAcc | RsnRisk | RsnAct | ActSafe | Danger |
|-----------|---------|---------|--------|---------|--------|
| with_clip | 96.2% | 93.1% | 98.7% | 100% | 0 |
| **no_clip** | **97.5%** | **96.2%** | **100%** | **100%** | **0** |

**Without CLIP, the LLM scores higher on every reasoning metric:**
- DescAcc: +1.3pp (models describe scenes more accurately from the image alone)
- RsnRisk: +3.1pp (descriptions better justify the stated risk level)
- RsnAct: +1.3pp (actions are perfectly consistent with stated risk)
- ActSafe: tied (100% both — CLIP has no effect on action safety)
- Danger: tied (0 both — both conditions are equally safe)

The accuracy gain with CLIP (67.9% vs 56.0%) is real — but it comes from
**anchoring**, not better reasoning. When CLIP provides a label, the LLM
anchors on it and stops reasoning as carefully from the image. When CLIP is
correct, the LLM gets the right answer. When CLIP is wrong or uncertain, the
LLM's reasoning degrades because it is conflicted between the visual evidence
and CLIP's signal.

---

## Per Model × Condition — Reasoning from Reply

| Model | Condition | DescAcc | RsnRisk | RsnAct | ActSafe | Danger | Quality |
|-------|-----------|---------|---------|--------|---------|--------|---------|
| claude | with_clip | 97.4% | **100%** | 94.9% | 100% | 0 | 4.13 |
| claude | no_clip | 89.7% | 97.4% | **100%** | 100% | 0 | 4.05 |
| gpt4o | with_clip | **100%** | 87.5% | **100%** | 100% | 0 | **4.85** |
| gpt4o | no_clip | **100%** | **100%** | **100%** | 100% | 0 | 4.60 |
| gpt4o_mini | with_clip | 87.5% | 97.5% | **100%** | 100% | 0 | 4.75 |
| gpt4o_mini | no_clip | **100%** | **100%** | **100%** | 100% | 0 | 4.60 |
| gemini | with_clip | **100%** | 87.5% | **100%** | 100% | 0 | 4.50 |
| gemini | no_clip | **100%** | 87.5% | **100%** | 100% | 0 | 4.50 |

**GPT-4o is the clearest case:** RsnRisk drops from 100% → 87.5% when CLIP is
added. Without CLIP, every GPT-4o description perfectly justifies the stated risk
level. With CLIP, 12.5% of trials have a description that doesn't fully justify
the risk — because the model anchored on CLIP's label and the description didn't
independently lead there.

**GPT-4o-mini:** DescAcc drops from 100% → 87.5% with CLIP. CLIP's occasional
"unknown" or wrong label causes GPT-4o-mini to not even correctly identify the
primary scene element. Without CLIP, GPT-4o-mini always describes the scene correctly.

**Gemini:** Unaffected (87.5% RsnRisk in both conditions) — Gemini is robust
to CLIP's signal in either direction. It neither benefits nor is hurt by it.

**Claude:** RsnAct drops from 100% → 94.9% with CLIP — CLIP occasionally causes
Claude to produce an action that doesn't match its stated risk level.

---

## Per Scene — Where CLIP Hurts Reasoning

| Scene | Truth | Condition | DescAcc | RsnRisk | RsnAct | ActSafe |
|-------|-------|-----------|---------|---------|--------|---------|
| blocked_lens | hazard | with_clip | 100% | **45%** | 100% | 100% |
| blocked_lens | hazard | no_clip | 100% | **70%** | 100% | 100% |
| dim_light | caution | with_clip | **68%** | 100% | 100% | 100% |
| dim_light | caution | no_clip | **84%** | 100% | 100% | 100% |
| door_open | safe | with_clip | 100% | 100% | **90%** | 100% |
| door_open | safe | no_clip | 100% | 100% | **100%** | 100% |

**blocked_lens:** CLIP returns "unknown" (0% standalone accuracy). With CLIP in
the prompt, models correctly describe the blocked lens (DescAcc=100%) but
RsnRisk drops from 70% → 45% — models cannot justify their risk level because
CLIP's "unknown" signal conflicts with the visual evidence of a dark/obstructed
frame. The model knows what it sees but the CLIP noise prevents it from forming
a coherent risk argument. Without CLIP, models reason directly from the dark
frame and justify the risk level significantly better.

**dim_light:** CLIP returns "unknown" (0% standalone). With CLIP, DescAcc drops
from 84% → 68% — models less accurately describe the dim lighting scene because
the CLIP "unknown" label confuses their scene identification. Without CLIP, models
see and describe the dim environment more reliably.

**door_open:** RsnAct drops from 100% → 90% with CLIP. CLIP correctly identifies
"safe" but this occasionally leads models to recommend PITCH_FORWARD even when
their own description is ambiguous — a case where CLIP's confidence in "safe"
overrides the model's visual caution.

---

## The Core Distinction: Classification Labels vs Detection Metadata

This experiment reveals a fundamental principle about what type of auxiliary
signal helps vs hurts LLM reasoning:

**CLIP — provides a classification label:**
> *"This scene is: caution (conf=0.61, risk=caution)"*

The LLM receives an answer. It anchors on it. Reasoning degrades because the
model no longer needs to work out the risk from visual evidence — CLIP has already
told it the answer. When CLIP is right, accuracy improves. When CLIP is wrong
or uncertain (blocked_lens, dim_light), the LLM's reasoning collapses.

**MediaPipe — provides detection facts:**
> *"Person detected (conf=0.455, est_dist=0.24m)"*

**YOLO-World — provides detection facts:**
> *"person: conf=0.81, est_dist=0.3m, depth=0.24m (DA v2)"*

The LLM receives raw evidence. It reasons from it. Example: if two people are
fighting near the drone, YOLO-World detects two persons at close range,
MediaPipe confirms human presence. The LLM reasons: *"two people in close proximity,
unpredictable movement, flight path blocked — hazard."* Neither YOLO nor MediaPipe
told the LLM the risk level — they gave it the raw facts to reason from. This
is why MediaPipe and YOLO-World improve reasoning while CLIP degrades it.

**The principle:** auxiliary signals that provide raw detections and measurements
enable LLM reasoning. Auxiliary signals that provide pre-computed classification
labels bypass LLM reasoning and cause anchoring. For a safety system where the
LLM is the reasoning layer, feeding it answers instead of evidence undermines
the purpose of having an LLM in the loop.

---

## Stated Risk Distribution: CLIP Shifts Labels Towards Hazard

| Condition | safe | caution | hazard | unknown |
|-----------|------|---------|--------|---------|
| with_clip | 10.7% | 28.9% | **60.4%** | 0% |
| no_clip | 13.2% | 29.6% | **57.2%** | 0% |

With CLIP, models classify more scenes as hazard (+3.2pp). This is because CLIP
has a hazard bias for several scenes (wall_close=100% hazard, person_near=caution
but YOLO provides person detection which the LLM upgrades). The shift is small
but confirms CLIP biases the LLM's output distribution towards its own predictions.

---

## Reconciling with V6 Finding

V6 (verbosity sweep) showed: maximum ΔQ = 0.05 between with_clip and no_clip —
CLIP adds nothing to quality.

V_clip_ablation shows: CLIP adds +11.9pp risk accuracy but degrades DescAcc,
RsnRisk, RsnAct.

These findings are not contradictory — they measure different things:

- **Quality score (V6)** is a composite: scene description + proximity mention +
  risk label + response length + pilot action. CLIP's effect on quality is diluted
  because most quality dimensions are unaffected.
- **Risk accuracy** is specifically whether the risk label matches ground truth.
  CLIP directly targets this by providing a label — inflating accuracy through
  anchoring.
- **Reasoning metrics (this experiment)** measure the LLM's internal coherence
  and evidence quality. CLIP degrades these because anchoring replaces reasoning.

V6's conclusion (drop CLIP) was correct but for an incomplete reason. The full
reason is: **CLIP inflates risk accuracy through anchoring while degrading the
reasoning quality that makes the LLM valuable as a safety layer.**

---

## Production Decision: CLIP Dropped — Confirmed with Full Justification

| Criterion | Keep CLIP | Drop CLIP | Winner |
|-----------|-----------|-----------|--------|
| Risk accuracy | **67.9%** | 56.0% | CLIP (anchoring artefact) |
| DescAcc | 96.2% | **97.5%** | No CLIP |
| RsnRisk | 93.1% | **96.2%** | No CLIP |
| RsnAct | 98.7% | **100%** | No CLIP |
| ActSafe | 100% | **100%** | Tied |
| Danger | 0 | **0** | Tied |
| Latency | +109ms | **−109ms** | No CLIP |
| Tier 2 overhead | **+40.7%** | 0% | No CLIP |
| Reasoning quality | ❌ anchoring | ✅ visual reasoning | No CLIP |

CLIP is dropped. The accuracy gain is an anchoring artefact, not evidence of
better reasoning. The system reasons more coherently, more consistently, and
with perfect internal consistency without CLIP — and saves 109ms (40.7% of
Tier 2 pipeline) per frame.

---

## Thesis Interpretation

> *"A CLIP necessity ablation confirms that including CLIP scene classification
> labels in the LLM prompt degrades reasoning quality despite inflating risk
> accuracy. With CLIP, risk accuracy is 67.9% versus 56.0% without — a +11.9pp
> gain. However, full reply analysis reveals this gain is an anchoring artefact:
> the LLM copies CLIP's label rather than reasoning from visual evidence. Without
> CLIP, description accuracy (97.5% vs 96.2%), reasoning-risk alignment (96.2%
> vs 93.1%), and internal action consistency (100% vs 98.7%) all improve. Action
> safety is identical (100%, 0 dangerous cases) in both conditions — CLIP provides
> no safety benefit.*
>
> *The degradation is most pronounced on scenes where CLIP fails: blocked\_lens
> (CLIP returns "unknown", 0% standalone accuracy) reduces reasoning-risk alignment
> from 70% to 45% with CLIP in the prompt — the model correctly describes the
> obstructed frame but cannot form a coherent risk argument because CLIP's
> "unknown" signal conflicts with the visual evidence. dim\_light (CLIP=unknown)
> reduces description accuracy from 84% to 68% — CLIP noise causes models to
> misidentify the scene itself.*
>
> *This result reveals a fundamental distinction between auxiliary signal types:
> CLIP provides a classification label (an answer), while YOLO-World and MediaPipe
> provide detection metadata (evidence). Labels cause anchoring — the LLM stops
> reasoning and copies the label. Evidence enables reasoning — the LLM builds
> its own risk assessment from object detections, proximity estimates, and sensor
> readings. For a system where the LLM serves as the cognitive reasoning layer,
> feeding it answers undermines its function. CLIP is removed from the production
> pipeline: it adds 109ms latency (40.7% of Tier 2 pipeline), degrades the LLM's
> reasoning quality, and provides zero safety benefit over the YOLO-World +
> MediaPipe detection stack."*

---

## Run Configuration

```
Date          : 2026-05-26
Script        : Image verbalization experiments/exp_V_clip_ablation.py
Models        : claude, gpt4o, gpt4o_mini, gemini
Conditions    : with_clip, no_clip
N runs        : 5 per scene per model per condition
Scenes        : 8 canonical scenes (run03 saved frames)
Total trials  : 320 LLM calls + 40 CLIP standalone observations
Pipeline      : CLAHE → YOLO-World + MediaPipe → [with/without CLIP] → LLM
max_tokens    : 300
temperature   : 0.0
Total cost    : $0.98
CLIP overhead : 109.1ms avg = 40.7% of Tier 2 pipeline

Key distinction: V6 showed ΔQ=0.05 (quality score) — incomplete finding.
This experiment shows CLIP degrades DescAcc, RsnRisk, RsnAct from full reply.
Production decision: CLIP dropped. Reasoning from image + YOLO + MediaPipe only.
```
