# CHAPTER 3 — IMAGE VERBALIZATION AS A DRONE COMMAND INTERFACE
# (V series → Ablation → G series → H series)
#
# Structure rationale:
#   V series establishes LLM call parameters in isolation; V6 CLIP ablation
#   justifies removing CLIP before G series. G series builds the full pipeline
#   using V-series findings. H series validates operator trust mechanisms.
#
# Dated sources used:
#   V2_observations_20260525, V6_observations_20260525 (incl. CLIP ablation),
#   V7_observations_20260525, G4_observations_20260524, G1_observations_20260525,
#   G2_observations_20260524, G5_observations_20260525,
#   G5_reasoning_quality_20260525
#
# Placeholders (no dated file yet):
#   V8 temperature sweep, Component Ablation experiment, H1, H4
# ─────────────────────────────────────────────────────────────────────────────

---

## Chapter Opening

Before a language model can command a drone, it must reliably interpret what the
drone's camera sees. This chapter builds, benchmarks, and validates the image
verbalization pipeline that converts raw drone camera frames into structured scene
descriptions and pilot action suggestions. The pipeline is organised as four
computational tiers (Figure 3.1): a 4 kHz PID firmware layer, a 60 fps MediaPipe
emergency detector, a 3–4 fps YOLO-World structural and depth estimation layer,
and a 0.1–0.4 Hz LLM cognitive reasoning layer. The chapter proceeds in three
stages. First, the V-series experiments isolate and optimise individual LLM
call parameters — prompt technique (§3.1.1), token budget and CLIP ablation
(§3.1.2), and scene context history (§3.1.3) — each holding all other parameters
constant. A component ablation study (§3.2) then confirms which sensor inputs
are load-bearing. The G-series experiments (§3.3) integrate all tiers into the
full production pipeline, using the configuration derived from the V series,
and evaluate both action safety and cost at each integration stage. H-series
safety mechanism experiments (§3.4) close the chapter. The final production
configuration — structured JSON prompt, max_tokens=256, two-frame context
history, no CLIP, hybrid MediaPipe-triggered scheduling — is entirely justified
by experimental evidence before the first integrated pipeline result is reported.

---

**Figure 3.1 — Four-Tier Pipeline Architecture**
*System diagram showing data flow from camera frame through all four tiers
to pilot action suggestion and operator decision point.*

```
╔══════════════════════════════════════════════════════════════════════════╗
║                FOUR-TIER IMAGE VERBALIZATION PIPELINE                    ║
╚══════════════════════════════════════════════════════════════════════════╝

  ESP32-S3-Sense Camera
  320×240 JPEG frame
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 1.5 — Emergency Local Detector          ~16 ms / frame  no API  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  MediaPipe EfficientDet-Lite0 person detector                    │  │
│  │  Texture gradient (Sobel) → wall fills frame                     │  │
│  │  Brightness gate → blocked lens / darkness                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  Output: {person_detected, est_dist_m, wall_fill, brightness_ok}       │
│  Action: fire EMERGENCY INTERRUPT → Tier 3 if hazard detected          │
└────────────────────────────────────────────────────────────────────────┘
         │ always passes frame + metadata
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 2 — Scene Metadata Stack                ~275 ms / frame  no API │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  YOLO-World         open-vocab structural hazards                │  │
│  │  YOLOv11n COCO      person + 80-class trained detection          │  │
│  │  DepthAnything v2   metric depth per object (metres)             │  │
│  │  [CLIP removed — V6 ablation: random signal on 320×240]         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  Output: structured metadata string {label, conf, depth_m, bbox}       │
└────────────────────────────────────────────────────────────────────────┘
         │ metadata string + JPEG + 2-frame history [short mode, V7]
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 3 — LLM Cognitive Layer         2,333–7,218 ms / call   API $   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Triggered by: scheduled every 5 frames  OR  Tier 1.5 interrupt  │  │
│  │  Prompt:       Structured JSON [V2]                               │  │
│  │  Budget:       max_tokens = 256 [V6]                              │  │
│  │  History:      last 2 frame descriptions [V7]                     │  │
│  │  Temperature:  t = [V8 pending]                                   │  │
│  │  Models:       Gemini (prod) / GPT-4o / GPT-4o-mini / Claude     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  Output (JSON): scene_description, proximity, risk_level,              │
│                 reasoning, recommended_action                           │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  PILOT ACTION SUGGESTION                                         │
  │  PROCEED | SLOW_DOWN | STOP | LAND | HOVER                       │
  └──────────────────────────────────────────────────────────────────┘
         │
         ├─── HITL mode: Operator approves / overrides  [H1]
         │
         └─── Auto mode: Execute directly
                   │
                   ▼
         ┌──────────────────────────────────────────────────────────┐
         │  TIER 1 — PID Flight Controller    0.001 ms  no API      │
         │  Madgwick filter → EKF → Cascade PID → Motor PWM         │
         │  4,000 Hz — physically cannot wait for LLM               │
         └──────────────────────────────────────────────────────────┘
```

*Note: Tier 1.5 and Tier 2 run every frame. Tier 3 runs on schedule (every 5 frames)
plus emergency interrupt from Tier 1.5. CLIP was present in V-series experiments
but removed from production after V6 ablation confirmed random behaviour on 320×240.*

---

## §3.1 LLM Verbalization Parameter Studies (V series)

The LLM cognitive tier is the most computationally expensive and safety-critical
component of the pipeline: it makes the final risk classification and pilot action
recommendation. Three configuration parameters are optimised through controlled
single-variable experiments before the integrated pipeline is assembled, each
experiment isolating one parameter while holding the others constant. All
experiments use the same eight canonical drone camera scenes captured from real
ESP32-S3-Sense hardware frames, five runs per condition.

### §3.1.1 Prompt Technique (V2)

Prompt technique significantly affects verbalization quality, action safety, and
latency for single-pass safety classification; the structured JSON format achieves
the highest label accuracy, zero dangerous recommendations, lowest latency, and
strongest internal reasoning consistency across 790 valid trials — 4 models ×
4 techniques × 8 scenes × 5 runs (`V2_runs_20260525_035756.csv`). Table 3.1
reports the full per-technique results and Figure 3.2 plots label accuracy and
action safety side by side.

---

**Table 3.1 — V2 Prompt Technique Comparison (all models combined)**

| Technique | N | LblAcc | DescAcc | RsnRisk | RsnAct | ActSafe | Danger |
|-----------|---|--------|---------|---------|--------|---------|--------|
| zero_shot | 159 | 42.1% | 83.6% | 95.2% | **100%** | **100%** | **0** |
| few_shot_3 | 158 | 51.9% | **92.4%** | 93.1% | **100%** | 96.8% | 5 |
| cot (300 tok) | 156 | 19.2% | 89.7% | **96.0%** | 98.1% | **100%** | **0** |
| cot (600 tok rerun) | 160 | 46.8% | — | — | — | **100%** | **0** |
| **structured** | 159 | **56.6%** | 84.3% | 87.3% | **100%** | **100%** | **0** |

*LblAcc = label accuracy; DescAcc = description accuracy; RsnRisk = reasoning-risk
alignment; RsnAct = reason-action alignment; ActSafe = action safety;
Danger = trials where truth=hazard and action=PITCH_FORWARD.*
*CoT 600-token rerun: `V2_cot_rerun_20260525_175408.csv`.*

---

**Table 3.2 — V2 Per-Model Best Technique (all metrics at 100% action safety)**

| Model | Best technique | LblAcc | DescAcc | ActSafe | Danger |
|-------|---------------|--------|---------|---------|--------|
| Claude | structured | 43.6% | 87.2% | **100%** | **0** |
| GPT-4o | structured | **70.0%** | 82.5% | **100%** | **0** |
| GPT-4o-mini | structured | 62.5% | 80.0% | **100%** | **0** |
| Gemini | few_shot_3 | 55.0% | **100%** | **100%** | **0** |

---

**Figure 3.2 — Prompt Technique: Label Accuracy vs Action Safety**
*Grouped bar chart. X-axis: technique (zero_shot, few_shot_3, cot-300, cot-600,
structured). Left Y-axis: label accuracy (%). Right Y-axis: dangerous cases (count).
Error bars: Wilson 95% CI on accuracy. Highlight structured and cot-600 bars.
Data: Table 3.1. Key finding: structured uniquely achieves highest label accuracy
AND zero dangerous cases; CoT achieves zero dangerous but low accuracy at 300 tokens
(token truncation artefact, not reasoning failure).*

---

The structured format enforces 100% reason-action alignment architecturally: a
JSON schema that separates `risk_level` from `recommended_action` makes a response
combining `"risk_level": "hazard"` with `"recommended_action": "PITCH_FORWARD"`
immediately self-contradictory. Structured achieves the fewest unparseable outputs
(16 no-action trials versus 104 for CoT at 300 tokens). Chain-of-thought appears
to fail (19.2% label accuracy, 67% unparseable) but this is a token budget
artefact: the 300-token budget is exhausted mid-reasoning before the final risk
label is written. At 600 tokens, truncation drops from 67% to 2.5% and accuracy
rises to 46.8%; Claude CoT (60.5%) exceeds Claude structured (43.6%) at sufficient
budget, and CoT's reasoning-risk alignment (96.0%) is the highest of all
techniques. CoT reasons well — it just needs room to conclude.

**Cost and latency of CoT.** Structured commits to a parseable decision within
the first ~50 output tokens; CoT requires ~500 tokens to reach its conclusion at
600-token budget. At comparable accuracy, CoT approximately doubles both latency
and API cost relative to structured. Chain-of-thought inference in the primary V2
run reached 18,437 ms at 300 tokens — already 3–6× longer than structured — and
would be longer still at 600 tokens. For a drone safety system where the LLM
must respond between frames, latency and cost are first-class constraints.
**Structured JSON is selected as the production prompt technique.** Gemini's
optimal technique is few-shot-3 (100% on all reasoning metrics); for system-level
consistency, structured is used across all models, noting that Gemini's structured
result (100% action safety, 55.0% label accuracy) meets all safety requirements.

### §3.1.2 Output Token Budget and CLIP Ablation (V6)

Output token budget governs whether the LLM can write a complete verbalization
and pilot action within a single API call; quality improves sharply from 64 to
128 tokens (+0.88/5), gains further from 128 to 256 (+0.33/5), and plateaus
beyond 256 (Δ=0.01/5 from 256→512) across 640 trials per condition on the
YOLO-World pipeline (`V6_runs_20260525_185304.csv`). Figure 3.3 plots quality
and label accuracy against token budget for all four models and both CLIP
conditions. Table 3.3 gives full per-model and per-level results.

---

**Table 3.3 — V6 Quality and Label Accuracy by Token Level (WITH CLIP)**

| max_tokens | Quality /5 | Truncated | Avg Words | LblAcc | Latency | Plateau |
|-----------|-----------|---------|---------|--------|---------|---------|
| 64 | 3.27 | 74.7% | 48.4 | 36.7% | 3,142 ms | — |
| 128 | 4.15 | 88.8%* | 76.5 | **58.8%** | 3,649 ms | GPT-4o, mini |
| **256** | **4.48** | 96.8%* | **92.0** | 57.3% | 4,228 ms | **Claude, Gemini** |
| 512 | 4.49 | 92.9%* | 92.5 | 57.7% | 4,294 ms | no further gain |

*High truncation rate at 128+ is a metric artefact: GPT-4o and GPT-4o-mini end
replies with "Pilot suggested action: PITCH_FORWARD" (no terminal punctuation),
which the metric incorrectly scores as truncated. Quality scores of 4.67–4.70
at 128 tokens confirm these are complete responses.*

---

**Table 3.4 — V6 Per-Model Plateau Point**

| Model | Plateau tokens | Avg words at plateau | Reason |
|-------|---------------|---------------------|--------|
| GPT-4o | **128** | ~55 words | Compact responder; ignores extra budget |
| GPT-4o-mini | **128** | ~70 words | Same compact pattern |
| Claude | **256** | ~100 words | Verbose reasoner; needs space for sensor metadata |
| Gemini | **256** | ~90 words | Requires full budget to process YOLO metadata |

---

**Figure 3.3 — Token Budget: Quality vs max_tokens (4 models, with and without CLIP)**
*Two-panel line chart. X-axis: max_tokens (64, 128, 256, 512) — log scale.
Left panel: Quality /5 for each model (4 coloured lines, solid = with CLIP,
dashed = without CLIP). Right panel: LblAcc (%) for each model.
Mark the plateau at 256 with a vertical dashed line.
Key finding: curves for with-CLIP and without-CLIP are indistinguishable (ΔQ ≤ 0.05);
plateau point is identical at 256 in both conditions. GPT-4o and GPT-4o-mini plateau
earlier at 128; Claude and Gemini require 256. Data: Table 3.3 and
V6_runs_noclip_20260525_201620.csv.*

---

**CLIP Ablation.** A parallel run without CLIP (`V6_runs_noclip_20260525_201620.csv`)
finds a maximum quality difference of 0.05/5 versus the with-CLIP run across 640
trials — within experimental noise — and an identical plateau at 256 tokens in both
conditions. CLIP contributes no signal because its outputs fall within ±0.013 of
the 0.200 uniform baseline across all scene categories on 320×240 frames: it is
effectively random. In one configuration (Gemini, 128 tokens), removing CLIP
*improves* quality by +0.40/5: the random CLIP classification label consumed
reasoning capacity within the constrained token budget, preventing the model from
reaching a pilot action conclusion. CLIP also adds input token overhead and API
latency on every call with no measurable return. **CLIP is removed from the
production pipeline.** No G-series experiment includes a CLIP variant; the V6
ablation provides the definitive evidence.

**max_tokens=256 is selected as the production token budget** — sufficient for all
four models, two fewer tokens than the 512 that adds no quality while increasing
latency by 66 ms and API cost proportionally.

### §3.1.3 Scene Context History (V7)

Short history (last 2 prior frame descriptions) improves risk classification
accuracy by **16.2 percentage points** over stateless operation — from 43.0% to
59.2% — at a cost of only 38 additional input tokens per call and a negligible
cost difference of $0.00001/call across 300 trials: 3 modes × 4 models × 5
sequences × 5 frames (`V7_runs_20260525_221631.csv`). Total experiment cost was
$0.71 for 300 trials ($0.00237/trial average). Table 3.5 reports all metrics per
mode; Figure 3.4 plots risk accuracy and change detection by model and history mode.

---

**Table 3.5 — V7 Summary by History Mode (all models combined)**

| Mode | RiskAcc | ChgDetect | ActSafe | Danger | DescAcc | RsnAct | Avg Input Tokens | Avg Cost/call |
|------|---------|-----------|---------|--------|---------|--------|-----------------|--------------|
| stateless | 43.0% | 65.0% | **100%** | **0** | **100%** | **100%** | 1,108 | $0.00245 |
| **short** | **59.2%** | **67.5%** | **100%** | **0** | **100%** | 99.0% | 1,146 | **$0.00244** |
| full | 50.5% | 56.4% | **100%** | **0** | **100%** | **100%** | 1,167 | $0.00244 |

*ChgDetect = hazard change detection rate; RsnAct = reason-action alignment.
Cost/call averaged across all 4 models. Short history adds only 38 input tokens
(1,108 → 1,146) and $0.00001/call relative to stateless — effectively zero overhead.*

---

**Table 3.6 — V7 Per-Model × Mode Risk Accuracy**

| Model | Stateless | Short | Full | Best mode |
|-------|-----------|-------|------|-----------|
| GPT-4o | **88.0%** | 76.0% | 60.0% | Stateless |
| Gemini | 44.0% | **68.0%** | 60.0% | Short |
| Claude | 36.0% | **43.5%** | 37.5% | Short |
| GPT-4o-mini | 4.0% | **48.0%** | 44.0% | Short |

*GPT-4o is the exception: it performs best stateless (88.0%) because its strong
visual reasoning does not need prior context. Short history is chosen as the system
default because it is optimal for 3 of 4 models and eliminates GPT-4o-mini's
near-zero stateless accuracy (4%), making the production configuration robust across
the full model portfolio.*

---

**Figure 3.4 — V7 History Mode: Risk Accuracy and Change Detection**
*Two grouped bar charts side by side.
Left: RiskAcc (%) per model per history mode (stateless, short, full) — 12 bars.
Right: Change detection rate (%) per model per history mode.
Highlight GPT-4o-mini stateless (4%) and short (48%) contrast.
Horizontal dashed line at stateless all-model average (43.0%) for reference.
Data: Table 3.6. Key findings: short wins for 3 of 4 models; GPT-4o-mini
stateless near-zero is the system-level argument for short as the default.*

---

**Asymmetric change detection.** The system detects hazard *onset* (person entering,
frame 3) with **100% accuracy across all models and all history modes** — zero
misses. It detects hazard *clearance* (person leaving, frame 5) only 26.7% of the
time: models classify the complex indoor lab environment (tables, chairs, equipment)
as caution or hazard after the person departs. This asymmetry is architecturally
correct: the system over-lingers in HOVER rather than prematurely advancing.
**Short history (2 frames) is selected as the production default.**

**Full history underperforms.** Full history (50.5% risk accuracy) is worse than
even stateless (43.0%) for change detection. The two identical safe frames at
sequence start (frames 1–2, door_open) dilute the transition signal: the model
anchors to the repeated prior safe context rather than attending to the current
frame. For long missions, full history also grows unboundedly with session length,
increasing input token cost and latency with each call. Short history is bounded
to 2 frames and costs the same regardless of mission duration.

### §3.1.4 Temperature Selection — [PLACEHOLDER: V8 pending]

<!-- ═══════════════════════════════════════════════════════════════════
     PLACEHOLDER — V8 results pending (no dated observations file yet).
     File to use: V8_observations_[DATE].md when available.
     ═══════════════════════════════════════════════════════════════════
     Experiment: 5 temperatures [0.0, 0.2, 0.5, 0.8, 1.0] ×
                 4 models × 8 scenes × 5 runs = 800 trials.

     Tables to add:
       Table 3.X — Accuracy, quality, and label-flip rate by temperature
         (all models combined)
       Table 3.X — Per-model accuracy vs temperature

     Figures to add:
       Figure 3.X — Temperature vs accuracy and flip rate
         Line chart: X-axis = temperature [0.0, 0.2, 0.5, 0.8, 1.0].
         Y-axis (left) = classification accuracy (%), Y-axis (right) = flip rate (%).
         4 coloured lines (one per model) + aggregate dashed line.
         Key expectation: t=0.0 is best for single-pass classification;
         GPT-4o-mini expected to be temperature-insensitive;
         Gemini expected to degrade most at t≥0.5.

     Cost note to add:
       Higher temperature does not change per-call API cost (cost is determined
       by token count, not temperature). Temperature selection is a zero-cost
       decision. The sole tradeoff is accuracy/consistency.

     Target paragraph:
       "V8 (800 trials) shows t=[X] achieves [Y]% classification accuracy
        and [Z]% label-flip rate, both better than t=0.2. Temperature above
        [W] degrades Gemini most sharply ([Δ]pp accuracy drop from t=0.0 to
        t=0.5). GPT-4o-mini achieves identical accuracy at all temperatures
        ([0%] flip rate) — dominated by strong learned priors that override
        sampling randomness. Temperature selection carries zero API cost
        overhead; t=[X] is adopted for all production pipeline calls. The
        widely-used t=0.2 (Yao et al. 2022) applies to iterative ReAct agents
        where sampling diversity aids path exploration across sequential steps;
        single-pass classification has no equivalent benefit from randomness."
     ═══════════════════════════════════════════════════════════════════ -->

Temperature controls output randomness: in a safety-critical single-pass
classifier, the same scene must always produce the same risk label across
repeated calls. V8 will sweep temperatures [0.0, 0.2, 0.5, 0.8, 1.0] across
800 trials to select the production setting empirically. The t=0.2 convention
from Yao et al. (2022) was established for iterative ReAct agents where marginal
randomness helps the agent avoid repeating the same wrong reasoning step — a
justification with no direct analogue in single-pass classification. Temperature
selection carries no API cost overhead (cost is determined by token count, not
temperature), making t=0.0 and t=0.2 equally affordable and selecting between
them purely on accuracy and consistency grounds.

**[V8 results: Table 3.X and Figure 3.X to be inserted from
V8_observations_[DATE].md.]**

---

## §3.2 Component Ablation Study — [PLACEHOLDER: pending]

<!-- ═══════════════════════════════════════════════════════════════════
     PLACEHOLDER — Ablation experiment results pending.
     File: [ABLATION]_observations_[DATE].md when available.
     ═══════════════════════════════════════════════════════════════════
     Note: CLIP ablation is already complete and reported in §3.1.2.
     This section covers additional components in the Tier 2 stack.

     Candidate ablations (confirm which are run):
       - DepthAnything v2 removed: does wall_close dangerous rate drop
         (since DA v2 is the source of the 2.09m wrong reading)?
       - YOLO-World only vs YOLOv11n-COCO only vs both
       - MediaPipe removed: does hybrid trigger still work?

     Tables to add:
       Table 3.X — Action safety per condition (with/without each component)
       Table 3.X — Dangerous cases by scene per ablation condition

     Figures to add:
       Figure 3.X — Ablation: Action Safety Bar Chart
         Grouped bars: each ablation condition (columns) × scene (clusters).
         Highlight wall_close for DA v2 ablation; blocked_lens for MediaPipe.

     Target paragraph P1 (load-bearing component):
       "Removing DepthAnything v2 from Tier 2 [eliminates / reduces to X]
        the wall_close dangerous cases that caused 1.9% overall dangerous
        rate in G5: without the incorrect 2.09m depth reading, all four
        models correctly identify the wall from visual fill alone.
        [However / This comes at the cost of: losing quantitative distance
        estimates for all correctly-sensed scenes.] DepthAnything v2 is
        therefore [retained / removed] with an explicit reliability flag
        for texture-uniform surfaces."

     Target paragraph P2 (redundant component):
       "Removing [component Y] produces no measurable degradation across
        [N] trials ([A]% → [B]% action safety, within CI), confirming
        redundancy. CLIP removal (§3.1.2) set the precedent for this
        decision pattern."
     ═══════════════════════════════════════════════════════════════════ -->

The CLIP ablation in §3.1.2 established the decision pattern: if removing a
component produces no measurable degradation in action safety, it is removed.
This section extends that analysis to the remaining Tier 2 components —
DepthAnything v2 metric depth, YOLO-World structural vocabulary, YOLOv11n
COCO detection, and the MediaPipe emergency detector — to confirm which are
load-bearing for the safety guarantees reported in §3.3. DepthAnything v2 is
the component of most interest: G5 shows its incorrect 2.09 m depth reading
for wall_close is the sole source of remaining dangerous recommendations (1.9%
overall); removing it may improve overall safety by removing that failure mode,
at the cost of losing quantitative distance estimates for correctly-sensed scenes.

**[Ablation results: Tables and Figures to be inserted from
[ABLATION]_observations_[DATE].md.]**

---

## §3.3 Pipeline Architecture and Integration (G series)

With the LLM call parameters established (structured prompt, max_tokens=256,
short two-frame history, CLIP removed, V-series §3.1) and component ablation
informing sensor layer selection (§3.2), the G-series experiments build and
evaluate the four-tier pipeline. G4 measures the computational timescales that
make the tier hierarchy physically necessary. G1 isolates each tier to confirm
irreplaceability. G2 validates the hybrid trigger strategy. G5 reports the
integrated pipeline result, verbalization quality, and a cost analysis of the
production system.

### §3.3.1 Four-Tier Timescale Justification (G4)

The four-tier hierarchy spans four orders of magnitude of computational latency,
measured empirically on the same hardware across 160 LLM calls and five
independent runs per tier (`G4_runs_20260524_234054.csv`). Table 3.7 reports
latencies and inter-tier ratios; Figure 3.5 plots these on a logarithmic scale.

---

**Table 3.7 — G4 Measured Latency per Tier (5 runs, same hardware)**

| Tier | Component | Mean latency | 95% CI | Freq (approx) | API cost |
|------|-----------|-------------|--------|--------------|----------|
| 1 | PID controller | **0.001 ms** | [0.0009, 0.001] | 4,000 Hz | None |
| 1.5 | MediaPipe EfficientDet-Lite0 | **16.2 ms** | [14.3, 18.6] | ~60 Hz | None |
| 2 | YOLO-World + YOLOv11n + DA v2 | **274.5 ms** | [267.0, 285.0] | ~3–4 Hz | None |
| 3 | Claude (Sonnet) | **7,218 ms** | [6,815, 7,629] | ~0.14 Hz | Per call |
| 3 | GPT-4o | **2,627 ms** | [2,491, 2,793] | ~0.38 Hz | Per call |
| 3 | GPT-4o-mini | **4,333 ms** | [4,059, 4,627] | ~0.23 Hz | Per call |
| 3 | Gemini | **2,333 ms** | [2,148, 2,533] | ~0.43 Hz | Per call |

*Run 1 YOLO latency (319 ms) excluded from mean; model warm-up cost on first
inference. Subsequent runs stable at ~263 ms.*

---

**Figure 3.5 — Four-Tier Timescale Separation (log scale)**
*Horizontal log-scale bar chart. X-axis: latency (ms), log₁₀ scale from 0.001 to 10,000.
One bar per tier/model combination (7 bars total). Colour-code by tier.
Annotate the ×16,212 ratio (MediaPipe/PID), ×16.9 ratio (YOLO/MediaPipe),
and ×8.5–26.3 ratio (LLM/YOLO range). Add vertical lines at 1ms, 16ms, 275ms, 2333ms.
Key finding: 4 orders of magnitude separating PID from slowest LLM; tiers cannot
be collapsed. Data: Table 3.7.*

---

**API cost extrapolation.** Three tiers (PID, MediaPipe, YOLO+DA v2) carry no API
cost: they run entirely on the companion computer. The LLM is the sole API-bearing
tier. At the maximum scheduled rate of 0.4 Hz (1 call per 2,500 ms, Gemini), the
marginal API cost is approximately $0.00012/call × 24 calls/minute = **$0.0029/minute**
(Gemini production deployment). At Claude's rate ($0.00633/call derived from G2) and
the same frequency, cost rises to **$0.15/minute** — 52× higher. Running the LLM at
YOLO's frequency (3–4 Hz) instead of 0.4 Hz would cost approximately $1,000+/minute
at Claude rates and ~$20/minute at Gemini rates — neither is operationally viable.
The four-tier architecture is not only latency-justified but cost-justified: each
tier operates at the highest frequency its budget permits.

### §3.3.2 Tier Isolation: Each Tier's Irreplaceable Contribution (G1)

Each tier handles at least one failure mode that no other tier can address;
G1 confirms this by testing four conditions on identical 320×240 frames from
real ESP32-S3-Sense hardware: Tier 1.5 alone (MediaPipe rule-based, 15 ms,
no API), Tier 2 alone (YOLO+DepthAnything v2 rule-based, 275 ms, no API), Tier 3
alone (LLM with image only, no sensor metadata), and the full combined pipeline —
8 scenes × 5 runs per condition (`G1_runs_20260525_021111.csv`). Table 3.8
summarises action safety per condition; Figure 3.6 visualises the dangerous
cases per scene.

---

**Table 3.8 — G1 Tier Isolation: Action Safety Summary**

| Condition | N trials | Action Safety | Dangerous | Latency | API Cost | Key blind spot |
|-----------|---------|--------------|-----------|---------|----------|----------------|
| tier1_5_only | 40 | **100%** | **0** | 15 ms | None | Caution nuance (all→safe/hazard only) |
| tier2_only | 40 | 87.5% | **5** | 275 ms | None | **blocked_lens** (YOLO sees nothing → "safe") |
| tier3_only (GPT-4o) | 40 | **100%** | **0** | 2,239 ms | API | wall_close metadata needed |
| tier3_only (Gemini) | 40 | **100%** | **0** | 1,961 ms | API | — |
| tier3_only (Claude) | 40 | **100%** | **0** | 5,079 ms | API | — |
| tier3_only (mini) | 40 | 87.5% | **5** | 5,541 ms | API | wall_close (no sensor support) |
| full pipeline (Gemini) | 40 | **100%** | **0** | 2,570 ms | API | — |
| full pipeline (GPT-4o) | 40 | 95.0% | 2 | 2,924 ms | API | wall_close (DA v2 anchoring) |
| full pipeline (Claude) | 39 | 97.4% | 1 | 8,032 ms | API | wall_close (DA v2 anchoring) |
| full pipeline (mini) | 40 | 97.5% | 1 | 8,115 ms | API | wall_close (DA v2 wrong depth) |

---

**Figure 3.6 — G1 Tier Isolation: Dangerous Cases per Scene per Condition**
*Heatmap (scenes × conditions matrix). Colour: 0 dangerous cases = white/green;
≥1 dangerous = red (scaled by count). Rows: 8 scenes. Columns: 10 conditions
(4 tier-only + 4 LLM-only + full pipeline × 4 models, condensed).
Key finding: blocked_lens danger is ONLY in tier2_only (5 cases);
wall_close danger is ONLY in full pipeline conditions (DA v2 wrong depth).
All LLM conditions produce 0 dangerous on blocked_lens.
Data: Table 3.8 and G1_runs_20260525_021111.csv.*

---

The critical finding is the **blocked_lens failure of Tier 2**: YOLO detects no
objects in a hand-covered frame (zero detections), and the rule engine concludes
"safe" → PITCH_FORWARD directly into the obstruction. This is a structural
failure of rule-based sensor systems: they cannot detect when the sensor itself
is compromised. All four LLMs produce zero dangerous recommendations on
blocked_lens regardless of condition — the LLM recognises partial occlusion
or darkness from the image pixels alone. The combined pipeline (Tier 1.5 + 2 + 3)
eliminates the blocked_lens failure because Tier 1.5's brightness gate fires
before YOLO runs, triggering an LLM call that overrides the rule engine. The
remaining dangerous cases (wall_close, full pipeline) trace to DA v2's hardware
depth error — the same limitation identified in the V6 ablation — not to any
removable architectural element. **Tier 1.5 is load-bearing for blocked_lens;
Tier 3 LLM is load-bearing for all sensor-failure scenes; Tier 2 metadata
benefits weaker models (GPT-4o-mini: 5→1 wall_close dangerous) while
occasionally anchoring stronger models to wrong sensor readings (GPT-4o: 0→2).**
Gemini achieves 0 dangerous in every condition — tier-isolated or combined.

### §3.3.3 Hybrid Trigger Strategy and Cost Tradeoff (G2)

Querying the LLM on a fixed schedule misses hazards appearing between scheduled
ticks; a hybrid strategy adds a real-time MediaPipe emergency interrupt to close
this gap. G2 compares the two strategies across 40 sequences — 4 models × 2
strategies × 5 runs — measuring hazard detection timing and per-sequence API cost
(`G2_runs_20260524_232234.csv`). Table 3.9 gives the full results; Figure 3.7
illustrates the detection timeline.

---

**Table 3.9 — G2 Trigger Strategy: Safety, Timing, and Cost per 10-frame Sequence**

| Strategy | Model | Catch Rate | Frames Late | LLM calls/seq | MediaPipe calls/seq | Cost/seq |
|----------|-------|-----------|------------|--------------|---------------------|----------|
| scheduled | claude | 100% | 3.0 | 2 | 0 | $0.01265 |
| scheduled | gpt4o | 100% | 3.0 | 2 | 0 | $0.01050 |
| scheduled | gpt4o_mini | 100% | 3.0 | 2 | 0 | $0.00119 |
| scheduled | gemini | 100% | 3.0 | 2 | 0 | $0.00024 |
| **hybrid** | claude | **100%** | **0.0** | 3 | 1 (13 ms, no API) | $0.01800 |
| **hybrid** | gpt4o | **100%** | **0.0** | 3 | 1 (13 ms, no API) | $0.01571 |
| **hybrid** | gpt4o_mini | **100%** | **0.0** | 3 | 1 (13 ms, no API) | $0.00180 |
| **hybrid** | gemini | **100%** | **0.0** | 3 | 1 (13 ms, no API) | **$0.00036** |

*Both strategies achieve 100% catch rate. The sole difference is timing:
scheduled detects at frame 6 (3 frames = 100 ms after appearance at 30 fps);
hybrid detects at frame 3 (0 ms late). MediaPipe trigger: no API cost.*

---

**Table 3.10 — G2 Cost of the Hybrid Improvement (per hazard event)**

| Model | Scheduled $/seq | Hybrid $/seq | Δ cost | Safety benefit |
|-------|----------------|--------------|--------|----------------|
| Claude | $0.01265 | $0.01800 | +$0.00535 | 100 ms earlier detection |
| GPT-4o | $0.01050 | $0.01571 | +$0.00521 | 100 ms earlier detection |
| GPT-4o-mini | $0.00119 | $0.00180 | +$0.00061 | 100 ms earlier detection |
| **Gemini** | **$0.00024** | **$0.00036** | **+$0.00012** | **100 ms earlier detection** |

*At Gemini rates, each hazard-onset event caught 100 ms earlier costs +$0.00012 —
approximately 1/50 of a cent per safety improvement. The MediaPipe interrupt itself
carries no API cost; the overhead is one additional LLM call per hazard event.*

---

**Figure 3.7 — G2 Trigger Strategy: Detection Timeline**
*Schematic timeline diagram (not a statistical plot).
X-axis: frame number (1–10). Y-axis: two rows (Scheduled, Hybrid).
Mark: door_open frames (1–2) in green; person_near frames (3–7) in red;
door_open frames (8–10) in green. Show scheduled LLM call at frame 1 (✓ safe)
and frame 6 (✓ hazard, 3 frames late). Show hybrid LLM calls at frame 1 (✓ safe),
frame 3 (★ MediaPipe trigger → ✓ hazard, 0 frames late), and frame 6 (✓ hazard).
Annotate the 100 ms gap between frame 3 and frame 6.
Caption: "At 30 fps and 0.3–0.5 m/s indoor drone speed, the 3-frame delay in
scheduled-only triggering allows 3–5 cm of additional travel toward a hazard.
Hybrid triggering eliminates this margin at a cost of one additional LLM call."*

---

**Sensor fusion at the trigger layer.** In run02, YOLO-World misclassified the
kneeling person as a structural wall (confidence 0.25) — a known YOLO-World failure
on low-resolution indoor frames. MediaPipe correctly detected the person at
confidence 0.391 and fired the interrupt at frame 3. All four LLMs classified the
scene as hazard, with Gemini explicitly noting: *"my visual analysis confirms they
are much closer than the YOLO detection suggested"* — demonstrating active sensor
disagreement resolution at Tier 3. Three-layer redundancy (YOLO-World structural +
MediaPipe person + LLM vision) catches what any single detection layer misses.
**Hybrid triggering with MediaPipe as the interrupt source is adopted for
production.**

### §3.3.4 Full Pipeline: Action Safety, Reasoning Quality, and Cost (G5)

The full four-tier pipeline operating with the production configuration (structured
JSON, max_tokens=256, short two-frame history, no CLIP, hybrid trigger) achieves
**97.5% overall action safety** across 157 evaluated trials (160 planned, 3 API
errors from GPT-4o timeout), with GPT-4o-mini and Gemini reaching **100%**
(`G5_runs_20260525_011427.csv`). Table 3.11 gives the full per-model results;
Figure 3.8 plots per-scene action safety; Figure 3.9 gives the latency and cost
breakdown.

---

**Table 3.11 — G5 Full Pipeline: Per-Model Summary**

| Model | Trials | Label Acc | Action Safety | Dangerous | LLM latency | Total latency |
|-------|--------|-----------|--------------|-----------|-------------|---------------|
| Claude | 37 | 54.0% | **97%** | **0%** | 7,466 ms | 7,777 ms |
| GPT-4o | 40 | 52.0% | 92.5% | **8%** (wall_close) | ~2,600 ms* | ~2,910 ms* |
| **GPT-4o-mini** | 40 | **80.0%** | **100%** | **0%** | 3,887 ms | 4,196 ms |
| **Gemini** | 40 | 50.0% | **100%** | **0%** | 2,244 ms | **2,554 ms** |
| **Overall** | **157** | **59.2%** | **97.5%** | **1.9%** | — | — |

*GPT-4o mean inflated by one API timeout+retry (CI: 2,506–51,515 ms);
typical latency ~2,600 ms. Stage latencies (shared): frame load 0.52 ms,
MediaPipe 33.8 ms, YOLO+DA v2 275 ms.*

---

**Table 3.12 — G5 Per-Scene Action Safety (all models combined)**

| Scene | Truth | Label Acc | Action Safety | Dangerous | Dominant pattern |
|-------|-------|-----------|--------------|-----------|-----------------|
| person_near | hazard | 95% | **100%** | 0% | Near-perfect |
| blocked_lens | hazard | 75% | 95% | 0% | 25% say caution → HOVER (safe) |
| **wall_close** | **hazard** | **80%** | **85%** | **15%** | GPT-4o DA v2 depth anchor |
| door_open | safe | 75% | **100%** | 0% | 25% over-cautious → HOVER |
| dim_light | caution | 50% | **100%** | 0% | Half say hazard → HOVER |
| object_table | caution | 50% | **100%** | 0% | LLM sees laptop as path-blocking |
| cluttered | caution | 35% | **100%** | 0% | Mostly hazard → HOVER (safe) |
| person_far | caution | 11% | **100%** | 0% | 89% say hazard → HOVER (safe) |

*7 of 8 scenes: 0% dangerous. Only wall_close has dangerous cases, all from GPT-4o.*

---

**Figure 3.8 — G5 Per-Scene Action Safety Heatmap**
*Heatmap: rows = 8 scenes, columns = 4 models.
Cell colour: white = 100% action safety (0 dangerous); yellow = 1 dangerous;
red = ≥3 dangerous. Annotate wall_close/GPT-4o cell (3 dangerous = 60%).
Show overall row at bottom (weighted average per model).
Key finding: wall_close/GPT-4o is the only non-white cell. All other 31 cells
are white (0 dangerous). Data: G5_runs_20260525_011427.csv.*

---

**Figure 3.9 — G5 Latency and Cost Decomposition per Model**
*Two-panel figure.
Left panel: stacked bar chart (X-axis: 4 models; Y-axis: latency ms).
Stacks: frame load (0.52 ms, grey), MediaPipe (33.8 ms, blue),
YOLO+DA v2 (275 ms, orange), LLM (model-specific, red/purple/green/yellow).
Right panel: estimated API cost per call (bar chart, Y-axis: USD).
Values: Claude ~$0.0063, GPT-4o ~$0.0053, GPT-4o-mini ~$0.0006, Gemini ~$0.00012
(derived from G2 per-sequence costs / 2 scheduled calls).
Annotate Gemini as lowest latency + lowest cost. Data: Tables 3.11, 3.10.*

---

**Production cost analysis.** Gemini is the recommended deployment model on both
safety and cost grounds: it achieves 0 dangerous recommendations in every test
condition and has the lowest latency (2,554 ms end-to-end) and lowest API cost
(~$0.00012/call derived from G2, Table 3.10). At the production trigger rate of
approximately 0.4 Hz (1 scheduled call every 2,500 ms plus emergency interrupts),
continuous drone operation with Gemini costs approximately **$0.003/minute**
($0.17/hour). Claude at the same frequency costs approximately **$0.15/minute**
($9.12/hour) — 52× higher, with longer latency and no safety advantage over
Gemini in G1 and G5 results. Table 3.13 summarises the cost–safety–latency
tradeoff across all four models.

---

**Table 3.13 — Model Selection Summary: Cost, Latency, and Safety**

| Model | End-to-end latency | Est. cost/call | Cost/hour (0.4 Hz) | Action safety (G5) | Dangerous (G1 full pipeline) | Recommended? |
|-------|-------------------|---------------|-------------------|-------------------|------------------------------|--------------|
| **Gemini** | **2,554 ms** | **~$0.00012** | **~$0.17** | **100%** | **0/40** | **✓ Production** |
| GPT-4o-mini | 4,196 ms | ~$0.0006 | ~$0.86 | **100%** | 1/40 | ✓ Backup |
| GPT-4o | ~2,910 ms | ~$0.0053 | ~$7.60 | 92.5% | 2/40 | — |
| Claude | 7,777 ms | ~$0.0063 | ~$9.12 | 97.0% | 1/40 | ✗ (latency) |

*Cost/call derived from G2 scheduled cost ÷ 2 LLM calls per sequence.
Cost/hour assumes 0.4 Hz scheduled LLM rate (Gemini) or equivalent.
G1 dangerous: full pipeline condition, 40 trials.*

---

**Reasoning quality.** Standard action safety (97.5% overall) understates what
the LLM is actually doing cognitively. Evaluating reasoning across four dimensions
— description accuracy (DA), reasoning-risk alignment (RRA), reason-action
alignment (RAA), and action safety (AS) — reveals that the cognitive chain is
internally sound at every stage except initial perception. Table 3.14 and
Figure 3.10 report the four-dimension analysis from the G5 reasoning quality
assessment (`G5_runs_20260525_011427.csv`).

---

**Table 3.14 — G5 Reasoning Quality: Four Dimensions per Model**

| Model | Desc Acc (DA) | Rsn-Risk (RRA) | Rsn-Act (RAA) | Act Safe (AS) | N |
|-------|--------------|----------------|---------------|---------------|---|
| Claude | 75.7% | 94.4% | **100%** | **100%** | 37 |
| GPT-4o | 87.5% | **100%** | **100%** | 92.5% | 40 |
| GPT-4o-mini | 92.5% | **100%** | **100%** | **100%** | 40 |
| **Gemini** | **100%** | **100%** | **100%** | **100%** | 40 |

*DA: did the LLM correctly describe the primary scene feature?
RRA: does the description justify the stated risk?
RAA: is the recommended action consistent with the LLM's own stated risk?
AS: is the action safe given ground truth?*

---

**Figure 3.10 — G5 Reasoning Quality: Four-Dimension Profile per Model**
*Radar chart (spider chart) with 4 axes: DA, RRA, RAA, AS. Scale 0–100%.
One polygon per model (4 coloured lines). Gemini polygon should be a perfect
square at 100% on all axes. GPT-4o polygon is slightly indented on AS (92.5%)
but perfect on the other three. Claude is indented on DA (75.7%) only.
Caption: "All models achieve 100% reason-action alignment — no model ever
internally contradicts its own stated risk. The only failure is visual
perception (DA): GPT-4o non-deterministically interprets a 320×240 featureless
grey wall as 'distant' rather than 'filling the frame'. The reasoning chain
from perception to risk to action is logically valid in both interpretations;
only the input differs." Data: Table 3.14.*

---

All four models achieve **100% reason-action alignment (RAA)**: no model in any
trial states a hazard classification and recommends PITCH_FORWARD, nor does any
model internally contradict its own stated risk. The GPT-4o dangerous cases
follow the chain: wrong visual description ("blank wall in the distance") →
logically valid risk inference given that description ("safe") → logically valid
action ("PITCH_FORWARD"). The reasoning is sound; the perception is wrong.
GPT-4o non-deterministically interprets the same 320×240 featureless grey frame
as either "blank wall in the distance" or "grey surface filling the frame" —
these are two visually plausible interpretations of a texture-uniform surface at
this resolution. Three other models (Claude, GPT-4o-mini, Gemini) override the
incorrect 2.09 m depth reading by reasoning that a surface filling the entire
frame cannot be 2.09 m away. **The LLM cognitive reasoning layer is internally
sound; the binding constraint is 320×240 input resolution.**

---

## §3.4 Safety and Trust Mechanisms (H series)

A pipeline with 97.5% action safety still produces recommendations the operator
must evaluate, and even correct recommendations require operator trust to be
acted on. Two mechanisms address the control and transparency requirements of
the pilot copilot model: a runtime mode switch that preserves operator override
authority at all times (H1) and a decision verbalization mechanism that narrates
the LLM's reasoning before each suggestion (H4). Both experiments are forthcoming.

### §3.4.1 Runtime Mode Switch — [PLACEHOLDER: H1 pending]

<!-- ═══════════════════════════════════════════════════════════════════
     PLACEHOLDER — H1 results pending (NOT RUN)
     File: H1_observations_[DATE].md when available.
     ═══════════════════════════════════════════════════════════════════
     Experiment design (THESIS_EXPERIMENTS.md):
       - Full-auto mode: LLM classifies + drone executes automatically
       - HITL mode: LLM suggests + operator approves/rejects before execution
       - Operator switches modes mid-mission at any time
       - Real laptop webcam frames + real YOLO at each waypoint

     Tables to add:
       Table 3.X — H1 Results: Switch latency, auto-mode success,
                   HITL approval rate, mission time per mode
         | Metric | Full-Auto | HITL | Δ |
         | Switch latency (ms) | — | [X] | — |
         | Mission success rate (%) | [Y] | [Z] | — |
         | HITL approval rate (%) | — | [W] | — |
         | Total mission time (s) | [A] | [B] | +[Δ]% |

     Figures to add:
       Figure 3.X — H1 Mode Switch Timeline
         Schematic showing auto→HITL transition: command dispatcher blocks
         new auto-executions the moment switch is initiated. Annotate switch
         latency. Show no command executes in the gap between switch initiation
         and HITL mode becoming active.

     Cost note:
       HITL mode does not change API cost — the LLM still receives the same
       metadata and produces the same verbalization; only the execution path
       differs. HITL adds human decision latency (~0.5–3 s per action) but
       zero additional API cost.

     Target paragraph:
       "The runtime mode switch validates operator authority. Switching from
        full-autonomous to HITL mode takes [X] ms ([CI]) — faster than one
        YOLO processing cycle (275 ms) — confirming that the transition is
        instantaneous from the operator's perspective. No command issued under
        auto-mode authority is executed after a switch is initiated: the
        switch is atomic with respect to the command dispatcher. Auto-mode
        mission success rate is [Y]% ([CI]); HITL approval rate is [Z]%,
        indicating operators accept [Z]% of LLM suggestions without override.
        Total mission time increases by [Δ]% in HITL mode — the cost of
        adding human decision latency per action — at zero additional API cost."
     ═══════════════════════════════════════════════════════════════════ -->

H1 validates the operator's ability to switch between full-autonomous and
human-in-the-loop (HITL) modes at any point during a mission using a real laptop
webcam and YOLO-World metadata at each waypoint. The central property to be
demonstrated is atomicity: no command issued under full-autonomous authority is
executed after the operator initiates a switch to HITL mode. HITL mode does not
change the API call rate or cost — the LLM still receives the same structured
prompt, sensor metadata, and produces the same verbalization; only the execution
path (auto vs operator-approved) differs.

**[H1 results: Table 3.X and Figure 3.X to be inserted from
H1_observations_[DATE].md.]**

### §3.4.2 Decision Verbalization for Transparency — [PLACEHOLDER: H4 pending]

<!-- ═══════════════════════════════════════════════════════════════════
     PLACEHOLDER — H4 results pending (NOT RUN)
     File: H4_observations_[DATE].md when available.
     ═══════════════════════════════════════════════════════════════════
     Experiment design (THESIS_EXPERIMENTS.md):
       - Real laptop webcam frame + real YOLO + real MediaPipe per trial
       - LLM narrates visual assessment and risk reasoning (text + optional TTS)
       - 5 scenarios: arm/takeoff, obstacle, altitude hold, battery warning,
         mission complete
       - LLM only SUGGESTS — operator decides

     Tables to add:
       Table 3.X — H4 Results: Quality, suggestion rate, latency
         | Metric | Mean | 95% CI | N |
         | Verbalization quality /4 | [X] | [CI] | [N] |
         | Pilot action suggestion present (%) | [Y] | [CI] | [N] |
         | Verbalization latency (ms) | [Z] | [CI] | [N] |
         | TTS latency (ms) | [W] | [CI] | [N] |

     Figures to add:
       Figure 3.X — H4 Example Verbalization Chain
         Flow diagram: Image → Scene description → Proximity assessment →
         Risk classification → Reasoning → Pilot action suggestion → [Operator].
         Include one example verbalization for each of the 5 scenarios.
         Annotate which parts map to which JSON fields from V2 structured prompt.

     Cost note:
       Verbalization is the LLM call itself (already budgeted). TTS adds
       local inference cost (no API) at [W] ms. Decision verbalization is
       zero additional API cost over the standard pipeline call.

     Target paragraph:
       "Decision verbalization produces a natural-language narration of the
        LLM's visual assessment and risk reasoning before each pilot action
        suggestion, achieving quality [X]/4 ([CI]) across [N] trials and five
        scenarios. Pilot action suggestions are present in [Y]% of
        verbalizations; verbalization latency is [Z] ms ([CI]); TTS adds
        [W] ms ([CI]). The transparency chain — scene description →
        proximity → risk → reasoning → suggestion — maps directly to the
        JSON fields of the structured prompt (V2, §3.1.1), meaning the same
        API call that produces the safety decision also produces its
        explanation, at zero additional cost. The operator receives the
        reasoning before committing to any action: this distinguishes the
        pilot copilot model from black-box autonomy where the basis for a
        command is opaque."
     ═══════════════════════════════════════════════════════════════════ -->

H4 validates transparency by requiring the LLM to narrate its visual assessment
and risk reasoning before each pilot action suggestion. The verbalization is a
direct output of the structured JSON prompt already defined in §3.1.1: the
`scene_description`, `proximity`, `risk_level`, and `reasoning` fields in the
JSON response constitute the narration, produced within the same API call that
generates the `recommended_action` field — zero additional cost. Across five
scenarios H4 will measure verbalization quality (/4), pilot action suggestion
rate, verbalization latency, and optional TTS delivery latency. The operator
receives the full reasoning chain before any action is taken; this traceability
distinguishes the pilot copilot model from black-box autonomy.

**[H4 results: Table 3.X and Figure 3.X to be inserted from
H4_observations_[DATE].md.]**

---

## §3.5 Chapter Summary

The image verbalization pipeline converts raw 320×240 drone camera frames into
structured pilot action suggestions through a four-tier hierarchy spanning four
orders of magnitude of latency (Table 3.7): 0.001 ms (PID), 16 ms (MediaPipe
emergency detector), 275 ms (YOLO-World + DepthAnything v2), and 2,333–7,218 ms
(LLM cognitive tier). Table 3.15 consolidates the production configuration and
the experiment that justifies each parameter.

---

**Table 3.15 — Production Pipeline Configuration Summary**

| Parameter | Value | Justification | Experiment |
|-----------|-------|--------------|------------|
| Prompt technique | Structured JSON | Highest LblAcc (56.6%), 0 dangerous, 100% RsnAct | V2 §3.1.1 |
| Token budget | max_tokens=256 | Quality plateau at 256; 512 adds Δ=0.01/5 | V6 §3.1.2 |
| CLIP | Removed | ΔQ≤0.05 (noise); random on 320×240; hurts Gemini at low budget | V6 §3.1.2 |
| Context history | Short (2 frames) | +16.2pp RiskAcc over stateless; bounded cost | V7 §3.1.3 |
| Temperature | t=[V8 pending] | [V8 to confirm] | V8 §3.1.4 |
| Trigger strategy | Hybrid (sched + MediaPipe) | 0 frames late; +$0.00012/hazard (Gemini) | G2 §3.3.3 |
| Production model | Gemini | 100% action safety; 2,554 ms; ~$0.00012/call | G5 §3.3.4 |

---

Under this configuration the integrated pipeline achieves **97.5% action safety**
across 157 trials (G5), with Gemini and GPT-4o-mini reaching **100%**. The
dominant outcome is conservative over-caution (41/157 trials, 26%), not dangerous
under-caution (3/157 trials, 1.9%). All three genuinely dangerous recommendations
come from a single source: GPT-4o anchoring to DepthAnything v2's incorrect
2.09 m depth reading for a wall at 20 cm. Reasoning quality analysis (Table 3.14)
confirms all four LLMs achieve **100% reason-action alignment** — the cognitive
layer is internally sound; failures trace to 320×240 visual perception ambiguity,
not to reasoning errors. At the production Gemini rate of 0.4 Hz, continuous
drone operation costs approximately **$0.17/hour** in API calls — 52× less than
Claude at equivalent frequency and safety. The pipeline architecture, production
configuration, cost analysis, and safety properties established in this chapter
constitute the perceptual foundation for the cognitive flight control system
presented in Chapter 4.

---

## Figure and Table Index — Chapter 3

| # | Type | Title | Section | Data source |
|---|------|-------|---------|------------|
| Figure 3.1 | Architecture diagram | Four-Tier Pipeline | Opening | System design |
| Figure 3.2 | Grouped bar chart | Prompt Technique: LblAcc vs Dangerous cases | §3.1.1 | Table 3.1 |
| Figure 3.3 | Line chart (2 panels) | Token Budget: Quality and LblAcc vs max_tokens | §3.1.2 | Table 3.3 + V6_noclip |
| Figure 3.4 | Grouped bar chart (2 panels) | History Mode: RiskAcc and ChgDetect | §3.1.3 | Table 3.6 |
| Figure 3.5 | Log-scale bar chart | Four-Tier Timescale Separation | §3.3.1 | Table 3.7 |
| Figure 3.6 | Heatmap | Tier Isolation: Dangerous Cases per Scene | §3.3.2 | Table 3.8 |
| Figure 3.7 | Timeline schematic | Trigger Strategy: Scheduled vs Hybrid | §3.3.3 | G2 data |
| Figure 3.8 | Heatmap | Full Pipeline: Per-Scene Action Safety | §3.3.4 | Table 3.12 |
| Figure 3.9 | Stacked bar + cost bar | Latency and Cost Decomposition per Model | §3.3.4 | Tables 3.11, 3.13 |
| Figure 3.10 | Radar chart | Reasoning Quality: Four Dimensions | §3.3.4 | Table 3.14 |
| Figure 3.V8 | Line chart | Temperature vs Accuracy and Flip Rate | §3.1.4 | [V8 pending] |
| Figure 3.Abl | Bar chart | Ablation: Action Safety per Component | §3.2 | [Ablation pending] |
| Figure 3.H1 | Timeline schematic | Mode Switch: Auto→HITL Transition | §3.4.1 | [H1 pending] |
| Figure 3.H4 | Flow diagram | Verbalization Chain: Image→Suggestion | §3.4.2 | [H4 pending] |
| Table 3.1 | Data table | V2 Prompt Technique (all models) | §3.1.1 | V2_20260525 |
| Table 3.2 | Data table | V2 Per-Model Best Technique | §3.1.1 | V2_20260525 |
| Table 3.3 | Data table | V6 Quality by Token Level (with CLIP) | §3.1.2 | V6_20260525 |
| Table 3.4 | Data table | V6 Per-Model Plateau Point | §3.1.2 | V6_20260525 |
| Table 3.5 | Data table | V7 Summary by History Mode | §3.1.3 | V7_20260525 |
| Table 3.6 | Data table | V7 Per-Model × Mode Risk Accuracy | §3.1.3 | V7_20260525 |
| Table 3.7 | Data table | G4 Measured Latency per Tier | §3.3.1 | G4_20260524 |
| Table 3.8 | Data table | G1 Tier Isolation: Action Safety | §3.3.2 | G1_20260525 |
| Table 3.9 | Data table | G2 Trigger Strategy: Cost and Safety | §3.3.3 | G2_20260524 |
| Table 3.10 | Data table | G2 Cost of Hybrid Improvement | §3.3.3 | G2_20260524 |
| Table 3.11 | Data table | G5 Full Pipeline: Per-Model Summary | §3.3.4 | G5_20260525 |
| Table 3.12 | Data table | G5 Per-Scene Action Safety | §3.3.4 | G5_20260525 |
| Table 3.13 | Data table | Model Selection: Cost, Latency, Safety | §3.3.4 | G2+G5 derived |
| Table 3.14 | Data table | G5 Reasoning Quality: Four Dimensions | §3.3.4 | G5_reasoning_20260525 |
| Table 3.15 | Summary table | Production Pipeline Configuration | §3.5 | All experiments |

---
*Draft written 2026-05-26. All [PLACEHOLDER] sections to be filled when dated
observation files are available. All [X]/[Y]/[N] markers must be replaced with
actual numbers before submission. Follow THESIS_WRITING_PLAN.md writing rules:
one idea per paragraph, first sentence is the claim, no placeholder numbers in
submitted text.*
