# Thesis Experiment Selection
# "Hierarchical Cognitive-Physical Autonomy for Hybrid Drones using Large Vision-Language Models"
#
# ═══════════════════════════════════════════════════════════════════════════════
# THREE THESIS CLAIMS — every kept experiment proves one of these
# ═══════════════════════════════════════════════════════════════════════════════
#
#   CLAIM 1 — VLM verbalization works as a drone command interface
#   CLAIM 2 — LLM cognitive layer can command and supervise a real drone
#   CLAIM 3 — Cognitive layer optimises hybrid locomotion
#
# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BODY EXPERIMENTS  (27 total)
# ═══════════════════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3 — IMAGE VERBALIZATION  (12 experiments)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CHAPTER 3 ARCHITECTURE (applies to all experiments except G1):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Camera frame captured                                          │
  │       ↓                                                         │
  │  YOLO runs on frame → detections (label, conf, est_dist, bbox)  │
  │       ↓ metadata passed alongside image                         │
  │  LLM called (scheduled OR YOLO-triggered emergency)             │
  │  Input  : image frame + YOLO metadata                           │
  │  Output : scene verbalization + pilot suggested action          │
  │           (PROCEED | SLOW_DOWN | STOP | LAND | HOLD)            │
  │       ↓                                                         │
  │  Pilot sees suggestion — accepts or overrides                   │
  └─────────────────────────────────────────────────────────────────┘
  Exception: G1 tests YOLO and LLM in pure isolation — no metadata sharing.

  SCORING RUBRIC (0–5, used in V and H series):
    s1 +1  scene content correctly described
    s2 +1  proximity/spatial information mentioned
    s3 +1  correct risk classification (safe/caution/hazard)
    s4 +1  response length 10–150 words
    s5 +1  pilot action explicitly suggested

── V series — Verbalization quality (5 experiments) ────────────────────────────

  V1  Model comparison
      CLAIM  : CLAIM 1
      PROVES : 4 models (Claude, GPT-4o, GPT-4o Mini, Gemini) × 8 scenes × 5 runs = 160 trials.
               Each model receives identical image + YOLO metadata.
               Gemini leads on accuracy (65%), quality (4.65/5), and cost ($0.0001/call).
               Gemini is 60× cheaper than Claude and 3.2× faster — recommended model
               for the production pipeline.
      KEY METRIC : Quality score /5 per model (Bootstrap CI);
                   classification accuracy (Wilson CI);
                   pilot action rate (%); latency (ms); cost (USD)
      WHY KEEP   : Essential — thesis must justify the model chosen for
                   the production pipeline
      STATUS : DONE  ✓  (results/V1_runs_20260521_015524.csv + V1_observations.md)

  V2  Prompt techniques
      CLAIM  : CLAIM 1
      PROVES : 5 prompting styles (zero-shot, few-shot, CoT, structured, ReAct)
               × 4 models × 8 scenes × 5 runs = 800 trials.
               No single technique dominates — best varies per model:
               Gemini→zero_shot (60%), GPT-4o→few_shot_3 (57.5%),
               GPT-4o Mini→react (45%), Claude→zero_shot (37.5%).
               Open-loop react_template collapses on door_open (5%) —
               ReAct framing without real feedback is just verbose zero_shot.
      KEY METRIC : Quality score /5 per technique (Bootstrap CI);
                   pilot action rate (%); classification accuracy (Wilson CI)
      WHY KEEP   : Essential — prompt choice directly affects whether
                   the pilot suggestion is actionable and trustworthy
      STATUS : DONE  ✓  (results/V2_runs_20260521_020936.csv + V2_observations.md)

  V2R ReAct agentic feedback loop vs open-loop template
      CLAIM  : CLAIM 1
      PROVES : Isolates the feedback loop contribution in ReAct prompting.
               Condition A (react_template, open-loop): loaded from V2 —
                 door_open accuracy = 5% across all 4 models.
               Condition B (react_agentic, 2-call feedback loop):
                 Call 1 [image only] → model observes.
                 Call 2 [observation + YOLO] → final classification.
                 door_open accuracy = 35% overall; gpt4o_mini = 100%.
               +30pp improvement on the worst open-loop failure proves the
               feedback loop — not the Reason-Observe-Act text structure —
               drives correct classification. Directly justifies the C-series
               architecture where ReAct uses real tool feedback per step.
      KEY METRIC : Classification accuracy per condition (Wilson CI);
                   per-model delta (agentic − template) on door_open
      WHY KEEP   : Closes the ReAct justification gap — without this,
                   using ReAct in C-series while rejecting it in V2 is
                   a contradiction. V2R proves they differ architecturally.
      STATUS : DONE  ✓  (results/V2R_runs_20260521_034438.csv + V2R_observations.md)

  V6  Verbosity vs quality
      CLAIM  : CLAIM 1
      PROVES : 4 token levels (64, 128, 256, 512) × 4 models × 8 scenes × 5 runs = 640 trials.
               Quality and accuracy plateau at 128 tokens — 256→512 adds nothing.
               Claude requires 256 to avoid truncation (97.5% truncation at 128).
               GPT-4o/Mini/Gemini plateau at ~40–53 words regardless of budget.
               max_tokens=256 selected as minimum sufficient budget for all 4 models.
      KEY METRIC : Quality score /5 vs token budget; truncation rate (Wilson CI);
                   efficiency (quality/USD); word count vs token budget
      WHY KEEP   : Practical design parameter — identifies minimum token
                   budget for a complete verbalization + pilot suggestion
      STATUS : DONE  ✓  (results/V6_runs_20260521_053358.csv + V6_observations.md)

  V7  Scene context history
      CLAIM  : CLAIM 1
      PROVES : Stateless vs short-history vs full-history — YOLO metadata
               from each frame is passed into the LLM call, and prior frame
               descriptions are optionally prepended as context.
               Does temporal context improve scene-change detection
               and pilot suggestion accuracy?
      KEY METRIC : Change detection rate (Wilson CI);
                   pilot action correctness at change frames (%);
                   input token count per history mode (Bootstrap CI)
      WHY KEEP   : Shows the scheduled LLM call benefits from temporal
                   context — relevant to real deployment where frames
                   are sequential
                   RESULT: Stateless is sufficient and best. Gemini stateless
                   achieves 72% risk accuracy and 90% change detection — best
                   across all modes and models. History adds token cost without
                   consistent benefit. Claude rerun with structured prompt fix:
                   stateless=0% (indoor caution bias), short/full=40% (history
                   enables caution→hazard escalation for person_near but never
                   fixes door_open safe-scene bias). Decision: LLM copilot
                   layer runs stateless.
      STATUS : DONE  ✓  (results/V7_runs_20260521_063401.csv + V7_observations.md)

  V8  Temperature sweep
      CLAIM  : CLAIM 1
      PROVES : Temperature [0.0, 0.2, 0.5, 0.8, 1.0] × 4 models × 8 scenes × 5 runs = 800 trials.
               t=0.0 achieves highest accuracy (40.6%) and lowest flip rate (5.5%).
               Higher temperatures increase variance without improving accuracy.
               Gemini degrades most sharply (37.5% → 25% at t=0.5).
               GPT-4o Mini is perfectly temperature-insensitive (0% flip at all temps).
               t=0.2 (Yao et al. 2022) is for iterative ReAct agents — not applicable
               to single-pass classification. t=0.0 selected for all V-series calls.
      KEY METRIC : Classification accuracy (Wilson CI) vs temperature;
                   label-flip rate vs temperature; quality score vs temperature
      WHY KEEP   : Design parameter — justifies t=0.0 for all production pipeline
                   calls with direct experimental evidence
      STATUS : DONE  ✓  (results/V8_runs_20260521_042814.csv + V8_observations.md)

── G series — Pipeline architecture (4 experiments) ────────────────────────────

  G1  YOLO vs Claude latency and accuracy  [ISOLATION EXPERIMENT — no metadata sharing]
      CLAIM  : CLAIM 1
      PROVES : Pure YOLO alone vs pure LLM alone — neither is sufficient.
               YOLO is fast but cannot suggest contextual pilot actions.
               LLM alone is slow and lacks structured detection metadata.
               Combined pipeline (G2, G5) outperforms both.
      KEY METRIC : Latency (ms); description richness score; detection accuracy
      WHY KEEP   : Justifies why both YOLO and LLM are needed in the pipeline
      STATUS : NOT RUN

  G2  Trigger strategy — scheduled-only vs scheduled+YOLO-triggered
      CLAIM  : CLAIM 1
      HARDWARE   : Laptop webcam (or ESP32-S3-Sense) — real camera frames required
      PROVES : YOLO always running on real camera frames in both conditions.
               Every LLM call receives real YOLO metadata and outputs pilot suggestion.
               Condition A — YOLO + scheduled LLM only (YOLO runs but never triggers extra LLM)
               Condition B — YOLO + scheduled LLM + YOLO emergency interrupt
               Hazards = real YOLO detections above confidence threshold.
               Condition A misses hazards between ticks even with YOLO running.
               Condition B catches them — real YOLO detection immediately triggers
               LLM interrupt call which suggests STOP to the pilot.
      KEY METRIC : Missed emergency rate (%) — A vs B (Wilson CI);
                   extra LLM calls per mission from YOLO interrupts (Bootstrap CI);
                   cost overhead of adding the interrupt (USD)
      WHY KEEP   : Directly proves the core Chapter 3 architectural claim —
                   YOLO-triggered LLM calls are not redundant; they close
                   the gap between scheduled ticks for time-critical hazards.
                   Uses real camera + real YOLO — not simulated text strings.
      STATUS : NOT RUN

  G4  Three-tier timescale analysis
      CLAIM  : CLAIM 1 + CLAIM 2
      HARDWARE   : Laptop webcam (or ESP32-S3-Sense) — real camera frames required
      PROVES : PID ~0.25ms (firmware arithmetic), YOLO measured on real camera frames,
               Claude measured with real YOLO metadata from those frames.
               Each tier operates at its natural timescale.
               Proves YOLO is fast enough to interrupt meaningfully before
               the next scheduled LLM tick.
      KEY METRIC : Latency distribution per tier on real hardware (Bootstrap CI);
                   tier separation ratios (YOLO/PID, LLM/YOLO)
      WHY KEEP   : The timescale data physically justifies the entire
                   three-tier hierarchy and the YOLO interrupt design.
                   Real camera + real YOLO — not synthetic frames or text strings.
      STATUS : NOT RUN

  G5  Real vision pipeline end-to-end
      CLAIM  : CLAIM 1
      PROVES : Full pipeline on live camera frames:
               camera → YOLO → YOLO metadata + image → LLM →
               scene verbalization + pilot action suggestion.
               Measures each stage latency and validates the LLM
               correctly suggests actions on real frames.
      KEY METRIC : End-to-end latency (ms) per stage (Bootstrap CI);
                   pilot action suggestion accuracy on live frames (Wilson CI)
      WHY KEEP   : Essential validation — proves the integrated pipeline
                   works on real hardware frames, not just simulation
      STATUS : NOT RUN

── H series — Safety and trust (2 experiments) ──────────────────────────────────

  H1  Runtime mode switch
      CLAIM  : CLAIM 1 + CLAIM 2
      HARDWARE   : Laptop webcam — real camera frame captured at each waypoint
      PROVES : YOLO + LLM pipeline runs continuously in both modes.
               Real camera frame captured + real YOLO run at each waypoint.
               Full-auto mode: LLM receives real YOLO metadata, suggests
                 pilot action, drone executes automatically.
               HITL mode: LLM still receives real YOLO metadata and
                 suggests pilot action, but operator must approve/reject
                 before execution.
               Operator can switch modes mid-mission at any time.
               Proves the pilot always has the last word regardless of
               what the LLM suggests.
      KEY METRIC : Switch latency (ms) (Bootstrap CI);
                   auto-mode success rate (Wilson CI);
                   HITL approval rate (Wilson CI);
                   total mission time (Bootstrap CI)
      WHY KEEP   : Safety — the pilot copilot model is only trustworthy if
                   the operator can interrupt any LLM suggestion and take
                   manual control instantly; this experiment proves it
      STATUS : NOT RUN

  H4  Decision verbalization
      CLAIM  : CLAIM 1 + CLAIM 2
      HARDWARE   : Laptop webcam — real camera frame captured per trial
      PROVES : Real camera frame captured + real YOLO run before each trial.
               LLM receives real YOLO metadata, narrates its assessment,
               and suggests a pilot action — across 5 scenarios:
               (arm/takeoff, obstacle scene description, altitude hold,
               battery warning, mission complete).
               LLM only SUGGESTS — pilot decides what to do.
               Pilot can read/hear the TTS verbalization and trust it.
      KEY METRIC : Verbalization quality score /4 (Bootstrap CI);
                   pilot action suggestion present (%);
                   verbalization latency (ms); TTS latency (ms)
      WHY KEEP   : Transparency — differentiates from black-box autonomy;
                   pilot copilot model requires explainable suggestions
      STATUS : NOT RUN



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4 — SYSTEM DESIGN AND HARDWARE  (16 experiments)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── C series — Manual cognitive commands in sim (6 experiments) ──────────────────

  C1  Natural language to toolchain
      CLAIM  : CLAIM 2
      PROVES : LLM correctly decomposes plain-English mission into tool calls
      KEY METRIC : Success rate (%), latency (ms), cost per run — 4 models
      WHY KEEP   : Foundation — proves cognitive layer can parse commands
      STATUS : DONE  ✓  (results/C1_runs.csv + multi-LLM variants)

  C2  Ambiguity handling
      CLAIM  : CLAIM 2
      PROVES : On vague commands, LLM asks clarification rather than guessing
      KEY METRIC : Clarification rate, success on ambiguous commands 3–6
      WHY KEEP   : Robustness — real operators give imprecise commands
      STATUS : DONE  ✓  (results/C2_runs.csv + multi-LLM variants)

  C3  Multi-turn context retention
      CLAIM  : CLAIM 2
      PROVES : LLM uses earlier context to correctly interpret later commands
      KEY METRIC : Context-dependent success rate across 5-turn dialogues
      WHY KEEP   : Unique capability — stateful cognition over a mission
      STATUS : DONE  ✓  (results/C3_runs.csv + multi-LLM variants)

  C5  Human describes problem → LLM diagnoses
      CLAIM  : CLAIM 2
      PROVES : Operator describes oscillation in plain English, LLM identifies
               the bad PID gain, suggests a fix, applies it, verifies improvement
      KEY METRIC : Diagnosis accuracy (%), RMSE before vs after (cm)
      WHY KEEP   : Most impressive result — LLM as autonomous diagnostician
      STATUS : DONE  ✓  (results/C5_runs.csv + guardrail on/off)

  C7  Safety guardrail
      CLAIM  : CLAIM 2
      PROVES : Dangerous commands blocked regardless of LLM output
               Adversarial disarm attempt also blocked
      KEY METRIC : Block rate on dangerous commands (%), false positive rate
      WHY KEEP   : Essential — safety is a core thesis claim
      STATUS : DONE  ✓  (results/C7_runs.csv + multi-LLM + guardrail on/off)

  C8  Three-mode comparison (manual vs semi-auto vs full-auto)
      CLAIM  : CLAIM 2
      PROVES : Full-auto matches human on structured missions
               Semi-auto best for novel environments
      KEY METRIC : Task completion time (s), error rate per mode
      WHY KEEP   : Evaluation framework — where does autonomy help vs hurt?
      STATUS : DONE  ✓  (results/C8_runs.csv + multi-LLM + guardrail on/off)

── D series — Vision-guided autonomy in sim (3 experiments) ─────────────────────

  D7  LLM iterative PID adaptation
      CLAIM  : CLAIM 2
      PROVES : Given a deliberately bad gain, LLM detects oscillation and
               corrects it iteratively to below 5 cm RMSE — 4 LLM models
      KEY METRIC : RMSE before/after (cm), iterations to stable, cost
      WHY KEEP   : Most unique technical result — LLM as control engineer
      STATUS : NOT RUN

  D8  Sensor dropout
      CLAIM  : CLAIM 2
      PROVES : When a sensor (ToF, barometer) drops mid-flight, the cognitive
               layer detects the telemetry anomaly and responds safely —
               switches mode, alerts operator, or initiates safe land —
               rather than issuing commands blind
      KEY METRIC : Detection latency (ms), correct response rate (%), false
                   positive rate
      WHY KEEP   : Robustness — proves cognitive layer handles hardware faults,
                   not just mission commands; pairs with H1 (mode switch)
      STATUS : NOT RUN

  D9  End-to-end integration
      CLAIM  : CLAIM 1 + CLAIM 2
      PROVES : Camera → verbalization (Ch3) → cognitive command → cascade PID
               — the three layers working as a unit
      KEY METRIC : Mission completion rate (%), end-to-end latency (ms)
      WHY KEEP   : Capstone experiment — validates entire system as integrated
      STATUS : NOT RUN

── E series — Architecture justification (5 experiments) ───────────────────────

  E1  API latency measurement
      CLAIM  : CLAIM 2
      PROVES : LLM round-trip is 500–2000 ms — 4 orders of magnitude above
               PID tick period — hierarchical separation is physically necessary
      KEY METRIC : Latency distribution (ms) per model, percentiles
      WHY KEEP   : Justifies the entire architecture — without this the
                   hierarchy looks arbitrary
      STATUS : NOT RUN

  E2  Human vs auto time
      CLAIM  : CLAIM 2
      PROVES : Full-auto LLM completes structured missions in comparable time
               to an experienced human operator
      KEY METRIC : Task completion time (s) per mode, error rate
      WHY KEEP   : Direct evidence that LLM autonomy is operationally viable
      STATUS : NOT RUN

  E3  Memory retention
      CLAIM  : CLAIM 2
      PROVES : Cognitive layer retains mission context across a long session —
               later commands correctly reference earlier decisions
      KEY METRIC : Context-dependent success rate across 10-turn sessions
      WHY KEEP   : Shows stateful cognition at session timescale, not just
                   turn-by-turn; complements C3 (5-turn)
      STATUS : NOT RUN

  E4  Token scaling
      CLAIM  : CLAIM 2
      PROVES : Performance plateaus beyond a token budget threshold —
               identifies the minimum viable context window for this task
      KEY METRIC : Task success rate vs input token count curve
      WHY KEEP   : Practical deployment parameter — informs API cost vs
                   performance tradeoff
      STATUS : NOT RUN

  E5  LLM vs rule-based supervisor
      CLAIM  : CLAIM 2
      PROVES : On combined/novel faults, rule-based fails — LLM recovers
               On simple faults, both succeed equally
      KEY METRIC : Recovery time (s), correct action rate per fault scenario
      WHY KEEP   : Justifies using LLM over simpler rule system
      STATUS : NOT RUN

── F series — Model selection (2 experiments) ───────────────────────────────────

  F1  Multi-model capability benchmark
      CLAIM  : CLAIM 2
      PROVES : 4 models × 5 capability dimensions — objective comparison
      KEY METRIC : Score per model per capability dimension (radar chart)
      WHY KEEP   : Principled model selection for hardware deployment
      STATUS : NOT RUN

  F2  Latency vs capability Pareto analysis
      CLAIM  : CLAIM 2
      PROVES : Claude sits on the Pareto frontier — best capability
               at lowest latency among tested models
      KEY METRIC : Bubble chart: latency (x) × capability score (y) × cost (size)
      WHY KEEP   : Final model selection justified — not an arbitrary choice
      STATUS : NOT RUN


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5 — HYBRID LOCOMOTION  (~3 experiments, to be designed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  L1  Energy per metre by locomotion mode
      CLAIM  : CLAIM 3
      PROVES : Rolling/walking costs less energy per metre than flying
               on flat terrain — quantified per terrain type
      KEY METRIC : J/m for fly vs roll vs walk per terrain type
      STATUS : NOT RUN — hardware + wheels required

  L2  Cognitive mode selection accuracy
      CLAIM  : CLAIM 3
      PROVES : LLM selects the energy-optimal mode on N terrain scenarios
               better than a fixed-rule baseline
      KEY METRIC : Optimal mode selection rate (%), vs rule baseline (%)
      STATUS : NOT RUN — hardware + wheels required

  L3  Range extension on benchmark mission
      CLAIM  : CLAIM 3
      PROVES : Mixed-terrain mission: cognitive selection extends range
               vs fly-only policy
      KEY METRIC : Distance reached at battery cutoff (m), % improvement
      STATUS : NOT RUN — hardware + wheels required


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX — SIMULATOR VALIDATION  (7 experiments)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These are not thesis contributions. They are credibility evidence.
# They show the simulator is faithful enough to trust the C series results.
# Full data in appendix; one paragraph + one table cited in §4.3.

  A1  Freefall                 — gravity correct to <1%
  A2  Madgwick filter RMSE     — 0.068° steady-state, matches literature
  A3  EKF altitude             — RMSE matches ToF noise floor
  A5  Motor lag                — τ = 30 ms matches Bitcraze measurement
  B1  Althold step response    — rise time, overshoot, RMSE vs Crazyflie benchmark
  B2  Poshold disturbance      — recovery from 0.05N impulse
  B3  Attitude step response   — rise time, overshoot vs Crazyflie benchmark

  STATUS : ALL DONE ✓  (results/A*.csv, results/B*.csv)


# ═══════════════════════════════════════════════════════════════════════════════
# DROPPED — not used in thesis
# ═══════════════════════════════════════════════════════════════════════════════

  V3  Multilingual input        — does not affect flight performance
  V4  Model × prompt matrix     — covered by V1 + V2 separately
  V5  YOLO threshold sweep      — implementation detail, not a claim
  V9  Model × params matrix     — overlaps V4 + V8

  I1  Multimodal vision benchmark — covered by V1 (same models × same scenes;
                                     V1 kept as it is already scripted and scoped)

  G3  Monocular depth accuracy  — tangential, depth not a core contribution

  H2  Face recognition auth     — not related to flight control or cognition
  H3  Blockchain integrity      — very tangential to thesis claims

  A4  Ground effect             — literature value used, no new finding
  A6  Battery model             — not a claim

  B4  Combined hold + wind      — redundant with B1 + B2 together
  B5  Hover SoC                 — redundant with A6

  C4  Mid-mission correction    — adequately covered by C3 multi-turn
  C6  Mission planning          — covered by C1 + C8 combined
  C2.1, C4.1, C6.1             — implementation fix variants, not results

  D1  Scene classification      — covered by G1 + I1
  D2  Autonomous wall approach  — camera+LLM too slow for collision response
                                   (500–2000 ms vs drone reaction need < 50 ms);
                                   correct layer is firmware-level LiDAR reflex
                                   (planned future hardware revision)
  D3  Human loop navigation     — overlaps C8
  D4  Obstacle avoidance        — same architectural reason as D2; LLM cannot
                                   command avoidance safely at flight speeds;
                                   LiDAR-based avoidance is future work
  D5  Autonomous waypoint       — overlaps D9
  D6  Anomaly detection         — covered by D8 (sensor dropout is the concrete
                                   anomaly scenario)

  F3  Open-source reproducibility — nice to have, not a claim

  MCP experiments (J–Q series)  — entire series not used in this thesis


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

  Original experiment count    : ~58 across all series
  Main body (kept)             : 31  (12 Ch3 + 16 Ch4 + 3 Ch5)
  Appendix                     : 7
  Dropped entirely             : ~29
  MCP series (never included)  : ~18

  Already done (main body)     : 12  (V1, V2, V2R, V6, V7, V8, C1, C2, C3, C5, C7, C8)
  Already done (appendix)      : 7   (A1–A3, A5, B1–B3)
  Still need to run            : 19  (G1,G2,G4,G5 + H1,H4
                                      + D7,D8,D9 + E1,E2,E3,E4,E5 + F1,F2 + L1,L2,L3)

  Experiments needing only laptop + API  : 18  (run these first)
  Experiments needing hardware           :  3  (L1, L2, L3)
  Experiments needing hardware + wheels  :  3  (same L series)
