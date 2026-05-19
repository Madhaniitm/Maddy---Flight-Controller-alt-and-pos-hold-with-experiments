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
CHAPTER 3 — IMAGE VERBALIZATION  (10 experiments)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── V series — Verbalization quality (4 experiments) ────────────────────────────

  V1  Model comparison
      CLAIM  : CLAIM 1
      PROVES : Which VLM produces the most command-compatible descriptions
      KEY METRIC : Scene classification accuracy per model (%)
      WHY KEEP   : Essential — thesis must justify the model chosen
      STATUS : NOT RUN

  V2  Prompt techniques
      CLAIM  : CLAIM 1
      PROVES : Zero-shot vs few-shot vs chain-of-thought on verbalization quality
      KEY METRIC : Command accuracy per prompt style (%)
      WHY KEEP   : Essential — prompt choice directly affects command reliability
      STATUS : NOT RUN

  V6  Verbosity vs quality
      CLAIM  : CLAIM 1
      PROVES : Concise descriptions produce better commands than long ones
      KEY METRIC : Command accuracy vs token count curve
      WHY KEEP   : Practical design parameter — tells reader how to prompt
      STATUS : NOT RUN

  V8  Temperature sweep
      CLAIM  : CLAIM 1
      PROVES : Low temperature (≤0.2) minimises hallucination in commands
      KEY METRIC : Hallucination rate vs temperature
      WHY KEEP   : Design parameter — justifies deterministic settings used
      STATUS : NOT RUN

── G series — Pipeline architecture (3 experiments) ────────────────────────────

  G1  YOLO vs Claude latency and accuracy
      CLAIM  : CLAIM 1
      PROVES : Why VLM is used over pure object detection
      KEY METRIC : Latency (ms), description richness score
      WHY KEEP   : Justifies pipeline design choice
      STATUS : NOT RUN

  G4  Three-tier timescale analysis
      CLAIM  : CLAIM 1 + CLAIM 2
      PROVES : YOLO at <10ms, VLM at ~1Hz, mission reasoning at ~0.1Hz
               — each tier operates at its natural timescale
      KEY METRIC : Latency distribution per tier
      WHY KEEP   : The entire architecture argument lives in this experiment
      STATUS : NOT RUN

  G5  Real vision pipeline end-to-end
      CLAIM  : CLAIM 1
      PROVES : Full pipeline working: camera → JPEG → VLM → JSON command
      KEY METRIC : End-to-end latency (ms), command accuracy on live frames
      WHY KEEP   : Essential validation — shows pipeline works on real frames
      STATUS : NOT RUN

── H series — Safety and trust (2 experiments) ──────────────────────────────────

  H1  Runtime mode switch
      CLAIM  : CLAIM 2
      PROVES : Cognitive layer can be interrupted and overridden at runtime
      KEY METRIC : Switch latency (ms), success rate
      WHY KEEP   : Safety — cognitive layer must not be a single point of failure
      STATUS : NOT RUN

  H4  Decision verbalization
      CLAIM  : CLAIM 2
      PROVES : LLM narrates its reasoning before each command — operator
               can understand and trust every decision
      KEY METRIC : Operator trust rating, verbalization latency (ms)
      WHY KEEP   : Transparency — differentiates from black-box autonomy
      STATUS : NOT RUN

── I series — Multi-model benchmark (1 experiment) ─────────────────────────────

  I1  Multimodal vision benchmark
      CLAIM  : CLAIM 1
      PROVES : 4 models × 10 scene classes — head-to-head on drone frames
      KEY METRIC : Accuracy per model per scene class, latency, cost
      WHY KEEP   : Objective model selection evidence for Chapter 3
      STATUS : NOT RUN


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4 — SYSTEM DESIGN AND HARDWARE  (14 experiments)
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

── D series — Vision-guided autonomy in sim (4 experiments) ─────────────────────

  D2  Autonomous wall approach and stop
      CLAIM  : CLAIM 1 + CLAIM 2
      PROVES : Verbalization → LLM command → drone stops before wall contact
      KEY METRIC : Stopping distance (m), success rate across 5 runs
      WHY KEEP   : Clearest end-to-end vision-to-action demonstration
      STATUS : NOT RUN

  D4  Obstacle avoidance
      CLAIM  : CLAIM 1 + CLAIM 2
      PROVES : Drone autonomously navigates around a mid-path obstacle
      KEY METRIC : Avoidance success rate, path deviation (m)
      WHY KEEP   : Core autonomous navigation claim
      STATUS : NOT RUN

  D7  LLM iterative PID adaptation
      CLAIM  : CLAIM 2
      PROVES : Given a deliberately bad gain, LLM detects oscillation and
               corrects it iteratively to below 5 cm RMSE — 4 LLM models
      KEY METRIC : RMSE before/after (cm), iterations to stable, cost
      WHY KEEP   : Most unique technical result — LLM as control engineer
      STATUS : NOT RUN

  D9  End-to-end integration
      CLAIM  : CLAIM 1 + CLAIM 2
      PROVES : Camera → verbalization (Ch3) → cognitive command → cascade PID
               — the three layers working as a unit
      KEY METRIC : Mission completion rate (%), end-to-end latency (ms)
      WHY KEEP   : Capstone experiment — validates entire system as integrated
      STATUS : NOT RUN

── E series — Architecture justification (2 experiments) ───────────────────────

  E1  API latency measurement
      CLAIM  : CLAIM 2
      PROVES : LLM round-trip is 500–2000 ms — 4 orders of magnitude above
               PID tick period — hierarchical separation is physically necessary
      KEY METRIC : Latency distribution (ms) per model, percentiles
      WHY KEEP   : Justifies the entire architecture — without this the
                   hierarchy looks arbitrary
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
  V7  Scene context history     — nice to have, not core
  V9  Model × params matrix     — overlaps V4 + V8

  G2  Event vs periodic         — cost optimisation, not a thesis claim
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
  D3  Human loop navigation     — overlaps C8
  D5  Autonomous waypoint       — overlaps D2 + D4
  D6  Anomaly detection         — covered by D7 (detection implied by adaptation)
  D8  Sensor dropout            — overlaps D6

  E2  Human vs auto time        — covered by C8
  E3  Memory retention          — covered by C3
  E4  Token scaling             — implementation detail

  F3  Open-source reproducibility — nice to have, not a claim

  MCP experiments (J–Q series)  — entire series not used in this thesis


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

  Original experiment count    : ~58 across all series
  Main body (kept)             : 27  (10 Ch3 + 14 Ch4 + 3 Ch5)
  Appendix                     : 7
  Dropped entirely             : ~31
  MCP series (never included)  : ~18

  Already done (main body)     : 6   (C1, C2, C3, C5, C7, C8)
  Already done (appendix)      : 7   (A1–A3, A5, B1–B3)
  Still need to run            : 21  (V1,V2,V6,V8 + G1,G4,G5 + H1,H4 + I1
                                      + D2,D4,D7,D9 + E1,E5 + F1,F2 + L1,L2,L3)

  Experiments needing only laptop + API  : 18  (run these first)
  Experiments needing hardware           :  3  (L1, L2, L3)
  Experiments needing hardware + wheels  :  3  (same L series)
