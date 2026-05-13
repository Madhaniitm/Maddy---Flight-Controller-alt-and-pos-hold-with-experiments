# Thesis Writing Plan
# "Hierarchical Cognitive-Physical Autonomy for Hybrid Drones using Large Vision-Language Models"
#
# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT STATUS AND WRITING ORDER
# ═══════════════════════════════════════════════════════════════════════════════
#
# ── What results exist right now ────────────────────────────────────────────
#
#   DONE — results CSVs confirmed in experiments/results/
#     A1  freefall              A2  madgwick            A3  ekf_altitude
#     A4  ground_effect         A5  motor_lag           A6  battery
#     B1  althold_step          B2  poshold_disturbance B3  attitude_step
#     B4  combined_hold_wind    B5  hover_soc
#     C1  nl_to_toolchain  (+ multi-LLM: Claude, GPT-4o, Gemini, LLaMA)
#     C2  ambiguity        (+ multi-LLM + rule baseline)
#     C3  multiturn         (+ multi-LLM)
#     C4  mid_mission       (+ multi-LLM)
#     C5  human_describes   (+ guardrail on/off)
#     C6  mission_planning  (+ multi-LLM)
#     C7  safety_override   (+ multi-LLM + guardrail on/off)
#     C8  three_mode        (+ multi-LLM + guardrail on/off)
#
#   NOT RUN — no results CSVs exist yet
#     V1–V9  Image verbalization experiments   (Image verbalization experiments/)
#     G1–G5  Vision pipeline experiments       (experiments/)
#     H1–H4  Safety and trust experiments      (experiments/)
#     I1     Multi-model vision benchmark      (experiments/)
#     D1–D9  Vision-guided autonomy            (experiments/)
#     E1–E5  System characterisation           (experiments/)
#     F1–F3  Multi-model benchmarks            (experiments/)
#     HW     Hardware trials                   (physical drone needed)
#     Ch5    Hybrid locomotion                 (hardware + wheels needed)
#
# ── Realistic writing order ──────────────────────────────────────────────────
#
# STAGE 1 — Write now, no new experiments needed
# ───────────────────────────────────────────────
#   STEP 1  Ch4 §4.2   Hardware Design                    ← START HERE
#                       Describe the physical drone you built.
#                       No results needed — factual description only.
#                       ESP32-S3, LiteWing, sensors, firmware architecture.
#
#   STEP 2  Ch4 §4.1   System Architecture
#                       Three-layer overview + timescale argument.
#                       No results needed — design rationale only.
#
#   STEP 3  Ch4 §4.3   Simulator — physics (A series)
#                       A1–A6 results all in hand. 5 paragraphs.
#
#   STEP 4  Ch4 §4.3   Simulator — control (B series)
#                       B1–B5 results all in hand. 3 paragraphs.
#
#   STEP 5  Ch4 §4.4   Phase 1 — C series (manual cognitive commands)
#                       C1–C8 + multi-LLM all in hand. 8 paragraphs.
#
# STAGE 2 — Run V, G, H, I experiments first  (laptop + API, ~1–2 days)
# ────────────────────────────────────────────────────────────────────────
#   These are Python scripts calling the LLM API. No hardware required.
#   Run in this order (each takes 30–60 min):
#     V1 → V2 → V3 → V6 → V7 → V8   (core verbalization experiments)
#     G1 → G2 → G3 → G4 → G5        (pipeline experiments)
#     H1 → H2 → H3 → H4             (safety experiments)
#     I1                              (multi-model benchmark)
#
#   STEP 6  Chapter 3   Image Verbalization (all sections)
#                       19 paragraphs. Write after V, G, H, I are run.
#
# STAGE 3 — Run D, E, F experiments  (laptop + API, ~2–3 days)
# ──────────────────────────────────────────────────────────────
#     D1 → D2 → D3 → D4 → D5 → D6 → D7 → D8 → D9
#     E1 → E2 → E3 → E4 → E5
#     F1 → F2 → F3  (F1 and F2 are analysis-only, no API calls needed
#                    once D and E are done)
#
#   STEP 7  Ch4 §4.4   Phase 2 — D series (vision-guided in sim)
#   STEP 8  Ch4 §4.5   E and F series — model selection + benchmarks
#
# STAGE 4 — Hardware trials  (physical drone required)
# ──────────────────────────────────────────────────────
#   Run hardware versions of B1, C1, then full integration.
#
#   STEP 9  Ch4 §4.6–4.7   Hardware validation
#
# STAGE 5 — Hybrid locomotion  (hardware + wheels required)
# ───────────────────────────────────────────────────────────
#   STEP 10  Chapter 5   Hybrid locomotion
#
# STAGE 6 — Framing chapters  (write after all core is done)
# ───────────────────────────────────────────────────────────
#   STEP 11  Chapter 2   Background and Related Work
#                         Now you know exactly which literature to cite.
#   STEP 12  Chapter 6   Results, Discussion, Conclusion
#
# STAGE 7 — Write absolutely last
# ────────────────────────────────
#   STEP 13  Chapter 1   Introduction
#                         Claims what you found — needs core written first.
#   STEP 14  Abstract     Final thing you write, ever.
#
# ── Progress tracker — update this as you go ────────────────────────────────
#
#   [ ] STEP 1   Ch4 §4.2  Hardware Design
#   [ ] STEP 2   Ch4 §4.1  System Architecture
#   [ ] STEP 3   Ch4 §4.3  A series (physics validation)
#   [ ] STEP 4   Ch4 §4.3  B series (control validation)
#   [ ] STEP 5   Ch4 §4.4  C series (manual cognitive commands)
#   [ ] RUN      V, G, H, I experiments
#   [ ] STEP 6   Chapter 3  Image Verbalization
#   [ ] RUN      D, E, F experiments
#   [ ] STEP 7   Ch4 §4.4  D series (vision-guided in sim)
#   [ ] STEP 8   Ch4 §4.5  E and F series (model selection)
#   [ ] RUN      Hardware trials
#   [ ] STEP 9   Ch4 §4.6–4.7  Hardware validation
#   [ ] RUN      Hybrid locomotion build + trials
#   [ ] STEP 10  Chapter 5  Hybrid locomotion
#   [ ] STEP 11  Chapter 2  Background and Related Work
#   [ ] STEP 12  Chapter 6  Results, Discussion, Conclusion
#   [ ] STEP 13  Chapter 1  Introduction
#   [ ] STEP 14  Abstract
#
# ═══════════════════════════════════════════════════════════════════════════════
# HOW TO USE THIS PLAN
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each section lists paragraphs in order.
# Every paragraph entry has:
#   WHAT   — the single idea this paragraph makes
#   CITE   — experiment / paper / figure to reference
#   OPENS  — first sentence you can draft from
#   LEADS  — what the next paragraph picks up
#
# Write one paragraph at a time. Do not move to the next until the current one
# says exactly one thing clearly.
# ═══════════════════════════════════════════════════════════════════════════════


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 1 — INTRODUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

§1.1  The Gap
─────────────
P1 — The problem with autonomous drones today
  WHAT   : Drones can stabilise and hold position reliably, but they cannot
           understand what they see or reason about what to do next.
  CITE   : General autonomous drone literature (Mahony 2012, Vemprala 2023)
  OPENS  : "Modern quadrotors can maintain stable hover and follow waypoints
            with centimetre precision, yet they remain fundamentally blind to
            the meaning of their environment."
  LEADS  : Why this gap matters — what it prevents.

P2 — Why this gap matters
  WHAT   : Without scene understanding, a drone cannot adapt its mission,
           detect faults, or decide between locomotion modes.
  CITE   : SayCan (Ahn 2022), InnerMonologue (Huang 2022)
  OPENS  : "Without the ability to interpret a scene, a drone cannot stop
            before a wall it was not pre-programmed to avoid, cannot diagnose
            why it is oscillating, and cannot decide whether flying or rolling
            is the more energy-efficient choice."
  LEADS  : Existing approaches and why they fall short.

P3 — Why existing approaches fall short
  WHAT   : Rule-based supervisors handle known faults but fail on novel or
           combined failures. End-to-end deep learning lacks interpretability
           and generalisation.
  CITE   : E5 (LLM vs rules), Vemprala 2023
  OPENS  : "Hand-coded rule supervisors address a finite set of anticipated
            faults; they fail silently when faults combine or when the
            environment deviates from design assumptions."
  LEADS  : The insight that motivates this thesis.

§1.2  The Central Insight
─────────────────────────
P4 — The key observation
  WHAT   : A vision-language model that can describe a drone's camera view
           in natural language can equally well generate structured commands
           — the same output serves both purposes.
  CITE   : Chapter 3 preview (V series, G series)
  OPENS  : "This thesis begins from a single observation: a model capable of
            describing what a drone's camera sees is, by the same token,
            capable of deciding what the drone should do next."
  LEADS  : How this observation becomes an architecture.

P5 — The architectural response
  WHAT   : Separate the system into three layers — perception (VLM),
           cognition (LLM reasoning), physical control (PID firmware) —
           each operating at its natural timescale.
  CITE   : E1 (4 orders of magnitude latency gap), Madgwick 2010
  OPENS  : "We exploit the four-orders-of-magnitude difference between LLM
            inference cadence (~0.5–2 Hz) and the PID inner loop (200 Hz)
            to build a hierarchy where each layer operates at the timescale
            it is physically capable of."
  LEADS  : What this thesis builds and demonstrates.

§1.3  Contributions
───────────────────
P6 — Contribution 1: image verbalization system
  WHAT   : A benchmarked pipeline that converts drone camera frames into
           structured natural-language descriptions that map to flight commands.
  CITE   : G, H, V, I series
  OPENS  : "First, we design, implement, and benchmark an image verbalization
            pipeline that converts raw drone camera frames into structured
            scene descriptions..."
  LEADS  : Next contribution.

P7 — Contribution 2: hardware flight controller with cognitive layer
  WHAT   : A physical ESP32-based drone with a firmware-identical simulator,
           validated with LLM cognitive commands from natural language through
           to full vision-guided autonomous operation.
  CITE   : A, B, C, D, E, F, HW series
  OPENS  : "Second, we build and validate a physical quadrotor flight
            controller and demonstrate that an LLM cognitive layer can command
            it reliably — first in simulation, then on hardware."
  LEADS  : Next contribution.

P8 — Contribution 3: hybrid locomotion with cognitive optimisation
  WHAT   : A wheel-and-tripod-gait extension to the drone where the cognitive
           layer chooses locomotion mode to maximise range.
  CITE   : Chapter 5 results
  OPENS  : "Third, we extend the drone with a wheeled locomotion mechanism
            and demonstrate that the same cognitive layer that commands flight
            can optimise energy use by selecting between flying, rolling, and
            walking."
  LEADS  : Thesis map.

§1.4  Thesis Map
────────────────
P9 — One paragraph walking through each chapter
  WHAT   : Tell the reader exactly where each contribution lives.
  OPENS  : "Chapter 2 reviews the literature on quadrotor control, LLMs for
            robotics, and hybrid locomotion. Chapter 3 presents the image
            verbalization system..."
  LEADS  : Chapter 2.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 2 — BACKGROUND AND RELATED WORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

§2.1  Quadrotor Dynamics and Cascade PID Control
─────────────────────────────────────────────────
P1 — Quadrotor physics fundamentals
  WHAT   : 6-DOF rigid body dynamics, thrust model, motor lag.
  CITE   : Mahony 2012, Förster 2015 (ETH), Faessler 2018
  OPENS  : "A quadrotor is a rigid body subject to gravity, aerodynamic drag,
            and four independent thrust forces..."

P2 — Cascade PID as the standard control architecture
  WHAT   : Inner rate loop + outer angle loop + altitude/position loop —
           bandwidth hierarchy must be satisfied.
  CITE   : Mahony 2012, Giernacki 2017 (Crazyflie)
  OPENS  : "The standard flight controller for micro-quadrotors uses a
            cascaded PID structure where each successive loop operates at
            a lower bandwidth than the one inside it."

P3 — State estimation: Madgwick and EKF
  WHAT   : Why complementary/Kalman filtering is needed; what each provides.
  CITE   : Madgwick 2010, Bitcraze kalman_core.c, Kan 2019
  OPENS  : "Raw IMU measurements are corrupted by noise and drift; a filter
            is required to fuse gyroscope and accelerometer data into a
            reliable attitude estimate."

§2.2  Large Language Models for Robotics
─────────────────────────────────────────
P4 — LLMs as task planners
  WHAT   : LLMs can decompose high-level instructions into executable
           sub-tasks via tool APIs.
  CITE   : Vemprala 2023, SayCan (Ahn 2022)
  OPENS  : "Recent work has demonstrated that large language models can act
            as high-level task planners for robotic systems..."

P5 — ReAct and closed-loop reasoning
  WHAT   : Interleaving reasoning and action (observe → think → act) allows
           LLMs to handle unexpected outcomes.
  CITE   : ReAct (Yao 2022), InnerMonologue (Huang 2022)
  OPENS  : "The ReAct framework (Yao et al. 2022) showed that an LLM
            alternating between reasoning steps and tool-call actions
            outperforms purely reactive or purely deliberative approaches."

P6 — Latency as a design constraint
  WHAT   : LLM inference at 0.5–2 Hz is incompatible with PID at 200 Hz —
           this gap motivates hierarchical separation.
  CITE   : E1 results (our own), Vemprala 2023
  OPENS  : "A fundamental constraint of LLM-based control is inference
            latency: typical API round-trips of 500–2000 ms place the
            cognitive layer at 0.5–2 Hz, four orders of magnitude below
            the millisecond timescales of inner PID loops."

§2.3  Vision-Language Models for Scene Understanding
─────────────────────────────────────────────────────
P7 — VLMs and their capabilities
  WHAT   : GPT-4V, Gemini, Claude vision — what they can and cannot do.
  CITE   : Achiam 2023 (GPT-4V), Reid 2024 (Gemini 1.5), Liu 2023 (LLaVA)
  OPENS  : "Vision-language models extend the text reasoning capability of
            LLMs to image inputs, enabling structured descriptions of
            visual scenes without task-specific training."

P8 — Gap in drone-specific VLM evaluation
  WHAT   : No prior work benchmarks VLMs on drone camera frames with the
           goal of generating flight commands.
  CITE   : I1, V series (our own — forward reference)
  OPENS  : "Despite the rapid growth of VLM capability, no prior work has
            systematically evaluated these models on the specific task of
            interpreting drone camera views and generating structured
            flight directives."

§2.4  Hybrid Terrestrial-Aerial Vehicles
─────────────────────────────────────────
P9 — Why hybrid locomotion
  WHAT   : Flying is energy-expensive; rolling/walking is cheaper on flat
           terrain. A hybrid system can extend operational range.
  CITE   : Relevant hybrid UAV literature
  OPENS  : "Quadrotors expend energy at a rate proportional to thrust
            squared; ground locomotion on flat terrain is an order of
            magnitude more efficient per metre travelled."

P10 — Existing hybrid designs and their limitations
  WHAT   : Prior hybrids use fixed switching rules; none use cognitive
           scene understanding to decide locomotion mode.
  CITE   : Relevant hybrid UAV papers
  OPENS  : "Existing hybrid vehicle designs switch between locomotion modes
            via pre-programmed rules tied to terrain classification or battery
            thresholds, without the ability to reason about scene context."

§2.5  Gap Summary and Positioning
──────────────────────────────────
P11 — What this thesis addresses that no prior work does
  WHAT   : Single paragraph tying all gaps together into the three
           contributions of this thesis.
  OPENS  : "This thesis addresses three gaps simultaneously: ..."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 3 — IMAGE VERBALIZATION AS A DRONE COMMAND INTERFACE
           (G, H, V, I series)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAPTER OPENING PARAGRAPH
  WHAT   : State the question this chapter answers and why it must be
           answered before anything else in the thesis.
  OPENS  : "Before a language model can command a drone, it must reliably
            interpret what the drone's camera sees. This chapter builds,
            benchmarks, and validates the image verbalization pipeline that
            converts raw camera frames into structured scene descriptions
            suitable for flight command generation."

§3.1  Pipeline Architecture
────────────────────────────
P1 — Three-tier design overview (G4)
  WHAT   : Fast YOLO tier (real-time detection), medium VLM tier
           (scene verbalization, ~1 Hz), slow reasoning tier
           (mission decisions, ~0.1 Hz).
  CITE   : G4 (three-tier timescale experiment)
  OPENS  : "The pipeline is structured as three tiers, each operating at
            the timescale its computational budget permits..."

P2 — Event-driven vs periodic triggering (G2)
  WHAT   : Triggering VLM calls on scene change events rather than
           periodically reduces API cost by X% with no accuracy loss.
  CITE   : G2 results
  OPENS  : "Rather than querying the vision-language model at a fixed rate,
            we trigger calls on detected scene changes..."

P3 — Real pipeline end-to-end (G5)
  WHAT   : Describe the actual implementation: camera → JPEG → VLM API →
           structured JSON output → command dispatcher.
  CITE   : G5 results, latency measured
  OPENS  : "In the full pipeline, a JPEG frame captured at [resolution] is
            transmitted to the VLM API; the structured JSON response is
            parsed by the command dispatcher within [X] ms end-to-end."

§3.2  Verbalization Quality Benchmarking
─────────────────────────────────────────
P4 — YOLO vs Claude accuracy/latency tradeoff (G1)
  WHAT   : YOLO is faster but produces bounding boxes, not scene
           descriptions. Claude produces richer output but at higher latency.
           Each has its role.
  CITE   : G1 results (latency, accuracy comparison)
  OPENS  : "Object detection models such as YOLO produce bounding-box
            outputs at sub-millisecond latency but cannot generate the
            natural-language scene descriptions required for cognitive
            command generation..."

P5 — Monocular depth for spatial awareness (G3)
  WHAT   : Depth estimation from a single camera — accuracy on drone frames,
           how it feeds into obstacle distance estimates.
  CITE   : G3 results (RMSE vs ground truth)
  OPENS  : "To enable distance-aware commands ('stop 30 cm before the wall'),
            the pipeline incorporates monocular depth estimation..."

§3.3  Model and Prompt Ablation (V series)
───────────────────────────────────────────
P6 — Model comparison across verbalization quality (V1)
  WHAT   : Which model produces the most command-compatible descriptions.
           Table: Claude vs GPT-4o vs Gemini vs LLaVA on scene accuracy.
  CITE   : V1 results
  OPENS  : "We evaluated four vision-language models on identical drone
            frames using a fixed evaluation rubric..."

P7 — Prompt technique comparison (V2)
  WHAT   : Zero-shot vs few-shot vs chain-of-thought on verbalization
           quality. Which technique wins and why.
  CITE   : V2 results
  OPENS  : "Prompt structure significantly affects verbalization quality:
            chain-of-thought prompting improved command-compatible output
            by [X]% over zero-shot on ambiguous scenes..."

P8 — Verbosity vs command quality tradeoff (V6)
  WHAT   : More verbose descriptions are not always better — they increase
           latency and can confuse the command dispatcher.
  CITE   : V6 results
  OPENS  : "Verbosity and command quality are not monotonically related:
            descriptions exceeding [N] tokens showed no improvement in
            command accuracy while increasing latency by [X] ms..."

P9 — Temperature sweep (V8)
  WHAT   : Low temperature (deterministic) is better for command generation;
           slightly higher temperature acceptable for scene description.
  CITE   : V8 results
  OPENS  : "Temperature controls output randomness; for command-critical
            applications, we find that temperature ≤ 0.2 minimises
            hallucination of non-existent scene elements..."

P10 — Scene context and history (V7)
  WHAT   : Providing the previous frame's description as context improves
           temporal consistency of commands.
  CITE   : V7 results
  OPENS  : "Providing the model with a one-sentence summary of the previous
            frame's interpretation improved temporal consistency of
            consecutive commands by [X]%..."

P11 — Multilingual input (V3)
  WHAT   : The pipeline accepts non-English operator instructions without
           performance degradation, broadening deployment scenarios.
  CITE   : V3 results
  OPENS  : "To assess deployment breadth, we tested operator commands in
            [N] languages..."

§3.4  Multi-Model Vision Benchmark (I1)
────────────────────────────────────────
P12 — Experimental setup
  WHAT   : 4 models × 10 scene classes × 5 frames = 200 trials. Same JPEG
           fed to each model. Apples-to-apples comparison.
  CITE   : I1 setup
  OPENS  : "To establish a direct model comparison, we presented identical
            drone camera frames to four vision-language models under identical
            prompt conditions..."

P13 — Accuracy results
  WHAT   : Table of per-scene and overall accuracy per model. Which scenes
           are hardest and why.
  CITE   : I1 accuracy + confusion matrix
  OPENS  : "Overall classification accuracy ranged from [X]% (LLaVA-13B)
            to [Y]% (Claude Sonnet), with the most confusion occurring
            between [scene A] and [scene B]..."

P14 — Latency and cost results
  WHAT   : Latency/cost table per model. Pareto observation (which model
           is dominated).
  CITE   : I1 latency + cost results
  OPENS  : "Latency and cost varied substantially across models: GPT-4o
            completed calls in [X] ms at [Y] USD per call, while
            LLaVA-13B via Ollama incurred no API cost at the expense of
            [Z] ms local inference latency..."

§3.5  Safety and Trust Mechanisms (H series)
─────────────────────────────────────────────
P15 — Why safety is needed in a cognitive-physical system (H1)
  WHAT   : A cognitive layer that can issue any command must also be able to
           refuse dangerous ones. Runtime mode switching adds a safety valve.
  CITE   : H1 (runtime mode switch)
  OPENS  : "A cognitive layer with unrestricted command authority presents
            a safety risk: an LLM hallucination or adversarial input could
            issue a command that damages hardware or injures bystanders..."

P16 — Operator authentication (H2)
  WHAT   : Face recognition gates who can issue cognitive commands.
  CITE   : H2 results (recognition accuracy, latency)
  OPENS  : "To prevent unauthorised command injection, we implemented a
            face-recognition gate at the cognitive layer input..."

P17 — Command integrity and audit trail (H3)
  WHAT   : Blockchain-style logging of every command ensures auditability.
  CITE   : H3 results
  OPENS  : "Every command issued by the cognitive layer is logged to an
            append-only blockchain-style ledger, providing a tamper-evident
            audit trail for post-incident analysis..."

P18 — Decision verbalization for transparency (H4)
  WHAT   : The LLM narrates its reasoning before issuing each command,
           making decisions interpretable to the human operator.
  CITE   : H4 results (operator trust rating, verbalization latency)
  OPENS  : "To make the cognitive layer's decisions legible to a human
            supervisor, we require the LLM to verbalize its reasoning before
            each command is dispatched..."

§3.6  Chapter Summary
──────────────────────
P19 — What this chapter established
  WHAT   : Verbalization pipeline works reliably, Claude is the strongest
           model, safety mechanisms are validated. This output is the input
           to Chapter 4.
  OPENS  : "The image verbalization system established in this chapter
            produces structured, command-compatible scene descriptions at
            [X] Hz with [Y]% scene classification accuracy. These outputs
            serve as the cognitive input to the flight control system
            developed in the following chapter."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 4 — SYSTEM DESIGN, VALIDATION AND HARDWARE IMPLEMENTATION
           (A, B, C, D, E, F series + Hardware)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAPTER OPENING PARAGRAPH
  WHAT   : Frame the chapter's narrative arc. Simulation is a design tool.
           Hardware is the result.
  OPENS  : "This chapter presents the design, simulation-based concept
            validation, and hardware implementation of the hierarchical
            drone control system. Simulation is used as a low-risk environment
            to tune and validate the control and cognitive layers before
            deployment; all primary results are drawn from physical hardware
            trials."

§4.1  System Architecture
──────────────────────────
P1 — Three-layer architecture description
  WHAT   : Name the layers, their roles, and their interfaces.
  CITE   : System diagram
  OPENS  : "The system comprises three layers: a perception layer (Chapter 3),
            a cognitive layer (LLM running at 0.5–2 Hz), and a physical
            control layer (cascade PID firmware at 200 Hz)."

P2 — Timescale separation as a design principle (E1)
  WHAT   : The 4-orders-of-magnitude gap between LLM and PID is not a
           limitation — it is the justification for the hierarchy.
  CITE   : E1 (measured API latency distribution)
  OPENS  : "API latency measurements across three LLM providers (§4.5.1)
            confirm a mean round-trip of [X] ms — [N]× the 5 ms PID tick
            period — making hierarchical separation not merely convenient
            but physically necessary."

P3 — WebSocket + HTTP command protocol
  WHAT   : The same JSON protocol is used by the browser controller,
           the AI agent, the simulator, and the hardware — zero code
           changes between environments.
  CITE   : drone_sim.py docstring, Maddy_Flight_Controller.ino
  OPENS  : "A shared WebSocket command protocol (port 81) and HTTP camera
            endpoint (port 8080) allow the cognitive layer, browser
            controller, and simulator to operate identically..."

§4.2  Hardware Design
──────────────────────
P4 — Frame and motor selection
  WHAT   : Physical build: ESP32-S3, LiteWing module, 50 g total mass,
           46 mm props, 300 mAh LiPo.
  CITE   : Physical build specs, Förster 2015 for parameter validation
  OPENS  : "The physical drone is built around an ESP32-S3 microcontroller
            paired with a LiteWing sensor module, giving a total all-up
            mass of [X] g..."

P5 — Sensor suite
  WHAT   : IMU (MPU-6050), ToF (VL53L0X), optical flow sensor — what
           each contributes to state estimation.
  OPENS  : "State estimation relies on three sensor modalities: a 6-axis
            IMU for attitude at 200 Hz, a time-of-flight sensor for
            altitude at 50 Hz, and an optical flow sensor for horizontal
            velocity..."

P6 — Firmware architecture
  WHAT   : Madgwick → 9-state EKF → cascade PID → motor mixer.
           All running on-board at 200 Hz.
  CITE   : Maddy_Flight_Controller.ino, dRehmFlight reference
  OPENS  : "The firmware executes a fixed 200 Hz control loop: the
            Madgwick filter estimates attitude from IMU data, the 9-state
            EKF fuses ToF and optical flow for position, and the cascade
            PID converts setpoints to motor PWM commands..."

§4.3  High-Fidelity Simulator as a Design Tool
───────────────────────────────────────────────
SECTION OPENING — one sentence framing
  "All gains and cognitive logic were first validated in a simulator
   built to exactly mirror the hardware firmware before any hardware
   trials were conducted."

P7 — Why a simulator and what it mirrors
  WHAT   : drone_sim.py is a drop-in replacement for the hardware —
           same protocol, same PID, same EKF, same sensor noise model.
  CITE   : drone_sim.py docstring
  OPENS  : "The simulator implements identical firmware algorithms in
            Python — the same Madgwick filter, the same 9-state EKF,
            the same cascade PID gains — so that a command sequence
            validated in simulation transfers to hardware without
            modification."

P8 — Physics fidelity validation (A1, A4, A5)
  WHAT   : Freefall matches gravity exactly. Ground effect matches
           Kan 2019. Motor lag τ = 30 ms matches Bitcraze measurement.
  CITE   : A1, A4, A5 results + literature benchmarks
  OPENS  : "Three targeted experiments validate the simulator's physics:
            a free-fall test confirms the gravitational acceleration to
            within [X]%, a ground-effect test matches the Kan et al.
            (2019) Crazyflie dataset with RMSE [Y]..."

P9 — Estimator fidelity (A2, A3)
  WHAT   : Madgwick steady-state RMSE = 0.068°. EKF altitude RMSE
           matches ToF noise floor.
  CITE   : A2, A3 results
  OPENS  : "The Madgwick filter implementation achieves a steady-state
            roll/pitch RMSE of [X]° at β = 0.03, consistent with
            Madgwick's (2010) reported performance..."

P10 — Battery model (A6)
  WHAT   : Simulated discharge curve matches LiPo OCV vs SoC model.
           Thrust degradation with voltage is captured.
  CITE   : A6 results
  OPENS  : "The battery model replicates the discharge behaviour of a
            1S 300 mAh LiPo, with terminal voltage dropping under load
            according to V = OCV − I·R_int..."

P11 — Altitude hold in simulation (B1, B4)
  WHAT   : Sim althold matches Crazyflie literature: rise time 1–2 s,
           overshoot ≤ 10%, RMSE < 2 cm.
  CITE   : B1 results + Giernacki 2017 benchmarks
  OPENS  : "The cascade altitude hold controller achieves a rise time of
            [X] s and steady-state RMSE of [Y] cm for a 0.3 m step
            command, within the [1–2 s, <2 cm] range reported for
            Crazyflie-class drones..."

P12 — Position hold and disturbance rejection (B2, B3, B5)
  WHAT   : Position hold recovers from a 0.05 N lateral impulse
           (Δv = 0.2 m/s) within [X] s with < 5 cm SS error.
  CITE   : B2, B3, B5 results
  OPENS  : "Position hold disturbance rejection was evaluated by
            injecting a 0.05 N lateral impulse at t = 5 s, equivalent
            to a realistic indoor gust on a 50 g drone..."

P13 — Simulator validation summary
  WHAT   : Sim matches literature on all key metrics → gains transfer.
  OPENS  : "Across all physics and control validation experiments, the
            simulator reproduces Crazyflie-class benchmarks to within
            measurement uncertainty, confirming that PID gains and EKF
            parameters tuned in simulation will transfer to hardware."
  LEADS  : Now we run the cognitive layer on top of this validated simulator.

§4.4  LLM Cognitive Control: Concept Validation in Simulation
──────────────────────────────────────────────────────────────
SECTION OPENING
  "With the physical control layer validated, the cognitive layer was
   integrated and tested in simulation. All cognitive experiments in
   this section use the identical tool API and guardrail stack later
   deployed on hardware."

── Phase 1: Natural Language Command Chain (C series) ──

P14 — NL-to-toolchain parsing (C1)
  WHAT   : LLM correctly decomposes a plain-English mission into
           sequential tool calls. Multi-LLM comparison.
  CITE   : C1 results (success rate, latency, cost — Claude vs GPT-4o
           vs Gemini)
  OPENS  : "The first test of the cognitive layer is whether it can
            translate a natural-language mission description into a
            correct sequence of tool calls without human intervention..."

P15 — Ambiguity handling (C2, C2.1)
  WHAT   : When commands are vague ('go there'), the LLM asks a
           clarifying question rather than guessing dangerously.
  CITE   : C2 results (clarification rate, success on ambiguous
           commands 3–6)
  OPENS  : "Real operator commands are often underspecified. We
            tested the cognitive layer on commands ranging from
            unambiguous to highly ambiguous..."

P16 — Multi-turn dialogue and context retention (C3)
  WHAT   : LLM retains mission context across turns, using earlier
           information to interpret later commands correctly.
  CITE   : C3 results + E3 (memory retention experiment)
  OPENS  : "A realistic mission involves multiple exchanges, not a
            single command. We tested whether the cognitive layer
            correctly uses earlier context when interpreting later
            instructions..."

P17 — Mid-mission correction (C4, C4.1)
  WHAT   : Operator changes the target mid-flight. LLM aborts current
           trajectory, replans, and reaches the new target.
  CITE   : C4 results (replanning success rate, time to new target)
  OPENS  : "Mid-mission corrections are a frequent operational
            requirement. We injected a target change after the drone
            had already begun moving toward the original objective..."

P18 — Human-describes-problem diagnosis (C5)
  WHAT   : Operator describes oscillation in plain English. LLM
           diagnoses the PID gain, suggests a fix, applies it, and
           verifies improvement.
  CITE   : C5 results (diagnosis accuracy, RMSE before/after)
  OPENS  : "We tested whether an operator with no control-theory
            knowledge could describe a flight anomaly in natural
            language and have the cognitive layer diagnose and
            correct it..."

P19 — Mission planning from a single sentence (C6, C6.1)
  WHAT   : 'Survey the room and return' decomposed into waypoints,
           altitude changes, and camera triggers automatically.
  CITE   : C6 results (waypoint accuracy, mission completion rate)
  OPENS  : "High-level mission planning was evaluated by giving the
            cognitive layer a single-sentence objective and measuring
            whether it could decompose this into a valid flight plan..."

P20 — Safety guardrail (C7, C7.2)
  WHAT   : Guardrail blocks dangerous commands (fly into wall, exceed
           battery limit) regardless of LLM output. Adversarial
           disarm test.
  CITE   : C7, C7.2, guardrail_validation results
  OPENS  : "A safety-critical cognitive layer must refuse dangerous
            commands even when they originate from the LLM itself.
            We implemented a rule-based guardrail layer between the
            cognitive and physical layers..."

P21 — Mode comparison: manual vs semi-auto vs full-auto (C8)
  WHAT   : Full-auto matches human performance on structured missions;
           semi-auto is best for novel environments.
  CITE   : C8 results (task completion time, error rate per mode)
  OPENS  : "Three operating modes were compared on identical missions:
            manual (joystick only), semi-autonomous (LLM suggests,
            human approves), and fully autonomous (LLM commands
            directly)..."

── Phase 2: Vision-Guided Autonomous Operation (D series) ──

TRANSITION PARAGRAPH
  WHAT   : C series used scripted/manual natural-language commands.
           D series feeds the Chapter 3 verbalization pipeline as
           the command source — this is the first full integration.
  OPENS  : "Having validated the cognitive command chain with manually
            crafted inputs, we now close the loop by connecting the
            image verbalization pipeline (Chapter 3) as the command
            source, creating a fully autonomous vision-guided system."

P22 — Scene classification accuracy in-flight (D1)
  WHAT   : LLM correctly classifies 10 scene types from live drone
           camera at [X]% accuracy with [Y] ms latency.
  CITE   : D1 results
  OPENS  : "Before autonomous navigation, we confirmed that the
            vision-classification step performs reliably on frames
            captured during active flight, where motion blur and
            attitude variation are present..."

P23 — Wall approach and stop (D2)
  WHAT   : Drone autonomously approaches a wall and stops before
           contact. All 5 runs successful. Stopping distance distribution.
  CITE   : D2 results (stopping_distance_m, time_to_stop_s)
  OPENS  : "The first navigation task required the drone to advance
            toward a virtual wall and stop before reaching 20 cm
            proximity, using only visual information..."

P24 — Obstacle avoidance and waypoint navigation (D3, D4, D5)
  WHAT   : Human-in-loop, full obstacle avoidance, multi-waypoint
           mission. Success rates and key failure modes.
  CITE   : D3, D4, D5 results
  OPENS  : "More complex navigation scenarios were tested progressively:
            human-in-the-loop approval of each action (D3), autonomous
            obstacle avoidance (D4), and multi-waypoint sequential
            navigation (D5)..."

P25 — Anomaly detection and fault supervision (D6, D8)
  WHAT   : LLM detects PID oscillation and sensor dropout from
           telemetry alone without human prompting.
  CITE   : D6 results (detection_rate per fault type), D8
  OPENS  : "A key capability of the cognitive layer is unsupervised
            fault detection: given only telemetry, can the LLM identify
            that something is wrong without being told to look?"

P26 — Iterative PID adaptation by LLM (D7)
  WHAT   : LLM iteratively tunes a deliberately bad gain over ≤ 3
           iterations to below 5 cm RMSE. Multi-LLM comparison.
  CITE   : D7 results (rmse_before, rmse_after, iterations_to_stable)
  OPENS  : "We tested whether the cognitive layer could act as an
            autonomous control engineer: given an oscillating drone
            with incorrect PID gains, can it diagnose and correct the
            problem iteratively?"

P27 — End-to-end integration (D9)
  WHAT   : The full pipeline — camera → verbalization (Ch3) → LLM
           command → cascade PID (§4.3) — validated as a unit.
  CITE   : D9 results (mission completion rate, end-to-end latency)
  OPENS  : "Experiment D9 validates the complete system as an
            integrated unit: the image verbalization pipeline,
            cognitive command layer, safety guardrail, and cascade
            PID firmware are exercised together on a multi-step
            indoor mission..."

§4.5  LLM Model Selection and System Characterisation
──────────────────────────────────────────────────────
SECTION OPENING
  "Before hardware trials, we conducted a systematic characterisation
   to select the LLM model for deployment and to confirm that the
   architecture's properties hold under real API conditions."

P28 — Latency justification (E1)
  WHAT   : Measured latency across Claude, GPT-4o, Gemini confirms
           0.5–2 Hz cognitive loop. Mean latency is [X]× the PID
           tick period.
  CITE   : E1 results (latency distribution per model)
  OPENS  : "Fifty consecutive API calls to each of three providers
            under a fixed drone-status prompt yielded latency
            distributions with means of [X], [Y], [Z] ms respectively..."

P29 — LLM vs rule-based supervisor (E5)
  WHAT   : On combined/novel faults (S2, S3), rule-based supervisor
           fails; LLM recovers. On simple faults (S1), both succeed.
  CITE   : E5 results (recovery_time_s, correct_sequence per scenario)
  OPENS  : "A hand-coded rule-based supervisor was implemented as a
            baseline. On simple faults (S1: steady drift), both
            supervisors performed equivalently. On combined faults
            (S2: oscillation + overshoot), the rule-based supervisor
            failed to recover in [X]% of runs..."

P30 — Human vs autonomous efficiency (E2)
  WHAT   : Autonomous cognitive layer completes structured missions
           [X]% faster than a human operator; human wins on novel
           environment first exposure.
  CITE   : E2 results
  OPENS  : "Task completion time was compared between a human operator
            using the browser joystick interface and the fully autonomous
            cognitive layer on identical missions..."

P31 — Token scaling and context strategy (E4)
  WHAT   : Performance plateaus above [N] tokens of context; longer
           context adds latency cost without accuracy gain.
  CITE   : E4 results
  OPENS  : "To determine the optimal context window for hardware
            deployment, we varied the number of previous telemetry
            readings provided to the LLM..."

P32 — Multi-model capability benchmark (F1)
  WHAT   : 4-model × 5-capability matrix. Claude leads on 4/5
           dimensions. LLaMA strong on anomaly detection only.
  CITE   : F1 benchmark_matrix.csv, radar chart
  OPENS  : "Five capability dimensions — anomaly detection, PID
            adaptation, fault supervision, ambiguity handling, and
            diagnosis accuracy — were scored across four LLM backends..."

P33 — Latency–capability Pareto analysis (F2)
  WHAT   : Claude is on the Pareto frontier: best capability at
           acceptable latency. GPT-4o dominated by Claude on both.
  CITE   : F2 bubble chart, pareto.csv
  OPENS  : "Plotting each model as a point in latency–capability space
            reveals that Claude occupies the Pareto frontier: no other
            tested model achieves higher capability at lower or equal
            latency..."

P34 — Model selection decision
  WHAT   : Claude selected for all hardware experiments. Rationale
           stated explicitly.
  OPENS  : "Based on the Pareto analysis and the capability benchmark,
            Claude Sonnet was selected as the cognitive layer model
            for all subsequent hardware experiments. GPT-4o was
            retained as a secondary model for comparison on hardware."

§4.6  Hardware Validation: Manual Cognitive Commands
─────────────────────────────────────────────────────
SECTION OPENING — the prime result begins here
  "The following experiments constitute the primary contribution of
   this chapter: the cognitive-physical architecture is validated on
   the physical drone. Each experiment mirrors its simulation
   counterpart to quantify the sim-to-real transfer."

P35 — Hardware flight control baseline (exp_B1_althold_step_HW)
  WHAT   : Althold step response on real drone. Side-by-side table:
           sim vs hardware. Transfer confirmed or gap quantified.
  CITE   : exp_B1_althold_step_HW.py results vs B1 sim results
  OPENS  : "As a baseline, the altitude hold step response was
            reproduced on the physical drone using identical PID gains
            to those validated in simulation..."

P36 — Sim-to-real comparison table
  WHAT   : One table: rise time, overshoot, SS RMSE for sim vs hardware
           on althold, poshold, attitude.
  CITE   : HW experiment results vs B-series sim results
  OPENS  : "Table [X] summarises the sim-to-real transfer across
            three control tasks. Altitude RMSE differed by [Y] cm,
            attributable to real-world vibration not modelled in
            simulation..."

P37 — Natural language commands on real hardware (exp_C1_HW)
  WHAT   : Same C1 NL command test run on physical drone. Success
           rate, latency breakdown (WS + LLM + PID response time).
  CITE   : exp_C1_nl_to_toolchain_HW.py results vs C1 sim
  OPENS  : "The NL-to-toolchain experiment (§4.4, C1) was replicated
            on the physical drone. Command success rate was [X]%,
            compared to [Y]% in simulation..."

P38 — Hardware-specific failure modes
  WHAT   : Failures that appeared on hardware but not in sim:
           radio dropout, prop wash interference with ToF, battery
           sag causing gain drift.
  OPENS  : "Hardware trials revealed three failure modes absent from
            simulation: intermittent WebSocket dropout under RF
            interference, ToF range corruption in prop wash, and
            thrust reduction under battery sag not fully captured
            by the voltage model..."

P39 — Guardrail validation on hardware (C7 HW)
  WHAT   : Dangerous commands blocked at the same rate on hardware
           as in simulation. Any hardware-specific edge case noted.
  CITE   : C7 HW results vs C7 sim results
  OPENS  : "The safety guardrail was validated on hardware by
            submitting the same adversarial command set used in
            simulation (§4.4, C7)..."

P40 — Fault supervision on hardware (D7 HW)
  WHAT   : Deliberately bad gain applied to real drone. LLM detects
           and corrects within [N] iterations. RMSE before/after.
  CITE   : HW replication of D7
  OPENS  : "To validate autonomous fault supervision on hardware,
            we intentionally multiplied the roll angle PID gain by 5×,
            inducing visible oscillation, and activated the cognitive
            supervisor without informing it of the fault..."

§4.7  Full Integration: Verbalization → Cognitive → Physical on Hardware
────────────────────────────────────────────────────────────────────────
P41 — Experimental setup of full integration
  WHAT   : Live camera → G5 pipeline → LLM command → ESP32 physical
           drone. Indoor obstacle scenario. N runs.
  OPENS  : "The complete three-layer system was evaluated on the
            physical drone in an indoor environment with [N] obstacle
            configurations. The drone's camera feed was processed by
            the verbalization pipeline in real time..."

P42 — Mission completion results
  WHAT   : Mission completion rate, end-to-end latency, battery
           consumed per mission. Compare to D9 sim result.
  CITE   : Full integration HW results vs D9 sim
  OPENS  : "Across [N] runs, the integrated system achieved a mission
            completion rate of [X]%, with a mean end-to-end latency
            of [Y] ms from frame capture to motor response..."

P43 — Cognitive layer contribution analysis
  WHAT   : What happened on runs where it failed? Was the failure
           in perception (Ch3), cognition (LLM), or physical (PID)?
  OPENS  : "Failure analysis across unsuccessful runs attributed
            [X]% of failures to verbalization errors (incorrect scene
            description), [Y]% to cognitive layer misinterpretation,
            and [Z]% to physical control limitations..."

P44 — Chapter summary
  WHAT   : The hardware is built and works. Cognitive commands
           transfer from sim. Full integration succeeds. Hands off
           to Chapter 5.
  OPENS  : "This chapter has demonstrated that the hierarchical
            cognitive-physical architecture transfers from simulation
            to hardware with quantifiable and bounded performance
            degradation. The full integration of image verbalization,
            LLM cognition, and cascade PID control operates reliably
            on the physical drone, establishing the foundation for
            the hybrid locomotion system presented in Chapter 5."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 5 — HYBRID LOCOMOTION AND COGNITIVE ENERGY OPTIMISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAPTER OPENING PARAGRAPH
  WHAT   : State why locomotion choice matters and how the cognitive
           layer connects to it.
  OPENS  : "Flight is energetically expensive. A 50 g drone at hover
            draws [X] W, limiting flight endurance to [Y] minutes on
            a 300 mAh cell. When the terrain permits ground locomotion,
            rolling or walking consumes a fraction of this energy per
            metre. This chapter integrates a wheeled and tripod-gait
            mechanism with the cognitive layer developed in Chapter 4,
            enabling the drone to extend its operational range by
            selecting the most energy-efficient locomotion mode."

§5.1  Mechanical Integration
──────────────────────────────
P1 — Wheel mechanism design
  WHAT   : Physical description of wheel attachment, motor selection,
           weight penalty, how it affects flight characteristics.
  OPENS  : "Three omni-wheels are mounted on retractable arms below
            the drone frame, adding [X] g to the all-up mass..."

P2 — Tripod gait walking mechanism
  WHAT   : How tripod gait is implemented — leg sequence, ground
           contact pattern, speed achieved.
  OPENS  : "For uneven surfaces where rolling is not feasible,
            a tripod gait alternates between two sets of three legs..."

P3 — Impact on flight performance
  WHAT   : Added mass increases hover throttle by [X]%. New PID
           gains required. Simulation re-validated with new MASS.
  CITE   : New A-series / B-series run with updated MASS
  OPENS  : "The added locomotion mechanism increases all-up mass
            from [X] g to [Y] g, raising the hover throttle from
            [Z]% to [W]% of maximum. PID gains were retuned in
            simulation and validated on hardware..."

§5.2  Cognitive Locomotion Mode Selection
──────────────────────────────────────────
P4 — Inputs to the mode decision
  WHAT   : The cognitive layer receives: scene verbalization (terrain
           type from Ch3), battery SoC, mission distance, current
           height — and outputs: fly, roll, or walk.
  OPENS  : "The cognitive layer receives four inputs to its locomotion
            decision: the terrain description from the verbalization
            pipeline, the current battery state of charge, the
            remaining mission distance, and the current altitude..."

P5 — Decision logic and prompt design
  WHAT   : How the LLM is prompted to make this decision. Few-shot
           examples used. Safety constraints (never walk if airborne
           above 0.3 m).
  OPENS  : "The locomotion decision prompt provides the LLM with the
            four inputs and asks it to select a mode and provide a
            one-sentence rationale. Three few-shot examples anchor
            the expected output format..."

P6 — Scene-to-mode mapping accuracy
  WHAT   : On [N] terrain scenarios, cognitive layer chose the
           energy-optimal mode [X]% of the time vs a fixed-rule
           baseline.
  OPENS  : "To evaluate mode selection accuracy, [N] terrain scenarios
            spanning flat indoor floor, carpet, threshold steps, and
            outdoor gravel were presented to the cognitive layer..."

§5.3  Energy and Range Results
───────────────────────────────
P7 — Energy per metre by mode
  WHAT   : Table: J/m for fly vs roll vs walk on each terrain type.
  OPENS  : "Energy consumption per metre was measured for each
            locomotion mode on each terrain type by logging motor
            current and integrating over measured distances..."

P8 — Range extension on a benchmark mission
  WHAT   : A [X] m mission with mixed terrain. Fly-only vs cognitive
           mode selection. Range improvement in %.
  OPENS  : "A benchmark mission of [X] m was defined traversing
            [N] terrain zones. Under a fly-only policy, the drone
            exhausted its battery at [Y] m. With cognitive mode
            selection, the same mission was completed with [Z]%
            battery remaining..."

P9 — Battery-to-range relationship
  WHAT   : Plot of range vs SoC decision threshold. What threshold
           maximises range without mission failure risk.
  OPENS  : "The cognitive layer's willingness to switch to ground
            locomotion depends implicitly on battery state. We
            characterised range as a function of the SoC threshold
            below which ground locomotion is preferred..."

§5.4  Chapter Summary
──────────────────────
P10 — What this chapter demonstrated
  WHAT   : Cognitive locomotion selection extends range by [X]% vs
           fly-only. The same verbalization pipeline used for flight
           commands also drives locomotion decisions.
  OPENS  : "The hybrid locomotion system demonstrates that the
            cognitive layer built for flight command in Chapter 4
            generalises without modification to locomotion mode
            selection, extending operational range by [X]% on mixed
            terrain missions."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTER 6 — RESULTS, DISCUSSION AND CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

§6.1  Consolidated Results
───────────────────────────
P1 — Results summary table
  WHAT   : One table covering all three contributions with their
           key metric and benchmark comparison.
  OPENS  : "Table [X] consolidates the primary quantitative results
            across all three system contributions..."

P2 — Sim-to-real transfer summary
  WHAT   : Across all hardware experiments, what was the mean
           performance degradation vs simulation? Is it bounded?
  OPENS  : "Sim-to-real transfer was quantified by pairing each
            hardware experiment with its simulation counterpart.
            The mean performance degradation was [X]%, with the
            largest gap in [metric] due to [cause]..."

§6.2  Discussion
──────────────────
P3 — What the architecture gets right
  WHAT   : Timescale separation works. Cognitive layer genuinely
           adds capability over rule-based systems (E5). Verbalization
           is the right interface between perception and cognition.
  OPENS  : "The hierarchical architecture succeeds because each layer
            operates at its natural timescale without blocking the
            others..."

P4 — Limitations
  WHAT   : LLM API dependency (no offline operation). Latency floor
           at 500 ms prevents sub-second cognitive response.
           Sim-to-real gap in turbulent conditions.
  OPENS  : "The architecture has three significant limitations.
            First, it depends on external API access..."

P5 — When the cognitive layer fails
  WHAT   : Specific failure scenarios: high-ambiguity scenes, rapid
           state changes faster than LLM cadence, adversarial inputs
           that bypass guardrail.
  OPENS  : "The cognitive layer degrades most in three scenarios:
            scenes with high visual ambiguity where the verbalization
            accuracy drops below [X]%..."

§6.3  Future Work
──────────────────
P6 — Onboard LLM inference
  WHAT   : Replacing API with an onboard quantised model (e.g.
           LLaVA-7B on Jetson) would eliminate the latency floor
           and API dependency.
  OPENS  : "The most impactful near-term extension is onboard LLM
            inference using a quantised model on an edge GPU module..."

P7 — Swarm extension
  WHAT   : Multiple drones sharing a single cognitive layer via a
           shared telemetry stream.
  OPENS  : "The cognitive layer's tool API architecture supports
            multi-drone operation: a single LLM instance can receive
            telemetry from N drones and issue commands to each..."

P8 — Outdoor GPS integration
  WHAT   : Replacing optical flow with GPS for outdoor position hold,
           enabling longer-range missions.
  OPENS  : "Outdoor deployment requires replacing the optical flow
            position estimator with GPS, extending the EKF state
            to include absolute position..."

§6.4  Conclusion
─────────────────
P9 — Restate the three contributions in past tense
  WHAT   : What was built, validated, and shown.
  OPENS  : "This thesis presented three contributions to the field
            of autonomous drone systems. First, an image verbalization
            pipeline was designed and benchmarked across four
            vision-language models..."

P10 — The unifying claim
  WHAT   : The verbalization output is the connective tissue of the
           entire system — from Chapter 3 through to Chapter 5.
           One pipeline, three uses.
  OPENS  : "The unifying contribution is the image verbalization
            system: the same pipeline that converts a camera frame
            to a structured scene description also generates flight
            commands, enables fault diagnosis, and drives locomotion
            mode selection — demonstrating that general-purpose
            vision-language models can serve as the cognitive core
            of a physical autonomous system."

P11 — Final sentence
  WHAT   : The field implication.
  OPENS  : "These results suggest that the primary barrier to
            generalised drone autonomy is not physical control —
            which is mature — but scene understanding and high-level
            reasoning, and that vision-language models are now capable
            enough to fill that role."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WRITING RULES (follow these for every paragraph)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.  One idea per paragraph. If you find yourself writing "also" or
    "additionally", that is a second paragraph.

2.  Fill in every [X], [Y], [N] with the actual number from your results
    CSV before considering a paragraph done.

3.  Every paragraph that cites an experiment must cite the specific
    metric (not just the experiment name). Wrong: "D7 showed improvement."
    Right: "D7 showed RMSE reduction from 8.3 cm to 3.1 cm across 5 runs."

4.  Simulation paragraphs (§4.3) end by pointing forward to the hardware
    paragraph that uses the same metric. Never let a sim result be the
    last word on a topic.

5.  The first sentence of each paragraph is the claim. The rest is
    evidence. If your first sentence does not make a falsifiable claim,
    rewrite it.

6.  Write §4.2 (hardware design) before §4.3 (simulator). The reader
    must know the hardware exists before being told the sim mirrors it.

PARAGRAPH COUNT ESTIMATE
  Ch1:  9 paragraphs   (~900 words)
  Ch2:  11 paragraphs  (~1,100 words)
  Ch3:  19 paragraphs  (~2,300 words)
  Ch4:  44 paragraphs  (~5,500 words)
  Ch5:  10 paragraphs  (~1,200 words)
  Ch6:  11 paragraphs  (~1,100 words)
  ─────────────────────────────────
  TOTAL: ~104 paragraphs / ~12,100 words (body text, excluding figures/tables)
  With figures, tables, captions, references: ~18,000–22,000 words
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
