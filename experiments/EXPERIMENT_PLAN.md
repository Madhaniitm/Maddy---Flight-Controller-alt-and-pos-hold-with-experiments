# Experiment Plan — LLM-Guided Autonomous Micro-Quadrotor Paper
# Total: 38 Experiments across 6 Sections
# Last updated: 2026-04-15
# Revision: Added multi-LLM comparison, real hardware validation, photorealistic vision, energy/cost analysis

---

## PAPER TITLE
"Vision-Language Model as Situational-Aware Autonomous Supervisor for Micro-Quadrotor UAVs:
A Hierarchical Architecture with Physics-Accurate Simulation and Hardware Validation"

## PAPER FLOW
Sim Validation → Controller Baseline → Human Command → Autonomous Supervision → Architecture Analysis → Multi-LLM Benchmark
      A                   B                  C                    D                       E                      F

---

## ARCHITECTURE (must be clear in paper)
```
Real/Rendered Camera + Telemetry
              ↓
   LLM Supervisor (Claude / GPT-4o / Gemini / LLaMA-3)
              ↓ tool calls
      keyboard_server.py   ← translates LLM decisions to PWM/WebSocket commands
              ↓ WebSocket
   ESP32-S3 firmware       ← low-level flight controller (Madgwick + 9-state EKF + cascaded PID)
              ↓
         Motors (brushed)
```
LLM is NOT the flight controller. LLM is the autonomous supervisor.
Timescale separation: inner loop 4kHz / LLM outer loop 0.5–2Hz — this is the key design principle.

---

## HARDWARE PLATFORM (for H-series and B-hardware)
- Flight controller: ESP32-S3 Sense (onboard camera OV2640, 2MP)
- Firmware: Madgwick (200Hz) + 9-state EKF + cascaded PID (altitude hold confirmed working)
- Camera stream: ESP32-S3 Sense → HTTP → LLM vision input (real hardware, not synthetic)
- This enables: real camera frames for D1–D4, real altitude hold for B1-HW, real end-to-end for D9-HW

---

## LLM MODELS TESTED (Section F and comparison columns in D/E)
| Model | Type | Size | Notes |
|-------|------|------|-------|
| Claude 3.7 Sonnet | Closed-source | — | Primary model (this paper's system) |
| GPT-4o | Closed-source | — | Comparison |
| Gemini 1.5 Pro | Closed-source | — | Comparison |
| LLaMA-3-70B (via Ollama) | Open-source | 70B | Reproducibility baseline |

Comparison experiments: D6, D7, E5, F1–F3.
Rationale: multi-LLM comparison transforms single-model demo into generalisable benchmarking study.

---

## SECTION A — Simulation Validation (6 experiments)
Purpose: Prove the simulator is physics-accurate before using it as the test bed.
Script dependency: drone_sim.py physics classes imported directly (no WebSocket, no API)

---

### EXP-A1: Free-Fall Physics Validation
**What:** Disarm drone at z=1m, let it fall. Compare simulated z(t) vs analytical z = 1 - ½gt²
**Measure:** RMS error between sim trajectory and analytical solution over 0–0.45s
**Expected result:** RMSE < 1 mm — proves Newton integration is correct
**Output:** Plot z(t) sim vs analytical, print RMSE
**Script:** exp_A1_freefall.py

### EXP-A2: Madgwick Filter Convergence
**What:** Start drone tilted at roll=30°, run Madgwick filter. Plot estimated roll vs true roll.
**Measure:** Convergence time (time to reach within 1° of true), steady-state error
**Expected result:** Convergence < 2s at β=0.03 matching firmware default
**Output:** Plot roll_estimated vs roll_true vs time
**Script:** exp_A2_madgwick.py

### EXP-A3: EKF Altitude Estimation Accuracy
**What:** Hover at true z=1m. Inject ToF noise σ=5mm. Compare EKF-estimated z vs true z.
**Measure:** Estimation RMSE, noise attenuation ratio (raw ToF noise / EKF error)
**Expected result:** EKF reduces noise by >80% (RMSE < 1mm vs 5mm raw ToF)
**Output:** Plot true_z, noisy_tof, ekf_z vs time. Print noise rejection ratio.
**Script:** exp_A3_ekf_altitude.py

### EXP-A4: Ground Effect Validation
**What:** Hover drone at altitudes: 2cm, 3cm, 5cm, 8cm, 10cm, 15cm, 20cm, 30cm, 50cm.
Record effective thrust multiplier at each height.
**Measure:** Thrust multiplier = F_actual / F_no_ground_effect at each height
**Expected result:** Exponential decay matching Cheeseman-Bennett: 1 + GE_COEFF*exp(-z/(GE_DECAY*R_ROTOR))
**Output:** Plot thrust_multiplier vs z/R_rotor with fitted curve
**Script:** exp_A4_ground_effect.py

### EXP-A5: Motor Lag Model Validation
**What:** Step PWM command from 0 to full. Record ω(t) from simulation.
Fit first-order lag: ω(t) = ω_max*(1 - exp(-t/τ))
**Measure:** Fitted τ vs TAU_MOTOR=0.030s, R² of fit
**Expected result:** Fitted τ ≈ 30ms, R² > 0.99
**Output:** Plot ω_sim vs ω_model vs time. Print fitted τ and R².
**Script:** exp_A5_motor_lag.py

### EXP-A6: Battery Discharge Thrust Degradation
**What:** Fly at constant hover throttle from 100% to 5% SOC. Record voltage and thrust vs time.
**Measure:** Thrust at 100%, 80%, 60%, 40%, 20% SOC. Verify thrust ∝ V_term²
**Expected result:** Thrust at 20% SOC ≈ 70% of full-charge thrust (voltage drops ~16%)
**Output:** Plot thrust vs SOC, voltage vs SOC. Show V² relationship.
**Script:** exp_A6_battery.py

---

## SECTION B — Low-Level Controller Baseline (5 sim + 2 hardware = 7 experiments)
Purpose: Characterise PID+EKF performance WITHOUT LLM. Baseline for later comparison.
Hardware sub-experiments validate that sim results transfer to real ESP32-S3 platform.

---

### EXP-B1: Altitude Hold Step Response (Simulation)
**What:** Arm and takeoff. Enable alt-hold. Step setpoint: 1.0→1.3→1.6m
**Measure:** Overshoot %, rise time (10→90%), settling time (±5%), steady-state RMSE
**Expected result:** Overshoot <10%, settling <5s, RMSE <2cm
**Output:** Plot z_true, z_ekf, z_setpoint vs time with annotations. Metrics table.
**Script:** exp_B1_althold_step.py ✓ DONE

### EXP-B2: Position Hold Disturbance Rejection (Simulation)
**What:** Hover at (0,0,1m) with pos-hold active. At t=5s apply lateral impulse (Fx=0.05N for 0.2s).
**Measure:** Max XY drift, return-to-hold time, RMSE after disturbance
**Expected result:** Max drift <20cm, return within 4s, SS-RMSE <5cm
**Output:** XY trajectory plot. Time series of XY error. Metrics table.
**Script:** exp_B2_poshold_disturbance.py ✓ DONE

### EXP-B3: Attitude Stabilisation Step Response (Simulation)
**What:** At hover, command roll_des=10°, hold 2s, return to 0°. Repeat for pitch.
**Measure:** Rise time, overshoot, settling time for roll and pitch axes
**Expected result:** Rise time <0.5s, overshoot <10%, settling <0.5s
**Output:** Plot roll_true vs roll_des vs time. Annotated metrics.
**Script:** exp_B3_attitude_step.py ✓ DONE

### EXP-B4: Combined Alt+Pos Hold Under Steady Wind (Simulation)
**What:** Hover with both holds active. Apply constant Fx=0.02N wind for 40s.
**Measure:** Steady-state XY RMSE, altitude RMSE, P-only vs PID integral comparison
**Expected result:** XY SS RMSE <5cm (integral eliminates wind error), Z RMSE <2cm
**Output:** 4-panel plot: trajectory, altitude, XY error, X error with decay envelope.
**Script:** exp_B4_combined_hold_wind.py ✓ DONE

### EXP-B5: Hover Throttle vs Battery SOC (Simulation)
**What:** Analytically compute and physics-validate hover PWM at each SOC (100%→20%).
**Measure:** Hover PWM vs SOC curve, sim hover-find vs analytical delta (ΔPW)
**Expected result:** ΔPW=238 over SOC range, sim-analytical match within ±6 PWM
**Output:** Hover PWM curve + terminal voltage vs SOC. Sim validation annotations.
**Script:** exp_B5_hover_soc.py ✓ DONE

### EXP-B1-HW: Altitude Hold Step Response (Real Hardware) ★ KEY
**What:** Same step profile as B1 (1.0→1.3→1.6m) on real ESP32-S3 drone with ToF sensor.
**Measure:** Same 4 metrics as B1-sim. Sim-to-real gap: Δovershoot, Δrise_time, ΔRMSE.
**Expected result:** Real RMSE within 2× sim RMSE. Qualitative response shape matches.
**Sim-to-real gap closes:** Physics validation (A series) → sim baseline (B1) → real flight (B1-HW).
**Output:** Same 4-panel plot as B1 overlaid with sim result. Gap table.
**Script:** exp_B1_hw_althold_step.py (hardware, requires ESP32-S3 drone)
**Note:** This single experiment is sufficient to establish sim-to-real credibility for RA-L.

### EXP-D9-HW: End-to-End Autonomous Mission (Real Hardware) ★ KEY
**What:** Same D9 mission on real drone. LLM reads real ESP32-S3 Sense camera frames.
LLM supervisor commands the real flight controller via WebSocket.
**Measure:** Mission success rate (3 runs). Real camera → LLM decision latency. Landing accuracy.
**Expected result:** >1/3 runs complete without crash. LLM correctly interprets real images.
**Output:** Real flight trajectory (from altimeter + heading). Camera frames at each decision.
**Note:** Even partial success (takeoff + hover + land) is publishable as "first real-hardware demo."
**Script:** exp_D9_hw_end_to_end.py (hardware, real camera)

---

## SECTION C — Human Command Experiments (8 experiments)
Purpose: Show LLM correctly interprets human natural language and executes precise flight actions.
Script dependency: keyboard_server.py tool execution logic + Claude API

---

### EXP-C1: Natural Language → Tool Chain Execution
**What:** Human types: "take off and hover at 1 metre". Log complete LLM tool call sequence.
**Measure:**
  - Tool call sequence correctness
  - Achieved altitude vs commanded (1.0m target)
  - Total API calls, tokens, wall time
  - Altitude error at steady state
**Expected result:** LLM calls tools in correct order, altitude within ±10cm of 1m
**Output:** Tool call trace table. Timeline plot of tool calls vs drone altitude.
**Script:** exp_C1_nl_to_toolchain.py

### EXP-C2: Command Ambiguity Resolution
**What:** Send 6 progressively ambiguous commands to LLM:
  1. "go to 2 metres"              (explicit)
  2. "climb to 2m"                 (paraphrase)
  3. "go higher"                   (relative, no number)
  4. "go up a bit"                 (vague relative)
  5. "ascend slowly to a safe height" (abstract)
  6. "I want it higher"            (indirect)
**Measure:** Correct altitude interpretation rate. Clarification requests on ambiguous commands.
**Expected result:** Commands 1-2 exact, 3-4 reasonable increment, 5-6 LLM asks or safe default
**Output:** Table: command → LLM interpretation → target_m → correct?
**Script:** exp_C2_ambiguity.py

### EXP-C3: Multi-Turn Conversational Mission
**What:** 5-turn conversation: arm → go to 1.5m → hold → rotate 90° → land
**Measure:** State tracking across turns. No repeated arm/takeoff. Tool calls per turn.
**Expected result:** LLM correctly infers drone state from history at each turn.
**Output:** Conversation transcript with tool calls. State tracking table per turn.
**Script:** exp_C3_multiturn.py

### EXP-C4: Human Correction Mid-Mission
**What:** "hover at 0.5m" → executing → interrupt: "actually go to 1.2m"
**Measure:** Does LLM set new target without re-arming? Time from correction to new command.
**Expected result:** LLM directly calls set_altitude_target(1.2) without full re-sequence
**Output:** Tool call trace before/after correction. Altitude plot showing smooth transition.
**Script:** exp_C4_mid_mission_correction.py

### EXP-C5: Human Describes Problem → LLM Diagnoses and Fixes
**What:** Inject high roll_angle_kp (×3) → human says "oscillating badly on roll"
LLM must: analyze_flight() → suggest_pid_tuning() → apply_tuning() without human specifying fix.
**Measure:** Roll RMSE before/after. Correct tool sequence. LLM-suggested Kp vs correct value.
**Expected result:** RMSE reduces >50%. LLM identifies roll_angle_kp as problem.
**Output:** Roll error plot before/after. Tool call trace. Gain table.
**Script:** exp_C5_human_describes_problem.py

### EXP-C6: Human Goal → LLM Mission Planning
**What:** "do a square pattern at 1 metre height" — LLM must decompose entirely.
**Measure:** Correct waypoint count. Trajectory squareness. Waypoint position error.
**Expected result:** LLM plans 4-leg square, executes in sequence.
**Output:** XY trajectory plot. LLM plan printout. Waypoint error table.
**Script:** exp_C6_mission_planning.py

### EXP-C7: Human Safety Override
**What:** LLM mid-mission → "stop everything and come down now"
**Measure:** Time from command to landing start. Correct disable sequence. No unnecessary steps.
**Expected result:** LLM immediately executes landing. Latency < 3s (1 API call).
**Output:** Tool call trace. Timeline from command to motors-off.
**Script:** exp_C7_safety_override.py

### EXP-C8: Three-Mode Comparison — Manual vs NL-Commanded vs Full-Auto
**What:** Same mission (takeoff → 1m → hold 10s → land) in 3 modes:
  Mode A — Manual: scripted fixed PWM, no LLM
  Mode B — NL-commanded: human gives NL, LLM executes
  Mode C — Full-auto: LLM + camera, no human
**Measure:** Altitude RMSE, mission time, human inputs, API calls, tokens per mode.
**Output:** Side-by-side altitude plots. Comparison table.
**Script:** exp_C8_three_mode_comparison.py

---

## SECTION D — LLM Autonomous Supervision (9 experiments)
Purpose: Show LLM reading camera + telemetry and making decisions without human in loop.
Core research contribution of the paper.

Camera note: D1–D4 use REAL ESP32-S3 Sense camera frames (OV2640) where the drone is
physically positioned in front of target scenes. Removes synthetic image criticism entirely.
Where physical setup is impractical, photorealistic Blender renders at matching FOV are used.

---

### EXP-D1: Vision-Only Scene Classification Accuracy
**What:** Position real drone (or photorealistic Blender render) in front of 10 scenes:
open space, wall close, wall far, floor pattern, ceiling, obstacle left, obstacle right,
dark room, textured floor, bright overexposure.
Capture real/rendered frames via ESP32-S3 Sense. Send to LLM via analyze_frame.
**Measure:** Classification accuracy %, response time per frame, token usage per frame.
**Expected result:** >80% correct on clear scenes, lower on ambiguous/dark.
**Output:** Confusion-matrix table. Accuracy vs scene type. Example frames shown.
**Script:** exp_D1_vision_classification.py

### EXP-D2: Vision-to-Action — Full Autonomy Navigation
**What:** autonomy_loop (full_auto mode): "Move forward slowly until wall is close, then stop"
Camera: real ESP32-S3 or photorealistic sim render.
**Measure:** Correct direction/stop decisions. Trajectory vs intended. False moves.
**Expected result:** LLM moves forward, detects wall approach, stops correctly.
**Output:** XY trajectory. Per-iteration: scene description, action, correct?
**Script:** exp_D2_full_auto_navigation.py

### EXP-D3: Vision-to-Action — Human-in-Loop Navigation
**What:** Same goal as D2 in human_loop mode. (a) human always approves, (b) 30% reject.
**Measure:** Mission time: full_auto vs approve_all vs 30pct_reject. LLM re-planning after reject.
**Output:** Timeline comparison. Rejection→adaptation trace.
**Script:** exp_D3_human_loop_navigation.py

### EXP-D4: Situational Awareness — Obstacle Avoidance Decision
**What:** Drone faces wall at 20cm, 30cm, 50cm, 100cm. 10 trials per distance = 40 total.
Real or photorealistic camera frames. LLM must decide: ascend, descend, or turn.
**Measure:** Correct decision rate per distance. LLM reasoning confidence.
**Expected result:** High accuracy at 20–30cm, lower at 100cm (barely visible).
**Output:** Decision accuracy vs distance. Example correct/wrong frames shown.
**Script:** exp_D4_obstacle_avoidance.py

### EXP-D5: Autonomous Multi-Waypoint Mission
**What:** "Fly a 1m × 1m square at 0.8m altitude and return to start" — no human in loop.
5 repeated runs.
**Measure:** Trajectory squareness. Waypoint error. Success rate. Tool calls per run.
**Expected result:** Recognisable square trajectory. >60% success rate.
**Output:** XY overlay of 5 runs. Success rate. Waypoint error table.
**Script:** exp_D5_autonomous_waypoint.py

### EXP-D6: Telemetry-Aware Anomaly Detection and Recovery ★ MULTI-LLM
**What:** Inject 4 faults: yaw spin, roll oscillation, motor imbalance, altitude drift.
LLM must detect via get_telemetry() and execute correct recovery action autonomously.
**Multi-LLM:** Run all 4 fault scenarios with Claude, GPT-4o, Gemini, LLaMA-3.
**Measure:** Detection rate (N/4). Correct recovery action rate. Time to recovery. Per-model comparison.
**Expected result:** Closed-source models: ≥3/4 detected + correct. LLaMA-3: ≥2/4.
**Output:** 4×4 table (fault × model): detected? response? correct? time.
Bar chart comparing models on detection rate and recovery accuracy.
**Script:** exp_D6_anomaly_detection.py

### EXP-D7: LLM PID Gain Adaptation — Iterative Closed Loop ★ MULTI-LLM
**What:** Inject bad gains (roll_angle_kp = 3×). LLM iteratively: analyze → suggest → apply.
3 iterations. Measure RMSE per iteration.
**Multi-LLM:** Run full 3-iteration sequence with Claude, GPT-4o, Gemini, LLaMA-3.
**Measure:** RMSE per iteration per model. Convergence rate. Suggested gain vs correct gain.
**Expected result:** RMSE decreases each iteration. Closed-source converges within 3 iterations.
**Output:** RMSE vs iteration per model (4 curves). Gain trajectory per model. Convergence table.
**Script:** exp_D7_pid_adaptation.py

### EXP-D8: Sensor Dropout — LLM Fault Tolerance Mid-Mission
**What:** LLM hovering at 1m. At t=5s: ToF dropout (altitude = None in telemetry).
LLM must detect → disable_altitude_hold → command safe descent → land.
**Measure:** Detection (Y/N). Correct recovery sequence. Time to safe descent. Landing outcome.
**Expected result:** LLM detects None altitude, commands safe descent.
**Output:** Timeline: dropout → detection → recovery → outcome. Altitude plot.
**Script:** exp_D8_sensor_dropout.py

### EXP-D9: End-to-End Autonomous Mission — Full Pipeline ★ FLAGSHIP
**What:** "Take off, explore the room for 15 seconds, then return to start and land"
Full autonomy: arm → hover → capture_image loop → navigate → return → land.
5 simulation runs + 3 real hardware runs (D9-HW, see B-series hardware section).
**Measure:**
  - Mission success rate (sim: 5 runs, hardware: 3 runs)
  - Tool calls, API calls, tokens, wall time per mission
  - Trajectory coverage area
  - Landing position error from start
**Expected result:** Sim: >60% success. Hardware: ≥1/3 success (real camera + real flight).
**Output:** Trajectory per run. Success/fail breakdown. API usage table. Real camera frames.
This is the flagship experiment.
**Script:** exp_D9_end_to_end.py + exp_D9_hw_end_to_end.py

---

## SECTION E — Architecture Analysis (5 experiments)
Purpose: Justify the hierarchical design. Show WHY LLM cannot be the inner-loop controller.

---

### EXP-E1: LLM API Latency Distribution + Cost Analysis ★ EXTENDED
**What:** 50 consecutive Claude API calls. Record latency + token count + cost per call.
**Extended:** Repeat for GPT-4o and Gemini to show cost/latency tradeoff across providers.
**Measure:** Min, mean, median, P95, max latency. Cost per call (USD). Calls/min throughput.
Show: LLM at 0.5–2Hz is fundamentally incompatible with 4kHz PID inner loop.
**Expected result:** Mean latency 0.5–2s. Inner loop requires 250µs — 4 orders of magnitude gap.
**Output:** Latency histogram per model. Cost per 100 calls table. Annotation: inner loop req = 0.25ms.
**Script:** exp_E1_api_latency.py

### EXP-E2: Human-in-Loop vs Full-Auto Mission Time
**What:** Same 5-step mission in human_loop vs full_auto mode. 3 runs each.
**Measure:** Total mission time, human approval wait time, trajectory quality (RMSE).
**Expected result:** human_loop 2–5× slower. Trajectory quality similar.
**Output:** Mission time comparison bar chart. Approval wait breakdown.
**Script:** exp_E2_human_vs_auto_time.py

### EXP-E3: Conversation Memory — Multi-Turn State Retention
**What:** 3-turn mission where turn 3 requires recalling turn 1 result.
Inject 20-message history between turn 1 and 3.
**Measure:** Does LLM correctly recall turn 1 result across 20 intervening messages?
**Expected result:** Correct recall. Memory survives up to MAX_HISTORY_MESSAGES=40.
**Output:** Conversation transcript. Memory recall accuracy: correct/incorrect.
**Script:** exp_E3_memory_retention.py

### EXP-E4: Token Usage vs Mission Complexity + Cost Scaling
**What:** 5 missions of increasing complexity (1-step to 5-step).
**Extended:** Record USD cost per mission (Claude + GPT-4o rates). Cost vs steps.
**Measure:** Input tokens, output tokens, API calls, USD cost per mission.
**Expected result:** Near-linear token growth. Cost per mission < $0.10 for 5-step.
**Output:** Tokens vs steps plot. Cost vs steps plot. API calls vs steps.
**Script:** exp_E4_token_scaling.py

### EXP-E5: LLM Supervisor vs Rule-Based Supervisor ★ MULTI-LLM + KEY
**What:** Rule-based supervisor (if roll>5°→set_trim; if alt_error>0.2m→adjust throttle).
Run LLM supervisor (4 models) vs rule-based on 3 scenarios:
  Scenario 1: Simple hover with steady drift
  Scenario 2: Combined roll oscillation + altitude overshoot
  Scenario 3: Sensor noise causing EKF jitter
**Measure:** Recovery time, RMSE, handling of combined/novel faults. Per-model comparison.
**Expected result:** Rule-based handles Scenario 1. LLM handles Scenarios 2–3 better.
All 4 LLMs outperform rule-based on combined/novel faults.
**Output:** Recovery time and RMSE table: 4 LLMs + rule-based × 3 scenarios.
Highlight: cases where rule-based fails and LLM succeeds — this is the key differentiator.
**Script:** exp_E5_llm_vs_rules.py

---

## SECTION F — Multi-LLM Benchmark Summary (3 experiments)
Purpose: Synthesise multi-LLM results from D6, D7, E5 into a standalone benchmark.
Makes the paper a reference for future LLM-for-UAV research.

---

### EXP-F1: LLM Benchmark — Task Success Rate Across All Capabilities
**What:** Compile results from D6, D7, E5, C2 (ambiguity), C5 (diagnosis) per model.
Score each model on: anomaly detection, PID adaptation, rule-based comparison, ambiguity handling.
**Measure:** Per-model score across 5 capability dimensions. Radar chart.
**Output:** 4×5 capability matrix (model × task). Radar chart. Overall rank table.
**Script:** exp_F1_benchmark_summary.py (analysis only, no new API calls)

### EXP-F2: LLM Benchmark — Latency vs Capability Tradeoff
**What:** Plot each model: (x=mean latency, y=task success rate, size=cost/call).
**Measure:** Pareto frontier of latency vs accuracy. Best model per deployment constraint.
**Expected result:** Claude/GPT-4o on Pareto front. LLaMA-3 lower accuracy but zero API cost.
**Output:** Scatter plot with Pareto frontier. Deployment recommendation table.
**Script:** exp_F2_latency_vs_capability.py (analysis only)

### EXP-F3: Open-Source vs Closed-Source — Reproducibility Assessment
**What:** Re-run D6 Scenario (a) and D7 Iteration 1 with LLaMA-3-70B (local, no API).
Measure performance gap vs Claude on same tasks.
**Rationale:** Any researcher can reproduce with LLaMA-3 via Ollama — removes closed-source reproducibility concern.
**Measure:** Task success rate gap. Latency gap. Qualitative reasoning quality comparison.
**Output:** Side-by-side tool call trace (Claude vs LLaMA-3). Performance gap table.
**Script:** exp_F3_opensource_reproducibility.py

---

## SUMMARY TABLE

| Section | Count | Needs API | Needs Hardware | Core Question |
|---------|-------|-----------|----------------|---------------|
| A — Sim Validation        | 6  | No  | No  | Is the simulator physics-accurate? |
| B — Controller Baseline   | 5+2| No  | 2 exp| How well does the PID work? Sim-to-real gap? |
| C — Human Commands        | 8  | Yes | No  | Can LLM interpret and execute human NL commands? |
| D — LLM Autonomous        | 9+1| Yes | 1 exp| Can LLM see, decide, act without human? |
| E — Architecture Analysis | 5  | Partial | No | Why hierarchical? LLM vs rule-based? |
| F — Multi-LLM Benchmark   | 3  | Partial | No | Which LLM is best for UAV supervision? |
| **TOTAL**                 | **38** | **26 API** | **3 HW** | |

Hardware experiments: B1-HW, D9-HW (both on ESP32-S3 drone with OV2640 camera)
Multi-LLM experiments: D6, D7, E5, F1, F2, F3 (Claude + GPT-4o + Gemini + LLaMA-3)

---

## SCRIPT STATUS

| Script | Section | Status |
|--------|---------|--------|
| exp_A1_freefall.py | A1 | TODO |
| exp_A2_madgwick.py | A2 | TODO |
| exp_A3_ekf_altitude.py | A3 | TODO |
| exp_A4_ground_effect.py | A4 | TODO |
| exp_A5_motor_lag.py | A5 | TODO |
| exp_A6_battery.py | A6 | TODO |
| exp_B1_althold_step.py | B1 | ✓ DONE |
| exp_B2_poshold_disturbance.py | B2 | ✓ DONE |
| exp_B3_attitude_step.py | B3 | ✓ DONE |
| exp_B4_combined_hold_wind.py | B4 | ✓ DONE |
| exp_B5_hover_soc.py | B5 | ✓ DONE |
| exp_B1_hw_althold_step.py | B1-HW | TODO (hardware) |
| exp_C1_nl_to_toolchain.py | C1 | ✓ DONE |
| exp_C2_ambiguity.py | C2 | ✓ DONE |
| exp_C3_multiturn.py | C3 | ✓ DONE |
| exp_C4_mid_mission_correction.py | C4 | ✓ DONE |
| exp_C5_human_describes_problem.py | C5 | ✓ DONE |
| exp_C6_mission_planning.py | C6 | ✓ DONE |
| exp_C7_safety_override.py | C7 | ✓ DONE |
| exp_C8_three_mode_comparison.py | C8 | ✓ DONE |
| exp_D1_vision_classification.py | D1 | TODO |
| exp_D2_full_auto_navigation.py | D2 | TODO |
| exp_D3_human_loop_navigation.py | D3 | TODO |
| exp_D4_obstacle_avoidance.py | D4 | TODO |
| exp_D5_autonomous_waypoint.py | D5 | TODO |
| exp_D6_anomaly_detection.py | D6 | TODO (multi-LLM) |
| exp_D7_pid_adaptation.py | D7 | TODO (multi-LLM) |
| exp_D8_sensor_dropout.py | D8 | TODO |
| exp_D9_end_to_end.py | D9 | TODO |
| exp_D9_hw_end_to_end.py | D9-HW | TODO (hardware) |
| exp_E1_api_latency.py | E1 | TODO (multi-model) |
| exp_E2_human_vs_auto_time.py | E2 | TODO |
| exp_E3_memory_retention.py | E3 | TODO |
| exp_E4_token_scaling.py | E4 | TODO (cost extended) |
| exp_E5_llm_vs_rules.py | E5 | TODO (multi-LLM) |
| exp_F1_benchmark_summary.py | F1 | TODO (analysis) |
| exp_F2_latency_vs_capability.py | F2 | TODO (analysis) |
| exp_F3_opensource_reproducibility.py | F3 | TODO (LLaMA-3) |

---

## KEY PAPER CLAIMS (experiments must support these)
1. The simulator is physics-accurate — validated against 6 peer-reviewed models (A1–A6)
2. The low-level PID controller works reliably — sim (B1–B5) + hardware confirmation (B1-HW)
3. LLM correctly interprets human NL commands and executes precise flight actions (C1–C8)
4. LLM can operate as a situational-aware autonomous supervisor using camera + telemetry (D1–D9)
5. The hierarchical architecture is correct — LLM is supervisor, not inner-loop (E1–E5)
6. LLM handles faults and novel situations that rule-based systems cannot (D6, D8, E5)
7. Performance generalises across LLM model families — not a single-model demo (D6, D7, E5, F1–F3)
8. The system works on real hardware with a real camera — not simulation-only (B1-HW, D9-HW)

---

## NOTES FOR PAPER WRITING
- Always describe LLM as "supervisor" or "autonomous companion", never "flight controller"
- Inner loop = Madgwick (200Hz) + EKF (250Hz) + cascaded PID (4kHz) — untouched by LLM
- Outer loop = LLM (0.5–2Hz) decides setpoints, mode switches, gain changes, mission steps
- Timescale separation is the KEY design principle — quantified in E1
- find_hover_throttle() is novel: LLM-initiated adaptive calibration using live telemetry
- suggest_pid_tuning() is novel: LLM closes the loop on gain selection (D7)
- autonomy_loop with human-in-loop vs full_auto: two deployment modes in one system
- Multi-LLM benchmark (F series) makes this a reference paper, not just a single-system demo
- B1-HW + D9-HW with real ESP32-S3 Sense camera removes simulation-only limitation
- LLaMA-3 baseline (F3) addresses reproducibility — anyone can replicate without API access
