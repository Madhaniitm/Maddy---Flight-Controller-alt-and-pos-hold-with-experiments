# MCP Experiments — Maddy Flight Controller
## Vision-Language MCP Agent for Real-Time Autonomous Drone Control

> **All experiments in this series run on REAL HARDWARE only.**
> No simulation. No DroneState/PhysicsLoop. Every trial touches the actual drone.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP AGENT SYSTEM                            │
│                                                                 │
│  Human Chat ──┐                                                 │
│  Vision Frame─┼──► MCP Client (mcp_client.py)                  │
│  Auto trigger─┘         │ picks model (Claude/GPT-4o/Gemini/   │
│                          │            LLaVA/Gemma)              │
│                          │ formats tools from MCP server        │
│                          ▼                                      │
│               MCP Server (mcp_server.py)  port 5001            │
│               JSON-RPC 2.0 over HTTP                           │
│               tools: arm/disarm/takeoff/land/move/yaw/         │
│                      get_telemetry/capture_frame/              │
│                      set_altitude/emergency_stop               │
│                          │                                      │
│                          ▼                                      │
│            Drone HTTP API   (ESP32-S3 on LAN)                  │
│            GET  http://{DRONE_IP}/telemetry                    │
│            POST http://{DRONE_IP}/command  {cmd, value}        │
│            GET  http://{DRONE_IP}/capture  → JPEG              │
│                          │                                      │
│                          ▼                                      │
│            Maddy Flight Controller.ino (Arduino firmware)      │
│            4 kHz PID loop — 3-axis stabilisation               │
└─────────────────────────────────────────────────────────────────┘

Outputs per decision:
  • Drone action (command sent to FC)
  • TTS speech (pyttsx3 / gTTS)
  • Chat reply in terminal
  • Decision log → results/
```

---

## Configuration (shared across all experiments)

```python
DRONE_IP         = "192.168.4.1"    # ESP32-S3 access-point IP
MCP_SERVER_URL   = "http://localhost:5001/mcp"
KS_URL           = "http://localhost:8080"   # keyboard_server (optional relay)
ESP32_CAPTURE    = f"http://{DRONE_IP}/capture"
N_TRIALS         = 5                # per condition (real hardware — conservative)
PREFLIGHT_CHECK  = True             # abort if drone unreachable
```

---

## Experiment Series

### J — MCP Server Creation & Validation (2 experiments)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| J1 | server_tool_reliability | Call each MCP tool 50×, measure success/fail | success_rate (Wilson CI) |
| J2 | server_latency_benchmark | Round-trip per tool: MCP call → drone ACK | latency_ms (Bootstrap CI) |

**J1** — 12 tools × 50 calls = 600 real drone commands. Covers: arm, disarm,
takeoff, land, move_forward/back/left/right, set_altitude, get_telemetry,
capture_frame, emergency_stop. Measures: success_rate, error_type breakdown.

**J2** — Measures three latency segments:
  - `client→mcp_server` (JSON-RPC overhead)
  - `mcp_server→drone HTTP` (network + FC processing)
  - `drone→telemetry_confirm` (round-trip confirmation)

---

### K — MCP Client Creation & Multi-LLM Validation (2 experiments)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| K1 | client_multi_llm_connect | All 5 models connect, call tools, get telemetry | connect_rate, tool_call_accuracy |
| K2 | client_vision_pipeline | Each model receives frame + prompt, issues correct action | vision_accuracy (Wilson CI) |

**K1** — Models: Claude-3.5-Sonnet, GPT-4o, Gemini-1.5-Flash, LLaVA-13B (Ollama),
Gemma-3 (Ollama). N=5 per model. Simple command: "arm and report status."
Measures: tool call issued, telemetry read correctly, reply quality score.

**K2** — Each model receives the same real JPEG from drone camera + "what action?"
10 frames (5 near-wall, 5 open space) × 5 models. Correct action = stop if wall
< 30 cm, move_forward if clear.

---

### L — Multilanguage Command Understanding (1 experiment)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| L1 | multilanguage_commands | Drone responds correctly to commands in 5 languages | command_accuracy (Wilson CI) per language |

**L1** — Languages: English, Hindi (हिंदी), Tamil (தமிழ்), Spanish, French.
Commands per language: arm, takeoff, land, move_forward, emergency_stop.
5 commands × 5 languages × N=5 = 125 trials. Model: Claude (best multilingual).
Measures: correct drone action issued, translation latency.

---

### M — Prompting Technique Comparison (4 experiments)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| M1 | prompt_zero_vs_few_shot | 0/1/3/5-shot examples for obstacle avoidance | task_success_rate, api_calls |
| M2 | prompt_cot_vs_react | CoT vs ReAct vs direct-action on waypoint mission | success_rate, latency, tokens |
| M3 | prompt_system_design | Terse vs verbose vs structured system prompt | decision_accuracy, cost |
| M4 | prompt_safety_constraints | Where safety rules live: system/user/tool desc | safety_violation_rate |

**M1** — Zero-shot: "Navigate to 1m, avoid wall." Few-shot: add 1/3/5 examples of
past decisions. Scenario: drone hovers, wall appears. N=5 per shot-count.

**M2** — Same waypoint task, three prompt strategies:
  - Direct: "Go to x=1,y=0,z=1."
  - CoT: "Think step by step then act."
  - ReAct: "Reason: [think]. Act: [tool]."

**M3** — System prompt variants: (a) 50-word terse, (b) 300-word full spec,
(c) structured with explicit JSON output schema. Same task all three.

**M4** — Safety constraint "stop if <25 cm" placed in: (a) system prompt only,
(b) user message each turn, (c) tool description only, (d) all three.
Measure: how often drone gets too close before stopping.

---

### N — LLM Parameter Tuning (3 experiments)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| N1 | param_temperature_sweep | temp 0.0→1.0 on real hover-and-navigate | action_variance, crash_rate |
| N2 | param_model_comparison | Claude Haiku vs Sonnet vs GPT-4o-mini vs GPT-4o | accuracy, latency, cost |
| N3 | param_context_length | Short vs full conversation history | decision_consistency, tokens |

**N1** — Temperature: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]. Task: hover at 1 m,
navigate to 3 waypoints. At each temp: N=5 runs. Measure: did drone follow
correct path, how much action variance (same scene → different action?).

**N2** — 5 models, identical real-world task (arm → takeoff → hover → land),
measured on real drone. Metrics: task completion, latency, USD cost per mission.

**N3** — History lengths: 0 (stateless), 5 turns, 10 turns, full mission.
Scenario: drone asked to "return to initial altitude" after several moves.
Does it remember what the initial altitude was?

---

### O — MCP Full Autonomy Validation (2 experiments)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| O1 | mcp_autonomy_mission | Multi-waypoint mission, full autonomy, no human | waypoint_success, safety_stops |
| O2 | mcp_autonomy_recovery | Inject disturbance mid-mission, measure auto-recovery | recovery_success_rate |

**O1** — 5-waypoint indoor mission. LLM plans and executes entirely via MCP tools.
No human approval. Measures: waypoints completed, wall-stop events, total cost.

**O2** — During flight, operator physically tilts drone or blocks camera.
LLM must detect anomaly via telemetry/vision and recover. N=5 disturbances.

---

### P — Human-in-the-Loop MCP Autonomy (2 experiments)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| P1 | hitl_approval_latency | Human approves/rejects each LLM action | approval_time, mission_time |
| P2 | hitl_override_quality | Human overrides LLM; does LLM incorporate feedback? | override_success, adaptation_score |

**P1** — LLM proposes action, human types approve/reject in terminal within 10 s
(auto-approve if timeout). Measures: human decision time, total mission time vs
full-auto (from O1). N=5 full missions.

**P2** — Human types "no, go left instead" mid-sequence. Does LLM correctly
change plan AND remember the override for the rest of the mission?
N=10 overrides across 5 missions.

---

### Q — Chat + Vision + Verbalization (2 experiments)

| ID | Name | Goal | Primary Metric |
|----|------|------|----------------|
| Q1 | vision_verbalization | LLM describes real frame to operator, TTS spoken aloud | description_quality_score (rubric) |
| Q2 | chat_while_flying | Operator asks questions mid-flight, LLM answers without stopping drone | answer_accuracy, flight_safety |

**Q1** — 20 real frames captured during hover. LLM gives spoken description
(TTS via pyttsx3). Rubric: mentions obstacle/clearance, gives distance estimate,
recommends action. Score 0–4. N=5 runs × 20 frames.

**Q2** — Drone in autonomous hover. Operator types: "where is the nearest obstacle?",
"what is my battery?", "describe what you see." LLM answers via TTS AND chat
without interrupting the hover loop. N=5 questions × 5 runs.

---

## Summary Table

| Series | Experiments | Real HW trials | Key claim |
|--------|------------|----------------|-----------|
| J — MCP Server | 2 | ~650 | MCP server is reliable (>95%) and low-latency |
| K — MCP Client | 2 | ~100 | All 5 LLMs successfully control drone via MCP |
| L — Multilanguage | 1 | 125 | Claude correctly interprets 5 languages for drone |
| M — Prompting | 4 | ~160 | ReAct + structured prompt = best accuracy/cost |
| N — LLM Params | 3 | ~175 | temp=0.2 optimal; Sonnet > Haiku for safety tasks |
| O — Full Autonomy | 2 | ~50 | MCP-only autonomy completes missions safely |
| P — HITL | 2 | ~75 | HITL adds safety with acceptable latency cost |
| Q — Chat/Vision | 2 | ~200 | Real-time verbalization quality ≥ 3.5/4 rubric |
| **TOTAL** | **18** | **~1535** | |

---

## Key Paper Claims This Set Supports

1. **MCP generalises drone control** — any LLM can control any drone via the MCP interface
2. **ReAct prompting is safest** for obstacle avoidance on real hardware
3. **Temperature > 0.4 increases crash risk** on real drone (novel safety result)
4. **Multilingual commands work** reliably for Hindi/Tamil/Spanish/French
5. **HITL adds ~40% mission time** but catches LLM errors that full-auto misses
6. **Real-time verbalization** achieves ≥ 3.5/4 rubric score with <2 s TTS latency

---

## Publication Target

These 18 experiments form a standalone paper:
> *"MCP-Drone: Model Context Protocol for Multi-LLM Autonomous UAV Control —
>  Architecture, Prompting, Parameters, and Safety"*

Target: **IEEE RA-L** (hardware novelty) or **IROS 2026** (systems contribution)
Supplement: hardware video + all raw CSVs from results/

---

## File Index

```
MCP experiments/
├── MCP_EXPERIMENT_PLAN.md          ← this file
├── mcp_server.py                   ← MCP protocol server (drone HTTP bridge)
├── mcp_client.py                   ← Multi-LLM MCP client (vision+TTS+chat)
├── exp_J1_server_tool_reliability.py
├── exp_J2_server_latency.py
├── exp_K1_client_multi_llm.py
├── exp_K2_client_vision_pipeline.py
├── exp_L1_multilanguage_commands.py
├── exp_M1_zero_vs_few_shot.py
├── exp_M2_cot_vs_react.py
├── exp_M3_system_prompt_design.py
├── exp_M4_safety_constraint_placement.py
├── exp_N1_temperature_sweep.py
├── exp_N2_model_comparison.py
├── exp_N3_context_length.py
├── exp_O1_full_autonomy_mission.py
├── exp_O2_autonomy_recovery.py
├── exp_P1_hitl_approval_latency.py
├── exp_P2_hitl_override_quality.py
├── exp_Q1_vision_verbalization.py
├── exp_Q2_chat_while_flying.py
└── results/
```
