# OBSERVATIONS — PAPER 2: Prompt-Configurable LLM-Vision Payload for UAVs
# G-Series + H-Series + V-Series
# Created: 2026-05-02 | Target: IEEE Robotics and Automation Letters (RA-L)

---

## PAPER FRAMING

### Core Contribution

> **Same hardware. Change the prompt. Change the domain. Change the language. Zero hardware modification.**

The system is a UAV-mounted payload (ESP32-S3 Sense camera) that combines a fast YOLO detection tier with a slow LLM reasoning tier. The YOLO tier is domain-agnostic — an obstacle is an obstacle regardless of mission type. The LLM tier is domain-configurable via the system prompt alone. By swapping the prompt, the same hardware transitions between:

- **Soldier mode**: detect threats, identify suspicious activity, suggest tactical actions
- **Farmer mode**: detect crop health, identify pests, suggest field interventions
- **Search & rescue**: detect survivors, identify hazards, suggest entry routes
- **Inspection**: detect damage, identify structural issues, suggest maintenance
- **Any domain**: prompt defines the domain, hardware never changes

Multilingual support (V3) extends this further: a Tamil-speaking farmer receives Tamil suggestions, a French soldier receives French tactical output — same flight, same hardware, different prompt language.

### Architecture

```
ESP32-S3 Sense (on drone)
        │
        ▼ JPEG frames
   ┌────────────┐
   │  YOLO Tier │  ← Fast (8ms), domain-agnostic, always-on safety
   │  (G1, G5, V5) │    obstacle/hazard detection trigger
   └─────┬──────┘
         │ event flag (hazard detected)
         ▼
   ┌────────────┐
   │  LLM Tier  │  ← Slow (600ms), domain-specific via prompt
   │ (V1–V4, G2)│    scene description + suggested next action
   └─────┬──────┘
         │
         ▼
   ┌──────────────────────┐
   │  Pilot Display/Audio │  ← Human-on-the-loop
   │  (H1, H4, V3)        │    pilot manually executes suggestion
   └──────────────────────┘
         │
         ▼ (Future Work — Paper 3)
   Flight Controller (FC)  ← Autonomous command execution
```

**Key architectural insight:** YOLO/LLM split is not merely a latency optimization — it is a **domain abstraction boundary**. YOLO handles universal low-level safety; LLM handles domain-specific high-level reasoning. This separation means domain switching costs zero hardware effort.

### Two Operating Modes

**Mode 1 — Real-time pilot assistance (during flight):**
Each frame → YOLO check → if event triggered, LLM API call → scene description + suggested action → pilot receives in real-time

**Mode 2 — Post-flight mission summary (after landing):**
Accumulated frame history across entire flight → single LLM call with full context → mission summary + overall safety verdict (e.g., "Room assessed as partially safe: 3 of 4 zones clear, zone 2 blocked by debris")

### Existing Flight Evidence

A qualitative flight video exists demonstrating the complete pipeline:
- ESP32-S3 Sense mounted on drone, drone airborne
- VLM describing scene in real-time during flight
- VLM suggesting next action to drone pilot
- End-to-end pipeline validated on real hardware

This video serves as Fig 1 / supplementary multimedia for RA-L submission.
**Quantification needed:** extract end-to-end latency (frame capture → pilot receives suggestion) from video logs.

### Future Work (Paper 3 — D+E+F+I series)

The suggested action currently goes to the human pilot. Paper 3 converts the suggested action into an FC command, closing the loop and making the system fully autonomous. This positions Paper 2 cleanly as the perception+assistance layer and Paper 3 as the autonomy layer.

---

## NOVEL CONTRIBUTIONS (for RA-L)

1. **First prompt-configurable domain-agnostic UAV payload** — same hardware, any domain, zero code change
2. **Multilingual operator interface** — V3 validates non-English instructions with accuracy and language-match metrics
3. **YOLO+LLM hierarchical architecture** with quantified timescale separation (G1, G4)
4. **Event-triggered LLM activation** reducing API cost 6× with zero safety regression (G2)
5. **Real-time + post-flight dual operating modes** (V7 provides the context-history backbone for post-flight mode)
6. **Real flight validation** with video evidence (qualitative, quantification in progress)
7. **Multi-model comparison** on real ESP32-S3 hardware frames — which VLM is best for each domain? (V1, V4)

---

## EXPERIMENT-BY-EXPERIMENT ROLE IN PAPER

---

### G-SERIES: Vision Tier Architecture

#### EXP-G1: YOLO vs Claude Emergency Stop Latency
**Role in paper:** Motivates the two-tier architecture. Quantifies the 75× latency gap (YOLO ~8ms vs Claude ~600ms). Justifies why LLM cannot be on the critical safety path.
**Key metric:** Latency distribution (ms), accuracy (%), gap ratio
**Expected result:** YOLO ~8ms, Claude ~600ms, both 100% accuracy on test scenario
**Paper position:** Architecture motivation, Section II

#### EXP-G2: Event-Triggered vs Periodic Claude Activation
**Role in paper:** Proves that gating the LLM behind YOLO events reduces cost 6× with zero safety regression. This is the efficiency contribution of the YOLO+LLM hierarchy.
**Key metric:** API calls per mission (event: ~3, periodic: ~18), cost ratio, hazard detection rate
**Expected result:** 6× cost reduction, 100% hazard detection maintained
**Paper position:** Architecture evaluation, core result

#### EXP-G3: Monocular Depth Estimation Accuracy
**Role in paper:** Validates depth estimation for obstacle proximity — feeds into D-series (Paper 3). Supporting experiment, may go to supplementary if page-limited.
**Key metric:** MAE (m), R² per method (heuristic, MiDaS, Depth Anything V2)
**Expected result:** Depth Anything V2 best (~0.05m MAE, R²~0.92)
**Paper position:** Supporting / supplementary

#### EXP-G4: Three-Tier Timescale Validation
**Role in paper:** Quantifies the tier separation ratios (PID→YOLO ~1300×, YOLO→LLM ~40×). Confirms clean timescale hierarchy with measured jitter.
**Key metric:** Mean period (ms), jitter CV, tier ratio
**Paper position:** Architecture validation, Table in Section II

#### EXP-G5: Real Vision Pipeline (ESP32-S3 + YOLO + Claude)
**Role in paper:** End-to-end latency of the full pipeline on commodity hardware. Laptop: ~350ms, ESP32: ~400ms total.
**Key metric:** Total latency (ms) = capture + YOLO + Claude, per mode
**Paper position:** System implementation, hardware validation

---

### H-SERIES: Safety, Security & Human Interface

#### EXP-H1: Runtime Mode Switch — Full-Auto ↔ Human-in-the-Loop
**Role in paper:** Validates that operator can insert/remove supervision mid-mission without restart. Mode-switch latency ~3ms (instantaneous). Supports the "human-on-the-loop" framing.
**Key metric:** Switch latency (ms), auto success rate, HITL compliance rate
**Paper position:** Operating modes section

#### EXP-H2: Face Recognition Operator Authentication
**Role in paper:** Before pilot receives suggestions, system authenticates them. TAR ~98%, FAR ~5%, latency ~6ms. Domain-security feature — a soldier mode system should not accept an unauthenticated operator.
**Key metric:** True Accept Rate, False Positive Rate, auth latency (ms)
**Paper position:** Security layer, one paragraph

#### EXP-H3: Blockchain Audit Trail
**Role in paper:** Every LLM decision logged in tamper-evident chain. For search-and-rescue or military domains, post-mission accountability is legally required. Chain validity 100%, tamper detection 100%, latency ~0.8µs.
**Key metric:** Chain validity %, tamper detection rate, hash latency
**Paper position:** Accountability layer, one paragraph. Strongest justification in high-stakes domains (S&R, military)

#### EXP-H4: Decision Verbalization + TTS
**Role in paper:** The LLM output needs to reach the pilot as speech, not text, during flight (pilot is watching the drone). Validates verbalization quality and TTS latency. Quality score 3.2/4, pipeline ~520ms.
**Key metric:** Quality score (0–4), word count, verbalization + TTS latency
**Paper position:** Pilot interface section. Directly supports the flight video evidence.

---

### V-SERIES: VLM Characterization on ESP32-S3

**All V-series experiments use the same ESP32-S3 Sense camera. Bench-tested (drone stationary or hand-held) for controlled characterization. Flight video provides real-world deployment evidence.**

#### EXP-V1: Multi-Model Comparison (Claude, GPT-4o, Gemini, LLaVA)
**Role in paper:** Establishes which VLM is best for drone safety classification. 200 trials, 10 scenes, 4 models. This is the primary model selection experiment.
**Key metric:** Classification accuracy (Wilson CI), latency, cost per model
**Expected result:** Claude highest accuracy, LLaVA lowest; latency/cost tradeoff for deployment
**Paper position:** Core experiment, Section III

#### EXP-V2: Prompt Technique Comparison (Zero-shot, Few-shot, CoT, Structured, ReAct)
**Role in paper:** Which prompting strategy gives best scene descriptions? Directly informs the domain-configurable prompt design — if CoT prompts work best, all domain prompts should use CoT structure.
**Key metric:** Quality score (0–4), accuracy, tokens, cost per technique
**Paper position:** Prompt design section. Links to domain-configurability contribution.

#### EXP-V3: Multilingual Input (English, Hindi, Tamil, Spanish, French)
**Role in paper:** CENTRAL to the domain-configurability contribution. Proves same hardware serves non-English operators with full accuracy and language consistency. Tamil farmer gets Tamil suggestions.
**Key metric:** Answer relevance, language match rate, classification accuracy per language
**Expected result:** Language match ~95%+, accuracy comparable to English across all languages
**Paper position:** Core contribution experiment. Must-include, not supplementary.

#### EXP-V4: Model × Prompt Interaction (3 models × 3 prompts, full factorial)
**Role in paper:** Does CoT help all models equally, or only weaker ones? Interaction analysis. Informs which model+prompt combination is optimal per deployment (e.g., Gemini at lower cost with structured prompts may match Claude zero-shot).
**Key metric:** Quality per cell, interaction delta (Δ CoT − zero-shot per model)
**Paper position:** Model selection guidance for different deployment budgets

#### EXP-V5: YOLO Confidence Threshold Sweep
**Role in paper:** Optimises the YOLO tier. What threshold minimises false alarms while catching all hazards? Directly determines the G2 event-trigger sensitivity.
**Key metric:** Precision, Recall, F1, false alarm rate per threshold
**Expected result:** F1-optimal threshold ~0.40–0.50
**Paper position:** YOLO tier tuning, one figure (PR curve)

#### EXP-V6: Verbosity vs Quality (max_tokens 64–512)
**Role in paper:** Determines optimal token budget for real-time pilot suggestions. Too short = incomplete description. Too long = high latency. Identifies sweet spot.
**Key metric:** Quality score, truncation rate, efficiency (quality/USD) per token budget
**Expected result:** Quality plateaus at 256 tokens; efficiency peaks at 256
**Paper position:** Deployment configuration, one paragraph

#### EXP-V7: Scene Context History (Stateless vs Short vs Full context)
**Role in paper:** BACKBONE of Mode 2 (post-flight summary). Does feeding the LLM accumulated frame history improve its safety verdict? If full-context mode detects scene changes at 90% vs stateless 60%, this validates the post-flight summary capability.
**Key metric:** Change detection rate (frames 3 and 5), risk accuracy per history mode
**Expected result:** Full context > short > stateless for change detection
**Paper position:** Post-flight mode validation, key result for Mode 2

#### EXP-V8: Temperature Sweep (0.0 – 1.0)
**Role in paper:** Determines optimal temperature for deployment. Low temp = high consistency, low creativity. High temp = higher variance, occasional hallucination. Label flip rate metric is novel.
**Key metric:** Accuracy, consistency std, label flip rate per temperature
**Expected result:** T=0.2 optimal (maximises accuracy − flip_rate)
**Paper position:** Deployment configuration, supporting figure

#### EXP-V9: Model × Temperature × Max-Tokens (3×3×3 factorial, 810 trials)
**Role in paper:** Complete parameter sensitivity map. Which parameters matter most — model choice, temperature, or token budget? Guides deployment decisions across cost/accuracy tradeoffs.
**Key metric:** Quality and accuracy per cell, interaction deltas
**Paper position:** Comprehensive parameter guide, can go to supplementary if page-limited

---

## EXPERIMENT PRIORITY FOR RUNNING

Run in this order — highest RA-L impact first:

| Priority | Experiment | Why first |
|---|---|---|
| 1 | V1 | Primary model comparison — everything references this |
| 2 | V3 | Core multilingual contribution — must have numbers |
| 3 | G1 | YOLO vs LLM latency — motivates architecture |
| 4 | G2 | Event-triggered efficiency — 6× cost result |
| 5 | V7 | Post-flight mode backbone — change detection result |
| 6 | V5 | YOLO threshold — sets G2 trigger sensitivity |
| 7 | V2 | Prompt technique — informs domain prompt design |
| 8 | H1 | Mode switching — operating mode validation |
| 9 | H4 | TTS verbalization — pilot interface |
| 10 | G4 | Timescale validation — architecture table |
| 11 | V4 | Model×prompt interaction — deployment guidance |
| 12 | H2 | Face auth — security layer |
| 13 | V8 | Temperature sweep — configuration |
| 14 | G3 | Depth estimation — supplementary |
| 15 | G5 | Full pipeline latency — hardware section |
| 16 | H3 | Blockchain audit — one paragraph |
| 17 | V6 | Verbosity tradeoff — configuration |
| 18 | V9 | Full factorial — supplementary |

---

## KEY FIGURES PLANNED FOR PAPER

| Fig | Content | Source |
|---|---|---|
| 1 | System architecture diagram (YOLO→LLM→Pilot) + flight photo | Diagram + video frame |
| 2 | YOLO vs LLM latency comparison (G1) + timescale hierarchy (G4) | G1, G4 |
| 3 | V1: Multi-model accuracy/latency/cost radar chart | V1 |
| 4 | V3: Multilingual accuracy and language-match bar chart | V3 |
| 5 | G2: Event-triggered vs periodic API calls and cost | G2 |
| 6 | V7: Context history vs scene change detection rate | V7 |
| 7 | Domain-switch demo: same frame, soldier prompt vs farmer prompt outputs | Qualitative / video |
| 8 | V4: Model × prompt interaction heatmap | V4 |

---

## SCORING METRICS (consistent across V-series)

- **Classification accuracy**: Wilson 95% CI (binary correct/incorrect per trial)
- **Quality score**: 0–4 rubric (scene content +1, proximity +1, risk correct +1, length 10–100 words +1); Bootstrap 2000 replicates
- **Latency**: Bootstrap CI on wall-clock ms
- **Language match** (V3): heuristic keyword detection per language; Wilson CI
- **Change detection** (V7): Wilson CI on frames 3 and 5 per history mode
- **Label flip rate** (V8): Wilson CI on consecutive run label changes

---

## RELATED WORK TO CITE

- Yao et al. 2022 (ReAct) — reasoning + acting loop
- Ahn et al. 2022 (SayCan) — grounded affordance selection
- Vemprala et al. 2023 (ChatGPT for Robotics) — tool API design
- Huang et al. 2022 (Inner Monologue) — observation-driven replanning
- White et al. 2023 (Prompt Pattern Catalog) — prompt engineering for robotics

---

## RA-L SUBMISSION NOTES

- **Page limit**: 8 pages including references. V9, G3, H3 go to supplementary if needed.
- **Multimedia attachment**: Submit flight video as RA-L multimedia supplement — drone describing scene + suggesting action + post-flight summary in one continuous clip
- **Data availability**: "Results available from corresponding author on request. ESP32-S3 capture and VLM pipeline scripts available on request. Drone firmware is proprietary."
- **Hardware statement**: "All experiments conducted on a custom ~50g quadrotor with proprietary WiFi-based firmware. The ESP32-S3 Sense payload interfaces via the LLM API only — no FC connection in the current work."
- **Domain-switch demo**: Include Figure showing identical frame processed under soldier prompt vs farmer prompt with different outputs — this is the single most impactful visual for the contribution claim.
