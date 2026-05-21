# V7 Observations — Scene Context History Effect

**Experiment**: EXP-V7  
**Date run**: 2026-05-21  
**Result files**: V7_runs_20260521_063401.csv, V7_summary_20260521_063401.csv  
**Trials**: 300 (3 modes × 5 sequences × 5 frames × 4 models)  
**Claude rerun**: exp_V7_claude_rerun.py — Claude rows replaced with structured prompt fix (2026-05-21)

---

## Setup Recap

Sequence: door_open (safe) → door_open (safe) → person_near (hazard) → person_near (hazard) → door_open (safe)  
Change events: frame 3 (safe→hazard) and frame 5 (hazard→safe)  
History modes: stateless (no prior context), short (last 2 frames), full (all prior frames)  
Frames: run02–run05 real hardware captures (run01 excluded as buggy; run02 used twice)

---

## Risk Accuracy by Mode × Model

| Mode       | Claude | GPT-4o | GPT-4o-mini | Gemini |
|------------|--------|--------|-------------|--------|
| stateless  | 0%     | 56%    | 32%         | **72%** |
| short      | 40%    | 60%    | 32%         | 60%    |
| full       | 40%    | 72%    | 24%         | 68%    |

**Verdict**: No consistent improvement from adding history across all models. Gemini stateless achieves the highest risk accuracy (72%) with no history at all.

---

## Change Detection by Mode × Model

Change detection = model correctly flags scene transition at frame 3 (safe→hazard) and frame 5 (hazard→safe).

| Mode       | Claude | GPT-4o | GPT-4o-mini | Gemini |
|------------|--------|--------|-------------|--------|
| stateless  | 50%    | 80%    | 70%         | **90%** |
| short      | 50%    | 90%    | 50%         | 60%    |
| full       | 50%    | 90%    | 50%         | 70%    |

**Verdict**: Gemini stateless achieves best change detection (90%) without any history. Adding history to Gemini degrades it (90% → 60–70%). GPT-4o improves marginally (80% → 90%) with history. Claude stays flat at 50% across all modes.

---

## Token Cost Growth

| Mode       | Claude | GPT-4o | GPT-4o-mini | Gemini |
|------------|--------|--------|-------------|--------|
| stateless  | 258    | 168    | 2935        | 356    |
| short      | 296    | 204    | 2970        | 393    |
| full       | 310    | 219    | 2984        | 407    |

Input tokens grow by ~15–50 per history mode. Over a flight, full history accumulates unboundedly. No accuracy gain justifies this cost.

---

## Per-Model Analysis

### Gemini
- Best overall: stateless 72% risk acc, 90% change detection
- History HURTS: short mode drops risk acc to 60% and change detection to 60%
- Consistently produces parseable output across all modes
- Conclusion: stateless is best — history is noise, not signal for Gemini

### GPT-4o
- Stateless 56% → improves to 72% with full history
- Change detection 80% → 90% with short/full history
- door_open scene still classified as "hazard" (known failure from V1/V2/failure_analysis)
- Improvement with history likely reflects anchoring to prior correct calls
- CIs overlap — improvement not statistically conclusive

### Claude (rerun with structured prompt fix)
Claude's original V7 rows used an unstructured prompt ("Classify as: safe | caution | hazard") which caused verbose markdown replies to score as "unknown" — inflating apparent accuracy via parser accidents. Rerun with structured "Risk: <safe|caution|hazard>" prompt reveals Claude's true behaviour.

**Stateless: 0% risk accuracy**
- door_open (expected safe) → Claude says **caution** every run — never says safe ❌
- person_near (expected hazard) → Claude says **caution** every run — never says hazard ❌
- Claude defaults to caution for every indoor scene regardless of content
- Change detection 50%: frame 3 (safe→hazard) partially credited because caution counts as detecting change; frame 5 (hazard→safe) 0% because caution ≠ safe

**Short/Full: 40% risk accuracy**
- door_open → still caution (or hazard) every time — never safe ❌
- person_near → **hazard** every time ✓ — 10/10 correct across both modes
- With history, Claude sees prior caution frames and escalates to hazard when person_near arrives — the escalation pattern triggers a correct response
- Change detection stays at 50% because frame 5 (back to door_open/safe) still returns caution not safe

**Root cause — indoor visual bias, not a prompt failure:**  
Claude sees metal shelving, glass partitions, industrial environment → always says caution. This is the same Type B failure documented in failure_analysis_observations.md (door_open scene: Claude over-weights visible structures as collision risk). History helps Claude correctly escalate to hazard for person_near (0% → 100% for those frames) but does not fix the door_open safe-scene bias.

Claude's V1/V2/V6/V8 results remain valid — those experiments used structured prompts and diverse scenes. The 0% stateless score here reflects Claude's persistent indoor caution bias on this specific two-scene sequence, not a general capability failure.

### GPT-4o-mini
- 2935–2984 input tokens across all modes (abnormally high — system prompt overhead in API path)
- Risk accuracy flat at 32% (stateless=short) then drops to 24% (full) — history hurts
- Change detection drops: 70% → 50% → 50% with more history

---

## Key Findings

1. **Stateless pipeline is sufficient**: The best result (Gemini stateless: 72% risk acc, 90% change detection) is achieved without any context history. History adds cost without reliable benefit across all four models.

2. **History helps Claude but not enough**: Claude improves from 0% (stateless) to 40% (short/full) by using prior frame context to escalate caution → hazard for person_near. But it never correctly classifies door_open as safe — the 40% ceiling is entirely from person_near frames. History is a partial fix for one model, not a general solution.

3. **History degrades Gemini**: Gemini drops 12pp on risk accuracy and 30pp on change detection when switching from stateless to short history. Prior frame context anchors the model to previous classifications, reducing responsiveness to new visual evidence.

4. **Change events are visually obvious**: The person_near (hazard) scene has a person at 0.31m — detectable by all models without context. Stateless Gemini detects 90% of change events. History is not required for scene-change detection.

5. **Cost grows unboundedly with full history**: At 0.1Hz over a 5-minute flight, full history accumulates 30 frames of context → ~300–600 extra tokens → 10–30% cost increase with no accuracy gain across the best-performing models.

6. **Claude's indoor bias is persistent**: Even with history and a fixed structured prompt, Claude never classifies the door_open (indoor safe) scene correctly. This is a model-prior issue consistent with the failure_analysis Type B finding — not a prompt or scoring issue.

---

## Decision / Thesis Justification

**Design the LLM copilot layer as stateless: each frame is evaluated independently.**

Rationale:
- Temporal continuity is already handled by YOLO at 30fps — the LLM at 0.1–1Hz does not need to track history
- V7 shows stateless achieves the best risk accuracy (72%, Gemini) and best change detection (90%, Gemini)
- Context history adds token cost and latency with no consistent accuracy improvement
- History helps Claude specifically (0% → 40%) but only because it enables escalation for one scene type — not because it improves the pipeline generally
- Full history is unbounded — impractical for long flights without truncation, which introduces its own errors
- Simpler architecture is more robust: each LLM call is independent, no state to maintain or corrupt

This is consistent with the overall system design: YOLO provides the real-time stream, the LLM provides periodic high-level semantic judgment on a fresh frame — not a running narrative.
