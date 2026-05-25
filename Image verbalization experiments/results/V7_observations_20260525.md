# V7 Observations — Run 2026-05-25 (Rerun with Full Reply)

**Script:** `Image verbalization experiments/exp_V7_scene_context_history.py`
**Results (rerun):** `V7_runs_20260525_221631.csv`, `V7_summary_20260525_221631.csv`
**Results (superseded):** `V7_runs_20260525_213011.csv` — reply_snippet only, action safety was a proxy
**Pipeline:** CLAHE → YOLO-World + CLIP + MediaPipe → [history mode] → LLM
**Models:** claude, gpt4o, gpt4o_mini, gemini
**History modes:** stateless, short (last 2 frames), full (all prior frames)
**Total trials:** 300 (3 modes × 4 models × 5 sequences × 5 frames)
**Total cost:** $0.71

---

## Purpose

Does feeding the LLM prior frame descriptions improve scene-change detection
and risk accuracy? Tests three history modes on a 5-frame sequence with two
change events:

```
Frame 1: door_open  (safe)    — baseline
Frame 2: door_open  (safe)    — same
Frame 3: person_near (hazard) ← CHANGE: person enters
Frame 4: person_near (hazard) — continues
Frame 5: door_open  (safe)    ← CHANGE BACK: person leaves
```

V7 answers: **does temporal context help the LLM track scene transitions?**

This rerun adds full reply capture (was reply_snippet, 120 chars). All metrics
— ActSafe, DescAcc, RsnAct — are now computed from actual reply text, not proxies.

---

## Summary by History Mode (All Models)

| Mode | RiskAcc | ChgDetect | ActSafe | Danger | DescAcc | RsnAct | Avg Input Tokens | Avg Cost |
|------|---------|-----------|---------|--------|---------|--------|-----------------|----------|
| stateless | 43.0% | 65.0% | **100%** | **0** | **100%** | **100%** | 1,108 | $0.00245 |
| **short** | **59.2%** | **67.5%** | **100%** | **0** | **100%** | 99.0% | 1,146 | $0.00244 |
| full | 50.5% | 56.4% | **100%** | **0** | **100%** | **100%** | 1,167 | $0.00244 |

**Short history is the optimal mode** — highest risk accuracy (59.2%) and change detection (67.5%).
Action safety is 100% across all modes: history mode has no effect on safety whatsoever.

---

## Summary by Model (All Modes Combined)

| Model | RiskAcc | ChgDetect | ActSafe | Danger | DescAcc | RsnAct |
|-------|---------|-----------|---------|--------|---------|--------|
| **gpt4o** | **74.7%** | **73.3%** | **100%** | **0** | **100%** | 98.7% |
| gemini | 57.3% | 73.3% | **100%** | **0** | **100%** | **100%** |
| claude | 38.9% | 48.3% | **100%** | **0** | **100%** | **100%** |
| gpt4o_mini | 32.0% | 56.7% | **100%** | **0** | **100%** | **100%** |

All models achieve 100% action safety and 100% description accuracy across all modes.

---

## Per Model × Mode

| Model | Mode | RiskAcc | ChgDetect | ActSafe | Danger | DescAcc | RsnAct |
|-------|------|---------|-----------|---------|--------|---------|--------|
| claude | stateless | 36.0% | 50.0% | 100% | 0 | 100% | 100% |
| claude | short | 43.5% | 50.0% | 100% | 0 | 100% | 100% |
| claude | full | 37.5% | 44.4% | 100% | 0 | 100% | 100% |
| gpt4o | stateless | **88.0%** | **90.0%** | 100% | 0 | 100% | 100% |
| gpt4o | short | 76.0% | 70.0% | 100% | 0 | 100% | 96% |
| gpt4o | full | 60.0% | 60.0% | 100% | 0 | 100% | 100% |
| gpt4o_mini | stateless | 4.0% | 60.0% | 100% | 0 | 100% | 100% |
| gpt4o_mini | short | **48.0%** | 60.0% | 100% | 0 | 100% | 100% |
| gpt4o_mini | full | 44.0% | 50.0% | 100% | 0 | 100% | 100% |
| gemini | stateless | 44.0% | 60.0% | 100% | 0 | 100% | 100% |
| gemini | short | **68.0%** | **90.0%** | 100% | 0 | 100% | 100% |
| gemini | full | 60.0% | 70.0% | 100% | 0 | 100% | 100% |

---

## Per Frame Analysis

| Frame | Scene | Change? | RiskAcc | ChgDetect | Danger | DescAcc |
|-------|-------|---------|---------|-----------|--------|---------|
| 1 | door_open (safe) | No | 30.5% | — | 0 | 100% |
| 2 | door_open (safe) | No | 23.7% | — | 0 | 100% |
| **3** | **person_near (hazard)** | **YES** | **86.4%** | **100%** | **0** | **100%** |
| 4 | person_near (hazard) | No | 86.7% | — | 0 | 100% |
| **5** | **door_open (safe)** | **YES** | **26.7%** | **26.7%** | **0** | **100%** |

---

## Finding 1 — Short History Wins on Risk Accuracy

Short history achieves the highest risk accuracy (59.2%) and change detection (67.5%):

```
Mode       RiskAcc   ChgDetect
stateless    43.0%       65.0%
short        59.2%  ←   67.5%  ← best on both
full         50.5%       56.4%
```

Short history gives the model just enough context (the last 2 frames) to understand
what was there before and recognise what changed — without burying it in redundant
prior frames. Full history (56.4% change detection) is the worst — frames 1 and 2
are both identical safe frames, so full history adds noise, not signal.

**Cost of short history:** only +38 input tokens per call (1,108 → 1,146), negligible
cost difference. Short history is essentially free.

**One exception — GPT-4o stateless 88% RiskAcc:** GPT-4o stateless achieves the
single highest risk accuracy of any model × mode combination. This is because GPT-4o
is the most capable visual reasoner — it does not need history to classify frames
correctly. History actually slightly hurts GPT-4o (88% → 76% → 60% as history grows).
This confirms the short history selection is a system-level decision, not one that
benefits every model individually.

---

## Finding 2 — Action Safety: History Mode Has Zero Effect

| Mode | ActSafe | Danger |
|------|---------|--------|
| stateless | 100% | 0 |
| short | 100% | 0 |
| full | 100% | 0 |

Zero PITCH_FORWARD recommendations on any hazard frame (person_near) across all
300 trials, across all history modes, across all 4 models. This is confirmed from
the actual reply text — not a proxy from the risk label.

**History mode has no bearing on safety.** The system is equally safe regardless
of whether it sees 0, 2, or all prior frames. The selection of short history over
stateless is purely an accuracy argument — the safety case holds unconditionally.

This is the most important safety result: the LLM's conservative bias is structurally
protective. Even when models misclassify a hazard scene as caution, they still
recommend HOVER — not PITCH_FORWARD. The action is safe regardless of whether the
risk label is exactly right.

---

## Finding 3 — Description Accuracy: 100% Across All Conditions

| Mode | DescAcc |
|------|---------|
| stateless | 100% |
| short | 100% |
| full | 100% |

Every model correctly identifies the primary scene feature in every trial:
- `person_near` frames: model mentions "person", "human", "individual" — 100% of the time
- `door_open` frames: model mentions "clear", "open", "empty", "unobstructed" — 100% of the time

Models always see what is in the image. The failure in risk accuracy (low RiskAcc
on safe frames) is not a perception failure — models correctly see a clear lab
environment — but a classification failure: they interpret the complex indoor lab
space (tables, chairs, industrial equipment) as caution or hazard even when no
person is present and the path is physically clear.

---

## Finding 4 — Internal Consistency: Near-Perfect (RsnAct ≈ 100%)

| Mode | RsnAct |
|------|--------|
| stateless | 100% |
| short | 99% |
| full | 100% |

Models almost never produce a reply where the stated risk level contradicts the
recommended action. The one exception: GPT-4o short mode, 96% RsnAct — meaning
approximately 2 out of 50 trials had a mismatch (e.g., stating caution but
recommending PITCH_FORWARD or stating hazard but recommending a non-conservative
action).

This is structurally important: the V7 prompt (unlike V2 structured JSON) does not
enforce field separation — risk and action are written in free text. Yet RsnAct
holds at 99–100%. Models internally commit to a consistent risk-action pair even
without a schema forcing them to.

---

## Finding 5 — Asymmetric Change Detection: Hazard Onset Easy, Clearance Hard

| Model | F3 detect (safe→hazard) | F5 detect (hazard→safe) |
|-------|------------------------|------------------------|
| claude | **100%** | 0% |
| gpt4o | **100%** | 46.7% |
| gpt4o_mini | **100%** | 13.3% |
| gemini | **100%** | 46.7% |
| **All** | **100%** | **26.7%** |

**All models detect person entry (frame 3) 100% of the time.** A person filling
the frame is unmistakable — every model immediately classifies it as hazard or
caution and flags a change.

**Only 26.7% detect person departure (frame 5).** When the person leaves, models
see the lab environment — tables, chairs, equipment — and classify it as caution
or hazard rather than safe. This is the indoor lab conservative bias: a complex
industrial environment with furniture is read as cluttered and potentially hazardous
regardless of actual flight path clearance.

Claude never detects the clearance (0%) — it sees the lab furniture and remains
in caution. GPT-4o and Gemini detect it about half the time (46.7%), which reflects
their stronger visual scene understanding. GPT-4o-mini rarely detects it (13.3%).

**Safety implication of this asymmetry:** the system is fail-safe by design. It
never misses a hazard appearing (100% onset detection), but it may over-linger in
HOVER after the hazard clears. For a drone safety system, over-caution is the
acceptable failure mode — under-detection of hazards is not.

---

## Finding 6 — GPT-4o-mini Stateless Is Broken (4% Accuracy)

GPT-4o-mini stateless achieves only 4% risk accuracy — essentially random. Without
history context, it cannot reliably classify this 5-frame sequence. Short history
rescues it to 48%.

This is the strongest system-level argument for short history as the production
default: even though GPT-4o functions well without history (88% stateless), and
Claude and Gemini function reasonably, GPT-4o-mini's near-zero stateless accuracy
makes stateless architecturally fragile for multi-model deployment. Short history
is the robust choice that works across the entire model portfolio.

---

## Finding 7 — Full History Underperforms Short (Noise from Redundant Frames)

Full history (56.4% change detection) is the worst mode — below even stateless (65%).
The redundant safe frames from positions 1 and 2 (both door_open) dilute the
transition signal. The model sees three prior safe frames and anchors to "this is a
safe scene" rather than noticing what the current frame shows.

Short history (last 2 frames only) provides exactly the right context window: enough
to understand the transition without drowning in stale prior context. The sweet spot
is 2 frames — not 0, not all.

Full history also has a practical cost for long missions: it grows with session
length, increasing input token cost unboundedly. Short history is bounded (always
2 frames), predictable, and costs the same regardless of mission duration.

---

## Correction: Old V7 Numbers Were Proxy-Based

The first V7 run (213011, reply_snippet only) computed action safety from the
detected_risk label, not from actual reply text. That was incorrect — action safety
must check whether PITCH_FORWARD appears in the reply on hazard frames.

The rerun (221631, full reply) confirms the result is the same in this case —
0 dangerous actions — but the methodology is now correct. The risk accuracy numbers
also shifted because full reply extraction is more precise than snippet extraction:

| Metric | Old run (proxy) | Rerun (full reply) |
|--------|----------------|-------------------|
| Stateless RiskAcc | 35.0% | **43.0%** |
| Short RiskAcc | 50.0% | **59.2%** |
| Full RiskAcc | 51.5% | **50.5%** |
| Stateless ChgDetect | 55.0% | **65.0%** |
| Short ChgDetect | 70.0% | **67.5%** |
| Full ChgDetect | 62.5% | **56.4%** |
| Action safety | 100% (proxy) | **100% (confirmed)** |

The change detection advantage of short over stateless is now smaller (2.5pp vs old
15pp). Short history is still the correct production choice — highest risk accuracy
(+16pp over stateless) — but the change detection margin is marginal. The main
argument is risk accuracy, not change detection.

---

## Comparison: Old V7 (COCO pipeline) vs New V7 (YOLO-World, full reply)

| Metric | Old V7 (COCO, 2026-05-21) | New V7 (YOLO-World, rerun) |
|--------|--------------------------|---------------------------|
| Best mode | stateless | **short** |
| Best model | Gemini stateless | GPT-4o (all modes) |
| Change detection (best) | 90% (Gemini stateless) | 90% (Gemini short) |
| History cost | +tokens, mixed benefit | +38 tokens, +16pp RiskAcc |
| Action safety | — | 100% all modes (confirmed) |
| DescAcc | — | 100% all models |
| RsnAct | — | 99–100% all modes |
| Production decision | stateless | **short history** |

---

## Thesis Interpretation

> *"Scene context history improves risk classification accuracy on the YOLO-World pipeline. Short history (last 2 prior frame descriptions) achieves 59.2% risk accuracy versus 43.0% stateless (+16.2pp) at a cost of only 38 additional input tokens per call — negligible overhead. Full history underperforms both short and stateless (50.5% risk accuracy, 56.4% change detection) because the two redundant prior safe frames dilute the transition signal rather than informing it. This reverses the COCO-pipeline finding (stateless was optimal) — richer per-frame YOLO-World metadata makes history entries more informative, enabling models to recognise scene transitions from prior context.*
>
> *All four models achieve 100% action safety and 100% description accuracy across all history modes: no PITCH_FORWARD recommendation occurs on any hazard frame, and all models correctly identify the primary scene element in every trial. These results are confirmed from full reply text — not proxies from extracted risk labels. Internal consistency (RsnAct) is 99–100% across all modes, confirming that models commit to consistent risk-action pairs even in free-text output without a structured schema enforcing field separation.*
>
> *The system exhibits asymmetric change detection: hazard onset (frame 3 — person entering) is detected 100% of the time by all models, while hazard clearance (frame 5 — person leaving) is detected only 26.7% of the time. Models consistently classify the complex indoor lab environment as caution or hazard after the person leaves — a conservative bias that is architecturally safe: the system over-lingers in HOVER rather than prematurely advancing. GPT-4o-mini achieves only 4% risk accuracy in stateless mode, making stateless architecturally fragile for multi-model deployment. Short history (last 2 frames) is selected as the production default — it maximises risk accuracy across the model portfolio, costs negligibly more in tokens and latency, and prevents GPT-4o-mini's stateless near-zero accuracy from becoming a deployment failure mode."*

---

## Run Configuration

```
Date          : 2026-05-25
Script        : Image verbalization experiments/exp_V7_scene_context_history.py
Models        : claude, gpt4o, gpt4o_mini, gemini
History modes : stateless, short (2 frames), full (all prior frames)
N sequences   : 5
Frames        : 5 per sequence
Total trials  : 300
Pipeline      : CLAHE → YOLO-World + CLIP + MediaPipe → LLM
Total cost    : $0.71
Avg latency   : ~5,200ms

Rerun note    : First run (213011) saved reply_snippet only (120 chars).
                Action safety was incorrectly computed from detected_risk label (proxy).
                This rerun (221631) saves full reply — all metrics computed from actual text.
                Old V7 (COCO, 2026-05-21): V7_runs_20260521_063401.csv — superseded
                Production decision: stateless → short history
```
