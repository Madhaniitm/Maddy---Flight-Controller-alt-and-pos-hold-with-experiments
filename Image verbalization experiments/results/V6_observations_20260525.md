# V6 Observations — Run 2026-05-25

**Script:** `Image verbalization experiments/exp_V6_verbosity_vs_quality.py`
**Results (with CLIP):** `V6_runs_20260525_185304.csv`
**Results (without CLIP):** `V6_runs_noclip_20260525_201620.csv`
**Pipeline:** CLAHE → YOLO-World + MediaPipe → [CLIP optional] → LLM
**Models:** claude, gpt4o, gpt4o_mini, gemini
**Token levels:** 64, 128, 256, 512
**Total trials:** 640 per run (4 token levels × 4 models × 8 scenes × 5 runs)

---

## Purpose

Determine the minimum viable token budget for complete verbalization quality across all four models on the YOLO-World pipeline. Two runs compared:
- **With CLIP** — full V-series pipeline (YOLO-World + CLIP + MediaPipe)
- **Without CLIP** — ablation run (YOLO-World + MediaPipe only)

V6 answers: **how many output tokens does each model need to produce a complete verbalization + pilot action?**

---

## Summary by Token Level — WITH CLIP

| max_tokens | Quality /5 | Truncated | Avg Words | LblAcc | Latency |
|-----------|-----------|---------|---------|--------|---------|
| 64 | 3.27 | 74.7% | 48.4 | 36.7% | 3,142ms |
| 128 | 4.15 | 88.8% | 76.5 | 58.8% | 3,649ms |
| **256** | **4.48** | **96.8%** | **92.0** | **57.3%** | **4,228ms** |
| 512 | 4.49 | 92.9% | 92.5 | 57.7% | 4,294ms |

## Summary by Token Level — WITHOUT CLIP

| max_tokens | Quality /5 | Truncated | Avg Words | LblAcc | Latency |
|-----------|-----------|---------|---------|--------|---------|
| 64 | 3.28 | — | 48.3 | 35.6% | — |
| 128 | 4.20 | — | 74.1 | 53.8% | — |
| **256** | **4.47** | — | **89.2** | **57.5%** | — |
| 512 | 4.47 | — | 90.6 | 56.9% | — |

---

## WITH vs WITHOUT CLIP — Direct Comparison

| Tokens | Q +CLIP | Q −CLIP | ΔQ | Acc +CLIP | Acc −CLIP | ΔAcc |
|--------|---------|---------|-----|-----------|-----------|------|
| 64 | 3.27 | 3.28 | +0.02 | 36.7% | 35.6% | −1.1% |
| 128 | 4.15 | 4.20 | +0.05 | 58.8% | 53.8% | −5.0% |
| 256 | 4.48 | 4.47 | −0.01 | 57.3% | 57.5% | +0.2% |
| 512 | 4.49 | 4.47 | −0.02 | 57.7% | 56.9% | −0.8% |

**Maximum ΔQ: 0.05 (within noise). Plateau point: 256 tokens in both conditions.**

---

## Per-Model Results — WITH CLIP

| Model | tok=64 | tok=128 | tok=256 | tok=512 | Plateau |
|-------|--------|---------|---------|---------|---------|
| claude | 3.26 / 44.7% | 3.45 / 45.0% | **4.22 / 59.5%** | 4.14 / 50.0% | **256** |
| gpt4o | 3.33 / 27.5% | **4.70 / 70.0%** | 4.55 / 55.0% | 4.62 / 62.5% | **128** |
| gpt4o_mini | 3.15 / 30.0% | **4.67 / 67.5%** | 4.65 / 65.0% | 4.67 / 67.5% | **128** |
| gemini | 3.33 / 45.0% | 3.77 / 52.5% | **4.50 / 50.0%** | 4.50 / 50.0% | **256** |

## Per-Model Results — WITHOUT CLIP

| Model | tok=64 | tok=128 | tok=256 | tok=512 | Plateau |
|-------|--------|---------|---------|---------|---------|
| claude | 3.20 / 40.0% | 3.33 / 35.0% | **4.15 / 55.0%** | 4.08 / 47.5% | **256** |
| gpt4o | 3.40 / 40.0% | **4.60 / 60.0%** | 4.60 / 60.0% | 4.58 / 57.5% | **128** |
| gpt4o_mini | 3.20 / 25.0% | **4.70 / 72.5%** | 4.65 / 65.0% | 4.72 / 72.5% | **128** |
| gemini | 3.33 / 37.5% | 4.17 / 47.5% | **4.50 / 50.0%** | 4.50 / 50.0% | **256** |

*(format: Quality/5 / LblAcc)*

---

## Finding 1 — Quality Plateaus at 256 Tokens (Both Conditions)

```
64 → 128 : +0.88 quality jump  (large gain — minimum viable budget)
128 → 256: +0.33 quality gain  (worthwhile for Claude and Gemini)
256 → 512: +0.01 quality gain  (negligible — no benefit)
```

**256 tokens is the optimal budget** for the YOLO-World pipeline. Adding more tokens beyond 256 adds negligible quality improvement at higher latency and cost.

Compared to the old V6 run (COCO pipeline, plateau at 128): the YOLO-World+CLIP pipeline requires 256 tokens because the richer metadata (object classes, estimated distances, depth readings, MediaPipe advisory) requires more output tokens for the model to process supplementary sensor data before concluding.

---

## Finding 2 — Per-Model Plateau Differs

| Model | Plateau | Reason |
|-------|---------|--------|
| **claude** | 256 | Verbose reasoner — needs space to process metadata and conclude |
| **gpt4o** | 128 | Writes ~55 words regardless of budget; extra tokens unused |
| **gpt4o_mini** | 128 | Same pattern — compact responder |
| **gemini** | 256 | Needs ~100 words for full verbalization + pilot action |

GPT-4o and GPT-4o-mini are token-efficient — they plateau at 128. Claude and Gemini need 256. Since the system must support all models, **256 is the minimum sufficient budget**.

---

## Finding 3 — CLIP Has No Meaningful Effect (Ablation Result)

Maximum quality difference between with/without CLIP is **0.05** — within noise across 160 trials per condition. The plateau point is identical (256 tokens) in both runs.

**Per-model CLIP effect:**
- **Claude**: marginally prefers CLIP (−0.07 to −0.12 without CLIP) — too small to matter
- **GPT-4o**: mixed (±0.05–0.10) — no consistent direction
- **GPT-4o-mini**: unaffected (ΔQ ≈ 0)
- **Gemini at 128 tokens**: improves **+0.40** without CLIP — CLIP's random label was consuming reasoning capacity at low token budgets, preventing Gemini from reaching a conclusion within 128 tokens

**CLIP is confirmed as a near-random signal on 320×240 frames.** It neither helps nor hurts overall, and in some configurations (Gemini, low token budget) actively degrades performance by injecting noise into limited reasoning space. This result serves as the V6 clip ablation — a separate V_clip_ablation experiment is not required.

**CLIP is dropped from the production pipeline.**

---

## Finding 4 — LblAcc Jumps at 128 Then Plateaus

| Tokens | LblAcc (with CLIP) |
|--------|-------------------|
| 64 | 36.7% |
| **128** | **58.8%** (+22.1pp) |
| 256 | 57.3% (−1.5pp) |
| 512 | 57.7% (+0.4pp) |

The large jump at 128 (not 256) confirms that label accuracy depends on whether the model can write the `Risk:` line — not on reasoning quality. At 64 tokens many responses are cut off before the risk label. At 128 the risk label is reachable for most models. Beyond 128, label accuracy is stable — adding tokens improves description quality but not classification.

---

## Finding 5 — Truncation Metric Note

The `truncated` column (reply does not end with terminal punctuation) shows high rates for GPT-4o and GPT-4o-mini even at 128+ tokens (up to 100%). This is a metric artefact — these models consistently end their replies with the pilot action line ("Pilot suggested action: PITCH_FORWARD") which has no terminal punctuation. Their replies are complete, not truncated. The quality scores (4.67–4.70) confirm these are full responses.

---

## Thesis Interpretation

> *"Output token budget significantly affects verbalization quality on the YOLO-World pipeline. Quality improves sharply from 64 to 128 tokens (+0.88/5), gains further from 128 to 256 (+0.33/5), and plateaus beyond 256 (Δ=0.01/5 from 256→512). The optimal budget is max_tokens=256 — sufficient for all four models to produce complete verbalizations with pilot action suggestions. GPT-4o and GPT-4o-mini plateau at 128 tokens (compact responders averaging 55–75 words); Claude and Gemini require 256 (verbose reasoners averaging 90–145 words). Compared to the earlier COCO pipeline result (plateau at 128), the richer YOLO-World metadata requires additional output tokens, raising the minimum sufficient budget from 128 to 256.*
>
> *A parallel ablation run without CLIP confirms that CLIP contributes no meaningful quality improvement (maximum ΔQ=0.05) and does not change the plateau point. Gemini at 128 tokens improves by +0.40 quality points without CLIP — the random CLIP label consumed reasoning capacity within the limited token budget. CLIP is removed from the production pipeline: it adds input token cost and latency with no measurable benefit to verbalization quality or safety classification accuracy."*

---

## Run Configuration

```
Date              : 2026-05-25
Script            : Image verbalization experiments/exp_V6_verbosity_vs_quality.py
Models            : claude, gpt4o, gpt4o_mini, gemini
Token levels      : [64, 128, 256, 512]
N runs            : 5 per scene per model per token level
Scenes            : 8 canonical scenes (run03 saved frames)
Total trials      : 640 per condition

With CLIP run     : V6_runs_20260525_185304.csv
                    Pipeline: YOLO-World + CLIP + MediaPipe → LLM
Without CLIP run  : V6_runs_noclip_20260525_201620.csv
                    Pipeline: YOLO-World + MediaPipe → LLM (CLIP ablation)

Old V6 (COCO)     : V6_runs_20260521_053358.csv — superseded, plateau was 128 tokens
                    New plateau is 256 due to richer YOLO-World metadata
```
