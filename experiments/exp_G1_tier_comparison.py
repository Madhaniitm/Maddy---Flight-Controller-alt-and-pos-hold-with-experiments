"""
EXP-G1 (Revised): Four-Tier Architecture Contribution Analysis
==============================================================
Goal:
    Compare the contribution of each tier to safety performance by running
    four conditions on the same saved real hardware frames (run03):

        Condition 1 — tier1_5_only
            MediaPipe EfficientDet-Lite0 → rule-based risk
            No API calls. Fast local detection only (~14ms).
            Tests what Tier 1.5 alone can achieve.

        Condition 2 — tier2_only
            YOLO-World + YOLOv11n COCO + DA v2 → rule-based risk
            No API calls. Full sensor stack, no cognitive reasoning (~275ms).
            Tests what Tier 2 alone can achieve.

        Condition 3 — tier3_only
            LLM receives camera image only (no sensor metadata).
            Tests LLM vision capability without any sensor support.
            Uses LLM_ONLY_PROMPT.

        Condition 4 — tier1_5_tier2_tier3
            Full pipeline: MediaPipe + YOLO + DA v2 metadata → LLM.
            Production condition. Tests the complete architecture.

    Rule-based action mapping (conditions 1 & 2):
        hazard  → HOVER   (stop, do not advance)
        caution → HOVER   (conservative: stop if uncertain)
        safe    → PITCH_FORWARD

    Action safety metric (primary):
        A recommendation is dangerous only if truth=hazard AND action=PITCH_FORWARD.
        Over-caution (HOVER for safe/caution scenes) is acceptable.

    Models (LLM conditions 3 & 4): claude, gpt4o, gpt4o_mini, gemini
    Scenes: all 8 canonical scenes (run03 saved frames)
    N runs: 5

    Total LLM calls: 2 conditions × 4 models × 8 scenes × 5 runs = 320 (~30 min)

Output: G1_runs_<ts>.csv
        G1_summary_<ts>.csv
"""

import sys, pathlib, datetime, time
import numpy as np
import cv2

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
EXP_DIR   = pathlib.Path(__file__).parent
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (          # noqa: E402
    get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR, SCENES
)
from enhanced_yolo_pipeline import (       # noqa: E402
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer, enhanced_rule_risk,
    COMBINED_PROMPT_TEMPLATE, LLM_ONLY_PROMPT
)
from robust_local_detector import (       # noqa: E402
    load_mediapipe_detector, detect_hazard
)

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

# ── Rule-based action: map risk level → pilot action (no LLM) ─────────────────
RULE_ACTION = {
    "hazard":  "HOVER",
    "caution": "HOVER",
    "safe":    "PITCH_FORWARD",
}


def is_action_dangerous(truth: str, action: str) -> bool:
    """Returns True only if the action is genuinely dangerous: flying forward into hazard."""
    return truth == "hazard" and action == "PITCH_FORWARD"


def is_action_safe(truth: str, action: str) -> bool:
    return not is_action_dangerous(truth, action)


def make_rule_row(condition, model, scene, run,
                  tier1_5_ms, tier2_ms, risk, load_ms=0.0):
    """Build a result row for rule-based conditions (no LLM)."""
    action = RULE_ACTION.get(risk, "HOVER")
    truth  = scene["truth"]
    return {
        "condition":    condition,
        "model":        model,
        "scene_label":  scene["label"],
        "truth":        truth,
        "run":          run,
        "load_ms":      load_ms,
        "tier1_5_ms":   tier1_5_ms,
        "tier2_ms":     tier2_ms,
        "llm_ms":       0,
        "total_ms":     round(load_ms + tier1_5_ms + tier2_ms, 1),
        "detected_risk": risk,
        "risk_correct":  int(risk == truth),
        "suggested_action": action,
        "action_safe":   int(is_action_safe(truth, action)),
        "action_dangerous": int(is_action_dangerous(truth, action)),
        "quality_score": 0,   # rubric not applicable for rule-based
        "input_tokens":  0,
        "output_tokens": 0,
        "cost_usd":      0.0,
        "reply":         f"[rule-based] risk={risk} → {action}",
        "error":         "",
    }


def main():
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_calls = 2 * len(MODELS) * len(SCENES) * N_RUNS   # conditions 3 + 4
    print("=" * 70)
    print("EXP-G1: Four-Tier Architecture Contribution Analysis")
    print("  Condition 1 — tier1_5_only     (rule-based, no API)")
    print("  Condition 2 — tier2_only        (rule-based, no API)")
    print("  Condition 3 — tier3_only        (LLM, no metadata)")
    print("  Condition 4 — tier1_5_tier2_tier3 (full pipeline)")
    print(f"Scenes={len(SCENES)}  N={N_RUNS}  LLM calls={llm_calls}")
    print(f"Estimated time: ~{llm_calls * 5 // 60} min")
    print("=" * 70)

    # ── Load models once ──────────────────────────────────────────────────────
    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5)…")
    mp_detector, mp_type = load_mediapipe_detector()

    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()

    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()

    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()

    all_rows = []

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITIONS 1 & 2: Rule-based (no API)
    # Run these once per scene per run — shared load across both conditions
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("CONDITIONS 1 & 2: Rule-based (no LLM API)")
    print("─" * 70)

    for scene in SCENES:
        label = scene["label"]
        truth = scene["truth"]
        print(f"\n── Scene: {label}  (truth={truth}) ──")

        for run in range(1, N_RUNS + 1):
            # Load frame
            t0      = time.perf_counter()
            jpeg    = get_saved_frame(label)
            load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            img_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

            # ── Condition 1: Tier 1.5 only ────────────────────────────────────
            t15        = time.perf_counter()
            local_res  = detect_hazard(img_bgr, depth_map=None,
                                       mp_detector=mp_detector, mp_type=mp_type)
            tier1_5_ms = round((time.perf_counter() - t15) * 1000.0, 2)
            risk_t15   = local_res["risk"]

            row_t15 = make_rule_row("tier1_5_only", "rule_based", scene, run,
                                    tier1_5_ms, 0.0, risk_t15, load_ms)
            all_rows.append(row_t15)
            status_t15 = "✓" if row_t15["risk_correct"] else "✗"
            print(f"   run={run}  tier1_5_only   risk={risk_t15:8s}  "
                  f"→ {row_t15['suggested_action']:14s}  "
                  f"t1.5={tier1_5_ms:.0f}ms  {status_t15}")

            # ── Condition 2: Tier 2 only ──────────────────────────────────────
            t2     = time.perf_counter()
            tier2  = enhanced_yolo_infer(yolo_model, yolo_type,
                                         None, None, None, jpeg,
                                         coco_model=coco_model,
                                         depth_pipe=depth_pipe)
            tier2_ms = round((time.perf_counter() - t2) * 1000.0, 2)
            risk_t2  = enhanced_rule_risk(tier2["yolo_meta"], clip_risk="")

            row_t2 = make_rule_row("tier2_only", "rule_based", scene, run,
                                   0.0, tier2_ms, risk_t2, load_ms)
            all_rows.append(row_t2)
            status_t2 = "✓" if row_t2["risk_correct"] else "✗"
            print(f"   run={run}  tier2_only      risk={risk_t2:8s}  "
                  f"→ {row_t2['suggested_action']:14s}  "
                  f"t2={tier2_ms:.0f}ms  {status_t2}")

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITIONS 3 & 4: LLM-based
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("CONDITIONS 3 & 4: LLM-based")
    print("─" * 70)

    for scene in SCENES:
        label = scene["label"]
        truth = scene["truth"]
        print(f"\n── Scene: {label}  (truth={truth}) ──")

        for run in range(1, N_RUNS + 1):
            # Load frame
            t0      = time.perf_counter()
            jpeg    = get_saved_frame(label)
            load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            img_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

            # ── Shared Tier 1.5 pass (used by condition 4 metadata) ──────────
            t15       = time.perf_counter()
            local_res = detect_hazard(img_bgr, depth_map=None,
                                      mp_detector=mp_detector, mp_type=mp_type)
            tier1_5_ms = round((time.perf_counter() - t15) * 1000.0, 2)

            # ── Shared Tier 2 pass (used by condition 4 metadata) ────────────
            t2    = time.perf_counter()
            tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                        None, None, None, jpeg,
                                        coco_model=coco_model,
                                        depth_pipe=depth_pipe)
            tier2_ms = round((time.perf_counter() - t2) * 1000.0, 2)

            # Combined metadata string (same as G5)
            yolo_meta_full = (
                f"{tier2['yolo_meta']}\n"
                f"  Local detector (Tier 1.5 — MediaPipe EfficientDet-Lite0, advisory — image overrides): "
                f"{local_res['metadata']}"
            )

            # Prompts for condition 3 (no metadata) and condition 4 (full metadata)
            prompt_t3 = LLM_ONLY_PROMPT                           # no metadata
            prompt_t4 = COMBINED_PROMPT_TEMPLATE.format(           # full metadata
                yolo_meta=yolo_meta_full
            )

            for model in MODELS:
                # ── Condition 3: Tier 3 only (LLM, no metadata) ──────────────
                res3    = call_vision_llm(jpeg, prompt_t3, model=model,
                                          max_tokens=300, temperature=0.0)
                sc3     = score_verbalization(res3["reply"], truth)
                action3 = sc3["detected_action"] or "HOVER"
                risk3   = sc3["detected_risk"] or ""

                all_rows.append({
                    "condition":    "tier3_only",
                    "model":        model,
                    "scene_label":  label,
                    "truth":        truth,
                    "run":          run,
                    "load_ms":      load_ms,
                    "tier1_5_ms":   0,
                    "tier2_ms":     0,
                    "llm_ms":       res3["latency_ms"],
                    "total_ms":     round(load_ms + res3["latency_ms"], 1),
                    "detected_risk": risk3,
                    "risk_correct":  sc3["s3_risk"],
                    "suggested_action": action3,
                    "action_safe":   int(is_action_safe(truth, action3)),
                    "action_dangerous": int(is_action_dangerous(truth, action3)),
                    "quality_score": sc3["quality_score"],
                    "input_tokens":  res3["input_tokens"],
                    "output_tokens": res3["output_tokens"],
                    "cost_usd":      res3["cost_usd"],
                    "reply":         res3["reply"],
                    "error":         res3["error"][:80] if res3["error"] else "",
                })

                # ── Condition 4: Full pipeline (Tier 1.5 + 2 + 3) ────────────
                res4    = call_vision_llm(jpeg, prompt_t4, model=model,
                                          max_tokens=300, temperature=0.0)
                sc4     = score_verbalization(res4["reply"], truth)
                action4 = sc4["detected_action"] or "HOVER"
                risk4   = sc4["detected_risk"] or ""

                all_rows.append({
                    "condition":    "tier1_5_tier2_tier3",
                    "model":        model,
                    "scene_label":  label,
                    "truth":        truth,
                    "run":          run,
                    "load_ms":      load_ms,
                    "tier1_5_ms":   tier1_5_ms,
                    "tier2_ms":     tier2_ms,
                    "llm_ms":       res4["latency_ms"],
                    "total_ms":     round(load_ms + tier1_5_ms + tier2_ms + res4["latency_ms"], 1),
                    "detected_risk": risk4,
                    "risk_correct":  sc4["s3_risk"],
                    "suggested_action": action4,
                    "action_safe":   int(is_action_safe(truth, action4)),
                    "action_dangerous": int(is_action_dangerous(truth, action4)),
                    "quality_score": sc4["quality_score"],
                    "input_tokens":  res4["input_tokens"],
                    "output_tokens": res4["output_tokens"],
                    "cost_usd":      res4["cost_usd"],
                    "reply":         res4["reply"],
                    "error":         res4["error"][:80] if res4["error"] else "",
                })

                s3   = "✓" if sc3["s3_risk"] else "✗"
                s4   = "✓" if sc4["s3_risk"] else "✗"
                safe3 = "🔴" if is_action_dangerous(truth, action3) else "🟢"
                safe4 = "🔴" if is_action_dangerous(truth, action4) else "🟢"
                print(f"   run={run}  {model:12s}  "
                      f"t3_only={sc3['detected_risk'] or '?':8s}{s3}{safe3}  "
                      f"t4_full={sc4['detected_risk'] or '?':8s}{s4}{safe4}  "
                      f"llm={res3['latency_ms']:.0f}/{res4['latency_ms']:.0f}ms")

            time.sleep(0.3)

    # ── Save runs ─────────────────────────────────────────────────────────────
    fields = [
        "condition", "model", "scene_label", "truth", "run",
        "load_ms", "tier1_5_ms", "tier2_ms", "llm_ms", "total_ms",
        "detected_risk", "risk_correct", "suggested_action",
        "action_safe", "action_dangerous", "quality_score",
        "input_tokens", "output_tokens", "cost_usd", "reply", "error",
    ]
    runs_csv = RESULTS_DIR / f"G1_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)
    print(f"\nRuns → {runs_csv}")

    # ── Summary per condition (× model for LLM conditions) ────────────────────
    print("\n" + "═" * 80)
    print("G1 Summary: Architecture Contribution")
    print("─" * 80)

    CONDITIONS = [
        ("tier1_5_only",         ["rule_based"]),
        ("tier2_only",           ["rule_based"]),
        ("tier3_only",           MODELS),
        ("tier1_5_tier2_tier3",  MODELS),
    ]

    summary_rows = []
    header = (f"  {'Condition':26s}  {'Model':12s}  {'N':>4s}  "
              f"{'LblAcc':>7s}  {'ActSafe':>7s}  {'Danger':>6s}  {'Latency':>9s}")
    print(header)
    print("  " + "─" * 76)

    for condition, models in CONDITIONS:
        for model in models:
            hr = [r for r in all_rows
                  if r["condition"] == condition
                  and r["model"] == model
                  and not r["error"]]
            if not hr:
                continue

            n = len(hr)
            acc,  a_lo, a_hi  = wilson_ci(sum(r["risk_correct"] for r in hr), n)
            safe, s_lo, s_hi  = wilson_ci(sum(r["action_safe"] for r in hr), n)
            ndang = sum(r["action_dangerous"] for r in hr)
            lat_m, _, _       = bootstrap_ci([r["total_ms"] for r in hr])

            print(f"  {condition:26s}  {model:12s}  {n:>4d}  "
                  f"{acc:.1%}  {safe:.1%}  {ndang:>6d}  {lat_m:>8.0f}ms")

            summary_rows.append({
                "condition":    condition,
                "model":        model,
                "n_trials":     n,
                "label_acc":    round(acc, 4),
                "acc_lo":       round(a_lo, 4),
                "acc_hi":       round(a_hi, 4),
                "action_safe":  round(safe, 4),
                "safe_lo":      round(s_lo, 4),
                "safe_hi":      round(s_hi, 4),
                "n_dangerous":  ndang,
                "total_ms":     round(lat_m, 1),
            })

    summary_csv = RESULTS_DIR / f"G1_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["condition", "model", "n_trials",
               "label_acc", "acc_lo", "acc_hi",
               "action_safe", "safe_lo", "safe_hi",
               "n_dangerous", "total_ms"])

    print(f"\nSummary → {summary_csv}")
    print(f"[G1] Done — {len(all_rows)} total rows recorded.")


if __name__ == "__main__":
    main()
