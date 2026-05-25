"""
EXP-G2: Trigger Strategy — Scheduled-Only vs Scheduled+YOLO-Hybrid
====================================================================
Goal:
    YOLO runs on every frame in both conditions. Compare two LLM trigger
    strategies on a simulated 10-frame mission sequence:

    Condition A — scheduled:
        LLM called every SCHEDULE_INTERVAL frames only.
        YOLO detects hazards but does NOT trigger extra LLM calls.
        Hazards appearing between ticks are missed until next tick.

    Condition B — hybrid:
        LLM called on the same schedule PLUS immediately on first YOLO
        hazard detection (with cooldown to avoid hammering).
        Catches hazards the moment YOLO sees them.

    Sequence (10 frames):
        Frames 1-2  : door_open (safe)   — baseline
        Frames 3-7  : person_near (hazard) ← hazard window
        Frames 8-10 : door_open (safe)   — hazard cleared

    Schedule interval = 5 frames → ticks at frames 1 and 6.
    Hazard starts at frame 3 (between ticks 1 and 6).

    Scheduled: first LLM call during hazard window = frame 6 (3 frames late).
    Hybrid:    YOLO triggers LLM at frame 3 immediately.

Models:  claude, gpt4o, gpt4o_mini, gemini
N runs:  5 sequences per condition per model
Total:   2 conditions × 4 models × 5 runs × ~3 LLM calls = ~120 LLM calls (~15 min)

Output: Image verbalization experiments/results/G2_runs_<ts>.csv
        Image verbalization experiments/results/G2_summary_<ts>.csv
"""

import sys, pathlib, datetime, time
import numpy as np
import cv2

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
EXP_DIR   = pathlib.Path(__file__).parent
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (  # noqa: E402
    call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR, FRAMES_DIR
)
from enhanced_yolo_pipeline import (  # noqa: E402
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer, enhanced_rule_risk, COMBINED_PROMPT_TEMPLATE
)
from robust_local_detector import (  # noqa: E402
    load_mediapipe_detector, detect_hazard
)

N_RUNS            = 5
MODELS            = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
SCHEDULE_INTERVAL = 5          # LLM called every 5 frames
YOLO_COOLDOWN     = 3          # min frames between YOLO-triggered calls
HAZARD_FRAME_RANGE = (3, 7)    # frames where hazard is active (inclusive)

# 10-frame sequence: door_open×2, person_near×5, door_open×3
# run02-run05 used (run01 buggy); run02 repeated for seq5
SEQ_TO_RUN = {1: 2, 2: 3, 3: 4, 4: 5, 5: 2}

SEQUENCE = [
    (1,  "door_open",   "safe"),
    (2,  "door_open",   "safe"),
    (3,  "person_near", "hazard"),   # ← hazard starts
    (4,  "person_near", "hazard"),
    (5,  "person_near", "hazard"),
    (6,  "person_near", "hazard"),
    (7,  "person_near", "hazard"),   # ← hazard ends
    (8,  "door_open",   "safe"),
    (9,  "door_open",   "safe"),
    (10, "door_open",   "safe"),
]



def get_frame(scene_label: str, run_n: int) -> bytes:
    path = FRAMES_DIR / f"{scene_label}_run{run_n:02d}_real.jpg"
    if path.exists():
        return path.read_bytes()
    path = FRAMES_DIR / f"{scene_label}_run03_real.jpg"
    return path.read_bytes()


def is_yolo_hazard(yolo_meta: str, clip_risk: str) -> bool:
    """True if enhanced rule risk is hazard or caution (not safe).
    Uses enhanced_rule_risk so door/window detections are not treated as hazards."""
    return enhanced_rule_risk(yolo_meta, clip_risk) in ("hazard", "caution")


def run_sequence(strategy: str, model: str, seq: int,
                 yolo_model, yolo_type, coco_model, depth_pipe,
                 mp_detector, mp_type) -> tuple[dict, list]:
    """
    Simulate one 10-frame mission sequence.
    Returns (per-sequence stats, per-call rows with reply).

    Tier 1.5 — robust_local_detector runs on every frame (no API, ~14 ms).
    In hybrid mode the local detector acts as the emergency LLM trigger
    (person close + wall + darkness checks) instead of raw YOLO flag.
    YOLO+DA v2 tier always runs for full metadata.
    """
    run_n = SEQ_TO_RUN[seq]
    llm_calls       = 0
    scheduled_calls = 0
    local_calls     = 0          # Tier-1.5 triggered LLM calls
    total_cost      = 0.0
    hazard_caught   = False
    hazard_catch_frame = None
    last_local_call_frame = -YOLO_COOLDOWN
    call_rows       = []

    for frame_num, scene_label, expected_risk in SEQUENCE:
        jpeg     = get_frame(scene_label, run_n)

        # Tier 1.5: local emergency detector (no API — runs every frame)
        img_bgr      = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        t_local      = time.perf_counter()
        local_result = detect_hazard(img_bgr, depth_map=None,
                                     mp_detector=mp_detector, mp_type=mp_type)
        local_ms     = round((time.perf_counter() - t_local) * 1000.0, 2)
        local_flag   = local_result["risk"] == "hazard"

        # Tier 2: dual-YOLO + DA v2 (runs every frame, provides LLM metadata)
        tier2    = enhanced_yolo_infer(yolo_model, yolo_type,
                                       None, None, None, jpeg,
                                       coco_model=coco_model,
                                       depth_pipe=depth_pipe)
        # Append local detector so LLM sees MediaPipe person detection
        # even when YOLO-World misclassifies person as wall/structural object
        yolo_meta = (
            f"{tier2['yolo_meta']}\n"
            f"  Local detector (Tier 1.5 — MediaPipe EfficientDet-Lite0, advisory — image overrides): "
            f"{local_result['metadata']}"
        )
        in_hazard = HAZARD_FRAME_RANGE[0] <= frame_num <= HAZARD_FRAME_RANGE[1]

        should_call  = False
        call_source  = None

        # Scheduled tick (both strategies)
        if (frame_num - 1) % SCHEDULE_INTERVAL == 0:
            should_call = True
            call_source = "scheduled"

        # Local-detector interrupt (hybrid only) — Tier 1.5 emergency trigger
        if (strategy == "hybrid"
                and not should_call
                and local_flag
                and not hazard_caught
                and (frame_num - last_local_call_frame) >= YOLO_COOLDOWN):
            should_call = True
            call_source = "local"

        if should_call:
            prompt = COMBINED_PROMPT_TEMPLATE.format(
                yolo_meta  = yolo_meta,
            )
            res    = call_vision_llm(jpeg, prompt, model=model,
                                     max_tokens=300, temperature=0.0)
            scores = score_verbalization(res["reply"], expected_risk)
            llm_calls  += 1
            total_cost += res["cost_usd"]

            if call_source == "scheduled":
                scheduled_calls += 1
            else:
                local_calls           += 1
                last_local_call_frame  = frame_num

            # Did this call catch the hazard?
            stop_or_hazard = (scores["detected_risk"] in ("hazard", "caution")
                               or "hover" in res["reply"].lower()
                               or "pitch_back" in res["reply"].lower()
                               or "stop" in res["reply"].lower())
            if in_hazard and not hazard_caught and stop_or_hazard:
                hazard_caught      = True
                hazard_catch_frame = frame_num

            call_rows.append({
                "strategy":      strategy,
                "model":         model,
                "seq":           seq,
                "frame_num":     frame_num,
                "scene_label":   scene_label,
                "expected_risk": expected_risk,
                "call_source":   call_source,
                "in_hazard":     int(in_hazard),
                "local_ms":      local_ms,
                "local_risk":    local_result["risk"],
                "local_trigger": int(local_flag),
                "yolo_meta":     yolo_meta,
                "clip_label":    tier2["clip_label"],
                "clip_risk":     tier2["clip_risk"],
                "clip_conf":     tier2["clip_conf"],
                "detected_risk": scores["detected_risk"] or "",
                "risk_correct":  scores["s3_risk"],
                "latency_ms":    res["latency_ms"],
                "cost_usd":      res["cost_usd"],
                "reply":         res["reply"],
                "error":         res["error"][:80] if res["error"] else "",
            })

            time.sleep(0.2)

    seq_stats = {
        "strategy":          strategy,
        "model":             model,
        "seq":               seq,
        "llm_calls":         llm_calls,
        "scheduled_calls":   scheduled_calls,
        "local_calls":       local_calls,
        "cost_usd":          round(total_cost, 7),
        "hazard_caught":     int(hazard_caught),
        "catch_frame":       hazard_catch_frame if hazard_caught else -1,
        "frames_late":       (hazard_catch_frame - HAZARD_FRAME_RANGE[0])
                             if hazard_caught else -1,
    }
    return seq_stats, call_rows


def main():
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = 2 * len(MODELS) * N_RUNS
    print("=" * 65)
    print("EXP-G2: Scheduled vs Hybrid YOLO Trigger")
    print(f"Sequence: safe×2 → hazard×5 → safe×3  (10 frames)")
    print(f"Schedule every {SCHEDULE_INTERVAL} frames  |  YOLO cooldown {YOLO_COOLDOWN} frames")
    print(f"Conditions: scheduled | hybrid  |  Models={MODELS}")
    print(f"Total runs: {total}  (~15 min)")
    print("=" * 65)

    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5 — local emergency detector)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()

    all_rows  = []
    all_calls = []

    for strategy in ("scheduled", "hybrid"):
        print(f"\n{'='*55}")
        print(f"=== Strategy: {strategy} ===")
        print(f"{'='*55}")
        for model in MODELS:
            print(f"\n  ── Model: {model} ──")
            for seq in range(1, N_RUNS + 1):
                row, call_rows = run_sequence(
                    strategy, model, seq,
                    yolo_model, yolo_type, coco_model, depth_pipe,
                    mp_detector, mp_type
                )
                all_rows.append(row)
                all_calls.extend(call_rows)
                status = "✓ CAUGHT" if row["hazard_caught"] else "✗ MISSED"
                print(f"    seq={seq}  {status}  "
                      f"catch_frame={row['catch_frame']}  "
                      f"frames_late={row['frames_late']}  "
                      f"llm_calls={row['llm_calls']}  "
                      f"cost=${row['cost_usd']:.5f}")

    # ── Save runs (sequence-level)
    fields = ["strategy","model","seq","llm_calls","scheduled_calls","local_calls",
              "cost_usd","hazard_caught","catch_frame","frames_late"]
    runs_csv = RESULTS_DIR / f"G2_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Save calls (per-LLM-call with reply)
    call_fields = ["strategy","model","seq","frame_num","scene_label","expected_risk",
                   "call_source","in_hazard","local_ms","local_risk","local_trigger",
                   "yolo_meta","clip_label","clip_risk","clip_conf",
                   "detected_risk","risk_correct",
                   "latency_ms","cost_usd","reply","error"]
    calls_csv = RESULTS_DIR / f"G2_calls_{ts}.csv"
    write_csv(calls_csv, all_calls, call_fields)
    print(f"Calls   → {calls_csv}")

    # ── Summary
    print(f"\n── G2 Summary ──────────────────────────────────────────────────")
    print(f"  {'strategy':12s}  {'model':12s}  {'catch_rate':>10s}  "
          f"{'frames_late':>11s}  {'llm_calls':>9s}  {'cost':>9s}")
    print("-" * 72)

    summary_rows = []
    for strategy in ("scheduled", "hybrid"):
        for model in MODELS:
            hr = [r for r in all_rows
                  if r["strategy"] == strategy and r["model"] == model]
            if not hr:
                continue

            catch_rate, cr_lo, cr_hi = wilson_ci(
                sum(r["hazard_caught"] for r in hr), len(hr))
            late_data = [r["frames_late"] for r in hr if r["frames_late"] >= 0]
            late_m, _, _  = bootstrap_ci(late_data) if late_data else (float("nan"),)*3
            calls_m, _, _ = bootstrap_ci([r["llm_calls"] for r in hr])
            cost_m, _, _  = bootstrap_ci([r["cost_usd"]  for r in hr])

            print(f"  {strategy:12s}  {model:12s}  "
                  f"{catch_rate:.3f}[{cr_lo:.3f},{cr_hi:.3f}]  "
                  f"late={late_m:.1f}fr    "
                  f"calls={calls_m:.1f}    ${cost_m:.5f}")

            local_calls_m, _, _ = bootstrap_ci([r["local_calls"] for r in hr])
            summary_rows.append({
                "strategy": strategy, "model": model, "n_runs": len(hr),
                "catch_rate": round(catch_rate, 4),
                "cr_lo": round(cr_lo, 4), "cr_hi": round(cr_hi, 4),
                "mean_frames_late": round(late_m, 2) if late_data else float("nan"),
                "mean_llm_calls": round(calls_m, 2),
                "mean_local_calls": round(local_calls_m, 2),
                "mean_cost_usd": round(cost_m, 6),
            })

    summary_csv = RESULTS_DIR / f"G2_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["strategy","model","n_runs","catch_rate","cr_lo","cr_hi",
               "mean_frames_late","mean_llm_calls","mean_local_calls","mean_cost_usd"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[G2] Done — {len(all_rows)} runs recorded.")


if __name__ == "__main__":
    main()
