"""
EXP-G4: Four-Tier Timescale Validation
=========================================
Goal:
    Validate that all four control tiers operate at the correct timescales
    and that no tier encroaches on the period of a faster tier.

    Tier 1  — PID inner loop       : target   0.25 ms  (4 kHz firmware arithmetic)
    Tier 1.5— Local detector       : target  ~14   ms  (MediaPipe + texture + brightness)
    Tier 2  — Dual-YOLO + DA v2   : target ~330   ms  (inference + metric depth)
    Tier 3  — LLM outer loop       : target ~1000  ms  (0.1–1 Hz API call)

    PID        : pure Python arithmetic (firmware-equivalent timing).
    Local      : robust_local_detector — MediaPipe EfficientDet-Lite0 person detect +
                 Sobel texture wall check + brightness gate. No API. Emergency trigger.
    Dual-YOLO  : YOLOv11n COCO + YOLO-World + CLAHE + DepthAnything v2 metric depth.
                 Model loaded once — inference timing only.
    LLM        : all 4 models timed on saved frames with real YOLO+depth metadata.

    Tier separation ratios (all should differ by ~10–100×):
        Tier1 → Local : local_mean / pid_mean
        Local → YOLO  : yolo_mean  / local_mean
        YOLO  → LLM   : llm_mean   / yolo_mean

Models  (Tier 3): claude, gpt4o, gpt4o_mini, gemini
N runs : 5 independent runs per tier
Samples: 500 PID ticks, 8×5=40 local/YOLO frames, 8×5=40 LLM calls per model

Output: Image verbalization experiments/results/G4_runs_<ts>.csv
        Image verbalization experiments/results/G4_summary_<ts>.csv
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
    get_saved_frame, call_vision_llm,
    bootstrap_ci, write_csv, RESULTS_DIR, SCENES
)
from enhanced_yolo_pipeline import (  # noqa: E402
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer, COMBINED_PROMPT_TEMPLATE
)
from robust_local_detector import (  # noqa: E402
    load_mediapipe_detector, detect_hazard
)

N_RUNS    = 5
N_PID     = 500
MODELS    = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

TARGET_PID_MS  = 0.25
TARGET_YOLO_MS = 33.3


def measure_pid_tier(n_ticks: int) -> list[float]:
    """Simulate PID arithmetic and measure actual Python tick period."""
    kp, ki, kd = 1.2, 0.05, 0.3
    err_prev = 0.0
    integral = 0.0
    setpoint = 1.0
    altitude = 0.0
    periods  = []
    t_prev   = time.perf_counter()
    for _ in range(n_ticks):
        err      = setpoint - altitude
        integral += err * TARGET_PID_MS / 1000.0
        deriv    = (err - err_prev) / (TARGET_PID_MS / 1000.0)
        thrust   = kp * err + ki * integral + kd * deriv
        altitude += thrust * (TARGET_PID_MS / 1000.0) * 0.5
        err_prev = err
        t_now    = time.perf_counter()
        periods.append((t_now - t_prev) * 1000.0)
        t_prev   = t_now
    return periods


def measure_local_tier(mp_detector, mp_type, scenes: list) -> list[float]:
    """
    Tier 1.5 timing: robust_local_detector wall-clock time.
    MediaPipe person detect + Sobel texture wall check + brightness gate.
    Runs WITHOUT DA v2 depth — represents real emergency-trigger latency.
    depth_map=None: gated check only; fast path identical to production.
    """
    periods = []
    for scene in scenes:
        jpeg    = get_saved_frame(scene["label"])
        img_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        t0      = time.perf_counter()
        detect_hazard(img_bgr, depth_map=None,
                      mp_detector=mp_detector, mp_type=mp_type)
        periods.append((time.perf_counter() - t0) * 1000.0)
    return periods


def measure_yolo_tier(yolo_model, yolo_type, coco_model, depth_pipe,
                      scenes: list) -> list[float]:
    """Tier 2 timing: full pipeline wall-clock (CLAHE + dual-YOLO + DA v2)."""
    periods = []
    for scene in scenes:
        jpeg = get_saved_frame(scene["label"])
        t0   = time.perf_counter()
        enhanced_yolo_infer(yolo_model, yolo_type,
                            None, None, None, jpeg,
                            coco_model=coco_model, depth_pipe=depth_pipe)
        periods.append((time.perf_counter() - t0) * 1000.0)
    return periods


def measure_llm_tier(model: str, scenes: list,
                     yolo_model, yolo_type,
                     coco_model, depth_pipe) -> tuple[list, list]:
    """LLM API call timing with dual-YOLO + DA v2 metadata."""
    periods   = []
    call_rows = []
    for scene in scenes:
        jpeg  = get_saved_frame(scene["label"])
        tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                    None, None, None, jpeg,
                                    coco_model=coco_model, depth_pipe=depth_pipe)
        prompt = COMBINED_PROMPT_TEMPLATE.format(
            yolo_meta  = tier2["yolo_meta"],
        )
        res = call_vision_llm(jpeg, prompt, model=model,
                              max_tokens=300, temperature=0.0)
        periods.append(res["latency_ms"])
        call_rows.append({
            "model":       model,
            "scene_label": scene["label"],
            "truth":       scene["truth"],
            "latency_ms":  res["latency_ms"],
            "input_tokens":  res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "cost_usd":    res["cost_usd"],
            "reply":       res["reply"],
            "error":       res["error"][:80] if res["error"] else "",
        })
        time.sleep(0.2)
    return periods, call_rows


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_calls = len(MODELS) * len(SCENES) * N_RUNS
    print("=" * 65)
    print("EXP-G4: Four-Tier Timescale Validation")
    print(f"PID: {N_PID} ticks × {N_RUNS} runs")
    print(f"Local detector: {len(SCENES)} frames × {N_RUNS} runs")
    print(f"Dual-YOLO+DA v2: {len(SCENES)} frames × {N_RUNS} runs")
    print(f"LLM: {len(SCENES)} frames × {N_RUNS} runs × {len(MODELS)} models = {llm_calls} calls")
    print("=" * 65)

    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5 — local emergency detector)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()

    all_pid_ms   = []
    all_local_ms = []
    all_yolo_ms  = []
    llm_ms_by_model = {m: [] for m in MODELS}
    run_rows  = []
    all_calls = []

    for run in range(1, N_RUNS + 1):
        print(f"\n── Run {run}/{N_RUNS} ──")

        # Tier 1: PID
        print(f"  PID ({N_PID} ticks)…")
        pid_ms = measure_pid_tier(N_PID)
        all_pid_ms.extend(pid_ms)
        print(f"    mean={np.mean(pid_ms):.5f}ms  std={np.std(pid_ms):.5f}ms")

        # Tier 1.5: Local emergency detector (MediaPipe + texture + brightness)
        print(f"  Local detector ({len(SCENES)} frames)…")
        local_ms = measure_local_tier(mp_detector, mp_type, SCENES)
        all_local_ms.extend(local_ms)
        print(f"    mean={np.mean(local_ms):.2f}ms  std={np.std(local_ms):.2f}ms")

        # Tier 2: dual-YOLO + DA v2 (CLAHE + YOLOv11n COCO + YOLO-World + depth)
        print(f"  Dual-YOLO + DA v2 ({len(SCENES)} frames)…")
        yolo_ms = measure_yolo_tier(yolo_model, yolo_type,
                                    coco_model, depth_pipe, SCENES)
        all_yolo_ms.extend(yolo_ms)
        print(f"    mean={np.mean(yolo_ms):.2f}ms  std={np.std(yolo_ms):.2f}ms")

        # Tier 3: LLM
        for model in MODELS:
            print(f"  LLM {model} ({len(SCENES)} calls)…")
            lms, call_rows = measure_llm_tier(model, SCENES,
                                              yolo_model, yolo_type,
                                              coco_model, depth_pipe)
            for cr in call_rows:
                cr["run"] = run
            all_calls.extend(call_rows)
            llm_ms_by_model[model].extend(lms)
            print(f"    mean={np.mean(lms):.0f}ms")

        pid_mean   = float(np.mean(pid_ms))
        local_mean = float(np.mean(local_ms))
        yolo_mean  = float(np.mean(yolo_ms))
        run_rows.append({
            "run":            run,
            "pid_mean_ms":    round(pid_mean, 5),
            "pid_std_ms":     round(float(np.std(pid_ms)), 5),
            "local_mean_ms":  round(local_mean, 2),
            "local_std_ms":   round(float(np.std(local_ms)), 2),
            "ratio_pid_local": round(local_mean / pid_mean, 1) if pid_mean > 0 else 0,
            "yolo_mean_ms":   round(yolo_mean, 2),
            "yolo_std_ms":    round(float(np.std(yolo_ms)), 2),
            "ratio_local_yolo": round(yolo_mean / local_mean, 1) if local_mean > 0 else 0,
            **{f"{m}_mean_ms": round(float(np.mean(llm_ms_by_model[m][-len(SCENES):])), 1)
               for m in MODELS},
        })

    # ── Save runs
    run_fields = (["run",
                   "pid_mean_ms","pid_std_ms",
                   "local_mean_ms","local_std_ms","ratio_pid_local",
                   "yolo_mean_ms","yolo_std_ms","ratio_local_yolo"]
                  + [f"{m}_mean_ms" for m in MODELS])
    runs_csv = RESULTS_DIR / f"G4_runs_{ts}.csv"
    write_csv(runs_csv, run_rows, run_fields)

    # ── Aggregate summary
    pm,  pm_lo,  pm_hi  = bootstrap_ci(all_pid_ms)
    lom, lom_lo, lom_hi = bootstrap_ci(all_local_ms)
    ym,  ym_lo,  ym_hi  = bootstrap_ci(all_yolo_ms)

    print(f"\n── G4 Summary — Four-Tier Timescale ───────────────────────────")
    print(f"  {'tier':25s}  {'mean_ms':>10s}  {'ci_lo':>8s}  {'ci_hi':>8s}  "
          f"{'ratio_to_prev':>14s}")
    print("-" * 80)
    print(f"  {'PID (Tier 1)':25s}  {pm:>10.5f}  {pm_lo:>8.5f}  {pm_hi:>8.5f}  "
          f"{'baseline':>14s}")
    print(f"  {'Local detector (Tier 1.5)':25s}  {lom:>10.2f}  {lom_lo:>8.2f}  "
          f"{lom_hi:>8.2f}  {lom/pm:>13.0f}×")
    print(f"  {'Dual-YOLO+DA v2 (Tier 2)':25s}  {ym:>10.2f}  {ym_lo:>8.2f}  "
          f"{ym_hi:>8.2f}  {ym/lom:>13.0f}×")

    summary_rows = [
        {"tier": "pid",   "mean_ms": pm,  "ci_lo": pm_lo,  "ci_hi": pm_hi,
         "ratio_to_prev": 1.0,           "model": "—"},
        {"tier": "local", "mean_ms": lom, "ci_lo": lom_lo, "ci_hi": lom_hi,
         "ratio_to_prev": round(lom/pm, 1),  "model": "—"},
        {"tier": "yolo",  "mean_ms": ym,  "ci_lo": ym_lo,  "ci_hi": ym_hi,
         "ratio_to_prev": round(ym/lom, 1),  "model": "—"},
    ]

    for model in MODELS:
        lm, lm_lo, lm_hi = bootstrap_ci(llm_ms_by_model[model])
        print(f"  {'LLM '+model+' (Tier 3)':25s}  {lm:>10.0f}  {lm_lo:>8.0f}  "
              f"{lm_hi:>8.0f}  {lm/ym:>13.0f}×")
        summary_rows.append({
            "tier": "llm", "mean_ms": lm, "ci_lo": lm_lo, "ci_hi": lm_hi,
            "ratio_to_prev": round(lm/ym, 1), "model": model,
        })

    summary_csv = RESULTS_DIR / f"G4_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["tier","model","mean_ms","ci_lo","ci_hi","ratio_to_prev"])

    calls_csv = RESULTS_DIR / f"G4_calls_{ts}.csv"
    write_csv(calls_csv, all_calls,
              ["run","model","scene_label","truth","latency_ms",
               "input_tokens","output_tokens","cost_usd","reply","error"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"Calls   → {calls_csv}")
    print(f"[G4] Done.")


if __name__ == "__main__":
    main()
