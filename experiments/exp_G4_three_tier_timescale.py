"""
EXP-G4: Three-Tier Timescale Validation
=========================================
Goal:
    Validate that the three control tiers operate at the correct timescales
    and that no tier encroaches on the period of a faster tier.

    Tier 1 — PID inner loop    : target 0.25ms (4kHz firmware arithmetic)
    Tier 2 — YOLO middle loop  : target ~33ms  (30fps inference on real frames)
    Tier 3 — LLM outer loop    : target ~5000ms (0.1–1Hz API call)

    PID: pure Python arithmetic (firmware-equivalent timing).
    YOLO: inference timing on all 8 saved run03 real hardware frames.
          Model loaded once — measures inference only, not model load.
    LLM: all 4 models timed on saved frames with real YOLO metadata.

    Tier separation ratios:
        Tier1→2: yolo_mean / pid_mean      (should be ~100–200×)
        Tier2→3: llm_mean  / yolo_mean     (should be ~100–300×)

Models  (Tier 3): claude, gpt4o, gpt4o_mini, gemini
N runs : 5 independent runs per tier
Samples: 500 PID ticks, 8×5=40 YOLO frames, 8×5=40 LLM calls per model

Output: Image verbalization experiments/results/G4_runs_<ts>.csv
        Image verbalization experiments/results/G4_summary_<ts>.csv
"""

import sys, pathlib, datetime, time
import numpy as np

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
    load_enhanced_yolo, load_clip, enhanced_yolo_infer,
    COMBINED_PROMPT_TEMPLATE
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


def measure_yolo_tier(yolo_model, yolo_type, clip_model,
                      clip_preprocess, clip_tokenizer,
                      scenes: list) -> list[float]:
    """Enhanced Tier 2 timing: full pipeline wall-clock time (CLAHE + YOLO-World + CLIP)."""
    periods = []
    for scene in scenes:
        jpeg = get_saved_frame(scene["label"])
        t0   = time.perf_counter()
        enhanced_yolo_infer(yolo_model, yolo_type,
                            clip_model, clip_preprocess,
                            clip_tokenizer, jpeg)
        periods.append((time.perf_counter() - t0) * 1000.0)
    return periods


def measure_llm_tier(model: str, scenes: list,
                     yolo_model, yolo_type,
                     clip_model, clip_preprocess,
                     clip_tokenizer) -> tuple[list, list]:
    """LLM API call timing with enhanced YOLO+CLIP metadata."""
    periods   = []
    call_rows = []
    for scene in scenes:
        jpeg  = get_saved_frame(scene["label"])
        tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                    clip_model, clip_preprocess,
                                    clip_tokenizer, jpeg)
        prompt = COMBINED_PROMPT_TEMPLATE.format(
            yolo_meta  = tier2["yolo_meta"],
            clip_label = tier2["clip_label"],
            clip_conf  = tier2["clip_conf"],
            clip_risk  = tier2["clip_risk"],
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
    print("EXP-G4: Three-Tier Timescale Validation")
    print(f"PID: {N_PID} ticks × {N_RUNS} runs")
    print(f"YOLO: {len(SCENES)} frames × {N_RUNS} runs  (model loaded once)")
    print(f"LLM: {len(SCENES)} frames × {N_RUNS} runs × {len(MODELS)} models = {llm_calls} calls")
    print("=" * 65)

    print("\nLoading enhanced YOLO tier…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading CLIP scene screener…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_pid_ms    = []
    all_yolo_ms   = []
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

        # Tier 2: YOLO (enhanced: CLAHE + YOLO-World + CLIP)
        print(f"  YOLO-enhanced ({len(SCENES)} frames)…")
        yolo_ms = measure_yolo_tier(yolo_model, yolo_type,
                                    clip_model, clip_preprocess,
                                    clip_tokenizer, SCENES)
        all_yolo_ms.extend(yolo_ms)
        print(f"    mean={np.mean(yolo_ms):.2f}ms  std={np.std(yolo_ms):.2f}ms")

        # Tier 3: LLM
        for model in MODELS:
            print(f"  LLM {model} ({len(SCENES)} calls)…")
            lms, call_rows = measure_llm_tier(model, SCENES,
                                              yolo_model, yolo_type,
                                              clip_model, clip_preprocess,
                                              clip_tokenizer)
            for cr in call_rows:
                cr["run"] = run
            all_calls.extend(call_rows)
            llm_ms_by_model[model].extend(lms)
            print(f"    mean={np.mean(lms):.0f}ms")

        pid_mean  = float(np.mean(pid_ms))
        yolo_mean = float(np.mean(yolo_ms))
        run_rows.append({
            "run":           run,
            "pid_mean_ms":   round(pid_mean, 5),
            "pid_std_ms":    round(float(np.std(pid_ms)), 5),
            "pid_cv":        round(float(np.std(pid_ms) / pid_mean), 4) if pid_mean > 0 else 0,
            "yolo_mean_ms":  round(yolo_mean, 3),
            "yolo_std_ms":   round(float(np.std(yolo_ms)), 3),
            "yolo_cv":       round(float(np.std(yolo_ms) / yolo_mean), 4) if yolo_mean > 0 else 0,
            "tier12_ratio":  round(yolo_mean / pid_mean, 1) if pid_mean > 0 else 0,
            **{f"{m}_mean_ms": round(float(np.mean(llm_ms_by_model[m][-len(SCENES):])), 1)
               for m in MODELS},
        })

    # ── Save runs
    run_fields = (["run","pid_mean_ms","pid_std_ms","pid_cv",
                   "yolo_mean_ms","yolo_std_ms","yolo_cv","tier12_ratio"]
                  + [f"{m}_mean_ms" for m in MODELS])
    runs_csv = RESULTS_DIR / f"G4_runs_{ts}.csv"
    write_csv(runs_csv, run_rows, run_fields)

    # ── Aggregate summary
    pm, pm_lo, pm_hi = bootstrap_ci(all_pid_ms)
    ym, ym_lo, ym_hi = bootstrap_ci(all_yolo_ms)

    print(f"\n── G4 Summary ──────────────────────────────────────────────────")
    print(f"  {'tier':20s}  {'mean_ms':>10s}  {'ci_lo':>8s}  {'ci_hi':>8s}  "
          f"{'target_ms':>10s}  {'ratio_to_pid':>12s}")
    print("-" * 78)
    print(f"  {'PID (Tier 1)':20s}  {pm:>10.5f}  {pm_lo:>8.5f}  {pm_hi:>8.5f}  "
          f"{TARGET_PID_MS:>10.2f}  {'—':>12s}")
    print(f"  {'YOLO (Tier 2)':20s}  {ym:>10.2f}  {ym_lo:>8.2f}  {ym_hi:>8.2f}  "
          f"{TARGET_YOLO_MS:>10.1f}  {ym/pm:>11.0f}×")

    summary_rows = [
        {"tier": "pid",  "mean_ms": pm, "ci_lo": pm_lo, "ci_hi": pm_hi,
         "target_ms": TARGET_PID_MS, "ratio_to_pid": 1.0, "model": "—"},
        {"tier": "yolo", "mean_ms": ym, "ci_lo": ym_lo, "ci_hi": ym_hi,
         "target_ms": TARGET_YOLO_MS, "ratio_to_pid": round(ym/pm, 1), "model": "—"},
    ]

    for model in MODELS:
        lm, lm_lo, lm_hi = bootstrap_ci(llm_ms_by_model[model])
        print(f"  {'LLM '+model+' (Tier 3)':20s}  {lm:>10.0f}  {lm_lo:>8.0f}  "
              f"{lm_hi:>8.0f}  {'—':>10s}  {lm/pm:>11.0f}×  "
              f"(Tier2→3: {lm/ym:.0f}×)")
        summary_rows.append({
            "tier": "llm", "mean_ms": lm, "ci_lo": lm_lo, "ci_hi": lm_hi,
            "target_ms": float("nan"), "ratio_to_pid": round(lm/pm, 1),
            "model": model,
        })

    summary_csv = RESULTS_DIR / f"G4_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["tier","model","mean_ms","ci_lo","ci_hi","target_ms","ratio_to_pid"])

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
