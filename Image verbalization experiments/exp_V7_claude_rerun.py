"""
EXP-V7-CLAUDE-RERUN: Rerun Claude rows with structured prompt fix
==================================================================
The original V7 prompt was unstructured ("Classify as: safe | caution | hazard")
which caused Claude's verbose markdown responses to produce "unknown" detections
because the risk keyword parser couldn't find safe/caution/hazard in the reply.

This script reruns only Claude with a structured prompt that enforces:
    Risk: <safe|caution|hazard>

All other model rows (gpt4o, gpt4o_mini, gemini) are kept unchanged.
Both the runs CSV and summary CSV are overwritten in place.

Overwrites: results/V7_runs_20260521_063401.csv
            results/V7_summary_20260521_063401.csv

Trials: 3 modes × 5 sequences × 5 frames × 1 model = 75 trials (~8 min)
"""

import sys, pathlib, csv, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    call_vision_llm, run_yolo_on_frame,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR, FRAMES_DIR
)

RUNS_CSV    = RESULTS_DIR / "V7_runs_20260521_063401.csv"
SUMMARY_CSV = RESULTS_DIR / "V7_summary_20260521_063401.csv"

N_SEQUENCES   = 5
HISTORY_MODES = ["stateless", "short", "full"]
ALL_MODELS    = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
SEQ_TO_RUN    = {1: 2, 2: 3, 3: 4, 4: 5, 5: 2}

SEQUENCE_DEFINITION = [
    (1, "safe",   "door_open"),
    (2, "safe",   "door_open"),
    (3, "hazard", "person_near"),
    (4, "hazard", "person_near"),
    (5, "safe",   "door_open"),
]

# Fixed prompt: enforces structured output so Claude's reply is always parseable
BASE_PROMPT = (
    "You are an AI copilot for a drone analysing a live sequence of frames.\n"
    "{context}"
    "YOLO detections on current frame: {yolo_meta}\n\n"
    "Using the YOLO metadata and the image:\n"
    "1. Describe what you see (1-2 sentences). Has the scene changed from before?\n"
    "2. Classify the scene.\n\n"
    "You MUST respond in this exact format:\n"
    "Description: <text>\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <PROCEED|SLOW_DOWN|STOP|LAND|HOLD>"
)


def get_frame_for_run(scene_label: str, run: int) -> bytes:
    path = FRAMES_DIR / f"{scene_label}_run{run:02d}_real.jpg"
    if path.exists():
        return path.read_bytes()
    from verbalization_utils import get_saved_frame
    return get_saved_frame(scene_label)


def build_context(history: list[dict], mode: str) -> str:
    if mode == "stateless" or not history:
        return ""
    prev = history[-2:] if mode == "short" else history
    lines = "\n".join(
        f"Frame {h['frame_num']}: [{h['detected_risk']}] {h['reply'][:80]}"
        for h in prev
    )
    return f"Previous frames:\n{lines}\n\n"


def extract_risk(reply: str) -> str:
    low = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if lvl in low:
            return lvl
    return "unknown"


def read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    total = len(HISTORY_MODES) * N_SEQUENCES * len(SEQUENCE_DEFINITION)
    print("=" * 65)
    print("EXP-V7-CLAUDE-RERUN: Claude structured prompt fix")
    print(f"Modes={HISTORY_MODES}  Sequences={N_SEQUENCES}  Model=claude")
    print(f"Total new trials: {total}  (~8 minutes)")
    print("=" * 65)

    existing = read_csv(RUNS_CSV)
    kept_rows = [r for r in existing if r["model"] != "claude"]
    replaced  = len(existing) - len(kept_rows)
    print(f"Loaded {len(existing)} rows — keeping {len(kept_rows)} non-claude, replacing {replaced} claude rows\n")

    new_claude_rows = []

    for hist_mode in HISTORY_MODES:
        print(f"\n{'='*55}")
        print(f"=== History mode: {hist_mode} ===")
        print(f"{'='*55}")

        for seq in range(1, N_SEQUENCES + 1):
            run_n = SEQ_TO_RUN[seq]
            print(f"\n  ── Sequence {seq}/{N_SEQUENCES} (run{run_n:02d}) ──")

            history = []

            for frame_num, expected_risk, scene_label in SEQUENCE_DEFINITION:
                jpeg      = get_frame_for_run(scene_label, run_n)
                yolo_meta = run_yolo_on_frame(jpeg)
                change_event = frame_num in (3, 5)

                context  = build_context(history, hist_mode)
                prompt   = BASE_PROMPT.format(context=context, yolo_meta=yolo_meta)

                res      = call_vision_llm(jpeg, prompt, model="claude",
                                           max_tokens=200, temperature=0.0)
                reply    = res["reply"]
                det_risk = extract_risk(reply)
                risk_correct = int(det_risk == expected_risk)

                if change_event:
                    change_detected = (int(det_risk in ("hazard", "caution"))
                                       if frame_num == 3 else int(det_risk == "safe"))
                else:
                    change_detected = -1

                row = {
                    "history_mode":    hist_mode,
                    "sequence":        seq,
                    "frame_num":       frame_num,
                    "scene_label":     scene_label,
                    "expected_risk":   expected_risk,
                    "model":           "claude",
                    "detected_risk":   det_risk,
                    "risk_correct":    risk_correct,
                    "change_event":    int(change_event),
                    "change_detected": change_detected,
                    "input_tokens":    res["input_tokens"],
                    "output_tokens":   res["output_tokens"],
                    "latency_ms":      res["latency_ms"],
                    "cost_usd":        res["cost_usd"],
                    "reply_snippet":   reply[:120].replace("\n", " "),
                    "error":           res["error"][:80] if res["error"] else "",
                }
                new_claude_rows.append(row)

                history.append({
                    "frame_num":     frame_num,
                    "detected_risk": det_risk,
                    "reply":         reply,
                })

                status = "✓" if risk_correct else "✗"
                print(f"    f{frame_num}  {det_risk:8s}  {status}  "
                      f"{'change_ok='+str(change_detected) if change_event else '':12s}  "
                      f"lat={res['latency_ms']:.0f}ms")

            time.sleep(0.3)

    # ── Merge and sort to match original CSV ordering
    all_rows = kept_rows + new_claude_rows
    mode_order  = {m: i for i, m in enumerate(HISTORY_MODES)}
    model_order = {m: i for i, m in enumerate(ALL_MODELS)}
    all_rows.sort(key=lambda r: (
        mode_order.get(r["history_mode"], 99),
        int(r["sequence"]),
        int(r["frame_num"]),
        model_order.get(r["model"], 99),
    ))

    fields = ["history_mode","sequence","frame_num","scene_label","expected_risk",
              "model","detected_risk","risk_correct","change_event","change_detected",
              "input_tokens","output_tokens","latency_ms","cost_usd","reply_snippet","error"]
    write_csv(RUNS_CSV, all_rows, fields)
    print(f"\n✓ Runs CSV overwritten: {RUNS_CSV.name}  ({len(all_rows)} rows total)")

    # ── Recompute summary for all models (kept + new claude)
    print(f"\n── Updated V7 Summary ───────────────────────────────────────────")
    print(f"  {'mode':10s}  {'model':12s}  {'risk_acc':>8s}  {'change_det':>10s}  {'tokens':>6s}  {'lat_ms':>7s}")
    print("-" * 68)

    summary_rows = []
    for hm in HISTORY_MODES:
        for model in ALL_MODELS:
            hr = [r for r in all_rows
                  if r["history_mode"] == hm and r["model"] == model
                  and not r.get("error", "")]
            if not hr:
                continue

            acc, alo, ahi = wilson_ci(
                sum(int(r["risk_correct"]) for r in hr), len(hr))

            cr = [r for r in hr
                  if int(r["change_event"]) == 1 and int(r["change_detected"]) >= 0]
            cd, cdlo, cdhi = (wilson_ci(sum(int(r["change_detected"]) for r in cr), len(cr))
                              if cr else (0., 0., 0.))

            tm, _, _ = bootstrap_ci([float(r["input_tokens"]) for r in hr])
            lm, _, _ = bootstrap_ci([float(r["latency_ms"])   for r in hr])
            cm, _, _ = bootstrap_ci([float(r["cost_usd"])     for r in hr])

            print(f"  {hm:10s}  {model:12s}  {acc:.3f}[{alo:.3f},{ahi:.3f}]  "
                  f"{cd:.3f}[{cdlo:.3f},{cdhi:.3f}]  {tm:.0f}  {lm:.0f}ms")

            summary_rows.append({
                "history_mode":      hm,
                "model":             model,
                "n_trials":          len(hr),
                "risk_accuracy":     round(acc, 4),
                "acc_lo":            round(alo, 4),
                "acc_hi":            round(ahi, 4),
                "change_detect":     round(cd, 4),
                "cd_lo":             round(cdlo, 4),
                "cd_hi":             round(cdhi, 4),
                "mean_input_tokens": round(tm, 2),
                "latency_ms":        round(lm, 3),
                "cost_usd":          round(cm, 6),
            })

    write_csv(SUMMARY_CSV, summary_rows,
              ["history_mode","model","n_trials","risk_accuracy","acc_lo","acc_hi",
               "change_detect","cd_lo","cd_hi","mean_input_tokens","latency_ms","cost_usd"])

    print(f"\n✓ Summary CSV overwritten: {SUMMARY_CSV.name}")
    print(f"[V7-RERUN] Done — {len(new_claude_rows)} new Claude trials recorded.")


if __name__ == "__main__":
    main()
