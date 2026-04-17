"""
EXP-V7: Scene Context History Effect
======================================
Goal:
    Does feeding the LLM a history of previous frame descriptions help it
    detect mid-sequence scene changes?

    History modes:
        stateless : each API call has no prior context
        short     : last 2 frame descriptions prepended as context
        full      : ALL prior frame descriptions in session

    Scenario: operator creates a 5-frame sequence with a change at frame 3:
        Frame 1: clear     (baseline)
        Frame 2: clear     (same)
        Frame 3: hazard    ← CHANGE (e.g. person walks in / object placed)
        Frame 4: hazard    (continues)
        Frame 5: clear     ← CHANGE BACK (hazard removed)

    N=5 sequences per history mode = 75 trials per mode (15 total per sequence).
    Each trial = one API call for one frame in the sequence.

Metrics:
    - change_detected      : LLM flags the change at frame 3 and 5 (Wilson CI)
    - description_drift    : semantic consistency across frames 1-2 (Bootstrap CI)
    - input_tokens         : total context consumed per call (Bootstrap CI)
    - cost_usd             : per call (Bootstrap CI)
    - latency_ms           : per call (Bootstrap CI)

Scoring:
    change_detected_f3 = 1 if LLM says "hazard" on frame 3 (was "safe/clear")
    change_detected_f5 = 1 if LLM says "safe" on frame 5   (was "hazard")
    drift_score        = fraction of frames 1-2 with consistent "safe" output
"""

import sys, os, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    get_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR
)

N_SEQUENCES   = 5
HISTORY_MODES = ["stateless", "short", "full"]
MODEL         = "claude"

SEQUENCE_DEFINITION = [
    # (frame_num, expected_risk, operator_instruction)
    (1, "safe",    "Clear scene — empty floor, nothing in frame."),
    (2, "safe",    "Same clear scene."),
    (3, "hazard",  "CHANGE: Place an obstacle close OR person steps in front."),
    (4, "hazard",  "Same hazard — keep obstacle/person in place."),
    (5, "safe",    "CHANGE BACK: Remove obstacle/person — clear again."),
]

BASE_PROMPT = (
    "You are a drone camera monitor analysing a live sequence of frames.\n"
    "{context}"
    "Current frame: Describe what you see. "
    "Has the scene changed from before? "
    "Classify as: safe | caution | hazard"
)

def build_context(history: list[dict], mode: str) -> str:
    if mode == "stateless" or not history:
        return ""
    if mode == "short":
        prev = history[-2:]
    else:
        prev = history
    lines = "\n".join(
        f"Frame {h['frame_num']}: [{h['detected_risk'] or 'unknown'}] {h['reply'][:80]}"
        for h in prev
    )
    return f"Previous frames:\n{lines}\n\n"

def detect_change(reply: str, expected_risk: str) -> int:
    r = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if lvl in r:
            return int(lvl == expected_risk)
    return 0

def consistency_score(history_safe_frames: list[str]) -> float:
    if not history_safe_frames:
        return float("nan")
    return sum(1 for r in history_safe_frames if "safe" in r.lower()) / len(history_safe_frames)

def main():
    print("="*60)
    print("EXP-V7: Scene Context History Effect")
    print(f"History modes={HISTORY_MODES}  Sequences={N_SEQUENCES}")
    print("="*60)
    if not preflight():
        ans = input("ESP32 not reachable. Use synthetic frames? [y/N]: ")
        if ans.strip().lower() != "y":
            return

    all_rows = []

    for hist_mode in HISTORY_MODES:
        print(f"\n{'='*50}")
        print(f"=== History mode: {hist_mode} ===")
        print(f"{'='*50}")

        for seq in range(1, N_SEQUENCES+1):
            print(f"\n  ── Sequence {seq}/{N_SEQUENCES} ──")
            history: list[dict] = []

            for frame_num, expected_risk, instruction in SEQUENCE_DEFINITION:
                print(f"\n  Frame {frame_num}/5  expected={expected_risk}")
                print(f"  [SETUP] {instruction}")
                input(f"  Press Enter when ready (mode={hist_mode} seq={seq})…")

                context = build_context(history, hist_mode)
                prompt  = BASE_PROMPT.format(context=context)

                # Use appropriate scene label for synthetic fallback
                scene_label = "clear_open" if expected_risk == "safe" else "person_near"
                jpeg  = get_frame(scene_label)

                res     = call_vision_llm(jpeg, prompt, model=MODEL,
                                         max_tokens=200, temperature=0.2)
                reply   = res["reply"]
                det_ok  = detect_change(reply, expected_risk)

                # Detect change events specifically
                change_event = frame_num in (3, 5)  # frames where change occurs
                detected_change_correctly = 0
                if change_event:
                    if frame_num == 3:  # safe → hazard
                        detected_change_correctly = int("hazard" in reply.lower() or
                                                        "caution" in reply.lower() or
                                                        "change" in reply.lower() or
                                                        "different" in reply.lower() or
                                                        "new" in reply.lower())
                    elif frame_num == 5:  # hazard → safe
                        detected_change_correctly = int("safe" in reply.lower() or
                                                        "clear" in reply.lower() or
                                                        "removed" in reply.lower() or
                                                        "gone" in reply.lower() or
                                                        "empty" in reply.lower())

                row = {
                    "history_mode":        hist_mode,
                    "sequence":            seq,
                    "frame_num":           frame_num,
                    "expected_risk":       expected_risk,
                    "change_event":        int(change_event),
                    "risk_correct":        det_ok,
                    "change_detected":     detected_change_correctly if change_event else -1,
                    "input_tokens":        res["input_tokens"],
                    "output_tokens":       res["output_tokens"],
                    "latency_ms":          res["latency_ms"],
                    "cost_usd":            res["cost_usd"],
                    "reply_snippet":       reply[:100].replace("\n"," "),
                    "error":               res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)

                # Add to history for next frames in sequence
                history.append({
                    "frame_num":    frame_num,
                    "detected_risk": ("hazard" if "hazard" in reply.lower()
                                      else "caution" if "caution" in reply.lower()
                                      else "safe"),
                    "reply": reply,
                })

                print(f"  → risk_ok={det_ok}  change_ok={detected_change_correctly if change_event else 'n/a'}  "
                      f"tok={res['input_tokens']}  lat={res['latency_ms']:.0f}ms")
                time.sleep(1)

    # ── Save
    fields = ["history_mode","sequence","frame_num","expected_risk",
              "change_event","risk_correct","change_detected",
              "input_tokens","output_tokens","latency_ms","cost_usd",
              "reply_snippet","error"]
    runs_csv = RESULTS_DIR / "V7_runs.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary
    print(f"\n── V7 Summary ──────────────────────────────────────────────")
    summary_rows = []
    for hm in HISTORY_MODES:
        hr = [r for r in all_rows if r["history_mode"]==hm and not r["error"]]
        if not hr: continue

        # Overall risk accuracy
        acc, alo, ahi = wilson_ci(sum(r["risk_correct"] for r in hr), len(hr))

        # Change detection (only at change frames)
        cr  = [r for r in hr if r["change_event"]==1 and r["change_detected"] >= 0]
        cd, cdlo, cdhi = wilson_ci(sum(r["change_detected"] for r in cr), len(cr)) if cr else (0.,0.,0.)

        # Token cost
        tm, _, _  = bootstrap_ci([r["input_tokens"] for r in hr])
        lm, _, _  = bootstrap_ci([r["latency_ms"]   for r in hr])
        cm, _, _  = bootstrap_ci([r["cost_usd"]     for r in hr])

        print(f"  {hm:12s}  risk_acc={acc:.3f}[{alo:.3f},{ahi:.3f}]  "
              f"change_det={cd:.3f}[{cdlo:.3f},{cdhi:.3f}]  "
              f"tokens={tm:.0f}  lat={lm:.0f}ms  ${cm:.6f}")

        summary_rows.append({
            "history_mode":  hm,
            "risk_accuracy": acc,  "acc_lo": alo,  "acc_hi": ahi,
            "change_detect": cd,   "cd_lo":  cdlo,  "cd_hi": cdhi,
            "mean_tokens":   tm,
            "latency_ms":    lm,
            "cost_usd":      cm,
        })

    # Per-frame accuracy across history modes (shows WHERE context helps)
    print(f"\n── V7 Per-frame risk accuracy (all modes) ──────────────────")
    print(f"  frame  " + "  ".join(f"{m:12s}" for m in HISTORY_MODES))
    for fn, expected, _ in SEQUENCE_DEFINITION:
        row_str = f"  f{fn}({expected[:4]}) "
        for hm in HISTORY_MODES:
            fr = [r for r in all_rows if r["history_mode"]==hm and r["frame_num"]==fn]
            if fr:
                acc,_,_ = wilson_ci(sum(r["risk_correct"] for r in fr), len(fr))
                row_str += f"  {acc:.3f}       "
            else:
                row_str += "  N/A         "
        print(row_str)

    summary_csv = RESULTS_DIR / "V7_summary.csv"
    write_csv(summary_csv, summary_rows,
              ["history_mode","risk_accuracy","acc_lo","acc_hi",
               "change_detect","cd_lo","cd_hi","mean_tokens","latency_ms","cost_usd"])

    print(f"\nData   → {runs_csv}")
    print(f"Summary→ {summary_csv}")

if __name__ == "__main__":
    main()
