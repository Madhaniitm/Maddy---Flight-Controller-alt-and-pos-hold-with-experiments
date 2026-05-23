"""
EXP-V7: Scene Context History Effect
======================================
Goal:
    Does feeding the LLM prior frame descriptions help it detect scene
    changes? Tests three history modes on a 5-frame sequence with two
    change events.

    History modes:
        stateless : each API call has no prior context
        short     : last 2 frame descriptions prepended
        full      : all prior frame descriptions prepended

    Sequence (5 frames, change at frame 3 and frame 5):
        Frame 1: door_open  (safe)   — baseline clear scene
        Frame 2: door_open  (safe)   — same
        Frame 3: person_near (hazard) ← CHANGE (person enters)
        Frame 4: person_near (hazard) — continues
        Frame 5: door_open  (safe)   ← CHANGE BACK (person leaves)

    Saved frames used: door_open_run{N}_real.jpg and
    person_near_run{N}_real.jpg. Run01 is excluded (buggy frames).
    Sequence→run mapping: [1→02, 2→03, 3→04, 4→05, 5→02]
    run02 is used twice to fill 5 sequences without run01.

    Models:  claude, gpt4o, gpt4o_mini, gemini
    N runs:  5 sequences × 3 modes × 4 models × 5 frames = 300 trials

Output:  results/V7_runs_<timestamp>.csv
         results/V7_summary_<timestamp>.csv
"""

import sys, pathlib, datetime, time

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = pathlib.Path(__file__).parent
EXP_DIR   = REPO_ROOT / "experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR, FRAMES_DIR
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_clip, enhanced_yolo_infer
)

N_SEQUENCES   = 5
HISTORY_MODES = ["stateless", "short", "full"]
MODELS        = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
# run01 excluded (buggy); run02 used twice to fill 5 sequences
SEQ_TO_RUN    = {1: 2, 2: 3, 3: 4, 4: 5, 5: 2}

# Frame 1,2,5 = door_open (safe); Frame 3,4 = person_near (hazard)
SEQUENCE_DEFINITION = [
    (1, "safe",   "door_open"),
    (2, "safe",   "door_open"),
    (3, "hazard", "person_near"),
    (4, "hazard", "person_near"),
    (5, "safe",   "door_open"),
]

BASE_PROMPT = (
    "You are an AI copilot for a small indoor drone flying at ~1m altitude.\n"
    "{context}"
    "YOLO-World detections on current frame: {yolo_meta}\n"
    "CLIP scene label: {clip_label} (conf={clip_conf:.3f}, risk={clip_risk})\n\n"
    "Using the YOLO-World metadata, CLIP label, and the image:\n"
    "Describe what you see. Has the scene changed from before?\n"
    "Classify as: safe | caution | hazard\n"
    "Pilot suggested action: HOVER | PITCH_FORWARD | PITCH_BACK | ROLL_LEFT | ROLL_RIGHT | ASCEND | DESCEND | LAND"
)


def get_frame_for_run(scene_label: str, run: int) -> bytes:
    """Load a specific run's saved frame (run01–run05)."""
    path = FRAMES_DIR / f"{scene_label}_run{run:02d}_real.jpg"
    if path.exists():
        return path.read_bytes()
    # fallback to run03 if specific run missing
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


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(HISTORY_MODES) * N_SEQUENCES * len(SEQUENCE_DEFINITION) * len(MODELS)
    print("=" * 65)
    print("EXP-V7: Scene Context History Effect")
    print(f"Modes={HISTORY_MODES}  Sequences={N_SEQUENCES}  Models={MODELS}")
    print(f"Sequence: safe→safe→hazard→hazard→safe  (door_open / person_near)")
    print(f"Total trials: {total}")
    print("=" * 65)

    print("\nLoading enhanced YOLO tier…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading CLIP scene screener…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows = []

    for hist_mode in HISTORY_MODES:
        print(f"\n{'='*55}")
        print(f"=== History mode: {hist_mode} ===")
        print(f"{'='*55}")

        for seq in range(1, N_SEQUENCES + 1):
            run_n = SEQ_TO_RUN[seq]
            print(f"\n  ── Sequence {seq}/{N_SEQUENCES} (using run{run_n:02d} frames) ──")

            # Per-model history — each model maintains its own context
            histories = {model: [] for model in MODELS}

            for frame_num, expected_risk, scene_label in SEQUENCE_DEFINITION:
                jpeg  = get_frame_for_run(scene_label, run_n)
                tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                            clip_model, clip_preprocess,
                                            clip_tokenizer, jpeg)
                change_event = frame_num in (3, 5)

                print(f"\n    Frame {frame_num}/5  scene={scene_label}  "
                      f"expected={expected_risk}  change={change_event}")

                for model in MODELS:
                    context = build_context(histories[model], hist_mode)
                    prompt  = BASE_PROMPT.format(
                        context    = context,
                        yolo_meta  = tier2["yolo_meta"],
                        clip_label = tier2["clip_label"],
                        clip_conf  = tier2["clip_conf"],
                        clip_risk  = tier2["clip_risk"],
                    )
                    res   = call_vision_llm(jpeg, prompt, model=model,
                                            max_tokens=300, temperature=0.0)
                    reply = res["reply"]
                    det_risk = extract_risk(reply)
                    risk_correct = int(det_risk == expected_risk)

                    # Change detection: did model flag the transition?
                    if change_event:
                        if frame_num == 3:  # safe → hazard
                            change_detected = int(det_risk in ("hazard", "caution"))
                        else:               # hazard → safe (frame 5)
                            change_detected = int(det_risk == "safe")
                    else:
                        change_detected = -1  # not a change frame

                    row = {
                        "history_mode":    hist_mode,
                        "sequence":        seq,
                        "frame_num":       frame_num,
                        "scene_label":     scene_label,
                        "expected_risk":   expected_risk,
                        "model":           model,
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
                    all_rows.append(row)

                    histories[model].append({
                        "frame_num":    frame_num,
                        "detected_risk": det_risk,
                        "reply":        reply,
                    })

                    print(f"      {model:12s}  risk={det_risk:8s}  "
                          f"correct={risk_correct}  "
                          f"{'change_ok='+str(change_detected) if change_event else '':12s}  "
                          f"lat={res['latency_ms']:.0f}ms")

                time.sleep(0.5)

    # ── Save runs CSV
    fields = ["history_mode","sequence","frame_num","scene_label","expected_risk",
              "model","detected_risk","risk_correct","change_event","change_detected",
              "input_tokens","output_tokens","latency_ms","cost_usd","reply_snippet","error"]
    runs_csv = RESULTS_DIR / f"V7_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary by history_mode × model
    print(f"\n── V7 Summary by Mode × Model ──────────────────────────────────")
    print(f"  {'mode':10s}  {'model':12s}  {'risk_acc':>8s}  {'change_det':>10s}  "
          f"{'tokens':>6s}  {'lat_ms':>7s}")
    print("-" * 68)

    summary_rows = []
    for hm in HISTORY_MODES:
        for model in MODELS:
            hr = [r for r in all_rows if r["history_mode"]==hm
                  and r["model"]==model and not r["error"]]
            if not hr:
                continue

            acc, alo, ahi = wilson_ci(sum(r["risk_correct"] for r in hr), len(hr))

            cr = [r for r in hr if r["change_event"]==1 and r["change_detected"] >= 0]
            cd, cdlo, cdhi = (wilson_ci(sum(r["change_detected"] for r in cr), len(cr))
                              if cr else (0., 0., 0.))

            tm, _, _ = bootstrap_ci([r["input_tokens"] for r in hr])
            lm, _, _ = bootstrap_ci([r["latency_ms"]   for r in hr])
            cm, _, _ = bootstrap_ci([r["cost_usd"]     for r in hr])

            print(f"  {hm:10s}  {model:12s}  {acc:.3f}[{alo:.3f},{ahi:.3f}]  "
                  f"{cd:.3f}[{cdlo:.3f},{cdhi:.3f}]  {tm:.0f}  {lm:.0f}ms")

            summary_rows.append({
                "history_mode": hm, "model": model, "n_trials": len(hr),
                "risk_accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                "change_detect": cd, "cd_lo": cdlo, "cd_hi": cdhi,
                "mean_input_tokens": tm, "latency_ms": lm, "cost_usd": cm,
            })

    # ── Per-frame accuracy across modes (all models avg)
    print(f"\n── V7 Per-frame risk accuracy (all models avg) ─────────────────")
    print(f"  {'frame':8s}  {'stateless':>10s}  {'short':>10s}  {'full':>10s}")
    for fn, expected, scene in SEQUENCE_DEFINITION:
        row_str = f"  f{fn}({expected[:4]}) "
        for hm in HISTORY_MODES:
            fr = [r for r in all_rows if r["history_mode"]==hm and r["frame_num"]==fn]
            if fr:
                acc, _, _ = wilson_ci(sum(r["risk_correct"] for r in fr), len(fr))
                row_str += f"  {acc:.3f}      "
            else:
                row_str += "  N/A        "
        print(row_str)

    # ── Input token growth (shows context cost)
    print(f"\n── Input token growth by mode (all models avg) ─────────────────")
    print(f"  {'mode':10s}  " + "  ".join(f"f{fn}" for fn, _, _ in SEQUENCE_DEFINITION))
    for hm in HISTORY_MODES:
        tok_str = f"  {hm:10s}  "
        for fn, _, _ in SEQUENCE_DEFINITION:
            fr = [r for r in all_rows if r["history_mode"]==hm and r["frame_num"]==fn]
            if fr:
                tm, _, _ = bootstrap_ci([r["input_tokens"] for r in fr])
                tok_str += f"{tm:.0f}   "
        print(tok_str)

    summary_csv = RESULTS_DIR / f"V7_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["history_mode","model","n_trials","risk_accuracy","acc_lo","acc_hi",
               "change_detect","cd_lo","cd_hi","mean_input_tokens","latency_ms","cost_usd"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V7] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
