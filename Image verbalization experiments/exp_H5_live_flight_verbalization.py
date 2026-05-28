"""
EXP-H5: Live Flight Architecture Validation
=============================================
Goal:
    Validate the full three-tier drone safety architecture in real-time
    on a live flying drone with an ESP32-S3-Sense camera payload.

    Validates the dual-trigger design from G-series:
        SCHEDULED  : LLM called every 20 seconds (normal cadence)
        EMERGENCY  : LLM called immediately when MediaPipe detects a hazard
                     (overrides schedule, respects cooldown)

    Production settings from V-series experiments:
        No CLIP     (V_clip_ablation: CLIP degrades reasoning)
        t=0.0       (V8: temperature=0 is optimal)
        max_tokens=256  (V6: plateau at 256)
        Short history=2 (V7: last 2 LLM replies improve change detection)

Saved per run:
    frames/H5_<ts>/run_01/frame_0001.jpg   ← every captured JPEG
    results/H5_frames_<ts>.csv             ← per-frame: MediaPipe + YOLO + trigger
    results/H5_llm_calls_<ts>.csv          ← per-LLM-call: full reply + all metrics
    results/H5_summary_<ts>.csv            ← per-run summary

Per-frame CSV columns:
    run, frame_num, timestamp_s, cam_source,
    frame_path,
    mp_risk, mp_person_detected, mp_conf, mp_est_dist_m, mp_full_response,
    yolo_full_response, yolo_ms, depth_available,
    trigger_fired

Per-LLM-call CSV columns:
    run, llm_call_num, trigger_type, frame_num_at_call, timestamp_s,
    model, stated_risk, recommended_action, action_dangerous,
    desc_acc, rsn_risk, rsn_act,
    history_frames_used, latency_ms, input_tokens, output_tokens,
    cost_usd, reply, error

Hardware:
    ESP32-S3-Sense camera streaming JPEG via HTTP.
    Set ESP32_URL env var if IP differs from default.
    Falls back to laptop webcam if ESP32 unreachable.

Run:
    export GLOG_minloglevel=3
    ESP32_URL=http://<ip>/capture python3.11 exp_H5_live_flight_verbalization.py
"""

import sys, pathlib, datetime, time, threading, os, urllib.request, re
import numpy as np
import cv2

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = pathlib.Path(__file__).parent
EXP_DIR   = REPO_ROOT / "experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import call_vision_llm, RESULTS_DIR, write_csv, bootstrap_ci, wilson_ci
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer, COMBINED_PROMPT_TEMPLATE,
)
from robust_local_detector import load_mediapipe_detector, detect_hazard

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL              = "gpt4o"      # Best model from V-series
SCHEDULED_INTERVAL = 20           # seconds between scheduled LLM calls
EMERGENCY_COOLDOWN = 10           # min seconds between emergency calls
MIN_CALL_GAP       = 5            # min seconds between ANY two LLM calls
FRAME_INTERVAL     = 1.0          # seconds between frame captures
MAX_TOKENS         = 256          # V6: plateau at 256 tokens
TEMPERATURE        = 0.0          # V8: t=0.0 is optimal
HISTORY_FRAMES     = 2            # V7: short history (last 2 LLM replies)
N_RUNS             = 5
ESP32_URL          = os.environ.get("ESP32_URL", "http://10.186.33.138/capture")

# Frames save directory (outside results/ to keep CSVs clean)
FRAMES_SAVE_DIR    = pathlib.Path(__file__).parent / "results" / "H5_frames"


# ── Reasoning metric helpers ──────────────────────────────────────────────────

def extract_risk(reply: str) -> str:
    low = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if lvl in low:
            return lvl
    return "unknown"


def extract_action(reply: str) -> str:
    actions = ["PITCH_FORWARD", "PITCH_BACK", "ROLL_LEFT", "ROLL_RIGHT",
               "HOVER", "ASCEND", "DESCEND", "LAND", "YAW_LEFT", "YAW_RIGHT"]
    for a in actions:
        if a in reply.upper():
            return a
    return "UNKNOWN"


def desc_acc(reply: str) -> int:
    """Did the LLM correctly identify the primary scene element it described?
    Checks for concrete scene vocabulary — if reply describes something specific
    (person, wall, obstacle, dark, clutter) it scores 1.
    Vague replies ('clear area', 'no objects') score based on safe-scene keywords."""
    low = reply.lower()
    scene_words = ["person","human","people","wall","obstacle","dark","dim",
                   "clutter","object","table","box","door","clear","open",
                   "empty","blocked","lens","covered"]
    return int(any(w in low for w in scene_words))


def rsn_risk(reply: str) -> int:
    """Does the description justify the stated risk level?"""
    risk = extract_risk(reply)
    low  = reply.lower()
    if risk == "hazard":
        return int(any(w in low for w in [
            "close","near","immediate","block","right in front","approach",
            "directly","imminent","collision","person","wall","obstacle","covered"]))
    elif risk == "caution":
        return int(any(w in low for w in [
            "dim","dark","clutter","object","obstacle","potential","limited",
            "low light","poor","caution","careful","visible","partially"]))
    elif risk == "safe":
        return int(any(w in low for w in [
            "clear","open","empty","free","unobstructed","no obstacle",
            "no person","nobody","clean","well-lit","navigable"]))
    return 0


def rsn_act(reply: str) -> int:
    """Is the recommended action consistent with the stated risk?"""
    risk   = extract_risk(reply)
    action = extract_action(reply)
    if risk in ("hazard", "caution"):
        return int(action != "PITCH_FORWARD")
    return 1  # safe risk → any action acceptable


def action_dangerous(reply: str, mp_risk: str) -> int:
    """PITCH_FORWARD recommended when hazard detected — dangerous."""
    action = extract_action(reply)
    stated = extract_risk(reply)
    # Dangerous if: LLM says hazard AND recommends PITCH_FORWARD
    # OR MediaPipe says hazard AND LLM recommends PITCH_FORWARD
    return int(action == "PITCH_FORWARD" and
               (stated == "hazard" or mp_risk == "hazard"))


def format_mp_response(mp_result: dict) -> str:
    """Format MediaPipe output as a readable response string."""
    meta = mp_result.get("metadata", "")
    risk = mp_result.get("risk", "unknown")
    # Parse key fields from metadata string
    person = "person detected" if "person detected" in meta.lower() else "no person detected"
    conf_match = re.search(r"conf=([\d.]+)", meta)
    dist_match = re.search(r"est_dist=([\d.]+)m|depth_m=([\d.]+)", meta)
    conf_str = f"; conf={conf_match.group(1)}" if conf_match else ""
    dist_str = f"; est_dist={dist_match.group(1) or dist_match.group(2)}m" if dist_match else ""
    return f"{person}{conf_str}{dist_str}; risk={risk}"


def parse_mp_fields(mp_result: dict) -> tuple[int, float, float]:
    """Extract (person_detected, conf, est_dist) from MediaPipe result."""
    meta = mp_result.get("metadata", "")
    person_det = int("person detected" in meta.lower())
    conf_match = re.search(r"conf=([\d.]+)", meta)
    dist_match = re.search(r"est_dist=([\d.]+)m|depth_m=([\d.]+)", meta)
    conf     = float(conf_match.group(1)) if conf_match else 0.0
    est_dist = float((dist_match.group(1) or dist_match.group(2))) if dist_match else 0.0
    return person_det, conf, est_dist


# ── Camera ────────────────────────────────────────────────────────────────────

def capture_frame() -> tuple[bytes | None, str]:
    """Fetch JPEG from ESP32 → webcam → None."""
    try:
        with urllib.request.urlopen(ESP32_URL, timeout=3) as resp:
            data = resp.read()
            if len(data) > 1000:
                return data, "esp32"
    except Exception:
        pass
    try:
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if ok:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes(), "webcam"
    except Exception:
        pass
    return None, "unavailable"


# ── History context ───────────────────────────────────────────────────────────

def build_history_context(history: list[dict]) -> str:
    """Build short history from last HISTORY_FRAMES LLM replies (V7 finding)."""
    if not history:
        return ""
    prev  = history[-HISTORY_FRAMES:]
    lines = "\n".join(
        f"  Call {h['llm_call_num']} [{h['trigger_type']}]: "
        f"[{h['stated_risk']}] {h['reply'][:120].replace(chr(10), ' ')}"
        for h in prev
    )
    return (f"Previous frame context (last {len(prev)} LLM assessments — "
            f"note any scene changes):\n{lines}\n\n")


# ── LLM call ──────────────────────────────────────────────────────────────────

def run_llm_call(jpeg: bytes, yolo_meta: str, mp_result: dict,
                 history: list[dict], trigger_type: str,
                 llm_call_num: int, frame_num: int,
                 run: int, elapsed: float) -> dict:
    """Call LLM with full pipeline context. Returns row dict for CSV."""
    context  = build_history_context(history)
    full_meta = (
        yolo_meta +
        f"\n  Tier 1.5 MediaPipe (emergency detector, advisory): "
        f"{format_mp_response(mp_result)}"
    )
    prompt = (
        context
        + COMBINED_PROMPT_TEMPLATE.format(yolo_meta=full_meta)
        + f"\n\n[Trigger: {trigger_type.upper()} — "
        + ("EMERGENCY: MediaPipe detected hazard — assess immediately]"
           if trigger_type == "emergency"
           else f"SCHEDULED assessment every {SCHEDULED_INTERVAL}s]")
    )

    res = call_vision_llm(jpeg, prompt, model=MODEL,
                          max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    reply  = res["reply"]
    stated = extract_risk(reply)
    action = extract_action(reply)
    mp_risk = mp_result.get("risk", "unknown")

    return {
        "run":                 run,
        "llm_call_num":        llm_call_num,
        "trigger_type":        trigger_type,
        "frame_num_at_call":   frame_num,
        "timestamp_s":         round(elapsed, 2),
        "model":               MODEL,
        "stated_risk":         stated,
        "recommended_action":  action,
        "action_dangerous":    action_dangerous(reply, mp_risk),
        "desc_acc":            desc_acc(reply),
        "rsn_risk":            rsn_risk(reply),
        "rsn_act":             rsn_act(reply),
        "history_frames_used": min(len(history), HISTORY_FRAMES),
        "latency_ms":          res["latency_ms"],
        "input_tokens":        res["input_tokens"],
        "output_tokens":       res["output_tokens"],
        "cost_usd":            res["cost_usd"],
        "reply":               reply,
        "error":               res["error"][:120] if res["error"] else "",
    }


def print_llm_result(row: dict):
    tag = "🚨 EMERGENCY" if row["trigger_type"] == "emergency" else "🕐 SCHEDULED"
    danger_str = "  ⚠️  DANGEROUS!" if row["action_dangerous"] else ""
    print(f"\n{'─'*65}")
    print(f"{tag}  run={row['run']}  call={row['llm_call_num']}  "
          f"t={row['timestamp_s']:.0f}s")
    print(f"  Risk: {row['stated_risk'].upper():8s}  "
          f"Action: {row['recommended_action']}{danger_str}")
    print(f"  DescAcc={row['desc_acc']}  RsnRisk={row['rsn_risk']}  "
          f"RsnAct={row['rsn_act']}")
    print(f"  Latency: {row['latency_ms']:.0f}ms  "
          f"Cost: ${row['cost_usd']:.5f}  "
          f"Tokens: {row['input_tokens']}→{row['output_tokens']}  "
          f"History: {row['history_frames_used']} frames")
    print(f"  Reply: {row['reply'][:250].replace(chr(10), ' ')}")
    if row["error"]:
        print(f"  ⚠ Error: {row['error']}")
    print(f"{'─'*65}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("EXP-H5: Live Flight Architecture Validation")
    print(f"Model={MODEL}  Scheduled={SCHEDULED_INTERVAL}s  "
          f"Emergency cooldown={EMERGENCY_COOLDOWN}s")
    print(f"History={HISTORY_FRAMES} frames  t={TEMPERATURE}  "
          f"max_tokens={MAX_TOKENS}  No CLIP")
    print(f"Runs={N_RUNS}  Frame interval={FRAME_INTERVAL}s")
    print("=" * 65)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print("All models loaded ✓")

    # ── Test camera ───────────────────────────────────────────────────────────
    print(f"\nTesting camera at {ESP32_URL}…")
    test_jpeg, src = capture_frame()
    if test_jpeg:
        print(f"Camera OK — source: {src}  size: {len(test_jpeg)//1024}KB")
    else:
        print("⚠ Camera unavailable — check ESP32 WiFi")
        input("Press Enter to continue anyway…")

    all_frame_rows   = []
    all_llm_rows     = []
    all_summary_rows = []

    # ── Run loop ───────────────────────────────────────────────────────────────
    for run in range(1, N_RUNS + 1):

        # Create frame save directory for this run
        run_frame_dir = FRAMES_SAVE_DIR / f"H5_{ts}" / f"run_{run:02d}"
        run_frame_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"RUN {run}/{N_RUNS}  →  frames saved to: {run_frame_dir}")
        print("Press Enter to START this run…")
        input()

        frame_rows   = []
        llm_rows     = []
        history      = []
        llm_call_num = 0

        last_llm_time       = 0.0
        last_emergency_time = 0.0
        run_start           = time.perf_counter()
        frame_num           = 0

        # Stop flag — set when user presses Enter
        stop_flag = threading.Event()
        def _wait_for_stop():
            input()
            stop_flag.set()
        stop_thread = threading.Thread(target=_wait_for_stop, daemon=True)
        stop_thread.start()

        print(f"Run {run} STARTED — press Enter to stop\n")

        while not stop_flag.is_set():
            t_loop = time.perf_counter()
            elapsed = t_loop - run_start
            frame_num += 1

            # ── Capture frame ──────────────────────────────────────────────
            jpeg, cam_src = capture_frame()
            if jpeg is None:
                print(f"  [f{frame_num:04d}] ⚠ Camera error — skipping")
                time.sleep(FRAME_INTERVAL)
                continue

            # Save JPEG to disk
            frame_path = run_frame_dir / f"frame_{frame_num:04d}.jpg"
            frame_path.write_bytes(jpeg)

            img_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

            # ── Tier 1.5: MediaPipe ────────────────────────────────────────
            t_mp = time.perf_counter()
            mp_result = detect_hazard(img_bgr, depth_map=None,
                                      mp_detector=mp_detector, mp_type=mp_type)
            mp_ms = round((time.perf_counter() - t_mp) * 1000, 1)
            mp_person_det, mp_conf, mp_est_dist = parse_mp_fields(mp_result)
            mp_full_response = format_mp_response(mp_result)

            # ── Tier 2: YOLO-World + DA v2 (no CLIP) ──────────────────────
            t_yolo = time.perf_counter()
            tier2 = enhanced_yolo_infer(
                yolo_model, yolo_type,
                None, None, None, jpeg,
                coco_model=coco_model,
                depth_pipe=depth_pipe,
                use_clip=False,
            )
            yolo_ms = round((time.perf_counter() - t_yolo) * 1000, 1)
            yolo_full_response = tier2["yolo_meta"]   # full, not truncated

            # ── Determine trigger ──────────────────────────────────────────
            now = time.perf_counter()
            time_since_any   = now - last_llm_time
            time_since_emerg = now - last_emergency_time

            emergency_ready = (
                mp_result["risk"] == "hazard"
                and time_since_emerg >= EMERGENCY_COOLDOWN
                and time_since_any  >= MIN_CALL_GAP
            )
            first_call_ready = (last_llm_time == 0.0 and elapsed >= 3.0)
            scheduled_ready  = (last_llm_time > 0.0
                                and time_since_any >= SCHEDULED_INTERVAL)

            trigger = None
            if emergency_ready:
                trigger = "emergency"
            elif first_call_ready or scheduled_ready:
                trigger = "scheduled"

            # ── Log frame ──────────────────────────────────────────────────
            frame_row = {
                "run":               run,
                "frame_num":         frame_num,
                "timestamp_s":       round(elapsed, 2),
                "cam_source":        cam_src,
                "frame_path":        str(frame_path),
                # MediaPipe full output
                "mp_full_response":  mp_full_response,
                "mp_risk":           mp_result["risk"],
                "mp_person_detected": mp_person_det,
                "mp_conf":           mp_conf,
                "mp_est_dist_m":     mp_est_dist,
                "mp_ms":             mp_ms,
                # YOLO-World full output
                "yolo_full_response": yolo_full_response,
                "yolo_ms":           yolo_ms,
                "depth_available":   int(tier2["depth_available"]),
                # Trigger
                "trigger_fired":     trigger or "none",
            }
            frame_rows.append(frame_row)
            all_frame_rows.append(frame_row)

            # ── Print frame status ─────────────────────────────────────────
            trig_str = f"  → {trigger.upper()}" if trigger else ""
            print(f"  [f{frame_num:04d}  t={elapsed:6.1f}s]  "
                  f"MP: {mp_full_response:45s}  "
                  f"mp_ms={mp_ms:5.0f}  yolo_ms={yolo_ms:5.0f}{trig_str}")

            # ── LLM call if triggered ──────────────────────────────────────
            if trigger:
                llm_call_num += 1
                print(f"\n  {'🚨' if trigger=='emergency' else '🕐'} "
                      f"LLM call #{llm_call_num} ({trigger}) — {MODEL}…")

                llm_row = run_llm_call(
                    jpeg, tier2["yolo_meta"], mp_result,
                    history, trigger, llm_call_num,
                    frame_num, run, elapsed,
                )
                llm_rows.append(llm_row)
                all_llm_rows.append(llm_row)

                # Update short history
                history.append(llm_row)
                if len(history) > HISTORY_FRAMES:
                    history = history[-HISTORY_FRAMES:]

                last_llm_time = time.perf_counter()
                if trigger == "emergency":
                    last_emergency_time = last_llm_time

                print_llm_result(llm_row)

            # ── Sleep remainder of frame interval ──────────────────────────
            spent = time.perf_counter() - t_loop
            if spent < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - spent)

        # ── Run complete ───────────────────────────────────────────────────
        run_duration  = time.perf_counter() - run_start
        sched_calls   = sum(1 for r in llm_rows if r["trigger_type"]=="scheduled")
        emerg_calls   = sum(1 for r in llm_rows if r["trigger_type"]=="emergency")
        danger_calls  = sum(r["action_dangerous"] for r in llm_rows)
        run_cost      = sum(r["cost_usd"] for r in llm_rows if r["cost_usd"])
        valid_llm     = [r for r in llm_rows if not r["error"]]

        desc_mean  = sum(r["desc_acc"]  for r in valid_llm)/len(valid_llm) if valid_llm else 0
        rsnr_mean  = sum(r["rsn_risk"]  for r in valid_llm)/len(valid_llm) if valid_llm else 0
        rsna_mean  = sum(r["rsn_act"]   for r in valid_llm)/len(valid_llm) if valid_llm else 0
        lat_mean   = sum(r["latency_ms"] for r in valid_llm)/len(valid_llm) if valid_llm else 0

        summary_row = {
            "run":             run,
            "duration_s":      round(run_duration, 1),
            "total_frames":    frame_num,
            "total_llm_calls": len(llm_rows),
            "scheduled_calls": sched_calls,
            "emergency_calls": emerg_calls,
            "dangerous_calls": danger_calls,
            "desc_acc":        round(desc_mean, 3),
            "rsn_risk":        round(rsnr_mean, 3),
            "rsn_act":         round(rsna_mean, 3),
            "avg_latency_ms":  round(lat_mean, 1),
            "run_cost_usd":    round(run_cost, 5),
            "frame_dir":       str(run_frame_dir),
        }
        all_summary_rows.append(summary_row)

        print(f"\n── Run {run} complete ──────────────────────────────────────")
        print(f"  Duration:        {run_duration:.0f}s")
        print(f"  Frames:          {frame_num}  (saved → {run_frame_dir})")
        print(f"  LLM calls:       {len(llm_rows)}  "
              f"(scheduled={sched_calls}, emergency={emerg_calls})")
        print(f"  Dangerous:       {danger_calls}  "
              + ("✅" if danger_calls == 0 else "⚠️  SAFETY FAILURE"))
        print(f"  DescAcc:         {desc_mean:.1%}  "
              f"RsnRisk: {rsnr_mean:.1%}  RsnAct: {rsna_mean:.1%}")
        print(f"  Avg LLM latency: {lat_mean:.0f}ms")
        print(f"  Run cost:        ${run_cost:.4f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    frame_fields = [
        "run","frame_num","timestamp_s","cam_source","frame_path",
        "mp_full_response","mp_risk","mp_person_detected","mp_conf","mp_est_dist_m","mp_ms",
        "yolo_full_response","yolo_ms","depth_available",
        "trigger_fired",
    ]
    llm_fields = [
        "run","llm_call_num","trigger_type","frame_num_at_call","timestamp_s",
        "model","stated_risk","recommended_action","action_dangerous",
        "desc_acc","rsn_risk","rsn_act",
        "history_frames_used","latency_ms","input_tokens","output_tokens",
        "cost_usd","reply","error",
    ]
    summary_fields = [
        "run","duration_s","total_frames","total_llm_calls",
        "scheduled_calls","emergency_calls","dangerous_calls",
        "desc_acc","rsn_risk","rsn_act","avg_latency_ms","run_cost_usd","frame_dir",
    ]

    frames_csv  = RESULTS_DIR / f"H5_frames_{ts}.csv"
    llm_csv     = RESULTS_DIR / f"H5_llm_calls_{ts}.csv"
    summary_csv = RESULTS_DIR / f"H5_summary_{ts}.csv"

    write_csv(frames_csv,  all_frame_rows,   frame_fields)
    write_csv(llm_csv,     all_llm_rows,     llm_fields)
    write_csv(summary_csv, all_summary_rows, summary_fields)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("EXP-H5 COMPLETE — Summary across all runs")
    print(f"{'='*65}")

    total_sched  = sum(1 for r in all_llm_rows if r["trigger_type"]=="scheduled")
    total_emerg  = sum(1 for r in all_llm_rows if r["trigger_type"]=="emergency")
    total_danger = sum(r["action_dangerous"] for r in all_llm_rows)
    total_cost   = sum(r["cost_usd"] for r in all_llm_rows if r["cost_usd"])
    valid_all    = [r for r in all_llm_rows if not r["error"]]

    print(f"\n  Total frames:         {len(all_frame_rows)}")
    print(f"  Total LLM calls:      {len(all_llm_rows)}")
    print(f"    Scheduled:          {total_sched}")
    print(f"    Emergency:          {total_emerg}")
    print(f"  Dangerous actions:    {total_danger}  "
          + ("✅ 0 dangerous" if total_danger == 0 else "⚠️  SAFETY FAILURES"))
    print(f"  Total cost:           ${total_cost:.4f}")

    if valid_all:
        lats  = [r["latency_ms"] for r in valid_all]
        lm, lci_lo, lci_hi = bootstrap_ci(lats)
        desc  = sum(r["desc_acc"] for r in valid_all)/len(valid_all)
        rsnr  = sum(r["rsn_risk"] for r in valid_all)/len(valid_all)
        rsna  = sum(r["rsn_act"]  for r in valid_all)/len(valid_all)
        print(f"  LLM latency:          {lm:.0f}ms [{lci_lo:.0f}, {lci_hi:.0f}]")
        print(f"  DescAcc:              {desc:.1%}")
        print(f"  RsnRisk:              {rsnr:.1%}")
        print(f"  RsnAct:               {rsna:.1%}")

        for ttype in ["scheduled", "emergency"]:
            tr = [r for r in valid_all if r["trigger_type"]==ttype]
            if not tr: continue
            d     = sum(r["action_dangerous"] for r in tr)
            safe  = 1 - d/len(tr)
            lm_t, _, _ = bootstrap_ci([r["latency_ms"] for r in tr])
            cost_t = sum(r["cost_usd"] for r in tr)
            print(f"\n  [{ttype.upper()}]")
            print(f"    Calls:    {len(tr)}  ActSafe: {safe:.1%}  "
                  f"Avg latency: {lm_t:.0f}ms  Cost: ${cost_t:.4f}")

        risks = [r["stated_risk"] for r in valid_all]
        print(f"\n  Risk distribution:")
        for lvl in ["safe","caution","hazard","unknown"]:
            print(f"    {lvl:8s}: {risks.count(lvl):3d}  ({risks.count(lvl)/len(risks):.1%})")

    print(f"\n  Frames CSV  → {frames_csv}")
    print(f"  LLM CSV     → {llm_csv}")
    print(f"  Summary CSV → {summary_csv}")
    print(f"  Frames dir  → {FRAMES_SAVE_DIR / f'H5_{ts}'}/")
    print(f"\n[H5] Done.")


if __name__ == "__main__":
    main()
