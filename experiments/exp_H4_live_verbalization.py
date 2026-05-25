"""
EXP-H4: Live Decision Verbalization — Real Flight Demo
=======================================================
Architecture (same three-tier as G series):

    Tier 2 (continuous): YOLO-World + CLIP runs on every frame from ESP32-S3 Sense
                         Monitors for emergency conditions in real time.

    Tier 3 (scheduled + event-driven): LLM called in two modes:
        PERIODIC  — every PERIODIC_S seconds (routine situation awareness)
        EMERGENCY — immediately when Tier 2 flags a hazard (clip_risk=hazard
                    OR high-confidence hazard-class detection at close range)

    LLM output: natural language scene description (2-3 sentences, as a co-pilot
                narrating to a pilot who cannot see the screen) followed by
                the recommended pilot action.

    TTS: pyttsx3 (local) → gTTS (cloud) → print fallback.
         Spoken aloud so pilot hears: "I can see a person standing directly
         in front of the drone about half a metre away. The path is completely
         blocked. Recommended action: PITCH_BACK."

Metrics:
    - verbalization_quality : 0–5 rubric (Bootstrap CI)
    - trigger_type          : periodic vs emergency
    - emergency_rate        : fraction of calls that were emergency-triggered (Wilson CI)
    - llm_latency_ms        : LLM call time (Bootstrap CI)
    - total_pipeline_ms     : Tier2 + Tier3 wall time (Bootstrap CI)
    - classification_acc    : detected risk vs ground truth (Wilson CI)

Setup:
    1. Drone flying, ESP32-S3 Sense streaming at http://<IP>/capture
    2. Run script: python3.11 exp_H4_live_verbalization.py
    3. Press Enter to start. Pilot flies through scenarios shown on screen.
    4. Script captures, classifies, speaks — continuously.
    5. Press Ctrl+C to stop and save results.

Scenarios (fly in this order, ~30s each):
    1. Open room — hover in clear space          (truth: safe)
    2. Approach wall — fly to ~25cm from wall    (truth: hazard)
    3. Person near — operator stands ~0.5m front (truth: hazard)
    4. Cluttered — fly over boxes/chairs area    (truth: caution)
    5. Dim light — turn off room lights          (truth: caution)
    6. Blocked lens — partially cover camera     (truth: hazard)
    Repeat 2–3 times for N≥15 trials total.
"""

import os, sys, time, csv, json, pathlib, datetime, threading, urllib.request

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXP_DIR   = pathlib.Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(REPO_ROOT / "Image verbalization experiments"))

import credentials  # noqa: F401 — sets env vars
from verbalization_utils import (
    call_vision_llm, score_verbalization, wilson_ci, bootstrap_ci, write_csv
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_clip, enhanced_yolo_infer
)

# ── Config ────────────────────────────────────────────────────────────────────
ESP32_URL    = os.environ.get("ESP32_URL", "http://10.186.33.138/capture")
PERIODIC_S   = 5.0        # scheduled LLM call every N seconds
COOLDOWN_S   = 2.0        # minimum gap between any two LLM calls
LLM_MODEL    = "gpt4o"    # best from V1; fastest capable model
MAX_TOKENS   = 300
RESULTS_DIR  = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Hazard classes that trigger emergency if detected at close range (<1.5m advisory)
EMERGENCY_YOLO_CLASSES = {"person", "wall", "hand", "fire", "smoke"}
EMERGENCY_DIST_M       = 1.5   # advisory est_dist threshold

# ── Prompt ────────────────────────────────────────────────────────────────────
VERBALIZATION_PROMPT = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "flying at ~1m altitude indoors. You are the AI voice of the drone — speaking "
    "to a pilot who cannot see the screen.\n\n"
    "YOUR PRIMARY INPUT IS THE CAMERA IMAGE. Look at it directly.\n\n"
    "Supplementary sensor data (advisory — you may agree or override):\n"
    "  YOLO-World: {yolo_meta}\n"
    "  CLIP scene: {clip_label} (confidence={clip_conf:.3f}, risk={clip_risk})\n\n"
    "YOLO provides distance estimates and class detections. CLIP labels the overall scene. "
    "Both can miss things. If you see something in the image that the sensors missed, "
    "it still exists — describe it.\n\n"
    "Describe what you see in 2-3 natural sentences as if narrating to the pilot. "
    "Talk about the space, lighting, people, obstacles, and path clarity at ~1m altitude. "
    "Speak from your own visual analysis — do NOT just read out sensor data.\n\n"
    "Then state:\n"
    "Recommended action: HOVER | PITCH_FORWARD | PITCH_BACK | "
    "ROLL_LEFT | ROLL_RIGHT | ASCEND | DESCEND | LAND\n"
    "Risk level: safe | caution | hazard"
)

# ── Camera ────────────────────────────────────────────────────────────────────
def capture_frame() -> bytes | None:
    """Try ESP32 → webcam → return None on failure."""
    try:
        with urllib.request.urlopen(ESP32_URL, timeout=3) as r:
            return r.read()
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes()
    except Exception:
        pass
    return None

# ── Emergency detection ───────────────────────────────────────────────────────
def is_emergency(tier2: dict) -> bool:
    """True if Tier 2 output indicates immediate hazard — triggers unscheduled LLM call."""
    if tier2["clip_risk"] == "hazard":
        return True
    # Parse YOLO meta for hazard classes at close range
    meta = tier2["yolo_meta"].lower()
    for cls in EMERGENCY_YOLO_CLASSES:
        if cls in meta:
            # Check est_dist — advisory only but useful as trigger heuristic
            import re
            pattern = rf"{cls}.*?est_dist~(\d+\.\d+)m"
            m = re.search(pattern, meta)
            if m and float(m.group(1)) < EMERGENCY_DIST_M:
                return True
    return False

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    """Speak text aloud. pyttsx3 → gTTS+playsound → print fallback."""
    print(f"\n🔊 SPEAKING: {text[:120]}{'…' if len(text)>120 else ''}\n")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass
    try:
        from gtts import gTTS
        import tempfile, subprocess
        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            subprocess.Popen(
                ["afplay", f.name],          # macOS
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).wait()
        return
    except Exception:
        pass
    # Final fallback: just print
    print(f"[TTS unavailable] {text}")

# ── Parse LLM reply ───────────────────────────────────────────────────────────
def parse_reply(reply: str) -> tuple[str, str]:
    """Extract (risk_level, recommended_action) from LLM reply."""
    low = reply.lower()
    risk = "unknown"
    for lvl in ("hazard", "caution", "safe"):
        if lvl in low:
            risk = lvl
            break
    action = "HOVER"
    for act in ("PITCH_BACK", "PITCH_FORWARD", "ROLL_LEFT", "ROLL_RIGHT",
                "ASCEND", "DESCEND", "LAND", "HOVER"):
        if act.lower() in low:
            action = act
            break
    return risk, action

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 65)
    print("EXP-H4: Live Decision Verbalization — Real Flight Demo")
    print(f"Model={LLM_MODEL}  Periodic={PERIODIC_S}s  Emergency=auto-trigger")
    print(f"ESP32: {ESP32_URL}")
    print("=" * 65)

    print("\nLoading YOLO-World pipeline…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading CLIP scene screener…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()
    print("\nPipeline ready.\n")

    # Scenario labels — operator annotates after each block
    SCENARIOS = [
        {"label": "open_room",      "truth": "safe"},
        {"label": "wall_close",     "truth": "hazard"},
        {"label": "person_near",    "truth": "hazard"},
        {"label": "cluttered",      "truth": "caution"},
        {"label": "dim_light",      "truth": "caution"},
        {"label": "blocked_lens",   "truth": "hazard"},
    ]

    print("Scenarios (fly each for ~30 seconds):")
    for i, s in enumerate(SCENARIOS, 1):
        print(f"  {i}. {s['label']:20s}  truth={s['truth']}")

    input("\nPress Enter when drone is airborne and ready…")
    print("\nStarting. Ctrl+C to stop and save.\n")

    all_rows   = []
    trial_num  = 0
    last_llm_t = 0.0    # time of last LLM call
    last_emrg  = 0.0    # time of last emergency call (separate cooldown)
    scenario_idx = 0

    speak("Live verbalization active. I will describe the scene every few seconds.")

    try:
        while True:
            # ── Tier 2: capture + YOLO-World + CLIP ──────────────────────────
            t_frame = time.perf_counter()
            jpeg    = capture_frame()
            if jpeg is None:
                print("[WARN] No frame — retrying in 1s")
                time.sleep(1)
                continue

            tier2    = enhanced_yolo_infer(yolo_model, yolo_type,
                                           clip_model, clip_preprocess,
                                           clip_tokenizer, jpeg)
            tier2_ms = round((time.perf_counter() - t_frame) * 1000, 1)

            now      = time.perf_counter()
            emrg     = is_emergency(tier2)
            periodic = (now - last_llm_t) >= PERIODIC_S
            cooldown = (now - last_llm_t) >= COOLDOWN_S

            trigger = None
            if emrg and (now - last_emrg) >= COOLDOWN_S:
                trigger    = "emergency"
                last_emrg  = now
            elif periodic:
                trigger = "periodic"

            if trigger is None or not cooldown:
                time.sleep(0.1)
                continue

            # ── Tier 3: LLM call ──────────────────────────────────────────────
            trial_num  += 1
            last_llm_t  = now
            prompt      = VERBALIZATION_PROMPT.format(
                yolo_meta  = tier2["yolo_meta"],
                clip_label = tier2["clip_label"],
                clip_conf  = tier2["clip_conf"],
                clip_risk  = tier2["clip_risk"],
            )

            print(f"\n── Trial {trial_num}  trigger={trigger}  "
                  f"clip_risk={tier2['clip_risk']}  "
                  f"yolo={tier2['yolo_meta'][:60]} ──")

            t_llm  = time.perf_counter()
            res    = call_vision_llm(jpeg, prompt, model=LLM_MODEL,
                                     max_tokens=MAX_TOKENS, temperature=0.0)
            llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)

            reply        = res["reply"]
            risk, action = parse_reply(reply)
            scores       = score_verbalization(reply, "")  # truth set per scenario below

            # Ask operator to annotate scenario (non-blocking — just record current)
            sc = SCENARIOS[scenario_idx % len(SCENARIOS)]
            truth = sc["truth"]

            correct = int(risk == truth)

            print(f"   REPLY: {reply[:200]}")
            print(f"   risk={risk}  action={action}  correct={correct}  llm={llm_ms}ms")

            # Speak: full description + action
            spoken = reply.strip()
            # Ensure action is spoken clearly at the end
            if action not in spoken.upper():
                spoken += f". Recommended action: {action}."
            speak(spoken)

            total_ms = round(tier2_ms + llm_ms, 1)

            row = {
                "trial":          trial_num,
                "timestamp":      datetime.datetime.now().isoformat(),
                "trigger":        trigger,
                "scenario_label": sc["label"],
                "truth":          truth,
                "detected_risk":  risk,
                "correct":        correct,
                "action":         action,
                "clip_risk":      tier2["clip_risk"],
                "clip_label":     tier2["clip_label"],
                "yolo_meta":      tier2["yolo_meta"][:120],
                "reply_snippet":  reply[:200].replace("\n", " "),
                "quality_score":  scores["quality_score"],
                "word_count":     scores["word_count"],
                "tier2_ms":       tier2_ms,
                "llm_ms":         llm_ms,
                "total_ms":       total_ms,
                "input_tokens":   res["input_tokens"],
                "output_tokens":  res["output_tokens"],
                "cost_usd":       res["cost_usd"],
                "error":          res["error"][:80] if res["error"] else "",
            }
            all_rows.append(row)

            # Advance scenario every 3 trials (operator can override)
            if trial_num % 3 == 0:
                scenario_idx += 1
                next_sc = SCENARIOS[scenario_idx % len(SCENARIOS)]
                msg = f"Moving to scenario: {next_sc['label'].replace('_', ' ')}."
                print(f"\n⟳  {msg}")
                speak(msg)

    except KeyboardInterrupt:
        print("\n\nStopping — saving results…")

    if not all_rows:
        print("No trials recorded.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fields = ["trial","timestamp","trigger","scenario_label","truth","detected_risk",
              "correct","action","clip_risk","clip_label","yolo_meta","reply_snippet",
              "quality_score","word_count","tier2_ms","llm_ms","total_ms",
              "input_tokens","output_tokens","cost_usd","error"]
    runs_csv = RESULTS_DIR / f"H4_live_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary ───────────────────────────────────────────────────────────────
    valid  = [r for r in all_rows if not r["error"]]
    emrg_r = [r for r in valid if r["trigger"] == "emergency"]
    perd_r = [r for r in valid if r["trigger"] == "periodic"]

    acc, alo, ahi = wilson_ci(sum(r["correct"]      for r in valid), len(valid))
    er,  _,   _   = wilson_ci(len(emrg_r), len(valid))
    qm,  _,   _   = bootstrap_ci([r["quality_score"] for r in valid])
    lm,  _,   _   = bootstrap_ci([r["llm_ms"]        for r in valid])
    t2m, _,   _   = bootstrap_ci([r["tier2_ms"]       for r in valid])
    tm,  _,   _   = bootstrap_ci([r["total_ms"]       for r in valid])
    cm,  _,   _   = bootstrap_ci([r["cost_usd"]       for r in valid])

    print(f"\n{'='*60}")
    print(f"EXP-H4 Summary — {len(valid)} trials ({len(emrg_r)} emergency, {len(perd_r)} periodic)")
    print(f"{'='*60}")
    print(f"  Classification accuracy : {acc:.3f}  [{alo:.3f}, {ahi:.3f}]")
    print(f"  Emergency trigger rate  : {er:.3f}")
    print(f"  Verbalization quality   : {qm:.2f}/5")
    print(f"  Tier 2 latency          : {t2m:.0f}ms")
    print(f"  LLM latency             : {lm:.0f}ms")
    print(f"  Total pipeline          : {tm:.0f}ms")
    print(f"  Cost/call               : ${cm:.5f}")

    # Per-scenario breakdown
    print(f"\n  Per-scenario accuracy:")
    for sc in set(r["scenario_label"] for r in valid):
        sr = [r for r in valid if r["scenario_label"] == sc]
        a, _, _ = wilson_ci(sum(r["correct"] for r in sr), len(sr))
        emrg_n  = sum(1 for r in sr if r["trigger"] == "emergency")
        print(f"    {sc:20s}  acc={a:.2f}  n={len(sr)}  emergency={emrg_n}")

    # Save summary
    summary = [{
        "n_trials": len(valid), "n_emergency": len(emrg_r),
        "n_periodic": len(perd_r), "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
        "emergency_rate": er, "quality": qm, "tier2_ms": t2m,
        "llm_ms": lm, "total_ms": tm, "cost_usd": cm,
    }]
    summary_csv = RESULTS_DIR / f"H4_live_summary_{ts}.csv"
    write_csv(summary_csv, summary,
              ["n_trials","n_emergency","n_periodic","accuracy","acc_lo","acc_hi",
               "emergency_rate","quality","tier2_ms","llm_ms","total_ms","cost_usd"])

    print(f"\nRuns    → {runs_csv}")
    print(f"Summary → {summary_csv}")


if __name__ == "__main__":
    main()
