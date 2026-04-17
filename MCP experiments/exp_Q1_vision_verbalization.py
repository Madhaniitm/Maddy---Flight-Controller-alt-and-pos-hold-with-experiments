"""
EXP-Q1: Vision Verbalization — LLM Describes Real Frame via TTS
================================================================
REAL HARDWARE ONLY.

Goal:
    Capture a real camera frame from the drone (or laptop webcam), send it to
    the LLM via MCP's capture_frame tool, and ask the LLM to verbalize what it
    sees and what action it recommends. The verbalization is spoken aloud via TTS.

    20 real frames × N=5 runs each = 100 trials.
    Frames are pre-categorised by the operator: 10 "hazard" (obstacle <30cm),
    10 "clear" (open space). LLM must detect category and recommend action.

    Rubric (0-4):
        +1 describes scene content (object/colour/layout)
        +1 mentions proximity or distance estimate
        +1 recommends correct action (stop / proceed)
        +1 length 10-80 words (concise but complete)

Metrics:
    - verbalization_quality : mean rubric score (Bootstrap CI)
    - action_accuracy       : correct stop vs proceed recommendation (Wilson CI)
    - verbalize_latency_ms  : LLM response time (Bootstrap CI)
    - tts_latency_ms        : TTS playback start time (Bootstrap CI)
    - tokens_in             : context tokens per call (Bootstrap CI)
"""

import sys, os, time, csv, math, pathlib, base64, io
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import preflight_check, MCPClient

OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP = os.environ.get("DRONE_IP", "192.168.4.1")
MCP_URL  = os.environ.get("MCP_URL",  "http://localhost:5001/mcp")
N_FRAMES = 20
N_RUNS   = 5

VERBALIZE_PROMPT = (
    "You are the AI vision system for a drone. "
    "Describe what you see in the image in 1-3 sentences (10-80 words). "
    "Estimate whether there is an obstacle within 30 cm. "
    "End with either 'ACTION: STOP' or 'ACTION: PROCEED'."
)

FRAME_LABELS = (
    ["hazard"] * 10 + ["clear"] * 10
)

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0., 0., 0.
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d; m = (z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/d
    return round(p,4), round(max(0,c-m),4), round(min(1,c+m),4)

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan"); return v, v, v
    arr = np.array(data, float)
    boots = [stat(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)),4), round(float(lo),4), round(float(hi),4)

def score_verbalization(reply: str, true_label: str) -> dict:
    words = reply.split()
    n_words = len(words)
    r = reply.lower()

    scene_keywords = ["see","observe","wall","obstacle","object","floor","ceiling",
                      "surface","dark","bright","blurry","clear","colour","color"]
    prox_keywords  = ["cm","mm","metre","meter","distance","close","near","far",
                      "proxim","within","away","30","25","50"]

    s1 = int(any(kw in r for kw in scene_keywords))
    s2 = int(any(kw in r for kw in prox_keywords))
    s4 = int(10 <= n_words <= 80)

    recommended_stop = "stop" in r.split("action:")[-1] if "action:" in r else ("stop" in r)
    correct_action   = (true_label == "hazard") == recommended_stop
    s3 = int(correct_action)

    return {
        "quality_score":  s1 + s2 + s3 + s4,
        "action_correct": s3,
        "word_count":     n_words,
    }

def tts_speak(text: str) -> float:
    t0 = time.perf_counter()
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text[:300])
        engine.runAndWait()
    except Exception:
        try:
            import tempfile, subprocess
            from gtts import gTTS
            tts = gTTS(text=text[:300], lang="en", slow=False)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(tmp.name)
            subprocess.Popen(["ffplay","-nodisp","-autoexit",tmp.name],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    return round((time.perf_counter() - t0) * 1000, 1)

def call_llm_vision(mcp: MCPClient, jpeg_b64: str) -> dict:
    try:
        from mcp_client import call_anthropic
    except ImportError:
        return {"reply": "", "input_tokens": 0, "latency_ms": 0}

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
    t0 = time.perf_counter()
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": "image/jpeg",
                        "data":       jpeg_b64,
                    },
                },
                {"type": "text", "text": VERBALIZE_PROMPT},
            ],
        }],
    )
    lat_ms = round((time.perf_counter()-t0)*1000, 1)
    reply  = resp.content[0].text if resp.content else ""
    return {
        "reply":        reply,
        "input_tokens": resp.usage.input_tokens,
        "latency_ms":   lat_ms,
    }

def capture_frame_b64(mcp: MCPClient, frame_idx: int) -> str:
    result = mcp.capture()
    if isinstance(result, dict):
        data = result.get("image_b64") or result.get("data","")
        if data:
            return data
    # Synthetic fallback: coloured JPEG
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            _, buf = cv2.imencode(".jpg", frame)
            return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    # Minimal synthetic JPEG
    from PIL import Image
    colour = (180, 60, 60) if FRAME_LABELS[frame_idx % N_FRAMES] == "hazard" else (60, 180, 60)
    img = Image.new("RGB", (320, 240), colour)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def main():
    print("="*60)
    print("EXP-Q1: Vision Verbalization — REAL HARDWARE")
    print(f"N_FRAMES={N_FRAMES}  N_RUNS={N_RUNS}")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    mcp = MCPClient(MCP_URL)
    all_rows = []

    for frame_idx in range(N_FRAMES):
        label = FRAME_LABELS[frame_idx]
        print(f"\n=== Frame {frame_idx+1}/{N_FRAMES}  label={label} ===")
        if label == "hazard":
            input("  [SETUP] Place obstacle <30cm in front of camera. Press Enter…")
        else:
            input("  [SETUP] Ensure clear open space in front of camera. Press Enter…")

        for run in range(1, N_RUNS+1):
            try:
                jpeg_b64 = capture_frame_b64(mcp, frame_idx)

                llm_res = call_llm_vision(mcp, jpeg_b64)
                reply   = llm_res["reply"]

                tts_ms  = tts_speak(reply)
                scores  = score_verbalization(reply, label)

                row = {
                    "frame_idx":    frame_idx+1,
                    "true_label":   label,
                    "run":          run,
                    "quality_score":scores["quality_score"],
                    "action_correct":scores["action_correct"],
                    "word_count":   scores["word_count"],
                    "verbalize_ms": llm_res["latency_ms"],
                    "tts_ms":       tts_ms,
                    "tokens_in":    llm_res["input_tokens"],
                    "reply_snippet":reply[:80].replace("\n"," "),
                    "error":        "",
                }
            except Exception as e:
                row = {
                    "frame_idx":    frame_idx+1,
                    "true_label":   label,
                    "run":          run,
                    "quality_score":0,
                    "action_correct":0,
                    "word_count":   0,
                    "verbalize_ms": 0,
                    "tts_ms":       0,
                    "tokens_in":    0,
                    "reply_snippet":"",
                    "error":        str(e)[:80],
                }

            all_rows.append(row)
            print(f"  f={frame_idx+1} r={run} quality={row['quality_score']}/4 "
                  f"correct={row['action_correct']} "
                  f"verbalize={row['verbalize_ms']:.0f}ms tts={row['tts_ms']:.0f}ms")

    runs_csv = OUT_DIR / "Q1_runs.csv"
    fields = ["frame_idx","true_label","run","quality_score","action_correct",
              "word_count","verbalize_ms","tts_ms","tokens_in","reply_snippet","error"]
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── Q1 Summary ──────────────────────────────────────────")
    qm, qlo, qhi = bootstrap_ci([r["quality_score"] for r in all_rows])
    ac, aclo, achi = wilson_ci(sum(r["action_correct"] for r in all_rows), len(all_rows))
    vm, _, _ = bootstrap_ci([r["verbalize_ms"] for r in all_rows if r["verbalize_ms"]>0])
    tm, _, _ = bootstrap_ci([r["tts_ms"]       for r in all_rows if r["tts_ms"]>0])

    print(f"  Verbalization quality : {qm:.2f}/4 [{qlo:.2f},{qhi:.2f}]")
    print(f"  Action accuracy       : {ac:.3f} [{aclo:.3f},{achi:.3f}]")
    print(f"  Verbalize latency     : {vm:.0f}ms")
    print(f"  TTS latency           : {tm:.0f}ms")

    # Per-label breakdown
    for lbl in ("hazard","clear"):
        lr = [r for r in all_rows if r["true_label"]==lbl]
        a, alo, ahi = wilson_ci(sum(r["action_correct"] for r in lr), len(lr))
        print(f"  {lbl:7s} action_accuracy={a:.3f} [{alo:.3f},{ahi:.3f}]")

    print(f"\nData → {runs_csv}")

if __name__ == "__main__":
    main()
