"""
Image Verbalization Server
===========================
Flask backend for the ESP32-S3 Sense camera verbalization system.

Architecture
------------
  ESP32 (192.168.x.x)           Python server (this file)           Browser
  ├── /stream  MJPEG  ◄──────── embedded in <img> by browser        │
  └── /capture JPEG   ◄──────── fetched by YOLO worker / LLM calls  │
                                 │                                    │
                                 ├── YOLO thread  ─► anomaly events  │
                                 ├── Scheduler thread ─► timed LLM   │
                                 ├── SSE /api/events ──────────────► │
                                 └── REST  /api/*  ◄──────────────── │

Run
---
  pip install -r requirements.txt
  python server.py

Then open http://localhost:5050 in a browser on the same WiFi.
Set DRONE_IP in env or edit ESP32_IP below to match the ESP32's IP.
"""

from __future__ import annotations
import os, time, base64, io, json, threading, queue, logging
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, render_template, stream_with_context

# ── Config ───────────────────────────────────────────────────────────────────
ESP32_IP          = os.environ.get("ESP32_IP",     "192.168.1.x")   # set to ESP32's IP
CAPTURE_URL       = f"http://{ESP32_IP}/capture"
STREAM_URL        = f"http://{ESP32_IP}/stream"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL         = os.environ.get("LLM_MODEL", "claude")           # claude | gpt4o | gemini
LLM_CLAUDE_MODEL  = "claude-opus-4-5"
LLM_GPT_MODEL     = "gpt-4o"
LLM_GEMINI_MODEL  = "gemini-1.5-flash"
YOLO_MODEL        = os.environ.get("YOLO_MODEL",   "yolov8n.pt")
YOLO_CONF         = float(os.environ.get("YOLO_CONF", "0.45"))
MAX_LOG_EVENTS    = 200
CAPTURE_TIMEOUT   = 5   # seconds

app = Flask(__name__, template_folder="templates", static_folder="static")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verbalize")

# ── Shared state (all guarded by _lock) ──────────────────────────────────────
_lock = threading.Lock()

state = {
    "yolo_running":     False,
    "schedule_running": False,
    "schedule_interval_s": 30,
    "last_frame_b64":   None,        # most recent captured frame (b64 JPEG)
    "last_detections":  [],          # list of {label, conf, x1,y1,x2,y2}
    "last_anomaly":     None,        # dict or None
    "last_verbalization": None,      # {text, timestamp, image_b64}
    "esp32_reachable":  False,
}

# Server-Sent Events queue (one per connected browser tab)
_sse_queues: list[queue.Queue] = []

def push_event(event_type: str, data: dict):
    payload = json.dumps({"type": event_type, "ts": datetime.now().isoformat(), **data})
    dead = []
    for q in _sse_queues:
        try:
            q.put_nowait(payload)
        except queue.Full:
            dead.append(q)
    for d in dead:
        _sse_queues.remove(d)

def append_log(event_type: str, data: dict):
    push_event(event_type, data)

# ── Camera helpers ────────────────────────────────────────────────────────────
def fetch_jpeg() -> bytes | None:
    """Fetch a JPEG snapshot from the ESP32."""
    try:
        r = requests.get(CAPTURE_URL, timeout=CAPTURE_TIMEOUT)
        if r.status_code == 200 and r.headers.get("Content-Type","").startswith("image/jpeg"):
            with _lock:
                state["esp32_reachable"] = True
            return r.content
    except Exception as e:
        log.warning(f"[CAM] fetch failed: {e}")
        with _lock:
            state["esp32_reachable"] = False
    return None

def jpeg_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

# ── YOLO worker ───────────────────────────────────────────────────────────────
_yolo_model = None

def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
        log.info(f"[YOLO] Model loaded: {YOLO_MODEL}")
    except Exception as e:
        log.error(f"[YOLO] Failed to load model: {e}")
        _yolo_model = None
    return _yolo_model

# Anomaly rules: if any of these classes appears, trigger LLM verbalization
ANOMALY_CLASSES = {
    "person", "fire", "smoke", "knife", "gun", "scissors",
    "cell phone", "laptop", "backpack", "suitcase",
    # motion anomaly: large bbox relative to frame
}
# Anomaly: sudden spike in object count (>3 new objects vs previous frame)
_prev_detection_count = 0

def run_yolo_on_frame(jpeg_bytes: bytes) -> tuple[list, bool]:
    """Returns (detections, is_anomaly)."""
    global _prev_detection_count
    model = _load_yolo()
    if model is None:
        return [], False

    try:
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        results = model(img, conf=YOLO_CONF, verbose=False)
        detections = []
        anomaly = False
        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                label   = model.names[cls_id]
                conf    = round(float(box.conf[0]), 3)
                x1,y1,x2,y2 = [round(float(v),1) for v in box.xyxy[0]]
                w, h = img.width, img.height
                area_pct = round((x2-x1)*(y2-y1)/(w*h)*100, 1)
                detections.append({
                    "label": label, "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area_pct": area_pct,
                })
                if label in ANOMALY_CLASSES:
                    anomaly = True
                if area_pct > 60:   # object fills >60% of frame — proximity alert
                    anomaly = True

        # Count spike anomaly
        delta = len(detections) - _prev_detection_count
        if delta >= 4:
            anomaly = True
        _prev_detection_count = len(detections)

        return detections, anomaly

    except Exception as e:
        log.error(f"[YOLO] inference error: {e}")
        return [], False

def yolo_worker():
    log.info("[YOLO] Worker started")
    while True:
        with _lock:
            running = state["yolo_running"]
        if not running:
            time.sleep(0.5)
            continue

        jpeg = fetch_jpeg()
        if jpeg is None:
            time.sleep(2)
            continue

        detections, anomaly = run_yolo_on_frame(jpeg)
        b64 = jpeg_to_b64(jpeg)

        with _lock:
            state["last_frame_b64"]  = b64
            state["last_detections"] = detections

        push_event("yolo_update", {
            "detections": detections,
            "anomaly": anomaly,
            "n": len(detections),
        })

        if anomaly:
            desc = describe_anomaly(detections)
            log.info(f"[YOLO] Anomaly detected — {desc}")
            with _lock:
                state["last_anomaly"] = {"description": desc, "timestamp": datetime.now().isoformat()}
            push_event("anomaly_detected", {"description": desc})
            # Auto-trigger LLM verbalization
            verbalization = call_vision_llm(jpeg, f"ANOMALY DETECTED: {desc}. Describe what you see and assess the risk.")
            if verbalization:
                with _lock:
                    state["last_verbalization"] = {
                        "text":      verbalization,
                        "timestamp": datetime.now().isoformat(),
                        "image_b64": b64,
                        "trigger":   "anomaly",
                    }
                push_event("verbalization", {
                    "text":      verbalization,
                    "image_b64": b64,
                    "trigger":   "anomaly",
                })

        time.sleep(0.5)  # ~2 fps YOLO (LLM is async on anomaly)

def describe_anomaly(detections: list) -> str:
    labels = [d["label"] for d in detections if d["label"] in ANOMALY_CLASSES]
    large  = [d["label"] for d in detections if d["area_pct"] > 60]
    parts  = []
    if labels:
        parts.append(f"hazard objects: {', '.join(set(labels))}")
    if large:
        parts.append(f"large proximity objects: {', '.join(set(large))}")
    if len(detections) >= 4 and not labels and not large:
        parts.append(f"sudden object count spike ({len(detections)} objects)")
    return "; ".join(parts) if parts else f"{len(detections)} objects detected"

# ── Scheduler worker ──────────────────────────────────────────────────────────
def scheduler_worker():
    log.info("[SCHED] Worker started")
    while True:
        with _lock:
            running  = state["schedule_running"]
            interval = state["schedule_interval_s"]
        if not running:
            time.sleep(1)
            continue

        jpeg = fetch_jpeg()
        if jpeg:
            b64  = jpeg_to_b64(jpeg)
            text = call_vision_llm(jpeg, "Describe this scene in detail. What do you observe? Any concerns?")
            if text:
                with _lock:
                    state["last_verbalization"] = {
                        "text":      text,
                        "timestamp": datetime.now().isoformat(),
                        "image_b64": b64,
                        "trigger":   "scheduled",
                    }
                push_event("verbalization", {
                    "text":      text,
                    "image_b64": b64,
                    "trigger":   "scheduled",
                })
        time.sleep(max(1, interval))

# ── Vision LLM ────────────────────────────────────────────────────────────────
def call_vision_llm(jpeg_bytes: bytes, prompt: str) -> str | None:
    model = LLM_MODEL.lower()
    b64   = jpeg_to_b64(jpeg_bytes)
    try:
        if model == "claude":
            return _llm_claude(b64, prompt)
        elif model == "gpt4o":
            return _llm_openai(b64, prompt)
        elif model == "gemini":
            return _llm_gemini(b64, prompt)
        else:
            return _llm_claude(b64, prompt)
    except Exception as e:
        log.error(f"[LLM] error: {e}")
        return f"[LLM error: {e}]"

def _llm_claude(b64: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp   = client.messages.create(
        model=LLM_CLAUDE_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return resp.content[0].text if resp.content else ""

def _llm_openai(b64: str, prompt: str) -> str:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp   = client.chat.completions.create(
        model=LLM_GPT_MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return resp.choices[0].message.content or ""

def _llm_gemini(b64: str, prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY",""))
    model  = genai.GenerativeModel(LLM_GEMINI_MODEL)
    import PIL.Image
    img    = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))
    resp   = model.generate_content([prompt, img])
    return resp.text or ""

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak_text(text: str):
    def _speak():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text[:400])
            engine.runAndWait()
        except Exception:
            try:
                import tempfile, subprocess
                from gtts import gTTS
                tts = gTTS(text=text[:400], lang="en")
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tts.save(tmp.name)
                subprocess.Popen(["ffplay","-nodisp","-autoexit", tmp.name],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log.warning(f"[TTS] failed: {e}")
    threading.Thread(target=_speak, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           stream_url=STREAM_URL,
                           esp32_ip=ESP32_IP,
                           llm_model=LLM_MODEL)

# ── SSE event stream
@app.route("/api/events")
def sse_events():
    q: queue.Queue = queue.Queue(maxsize=50)
    _sse_queues.append(q)

    def generate():
        try:
            # Send initial state
            with _lock:
                s = dict(state)
            yield f"data: {json.dumps({'type':'init','state': {k: v for k,v in s.items() if k != 'last_frame_b64'}})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            if q in _sse_queues:
                _sse_queues.remove(q)

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

# ── Status
@app.route("/api/status")
def api_status():
    with _lock:
        s = dict(state)
    s.pop("last_frame_b64", None)
    s.pop("last_verbalization", None)
    try:
        r = requests.get(f"http://{ESP32_IP}/status", timeout=2)
        s["esp32_status"] = r.json()
    except Exception:
        s["esp32_status"] = None
    return jsonify(s)

# ── YOLO control
@app.route("/api/yolo/start", methods=["POST"])
def yolo_start():
    with _lock:
        state["yolo_running"] = True
    push_event("yolo_status", {"running": True})
    return jsonify({"ok": True})

@app.route("/api/yolo/stop", methods=["POST"])
def yolo_stop():
    with _lock:
        state["yolo_running"] = False
    push_event("yolo_status", {"running": False})
    return jsonify({"ok": True})

@app.route("/api/yolo/detections")
def yolo_detections():
    with _lock:
        return jsonify({
            "detections": state["last_detections"],
            "anomaly":    state["last_anomaly"],
        })

# ── Scheduler control
@app.route("/api/schedule/set", methods=["POST"])
def schedule_set():
    data = request.get_json(force=True, silent=True) or {}
    interval = int(data.get("interval_s", 30))
    interval = max(5, min(3600, interval))
    with _lock:
        state["schedule_interval_s"] = interval
    push_event("schedule_config", {"interval_s": interval})
    return jsonify({"ok": True, "interval_s": interval})

@app.route("/api/schedule/start", methods=["POST"])
def schedule_start():
    with _lock:
        state["schedule_running"] = True
        interval = state["schedule_interval_s"]
    push_event("schedule_status", {"running": True, "interval_s": interval})
    return jsonify({"ok": True})

@app.route("/api/schedule/stop", methods=["POST"])
def schedule_stop():
    with _lock:
        state["schedule_running"] = False
    push_event("schedule_status", {"running": False})
    return jsonify({"ok": True})

# ── Manual capture + verbalize
@app.route("/api/capture_and_describe", methods=["POST"])
def capture_and_describe():
    data   = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "Describe this scene in detail. What do you observe?")
    jpeg   = fetch_jpeg()
    if jpeg is None:
        return jsonify({"ok": False, "error": "Could not reach ESP32 camera"}), 503

    b64  = jpeg_to_b64(jpeg)
    text = call_vision_llm(jpeg, prompt)

    result = {
        "text":      text,
        "timestamp": datetime.now().isoformat(),
        "image_b64": b64,
        "trigger":   "manual",
    }
    with _lock:
        state["last_verbalization"] = result
    push_event("verbalization", {"text": text, "image_b64": b64, "trigger": "manual"})

    speak_text(text)
    return jsonify({"ok": True, "text": text, "image_b64": b64})

# ── Chat
@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # Always fetch a fresh frame for context
    jpeg = fetch_jpeg()
    if jpeg:
        b64     = jpeg_to_b64(jpeg)
        prompt  = (
            f"You are an AI vision assistant monitoring a camera feed. "
            f"A user just asked: \"{message}\"\n\n"
            f"Look at the current camera image and answer based on what you see. "
            f"Be specific about what is visible in the image. Keep answer under 100 words."
        )
        reply = call_vision_llm(jpeg, prompt)

        with _lock:
            state["last_frame_b64"]   = b64
            state["last_verbalization"] = {
                "text":      reply,
                "timestamp": datetime.now().isoformat(),
                "image_b64": b64,
                "trigger":   "chat",
            }
        push_event("verbalization", {"text": reply, "image_b64": b64, "trigger": "chat"})
        speak_text(reply)
    else:
        # No camera — answer from knowledge
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            resp   = client.messages.create(
                model=LLM_CLAUDE_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": message}],
            )
            reply = resp.content[0].text if resp.content else "[No response]"
        except Exception as e:
            reply = f"[Error: {e}]"
        speak_text(reply)

    return jsonify({"ok": True, "reply": reply})

# ── Proxy ESP32 stream URL for the browser (avoids CORS issues)
@app.route("/api/stream_url")
def stream_url():
    return jsonify({"url": STREAM_URL, "capture_url": CAPTURE_URL})

# ── Latest verbalization
@app.route("/api/last_verbalization")
def last_verbalization():
    with _lock:
        v = state["last_verbalization"]
    return jsonify(v or {})

# ── Change LLM model at runtime
@app.route("/api/set_model", methods=["POST"])
def set_model():
    global LLM_MODEL
    data  = request.get_json(force=True, silent=True) or {}
    model = data.get("model","claude").lower()
    if model not in ("claude","gpt4o","gemini"):
        return jsonify({"ok": False, "error": "unknown model"}), 400
    LLM_MODEL = model
    push_event("model_changed", {"model": model})
    return jsonify({"ok": True, "model": model})

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start background workers
    threading.Thread(target=yolo_worker,      daemon=True, name="yolo").start()
    threading.Thread(target=scheduler_worker, daemon=True, name="sched").start()

    log.info("="*60)
    log.info(" Image Verbalization Server")
    log.info(f" ESP32 stream : {STREAM_URL}")
    log.info(f" ESP32 capture: {CAPTURE_URL}")
    log.info(f" LLM model    : {LLM_MODEL}")
    log.info(f" Open browser : http://localhost:5050")
    log.info("="*60)

    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
