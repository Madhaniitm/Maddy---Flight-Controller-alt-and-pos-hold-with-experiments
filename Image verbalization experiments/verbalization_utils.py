"""
verbalization_utils.py
======================
Shared utilities for V-series image verbalization experiments.

All experiments import from here:
    from verbalization_utils import (
        fetch_jpeg, call_vision_llm, score_verbalization,
        bootstrap_ci, wilson_ci, SCENES, HAZARD_LABELS,
    )
"""

from __future__ import annotations
import os, sys, time, base64, io, math, csv

# Disable MediaPipe telemetry BEFORE any mediapipe/cv2 import
# Prevents clearcut upload attempts that cause periodic latency spikes
os.environ["GLOG_minloglevel"]   = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF/TFLITE logs too
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# ── Load API credentials from experiments/credentials.py (gitignored) ─────────
try:
    _exp_dir = str(Path(__file__).parent.parent / "experiments")
    if _exp_dir not in sys.path:
        sys.path.insert(0, _exp_dir)
    import credentials  # noqa: F401 — calls _load_credentials() at import time
except ImportError:
    pass  # credentials.py absent — keys must come from shell environment

# ── Config ────────────────────────────────────────────────────────────────────
ESP32_IP          = os.environ.get("ESP32_IP",  "10.186.33.138")
CAPTURE_URL       = f"http://{ESP32_IP}/capture"
CAPTURE_TIMEOUT   = 5
RESULTS_DIR       = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Keys and endpoints loaded from environment (set by credentials.py — never committed)
ANTHROPIC_API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
AZURE_CLAUDE_ENDPOINT = os.environ.get("AZURE_CLAUDE_ENDPOINT", "")
AZURE_CLAUDE_VERSION  = "2023-06-01"
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL   = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MINI_KEY   = os.environ.get("OPENAI_MINI_KEY", "")
OPENAI_MINI_URL   = os.environ.get("OPENAI_MINI_URL", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
OLLAMA_URL        = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ── Model identifiers ─────────────────────────────────────────────────────────
CLAUDE_MODEL  = "claude-sonnet-4-6"
GPT4O_MODEL   = "gpt-4o"
GEMINI_MODEL  = "gemini-2.5-flash"
LLAVA_MODEL   = "llava"               # via Ollama

# ── Canonical scenes ──────────────────────────────────────────────────────────
SCENES = [
    {"id": 1,  "label": "person_near",   "truth": "hazard",  "setup": "Operator stands ~1m in front of camera."},
    {"id": 2,  "label": "wall_close",    "truth": "hazard",  "setup": "Point camera at wall from ~25cm away."},
    {"id": 3,  "label": "object_table",  "truth": "caution", "setup": "Laptop close-up on table, fills frame."},
    {"id": 4,  "label": "dim_light",     "truth": "caution", "setup": "Room lights off, single dim lamp only."},
    {"id": 5,  "label": "cluttered",     "truth": "caution", "setup": "Multiple objects scattered on floor."},
    {"id": 6,  "label": "door_open",     "truth": "safe",    "setup": "Open doorway visible in frame."},
    {"id": 7,  "label": "person_far",    "truth": "caution", "setup": "Operator stands ~3m away from camera in cluttered lab."},
    {"id": 8,  "label": "blocked_lens",  "truth": "hazard",  "setup": "Partially cover camera lens with hand."},
]

HAZARD_LABELS = {"person", "fire", "smoke", "knife", "gun", "scissors",
                 "cell phone", "laptop", "backpack", "suitcase"}

RISK_LEVELS = ("safe", "caution", "hazard")

# ── Statistics ────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0: return 0., 0., 0.
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d
    return round(p, 4), round(max(0., c-m), 4), round(min(1., c+m), 4)

def bootstrap_ci(data, stat=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr = np.array(data, float)
    boots = [stat(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)), 4), round(float(lo), 4), round(float(hi), 4)

# ── Camera ────────────────────────────────────────────────────────────────────
def fetch_jpeg(url: str = CAPTURE_URL, timeout: int = CAPTURE_TIMEOUT) -> Optional[bytes]:
    """Fetch a single JPEG from the ESP32. Returns None on failure."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and "image/jpeg" in r.headers.get("Content-Type",""):
            return r.content
    except Exception as e:
        print(f"[CAM] fetch failed: {e}")
    return None

def jpeg_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

def synthetic_jpeg(label: str = "clear_open") -> bytes:
    """Fallback synthetic JPEG for offline testing — colour-coded by label."""
    from PIL import Image, ImageDraw, ImageFont
    colour_map = {
        "clear_open":     (60, 180, 60),
        "person_near":    (200, 60, 60),
        "wall_close":     (180, 50, 50),
        "object_table":   (60, 120, 200),
        "dim_light":      (60, 60, 60),
        "cluttered":      (160, 120, 60),
        "door_open":      (80, 200, 120),
        "person_far":     (100, 180, 100),
        "blocked_lens":   (20, 20, 20),
        "outdoor_bright": (240, 240, 180),
    }
    colour = colour_map.get(label, (128, 128, 128))
    img = Image.new("RGB", (320, 240), colour)
    try:
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label, fill=(255, 255, 255))
    except Exception:
        pass
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

_frame_counters: dict = {}

def get_frame(scene_label: str, allow_synthetic: bool = True) -> bytes:
    """Fetch real frame; fall back to synthetic if ESP32 unreachable.
    Saves every frame to results/frames/<scene_label>_run<N>.jpg for manual review."""
    data = fetch_jpeg()
    is_synthetic = False
    if not data:
        if allow_synthetic:
            print(f"[CAM] Using synthetic frame for '{scene_label}'")
            data = synthetic_jpeg(scene_label)
            is_synthetic = True
        else:
            raise RuntimeError("ESP32 unreachable and synthetic fallback disabled.")

    # Save frame to disk
    frames_dir = RESULTS_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)
    _frame_counters[scene_label] = _frame_counters.get(scene_label, 0) + 1
    tag = "syn" if is_synthetic else "real"
    run_n = _frame_counters[scene_label]
    frame_path = frames_dir / f"{scene_label}_run{run_n:02d}_{tag}.jpg"
    frame_path.write_bytes(data)

    return data

FRAMES_DIR = RESULTS_DIR / "frames"

def get_saved_frame(scene_label: str) -> bytes:
    """Load run03 real hardware frame captured during V1/V2. Always uses run 3."""
    path = FRAMES_DIR / f"{scene_label}_run03_real.jpg"
    if path.exists():
        return path.read_bytes()
    raise FileNotFoundError(
        f"No saved frame for '{scene_label}' (expected {path}). Run V1/V2 first.")

# ── YOLO preprocessing ────────────────────────────────────────────────────────
def run_yolo_on_frame(jpeg_bytes: bytes) -> str:
    """
    Run YOLOv8n on a frame and return formatted detection metadata string.
    This metadata is passed to every LLM call alongside the image.
    Falls back to a descriptive unavailable message if ultralytics not installed.
    """
    try:
        from ultralytics import YOLO as UltralyticsYOLO
        import cv2

        yolo   = UltralyticsYOLO("yolov8n.pt")
        img    = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        result = yolo(img, verbose=False, conf=0.25)[0]

        detections = []
        h_img = img.shape[0] if img is not None else 480
        for box in result.boxes:
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            label    = result.names[int(box.cls[0])]
            conf     = round(float(box.conf[0]), 2)
            bbox_h   = max(y2 - y1, 1)
            est_dist = round(h_img / bbox_h * 0.3, 2)   # rough metres heuristic
            detections.append(
                f"{label} (conf={conf}, est_dist~{est_dist}m, "
                f"bbox=[{x1},{y1},{x2},{y2}])"
            )
        if detections:
            return "YOLO detections: " + "; ".join(detections)
        return "YOLO detections: none"
    except Exception as e:
        return f"YOLO detections: unavailable ({e})"

def build_llm_prompt(yolo_metadata: str, task_prompt: str) -> str:
    """
    Prepend YOLO metadata to any LLM prompt and append the pilot action request.
    Used in every LLM call except G1 (pure isolation experiment).
    """
    return (
        f"{yolo_metadata}\n\n"
        f"{task_prompt}\n\n"
        "Pilot suggested action (choose one and state it clearly): "
        "PROCEED | SLOW_DOWN | STOP | LAND | HOLD"
    )

# ── Vision LLM calls ──────────────────────────────────────────────────────────
def call_vision_llm(
    jpeg_bytes: bytes,
    prompt: str,
    model: str = "claude",
    max_tokens: int = 256,
    temperature: float = 0.2,
    system: str = "",
) -> dict:
    """
    Unified vision LLM call. Returns:
        {reply, input_tokens, output_tokens, latency_ms, cost_usd, error}
    """
    b64 = jpeg_to_b64(jpeg_bytes)
    t0  = time.perf_counter()
    try:
        if model == "claude":
            r = _call_claude(b64, prompt, max_tokens, temperature, system)
        elif model == "gpt4o":
            r = _call_openai(b64, prompt, max_tokens, temperature)
        elif model == "gpt4o_mini":
            r = _call_openai_mini(b64, prompt, max_tokens, temperature)
        elif model == "gemini":
            r = _call_gemini(b64, prompt, max_tokens, temperature)
        elif model in ("llava", "ollama"):
            r = _call_ollama(b64, prompt, max_tokens)
        else:
            r = _call_claude(b64, prompt, max_tokens, temperature, system)
        r["latency_ms"] = round((time.perf_counter()-t0)*1000, 1)
        r["error"]      = ""
        return r
    except Exception as e:
        return {
            "reply": "", "input_tokens": 0, "output_tokens": 0,
            "latency_ms": round((time.perf_counter()-t0)*1000, 1),
            "cost_usd": 0.0, "error": str(e)[:120],
        }

def _call_claude(b64, prompt, max_tokens, temperature, system):
    import json, urllib.request
    body = {
        "model":      CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    if system:
        body["system"] = system
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        AZURE_CLAUDE_ENDPOINT,
        data=data,
        headers={
            "Content-Type":      "application/json",
            "Authorization":     f"Bearer {ANTHROPIC_API_KEY}",
            "anthropic-version": AZURE_CLAUDE_VERSION,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    text = raw["content"][0]["text"] if raw.get("content") else ""
    i    = raw.get("usage", {}).get("input_tokens",  0)
    o    = raw.get("usage", {}).get("output_tokens", 0)
    # Claude Sonnet pricing: $3/M in, $15/M out
    cost = round(i*3e-6 + o*15e-6, 6)
    return {"reply": text, "input_tokens": i, "output_tokens": o, "cost_usd": cost}

def _call_openai(b64, prompt, max_tokens, temperature):
    import openai
    kwargs = dict(api_key=OPENAI_API_KEY)
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    client = openai.OpenAI(**kwargs)
    resp   = client.chat.completions.create(
        model=GPT4O_MODEL, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    text = resp.choices[0].message.content or ""
    i    = resp.usage.prompt_tokens
    o    = resp.usage.completion_tokens
    # GPT-4o pricing: $5/M in, $15/M out
    cost = round(i*5e-6 + o*15e-6, 6)
    return {"reply": text, "input_tokens": i, "output_tokens": o, "cost_usd": cost}

def _call_gemini(b64, prompt, max_tokens, temperature):
    """
    Calls Gemini via REST API directly — same approach as G-series GeminiProvider
    in multi_llm_provider.py (avoids google.genai SDK version issues).
    thinkingBudget=0 disables thinking so all max_tokens go to the reply.
    """
    import json, urllib.request
    url  = (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            ],
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     temperature,
            "thinkingConfig":  {"thinkingBudget": 0},   # disable thinking — saves tokens
        },
    }
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    # Parse — same logic as GeminiProvider._parse in multi_llm_provider.py
    candidate = raw.get("candidates", [{}])[0]
    parts     = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if "text" in p)

    usage = raw.get("usageMetadata", {})
    i = usage.get("promptTokenCount",     0)
    o = usage.get("candidatesTokenCount", 0)
    # Gemini 2.5 Flash pricing: $0.075/M in, $0.30/M out
    cost = round(i*0.075e-6 + o*0.30e-6, 6)
    return {"reply": text, "input_tokens": i, "output_tokens": o, "cost_usd": cost}

def _call_openai_mini(b64, prompt, max_tokens, temperature):
    import openai
    client = openai.OpenAI(api_key=OPENAI_MINI_KEY, base_url=OPENAI_MINI_URL)
    resp   = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    text = resp.choices[0].message.content or ""
    i    = resp.usage.prompt_tokens
    o    = resp.usage.completion_tokens
    # GPT-4o-mini pricing: $0.15/M in, $0.60/M out
    cost = round(i*0.15e-6 + o*0.60e-6, 6)
    return {"reply": text, "input_tokens": i, "output_tokens": o, "cost_usd": cost}

def _call_ollama(b64, prompt, max_tokens):
    payload = {
        "model": LLAVA_MODEL, "prompt": prompt,
        "images": [b64], "stream": False,
        "options": {"num_predict": max_tokens},
    }
    r    = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
    data = r.json()
    text = data.get("response","")
    # Ollama is local — zero cost; token counts approximated
    n = len(text.split())
    return {"reply": text, "input_tokens": n, "output_tokens": n, "cost_usd": 0.0}

# ── Verbalization scoring ─────────────────────────────────────────────────────
PILOT_ACTION_KEYWORDS = {
    # Drone control commands (primary vocabulary)
    "hover", "pitch_forward", "pitch_back", "pitch forward", "pitch back",
    "roll_left", "roll_right", "roll left", "roll right",
    "yaw_left", "yaw_right", "yaw left", "yaw right",
    "ascend", "descend", "land",
    # Legacy / fallback words still accepted
    "proceed", "slow_down", "slow down", "slowdown", "stop", "hold",
}

def score_verbalization(reply: str, true_risk: str) -> dict:
    """
    5-point rubric:
      s1 +1 scene content mentioned
      s2 +1 proximity/spatial info
      s3 +1 correct risk classification
      s4 +1 word count 10-100
      s5 +1 pilot action suggested (drone control command)
    Returns dict with individual scores and total (max=5).
    """
    r = reply.lower()
    words = reply.split()
    n_words = len(words)

    scene_kw = {"see","observe","wall","obstacle","object","floor","ceiling",
                "surface","dark","bright","blurry","clear","colour","color",
                "person","table","door","lamp","light","room","outdoor","window",
                "laptop","chair","bag","box","covered","hand","partial"}
    prox_kw  = {"cm","mm","metre","meter","distance","close","near","far",
                "proxim","within","away","behind","front","side","left","right",
                "above","below","approximately","roughly","about","adjacent"}

    s1 = int(any(kw in r for kw in scene_kw))
    s2 = int(any(kw in r for kw in prox_kw))
    s4 = int(10 <= n_words <= 150)
    s5 = int(any(a in r for a in PILOT_ACTION_KEYWORDS))

    # Risk classification
    detected_risk = None
    for lvl in ("hazard","caution","safe"):
        if lvl in r:
            detected_risk = lvl
            break
    if detected_risk is None:
        if any(w in r for w in ("danger","obstacle","block","emergency","covered")):
            detected_risk = "hazard"
        elif any(w in r for w in ("warning","unclear","dim","cluttered","concern")):
            detected_risk = "caution"
        elif any(w in r for w in ("clear","open","fine","okay","no obstacle")):
            detected_risk = "safe"

    s3 = int(detected_risk == true_risk) if detected_risk else 0

    # Extract pilot action from reply (drone control vocabulary, longest match first)
    detected_action = None
    for a in ("pitch_forward", "pitch forward", "pitch_back", "pitch back",
              "roll_left", "roll left", "roll_right", "roll right",
              "yaw_left", "yaw left", "yaw_right", "yaw right",
              "hover", "ascend", "descend", "land",
              "stop", "slow_down", "slow down", "hold", "proceed"):
        if a in r:
            detected_action = a.replace(" ", "_").upper()
            break

    return {
        "s1_scene":       s1,
        "s2_proximity":   s2,
        "s3_risk":        s3,
        "s4_length":      s4,
        "s5_pilot_action": s5,
        "quality_score":  s1 + s2 + s3 + s4 + s5,
        "detected_risk":  detected_risk,
        "detected_action": detected_action,
        "word_count":     n_words,
    }

def extract_json_risk(reply: str) -> Optional[str]:
    """Extract risk_level from a JSON-format LLM reply."""
    import re, json as _json
    m = re.search(r'\{.*?\}', reply, re.DOTALL)
    if m:
        try:
            d = _json.loads(m.group())
            r = str(d.get("risk_level", "")).lower()
            if r in RISK_LEVELS: return r
        except Exception:
            pass
    return None

# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text: str):
    import threading
    def _run():
        try:
            import pyttsx3
            e = pyttsx3.init(); e.setProperty("rate", 165)
            e.say(text[:400]); e.runAndWait()
        except Exception:
            try:
                import tempfile, subprocess
                from gtts import gTTS
                t = gTTS(text=text[:400], lang="en")
                f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                t.save(f.name)
                subprocess.Popen(["ffplay","-nodisp","-autoexit", f.name],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()

# ── CSV helpers ───────────────────────────────────────────────────────────────
def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def preflight(skip: bool = False) -> bool:
    """Check ESP32 is reachable. Returns True if OK or skip=True."""
    if skip:
        print("[PREFLIGHT] Skipped (running in synthetic mode)")
        return True
    try:
        r = requests.get(f"http://{ESP32_IP}/status", timeout=3)
        if r.status_code == 200:
            d = r.json()
            print(f"[PREFLIGHT] ESP32 online — heap={d.get('heap')} PSRAM={d.get('psram')}")
            return True
    except Exception as e:
        print(f"[PREFLIGHT] ESP32 unreachable: {e}")
    print("[PREFLIGHT] FAIL — set ESP32_IP env var and ensure camera is on the same WiFi")
    return False
