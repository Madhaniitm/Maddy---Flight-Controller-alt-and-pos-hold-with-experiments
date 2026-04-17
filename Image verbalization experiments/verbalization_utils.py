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
import os, time, base64, io, math, csv
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────
ESP32_IP          = os.environ.get("ESP32_IP",  "192.168.1.x")   # set via env
CAPTURE_URL       = f"http://{ESP32_IP}/capture"
CAPTURE_TIMEOUT   = 5
RESULTS_DIR       = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY",    "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",    "")
OLLAMA_URL        = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ── Model identifiers ─────────────────────────────────────────────────────────
CLAUDE_MODEL  = "claude-opus-4-5"
GPT4O_MODEL   = "gpt-4o"
GEMINI_MODEL  = "gemini-1.5-flash"
LLAVA_MODEL   = "llava"               # via Ollama

# ── Canonical scenes ──────────────────────────────────────────────────────────
SCENES = [
    {"id": 1,  "label": "clear_open",    "truth": "safe",    "setup": "Empty floor, good lighting. Nothing in camera view."},
    {"id": 2,  "label": "person_near",   "truth": "hazard",  "setup": "Operator stands ~1m in front of camera."},
    {"id": 3,  "label": "wall_close",    "truth": "hazard",  "setup": "Point camera at wall from ~25cm away."},
    {"id": 4,  "label": "object_table",  "truth": "safe",    "setup": "Laptop on table, clear surroundings."},
    {"id": 5,  "label": "dim_light",     "truth": "caution", "setup": "Room lights off, single dim lamp only."},
    {"id": 6,  "label": "cluttered",     "truth": "caution", "setup": "Multiple objects scattered on floor."},
    {"id": 7,  "label": "door_open",     "truth": "safe",    "setup": "Open doorway visible in frame."},
    {"id": 8,  "label": "person_far",    "truth": "safe",    "setup": "Operator stands ~3m away from camera."},
    {"id": 9,  "label": "blocked_lens",  "truth": "hazard",  "setup": "Partially cover camera lens with hand."},
    {"id": 10, "label": "outdoor_bright","truth": "safe",    "setup": "Point camera at bright window / outdoor."},
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

def get_frame(scene_label: str, allow_synthetic: bool = True) -> bytes:
    """Fetch real frame; fall back to synthetic if ESP32 unreachable."""
    data = fetch_jpeg()
    if data:
        return data
    if allow_synthetic:
        print(f"[CAM] Using synthetic frame for '{scene_label}'")
        return synthetic_jpeg(scene_label)
    raise RuntimeError("ESP32 unreachable and synthetic fallback disabled.")

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
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    kwargs = dict(
        model=CLAUDE_MODEL, max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": b64}},
                {"type": "text",  "text": prompt},
            ],
        }],
    )
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    text = resp.content[0].text if resp.content else ""
    i, o = resp.usage.input_tokens, resp.usage.output_tokens
    # Claude pricing (claude-opus-4-5): $15/M in, $75/M out
    cost = round(i*15e-6 + o*75e-6, 6)
    return {"reply": text, "input_tokens": i, "output_tokens": o, "cost_usd": cost}

def _call_openai(b64, prompt, max_tokens, temperature):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
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
    import google.generativeai as genai
    from PIL import Image as PILImage
    genai.configure(api_key=GEMINI_API_KEY)
    model  = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature),
    )
    img    = PILImage.open(io.BytesIO(base64.b64decode(b64)))
    resp   = model.generate_content([prompt, img])
    text   = resp.text or ""
    try:
        i = resp.usage_metadata.prompt_token_count
        o = resp.usage_metadata.candidates_token_count
    except Exception:
        i = o = 0
    # Gemini 1.5 Flash: $0.075/M in, $0.30/M out
    cost = round(i*0.075e-6 + o*0.30e-6, 6)
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
def score_verbalization(reply: str, true_risk: str) -> dict:
    """
    4-point rubric:
      s1 +1 scene content mentioned
      s2 +1 proximity/spatial info
      s3 +1 correct risk classification
      s4 +1 word count 10-100
    Returns dict with individual scores and total.
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
    risk_kw  = {"safe","danger","hazard","warning","caution","risk","concern",
                "obstacle","block","proceed","stop","alert","emergency"}

    s1 = int(any(kw in r for kw in scene_kw))
    s2 = int(any(kw in r for kw in prox_kw))
    s4 = int(10 <= n_words <= 100)

    # Risk classification: look for the risk level in the reply
    detected_risk = None
    for lvl in ("hazard","caution","safe"):
        if lvl in r:
            detected_risk = lvl
            break
    # Also accept synonyms
    if detected_risk is None:
        if any(w in r for w in ("danger","obstacle","block","emergency","covered")):
            detected_risk = "hazard"
        elif any(w in r for w in ("warning","unclear","dim","cluttered","concern")):
            detected_risk = "caution"
        elif any(w in r for w in ("clear","open","proceed","fine","okay","no obstacle")):
            detected_risk = "safe"

    s3 = int(detected_risk == true_risk) if detected_risk else 0

    return {
        "s1_scene":      s1,
        "s2_proximity":  s2,
        "s3_risk":       s3,
        "s4_length":     s4,
        "quality_score": s1 + s2 + s3 + s4,
        "detected_risk": detected_risk,
        "word_count":    n_words,
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
