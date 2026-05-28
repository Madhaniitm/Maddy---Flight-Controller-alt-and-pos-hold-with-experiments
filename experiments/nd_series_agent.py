"""
ND-Series Experiment Infrastructure — nd_series_agent.py
=========================================================
Room Emergency Surveillance Drone — camera pipeline as a tool call.

Mission:
  The drone patrols an indoor space and uses its camera to detect emergencies
  IN THE ROOM (person collapsed, fire, flooding, structural damage, etc.).
  The drone is not protecting itself — it is the observer, monitoring for
  situations that require human intervention.

Architecture:
  LLM Agent (surveillance orchestrator)
      ↓ calls
  analyze_scene()  ←  MediaPipe + YOLO-World + DepthAnything + LLM vision
      ↓ returns structured JSON with room status
  LLM decides next action:
      EMERGENCY → hover + alert operator + investigate closer
      ALERT     → move toward subject, get closer view
      CLEAR     → continue patrol sweep
      ↓ calls
  drone control tools (navigate, hover, rotate_yaw, etc.)
      ↓ telemetry read after every command
  get_sensor_status() → verify + log

Room status labels (what the drone reports about the room):
  EMERGENCY : Active emergency — person collapsed/injured, fire, severe flooding,
              hazardous material visible, violent incident.
  ALERT     : Something needs investigation — person in unusual position,
              smoke/haze, restricted area breach, unusual object.
  CLEAR     : No emergency. Room appears normal.

Drone action labels (what the drone should do next):
  INVESTIGATE_CLOSER  — fly toward the subject of interest
  HOVER_MONITOR       — hold position, keep observing
  ALERT_OPERATOR      — notify human operator immediately
  CONTINUE_PATROL     — continue the room surveillance sweep
  RETURN_TO_BASE      — mission complete or battery low

Room-event trigger (background MediaPipe):
  Background thread runs MediaPipe every MONITOR_INTERVAL_S seconds.
  Person/object detected → sets _room_event_flag → main loop issues hover()
  and injects an event notification into the conversation → LLM calls
  analyze_scene(context="investigate") to assess and plan response.

Logging per run:
  _camera_rows   — every analyze_scene() call (all pipeline outputs)
  _workflow_rows — every drone command (telemetry before + after)
"""

import sys, os, json, time, math, pathlib, threading, re, base64
import numpy as np
import cv2
import urllib.request

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXP_DIR   = pathlib.Path(__file__).parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from d_series_agent import DAgent, D_ALL_TOOLS, D_SYSTEM_PROMPT, _claude_tools_to_openai
from c_series_agent import (
    _ENDPOINT, _API_KEY, _MODEL, _VERSION,
    COST_IN, COST_OUT, MAX_TURNS,
)
from verbalization_utils import call_vision_llm, RESULTS_DIR, write_csv
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer, COMBINED_PROMPT_TEMPLATE,
)
from robust_local_detector import load_mediapipe_detector, detect_hazard

MONITOR_INTERVAL_S = 1.0   # background MediaPipe polling interval


# ── Tool format converters (used by OpenAI and Gemini orchestrator loops) ──────

def _nd_tools_to_openai(tools):
    """
    Wrap _claude_tools_to_openai (imported from d_series_agent) for ND_ALL_TOOLS.
    Identical format — reuse the existing converter.
    """
    return _claude_tools_to_openai(tools)


def _gemini_schema(schema: dict) -> dict:
    """
    Recursively convert a JSON schema dict from Anthropic format to Gemini format.

    Gemini REST API requirements:
      • Type names must be uppercase: OBJECT, STRING, NUMBER, INTEGER, BOOLEAN, ARRAY
      • Array types MUST include an "items" field (Gemini raises INVALID_ARGUMENT if missing)
      • "required" is omitted — Gemini doesn't use it at the property level
    """
    TYPE_UP = {
        "object": "OBJECT", "string": "STRING", "number": "NUMBER",
        "integer": "INTEGER", "boolean": "BOOLEAN", "array": "ARRAY",
    }
    out = {}
    if "type" in schema:
        out["type"] = TYPE_UP.get(schema["type"], schema["type"].upper())
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if "properties" in schema:
        out["properties"] = {k: _gemini_schema(v)
                             for k, v in schema["properties"].items()}
    # Arrays: recursively convert items if present; add default if missing.
    # Gemini rejects ARRAY types without an items field (INVALID_ARGUMENT).
    if "items" in schema:
        out["items"] = _gemini_schema(schema["items"])
    elif out.get("type") == "ARRAY":
        out["items"] = {"type": "STRING"}   # safe default — steps/waypoints are strings
    # Note: "required" is omitted — Gemini ignores it and it can cause API warnings
    return out


def _nd_tools_to_gemini(tools):
    """
    Convert ND_ALL_TOOLS (Anthropic format) to a Gemini functionDeclarations list.
    All tools are declared so all 4 orchestrators can fly the drone equally.
    """
    decls = []
    for t in tools:
        decls.append({
            "name":        t["name"],
            "description": t["description"],
            "parameters":  _gemini_schema(t["input_schema"]),
        })
    return [{"functionDeclarations": decls}]

# Orchestrators supported for tool-calling agentic loop
ORCHESTRATORS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

# Navigation tools — after these execute, inject updated camera frame
NAV_TOOLS = {
    "navigate_to_waypoint", "move_forward", "move_backward",
    "move_left", "move_right", "rotate_yaw", "set_altitude_target",
}

# ── analyze_scene tool definition ──────────────────────────────────────────────

ANALYZE_SCENE_TOOL = {
    "name": "analyze_scene",
    "description": (
        "Run the sensor pipeline on the current camera frame and return raw detection metadata.\n"
        "  Tier 1.5 — MediaPipe EfficientDet-Lite0: person detection, confidence, estimated distance (~14ms).\n"
        "  Tier 2   — YOLO-World + DepthAnything v2: object detections with metric depth (~275ms).\n\n"
        "Returns raw JSON sensor data — NO internal LLM assessment.\n"
        "YOU have the camera image in your context. Use the sensor metadata to assist\n"
        "your own visual assessment and decide the room status and next action.\n\n"
        "Call this at every patrol waypoint. After receiving the metadata, classify:\n"
        "  EMERGENCY — person collapsed/injured, fire, flooding, hazardous condition\n"
        "  ALERT     — unusual person position, faint smoke, restricted area breach\n"
        "  CLEAR     — no emergency, room appears normal\n\n"
        "Then decide drone action:\n"
        "  INVESTIGATE_CLOSER / HOVER_MONITOR / ALERT_OPERATOR / CONTINUE_PATROL / RETURN_TO_BASE"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": (
                    "Why this call is made: "
                    "'patrol' (routine surveillance sweep), "
                    "'investigate' (moving toward detected subject), "
                    "'verify' (confirming earlier finding), "
                    "'room_event' (background sensor triggered)."
                ),
            }
        },
        "required": [],
    },
}

# ── Drone control tools — need telemetry logging + optional human approval ─────

DRONE_CONTROL_TOOLS = {
    "arm", "disarm", "find_hover_throttle",
    "enable_altitude_hold", "disable_altitude_hold",
    "set_altitude_target", "hover",
    "set_throttle", "wait",
    "move_forward", "move_backward", "move_left", "move_right",
    "stop_movement", "navigate_to_waypoint", "return_to_home",
    "rotate_yaw", "set_home_position",
    "land", "emergency_land",
}

ND_ALL_TOOLS = D_ALL_TOOLS + [ANALYZE_SCENE_TOOL]

# ── ND system prompt ───────────────────────────────────────────────────────────

ND_SYSTEM_PROMPT = D_SYSTEM_PROMPT + """
━━ MISSION: INDOOR ROOM EMERGENCY SURVEILLANCE ━━
  You are an autonomous drone deployed to surveil an indoor space and detect
  emergencies. You are NOT protecting yourself — you are monitoring the room
  for situations that require human intervention.

  Your job:
    1. Patrol the room systematically
    2. Call analyze_scene at every waypoint to get sensor metadata
    3. Use the camera image (already in your context) + sensor data to assess the room
    4. Investigate any alert or emergency by flying closer
    5. Alert the operator with a clear, structured report

━━ SENSOR TOOL ━━
  analyze_scene(context=...)
      Runs MediaPipe + YOLO-World + DepthAnything on the current frame.
      Returns raw sensor JSON — NO internal LLM assessment.
      YOU look at the camera image directly and decide.

      Fields returned:
        mediapipe          : person detection result (confidence, distance, risk)
        mp_person_detected : True/False
        mp_risk            : MediaPipe risk level
        yolo_detections    : YOLO-World objects with metric depth (conf ≥ 0.35)
        pipeline_ms        : sensor latency

      contexts: "patrol" | "investigate" | "verify" | "room_event"

━━ TELEMETRY VERIFICATION PROTOCOL (same as C-series) ━━
  After EVERY drone command, verify it succeeded before proceeding:
    • After set_altitude_target(h) or takeoff:
        check_altitude_reached(target_m=h, tolerance_m=0.10)
        If ✗ → wait(2.0) → re-check. After 3 failures → land() and report.
    • After arm() or any movement command:
        check_drone_stable(max_degrees=5.0)
        If ✗ → wait(1.0) → re-check. After 3 failures → hover() and report.
    • After navigate_to_waypoint or move_*:
        get_sensor_status() → confirm position changed as expected.
  NEVER proceed to the next step without confirming the current one succeeded.

━━ SURVEILLANCE PATROL PROTOCOL ━━
  At each patrol waypoint:
    1. navigate_to_waypoint(...)          — move to position
    2. check_drone_stable()               — verify stable after movement
    3. analyze_scene(context="patrol")    — get sensor data; you see the image
    4. get_sensor_status()                — log altitude and position
    5. Reason: what does the camera show? what do sensors confirm?
    6. Reply in the REQUIRED FORMAT (see below)
    7. Decide drone action based on Risk

━━ ROOM EMERGENCY RESPONSE PROTOCOL ━━
  If Risk is EMERGENCY or ALERT:
    1. hover()                              — hold for stable view
    2. check_drone_stable()                 — verify stable before investigation
    3. analyze_scene(context="investigate") — get fresh sensor data
    4. Reply in REQUIRED FORMAT with full findings
    5. Execute appropriate action:
         EMERGENCY → INVESTIGATE_CLOSER + ALERT_OPERATOR
         ALERT     → navigate closer + analyze_scene again
    6. NEVER ignore EMERGENCY

━━ REQUIRED REPLY FORMAT ━━
  After every analyze_scene call, reply in EXACTLY this format:
  Description: <1-2 sentences — your own visual analysis of the camera image>
  Sensor note: <discrepancy between image and YOLO/MediaPipe, or 'consistent'>
  Proximity: <your estimate of closest person/object, cross-check with depth_m>
  Risk: <EMERGENCY | ALERT | CLEAR>
  Pilot suggested action: <INVESTIGATE_CLOSER | HOVER_MONITOR | ALERT_OPERATOR | CONTINUE_PATROL | RETURN_TO_BASE>
  Confidence: <0.0–1.0 — your confidence in the Risk classification above>
"""

# ── ND_SURVEILLANCE_PROMPT removed ────────────────────────────────────────────
# There is no longer an inner Tier-3 LLM inside analyze_scene.
# The orchestrator LLM receives the camera image directly in its message context
# and calls analyze_scene only to get raw sensor metadata (MediaPipe + YOLO +
# DepthAnything). The orchestrator itself classifies and decides next action.

# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_mp_response(mp_result: dict) -> str:
    meta   = mp_result.get("metadata", "")
    risk   = mp_result.get("risk", "unknown")
    person = ("person detected"
              if "person detected" in meta.lower()
              else "no person detected")
    conf_m = re.search(r"conf=([\d.]+)", meta)
    dist_m = re.search(r"est_dist=([\d.]+)m|depth_m=([\d.]+)", meta)
    conf_s = f"; conf={conf_m.group(1)}" if conf_m else ""
    dist_s = (f"; est_dist={dist_m.group(1) or dist_m.group(2)}m"
              if dist_m else "")
    return f"{person}{conf_s}{dist_s}; risk={risk}"


def _extract_room_status(text: str) -> str:
    """Extract room status from ND surveillance prompt response."""
    up = text.upper()
    # Look for explicit STATUS: line first
    import re as _re
    m = _re.search(r"STATUS:\s*(EMERGENCY|ALERT|CLEAR)", up)
    if m:
        return m.group(1)
    # Fall back to keyword scan
    for lvl in ("EMERGENCY", "ALERT", "CLEAR"):
        if lvl in up:
            return lvl
    return "UNKNOWN"


def _extract_urgency(text: str) -> str:
    up = text.upper()
    import re as _re
    m = _re.search(r"URGENCY:\s*(IMMEDIATE|MONITOR|NONE)", up)
    if m:
        return m.group(1)
    if "IMMEDIATE" in up:
        return "IMMEDIATE"
    if "MONITOR" in up:
        return "MONITOR"
    return "NONE"


def _extract_action(text: str) -> str:
    """Extract recommended drone action from surveillance prompt response."""
    up = text.upper()
    import re as _re
    m = _re.search(
        r"DRONE_ACTION:\s*(INVESTIGATE_CLOSER|HOVER_MONITOR|ALERT_OPERATOR|"
        r"CONTINUE_PATROL|RETURN_TO_BASE)", up)
    if m:
        return m.group(1)
    # Keyword scan fallback
    for a in ["INVESTIGATE_CLOSER", "HOVER_MONITOR", "ALERT_OPERATOR",
              "CONTINUE_PATROL", "RETURN_TO_BASE"]:
        if a in up:
            return a
    return "UNKNOWN"


def _extract_confidence(text: str) -> float:
    """
    Extract LLM self-reported confidence score from structured response.

    Rationale (AgentThink arXiv:2505.15298 + GSCE arXiv:2502.12531):
      Eliciting an explicit confidence score enables calibration analysis —
      comparing stated confidence against actual accuracy across scenes and models.
      Returns -1.0 if not found (sentinel for 'not reported').
    """
    m = re.search(r"CONFIDENCE:\s*([\d.]+)", text.upper())
    if m:
        try:
            return round(min(1.0, max(0.0, float(m.group(1)))), 3)
        except ValueError:
            pass
    return -1.0   # sentinel: not found / not reported


def _encode_frame(jpeg: bytes | None) -> str | None:
    """Base64-encode a JPEG frame for direct inclusion in vision-LLM messages."""
    if jpeg is None:
        return None
    return base64.b64encode(jpeg).decode()


def _filter_yolo_by_confidence(yolo_meta: str, min_conf: float = 0.35) -> str:
    """
    Filter YOLO detection lines to keep only those with conf >= min_conf.

    Rationale (PMC12583037 accuracy-latency tradeoff + LiteVLM arXiv:2506.07416):
      Passing low-confidence detections to the Tier-3 LLM introduces noise that
      degrades reasoning quality — same mechanism as CLIP score anchoring that
      degraded LLM reasoning in the V_clip_ablation experiments.
      Filtering to high-confidence detections only also reduces input token count
      → lower Tier-3 latency and cost.

    Preserves non-detection lines (header context, depth summaries, MediaPipe
    summary) so the LLM always receives scene geometry and person-detection data.
    Lines without a 'conf=' token are always kept.
    """
    lines = yolo_meta.split("\n")
    filtered = []
    for line in lines:
        conf_m = re.search(r"conf=([\d.]+)", line)
        if conf_m:
            if float(conf_m.group(1)) >= min_conf:
                filtered.append(line)
            # else: drop low-confidence detection line silently
        else:
            # Non-detection line (header, MediaPipe, depth context) — always keep
            filtered.append(line)
    return "\n".join(filtered)


# ── NDAgent ────────────────────────────────────────────────────────────────────

class NDAgent(DAgent):
    """
    DAgent extended with the sensor pipeline as a tool call.

    analyze_scene runs MediaPipe + YOLO-World + DepthAnything and returns raw
    sensor metadata JSON. No inner LLM — the orchestrator LLM receives the
    camera image directly in its message context and assesses it itself.

    Parameters
    ----------
    session_id   : experiment session label
    mp_detector  : loaded MediaPipe detector
    mp_type      : detector type string
    yolo_model   : loaded YOLO-World model
    yolo_type    : YOLO type string
    coco_model   : loaded COCO YOLO model
    depth_pipe   : loaded DepthAnything pipeline
    frame_source : "saved" | "esp32" | "webcam"
    esp32_url    : ESP32 camera URL
    """

    def __init__(self, session_id="ND_default",
                 mp_detector=None, mp_type=None,
                 yolo_model=None, yolo_type=None,
                 coco_model=None, depth_pipe=None,
                 frame_source="saved",
                 esp32_url="http://10.186.33.138/capture"):
        super().__init__(session_id=session_id)

        self.mp_detector  = mp_detector
        self.mp_type      = mp_type
        self.yolo_model   = yolo_model
        self.yolo_type    = yolo_type
        self.coco_model   = coco_model
        self.depth_pipe   = depth_pipe
        self.frame_source = frame_source
        self.esp32_url    = esp32_url

        # Frame state
        self._current_frame: bytes | None = None

        # Run-level counters
        self._current_run     = 1
        self._camera_call_num = 0
        self._action_num      = 0

        # Per-run log rows (cleared by experiment at start of each run)
        self._camera_rows:   list[dict] = []
        self._workflow_rows: list[dict] = []

        # Emergency state
        self._emergency_flag      = threading.Event()
        self._mp_stop_flag        = threading.Event()
        self._mp_monitor_thread: threading.Thread | None = None
        self._last_mp_summary     = ""    # last MediaPipe result, for advisory injection
        self._last_emergency_frame: bytes | None = None  # frame that triggered the alert

    # ── Frame management ───────────────────────────────────────────────────────

    def set_frame(self, jpeg_bytes: bytes):
        """Explicitly set the current frame (for 'saved' frame source)."""
        self._current_frame = jpeg_bytes

    def _get_frame(self) -> bytes | None:
        """Fetch a JPEG frame from the configured source."""
        if self.frame_source == "saved":
            return self._current_frame

        if self.frame_source == "esp32":
            try:
                with urllib.request.urlopen(self.esp32_url, timeout=3) as r:
                    data = r.read()
                if len(data) > 1000:
                    self._current_frame = data
                    return data
            except Exception:
                pass

        # Webcam (also used as esp32 fallback)
        try:
            cap = cv2.VideoCapture(0)
            ok, frame = cap.read()
            cap.release()
            if ok:
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
                data = buf.tobytes()
                self._current_frame = data
                return data
        except Exception:
            pass

        return self._current_frame   # last known frame

    # ── Telemetry snapshot ─────────────────────────────────────────────────────

    def _get_telemetry_snapshot(self) -> dict:
        s = self.state
        with s.lock:
            return {
                "alt_m":     round(s.ekf_z, 3),
                "roll_deg":  round(s.roll,  2),
                "pitch_deg": round(s.pitch, 2),
                "yaw_deg":   round(s.yaw,   2),
                "bat_pct":   round(s.bat_pct, 1),
                "armed":     s.armed,
                "althold":   s.althold,
                "pos_x":     round(s.ekf_x, 3),
                "pos_y":     round(s.ekf_y, 3),
            }

    # ── Emergency monitor ──────────────────────────────────────────────────────

    def start_emergency_monitor(self):
        """
        Start background MediaPipe room-event monitoring thread.

        In the room-surveillance mission, any person detection is a trigger
        for investigation — the drone is there to check on people, not avoid
        them. The monitor fires when:
          - A person is detected (any confidence) → investigate
          - A high-risk detection (e.g. very close object, occluded lens) → investigate

        This replaces the old obstacle-avoidance trigger with a room-event trigger.
        """
        if self.mp_detector is None:
            return
        self._mp_stop_flag.clear()
        self._emergency_flag.clear()

        def _monitor():
            while not self._mp_stop_flag.is_set():
                jpeg = self._get_frame()
                if jpeg is not None:
                    try:
                        img = cv2.imdecode(
                            np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        result = detect_hazard(
                            img, depth_map=None,
                            mp_detector=self.mp_detector,
                            mp_type=self.mp_type)
                        self._last_mp_summary = _format_mp_response(result)
                        meta = result.get("metadata", "").lower()
                        person_present = "person detected" in meta
                        high_risk      = result.get("risk") in ("hazard", "caution")
                        if person_present or high_risk:
                            # Store the exact frame that triggered the alert.
                            # This is what gets shown to the LLM in the advisory —
                            # it may differ from the frame the LLM is currently
                            # working on, which is how it decides to accept or ignore.
                            self._last_emergency_frame = jpeg
                            self._emergency_flag.set()
                    except Exception:
                        pass
                self._mp_stop_flag.wait(MONITOR_INTERVAL_S)

        self._mp_monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self._mp_monitor_thread.start()

    def stop_emergency_monitor(self):
        self._mp_stop_flag.set()

    def handle_emergency_if_flagged(self) -> tuple:
        """
        Advisory emergency notification — background MediaPipe sensor trigger.

        When the background thread detects a person or high-risk condition, it
        sets _emergency_flag. On the next LLM turn, this injects an ADVISORY
        notification — the LLM decides what to do based on its own history.

        Design rationale:
          The LLM already has the camera image in its message context and has
          been running analyze_scene calls. It can check its recent history:
            - If it just analyzed the same scene → ignore as redundant trigger
            - If this looks new or different → call analyze_scene + decide action
          This avoids forced double-analysis and lets the LLM own the decision.
          (C-series finding: short history works well — LLM can deduplicate events.)

        No auto-hover. No mandatory redirect. The LLM decides.

        Returns:
            (flagged: bool, advisory_message: str | None)
        """
        if not self._emergency_flag.is_set():
            return False, None

        self._emergency_flag.clear()
        timestamp  = round(self.sim_time, 2)
        mp_summary = self._last_mp_summary or "detection details unavailable"

        print(f"  🔔 [ND SENSOR t={timestamp}s] Background MediaPipe advisory — "
              f"{mp_summary}")

        advisory = (
            f"[BACKGROUND SENSOR NOTIFICATION — t={timestamp}s]\n"
            f"MediaPipe detected: {mp_summary}\n\n"
            f"Check your recent analysis history. You have two options:\n"
            f"  • If you have already assessed this scene recently and the sensor "
            f"reading matches what you saw — you may IGNORE this notification "
            f"and continue your current task.\n"
            f"  • If this appears to be a NEW or DIFFERENT situation from what "
            f"you last saw — call analyze_scene(context='room_event') to get "
            f"fresh sensor data, look at the camera image, and decide whether "
            f"to investigate or resume patrol.\n\n"
            f"The decision is yours based on context. No action has been taken automatically."
        )
        return True, advisory

    def reset_run(self, run: int):
        """Reset per-run state at the start of each run."""
        self._current_run          = run
        self._camera_call_num      = 0
        self._action_num           = 0
        self._camera_rows          = []
        self._workflow_rows        = []
        self._emergency_flag.clear()
        self._last_emergency_frame = None
        self._last_mp_summary      = ""

    # ── Tool execution override ────────────────────────────────────────────────

    def execute_tool(self, name: str, args: dict) -> str:

        # ── analyze_scene ──────────────────────────────────────────────────────
        if name == "analyze_scene":
            context = args.get("context", "routine")
            jpeg    = self._get_frame()

            if jpeg is None:
                return json.dumps({
                    "error":              "No frame available",
                    "risk":               "unknown",
                    "recommended_action": "HOVER",
                    "scene_description":  "Camera unavailable — hold position.",
                    "mediapipe":          "unavailable",
                    "yolo_summary":       "unavailable",
                    "mp_risk":            "unknown",
                    "pipeline_ms":        0,
                    "cost_usd":           0.0,
                })

            img_bgr = cv2.imdecode(
                np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

            # Tier 1.5 — MediaPipe
            t_mp = time.perf_counter()
            if self.mp_detector is not None:
                mp_result = detect_hazard(
                    img_bgr, depth_map=None,
                    mp_detector=self.mp_detector,
                    mp_type=self.mp_type)
            else:
                mp_result = {"risk": "unknown",
                             "metadata": "MediaPipe not loaded"}
            mp_ms   = round((time.perf_counter() - t_mp) * 1000, 1)
            mp_full = _format_mp_response(mp_result)

            # Tier 2 — YOLO-World + DepthAnything
            t_yolo = time.perf_counter()
            if self.yolo_model is not None:
                tier2     = enhanced_yolo_infer(
                    self.yolo_model, self.yolo_type,
                    None, None, None, jpeg,
                    coco_model=self.coco_model,
                    depth_pipe=self.depth_pipe,
                    use_clip=False,
                )
                yolo_meta = tier2["yolo_meta"]
            else:
                yolo_meta = "YOLO not loaded"
            yolo_ms = round((time.perf_counter() - t_yolo) * 1000, 1)

            # Filter low-confidence YOLO detections before returning to orchestrator
            # (PMC12583037 + LiteVLM: reduces noise without losing high-signal detections)
            yolo_filtered = _filter_yolo_by_confidence(yolo_meta, min_conf=0.35)
            mp_person     = "person detected" in mp_full.lower()

            # Log sensor call (no inner LLM — orchestrator assesses from image + metadata)
            self._camera_rows.append({
                "run":                self._current_run,
                "call_num":           self._camera_call_num,
                "context":            context,
                "mp_risk":            mp_result.get("risk", "unknown"),
                "mp_full_response":   mp_full,
                "mp_ms":              mp_ms,
                "yolo_full_response": yolo_meta,
                "yolo_ms":            yolo_ms,
                "total_pipeline_ms":  round(mp_ms + yolo_ms, 1),
                "error":              "",
            })
            self._camera_call_num += 1

            print(f"  📷 [analyze_scene/{context}] "
                  f"mp_person={mp_person}  mp_risk={mp_result.get('risk','?')}  "
                  f"mp={mp_ms:.0f}ms  yolo={yolo_ms:.0f}ms  "
                  f"total={mp_ms+yolo_ms:.0f}ms")

            # Return raw sensor metadata — orchestrator assesses from image + this data
            return json.dumps({
                "context":            context,
                "mediapipe":          mp_full,
                "mp_person_detected": mp_person,
                "mp_risk":            mp_result.get("risk", "unknown"),
                "yolo_detections":    yolo_filtered,
                "pipeline_ms":        round(mp_ms + yolo_ms, 1),
            }, indent=2)

        # ── Drone control tools — log telemetry before + after ─────────────────
        if name in DRONE_CONTROL_TOOLS:
            tel_before = self._get_telemetry_snapshot()
            t0 = time.perf_counter()
            result = super().execute_tool(name, args)
            exec_ms = round((time.perf_counter() - t0) * 1000, 1)
            tel_after = self._get_telemetry_snapshot()

            self._workflow_rows.append({
                "run":               self._current_run,
                "action_num":        self._action_num,
                "tool_name":         name,
                "tool_args":         json.dumps(args),
                "telemetry_before":  json.dumps(tel_before),
                "telemetry_after":   json.dumps(tel_after),
                "execution_result":  result,   # full result — no truncation
                "execution_ms":      exec_ms,
                "is_drone_command":  1,
                "approved":          1,   # ND3 override sets 0 on rejection
            })
            self._action_num += 1
            return result

        return super().execute_tool(name, args)

    # ── Agentic loop with ND tools + optional human approval ──────────────────

    def run_nd_agent_loop(self, user_prompt: str,
                          history=None,
                          max_turns: int = MAX_TURNS,
                          max_tokens: int = 2048,
                          approve_callback=None) -> tuple:
        """
        Agent loop using ND_ALL_TOOLS and ND_SYSTEM_PROMPT.

        approve_callback (ND3 only):
            Signature: (tool_name: str, tool_args: dict, context_str: str)
                       -> (approved: bool, reason: str)
            Called before every DRONE_CONTROL_TOOLS call.
            Camera reads and telemetry reads are never gated.

        Returns: (final_text, api_stats, tool_trace)
        """
        # Include current camera frame in initial message (Anthropic image block).
        # The orchestrator LLM is vision-capable and assesses the image directly —
        # analyze_scene only provides supplementary sensor metadata.
        jpeg = self._get_frame()
        b64  = _encode_frame(jpeg)
        if b64:
            first_content = [
                {"type": "image",
                 "source": {"type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64}},
                {"type": "text", "text": user_prompt},
            ]
        else:
            first_content = user_prompt

        messages = list(history or [])
        messages.append({"role": "user", "content": first_content})

        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):

            # ── Advisory emergency check between turns ─────────────────────────
            # Inject the emergency frame that triggered MediaPipe alongside the
            # advisory text. LLM compares it to its working history and decides:
            # same frame → ignore; different frame → accept and investigate.
            emg_handled, emg_msg = self.handle_emergency_if_flagged()
            if emg_handled and emg_msg:
                emg_b64 = _encode_frame(self._last_emergency_frame)
                if emg_b64:
                    messages.append({"role": "user", "content": [
                        {"type": "image",
                         "source": {"type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": emg_b64}},
                        {"type": "text", "text": emg_msg},
                    ]})
                else:
                    messages.append({"role": "user", "content": emg_msg})

            payload = {
                "model":      _MODEL,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "system":     ND_SYSTEM_PROMPT,
                "messages":   messages,
                "tools":      ND_ALL_TOOLS,
            }
            headers = {
                "Content-Type":      "application/json",
                "Authorization":     f"Bearer {_API_KEY}",
                "anthropic-version": _VERSION,
            }

            t0 = time.time()
            try:
                req = urllib.request.Request(
                    _ENDPOINT,
                    data=json.dumps(payload).encode(),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read().decode())
            except Exception as e:
                print(f"[ND API ERROR turn {turn}] {e}")
                break

            latency = time.time() - t0
            usage   = resp.get("usage", {})
            in_tok  = usage.get("input_tokens",  0)
            out_tok = usage.get("output_tokens", 0)
            content     = resp.get("content", [])
            stop_reason = resp.get("stop_reason", "end_turn")
            turn_text   = ""
            for block in content:
                if block.get("type") == "text":
                    turn_text  = block["text"]
                    final_text = turn_text

            api_stats.append({
                "turn":          turn,
                "latency_s":     round(latency, 3),
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "cost_usd":      round(in_tok * COST_IN + out_tok * COST_OUT, 6),
                "text":          turn_text,
            })

            messages.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            if not tool_uses or stop_reason == "end_turn":
                break

            results = []
            for tu in tool_uses:
                t_name = tu["name"]
                t_args = tu.get("input", {})
                t_id   = tu["id"]

                # Human approval gate (ND3 only, drone commands only)
                approved    = True
                deny_reason = ""
                if (approve_callback is not None
                        and t_name in DRONE_CONTROL_TOOLS):
                    tel = self._get_telemetry_snapshot()
                    context_str = (
                        f"Proposed: {t_name}({json.dumps(t_args)})\n"
                        f"Telemetry: {json.dumps(tel)}"
                    )
                    approved, deny_reason = approve_callback(
                        t_name, t_args, context_str)

                    # Log approval decision (before execution)
                    self._workflow_rows.append({
                        "run":               self._current_run,
                        "action_num":        self._action_num,
                        "tool_name":         t_name,
                        "tool_args":         json.dumps(t_args),
                        "telemetry_before":  json.dumps(tel),
                        "telemetry_after":   "",
                        "execution_result":  "REJECTED" if not approved else "PENDING",
                        "execution_ms":      0,
                        "is_drone_command":  1,
                        "approved":          int(approved),
                    })
                    self._action_num += 1

                if not approved:
                    result = (f"[Human rejected '{t_name}': {deny_reason}. "
                              f"Choose a different action.]")
                    print(f"  ❌ [REJECTED] {t_name} — {deny_reason}")
                else:
                    print(f"  [TOOL ND t{turn}] "
                          f"{t_name}({json.dumps(t_args)[:60]})")
                    result = self.execute_tool(t_name, t_args)

                tool_trace.append({
                    "turn":       turn,
                    "name":       t_name,
                    "args":       t_args,
                    "result":     result,   # full result — no truncation
                    "approved":   int(approved),
                    "sim_time_s": round(self.sim_time, 2),
                })
                results.append({
                    "type":        "tool_result",
                    "tool_use_id": t_id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": results})

            # For real-time sources: inject updated frame after navigation commands
            if (self.frame_source != "saved"
                    and any(tu["name"] in NAV_TOOLS for tu in tool_uses)):
                new_b64 = _encode_frame(self._get_frame())
                if new_b64:
                    messages.append({"role": "user", "content": [
                        {"type": "image",
                         "source": {"type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": new_b64}},
                        {"type": "text",
                         "text": "Updated camera view after movement."},
                    ]})

        return final_text, api_stats, tool_trace

    # ── OpenAI orchestrator loop (GPT-4o / GPT-4o-mini) ───────────────────────

    def _run_openai_orchestrator_loop(self, orchestrator: str,
                                       user_prompt: str,
                                       max_turns: int,
                                       max_tokens: int) -> tuple:
        """
        Agentic loop using OpenAI function-calling API as the orchestrator.
        analyze_scene is registered as a function; the model calls it and we
        execute it, returning the result as a tool message.
        """
        from verbalization_utils import (
            OPENAI_API_KEY, OPENAI_BASE_URL,
            OPENAI_MINI_KEY, OPENAI_MINI_URL,
            GPT4O_MODEL,
        )
        import openai as _openai

        if orchestrator == "gpt4o_mini":
            client    = _openai.OpenAI(api_key=OPENAI_MINI_KEY,
                                       base_url=OPENAI_MINI_URL)
            oai_model = "gpt-4o-mini"
            cost_in, cost_out = 0.15e-6, 0.60e-6
        else:  # gpt4o
            kwargs = {"api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            client    = _openai.OpenAI(**kwargs)
            oai_model = GPT4O_MODEL
            cost_in, cost_out = 5e-6, 15e-6

        # All ND tools in OpenAI function format — drone control + analyze_scene.
        # Previously only analyze_scene was declared, making GPT-4o/mini unable
        # to fly the drone. Now all 30+ tools are declared for a fair comparison.
        oai_tools = _nd_tools_to_openai(ND_ALL_TOOLS)

        # Include current frame in initial message (OpenAI image_url format)
        jpeg = self._get_frame()
        b64  = _encode_frame(jpeg)
        if b64:
            first_content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                               "detail": "low"}},
                {"type": "text", "text": user_prompt},
            ]
        else:
            first_content = [{"type": "text", "text": user_prompt}]

        messages   = [
            {"role": "system", "content": ND_SYSTEM_PROMPT},
            {"role": "user",   "content": first_content},
        ]
        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):
            emg_handled, emg_msg = self.handle_emergency_if_flagged()
            if emg_handled and emg_msg:
                emg_b64 = _encode_frame(self._last_emergency_frame)
                if emg_b64:
                    messages.append({"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{emg_b64}",
                                       "detail": "low"}},
                        {"type": "text", "text": emg_msg},
                    ]})
                else:
                    messages.append({"role": "user", "content": emg_msg})

            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model       = oai_model,
                    messages    = messages,
                    tools       = oai_tools,
                    tool_choice = "auto",
                    max_tokens  = max_tokens,
                    temperature = 0.0,
                )
            except Exception as e:
                print(f"[OpenAI API ERROR turn {turn}] {e}")
                break

            latency = time.time() - t0
            usage   = resp.usage
            in_tok  = usage.prompt_tokens
            out_tok = usage.completion_tokens
            msg        = resp.choices[0].message
            finish     = resp.choices[0].finish_reason
            turn_text  = msg.content or ""
            final_text = turn_text

            api_stats.append({
                "turn":         turn,
                "latency_s":    round(latency, 3),
                "input_tokens": in_tok,
                "output_tokens":out_tok,
                "cost_usd":     round(in_tok * cost_in + out_tok * cost_out, 6),
                "text":         turn_text,
            })

            # Append assistant message (must include tool_calls if present)
            messages.append(msg)

            if not msg.tool_calls or finish == "stop":
                break

            for tc in msg.tool_calls:
                t_name = tc.function.name
                try:
                    t_args = json.loads(tc.function.arguments)
                except Exception:
                    t_args = {}
                print(f"  [TOOL OAI t{turn}] {t_name}({json.dumps(t_args)[:60]})")
                result = self.execute_tool(t_name, t_args)
                tool_trace.append({
                    "turn": turn, "name": t_name, "args": t_args,
                    "result": result,   # full result — no truncation
                    "approved": 1,
                    "sim_time_s": round(self.sim_time, 2),
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

            # For real-time sources: inject updated frame after navigation
            called_names = [tc.function.name for tc in msg.tool_calls]
            if (self.frame_source != "saved"
                    and any(n in NAV_TOOLS for n in called_names)):
                new_b64 = _encode_frame(self._get_frame())
                if new_b64:
                    messages.append({"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{new_b64}",
                                       "detail": "low"}},
                        {"type": "text", "text": "Updated camera view after movement."},
                    ]})

        return final_text, api_stats, tool_trace

    # ── Gemini orchestrator loop ───────────────────────────────────────────────

    def _run_gemini_orchestrator_loop(self, user_prompt: str,
                                       max_turns: int,
                                       max_tokens: int) -> tuple:
        """
        Agentic loop using Gemini REST API as the orchestrator.
        Uses function declarations + functionCall / functionResponse pattern.
        Follows the same REST approach as _call_gemini in verbalization_utils.
        """
        from verbalization_utils import GEMINI_API_KEY, GEMINI_MODEL

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        cost_in, cost_out = 0.075e-6, 0.30e-6

        # All ND tools in Gemini function declaration format — drone control + analyze_scene.
        # Previously only analyze_scene was declared. Gemini was calling undeclared
        # drone tools by reading the system prompt (prompt-driven), which is
        # unpredictable. Now all tools are declared explicitly for a fair comparison.
        gemini_tools = _nd_tools_to_gemini(ND_ALL_TOOLS)

        # Include current frame in initial message (Gemini inline_data format)
        jpeg = self._get_frame()
        b64  = _encode_frame(jpeg)
        if b64:
            first_parts = [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": user_prompt},
            ]
        else:
            first_parts = [{"text": user_prompt}]

        # Gemini uses a rolling `contents` list for multi-turn
        contents   = [{"role": "user", "parts": first_parts}]
        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):
            emg_handled, emg_msg = self.handle_emergency_if_flagged()
            if emg_handled and emg_msg:
                emg_b64 = _encode_frame(self._last_emergency_frame)
                if emg_b64:
                    contents.append({"role": "user", "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": emg_b64}},
                        {"text": emg_msg},
                    ]})
                else:
                    contents.append({"role": "user", "parts": [{"text": emg_msg}]})

            body = {
                "contents":         contents,
                "tools":            gemini_tools,
                "toolConfig":       {"functionCallingConfig": {"mode": "AUTO"}},
                "systemInstruction":{"parts": [{"text": ND_SYSTEM_PROMPT}]},
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature":     0.0,
                    "thinkingConfig":  {"thinkingBudget": 0},
                },
            }

            t0 = time.time()
            try:
                req = urllib.request.Request(
                    endpoint,
                    data    = json.dumps(body).encode(),
                    headers = {"Content-Type": "application/json"},
                    method  = "POST",
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = json.loads(r.read().decode())
            except Exception as e:
                print(f"[Gemini API ERROR turn {turn}] {e}")
                break

            latency = time.time() - t0
            usage   = raw.get("usageMetadata", {})
            in_tok  = usage.get("promptTokenCount",     0)
            out_tok = usage.get("candidatesTokenCount", 0)
            candidate  = raw.get("candidates", [{}])[0]
            parts      = candidate.get("content", {}).get("parts", [])
            text_parts = [p["text"] for p in parts if "text" in p]
            turn_text  = " ".join(text_parts) if text_parts else ""
            if turn_text:
                final_text = turn_text

            api_stats.append({
                "turn":         turn,
                "latency_s":    round(latency, 3),
                "input_tokens": in_tok,
                "output_tokens":out_tok,
                "cost_usd":     round(in_tok * cost_in + out_tok * cost_out, 6),
                "text":         turn_text,
            })

            # Check for function call
            fn_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not fn_calls:
                break  # Gemini is done

            # Append model turn to contents
            contents.append({"role": "model", "parts": parts})

            # Execute each function call, collect responses
            fn_responses = []
            for fc in fn_calls:
                t_name = fc.get("name", "")
                t_args = dict(fc.get("args", {}))
                print(f"  [TOOL GEMINI t{turn}] {t_name}({json.dumps(t_args)[:60]})")
                result = self.execute_tool(t_name, t_args)
                tool_trace.append({
                    "turn": turn, "name": t_name, "args": t_args,
                    "result": result,   # full result — no truncation
                    "approved": 1,
                    "sim_time_s": round(self.sim_time, 2),
                })
                fn_responses.append({
                    "functionResponse": {
                        "name":     t_name,
                        "response": {"output": result},
                    }
                })

            contents.append({"role": "user", "parts": fn_responses})

            # For real-time sources: inject updated frame after navigation
            called_names = [fc.get("name", "") for fc in fn_calls]
            if (self.frame_source != "saved"
                    and any(n in NAV_TOOLS for n in called_names)):
                new_b64 = _encode_frame(self._get_frame())
                if new_b64:
                    contents.append({"role": "user", "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": new_b64}},
                        {"text": "Updated camera view after movement."},
                    ]})

        return final_text, api_stats, tool_trace

    # ── Public dispatcher ──────────────────────────────────────────────────────

    def run_orchestrator_loop(self, orchestrator: str,
                               user_prompt: str,
                               history=None,
                               max_turns: int = MAX_TURNS,
                               max_tokens: int = 2048,
                               approve_callback=None) -> tuple:
        """
        Run the agentic loop with the specified orchestrator LLM.

        orchestrator : "claude" | "gpt4o" | "gpt4o_mini" | "gemini"

        All orchestrators have analyze_scene available as a native tool call.
        The inner vision model inside analyze_scene is fixed (self.vision_models[0]).

        Returns: (final_text, api_stats, tool_trace)
        """
        if orchestrator == "claude":
            return self.run_nd_agent_loop(
                user_prompt      = user_prompt,
                history          = history,
                max_turns        = max_turns,
                max_tokens       = max_tokens,
                approve_callback = approve_callback,
            )
        elif orchestrator in ("gpt4o", "gpt4o_mini"):
            return self._run_openai_orchestrator_loop(
                orchestrator = orchestrator,
                user_prompt  = user_prompt,
                max_turns    = max_turns,
                max_tokens   = max_tokens,
            )
        elif orchestrator == "gemini":
            return self._run_gemini_orchestrator_loop(
                user_prompt = user_prompt,
                max_turns   = max_turns,
                max_tokens  = max_tokens,
            )
        else:
            raise ValueError(f"Unknown orchestrator: {orchestrator!r}. "
                             f"Use 'claude', 'gpt4o', 'gpt4o_mini', or 'gemini'.")


# ── CSV field lists (shared by experiments) ────────────────────────────────────

CAMERA_CSV_FIELDS = [
    "run", "call_num", "context",
    "mp_risk", "mp_full_response", "mp_ms",
    "yolo_full_response", "yolo_ms",
    "total_pipeline_ms", "error",
]

WORKFLOW_CSV_FIELDS = [
    "run", "action_num", "tool_name", "tool_args",
    "telemetry_before", "telemetry_after",
    "execution_result", "execution_ms",
    "is_drone_command", "approved",
]

# Per-turn LLM metrics (C-series style) — saved to ND2_api_stats_<ts>.csv
LLM_STATS_CSV_FIELDS = [
    "orchestrator", "global_run", "scenario", "scene_label",
    "turn", "latency_s", "input_tokens", "output_tokens", "cost_usd", "text",
]
