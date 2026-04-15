#!/usr/bin/env python3
"""
ESP32-S3 Drone Controller – Local Web Server with AI Chat Agent

Serves the HTML controller UI and provides an AI assistant endpoint
that understands natural language flight commands.

Usage:
    python3 keyboard_server.py [port]

Default port: 8080

Requires:
    pip install requests   (or uses built-in urllib if not available)
"""

import http.server
import socketserver
import socket
import os
import sys
import json
import threading
import time
import base64
import urllib.request
import urllib.error

# ── Optional: tag AI commands in drone_sim.py CSV log ─────────────────────────
try:
    import drone_sim as _drone_sim_module
    _log_ai_event = _drone_sim_module.log_ai_event
except Exception:
    _log_ai_event = None   # Running against real hardware — no-op
# ──────────────────────────────────────────────────────────────────────────────

# ── Anthropic Claude API Configuration ────────────────────────────────────────
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY  = "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"
ANTHROPIC_MODEL    = "claude-sonnet-4-6"             # fast + capable
ANTHROPIC_VERSION  = "2023-06-01"
# ──────────────────────────────────────────────────────────────────────────────

def _anthropic_post(system: str, messages: list, tools: list = None,
                    max_tokens: int = 1024, temperature: float = 0.2) -> dict:
    """Low-level Anthropic API call. Returns parsed response dict."""
    payload = {
        "model":      ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   messages,
    }
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        ANTHROPIC_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type":    "application/json",
            "x-api-key":       ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[API] HTTP {e.code} error: {body}")
        raise

def _to_anthropic_tools(openai_tools: list) -> list:
    """Convert OpenAI tool format → Anthropic tool format."""
    out = []
    for t in openai_tools:
        fn = t["function"]
        out.append({
            "name":         fn["name"],
            "description":  fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out

def _extract_text(content_blocks: list) -> str:
    """Extract text from Anthropic content block list."""
    return " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text").strip()
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PORT = 8080
HTML_FILE    = "ESP32S3_DroneController.html"
SERVE_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── In-memory conversation histories (keyed by session_id) ────────────────────
_histories      = {}
_histories_lock = threading.Lock()
MAX_HISTORY_MESSAGES = 40   # keep last N messages to stay within context limit
# ──────────────────────────────────────────────────────────────────────────────

# ── Autonomy state ─────────────────────────────────────────────────────────────
_autonomy_lock   = threading.Lock()
_autonomy_thread = None
_autonomy_stop   = threading.Event()
_autonomy_status = {
    "running":        False,
    "goal":           "",
    "mode":           "",       # "human_loop" | "full_auto"
    "interval_sec":   3.0,
    "iteration":      0,
    "last_scene":     "",
    "last_action":    "",
    "pending_action": None,     # waiting for human approval (human_loop mode)
    "drone_ip":       "",
    "log":            [],       # last 20 log entries
}
_autonomy_approve_event = threading.Event()   # set by POST /ai/autonomy/approve
_autonomy_approve_value = None                # True = approved, False = rejected
# ──────────────────────────────────────────────────────────────────────────────

# ── Workflow state (background agentic execution) ─────────────────────────────
_workflow_lock   = threading.Lock()
_workflow_status = {
    "running":          False,
    "workflow_id":      None,
    "goal":             "",
    "plan":             [],        # list of step description strings declared by AI
    "current_step":     0,         # 1-based
    "total_steps":      0,
    "status":           "idle",    # idle | planning | executing | waiting | done | failed
    "log":              [],        # last 30 entries
    "reply":            "",        # final reply text when done
    "pending_commands": [],        # commands queued for browser to execute
}

def _workflow_log(msg: str):
    entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(f"[WF] {entry}")
    with _workflow_lock:
        _workflow_status["log"].append(entry)
        if len(_workflow_status["log"]) > 30:
            _workflow_status["log"].pop(0)

def _workflow_push_commands(cmds: list):
    """Push commands to browser immediately (async delivery via polling)."""
    if not cmds:
        return
    with _workflow_lock:
        _workflow_status["pending_commands"].extend(cmds)
# ──────────────────────────────────────────────────────────────────────────────

# ── Telemetry store ────────────────────────────────────────────────────────────
_telemetry_lock = threading.Lock()
_telemetry_buf  = []          # latest snapshot posted by the browser
# ──────────────────────────────────────────────────────────────────────────────

# ── Camera / Vision helpers ────────────────────────────────────────────────────
def fetch_frame(drone_ip: str):
    """Fetch one JPEG frame from the drone's /capture endpoint."""
    try:
        req = urllib.request.Request(
            f"http://{drone_ip}/capture",
            headers={"User-Agent": "DroneAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            return resp.read()
    except Exception as e:
        print(f"[CAM] fetch_frame failed: {e}")
        return None


def analyze_frame(frame_bytes: bytes, prompt: str) -> str:
    """Send a JPEG frame to Claude vision and return the text description."""
    b64 = base64.b64encode(frame_bytes).decode("utf-8")
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       b64,
                },
            },
            {"type": "text", "text": prompt},
        ],
    }]
    system = (
        "You are a drone vision assistant. Analyze the camera image and "
        "describe what you see concisely for autonomous navigation. "
        "Note obstacles, open space, surfaces, lighting, and any objects."
    )
    result = _anthropic_post(system, messages, max_tokens=300, temperature=0.1)
    return _extract_text(result.get("content", []))


# ── Flight data AI analysis ────────────────────────────────────────────────────
def analyze_telemetry(tel: list, task: str) -> str:
    """Send a telemetry snapshot to the LLM and return an analysis string."""
    if not tel:
        return "No telemetry data available yet. Fly the drone first."

    # Build a compact summary so we don't blow the token budget
    n   = len(tel)
    dur = (tel[-1]["t"] - tel[0]["t"]) / 1000.0 if n > 1 else 0

    def col(key):
        return [e.get(key, 0) for e in tel]

    def stats(vals):
        mn, mx = min(vals), max(vals)
        avg = sum(vals) / len(vals)
        return {"min": round(mn,2), "max": round(mx,2), "avg": round(avg,2)}

    summary = {
        "duration_sec": round(dur, 1),
        "samples":      n,
        "roll_deg":     stats(col("r")),
        "pitch_deg":    stats(col("p")),
        "yaw_deg":      stats(col("y")),
        "gyroX_dps":    stats(col("gx")),
        "gyroY_dps":    stats(col("gy")),
        "gyroZ_dps":    stats(col("gz")),
        "error_roll":   stats(col("er")),
        "error_pitch":  stats(col("ep")),
        "error_yaw":    stats(col("ey")),
        "throttle":     stats(col("ch1")),
        "motor1_pwm":   stats(col("m1")),
        "motor2_pwm":   stats(col("m2")),
        "motor3_pwm":   stats(col("m3")),
        "motor4_pwm":   stats(col("m4")),
    }

    # Count sign-flip oscillations in roll/pitch errors
    def count_flips(vals):
        flips, prev = 0, 0
        for v in vals:
            s = 1 if v > 0 else -1
            if prev and s != prev: flips += 1
            prev = s
        return flips

    summary["roll_error_flips"]  = count_flips(col("er"))
    summary["pitch_error_flips"] = count_flips(col("ep"))
    summary["yaw_error_flips"]   = count_flips(col("ey"))

    prompt = (
        f"Task: {task}\n\n"
        f"Flight telemetry summary ({n} samples, {dur:.1f} s):\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "Known drone: 7×7 cm, 50 g micro quad, brushed motors, cascade PID "
        "(angle outer loop → rate inner loop for roll/pitch, rate-only for yaw).\n"
        "Provide a concise expert analysis. Be specific with numbers."
    )

    system = (
        "You are an expert quadrotor flight dynamics and PID tuning engineer. "
        "Analyse telemetry data and give actionable, specific advice."
    )
    messages = [{"role": "user", "content": prompt}]
    result = _anthropic_post(system, messages, max_tokens=600, temperature=0.2)
    return _extract_text(result.get("content", []))


# ── Autonomy loop (runs in a background thread) ────────────────────────────────
def _autonomy_log(msg: str):
    entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(f"[AUTO] {entry}")
    with _autonomy_lock:
        _autonomy_status["log"].append(entry)
        if len(_autonomy_status["log"]) > 20:
            _autonomy_status["log"].pop(0)


def autonomy_loop(session_id: str, goal: str, mode: str,
                  drone_ip: str, interval_sec: float, drone_state: dict):
    """
    Background autonomy loop.
    Captures a frame, asks the vision model what to do next, then either
    waits for human approval (human_loop) or executes immediately (full_auto).
    Commands are appended to a shared queue that the browser polls via
    GET /ai/autonomy/status and executes via its existing executeAICommands().
    """
    global _autonomy_approve_value
    _autonomy_log(f"Started — goal='{goal}' mode={mode}")

    iteration = 0
    while not _autonomy_stop.is_set():
        iteration += 1
        with _autonomy_lock:
            _autonomy_status["iteration"] = iteration

        # 1. Auto-trim check: correct steady-state pitch/roll drift at hover
        with _telemetry_lock:
            recent_tel = list(_telemetry_buf[-20:])  # last ~2 s at 10 Hz
        if len(recent_tel) >= 10:
            pitch_stick = drone_state.get("pitch", 1500)
            roll_stick  = drone_state.get("roll",  1500)
            # Only trim when sticks are centred (no intentional pitch/roll command)
            if abs(pitch_stick - 1500) < 50 and abs(roll_stick - 1500) < 50:
                avg_p = sum(d.get("p", 0) for d in recent_tel) / len(recent_tel)
                avg_r = sum(d.get("r", 0) for d in recent_tel) / len(recent_tel)
                TRIM_THRESHOLD = 3.0   # degrees
                TRIM_STEP      = 3     # PWM per cycle
                TRIM_MAX       = 100   # ±100 PWM hard cap
                trim_fields = {}
                cur_pt = int(drone_state.get("trim_pitch", 0))
                cur_rt = int(drone_state.get("trim_roll",  0))
                if avg_p > TRIM_THRESHOLD:
                    new_pt = max(-TRIM_MAX, min(TRIM_MAX, cur_pt + TRIM_STEP))
                    trim_fields["trim-pitch"] = new_pt
                elif avg_p < -TRIM_THRESHOLD:
                    new_pt = max(-TRIM_MAX, min(TRIM_MAX, cur_pt - TRIM_STEP))
                    trim_fields["trim-pitch"] = new_pt
                if avg_r > TRIM_THRESHOLD:
                    new_rt = max(-TRIM_MAX, min(TRIM_MAX, cur_rt + TRIM_STEP))
                    trim_fields["trim-roll"] = new_rt
                elif avg_r < -TRIM_THRESHOLD:
                    new_rt = max(-TRIM_MAX, min(TRIM_MAX, cur_rt - TRIM_STEP))
                    trim_fields["trim-roll"] = new_rt
                if trim_fields:
                    _autonomy_log(f"Auto-trim: {trim_fields} (pitch_IMU={avg_p:.1f}° roll_IMU={avg_r:.1f}°)")
                    with _autonomy_lock:
                        _autonomy_status["approved_commands"] = [
                            {"action": "set_ui_fields", "fields": trim_fields}
                        ]
                    _autonomy_stop.wait(0.6)  # brief pause so trim applies before next action

        # 2. Capture frame
        frame = fetch_frame(drone_ip)
        if not frame:
            _autonomy_log("Frame capture failed — retrying next cycle")
            _autonomy_stop.wait(interval_sec)
            continue

        # 3. Analyse scene
        try:
            scene_prompt = (
                f"Goal: {goal}\n"
                "Given this drone camera view, describe what you see and "
                "suggest the single next action (move forward/back/left/right, "
                "ascend, descend, hover, or land). Be brief."
            )
            scene = analyze_frame(frame, scene_prompt)
        except Exception as e:
            _autonomy_log(f"Vision analysis failed: {e}")
            _autonomy_stop.wait(interval_sec)
            continue

        with _autonomy_lock:
            _autonomy_status["last_scene"] = scene

        _autonomy_log(f"Scene: {scene[:120]}")

        # 4. Decide action via LLM tool-calling
        with _telemetry_lock:
            tel_snap = list(_telemetry_buf[-20:])
        tel_note = ""
        if len(tel_snap) >= 5:
            avg_p = sum(d.get("p", 0) for d in tel_snap) / len(tel_snap)
            avg_r = sum(d.get("r", 0) for d in tel_snap) / len(tel_snap)
            tel_note = (
                f"\nLive telemetry (last 2 s avg): pitch_IMU={avg_p:.1f}°, roll_IMU={avg_r:.1f}°. "
                "If sticks are centred and either angle exceeds ±3°, call set_trim() to correct drift "
                "instead of a movement command."
            )
        action_prompt = (
            f"You are controlling a drone autonomously.\n"
            f"Goal: {goal}\n"
            f"Current scene: {scene}\n"
            f"Current drone state: {json.dumps(drone_state)}\n"
            f"{tel_note}\n"
            "Call exactly ONE flight tool to make the best next move toward the goal. "
            "Do NOT call get_drone_state or tuning tools. "
            "If the goal is complete, call land()."
        )
        try:
            reply, commands = run_agent(session_id, action_prompt, drone_state)
        except Exception as e:
            _autonomy_log(f"Agent failed: {e}")
            _autonomy_stop.wait(interval_sec)
            continue

        action_desc = reply[:120]
        with _autonomy_lock:
            _autonomy_status["last_action"] = action_desc

        _autonomy_log(f"Action: {action_desc}")

        if mode == "human_loop":
            # Park the commands and wait for browser approval
            with _autonomy_lock:
                _autonomy_status["pending_action"] = {
                    "description": action_desc,
                    "commands":    commands,
                    "scene":       scene,
                }
            _autonomy_approve_event.clear()
            approved = _autonomy_approve_event.wait(timeout=30)   # 30 s timeout
            with _autonomy_lock:
                decision = _autonomy_approve_value
                _autonomy_status["pending_action"] = None
            if approved and decision:
                _autonomy_log("Human approved — executing")
                # Commands are sent back in the next status poll — browser executes them
                with _autonomy_lock:
                    _autonomy_status["approved_commands"] = commands
            else:
                _autonomy_log("Human rejected / timed out — skipping")
        else:
            # full_auto: browser will pick up and execute via polling
            with _autonomy_lock:
                _autonomy_status["approved_commands"] = commands

        _autonomy_stop.wait(interval_sec)

    with _autonomy_lock:
        _autonomy_status["running"] = False
    _autonomy_log("Stopped")


# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AUTONOMOUS FLIGHT AGENT for a micro quadrotor drone (7 cm × 7 cm, 50 g, brushed motors). You execute multi-step flight workflows automatically — do NOT wait for user approval between steps.

━━ AGENTIC WORKFLOW PROTOCOL ━━
For EVERY multi-step request, follow this exact sequence:
  1. plan_workflow(goal, steps)  — declare your full plan FIRST, before any action
  2. report_progress(1, N, "description")  — before each step
  3. Execute the tool for that step
  4. Observe: use check_altitude_reached() / check_drone_stable() / get_sensor_status() after physical actions
  5. If observation fails → retry or adapt (wait longer, adjust params)
  6. report_progress(N, N, "Complete")  — when done

Example — "climb and hold at 1 meter":
  plan_workflow("climb and hold at 1m", ["Check sensors","Arm","Ramp throttle","Detect liftoff","Enable alt hold at hover","Step to 1m","Verify"])
  report_progress(1,7,"Checking sensors")         →  get_sensor_status()   ← confirm EKF altitude is tracking (not None)
  report_progress(2,7,"Arming")                   →  arm()  →  wait(1.0)
  report_progress(3,7,"Finding hover throttle")   →  find_hover_throttle()
    ← ramps from 1200 upward until airborne and vz≈0; returns exact hover PWM
  report_progress(4,7,"Confirming liftoff")       →  get_sensor_status()
    Once ekf_altitude_m > 0.10 and |vz_m_s| < 0.10: liftoff confirmed
  report_progress(5,7,"Enabling alt hold at hover altitude")
    →  enable_altitude_hold()   ← firmware captures current altitude + throttle as baseline
    →  wait(1.0)
  report_progress(6,7,"Stepping to 1m")
    →  set_altitude_target(0.5)  →  wait(3.0)   ← step to 0.5m first
    →  set_altitude_target(1.0)  →  wait(4.0)   ← then to 1m (max climb rate 0.4 m/s)
  report_progress(7,7,"Verifying")  →  check_altitude_reached(1.0, 0.15)

━━ DRONE CONTROL CHANNELS ━━
• ch1  Throttle   PWM 1000–2000  (1000=off, 2000=full, ~1500–1600=hover)
• ch2  Roll       PWM 1000–2000  (1500=level, 1000=left, 2000=right)
• ch3  Pitch      PWM 1000–2000  (1500=level, 1000=back, 2000=forward)
• ch4  Yaw        PWM 1000–2000  (1500=no spin, 1000=CCW, 2000=CW)
• ch5  Arm switch (1000=ARMED, 2000=DISARMED)

━━ CONNECTION ━━
Known drones (use automatically — never ask user):
  "sim" / "simulator" / "simulation" → localhost
  "nano drone"  → 10.198.219.186
  "micro drone" → 10.52.205.30
If drone_state.connected is already True, do NOT call connect_drone() — the user has already connected manually.
Only call connect_drone() if drone_state.connected is explicitly False AND the user has not already set an IP in the browser.

━━ ARM / TAKEOFF SEQUENCE ━━
1. connect_drone() if not connected
2. get_sensor_status() — confirm ekf_altitude_m is not None (sensors live)
3. arm() — sets ch5=1000, wait(1.0)
4. find_hover_throttle() — adaptively ramps throttle until drone lifts off and vz≈0.
   This finds the true hover thrust for the current battery state automatically.
   Do NOT use a fixed takeoff(hover_power=…) when precision or alt-hold is needed.
5. check_drone_stable() — confirm roll/pitch within 5°
NEVER assume the drone lifted off — always verify with ekf_altitude_m from telemetry.

━━ LANDING SEQUENCE ━━
1. disable_altitude_hold() + disable_position_hold() if active
2. hover() — centre all controls
3. Gradually reduce throttle: set_throttle(1400) → wait(1) → set_throttle(1200) → wait(1) → set_throttle(1100) → wait(1)
4. disarm()

━━ ALTITUDE & POSITION HOLD (LiteWing EKF) ━━
The drone uses a 9-state Kalman filter with VL53L1X ToF + PMW3901 optical flow.
  enable_altitude_hold()      — activates EKF altitude hold
                                IMPORTANT: firmware captures current altitude + throttle at the moment
                                of enable. So ONLY enable when drone is already stably hovering in air.
                                Do NOT enable on the ground.
  set_altitude_target(meters) — sets target (0.2–2.5 m). Firmware max climb rate = 0.4 m/s.
                                For large changes (>0.5m), step in increments: e.g. 0.5m → 1.0m.
  enable_position_hold()      — locks XY position at current EKF estimate
  disable_altitude_hold/position_hold() — returns to manual

CEILING SAFETY: firmware hard-limits setpoint to 2.5m and forces descent above 2.8m.
After enabling holds, ALWAYS wait(2.0) then check_altitude_reached() before declaring success.
Use get_sensor_status() to verify EKF altitude is tracking before enabling hold.

━━ OBSERVATION TOOLS ━━
  get_sensor_status()                           — reads live telemetry: EKF altitude, roll, pitch, hold status
  check_altitude_reached(target_m, tolerance_m) — returns ✓/✗ based on EKF Z state
  check_drone_stable(max_degrees=5.0)           — returns ✓/✗ based on roll/pitch

If check returns ✗: wait longer and re-check. After 3 failures, report the issue and stop.

━━ TRIM ━━
  set_trim(pitch_trim, roll_trim) — adds PWM offset to ch3/ch2. Range ±150.
  Positive pitch_trim → nose-down correction. Use when drone drifts forward at hover.

━━ TUNING PANEL ━━
Workflow for PID or LiteWing param change:
  1. get_tuning_params() — see current values (includes litewing_tuning section)
  2. set_tuning_params(param=value) — update UI field
  3. apply_tuning() — push to drone immediately

Attitude PID (roll_angle_kp/ki/kd, roll_rate_kp/ki/kd, pitch_*, yaw_rate_*):
  Motor comp (m1-m4_comp: 0.80–1.05), duty_idle (0–100), yaw_scale (0.1–2.0), loop_rate (100–2000)

LiteWing EKF & Hold (all via set_tuning_params + apply_tuning):
  lw_pidZ_kp/ki   Alt hold outer (pos→vel). Default 1.6/0.5. Reduce kp if overshoots.
  lw_pidVZ_kp/ki  Alt hold inner (vel→thrust). Default 0.50/0.30. HALVE kp if oscillating.
  lw_hover_thr    Throttle baseline 0.2–0.9. Lower if climbs on hold engage.
  lw_pidX_kp/ki   Pos hold outer. Default 1.9/0.1.
  lw_pidVX_kp/ki  Pos hold inner. Default 0.60/0.08. Halve if position oscillates.
  lw_kf_q         EKF Z process noise. Default 1.0.
  lw_kf_r         ToF measurement noise. Default 0.05. Lower if altitude lags sensor.
  lw_flow_std     Flow pixel noise. Default 2.0.

━━ SAFETY RULES ━━
• Never jump throttle suddenly — always ramp.
• Max safe throttle: 1800.
• Always check_drone_stable() after takeoff before enabling holds.
• Warn before applying PID changes while airborne.
• If check_drone_stable returns ✗ 3 times → land immediately.

━━ YOUR BEHAVIOUR ━━
• ALWAYS plan_workflow() first for multi-step tasks.
• ALWAYS report_progress() before each step.
• ALWAYS observe with check_* after physical actions.
• Chain tools without waiting for user — execute the full plan autonomously.
• Speak concisely at the end with what was accomplished.
"""

# ── Tool definitions ───────────────────────────────────────────────────────────
DRONE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_drone_state",
            "description": (
                "Returns the current drone state: connected, armed, throttle, roll, pitch, "
                "yaw, altitudeHold, positionHold, altitudeTarget."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arm",
            "description": (
                "Arm the drone motors. If throttle > 1050, the system will lower it first. "
                "Sets ch5 = 1000."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disarm",
            "description": "Disarm the drone motors (ch5 = 2000). Call only when landed.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "takeoff",
            "description": (
                "Full safe takeoff sequence: lower throttle if needed, arm, wait, ramp throttle to hover. "
                "Use this instead of individual arm/throttle calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hover_power": {
                        "type": "integer",
                        "description": "Target hover throttle PWM (1400–1700). Default 1550.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "land",
            "description": "Full safe landing sequence: center controls, ramp throttle down to zero, disarm.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hover",
            "description": "Center all roll/pitch/yaw to 1500 and keep current throttle for a stable hover.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emergency_stop",
            "description": "EMERGENCY ONLY — immediately disarm all motors. Drone will drop.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_throttle",
            "description": "Set throttle to a specific PWM value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pwm": {
                        "type": "integer",
                        "description": "Throttle PWM 1000–2000. Hover ≈ 1500–1600.",
                    }
                },
                "required": ["pwm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_yaw",
            "description": "Set yaw (rotation around vertical axis). Momentary — drone rotates while applied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pwm": {
                        "type": "integer",
                        "description": "Yaw PWM 1000–2000. 1500 = no rotation.",
                    }
                },
                "required": ["pwm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_roll",
            "description": "Set roll tilt for lateral (left/right) movement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pwm": {
                        "type": "integer",
                        "description": "Roll PWM 1000–2000. 1500 = level.",
                    }
                },
                "required": ["pwm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_pitch",
            "description": "Set pitch tilt for forward/backward movement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pwm": {
                        "type": "integer",
                        "description": "Pitch PWM 1000–2000. 1500 = level.",
                    }
                },
                "required": ["pwm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_altitude_hold",
            "description": "Enable altitude hold mode (rangefinder-based height maintenance).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_altitude_hold",
            "description": "Disable altitude hold — pilot controls altitude manually.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_position_hold",
            "description": "Enable position hold (optical flow). Drone stays in place horizontally.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_position_hold",
            "description": "Disable position hold.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_altitude_target",
            "description": "Set the target altitude for altitude hold mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meters": {
                        "type": "number",
                        "description": "Target altitude in metres (0.2–3.0).",
                    }
                },
                "required": ["meters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Pause for a specified duration before the next action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "Duration in seconds.",
                    }
                },
                "required": ["seconds"],
            },
        },
    },
    # ── Connection ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "connect_drone",
            "description": "Connect the controller to the ESP32-S3 drone via WebSocket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip_address": {
                        "type": "string",
                        "description": "IP address of the ESP32-S3 drone (e.g. '192.168.1.42').",
                    }
                },
                "required": ["ip_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disconnect_drone",
            "description": "Disconnect the controller from the drone.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Tuning ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_tuning_params",
            "description": (
                "Returns all current tuning parameters: motor compensation, duty_idle, "
                "yaw_scale, loop_rate, all attitude PID gains, and LiteWing EKF & hold params "
                "(litewing_tuning section: pidZ, pidVZ, pidX, pidVX, kf_q, kf_r, flow_std, hover_thr)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_tuning_params",
            "description": (
                "Update one or more tuning parameters in the UI. "
                "Only include the fields you want to change. "
                "Call apply_tuning() afterwards to push changes to the drone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "m1_comp":         {"type": "number", "description": "Motor 1 (Front-Right CCW) compensation, 0.80–1.05"},
                    "m2_comp":         {"type": "number", "description": "Motor 2 (Back-Right CW) compensation, 0.80–1.05"},
                    "m3_comp":         {"type": "number", "description": "Motor 3 (Back-Left CCW) compensation, 0.80–1.05"},
                    "m4_comp":         {"type": "number", "description": "Motor 4 (Front-Left CW) compensation, 0.80–1.05"},
                    "duty_idle":       {"type": "integer", "description": "Min motor PWM when armed (0–100)"},
                    "yaw_scale":       {"type": "number", "description": "Yaw mixer scaling (0.1–2.0)"},
                    "loop_rate":       {"type": "integer", "description": "Main loop rate in Hz (100–2000)"},
                    "roll_angle_kp":   {"type": "number"},
                    "roll_angle_ki":   {"type": "number"},
                    "roll_angle_kd":   {"type": "number"},
                    "roll_rate_kp":    {"type": "number"},
                    "roll_rate_ki":    {"type": "number"},
                    "roll_rate_kd":    {"type": "number"},
                    "pitch_angle_kp":  {"type": "number"},
                    "pitch_angle_ki":  {"type": "number"},
                    "pitch_angle_kd":  {"type": "number"},
                    "pitch_rate_kp":   {"type": "number"},
                    "pitch_rate_ki":   {"type": "number"},
                    "pitch_rate_kd":   {"type": "number"},
                    "yaw_rate_kp":     {"type": "number"},
                    "yaw_rate_ki":     {"type": "number"},
                    "yaw_rate_kd":     {"type": "number"},
                    # ── LiteWing EKF & Hold ─────────────────────────────────
                    "lw_pidZ_kp":  {"type": "number", "description": "Alt hold outer Kp — Z pos→vel (0–5, default 1.6). Reduce if altitude overshoots."},
                    "lw_pidZ_ki":  {"type": "number", "description": "Alt hold outer Ki (0–2, default 0.5)"},
                    "lw_pidVZ_kp": {"type": "number", "description": "Alt hold inner Kp — vel→thrust (0–5, default 0.50). Halve first if oscillating."},
                    "lw_pidVZ_ki": {"type": "number", "description": "Alt hold inner Ki (0–2, default 0.30)"},
                    "lw_hover_thr":{"type": "number", "description": "Hover throttle baseline 0.2–0.9 (default 0.50). Lower if drone climbs on hold engage."},
                    "lw_pidX_kp":  {"type": "number", "description": "Pos hold outer Kp — XY pos→vel (0–5, default 1.9). X and Y share same gain."},
                    "lw_pidX_ki":  {"type": "number", "description": "Pos hold outer Ki (0–2, default 0.1)"},
                    "lw_pidVX_kp": {"type": "number", "description": "Pos hold inner Kp — vel→roll/pitch (0–5, default 0.60). Halve if position oscillates."},
                    "lw_pidVX_ki": {"type": "number", "description": "Pos hold inner Ki (0–2, default 0.08)"},
                    "lw_kf_q":     {"type": "number", "description": "EKF Z process noise (0–10, default 1.0). Increase if altitude estimate is sluggish."},
                    "lw_kf_r":     {"type": "number", "description": "EKF ToF measurement noise in m (0.001–1.0, default 0.05). Decrease if altitude lags ToF."},
                    "lw_flow_std": {"type": "number", "description": "Optical flow pixel noise (0.1–20, default 2.0). Increase if XY estimate is jittery."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_tuning",
            "description": (
                "Send all current tuning parameters from the UI to the drone via WebSocket. "
                "Always call this after set_tuning_params()."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Takeoff helpers ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_takeoff_percent",
            "description": "Set the takeoff power percentage used by the UI auto-takeoff button (10–100 %).",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "integer",
                        "description": "Takeoff power 10–100 %.",
                    }
                },
                "required": ["percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_takeoff",
            "description": (
                "Trigger the UI's built-in auto-takeoff sequence at the saved takeoff percentage. "
                "Optionally set a new percentage first with set_takeoff_percent()."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_land",
            "description": "Trigger the UI's built-in auto-landing sequence.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Vision ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": (
                "Take a photo with the drone's camera and analyse it with AI vision. "
                "Returns a natural-language description of what the drone sees. "
                "Use this to answer 'what do you see?', check surroundings, or "
                "inform navigation decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What to look for or ask about the image. "
                            "Default: describe the scene for navigation."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    # ── Autonomy ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "start_autonomy",
            "description": (
                "Start the autonomous AI vision loop. The drone will repeatedly "
                "capture images, analyse them, and take actions toward the goal. "
                "Use mode='human_loop' for human approval of each action, or "
                "mode='full_auto' for fully autonomous operation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What the drone should try to achieve, e.g. 'explore the room' or 'find the red object'.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["human_loop", "full_auto"],
                        "description": "'human_loop' = propose each action and wait for approval. 'full_auto' = execute immediately.",
                    },
                    "interval_sec": {
                        "type": "number",
                        "description": "Seconds between each capture-analyse-act cycle (2–30). Default 5.",
                    },
                },
                "required": ["goal", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_autonomy",
            "description": "Stop the autonomous AI vision loop immediately.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_autonomy_status",
            "description": "Return the current status of the autonomy loop: running, goal, last scene, last action, pending approval, log.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Flight analysis ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_flight",
            "description": (
                "Analyse the most recent flight telemetry using AI. "
                "Returns a detailed report covering stability, oscillation, motor balance, "
                "and — critically — whether the observed drift/instability is a TRIM problem "
                "(steady-state angle offset → call set_trim) or a PID problem "
                "(oscillation / overshoot → call suggest_pid_tuning). "
                "Always call this first before deciding what to tune."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "How many seconds of recent telemetry to analyse (5–60). Default 30.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomaly",
            "description": (
                "Scan recent telemetry for anomalies: yaw spin, roll/pitch oscillation, "
                "motor imbalance, sudden throttle loss, or crash signatures. "
                "Returns a list of detected issues with severity."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_pid_tuning",
            "description": (
                "Analyse flight telemetry and suggest specific PID value changes. "
                "Only call this when telemetry shows OSCILLATION or OVERSHOOT — "
                "i.e. rapid error sign-flips or large underdamped response. "
                "If the problem is a steady-state drift/offset instead, use set_trim(). "
                "Returns exact param: old → new recommendations with reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["roll", "pitch", "yaw", "all"],
                        "description": "Which axis to tune. Default 'all'.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_crash",
            "description": (
                "Analyse the telemetry captured just before the last disarm/crash event. "
                "Identifies the likely cause: yaw runaway, oscillation, low battery, etc."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_trim",
            "description": (
                "Set pitch and/or roll trim offsets (PWM units added to ch3/ch2). "
                "Use after analysing telemetry to correct steady-state hover drift. "
                "Positive pitch_trim pushes nose forward; positive roll_trim tilts right. "
                "Range: -150 to +150 for each axis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pitch_trim": {
                        "type": "integer",
                        "description": "Pitch trim PWM offset (-150 to +150).",
                    },
                    "roll_trim": {
                        "type": "integer",
                        "description": "Roll trim PWM offset (-150 to +150).",
                    },
                },
                "required": [],
            },
        },
    },
    # ── Agentic workflow tools ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "plan_workflow",
            "description": (
                "Declare the full execution plan BEFORE any action. "
                "Required as the first call for any multi-step task. "
                "Shows the plan in the UI and sets up progress tracking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal":  {"type": "string", "description": "One-line description of the overall goal."},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of step descriptions (3–15 steps).",
                    },
                },
                "required": ["goal", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_progress",
            "description": "Report current step progress. Call before executing each step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step":        {"type": "integer", "description": "Current step number (1-based)."},
                    "total_steps": {"type": "integer", "description": "Total number of steps in the plan."},
                    "description": {"type": "string",  "description": "What this step is doing."},
                },
                "required": ["step", "total_steps", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sensor_status",
            "description": (
                "Read live sensor data from telemetry: EKF altitude (lw_z), roll, pitch, "
                "altitude hold active, position hold active, motor PWMs. "
                "Use before enabling holds to confirm sensors are live."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_altitude_reached",
            "description": (
                "Check if the drone's EKF altitude is within tolerance of target. "
                "Returns ✓ if reached, ✗ with current altitude if not. "
                "Use after set_altitude_target + wait to verify."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_m":     {"type": "number", "description": "Target altitude in metres."},
                    "tolerance_m":  {"type": "number", "description": "Acceptable error in metres (default 0.10)."},
                },
                "required": ["target_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_drone_stable",
            "description": (
                "Check if the drone is hovering stably by reading recent roll/pitch telemetry. "
                "Returns ✓ if both are within max_degrees, ✗ otherwise. "
                "Use after takeoff before enabling holds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_degrees": {"type": "number", "description": "Max acceptable roll or pitch in degrees (default 5.0)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_hover_throttle",
            "description": (
                "Adaptively ramp throttle upward from start_pwm until the drone lifts off and "
                "vertical velocity (vz) settles near zero — that is the true hover throttle for "
                "this battery state. Blocks until hover is found or max_pwm is reached. "
                "Use this instead of takeoff() when you need precise hover throttle capture "
                "before enabling altitude hold. Drone must already be armed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_pwm": {"type": "integer", "description": "PWM to begin ramping from (default 1200)."},
                    "max_pwm":   {"type": "integer", "description": "PWM ceiling — abort if reached without hover (default 1750)."},
                    "step_pwm":  {"type": "integer", "description": "PWM increment per step (default 20)."},
                    "step_wait_s": {"type": "number", "description": "Seconds to wait between steps for drone to respond (default 0.4)."},
                },
                "required": [],
            },
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict, drone_state: dict, session_id: str = "default"):
    """
    Execute one tool call.
    Returns (result_str, [command_dicts]).
    Command dicts are forwarded to the browser to be executed via WebSocket.
    """
    cmds = []

    # Tag AI tool calls in simulation CSV log
    if _log_ai_event and name not in ("get_drone_state", "get_telemetry_history"):
        _log_ai_event(f"{name}({json.dumps(args, separators=(',', ':'))})")

    if name == "get_drone_state":
        return json.dumps(drone_state), cmds

    if name == "arm":
        if drone_state.get("throttle", 1000) > 1050:
            cmds.append({"action": "set_throttle", "ch1": 1000})
            cmds.append({"action": "wait", "seconds": 0.3})
        cmds.append({"action": "arm"})
        return "Arm command queued (ch5=1000).", cmds

    if name == "disarm":
        cmds.append({"action": "disarm"})
        return "Disarm command queued (ch5=2000).", cmds

    if name == "takeoff":
        hover_power = max(1400, min(1700, int(args.get("hover_power", 1550))))
        already_armed = drone_state.get("armed", False)
        if not already_armed:
            cmds.append({"action": "set_throttle", "ch1": 1000})
            cmds.append({"action": "wait", "seconds": 0.2})
            cmds.append({"action": "arm"})
            cmds.append({"action": "wait", "seconds": 0.5})
        cmds.append({"action": "set_throttle", "ch1": 1200})
        cmds.append({"action": "wait", "seconds": 0.4})
        cmds.append({"action": "set_throttle", "ch1": hover_power})
        cmds.append({"action": "hover"})
        return f"Takeoff sequence queued. Hover power = {hover_power}.", cmds

    if name == "land":
        current = drone_state.get("throttle", 1500)
        cmds.append({"action": "hover"})
        for step in [1400, 1200, 1100, 1000]:
            if step < current:
                cmds.append({"action": "set_throttle", "ch1": step})
                cmds.append({"action": "wait", "seconds": 0.5})
        cmds.append({"action": "set_throttle", "ch1": 1000})
        cmds.append({"action": "wait", "seconds": 0.4})
        cmds.append({"action": "disarm"})
        return "Landing sequence queued.", cmds

    if name == "hover":
        cmds.append({"action": "hover"})
        return "Hover queued (roll/pitch/yaw = 1500).", cmds

    if name == "emergency_stop":
        cmds.append({"action": "emergency_stop"})
        return "EMERGENCY STOP queued.", cmds

    if name == "set_throttle":
        pwm = max(1000, min(2000, int(args.get("pwm", 1000))))
        cmds.append({"action": "set_throttle", "ch1": pwm})
        return f"Throttle → {pwm}.", cmds

    if name == "set_yaw":
        pwm = max(1000, min(2000, int(args.get("pwm", 1500))))
        cmds.append({"action": "set_yaw", "ch4": pwm})
        return f"Yaw → {pwm}.", cmds

    if name == "set_roll":
        pwm = max(1000, min(2000, int(args.get("pwm", 1500))))
        cmds.append({"action": "set_roll", "ch2": pwm})
        return f"Roll → {pwm}.", cmds

    if name == "set_pitch":
        pwm = max(1000, min(2000, int(args.get("pwm", 1500))))
        cmds.append({"action": "set_pitch", "ch3": pwm})
        return f"Pitch → {pwm}.", cmds

    if name == "enable_altitude_hold":
        cmds.append({"action": "enable_altitude_hold"})
        return "Altitude hold enabled.", cmds

    if name == "disable_altitude_hold":
        cmds.append({"action": "disable_altitude_hold"})
        return "Altitude hold disabled.", cmds

    if name == "enable_position_hold":
        cmds.append({"action": "enable_position_hold"})
        return "Position hold enabled.", cmds

    if name == "disable_position_hold":
        cmds.append({"action": "disable_position_hold"})
        return "Position hold disabled.", cmds

    if name == "set_altitude_target":
        meters = max(0.2, min(3.0, float(args.get("meters", 1.0))))
        cmds.append({"action": "set_altitude_target", "meters": meters})
        return f"Altitude target → {meters} m.", cmds

    if name == "wait":
        seconds = max(0.05, min(15.0, float(args.get("seconds", 1.0))))
        cmds.append({"action": "wait", "seconds": seconds})
        _workflow_push_commands(cmds)       # push to browser NOW before sleeping
        _workflow_log(f"Waiting {seconds}s…")
        with _workflow_lock:
            _workflow_status["status"] = "waiting"
        time.sleep(seconds)                 # real blocking wait in background thread
        with _workflow_lock:
            _workflow_status["status"] = "executing"
        return f"Waited {seconds}s.", []    # cmds already pushed above

    # ── Connection ────────────────────────────────────────────────────────────
    if name == "connect_drone":
        ip = args.get("ip_address", "").strip()
        if not ip:
            return "No IP address provided.", cmds
        cmds.append({"action": "connect_drone", "ip": ip})
        return f"Connecting to drone at {ip}.", cmds

    if name == "disconnect_drone":
        cmds.append({"action": "disconnect_drone"})
        return "Disconnecting from drone.", cmds

    # ── Tuning ────────────────────────────────────────────────────────────────
    if name == "get_tuning_params":
        tuning = drone_state.get("tuning", {})
        lw     = drone_state.get("litewing_tuning", {})
        if not tuning and not lw:
            return "No tuning data in state. Ask the user to open the tuning panel first.", cmds
        result = dict(tuning)
        if lw:
            result["litewing_tuning"] = lw
        return json.dumps(result), cmds

    if name == "set_tuning_params":
        # Map of tool arg → html element id and clamp (min, max)
        FIELD_MAP = {
            "m1_comp":        ("m1-comp",        0.80, 1.05),
            "m2_comp":        ("m2-comp",        0.80, 1.05),
            "m3_comp":        ("m3-comp",        0.80, 1.05),
            "m4_comp":        ("m4-comp",        0.80, 1.05),
            "duty_idle":      ("duty-idle",       0,   100),
            "yaw_scale":      ("yaw-scale",      0.1,  2.0),
            "loop_rate":      ("loop-rate",      100, 2000),
            "roll_angle_kp":  ("roll-angle-kp",  0.0,  5.0),
            "roll_angle_ki":  ("roll-angle-ki",  0.0,  1.0),
            "roll_angle_kd":  ("roll-angle-kd",  0.0,  1.0),
            "roll_rate_kp":   ("roll-rate-kp",   0.0,  5.0),
            "roll_rate_ki":   ("roll-rate-ki",   0.0,  1.0),
            "roll_rate_kd":   ("roll-rate-kd",   0.0,  1.0),
            "pitch_angle_kp": ("pitch-angle-kp", 0.0,  5.0),
            "pitch_angle_ki": ("pitch-angle-ki", 0.0,  1.0),
            "pitch_angle_kd": ("pitch-angle-kd", 0.0,  1.0),
            "pitch_rate_kp":  ("pitch-rate-kp",  0.0,  5.0),
            "pitch_rate_ki":  ("pitch-rate-ki",  0.0,  1.0),
            "pitch_rate_kd":  ("pitch-rate-kd",  0.0,  1.0),
            "yaw_rate_kp":    ("yaw-rate-kp",    0.0,  5.0),
            "yaw_rate_ki":    ("yaw-rate-ki",    0.0,  1.0),
            "yaw_rate_kd":    ("yaw-rate-kd",    0.0,  1.0),
            # ── LiteWing EKF & Hold ──────────────────────────────────────────
            "lw_pidZ_kp":     ("lw-pidz-kp",     0.0,  5.0),
            "lw_pidZ_ki":     ("lw-pidz-ki",     0.0,  2.0),
            "lw_pidVZ_kp":    ("lw-pidvz-kp",    0.0,  5.0),
            "lw_pidVZ_ki":    ("lw-pidvz-ki",    0.0,  2.0),
            "lw_hover_thr":   ("lw-hover-thr",   0.2,  0.9),
            "lw_pidX_kp":     ("lw-pidx-kp",     0.0,  5.0),
            "lw_pidX_ki":     ("lw-pidx-ki",     0.0,  2.0),
            "lw_pidVX_kp":    ("lw-pidvx-kp",    0.0,  5.0),
            "lw_pidVX_ki":    ("lw-pidvx-ki",    0.0,  2.0),
            "lw_kf_q":        ("lw-kf-q",        0.0, 10.0),
            "lw_kf_r":        ("lw-kf-r",      0.001,  1.0),
            "lw_flow_std":    ("lw-flow-std",    0.1, 20.0),
        }
        fields = {}
        for key, (elem_id, lo, hi) in FIELD_MAP.items():
            if key in args and args[key] is not None:
                fields[elem_id] = max(lo, min(hi, float(args[key])))
        if not fields:
            return "No tuning params provided.", cmds
        cmds.append({"action": "set_ui_fields", "fields": fields})
        changed = ", ".join(f"{k}={v}" for k, v in fields.items())
        return f"UI fields updated: {changed}. Call apply_tuning() to push to drone.", cmds

    if name == "apply_tuning":
        cmds.append({"action": "apply_tuning"})
        return "Tuning parameters sent to drone via WebSocket.", cmds

    # ── Takeoff helpers ────────────────────────────────────────────────────────
    if name == "set_takeoff_percent":
        pct = max(10, min(100, int(args.get("percent", 60))))
        cmds.append({"action": "set_ui_fields", "fields": {"takeoff-percent": pct}})
        return f"Takeoff percent set to {pct} %.", cmds

    if name == "auto_takeoff":
        cmds.append({"action": "auto_takeoff"})
        return "UI auto-takeoff sequence triggered.", cmds

    if name == "auto_land":
        cmds.append({"action": "auto_land"})
        return "UI auto-landing sequence triggered.", cmds

    # ── Vision ────────────────────────────────────────────────────────────────
    if name == "capture_image":
        drone_ip = drone_state.get("droneIP", "").strip()
        if not drone_ip:
            return "No drone IP in state — connect to the drone first.", cmds
        frame = fetch_frame(drone_ip)
        if not frame:
            return f"Could not capture image from {drone_ip}. Is the camera enabled?", cmds
        prompt = args.get("prompt") or (
            "Describe what this drone camera sees. Focus on obstacles, open space, "
            "ground features, and anything relevant for navigation."
        )
        try:
            description = analyze_frame(frame, prompt)
            return description, cmds
        except Exception as e:
            return f"Vision analysis failed: {e}", cmds

    # ── Autonomy ──────────────────────────────────────────────────────────────
    if name == "start_autonomy":
        global _autonomy_thread, _autonomy_stop, _autonomy_approve_value
        goal         = args.get("goal", "explore surroundings")
        mode         = args.get("mode", "human_loop")
        interval_sec = max(2.0, min(30.0, float(args.get("interval_sec", 5.0))))
        drone_ip     = drone_state.get("droneIP", "").strip()
        if not drone_ip:
            return "No drone IP in state — connect first.", cmds
        with _autonomy_lock:
            if _autonomy_status["running"]:
                return "Autonomy loop already running. Call stop_autonomy() first.", cmds
            _autonomy_status.update({
                "running": True, "goal": goal, "mode": mode,
                "interval_sec": interval_sec, "iteration": 0,
                "last_scene": "", "last_action": "", "pending_action": None,
                "drone_ip": drone_ip, "log": [],
                "approved_commands": [],
            })
        _autonomy_stop.clear()
        _autonomy_approve_value = None
        _autonomy_thread = threading.Thread(
            target=autonomy_loop,
            args=(session_id, goal, mode, drone_ip, interval_sec, drone_state),
            daemon=True,
        )
        _autonomy_thread.start()
        return (
            f"Autonomy loop started — goal='{goal}' mode={mode} "
            f"interval={interval_sec}s drone={drone_ip}."
        ), cmds

    if name == "stop_autonomy":
        _autonomy_stop.set()
        with _autonomy_lock:
            _autonomy_status["running"] = False
            _autonomy_status["pending_action"] = None
        return "Autonomy loop stop signal sent.", cmds

    if name == "get_autonomy_status":
        with _autonomy_lock:
            snap = dict(_autonomy_status)
        snap.pop("approved_commands", None)   # internal field, not needed here
        return json.dumps(snap), cmds

    # ── Flight analysis ───────────────────────────────────────────────────────
    if name == "analyze_flight":
        seconds = max(5, min(60, int(args.get("seconds", 30))))
        with _telemetry_lock:
            n = seconds * 10
            tel = _telemetry_buf[-n:] if len(_telemetry_buf) >= 5 else list(_telemetry_buf)
        if len(tel) < 5:
            return "Not enough telemetry yet — fly the drone for a few seconds first.", cmds
        try:
            result = analyze_telemetry(tel, "Provide a complete flight health report.")
            return result, cmds
        except Exception as e:
            return f"Analysis failed: {e}", cmds

    if name == "detect_anomaly":
        with _telemetry_lock:
            tel = _telemetry_buf[-300:]   # last 30 s
        if len(tel) < 10:
            return "Not enough telemetry yet.", cmds
        try:
            result = analyze_telemetry(
                tel,
                "Detect and list any flight anomalies (yaw spin, oscillations, "
                "motor imbalance, vibration, instability). Rate each as LOW/MED/HIGH severity. "
                "If no anomalies found, say 'No anomalies detected'.",
            )
            return result, cmds
        except Exception as e:
            return f"Anomaly detection failed: {e}", cmds

    if name == "suggest_pid_tuning":
        axis = args.get("axis", "all")
        with _telemetry_lock:
            tel = _telemetry_buf[-300:]
        if len(tel) < 10:
            return "Not enough telemetry yet — fly first.", cmds
        try:
            current_pids  = drone_state.get("tuning", {})
            current_trim  = {
                "trim_pitch": drone_state.get("trim_pitch", 0),
                "trim_roll":  drone_state.get("trim_roll",  0),
            }
            task = (
                f"Analyse this flight telemetry for axis='{axis}'. "
                f"Current PID values: {json.dumps(current_pids)}. "
                f"Current trim values: {json.dumps(current_trim)}. "
                "FIRST determine whether the problem is:\n"
                "  A) TRIM (steady constant pitch_IMU/roll_IMU offset, low variance) → recommend set_trim values.\n"
                "  B) PID (oscillation: rapid error sign-flips, high variance) → recommend PID changes.\n"
                "  C) Both → recommend trim fix first, then PID changes.\n"
                "Give exact numerical recommendations with brief reasoning. "
                "Format PID changes as: param_name: old → new (reason). "
                "Format trim changes as: trim_pitch/trim_roll: old → new (reason)."
            )
            result = analyze_telemetry(tel, task)
            return result, cmds
        except Exception as e:
            return f"PID suggestion failed: {e}", cmds

    if name == "analyze_crash":
        with _telemetry_lock:
            tel = list(_telemetry_buf[-100:])   # last 10 s as crash snapshot
        if len(tel) < 5:
            return "No crash telemetry captured yet. This updates automatically at disarm.", cmds
        try:
            result = analyze_telemetry(
                tel,
                "Analyse this pre-crash/pre-disarm telemetry. "
                "Identify the most likely cause of the crash or emergency disarm. "
                "Was it yaw runaway? Oscillation? Loss of control? Pilot error? "
                "Be specific about which signals indicate the problem.",
            )
            return result, cmds
        except Exception as e:
            return f"Crash analysis failed: {e}", cmds

    if name == "set_trim":
        pitch_trim = args.get("pitch_trim")
        roll_trim  = args.get("roll_trim")
        fields = {}
        parts  = []
        if pitch_trim is not None:
            pt = max(-150, min(150, int(pitch_trim)))
            fields["trim-pitch"] = pt
            parts.append(f"pitch trim → {pt}")
        if roll_trim is not None:
            rt = max(-150, min(150, int(roll_trim)))
            fields["trim-roll"] = rt
            parts.append(f"roll trim → {rt}")
        if fields:
            cmds.append({"action": "set_ui_fields", "fields": fields})
            return f"Trim updated: {', '.join(parts)}", cmds
        return "No trim values provided.", cmds

    # ── Agentic workflow tools ────────────────────────────────────────────────
    if name == "plan_workflow":
        goal  = args.get("goal", "")
        steps = args.get("steps", [])
        with _workflow_lock:
            _workflow_status["goal"]        = goal
            _workflow_status["plan"]        = list(steps)
            _workflow_status["total_steps"] = len(steps)
            _workflow_status["current_step"]= 0
            _workflow_status["status"]      = "executing"
        _workflow_log(f"Plan: {goal} ({len(steps)} steps)")
        return f"Plan registered ({len(steps)} steps): " + " → ".join(steps[:5]) + ("…" if len(steps) > 5 else ""), cmds

    if name == "report_progress":
        step  = int(args.get("step", 0))
        total = int(args.get("total_steps", 0))
        desc  = args.get("description", "")
        with _workflow_lock:
            _workflow_status["current_step"] = step
            _workflow_status["total_steps"]  = total
            _workflow_status["status"]       = "executing" if step < total else "done"
        _workflow_log(f"[{step}/{total}] {desc}")
        return f"Progress {step}/{total}: {desc}", cmds

    if name == "get_sensor_status":
        with _telemetry_lock:
            recent = list(_telemetry_buf[-5:])
        if not recent:
            return "No telemetry data — browser must be connected and sending data.", cmds
        last = recent[-1]
        lw_z_mm = last.get("lw_z", None)
        result = {
            "ekf_altitude_m":  round(lw_z_mm / 1000.0, 3) if lw_z_mm is not None else None,
            "vz_m_s":          round(last.get("vz", 0.0), 3),
            "althold_active":  bool(last.get("althold", 0)),
            "poshold_active":  bool(last.get("poshold", 0)),
            "roll_deg":        round(last.get("r", 0), 1),
            "pitch_deg":       round(last.get("p", 0), 1),
            "motor1_pwm":      last.get("m1", 0),
            "motor2_pwm":      last.get("m2", 0),
            "motor3_pwm":      last.get("m3", 0),
            "motor4_pwm":      last.get("m4", 0),
        }
        return json.dumps(result), cmds

    if name == "check_altitude_reached":
        target_m    = float(args.get("target_m", 1.0))
        tolerance_m = float(args.get("tolerance_m", 0.10))
        with _telemetry_lock:
            recent = list(_telemetry_buf[-5:])
        if not recent:
            return "No telemetry data available yet.", cmds
        lw_z_vals = [d["lw_z"] for d in recent if d.get("lw_z") is not None]
        if not lw_z_vals:
            return "No EKF altitude in telemetry. Is altitude hold active? Is VL53L1X connected?", cmds
        current_m = (sum(lw_z_vals) / len(lw_z_vals)) / 1000.0
        diff = abs(current_m - target_m)
        if diff <= tolerance_m:
            return f"✓ Altitude reached: {current_m:.2f}m (target={target_m}m, error={diff:.3f}m)", cmds
        return f"✗ Not there yet: current={current_m:.2f}m target={target_m}m error={diff:.3f}m", cmds

    if name == "check_drone_stable":
        max_deg = float(args.get("max_degrees", 5.0))
        with _telemetry_lock:
            recent = list(_telemetry_buf[-10:])
        if not recent:
            return "No telemetry data available yet.", cmds
        avg_roll  = abs(sum(d.get("r", 0) for d in recent) / len(recent))
        avg_pitch = abs(sum(d.get("p", 0) for d in recent) / len(recent))
        if avg_roll <= max_deg and avg_pitch <= max_deg:
            return f"✓ Stable: roll={avg_roll:.1f}° pitch={avg_pitch:.1f}° (max={max_deg}°)", cmds
        return f"✗ Unstable: roll={avg_roll:.1f}° pitch={avg_pitch:.1f}° (max={max_deg}°)", cmds

    if name == "find_hover_throttle":
        start_pwm  = max(1100, min(1600, int(args.get("start_pwm",   1200))))
        max_pwm    = max(1400, min(1800, int(args.get("max_pwm",     1750))))
        step_pwm   = max(5,    min(50,   int(args.get("step_pwm",      20))))
        step_wait  = max(0.2,  min(2.0, float(args.get("step_wait_s",  0.4))))

        _workflow_log(f"find_hover_throttle: ramp {start_pwm}→{max_pwm} step={step_pwm} wait={step_wait}s")

        # Push starting throttle and let drone settle
        _workflow_push_commands([{"action": "set_throttle", "ch1": start_pwm}])
        time.sleep(0.6)

        current_pwm = start_pwm
        hover_pwm   = None

        while current_pwm <= max_pwm:
            # Read latest telemetry from server-side buffer
            with _telemetry_lock:
                recent = list(_telemetry_buf[-4:])

            if recent:
                last   = recent[-1]
                lw_z   = last.get("lw_z")
                alt_m  = (lw_z / 1000.0) if lw_z is not None else 0.0
                vz     = last.get("vz", 0.0)

                _workflow_log(f"  PWM={current_pwm}  alt={alt_m:.3f}m  vz={vz:+.3f}m/s")

                if alt_m > 0.08:
                    # Airborne — now hunt for vz ≈ 0
                    if vz > 0.12:
                        # Still climbing — back off one step
                        current_pwm -= step_pwm
                        current_pwm  = max(start_pwm, current_pwm)
                        _workflow_push_commands([{"action": "set_throttle", "ch1": current_pwm}])
                        time.sleep(step_wait)
                        continue
                    elif vz < -0.10:
                        # Descending — add one step
                        current_pwm += step_pwm
                        _workflow_push_commands([{"action": "set_throttle", "ch1": current_pwm}])
                        time.sleep(step_wait)
                        continue
                    else:
                        # |vz| < 0.10 m/s while airborne → this is hover
                        hover_pwm = current_pwm
                        break

            # Not airborne yet — step up
            current_pwm += step_pwm
            _workflow_push_commands([{"action": "set_throttle", "ch1": current_pwm}])
            time.sleep(step_wait)

        if hover_pwm is None:
            return (f"✗ Drone did not lift off below {max_pwm} PWM. "
                    f"Check motors, props, and battery. Last PWM tried: {current_pwm}."), cmds

        _workflow_log(f"Hover throttle found: {hover_pwm} PWM")
        return (f"✓ Hover throttle found: {hover_pwm} PWM  "
                f"(alt={alt_m:.2f}m, vz={vz:+.3f}m/s). "
                f"Drone is airborne and stable — safe to enable_altitude_hold()."), cmds

    return f"Unknown tool: {name}", cmds


# ── Agentic loop (Anthropic Claude API) ───────────────────────────────────────
_ANTHROPIC_TOOLS = _to_anthropic_tools(DRONE_TOOLS)

def run_agent(session_id: str, user_message: str, drone_state: dict):
    """
    Full agentic tool-calling loop using Anthropic Claude API.
    Runs to completion (up to 30 iterations).
    Pushes commands to _workflow_status["pending_commands"] after each tool call
    so the browser can execute them in real-time while the agent continues working.
    Returns (reply_text, all_commands) for backward compatibility.
    """
    with _histories_lock:
        if session_id not in _histories:
            _histories[session_id] = []
        history = _histories[session_id]

    state_note = f"\n[Drone state: {json.dumps(drone_state)}]"
    history.append({"role": "user", "content": user_message + state_note})

    all_commands = []

    for iteration in range(30):
        response = _anthropic_post(
            system=SYSTEM_PROMPT,
            messages=history,
            tools=_ANTHROPIC_TOOLS,
            max_tokens=1024,
            temperature=0.2,
        )

        content_blocks = response.get("content", [])
        stop_reason    = response.get("stop_reason", "end_turn")

        # Append assistant message to history
        history.append({"role": "assistant", "content": content_blocks})

        # Extract tool calls
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_uses or stop_reason == "end_turn":
            reply_text = _extract_text(content_blocks) or "Done."
            with _histories_lock:
                if len(history) > MAX_HISTORY_MESSAGES:
                    _histories[session_id] = history[-(MAX_HISTORY_MESSAGES):]
            with _workflow_lock:
                _workflow_status["status"] = "done"
                _workflow_status["reply"]  = reply_text
            _workflow_log(f"Agent done: {reply_text[:80]}")
            return reply_text, all_commands

        # Execute all tool calls and collect results
        tool_results = []
        for tu in tool_uses:
            tool_name = tu["name"]
            tool_args = tu.get("input", {})
            tool_id   = tu["id"]

            print(f"  [tool] {tool_name}({tool_args})")
            result_str, cmds = execute_tool(tool_name, tool_args, drone_state, session_id)

            if tool_name != "wait" and cmds:
                _workflow_push_commands(cmds)
            all_commands.extend(cmds)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tool_id,
                "content":     result_str,
            })

        # Append tool results as a user message (Anthropic format)
        history.append({"role": "user", "content": tool_results})

    with _workflow_lock:
        _workflow_status["status"] = "done"
    return "Workflow completed.", all_commands


def _run_agent_background(session_id: str, user_message: str, drone_state: dict, workflow_id: str):
    """Runs run_agent in a background thread. Updates _workflow_status on completion."""
    try:
        _workflow_log(f"Starting workflow: {user_message[:60]}")
        reply, _ = run_agent(session_id, user_message, drone_state)
        with _workflow_lock:
            _workflow_status["running"] = False
            _workflow_status["status"]  = "done"
            _workflow_status["reply"]   = reply
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _workflow_lock:
            _workflow_status["running"] = False
            _workflow_status["status"]  = "failed"
            _workflow_status["reply"]   = f"Error: {e}"
        _workflow_log(f"Workflow failed: {e}")


# ── HTTP handler ───────────────────────────────────────────────────────────────
class DroneControllerHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    # ── GET: serve static files + autonomy status ─────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/" + HTML_FILE
            return super().do_GET()
        if self.path == "/ai/autonomy/status":
            self._handle_autonomy_status()
            return
        if self.path == "/ai/workflow/status":
            self._handle_workflow_status()
            return
        return super().do_GET()

    # ── OPTIONS: CORS preflight ────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ── POST: AI chat + autonomy control ──────────────────────────────────────
    def do_POST(self):
        if self.path == "/ai/chat":
            self._handle_ai_chat()
        elif self.path == "/ai/reset":
            self._handle_ai_reset()
        elif self.path == "/ai/autonomy/approve":
            self._handle_autonomy_approve(approved=True)
        elif self.path == "/ai/autonomy/reject":
            self._handle_autonomy_approve(approved=False)
        elif self.path == "/ai/telemetry":
            self._handle_telemetry()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_ai_chat(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body)

            user_message = data.get("message", "").strip()
            drone_state  = data.get("state", {})
            session_id   = data.get("session_id", "default")

            if not user_message:
                raise ValueError("Empty message")

            print(f"\n[AI] [{session_id}] User: {user_message}")

            workflow_id = f"wf_{int(time.time() * 1000)}"

            # Initialise workflow status for this run
            with _workflow_lock:
                _workflow_status["running"]          = True
                _workflow_status["workflow_id"]      = workflow_id
                _workflow_status["goal"]             = user_message
                _workflow_status["plan"]             = []
                _workflow_status["current_step"]     = 0
                _workflow_status["total_steps"]      = 0
                _workflow_status["status"]           = "planning"
                _workflow_status["log"]              = []
                _workflow_status["reply"]            = ""
                _workflow_status["pending_commands"] = []

            # Start agent in background thread — returns HTTP immediately
            t = threading.Thread(
                target=_run_agent_background,
                args=(session_id, user_message, drone_state, workflow_id),
                daemon=True,
            )
            t.start()

            self._json_response({
                "reply":       "⏳ Working on it…",
                "commands":    [],
                "workflow_id": workflow_id,
                "async":       True,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json_response({"reply": f"Server error: {e}", "commands": []}, 500)

    def _handle_ai_reset(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body) if length else {}
            session_id = data.get("session_id", "default")
            with _histories_lock:
                _histories.pop(session_id, None)
            self._json_response({"ok": True})
        except Exception as e:
            self._json_response({"ok": False, "error": str(e)}, 500)

    def _handle_telemetry(self):
        """POST /ai/telemetry — browser pushes its telemetry buffer here."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body)
            tel    = data.get("telemetry", [])
            with _telemetry_lock:
                _telemetry_buf.clear()
                _telemetry_buf.extend(tel)
            self._json_response({"ok": True, "stored": len(tel)})
        except Exception as e:
            self._json_response({"ok": False, "error": str(e)}, 500)

    def _handle_autonomy_status(self):
        """GET /ai/autonomy/status — returns current autonomy state.
        Also flushes any approved_commands so the browser can execute them."""
        with _autonomy_lock:
            snap = dict(_autonomy_status)
            commands = snap.pop("approved_commands", [])
            _autonomy_status["approved_commands"] = []   # clear after delivery
        self._json_response({"status": snap, "commands": commands})

    def _handle_autonomy_approve(self, approved: bool):
        """POST /ai/autonomy/approve  or  /ai/autonomy/reject"""
        global _autonomy_approve_value
        _autonomy_approve_value = approved
        _autonomy_approve_event.set()
        self._json_response({"ok": True, "approved": approved})

    def _handle_workflow_status(self):
        """GET /ai/workflow/status — returns workflow progress + pending commands for browser."""
        with _workflow_lock:
            snap = {k: v for k, v in _workflow_status.items() if k != "pending_commands"}
            commands = list(_workflow_status["pending_commands"])
            _workflow_status["pending_commands"] = []   # flush after delivery
        self._json_response({"status": snap, "commands": commands})

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress noisy asset requests
        if any(self.path.endswith(ext) for ext in (".ico", ".png", ".jpg", ".gif", ".css")):
            return
        super().log_message(fmt, *args)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT

    html_path = os.path.join(SERVE_DIR, HTML_FILE)
    if not os.path.exists(html_path):
        print(f"ERROR: '{HTML_FILE}' not found in:\n  {SERVE_DIR}")
        sys.exit(1)

    if ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY_HERE":
        print("⚠  WARNING: ANTHROPIC_API_KEY is not set in keyboard_server.py")
        print("   Get your key from console.anthropic.com → API Keys")
        print("   Then set ANTHROPIC_API_KEY at the top of keyboard_server.py\n")

    local_ip = get_local_ip()
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), DroneControllerHandler) as httpd:
        print()
        print("╔══════════════════════════════════════════════════╗")
        print("║   ESP32-S3 Drone Controller + AI Assistant       ║")
        print("╠══════════════════════════════════════════════════╣")
        print(f"║  Local   →  http://localhost:{port:<21}║")
        print(f"║  Network →  http://{local_ip}:{port:<{28 - len(local_ip)}}║")
        print("╠══════════════════════════════════════════════════╣")
        print("║  AI endpoint →  POST /ai/chat                    ║")
        print("║  Reset chat  →  POST /ai/reset                   ║")
        print("╠══════════════════════════════════════════════════╣")
        print("║  Press  Ctrl+C  to stop.                         ║")
        print("╚══════════════════════════════════════════════════╝")
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
