"""
C-Series Experiment Infrastructure — c_series_agent.py
=======================================================
SimAgent: drone_sim-backed tool executor + Claude API agent runner.
Imported by all exp_C*_*.py scripts.

Architecture:
  - DroneState + PhysicsLoop from drone_sim run synchronously (200 Hz)
  - Tool calls map directly to sim state mutations + physics ticks
  - Claude API called via urllib (no extra SDK dependency)
  - Records telemetry, tool call trace, and API stats per experiment

Usage:
    from c_series_agent import SimAgent
    agent = SimAgent()
    final_text, api_stats, tool_trace = agent.run_agent_loop("take off and hover at 1 metre")
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import urllib.request, urllib.error
import matplotlib
matplotlib.use("Agg")

from drone_sim import (
    PhysicsLoop, DroneState, PID,
    MASS, GRAVITY,
    Kp_roll_angle, Ki_roll_angle, Kd_roll_angle,
    Kp_roll_rate,  Ki_roll_rate,  Kd_roll_rate,
    Kp_pitch_angle,Ki_pitch_angle,Kd_pitch_angle,
    Kp_pitch_rate, Ki_pitch_rate, Kd_pitch_rate,
    Kp_yaw,        Ki_yaw,        Kd_yaw,
    lw_pidZ_kp,  lw_pidZ_ki,
    lw_pidVZ_kp, lw_pidVZ_ki, lw_hover_thr,
    lw_pidX_kp,  lw_pidX_ki,
    lw_pidVX_kp, lw_pidVX_ki, lw_pidVX_kd,
    lw_kf_q,     lw_kf_r_tof, lw_flow_std,
    i_limit
)

# ── Anthropic API configuration ────────────────────────────────────────────────
_ENDPOINT = "https://claude-test-madhan-resource.services.ai.azure.com/anthropic/v1/messages?api-version=2025-01-01-preview"
_API_KEY   = os.environ.get(
    "ANTHROPIC_API_KEY",
    "EpilO2YT1tLIiwwKoIqCv9oodffWWedT4R7gJdocTTrSVwCC2GEUJQQJ99CCACfhMk5XJ3w3AAAAACOGGVL2"
)
_MODEL   = "claude-sonnet-4-6"   # Azure deployment name
_VERSION = "2023-06-01"
_USE_AZURE = True   # use Authorization: Bearer instead of x-api-key
COST_IN  = 3.0  / 1_000_000   # $/token  (input)
COST_OUT = 15.0 / 1_000_000   # $/token  (output)

SIM_HZ      = 200
DT          = 1.0 / SIM_HZ
TEL_STRIDE  = 20          # record telemetry every N ticks  → 10 Hz
MAX_TURNS   = 30          # max LLM → tool iterations

# ── System prompt (sim-adapted from keyboard_server.py) ───────────────────────
SYSTEM_PROMPT = """\
You are an AUTONOMOUS FLIGHT AGENT for a SIMULATED micro quadrotor drone
(7 cm × 7 cm, 50 g, brushed motors). The drone physics are simulated at 200 Hz.

━━ AGENTIC WORKFLOW PROTOCOL ━━
For EVERY multi-step request, follow this exact sequence:
  1. plan_workflow(goal, steps)       — declare full plan FIRST
  2. report_progress(step, N, "...")  — before each step
  3. Execute the tool for that step
  4. Observe: check_altitude_reached / check_drone_stable / get_sensor_status
  5. If observation fails → retry or adapt
  6. report_progress(N, N, "Complete") — when done

━━ STANDARD TAKEOFF SEQUENCE ━━
  arm()
  find_hover_throttle()               ← ramps from 1200 until vz≈0
  check_drone_stable()
  enable_altitude_hold()              ← captures current z + throttle
  wait(2.0)
  set_altitude_target(target_m)
  wait(4.0)
  check_altitude_reached(target_m, 0.10)

━━ LANDING SEQUENCE ━━
  disable_altitude_hold()
  hover()
  set_throttle(1400) → wait(1.0) → set_throttle(1200) → wait(1.0) → set_throttle(1000) → wait(0.5)
  disarm()

━━ ALTITUDE HOLD ━━
  enable_altitude_hold()      — ONLY when already stably airborne
  set_altitude_target(meters) — 0.20–2.50 m; step in ≤0.5 m increments for large changes
  disable_altitude_hold()     — return to manual throttle

━━ OBSERVATION TOOLS ━━
  get_sensor_status()                            — live telemetry snapshot
  check_altitude_reached(target_m, tolerance_m)  — ✓/✗ based on EKF Z
  check_drone_stable(max_degrees=5.0)            — ✓/✗ based on roll/pitch
  If ✗: wait longer then re-check. After 3 failures: report issue and stop.

━━ TUNING (for PID adjustment tasks) ━━
  get_tuning_params()               — see current PID values
  set_tuning_params(param=value)    — change one or more gains
  apply_tuning()                    — push to drone
  analyze_flight(seconds=30)        — AI analysis of recent telemetry
  suggest_pid_tuning(axis="all")    — targeted gain recommendations
  detect_anomaly()                  — scan for yaw spin, oscillation, etc.

━━ MISSION TOOLS ━━
  plan_workflow(goal, steps)
  report_progress(step, total_steps, description)

━━ SAFETY ━━
  Never jump throttle — always ramp.
  Max safe throttle: 1800 PWM.
  Always check_drone_stable after takeoff before enabling holds.
  If check_drone_stable fails 3× → land immediately.

Respond concisely. Chain tools without waiting for user confirmation.
"""

# ── Tool definitions (Anthropic format) ───────────────────────────────────────
SIM_TOOLS = [
    {"name": "arm",
     "description": "Arm the drone motors. Must precede takeoff.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "disarm",
     "description": "Disarm motors. Only when landed or emergency.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "takeoff",
     "description": "Full safe takeoff: arm, ramp throttle to hover.",
     "input_schema": {
         "type": "object",
         "properties": {"hover_power": {"type": "integer",
                                        "description": "Target hover PWM 1400–1700. Default 1550."}},
         "required": []}},

    {"name": "land",
     "description": "Full safe landing: centre controls, ramp throttle down, disarm.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "hover",
     "description": "Centre roll/pitch/yaw to 1500 for stable hover. Throttle unchanged.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "set_throttle",
     "description": "Set throttle PWM directly.",
     "input_schema": {
         "type": "object",
         "properties": {"pwm": {"type": "integer", "description": "1000–2000"}},
         "required": ["pwm"]}},

    {"name": "set_roll",
     "description": "Set roll tilt. 1500=level, 1000=left, 2000=right.",
     "input_schema": {
         "type": "object",
         "properties": {"pwm": {"type": "integer"}},
         "required": ["pwm"]}},

    {"name": "set_pitch",
     "description": "Set pitch tilt. 1500=level, 1000=back, 2000=forward.",
     "input_schema": {
         "type": "object",
         "properties": {"pwm": {"type": "integer"}},
         "required": ["pwm"]}},

    {"name": "set_yaw",
     "description": "Set yaw rotation. 1500=no spin, 1000=CCW, 2000=CW.",
     "input_schema": {
         "type": "object",
         "properties": {"pwm": {"type": "integer"}},
         "required": ["pwm"]}},

    {"name": "enable_altitude_hold",
     "description": "Enable altitude hold. Drone must be stably airborne first.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "disable_altitude_hold",
     "description": "Disable altitude hold — pilot controls altitude manually.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "enable_position_hold",
     "description": "Enable XY position hold using optical flow.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "disable_position_hold",
     "description": "Disable position hold.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "set_altitude_target",
     "description": "Set target altitude for altitude-hold mode (0.20–2.50 m).",
     "input_schema": {
         "type": "object",
         "properties": {"meters": {"type": "number"}},
         "required": ["meters"]}},

    {"name": "wait",
     "description": "Pause for given duration (simulation advances this much).",
     "input_schema": {
         "type": "object",
         "properties": {"seconds": {"type": "number"}},
         "required": ["seconds"]}},

    {"name": "get_drone_state",
     "description": "Returns drone state: armed, throttle, althold, poshold, altitude target.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "get_sensor_status",
     "description": "Live sensor snapshot: EKF altitude, roll, pitch, hold modes, motor PWMs.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "check_altitude_reached",
     "description": "Check if EKF altitude is within tolerance of target. Returns ✓ or ✗.",
     "input_schema": {
         "type": "object",
         "properties": {
             "target_m":    {"type": "number"},
             "tolerance_m": {"type": "number", "description": "Default 0.10"}},
         "required": ["target_m"]}},

    {"name": "check_drone_stable",
     "description": "Check if roll and pitch are within max_degrees. Returns ✓ or ✗.",
     "input_schema": {
         "type": "object",
         "properties": {"max_degrees": {"type": "number", "description": "Default 5.0"}},
         "required": []}},

    {"name": "find_hover_throttle",
     "description": ("Ramp throttle from start_pwm until vz≈0 — "
                     "finds true hover throttle for current battery state. Drone must be armed."),
     "input_schema": {
         "type": "object",
         "properties": {
             "start_pwm":   {"type": "integer", "description": "Default 1200"},
             "max_pwm":     {"type": "integer", "description": "Default 1750"},
             "step_pwm":    {"type": "integer", "description": "Default 20"},
             "step_wait_s": {"type": "number",  "description": "Default 0.4"}},
         "required": []}},

    {"name": "plan_workflow",
     "description": "Declare the full plan before executing any steps.",
     "input_schema": {
         "type": "object",
         "properties": {
             "goal":  {"type": "string"},
             "steps": {"type": "array", "items": {"type": "string"}}},
         "required": ["goal", "steps"]}},

    {"name": "report_progress",
     "description": "Report current step number and description.",
     "input_schema": {
         "type": "object",
         "properties": {
             "step":        {"type": "integer"},
             "total_steps": {"type": "integer"},
             "description": {"type": "string"}},
         "required": ["step", "total_steps", "description"]}},

    {"name": "get_tuning_params",
     "description": "Return all current PID and EKF tuning parameters.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "set_tuning_params",
     "description": "Update one or more PID/EKF parameters. Call apply_tuning() afterwards.",
     "input_schema": {
         "type": "object",
         "properties": {
             "roll_angle_kp": {"type": "number"}, "roll_angle_ki": {"type": "number"},
             "roll_angle_kd": {"type": "number"}, "roll_rate_kp":  {"type": "number"},
             "roll_rate_ki":  {"type": "number"}, "roll_rate_kd":  {"type": "number"},
             "pitch_angle_kp":{"type": "number"}, "pitch_angle_ki":{"type": "number"},
             "pitch_angle_kd":{"type": "number"}, "pitch_rate_kp": {"type": "number"},
             "pitch_rate_ki": {"type": "number"}, "pitch_rate_kd": {"type": "number"},
             "yaw_rate_kp":   {"type": "number"}, "yaw_rate_ki":   {"type": "number"},
             "yaw_rate_kd":   {"type": "number"},
             "lw_pidZ_kp":    {"type": "number"}, "lw_pidZ_ki":    {"type": "number"},
             "lw_pidVZ_kp":   {"type": "number"}, "lw_pidVZ_ki":   {"type": "number"},
             "lw_hover_thr":  {"type": "number"},
             "lw_pidX_kp":    {"type": "number"}, "lw_pidX_ki":    {"type": "number"},
             "lw_pidVX_kp":   {"type": "number"}, "lw_pidVX_ki":   {"type": "number"},
         },
         "required": []}},

    {"name": "apply_tuning",
     "description": "Push tuning parameters to drone immediately.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "set_trim",
     "description": "Set pitch/roll trim offset (±150 PWM). Corrects steady-state hover drift.",
     "input_schema": {
         "type": "object",
         "properties": {
             "pitch_trim": {"type": "integer", "description": "-150 to 150"},
             "roll_trim":  {"type": "integer", "description": "-150 to 150"}},
         "required": []}},

    {"name": "analyze_flight",
     "description": ("Analyse recent flight telemetry. Returns stability report, "
                     "oscillation counts, and whether problem is trim or PID."),
     "input_schema": {
         "type": "object",
         "properties": {"seconds": {"type": "integer", "description": "Default 30"}},
         "required": []}},

    {"name": "suggest_pid_tuning",
     "description": ("Analyse telemetry and suggest specific PID gain changes. "
                     "Only call when oscillation is detected."),
     "input_schema": {
         "type": "object",
         "properties": {"axis": {"type": "string", "enum": ["roll","pitch","yaw","all"],
                                 "description": "Default 'all'"}},
         "required": []}},

    {"name": "detect_anomaly",
     "description": "Scan telemetry for yaw spin, roll/pitch oscillation, motor imbalance, altitude drift.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "emergency_stop",
     "description": "EMERGENCY: immediately disarm all motors. Drone drops.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SimAgent
# ═══════════════════════════════════════════════════════════════════════════════

class SimAgent:
    """
    Drone physics simulation + Claude API agent runner.

    Each instance owns one DroneState + PhysicsLoop. All tool calls execute
    synchronously against the sim (no WebSocket, no threading during tools).
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.state   = DroneState()
        self.physics = PhysicsLoop(self.state)

        self.sim_time  = 0.0    # simulated seconds elapsed
        self.tel_buf   = []     # telemetry ring buffer (10 Hz)
        self._tel_ctr  = 0      # tick counter for decimation
        self._trim_pitch = 0    # PWM offset applied to ch3
        self._trim_roll  = 0    # PWM offset applied to ch2

        # Pending tuning changes (applied by apply_tuning)
        self._pending_gains: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  Physics helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _tick_one(self):
        """Run one 200 Hz physics tick and optionally record telemetry."""
        # Apply trim offsets to roll/pitch channels before ticking
        with self.state.lock:
            if not self.state.althold:
                self.state.ch2 = max(1000, min(2000,
                    self.state.ch2 + self._trim_roll))
                self.state.ch3 = max(1000, min(2000,
                    self.state.ch3 + self._trim_pitch))
        self.physics.tick()
        self.sim_time += DT

        self._tel_ctr += 1
        if self._tel_ctr >= TEL_STRIDE:
            self._tel_ctr = 0
            self._record_tel()

    def _record_tel(self):
        s = self.state
        with s.lock:
            sample = {
                "t":        round(self.sim_time * 1000),  # ms
                "r":        round(s.roll,  2),
                "p":        round(s.pitch, 2),
                "y":        round(s.yaw,   2),
                "gx":       round(s.p, 1),
                "gy":       round(s.q, 1),
                "gz":       round(s.r, 1),
                "er":       round(s.error_roll,  3),
                "ep":       round(s.error_pitch, 3),
                "ey":       round(s.error_yaw,   3),
                "ch1":      s.ch1,
                "m1": s.m1, "m2": s.m2, "m3": s.m3, "m4": s.m4,
                "lw_z":     round(s.ekf_z * 1000, 1),   # mm
                "z_true":   round(s.z, 4),
                "altsp":    round(s.alt_sp_mm, 1),
                "vz":       round(s.ekf_vz, 3),
                "kx":       round(s.ekf_x,  3),
                "ky":       round(s.ekf_y,  3),
                "althold":  1 if s.althold  else 0,
                "poshold":  1 if s.poshold  else 0,
                "bat_pct":  round(s.bat_pct, 1),
            }
        self.tel_buf.append(sample)
        if len(self.tel_buf) > 3000:      # keep last 5 minutes @ 10 Hz
            self.tel_buf.pop(0)

    def wait_sim(self, seconds: float):
        """Advance simulation by `seconds` of simulated time."""
        n = max(1, int(round(seconds * SIM_HZ)))
        for _ in range(n):
            self._tick_one()

    def _find_hover(self, start_pwm=1200, max_pwm=1750,
                    step_pwm=20, step_wait_s=0.4) -> int:
        """
        Ramp throttle from start_pwm until the drone lifts off and vz≈0.
        Returns the found hover PWM. State.armed must be True.
        """
        pwm = start_pwm
        with self.state.lock:
            self.state.ch1 = pwm
        self.wait_sim(0.5)

        for _ in range(200):
            self.wait_sim(step_wait_s)
            with self.state.lock:
                vz = self.state.vz
                z  = self.state.z

            if z > 0.05 and abs(vz) < 0.015:
                break   # stable hover found

            if z < 0.05:
                pwm = min(max_pwm, pwm + step_pwm)
            elif vz > 0.04:
                pwm -= 2
            elif vz < -0.04:
                pwm += 2

            with self.state.lock:
                self.state.ch1 = max(1000, min(max_pwm, pwm))

        return pwm

    # ─────────────────────────────────────────────────────────────────────────
    #  Telemetry analysis helpers (used by analyze_flight / suggest_pid_tuning)
    # ─────────────────────────────────────────────────────────────────────────

    def _recent_tel(self, seconds: float):
        """Return telemetry samples from the last `seconds`."""
        cutoff_ms = self.sim_time * 1000 - seconds * 1000
        return [s for s in self.tel_buf if s["t"] >= cutoff_ms]

    @staticmethod
    def _sign_flips(values):
        flips, prev = 0, 0
        for v in values:
            s = 1 if v > 0 else -1
            if prev and s != prev:
                flips += 1
            prev = s
        return flips

    @staticmethod
    def _stats(vals):
        if not vals:
            return {"min": 0, "max": 0, "avg": 0, "std": 0}
        mn, mx = min(vals), max(vals)
        avg = sum(vals) / len(vals)
        std = math.sqrt(sum((v - avg)**2 for v in vals) / len(vals))
        return {"min": round(mn,3), "max": round(mx,3),
                "avg": round(avg,3), "std": round(std,3)}

    # ─────────────────────────────────────────────────────────────────────────
    #  Claude API
    # ─────────────────────────────────────────────────────────────────────────

    def _api_call(self, messages: list, max_tokens: int = 2048) -> dict:
        """Single Anthropic API call. Returns parsed response dict."""
        payload = {
            "model":      _MODEL,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "system":     SYSTEM_PROMPT,
            "messages":   messages,
            "tools":      SIM_TOOLS,
        }
        req = urllib.request.Request(
            _ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type":      "application/json",
                "Authorization":     f"Bearer {_API_KEY}",
                "anthropic-version": _VERSION,
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8")
                print(f"[API] HTTP {e.code}: {body[:500]}")
                raise
            except Exception as e:
                if attempt < 2:
                    print(f"[API] Attempt {attempt+1} failed ({e}), retrying …")
                    time.sleep(3)
                    # Rebuild request (body consumed)
                    req = urllib.request.Request(
                        _ENDPOINT,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type":      "application/json",
                            "Authorization":     f"Bearer {_API_KEY}",
                            "anthropic-version": _VERSION,
                        },
                        method="POST",
                    )
                else:
                    raise

    # ─────────────────────────────────────────────────────────────────────────
    #  Tool executor
    # ─────────────────────────────────────────────────────────────────────────

    def execute_tool(self, name: str, args: dict) -> str:
        """Execute one tool, return result string for LLM."""
        s = self.state
        p = self.physics

        # ── Metacognitive ──────────────────────────────────────────────────
        if name == "plan_workflow":
            goal  = args.get("goal", "")
            steps = args.get("steps", [])
            plan_str = "\n".join(f"  {i+1}. {st}" for i, st in enumerate(steps))
            print(f"\n[PLAN] {goal}\n{plan_str}")
            return f"Plan recorded ({len(steps)} steps): {goal}"

        if name == "report_progress":
            step  = args.get("step", 0)
            total = args.get("total_steps", 0)
            desc  = args.get("description", "")
            print(f"  [{step}/{total}] {desc}")
            return f"Progress: step {step}/{total} — {desc}"

        # ── State query ────────────────────────────────────────────────────
        if name == "get_drone_state":
            with s.lock:
                return json.dumps({
                    "connected":       True,
                    "armed":           s.armed,
                    "throttle":        s.ch1,
                    "roll":            s.ch2,
                    "pitch":           s.ch3,
                    "yaw":             s.ch4,
                    "altitudeHold":    s.althold,
                    "positionHold":    s.poshold,
                    "altitudeTarget":  round(s.alt_sp, 3),
                    "ekf_altitude_m":  round(s.ekf_z, 3),
                    "bat_pct":         round(s.bat_pct, 1),
                })

        if name == "get_sensor_status":
            with s.lock:
                return json.dumps({
                    "ekf_altitude_m": round(s.ekf_z, 3),
                    "ekf_vz_m_s":     round(s.ekf_vz, 3),
                    "roll_deg":       round(s.roll,  2),
                    "pitch_deg":      round(s.pitch, 2),
                    "yaw_deg":        round(s.yaw,   2),
                    "althold_active": s.althold,
                    "poshold_active": s.poshold,
                    "alt_setpoint_m": round(s.alt_sp, 3),
                    "motor_m1":       s.m1, "motor_m2": s.m2,
                    "motor_m3":       s.m3, "motor_m4": s.m4,
                    "bat_pct":        round(s.bat_pct, 1),
                    "armed":          s.armed,
                })

        if name == "check_altitude_reached":
            target = args.get("target_m", 1.0)
            tol    = args.get("tolerance_m", 0.10)
            with s.lock:
                z_ekf = s.ekf_z
            err = abs(z_ekf - target)
            if err <= tol:
                return f"✓ Altitude reached: {z_ekf:.3f} m (target {target} m, err {err*100:.1f} cm)"
            return f"✗ Not reached: {z_ekf:.3f} m (target {target} m, err {err*100:.1f} cm)"

        if name == "check_drone_stable":
            max_deg = args.get("max_degrees", 5.0)
            with s.lock:
                roll  = abs(s.roll)
                pitch = abs(s.pitch)
            if roll <= max_deg and pitch <= max_deg:
                return f"✓ Stable: roll={roll:.1f}°, pitch={pitch:.1f}° (max {max_deg}°)"
            return f"✗ Unstable: roll={roll:.1f}°, pitch={pitch:.1f}° (max {max_deg}°)"

        # ── Arm / disarm ───────────────────────────────────────────────────
        if name == "arm":
            with s.lock:
                if s.ch1 > 1050:
                    s.ch1 = 1000
                s.ch5   = 1000
                s.armed = True
            self.wait_sim(0.5)
            return "Armed. ch5=1000, motors spinning at idle."

        if name == "disarm":
            with s.lock:
                s.ch5      = 2000
                s.armed    = False
                s.ch1      = 1000
                s.althold  = False
                s.poshold  = False
            return "Disarmed. Motors stopped."

        if name == "emergency_stop":
            with s.lock:
                s.ch5      = 2000
                s.armed    = False
                s.ch1      = 1000
                s.althold  = False
                s.poshold  = False
            return "EMERGENCY STOP — drone disarmed, falling."

        # ── Takeoff / land ─────────────────────────────────────────────────
        if name == "takeoff":
            hover_pwr = int(args.get("hover_power", 1550))
            with s.lock:
                s.ch1   = 1000
                s.ch5   = 1000
                s.armed = True
            self.wait_sim(0.5)
            # Ramp to hover
            pwm = 1000
            with s.lock:
                s.ch1 = pwm
            for target in range(1000, hover_pwr + 1, 10):
                with s.lock:
                    s.ch1 = target
                self.wait_sim(0.05)
            self.wait_sim(2.0)
            with s.lock:
                z = round(s.z, 3)
            return (f"Takeoff complete. Hovering at z={z:.3f} m, "
                    f"throttle={hover_pwr}.")

        if name == "land":
            with s.lock:
                s.althold = False
                s.poshold = False
                s.ch2 = 1500; s.ch3 = 1500; s.ch4 = 1500
            for pwm in [1400, 1300, 1200, 1100, 1000]:
                with s.lock:
                    s.ch1 = pwm
                self.wait_sim(1.0)
            with s.lock:
                s.ch5   = 2000
                s.armed = False
                z = round(s.z, 3)
            return f"Landed and disarmed. Final z={z:.3f} m."

        # ── Manual control channels ────────────────────────────────────────
        if name == "hover":
            with s.lock:
                s.ch2 = 1500; s.ch3 = 1500; s.ch4 = 1500
            return "Hover: roll/pitch/yaw centred at 1500."

        if name == "set_throttle":
            pwm = max(1000, min(2000, int(args.get("pwm", 1500))))
            with s.lock:
                s.ch1 = pwm
            return f"Throttle set to {pwm}."

        if name == "set_roll":
            pwm = max(1000, min(2000, int(args.get("pwm", 1500))))
            with s.lock:
                s.ch2 = pwm
            return f"Roll set to {pwm}."

        if name == "set_pitch":
            pwm = max(1000, min(2000, int(args.get("pwm", 1500))))
            with s.lock:
                s.ch3 = pwm
            return f"Pitch set to {pwm}."

        if name == "set_yaw":
            pwm = max(1000, min(2000, int(args.get("pwm", 1500))))
            with s.lock:
                s.ch4 = pwm
            return f"Yaw set to {pwm}."

        # ── Altitude / position hold ───────────────────────────────────────
        if name == "enable_altitude_hold":
            with s.lock:
                z_now  = s.z
                ch1    = s.ch1
            if z_now < 0.05:
                return "ERROR: Cannot enable altitude hold on the ground. Arm and take off first."
            hover_thr = (ch1 - 1000) / 1000.0
            hover_thr = max(0.20, min(0.80, hover_thr))
            with s.lock:
                s.althold           = True
                s.hover_thr_locked  = hover_thr
                s.alt_sp            = z_now
                s.alt_sp_mm         = z_now * 1000
            p.pid_alt_pos.reset()
            p.pid_alt_vel.reset()
            return (f"Altitude hold enabled at {z_now:.3f} m, "
                    f"hover_thr={hover_thr:.3f}.")

        if name == "disable_altitude_hold":
            with s.lock:
                s.althold = False
            return "Altitude hold disabled."

        if name == "enable_position_hold":
            with s.lock:
                s.pos_sp_x = s.ekf_x
                s.pos_sp_y = s.ekf_y
                s.poshold  = True
            p.pid_px.reset(); p.pid_py.reset()
            p.pid_pvx.reset(); p.pid_pvy.reset()
            with s.lock:
                x, y = round(s.ekf_x, 3), round(s.ekf_y, 3)
            return f"Position hold enabled at x={x:.3f} m, y={y:.3f} m."

        if name == "disable_position_hold":
            with s.lock:
                s.poshold = False
            return "Position hold disabled."

        if name == "set_altitude_target":
            meters = float(args.get("meters", 1.0))
            meters = max(0.20, min(2.50, meters))
            with s.lock:
                if not s.althold:
                    return "ERROR: Altitude hold not active. Call enable_altitude_hold() first."
                s.alt_sp    = meters
                s.alt_sp_mm = meters * 1000
            p.pid_alt_pos.reset()
            return f"Altitude target set to {meters:.2f} m."

        # ── Wait ───────────────────────────────────────────────────────────
        if name == "wait":
            sec = float(args.get("seconds", 1.0))
            sec = max(0.1, min(30.0, sec))
            self.wait_sim(sec)
            with s.lock:
                z = round(s.ekf_z, 3)
            return f"Waited {sec:.1f} s. Current EKF altitude: {z:.3f} m."

        # ── Find hover throttle ────────────────────────────────────────────
        if name == "find_hover_throttle":
            with s.lock:
                if not s.armed:
                    return "ERROR: Drone is not armed. Call arm() first."
            start = int(args.get("start_pwm",   1200))
            maxp  = int(args.get("max_pwm",     1750))
            step  = int(args.get("step_pwm",      20))
            wait  = float(args.get("step_wait_s", 0.4))
            hover_pwm = self._find_hover(start, maxp, step, wait)
            hover_thr = (hover_pwm - 1000) / 1000.0
            with s.lock:
                z = round(s.z, 3)
            return (f"Hover throttle found: PWM={hover_pwm}, "
                    f"thr={hover_thr:.3f}, z={z:.3f} m.")

        # ── Trim ───────────────────────────────────────────────────────────
        if name == "set_trim":
            self._trim_pitch = max(-150, min(150, int(args.get("pitch_trim", 0))))
            self._trim_roll  = max(-150, min(150, int(args.get("roll_trim",  0))))
            return f"Trim set: pitch={self._trim_pitch}, roll={self._trim_roll}."

        # ── Tuning ────────────────────────────────────────────────────────
        if name == "get_tuning_params":
            return json.dumps({
                "attitude_pid": {
                    "roll_angle_kp":  p.pid_roll_angle.kp,
                    "roll_angle_ki":  p.pid_roll_angle.ki,
                    "roll_angle_kd":  p.pid_roll_angle.kd,
                    "roll_rate_kp":   p.pid_roll_rate.kp,
                    "roll_rate_ki":   p.pid_roll_rate.ki,
                    "roll_rate_kd":   p.pid_roll_rate.kd,
                    "pitch_angle_kp": p.pid_pitch_angle.kp,
                    "pitch_angle_ki": p.pid_pitch_angle.ki,
                    "pitch_angle_kd": p.pid_pitch_angle.kd,
                    "pitch_rate_kp":  p.pid_pitch_rate.kp,
                    "pitch_rate_ki":  p.pid_pitch_rate.ki,
                    "pitch_rate_kd":  p.pid_pitch_rate.kd,
                    "yaw_rate_kp":    p.pid_yaw_rate.kp,
                    "yaw_rate_ki":    p.pid_yaw_rate.ki,
                    "yaw_rate_kd":    p.pid_yaw_rate.kd,
                },
                "litewing_tuning": {
                    "lw_pidZ_kp":   p.pid_alt_pos.kp,
                    "lw_pidZ_ki":   p.pid_alt_pos.ki,
                    "lw_pidVZ_kp":  p.pid_alt_vel.kp,
                    "lw_pidVZ_ki":  p.pid_alt_vel.ki,
                    "lw_hover_thr": s.hover_thr_locked,
                    "lw_pidX_kp":   p.pid_px.kp,
                    "lw_pidX_ki":   p.pid_px.ki,
                    "lw_pidVX_kp":  p.pid_pvx.kp,
                    "lw_pidVX_ki":  p.pid_pvx.ki,
                },
            })

        if name == "set_tuning_params":
            # Map keys → physics objects
            _map = {
                "roll_angle_kp":  (p.pid_roll_angle,  "kp"),
                "roll_angle_ki":  (p.pid_roll_angle,  "ki"),
                "roll_angle_kd":  (p.pid_roll_angle,  "kd"),
                "roll_rate_kp":   (p.pid_roll_rate,   "kp"),
                "roll_rate_ki":   (p.pid_roll_rate,   "ki"),
                "roll_rate_kd":   (p.pid_roll_rate,   "kd"),
                "pitch_angle_kp": (p.pid_pitch_angle, "kp"),
                "pitch_angle_ki": (p.pid_pitch_angle, "ki"),
                "pitch_angle_kd": (p.pid_pitch_angle, "kd"),
                "pitch_rate_kp":  (p.pid_pitch_rate,  "kp"),
                "pitch_rate_ki":  (p.pid_pitch_rate,  "ki"),
                "pitch_rate_kd":  (p.pid_pitch_rate,  "kd"),
                "yaw_rate_kp":    (p.pid_yaw_rate,    "kp"),
                "yaw_rate_ki":    (p.pid_yaw_rate,    "ki"),
                "yaw_rate_kd":    (p.pid_yaw_rate,    "kd"),
                "lw_pidZ_kp":     (p.pid_alt_pos,     "kp"),
                "lw_pidZ_ki":     (p.pid_alt_pos,     "ki"),
                "lw_pidVZ_kp":    (p.pid_alt_vel,     "kp"),
                "lw_pidVZ_ki":    (p.pid_alt_vel,     "ki"),
                "lw_pidX_kp":     (p.pid_px,          "kp"),
                "lw_pidX_ki":     (p.pid_px,          "ki"),
                "lw_pidVX_kp":    (p.pid_pvx,         "kp"),
                "lw_pidVX_ki":    (p.pid_pvx,         "ki"),
            }
            changed = []
            for k, v in args.items():
                if k in _map:
                    obj, attr = _map[k]
                    old = getattr(obj, attr)
                    setattr(obj, attr, float(v))
                    changed.append(f"{k}: {old:.4f} → {v:.4f}")
                elif k == "lw_hover_thr":
                    with s.lock:
                        old = s.hover_thr_locked
                        s.hover_thr_locked = float(v)
                    changed.append(f"lw_hover_thr: {old:.3f} → {v:.3f}")
            self._pending_gains = args
            if changed:
                return "Parameters updated:\n" + "\n".join(changed)
            return "No recognised parameters found in request."

        if name == "apply_tuning":
            if self._pending_gains:
                result = f"Tuning applied to drone: {list(self._pending_gains.keys())}"
                self._pending_gains = {}
            else:
                result = "Tuning applied (no changes pending)."
            return result

        # ── Telemetry analysis ────────────────────────────────────────────
        if name == "analyze_flight":
            seconds = float(args.get("seconds", 30.0))
            recent = self._recent_tel(seconds)
            if len(recent) < 5:
                return "Insufficient telemetry data. Fly the drone first."

            roll_errs  = [s["er"] for s in recent]
            pitch_errs = [s["ep"] for s in recent]
            z_vals     = [s["lw_z"] / 1000.0 for s in recent]
            z_sp_vals  = [s["altsp"] / 1000.0 for s in recent]
            alt_errs   = [abs(z - sp) for z, sp in zip(z_vals, z_sp_vals)]
            m1_vals    = [s["m1"] for s in recent]
            m3_vals    = [s["m3"] for s in recent]

            r_flips = self._sign_flips(roll_errs)
            p_flips = self._sign_flips(pitch_errs)
            r_stats = self._stats(roll_errs)
            p_stats = self._stats(pitch_errs)
            z_rmse  = math.sqrt(sum(e**2 for e in alt_errs) / len(alt_errs))

            n_sec   = len(recent) / 10.0  # 10 Hz telemetry
            r_hz    = r_flips / n_sec if n_sec > 0 else 0
            p_hz    = p_flips / n_sec if n_sec > 0 else 0

            diagnosis = []
            if r_hz > 4:
                diagnosis.append(
                    f"OSCILLATION: roll error sign-flipping at {r_hz:.1f} Hz "
                    f"(>{4} Hz threshold). Likely cause: roll_angle_kp too high "
                    f"(current={p.pid_roll_angle.kp:.4f}). Reduce by 50–70%.")
            if p_hz > 4:
                diagnosis.append(
                    f"OSCILLATION: pitch error sign-flipping at {p_hz:.1f} Hz. "
                    f"Likely cause: pitch_angle_kp too high "
                    f"(current={p.pid_pitch_angle.kp:.4f}). Reduce by 50–70%.")
            if not diagnosis:
                diagnosis.append("No oscillation detected. Flight appears stable.")
                if max(alt_errs) > 0.10:
                    diagnosis.append(
                        f"Altitude error up to {max(alt_errs)*100:.1f} cm — "
                        "consider reducing lw_pidZ_kp if overshooting.")

            return json.dumps({
                "duration_sec":     round(n_sec, 1),
                "samples":          len(recent),
                "roll_error_stats": r_stats,
                "pitch_error_stats":p_stats,
                "roll_flips_per_s": round(r_hz, 2),
                "pitch_flips_per_s":round(p_hz, 2),
                "alt_rmse_cm":      round(z_rmse * 100, 2),
                "diagnosis":        diagnosis,
            })

        if name == "suggest_pid_tuning":
            axis    = args.get("axis", "all")
            seconds = 30.0
            recent  = self._recent_tel(seconds)
            if len(recent) < 5:
                return "Insufficient telemetry. Fly more before requesting tuning."

            suggestions = []
            n_sec = len(recent) / 10.0

            if axis in ("roll", "all"):
                roll_errs = [s["er"] for s in recent]
                r_flips   = self._sign_flips(roll_errs)
                r_hz      = r_flips / n_sec if n_sec > 0 else 0
                cur_kp    = p.pid_roll_angle.kp
                if r_hz > 4:
                    new_kp = round(cur_kp * 0.40, 5)
                    suggestions.append({
                        "param": "roll_angle_kp",
                        "current": cur_kp,
                        "recommended": new_kp,
                        "reason": (f"Roll error oscillating at {r_hz:.1f} Hz. "
                                   f"Reduce kp from {cur_kp:.4f} to {new_kp:.4f} "
                                   f"(−60%).")
                    })
                elif r_hz > 2:
                    new_kp = round(cur_kp * 0.70, 5)
                    suggestions.append({
                        "param": "roll_angle_kp",
                        "current": cur_kp,
                        "recommended": new_kp,
                        "reason": (f"Mild roll oscillation at {r_hz:.1f} Hz. "
                                   f"Reduce kp from {cur_kp:.4f} to {new_kp:.4f} "
                                   f"(−30%).")
                    })

            if axis in ("pitch", "all"):
                pitch_errs = [s["ep"] for s in recent]
                p_flips    = self._sign_flips(pitch_errs)
                p_hz       = p_flips / n_sec if n_sec > 0 else 0
                cur_kp     = p.pid_pitch_angle.kp
                if p_hz > 4:
                    new_kp = round(cur_kp * 0.40, 5)
                    suggestions.append({
                        "param": "pitch_angle_kp",
                        "current": cur_kp,
                        "recommended": new_kp,
                        "reason": f"Pitch oscillation at {p_hz:.1f} Hz. Reduce kp by 60%."
                    })

            if not suggestions:
                return "No significant oscillation detected. PID gains appear well-tuned."

            return json.dumps({"suggestions": suggestions})

        if name == "detect_anomaly":
            recent = self._recent_tel(10.0)
            if len(recent) < 5:
                return json.dumps({"anomalies": [], "note": "Insufficient data."})

            anomalies = []
            n_sec = len(recent) / 10.0

            # Yaw spin
            gz_vals = [abs(s["gz"]) for s in recent]
            if max(gz_vals) > 50 and sum(1 for v in gz_vals if v > 50) / len(gz_vals) > 0.3:
                anomalies.append({"type": "yaw_spin", "severity": "HIGH",
                                  "detail": f"Max yaw rate {max(gz_vals):.0f} deg/s"})

            # Roll oscillation
            roll_errs = [s["er"] for s in recent]
            r_hz = self._sign_flips(roll_errs) / n_sec if n_sec > 0 else 0
            if r_hz > 4:
                anomalies.append({"type": "roll_oscillation", "severity": "HIGH",
                                  "detail": f"Roll error oscillating at {r_hz:.1f} Hz"})

            # Pitch oscillation
            pitch_errs = [s["ep"] for s in recent]
            p_hz = self._sign_flips(pitch_errs) / n_sec if n_sec > 0 else 0
            if p_hz > 4:
                anomalies.append({"type": "pitch_oscillation", "severity": "HIGH",
                                  "detail": f"Pitch error oscillating at {p_hz:.1f} Hz"})

            # Motor imbalance
            m_vals = [[s[f"m{i}"] for s in recent] for i in range(1, 5)]
            m_avgs = [sum(v) / len(v) for v in m_vals]
            imbal  = max(m_avgs) - min(m_avgs)
            if imbal > 50:
                anomalies.append({"type": "motor_imbalance", "severity": "MEDIUM",
                                  "detail": f"Motor PWM spread: {imbal:.0f} (avg per motor: {[round(a) for a in m_avgs]})"})

            # Altitude drift
            z_vals  = [s["lw_z"] / 1000.0 for s in recent]
            z_sp    = [s["altsp"] / 1000.0 for s in recent]
            ah_vals = [s["althold"] for s in recent]
            if all(v == 1 for v in ah_vals[-5:]):
                recent_err = abs(z_vals[-1] - z_sp[-1]) if z_sp[-1] > 0 else 0
                if recent_err > 0.20:
                    anomalies.append({"type": "altitude_drift", "severity": "MEDIUM",
                                      "detail": f"Altitude error {recent_err*100:.0f} cm from setpoint"})

            return json.dumps({
                "anomalies": anomalies,
                "total_detected": len(anomalies),
            })

        # ── Fallback ───────────────────────────────────────────────────────
        return f"Tool '{name}' executed (sim stub). args={args}"

    # ─────────────────────────────────────────────────────────────────────────
    #  Main agent loop
    # ─────────────────────────────────────────────────────────────────────────

    def run_agent_loop(self, user_prompt: str,
                       history: list = None,
                       max_turns: int = MAX_TURNS,
                       max_tokens: int = 2048) -> tuple:
        """
        Run full LLM → tool → observe loop to completion.

        Returns:
            final_text  (str)  — last LLM text response
            api_stats   (list) — [{turn, latency_s, input_tokens, output_tokens, cost_usd}]
            tool_trace  (list) — [{turn, name, args, result, sim_time_s}]
        """
        messages   = list(history or [])
        messages.append({"role": "user", "content": user_prompt})

        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):
            t0 = time.time()
            try:
                resp = self._api_call(messages, max_tokens=max_tokens)
            except Exception as e:
                print(f"[API ERROR turn {turn}] {e}")
                break

            latency = time.time() - t0
            usage   = resp.get("usage", {})
            in_tok  = usage.get("input_tokens",  0)
            out_tok = usage.get("output_tokens", 0)
            cost    = in_tok * COST_IN + out_tok * COST_OUT

            api_stats.append({
                "turn":          turn,
                "latency_s":     round(latency, 3),
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "cost_usd":      round(cost, 6),
            })

            content     = resp.get("content", [])
            stop_reason = resp.get("stop_reason", "end_turn")

            # Extract assistant text
            for block in content:
                if block.get("type") == "text":
                    final_text = block["text"]

            # Append assistant turn
            messages.append({"role": "assistant", "content": content})

            # Done?
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            if not tool_uses or stop_reason == "end_turn":
                break

            # Execute tools
            results = []
            for tu in tool_uses:
                t_name = tu["name"]
                t_args = tu.get("input", {})
                t_id   = tu["id"]
                print(f"  [TOOL t{turn}] {t_name}({json.dumps(t_args)[:80]})")
                result = self.execute_tool(t_name, t_args)
                tool_trace.append({
                    "turn":       turn,
                    "name":       t_name,
                    "args":       t_args,
                    "result":     result[:300],
                    "sim_time_s": round(self.sim_time, 2),
                })
                results.append({
                    "type":        "tool_result",
                    "tool_use_id": t_id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": results})

        return final_text, api_stats, tool_trace

    # ─────────────────────────────────────────────────────────────────────────
    #  Convenience: telemetry arrays for plotting
    # ─────────────────────────────────────────────────────────────────────────

    def get_telem_arrays(self):
        """Return dict of numpy arrays from telemetry buffer."""
        if not self.tel_buf:
            return {}
        keys = self.tel_buf[0].keys()
        return {k: np.array([s[k] for s in self.tel_buf]) for k in keys}
