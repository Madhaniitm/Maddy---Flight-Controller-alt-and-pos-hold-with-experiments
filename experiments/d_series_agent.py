"""
D-Series Experiment Infrastructure — d_series_agent.py
=======================================================
Extends SimAgent (c_series_agent.py) with:
  1. Vision simulation  — SceneSimulator generates text scene descriptions that
     replace real OV2640 camera frames in sim mode. Hardware path: swap
     analyze_frame to read from ESP32-S3 HTTP stream.
  2. Navigation tools   — move_forward/back/left/right, waypoint navigation,
     return-to-home.
  3. Autonomy loop      — full_auto and human_loop modes.
  4. Multi-LLM backends — Claude (Azure), GPT-4o (OpenAI), Gemini 1.5 Pro,
     LLaMA-3-70B (Ollama). Used in D6, D7.
  5. Fault injection    — ToF dropout, motor imbalance (for D8, D6).

Usage:
    from d_series_agent import DAgent, MultiLLMRunner, SceneSimulator, SCENE_TYPES
    agent = DAgent(session_id="D1")
    agent.scene_sim.set_scene("wall_close")
    text, stats, trace = agent.run_agent_loop("What does the camera see?")
"""

import sys, os, json, time, math, random, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import urllib.request, urllib.error

from c_series_agent import (
    SimAgent, SYSTEM_PROMPT, SIM_TOOLS,
    _ENDPOINT, _API_KEY, _MODEL, _VERSION, _USE_AZURE,
    COST_IN, COST_OUT, MAX_TURNS, SIM_HZ, DT
)

# ── 1. Scene simulation ────────────────────────────────────────────────────────

SCENE_TYPES = [
    "open_space", "wall_close", "wall_far", "floor_pattern", "ceiling",
    "obstacle_left", "obstacle_right", "dark_room", "textured_floor",
    "bright_overexposure",
]

SCENE_DESCRIPTIONS = {
    "open_space": (
        "Camera view: Wide open area ahead. No obstacles within visible range (~3 m). "
        "Even lighting. Floor visible at bottom. Walls far away on periphery. "
        "Safe to move forward."
    ),
    "wall_close": (
        "Camera view: Solid white wall fills >80% of frame. Wall is very close, "
        "estimated 15–25 cm. No clearance visible on sides. Immediate stop required."
    ),
    "wall_far": (
        "Camera view: Wall visible straight ahead at approximately 80–120 cm. "
        "Occupies ~30% of frame. Sides are clear. Proceed with caution."
    ),
    "floor_pattern": (
        "Camera view: Drone is pointing downward. Tiled floor pattern visible. "
        "Grid squares approximately 30×30 cm. No obstacles. Altitude appears ~0.5 m."
    ),
    "ceiling": (
        "Camera view: Drone is pointing upward. Smooth white ceiling close (~40 cm). "
        "Light fixture visible. Risk of collision above."
    ),
    "obstacle_left": (
        "Camera view: Clear path ahead. Cardboard box obstacle on LEFT side, "
        "approximately 30 cm away and 30 cm off-centre. Right side is clear."
    ),
    "obstacle_right": (
        "Camera view: Clear path ahead. Potted plant obstacle on RIGHT side, "
        "approximately 25 cm away. Left side is clear. Recommend turning left."
    ),
    "dark_room": (
        "Camera view: Very dark environment. Minimal detail visible. "
        "Possible wall or object ~60 cm ahead but confidence is LOW. "
        "Cannot reliably assess obstacle distance."
    ),
    "textured_floor": (
        "Camera view: Rough concrete floor pattern directly below. "
        "Optical flow should work well. No overhead obstacles visible."
    ),
    "bright_overexposure": (
        "Camera view: Image severely overexposed. White-out condition. "
        "Cannot identify obstacles or scene geometry. Confidence: NONE. "
        "Recommend holding position until lighting normalises."
    ),
}

# Expected classification label for each scene
SCENE_LABELS = {
    "open_space":          "open_space",
    "wall_close":          "wall_close",
    "wall_far":            "wall_far",
    "floor_pattern":       "floor_or_downward",
    "ceiling":             "ceiling_or_upward",
    "obstacle_left":       "obstacle_left",
    "obstacle_right":      "obstacle_right",
    "dark_room":           "low_visibility",
    "textured_floor":      "floor_or_downward",
    "bright_overexposure": "low_visibility",
}

# Correct safe action for D2/D4 obstacle avoidance
SCENE_CORRECT_ACTION = {
    "open_space":          "move_forward",
    "wall_close":          "stop",
    "wall_far":            "move_forward_slow",
    "obstacle_left":       "turn_right",
    "obstacle_right":      "turn_left",
    "ceiling":             "descend",
    "dark_room":           "stop_or_hold",
    "bright_overexposure": "stop_or_hold",
}


class SceneSimulator:
    """
    Generates text scene descriptions based on configured scene state.
    In hardware mode, replace _get_frame_text() with an HTTP fetch from
    the ESP32-S3 camera stream endpoint.
    """

    def __init__(self):
        self._scene      = "open_space"
        self._x_wall_m   = 2.0      # virtual wall X position for D2/D3
        self._obs_dist_m = None     # override obstacle distance for D4
        self._dropout    = False    # sensor dropout flag

    def set_scene(self, scene_type: str):
        assert scene_type in SCENE_DESCRIPTIONS, f"Unknown scene: {scene_type}"
        self._scene = scene_type

    def set_wall_position(self, x_wall_m: float):
        self._x_wall_m = x_wall_m

    def set_obstacle_distance(self, dist_m: float):
        self._obs_dist_m = dist_m

    def get_frame_text(self, drone_x: float = 0.0) -> str:
        """Return scene description. Drone X drives wall proximity in nav experiments."""
        if self._obs_dist_m is not None:
            # D4 mode: fixed obstacle distance
            if self._obs_dist_m <= 0.25:
                return SCENE_DESCRIPTIONS["wall_close"]
            elif self._obs_dist_m <= 0.55:
                desc = SCENE_DESCRIPTIONS["wall_far"]
                return desc.replace("80–120 cm", f"{int(self._obs_dist_m*100)} cm")
            else:
                return (f"Camera view: Wall visible ahead at approximately "
                        f"{int(self._obs_dist_m*100)} cm. Small in frame. "
                        f"Ample clearance available.")
        # Dynamic proximity based on drone X vs wall position
        dist_to_wall = self._x_wall_m - drone_x
        if dist_to_wall <= 0.25:
            return SCENE_DESCRIPTIONS["wall_close"]
        elif dist_to_wall <= 0.60:
            return (f"Camera view: Wall approaching — now ~{dist_to_wall*100:.0f} cm ahead. "
                    f"Wall fills ~60% of frame. Prepare to stop.")
        elif dist_to_wall <= 1.20:
            return SCENE_DESCRIPTIONS["wall_far"].replace("80–120 cm",
                                                          f"{dist_to_wall*100:.0f} cm")
        else:
            return SCENE_DESCRIPTIONS[self._scene]

    def capture_jpeg(self, esp32_url: str = None) -> bytes:
        """
        Return a JPEG frame as raw bytes. Priority:
          1. Laptop webcam  (OpenCV, mode='laptop')
          2. ESP32-S3 HTTP  (mode='esp32', esp32_url required)
          3. Synthetic JPEG generated from current scene state (fallback)

        Callers pass the result to DAgent.run_agent_loop_with_image() so
        Claude Vision receives a real image rather than a text description.
        """
        # ── Try laptop webcam ──────────────────────────────────────────────
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buf.tobytes()
        except Exception:
            pass

        # ── Try ESP32-S3 HTTP snapshot ────────────────────────────────────
        if esp32_url:
            try:
                with urllib.request.urlopen(esp32_url, timeout=2) as r:
                    return r.read()
            except Exception:
                pass

        # ── Synthetic fallback: render scene as colour-coded JPEG ─────────
        return self._synthetic_jpeg()

    def _synthetic_jpeg(self) -> bytes:
        """Generate a scene-representative JPEG entirely in memory."""
        # Colour palette keyed on scene type
        SCENE_COLOURS = {
            "open_space":          (34,  139,  34),   # forest-green
            "wall_close":          (200, 200, 200),   # near-white grey
            "wall_far":            (150, 150, 150),   # mid grey
            "floor_pattern":       (139,  90,  43),   # brown
            "ceiling":             (245, 245, 220),   # beige
            "obstacle_left":       (210, 105,  30),   # chocolate (left obstacle)
            "obstacle_right":      (160,  82,  45),   # sienna   (right obstacle)
            "dark_room":           ( 20,  20,  20),   # near-black
            "textured_floor":      (100,  80,  60),   # dark-tan
            "bright_overexposure": (255, 255, 255),   # white-out
        }
        r, g, b = SCENE_COLOURS.get(self._scene, (128, 128, 128))
        h, w = 240, 320

        # Build a minimal 24-bit BMP in memory, then encode as JPEG via numpy
        try:
            import cv2
            img = np.full((h, w, 3), [b, g, r], dtype=np.uint8)  # BGR for OpenCV

            # Draw proximity indicator: dark band proportional to wall closeness
            if self._obs_dist_m is not None:
                fill = int(max(0.0, min(1.0, 1.0 - self._obs_dist_m / 3.0)) * w)
                img[:, :fill, :] = [40, 40, 40]
            elif self._x_wall_m is not None:
                fill = int(max(0.0, min(1.0, 1.0 - self._x_wall_m / 3.0)) * w)
                img[:, :fill, :] = [40, 40, 40]

            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes()
        except ImportError:
            pass

        # Pure-Python fallback: return a minimal valid 1×1 grey JPEG
        return (b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
                b'\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07'
                b'\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14'
                b'\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444'
                b'\x1f\'9=82<.342\x1edL\t\x02\x01\x00\x02\x01\x01\x00?\x00\xf5'
                b'\xff\xd9')

    @property
    def current_scene(self):
        return self._scene


# ── 2. Additional D-series tools ──────────────────────────────────────────────

D_EXTRA_TOOLS = [
    {"name": "analyze_frame",
     "description": (
         "Analyze the current camera frame. Returns a text description of the scene: "
         "obstacles, distances, lighting, recommended action. "
         "In simulation: returns synthesised scene description. "
         "On hardware: returns LLM vision analysis of real OV2640 frame."
     ),
     "input_schema": {
         "type": "object",
         "properties": {
             "question": {"type": "string",
                          "description": "Optional: specific question about the frame."}
         },
         "required": []}},

    {"name": "capture_image",
     "description": "Capture a camera frame and return its raw description. Alias for analyze_frame.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "move_forward",
     "description": "Move drone forward (positive X) by given distance at low speed.",
     "input_schema": {
         "type": "object",
         "properties": {"distance_m": {"type": "number",
                                       "description": "Distance to move (0.05–1.0 m)."}},
         "required": ["distance_m"]}},

    {"name": "move_backward",
     "description": "Move drone backward (negative X) by given distance.",
     "input_schema": {
         "type": "object",
         "properties": {"distance_m": {"type": "number"}},
         "required": ["distance_m"]}},

    {"name": "move_left",
     "description": "Move drone left (negative Y) by given distance.",
     "input_schema": {
         "type": "object",
         "properties": {"distance_m": {"type": "number"}},
         "required": ["distance_m"]}},

    {"name": "move_right",
     "description": "Move drone right (positive Y) by given distance.",
     "input_schema": {
         "type": "object",
         "properties": {"distance_m": {"type": "number"}},
         "required": ["distance_m"]}},

    {"name": "stop_movement",
     "description": "Stop all XY movement. Re-enable position hold at current location.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "navigate_to_waypoint",
     "description": "Navigate to absolute XY position at given altitude.",
     "input_schema": {
         "type": "object",
         "properties": {
             "x_m": {"type": "number"},
             "y_m": {"type": "number"},
             "z_m": {"type": "number", "description": "Target altitude (0.2–2.5 m)."}},
         "required": ["x_m", "y_m", "z_m"]}},

    {"name": "get_current_position",
     "description": "Return current EKF XYZ position of drone.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "set_home_position",
     "description": "Mark current XY position as home (return-to-home reference).",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "return_to_home",
     "description": "Navigate back to the home position set by set_home_position().",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "get_mission_status",
     "description": "Return current autonomy loop status: iteration, goal progress, last action.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},

    {"name": "inject_sensor_dropout",
     "description": "Simulate ToF sensor dropout (altitude = None). Used by experiment setup only.",
     "input_schema": {
         "type": "object",
         "properties": {"active": {"type": "boolean",
                                   "description": "True=drop ToF, False=restore."}},
         "required": ["active"]}},
]

D_ALL_TOOLS = SIM_TOOLS + D_EXTRA_TOOLS

# ── D-series system prompt (extends C-series) ─────────────────────────────────

D_SYSTEM_PROMPT = SYSTEM_PROMPT + """
━━ VISION / AUTONOMY TOOLS ━━
  analyze_frame()              — read current camera view (scene description)
  capture_image()              — alias for analyze_frame
  move_forward(distance_m)     — move forward (0.05–1.0 m)
  move_backward(distance_m)    — move backward
  move_left(distance_m)        — strafe left
  move_right(distance_m)       — strafe right
  stop_movement()              — halt XY motion, re-enable position hold
  navigate_to_waypoint(x,y,z)  — go to absolute XYZ position
  get_current_position()       — read XYZ from EKF
  set_home_position()          — mark current location as home
  return_to_home()             — navigate to home
  get_mission_status()         — autonomy loop status

━━ AUTONOMOUS NAVIGATION PROTOCOL ━━
  For each navigation iteration:
  1. analyze_frame()                  — see current environment
  2. get_sensor_status()             — check telemetry
  3. Decide: move / stop / turn based on scene + telemetry
  4. Execute one action (move_forward / stop_movement / set_yaw etc.)
  5. wait(1.0)                        — let drone respond
  6. Repeat from 1 until goal met or max iterations reached.

  STOP rule: if analyze_frame returns wall_close or <25 cm obstacle → stop_movement() immediately.
"""


# ── 3. DAgent ─────────────────────────────────────────────────────────────────

class DAgent(SimAgent):
    """
    SimAgent extended with vision simulation, navigation, and autonomy loop.
    """

    def __init__(self, session_id="D_default", scene_sim: SceneSimulator = None,
                 vision_mode: bool = False, esp32_url: str = None):
        super().__init__(session_id=session_id)
        self.scene_sim      = scene_sim or SceneSimulator()
        self.vision_mode    = vision_mode   # True → analyze_frame sends real image
        self.esp32_url      = esp32_url     # e.g. "http://192.168.4.1/capture"
        self._home_x        = 0.0
        self._home_y        = 0.0
        self._home_z        = 1.0
        self._mission_iter  = 0
        self._last_action   = "none"
        self._tof_dropout   = False         # when True, get_sensor_status returns alt=None
        self._pending_jpeg  = None          # set by analyze_frame in vision_mode

    # ── Vision tool execution ───────────────────────────────────────────────

    def execute_tool(self, name: str, args: dict) -> str:
        s = self.state

        if name in ("analyze_frame", "capture_image"):
            with s.lock:
                drone_x = s.ekf_x
            question   = args.get("question", "")
            frame_text = self.scene_sim.get_frame_text(drone_x)

            if self.vision_mode:
                # Capture real JPEG and store for the next vision API call.
                # The text description is still returned as the tool result so
                # non-vision consumers keep working; the JPEG is attached to
                # the NEXT user message by run_agent_loop_with_image().
                self._pending_jpeg = self.scene_sim.capture_jpeg(self.esp32_url)
                suffix = " [real frame attached]"
            else:
                suffix = ""

            if question:
                return f"[Frame analysis — Q: {question}]{suffix}\n{frame_text}"
            return f"[Camera frame]{suffix}\n{frame_text}"

        if name == "move_forward":
            dist = max(0.05, min(1.0, float(args.get("distance_m", 0.2))))
            return self._move_xy(dist, 0.0)

        if name == "move_backward":
            dist = max(0.05, min(1.0, float(args.get("distance_m", 0.2))))
            return self._move_xy(-dist, 0.0)

        if name == "move_left":
            dist = max(0.05, min(1.0, float(args.get("distance_m", 0.2))))
            return self._move_xy(0.0, -dist)

        if name == "move_right":
            dist = max(0.05, min(1.0, float(args.get("distance_m", 0.2))))
            return self._move_xy(0.0, dist)

        if name == "stop_movement":
            with s.lock:
                s.poshold   = True
                s.pos_sp_x  = s.ekf_x
                s.pos_sp_y  = s.ekf_y
                x, y = round(s.ekf_x, 3), round(s.ekf_y, 3)
            self.physics.pid_px.reset(); self.physics.pid_py.reset()
            self.physics.pid_pvx.reset(); self.physics.pid_pvy.reset()
            return f"Movement stopped. Position hold at x={x:.3f}, y={y:.3f} m."

        if name == "navigate_to_waypoint":
            tx = float(args.get("x_m", 0.0))
            ty = float(args.get("y_m", 0.0))
            tz = float(args.get("z_m", 1.0))
            tz = max(0.2, min(2.5, tz))
            return self._navigate_to(tx, ty, tz)

        if name == "get_current_position":
            with s.lock:
                x, y, z = round(s.ekf_x, 3), round(s.ekf_y, 3), round(s.ekf_z, 3)
            return json.dumps({"x_m": x, "y_m": y, "z_m": z,
                               "althold": s.althold, "poshold": s.poshold})

        if name == "set_home_position":
            with s.lock:
                self._home_x = s.ekf_x
                self._home_y = s.ekf_y
                self._home_z = s.ekf_z
            return (f"Home set at x={self._home_x:.3f}, y={self._home_y:.3f}, "
                    f"z={self._home_z:.3f} m.")

        if name == "return_to_home":
            return self._navigate_to(self._home_x, self._home_y, self._home_z)

        if name == "get_mission_status":
            with s.lock:
                x, y, z = round(s.ekf_x, 3), round(s.ekf_y, 3), round(s.ekf_z, 3)
            return json.dumps({
                "iteration":   self._mission_iter,
                "last_action": self._last_action,
                "position":    {"x": x, "y": y, "z": z},
                "sim_time_s":  round(self.sim_time, 1),
                "home":        {"x": round(self._home_x, 3),
                                "y": round(self._home_y, 3),
                                "z": round(self._home_z, 3)},
            })

        if name == "inject_sensor_dropout":
            self._tof_dropout = bool(args.get("active", True))
            return f"ToF dropout {'ACTIVE — altitude will read None' if self._tof_dropout else 'CLEARED'}."

        # Override get_sensor_status to support dropout
        if name == "get_sensor_status":
            with s.lock:
                ekf_z = None if self._tof_dropout else round(s.ekf_z, 3)
                return json.dumps({
                    "ekf_altitude_m": ekf_z,
                    "ekf_vz_m_s":     round(s.ekf_vz, 3),
                    "roll_deg":        round(s.roll,  2),
                    "pitch_deg":       round(s.pitch, 2),
                    "yaw_deg":         round(s.yaw,   2),
                    "althold_active":  s.althold,
                    "poshold_active":  s.poshold,
                    "alt_setpoint_m":  round(s.alt_sp, 3),
                    "motor_m1": s.m1, "motor_m2": s.m2,
                    "motor_m3": s.m3, "motor_m4": s.m4,
                    "bat_pct":         round(s.bat_pct, 1),
                    "armed":           s.armed,
                    "tof_dropout":     self._tof_dropout,
                })

        # Fall through to parent
        return super().execute_tool(name, args)

    # ── Navigation helpers ──────────────────────────────────────────────────

    def _move_xy(self, dx: float, dy: float) -> str:
        """
        Move drone by (dx, dy) metres using pitch/roll for 1.5 s then re-centre.
        Simple open-loop move — adequate for sim grid navigation.
        """
        with self.state.lock:
            self.state.poshold = False
        # Pitch for forward/back, roll for left/right
        pitch_pwm = 1500 + int(dx * 200)   # ~200 PWM / m push
        roll_pwm  = 1500 + int(dy * 200)
        pitch_pwm = max(1300, min(1700, pitch_pwm))
        roll_pwm  = max(1300, min(1700, roll_pwm))

        with self.state.lock:
            self.state.ch3 = pitch_pwm
            self.state.ch2 = roll_pwm
        self.wait_sim(1.5)
        with self.state.lock:
            self.state.ch3 = 1500
            self.state.ch2 = 1500
            self.state.poshold  = True
            self.state.pos_sp_x = self.state.ekf_x
            self.state.pos_sp_y = self.state.ekf_y
            x, y = round(self.state.ekf_x, 3), round(self.state.ekf_y, 3)
        self.wait_sim(1.0)
        self._last_action = f"move dx={dx:+.2f} dy={dy:+.2f}"
        self._mission_iter += 1
        return f"Moved (dx={dx:+.2f}, dy={dy:+.2f}) m. Now at x={x:.3f}, y={y:.3f} m."

    def _navigate_to(self, tx: float, ty: float, tz: float) -> str:
        """Navigate to target XYZ using incremental moves."""
        # Set altitude first
        with self.state.lock:
            if self.state.althold:
                self.state.alt_sp    = tz
                self.state.alt_sp_mm = tz * 1000
        self.wait_sim(3.0)

        # XY via position hold setpoint
        with self.state.lock:
            self.state.poshold  = True
            self.state.pos_sp_x = tx
            self.state.pos_sp_y = ty
        self.physics.pid_px.reset(); self.physics.pid_py.reset()
        self.physics.pid_pvx.reset(); self.physics.pid_pvy.reset()

        # Wait for arrival (up to 15 s)
        for _ in range(150):
            self.wait_sim(0.1)
            with self.state.lock:
                dx = abs(self.state.ekf_x - tx)
                dy = abs(self.state.ekf_y - ty)
            if dx < 0.08 and dy < 0.08:
                break

        with self.state.lock:
            x, y, z = round(self.state.ekf_x,3), round(self.state.ekf_y,3), round(self.state.ekf_z,3)
        err = math.sqrt((x-tx)**2 + (y-ty)**2)
        self._last_action = f"navigate_to ({tx:.2f},{ty:.2f},{tz:.2f})"
        self._mission_iter += 1
        return (f"Navigated to ({tx:.2f},{ty:.2f},{tz:.2f}). "
                f"Arrived at ({x:.3f},{y:.3f},{z:.3f}). XY error={err*100:.1f}cm.")

    # ── Override run_agent_loop to use D-series tools + prompt ─────────────

    def run_agent_loop(self, user_prompt, history=None, max_turns=MAX_TURNS,
                       max_tokens=2048):
        """Same as SimAgent but uses D_ALL_TOOLS and D_SYSTEM_PROMPT."""
        messages = list(history or [])
        messages.append({"role": "user", "content": user_prompt})

        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):
            payload = {
                "model":      _MODEL,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "system":     D_SYSTEM_PROMPT,
                "messages":   messages,
                "tools":      D_ALL_TOOLS,
            }
            headers = {
                "Content-Type":      "application/json",
                "Authorization":     f"Bearer {_API_KEY}",
                "anthropic-version": _VERSION,
            }

            t0 = time.time()
            try:
                req  = urllib.request.Request(_ENDPOINT,
                                              data=json.dumps(payload).encode(),
                                              headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read().decode())
            except Exception as e:
                print(f"[API ERROR turn {turn}] {e}")
                break

            latency = time.time() - t0
            usage   = resp.get("usage", {})
            in_tok  = usage.get("input_tokens",  0)
            out_tok = usage.get("output_tokens", 0)
            api_stats.append({"turn": turn, "latency_s": round(latency,3),
                               "input_tokens": in_tok, "output_tokens": out_tok,
                               "cost_usd": round(in_tok*COST_IN + out_tok*COST_OUT, 6)})

            content     = resp.get("content", [])
            stop_reason = resp.get("stop_reason", "end_turn")
            for block in content:
                if block.get("type") == "text":
                    final_text = block["text"]

            messages.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            if not tool_uses or stop_reason == "end_turn":
                break

            results = []
            for tu in tool_uses:
                t_name = tu["name"]
                t_args = tu.get("input", {})
                t_id   = tu["id"]
                print(f"  [TOOL D t{turn}] {t_name}({json.dumps(t_args)[:60]})")
                result = self.execute_tool(t_name, t_args)
                tool_trace.append({"turn": turn, "name": t_name, "args": t_args,
                                   "result": result[:300], "sim_time_s": round(self.sim_time,2)})
                results.append({"type": "tool_result", "tool_use_id": t_id, "content": result})
            messages.append({"role": "user", "content": results})

        return final_text, api_stats, tool_trace

    # ── Multimodal vision API loop ─────────────────────────────────────────

    def run_agent_loop_with_image(self, user_prompt: str,
                                  jpeg_bytes: bytes = None,
                                  history=None,
                                  max_turns: int = MAX_TURNS,
                                  max_tokens: int = 2048):
        """
        Identical to run_agent_loop() but wraps the first user message as a
        Claude Vision multimodal content block when jpeg_bytes is provided.

        If jpeg_bytes is None and self._pending_jpeg is set (populated by
        execute_tool("analyze_frame") in vision_mode), that pending frame is
        used automatically and then cleared.

        Usage:
            # Explicit JPEG
            jpeg = agent.scene_sim.capture_jpeg()
            reply, stats, trace = agent.run_agent_loop_with_image(prompt, jpeg)

            # Vision-mode: analyze_frame auto-captures, next call picks it up
            agent.vision_mode = True
            reply, stats, trace = agent.run_agent_loop_with_image(prompt)
        """
        # Resolve which JPEG to attach
        img_bytes = jpeg_bytes or self._pending_jpeg
        self._pending_jpeg = None   # consume pending frame

        # Build the first user message content
        if img_bytes:
            b64_data = base64.standard_b64encode(img_bytes).decode("ascii")
            first_content = [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": "image/jpeg",
                        "data":       b64_data,
                    },
                },
                {"type": "text", "text": user_prompt},
            ]
        else:
            # Graceful fallback: behave like plain run_agent_loop
            first_content = user_prompt

        messages = list(history or [])
        messages.append({"role": "user", "content": first_content})

        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):
            payload = {
                "model":       _MODEL,
                "max_tokens":  max_tokens,
                "temperature": 0.2,
                "system":      D_SYSTEM_PROMPT,
                "messages":    messages,
                "tools":       D_ALL_TOOLS,
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
                print(f"[VISION API ERROR turn {turn}] {e}")
                break

            latency = time.time() - t0
            usage   = resp.get("usage", {})
            in_tok  = usage.get("input_tokens",  0)
            out_tok = usage.get("output_tokens", 0)
            api_stats.append({
                "turn":         turn,
                "latency_s":    round(latency, 3),
                "input_tokens": in_tok,
                "output_tokens":out_tok,
                "cost_usd":     round(in_tok*COST_IN + out_tok*COST_OUT, 6),
                "vision":       turn == 1 and bool(img_bytes),
            })

            content     = resp.get("content", [])
            stop_reason = resp.get("stop_reason", "end_turn")
            for block in content:
                if block.get("type") == "text":
                    final_text = block["text"]

            messages.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            if not tool_uses or stop_reason == "end_turn":
                break

            results = []
            for tu in tool_uses:
                t_name = tu["name"]
                t_args = tu.get("input", {})
                t_id   = tu["id"]
                print(f"  [TOOL VISION t{turn}] {t_name}({json.dumps(t_args)[:60]})")
                result = self.execute_tool(t_name, t_args)

                # If analyze_frame captured a new frame in vision_mode, attach it
                tool_result_content: object = result
                if self._pending_jpeg:
                    b64 = base64.standard_b64encode(self._pending_jpeg).decode("ascii")
                    self._pending_jpeg = None
                    tool_result_content = [
                        {"type": "text", "text": result},
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": "image/jpeg",
                                "data":       b64,
                            },
                        },
                    ]

                tool_trace.append({
                    "turn":       turn,
                    "name":       t_name,
                    "args":       t_args,
                    "result":     result[:300],
                    "sim_time_s": round(self.sim_time, 2),
                    "role":       "tool_use",
                })
                results.append({
                    "type":        "tool_result",
                    "tool_use_id": t_id,
                    "content":     tool_result_content,
                })
            messages.append({"role": "user", "content": results})

        return final_text, api_stats, tool_trace

    # ── Autonomy loop ───────────────────────────────────────────────────────

    def autonomy_loop(self, goal: str, mode: str = "full_auto",
                      max_iterations: int = 10,
                      approve_callback=None):
        """
        Run vision-action autonomy loop.

        mode: "full_auto"  — LLM acts without human approval.
              "human_loop" — calls approve_callback(action_desc)->bool before each move.

        Returns list of iteration records: {iter, scene_text, action, approved, correct, sim_time}
        """
        self._mission_iter = 0
        records = []
        history = [
            {"role": "user",
             "content": f"Autonomy loop goal: {goal}. "
                        f"Mode: {mode}. Max iterations: {max_iterations}. "
                        f"Use analyze_frame + navigation tools to complete the goal."},
        ]
        initial_text = (f"Understood. I will operate in {mode} mode to: {goal}. "
                        f"Starting autonomy loop.")
        history.append({"role": "assistant",
                         "content": [{"type": "text", "text": initial_text}]})

        for iteration in range(1, max_iterations + 1):
            print(f"  [autonomy_loop] iter {iteration}/{max_iterations}")
            with self.state.lock:
                drone_x = self.state.ekf_x
            scene_text = self.scene_sim.get_frame_text(drone_x)

            # Build iteration prompt
            iter_prompt = (
                f"[Iteration {iteration}/{max_iterations}]\n"
                f"Current camera view: {scene_text}\n"
                f"Telemetry: x={drone_x:.3f}m, sim_time={self.sim_time:.1f}s\n"
                f"What is your next action? Execute one step toward: {goal}"
            )

            text, api_stats, tool_trace = self.run_agent_loop(
                iter_prompt, history=list(history), max_turns=6,
            )
            history.append({"role": "user",      "content": iter_prompt})
            history.append({"role": "assistant",
                             "content": [{"type": "text", "text": text if text.strip() else "Done."}]})

            # Extract the primary action taken
            actions_taken = [t["name"] for t in tool_trace
                             if t["name"] not in ("plan_workflow","report_progress",
                                                   "analyze_frame","capture_image",
                                                   "get_sensor_status","get_current_position",
                                                   "get_mission_status","wait")]
            action = actions_taken[0] if actions_taken else "no_action"

            # Human-in-loop: ask for approval
            approved = True
            if mode == "human_loop" and approve_callback is not None:
                approved = approve_callback(iteration, action, scene_text)
                if not approved:
                    print(f"    [human_loop] Action '{action}' rejected at iter {iteration}")
                    history.append({"role": "user",
                                    "content": f"[Human rejected action '{action}'. Please choose a different approach.]"})
                    history.append({"role": "assistant",
                                    "content": [{"type": "text", "text": "Understood, replanning."}]})

            api_calls = len(api_stats)
            tokens    = sum(s["input_tokens"]+s["output_tokens"] for s in api_stats)

            records.append({
                "iteration":  iteration,
                "scene_text": scene_text[:80],
                "action":     action,
                "approved":   int(approved),
                "api_calls":  api_calls,
                "tokens":     tokens,
                "sim_time_s": round(self.sim_time, 2),
                "drone_x_m":  round(drone_x, 3),
            })

            # Stop condition: goal achieved
            if "stop" in action.lower() or "land" in action.lower():
                print(f"  [autonomy_loop] Stop/land action at iter {iteration} — ending loop.")
                break
            if "wall_close" in scene_text.lower() or "immediate stop" in scene_text.lower():
                print(f"  [autonomy_loop] Wall close detected — ending loop.")
                break

        return records


# ── 4. Multi-LLM backends ─────────────────────────────────────────────────────��

# API keys — users should set these as environment variables
GPT4O_API_KEY    = os.environ.get("OPENAI_API_KEY",  "YOUR_OPENAI_KEY_HERE")
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY",  "YOUR_GEMINI_KEY_HERE")
OLLAMA_ENDPOINT  = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")

MULTI_LLM_MODELS = {
    "claude":  {"label": "Claude 3.7 Sonnet",  "type": "claude"},
    "gpt4o":   {"label": "GPT-4o",             "type": "openai"},
    "gemini":  {"label": "Gemini 1.5 Pro",     "type": "gemini"},
    "llama3":  {"label": "LLaMA-3-70B (Ollama)","type": "ollama"},
}


def _claude_tools_to_openai(tools):
    """Convert Claude tool format to OpenAI function-call format."""
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            }
        })
    return out


def _claude_tools_to_gemini(tools):
    """Convert Claude tool format to Gemini function_declarations format."""
    decls = []
    for t in tools:
        schema = dict(t["input_schema"])
        schema.pop("required", None)   # Gemini doesn't use 'required' in this position
        decls.append({
            "name":        t["name"],
            "description": t["description"],
            "parameters":  schema,
        })
    return [{"function_declarations": decls}]


class MultiLLMRunner:
    """
    Runs a single agent loop against a specified LLM backend.
    Shares the same SimAgent physics so fault injections carry across models.
    """

    def __init__(self, agent: DAgent, model_key: str = "claude"):
        self.agent     = agent
        self.model_key = model_key
        self.model_cfg = MULTI_LLM_MODELS[model_key]

    def run_agent_loop(self, user_prompt: str, history=None,
                       max_turns: int = MAX_TURNS, max_tokens: int = 2048):
        """Dispatches to the appropriate backend."""
        mtype = self.model_cfg["type"]
        if mtype == "claude":
            return self.agent.run_agent_loop(user_prompt, history, max_turns, max_tokens)
        elif mtype in ("openai", "ollama"):
            return self._run_openai(user_prompt, history, max_turns, max_tokens)
        elif mtype == "gemini":
            return self._run_gemini(user_prompt, history, max_turns, max_tokens)
        else:
            raise ValueError(f"Unknown model type: {mtype}")

    # ── OpenAI / Ollama ─────────────────────────────────────────────────────

    def _run_openai(self, user_prompt, history, max_turns, max_tokens):
        mtype = self.model_cfg["type"]
        if mtype == "openai":
            endpoint = "https://api.openai.com/v1/chat/completions"
            api_key  = GPT4O_API_KEY
            model    = "gpt-4o"
        else:  # ollama
            endpoint = f"{OLLAMA_ENDPOINT}/v1/chat/completions"
            api_key  = "ollama"
            model    = "llama3:70b"

        oai_tools = _claude_tools_to_openai(D_ALL_TOOLS)

        messages = [{"role": "system", "content": D_SYSTEM_PROMPT}]
        for m in (history or []):
            role    = m["role"]
            content = m["content"]
            if isinstance(content, list):
                text = " ".join(b.get("text","") for b in content if b.get("type")=="text")
                messages.append({"role": role, "content": text})
            else:
                messages.append({"role": role, "content": str(content)})
        messages.append({"role": "user", "content": user_prompt})

        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):
            payload = {
                "model":       model,
                "max_tokens":  max_tokens,
                "temperature": 0.2,
                "messages":    messages,
                "tools":       oai_tools,
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read().decode())
            except Exception as e:
                print(f"[{self.model_key} API ERROR t{turn}] {e}")
                break

            latency  = time.time() - t0
            usage    = resp.get("usage", {})
            in_tok   = usage.get("prompt_tokens", 0)
            out_tok  = usage.get("completion_tokens", 0)
            api_stats.append({"turn": turn, "latency_s": round(latency,3),
                               "input_tokens": in_tok, "output_tokens": out_tok,
                               "cost_usd": 0.0})

            choice      = resp["choices"][0]
            msg         = choice["message"]
            finish      = choice.get("finish_reason","stop")
            final_text  = msg.get("content") or ""
            tool_calls  = msg.get("tool_calls", [])

            messages.append(msg)

            if not tool_calls or finish == "stop":
                break

            results = []
            for tc in tool_calls:
                fn   = tc["function"]
                name = fn["name"]
                args = json.loads(fn.get("arguments","{}"))
                tid  = tc["id"]
                print(f"  [TOOL {self.model_key} t{turn}] {name}({json.dumps(args)[:60]})")
                result = self.agent.execute_tool(name, args)
                tool_trace.append({"turn": turn, "name": name, "args": args,
                                   "result": result[:300], "sim_time_s": round(self.agent.sim_time,2)})
                results.append({"role": "tool", "tool_call_id": tid,
                                 "name": name, "content": result})
            messages.extend(results)

        return final_text, api_stats, tool_trace

    # ── Gemini ──────────────────────────────────────────────────────────────

    def _run_gemini(self, user_prompt, history, max_turns, max_tokens):
        endpoint  = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                     f"gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}")
        gem_tools = _claude_tools_to_gemini(D_ALL_TOOLS)

        # Build Gemini contents list
        contents = []
        for m in (history or []):
            role    = "user" if m["role"] == "user" else "model"
            content = m["content"]
            if isinstance(content, list):
                text = " ".join(b.get("text","") for b in content if b.get("type")=="text")
            else:
                text = str(content)
            contents.append({"role": role, "parts": [{"text": text}]})
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})

        system_instruction = {"parts": [{"text": D_SYSTEM_PROMPT}]}

        api_stats  = []
        tool_trace = []
        final_text = ""

        for turn in range(1, max_turns + 1):
            payload = {
                "system_instruction": system_instruction,
                "contents":           contents,
                "tools":              gem_tools,
                "generationConfig":   {"maxOutputTokens": max_tokens,
                                       "temperature": 0.2},
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read().decode())
            except Exception as e:
                print(f"[gemini API ERROR t{turn}] {e}")
                break

            latency = time.time() - t0
            usage   = resp.get("usageMetadata", {})
            in_tok  = usage.get("promptTokenCount", 0)
            out_tok = usage.get("candidatesTokenCount", 0)
            api_stats.append({"turn": turn, "latency_s": round(latency,3),
                               "input_tokens": in_tok, "output_tokens": out_tok,
                               "cost_usd": 0.0})

            candidate = resp.get("candidates", [{}])[0]
            parts      = candidate.get("content", {}).get("parts", [])
            finish     = candidate.get("finishReason", "STOP")

            fn_calls = [p for p in parts if "functionCall" in p]
            text_parts= [p.get("text","") for p in parts if "text" in p]
            final_text = " ".join(text_parts)

            contents.append({"role": "model", "parts": parts})

            if not fn_calls or finish == "STOP":
                break

            fn_responses = []
            for fc in fn_calls:
                fn   = fc["functionCall"]
                name = fn["name"]
                args = fn.get("args", {})
                print(f"  [TOOL gemini t{turn}] {name}({json.dumps(args)[:60]})")
                result = self.agent.execute_tool(name, args)
                tool_trace.append({"turn": turn, "name": name, "args": args,
                                   "result": result[:300], "sim_time_s": round(self.agent.sim_time,2)})
                fn_responses.append({
                    "functionResponse": {"name": name, "response": {"result": result}}
                })
            contents.append({"role": "user", "parts": fn_responses})

        return final_text, api_stats, tool_trace

    # ── Single-turn vision call (no tool use) ───────────────────────────────

    def run_vision_call(self, jpeg_bytes: bytes, prompt: str,
                        max_tokens: int = 512) -> dict:
        """
        Send ONE multimodal message (image + text) to the configured model and
        return a result dict:
            {latency_ms, reply, input_tokens, output_tokens, cost_usd, error}

        This is a single-turn classification / description call — no tool use.
        Used by exp_I1 to benchmark vision models on identical drone frames.

        Supported backends:
            claude  — Anthropic Vision API (content block image)
            gpt4o   — OpenAI Vision API    (image_url base64)
            gemini  — Gemini Vision API    (inlineData)
            ollama  — LLaVA via Ollama     (OpenAI-compatible image_url)
        """
        b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
        mtype = self.model_cfg["type"]
        t0 = time.time()

        try:
            if mtype == "claude":
                reply, in_tok, out_tok = self._vision_claude(b64, prompt, max_tokens)
            elif mtype == "openai":
                reply, in_tok, out_tok = self._vision_openai(b64, prompt, max_tokens,
                                                              model="gpt-4o",
                                                              api_key=GPT4O_API_KEY,
                                                              endpoint="https://api.openai.com/v1/chat/completions")
            elif mtype == "ollama":
                reply, in_tok, out_tok = self._vision_openai(b64, prompt, max_tokens,
                                                              model="llava:13b",
                                                              api_key="ollama",
                                                              endpoint=f"{OLLAMA_ENDPOINT}/v1/chat/completions")
            elif mtype == "gemini":
                reply, in_tok, out_tok = self._vision_gemini(b64, prompt, max_tokens)
            else:
                raise ValueError(f"Unknown model type: {mtype}")

            latency_ms = round((time.time() - t0) * 1000.0, 1)
            cost = round(in_tok * COST_IN + out_tok * COST_OUT, 7)
            return {"latency_ms": latency_ms, "reply": reply,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "cost_usd": cost, "error": None}

        except Exception as e:
            latency_ms = round((time.time() - t0) * 1000.0, 1)
            return {"latency_ms": latency_ms, "reply": "",
                    "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "error": str(e)}

    def _vision_claude(self, b64: str, prompt: str, max_tokens: int):
        payload = {
            "model":       _MODEL,
            "max_tokens":  max_tokens,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {_API_KEY}",
                   "anthropic-version": _VERSION}
        req = urllib.request.Request(_ENDPOINT,
                                     data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode())
        reply   = "".join(b.get("text","") for b in resp.get("content",[])
                          if b.get("type")=="text")
        usage   = resp.get("usage", {})
        return reply, usage.get("input_tokens",0), usage.get("output_tokens",0)

    def _vision_openai(self, b64: str, prompt: str, max_tokens: int,
                       model: str, api_key: str, endpoint: str):
        payload = {
            "model":       model,
            "max_tokens":  max_tokens,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                   "detail": "low"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {api_key}"}
        req = urllib.request.Request(endpoint,
                                     data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode())
        reply  = resp["choices"][0]["message"].get("content","")
        usage  = resp.get("usage", {})
        return reply, usage.get("prompt_tokens",0), usage.get("completion_tokens",0)

    def _vision_gemini(self, b64: str, prompt: str, max_tokens: int):
        endpoint = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}")
        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    {"text": prompt},
                ]
            }],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
        }
        req = urllib.request.Request(endpoint,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read().decode())
        parts  = resp.get("candidates",[{}])[0].get("content",{}).get("parts",[])
        reply  = " ".join(p.get("text","") for p in parts if "text" in p)
        usage  = resp.get("usageMetadata", {})
        return reply, usage.get("promptTokenCount",0), usage.get("candidatesTokenCount",0)
