"""
LiteWing Position Hold for Maddy Flight Controller
===================================================
Reads Kalman-filtered state estimates (stateEstimate.x/y/vx/vy) from the
LiteWing Drone Positioning Module via cflib, runs a 2-stage XY position
controller (matching position_controller_pid.c), and sends corrected
ch2/ch3 PWM commands to Maddy via WebSocket.

Controller architecture (matches Crazyflie position_controller_pid.c):
  Outer loop:  pidX/Y  (position  -> velocity setpoint, ±xyVelMax)
  Inner loop:  pidVX/Y (velocity  -> roll/pitch angle,  ±rpLimit)
  Yaw compensation: rotate roll/pitch into body frame before sending

Usage:
  1. Connect PC to LiteWing WiFi AP (192.168.43.42)
  2. Start Maddy Flight Controller and connect browser (WebSocket port 81)
  3. Run this script: python3 litewing_pos_hold.py
  4. Arm Maddy and get to a stable hover
  5. Click "Enable Position Hold" — script takes over ch2/ch3

Author: generated for Maddy Flight Controller project
"""

import math
import time
import threading
import json
import tkinter as tk
from tkinter import ttk

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

try:
    import websocket  # websocket-client library
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[WARN] websocket-client not installed. Install with: pip install websocket-client")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

LITEWING_URI   = "udp://192.168.43.42"   # LiteWing module UDP address
MADDY_WS_URI   = "ws://192.168.4.1:81"  # Maddy WebSocket server (change IP as needed)
LOG_PERIOD_MS  = 20                       # 50 Hz — same as POSITION_RATE

# Maddy roll/pitch limits (must match firmware maxRoll / maxPitch)
MADDY_MAX_ROLL  = 10.0   # degrees
MADDY_MAX_PITCH = 10.0   # degrees

# ─── DEFAULT PID GAINS (matching position_controller_pid.c, scaled for Maddy) ─

# Outer loop: position → velocity setpoint
PIDX_KP = 1.9
PIDX_KI = 0.1
PIDX_KD = 0.0

PIDY_KP = 1.9
PIDY_KI = 0.1
PIDY_KD = 0.0

XY_VEL_MAX = 1.0   # m/s — output limit for outer loop

# Inner loop: velocity → roll/pitch angle (degrees)
# Original kp=25 was for a 20-deg rpLimit drone. Scaled to 10-deg Maddy limit.
PIDVX_KP = 10.0
PIDVX_KI = 0.5
PIDVX_KD = 0.0

PIDVY_KP = 10.0
PIDVY_KI = 0.5
PIDVY_KD = 0.0

RP_LIMIT = 10.0    # degrees — final roll/pitch clamp (= Maddy maxRoll/maxPitch)

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────

# LiteWing Kalman state (updated by log callback at 50 Hz)
lw_state = {
    "x": 0.0, "y": 0.0,
    "vx": 0.0, "vy": 0.0,
    "yaw": 0.0,           # stateEstimate.yaw in degrees
    "z": 0.0,
    "ready": False
}
lw_state_lock = threading.Lock()

# Target position (set when pos hold is first enabled)
target_x = 0.0
target_y = 0.0

# PID integrators / previous errors (reset when pos hold is toggled)
pid_state = {
    "x_int": 0.0, "y_int": 0.0,
    "vx_int": 0.0, "vy_int": 0.0,
    "x_prev_err": 0.0, "y_prev_err": 0.0,
    "vx_prev_err": 0.0, "vy_prev_err": 0.0,
}

# Control outputs
ctrl_roll  = 0.0   # degrees
ctrl_pitch = 0.0   # degrees

# Flags
poshold_enabled = False
ws_connected    = False
lw_connected    = False
stop_event      = threading.Event()

# WebSocket handle
ws_handle = None
ws_lock   = threading.Lock()

# GUI gains (live-tunable)
gains = {
    "pidx_kp": PIDX_KP,  "pidx_ki": PIDX_KI,  "pidx_kd": PIDX_KD,
    "pidy_kp": PIDY_KP,  "pidy_ki": PIDY_KI,  "pidy_kd": PIDY_KD,
    "pidvx_kp": PIDVX_KP, "pidvx_ki": PIDVX_KI, "pidvx_kd": PIDVX_KD,
    "pidvy_kp": PIDVY_KP, "pidvy_ki": PIDVY_KI, "pidvy_kd": PIDVY_KD,
    "xy_vel_max": XY_VEL_MAX,
    "rp_limit": RP_LIMIT,
}
gains_lock = threading.Lock()


# ─── HELPER: 1-D PID ──────────────────────────────────────────────────────────

def run_pid(error, prev_err, integrator, kp, ki, kd, dt, out_limit):
    """Returns (output, new_integrator, new_prev_err)."""
    integrator = integrator + error * dt
    # Anti-windup: clamp integrator
    integrator = max(-out_limit, min(out_limit, integrator))
    derivative = (error - prev_err) / dt if dt > 1e-6 else 0.0
    output = kp * error + ki * integrator + kd * derivative
    output = max(-out_limit, min(out_limit, output))
    return output, integrator, error


# ─── CONTROL LOOP ─────────────────────────────────────────────────────────────

def control_loop():
    """Runs at ~50 Hz. Computes roll/pitch from Kalman state; sends to Maddy."""
    global ctrl_roll, ctrl_pitch, pid_state, target_x, target_y

    dt = LOG_PERIOD_MS / 1000.0
    last_time = time.time()

    while not stop_event.is_set():
        now = time.time()
        dt_actual = now - last_time
        last_time = now
        if dt_actual <= 0 or dt_actual > 0.5:
            dt_actual = dt

        time.sleep(dt)

        if not poshold_enabled:
            continue

        with lw_state_lock:
            x   = lw_state["x"]
            y   = lw_state["y"]
            vx  = lw_state["vx"]
            vy  = lw_state["vy"]
            yaw = lw_state["yaw"]
            ready = lw_state["ready"]

        if not ready:
            continue

        with gains_lock:
            g = dict(gains)

        # ── Outer loop: position → velocity setpoint ──────────────────────────
        # pidX: position error → vx setpoint
        ex = target_x - x
        vel_x_sp, pid_state["x_int"], pid_state["x_prev_err"] = run_pid(
            ex, pid_state["x_prev_err"], pid_state["x_int"],
            g["pidx_kp"], g["pidx_ki"], g["pidx_kd"],
            dt_actual, g["xy_vel_max"]
        )

        # pidY: position error → vy setpoint
        ey = target_y - y
        vel_y_sp, pid_state["y_int"], pid_state["y_prev_err"] = run_pid(
            ey, pid_state["y_prev_err"], pid_state["y_int"],
            g["pidy_kp"], g["pidy_ki"], g["pidy_kd"],
            dt_actual, g["xy_vel_max"]
        )

        # ── Inner loop: velocity → roll/pitch (degrees) ───────────────────────
        # pidVX: velocity error → rollRaw (degrees)
        evx = vel_x_sp - vx
        roll_raw, pid_state["vx_int"], pid_state["vx_prev_err"] = run_pid(
            evx, pid_state["vx_prev_err"], pid_state["vx_int"],
            g["pidvx_kp"], g["pidvx_ki"], g["pidvx_kd"],
            dt_actual, g["rp_limit"]
        )

        # pidVY: velocity error → pitchRaw (degrees)
        evy = vel_y_sp - vy
        pitch_raw, pid_state["vy_int"], pid_state["vy_prev_err"] = run_pid(
            evy, pid_state["vy_prev_err"], pid_state["vy_int"],
            g["pidvy_kp"], g["pidvy_ki"], g["pidvy_kd"],
            dt_actual, g["rp_limit"]
        )

        # ── Yaw rotation compensation (position_controller_pid.c:230-231) ─────
        # attitude->pitch = -(rollRaw*cos(yaw)) - (pitchRaw*sin(yaw))
        # attitude->roll  = -(pitchRaw*cos(yaw)) + (rollRaw*sin(yaw))
        yaw_rad = math.radians(yaw)
        cy = math.cos(yaw_rad)
        sy = math.sin(yaw_rad)

        pitch_cmd = -(roll_raw * cy) - (pitch_raw * sy)
        roll_cmd  = -(pitch_raw * cy) + (roll_raw * sy)

        rp_lim = g["rp_limit"]
        roll_cmd  = max(-rp_lim, min(rp_lim, roll_cmd))
        pitch_cmd = max(-rp_lim, min(rp_lim, pitch_cmd))

        ctrl_roll  = roll_cmd
        ctrl_pitch = pitch_cmd

        # ── Convert to Maddy PWM and send ─────────────────────────────────────
        # Maddy: roll_des = (ch2-1500)/500 * maxRoll  →  ch2 = 1500 + (roll/maxRoll)*500
        ch2 = int(1500 + (roll_cmd  / MADDY_MAX_ROLL)  * 500)
        ch3 = int(1500 + (pitch_cmd / MADDY_MAX_PITCH) * 500)

        ch2 = max(1000, min(2000, ch2))
        ch3 = max(1000, min(2000, ch3))

        send_maddy({"ch2": ch2, "ch3": ch3})


def reset_pid():
    """Reset all PID integrators and previous errors."""
    global pid_state
    pid_state = {
        "x_int": 0.0, "y_int": 0.0,
        "vx_int": 0.0, "vy_int": 0.0,
        "x_prev_err": 0.0, "y_prev_err": 0.0,
        "vx_prev_err": 0.0, "vy_prev_err": 0.0,
    }


# ─── MADDY WEBSOCKET ──────────────────────────────────────────────────────────

def send_maddy(payload: dict):
    """Send JSON payload to Maddy via WebSocket."""
    global ws_handle
    if not WS_AVAILABLE:
        return
    with ws_lock:
        if ws_handle is None:
            return
        try:
            ws_handle.send(json.dumps(payload))
        except Exception as e:
            print(f"[WS] Send error: {e}")


def maddy_ws_thread():
    """Maintains persistent WebSocket connection to Maddy."""
    global ws_handle, ws_connected

    if not WS_AVAILABLE:
        return

    def on_open(ws):
        global ws_connected
        ws_connected = True
        print("[WS] Connected to Maddy")

    def on_close(ws, *args):
        global ws_connected
        ws_connected = False
        print("[WS] Disconnected from Maddy")

    def on_error(ws, err):
        print(f"[WS] Error: {err}")

    def on_message(ws, msg):
        pass  # Telemetry from Maddy (not used here)

    while not stop_event.is_set():
        try:
            ws = websocket.WebSocketApp(
                MADDY_WS_URI,
                on_open=on_open,
                on_close=on_close,
                on_error=on_error,
                on_message=on_message
            )
            with ws_lock:
                ws_handle = ws
            ws.run_forever(ping_interval=5, ping_timeout=3)
        except Exception as e:
            print(f"[WS] Connection failed: {e}")
        finally:
            with ws_lock:
                ws_handle = None
            ws_connected = False
        if not stop_event.is_set():
            print("[WS] Reconnecting in 3s...")
            time.sleep(3)


# ─── LITEWING CFLIB THREAD ────────────────────────────────────────────────────

def lw_log_callback(timestamp, data, logconf):
    """Called at 50 Hz by cflib with Kalman state from LiteWing."""
    with lw_state_lock:
        lw_state["x"]   = data.get("stateEstimate.x",  0.0)
        lw_state["y"]   = data.get("stateEstimate.y",  0.0)
        lw_state["vx"]  = data.get("stateEstimate.vx", 0.0)
        lw_state["vy"]  = data.get("stateEstimate.vy", 0.0)
        lw_state["yaw"] = data.get("stateEstimate.yaw", 0.0)
        lw_state["z"]   = data.get("stateEstimate.z",  0.0)
        lw_state["ready"] = True


def litewing_thread():
    """Connects to LiteWing module and streams Kalman state."""
    global lw_connected

    cflib.crtp.init_drivers(enable_debug_driver=False)

    while not stop_event.is_set():
        try:
            print(f"[LW] Connecting to {LITEWING_URI} ...")
            with SyncCrazyflie(LITEWING_URI, cf=Crazyflie(rw_cache="./cache")) as scf:
                lw_connected = True
                print("[LW] Connected to LiteWing module")

                log_cfg = LogConfig(name="KalmanState", period_in_ms=LOG_PERIOD_MS)
                log_vars = [
                    ("stateEstimate.x",   "float"),
                    ("stateEstimate.y",   "float"),
                    ("stateEstimate.vx",  "float"),
                    ("stateEstimate.vy",  "float"),
                    ("stateEstimate.yaw", "float"),
                    ("stateEstimate.z",   "float"),
                ]

                # Add only variables present in firmware TOC
                toc = scf.cf.log.toc.toc
                added = 0
                for name, vtype in log_vars:
                    group, var = name.split(".", 1)
                    if group in toc and var in toc[group]:
                        log_cfg.add_variable(name, vtype)
                        added += 1
                        print(f"[LW] Logging {name}")
                    else:
                        print(f"[LW] Variable not found: {name}")

                if added == 0:
                    print("[LW] No log variables available — check firmware")
                    break

                log_cfg.data_received_cb.add_callback(lw_log_callback)
                scf.cf.log.add_config(log_cfg)
                log_cfg.start()

                while not stop_event.is_set():
                    time.sleep(0.1)

                log_cfg.stop()

        except Exception as e:
            print(f"[LW] Error: {e}")
        finally:
            lw_connected = False
            with lw_state_lock:
                lw_state["ready"] = False

        if not stop_event.is_set():
            print("[LW] Reconnecting in 5s...")
            time.sleep(5)


# ─── GUI ──────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LiteWing Position Hold — Maddy Flight Controller")
        self.resizable(False, False)
        self._build_ui()
        self._update_ui()

    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}

        # ── Status bar ──────────────────────────────────────────────────────
        status_frame = ttk.LabelFrame(self, text="Connection Status")
        status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        self.lbl_lw  = ttk.Label(status_frame, text="LiteWing: DISCONNECTED", foreground="red")
        self.lbl_lw.grid(row=0, column=0, **pad)
        self.lbl_ws  = ttk.Label(status_frame, text="Maddy WS: DISCONNECTED", foreground="red")
        self.lbl_ws.grid(row=0, column=1, **pad)

        # ── State display ───────────────────────────────────────────────────
        state_frame = ttk.LabelFrame(self, text="LiteWing Kalman State")
        state_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        labels = ["x (m)", "y (m)", "vx (m/s)", "vy (m/s)", "z (m)", "yaw (°)"]
        self.state_vars = []
        for i, lbl in enumerate(labels):
            ttk.Label(state_frame, text=lbl + ":").grid(row=i//3, column=(i%3)*2,     sticky="e", **pad)
            sv = tk.StringVar(value="0.000")
            ttk.Label(state_frame, textvariable=sv, width=8).grid(row=i//3, column=(i%3)*2+1, sticky="w", **pad)
            self.state_vars.append(sv)

        self.sv_target_x = tk.StringVar(value="0.000")
        self.sv_target_y = tk.StringVar(value="0.000")
        ttk.Label(state_frame, text="target x:").grid(row=2, column=0, sticky="e", **pad)
        ttk.Label(state_frame, textvariable=self.sv_target_x, width=8).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(state_frame, text="target y:").grid(row=2, column=2, sticky="e", **pad)
        ttk.Label(state_frame, textvariable=self.sv_target_y, width=8).grid(row=2, column=3, sticky="w", **pad)

        # ── Control output ──────────────────────────────────────────────────
        ctrl_frame = ttk.LabelFrame(self, text="Control Output")
        ctrl_frame.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        self.sv_roll  = tk.StringVar(value="0.00")
        self.sv_pitch = tk.StringVar(value="0.00")
        self.sv_ch2   = tk.StringVar(value="1500")
        self.sv_ch3   = tk.StringVar(value="1500")

        ttk.Label(ctrl_frame, text="roll cmd (°):").grid(row=0, column=0, sticky="e", **pad)
        ttk.Label(ctrl_frame, textvariable=self.sv_roll,  width=8).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(ctrl_frame, text="pitch cmd (°):").grid(row=0, column=2, sticky="e", **pad)
        ttk.Label(ctrl_frame, textvariable=self.sv_pitch, width=8).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(ctrl_frame, text="ch2 PWM:").grid(row=1, column=0, sticky="e", **pad)
        ttk.Label(ctrl_frame, textvariable=self.sv_ch2, width=8).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(ctrl_frame, text="ch3 PWM:").grid(row=1, column=2, sticky="e", **pad)
        ttk.Label(ctrl_frame, textvariable=self.sv_ch3, width=8).grid(row=1, column=3, sticky="w", **pad)

        # ── PID gains ───────────────────────────────────────────────────────
        gain_frame = ttk.LabelFrame(self, text="PID Gains")
        gain_frame.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        gain_defs = [
            ("Pos X kp", "pidx_kp",  PIDX_KP),
            ("Pos X ki", "pidx_ki",  PIDX_KI),
            ("Pos Y kp", "pidy_kp",  PIDY_KP),
            ("Pos Y ki", "pidy_ki",  PIDY_KI),
            ("Vel X kp", "pidvx_kp", PIDVX_KP),
            ("Vel X ki", "pidvx_ki", PIDVX_KI),
            ("Vel Y kp", "pidvy_kp", PIDVY_KP),
            ("Vel Y ki", "pidvy_ki", PIDVY_KI),
            ("XY vel max", "xy_vel_max", XY_VEL_MAX),
            ("RP limit °", "rp_limit",   RP_LIMIT),
        ]

        self.gain_entries = {}
        for i, (label, key, default) in enumerate(gain_defs):
            r, c = divmod(i, 2)
            ttk.Label(gain_frame, text=label + ":").grid(row=r, column=c*3,   sticky="e", **pad)
            sv = tk.StringVar(value=str(default))
            entry = ttk.Entry(gain_frame, textvariable=sv, width=7)
            entry.grid(row=r, column=c*3+1, sticky="w", **pad)
            self.gain_entries[key] = sv

        ttk.Button(gain_frame, text="Apply Gains", command=self._apply_gains).grid(
            row=len(gain_defs)//2 + 1, column=0, columnspan=6, pady=4)

        # ── Pos hold controls ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=4, column=0, columnspan=2, **pad)

        self.btn_poshold = ttk.Button(btn_frame, text="Enable Position Hold",
                                       command=self._toggle_poshold, width=24)
        self.btn_poshold.grid(row=0, column=0, padx=8)

        ttk.Button(btn_frame, text="Reset Target Here",
                   command=self._reset_target).grid(row=0, column=1, padx=8)

        ttk.Button(btn_frame, text="Emergency Stop",
                   command=self._emergency_stop, style="Danger.TButton").grid(row=0, column=2, padx=8)

        # Style
        style = ttk.Style()
        style.configure("Danger.TButton", foreground="red")

        # ── URI config ──────────────────────────────────────────────────────
        uri_frame = ttk.LabelFrame(self, text="Connection Config")
        uri_frame.grid(row=5, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(uri_frame, text="LiteWing URI:").grid(row=0, column=0, sticky="e", **pad)
        self.sv_lw_uri = tk.StringVar(value=LITEWING_URI)
        ttk.Entry(uri_frame, textvariable=self.sv_lw_uri, width=28).grid(row=0, column=1, **pad)

        ttk.Label(uri_frame, text="Maddy WS URI:").grid(row=1, column=0, sticky="e", **pad)
        self.sv_ws_uri = tk.StringVar(value=MADDY_WS_URI)
        ttk.Entry(uri_frame, textvariable=self.sv_ws_uri, width=28).grid(row=1, column=1, **pad)

    def _apply_gains(self):
        with gains_lock:
            for key, sv in self.gain_entries.items():
                try:
                    gains[key] = float(sv.get())
                except ValueError:
                    pass
        print("[GUI] Gains updated:", {k: gains[k] for k in gains})

    def _toggle_poshold(self):
        global poshold_enabled, target_x, target_y
        if not poshold_enabled:
            # Capture current position as target
            with lw_state_lock:
                target_x = lw_state["x"]
                target_y = lw_state["y"]
                ready    = lw_state["ready"]
            if not ready:
                print("[GUI] LiteWing not ready — cannot enable pos hold")
                return
            reset_pid()
            poshold_enabled = True
            send_maddy({"poshold": 1})
            self.btn_poshold.config(text="Disable Position Hold")
            print(f"[GUI] Position hold ENABLED — target ({target_x:.3f}, {target_y:.3f})")
        else:
            poshold_enabled = False
            send_maddy({"poshold": 0, "ch2": 1500, "ch3": 1500})
            self.btn_poshold.config(text="Enable Position Hold")
            print("[GUI] Position hold DISABLED")

    def _reset_target(self):
        global target_x, target_y
        with lw_state_lock:
            target_x = lw_state["x"]
            target_y = lw_state["y"]
        print(f"[GUI] Target reset to ({target_x:.3f}, {target_y:.3f})")

    def _emergency_stop(self):
        global poshold_enabled
        poshold_enabled = False
        send_maddy({"poshold": 0, "ch2": 1500, "ch3": 1500})
        print("[GUI] EMERGENCY STOP — pos hold off, ch2/ch3 centered")

    def _update_ui(self):
        """Refresh GUI labels at ~10 Hz."""
        # Connection status
        self.lbl_lw.config(
            text="LiteWing: " + ("CONNECTED" if lw_connected else "DISCONNECTED"),
            foreground="green" if lw_connected else "red"
        )
        self.lbl_ws.config(
            text="Maddy WS: " + ("CONNECTED" if ws_connected else "DISCONNECTED"),
            foreground="green" if ws_connected else "red"
        )

        # Kalman state
        with lw_state_lock:
            vals = [lw_state["x"], lw_state["y"],
                    lw_state["vx"], lw_state["vy"],
                    lw_state["z"],  lw_state["yaw"]]
        for sv, v in zip(self.state_vars, vals):
            sv.set(f"{v:+.3f}")

        self.sv_target_x.set(f"{target_x:+.3f}")
        self.sv_target_y.set(f"{target_y:+.3f}")

        # Control output
        self.sv_roll.set(f"{ctrl_roll:+.2f}")
        self.sv_pitch.set(f"{ctrl_pitch:+.2f}")
        ch2 = int(1500 + (ctrl_roll  / MADDY_MAX_ROLL)  * 500)
        ch3 = int(1500 + (ctrl_pitch / MADDY_MAX_PITCH) * 500)
        self.sv_ch2.set(str(max(1000, min(2000, ch2))))
        self.sv_ch3.set(str(max(1000, min(2000, ch3))))

        self.after(100, self._update_ui)

    def on_close(self):
        stop_event.set()
        if poshold_enabled:
            send_maddy({"poshold": 0, "ch2": 1500, "ch3": 1500})
        self.destroy()


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LiteWing Position Hold for Maddy Flight Controller")
    print("=" * 60)
    print(f"  LiteWing URI : {LITEWING_URI}")
    print(f"  Maddy WS URI : {MADDY_WS_URI}")
    print(f"  Control rate : {1000//LOG_PERIOD_MS} Hz")
    print()

    # Start background threads
    t_lw   = threading.Thread(target=litewing_thread, daemon=True)
    t_ctrl = threading.Thread(target=control_loop,    daemon=True)
    t_ws   = threading.Thread(target=maddy_ws_thread, daemon=True)

    t_lw.start()
    t_ctrl.start()
    t_ws.start()

    # Launch GUI
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

    stop_event.set()
    print("Exiting.")


if __name__ == "__main__":
    main()
