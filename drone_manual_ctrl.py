#!/usr/bin/env python3.11
"""
Drone Manual Controller — standalone keyboard control window.

Key mapping:
  W / S        : Throttle up / down
  A / D        : Yaw left / right
  ↑ / ↓        : Pitch forward / back
  ← / →        : Roll left / right
  SPACE        : Arm / Disarm toggle
  ESC          : Emergency disarm

Requires:
  pip3.11 install websocket-client
"""

import tkinter as tk
import threading
import json
import websocket   # pip3.11 install websocket-client

# ── Tunable defaults ──────────────────────────────────────────────────────────
DEFAULT_IP        = "10.198.219.30"
THROTTLE_STEP     = 10      # PWM per 50 ms tick
DEFLECTION        = 300     # PWM deflection for pitch / roll / yaw
SEND_HZ           = 20      # update rate
AUTO_TRIM_THR_DEG = 3.0     # degrees before auto-trim kicks in
AUTO_TRIM_STEP    = 2       # PWM per auto-trim cycle (every 800 ms)
AUTO_TRIM_MAX     = 100     # hard cap ±100 PWM

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG   = "#0a0a0a"
C_PANEL= "#111111"
C_FG   = "#00ff41"
C_DIM  = "#336633"
C_WARN = "#ffaa00"
C_ERR  = "#ff4444"
C_VAL  = "#88ff88"


class DroneCtrl:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Drone Manual Controller")
        root.configure(bg=C_BG)
        root.resizable(False, False)

        # ── Flight state ──────────────────────────────────────────────────
        self.throttle  = 1000
        self.roll      = 1500
        self.pitch     = 1500
        self.yaw       = 1500
        self.armed     = False

        self.trim_pitch = 0
        self.trim_roll  = 0
        self.auto_trim  = False

        self.pressed   = set()
        self.ws        = None
        self.connected = False

        # Telemetry ring buffer (last 200 packets ≈ 20 s at 10 Hz)
        self.tel_buf  = []
        self.tel_lock = threading.Lock()

        self._auto_trim_counter = 0   # counts send ticks → triggers trim every 800 ms

        self._build_ui()
        self._bind_keys()
        root.after(int(1000 / SEND_HZ), self._loop)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        F  = ("Courier", 10)
        FB = ("Courier", 10, "bold")
        FL = ("Courier", 24, "bold")
        FM = ("Courier", 12)
        p  = dict(padx=8, pady=3)

        def panel(parent=None):
            return tk.Frame(parent or self.root, bg=C_PANEL,
                            bd=1, relief="groove")

        def lbl(parent, text, font=F, fg=C_FG, **kw):
            return tk.Label(parent, text=text, font=font,
                            fg=fg, bg=C_PANEL, **kw)

        def val_lbl(parent, init="—", width=8):
            l = tk.Label(parent, text=init, font=FM,
                         fg=C_VAL, bg=C_PANEL, width=width)
            return l

        # ── Title ─────────────────────────────────────────────────────────
        tk.Label(self.root, text="DRONE MANUAL CONTROLLER",
                 font=("Courier", 13, "bold"), fg=C_FG, bg=C_BG
                 ).pack(fill="x", pady=(10, 2))

        # ── Connection ────────────────────────────────────────────────────
        cf = panel(); cf.pack(fill="x", padx=10, pady=4)
        lbl(cf, "IP:").grid(row=0, column=0, **p)
        self.ip_var   = tk.StringVar(value=DEFAULT_IP)
        self.ip_entry = tk.Entry(cf, textvariable=self.ip_var,
                                 font=F, fg=C_FG, bg="#1a1a1a",
                                 insertbackground=C_FG, width=20)
        self.ip_entry.grid(row=0, column=1, **p)
        self.conn_btn = tk.Button(cf, text="CONNECT", font=FB,
                                  fg=C_BG, bg=C_FG, width=10,
                                  command=self._toggle_connect)
        self.conn_btn.grid(row=0, column=2, **p)
        self.status_lbl = tk.Label(cf, text="● DISCONNECTED",
                                   font=F, fg=C_ERR, bg=C_PANEL)
        self.status_lbl.grid(row=0, column=3, **p)

        # ── Throttle ──────────────────────────────────────────────────────
        tf = panel(); tf.pack(fill="x", padx=10, pady=4)
        lbl(tf, "THROTTLE", font=FB).pack()
        self.thr_pct = tk.Label(tf, text="0%", font=FL, fg=C_FG, bg=C_PANEL)
        self.thr_pct.pack()
        self.thr_bar = tk.Canvas(tf, height=16, bg="#1a1a1a",
                                 highlightthickness=0)
        self.thr_bar.pack(fill="x", padx=10, pady=(0, 6))

        # ── Channel readout ───────────────────────────────────────────────
        ch_f = panel(); ch_f.pack(fill="x", padx=10, pady=4)
        for col, (name, attr) in enumerate([
                ("CH1 THR",  "ch1_v"),
                ("CH2 ROLL", "ch2_v"),
                ("CH3 PITCH","ch3_v"),
                ("CH4 YAW",  "ch4_v"),
        ]):
            lbl(ch_f, name).grid(row=0, column=col*2, **p)
            v = val_lbl(ch_f, "1000")
            v.grid(row=0, column=col*2+1)
            setattr(self, attr, v)
        self.arm_lbl = tk.Label(ch_f, text="DISARMED",
                                font=FB, fg=C_ERR, bg=C_PANEL)
        self.arm_lbl.grid(row=1, column=0, columnspan=8, pady=3)

        # ── Sensitivity ───────────────────────────────────────────────────
        sf = panel(); sf.pack(fill="x", padx=10, pady=4)
        lbl(sf, "SENSITIVITY", font=FB).grid(row=0, column=0,
                                             columnspan=4, pady=(4,2))
        lbl(sf, "Throttle step:").grid(row=1, column=0, **p)
        self.step_var = tk.IntVar(value=THROTTLE_STEP)
        tk.Spinbox(sf, from_=5, to=50, increment=5,
                   textvariable=self.step_var, width=5,
                   font=F, fg=C_FG, bg="#1a1a1a",
                   buttonbackground="#1a1a1a").grid(row=1, column=1, **p)
        lbl(sf, "Deflection:").grid(row=1, column=2, **p)
        self.defl_var = tk.IntVar(value=DEFLECTION)
        tk.Spinbox(sf, from_=50, to=500, increment=50,
                   textvariable=self.defl_var, width=5,
                   font=F, fg=C_FG, bg="#1a1a1a",
                   buttonbackground="#1a1a1a").grid(row=1, column=3, **p)

        # ── Trim ──────────────────────────────────────────────────────────
        tr_f = panel(); tr_f.pack(fill="x", padx=10, pady=4)
        lbl(tr_f, "TRIM", font=FB).grid(row=0, column=0,
                                        columnspan=6, pady=(4, 2))

        for row, axis in enumerate(("PITCH", "ROLL"), start=1):
            lbl(tr_f, f"{axis}:").grid(row=row, column=0, **p)
            tk.Button(tr_f, text=" – ", font=F, fg=C_FG, bg="#1a1a1a",
                      command=lambda a=axis.lower(): self._adj_trim(a, -5)
                      ).grid(row=row, column=1, padx=2)
            disp = tk.Label(tr_f, text="0", font=FM, fg=C_VAL,
                            bg=C_PANEL, width=5)
            disp.grid(row=row, column=2)
            setattr(self, f"trim_{axis.lower()}_lbl", disp)
            tk.Button(tr_f, text=" + ", font=F, fg=C_FG, bg="#1a1a1a",
                      command=lambda a=axis.lower(): self._adj_trim(a, +5)
                      ).grid(row=row, column=3, padx=2)

        self.auto_trim_var = tk.BooleanVar()
        tk.Checkbutton(tr_f, text="AUTO-TRIM (when hovering)",
                       variable=self.auto_trim_var,
                       command=self._on_auto_trim_toggle,
                       font=F, fg=C_FG, bg=C_PANEL,
                       selectcolor="#1a1a1a",
                       activebackground=C_PANEL,
                       activeforeground=C_FG
                       ).grid(row=3, column=0, columnspan=4, pady=4)
        tk.Button(tr_f, text="RESET TRIM", font=F, fg=C_BG, bg=C_FG,
                  command=self._reset_trim
                  ).grid(row=3, column=4, columnspan=2, padx=6)

        self.auto_trim_status = tk.Label(tr_f, text="",
                                         font=("Courier", 9),
                                         fg=C_DIM, bg=C_PANEL)
        self.auto_trim_status.grid(row=4, column=0, columnspan=6, pady=2)

        # ── Live telemetry ────────────────────────────────────────────────
        tel_f = panel(); tel_f.pack(fill="x", padx=10, pady=4)
        lbl(tel_f, "TELEMETRY", font=FB).grid(row=0, column=0,
                                              columnspan=4, pady=(4, 2))
        tels = [
            ("Pitch IMU", "tel_p"), ("Roll IMU", "tel_r"),
            ("Yaw IMU",   "tel_y"), ("M1-M4",    "tel_m"),
        ]
        for i, (name, attr) in enumerate(tels):
            lbl(tel_f, f"{name}:").grid(row=1+i//2, column=(i%2)*2,
                                         padx=8, pady=1, sticky="e")
            v = val_lbl(tel_f)
            v.grid(row=1+i//2, column=(i%2)*2+1, sticky="w")
            setattr(self, attr, v)

        # ── Key reference ─────────────────────────────────────────────────
        ref_f = tk.Frame(self.root, bg=C_BG)
        ref_f.pack(fill="x", padx=10, pady=(4, 10))
        tk.Label(ref_f,
                 text=("  W/S: Throttle ↑↓      A/D: Yaw ←→\n"
                       "  ↑/↓: Pitch F/B      ←/→: Roll L/R\n"
                       "  SPACE: ARM/DISARM      ESC: Disarm"),
                 font=("Courier", 9), fg=C_DIM, bg=C_BG,
                 justify="left").pack(anchor="w")

    # ── Key binding ───────────────────────────────────────────────────────────
    def _bind_keys(self):
        self.root.bind_all("<KeyPress>",   self._on_key_down)
        self.root.bind_all("<KeyRelease>", self._on_key_up)
        self.root.focus_set()

    def _on_key_down(self, e):
        if e.widget is self.ip_entry:
            return
        self.pressed.add(e.keysym)
        if e.keysym == "space":
            self._toggle_arm()
        elif e.keysym == "Escape":
            self._disarm()

    def _on_key_up(self, e):
        self.pressed.discard(e.keysym)
        # Re-centre attitude axes on release
        if e.keysym in ("Up", "Down") and \
                "Up" not in self.pressed and "Down" not in self.pressed:
            self.pitch = 1500
        if e.keysym in ("Left", "Right") and \
                "Left" not in self.pressed and "Right" not in self.pressed:
            self.roll = 1500
        if e.keysym in ("a", "d") and \
                "a" not in self.pressed and "d" not in self.pressed:
            self.yaw = 1500

    # ── Key → channel values ──────────────────────────────────────────────────
    def _process_keys(self):
        step = self.step_var.get()
        defl = self.defl_var.get()

        if "w" in self.pressed:
            self.throttle = min(2000, self.throttle + step)
        if "s" in self.pressed:
            self.throttle = max(1000, self.throttle - step)

        if "a" in self.pressed and "d" not in self.pressed:
            self.yaw = 1500 - defl
        elif "d" in self.pressed and "a" not in self.pressed:
            self.yaw = 1500 + defl

        if "Up" in self.pressed and "Down" not in self.pressed:
            self.pitch = 1500 + defl
        elif "Down" in self.pressed and "Up" not in self.pressed:
            self.pitch = 1500 - defl

        if "Right" in self.pressed and "Left" not in self.pressed:
            self.roll = 1500 + defl
        elif "Left" in self.pressed and "Right" not in self.pressed:
            self.roll = 1500 - defl

        self.throttle = max(1000, min(2000, self.throttle))
        self.roll     = max(1000, min(2000, self.roll))
        self.pitch    = max(1000, min(2000, self.pitch))
        self.yaw      = max(1000, min(2000, self.yaw))

    # ── Main loop (20 Hz) ─────────────────────────────────────────────────────
    def _loop(self):
        self._process_keys()
        self._update_display()
        if self.connected:
            self._maybe_auto_trim()
            self._send_update()
        self.root.after(int(1000 / SEND_HZ), self._loop)

    # ── WebSocket send ────────────────────────────────────────────────────────
    def _send_update(self):
        if not self.ws:
            return
        fr = max(1000, min(2000, self.roll  + self.trim_roll))
        fp = max(1000, min(2000, self.pitch + self.trim_pitch))
        data = {
            "ch1": self.throttle,
            "ch2": fr,
            "ch3": fp,
            "ch4": self.yaw,
            "ch5": 1000 if self.armed else 2000,
            "ch6": 1000,
            "althold": 0,
            "poshold": 0,
        }
        try:
            self.ws.send(json.dumps(data))
        except Exception:
            self.root.after(0, self._on_disconnect)

    # ── Display update ────────────────────────────────────────────────────────
    def _update_display(self):
        pct = round((self.throttle - 1000) / 10)
        self.thr_pct.config(text=f"{pct}%")

        w = self.thr_bar.winfo_width()
        if w > 4:
            self.thr_bar.delete("all")
            fw = max(0, int(w * pct / 100))
            colour = C_FG if pct < 70 else C_WARN if pct < 85 else C_ERR
            if fw:
                self.thr_bar.create_rectangle(0, 0, fw, 16,
                                              fill=colour, outline="")

        fr = max(1000, min(2000, self.roll  + self.trim_roll))
        fp = max(1000, min(2000, self.pitch + self.trim_pitch))
        self.ch1_v.config(text=str(self.throttle))
        self.ch2_v.config(text=str(fr))
        self.ch3_v.config(text=str(fp))
        self.ch4_v.config(text=str(self.yaw))

        if self.armed:
            self.arm_lbl.config(text="ARMED", fg=C_FG)
        else:
            self.arm_lbl.config(text="DISARMED", fg=C_ERR)

    # ── Trim helpers ──────────────────────────────────────────────────────────
    def _adj_trim(self, axis: str, delta: int):
        if axis == "pitch":
            self.trim_pitch = max(-AUTO_TRIM_MAX,
                                  min(AUTO_TRIM_MAX, self.trim_pitch + delta))
            self.trim_pitch_lbl.config(text=str(self.trim_pitch))
        else:
            self.trim_roll = max(-AUTO_TRIM_MAX,
                                 min(AUTO_TRIM_MAX, self.trim_roll + delta))
            self.trim_roll_lbl.config(text=str(self.trim_roll))

    def _reset_trim(self):
        self.trim_pitch = 0
        self.trim_roll  = 0
        self.trim_pitch_lbl.config(text="0")
        self.trim_roll_lbl.config(text="0")
        self.auto_trim_status.config(text="Trim reset.")

    def _on_auto_trim_toggle(self):
        self.auto_trim = self.auto_trim_var.get()
        self.auto_trim_status.config(
            text="Auto-trim active — monitoring…" if self.auto_trim else "")

    # Auto-trim runs every ~800 ms (every 16 send ticks at 20 Hz)
    def _maybe_auto_trim(self):
        self._auto_trim_counter += 1
        if self._auto_trim_counter < 16:
            return
        self._auto_trim_counter = 0
        if not self.auto_trim or not self.armed:
            return
        if abs(self.pitch - 1500) > 50 or abs(self.roll - 1500) > 50:
            return   # user is commanding, don't interfere

        with self.tel_lock:
            recent = self.tel_buf[-20:]
        if len(recent) < 5:
            return

        avg_p = sum(d.get("p", 0) for d in recent) / len(recent)
        avg_r = sum(d.get("r", 0) for d in recent) / len(recent)

        changed = []
        if avg_p > AUTO_TRIM_THR_DEG:
            self.trim_pitch = min(AUTO_TRIM_MAX,
                                  self.trim_pitch + AUTO_TRIM_STEP)
            changed.append(f"P→{self.trim_pitch}")
        elif avg_p < -AUTO_TRIM_THR_DEG:
            self.trim_pitch = max(-AUTO_TRIM_MAX,
                                  self.trim_pitch - AUTO_TRIM_STEP)
            changed.append(f"P→{self.trim_pitch}")

        if avg_r > AUTO_TRIM_THR_DEG:
            self.trim_roll = min(AUTO_TRIM_MAX,
                                 self.trim_roll + AUTO_TRIM_STEP)
            changed.append(f"R→{self.trim_roll}")
        elif avg_r < -AUTO_TRIM_THR_DEG:
            self.trim_roll = max(-AUTO_TRIM_MAX,
                                 self.trim_roll - AUTO_TRIM_STEP)
            changed.append(f"R→{self.trim_roll}")

        if changed:
            self.trim_pitch_lbl.config(text=str(self.trim_pitch))
            self.trim_roll_lbl.config(text=str(self.trim_roll))
            self.auto_trim_status.config(
                text=f"Corrected: {' '.join(changed)}  "
                     f"(P={avg_p:.1f}° R={avg_r:.1f}°)")
        else:
            self.auto_trim_status.config(
                text=f"Stable ✓  (P={avg_p:.1f}° R={avg_r:.1f}°)")

    # ── Arm / disarm ──────────────────────────────────────────────────────────
    def _toggle_arm(self):
        if self.armed:
            self._disarm()
        else:
            if self.throttle > 1050:
                self.auto_trim_status.config(
                    text="Lower throttle to 0% before arming!")
                return
            self.armed = True

    def _disarm(self):
        self.armed    = False
        self.throttle = 1000

    # ── WebSocket connection ──────────────────────────────────────────────────
    def _toggle_connect(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        ip  = self.ip_var.get().strip()
        url = f"ws://{ip}:81"

        def on_open(ws):
            self.connected = True
            self.root.after(0, lambda: self.status_lbl.config(
                text=f"● CONNECTED  ({ip})", fg=C_FG))
            self.root.after(0, lambda: self.conn_btn.config(text="DISCONNECT"))

        def on_message(ws, msg):
            try:
                d = json.loads(msg)
                if d.get("tel") == 1:
                    with self.tel_lock:
                        self.tel_buf.append(d)
                        if len(self.tel_buf) > 200:
                            self.tel_buf.pop(0)
                    self.root.after(0, lambda dd=d: self._update_telemetry(dd))
            except Exception:
                pass

        def on_error(ws, err):
            self.root.after(0, lambda: self.auto_trim_status.config(
                text=f"WS error: {err}"))

        def on_close(ws, *args):
            self.root.after(0, self._on_disconnect)

        self.ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        self.status_lbl.config(text=f"Connecting to {ip}…", fg=C_WARN)

    def _disconnect(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    def _on_disconnect(self):
        self.connected = False
        self.ws        = None
        self.armed     = False
        self.throttle  = 1000
        self.status_lbl.config(text="● DISCONNECTED", fg=C_ERR)
        self.conn_btn.config(text="CONNECT")

    def _update_telemetry(self, d: dict):
        self.tel_p.config(text=f"{d.get('p', 0):.1f}°")
        self.tel_r.config(text=f"{d.get('r', 0):.1f}°")
        self.tel_y.config(text=f"{d.get('y', 0):.1f}°")
        m = [d.get(f"m{i}", 0) for i in range(1, 5)]
        self.tel_m.config(text=" ".join(str(x) for x in m))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    DroneCtrl(root)
    root.mainloop()
