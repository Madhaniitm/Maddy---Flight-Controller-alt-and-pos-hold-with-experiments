"""
EXP-C1-HW: Natural Language → Tool Chain — HARDWARE  (N=3 runs)
================================================================
Sends the command "take off and hover at 1 metre" to the LLM agent.
The agent executes the tool sequence on the REAL Maddy drone via WebSocket.

Produces:
  results/C1_hw_runs.csv          — per-run metrics
  results/C1_hw_vs_sim.png        — hardware altitude traces + sim overlay

Usage:
  python exp_C1_nl_to_toolchain_HW.py --ip 10.198.219.30
  python exp_C1_nl_to_toolchain_HW.py --ip 192.168.4.1 --n_runs 3

Safety:
  Ctrl+C → emergency land + disarm
  LLM cannot disarm mid-air (disarm tool blocked while z > 0.1 m)
"""

import sys, os, csv, math, time, json, threading, argparse, urllib.request, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import websocket
except ImportError:
    print("ERROR: pip install websocket-client")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ip",     default="10.198.219.30")
parser.add_argument("--n_runs", type=int, default=3)
args = parser.parse_args()

DRONE_IP = args.ip
WS_URL   = f"ws://{DRONE_IP}:81"
N_RUNS   = args.n_runs

OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_RUNS = os.path.join(OUT_DIR, "C1_hw_runs.csv")
OUT_PNG  = os.path.join(OUT_DIR, "C1_hw_vs_sim.png")
SIM_PNG  = os.path.join(OUT_DIR, "C1_nl_to_toolchain_guardrail_on.png")

COMMAND    = "take off and hover at 1 metre"
TARGET_ALT = 1.0
TOLERANCE  = 0.10
SEND_HZ    = 20
SEND_DT    = 1.0 / SEND_HZ

# ── LLM API (same Azure endpoint as c_series_agent.py) ───────────────────────
_ENDPOINT = "https://claude-test-madhan-resource.services.ai.azure.com/anthropic/v1/messages?api-version=2025-01-01-preview"
_API_KEY  = os.environ.get(
    "ANTHROPIC_API_KEY",
    "EpilO2YT1tLIiwwKoIqCv9oodffWWedT4R7gJdocTTrSVwCC2GEUJQQJ99CCACfhMk5XJ3w3AAAAACOGGVL2"
)
_MODEL    = "claude-sonnet-4-6"
MAX_TURNS = 30

# ── System prompt (hardware version) ─────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an AUTONOMOUS FLIGHT AGENT for a REAL 50 g nano-quadrotor drone
(custom firmware, brushed motors, ToF altitude sensor, ESP32-S3).

━━ STANDARD TAKEOFF SEQUENCE ━━
  arm()
  find_hover_throttle()           ← ramps throttle until drone hovers; locks baseline
  check_drone_stable()
  enable_altitude_hold()          ← engages firmware althold at current altitude
  wait(2.0)
  set_altitude_target(target_m)
  wait(4.0)
  check_altitude_reached(target_m, 0.10)

━━ LANDING SEQUENCE ━━
  land()                          ← ramps throttle down, waits for z < 0.05 m, disarms

━━ OBSERVATION TOOLS ━━
  get_sensor_status()             ← live EKF altitude, roll, pitch, althold status
  check_altitude_reached(m, tol)  ← ✓/✗ based on EKF altitude
  check_drone_stable(max_deg=5.0) ← ✓/✗ based on roll/pitch

━━ SAFETY ━━
  disarm() is BLOCKED while z > 0.10 m. Always use land() to descend.
  Do NOT call enable_altitude_hold() more than once per flight.

━━ IMPORTANT ━━
  This is a REAL drone. Every tool call is executed immediately on hardware.
  Be deliberate. Verify each step with check_altitude_reached or get_sensor_status.
"""

# ── Hardware agent ─────────────────────────────────────────────────────────────
class HardwareAgent:
    def __init__(self, run_idx):
        self.run_idx    = run_idx
        self._lock      = threading.Lock()
        self._ctrl      = {"ch1": 1000, "altset": 1.0, "althold": 0, "armed": False}
        self._tel       = []
        self._ws        = None
        self._stop      = threading.Event()
        self._tel_log   = []    # (mono_t, lw_z_mm) for this run
        self._hover_thr = 0.54  # default; updated by find_hover_throttle
        self._t0        = None  # run start time

    # ── WebSocket ─────────────────────────────────────────────────────────────
    def connect(self):
        ws = websocket.WebSocketApp(
            WS_URL,
            on_message=self._on_msg,
            on_error=lambda ws, e: print(f"[WS] {e}"),
            on_close=lambda ws, *a: None,
            on_open=lambda ws: print(f"[HW R{self.run_idx+1}] WS connected"),
        )
        self._ws = ws
        threading.Thread(target=ws.run_forever, daemon=True).start()
        time.sleep(1.0)
        threading.Thread(target=self._sender, daemon=True).start()

    def disconnect(self):
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _on_msg(self, ws, msg):
        try:
            d = json.loads(msg)
            if d.get("tel") == 1:
                d["_mono"] = time.monotonic()
                with self._lock:
                    self._tel.append(d)
                    if len(self._tel) > 1000:
                        self._tel.pop(0)
                lw_z = d.get("lw_z")
                if lw_z is not None and self._t0 is not None:
                    self._tel_log.append((d["_mono"] - self._t0, lw_z))
        except Exception:
            pass

    def _sender(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._lock:
                if self._ws:
                    pkt = {
                        "ch1":     self._ctrl["ch1"],
                        "ch2":     1500,
                        "ch3":     1500,
                        "ch4":     1500,
                        "ch5":     1000 if self._ctrl["armed"] else 2000,
                        "ch6":     1000,
                        "altset":  self._ctrl["altset"],
                        "althold": self._ctrl["althold"],
                        "poshold": 0,
                    }
                    try:
                        self._ws.send(json.dumps(pkt))
                    except Exception:
                        pass
            rem = SEND_DT - (time.monotonic() - t0)
            if rem > 0:
                time.sleep(rem)

    # ── Telemetry helpers ─────────────────────────────────────────────────────
    def _last_tel(self):
        with self._lock:
            return self._tel[-1] if self._tel else None

    def _alt_m(self):
        d = self._last_tel()
        if d is None:
            return None
        lw_z = d.get("lw_z")
        return lw_z / 1000.0 if lw_z is not None else None

    def _wait_tel(self, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._last_tel() is not None:
                return True
            time.sleep(0.05)
        return False

    # ── Internal control ──────────────────────────────────────────────────────
    def _set_thr(self, pwm):
        with self._lock:
            self._ctrl["ch1"] = int(max(1000, min(2000, pwm)))

    def _set_altset(self, m):
        with self._lock:
            self._ctrl["altset"] = float(m)

    def _safe_disarm(self):
        with self._lock:
            self._ctrl["armed"]   = False
            self._ctrl["althold"] = 0
            self._ctrl["ch1"]     = 1000

    # ── Tool implementations ──────────────────────────────────────────────────
    def _tool_arm(self):
        with self._lock:
            self._ctrl["armed"] = True
            self._ctrl["ch1"]   = 1000
        time.sleep(0.5)
        self._t0 = time.monotonic()
        return "Drone armed. Motors live. Proceed to find_hover_throttle()."

    def _tool_disarm(self):
        alt = self._alt_m()
        if alt is not None and alt > 0.10:
            return f"BLOCKED: altitude {alt:.3f} m > 0.10 m. Call land() first."
        self._safe_disarm()
        return "Drone disarmed."

    def _tool_find_hover_throttle(self):
        print(f"  [find_hover_throttle] ramping …")
        prev_alts = []
        pwm = 1200
        while pwm <= 1700:
            self._set_thr(pwm)
            time.sleep(0.08)
            alt = self._alt_m()
            if alt is not None:
                prev_alts.append(alt)
                if len(prev_alts) > 5:
                    prev_alts.pop(0)
                    vz_est = (prev_alts[-1] - prev_alts[0]) / (5 * 0.08)
                    if alt > 0.40 and abs(vz_est) < 0.08:
                        self._hover_thr = round(pwm / 2000.0, 3)
                        return (f"Hover throttle found: PWM={pwm} ({self._hover_thr*100:.1f}%) "
                                f"at alt={alt:.3f} m. vz≈{vz_est:.3f} m/s.")
            pwm += 5
        self._hover_thr = 0.54
        return f"Hover search complete. Using PWM={pwm} as baseline."

    def _tool_check_drone_stable(self, max_degrees=5.0):
        d = self._last_tel()
        if d is None:
            return "No telemetry. Wait and retry."
        roll  = abs(d.get("r", 0))
        pitch = abs(d.get("p", 0))
        if roll <= max_degrees and pitch <= max_degrees:
            return f"Drone stable ✓  roll={roll:.1f}°  pitch={pitch:.1f}°"
        return f"Drone unstable ✗  roll={roll:.1f}°  pitch={pitch:.1f}° (limit={max_degrees}°)"

    def _tool_enable_altitude_hold(self):
        alt = self._alt_m()
        sp = round(alt, 2) if alt is not None else 1.0
        with self._lock:
            self._ctrl["altset"]  = sp
            self._ctrl["althold"] = 1
        return f"Altitude hold engaged at {sp:.3f} m."

    def _tool_disable_altitude_hold(self):
        with self._lock:
            self._ctrl["althold"] = 0
        return "Altitude hold disabled."

    def _tool_set_altitude_target(self, meters):
        meters = float(meters)
        if not (0.20 <= meters <= 2.50):
            return f"REJECTED: {meters:.2f} m outside safe range [0.20, 2.50] m."
        self._set_altset(meters)
        return f"Altitude target set to {meters:.2f} m."

    def _tool_wait(self, seconds):
        seconds = float(seconds)
        time.sleep(min(seconds, 30.0))
        alt = self._alt_m()
        return f"Waited {seconds:.1f} s. Current altitude: {alt:.3f} m" if alt else f"Waited {seconds:.1f} s."

    def _tool_check_altitude_reached(self, target_m, tolerance_m=0.10):
        target_m    = float(target_m)
        tolerance_m = float(tolerance_m)
        alts = []
        with self._lock:
            recent = self._tel[-20:] if len(self._tel) >= 20 else list(self._tel)
        for d in recent:
            lw_z = d.get("lw_z")
            if lw_z is not None:
                alts.append(lw_z / 1000.0)
        if not alts:
            return "No altitude data. Retry."
        mean_alt = sum(alts) / len(alts)
        rmse     = math.sqrt(sum((z - target_m) ** 2 for z in alts) / len(alts))
        if abs(mean_alt - target_m) <= tolerance_m:
            return (f"Altitude reached ✓  mean={mean_alt:.3f} m  "
                    f"RMSE={rmse*100:.2f} cm  target={target_m:.2f} m")
        return (f"Altitude NOT reached ✗  mean={mean_alt:.3f} m  "
                f"error={abs(mean_alt-target_m)*100:.1f} cm  target={target_m:.2f} m")

    def _tool_get_sensor_status(self):
        d = self._last_tel()
        if d is None:
            return "No telemetry available."
        lw_z = d.get("lw_z")
        alt  = lw_z / 1000.0 if lw_z is not None else None
        with self._lock:
            ah = self._ctrl["althold"]
            sp = self._ctrl["altset"]
        lines = [
            f"ekf_altitude_m: {alt:.3f}" if alt is not None else "ekf_altitude_m: N/A",
            f"althold_active: {bool(ah)}",
            f"altset_m:       {sp:.3f}",
            f"roll_deg:       {d.get('r', 0):.1f}",
            f"pitch_deg:      {d.get('p', 0):.1f}",
            f"yaw_rate_dps:   {d.get('gz', 0):.1f}",
            f"throttle_pwm:   {d.get('ch1', 0)}",
        ]
        return "\n".join(lines)

    def _tool_land(self):
        print("  [land] ramping down …")
        self._set_altset(0.30)
        time.sleep(3.0)
        self._set_altset(0.10)
        time.sleep(2.5)
        with self._lock:
            self._ctrl["althold"] = 0
        for pwm in range(1400, 999, -50):
            self._set_thr(pwm)
            time.sleep(0.25)
        time.sleep(0.5)
        self._safe_disarm()
        alt = self._alt_m() or 0.0
        return f"Landed ✓  final altitude: {alt:.3f} m. Drone disarmed."

    # ── Tool dispatcher ───────────────────────────────────────────────────────
    TOOLS = [
        {"name": "arm",                   "description": "Arm the drone motors. Always the first step.",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "disarm",                "description": "Disarm motors. Blocked while altitude > 0.10 m — use land() instead.",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "find_hover_throttle",   "description": "Ramp throttle from 1200 until drone reaches ~0.4 m and vz≈0. Must call before enable_altitude_hold().",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "check_drone_stable",    "description": "Check roll and pitch are within max_degrees of level.",
         "input_schema": {"type": "object", "properties": {"max_degrees": {"type": "number"}}, "required": []}},
        {"name": "enable_altitude_hold",  "description": "Engage firmware altitude hold at current EKF altitude. Call ONCE per flight when already stably airborne.",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "disable_altitude_hold", "description": "Disengage altitude hold, returning to manual throttle.",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "set_altitude_target",   "description": "Set altitude hold setpoint (0.20–2.50 m). Drone moves to this altitude.",
         "input_schema": {"type": "object", "properties": {"meters": {"type": "number"}}, "required": ["meters"]}},
        {"name": "wait",                  "description": "Wait for the specified number of seconds (real time).",
         "input_schema": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}},
        {"name": "check_altitude_reached","description": "Check if EKF altitude is within tolerance of target. Returns ✓/✗.",
         "input_schema": {"type": "object", "properties": {"target_m": {"type": "number"}, "tolerance_m": {"type": "number"}}, "required": ["target_m"]}},
        {"name": "get_sensor_status",     "description": "Read live telemetry: EKF altitude, roll, pitch, althold status, throttle.",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
        {"name": "land",                  "description": "Full safe landing: lowers altitude setpoint, ramps throttle to zero, confirms z<0.05 m, disarms. Use for ALL landing scenarios.",
         "input_schema": {"type": "object", "properties": {}, "required": []}},
    ]

    def execute_tool(self, name, inputs):
        if name == "arm":                   return self._tool_arm()
        if name == "disarm":                return self._tool_disarm()
        if name == "find_hover_throttle":   return self._tool_find_hover_throttle()
        if name == "check_drone_stable":    return self._tool_check_drone_stable(**inputs)
        if name == "enable_altitude_hold":  return self._tool_enable_altitude_hold()
        if name == "disable_altitude_hold": return self._tool_disable_altitude_hold()
        if name == "set_altitude_target":   return self._tool_set_altitude_target(**inputs)
        if name == "wait":                  return self._tool_wait(**inputs)
        if name == "check_altitude_reached":return self._tool_check_altitude_reached(**inputs)
        if name == "get_sensor_status":     return self._tool_get_sensor_status()
        if name == "land":                  return self._tool_land()
        return f"Unknown tool: {name}"

    # ── LLM API ───────────────────────────────────────────────────────────────
    def _llm_call(self, messages):
        body = json.dumps({
            "model":      _MODEL,
            "max_tokens": 4096,
            "temperature": 0.2,
            "system":     SYSTEM_PROMPT,
            "tools":      self.TOOLS,
            "messages":   messages,
        }).encode()
        req = urllib.request.Request(
            _ENDPOINT,
            data=body,
            headers={
                "Content-Type":  "application/json",
                "anthropic-version": "2023-06-01",
                "Authorization": f"Bearer {_API_KEY}",
            },
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        latency = time.monotonic() - t0
        in_tok  = data.get("usage", {}).get("input_tokens",  0)
        out_tok = data.get("usage", {}).get("output_tokens", 0)
        cost    = in_tok * 3.0e-6 + out_tok * 15.0e-6
        return data, latency, in_tok, out_tok, cost

    # ── ReAct loop ────────────────────────────────────────────────────────────
    def run(self, command):
        messages    = [{"role": "user", "content": command}]
        tool_trace  = []
        api_stats   = []
        total_turns = 0

        while total_turns < MAX_TURNS:
            total_turns += 1
            resp, lat, in_tok, out_tok, cost = self._llm_call(messages)
            api_stats.append({"latency_s": lat, "input_tokens": in_tok,
                               "output_tokens": out_tok, "cost_usd": cost})

            content      = resp.get("content", [])
            stop_reason  = resp.get("stop_reason", "")

            # Collect text + tool_use blocks
            text_parts   = []
            tool_calls   = []
            for blk in content:
                if blk.get("type") == "text":
                    text_parts.append(blk["text"])
                elif blk.get("type") == "tool_use":
                    tool_calls.append(blk)

            if text_parts:
                print(f"  [LLM] {' '.join(text_parts)[:200]}")

            if stop_reason == "end_turn" and not tool_calls:
                break

            # Add assistant message
            messages.append({"role": "assistant", "content": content})

            if not tool_calls:
                break

            # Execute tools
            tool_results = []
            for tc in tool_calls:
                name   = tc["name"]
                inputs = tc.get("input", {})
                print(f"  [TOOL] {name}({inputs})")
                result = self.execute_tool(name, inputs)
                print(f"  [RESULT] {result[:120]}")
                tool_trace.append({"name": name, "inputs": inputs, "result": result})
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tc["id"],
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

        return api_stats, tool_trace

# ── Statistics ────────────────────────────────────────────────────────────────
def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0.0, c - m), min(1.0, c + m)

def _bootstrap_ci(vals, n=2000, a=0.05):
    if len(vals) < 2:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n)]
    return float(np.percentile(boots, 100*a/2)), float(np.percentile(boots, 100*(1-a/2)))

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[C1-HW] Command: \"{COMMAND}\"  N={N_RUNS}  drone={DRONE_IP}")

    # Quick connectivity check
    _probe = HardwareAgent(run_idx=-1)
    _probe.connect()
    if not _probe._wait_tel(timeout=6.0):
        print("[C1-HW] ERROR: No telemetry. Check drone IP and WiFi.")
        sys.exit(1)
    _probe.disconnect()
    print("[C1-HW] Drone reachable. Starting runs …\n")

    all_runs   = []
    all_tel    = []   # list of (t_arr, z_arr) per run

    for run_idx in range(N_RUNS):
        print(f"\n[C1-HW] ── Run {run_idx+1}/{N_RUNS} ──────────────────────────────")
        agent = HardwareAgent(run_idx)
        agent.connect()

        if not agent._wait_tel(timeout=6.0):
            print(f"[C1-HW] Run {run_idx+1}: no telemetry — skipping")
            agent.disconnect()
            continue

        t_wall = time.monotonic()
        api_stats, tool_trace = [], []
        try:
            api_stats, tool_trace = agent.run(COMMAND)
            # Let drone stabilise for final measurement
            agent._tool_wait(8.0)
        except KeyboardInterrupt:
            print(f"\n[C1-HW] Ctrl+C — emergency land (run {run_idx+1})")
            try:
                agent._tool_land()
            except Exception:
                agent._safe_disarm()
        finally:
            agent.disconnect()

        wall_s = time.monotonic() - t_wall

        # ── Per-run metrics ───────────────────────────────────────────────────
        recent_alts = [z / 1000.0 for _, z in agent._tel_log[-30:]] if agent._tel_log else []
        z_ss   = float(np.mean(recent_alts))   if recent_alts else float("nan")
        z_rmse = float(np.sqrt(np.mean((np.array(recent_alts) - TARGET_ALT) ** 2))) if recent_alts else float("nan")
        err_cm = abs(z_ss - TARGET_ALT) * 100

        tool_names = [t["name"] for t in tool_trace]
        expected   = ["arm", "find_hover_throttle", "enable_altitude_hold", "set_altitude_target"]
        seq_score  = sum(1 for e in expected if e in tool_names)
        alt_pass   = err_cm <= TOLERANCE * 100
        seq_pass   = seq_score >= 3
        passed     = alt_pass and seq_pass

        total_cost = sum(s["cost_usd"] for s in api_stats)
        api_calls  = len(api_stats)

        print(f"  z_ss={z_ss:.3f}m  err={err_cm:.1f}cm  seq={seq_score}/4  "
              f"api={api_calls}  pass={passed}")

        all_runs.append({
            "run":          run_idx + 1,
            "z_ss_m":       round(z_ss, 4),
            "alt_error_cm": round(err_cm, 2),
            "z_rmse_cm":    round(z_rmse * 100, 3),
            "seq_score":    seq_score,
            "passed":       int(passed),
            "api_calls":    api_calls,
            "cost_usd":     round(total_cost, 6),
            "wall_time_s":  round(wall_s, 1),
        })

        if agent._tel_log:
            t_arr = np.array([t for t, _ in agent._tel_log])
            z_arr = np.array([z / 1000.0 for _, z in agent._tel_log])
            all_tel.append((t_arr, z_arr))

        # Brief rest between runs
        if run_idx < N_RUNS - 1:
            print("[C1-HW] Resting 15 s before next run …")
            time.sleep(15.0)

    if not all_runs:
        print("[C1-HW] No successful runs.")
        sys.exit(1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_pass = sum(r["passed"] for r in all_runs)
    lo, hi = _wilson_ci(n_pass, len(all_runs))
    err_vals  = [r["alt_error_cm"] for r in all_runs]
    rmse_vals = [r["z_rmse_cm"]    for r in all_runs]
    z_ss_vals = [r["z_ss_m"]       for r in all_runs]
    rm_ci     = _bootstrap_ci(rmse_vals)

    print(f"\n[C1-HW] ── AGGREGATE ({len(all_runs)} runs) ─────────────────────────")
    print(f"  Success rate:    {n_pass}/{len(all_runs)}  (95% CI: {lo:.2f}–{hi:.2f})")
    print(f"  Alt error (cm):  {np.mean(err_vals):.2f} ± {np.std(err_vals):.2f}")
    print(f"  RMSE (cm):       {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}  "
          f"(CI: {rm_ci[0]:.3f}–{rm_ci[1]:.3f})")
    print(f"  z_ss (m):        {np.mean(z_ss_vals):.4f} ± {np.std(z_ss_vals):.4f}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(OUT_RUNS, "w", newline="") as f:
        keys = list(all_runs[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_runs)
    print(f"[C1-HW] CSV: {OUT_RUNS}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.Oranges(np.linspace(0.45, 0.9, len(all_tel)))

    for i, (t_arr, z_arr) in enumerate(all_tel):
        r = all_runs[i]
        ax.plot(t_arr, z_arr, color=colors[i], lw=1.5, alpha=0.7,
                label=f"HW run {i+1}  z_ss={r['z_ss_m']:.3f}m  ({'✓' if r['passed'] else '✗'})")

    if len(all_tel) > 1:
        min_len = min(len(z) for _, z in all_tel)
        z_stack = np.array([z[:min_len] for _, z in all_tel])
        t_ref   = all_tel[0][0][:min_len]
        z_mean  = z_stack.mean(axis=0)
        z_std   = z_stack.std(axis=0)
        ax.plot(t_ref, z_mean, color="darkorange", lw=2.2, zorder=5,
                label=f"HW mean (n={len(all_tel)})")
        ax.fill_between(t_ref, z_mean - z_std, z_mean + z_std,
                        alpha=0.15, color="darkorange", label="HW ±1σ")

    ax.axhline(TARGET_ALT, color="red", ls="--", lw=1.2, alpha=0.7,
               label=f"Target {TARGET_ALT} m")
    ax.axhspan(TARGET_ALT - TOLERANCE, TARGET_ALT + TOLERANCE,
               alpha=0.07, color="green", label=f"±{TOLERANCE*100:.0f} cm tolerance")

    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Time since arm (s)")
    ax.set_title(
        f'EXP-C1-HW: Natural Language → Tool Chain  (HARDWARE, N={len(all_runs)})\n'
        f'Command: "{COMMAND}"  |  Drone: {DRONE_IP}\n'
        f'Success: {n_pass}/{len(all_runs)}  (95% CI: {lo:.2f}–{hi:.2f})  |  '
        f'RMSE = {np.mean(rmse_vals):.2f} ± {np.std(rmse_vals):.2f} cm'
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"[C1-HW] Plot: {OUT_PNG}")

    print(f"\n[C1-HW] DONE  {n_pass}/{len(all_runs)} passed  "
          f"RMSE={np.mean(rmse_vals):.2f}±{np.std(rmse_vals):.2f} cm")
    if os.path.exists(SIM_PNG):
        print(f"[C1-HW] For overlay comparison, see sim figure: {SIM_PNG}")
