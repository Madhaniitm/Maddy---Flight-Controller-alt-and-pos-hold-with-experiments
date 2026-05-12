"""
EXP-B1-HW: Altitude Hold Step Response — HARDWARE
===================================================
Connects to the real Maddy drone via WebSocket (ws://<IP>:81).
Runs the same altitude step sequence as EXP-B1 (sim):
  baseline 1.0 m → step to 1.3 m → step to 1.6 m → land
Logs lw_z (EKF altitude in mm from ESP32) at telemetry rate (~10 Hz).
Produces sim-vs-hardware overlay figure: results/B1_hw_vs_sim.png

Usage:
  python exp_B1_althold_step_HW.py --ip 10.198.219.30
  python exp_B1_althold_step_HW.py --ip 192.168.4.1

Safety:
  Ctrl+C at any time → controlled land + disarm
  If no telemetry within 5 s → abort before arming
"""

import sys, os, csv, math, time, json, threading, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import websocket
except ImportError:
    print("ERROR: websocket-client not installed. Run: pip install websocket-client")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="10.198.219.30", help="Drone IP address")
args = parser.parse_args()

DRONE_IP = args.ip
WS_URL   = f"ws://{DRONE_IP}:81"
SEND_HZ  = 20
SEND_DT  = 1.0 / SEND_HZ

OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV  = os.path.join(OUT_DIR, "B1_hw_althold_step.csv")
OUT_PNG  = os.path.join(OUT_DIR, "B1_hw_vs_sim.png")
SIM_CSV  = os.path.join(OUT_DIR, "B1_althold_step.csv")

SP0, SP1, SP2 = 1.0, 1.3, 1.6   # altitude setpoints (m)

# ── Shared drone state (written by main, read by sender thread) ───────────────
_lock  = threading.Lock()
_ctrl  = {"ch1": 1000, "altset": 1.0, "althold": 0, "armed": False}
_tel   = []        # telemetry ring buffer (dicts from ESP32)
_ws    = None
_stop  = threading.Event()

# ── WebSocket receiver ────────────────────────────────────────────────────────
def _on_message(ws, msg):
    try:
        d = json.loads(msg)
        if d.get("tel") == 1:
            d["_mono"] = time.monotonic()
            with _lock:
                _tel.append(d)
                if len(_tel) > 1000:
                    _tel.pop(0)
    except Exception:
        pass

def _connect():
    global _ws
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=_on_message,
        on_error=lambda ws, e: print(f"[WS] {e}"),
        on_close=lambda ws, *a: print("[WS] closed"),
        on_open=lambda ws: print(f"[WS] connected → {WS_URL}"),
    )
    _ws = ws
    threading.Thread(target=ws.run_forever, daemon=True).start()
    time.sleep(1.2)

# ── 20 Hz sender thread ───────────────────────────────────────────────────────
def _sender():
    while not _stop.is_set():
        t0 = time.monotonic()
        with _lock:
            if _ws is not None:
                pkt = {
                    "ch1":     _ctrl["ch1"],
                    "ch2":     1500,
                    "ch3":     1500,
                    "ch4":     1500,
                    "ch5":     1000 if _ctrl["armed"] else 2000,
                    "ch6":     1000,
                    "altset":  _ctrl["altset"],
                    "althold": _ctrl["althold"],
                    "poshold": 0,
                }
                try:
                    _ws.send(json.dumps(pkt))
                except Exception:
                    pass
        rem = SEND_DT - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)

def _start_sender():
    threading.Thread(target=_sender, daemon=True).start()

# ── Telemetry helpers ─────────────────────────────────────────────────────────
def _last():
    with _lock:
        return _tel[-1] if _tel else None

def _alt_m():
    d = _last()
    if d is None:
        return None
    lw_z = d.get("lw_z")
    return lw_z / 1000.0 if lw_z is not None else None

def _wait_tel(timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if _last() is not None:
            return True
        time.sleep(0.05)
    return False

# ── Control commands ──────────────────────────────────────────────────────────
def _arm():
    with _lock:
        _ctrl["armed"] = True
        _ctrl["ch1"]   = 1000
    time.sleep(0.5)
    print("[B1-HW] Armed")

def _disarm():
    with _lock:
        _ctrl["armed"]   = False
        _ctrl["althold"] = 0
        _ctrl["ch1"]     = 1000
    time.sleep(0.3)
    print("[B1-HW] Disarmed")

def _set_thr(pwm):
    with _lock:
        _ctrl["ch1"] = int(max(1000, min(2000, pwm)))

def _set_altset(m):
    with _lock:
        _ctrl["altset"] = float(m)

def _althold_on(sp_m):
    with _lock:
        _ctrl["altset"]  = float(sp_m)
        _ctrl["althold"] = 1
    print(f"[B1-HW] althold ON  sp={sp_m:.2f} m")

def _althold_off():
    with _lock:
        _ctrl["althold"] = 0

def _land():
    print("[B1-HW] Landing …")
    _set_altset(0.3)
    time.sleep(3.0)
    _set_altset(0.10)
    time.sleep(2.5)
    _althold_off()
    for pwm in range(1400, 999, -50):
        _set_thr(pwm)
        time.sleep(0.25)
    time.sleep(0.5)
    _disarm()

# ── Recording helper ──────────────────────────────────────────────────────────
def _record(rows, t_ref, duration_s):
    seen = set()
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        d = _last()
        if d is not None:
            key = id(d)
            if key not in seen:
                seen.add(key)
                lw_z = d.get("lw_z")
                if lw_z is not None:
                    with _lock:
                        sp = _ctrl["altset"]
                    rows.append({
                        "t_s":      round(d["_mono"] - t_ref, 3),
                        "z_hw_m":   round(lw_z / 1000.0, 4),
                        "altset_m": round(sp, 3),
                    })
        time.sleep(0.02)

# ── Metrics (identical to sim version) ────────────────────────────────────────
def _metrics(rows, t_step, t_end, sp_prev, sp_new):
    sr = [r for r in rows if t_step <= r["t_s"] < t_end]
    if not sr:
        return {}
    step = sp_new - sp_prev
    zv   = [r["z_hw_m"] for r in sr]
    tv   = [r["t_s"]    for r in sr]
    peak = max(zv) if step > 0 else min(zv)
    os_p = max(0, (peak - sp_new) / abs(step) * 100)
    t10  = sp_prev + 0.10 * step
    t90  = sp_prev + 0.90 * step
    i10  = next((i for i, z in enumerate(zv) if z >= t10), None)
    i90  = next((i for i, z in enumerate(zv) if z >= t90), None)
    rise = round(tv[i90] - tv[i10], 3) if (i10 is not None and i90 is not None) else None
    band = 0.05 * abs(step)
    uset = [i for i, z in enumerate(zv) if abs(z - sp_new) > band]
    settl = round(tv[uset[-1]] - t_step, 3) if uset else 0.0
    ss   = zv[-40:]  # last 4 s at ~10 Hz
    ss_rmse = round(math.sqrt(sum((z - sp_new) ** 2 for z in ss) / max(len(ss), 1)) * 100, 3)
    return {"overshoot_pct": round(os_p, 2), "rise_time_s": rise,
            "settling_time_s": settl, "ss_rmse_cm": ss_rmse}

# ── Plot (sim overlay if CSV available) ───────────────────────────────────────
def _plot(rows, t_s1, t_s2, m1, m2):
    t_hw  = [r["t_s"]      for r in rows]
    z_hw  = [r["z_hw_m"]   for r in rows]
    sp_hw = [r["altset_m"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(t_hw, z_hw, color="orangered", lw=2.0, label="Hardware EKF (lw_z)", zorder=4)
    ax1.step(t_hw, sp_hw, color="red", lw=1.5, ls="--", where="post",
             label="Setpoint", alpha=0.7)

    if os.path.exists(SIM_CSV):
        st, sz, ssp = [], [], []
        with open(SIM_CSV) as f:
            for row in csv.DictReader(f):
                st.append(float(row["t_s"]))
                sz.append(float(row["z_ekf_m"]))
        ax1.plot(st, sz, color="steelblue", lw=1.5, ls=":", alpha=0.75,
                 label="Simulation EKF (reference)", zorder=3)

    for v in (t_s1, t_s2):
        if v is not None:
            ax1.axvline(v, color="gray", ls=":", lw=1)
    for sp_val, m, t_ann in [(SP1, m1, t_s1), (SP2, m2, t_s2)]:
        ax1.axhspan(sp_val - 0.015, sp_val + 0.015, alpha=0.08, color="green")
        if m and t_ann is not None:
            ax1.annotate(
                f"OS={m.get('overshoot_pct')}%\nRise={m.get('rise_time_s')}s\nSS={m.get('ss_rmse_cm')}cm",
                xy=(t_ann + 0.3, sp_val - 0.07),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9),
            )

    ax1.set_ylabel("Altitude (m)")
    ax1.set_title(
        f"EXP-B1-HW: Altitude Hold Step Response — HARDWARE vs SIM\n"
        f"Drone: {DRONE_IP}  |  Sequence: {SP0}→{SP1}→{SP2} m\n"
        f"Literature benchmark: rise 1–2 s, overshoot ≤10%, SS RMSE <2 cm"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    err_cm = [abs(z - s) * 100 for z, s in zip(z_hw, sp_hw)]
    ax2.plot(t_hw, err_cm, color="purple", lw=1.2, label="|HW altitude error| (cm)")
    ax2.axhline(5.0, color="red", ls="--", lw=1, label="5 cm threshold")
    ax2.set_ylabel("Altitude error (cm)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"[B1-HW] Plot saved: {OUT_PNG}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[B1-HW] Connecting to {WS_URL} …")
    _connect()

    if not _wait_tel(timeout=6.0):
        print("[B1-HW] ERROR: No telemetry after 6 s. Check drone IP and WiFi.")
        sys.exit(1)

    print(f"[B1-HW] Telemetry OK. Starting B1 step-response experiment.")
    _start_sender()

    rows   = []
    t_s1   = None
    t_s2   = None
    t_ref  = None

    try:
        # ── 1. Arm ────────────────────────────────────────────────────────────
        _arm()
        time.sleep(0.5)

        # ── 2. Ramp throttle until ~0.9 m ────────────────────────────────────
        print("[B1-HW] Ramping to ~0.9 m …")
        pwm = 1200
        while pwm < 1700:
            _set_thr(pwm)
            time.sleep(0.06)
            alt = _alt_m()
            if alt is not None and alt > 0.85:
                print(f"[B1-HW] Altitude {alt:.2f} m — enabling althold")
                break
            pwm += 5
        else:
            print("[B1-HW] WARNING: could not reach 0.85 m in ramp — proceeding anyway")

        # ── 3. Enable althold → settle at SP0 ────────────────────────────────
        _althold_on(SP0)
        print(f"[B1-HW] Settling at {SP0} m (6 s) …")
        time.sleep(6.0)

        # ── 4. Baseline (5 s) ─────────────────────────────────────────────────
        print(f"[B1-HW] Recording baseline at {SP0} m …")
        t_ref = time.monotonic()
        _record(rows, t_ref, 5.0)

        # ── 5. Step 1 → SP1 (9 s) ─────────────────────────────────────────────
        t_s1 = rows[-1]["t_s"] if rows else 5.0
        _set_altset(SP1)
        print(f"[B1-HW] Step 1 → {SP1} m")
        _record(rows, t_ref, 9.0)

        # ── 6. Step 2 → SP2 (11 s) ────────────────────────────────────────────
        t_s2 = rows[-1]["t_s"] if rows else 14.0
        _set_altset(SP2)
        print(f"[B1-HW] Step 2 → {SP2} m")
        _record(rows, t_ref, 11.0)

        print("[B1-HW] Sequence done. Landing …")
        _land()

    except KeyboardInterrupt:
        print("\n[B1-HW] Ctrl+C — emergency land")
        try:
            _land()
        except Exception:
            _disarm()
    finally:
        _stop.set()

    if not rows:
        print("[B1-HW] No data collected.")
        sys.exit(1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    t_end = rows[-1]["t_s"]
    m1 = _metrics(rows, t_s1 or 5.0,  t_s2 or 14.0, SP0, SP1)
    m2 = _metrics(rows, t_s2 or 14.0, t_end,          SP1, SP2)

    print(f"\n[B1-HW] ── Step 1 ({SP0}→{SP1} m) ──")
    for k, v in m1.items():
        print(f"  {k}: {v}")
    print(f"[B1-HW] ── Step 2 ({SP1}→{SP2} m) ──")
    for k, v in m2.items():
        print(f"  {k}: {v}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "z_hw_m", "altset_m"])
        w.writeheader()
        w.writerows(rows)
    print(f"[B1-HW] CSV saved: {OUT_CSV}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(rows, t_s1, t_s2, m1, m2)

    print(f"\n[B1-HW] DONE")
    print(f"  Step1({SP0}→{SP1}m): OS={m1.get('overshoot_pct')}%  rise={m1.get('rise_time_s')}s  SS={m1.get('ss_rmse_cm')}cm")
    print(f"  Step2({SP1}→{SP2}m): OS={m2.get('overshoot_pct')}%  rise={m2.get('rise_time_s')}s  SS={m2.get('ss_rmse_cm')}cm")
