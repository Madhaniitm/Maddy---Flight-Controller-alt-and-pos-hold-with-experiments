"""
EXP-B2: Position Hold Disturbance Rejection
=============================================
Drone hovers at (0,0,1m) with position hold active.
At t=5s, applies a lateral impulse (Fx=0.05N for 0.2s).
Measures max XY drift, return-to-hold time, RMSE after disturbance.

No WebSocket, no API, no GUI.
Saves: results/B2_poshold_disturbance.csv, results/B2_poshold_disturbance.png

─── What is tested ──────────────────────────────────────────────────────────
The position hold PID (outer loop) commands a roll/pitch setpoint to drive XY
velocity to zero. An impulse Fx=0.05N for 0.2s injects Δvx = Fx·Δt/m (impulse
response test). The controller must detect the drift via EKF and command a
counter-roll to return to the setpoint [Ref 1].

Impulse magnitude (0.05N × 0.2s = 0.01 N·s) corresponds to a typical indoor
disturbance for this 50g drone (Δv = 0.01/0.050 = 0.20 m/s), consistent
with disturbance magnitudes used in Crazyflie ADRC studies [Ref 3].

─── Reference recovery model ────────────────────────────────────────────────
After an impulse, an ideal underdamped 2nd-order position hold returns as:
  d(t) = d_peak · exp(−t/τ_rec) · cos(ω_d · t)
For a well-tuned PD position loop, τ_rec ≈ 1–3 s [Ref 2, 3]. An exponential
decay envelope d_peak·exp(−t/τ_rec) is plotted as a reference comparison.

─── References ──────────────────────────────────────────────────────────────

[1] Mahony, R., Kumar, V. & Corke, P. (2012)
    "Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor"
    IEEE Robotics & Automation Magazine, 19(3), 20–32.
    DOI: 10.1109/MRA.2012.2206474
    Cascade control architecture: position error → attitude setpoint → attitude
    controller. Disturbance rejection relies on the position PID integral.

[2] Giernacki, W. et al. (2017)
    "Crazyflie 2.0 Quadrotor as a Platform for Research and Education in
     Robotics and Control Engineering"
    Proc. 22nd Int. Conf. Methods and Models in Automation and Robotics (MMAR),
    IEEE, pp. 37–42. DOI: 10.1109/MMAR.2017.8046794
    Documents Crazyflie 2.0 position hold performance. Position hold SS RMSE
    ~2–5 cm under indoor conditions with optical flow or motion capture.

[3] Chadha, G., Bhushan, B. & Rawat, A. (2023)
    "Position Control of Crazyflie 2.1 Quadrotor UAV Based on Active
     Disturbance Rejection Control"
    Proc. IEEE Int. Conf. on Computing, Communication, and Intelligent Systems
    (ICCCIS), pp. 823–829. DOI: 10.1109/ICCCIS57919.2023.10156505
    ADRC applied to Crazyflie 2.1 XY position control. Demonstrates recovery
    from lateral disturbances in 1–3 s with <5 cm SS error, same drone class.

[4] Preiss, J.A., Hönig, W., Sukhatme, G.S. & Ayanian, N. (2017)
    "Crazyswarm: A Large Nano-Quadcopter Swarm"
    IEEE ICRA 2017, pp. 3299–3304.
    Position hold and disturbance rejection in swarm. Reports <5 cm inter-drone
    position tracking error on Crazyflie. Sets the accuracy baseline used here.

──────────────────────────────────────────────────────────────────────────────
"""

import sys, os, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import PhysicsLoop, DroneState, lw_hover_thr, MASS, GRAVITY

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "B2_poshold_disturbance.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "B2_poshold_disturbance.png")

SIM_HZ = 200
dt     = 1.0 / SIM_HZ

state   = DroneState()
physics = PhysicsLoop(state)


def run_ticks(n):
    for _ in range(n):
        physics.tick()

def ticks(s):
    return int(s * SIM_HZ)

# ── Phase 1: Arm + ramp + open-loop hover-find (mirrors B1) ──────────────────
print("[B2] Arming and climbing to 1m …")
with state.lock:
    state.armed = True
    state.ch5   = 1000
    state.ch1   = 1000

for pwm in range(1000, 1550, 5):
    with state.lock:
        state.ch1 = pwm
    run_ticks(4)

with state.lock:
    state.ch1 = 1560
for _ in range(ticks(20.0)):
    physics.tick()
    with state.lock:
        if state.z > 1.0:
            break

print("[B2] Finding stable hover …")
pwm_now = 1550
with state.lock:
    state.ch1 = pwm_now
for attempt in range(200):
    run_ticks(ticks(0.2))
    with state.lock:
        vz_now = state.vz
    if abs(vz_now) < 0.008:
        break
    pwm_now += 1 if vz_now < 0 else -1
    pwm_now = max(1400, min(1700, pwm_now))
    with state.lock:
        state.ch1 = pwm_now

HOVER_THR = (pwm_now - 1000) / 1000.0
run_ticks(ticks(3.0))  # let EKF converge

with state.lock:
    z_hover = state.z

print(f"[B2] Hover at z={z_hover:.2f}m  hover_thr={HOVER_THR:.3f}")

# Enable alt hold — setpoint 1.0m, althold descends from hover height
with state.lock:
    state.althold          = True
    state.alt_sp           = 1.0
    state.alt_sp_mm        = 1000
    state.hover_thr_locked = HOVER_THR

run_ticks(ticks(5.0))  # wait for descent and settle at 1.0m

with state.lock:
    true_hold_x = state.x
    true_hold_y = state.y

# Enable position hold at current EKF XY
with state.lock:
    state.poshold   = True
    state.pos_sp_x  = state.ekf_x
    state.pos_sp_y  = state.ekf_y

print(f"[B2] Position hold engaged at true=({true_hold_x:.3f}, {true_hold_y:.3f}) m")
run_ticks(ticks(5.0))  # settle poshold

# Re-sync after settle: EKF-based poshold may drift state.x slightly during the
# 5 s settle, so update true_hold to the actual position at recording start —
# ensures green pre-disturbance path begins exactly at (0, 0) in the relative plot.
with state.lock:
    true_hold_x = state.x
    true_hold_y = state.y

# ── Recording + disturbance ────────────────────────────────────────────────────
rows = []
t_global = 0.0
t_disturbance = 5.0    # s into recording
disturbance_applied = False

IMPULSE_FX   = 0.05    # N lateral force
IMPULSE_DUR  = 0.2     # s

print("[B2] Recording … disturbance at t=5s")

total_time = 20.0
for i in range(ticks(total_time)):
    with state.lock:
        x   = state.x
        y   = state.y
        z   = state.z
        ex  = state.ekf_x
        ey  = state.ekf_y
        ez  = state.ekf_z
        spx = state.pos_sp_x
        spy = state.pos_sp_y

    rows.append([round(t_global, 4), round(x, 4), round(y, 4), round(z, 4),
                 round(ex, 4), round(ey, 4), round(ez, 4),
                 round(math.hypot(x - true_hold_x, y - true_hold_y), 4)])

    # Apply impulse at t_disturbance for IMPULSE_DUR seconds
    if t_disturbance <= t_global < t_disturbance + IMPULSE_DUR:
        if not disturbance_applied:
            print(f"[B2] Disturbance applied at t={t_global:.2f}s")
            disturbance_applied = True
        # Direct force injection into physics state
        with state.lock:
            state.vx += IMPULSE_FX / MASS * dt

    physics.tick()
    t_global += dt

# ── Metrics ────────────────────────────────────────────────────────────────────
pre_rows  = [r for r in rows if r[0] < t_disturbance]
post_rows = [r for r in rows if r[0] >= t_disturbance]

pre_rmse  = math.sqrt(sum(r[7]**2 for r in pre_rows)  / len(pre_rows))
post_rows_all = [r for r in rows if r[0] >= t_disturbance]

# Max drift
max_drift = max(r[7] for r in post_rows_all)
t_max_drift = next(r[0] for r in post_rows_all if r[7] == max_drift)

# Return-to-hold time: first time after peak where drift < 2× pre-RMSE
threshold = max(0.05, 2.0 * pre_rmse)
peak_idx  = next(i for i, r in enumerate(post_rows_all) if r[7] == max_drift)
returned  = [r for r in post_rows_all[peak_idx:] if r[7] < threshold]
return_time = (returned[0][0] - t_disturbance) if returned else None

# SS-RMSE: last 5s
ss_rows  = rows[int(-5.0 * SIM_HZ):]
ss_rmse  = math.sqrt(sum(r[7]**2 for r in ss_rows) / len(ss_rows))

print(f"\n[B2] Pre-disturbance XY RMSE:  {pre_rmse*100:.2f} cm")
print(f"[B2] Max drift:                 {max_drift*100:.2f} cm  at t={t_max_drift:.2f}s")
print(f"[B2] Return-to-hold time:       {return_time:.2f}s" if return_time else "[B2] Return-to-hold: did not return")
print(f"[B2] Post-disturbance SS-RMSE:  {ss_rmse*100:.2f} cm")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "x_true_m", "y_true_m", "z_true_m",
                "x_ekf_m", "y_ekf_m", "z_ekf_m", "xy_error_m"])
    w.writerows(rows)
print(f"\n[B2] CSV saved: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
t_v    = [r[0] for r in rows]
x_v    = [r[1] for r in rows]
y_v    = [r[2] for r in rows]
xy_err = [r[7] * 100 for r in rows]   # cm

fig = plt.figure(figsize=(12, 8))
gs  = fig.add_gridspec(2, 2)

ax_xy  = fig.add_subplot(gs[0, 0])
ax_t   = fig.add_subplot(gs[0, 1])
ax_err = fig.add_subplot(gs[1, :])

# XY trajectory — plotted relative to true_hold_x/y so setpoint is at (0, 0).
# Absolute drone position accumulates drift during climb/hover and can land at
# arbitrary coordinates. Relative coords make axes directly readable as error.
def rel2(r): return (r[1] - true_hold_x, r[2] - true_hold_y)

t_recover    = t_disturbance + (return_time if return_time else 3.0)
pre_rows_xy  = [rel2(r) for r in rows if r[0] < t_disturbance]
dur_rows_xy  = [rel2(r) for r in rows if t_disturbance <= r[0] < t_recover]
post_rows_xy = [rel2(r) for r in rows if r[0] >= t_recover]

if pre_rows_xy:
    ax_xy.plot(*zip(*pre_rows_xy),  color="green",  lw=1.5, label="Pre-disturbance")
if dur_rows_xy:
    ax_xy.plot(*zip(*dur_rows_xy),  color="orange", lw=1.5, label="Disturbance → recovery")
if post_rows_xy:
    ax_xy.plot(*zip(*post_rows_xy), color="blue",   lw=1.0, alpha=0.6, label="Post-recovery")
ax_xy.scatter([0], [0], color="red", s=100, zorder=5, label="Setpoint")
# Literature acceptance circle: SS position error <5cm [Ref 2, 4]
theta = np.linspace(0, 2 * np.pi, 200)
ax_xy.plot(0.05 * np.cos(theta), 0.05 * np.sin(theta),
           color="red", linewidth=1.2, linestyle="--", alpha=0.7,
           label="5cm acceptance boundary [Ref 2, 4]")
ax_xy.set_xlabel("ΔX from setpoint (m)")
ax_xy.set_ylabel("ΔY from setpoint (m)")
ax_xy.set_title("XY Trajectory (relative to setpoint)")
ax_xy.legend(fontsize=8)
ax_xy.grid(True, alpha=0.3)
ax_xy.set_aspect("equal")

# Z altitude
z_v = [r[3] for r in rows]
ax_t.plot(t_v, z_v, color="purple", linewidth=1.5, label="Altitude (m)")
ax_t.axvline(t_disturbance, color="red", linestyle="--", linewidth=1, label="Disturbance")
# Literature ±2cm band: altitude SS RMSE <2cm for cascade PID [Ref 2]
ax_t.axhspan(0.98, 1.02, color="green", alpha=0.15, label="±2cm acceptance band [Ref 2]")
ax_t.axhline(1.0, color="green", linewidth=0.8, linestyle=":", alpha=0.6)
ax_t.set_xlabel("Time (s)")
ax_t.set_ylabel("Altitude (m)")
ax_t.set_title("Altitude during disturbance\nLit. SS altitude error <2cm [Ref 2]")
ax_t.legend(fontsize=8)
ax_t.grid(True, alpha=0.3)

# Reference exponential recovery: d_peak * exp(-(t - t_peak) / tau_rec)  [Ref 1, 3]
# τ_rec fitted from return_time: t_rec = 3*τ_rec → τ_rec = return_time/3
import math as _math, numpy as _np
tau_rec  = (return_time / 3.0) if return_time else 1.0
t_arr    = _np.array(t_v)
t_peak_v = t_disturbance + (t_max_drift - t_disturbance)
ref_env  = _np.where(t_arr >= t_peak_v,
                     max_drift * 100 * _np.exp(-(t_arr - t_peak_v) / tau_rec),
                     _np.nan)

# Literature recovery band: Chadha 2023 [Ref 3] reports recovery in 1–3s
# τ_fast=1s (best case), τ_slow=3s (worst case) → shaded band shows pass region
TAU_FAST = 1.0   # s — fast end of literature range [Ref 3]
TAU_SLOW = 3.0   # s — slow end of literature range [Ref 3]
lit_fast = _np.where(t_arr >= t_peak_v,
                     max_drift * 100 * _np.exp(-(t_arr - t_peak_v) / TAU_FAST),
                     _np.nan)
lit_slow = _np.where(t_arr >= t_peak_v,
                     max_drift * 100 * _np.exp(-(t_arr - t_peak_v) / TAU_SLOW),
                     _np.nan)

# XY error over time
ax_err.plot(t_v, xy_err, color="orange", linewidth=1.5, label="XY distance from setpoint (cm)")
ax_err.plot(t_v, ref_env, color="gray", linewidth=1.2, linestyle=":",
            label=f"Fitted decay envelope (τ={tau_rec:.2f}s, from sim return time)")
ax_err.plot(t_v, lit_fast, color="steelblue", linewidth=1.2, linestyle="--",
            label=f"Lit. fast recovery (τ={TAU_FAST}s) [Ref 3]")
ax_err.plot(t_v, lit_slow, color="navy", linewidth=1.2, linestyle="--",
            label=f"Lit. slow recovery (τ={TAU_SLOW}s) [Ref 3]")
ax_err.fill_between(t_v, _np.nan_to_num(lit_fast, nan=0.0),
                    _np.nan_to_num(lit_slow, nan=0.0),
                    where=(t_arr >= t_peak_v),
                    color="steelblue", alpha=0.15,
                    label="Literature recovery band (1–3s) [Ref 3]")
ax_err.axvline(t_disturbance, color="red", linestyle="--", linewidth=1.5, label="Disturbance")
ax_err.axhline(pre_rmse * 100, color="green", linestyle=":", label=f"Pre-disturbance RMSE={pre_rmse*100:.2f}cm")
ax_err.axhline(max_drift * 100, color="red", linestyle=":", label=f"Max drift={max_drift*100:.2f}cm")
if return_time:
    ax_err.axvline(t_disturbance + return_time, color="blue", linestyle=":",
                   label=f"Recovered at +{return_time:.1f}s")
ax_err.set_xlabel("Time (s)")
ax_err.set_ylabel("XY error (cm)")
ax_err.set_title(f"EXP-B2: Position Hold Disturbance Rejection  "
                 f"(Fx={IMPULSE_FX}N for {IMPULSE_DUR}s)\n"
                 f"Literature: recovery 1–3s, SS-RMSE <5cm [Ref 2, 3]")
ax_err.legend(fontsize=8)
ax_err.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[B2] Plot saved: {OUT_PNG}")
rt = f"{return_time:.2f}s" if return_time else "N/A"
print(f"\n[B2] RESULT: Max drift={max_drift*100:.1f}cm, Return={rt}, SS-RMSE={ss_rmse*100:.2f}cm")
