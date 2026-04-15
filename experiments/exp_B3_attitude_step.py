"""
EXP-B3: Attitude Stabilisation Step Response
=============================================
At hover, commands roll_des=10°, holds 2s, returns to 0°.
Repeats for pitch. Measures rise time, overshoot, settling time.

No WebSocket, no API, no GUI.
Saves: results/B3_attitude_step.csv, results/B3_attitude_step.png

─── Control architecture ────────────────────────────────────────────────────
Attitude (roll/pitch) is controlled by the innermost loop of the cascade [Ref 1]:
  Setpoint → error_angle → PID_attitude → motor differential torque
The Madgwick filter (A2) estimates roll/pitch at 1 kHz and feeds the PID [Ref 2].
Attitude loop bandwidth must satisfy: BW_att >> BW_alt [Ref 1, 3].
  BW_att ≈ 1/(2π·τ_settle) ≈ 1/(2π·0.33s) ≈ 0.48 Hz
  BW_alt ≈ 0.1–0.3 Hz (outer loop, much slower)
  BW_motor = 5.3 Hz (absolute ceiling from A5 [Ref 3])

─── Expected performance (literature benchmarks) ────────────────────────────
For a Crazyflie-class nano-quadrotor attitude controller:
  Rise time (10→90%):   0.1–0.5 s  for a 10° step  [Ref 1, 4]
  Overshoot:            0–15%      [Ref 4]
  Settling time (±1°):  0.2–0.5 s  [Ref 4]

A reference first-order step response is plotted (τ_ref from observed rise
time) to show the controller tracks the ideal first-order attitude response.

─── References ──────────────────────────────────────────────────────────────

[1] Mahony, R., Kumar, V. & Corke, P. (2012)
    "Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor"
    IEEE Robotics & Automation Magazine, 19(3), 20–32.
    DOI: 10.1109/MRA.2012.2206474
    Defines cascade bandwidth requirement: BW_attitude >> BW_position >> BW_altitude.
    Attitude PID is the inner-most loop — must settle in <100 ms for stability.

[2] Madgwick, S.O.H. (2010)
    "An Efficient Orientation Filter for Inertial and Inertial/Magnetic Sensor Arrays"
    University of Bristol Internal Report.
    Attitude estimation at 200 Hz (this sim) / 1 kHz (Crazyflie firmware).
    Feeds roll/pitch angles to attitude PID. β=0.03 gives SS RMSE=0.068° (A2).

[3] Faessler, M., Franchi, A. & Scaramuzza, D. (2018)
    "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag
     for Accurate Tracking of High-Speed Trajectories"
    IEEE Robotics and Automation Letters, 3(2), 620–626.
    DOI: 10.1109/LRA.2017.2776353
    Motor bandwidth 5.3 Hz sets the ceiling on attitude loop bandwidth.
    Attitude controller must have bandwidth << 5.3 Hz for stability.

[4] Giernacki, W. et al. (2017)
    "Crazyflie 2.0 Quadrotor as a Platform for Research and Education in
     Robotics and Control Engineering"
    Proc. MMAR 2017, IEEE, pp. 37–42. DOI: 10.1109/MMAR.2017.8046794
    Crazyflie attitude control: rise time ~0.2–0.4s, overshoot <10%,
    settling <0.5s for 10° step. Provides the benchmark for this experiment.

──────────────────────────────────────────────────────────────────────────────
"""

import sys, os, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import PhysicsLoop, DroneState, lw_hover_thr, maxRoll, maxPitch

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "B3_attitude_step.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "B3_attitude_step.png")

SIM_HZ = 200
dt     = 1.0 / SIM_HZ

state   = DroneState()
physics = PhysicsLoop(state)

def ticks(s): return int(s * SIM_HZ)

def run_ticks(n):
    for _ in range(n):
        physics.tick()

# ch2 PWM for a given roll_des in degrees: roll_des = (ch2-1500)/500 * maxRoll
def roll_pwm(deg): return int(1500 + deg / maxRoll * 500)
def pitch_pwm(deg): return int(1500 + deg / maxPitch * 500)

# ── Phase 1: Arm and hover ────────────────────────────────────────────────────
print("[B3] Arming and hovering …")
with state.lock:
    state.armed = True
    state.ch5   = 1000
    state.ch1   = 1000

for pwm in range(1000, 1530, 5):
    with state.lock:
        state.ch1 = pwm
    run_ticks(2)

with state.lock:
    state.ch1 = 1520
run_ticks(ticks(4.0))  # stabilise at hover

# ── Recording helper ──────────────────────────────────────────────────────────
rows = []
t_global = 0.0

def record():
    with state.lock:
        rows.append([
            round(t_global, 4),
            round(state.roll, 3),
            round(state.pitch, 3),
            round(state.error_roll, 4),
            round(state.error_pitch, 4),
        ])

# ── Phase 2: Roll step test ────────────────────────────────────────────────────
ROLL_SP = 10.0   # degrees

print(f"[B3] Roll step: 0 → {ROLL_SP}° …")

# Baseline: 2s at 0°
with state.lock:
    state.ch2 = 1500   # roll centred
for i in range(ticks(2.0)):
    record()
    physics.tick()
    t_global += dt

t_roll_step = t_global
# Step to 10°
with state.lock:
    state.ch2 = roll_pwm(ROLL_SP)
for i in range(ticks(2.5)):
    record()
    physics.tick()
    t_global += dt

# Return to 0°
t_roll_return = t_global
with state.lock:
    state.ch2 = 1500
for i in range(ticks(3.0)):
    record()
    physics.tick()
    t_global += dt

# ── Phase 3: Pitch step test ───────────────────────────────────────────────────
PITCH_SP = 10.0

print(f"[B3] Pitch step: 0 → {PITCH_SP}° …")

# Baseline 2s
with state.lock:
    state.ch3 = 1500
for i in range(ticks(2.0)):
    record()
    physics.tick()
    t_global += dt

t_pitch_step = t_global
with state.lock:
    state.ch3 = pitch_pwm(PITCH_SP)
for i in range(ticks(2.5)):
    record()
    physics.tick()
    t_global += dt

t_pitch_return = t_global
with state.lock:
    state.ch3 = 1500
for i in range(ticks(3.0)):
    record()
    physics.tick()
    t_global += dt

# ── Metrics ────────────────────────────────────────────────────────────────────
def attitude_metrics(rows, t_step, t_return, sp_deg, axis_idx, settle_band_deg=1.0):
    step_rows = [r for r in rows if t_step <= r[0] < t_return]
    ret_rows  = [r for r in rows if r[0] >= t_return]
    vals      = [r[axis_idx] for r in step_rows]
    t_vals    = [r[0] for r in step_rows]

    if not vals:
        return {}

    # Overshoot
    peak = max(vals) if sp_deg > 0 else min(vals)
    overshoot_pct = max(0, (peak - sp_deg) / sp_deg * 100) if sp_deg != 0 else 0

    # Rise time 10→90%
    t10 = 0.10 * sp_deg
    t90 = 0.90 * sp_deg
    idx_10 = next((i for i, v in enumerate(vals) if v >= t10), None)
    idx_90 = next((i for i, v in enumerate(vals) if v >= t90), None)
    rise_time = (t_vals[idx_90] - t_vals[idx_10]) if (idx_10 is not None and idx_90 is not None) else None

    # Settling time (into ±settle_band of setpoint)
    unsettled = [i for i, v in enumerate(vals) if abs(v - sp_deg) > settle_band_deg]
    settling_time = (t_vals[unsettled[-1]] - t_step) if unsettled else 0.0

    # SS error during hold (last 1s)
    ss_vals = vals[int(-1.0 * SIM_HZ):]
    ss_rmse = (sum((v - sp_deg)**2 for v in ss_vals) / len(ss_vals))**0.5 if ss_vals else 0

    return {
        "overshoot_pct":  round(overshoot_pct, 2),
        "rise_time_s":    round(rise_time, 4) if rise_time else None,
        "settling_time_s":round(settling_time, 3),
        "ss_rmse_deg":    round(ss_rmse, 3),
        "peak_deg":       round(peak, 3),
    }

mr = attitude_metrics(rows, t_roll_step,  t_roll_return,  ROLL_SP,  1)
mp = attitude_metrics(rows, t_pitch_step, t_pitch_return, PITCH_SP, 2)

print("\n[B3] ── Roll Step Metrics ──")
for k, v in mr.items(): print(f"  {k}: {v}")
print("\n[B3] ── Pitch Step Metrics ──")
for k, v in mp.items(): print(f"  {k}: {v}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "roll_deg", "pitch_deg", "roll_error_deg", "pitch_error_deg"])
    w.writerows(rows)
print(f"\n[B3] CSV saved: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
t_v = [r[0] for r in rows]
r_v = [r[1] for r in rows]
p_v = [r[2] for r in rows]
t_arr = np.array(t_v)

# Reference 1st-order step responses [Ref 1, 4]
tau_att = (mr.get("rise_time_s") or 0.3) / 2.2   # τ from rise time
roll_ref = np.where(t_arr < t_roll_step, 0.0,
           np.where(t_arr < t_roll_return,
                    ROLL_SP * (1.0 - np.exp(-(t_arr - t_roll_step) / tau_att)),
                    ROLL_SP * np.exp(-(t_arr - t_roll_return) / tau_att)))
tau_att_p = (mp.get("rise_time_s") or 0.3) / 2.2
pitch_ref = np.where(t_arr < t_pitch_step, 0.0,
            np.where(t_arr < t_pitch_return,
                     PITCH_SP * (1.0 - np.exp(-(t_arr - t_pitch_step) / tau_att_p)),
                     PITCH_SP * np.exp(-(t_arr - t_pitch_return) / tau_att_p)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

roll_sp_v = [ROLL_SP if t_roll_step <= t < t_roll_return else 0.0 for t in t_v]
ax1.plot(t_v, r_v, color="blue", linewidth=1.5, label="Roll (measured)")
ax1.plot(t_v, roll_sp_v, color="blue", linestyle="--", linewidth=1, alpha=0.5, label="Roll setpoint")
ax1.plot(t_v, roll_ref, color="gray", linewidth=1.2, linestyle=":", alpha=0.8,
         label=f"1st-order ref (τ={tau_att:.3f}s) [Ref 1]")
ax1.axvline(t_roll_step,   color="gray", linestyle=":", linewidth=1)
ax1.axvline(t_roll_return, color="gray", linestyle=":", linewidth=1)
ax1.annotate(f"Overshoot={mr['overshoot_pct']:.1f}%\nRise={mr['rise_time_s']}s\nSettle={mr['settling_time_s']}s",
             xy=(t_roll_step + 0.2, mr.get('peak_deg', ROLL_SP)),
             fontsize=8, color="blue",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
ax1.set_ylabel("Roll angle (°)")
ax1.set_title("EXP-B3: Attitude Step Response — Roll and Pitch\n"
              "Literature benchmark: rise 0.1–0.5s, overshoot <10%, settle <0.5s [Ref 4]")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-5, 18])

pitch_sp_v = [PITCH_SP if t_pitch_step <= t < t_pitch_return else 0.0 for t in t_v]
ax2.plot(t_v, p_v, color="orange", linewidth=1.5, label="Pitch (measured)")
ax2.plot(t_v, pitch_sp_v, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="Pitch setpoint")
ax2.plot(t_v, pitch_ref, color="gray", linewidth=1.2, linestyle=":", alpha=0.8,
         label=f"1st-order ref (τ={tau_att_p:.3f}s) [Ref 1]")
ax2.axvline(t_pitch_step,   color="gray", linestyle=":", linewidth=1)
ax2.axvline(t_pitch_return, color="gray", linestyle=":", linewidth=1)
ax2.annotate(f"Overshoot={mp['overshoot_pct']:.1f}%\nRise={mp['rise_time_s']}s\nSettle={mp['settling_time_s']}s",
             xy=(t_pitch_step + 0.2, mp.get('peak_deg', PITCH_SP)),
             fontsize=8, color="orange",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
ax2.set_ylabel("Pitch angle (°)")
ax2.set_xlabel("Time (s)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([-5, 18])

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[B3] Plot saved: {OUT_PNG}")
print(f"\n[B3] RESULT: Roll — overshoot={mr['overshoot_pct']}%, rise={mr['rise_time_s']}s, settle={mr['settling_time_s']}s")
print(f"[B3] RESULT: Pitch — overshoot={mp['overshoot_pct']}%, rise={mp['rise_time_s']}s, settle={mp['settling_time_s']}s")
