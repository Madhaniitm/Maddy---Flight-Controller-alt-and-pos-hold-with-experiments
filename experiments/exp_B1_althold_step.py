"""
EXP-B1: Altitude Hold Step Response
======================================
Finds stable hover altitude, enables altitude hold there,
then steps setpoint UP: hover → hover+0.3m → hover+0.6m.
Measures overshoot %, rise time, settling time, steady-state RMSE.

No WebSocket, no API, no GUI.
Saves: results/B1_althold_step.csv, results/B1_althold_step.png

─── Control architecture ────────────────────────────────────────────────────
Cascade PID structure [Ref 1]:
  Outer loop  (altitude position): error_z → velocity setpoint (vel_sp) → PID_alt_pos
  Inner loop  (altitude velocity): vel_sp − vz → throttle correction → PID_alt_vel

This two-loop structure is the standard architecture for quadrotor altitude hold
[Ref 1, 2]. The inner (velocity) loop has higher bandwidth than the outer
(position) loop, ensuring the velocity command is tracked before the position
error accumulates. Motor bandwidth (5.3 Hz from A5) sets the absolute ceiling
for inner-loop bandwidth [Ref 3].

─── Expected performance (literature benchmarks) ────────────────────────────
For a Crazyflie-class nano-quadrotor with cascade PID altitude hold:
  Rise time (10→90%):   1–2 s  for a 0.3 m step  [Ref 2, 4]
  Overshoot:            5–25%  [Ref 4]; well-tuned ≤10% [Ref 2]
  Settling time (±5%):  1–5 s  [Ref 4]
  SS RMSE:             <2 cm   [Ref 2]

A second-order underdamped reference step response is plotted alongside the
sim result [Ref 1]. The cascade PID is inherently 2nd-order (outer position
loop + inner velocity loop = two poles). Damping ratio ζ is fitted from
observed overshoot (OS% → ζ via logarithmic decrement), and ωn from rise time.
  z_ref(t) = z_sp·[1 − exp(−ζωn·t)/√(1−ζ²)·sin(ωd·t + arccos(ζ))]

─── References ──────────────────────────────────────────────────────────────

[1] Mahony, R., Kumar, V. & Corke, P. (2012)
    "Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor"
    IEEE Robotics & Automation Magazine, 19(3), 20–32.
    DOI: 10.1109/MRA.2012.2206474
    Establishes the cascade (inner velocity / outer position) altitude hold
    architecture used here. Defines bandwidth hierarchy: BW_alt << BW_att << BW_motor.

[2] Giernacki, W. et al. (2017)
    "Crazyflie 2.0 Quadrotor as a Platform for Research and Education in
     Robotics and Control Engineering"
    Proc. 22nd Int. Conf. Methods and Models in Automation and Robotics (MMAR),
    IEEE, pp. 37–42. DOI: 10.1109/MMAR.2017.8046794.
    Documents Crazyflie 2.0 control performance. Cascade PID achieves
    SS altitude error <2 cm and settling within 2 s for small steps.

[3] Faessler, M., Franchi, A. & Scaramuzza, D. (2018)
    "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag
     for Accurate Tracking of High-Speed Trajectories"
    IEEE Robotics and Automation Letters, 3(2), 620–626.
    DOI: 10.1109/LRA.2017.2776353
    Motor lag τ ≈ 30 ms → bandwidth 5.3 Hz. Altitude hold BW << 5.3 Hz
    (typically 0.5–2 Hz). Validates the cascade bandwidth hierarchy.

[4] Teppa-Garran, P.D. & Garcia, G. (2013)
    "PID Autotuning for Altitude Control of Unmanned Helicopter"
    J. Intelligent & Robotic Systems, 70(1–4), 359–373.
    DOI: 10.1007/s10846-012-9762-3
    PID cascade altitude hold: rise time 1–2 s, overshoot 5–25%,
    settling 1–5 s for 0.3–1 m steps. Establishes acceptance criteria used here.

──────────────────────────────────────────────────────────────────────────────
"""

import sys, os, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import (
    PhysicsLoop, DroneState, PID,
    BAT_V_FULL, BAT_V_EMPTY, BAT_R_INT, BAT_MAX_CURRENT,
    K_F, OMEGA_MAX, DUTY_MAX, DUTY_IDLE, MASS, GRAVITY,
    GE_COEFF, GE_DECAY, R_ROTOR
)

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "B1_althold_step.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "B1_althold_step.png")

SIM_HZ = 200
dt     = 1.0 / SIM_HZ

state   = DroneState()
physics = PhysicsLoop(state)

# Tuned gains for better step response (lower OS, faster settling):
#   outer: kp=1.6, ki=0.25 (↓ less integral windup on rise), vel_sp ±0.4 m/s, integral ±0.5
#   inner: kp=0.80 (↑ stronger velocity braking), ki=0.15 (↓ less oscillation), thr_corr ±0.30, integral ±0.5
# Firmware gains with back-calculation anti-windup on outer loop (in drone_sim tick)
# Anti-windup prevents integral accumulation when vel_sp is saturated → less overshoot
physics.pid_alt_pos.limit    = 0.20   # ±0.20 m/s for autonomous indoor safety

def ticks(s): return int(s * SIM_HZ)

# ── Phase 1: Arm + ramp + find stable hover ───────────────────────────────────
# Target ~1.0 m — well within firmware's operating envelope (ceiling 3.5 m).
print("[B1] Arming …")
with state.lock:
    state.armed = True
    state.ch5   = 1000
    state.ch1   = 1000

# Ramp slowly to just below hover
for pwm in range(1000, 1550, 5):
    with state.lock:
        state.ch1 = pwm
    for _ in range(4):
        physics.tick()

# Climb to ~1.0 m
print("[B1] Climbing …")
with state.lock:
    state.ch1 = 1560
for _ in range(ticks(20.0)):
    physics.tick()
    with state.lock:
        if state.z > 1.0:
            break

# ── Fine hover-search: start just above expected hover (1550) ────────────────
print("[B1] Finding stable hover …")
pwm_now = 1550
with state.lock:
    state.ch1 = pwm_now
for attempt in range(200):
    for _ in range(ticks(0.2)):
        physics.tick()
    with state.lock:
        vz_now = state.vz
    if abs(vz_now) < 0.008:
        break
    pwm_now += 1 if vz_now < 0 else -1
    pwm_now = max(1400, min(1700, pwm_now))
    with state.lock:
        state.ch1 = pwm_now

HOVER_PWM_SIM = pwm_now
HOVER_THR_SIM = (HOVER_PWM_SIM - 1000) / 1000.0

# Hold 3 s to confirm stable, let EKF converge
for _ in range(ticks(3.0)):
    physics.tick()

with state.lock:
    z_stable  = state.z
    vz_stable = state.vz

print(f"[B1] Stable hover: z={z_stable:.3f}m  vz={vz_stable:.4f}m/s  thr={HOVER_THR_SIM:.3f}")

# ── Phase 2: engage alt-hold, descend to 1.0m ────────────────────────────────
with state.lock:
    state.althold          = True
    state.alt_sp           = 1.0
    state.alt_sp_mm        = 1000
    state.hover_thr_locked = HOVER_THR_SIM

print("[B1] Descending to 1.0m …")
for _ in range(ticks(6.0)):   # wait for descent and settle at 1.0m
    physics.tick()

SP0 = 1.00
SP1 = 1.30
SP2 = 1.60
print(f"[B1] Setpoints: {SP0}m → {SP1}m → {SP2}m")

# ── Phase 3: Record steps ─────────────────────────────────────────────────────
rows = []
t_global = 0.0

def record():
    with state.lock:
        rows.append([round(t_global, 4),
                     round(state.z, 4), round(state.ekf_z, 4),
                     round(state.alt_sp, 4), round(state.ekf_vz, 4)])

# Baseline at SP0
with state.lock:
    state.alt_sp    = SP0
    state.alt_sp_mm = SP0 * 1000
print(f"[B1] Baseline at {SP0}m …")
for _ in range(ticks(5.0)):
    record(); physics.tick(); t_global += dt

# Step 1: SP0 → SP1
t_step1 = t_global
physics.pid_alt_pos.integral = 0  # firmware resets only outer integral on altset
with state.lock:
    state.alt_sp    = SP1
    state.alt_sp_mm = SP1 * 1000
print(f"[B1] Step 1 → {SP1}m")
for _ in range(ticks(9.0)):
    record(); physics.tick(); t_global += dt

# Step 2: SP1 → SP2
t_step2 = t_global
physics.pid_alt_pos.integral = 0  # firmware resets only outer integral on altset
with state.lock:
    state.alt_sp    = SP2
    state.alt_sp_mm = SP2 * 1000
print(f"[B1] Step 2 → {SP2}m")
for _ in range(ticks(11.0)):
    record(); physics.tick(); t_global += dt

# ── Metrics ────────────────────────────────────────────────────────────────────
def step_metrics(rows, t_step, t_end, sp_prev, sp_new):
    sr = [r for r in rows if t_step <= r[0] < t_end]
    if not sr: return {}
    step_size = sp_new - sp_prev
    z_vals = [r[2] for r in sr]; t_vals = [r[0] for r in sr]
    peak = max(z_vals) if step_size > 0 else min(z_vals)
    overshoot = max(0, (peak - sp_new) / abs(step_size) * 100)
    t10 = sp_prev + 0.10*step_size; t90 = sp_prev + 0.90*step_size
    idx10 = next((i for i,z in enumerate(z_vals) if z >= t10), None)
    idx90 = next((i for i,z in enumerate(z_vals) if z >= t90), None)
    rise = round(t_vals[idx90] - t_vals[idx10], 3) if (idx10 is not None and idx90 is not None) else None
    band = 0.05 * abs(step_size)
    unsettled = [i for i,z in enumerate(z_vals) if abs(z - sp_new) > band]
    settle = round(t_vals[unsettled[-1]] - t_step, 3) if unsettled else 0.0
    ss = z_vals[int(-4*SIM_HZ):]
    ss_rmse = round(math.sqrt(sum((z-sp_new)**2 for z in ss)/len(ss)) * 100, 3)
    return {"overshoot_pct": round(overshoot,2), "rise_time_s": rise,
            "settling_time_s": settle, "ss_rmse_cm": ss_rmse, "peak_m": round(peak,4)}

t_end = t_global
m1 = step_metrics(rows, t_step1, t_step2, SP0, SP1)
m2 = step_metrics(rows, t_step2, t_end,   SP1, SP2)

print(f"\n[B1] ── Step 1 ({SP0}→{SP1}m) ──")
for k,v in m1.items(): print(f"  {k}: {v}")
print(f"\n[B1] ── Step 2 ({SP1}→{SP2}m) ──")
for k,v in m2.items(): print(f"  {k}: {v}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_s","z_true_m","z_ekf_m","z_setpoint_m","vz_m_s"])
    w.writerows(rows)
print(f"\n[B1] CSV: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
t_v=[r[0] for r in rows]; z_v=[r[1] for r in rows]
ze_v=[r[2] for r in rows]; zs_v=[r[3] for r in rows]

# ── 2nd-order reference step response (cascade PID is inherently 2nd order) ──
# The cascade PID (outer pos loop + inner vel loop) produces 2nd-order dynamics.
# A 1st-order reference never overshoots — wrong for a controller that does.
#
# Fit ζ from observed overshoot: OS% = exp(-π·ζ/√(1−ζ²)) × 100  [Ref 1]
#   → ζ = -ln(OS/100) / √(π² + ln²(OS/100))
# Fit ωn from peak time: tp = π/ωd = π/(ωn√(1−ζ²))
#   → ωn = π / (tp · √(1−ζ²))
# Step response: z(t) = z_sp·[1 − exp(−ζωn·t)/√(1−ζ²)·sin(ωd·t + arccos(ζ))]  [Ref 1]

def second_order_step(t_rel, step_size, sp_prev, zeta, wn):
    """Underdamped 2nd-order step response shifted to start at sp_prev."""
    if zeta >= 1.0:  # critically / over-damped — fall back to 1st order
        tau = 1.0 / wn
        return sp_prev + step_size * (1.0 - np.exp(-t_rel / tau))
    wd  = wn * np.sqrt(1.0 - zeta**2)
    phi = np.arccos(zeta)
    resp = 1.0 - np.exp(-zeta * wn * t_rel) / np.sqrt(1.0 - zeta**2) * np.sin(wd * t_rel + phi)
    return sp_prev + step_size * resp

# Derive ζ and ωn from Step 1 metrics
os1   = (m1.get("overshoot_pct") or 5.0) / 100.0
rt1   = m1.get("rise_time_s") or 1.5
# Clamp OS to valid range for formula
os1   = max(0.001, min(os1, 0.999))
zeta  = -math.log(os1) / math.sqrt(math.pi**2 + math.log(os1)**2)
# ωn from rise time approximation: tr ≈ (1 - 0.4167·ζ + 2.917·ζ²) / ωn  [Mahony 2012]
wn    = (1.0 - 0.4167*zeta + 2.917*zeta**2) / rt1

t_arr = np.array(t_v)

# Build 2nd-order reference across both steps
def z2_ref_full(t_arr, t1, t2, sp0, sp1, sp2, zeta, wn):
    out = np.empty_like(t_arr)
    for i, t in enumerate(t_arr):
        if t < t1:
            out[i] = sp0
        elif t < t2:
            out[i] = second_order_step(np.array([t - t1]), sp1 - sp0, sp0, zeta, wn)[0]
        else:
            out[i] = second_order_step(np.array([t - t2]), sp2 - sp1, sp1, zeta, wn)[0]
    return out

z_ref2 = z2_ref_full(t_arr, t_step1, t_step2, SP0, SP1, SP2, zeta, wn)

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(11,7),sharex=True)
ax1.plot(t_v,z_v, color="blue",  lw=1.5, label="True altitude (sim)")
ax1.plot(t_v,ze_v,color="green", lw=1.5, label="EKF estimate", alpha=0.8)
ax1.step(t_v,zs_v,color="red",  lw=1.5, ls="--", label="Setpoint", where="post")
ax1.plot(t_v, z_ref2, color="purple", lw=1.4, ls=":",  alpha=0.9,
         label=f"2nd-order ref (ζ={zeta:.2f}, ωn={wn:.2f} rad/s) [Ref 1]")
ax1.axvline(t_step1,color="gray",ls=":",lw=1)
ax1.axvline(t_step2,color="gray",ls=":",lw=1)
for sp,m in [(SP1,m1),(SP2,m2)]:
    ax1.axhspan(sp-sp*0.015, sp+sp*0.015, alpha=0.1, color="green")
    if m:
        ax1.annotate(f"OS={m['overshoot_pct']}%\nRise={m['rise_time_s']}s\nSettle={m['settling_time_s']}s\nSS={m['ss_rmse_cm']}cm",
                     xy=(t_step1+0.3 if sp==SP1 else t_step2+0.3, sp-0.05),
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.2",facecolor="lightyellow",alpha=0.9))
ax1.set_ylabel("Altitude (m)")
ax1.set_title(f"EXP-B1: Altitude Hold Step Response  (hold=1.0m → 1.3m → 1.6m)\n"
              f"Literature benchmark: rise 1–2s, overshoot ≤10%, SS <2cm [Ref 2, 4]")
ax1.legend(fontsize=9); ax1.grid(True,alpha=0.3)

err_v=[abs(ze-zs)*100 for ze,zs in zip(ze_v,zs_v)]
ax2.plot(t_v,err_v,color="purple",lw=1,label="|EKF error| (cm)")
ax2.axhline(5.0,color="red",ls="--",lw=1,label="5cm band")
ax2.set_ylabel("Altitude error (cm)"); ax2.set_xlabel("Time (s)")
ax2.legend(fontsize=8); ax2.grid(True,alpha=0.3)

plt.tight_layout(); plt.savefig(OUT_PNG,dpi=150); plt.close()
print(f"[B1] Plot: {OUT_PNG}")
print(f"\n[B1] RESULT: hover_thr={HOVER_THR_SIM:.3f} (PWM={HOVER_PWM_SIM})")
print(f"[B1]   Step1({SP0}→{SP1}m): OS={m1.get('overshoot_pct')}%, rise={m1.get('rise_time_s')}s, settle={m1.get('settling_time_s')}s, SS={m1.get('ss_rmse_cm')}cm")
print(f"[B1]   Step2({SP1}→{SP2}m): OS={m2.get('overshoot_pct')}%, rise={m2.get('rise_time_s')}s, settle={m2.get('settling_time_s')}s, SS={m2.get('ss_rmse_cm')}cm")
