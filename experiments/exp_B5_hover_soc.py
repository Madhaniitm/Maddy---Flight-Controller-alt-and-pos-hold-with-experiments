"""
EXP-B5: Hover Throttle vs Battery SOC
========================================
At 100%, 90%, …, 20% SOC, analytically computes the exact throttle fraction
that produces hover (vz≈0, z stable) and validates with physics simulation.
Records hover_pwm vs SOC — baseline for LLM adaptive hover experiment.

No WebSocket, no API, no GUI.
Saves: results/B5_hover_soc.csv, results/B5_hover_soc.png

─── What is tested ──────────────────────────────────────────────────────────
Hover condition [Ref 4]:  4 · K_F · ω² · v_factor · k_ge = M · g
  v_factor = (V_term / V_full)²   (from A6 battery model [Ref 1])
  k_ge = 1 + Ca·exp(−z/(GD·R))   (from A4 ground effect model)

As SOC drops, V_term falls → v_factor falls → same ω produces less thrust
→ a higher ω (higher throttle) is needed to maintain hover. This experiment
quantifies the throttle increase required across the battery discharge range.

The A6 experiment showed that thrust ∝ V_term² [Ref 1, 2]. This experiment
applies that model to compute the exact adaptive hover throttle at each SOC.
If the flight controller uses a fixed hover throttle (tuned at 100% SOC),
it will cause altitude loss as the battery drains — a documented failure mode
in Crazyflie firmware [Ref 3]. B5 provides the look-up table that an adaptive
controller would use to compensate.

─── References ──────────────────────────────────────────────────────────────

[1] Soto-García, L. et al. (2023)
    "Battery Testing and Discharge Model Validation for Electric Unmanned
     Aerial Vehicles (UAV)"
    Sensors, 23(15), 6937. DOI: 10.3390/s23156937
    Validates V_term = V_oc − I·R_int model for LiPo UAVs. The SOC-dependent
    terminal voltage is the input to the hover throttle computation here.

[2] Bitcraze AB (2025)
    "Keeping Thrust Consistent as the Battery Drains"
    Bitcraze Blog, October 2025.
    URL: https://www.bitcraze.io/2025/10/keeping-thrust-consistent-as-the-battery-drains/
    Confirms that on the Crazyflie, thrust drops as battery drains at constant
    PWM. The V²-based compensation implemented in firmware is the same model
    this experiment validates analytically.

[3] Bitcraze AB (n.d.)
    "PWM to Thrust"
    Crazyflie Firmware Documentation.
    URL: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/pwm-to-thrust/
    Documents the quadratic voltage-thrust relationship on Crazyflie hardware.
    Fixed hover throttle at 100% SOC causes altitude loss at low SOC — B5
    provides the corrective look-up table.

[4] Mahony, R., Kumar, V. & Corke, P. (2012)
    "Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor"
    IEEE Robotics & Automation Magazine, 19(3), 20–32.
    DOI: 10.1109/MRA.2012.2206474
    Hover condition: 4·K_F·ω² = M·g (no battery/GE effects). Extended here
    to include v_factor and k_ge for realistic hover throttle computation.

[5] Vančura, J., Straka, M. & Pěnička, R. (2022)
    "Modeling and Validation of Electric Multirotor Unmanned Aerial Vehicle
     System Energy Dynamics"
    eTransportation, 12, 100166. DOI: 10.1016/j.etran.2022.100166
    Full energy dynamics model validates SOC-dependent hover power. Shows that
    hover current increases as battery voltage falls, consistent with B5 results.

──────────────────────────────────────────────────────────────────────────────
"""

import sys, os, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import (
    PhysicsLoop, DroneState,
    BAT_CAPACITY_MAH, BAT_V_FULL, BAT_V_EMPTY, BAT_R_INT, BAT_MAX_CURRENT,
    K_F, OMEGA_MAX, DUTY_MAX, DUTY_IDLE, MASS, GRAVITY,
    GE_COEFF, GE_DECAY, R_ROTOR
)

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "B5_hover_soc.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "B5_hover_soc.png")

SIM_HZ = 200
dt     = 1.0 / SIM_HZ
TARGET_Z = 0.5   # m — hover height for this test (above GE region)

def compute_hover_throttle_frac(soc_pct):
    """
    Analytically compute the hover throttle fraction at a given SOC.
    Hover condition: 4 * K_F * omega^2 * v_factor * k_ge = MASS * GRAVITY
    Correct omega mapping (mirrors physics):
      omega = ((DUTY_IDLE + thr*(DUTY_MAX-DUTY_IDLE)) / DUTY_MAX) * OMEGA_MAX
    Battery current uses physics-matched formula:
      throttle_frac = (DUTY_IDLE + thr*(DUTY_MAX-DUTY_IDLE)) / DUTY_MAX
      bat_i = throttle_frac * BAT_MAX_CURRENT
    This is identical to how drone_sim computes bat_current from motor omega.
    """
    soc = soc_pct / 100.0
    v_oc = BAT_V_EMPTY + (BAT_V_FULL - BAT_V_EMPTY) * soc

    thr_guess = 0.52
    for _ in range(30):
        # Match physics battery current model exactly:
        # throttle_frac = sum(omega)/(4*OMEGA_MAX) = duty/DUTY_MAX at level hover
        duty_iter  = DUTY_IDLE + thr_guess * (DUTY_MAX - DUTY_IDLE)
        thr_frac   = duty_iter / DUTY_MAX
        bat_i  = thr_frac * BAT_MAX_CURRENT
        v_term = max(BAT_V_EMPTY, v_oc - bat_i * BAT_R_INT)
        v_fac  = (v_term / BAT_V_FULL) ** 2

        k_ge = 1.0 + GE_COEFF * math.exp(-TARGET_Z / (GE_DECAY * R_ROTOR))

        # Solve for omega, then back-convert to throttle fraction
        # 4 * K_F * omega^2 * v_fac * k_ge = M*g
        omega_hover = math.sqrt(MASS * GRAVITY / (4.0 * K_F * v_fac * k_ge))
        # omega = ((DUTY_IDLE + thr*(DUTY_MAX-DUTY_IDLE)) / DUTY_MAX) * OMEGA_MAX
        duty_hover  = (omega_hover / OMEGA_MAX) * DUTY_MAX
        thr_new = (duty_hover - DUTY_IDLE) / (DUTY_MAX - DUTY_IDLE)
        thr_new = max(0.0, min(1.0, thr_new))

        if abs(thr_new - thr_guess) < 1e-7:
            break
        thr_guess = thr_new

    hover_pwm = int(1000 + thr_guess * 1000)
    return thr_guess, hover_pwm, v_term, v_fac

# ── Run for each SOC level ─────────────────────────────────────────────────────
SOC_LEVELS = [100, 90, 80, 70, 60, 50, 40, 30, 20]

rows = []
print("[B5] Computing hover throttle vs SOC …")
print(f"{'SOC%':>6} {'hover_frac':>12} {'hover_PWM':>10} {'V_term(V)':>10} {'v_factor':>10}")
print("-" * 55)

for soc in SOC_LEVELS:
    thr_frac, pwm, v_term, v_fac = compute_hover_throttle_frac(soc)
    print(f"{soc:6d}% {thr_frac:12.5f} {pwm:10d} {v_term:10.3f} {v_fac:10.5f}")
    rows.append([soc, round(thr_frac, 6), pwm, round(v_term, 4), round(v_fac, 6)])

# ── Validate with simulation ───────────────────────────────────────────────────
# Use hover-find (same algorithm as B1) at each SOC to find the actual hover PWM
# in physics, then compare with the analytical prediction.
# This is robust: it finds the true hover operating point regardless of initial
# conditions, motor lag, or GE transients.
print("\n[B5] Validating with physics simulation (hover-find method) …")
sim_results = []

for test_soc in [100, 60]:
    state   = DroneState()
    physics = PhysicsLoop(state)

    # Pre-discharge battery to target SOC
    state._bat_charge_used = BAT_CAPACITY_MAH * (1.0 - test_soc / 100.0)

    # Get analytically predicted hover PWM for this SOC
    thr_pred, pwm_pred, _, _ = compute_hover_throttle_frac(test_soc)

    with state.lock:
        state.armed = True
        state.ch5   = 1000
        state.ch1   = 1000

    # Ramp up gently
    for pwm in range(1000, 1560, 5):
        with state.lock:
            state.ch1 = pwm
        for _ in range(4):
            physics.tick()

    # Climb to ~0.5m
    with state.lock:
        state.ch1 = 1560
    for _ in range(int(15.0 * SIM_HZ)):
        physics.tick()
        with state.lock:
            if state.z > 0.5:
                break

    # Hover-find: start at analytical prediction, adjust until vz ≈ 0.
    # Starting at pwm_pred avoids the drone falling before converging —
    # the analytical estimate is close enough that only ±few PWM correction needed.
    pwm_now = pwm_pred
    with state.lock:
        state.ch1 = pwm_now
    for _ in range(200):
        for _ in range(int(0.2 * SIM_HZ)):
            physics.tick()
        with state.lock:
            vz_now = state.vz
        if abs(vz_now) < 0.008:
            break
        pwm_now += 1 if vz_now < 0 else -1
        pwm_now = max(1400, min(1700, pwm_now))
        with state.lock:
            state.ch1 = pwm_now

    # Settle
    for _ in range(int(2.0 * SIM_HZ)):
        physics.tick()

    with state.lock:
        z_actual  = state.z
        vz_actual = state.vz

    delta_pwm = pwm_now - pwm_pred
    print(f"  SOC={test_soc}%  Analytical={pwm_pred}  Sim hover-find={pwm_now}"
          f"  ΔPW={delta_pwm:+d}  z={z_actual:.3f}m  vz={vz_actual:+.4f}m/s")
    sim_results.append((test_soc, pwm_pred, pwm_now, z_actual, vz_actual))

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["soc_pct", "hover_throttle_frac", "hover_pwm",
                "v_term_V", "v_factor"])
    w.writerows(rows)
print(f"\n[B5] CSV saved: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
soc_v  = [r[0] for r in rows]
pwm_v  = [r[2] for r in rows]
thr_v  = [r[1] * 100 for r in rows]   # %
vt_v   = [r[3] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("EXP-B5: Hover Throttle vs Battery SOC  —  "
             "Fixed throttle causes progressive altitude loss as battery drains [Ref 2, 3]",
             fontsize=10, fontweight="bold")

# A6 analytical prediction: required throttle = thr_100 * (V_term_100 / V_term_soc)
# This is the A6 curve from the required-throttle subplot, plotted here as comparison [Ref 1, 2]
thr_100      = rows[0][1]         # hover throttle fraction at 100% SOC
vt_100       = rows[0][3]         # V_term at 100% SOC
a6_thr_v     = [thr_100 * (vt_100 / r[3]) * 1000 + 1000 for r in rows]  # as PWM

ax1.plot(soc_v, pwm_v, color="blue", linewidth=2, marker="o", markersize=6,
         label="Analytical hover PWM (this exp)")
ax1.plot(soc_v, a6_thr_v, color="orange", linewidth=1.5, linestyle="--", marker="^",
         markersize=5, label="A6 model prediction (V² scaling) [Ref 1, 2]")
ax1.set_xlabel("Battery SOC (%)")
ax1.set_ylabel("Hover throttle (PWM)")
ax1.set_title("Required Hover PWM vs SOC")
ax1.invert_xaxis()
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Annotate sim validation points (hover-find result vs analytical prediction)
for soc, pwm_pred_s, pwm_found, z, vz in sim_results:
    ax1.scatter([soc], [pwm_found], color="red", s=80, zorder=5,
                label="Sim hover-find" if soc == sim_results[0][0] else "")
    delta = pwm_found - pwm_pred_s
    ax1.annotate(f"Sim: {pwm_found} (ΔPW={delta:+d})", xy=(soc, pwm_found),
                 xytext=(soc + 5, pwm_found + 5), fontsize=8,
                 arrowprops=dict(arrowstyle="->"))

# PWM change across SOC range
delta_pwm = pwm_v[-1] - pwm_v[0]
ax1.annotate(f"ΔPW={delta_pwm} over SOC range",
             xy=(50, (pwm_v[0] + pwm_v[-1]) / 2),
             fontsize=9, color="darkblue")

ax2.plot(soc_v, vt_v, color="orange", linewidth=2, marker="s", markersize=6,
         label="V_term at hover current")
ax2.set_xlabel("Battery SOC (%)")
ax2.set_ylabel("Terminal voltage V_term (V)")
ax2.set_title("Terminal Voltage vs SOC at Hover Current [Ref 1]")
fig.subplots_adjust(top=0.88)
ax2.invert_xaxis()
ax2.axhline(BAT_V_EMPTY, color="red", linestyle="--", label=f"V_empty={BAT_V_EMPTY}V")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[B5] Plot saved: {OUT_PNG}")

pwm_100 = next(r[2] for r in rows if r[0] == 100)
pwm_40  = next(r[2] for r in rows if r[0] == 40)
print(f"\n[B5] RESULT: Hover PWM: {pwm_100} at 100% SOC → {pwm_40} at 40% SOC (ΔPW={pwm_40-pwm_100})")
print(f"[B5]         Fixed hover throttle tuned at 100% SOC will cause altitude loss at 40% SOC.")
