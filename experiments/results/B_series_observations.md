# B Series Experiment Observations

---

## EXP-B1: Altitude Hold Step Response
**What it tests:** Althold PID step response — drone hovers at 1.0m, then setpoint steps to 1.3m then 1.6m.

**Plot:** 2 subplots — altitude (true + EKF + setpoint) over time, and absolute altitude error (cm).

**Timeline:**
- t=0–5s: baseline hold at 1.0m
- t=5s: setpoint steps to 1.3m (+0.3m)
- t=14s: setpoint steps to 1.6m (+0.3m more)

**Key results:**
- Step 1 (1.0→1.3m): OS=5.8%, Rise=1.53s, Settle=3.47s, SS RMSE=0.71cm
- Step 2 (1.3→1.6m): OS=6.77%, Rise=1.49s, Settle=5.42s, SS RMSE=0.24cm
- EKF tracks true altitude closely (green ≈ blue line)
- Error subplot spikes at each step then decays below 5cm after settling

**Inferences:**
- Althold PID (outer Kp=1.6, Ki=0.5; inner Kp=0.70, Ki=0.30) gives clean, low-overshoot step response
- ~6% overshoot is small and acceptable for indoor altitude hold
- Anti-windup on outer loop prevents excessive overshoot
- Rise time ~1.5s — moderate climb speed (limited to ±0.20 m/s for indoor safety)
- SS RMSE <1cm — excellent steady-state accuracy

---

## EXP-B2: Position Hold Disturbance Rejection
**What it tests:** Poshold holds at 1.0m altitude. At t=5s, lateral impulse (Fx=0.05N for 0.2s). Measures max drift, return time, SS RMSE.

**Plot:** 3 subplots — XY top-down trajectory, altitude during recording, XY error over time.

**Key results:**
- Pre-disturbance XY RMSE: 3.29cm (EKF optical flow drift — irreducible without GPS)
- Max drift after impulse: 12.88cm at t=6.13s
- Return-to-hold time: 1.79s
- Post-disturbance SS RMSE: 2.87cm
- Altitude: holds tightly at 1.0m throughout (±8mm oscillation)

**Plot reading guide:**
- Red dot = average true hold position (pre-disturbance mean)
- Green path = pre-disturbance cluster
- Blue path = full trajectory including post-disturbance swing and return
- XY error subplot: red dashed = disturbance moment, blue dotted = recovery time

**Inferences:**
- Poshold integral successfully rejects impulse and returns drone within 1.79s
- Pre-disturbance RMSE of 3.29cm is EKF/optical-flow drift, not a tuning problem — irreducible without GPS
- Red dot offset from blue path is normal — EKF coordinate frame drifts from true physical frame
- Outer Kp=1.2 prevents oscillation after disturbance
- Altitude is fully decoupled from lateral disturbance — althold unaffected by impulse

---

## EXP-B3: Attitude Step Response (Roll and Pitch)
**What it tests:** At hover, commands roll_des=10° for 2.5s then returns to 0°. Same for pitch. No althold or poshold active — pure attitude control test.

**Plot:** 2 subplots — roll angle (measured + step setpoint) and pitch angle (measured + step setpoint). Setpoints shown as step functions so timing is clear.

**Timeline:**
- t=0–2s: roll baseline at 0°
- t=2s: roll steps to 10°, t≈4.5s returns to 0°
- t=7.5–9.5s: pitch baseline at 0°
- t=9.5s: pitch steps to 10°, t≈12s returns to 0°

**Key results:**
- Rise time: 0.29s (roll), 0.285s (pitch)
- Overshoot: 0% — no ringing, clean response
- Peak during hold: ~9.7° (0.3° undershoot / SS error)
- SS RMSE during hold: ~0.34°
- Residual after returning to 0°: ~0.35–0.4°

**Inferences:**
- Attitude PID is well-tuned and symmetric (roll ≈ pitch performance)
- 0% overshoot — stable, no oscillation
- ~0.4° residual at 0° after step is Madgwick filter drift caused by lateral acceleration during unconstrained tilt (no poshold active in B3)
- In poshold mode this residual is automatically corrected by position error feedback
- In attitude-only mode (no optical flow/GPS), drone slowly drifts laterally — pilot must correct manually
- Any filter (Madgwick, Kalman) using accelerometers has this drift under lateral acceleration — not fixable without a position sensor
- Kalman9D alone does not fix this — it estimates position, not attitude; poshold is what closes the loop

---

## EXP-B4: Combined Alt+Pos Hold Under Steady Wind
**What it tests:** Both althold and poshold active simultaneously. Constant wind force (Fx=0.02N) applied for 20s from the start of recording.

**Plot:** 4 subplots — XY error over time, XY trajectory (top-down), X-axis error, altitude error.

**Key results:**
- SS XY RMSE (last 10s): 9.21cm
- SS Z RMSE (last 10s): 0.30cm
- Max XY error: 32.00cm (first few seconds while integral builds)
- XY settled below 10cm at t=19s
- Altitude error stays within ±0.76cm throughout

**Plot reading guide:**
- XY error = straight-line distance from setpoint (always positive, both axes combined)
- X error = signed error in wind direction — shows integral gradually cancelling wind drift
- XY trajectory = drone path viewed from above; red dot = hold setpoint
- Altitude error subplot = Z error in cm (separate from XY)

**Inferences:**
- Altitude hold is excellent under lateral wind (0.30cm RMSE) — althold fully decoupled from lateral disturbances
- Position hold integral takes ~8–19s to learn and cancel constant wind load
- Initial oscillations (~32cm peak) are normal — poshold PID hunts until integral accumulates enough correction
- 9.21cm SS RMSE is acceptable for indoor hover under 0.02N constant wind
- Wind only pushes X, so X error is the key channel to watch; Y error barely changes

---

## EXP-B5: Hover Throttle vs Battery SOC
**What it tests:** How much hover throttle (PWM) must increase as battery discharges from 100% to 20% SOC.

**Plot:** 2 subplots — hover PWM vs SOC (left, x-axis inverted: 100%→20%), terminal voltage vs SOC (right).

**Key results:**
- Hover PWM at 100% SOC: 1534
- Hover PWM at 20% SOC: 1772
- ΔPW = 238 across full discharge range
- Terminal voltage: 4.12V (100%) → 3.12V (20%)
- v_factor drops from 0.962 → 0.553

**Battery model (corrected values in drone_sim.py):**
- BAT_R_INT = 0.05Ω (was 0.20Ω — too high, caused v_term clamping at V_empty for all SOC)
- BAT_MAX_CURRENT = 3.0A (was 10A — too high, same clamping issue)
- Old values made the voltage plot flat at 3.0V and ΔPW only ~13 — unrealistic

**Inferences:**
- Althold and poshold work correctly WITHOUT battery voltage monitoring
- As battery drains → thrust drops → altitude drops → althold integral auto-increases throttle → compensates transparently
- No explicit battery check needed for althold/poshold to function during normal flight
- Risk zone: below ~3.5V (≈40% SOC), voltage sag under full load may exceed althold correction headroom
- Practical guideline: land when battery warning LED blinks; do not rely on althold/poshold at very low SOC
- Analytical hover formula must use correct omega mapping: omega = ((DUTY_IDLE + thr*(DUTY_MAX-DUTY_IDLE))/DUTY_MAX) * OMEGA_MAX — not thr * OMEGA_MAX

---

## General Notes (All B Series)
- All experiments start hover at **1.0m** (althold descends from hover-find altitude to 1.0m setpoint)
- Hover-find converges at PWM ≈ 1535 with corrected battery model (was ~1830 with old model)
- Hover-find minimum clamp set to 1400 (was 1700 — too high for corrected battery model)
- B3 is the only experiment without althold/poshold — pure attitude mode
- B1, B2, B4 all use althold; B2 and B4 additionally use poshold
