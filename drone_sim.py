#!/usr/bin/env python3
"""
Maddy Drone Simulator  ·  drone_sim.py
=======================================
Drop-in Python replacement for the ESP32-S3 + LiteWing hardware stack.
Speaks the IDENTICAL WebSocket + HTTP protocol so the browser controller
and AI agent (keyboard_server.py) work with ZERO code changes.

Simulates:
  • 6-DOF quadrotor physics (100 Hz)
  • Cascaded angle+rate PID — exact copy of controlANGLE2() firmware
  • LiteWing 1-D Kalman altitude estimator + cascaded alt-hold PID
  • LiteWing 2-D Kalman XY estimator + cascaded position-hold PID
  • ToF, optical-flow, and IMU sensor noise models
  • Synthetic forward-facing camera (JPEG via HTTP)
  • Real-time matplotlib visualiser (top-down + altitude strip)

Endpoints (match real ESP32 exactly):
  ws://localhost:81           ← WebSocket flight control
  http://localhost:8080/capture ← JPEG camera frame

Install:
  pip install websockets numpy pillow matplotlib

Run:
  python3 drone_sim.py

Then open ESP32S3_DroneController.html in your browser and enter IP: localhost
(change the camera fetch port in keyboard_server.py to 8080 for vision tools)
"""

import asyncio, threading, time, json, math, random, io, struct
import http.server, socketserver, logging
import csv, os, datetime
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[SIM] Pillow not installed — camera disabled. pip install pillow")

try:
    import matplotlib
    matplotlib.use("MacOSX")   # native macOS backend — no TkAgg needed
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False
    print("[SIM] matplotlib unavailable — visualiser disabled.")

try:
    import websockets
    WS_OK = True
except ImportError:
    WS_OK = False
    print("[SIM] websockets not installed. pip install websockets")

logging.basicConfig(level=logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SIM_HZ     = 200    # physics update rate (Hz)
TEL_HZ     = 10     # telemetry broadcast rate (Hz)
WS_PORT    = 81     # WebSocket port — matches ESP32
HTTP_PORT  = 8080   # Camera HTTP port (ESP32 uses 80; 8080 avoids root on Mac/Linux)
CAM_W, CAM_H = 320, 240

# ── Drone physical parameters — Crazyflie 2.x characterisation ────────────────
# Primary source: Förster (2015) "System Identification of the Crazyflie 2.0"
#                 ETH Zürich, Autonomous Systems Lab
# Your drone resembles CF brushed: frame/props identical, heavier due to ESP32
# + LiteWing module (~20 g extra vs bare CF2.0 at 30 g) → MASS = 0.050 kg

MASS      = 0.050      # kg  — weigh yours; CF2.0+battery=0.030, yours ≈ 0.050
GRAVITY   = 9.81       # m/s²

# CF2.0 geometry: 46 mm diagonal → motor-to-centre = 46/√2/2 = 16.3 mm … no:
# diagonal tip-to-tip = 92 mm → arm = 92/(2√2) ≈ 32.5 mm
ARM       = 0.0325     # m  — Förster 2015 (measured, ±0.5 mm)

# Moments of inertia — Förster 2015, Table 1 (bifilar pendulum measurement)
# Your drone is heavier but extra mass (PCB, sensor) sits near centre → Ixx/Iyy
# scale less than linearly. Use CF value × (50/30)^0.6 ≈ 1.35 as correction.
I_XX      = 1.657e-5 * 1.35   # kg·m²  ≈ 2.24e-5  (CF measured × mass correction)
I_YY      = 1.657e-5 * 1.35   # kg·m²  ≈ 2.24e-5
I_ZZ      = 2.900e-5 * 1.35   # kg·m²  ≈ 3.92e-5  (CF measured × mass correction)

# Translational drag — Landry et al. (2020), Crazyflie drag identification
# DRAG_XY: horizontal, ~0.01 N·s/m gives ~5 m/s terminal at 10° tilt (realistic)
# DRAG_Z:  vertical,  terminal fall ≈ MASS×g/DRAG_Z → 0.050×9.81/0.04 ≈ 12 m/s
DRAG_XY   = 0.010     # N·s/m  — Landry et al. identified value for CF
DRAG_Z    = 0.04      # N·s/m  — estimated from drop test terminal velocity

DUTY_IDLE = 45
DUTY_MAX  = 255

# ── Motor model — Crazyflie brushed (7×16 mm coreless DC motor, 46 mm props) ──
# Source: Bitcraze official PWM-to-thrust characterisation (measured under prop load)
# No-load RPM at 4.2V ≈ 55,000 RPM (14,000 KV × 4.2V)
# Full-throttle loaded RPM ≈ 25,000 RPM at ~3.3V (Bitcraze measured, 93–100% PWM)
# 25,000 RPM × 2π/60 = 2618 rad/s
# Brushed motors respond FASTER than brushless → TAU_MOTOR ≈ 30 ms
OMEGA_MAX  = 2618.0    # rad/s  — 25,000 RPM loaded (Bitcraze measured, 7×16mm + 46mm prop)
TAU_MOTOR  = 0.030     # s  — brushed coreless DC: ~30 ms (brushless would be 50 ms)

# Thrust: F = K_F × ω²  — self-calibrated to hover condition
# Hover at ch1≈1520 → thro_des≈0.52 → ω_hover ≈ 0.52 × OMEGA_MAX
# 4 × K_F × ω_hover² = MASS × g  →  K_F derived automatically below
K_F        = MASS * GRAVITY / (4.0 * (DUTY_IDLE + 0.52*(DUTY_MAX-DUTY_IDLE))**2
                                / DUTY_MAX**2 * OMEGA_MAX**2)

# Torque coefficient — CF characterisation: kM/kF ≈ 0.006 (Förster 2015 Table 1)
K_Q        = K_F * 0.006

# Rotor gyroscopic inertia — CF 46 mm prop, ≈ 0.3 g mass:
# J ≈ 0.5 × m_prop × R² = 0.5 × 3e-4 × 0.023² = 7.9e-8 kg·m²
J_ROTOR    = 8.0e-8              # kg·m²  (Förster 2015 estimate)
MOTOR_SPIN = np.array([+1,-1,+1,-1], dtype=float)  # CCW=+1, CW=-1

# ── Ground effect — CF 46 mm props (R_ROTOR = 23 mm) ─────────────────────────
# Model: k_ge = 1 + GE_COEFF * exp(-z / (GE_DECAY * R_ROTOR))  [He et al. 2019]
# Coefficients GE_COEFF=0.37, GE_DECAY=1.43 fitted via least-squares to
# Kan et al. 2019 (RA-L) data — the only GE model validated on Crazyflie 2.0.
# Fit range: z/R = 0.5–5.0 (valid Kan domain). RMSE vs Kan = 0.069 (vs 0.090 generic).
# Ref: Kan, X. et al. (2019) "Analysis of Ground Effect for Small-scale UAVs"
#      IEEE Robotics and Automation Letters, accepted July 2019.
R_ROTOR    = 0.023     # m  (CF prop radius = 46mm/2)
GE_COEFF   = 0.37      # fitted to Kan 2019 Crazyflie data (was 0.25 generic He 2019)
GE_DECAY   = 1.43      # fitted to Kan 2019 Crazyflie data (was 2.0 generic He 2019)

# Legacy per-PWM constants for telemetry display
K_THRUST  = K_F * OMEGA_MAX**2 / DUTY_MAX
K_YAW_TQ  = K_Q * OMEGA_MAX**2 / DUTY_MAX

# ── Battery — CF2.0 compatible 1S LiPo ────────────────────────────────────────
# CF2.0 uses 250 mAh; your drone is heavier → assume 300 mAh pack
# CF brushed hover current ≈ 2–3 A total; max ≈ 6 A (4 × 1.5 A brushed motors)
BAT_CAPACITY_MAH = 300.0          # mAh
BAT_V_FULL       = 4.20           # V
BAT_V_NOMINAL    = 3.70           # V
BAT_V_EMPTY      = 3.00           # V  (cutoff)
BAT_R_INT        = 0.05           # Ω  internal resistance (realistic LiPo ~0.03-0.05 Ω)
BAT_MAX_CURRENT  = 3.0            # A  hover current total (realistic for this drone class)
# Thrust ∝ V² (brushed motor speed ∝ V directly, unlike brushless with ESC)

# ── Firmware default PID gains (exact copy from Maddy_Flight_Controller.ino) ──
Kp_roll_angle  = 0.3;   Ki_roll_angle  = 0.01;  Kd_roll_angle  = 0.0
Kp_pitch_angle = 0.3;   Ki_pitch_angle = 0.01;  Kd_pitch_angle = 0.0
Kp_roll_rate   = 0.08;  Ki_roll_rate   = 0.0;   Kd_roll_rate   = 0.01
Kp_pitch_rate  = 0.08;  Ki_pitch_rate  = 0.0;   Kd_pitch_rate  = 0.01
Kp_yaw         = 0.06;  Ki_yaw         = 0.0;   Kd_yaw         = 0.008
B_loop_roll    = 0.95;  B_loop_pitch   = 0.95
KL             = 30.0   # outer-loop gain (firmware line 3121)
i_limit        = 20.0
maxRoll = 10.0; maxPitch = 10.0; maxYaw = 90.0
yaw_scale_val  = 1.0

# ── LiteWing EKF + hold defaults ───────────────────────────────────────────────
lw_pidZ_kp    = 1.6;   lw_pidZ_ki    = 0.5    # alt-hold outer  (pos → vel)
lw_pidVZ_kp   = 0.70;  lw_pidVZ_ki   = 0.30   # alt-hold inner  (vel → thrust)
lw_hover_thr  = 0.50                           # hover throttle baseline (0-1)
lw_pidX_kp    = 1.2;   lw_pidX_ki    = 0.30   # pos-hold outer  (pos → vel)
lw_pidVX_kp   = 0.90;  lw_pidVX_ki   = 0.15;  lw_pidVX_kd = 0.05  # pos-hold inner  (vel → roll/pitch)
lw_rp_limit   = 10.0                           # roll/pitch clamp for pos-hold (°)
lw_xy_vel_max = 1.0                            # velocity setpoint limit (m/s)
lw_kf_q       = 1.0                            # Kalman process noise
lw_kf_r_tof   = 0.05                           # Kalman ToF measurement noise (m)
lw_flow_std   = 2.0                            # Optical flow noise (px)

# ── Sensor noise ───────────────────────────────────────────────────────────────
GYRO_NOISE = 0.5    # deg/s  RMS
ANGLE_NOISE = 0.2   # deg    RMS
TOF_NOISE   = 0.005 # m      RMS
FLOW_NOISE  = 0.015 # m/s    RMS (converted from pixels via altitude)

# ── Crash / tip-over thresholds ────────────────────────────────────────────────
# TIPOVER_ANGLE: arm tip contacts floor when |roll| or |pitch| exceeds this.
# For a 65 mm diagonal frame (arm = 32.5 mm, body height ≈ 15 mm):
#   arm tip clears floor by ARM*sin(θ) − body_h/2 ≈ 0 at θ ≈ 27°.
# Use 45° as conservative threshold (props definitely strike at this angle).
TIPOVER_ANGLE = 45.0   # deg  — arm+prop contacts ground
CRASH_VZ      = -2.0   # m/s  — hard-landing impact threshold

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def noisy(val, std):
    return val + random.gauss(0, std)


def quat_to_R(q):
    """Rotation matrix (body→world) from unit quaternion [w,x,y,z]."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def quat_euler_deg(q):
    """Roll, pitch, yaw (deg) from unit quaternion [w,x,y,z]."""
    w, x, y, z = q
    roll  = math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = math.degrees(math.asin(clamp(2*(w*y-z*x), -1.0, 1.0)))
    yaw   = math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw


# ═══════════════════════════════════════════════════════════════════════════════
#  PID CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class PID:
    """Standard PID — derivative on error (outer angle / position loops).

    ilimit : separate clamp for the integrator accumulator (mirrors firmware's
             constrain(integral, -0.5, 0.5) before the output constrain).
             If None, falls back to limit for backwards compatibility.
    """
    def __init__(self, kp=0, ki=0, kd=0, limit=None, ilimit=None):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.limit  = limit
        self.ilimit = ilimit if ilimit is not None else limit
        self.integral = 0.0
        self.prev_err = 0.0

    def update(self, error, dt, integrator_active=True):
        if integrator_active:
            self.integral = clamp(self.integral + error * dt,
                                  -self.ilimit if self.ilimit else -1e9,
                                   self.ilimit if self.ilimit else  1e9)
        deriv = (error - self.prev_err) / dt if dt > 1e-9 else 0.0
        self.prev_err = error
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        if self.limit:
            out = clamp(out, -self.limit, self.limit)
        return out

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0


class RatePID:
    """
    Rate PID with derivative-on-measurement — matches dRehmFlight firmware exactly:
        output = Kp*error + Ki*integral - Kd*gyro_rate
    Avoids the derivative spike when the rate setpoint jumps.
    """
    def __init__(self, kp=0, ki=0, kd=0, limit=None):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.limit = limit
        self.integral = 0.0

    def update(self, error, measurement, dt, integrator_active=True):
        if integrator_active:
            self.integral = clamp(self.integral + error * dt,
                                  -self.limit if self.limit else -1e9,
                                   self.limit if self.limit else  1e9)
        out = self.kp * error + self.ki * self.integral - self.kd * measurement
        if self.limit:
            out = clamp(out, -self.limit, self.limit)
        return out

    def reset(self):
        self.integral = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MADGWICK FILTER  (attitude estimation — exact match to firmware)
#  Reference: Madgwick et al. 2010, same beta=0.03 as firmware default
# ═══════════════════════════════════════════════════════════════════════════════

class MadgwickFilter:
    """
    Quaternion-based Madgwick AHRS filter.
    Fuses gyroscope + accelerometer exactly as the ESP32 firmware does.
    Beta = 0.03 matches firmware default (B_madgwick = 0.03).
    """
    def __init__(self, beta=0.03):
        self.beta = beta
        # Quaternion [w, x, y, z] — start level
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gx, gy, gz, ax, ay, az, dt):
        """
        gx/gy/gz : gyro rates in deg/s  (converted to rad/s internally)
        ax/ay/az : accelerometer in G's
        dt       : timestep in seconds
        """
        q = self.q
        gx = math.radians(gx);  gy = math.radians(gy);  gz = math.radians(gz)

        # Normalise accelerometer
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm < 1e-6:
            return   # free-fall — skip update
        ax /= norm;  ay /= norm;  az /= norm

        q0, q1, q2, q3 = q

        # Gradient descent step (objective function f = q ⊗ g - a, g=[0,0,1])
        f1 = 2*(q1*q3 - q0*q2) - ax
        f2 = 2*(q0*q1 + q2*q3) - ay
        f3 = 2*(0.5 - q1*q1 - q2*q2) - az

        j11 = -2*q2;  j12 =  2*q3;  j13 = -2*q0;  j14 =  2*q1
        j21 =  2*q1;  j22 =  2*q0;  j23 =  2*q3;  j24 =  2*q2
        j31 =  0;     j32 = -4*q1;  j33 = -4*q2;  j34 =  0

        step0 = j11*f1 + j21*f2 + j31*f3
        step1 = j12*f1 + j22*f2 + j32*f3
        step2 = j13*f1 + j23*f2 + j33*f3
        step3 = j14*f1 + j24*f2 + j34*f3

        # Normalise step
        norm_s = math.sqrt(step0**2 + step1**2 + step2**2 + step3**2)
        if norm_s > 1e-6:
            step0 /= norm_s;  step1 /= norm_s
            step2 /= norm_s;  step3 /= norm_s

        # Rate of change of quaternion from gyro
        qdot0 = 0.5*(-q1*gx - q2*gy - q3*gz) - self.beta*step0
        qdot1 = 0.5*( q0*gx + q2*gz - q3*gy) - self.beta*step1
        qdot2 = 0.5*( q0*gy - q1*gz + q3*gx) - self.beta*step2
        qdot3 = 0.5*( q0*gz + q1*gy - q2*gx) - self.beta*step3

        # Integrate
        q0 += qdot0 * dt;  q1 += qdot1 * dt
        q2 += qdot2 * dt;  q3 += qdot3 * dt

        # Normalise quaternion
        norm_q = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q = np.array([q0, q1, q2, q3]) / norm_q

    def euler_deg(self):
        """Return (roll, pitch, yaw) in degrees from quaternion."""
        q0, q1, q2, q3 = self.q
        roll  = math.degrees(math.atan2(2*(q0*q1 + q2*q3),
                                         1 - 2*(q1*q1 + q2*q2)))
        pitch = math.degrees(math.asin( clamp(2*(q0*q2 - q3*q1), -1.0, 1.0)))
        yaw   = math.degrees(math.atan2(2*(q0*q3 + q1*q2),
                                         1 - 2*(q2*q2 + q3*q3)))
        return roll, pitch, yaw

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])


# ═══════════════════════════════════════════════════════════════════════════════
#  9-STATE EKF  — exact port of kalman_core.c (Bitcraze / Crazyflie)
#  States: [X, Y, Z, PX, PY, PZ, D0, D1, D2]
#    X/Y/Z   : world position (m)
#    PX/PY/PZ: body-frame velocity (m/s)
#    D0/D1/D2: attitude error (rad)
#  Quaternion q[4]=[w,x,y,z] and rotation matrix R[3×3] maintained alongside.
#  Replaces the old Kalman1D (altitude) and Kalman2D (XY) — one filter for all.
# ═══════════════════════════════════════════════════════════════════════════════

class Kalman9D:
    # State indices (mirrors KC_STATE_* in firmware)
    X, Y, Z   = 0, 1, 2
    PX, PY, PZ = 3, 4, 5
    D0, D1, D2 = 6, 7, 8
    DIM = 9

    def __init__(self):
        # Process / measurement noise — firmware defaults
        self.proc_noise_acc_xy   = 0.5
        self.proc_noise_acc_z    = 1.0
        self.proc_noise_vel      = 0.0
        self.proc_noise_pos      = 0.0
        self.proc_noise_att      = 0.0
        self.meas_noise_gyro_rp  = 0.1   # rad/s
        self.meas_noise_gyro_yaw = 0.1   # rad/s
        self.tof_std_dev         = 0.05  # m  (kc_tof_stdDev)
        self.flow_std_dev        = 2.0   # px (kc_flow_stdDev)

        self.S = np.zeros(9)             # states
        self.P = np.zeros((9, 9))        # covariance
        self.q = np.array([1., 0., 0., 0.])   # quaternion [w,x,y,z]
        self.R = np.eye(3)               # rotation matrix body→world
        self.initialized = False
        self._zacc_filt = 0.0
        self._zacc_init = False

    # ── Initialisation ────────────────────────────────────────────────────────
    def init(self, q_true):
        """Bootstrap from true attitude quaternion (mirrors ekfInit)."""
        self.S[:] = 0.0
        qw, qx, qy, qz = q_true
        self.q = np.array([qw, qx, qy, qz], dtype=float)
        self._rebuild_R()
        P = np.zeros((9, 9))
        P[self.X,  self.X]  = 100.0**2   # stdDevInitialPosition_xy = 100 m
        P[self.Y,  self.Y]  = 100.0**2
        P[self.Z,  self.Z]  = 1.0**2     # stdDevInitialPosition_z  = 1 m
        P[self.PX, self.PX] = 0.01**2    # stdDevInitialVelocity    = 0.01 m/s
        P[self.PY, self.PY] = 0.01**2
        P[self.PZ, self.PZ] = 0.01**2
        P[self.D0, self.D0] = 0.01**2    # stdDevInitialAttitude    = 0.01 rad
        P[self.D1, self.D1] = 0.01**2
        P[self.D2, self.D2] = 0.01**2
        self.P = P
        self.initialized = True

    def _rebuild_R(self):
        """Rebuild body→world rotation matrix from quaternion."""
        qw, qx, qy, qz = self.q
        self.R[0,0] = qw*qw+qx*qx-qy*qy-qz*qz
        self.R[0,1] = 2*(qx*qy-qw*qz)
        self.R[0,2] = 2*(qx*qz+qw*qy)
        self.R[1,0] = 2*(qx*qy+qw*qz)
        self.R[1,1] = qw*qw-qx*qx+qy*qy-qz*qz
        self.R[1,2] = 2*(qy*qz-qw*qx)
        self.R[2,0] = 2*(qx*qz-qw*qy)
        self.R[2,1] = 2*(qy*qz+qw*qx)
        self.R[2,2] = qw*qw-qx*qx-qy*qy+qz*qz

    def _symmetrize_bound(self):
        """Symmetrize and bound covariance (mirrors kalman_core.c)."""
        for i in range(9):
            for j in range(i, 9):
                p = 0.5*(self.P[i,j]+self.P[j,i])
                if math.isnan(p) or p > 100.0:   p = 100.0
                elif i == j and p < 1e-6:         p = 1e-6
                self.P[i,j] = self.P[j,i] = p

    # ── Scalar measurement update (mirrors ekfScalarUpdate) ───────────────────
    def _scalar_update(self, h, error, std):
        PHT = self.P @ h
        HPHR = std*std + float(h @ PHT)
        if HPHR < 1e-10:
            return
        K = PHT / HPHR
        self.S += K * error
        # Joseph-form covariance update (symmetric, bounded)
        for i in range(9):
            for j in range(i, 9):
                p = 0.5*(self.P[i,j]+self.P[j,i]) - K[i]*PHT[j] - K[j]*PHT[i] + K[i]*K[j]*HPHR
                if math.isnan(p) or p > 100.0:  p = 100.0
                elif i == j and p < 1e-6:        p = 1e-6
                self.P[i,j] = self.P[j,i] = p

    # ── Prediction (mirrors ekfPredict) ───────────────────────────────────────
    def predict(self, dt, gx_dps, gy_dps, gz_dps, zacc_ms2, quad_is_flying):
        """Propagate state and covariance with IMU data."""
        if dt <= 0 or dt > 0.1:
            return
        G = 9.81
        DEG2RAD = math.pi / 180.0
        gx = gx_dps * DEG2RAD
        gy = gy_dps * DEG2RAD
        gz = gz_dps * DEG2RAD
        d0 = gx*dt/2;  d1 = gy*dt/2;  d2 = gz*dt/2

        Rv = self.R
        px, py, pz = self.S[self.PX], self.S[self.PY], self.S[self.PZ]

        # ── Build sparse A matrix (9×9 linearised dynamics) ──────────────────
        A = np.zeros((9, 9))
        for i in range(9): A[i,i] = 1.0

        # Pos ← body-vel (A[X:Z, PX:PZ] = R * dt)
        A[self.X, self.PX]=Rv[0,0]*dt; A[self.Y, self.PX]=Rv[1,0]*dt; A[self.Z, self.PX]=Rv[2,0]*dt
        A[self.X, self.PY]=Rv[0,1]*dt; A[self.Y, self.PY]=Rv[1,1]*dt; A[self.Z, self.PY]=Rv[2,1]*dt
        A[self.X, self.PZ]=Rv[0,2]*dt; A[self.Y, self.PZ]=Rv[1,2]*dt; A[self.Z, self.PZ]=Rv[2,2]*dt

        # Pos ← attitude error
        A[self.X, self.D0]=(py*Rv[0,2]-pz*Rv[0,1])*dt
        A[self.Y, self.D0]=(py*Rv[1,2]-pz*Rv[1,1])*dt
        A[self.Z, self.D0]=(py*Rv[2,2]-pz*Rv[2,1])*dt
        A[self.X, self.D1]=(-px*Rv[0,2]+pz*Rv[0,0])*dt
        A[self.Y, self.D1]=(-px*Rv[1,2]+pz*Rv[1,0])*dt
        A[self.Z, self.D1]=(-px*Rv[2,2]+pz*Rv[2,0])*dt
        A[self.X, self.D2]=(px*Rv[0,1]-py*Rv[0,0])*dt
        A[self.Y, self.D2]=(px*Rv[1,1]-py*Rv[1,0])*dt
        A[self.Z, self.D2]=(px*Rv[2,1]-py*Rv[2,0])*dt

        # Body-vel ← gyro coupling
        A[self.PY, self.PX]=-gz*dt;  A[self.PZ, self.PX]= gy*dt
        A[self.PX, self.PY]= gz*dt;  A[self.PZ, self.PY]=-gx*dt
        A[self.PX, self.PZ]=-gy*dt;  A[self.PY, self.PZ]= gx*dt

        # Body-vel ← gravity / attitude error
        A[self.PX, self.D1]= G*Rv[2,2]*dt;  A[self.PX, self.D2]=-G*Rv[2,1]*dt
        A[self.PY, self.D0]=-G*Rv[2,2]*dt;  A[self.PY, self.D2]= G*Rv[2,0]*dt
        A[self.PZ, self.D0]= G*Rv[2,1]*dt;  A[self.PZ, self.D1]=-G*Rv[2,0]*dt

        # Att-error rotation (2nd-order approx)
        A[self.D0, self.D0]= 1-d1*d1/2-d2*d2/2; A[self.D0, self.D1]= d2+d0*d1/2; A[self.D0, self.D2]=-d1+d0*d2/2
        A[self.D1, self.D0]=-d2+d0*d1/2;         A[self.D1, self.D1]= 1-d0*d0/2-d2*d2/2; A[self.D1, self.D2]= d0+d1*d2/2
        A[self.D2, self.D0]= d1+d0*d2/2;         A[self.D2, self.D1]=-d0+d1*d2/2; A[self.D2, self.D2]= 1-d0*d0/2-d1*d1/2

        # ── Covariance propagation P = A P Aᵀ ────────────────────────────────
        self.P = A @ self.P @ A.T

        # ── State propagation ─────────────────────────────────────────────────
        dt2 = dt*dt
        dx = px*dt;  dy = py*dt;  dz = pz*dt + zacc_ms2*dt2/2.0
        self.S[self.X] += Rv[0,0]*dx+Rv[0,1]*dy+Rv[0,2]*dz
        self.S[self.Y] += Rv[1,0]*dx+Rv[1,1]*dy+Rv[1,2]*dz
        self.S[self.Z] += Rv[2,0]*dx+Rv[2,1]*dy+Rv[2,2]*dz - G*dt2/2.0

        # Flying mode: body-z thrust only (mirrors kalmanCorePredict flying branch)
        self.S[self.PX] += dt*(       gz*py - gy*pz - G*Rv[2,0])
        self.S[self.PY] += dt*(-gz*px        + gx*pz - G*Rv[2,1])
        self.S[self.PZ] += dt*(zacc_ms2 + gy*px - gx*py - G*Rv[2,2])

        # ── Quaternion integration ────────────────────────────────────────────
        dtwx = gx*dt;  dtwy = gy*dt;  dtwz = gz*dt
        angle = math.sqrt(dtwx**2+dtwy**2+dtwz**2)
        if angle > 1e-10:
            ca = math.cos(angle/2);  sa = math.sin(angle/2)
            dq0=ca; dq1=sa*dtwx/angle; dq2=sa*dtwy/angle; dq3=sa*dtwz/angle
        else:
            dq0=1.0; dq1=dq2=dq3=0.0
        qw, qx, qy, qz = self.q
        tmpq = np.array([
            dq0*qw - dq1*qx - dq2*qy - dq3*qz,
            dq1*qw + dq0*qx + dq3*qy - dq2*qz,
            dq2*qw - dq3*qx + dq0*qy + dq1*qz,
            dq3*qw + dq2*qx - dq1*qy + dq0*qz,
        ])
        norm = math.sqrt(float(tmpq @ tmpq))
        if norm > 1e-10:
            self.q = tmpq / norm

    # ── Process noise (mirrors ekfAddProcessNoise) ────────────────────────────
    def add_process_noise(self, dt):
        n = self
        self.P[n.X,  n.X]  += (n.proc_noise_acc_xy*dt**2 + n.proc_noise_vel*dt + n.proc_noise_pos)**2
        self.P[n.Y,  n.Y]  += (n.proc_noise_acc_xy*dt**2 + n.proc_noise_vel*dt + n.proc_noise_pos)**2
        self.P[n.Z,  n.Z]  += (n.proc_noise_acc_z *dt**2 + n.proc_noise_vel*dt + n.proc_noise_pos)**2
        self.P[n.PX, n.PX] += (n.proc_noise_acc_xy*dt + n.proc_noise_vel)**2
        self.P[n.PY, n.PY] += (n.proc_noise_acc_xy*dt + n.proc_noise_vel)**2
        self.P[n.PZ, n.PZ] += (n.proc_noise_acc_z *dt + n.proc_noise_vel)**2
        self.P[n.D0, n.D0] += (n.meas_noise_gyro_rp *dt + n.proc_noise_att)**2
        self.P[n.D1, n.D1] += (n.meas_noise_gyro_rp *dt + n.proc_noise_att)**2
        self.P[n.D2, n.D2] += (n.meas_noise_gyro_yaw*dt + n.proc_noise_att)**2
        self._symmetrize_bound()

    # ── ToF update (tilt-corrected, mirrors ekfUpdateTof) ─────────────────────
    def update_tof(self, z_meas):
        R22 = self.R[2,2]
        if abs(R22) < 0.1 or R22 <= 0:
            return
        predicted_dist = self.S[self.Z] / R22
        h = np.zeros(9)
        h[self.Z] = 1.0 / R22
        self._scalar_update(h, z_meas - predicted_dist, self.tof_std_dev)
        if self.S[self.Z] < 0.0:
            self.S[self.Z] = 0.0

    # ── Optical-flow update (mirrors ekfUpdateFlow) ───────────────────────────
    def update_flow(self, dpx, dpy, flow_dt, gx_dps, gy_dps, flow_std):
        DEG2RAD  = math.pi / 180.0
        NPIX     = 30.0
        THETAPIX = 4.2 * DEG2RAD
        OMEGA_F  = 1.25                         # omegaFactor
        omegax_b = gx_dps * DEG2RAD
        omegay_b = gy_dps * DEG2RAD
        dx_g = self.S[self.PX]
        dy_g = self.S[self.PY]
        z_g  = max(0.1, self.S[self.Z])
        R22  = self.R[2,2]

        scale = NPIX * flow_dt / THETAPIX

        # X pixel update
        pred_nx = scale * ((dx_g*R22/z_g) - OMEGA_F*omegay_b)
        hx = np.zeros(9)
        hx[self.Z]  = scale * (R22*dx_g / (-z_g*z_g))
        hx[self.PX] = scale * (R22 / z_g)
        self._scalar_update(hx, dpx - pred_nx, flow_std)

        # Y pixel update
        pred_ny = scale * ((dy_g*R22/z_g) + OMEGA_F*omegax_b)
        hy = np.zeros(9)
        hy[self.Z]  = scale * (R22*dy_g / (-z_g*z_g))
        hy[self.PY] = scale * (R22 / z_g)
        self._scalar_update(hy, dpy - pred_ny, flow_std)

    # ── Finalize (mirrors ekfFinalize) ────────────────────────────────────────
    def finalize(self):
        """Incorporate attitude error into quaternion, rebuild R, reset D states."""
        v0, v1, v2 = self.S[self.D0], self.S[self.D1], self.S[self.D2]
        if ((abs(v0)>0.1e-3 or abs(v1)>0.1e-3 or abs(v2)>0.1e-3) and
                abs(v0)<10 and abs(v1)<10 and abs(v2)<10):
            angle = math.sqrt(v0**2+v1**2+v2**2)
            ca = math.cos(angle/2);  sa = math.sin(angle/2)
            dq0=ca; dq1=sa*v0/angle; dq2=sa*v1/angle; dq3=sa*v2/angle
            qw, qx, qy, qz = self.q
            tmpq = np.array([
                dq0*qw-dq1*qx-dq2*qy-dq3*qz,
                dq1*qw+dq0*qx+dq3*qy-dq2*qz,
                dq2*qw-dq3*qx+dq0*qy+dq1*qz,
                dq3*qw+dq2*qx-dq1*qy+dq0*qz,
            ])
            norm = math.sqrt(float(tmpq @ tmpq))
            if norm > 1e-10:
                self.q = tmpq / norm
            # Rotate covariance (2nd-order approx)
            d0=v0/2; d1=v1/2; d2=v2/2
            Af = np.eye(9)
            Af[6:9,6:9] = np.array([
                [ 1-d1*d1/2-d2*d2/2,  d2+d0*d1/2,         -d1+d0*d2/2],
                [-d2+d0*d1/2,          1-d0*d0/2-d2*d2/2,   d0+d1*d2/2],
                [ d1+d0*d2/2,         -d0+d1*d2/2,          1-d0*d0/2-d1*d1/2],
            ])
            self.P = Af @ self.P @ Af.T
        self._rebuild_R()
        self.S[self.D0] = self.S[self.D1] = self.S[self.D2] = 0.0
        self._symmetrize_bound()

    # ── EKF supervisor (mirrors kalman_supervisor.c) ──────────────────────────
    def is_diverged(self):
        for i in (self.X, self.Y, self.Z):
            if abs(self.S[i]) > 100.0: return True
        for i in (self.PX, self.PY, self.PZ):
            if abs(self.S[i]) > 10.0:  return True
        return False

    # ── Convenience properties (world-frame outputs) ──────────────────────────
    @property
    def z(self):
        return self.S[self.Z]
    @property
    def vz(self):
        Rv = self.R;  S = self.S
        return Rv[2,0]*S[self.PX] + Rv[2,1]*S[self.PY] + Rv[2,2]*S[self.PZ]
    @property
    def x(self):  return self.S[self.X]
    @property
    def y(self):  return self.S[self.Y]
    @property
    def vx(self):
        Rv = self.R;  S = self.S
        return Rv[0,0]*S[self.PX] + Rv[0,1]*S[self.PY] + Rv[0,2]*S[self.PZ]
    @property
    def vy(self):
        Rv = self.R;  S = self.S
        return Rv[1,0]*S[self.PX] + Rv[1,1]*S[self.PY] + Rv[1,2]*S[self.PZ]

    def reset(self, z0=0.0):
        self.S[:] = 0.0;  self.S[self.Z] = z0
        self.P = np.eye(9) * 0.5
        self.initialized = False


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED DRONE STATE  (physics thread ↔ WebSocket thread)
# ═══════════════════════════════════════════════════════════════════════════════

class DroneState:
    def __init__(self):
        self.lock = threading.Lock()

        # ── Kinematics ─────────────────────────────────────────────────────────
        self.x  = 0.0;  self.y  = 0.0;  self.z  = 0.0   # position  m
        self.vx = 0.0;  self.vy = 0.0;  self.vz = 0.0   # velocity  m/s
        self.roll  = 0.0   # deg  (+ = right tilt)
        self.pitch = 0.0   # deg  (+ = nose up)
        self.yaw   = 0.0   # deg
        self.p = 0.0;  self.q = 0.0;  self.r = 0.0      # body rates deg/s

        # ── Commands from WebSocket ────────────────────────────────────────────
        self.ch1 = 1000;  self.ch2 = 1500
        self.ch3 = 1500;  self.ch4 = 1500;  self.ch5 = 2000

        # ── Flight modes ───────────────────────────────────────────────────────
        self.armed    = False
        self.althold  = False
        self.poshold  = False
        self.alt_sp   = 0.5    # m  altitude setpoint
        self.pos_sp_x = 0.0    # m
        self.pos_sp_y = 0.0    # m
        self.hover_thr_locked = lw_hover_thr  # captured at althold engage

        # ── Motor outputs (0-255 PWM duty) ────────────────────────────────────
        self.m1 = 0;  self.m2 = 0;  self.m3 = 0;  self.m4 = 0

        # ── PID error outputs (for telemetry) ─────────────────────────────────
        self.error_roll  = 0.0
        self.error_pitch = 0.0
        self.error_yaw   = 0.0

        # ── EKF outputs ───────────────────────────────────────────────────────
        self.ekf_z  = 0.0;  self.ekf_vz = 0.0
        self.ekf_x  = 0.0;  self.ekf_y  = 0.0
        self.ekf_vx = 0.0;  self.ekf_vy = 0.0
        self.alt_sp_mm = 0.0

        # ── Battery state ─────────────────────────────────────────────────────
        self.bat_pct     = 100.0   # %  state of charge
        self.bat_voltage = BAT_V_FULL  # V  terminal voltage
        self.bat_current = 0.0     # A  instantaneous draw
        self._bat_charge_used = 0.0  # mAh consumed

        # ── Connected WebSocket clients (set, thread-safe via asyncio) ─────────
        self.clients     = set()
        self.clients_lock = threading.Lock()

        # ── Timestamp ─────────────────────────────────────────────────────────
        self.t_ms = 0

        # ── Crash state ───────────────────────────────────────────────────────
        self.crashed         = False
        self.crash_reason    = ""
        self.reset_requested = False

    def telemetry_dict(self):
        """Build telemetry dict matching ESP32 format exactly."""
        z_mm = self.ekf_z * 1000.0
        return {
            "tel": 1,
            "t":   self.t_ms,
            "r":   round(self.roll,  2),
            "p":   round(self.pitch, 2),
            "y":   round(self.yaw,   2),
            "gx":  round(self.p, 1),
            "gy":  round(self.q, 1),
            "gz":  round(self.r, 1),
            "er":  round(self.error_roll,  3),
            "ep":  round(self.error_pitch, 3),
            "ey":  round(self.error_yaw,   3),
            "ch1": self.ch1,
            "ch5": self.ch5,
            "m1":  self.m1,
            "m2":  self.m2,
            "m3":  self.m3,
            "m4":  self.m4,
            # LiteWing fields — send both 'alt' and 'lw_z' for compatibility
            "alt":    round(z_mm, 0),
            "lw_z":   round(z_mm, 0),       # keyboard_server.py uses lw_z
            "altsp":  round(self.alt_sp_mm, 0),
            "vz":     round(self.ekf_vz, 3),
            "kx":     round(self.ekf_x, 3),
            "ky":     round(self.ekf_y, 3),
            "kvx":    round(self.ekf_vx, 3),
            "kvy":    round(self.ekf_vy, 3),
            "althold": 1 if self.althold else 0,
            "poshold": 1 if self.poshold else 0,
            "bat_pct": round(self.bat_pct, 1),
            "bat_v":   round(self.bat_voltage, 2),
            "bat_a":   round(self.bat_current, 2),
            "crashed":      1 if self.crashed else 0,
            "crash_reason": self.crash_reason,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICS LOOP  (runs in a dedicated thread at SIM_HZ)
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsLoop:
    def __init__(self, state: DroneState):
        self.s = state
        self.dt = 1.0 / SIM_HZ

        # ── Attitude PID controllers (exact firmware copy) ─────────────────────
        self.pid_roll_angle  = PID(Kp_roll_angle,  Ki_roll_angle,  Kd_roll_angle,  i_limit)
        self.pid_pitch_angle = PID(Kp_pitch_angle, Ki_pitch_angle, Kd_pitch_angle, i_limit)
        self.pid_roll_rate   = RatePID(Kp_roll_rate,   Ki_roll_rate,   Kd_roll_rate,   i_limit)
        self.pid_pitch_rate  = RatePID(Kp_pitch_rate,  Ki_pitch_rate,  Kd_pitch_rate,  i_limit)
        self.pid_yaw_rate    = RatePID(Kp_yaw,         Ki_yaw,         Kd_yaw,         i_limit)

        # ── Madgwick filter (attitude estimation — same beta as firmware) ─────────
        self.madgwick = MadgwickFilter(beta=0.03)

        # ── Outer-loop LP filter state ─────────────────────────────────────────
        self._roll_des_ol  = 0.0
        self._pitch_des_ol = 0.0

        # ── Altitude hold PIDs ─────────────────────────────────────────────────
        # Limits match firmware exactly:
        #   outer: vel_sp  constrained to ±0.2 m/s (firmware line 2208),  integral to ±0.5
        #   inner: thr_corr constrained to ±0.30,                          integral to ±0.5
        # Note: firmware also has an anti-windup check at ±0.4 (line 2206) — that is NOT the output limit
        self.pid_alt_pos = PID(lw_pidZ_kp,  lw_pidZ_ki,  0, limit=0.2,  ilimit=0.5)  # pos→vel
        self.pid_alt_vel = PID(lw_pidVZ_kp, lw_pidVZ_ki, 0, limit=0.30, ilimit=0.5)  # vel→thrust

        # ── Position hold PIDs ─────────────────────────────────────────────────
        self.pid_px  = PID(lw_pidX_kp,  lw_pidX_ki,  0, limit=lw_xy_vel_max)
        self.pid_py  = PID(lw_pidX_kp,  lw_pidX_ki,  0, limit=lw_xy_vel_max)
        self.pid_pvx = PID(lw_pidVX_kp, lw_pidVX_ki, lw_pidVX_kd, limit=lw_rp_limit)
        self.pid_pvy = PID(lw_pidVX_kp, lw_pidVX_ki, lw_pidVX_kd, limit=lw_rp_limit)

        # ── Kalman filters ─────────────────────────────────────────────────────
        self.kf9 = Kalman9D()   # 9-state EKF — mirrors firmware kalman_core.c

        # ── Internal counters ──────────────────────────────────────────────────
        self._t0  = time.time()
        self._tick = 0
        self._lw_thrust_corr = 0.0  # altitude-hold throttle correction

        # ── Proper flight dynamics state ────────────────────────────────────────
        # Motor angular speeds (rad/s) — lag behind PWM commands
        self._motor_omega = np.zeros(4)
        # True physics quaternion [w,x,y,z] — evolved by Euler's equations
        self._q_true = np.array([1.0, 0.0, 0.0, 0.0])
        # True body angular rates (rad/s)
        self._omega_b = np.zeros(3)

        self.running = True

    # ── Physics reset (called when reset_requested is set) ────────────────────
    def _reset_physics(self, s):
        """Restore drone to a level, stationary state at current XY position."""
        self._q_true      = np.array([1.0, 0.0, 0.0, 0.0])
        self._omega_b     = np.zeros(3)
        self._motor_omega = np.zeros(4)
        self._roll_des_ol  = 0.0
        self._pitch_des_ol = 0.0
        self.madgwick.reset()
        self.kf9.reset(z0=s.z)
        for pid in [self.pid_roll_angle, self.pid_pitch_angle,
                    self.pid_roll_rate,  self.pid_pitch_rate, self.pid_yaw_rate,
                    self.pid_alt_pos, self.pid_alt_vel,
                    self.pid_px, self.pid_py, self.pid_pvx, self.pid_pvy]:
            pid.reset()
        with s.lock:
            s.z  = 0.0;  s.vx = 0.0;  s.vy = 0.0;  s.vz = 0.0
            s.roll = 0.0;  s.pitch = 0.0
            s.p = 0.0;  s.q = 0.0;  s.r = 0.0
            s.crashed = False;  s.crash_reason = ""
            s.reset_requested = False
            s.armed = False
            s.ch1 = 1000   # cut throttle

    # ── Main physics tick ──────────────────────────────────────────────────────
    def tick(self):
        s  = self.s
        dt = self.dt

        # ── Handle reset / crash early-exit ────────────────────────────────────
        with s.lock:
            reset_req = s.reset_requested
            crashed   = s.crashed

        if reset_req:
            self._reset_physics(s)
            return

        if crashed:
            # Drone is crashed — freeze it: zero motors, hold position
            with s.lock:
                s.m1 = s.m2 = s.m3 = s.m4 = 0
                s.armed = False
                s.vx = 0.0;  s.vy = 0.0;  s.vz = 0.0
            return

        with s.lock:
            ch1 = s.ch1;  ch2 = s.ch2;  ch3 = s.ch3;  ch4 = s.ch4;  ch5 = s.ch5
            armed    = s.armed
            althold  = s.althold
            poshold  = s.poshold
            alt_sp   = s.alt_sp
            pos_sp_x = s.pos_sp_x
            pos_sp_y = s.pos_sp_y
            hthr     = s.hover_thr_locked
            # Current kinematics (read)
            z  = s.z;   vz = s.vz
            x  = s.x;   y  = s.y
            vx = s.vx;  vy = s.vy
            roll  = s.roll;   pitch = s.pitch;   yaw = s.yaw
            p_r   = s.p;      q_r   = s.q;       r_r = s.r

        # active: motors engage only when throttle pushed past 1700 (intentional takeoff)
        # OR when already airborne under althold (so it doesn't cut out mid-flight)
        active = armed and (ch1 > 1100 or (althold and z > 0.05))

        # ── 1. Desired states (getDesState equivalent) ──────────────────────────
        thro_des  = (ch1 - 1000.0) / 1000.0
        roll_des  = clamp((ch2 - 1500.0) / 500.0, -1, 1) * maxRoll
        pitch_des = clamp((ch3 - 1500.0) / 500.0, -1, 1) * maxPitch
        yaw_des   = clamp((ch4 - 1500.0) / 500.0, -1, 1) * maxYaw

        # ── 2. Altitude hold override (LiteWing) ────────────────────────────────
        if althold and z > 0.05:
            ez  = alt_sp - self.kf9.z
            # Back-calculation anti-windup on outer loop:
            # if vel_sp is saturated, undo integral accumulation for this tick.
            _old_pos_int = self.pid_alt_pos.integral
            vel_sp = self.pid_alt_pos.update(ez, dt)
            if self.pid_alt_pos.limit and abs(vel_sp) >= self.pid_alt_pos.limit - 1e-4:
                self.pid_alt_pos.integral = _old_pos_int
            evz = vel_sp - self.kf9.vz
            self._lw_thrust_corr = self.pid_alt_vel.update(evz, dt)
            thro_des = clamp(hthr + self._lw_thrust_corr, 0.0, 1.0)
        else:
            self._lw_thrust_corr = 0.0
            if not althold:
                self.pid_alt_pos.reset()
                self.pid_alt_vel.reset()

        # ── 3. Position hold (LiteWing) — toggled by poshold switch ─────────────
        # poshold ON:  position PID overrides roll/pitch, joystick ignored
        # poshold OFF: joystick controls roll/pitch directly (no change to roll_des/pitch_des)
        if poshold and z > 0.10:
            ex = pos_sp_x - self.kf9.x
            ey = pos_sp_y - self.kf9.y
            vx_sp = self.pid_px.update(ex, dt)
            vy_sp = self.pid_py.update(ey, dt)
            evx = vx_sp - self.kf9.vx
            evy = vy_sp - self.kf9.vy
            roll_raw  = self.pid_pvx.update(evx, dt)
            pitch_raw = self.pid_pvy.update(evy, dt)
            yaw_rad = math.radians(yaw)
            cy = math.cos(yaw_rad);  sy = math.sin(yaw_rad)
            pos_pitch_corr =  (roll_raw * cy) + (pitch_raw * sy)   # +ve pitch = nose down = +X force
            pos_roll_corr  = -(pitch_raw * cy) + (roll_raw * sy)
            roll_des  = clamp(pos_roll_corr  * lw_rp_limit, -lw_rp_limit, lw_rp_limit)
            pitch_des = clamp(pos_pitch_corr * lw_rp_limit, -lw_rp_limit, lw_rp_limit)
        else:
            if not poshold:
                self.pid_px.reset();  self.pid_py.reset()
                self.pid_pvx.reset(); self.pid_pvy.reset()

        # ── 4. Attitude PID — controlANGLE2 exact copy ─────────────────────────
        int_active = active

        # Outer loop: angle error → desired rate
        e_roll  = roll_des  - roll
        e_pitch = pitch_des - pitch
        roll_des_ol  = self.pid_roll_angle.update(e_roll,  dt, int_active) * KL
        pitch_des_ol = self.pid_pitch_angle.update(e_pitch, dt, int_active) * KL
        roll_des_ol  = clamp(roll_des_ol,  -240.0, 240.0)
        pitch_des_ol = clamp(pitch_des_ol, -240.0, 240.0)
        # LP filter
        self._roll_des_ol  = (1-B_loop_roll)  * self._roll_des_ol  + B_loop_roll  * roll_des_ol
        self._pitch_des_ol = (1-B_loop_pitch) * self._pitch_des_ol + B_loop_pitch * pitch_des_ol

        # Inner loop: rate error → RatePID (derivative-on-measurement = gyro rate)
        # Matches firmware: Kp*e + Ki*∫e - Kd*GyroRate
        e_roll_r  = self._roll_des_ol  - p_r
        e_pitch_r = self._pitch_des_ol - q_r
        e_yaw_r   = yaw_des - r_r          # firmware: yaw_des - GyroZ
        roll_PID  = 0.01 * self.pid_roll_rate.update(e_roll_r,  p_r, dt, int_active)
        pitch_PID = 0.01 * self.pid_pitch_rate.update(e_pitch_r, q_r, dt, int_active)
        yaw_PID   = 0.01 * self.pid_yaw_rate.update(e_yaw_r,   r_r, dt, int_active)

        # Reset integrators when throttle too low (firmware behaviour)
        if not int_active:
            for pid in [self.pid_roll_angle, self.pid_pitch_angle,
                        self.pid_roll_rate, self.pid_pitch_rate, self.pid_yaw_rate]:
                pid.integral = 0.0
            self._roll_des_ol = 0.0;  self._pitch_des_ol = 0.0

        # ── 5. Motor mixer (exact firmware formula) ─────────────────────────────
        ys = yaw_scale_val
        if active:
            m1s = clamp(thro_des - pitch_PID - roll_PID + yaw_PID * ys, 0, 1)
            m2s = clamp(thro_des + pitch_PID - roll_PID - yaw_PID * ys, 0, 1)
            m3s = clamp(thro_des + pitch_PID + roll_PID + yaw_PID * ys, 0, 1)
            m4s = clamp(thro_des - pitch_PID + roll_PID - yaw_PID * ys, 0, 1)
        else:
            m1s = m2s = m3s = m4s = 0.0

        m1_pwm = int(DUTY_IDLE + m1s * (DUTY_MAX - DUTY_IDLE))
        m2_pwm = int(DUTY_IDLE + m2s * (DUTY_MAX - DUTY_IDLE))
        m3_pwm = int(DUTY_IDLE + m3s * (DUTY_MAX - DUTY_IDLE))
        m4_pwm = int(DUTY_IDLE + m4s * (DUTY_MAX - DUTY_IDLE))

        # Zero motors when disarmed OR throttle not yet past takeoff threshold
        if not active:
            m1_pwm = m2_pwm = m3_pwm = m4_pwm = 0

        # ── 6. Motor dynamics — first-order lag, thrust ∝ ω² ──────────────────
        pwm_cmd = np.array([m1_pwm, m2_pwm, m3_pwm, m4_pwm], dtype=float)
        omega_cmd = pwm_cmd / DUTY_MAX * OMEGA_MAX
        # First-order lag: dω/dt = (ω_cmd − ω) / τ
        self._motor_omega += (omega_cmd - self._motor_omega) / TAU_MOTOR * dt
        self._motor_omega  = np.maximum(self._motor_omega, 0.0)
        omega = self._motor_omega   # shorthand

        # ── 6b. Battery model ──────────────────────────────────────────────────
        # Current draw proportional to total motor power (ω³ ∝ power for fixed pitch)
        throttle_frac = float(np.sum(omega)) / (4.0 * OMEGA_MAX)
        bat_current   = throttle_frac * BAT_MAX_CURRENT
        # Coulombs consumed this tick → mAh
        s._bat_charge_used += bat_current * dt / 3.6   # A·s → mAh
        soc = max(0.0, 1.0 - s._bat_charge_used / BAT_CAPACITY_MAH)
        # Open-circuit voltage: linear discharge curve
        v_oc = BAT_V_EMPTY + (BAT_V_FULL - BAT_V_EMPTY) * soc
        # Terminal voltage drops under load (internal resistance)
        v_term = max(BAT_V_EMPTY, v_oc - bat_current * BAT_R_INT)
        # Thrust scales as (V_term / V_full)² — motor speed ∝ voltage
        v_factor = (v_term / BAT_V_FULL) ** 2
        with s.lock:
            s.bat_pct     = round(soc * 100.0, 1)
            s.bat_voltage = round(v_term, 3)
            s.bat_current = round(bat_current, 2)
            # ── Low-battery warnings and critical cutoff ───────────────────────
            if soc <= 0.05 and s.armed and not getattr(s, '_bat_critical_warned', False):
                print("[BAT] CRITICAL (<5%) — disarming to prevent crash")
                s._bat_critical_warned = True
                s.armed   = False
                s.althold = False
                s.poshold = False
                s.ch1     = 1000
            elif soc <= 0.15 and not getattr(s, '_bat_low_warned', False):
                print("[BAT] WARNING: battery low (<15%)")
                s._bat_low_warned = True

        # Individual motor thrusts: F = K_F * ω²  (scaled by battery voltage)
        F_motors = K_F * omega**2 * v_factor

        # Ground effect — exponential Cheeseman-Bennett approximation
        # More thrust when close to ground; negligible above ~5×R_ROTOR
        if z > 1e-3:
            k_ge = 1.0 + GE_COEFF * math.exp(-z / (GE_DECAY * R_ROTOR))
        else:
            k_ge = 1.0 + GE_COEFF
        F_motors *= k_ge
        F_total   = float(np.sum(F_motors))

        # Body torques
        # Roll:  (m3+m4 − m1−m2) × ARM
        # Pitch: (m2+m3 − m1−m4) × ARM
        # Yaw:   reactive torques from spinning rotors (CCW+ vs CW−)
        tau_roll  = (F_motors[2]+F_motors[3] - F_motors[0]-F_motors[1]) * ARM
        tau_pitch = (F_motors[1]+F_motors[2] - F_motors[0]-F_motors[3]) * ARM
        Q_motors  = K_Q * omega**2 * MOTOR_SPIN          # reactive torques
        tau_yaw   = float(np.sum(Q_motors))
        tau_body  = np.array([tau_roll, tau_pitch, tau_yaw])

        # ── 7. Full rigid-body dynamics ────────────────────────────────────────
        # 7a. World-frame forces using exact rotation from true quaternion
        R = quat_to_R(self._q_true)                      # body → world
        F_thrust_world = R @ np.array([0.0, 0.0, F_total])  # thrust in world frame
        v_world = np.array([vx, vy, vz])
        # Aerodynamic drag (world frame, separate XY and Z coefficients)
        F_drag = np.array([-DRAG_XY*vx, -DRAG_XY*vy, -DRAG_Z*vz])
        F_world = F_thrust_world + F_drag + np.array([0.0, 0.0, -MASS*GRAVITY])
        a_world = F_world / MASS

        # 7b. Angular dynamics — Euler's equation with gyroscopic coupling
        #     I·α = τ_body − ω×(I·ω) + τ_gyro
        I_diag = np.array([I_XX, I_YY, I_ZZ])
        ob = self._omega_b                               # body rates (rad/s)
        gyro_coupling = np.cross(ob, I_diag * ob)       # ω × (I·ω)

        # Gyroscopic effect from spinning rotors: τ_g = J·Ω_net·(ω_body × ẑ_body)
        Omega_net = float(np.sum(omega * MOTOR_SPIN))   # net rotor angular momentum
        tau_gyro  = J_ROTOR * Omega_net * np.cross(ob, np.array([0.0, 0.0, 1.0]))

        alpha_body = (tau_body - gyro_coupling + tau_gyro) / I_diag

        # Integrate body rates
        new_ob = ob + alpha_body * dt

        # 7c. Quaternion integration: q̇ = ½ q ⊗ [0, ω]
        w, qx, qy, qz = self._q_true
        p_b, q_b, r_b = new_ob
        qdot = 0.5 * np.array([
            -qx*p_b - qy*q_b - qz*r_b,
             w *p_b + qy*r_b - qz*q_b,
             w *q_b - qx*r_b + qz*p_b,
             w *r_b + qx*q_b - qy*p_b,
        ])
        new_q = self._q_true + qdot * dt
        new_q /= np.linalg.norm(new_q)
        self._q_true  = new_q
        self._omega_b = new_ob

        # True Euler angles from physics quaternion
        true_roll, true_pitch, true_yaw = quat_euler_deg(new_q)

        # 7d. Translational integration
        new_vx     = vx + a_world[0] * dt
        new_vy     = vy + a_world[1] * dt
        new_vz_raw = vz + a_world[2] * dt          # pre-clamp — used for crash detection
        new_x  = x  + vx * dt
        new_y  = y  + vy * dt
        new_z  = max(0.0, z + vz * dt)
        new_vz = 0.0 if (new_z == 0.0 and new_vz_raw < 0) else new_vz_raw

        # ── Ground-contact crash / tip-over detection ──────────────────────────
        new_crashed = False;  new_crash_reason = ""
        if new_z == 0.0:
            # Hard landing: struck ground at > |CRASH_VZ| m/s
            if z > 0.02 and new_vz_raw < CRASH_VZ:
                new_crashed = True
                new_crash_reason = f"hard landing (vz={new_vz_raw:.1f} m/s)"
                print(f"[CRASH] {new_crash_reason}")
            # Tip-over: arm/prop contacts floor when tilted past threshold
            elif abs(true_roll) > TIPOVER_ANGLE or abs(true_pitch) > TIPOVER_ANGLE:
                new_crashed = True
                new_crash_reason = (f"tip-over (roll={true_roll:.0f}°,"
                                    f" pitch={true_pitch:.0f}°)")
                print(f"[CRASH] {new_crash_reason}")
        if new_crashed:
            new_vx = new_vy = new_vz = 0.0

        # ── 8. Sensor simulation ───────────────────────────────────────────────
        # ToF: measures z when z > 0.03 m
        tof_reading = max(0.0, new_z + random.gauss(0, TOF_NOISE)) if new_z > 0.03 else 0.0

        # ── 9. Madgwick attitude estimation — fed from true physics state ───────
        # Noisy gyro readings (deg/s) from true body rates
        gx_n = math.degrees(new_ob[0]) + random.gauss(0, GYRO_NOISE)
        gy_n = math.degrees(new_ob[1]) + random.gauss(0, GYRO_NOISE)
        gz_n = math.degrees(new_ob[2]) + random.gauss(0, GYRO_NOISE)
        # Accelerometer: measures specific force = (F_thrust + F_drag) / MASS
        # a_world = (F_thrust + F_drag - mg) / MASS  →  add back gravity to get specific force
        spec_force_world = a_world + np.array([0.0, 0.0, GRAVITY])   # m/s²
        spec_force_body  = R.T @ spec_force_world                     # body frame, m/s²
        ax_b = float(spec_force_body[0]) / GRAVITY + random.gauss(0, 0.02)
        ay_b = float(spec_force_body[1]) / GRAVITY + random.gauss(0, 0.02)
        az_b = float(spec_force_body[2]) / GRAVITY + random.gauss(0, 0.02)
        self.madgwick.update(gx_n, gy_n, gz_n, ax_b, ay_b, az_b, dt)
        mw_roll, mw_pitch, mw_yaw = self.madgwick.euler_deg()

        # ── 10. 9-state EKF update (mirrors litewingAltitudeHold EKF sequence) ──
        # zacc: body-z specific force in m/s² INCLUDING gravity reaction
        # Firmware: zacc = (AccZ_g + 1.0) * 9.81  where AccZ_g is gravity-removed g's
        # Sim:      az_b = a_body[2] ≈ 1.0 when level/stationary → zacc = az_b * 9.81
        # Apply same heavy LP (B=0.05) the firmware uses to reject vibration spikes.
        zacc_raw = az_b * 9.81
        if not self.kf9._zacc_init:
            self.kf9._zacc_filt = zacc_raw;  self.kf9._zacc_init = True
        self.kf9._zacc_filt = 0.95 * self.kf9._zacc_filt + 0.05 * zacc_raw
        zacc_ms2 = float(np.clip(self.kf9._zacc_filt, -19.62, 19.62))

        # Bootstrap EKF from Madgwick quaternion on first call (mirrors ekfInit)
        if not self.kf9.initialized:
            mw_q = self.madgwick.q                      # [w, x, y, z]
            self.kf9.init(mw_q)
            self.kf9.S[Kalman9D.Z] = max(0.0, new_z)   # seed altitude from physics

        quad_is_flying = active

        # EKF sequence: predict → process noise → ToF update → flow update → finalize
        self.kf9.predict(dt, gx_n, gy_n, gz_n, zacc_ms2, quad_is_flying)
        self.kf9.add_process_noise(dt)

        if new_z > 0.03:
            self.kf9.update_tof(tof_reading)

        # Optical flow: generate raw pixel counts (translational + noise).
        # The 9-state EKF handles rotational compensation internally via gyro input.
        if new_z > 0.05:
            NPIX     = 30.0
            THETAPIX = 4.2 * math.pi / 180.0
            z_scale  = max(0.01, new_z)
            dpx = (new_vx / z_scale + random.gauss(0, FLOW_NOISE)) * NPIX * dt / THETAPIX
            dpy = (new_vy / z_scale + random.gauss(0, FLOW_NOISE)) * NPIX * dt / THETAPIX
            self.kf9.update_flow(dpx, dpy, dt, gx_n, gy_n, lw_flow_std)

        self.kf9.finalize()

        # Supervisor: reinit if EKF diverges (mirrors kalman_supervisor.c)
        if self.kf9.is_diverged():
            mw_q = self.madgwick.q
            self.kf9.init(mw_q)
            self.kf9.S[Kalman9D.Z] = max(0.0, new_z)

        # ── 11. Write back to shared state ─────────────────────────────────────
        t_ms = int((time.time() - self._t0) * 1000)

        with s.lock:
            s.x  = new_x;  s.y  = new_y;  s.z  = new_z
            s.vx = new_vx; s.vy = new_vy; s.vz = new_vz
            # Madgwick estimated attitude — fed to PID (same as firmware)
            s.roll  = mw_roll
            s.pitch = mw_pitch
            s.yaw   = mw_yaw % 360.0
            # Noisy gyro rates fed to Madgwick & inner-rate PID
            s.p = gx_n;  s.q = gy_n;  s.r = gz_n
            s.m1 = m1_pwm;  s.m2 = m2_pwm
            s.m3 = m3_pwm;  s.m4 = m4_pwm
            s.error_roll  = e_roll
            s.error_pitch = e_pitch
            s.error_yaw   = e_yaw_r
            s.ekf_z  = self.kf9.z;    s.ekf_vz = self.kf9.vz
            s.ekf_x  = self.kf9.x;    s.ekf_y  = self.kf9.y
            s.ekf_vx = self.kf9.vx;   s.ekf_vy = self.kf9.vy
            s.alt_sp_mm = s.alt_sp * 1000.0
            s.t_ms = t_ms
            if new_crashed:
                s.crashed      = True
                s.crash_reason = new_crash_reason
                s.armed        = False

    # ── Thread run ─────────────────────────────────────────────────────────────
    def run(self):
        interval = self.dt
        next_tick = time.perf_counter()
        while self.running:
            self.tick()
            self._tick += 1
            # Debug print every 2 seconds (every 2*SIM_HZ ticks)
            if self._tick % (2 * SIM_HZ) == 0:
                s = self.s
                with s.lock:
                    tr, tp, ty = quat_euler_deg(self._q_true)
                    print(f"[DBG] ch2={s.ch2} ch3={s.ch3} "
                          f"mw=({s.roll:+.1f}°,{s.pitch:+.1f}°) "
                          f"true=({tr:+.1f}°,{tp:+.1f}°) "
                          f"x={s.x:.3f}m y={s.y:.3f}m z={s.z:.3f}m "
                          f"ω=({math.degrees(self._omega_b[0]):+.1f},{math.degrees(self._omega_b[1]):+.1f}) deg/s "
                          f"motors_ω={self._motor_omega.astype(int).tolist()}")
            next_tick += interval
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA SIMULATOR  (generates synthetic JPEG of what drone sees)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_camera_frame(state: DroneState) -> bytes:
    """Generate a synthetic forward-facing camera JPEG from drone state."""
    if not PIL_OK:
        return b""

    with state.lock:
        z     = state.z
        roll  = state.roll
        pitch = state.pitch
        x     = state.x;   y = state.y
        armed = state.armed
        althold = state.althold
        poshold = state.poshold

    img = Image.new("RGB", (CAM_W, CAM_H), (135, 160, 200))  # sky blue
    draw = ImageDraw.Draw(img)

    # ── Horizon ────────────────────────────────────────────────────────────────
    horizon_y = int(CAM_H / 2 - pitch * 4)   # pitch tilts horizon
    horizon_y = clamp(horizon_y, 0, CAM_H)

    # Ground (below horizon): checkerboard floor texture
    cell = max(4, int(60 / max(0.1, z)))      # cells appear smaller with altitude
    for row in range(horizon_y, CAM_H, cell):
        for col in range(0, CAM_W, cell):
            ix = int(col / cell + x * 2)
            iy = int(row / cell + y * 2)
            if (ix + iy) % 2 == 0:
                draw.rectangle([col, row, col+cell, row+cell], fill=(180, 150, 100))
            else:
                draw.rectangle([col, row, col+cell, row+cell], fill=(160, 130, 80))

    # Sky (above horizon): gradient
    for row in range(0, horizon_y):
        shade = int(200 - row * 0.5)
        draw.rectangle([0, row, CAM_W, row+1], fill=(shade, shade+20, 220))

    # Horizon line with roll tilt
    roll_px = int(roll * 3)
    draw.line([(0, horizon_y + roll_px), (CAM_W, horizon_y - roll_px)],
              fill=(255, 255, 255), width=2)

    # ── HUD overlays ───────────────────────────────────────────────────────────
    # Altitude tape (right side)
    draw.rectangle([CAM_W-40, 10, CAM_W-5, CAM_H-10], fill=(0,0,0,120))
    alt_bar_h = int(clamp(z / 3.0, 0, 1) * (CAM_H - 20))
    draw.rectangle([CAM_W-38, CAM_H-10-alt_bar_h, CAM_W-7, CAM_H-10],
                   fill=(0, 220, 80))

    # Center crosshair
    cx, cy = CAM_W // 2, CAM_H // 2
    draw.line([(cx-15, cy), (cx+15, cy)], fill=(255,255,0), width=2)
    draw.line([(cx, cy-15), (cx, cy+15)], fill=(255,255,0), width=2)

    # Text info
    lines = [
        f"ALT  {z:.2f}m",
        f"X {x:+.2f} Y {y:+.2f}",
        f"R{roll:+.1f} P{pitch:+.1f}",
        f"{'ARMED' if armed else 'DISARMED'}",
        f"{'ALTHOLD ' if althold else ''}{'POSHOLD' if poshold else ''}",
    ]
    ty = 8
    for line in lines:
        # Shadow
        draw.text((9, ty+1), line, fill=(0,0,0))
        draw.text((8, ty),   line, fill=(0,255,0))
        ty += 16

    # Obstacle simulation: red box if close to origin (for AI vision testing)
    if abs(x) < 0.3 and abs(y) < 0.3 and z > 0.2:
        draw.rectangle([cx-20, horizon_y-30, cx+20, horizon_y+10],
                       outline=(255,0,0), width=3)
        draw.text((cx-18, horizon_y-28), "OBSTACLE", fill=(255,0,0))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP CAMERA SERVER  (serves /capture endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

_sim_state_ref: DroneState = None   # set in main()

class CameraHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/capture", "/"):
            frame = generate_camera_frame(_sim_state_ref) if PIL_OK else b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(frame)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, fmt, *args):
        pass   # suppress HTTP logs


def run_camera_server(port: int):
    with socketserver.TCPServer(("", port), CameraHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"[SIM] Camera server  http://localhost:{port}/capture")
        httpd.serve_forever()


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET SERVER  (WebSocket on port 81 — identical to ESP32)
# ═══════════════════════════════════════════════════════════════════════════════

_ws_state: DroneState = None   # set in main()

async def handle_client(websocket):
    """Handle one browser WebSocket connection."""
    s = _ws_state
    addr = websocket.remote_address
    print(f"[WS]  Client connected: {addr}")

    with s.clients_lock:
        s.clients.add(websocket)

    try:
        async for raw in websocket:
            try:
                doc = json.loads(raw)
            except Exception:
                continue

            with s.lock:
                # ── Channel PWM ────────────────────────────────────────────────
                if "ch1" in doc: s.ch1 = clamp(int(doc["ch1"]), 1000, 2000)
                if "ch2" in doc: s.ch2 = clamp(int(doc["ch2"]), 1000, 2000)
                if "ch3" in doc: s.ch3 = clamp(int(doc["ch3"]), 1000, 2000)
                if "ch4" in doc: s.ch4 = clamp(int(doc["ch4"]), 1000, 2000)
                if "ch5" in doc:
                    s.ch5 = clamp(int(doc["ch5"]), 1000, 2000)
                    s.armed = (s.ch5 < 1500) and (s.ch1 < 1050 or s.armed)

                # ── Arm / disarm implicit from ch5 ─────────────────────────────
                s.armed = (s.ch5 < 1500)
                if not s.armed:
                    s.ch1 = 1000    # cut throttle on disarm

                # ── Altitude hold ──────────────────────────────────────────────
                if "althold" in doc:
                    new_ah = bool(int(doc["althold"]))
                    if new_ah and not s.althold and s.z > 0.05:
                        # Capture hover throttle and current altitude at engagement
                        s.hover_thr_locked = clamp((s.ch1 - 1000.0) / 1000.0, 0.2, 0.9)
                        s.alt_sp = s.ekf_z   # hold wherever the drone currently is
                    if not new_ah:
                        s.hover_thr_locked = lw_hover_thr
                    s.althold = new_ah

                # ── Position hold ──────────────────────────────────────────────
                if "poshold" in doc:
                    new_ph = bool(int(doc["poshold"]))
                    if new_ph and not s.poshold:
                        s.pos_sp_x = s.ekf_x
                        s.pos_sp_y = s.ekf_y
                    s.poshold = new_ph

                # ── Altitude setpoint ── HTML sends "altset", accept both ──────
                _alt_val = doc.get("altset") or doc.get("altitude")
                if _alt_val is not None:
                    s.alt_sp = clamp(float(_alt_val), 0.1, 3.0)

                # ── Reset XY position ──────────────────────────────────────────
                if doc.get("resetpos"):
                    s.pos_sp_x = s.ekf_x
                    s.pos_sp_y = s.ekf_y

                # ── Crash reset — restores drone to level hover-ready state ────
                if doc.get("resetcrash"):
                    s.crashed         = False
                    s.crash_reason    = ""
                    s.reset_requested = True

    except Exception as e:
        print(f"[WS]  Client {addr} error: {e}")
    finally:
        with s.clients_lock:
            s.clients.discard(websocket)
        print(f"[WS]  Client disconnected: {addr}")


async def telemetry_broadcaster(state: DroneState):
    """Broadcast telemetry to all connected clients at TEL_HZ."""
    interval = 1.0 / TEL_HZ
    while True:
        await asyncio.sleep(interval)
        with state.clients_lock:
            clients = list(state.clients)
        if not clients:
            continue
        tel = json.dumps(state.telemetry_dict())
        dead = []
        for ws in clients:
            try:
                await ws.send(tel)
            except Exception:
                dead.append(ws)
        if dead:
            with state.clients_lock:
                for ws in dead:
                    state.clients.discard(ws)


async def ws_main(state: DroneState):
    global _ws_state
    _ws_state = state
    print(f"[SIM] WebSocket server ws://localhost:{WS_PORT}")
    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        await telemetry_broadcaster(state)


# ═══════════════════════════════════════════════════════════════════════════════
#  REAL-TIME VISUALISER  (matplotlib — optional)
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualiser(state: DroneState):
    if not MPL_OK:
        return

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Maddy Drone Simulator", fontsize=13)

    ax_top, ax_alt, ax_att = axes

    t_hist = [];  z_hist = [];  sp_hist = []
    roll_hist = [];  pitch_hist = []
    x_hist = [];  y_hist = []
    T0 = time.time()

    while plt.fignum_exists(fig.number):
        try:
            with state.lock:
                x = state.x;   y = state.y;   z = state.ekf_z
                roll = state.roll;  pitch = state.pitch
                sp_x = state.pos_sp_x;  sp_y = state.pos_sp_y
                alt_sp = state.alt_sp if state.althold else None
                armed  = state.armed
                bat_pct = state.bat_pct
                bat_v   = state.bat_voltage

            t = time.time() - T0
            t_hist.append(t);  z_hist.append(z)
            sp_hist.append(alt_sp if alt_sp else z)
            roll_hist.append(roll);  pitch_hist.append(pitch)
            x_hist.append(x);  y_hist.append(y)

            # Keep only last 30 s
            while t_hist and t_hist[-1] - t_hist[0] > 30:
                t_hist.pop(0);  z_hist.pop(0);  sp_hist.pop(0)
                roll_hist.pop(0);  pitch_hist.pop(0)
                x_hist.pop(0);  y_hist.pop(0)

            # ── Top-down XY — autoscale around drone position ─────────────────
            ax_top.cla()
            ax_top.set_title("Top-down XY position")
            ax_top.set_aspect("equal")
            ax_top.grid(True, alpha=0.3)
            ax_top.set_xlabel("X (m)");  ax_top.set_ylabel("Y (m)")
            pad = max(0.5, max(abs(x), abs(y)) + 0.3)   # auto zoom
            ax_top.set_xlim(-pad, pad);  ax_top.set_ylim(-pad, pad)
            # Draw trail
            if len(x_hist) > 1:
                ax_top.plot(x_hist, y_hist, 'r-', alpha=0.3, linewidth=1)
            ax_top.plot(x, y, 'ro', markersize=14, label="drone")
            ax_top.plot(sp_x, sp_y, 'b+', markersize=14, markeredgewidth=2,
                        label="pos setpoint")
            # Draw drone heading arrow (yaw not tracked here, use pitch as proxy)
            ax_top.annotate("", xy=(x + 0.05*math.sin(math.radians(pitch)),
                                     y + 0.05*math.cos(math.radians(pitch))),
                            xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color="orange", lw=2))
            status = "ARMED" if armed else "DISARMED"
            bat_color = "green" if bat_pct > 30 else ("orange" if bat_pct > 15 else "red")
            ax_top.set_title(f"Top-down XY  [{status}]  z={z:.2f}m  "
                             f"BAT:{bat_pct:.0f}% {bat_v:.2f}V", color="k")
            ax_top.title.set_color(bat_color if bat_pct < 30 else "k")
            ax_top.legend(loc="upper right", fontsize=8)

            # ── Altitude ──────────────────────────────────────────────────────
            ax_alt.cla()
            ax_alt.set_title("Altitude (m)")
            ax_alt.set_ylim(0, max(3.5, max(z_hist) + 0.3) if z_hist else 3.5)
            ax_alt.set_xlabel("Time (s)");  ax_alt.set_ylabel("m")
            ax_alt.plot(t_hist, z_hist, 'g', label="EKF z")
            ax_alt.plot(t_hist, sp_hist, 'b--', alpha=0.5, label="setpoint")
            ax_alt.legend(fontsize=8);  ax_alt.grid(True, alpha=0.3)

            # ── Attitude ──────────────────────────────────────────────────────
            ax_att.cla()
            att_max = max(15, max((abs(v) for v in roll_hist+pitch_hist), default=5) + 2)
            ax_att.set_title("Attitude (°)")
            ax_att.set_ylim(-att_max, att_max)
            ax_att.set_xlabel("Time (s)");  ax_att.set_ylabel("deg")
            ax_att.plot(t_hist, roll_hist,  'r', label="roll")
            ax_att.plot(t_hist, pitch_hist, 'b', label="pitch")
            ax_att.axhline(0, color='k', linewidth=0.5)
            ax_att.legend(fontsize=8);  ax_att.grid(True, alpha=0.3)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)
        except Exception:
            break


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSOLE STATUS PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def run_console_printer(state: DroneState):
    """Print compact status every 2 seconds."""
    while True:
        time.sleep(2.0)
        with state.lock:
            z    = state.z;    vz  = state.vz
            roll = state.roll; pitch = state.pitch; yaw = state.yaw
            ch1  = state.ch1;  armed = state.armed
            ah   = state.althold;  ph = state.poshold
            m1   = state.m1;   m2  = state.m2;  m3 = state.m3;  m4 = state.m4
            ekf_z = state.ekf_z
            n_clients = len(state.clients)
        mode = []
        if armed:  mode.append("ARMED")
        if ah:     mode.append("ALTHOLD")
        if ph:     mode.append("POSHOLD")
        mode_str = "|".join(mode) if mode else "DISARMED"
        print(f"[SIM] z={z:.2f}m ekf={ekf_z:.2f}m vz={vz:+.2f}  "
              f"R={roll:+.1f}° P={pitch:+.1f}° Y={yaw:.0f}°  "
              f"ch1={ch1}  motors={m1}/{m2}/{m3}/{m4}  "
              f"{mode_str}  clients={n_clients}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTOMATIC CSV DATA LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "t_s",          # elapsed time from arm event (seconds)
    "wall_time",    # absolute wall-clock timestamp
    "armed",        # 0/1
    "althold",      # 0/1
    "poshold",      # 0/1
    # Position
    "x_m", "y_m", "z_m",       # true physics position
    "ekf_z_m",                  # Kalman-filtered altitude
    "ekf_x_m", "ekf_y_m",      # Kalman-filtered XY
    # Velocity
    "vx_ms", "vy_ms", "vz_ms",
    # Attitude (degrees)
    "roll_deg", "pitch_deg", "yaw_deg",
    # Angular rates (deg/s)
    "gx_dps", "gy_dps", "gz_dps",
    # Setpoints
    "alt_sp_m", "pos_sp_x_m", "pos_sp_y_m",
    # PID error signals
    "roll_err_deg", "pitch_err_deg",
    # Channel inputs
    "ch1", "ch2", "ch3", "ch4", "ch5",
    # Motor outputs (0-255 PWM)
    "m1", "m2", "m3", "m4",
    # Battery
    "bat_pct", "bat_v", "bat_a",
    # AI / agent events (populated by logger, filled 0 normally)
    "ai_event",
]

# Global event log for AI-issued commands (keyboard_server.py can append here)
_ai_event_queue: list = []
_ai_event_lock = threading.Lock()

def log_ai_event(description: str):
    """Call from keyboard_server.py to tag AI commands in the CSV."""
    with _ai_event_lock:
        _ai_event_queue.append(description[:120])


def run_data_logger(state: DroneState, log_dir: str = "logs"):
    """
    Logs drone state to CSV at ~10 Hz.

    A new file is created each time the drone is ARMED.
    Files are written to  logs/run_YYYYMMDD_HHMMSS.csv
    """
    os.makedirs(log_dir, exist_ok=True)

    csv_file = None
    writer   = None
    t_arm    = None
    was_armed = False

    print(f"[LOG] Data logger active — files saved to ./{log_dir}/")

    while True:
        time.sleep(0.1)   # 10 Hz

        with state.lock:
            armed   = state.armed
            althold = state.althold
            poshold = state.poshold
            x, y, z   = state.x, state.y, state.z
            ekf_z     = state.ekf_z
            ekf_x, ekf_y = state.ekf_x, state.ekf_y
            vx, vy, vz = state.vx, state.vy, state.vz
            roll, pitch, yaw = state.roll, state.pitch, state.yaw
            gx, gy, gz = state.gx, state.gy, state.gz
            alt_sp   = state.alt_sp
            psp_x    = state.pos_sp_x
            psp_y    = state.pos_sp_y
            re, pe = state.error_roll, state.error_pitch
            ch1, ch2, ch3, ch4, ch5 = state.ch1, state.ch2, state.ch3, state.ch4, state.ch5
            m1, m2, m3, m4 = state.m1, state.m2, state.m3, state.m4

        # ── Open new file on ARM ───────────────────────────────────────────────
        if armed and not was_armed:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"run_{ts}.csv")
            csv_file = open(path, "w", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            csv_file.flush()
            t_arm = time.time()
            print(f"[LOG] Armed — logging to {path}")

        # ── Close file on DISARM ───────────────────────────────────────────────
        if not armed and was_armed and csv_file:
            csv_file.close()
            csv_file = None
            writer   = None
            t_arm    = None
            print("[LOG] Disarmed — log closed.")

        was_armed = armed

        # ── Write row ──────────────────────────────────────────────────────────
        if armed and writer:
            with _ai_event_lock:
                ai_ev = _ai_event_queue.pop(0) if _ai_event_queue else ""

            row = {
                "t_s":        round(time.time() - t_arm, 3),
                "wall_time":  datetime.datetime.now().isoformat(timespec="milliseconds"),
                "armed":      1,
                "althold":    int(althold),
                "poshold":    int(poshold),
                "x_m":        round(x,    4),
                "y_m":        round(y,    4),
                "z_m":        round(z,    4),
                "ekf_z_m":    round(ekf_z, 4),
                "ekf_x_m":    round(ekf_x, 4),
                "ekf_y_m":    round(ekf_y, 4),
                "vx_ms":      round(vx,   4),
                "vy_ms":      round(vy,   4),
                "vz_ms":      round(vz,   4),
                "roll_deg":   round(roll,  3),
                "pitch_deg":  round(pitch, 3),
                "yaw_deg":    round(yaw,   2),
                "gx_dps":     round(gx,   3),
                "gy_dps":     round(gy,   3),
                "gz_dps":     round(gz,   3),
                "alt_sp_m":   round(alt_sp, 4),
                "pos_sp_x_m": round(psp_x, 4),
                "pos_sp_y_m": round(psp_y, 4),
                "roll_err_deg":  round(re, 3),
                "pitch_err_deg": round(pe, 3),
                "ch1": ch1, "ch2": ch2, "ch3": ch3, "ch4": ch4, "ch5": ch5,
                "m1": m1,  "m2": m2,  "m3": m3,  "m4": m4,
                "bat_pct": state.bat_pct,
                "bat_v":   state.bat_voltage,
                "bat_a":   state.bat_current,
                "ai_event": ai_ev,
            }
            writer.writerow(row)
            csv_file.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not WS_OK:
        print("[SIM] ERROR: websockets library missing. pip install websockets")
        return

    print("=" * 62)
    print("  Maddy Drone Simulator")
    print("=" * 62)
    print(f"  WebSocket : ws://localhost:{WS_PORT}  (enter 'localhost' in browser)")
    print(f"  Camera    : http://localhost:{HTTP_PORT}/capture")
    print(f"  Physics   : {SIM_HZ} Hz   Telemetry: {TEL_HZ} Hz")
    print(f"  Hover     : ~ch1 1520  |  Max alt: 3 m")
    print()
    print("  In keyboard_server.py, change fetch_frame() URL port to 8080")
    print("  for camera/vision AI tools to work with the simulation.")
    print("=" * 62)

    # Shared state
    state = DroneState()
    global _sim_state_ref
    _sim_state_ref = state

    # Physics thread
    phys = PhysicsLoop(state)
    t_phys = threading.Thread(target=phys.run, daemon=True, name="physics")
    t_phys.start()

    # Camera HTTP server thread
    t_cam = threading.Thread(
        target=run_camera_server, args=(HTTP_PORT,), daemon=True, name="camera")
    t_cam.start()

    # Console status thread
    t_con = threading.Thread(
        target=run_console_printer, args=(state,), daemon=True, name="console")
    t_con.start()

    # CSV data logger thread
    t_log = threading.Thread(
        target=run_data_logger, args=(state,), daemon=True, name="datalog")
    t_log.start()

    # WebSocket server in background thread (frees main thread for GUI)
    def _ws_thread():
        asyncio.run(ws_main(state))
    t_ws = threading.Thread(target=_ws_thread, daemon=True, name="websocket")
    t_ws.start()

    # Visualiser MUST run on main thread on macOS
    if MPL_OK:
        run_visualiser(state)
    else:
        # No GUI — just block keeping background threads alive
        t_ws.join()


if __name__ == "__main__":
    main()
