# A2 Experiment Observations

---

## EXP-A2: AHRS Filter Comparison — Madgwick vs Alternatives

**What it tests:** 8 orientation filters converging from 0° to 30° roll; compares convergence speed, steady-state accuracy, and computational cost.

**Filters compared:** Low-pass, Complementary, Simple KF, EKF (4-state), Mahony, Madgwick (firmware default), 9-state EKF.

---

### Observation A2-OBS-1: Madgwick convergence at t ≈ 9 s is correct and expected

**Finding:** The Madgwick filter takes approximately 9 seconds to converge from 0° to the true 30° roll angle. This is marked with a vertical dashed line in the plot. At first glance this appears slow compared to filters like Mahony (~4–5 s) or EKF (which jumps near-instantly due to high process noise), but it is the correct and intended behaviour for the firmware-default gain setting.

**Root cause — the beta (β) parameter:**
The Madgwick algorithm uses a gradient-descent step of size β to iteratively rotate the quaternion estimate toward the accelerometer-corrected reference. The step size directly controls convergence speed:

> β = √(3/4) × GyroMeasError

With the firmware default of GyroMeasError ≈ 40°/s (in radians ≈ 0.6981 rad/s):

> β ≈ √(3/4) × 0.6981 ≈ 0.0331

This experiment uses β = 0.03, which matches the firmware default and produces convergence in approximately 8–12 s from a cold start — consistent with the ~9 s observed.

**Beta vs convergence trade-off:**

| β value         | Convergence time | Trade-off                        |
|-----------------|-----------------|----------------------------------|
| 0.03 (firmware) | ~8–12 s         | Smooth, noise-resistant          |
| 0.1–0.2         | ~2–4 s          | Mild oscillation at steady state |
| 0.3–0.5         | < 1 s           | Noisy, oscillatory output        |

**Significance for the thesis:**
- The 10-second initialization window used in Madgwick's original experiments was chosen precisely because β = 0.033 needs that long to converge from arbitrary initial conditions.
- Other filters that appear to "win" early (e.g., EKF jumps to ~30° within 0.08 s) do so at the cost of large initial noise spikes or high computational load.
- Once converged (t > 9 s), Madgwick achieves the best balance of SS RMSE, noise floor, and CPU cost — which is why it was selected for the firmware.

**References:**

[1] Madgwick, S.O.H. (2010). *An Efficient Orientation Filter for Inertial and Inertial/Magnetic Sensor Arrays.* University of Bristol Internal Report.
Defines β and derives the default value from GyroMeasError. States that a 10-second stationary initialization period is used to allow filter states to converge.

[2] Madgwick, S.O.H., Harrison, A.J.L. & Vaidyanathan, R. (2011). *Estimation of IMU and MARG Orientation Using a Gradient Descent Algorithm.* IEEE ICORR 2011, pp. 1–7. DOI: 10.1109/ICORR.2011.5975346.
Published version. Confirms that convergence rate scales with β and that the firmware default (β ≈ 0.033) prioritises noise rejection over convergence speed.

[3] AHRS Python Library Documentation — Madgwick Filter. https://ahrs.readthedocs.io/en/latest/filters/madgwick.html
Documents β formula and its effect on convergence speed. Notes that higher β values accelerate convergence but increase sensitivity to accelerometer noise.

[4] Valenti, R.G., Dryanovski, I. & Xiao, J. (2015). *Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs.* Sensors, 15(8), 19302–19330. DOI: 10.3390/s150819302.
Comparison study showing EKF converges faster from cold start but at 5.5× higher CPU cost than Madgwick. Validates that slower Madgwick convergence is a known and accepted trade-off.
