"""
EXP-A4: Ground Effect Validation
===================================
Computes thrust multiplier at different hover altitudes and compares
to multiple ground effect models from literature.

No WebSocket, no API, no GUI.
Saves: results/A4_ground_effect.csv, results/A4_ground_effect.png

─── Ground Effect Model References ────────────────────────────────────────────

[1] Cheeseman, I.C. & Bennett, W.E. (1955)
    "The Effect of the Ground on a Helicopter Rotor in Forward Flight"
    ARC R&M No. 3021, Aeronautical Research Council, UK.
    Formula: k_ge = 1 / (1 - (R/4h)²)
    Note: Derived for single-rotor helicopters. Has singularity at h < R/4.
    Overestimates ground effect for small quadrotors.

[2] Li, J., Zhou, Z., Shi, Z. & Lu, Y. (2015)
    "Autonomous Landing of Quadrotor Based on Ground Effect Modelling"
    Proc. 34th Chinese Control Conference, IEEE, 2015.
    Formula: k_ge = 1 / (1 - ρ*(R/4h)²),  ρ = 3.4 (fitted to quadrotor)
    Note: Modified Cheeseman-Bennett with ρ fitted to quadrotor data.
    ρ=3.4 re-identified by Kan et al. (2019) on AscTec Hummingbird.

[3] He, X., Kou, G., Calaf, M. & Leang, K.K. (2019)
    "In-Ground-Effect Modeling and Nonlinear-Disturbance Observer for
     Multirotor Unmanned Aerial Vehicle Control"
    ASME Journal of Dynamic Systems, Measurement, and Control,
    Vol. 141, No. 7, 071013, 2019.
    Formula: k_ge = 1 + Ca * exp(-Cb * z/R)
    Note: Singularity-free exponential model. THIS IS THE MODEL USED IN SIM.
    Coefficients Ca and Cb derived from blade element theory + experiment.
    Achieved ~23% reduction in altitude tracking error vs baseline.

[4] Kan, X., Thomas, J., Teng, H., Tanner, H.G., Kumar, V. & Karydis, K. (2019)
    "Analysis of Ground Effect for Small-scale UAVs in Forward Flight"
    IEEE Robotics and Automation Letters, accepted July 2019.
    Formula (hover only): k_ge = 1 / (1.02 - 0.171*(R/z))
    Note: Validated specifically on Crazyflie 2.0 (R=23mm) — same as this drone.
    Ground effect significant up to z = 5R (vs 2R in Cheeseman-Bennett).
    Only model validated for forward flight (0–8 m/s).

[5] Sanchez-Cuevas, P., Heredia, G. & Ollero, A. (2017)
    "Characterization of the Aerodynamic Ground Effect and Its Influence
     in Multirotor Control"
    International Journal of Aerospace Engineering, Wiley, 2017.
    Note: Accounts for multi-rotor inter-rotor interaction and body lift.
    More complex, requires rotor separation geometry as input.

[6] Yang, Z., Chai, J., Ji, J., Wu, X., Xu, G. & Gao, F. (2025)
    "Ground-Effect-Aware Modeling and Control for Multicopters"
    arXiv:2506.19424, Zhejiang University, 2025.
    Formula: F_G(h) = g2 / (h² + g1)  (rational function, singularity-free)
    Note: Also models ground-induced leveling torque — novel contribution.
    Found rotor drag is only 59.6–61.8% of free-air value near ground.
    Achieved 45.3% RMSE reduction vs baseline controller.

Key findings from recent literature:
  - Ground effect for small quadrotors extends to z ≈ 5R (not 2R as CB predicts)
  - Exponential models (He 2019) are preferred — no singularity, physically grounded
  - Kan et al. (2019) is the most relevant for this drone (Crazyflie R=23mm validated)
  - The sim already uses the He et al. (2019) structure — modern and correct
──────────────────────────────────────────────────────────────────────────────
"""

import sys, os, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import GE_COEFF, GE_DECAY, R_ROTOR, MASS, GRAVITY

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "A4_ground_effect.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "A4_ground_effect.png")

# Test altitudes from ground contact to 6× rotor radius
z_vals_m = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20,
            0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]

z_norm = [z / R_ROTOR for z in z_vals_m]

# ── Model 1: Sim (He et al. 2019 exponential — singularity-free) [Ref 3] ─────
k_ge_sim = [1.0 + GE_COEFF * math.exp(-z / (GE_DECAY * R_ROTOR)) for z in z_vals_m]

# ── Model 2: Cheeseman-Bennett 1955 [Ref 1] ──────────────────────────────────
k_ge_cb = []
for z in z_vals_m:
    if z == 0.0:
        k_ge_cb.append(float("inf"))
    else:
        ratio = R_ROTOR / (4.0 * z)
        k_ge_cb.append(float("inf") if ratio >= 1.0 else 1.0 / (1.0 - ratio**2))

# ── Model 3: Li et al. 2015 — modified CB with ρ=3.4 for quadrotors [Ref 2] ─
# k_ge = 1 / (1 - rho*(R/4z)^2),  rho=3.4 re-identified on quadrotor
RHO_LI = 3.4
k_ge_li = []
for z in z_vals_m:
    if z == 0.0:
        k_ge_li.append(float("inf"))
    else:
        denom = 1.0 - RHO_LI * (R_ROTOR / (4.0 * z))**2
        k_ge_li.append(float("inf") if denom <= 0 else 1.0 / denom)

# ── Model 4: Kan et al. 2019 — Crazyflie-validated, hover only [Ref 4] ───────
# k_ge = 1 / (1.02 - 0.171*(R/z))
# Valid when denominator > 0, i.e., z > 0.171*R/1.02 = 0.168*R
k_ge_kan = []
for z in z_vals_m:
    if z == 0.0:
        k_ge_kan.append(float("inf"))
    else:
        denom = 1.02 - 0.171 * (R_ROTOR / z)
        k_ge_kan.append(float("inf") if denom <= 0 else 1.0 / denom)

# ── Print table ───────────────────────────────────────────────────────────────
print("[A4] Ground Effect Thrust Multiplier vs Altitude")
print(f"{'z(m)':>6} {'z/R':>6} {'Sim(He19)':>10} {'CB1955':>10} {'Li2015':>10} {'Kan2019':>10}")
print("-" * 60)
rows = []
for z, zn, ks, kc, kl, kk in zip(z_vals_m, z_norm, k_ge_sim, k_ge_cb, k_ge_li, k_ge_kan):
    fmt = lambda k: f"{k:10.4f}" if k < 99 else "      inf "
    print(f"{z:6.3f} {zn:6.2f} {fmt(ks)} {fmt(kc)} {fmt(kl)} {fmt(kk)}")
    rows.append([round(z,3), round(zn,3),
                 round(ks,5),
                 round(kc,5) if kc < 99 else "inf",
                 round(kl,5) if kl < 99 else "inf",
                 round(kk,5) if kk < 99 else "inf"])

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["z_m", "z_over_R", "k_ge_sim_He2019",
                "k_ge_Cheeseman_Bennett_1955",
                "k_ge_Li_2015_rho3p4",
                "k_ge_Kan_2019_Crazyflie"])
    w.writerows(rows)
print(f"\n[A4] CSV saved: {OUT_CSV}")

# ── Fine curves ───────────────────────────────────────────────────────────────
z_fine  = np.linspace(0.1 * R_ROTOR, 3.0, 500)
zn_fine = z_fine / R_ROTOR

# Sim / He 2019 — starts at z = 0.01 m (z/R = 0.43), Crazyflie leg clearance lower bound.
# Below z/R = 0.43 the drone is essentially on the ground; GE model not operationally relevant.
# 0.43R also sits just above the CB 1955 singularity at R/4 = 0.25R.
z_fine_sim  = np.linspace(0.01, 3.0, 500)
zn_fine_sim = z_fine_sim / R_ROTOR
k_fine_sim  = [1.0 + GE_COEFF * math.exp(-z / (GE_DECAY * R_ROTOR)) for z in z_fine_sim]

# Cheeseman-Bennett 1955
k_fine_cb = []
for z in z_fine:
    r = R_ROTOR / (4.0 * z)
    k_fine_cb.append(None if r >= 1.0 else 1.0/(1.0-r**2))
k_fine_cb_plot = [k if k and k < 5.0 else None for k in k_fine_cb]

# Li et al. 2015 (ρ=3.4)
k_fine_li = []
for z in z_fine:
    d = 1.0 - RHO_LI * (R_ROTOR/(4.0*z))**2
    k_fine_li.append(None if d <= 0 else 1.0/d)
k_fine_li_plot = [k if k and k < 5.0 else None for k in k_fine_li]

# Kan et al. 2019
k_fine_kan = []
for z in z_fine:
    d = 1.02 - 0.171 * (R_ROTOR / z)
    k_fine_kan.append(None if d <= 0 else 1.0/d)
k_fine_kan_plot = [k if k and k < 5.0 else None for k in k_fine_kan]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

# Sim (He 2019) — primary model (starts at z/R = 0.5)
ax.plot(zn_fine_sim, k_fine_sim, color="blue", linewidth=2.5,
        label="Sim — He et al. 2019: 1 + Ca·exp(−Cb·z/R)  [THIS DRONE]")
ax.scatter([z/R_ROTOR for z in z_vals_m if z >= 0.01],
           [k for z, k in zip(z_vals_m, k_ge_sim) if z >= 0.01],
           color="blue", s=50, zorder=5)

# Cheeseman-Bennett 1955
cb_pairs = [(zn, k) for zn, k in zip(zn_fine, k_fine_cb_plot) if k is not None]
if cb_pairs:
    zn_cb, k_cb = zip(*cb_pairs)
    ax.plot(zn_cb, k_cb, color="orange", linestyle="--", linewidth=1.8,
            label="Cheeseman-Bennett 1955: 1/(1−(R/4h)²)  [helicopter, has singularity]")

# Li et al. 2015
li_pairs = [(zn, k) for zn, k in zip(zn_fine, k_fine_li_plot) if k is not None]
if li_pairs:
    zn_li, k_li = zip(*li_pairs)
    ax.plot(zn_li, k_li, color="green", linestyle="-.", linewidth=1.8,
            label="Li et al. 2015: 1/(1−3.4(R/4h)²)  [quadrotor, ρ=3.4]")

# Kan et al. 2019
kan_pairs = [(zn, k) for zn, k in zip(zn_fine, k_fine_kan_plot) if k is not None]
if kan_pairs:
    zn_kan, k_kan = zip(*kan_pairs)
    ax.plot(zn_kan, k_kan, color="red", linestyle=":", linewidth=2.0,
            label="Kan et al. 2019: 1/(1.02−0.171·R/z)  [Crazyflie validated ✓]")

# CB singularity shaded region
singularity_zR = (R_ROTOR / 4) / R_ROTOR
ax.axvspan(0, singularity_zR, color="orange", alpha=0.08,
           label=f"CB undefined (h < R/4 = {R_ROTOR/4*1000:.1f}mm)")

# GE significant zone (5R — from Kan et al. 2019)
ax.axvline(5.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(5.05, 1.02, "z=5R\n(GE limit\nKan 2019)", fontsize=7, color="gray")

ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Normalised height  z / R_rotor")
ax.set_ylabel("Thrust multiplier  k_ge")
ax.set_title("EXP-A4: Ground Effect — Sim vs Literature Models\n"
             "(Cheeseman-Bennett 1955 · Li 2015 · He 2019 · Kan 2019)")
ax.set_xlim([0, 6])
ax.set_ylim([0.95, 2.0])
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# Annotate sim start at z/R = 0.43 (z = 10 mm — leg clearance / operational lower bound)
val_043R = 1.0 + GE_COEFF * math.exp(-0.01 / (GE_DECAY * R_ROTOR))
ax.annotate(f"Sim starts at z/R=0.43\n(z=10 mm, leg clearance)",
            xy=(0.43, val_043R),
            xytext=(1.4, 1.62),
            arrowprops=dict(arrowstyle="->"),
            fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[A4] Plot saved: {OUT_PNG}")
print(f"\n[A4] RESULT: Sim (He 2019) peak = {max(k_ge_sim):.4f}× at z=0m.")
print(f"[A4]   Models compared: Cheeseman-Bennett 1955, Li et al. 2015, Kan et al. 2019")
print(f"[A4]   Kan et al. 2019 is most relevant — validated on Crazyflie 2.0 (R=23mm)")
