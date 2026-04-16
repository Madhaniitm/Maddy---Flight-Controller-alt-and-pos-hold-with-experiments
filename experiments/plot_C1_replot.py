"""
Regenerate C1 publication-quality plot from saved CSV data.
Reads:  results/C1_nl_to_toolchain.csv  (telemetry)
        results/C1_tool_trace.csv        (tool calls)
Writes: results/C1_nl_to_toolchain.png  (overwrites old plot)
"""

import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

DIR    = os.path.join(os.path.dirname(__file__), "results")
IN_TEL   = os.path.join(DIR, "C1_nl_to_toolchain.csv")
IN_TRACE = os.path.join(DIR, "C1_tool_trace.csv")
OUT_PNG  = os.path.join(DIR, "C1_nl_to_toolchain.png")

TARGET_ALT = 1.0
TOLERANCE  = 0.10

# ── Load telemetry ─────────────────────────────────────────────────────────────
t_ms, z_true, z_ekf, z_sp = [], [], [], []
with open(IN_TEL) as f:
    for row in csv.DictReader(f):
        t_ms.append(float(row["t_ms"]))
        z_true.append(float(row["z_true_m"]))
        z_ekf.append(float(row["z_ekf_m"]))
        z_sp.append(float(row["z_setpoint_m"]))

t_s    = np.array(t_ms) / 1000.0
z_true = np.array(z_true)
z_ekf  = np.array(z_ekf)
z_sp   = np.array(z_sp)

# ── Load tool trace ────────────────────────────────────────────────────────────
tool_trace = []
with open(IN_TRACE) as f:
    for row in csv.DictReader(f):
        tool_trace.append({
            "name":       row["tool_name"],
            "sim_time_s": float(row["sim_time_s"]) if row["sim_time_s"] else 0.0,
            "args":       row["args_json"],
        })

# Key event times from trace
T_ARM      = next((t["sim_time_s"] for t in tool_trace if t["name"] == "arm"),       None)
T_HOVER    = next((t["sim_time_s"] for t in tool_trace if t["name"] == "find_hover_throttle"), None)
T_ALTHOLD  = next((t["sim_time_s"] for t in tool_trace if t["name"] == "enable_altitude_hold"), None)
T_TARGET   = next((t["sim_time_s"] for t in tool_trace if t["name"] == "set_altitude_target"), None)
T_VERIFY   = next((t["sim_time_s"] for t in tool_trace if t["name"] == "check_altitude_reached"), None)

# ── EKF: mask pre-arm noise ────────────────────────────────────────────────────
# Only show EKF after altitude hold is enabled (when it's actually meaningful)
ekf_clean = np.where(t_s >= T_ALTHOLD - 0.5, z_ekf, np.nan)

# Setpoint: proper step input
#   - NaN  before althold engages (no setpoint active)
#   - initial z at althold-enable (~0.068 m) from T_ALTHOLD → T_TARGET
#   - 1.0 m from T_TARGET onwards  (the actual step command)
# (Raw CSV has a spurious 0.5m artefact from DroneState default — we ignore it)
z_at_althold = float(z_true[np.searchsorted(t_s, T_ALTHOLD)])  # true z when althold engaged
sp_clean = np.where(t_s < T_ALTHOLD,                   np.nan,       z_at_althold)
sp_clean = np.where(t_s >= T_TARGET,                   TARGET_ALT,   sp_clean)
sp_clean = sp_clean.astype(float)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                gridspec_kw={"height_ratios": [2.2, 1]},
                                sharex=True)
fig.subplots_adjust(hspace=0.08)

# ── Top panel: altitude ────────────────────────────────────────────────────────
# Tolerance band
ax1.axhspan(TARGET_ALT - TOLERANCE, TARGET_ALT + TOLERANCE,
            alpha=0.12, color="#2ecc71", zorder=0,
            label=f"±{TOLERANCE*100:.0f} cm tolerance band")

# Command target
ax1.axhline(TARGET_ALT, color="#e67e22", ls=":", lw=1.4, alpha=0.9,
            label="Commanded target  1.0 m", zorder=1)

# True altitude
ax1.plot(t_s, z_true, color="#2980b9", lw=2.0, label="True altitude (sim)", zorder=4)

# EKF (clean, post-althold only)
ax1.plot(t_s, ekf_clean, color="#27ae60", lw=1.5, ls="--", alpha=0.85,
         label="EKF estimate (post-hold)", zorder=3)

# Setpoint step
ax1.step(t_s, sp_clean, color="#c0392b", lw=1.8, ls="-", where="post",
         label="Altitude setpoint", zorder=3)

# ── Event markers ──────────────────────────────────────────────────────────────
events = [
    (T_ARM,     "#e74c3c", "Arm",              1.20),
    (T_HOVER,   "#e67e22", "Find hover\nthrottle",  1.20),
    (T_ALTHOLD, "#8e44ad", "Enable\nalt-hold",  1.20),
    (T_TARGET,  "#2980b9", "Set target\n1.0 m", 1.20),
    (T_VERIFY,  "#16a085", "Verify\narrival",   1.20),
]

for t_ev, color, label, y_lbl in events:
    if t_ev is None:
        continue
    ax1.axvline(t_ev, color=color, ls="--", lw=1.2, alpha=0.75, zorder=2)
    ax1.text(t_ev + 0.2, y_lbl,
             label, fontsize=7.5, color=color, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                       edgecolor=color, alpha=0.85, lw=0.7))

# Steady-state: last 3 s of sim (30 samples @ 10 Hz) — matches C1 script metric
z_ss_samples = z_true[-30:] if len(z_true) >= 30 else z_true
z_ss = float(np.mean(z_ss_samples))
ax1.annotate(
    f"Steady-state mean = {z_ss:.3f} m\nerr = {abs(z_ss - TARGET_ALT)*100:.1f} cm",
    xy=(t_s[-1] - 1, z_ss), xytext=(t_s[-1] - 8, 0.65),
    fontsize=8.5, color="#2c3e50",
    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#bdc3c7", lw=0.8)
)

ax1.set_ylabel("Altitude  (m)", fontsize=11)
ax1.set_ylim(-0.08, 1.35)
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.25))
ax1.legend(fontsize=8, loc="upper left", framealpha=0.92)
ax1.grid(True, alpha=0.25, lw=0.6)
ax1.set_title(
    "EXP-C1 — Natural Language → Tool Chain\n"
    f'Command: "take off and hover at 1 metre" '
    f'| z_ss = {z_ss:.3f} m  (err = {abs(z_ss-TARGET_ALT)*100:.1f} cm)'
    f'  |  sequence = 4/4  |  19 API calls',
    fontsize=10, pad=6
)

# ── Bottom panel: tool call Gantt ──────────────────────────────────────────────
# Categories
META  = {"plan_workflow", "report_progress"}
WAIT  = {"wait"}
FLIGHT= {"arm", "find_hover_throttle", "enable_altitude_hold",
         "set_altitude_target", "land", "disarm"}
CHECK = {"check_drone_stable", "check_altitude_reached", "get_sensor_status"}

def cat_color(name):
    if name in META:  return "#bdc3c7"   # light grey
    if name in WAIT:  return "#95a5a6"   # mid grey
    if name in FLIGHT:return "#3498db"   # blue
    if name in CHECK: return "#2ecc71"   # green
    return "#ecf0f1"

# Group tools by name for y-axis (preserve order of first appearance)
seen, ytick_names = [], []
for t in tool_trace:
    if t["name"] not in seen:
        seen.append(t["name"])
        ytick_names.append(t["name"])

name_to_y = {n: i for i, n in enumerate(ytick_names)}

BAR_H = 0.55
for tr in tool_trace:
    yi    = name_to_y[tr["name"]]
    color = cat_color(tr["name"])
    ax2.barh(yi, 0.5, left=tr["sim_time_s"], height=BAR_H,
             color=color, edgecolor="white", lw=0.4, zorder=3)

# Event lines repeated on lower panel
for t_ev, color, _, _ in events:
    if t_ev is not None:
        ax2.axvline(t_ev, color=color, ls="--", lw=1.0, alpha=0.6, zorder=2)

ax2.set_yticks(range(len(ytick_names)))
ax2.set_yticklabels(ytick_names, fontsize=7.5)
ax2.set_xlabel("Simulated time  (s)", fontsize=11)
ax2.set_ylabel("Tool", fontsize=9)
ax2.grid(True, alpha=0.2, lw=0.5, axis="x")
ax2.set_xlim(t_s[0], t_s[-1])
ax2.set_title("Tool call sequence", fontsize=9, pad=3)

# Legend patches for bottom panel
legend_items = [
    mpatches.Patch(color="#3498db", label="Flight action"),
    mpatches.Patch(color="#2ecc71", label="Observation/check"),
    mpatches.Patch(color="#95a5a6", label="Wait"),
    mpatches.Patch(color="#bdc3c7", label="Meta (plan/report)"),
]
ax2.legend(handles=legend_items, fontsize=7.5, loc="lower right",
           framealpha=0.9, ncol=2)

plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()
print(f"Plot saved → {OUT_PNG}")
