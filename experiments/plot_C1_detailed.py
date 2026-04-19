"""
plot_C1_detailed.py  —  Comprehensive diagnostic plots for EXP-C1
==================================================================
Generates 9 figures covering every measurable aspect of the C1 experiment
(NL→tool-chain: "take off and hover at 1 metre").

Figures saved to experiments/results/:
  C1_fig1_flight_timeline.png      Full altitude timeline with LLM event markers
  C1_fig2_phase_zoom.png           Four zoomed phase panels (hover-find, stabilise, climb, hold)
  C1_fig3_error_analysis.png       Tracking error + rolling |error| convergence
  C1_fig4_ekf_fidelity.png         EKF vs truth: time-series overlay + scatter
  C1_fig5_cross_run_stats.png      Per-run bars: z_ss, RMSE, API calls, latency, cost
  C1_fig6_llm_tool_timeline.png    LLM decision Gantt chart with tool categories
  C1_fig7_token_cost.png           Token usage and cost breakdown per run
  C1_fig8_steadystate_dist.png     Steady-state altitude distribution + statistics
  C1_fig9_phase_timing.png         Phase duration breakdown + cumulative cost
"""

import pathlib, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS = pathlib.Path(__file__).parent / "results"

def rcsv(name):
    p = RESULTS / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save(fig, name):
    out = RESULTS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")

# ── Colours ───────────────────────────────────────────────────────────────────
C_TRUE    = "#0072B2"   # blue  — z_true
C_EKF     = "#E69F00"   # amber — z_ekf
C_SP      = "#CC0000"   # red   — setpoint
C_ERR     = "#009E73"   # green — error
C_ANNOT   = "#555555"   # grey  — annotations
C_BAND    = "#D0E8FF"   # light blue — CI band

TOOL_COLORS = {
    "plan_workflow":        "#9B59B6",
    "arm":                  "#E74C3C",
    "find_hover_throttle":  "#F39C12",
    "check_drone_stable":   "#27AE60",
    "enable_altitude_hold": "#2980B9",
    "wait":                 "#95A5A6",
    "set_altitude_target":  "#E67E22",
    "check_altitude_reached":"#1ABC9C",
    "report_progress":      "#BDC3C7",
}

# ── Load data ─────────────────────────────────────────────────────────────────
ts_rows  = rcsv("C1_nl_to_toolchain.csv")   # flight time-series
run_rows = rcsv("C1_runs.csv")              # per-run aggregate
trace    = rcsv("C1_tool_trace.csv")        # LLM tool calls
sum_rows = rcsv("C1_summary.csv")           # experiment summary

if not ts_rows:
    print("ERROR: C1_nl_to_toolchain.csv not found — run the experiment first.")
    raise SystemExit(1)

# Parse time-series
t_s   = np.array([float(r["t_ms"]) / 1000.0 for r in ts_rows])
z_t   = np.array([float(r["z_true_m"])    for r in ts_rows])
z_e   = np.array([float(r["z_ekf_m"])     for r in ts_rows])
z_sp  = np.array([float(r["z_setpoint_m"])for r in ts_rows])

# Parse per-run data
runs = []
for r in run_rows:
    try:
        runs.append({
            "run":       int(r["run"]),
            "z_ss":      float(r["z_ss_m"]),
            "rmse":      float(r["z_rmse_cm"]),
            "api":       int(r["api_calls"]),
            "lat":       float(r["mean_latency_s"]),
            "cost":      float(r["cost_usd"]),
            "itok":      int(r["input_tokens"]),
            "otok":      int(r["output_tokens"]),
            "wall":      float(r["wall_time_s"]),
            "passed":    int(r["passed"]),
        })
    except (KeyError, ValueError):
        pass

# Parse tool trace
tool_events = []
for r in trace:
    try:
        tool_events.append({
            "turn":    int(r["turn"]),
            "tool":    r["tool_name"],
            "sim_t":   float(r["sim_time_s"]),
            "preview": r["result_preview"],
        })
    except (KeyError, ValueError):
        pass

# Parse summary dict
summ = {r["metric"]: r["value"] for r in sum_rows if "metric" in r}

# ── Phase boundaries (derived from tool trace / setpoint changes) ─────────────
# Phase 0: idle/arm        t < 0.5 s
# Phase 1: hover find      0.5 – 9.4 s  (find_hover_throttle active)
# Phase 2: hover hold      9.4 – 11.4 s (enable_altitude_hold + wait 2s)
# Phase 3: climb           11.4 – 15.4 s (set_altitude_target 1.0m + wait 4s)
# Phase 4: steady-state    15.4 – end
PHASES = [
    (0.0,  0.5,  "Arm",         "#FDEBD0"),
    (0.5,  9.4,  "Hover Find",  "#D5F5E3"),
    (9.4,  11.4, "Hold Settle", "#D6EAF8"),
    (11.4, 15.4, "Climb",       "#FCF3CF"),
    (15.4, t_s[-1], "Steady State", "#F9EBEA"),
]


# =============================================================================
# Figure 1 — Full Flight Timeline with LLM Event Annotations
# =============================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle("C1 — Full Altitude Timeline with LLM Tool-Call Events",
                 fontsize=13, fontweight="bold")

    # Phase shading
    for t0, t1, label, col in PHASES:
        ax.axvspan(t0, t1, alpha=0.18, color=col, label=label)
        ax.text((t0 + t1) / 2, 1.12, label, ha="center", va="bottom",
                fontsize=8, color="#555", fontstyle="italic")

    # Altitude lines
    ax.plot(t_s, z_t,  color=C_TRUE, linewidth=1.8, label="z_true (physical)", zorder=3)
    ax.plot(t_s, z_e,  color=C_EKF,  linewidth=1.2, linestyle="--",
            label="z_ekf (sensor estimate)", zorder=2, alpha=0.85)
    ax.plot(t_s, z_sp, color=C_SP,   linewidth=1.2, linestyle=":",
            label="z_setpoint (LLM command)", zorder=2, alpha=0.85)

    # Target line at 1.0 m
    ax.axhline(1.0, color=C_SP, linewidth=0.8, linestyle="-.", alpha=0.5)
    ax.text(t_s[-1] + 0.1, 1.0, "1.0 m target", va="center", fontsize=8, color=C_SP)

    # Tolerance band ±0.1 m
    ax.axhspan(0.9, 1.1, alpha=0.07, color=C_SP, label="±0.1 m tolerance")

    # LLM tool call markers (skip report_progress for clarity)
    annotated = [e for e in tool_events if e["tool"] != "report_progress"]
    for ev in annotated:
        col = TOOL_COLORS.get(ev["tool"], "#333")
        ax.axvline(ev["sim_t"], color=col, linewidth=1.0, alpha=0.7, linestyle="--")
        # Find z at this sim_t
        idx = np.searchsorted(t_s, ev["sim_t"])
        if idx < len(z_t):
            ax.scatter(ev["sim_t"], z_t[idx], color=col, s=55, zorder=5)
        ax.text(ev["sim_t"] + 0.05, -0.12 + (annotated.index(ev) % 3) * 0.07,
                ev["tool"].replace("_", "\n"), fontsize=5.5, color=col,
                rotation=0, va="bottom", ha="left")

    ax.set_xlabel("Simulation time (s)", fontsize=10)
    ax.set_ylabel("Altitude (m)", fontsize=10)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlim(t_s[0], t_s[-1] + 0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, "C1_fig1_flight_timeline.png")


# =============================================================================
# Figure 2 — Four Zoomed Phase Panels
# =============================================================================
def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("C1 — Flight Phase Zoom-In", fontsize=13, fontweight="bold")

    def zoom(ax, t0, t1, title, show_sp=True):
        mask = (t_s >= t0) & (t_s <= t1)
        ax.plot(t_s[mask], z_t[mask], color=C_TRUE, linewidth=2.0, label="z_true")
        ax.plot(t_s[mask], z_e[mask],  color=C_EKF,  linewidth=1.3,
                linestyle="--", label="z_ekf", alpha=0.8)
        if show_sp:
            ax.plot(t_s[mask], z_sp[mask], color=C_SP, linewidth=1.3,
                    linestyle=":", label="setpoint", alpha=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Sim time (s)"); ax.set_ylabel("Altitude (m)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 1: Hover find (LLM ramps throttle; z_true rises slowly)
    zoom(axes[0,0], 0.5, 9.5, "Phase 1 — Hover Throttle Find (0.5–9.5 s)", show_sp=False)
    axes[0,0].set_ylim(-0.01, 0.12)
    axes[0,0].axhline(0.068, color="#F39C12", linestyle="-.", linewidth=1.0,
                      label="hover z = 6.8 cm")
    axes[0,0].legend(fontsize=8)

    # Panel 2: Hold settle (z_true holds ~6.8 cm, setpoint = 6.85 cm)
    zoom(axes[0,1], 9.4, 11.6, "Phase 2 — Altitude Hold Settle (9.4–11.6 s)")
    axes[0,1].set_ylim(0.055, 0.085)
    axes[0,1].axhline(0.0685, color=C_SP, linestyle="-.", linewidth=1.0, alpha=0.6,
                      label="hold setpoint 6.85 cm")
    axes[0,1].legend(fontsize=8)

    # Panel 3: Climb to 1.0 m (smooth, near-linear)
    zoom(axes[1,0], 11.4, 15.5, "Phase 3 — Climb to 1.0 m (11.4–15.5 s)")
    axes[1,0].axhline(1.0, color=C_SP, linestyle="-.", linewidth=1.0, alpha=0.6,
                      label="target 1.0 m")
    axes[1,0].axhspan(0.9, 1.1, alpha=0.08, color=C_SP, label="±0.1 m tolerance")
    axes[1,0].legend(fontsize=8)
    # Climb rate annotation
    t_start_climb = 11.5; t_end_climb = 14.5
    m_s = (1.0 - 0.068) / (t_end_climb - t_start_climb)
    axes[1,0].text(12.5, 0.55, f"climb rate ≈ {m_s:.2f} m/s",
                   fontsize=8, color=C_TRUE, fontstyle="italic")

    # Panel 4: Steady-state hold (t > 15.4 s)
    zoom(axes[1,1], 15.4, t_s[-1], "Phase 4 — Steady-State Hold (15.4 s → end)")
    ss_mask = (t_s >= 15.4)
    ss_mean = np.mean(z_t[ss_mask])
    ss_std  = np.std(z_t[ss_mask])
    axes[1,1].axhline(1.0,     color=C_SP,   linestyle="-.", linewidth=1.0, alpha=0.5)
    axes[1,1].axhline(ss_mean, color=C_TRUE,  linestyle="-", linewidth=1.0,
                      label=f"mean = {ss_mean:.4f} m")
    axes[1,1].axhspan(ss_mean - ss_std, ss_mean + ss_std, alpha=0.15, color=C_TRUE,
                      label=f"±1σ = {ss_std*100:.2f} cm")
    axes[1,1].set_ylim(0.98, 1.04)
    axes[1,1].legend(fontsize=8)

    plt.tight_layout()
    save(fig, "C1_fig2_phase_zoom.png")


# =============================================================================
# Figure 3 — Tracking Error Analysis
# =============================================================================
def fig3():
    # Only meaningful once setpoint = 1.0 m (after climb starts)
    climb_start = 11.4
    mask_all = t_s >= climb_start
    t_err    = t_s[mask_all]
    err      = z_t[mask_all] - z_sp[mask_all]   # signed error
    abs_err  = np.abs(err)

    # Rolling mean of |error| over 1-second window
    win = max(1, int(1.0 / (t_s[1] - t_s[0])))
    rolling_abs = np.convolve(abs_err, np.ones(win)/win, mode="same")

    # Compute RMSE for the steady-state window only
    ss_mask = t_err >= 15.4
    rmse_ss = np.sqrt(np.mean(err[ss_mask]**2)) * 100  # cm

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("C1 — Tracking Error Analysis (from climb start)", fontsize=13, fontweight="bold")

    # Error time-series
    axes[0].plot(t_err, err * 100, color=C_ERR, linewidth=1.5)
    axes[0].axhline(0,   color="black",  linewidth=0.8)
    axes[0].axhline( 10, color=C_SP, linewidth=0.8, linestyle="--", alpha=0.6, label="+10 cm tolerance")
    axes[0].axhline(-10, color=C_SP, linewidth=0.8, linestyle="--", alpha=0.6, label="-10 cm tolerance")
    axes[0].axvspan(15.4, t_err[-1], alpha=0.1, color="grey", label="steady-state window")
    axes[0].set_ylabel("Error (cm)"); axes[0].legend(fontsize=8)
    axes[0].set_title("Signed tracking error: z_true − z_setpoint")
    axes[0].grid(alpha=0.3)

    # Absolute error + rolling mean
    axes[1].plot(t_err, abs_err * 100, color=C_TRUE, linewidth=1.0, alpha=0.6, label="|error|")
    axes[1].plot(t_err, rolling_abs * 100, color=C_EKF, linewidth=2.0, label="1 s rolling mean")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axvspan(15.4, t_err[-1], alpha=0.1, color="grey", label="steady-state window")
    axes[1].set_ylabel("|Error| (cm)"); axes[1].legend(fontsize=8)
    axes[1].set_title("Absolute error + 1-second rolling mean (convergence)")
    axes[1].grid(alpha=0.3)

    # Cumulative RMSE (running)
    cum_mse  = np.cumsum(err**2) / (np.arange(len(err)) + 1)
    cum_rmse = np.sqrt(cum_mse) * 100
    axes[2].plot(t_err, cum_rmse, color="#9B59B6", linewidth=1.8)
    axes[2].axhline(rmse_ss, color=C_SP, linestyle="--", linewidth=1.0,
                    label=f"SS RMSE = {rmse_ss:.3f} cm")
    axes[2].axvspan(15.4, t_err[-1], alpha=0.1, color="grey", label="steady-state window")
    axes[2].set_ylabel("Running RMSE (cm)"); axes[2].set_xlabel("Sim time (s)")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Cumulative RMSE — converges to steady-state value")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "C1_fig3_error_analysis.png")


# =============================================================================
# Figure 4 — EKF Fidelity: Time-series overlay + Scatter
# =============================================================================
def fig4():
    valid = (z_e != 0) | (t_s > 9.0)   # EKF resets to 0 before arming; filter those
    mask  = (t_s >= 9.4) & (np.abs(z_e) < 20)  # post-arm, sane range

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("C1 — EKF Altitude Estimate Fidelity", fontsize=13, fontweight="bold")

    # Time series overlay
    ax = axes[0]
    ax.plot(t_s[mask], z_t[mask], color=C_TRUE, linewidth=1.8, label="z_true (ground truth)")
    ax.plot(t_s[mask], z_e[mask], color=C_EKF,  linewidth=1.4, linestyle="--",
            label="z_ekf (sensor estimate)")
    ax.fill_between(t_s[mask], z_t[mask], z_e[mask],
                    alpha=0.2, color="#E69F00", label="estimation error")
    ekf_err = z_e[mask] - z_t[mask]
    mean_ekf_err = np.mean(ekf_err) * 100
    std_ekf_err  = np.std(ekf_err)  * 100
    ax.set_title(f"EKF vs Truth  (bias={mean_ekf_err:+.2f} cm, σ={std_ekf_err:.2f} cm)")
    ax.set_xlabel("Sim time (s)"); ax.set_ylabel("Altitude (m)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Scatter: z_ekf vs z_true — ideal = diagonal
    ax2 = axes[1]
    sc = ax2.scatter(z_t[mask], z_e[mask],
                     c=t_s[mask], cmap="viridis", s=8, alpha=0.7)
    lo = min(z_t[mask].min(), z_e[mask].min()) - 0.01
    hi = max(z_t[mask].max(), z_e[mask].max()) + 0.01
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="ideal (y=x)")
    fig.colorbar(sc, ax=ax2, label="Sim time (s)")
    # R²
    ss_res = np.sum((z_e[mask] - z_t[mask])**2)
    ss_tot = np.sum((z_e[mask] - np.mean(z_e[mask]))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax2.set_title(f"EKF vs Truth Scatter  (R²={r2:.5f})")
    ax2.set_xlabel("z_true (m)"); ax2.set_ylabel("z_ekf (m)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "C1_fig4_ekf_fidelity.png")


# =============================================================================
# Figure 5 — Cross-Run Statistics (all 5 runs)
# =============================================================================
def fig5():
    if not runs:
        print("  skip fig5 — C1_runs.csv missing or empty")
        return

    run_ids  = [r["run"]  for r in runs]
    z_ss     = [r["z_ss"] for r in runs]
    rmse     = [r["rmse"] for r in runs]
    api      = [r["api"]  for r in runs]
    lat      = [r["lat"]  for r in runs]
    cost     = [r["cost"] for r in runs]
    x        = np.arange(len(run_ids))
    labels   = [f"Run {i}" for i in run_ids]
    col      = ["#2ECC71" if r["passed"] else "#E74C3C" for r in runs]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("C1 — Cross-Run Statistics (N=5)", fontsize=13, fontweight="bold")

    # z_ss per run
    ax = axes[0,0]
    bars = ax.bar(x, z_ss, color=col, width=0.5)
    ax.axhline(1.0,    color=C_SP,   linestyle="--", linewidth=1.2, label="target 1.0 m")
    ax.axhspan(0.9, 1.1, alpha=0.1, color=C_SP, label="±0.1 m tolerance")
    mean_z = np.mean(z_ss); std_z = np.std(z_ss)
    ax.axhline(mean_z, color=C_TRUE,  linestyle="-",  linewidth=1.0,
               label=f"mean = {mean_z:.4f} m")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Steady-State Altitude per Run"); ax.set_ylabel("z_ss (m)")
    ax.set_ylim(0.95, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, z_ss):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # RMSE per run
    ax = axes[0,1]
    bars = ax.bar(x, rmse, color=col, width=0.5)
    mean_r = np.mean(rmse)
    ax.axhline(mean_r, color=C_TRUE, linestyle="--", linewidth=1.0,
               label=f"mean = {mean_r:.3f} cm")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("RMSE per Run"); ax.set_ylabel("RMSE (cm)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # API calls per run
    ax = axes[0,2]
    ax.bar(x, api, color=col, width=0.5)
    ax.axhline(np.mean(api), color=C_TRUE, linestyle="--", linewidth=1.0,
               label=f"mean = {np.mean(api):.1f}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("API Calls per Run"); ax.set_ylabel("N API calls")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(api):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=9)

    # Mean API latency per run
    ax = axes[1,0]
    ax.bar(x, lat, color=col, width=0.5)
    ax.axhline(np.mean(lat), color=C_TRUE, linestyle="--", linewidth=1.0,
               label=f"mean = {np.mean(lat):.2f} s")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Mean API Latency per Run"); ax.set_ylabel("Latency (s)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(lat):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Cost per run
    ax = axes[1,1]
    ax.bar(x, cost, color=col, width=0.5)
    ax.axhline(np.mean(cost), color=C_TRUE, linestyle="--", linewidth=1.0,
               label=f"mean = ${np.mean(cost):.4f}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("API Cost per Run"); ax.set_ylabel("Cost (USD)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(cost):
        ax.text(i, v + 0.0002, f"${v:.4f}", ha="center", va="bottom", fontsize=8)

    # Pass/fail summary (pie)
    ax = axes[1,2]
    n_pass = sum(r["passed"] for r in runs)
    n_fail = len(runs) - n_pass
    ax.pie([n_pass, n_fail] if n_fail > 0 else [n_pass],
           labels=([f"Pass ({n_pass})", f"Fail ({n_fail})"] if n_fail > 0
                   else [f"Pass ({n_pass})"]),
           colors=["#2ECC71", "#E74C3C"][:1 if n_fail == 0 else 2],
           autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 11})
    ax.set_title("Pass/Fail (N=5)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save(fig, "C1_fig5_cross_run_stats.png")


# =============================================================================
# Figure 6 — LLM Tool-Call Gantt Chart
# =============================================================================
def fig6():
    if not tool_events:
        print("  skip fig6 — C1_tool_trace.csv missing")
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle("C1 — LLM Decision Timeline (Tool-Call Gantt)", fontsize=13, fontweight="bold")

    unique_tools = list(dict.fromkeys(e["tool"] for e in tool_events))
    y_map = {t: i for i, t in enumerate(unique_tools)}

    for ev in tool_events:
        y    = y_map[ev["tool"]]
        col  = TOOL_COLORS.get(ev["tool"], "#888")
        ax.barh(y, 0.4, left=ev["sim_t"] - 0.2, height=0.55,
                color=col, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Tiny turn label
        ax.text(ev["sim_t"], y + 0.32, f"T{ev['turn']}", ha="center",
                va="bottom", fontsize=6.5, color="white", fontweight="bold")

    # Add vertical phase lines
    for t0, t1, label, _ in PHASES:
        ax.axvline(t0, color="#aaa", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(range(len(unique_tools)))
    ax.set_yticklabels(unique_tools, fontsize=9)
    ax.set_xlabel("Simulation time (s)", fontsize=10)
    ax.set_xlim(-0.5, t_s[-1] + 0.5)
    ax.set_title("Each bar = one API call. T-number = conversation turn.")
    ax.grid(axis="x", alpha=0.25)

    # Phase labels at top
    for t0, t1, label, _ in PHASES:
        ax.text((t0 + t1) / 2, len(unique_tools) - 0.3, label,
                ha="center", va="bottom", fontsize=7.5, color="#555",
                fontstyle="italic")

    # Legend for tool colors
    patches = [mpatches.Patch(color=TOOL_COLORS.get(t, "#888"), label=t)
               for t in unique_tools]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5,
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    save(fig, "C1_fig6_llm_tool_timeline.png")


# =============================================================================
# Figure 7 — Token Usage & Cost Breakdown
# =============================================================================
def fig7():
    if not runs:
        print("  skip fig7 — C1_runs.csv missing or empty")
        return

    run_ids = [r["run"]  for r in runs]
    itok    = [r["itok"] for r in runs]
    otok    = [r["otok"] for r in runs]
    cost    = [r["cost"] for r in runs]
    x       = np.arange(len(run_ids))
    col     = ["#2ECC71" if r["passed"] else "#E74C3C" for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("C1 — Token Usage & Cost Breakdown", fontsize=13, fontweight="bold")

    # Stacked bar: input + output tokens
    ax = axes[0]
    ax.bar(x, itok, label="Input tokens",  color="#2980B9", width=0.5)
    ax.bar(x, otok, bottom=itok, label="Output tokens", color="#E67E22", width=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"R{i}" for i in run_ids])
    ax.set_title("Token Count per Run (input + output)")
    ax.set_ylabel("Tokens"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for i, (it, ot) in enumerate(zip(itok, otok)):
        ax.text(i, it + ot + 200, f"{it+ot:,}", ha="center", va="bottom", fontsize=8)

    # Cost per run
    ax = axes[1]
    bars = ax.bar(x, cost, color=col, width=0.5)
    ax.axhline(np.mean(cost), color="#555", linestyle="--", linewidth=1.0,
               label=f"mean = ${np.mean(cost):.4f}")
    ax.set_xticks(x); ax.set_xticklabels([f"R{i}" for i in run_ids])
    ax.set_title("API Cost per Run"); ax.set_ylabel("Cost (USD)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, cost):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0002,
                f"${v:.4f}", ha="center", va="bottom", fontsize=8)

    # Input vs output cost split pie (Claude pricing: input cheaper than output)
    ax = axes[2]
    # Claude 3 Sonnet approximate pricing used in verbalization_utils
    PRICE_IN  = 3.0   # $ per 1M input tokens
    PRICE_OUT = 15.0  # $ per 1M output tokens
    total_in  = sum(itok)
    total_out = sum(otok)
    cost_in   = total_in  * PRICE_IN  / 1e6
    cost_out  = total_out * PRICE_OUT / 1e6
    ax.pie([cost_in, cost_out],
           labels=[f"Input tokens\n${cost_in:.3f}\n({total_in/1000:.1f}k tokens)",
                   f"Output tokens\n${cost_out:.3f}\n({total_out/1000:.1f}k tokens)"],
           colors=["#2980B9", "#E67E22"],
           autopct="%1.1f%%", startangle=140,
           textprops={"fontsize": 9})
    ax.set_title(f"Cost Split: Input vs Output Tokens\nTotal: ${sum(cost):.4f}")

    plt.tight_layout()
    save(fig, "C1_fig7_token_cost.png")


# =============================================================================
# Figure 8 — Steady-State Altitude Distribution
# =============================================================================
def fig8():
    ss_mask = t_s >= 15.4
    z_ss_ts = z_t[ss_mask]

    mean_z = np.mean(z_ss_ts)
    std_z  = np.std(z_ss_ts)
    rmse   = np.sqrt(np.mean((z_ss_ts - 1.0)**2)) * 100
    bias   = (mean_z - 1.0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C1 — Steady-State Altitude Distribution (t ≥ 15.4 s)",
                 fontsize=13, fontweight="bold")

    # Histogram
    ax = axes[0]
    n, bins, patches = ax.hist(z_ss_ts * 100, bins=25, color=C_TRUE,
                               alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(100.0,      color=C_SP,    linewidth=1.8, linestyle="--", label="Target 100 cm")
    ax.axvline(mean_z*100, color="black", linewidth=1.5, linestyle="-",
               label=f"Mean = {mean_z*100:.2f} cm")
    ax.axvspan((mean_z - std_z)*100, (mean_z + std_z)*100,
               alpha=0.2, color=C_TRUE, label=f"±1σ = {std_z*100:.2f} cm")
    ax.axvspan((mean_z - 2*std_z)*100, (mean_z + 2*std_z)*100,
               alpha=0.08, color=C_TRUE, label=f"±2σ = {2*std_z*100:.2f} cm")
    ax.set_xlabel("Altitude (cm)"); ax.set_ylabel("Count")
    ax.set_title(f"Distribution of z_true in steady-state\nN={len(z_ss_ts)} samples")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Stats table
    ax.text(0.98, 0.97,
            f"Mean:  {mean_z*100:.4f} cm\n"
            f"Bias:  {bias:+.4f} cm\n"
            f"σ:     {std_z*100:.4f} cm\n"
            f"RMSE:  {rmse:.4f} cm\n"
            f"Min:   {z_ss_ts.min()*100:.4f} cm\n"
            f"Max:   {z_ss_ts.max()*100:.4f} cm\n"
            f"Range: {(z_ss_ts.max()-z_ss_ts.min())*100:.4f} cm",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Time series in steady state (to show stationarity)
    ax2 = axes[1]
    t_ss = t_s[ss_mask]
    ax2.plot(t_ss, z_ss_ts * 100, color=C_TRUE, linewidth=1.0, alpha=0.8)
    ax2.axhline(100.0,      color=C_SP,   linestyle="--", linewidth=1.2, label="Target 100 cm")
    ax2.axhline(mean_z*100, color="black", linestyle="-", linewidth=1.0,
                label=f"Mean = {mean_z*100:.2f} cm")
    ax2.axhspan((mean_z - std_z)*100, (mean_z + std_z)*100,
                alpha=0.2, color=C_TRUE, label=f"±1σ band")
    ax2.set_xlabel("Sim time (s)"); ax2.set_ylabel("Altitude (cm)")
    ax2.set_title("Steady-State Time Series — stationarity check")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "C1_fig8_steadystate_dist.png")


# =============================================================================
# Figure 9 — Phase Timing + Cumulative API Cost
# =============================================================================
def fig9():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C1 — Phase Timing & Cumulative Cost", fontsize=13, fontweight="bold")

    # Phase durations (from PHASES list)
    phase_names = [p[2] for p in PHASES]
    phase_durs  = [p[1] - p[0] for p in PHASES]
    phase_cols  = ["#E74C3C", "#F39C12", "#2980B9", "#27AE60", "#9B59B6"]

    ax = axes[0]
    bars = ax.barh(phase_names, phase_durs,
                   color=phase_cols[:len(PHASES)], edgecolor="white", linewidth=0.5)
    for bar, d in zip(bars, phase_durs):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{d:.1f} s", va="center", fontsize=9)
    ax.set_xlabel("Duration (s)"); ax.set_title("Time in Each Flight Phase")
    ax.grid(axis="x", alpha=0.3); ax.set_xlim(0, max(phase_durs) * 1.2)

    # Cumulative API call cost (from tool trace sim_times)
    if runs and tool_events:
        # Approximate per-call cost from total
        total_cost = runs[0]["cost"]
        n_calls    = len(tool_events)
        cost_per   = total_cost / max(n_calls, 1)
        sim_times  = sorted(ev["sim_t"] for ev in tool_events)
        cum_costs  = [(i + 1) * cost_per for i in range(len(sim_times))]

        ax2 = axes[1]
        ax2.step(sim_times, cum_costs, where="post", color=C_EKF, linewidth=2.0,
                 label="Cumulative cost (Run 1)")
        for st, cc, ev in zip(sim_times, cum_costs, tool_events):
            if ev["tool"] != "report_progress":
                col = TOOL_COLORS.get(ev["tool"], "#888")
                ax2.scatter([st], [cc], color=col, s=55, zorder=5)

        ax2.set_xlabel("Simulation time (s)"); ax2.set_ylabel("Cumulative cost (USD)")
        ax2.set_title(f"Cumulative API Cost vs Time\n(total = ${total_cost:.4f})")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        ax2.text(0.02, 0.95,
                 f"{n_calls} API calls\n"
                 f"≈ ${cost_per:.5f} / call",
                 transform=ax2.transAxes, va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    save(fig, "C1_fig9_phase_timing.png")


# =============================================================================
# Run all figures
# =============================================================================
if __name__ == "__main__":
    import sys
    targets = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else \
              ["FIG1","FIG2","FIG3","FIG4","FIG5","FIG6","FIG7","FIG8","FIG9"]

    fns = {
        "FIG1": fig1, "FIG2": fig2, "FIG3": fig3,
        "FIG4": fig4, "FIG5": fig5, "FIG6": fig6,
        "FIG7": fig7, "FIG8": fig8, "FIG9": fig9,
    }

    print(f"Generating C1 detailed plots → {RESULTS}")
    print()
    for key in targets:
        if key in fns:
            fns[key]()
        else:
            print(f"  unknown: {key}")
    print("\nDone.")
