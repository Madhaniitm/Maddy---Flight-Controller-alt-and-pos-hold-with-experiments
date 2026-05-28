"""
exp_ND2_plots.py — Flight Path & Altitude Plots for ND2 Results
================================================================
Reads the ND2_workflow_<ts>.csv produced by exp_ND2_agentic_vision_control.py
and generates C-series-style flight path and altitude plots with THREE sensor
scenario traces per plot:

  1. Commanded path          — the exact target from tool_args of each movement
                               command (navigate_to_waypoint x/y/z, relative
                               move accumulation). The "ideal" the LLM asked for.

  2. ToF + Optical Flow      — telemetry_after.pos_x/y/alt_m from the physics sim.
     (sim, best case)          drone_sim.py already runs a Kalman filter fusing:
                                 • ToF:           TOF_NOISE  = 5 mm RMS  → ekf_z
                                 • Optical flow:  FLOW_NOISE = 15 mm/s RMS → ekf_x/y
                                 • Gyro:          GYRO_NOISE = 0.5 deg/s
                               ekf_z residual  ≈ 5–10 mm   (barometer-class accuracy)
                               ekf_x/y residual ≈ 1–3 cm   (optical flow class accuracy)

  3. ToF only, no optical flow — altitude from ToF (accurate), XY position from
     (common low-cost drone)    IMU integration only. Without optical flow, XY
                                position estimation drifts with accelerometer bias.
                                Modelled as:
                                  σ_xy = 3.0 cm/cmd, bias_xy = 0.8 cm/cmd
                                  σ_z  = 0.3 cm/cmd, bias_z  = 0.1 cm/cmd (ToF stable)
                                Expected XY drift over 10 cmd steps ≈ 12–25 cm

  4. IMU only, no sensors    — no ToF correction, no optical flow. Pure IMU
     (worst case)              integration (baro for alt, accel for XY).
                               Reflects a drone without ToF or optical flow module.
                               Modelled as:
                                 σ_xy = 7.0 cm/cmd, bias_xy = 1.5 cm/cmd
                                 σ_z  = 3.0 cm/cmd, bias_z  = 0.5 cm/cmd
                               Expected XY drift over 10 cmd steps ≈ 30–60 cm

  The three sensor scenarios span the real hardware design space and justify
  including both ToF and optical flow in an autonomous indoor patrol drone.

Plots generated (saved to results/ND2_plots_<ts>/):
  ─ Fig 1: top-down XY flight path per orchestrator (4 subplots, 3 sensor traces)
  ─ Fig 2: altitude vs step per orchestrator (3 sensor traces)
  ─ Fig 3: XY drift magnitude vs step — all three sensor scenarios compared
  ─ Fig 4: combined overlay — all orchestrators, run 1, XY path
  ─ Fig 5: altitude mean ± 1σ across 5 runs per orchestrator
  ─ Fig 6: commanded waypoints annotated on actual path
  ─ Fig 7: sensor scenario comparison — side-by-side 3-trace XY + altitude

Run:
    /opt/homebrew/bin/python3.11 experiments/exp_ND2_plots.py
    /opt/homebrew/bin/python3.11 experiments/exp_ND2_plots.py results/ND2_workflow_20260528_120000.csv
"""

import sys, pathlib, json, csv, re, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Path setup ─────────────────────────────────────────────────────────────────
EXP_DIR     = pathlib.Path(__file__).parent
RESULTS_DIR = EXP_DIR / "results"

ORCHESTRATORS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
ORCH_COLORS   = {
    "claude":     "#E87040",   # orange (Anthropic)
    "gpt4o":      "#10A37F",   # green  (OpenAI)
    "gpt4o_mini": "#2196F3",   # blue
    "gemini":     "#8B5CF6",   # purple (Google)
}
SCENARIOS = ["clear_room", "alert_room"]

# ── Sensor scenario drift models ───────────────────────────────────────────────
# Sourced from drone_sim.py noise constants:
#   TOF_NOISE  = 0.005 m RMS  (ekf_z)
#   FLOW_NOISE = 0.015 m/s RMS (ekf_x/y)
#   GYRO_NOISE = 0.5 deg/s RMS
#
# "tof_flow" scenario = drone_sim.py telemetry_after (already the KF output).
# "tof_only" and "imu_only" are applied POST-HOC to the tof_flow positions
# to simulate removing sensors from the same commanded trajectory.

DRIFT_PROFILES = {
    # ── Scenario 1: ToF + Optical Flow ─────────────────────────────────────────
    # This is what drone_sim.py already outputs in telemetry_after.
    # No additional drift added — the KF residuals (5–10 mm alt, 1–3 cm XY)
    # are already present in the actual telemetry_after values.
    "tof_flow": {
        "label":       "ToF + Optical Flow (sim)",
        "linestyle":   "-",
        "alpha":        0.90,
        "sigma_xy":    0.000,   # no additional drift — KF already running
        "sigma_z":     0.000,
        "bias_xy":     0.000,
        "bias_z":      0.000,
    },
    # ── Scenario 2: ToF only, no optical flow ──────────────────────────────────
    # Altitude from ToF (accurate). XY from IMU integration only.
    # Without optical flow: accelerometer bias accumulates → XY drift.
    # drone_sim accel noise is 0.02g = 0.196 m/s². Over a ~3s movement step
    # the random position error from accel noise ≈ 0.5 * 0.02g * (3s)² ≈ 9 cm.
    # Combined with gyro attitude error feeding back into position, realistic
    # XY drift per command step on a low-cost board (MPU6050) ≈ 3–8 cm.
    "tof_only": {
        "label":       "ToF only (no optical flow)",
        "linestyle":   "--",
        "alpha":        0.85,
        "sigma_xy":    0.030,   # 3.0 cm σ per movement command
        "sigma_z":     0.003,   # 3 mm σ (ToF still running for altitude)
        "bias_xy":     0.008,   # 0.8 cm per command in fixed direction
        "bias_z":      0.001,   # 1 mm bias (ToF stable)
    },
    # ── Scenario 3: IMU only — no ToF, no optical flow ─────────────────────────
    # Altitude from barometer + IMU (baro noise ~5–20 cm, slow drift).
    # XY from pure IMU integration — very fast drift indoors.
    # Without any correction, a 50g drone at ~10 cm/s attitude-induced
    # velocity error would drift ~0.5 m in 5 s. We model the residual
    # after attitude-based velocity zeroing: still ~7–15 cm per step.
    "imu_only": {
        "label":       "IMU only (no ToF, no optical flow)",
        "linestyle":   ":",
        "alpha":        0.80,
        "sigma_xy":    0.070,   # 7 cm σ per movement command
        "sigma_z":     0.030,   # 3 cm σ (baro is noisy)
        "bias_xy":     0.015,   # 1.5 cm per command bias
        "bias_z":      0.005,   # 5 mm bias (baro drift)
    },
}

# Ordered list for consistent plot ordering
DRIFT_ORDER = ["tof_flow", "tof_only", "imu_only"]
DRIFT_COLORS = {
    "tof_flow": None,      # inherits orchestrator colour (same as actual)
    "tof_only": "darkorange",
    "imu_only": "crimson",
}

# Movement commands whose execution changes drone XY position
XY_MOVE_CMDS = {
    "navigate_to_waypoint", "move_forward", "move_backward",
    "move_left", "move_right", "return_to_home",
}
ALT_CMDS = {"set_altitude_target", "navigate_to_waypoint", "land", "arm"}


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _load_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_tel(json_str: str) -> dict:
    """Parse telemetry JSON string, return {} on failure."""
    try:
        return json.loads(json_str) if json_str else {}
    except Exception:
        return {}


def _parse_args(json_str: str) -> dict:
    try:
        return json.loads(json_str) if json_str else {}
    except Exception:
        return {}


def _find_latest_workflow_csv() -> pathlib.Path | None:
    csvs = sorted(RESULTS_DIR.glob("ND2_workflow_*.csv"))
    return csvs[-1] if csvs else None


# ── Commanded-position tracker ─────────────────────────────────────────────────

def _commanded_positions(rows: list[dict]) -> list[dict]:
    """
    Walk through workflow rows and reconstruct the COMMANDED position sequence.

    For navigate_to_waypoint: commanded = {x_m, y_m, z_m} from tool_args.
    For relative moves (move_forward etc.): commanded = last known position
        + displacement (from tool_args distance_m, in body frame).
        Body frame: forward = +X, right = +Y (matches DAgent._move_xy convention).
    For set_altitude_target: commanded_z = target_m.
    For arm: commanded = (0, 0, 0).
    For land: commanded_z = 0.

    Returns list of dicts:
      {action_num, tool_name, cmd_x, cmd_y, cmd_z, step}
    """
    pos   = [0.0, 0.0, 0.0]   # [x, y, z] — running commanded position
    track = []
    step  = 0

    for r in rows:
        name   = r.get("tool_name", "")
        args   = _parse_args(r.get("tool_args", "{}"))
        tel_b  = _parse_tel(r.get("telemetry_before", "{}"))

        if name == "arm":
            pos = [0.0, 0.0, 0.0]
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name in ("set_altitude_target", "enable_altitude_hold"):
            h = float(args.get("target_m", args.get("altitude_m", pos[2])))
            pos[2] = h
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name == "navigate_to_waypoint":
            pos = [float(args.get("x_m", pos[0])),
                   float(args.get("y_m", pos[1])),
                   float(args.get("z_m", pos[2]))]
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name == "move_forward":
            d = float(args.get("distance_m", 0.2))
            pos[0] += d   # forward = +X in DAgent body frame
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name == "move_backward":
            d = float(args.get("distance_m", 0.2))
            pos[0] -= d
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name == "move_left":
            d = float(args.get("distance_m", 0.2))
            pos[1] -= d   # left = -Y in DAgent (_move_xy: left → dy=-dist)
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name == "move_right":
            d = float(args.get("distance_m", 0.2))
            pos[1] += d   # right = +Y
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name == "hover":
            # No position change — just record current commanded position
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

        elif name in ("land", "emergency_land"):
            pos[2] = 0.0
            track.append({"step": step, "tool_name": name,
                           "cmd_x": pos[0], "cmd_y": pos[1], "cmd_z": pos[2]})
            step += 1

    return track


# ── Drift accumulator ──────────────────────────────────────────────────────────

def _apply_drift(actual_pts: list[dict], rng: np.random.Generator,
                 move_steps: set[int], profile: dict) -> list[dict]:
    """
    Add accumulated sensor drift to a list of actual positions.

    Drift only accumulates on movement-command steps — IMU integration error
    grows with time in flight, not while hovering stationary.

    Parameters
    ----------
    actual_pts : list of {step, actual_x, actual_y, actual_z}
    rng        : reproducible numpy RNG (seeded per run, same seed for all profiles
                 so profiles are comparable on identical trajectories)
    move_steps : set of step indices that involved actual XY/Z movement
    profile    : one of DRIFT_PROFILES — contains sigma_xy, sigma_z, bias_xy, bias_z

    Returns list of same length with added drifted_{x,y,z} and drift_mag_xy fields.
    """
    sigma_xy = profile["sigma_xy"]
    sigma_z  = profile["sigma_z"]
    bias_xy  = profile["bias_xy"]
    bias_z   = profile["bias_z"]

    if sigma_xy == 0.0 and bias_xy == 0.0:
        # tof_flow scenario — no additional drift, return actual positions unchanged
        return [{**pt,
                 "drifted_x":    pt["actual_x"],
                 "drifted_y":    pt["actual_y"],
                 "drifted_z":    pt["actual_z"],
                 "drift_mag_xy": 0.0} for pt in actual_pts]

    # Fixed bias direction — random unit vector (constant for this run + profile)
    bias_angle = rng.uniform(0, 2 * np.pi)
    b_x = bias_xy * np.cos(bias_angle)
    b_y = bias_xy * np.sin(bias_angle)
    b_z = bias_z  * rng.choice([-1, 1])

    drift_x = drift_y = drift_z = 0.0
    drifted = []
    for pt in actual_pts:
        step = pt["step"]
        if step in move_steps:
            drift_x += b_x + rng.normal(0, sigma_xy)
            drift_y += b_y + rng.normal(0, sigma_xy)
            drift_z += b_z + rng.normal(0, sigma_z)
        drifted.append({
            **pt,
            "drifted_x":    pt["actual_x"] + drift_x,
            "drifted_y":    pt["actual_y"] + drift_y,
            "drifted_z":    pt["actual_z"] + drift_z,
            "drift_mag_xy": round(np.hypot(drift_x, drift_y), 4),
        })
    return drifted


# ── Data extraction per run ────────────────────────────────────────────────────

def _extract_run_data(rows: list[dict], run_num: int) -> dict:
    """
    Extract commanded + actual + all three sensor scenario positions for one run.

    Returns {
      "commanded":  list of {step, cmd_x, cmd_y, cmd_z, tool_name},
      "actual":     list of {step, actual_x, actual_y, actual_z, tool_name},
                    — this IS the tof_flow scenario (Kalman-filtered sim output)
      "tof_flow":   same as actual (no additional drift)
      "tof_only":   list with drifted_{x,y,z}, drift_mag_xy added (XY drift only)
      "imu_only":   list with drifted_{x,y,z}, drift_mag_xy added (XY + alt drift)
    }
    """
    run_rows = [r for r in rows if str(r.get("run", "")) == str(run_num)
                and r.get("is_drone_command", "0") == "1"]
    if not run_rows:
        return {k: [] for k in ["commanded", "actual", "tof_flow", "tof_only", "imu_only"]}

    # Commanded positions
    commanded = _commanded_positions(run_rows)

    # Actual positions from telemetry_after (= ToF + optical flow Kalman output)
    actual = []
    for i, r in enumerate(run_rows):
        tel = _parse_tel(r.get("telemetry_after", "{}"))
        if not tel:
            continue
        actual.append({
            "step":     i,
            "actual_x": float(tel.get("pos_x", tel.get("ekf_x", 0.0))),
            "actual_y": float(tel.get("pos_y", tel.get("ekf_y", 0.0))),
            "actual_z": float(tel.get("alt_m", tel.get("ekf_z", 0.0))),
            "tool_name":r.get("tool_name", ""),
        })

    # Steps where XY movement commands ran (drift accumulates on these)
    move_steps = {i for i, r in enumerate(run_rows)
                  if r.get("tool_name", "") in XY_MOVE_CMDS}

    # Seed per run — same base seed for all scenarios so trajectories are comparable
    base_seed = run_num * 7 + 13

    scenarios = {}
    for key in DRIFT_ORDER:
        profile = DRIFT_PROFILES[key]
        rng = np.random.default_rng(seed=base_seed)   # reset so each profile starts same
        scenarios[key] = _apply_drift(actual, rng, move_steps, profile)

    return {
        "commanded": commanded,
        "actual":    actual,           # convenience alias = tof_flow
        **scenarios,
    }


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _add_room_outline(ax, size_m: float = 3.0):
    """Draw a simple square room outline at (0,0)–(size_m, size_m)."""
    rect = plt.Rectangle((-0.1, -0.1), size_m + 0.2, size_m + 0.2,
                          fill=False, edgecolor="#CCCCCC", linewidth=1.0, linestyle="--")
    ax.add_patch(rect)
    ax.text(size_m / 2, -0.3, "Room outline (3 m × 3 m)",
            ha="center", va="top", fontsize=7, color="#AAAAAA")


def _plot_xy_three_scenarios(ax, cmd_pts, scenario_data: dict, orch_color: str,
                              show_legend: bool = True, alpha_scale: float = 1.0):
    """
    Plot commanded path + three sensor scenarios on one XY axes.

    scenario_data: dict keyed by "tof_flow", "tof_only", "imu_only"
    orch_color   : orchestrator colour for tof_flow trace
    """
    # Commanded target positions (faint dashed)
    cx = [p["cmd_x"] for p in cmd_pts]
    cy = [p["cmd_y"] for p in cmd_pts]
    if cx:
        ax.plot(cx, cy, "--", color=orch_color, alpha=0.25 * alpha_scale,
                linewidth=0.8,
                label="Commanded" if show_legend else None, zorder=2)

    # Three sensor scenarios
    for key in DRIFT_ORDER:
        pts    = scenario_data.get(key, [])
        prof   = DRIFT_PROFILES[key]
        color  = orch_color if key == "tof_flow" else DRIFT_COLORS[key]
        ls     = prof["linestyle"]
        lw     = 2.0 if key == "tof_flow" else 1.5
        alpha  = prof["alpha"] * alpha_scale
        label  = prof["label"] if show_legend else None

        dx = [p["drifted_x"] for p in pts]
        dy = [p["drifted_y"] for p in pts]
        if dx:
            ax.plot(dx, dy, ls, color=color, linewidth=lw, alpha=alpha,
                    label=label, zorder=4 if key == "tof_flow" else 3)

        # Start / end markers on tof_flow only
        if key == "tof_flow" and dx:
            ax.scatter([dx[0]], [dy[0]], marker="o", color="green", s=60, zorder=6)
            ax.scatter([dx[-1]], [dy[-1]], marker="s", color="red", s=60, zorder=6)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Top-down XY flight path — 4 orchestrators × 2 scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def plot_xy_paths(wf_rows: list[dict], out_dir: pathlib.Path, ts: str):
    """XY flight path per orchestrator × scenario.
       Three sensor scenario traces + commanded path.
       Run 1 is full opacity; runs 2-5 are faded behind.
    """
    n_orch = len(ORCHESTRATORS)
    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(n_orch, n_scen,
                              figsize=(5.5 * n_scen, 4.5 * n_orch),
                              squeeze=False)
    fig.suptitle(
        "ND2: Top-Down XY Flight Path\n"
        "dashed=commanded  ─ ToF+OptFlow(sim)  -- ToF only  ··· IMU only",
        fontsize=12, y=0.995)

    for oi, orch in enumerate(ORCHESTRATORS):
        color = ORCH_COLORS[orch]
        for si, scen in enumerate(SCENARIOS):
            ax = axes[oi][si]
            _add_room_outline(ax)

            orch_rows = [r for r in wf_rows
                         if r.get("orchestrator") == orch
                         and r.get("scenario") == scen]
            runs = sorted({int(r["run"]) for r in orch_rows if r.get("run")})

            for run in runs:
                d = _extract_run_data(orch_rows, run)
                if not d["actual"]:
                    continue
                alpha  = 1.0 if run == runs[0] else 0.15
                legend = (run == runs[0] and oi == 0 and si == 0)
                _plot_xy_three_scenarios(ax, d["commanded"],
                                         {k: d[k] for k in DRIFT_ORDER},
                                         color, show_legend=legend,
                                         alpha_scale=alpha)

            ax.set_title(f"{orch} / {scen}", fontsize=9, fontweight="bold")
            ax.set_xlabel("X position (m)", fontsize=8)
            ax.set_ylabel("Y position (m)", fontsize=8)
            if oi == 0 and si == 0:
                ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"ND2_xy_path_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Altitude vs step
# ═══════════════════════════════════════════════════════════════════════════════

def plot_altitude(wf_rows: list[dict], out_dir: pathlib.Path, ts: str):
    """Altitude vs command step — three sensor scenarios per subplot."""
    n_orch = len(ORCHESTRATORS)
    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(n_orch, n_scen,
                              figsize=(5.5 * n_scen, 3.5 * n_orch),
                              squeeze=False)
    fig.suptitle(
        "ND2: Altitude vs Command Step\n"
        "─ ToF+OptFlow(sim)  -- ToF only  ··· IMU only",
        fontsize=12, y=0.995)

    for oi, orch in enumerate(ORCHESTRATORS):
        color = ORCH_COLORS[orch]
        for si, scen in enumerate(SCENARIOS):
            ax = axes[oi][si]
            orch_rows = [r for r in wf_rows
                         if r.get("orchestrator") == orch
                         and r.get("scenario") == scen]
            runs = sorted({int(r["run"]) for r in orch_rows if r.get("run")})

            ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(0.02, 1.05, "target 1.0 m", fontsize=7, color="#999999",
                    transform=ax.get_yaxis_transform())

            for run in runs:
                d = _extract_run_data(orch_rows, run)
                if not d["actual"]:
                    continue
                alpha = 0.9 if run == runs[0] else 0.15
                lw    = 1.8 if run == runs[0] else 0.5

                # Commanded altitude
                steps_c = [p["step"] for p in d["commanded"] if "cmd_z" in p]
                alts_c  = [p["cmd_z"] for p in d["commanded"] if "cmd_z" in p]
                if steps_c:
                    ax.step(steps_c, alts_c, "--", where="post", color=color,
                            alpha=alpha * 0.3, linewidth=0.8,
                            label="Commanded" if (run == runs[0] and oi == 0 and si == 0) else None)

                # Three sensor scenarios
                for key in DRIFT_ORDER:
                    pts   = d.get(key, [])
                    prof  = DRIFT_PROFILES[key]
                    sc    = color if key == "tof_flow" else DRIFT_COLORS[key]
                    steps = [p["step"] for p in pts]
                    alts  = [p["drifted_z"] for p in pts]
                    ax.plot(steps, alts, prof["linestyle"], color=sc,
                            linewidth=lw, alpha=prof["alpha"] * alpha,
                            label=prof["label"] if (run == runs[0] and oi == 0 and si == 0) else None)

            ax.set_title(f"{orch} / {scen}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Command step", fontsize=8)
            ax.set_ylabel("Altitude (m)", fontsize=8)
            ax.set_ylim(-0.15, 2.2)
            ax.grid(True, alpha=0.3)
            if oi == 0 and si == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"ND2_altitude_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Drift magnitude vs step — shows how drift grows during patrol
# ═══════════════════════════════════════════════════════════════════════════════

def plot_drift_error(wf_rows: list[dict], out_dir: pathlib.Path, ts: str):
    """
    Accumulated XY drift magnitude vs step for all three sensor scenarios.
    Shows mean ± min/max band across 5 runs per scenario.
    All three scenarios plotted on the SAME axes per subplot so the
    separation between sensor quality levels is immediately visible.
    """
    n_orch = len(ORCHESTRATORS)
    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(n_orch, n_scen,
                              figsize=(5.5 * n_scen, 3.2 * n_orch),
                              squeeze=False)
    fig.suptitle(
        "ND2: Accumulated XY Drift Magnitude vs Command Step\n"
        "ToF+OptFlow(sim) vs ToF-only vs IMU-only  |  mean ± range over 5 runs",
        fontsize=12, y=0.995)

    for oi, orch in enumerate(ORCHESTRATORS):
        for si, scen in enumerate(SCENARIOS):
            ax = axes[oi][si]
            orch_rows = [r for r in wf_rows
                         if r.get("orchestrator") == orch
                         and r.get("scenario") == scen]
            runs = sorted({int(r["run"]) for r in orch_rows if r.get("run")})

            for key in DRIFT_ORDER:
                prof  = DRIFT_PROFILES[key]
                dc    = DRIFT_COLORS[key] or ORCH_COLORS[orch]
                ls    = prof["linestyle"]

                all_mags = []
                for run in runs:
                    d = _extract_run_data(orch_rows, run)
                    pts = d.get(key, [])
                    if not pts:
                        continue
                    mags = [p["drift_mag_xy"] for p in pts]
                    ax.plot(range(len(mags)), mags, ls, color=dc,
                            alpha=0.15, linewidth=0.6)
                    all_mags.append(mags)

                if all_mags:
                    max_len = max(len(m) for m in all_mags)
                    padded  = np.array([m + [m[-1]] * (max_len - len(m))
                                        for m in all_mags])
                    mean_d  = padded.mean(axis=0)
                    ax.plot(range(max_len), mean_d, ls, color=dc,
                            linewidth=2.0, alpha=0.9, label=prof["label"])
                    ax.fill_between(range(max_len),
                                    padded.min(axis=0), padded.max(axis=0),
                                    color=dc, alpha=0.08)

            # Reference thresholds
            ax.axhline(0.05, color="#FF8C00", linestyle=":", linewidth=1.0, alpha=0.7)
            ax.text(0.5, 0.052, "5 cm — patrol accuracy limit",
                    fontsize=6.5, color="#FF8C00", transform=ax.get_yaxis_transform())
            ax.axhline(0.20, color="red", linestyle=":", linewidth=1.0, alpha=0.7)
            ax.text(0.5, 0.205, "20 cm — room width / 15",
                    fontsize=6.5, color="red", transform=ax.get_yaxis_transform())

            ax.set_title(f"{orch} / {scen}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Command step", fontsize=8)
            ax.set_ylabel("XY drift magnitude (m)", fontsize=8)
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.3)
            if oi == 0 and si == 0:
                ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"ND2_drift_error_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Combined overlay — run 1 of all orchestrators on one XY plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_combined_overlay(wf_rows: list[dict], out_dir: pathlib.Path, ts: str):
    """All 4 orchestrators on one XY plot per scenario — run 1 only.
       Shows ToF+OptFlow trace per orchestrator (no cross-scenario drift clutter).
    """
    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(1, n_scen, figsize=(6.5 * n_scen, 5.5), squeeze=False)
    fig.suptitle("ND2: All Orchestrators — Run 1 Flight Path Overlay\n"
                 "solid = ToF+OptFlow(sim)  ··· = IMU only",
                 fontsize=12, y=1.01)

    for si, scen in enumerate(SCENARIOS):
        ax = axes[0][si]
        _add_room_outline(ax)

        for orch in ORCHESTRATORS:
            color     = ORCH_COLORS[orch]
            orch_rows = [r for r in wf_rows
                         if r.get("orchestrator") == orch
                         and r.get("scenario") == scen]
            d = _extract_run_data(orch_rows, run_num=1)
            if not d["actual"]:
                continue

            # ToF+OptFlow (solid, orchestrator colour)
            tf_pts = d.get("tof_flow", [])
            ax_x = [p["drifted_x"] for p in tf_pts]
            ay   = [p["drifted_y"] for p in tf_pts]
            ax.plot(ax_x, ay, "-", color=color, linewidth=2.2, label=orch, zorder=4)

            # IMU only (dotted, same colour, faint)
            im_pts = d.get("imu_only", [])
            ix = [p["drifted_x"] for p in im_pts]
            iy = [p["drifted_y"] for p in im_pts]
            ax.plot(ix, iy, ":", color=color, linewidth=1.4, alpha=0.6, zorder=3)

            if ax_x:
                ax.scatter([ax_x[0]], [ay[0]], marker="o", color=color, s=80, zorder=6)
                ax.scatter([ax_x[-1]], [ay[-1]], marker="s", color=color, s=80, zorder=6)

        patches = [mpatches.Patch(color=ORCH_COLORS[o], label=o) for o in ORCHESTRATORS]
        line_leg = [
            plt.Line2D([0],[0], color="grey", lw=2.0, ls="-",  label="ToF+OptFlow"),
            plt.Line2D([0],[0], color="grey", lw=1.4, ls=":",  label="IMU only"),
        ]
        ax.legend(handles=patches + line_leg, fontsize=8, loc="upper right")
        ax.set_title(f"Scenario: {scen}", fontsize=10, fontweight="bold")
        ax.set_xlabel("X position (m)", fontsize=9)
        ax.set_ylabel("Y position (m)", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"ND2_overlay_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Altitude mean ± σ across 5 runs per orchestrator × scenario
# ═══════════════════════════════════════════════════════════════════════════════

def plot_altitude_mean_std(wf_rows: list[dict], out_dir: pathlib.Path, ts: str):
    n_orch = len(ORCHESTRATORS)
    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(1, n_scen, figsize=(6.5 * n_scen, 4.0), squeeze=False)
    fig.suptitle("ND2: Altitude Mean ± 1σ across 5 Runs\n"
                 "solid=actual · dotted=drifted",
                 fontsize=12, y=1.01)

    for si, scen in enumerate(SCENARIOS):
        ax = axes[0][si]
        ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)

        for orch in ORCHESTRATORS:
            color     = ORCH_COLORS[orch]
            orch_rows = [r for r in wf_rows
                         if r.get("orchestrator") == orch
                         and r.get("scenario") == scen]
            runs = sorted({int(r["run"]) for r in orch_rows if r.get("run")})

            all_actual  = []
            all_drifted = []
            for run in runs:
                d = _extract_run_data(orch_rows, run)
                if d["actual"]:
                    all_actual.append([p["actual_z"] for p in d["actual"]])
                if d["drifted"]:
                    all_drifted.append([p["drifted_z"] for p in d["drifted"]])

            if not all_actual:
                continue

            max_len = max(len(a) for a in all_actual)
            padded_a = np.array([a + [a[-1]] * (max_len - len(a)) for a in all_actual])
            mean_a   = padded_a.mean(axis=0)
            std_a    = padded_a.std(axis=0)
            steps    = np.arange(max_len)

            ax.plot(steps, mean_a, "-", color=color, linewidth=2.0, label=f"{orch} actual")
            ax.fill_between(steps, mean_a - std_a, mean_a + std_a, color=color, alpha=0.12)

            if all_drifted:
                padded_d = np.array([a + [a[-1]] * (max_len - len(a)) for a in all_drifted])
                mean_d   = padded_d.mean(axis=0)
                ax.plot(steps, mean_d, ":", color=color, linewidth=1.5, alpha=0.8,
                        label=f"{orch} drifted")

        ax.set_title(f"Scenario: {scen}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Command step", fontsize=9)
        ax.set_ylabel("Altitude (m)", fontsize=9)
        ax.set_ylim(-0.1, 2.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out = out_dir / f"ND2_altitude_mean_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: XY path + waypoint positions per orchestrator (run 1)
#           Shows commanded waypoints as X markers with numeric labels
# ═══════════════════════════════════════════════════════════════════════════════

def plot_waypoint_map(wf_rows: list[dict], out_dir: pathlib.Path, ts: str):
    """Show commanded waypoints explicitly — where the LLM told the drone to go."""
    n_orch = len(ORCHESTRATORS)
    n_scen = len(SCENARIOS)
    fig, axes = plt.subplots(n_orch, n_scen,
                              figsize=(5 * n_scen, 4.5 * n_orch),
                              squeeze=False)
    fig.suptitle("ND2: Commanded Waypoints vs Actual Path (Run 1)\n"
                 "× = commanded · ○ start · □ land · solid = actual · dotted = drifted",
                 fontsize=12, y=0.995)

    for oi, orch in enumerate(ORCHESTRATORS):
        color = ORCH_COLORS[orch]
        for si, scen in enumerate(SCENARIOS):
            ax = axes[oi][si]
            _add_room_outline(ax)

            orch_rows = [r for r in wf_rows
                         if r.get("orchestrator") == orch
                         and r.get("scenario") == scen]
            d = _extract_run_data(orch_rows, run_num=1)
            if not d["actual"]:
                ax.set_title(f"{orch} / {scen} (no data)", fontsize=9)
                continue

            # Actual path
            ax_x = [p["actual_x"] for p in d["actual"]]
            ay   = [p["actual_y"] for p in d["actual"]]
            ax.plot(ax_x, ay, "-", color=color, linewidth=2.0, alpha=0.85, zorder=4)

            # Drifted path
            dx_x = [p["drifted_x"] for p in d["drifted"]]
            dy_y = [p["drifted_y"] for p in d["drifted"]]
            ax.plot(dx_x, dy_y, ":", color=color, linewidth=1.5, alpha=0.7, zorder=3)

            # Commanded waypoints (navigate_to_waypoint only)
            nav_pts = [p for p in d["commanded"]
                       if p["tool_name"] == "navigate_to_waypoint"]
            for k, wp in enumerate(nav_pts):
                ax.scatter(wp["cmd_x"], wp["cmd_y"],
                           marker="x", color="black", s=80, linewidths=2, zorder=6)
                ax.annotate(f"WP{k+1}\n({wp['cmd_x']:.2f},{wp['cmd_y']:.2f})",
                            (wp["cmd_x"], wp["cmd_y"]),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=6, color="black", zorder=7)

            # Start / land markers
            if ax_x:
                ax.scatter([ax_x[0]], [ay[0]], marker="o", color="green",
                           s=80, zorder=7, label="Start")
                ax.scatter([ax_x[-1]], [ay[-1]], marker="s", color="red",
                           s=80, zorder=7, label="Land")

            ax.set_title(f"{orch} / {scen}", fontsize=9, fontweight="bold")
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if oi == 0 and si == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"ND2_waypoints_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Sensor scenario comparison panel
#           One orchestrator × one scenario — three XY + three altitude subplots
#           side-by-side so the visual separation between sensor quality is clear
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensor_comparison(wf_rows: list[dict], out_dir: pathlib.Path, ts: str,
                            orch: str = "claude", scen: str = "clear_room"):
    """
    Three-column panel: [ToF+OptFlow] | [ToF only] | [IMU only]
    Top row = XY path; bottom row = altitude vs step.
    All runs overlaid (faded) + run-1 bold.
    Single orchestrator × single scenario for maximum clarity.
    """
    orch_rows = [r for r in wf_rows
                 if r.get("orchestrator") == orch
                 and r.get("scenario") == scen]
    runs  = sorted({int(r["run"]) for r in orch_rows if r.get("run")})
    color = ORCH_COLORS.get(orch, "steelblue")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    col_titles = [DRIFT_PROFILES[k]["label"] for k in DRIFT_ORDER]
    row_titles = ["XY Flight Path (top-down)", "Altitude vs Command Step"]

    for col, key in enumerate(DRIFT_ORDER):
        prof  = DRIFT_PROFILES[key]
        dc    = color if key == "tof_flow" else DRIFT_COLORS[key]
        ls    = prof["linestyle"]

        ax_xy  = axes[0][col]
        ax_alt = axes[1][col]

        _add_room_outline(ax_xy)
        ax_alt.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_alt.text(0.02, 1.05, "target 1.0 m", fontsize=7, color="#999999",
                    transform=ax_alt.get_yaxis_transform())

        for run in runs:
            d    = _extract_run_data(orch_rows, run)
            pts  = d.get(key, [])
            if not pts:
                continue
            a    = 0.9 if run == runs[0] else 0.15
            lw   = 2.0 if run == runs[0] else 0.6

            # XY
            dx = [p["drifted_x"] for p in pts]
            dy = [p["drifted_y"] for p in pts]
            ax_xy.plot(dx, dy, ls, color=dc, linewidth=lw, alpha=a)
            if run == runs[0] and dx:
                ax_xy.scatter([dx[0]], [dy[0]], marker="o", color="green", s=70, zorder=6)
                ax_xy.scatter([dx[-1]], [dy[-1]], marker="s", color="red",  s=70, zorder=6)

                # Commanded waypoints
                nav = [p for p in d["commanded"]
                       if p["tool_name"] == "navigate_to_waypoint"]
                for k_wp, wp in enumerate(nav):
                    ax_xy.scatter(wp["cmd_x"], wp["cmd_y"],
                                  marker="x", color="black", s=60, linewidths=1.5, zorder=7)
                    ax_xy.annotate(f"WP{k_wp+1}", (wp["cmd_x"], wp["cmd_y"]),
                                   xytext=(5, 5), textcoords="offset points",
                                   fontsize=6.5, color="black")

            # Altitude
            steps = [p["step"] for p in pts]
            alts  = [p["drifted_z"] for p in pts]
            ax_alt.plot(steps, alts, ls, color=dc, linewidth=lw, alpha=a)

        # Drift stats annotation
        all_mags = []
        for run in runs:
            d   = _extract_run_data(orch_rows, run)
            pts = d.get(key, [])
            if pts:
                all_mags.append([p["drift_mag_xy"] for p in pts])
        if all_mags:
            final_mags = [m[-1] for m in all_mags]
            mean_f = np.mean(final_mags)
            ax_xy.set_title(
                f"{col_titles[col]}\nFinal drift: {mean_f*100:.1f} cm (mean)",
                fontsize=9, fontweight="bold", color=dc)
        else:
            ax_xy.set_title(col_titles[col], fontsize=9, fontweight="bold", color=dc)

        ax_xy.set_xlabel("X (m)", fontsize=8)
        ax_xy.set_ylabel("Y (m)", fontsize=8)
        ax_xy.set_aspect("equal")
        ax_xy.grid(True, alpha=0.3)

        ax_alt.set_xlabel("Command step", fontsize=8)
        ax_alt.set_ylabel("Altitude (m)", fontsize=8)
        ax_alt.set_ylim(-0.15, 2.2)
        ax_alt.grid(True, alpha=0.3)

    # Row labels
    for row_i, rlab in enumerate(row_titles):
        axes[row_i][0].set_ylabel(f"{rlab}\n{axes[row_i][0].get_ylabel()}",
                                  fontsize=8)

    fig.suptitle(
        f"ND2 Sensor Scenario Comparison — {orch} / {scen}\n"
        f"Left: ToF + Optical Flow (sim)  |  Centre: ToF only  |  Right: IMU only\n"
        f"Shaded runs 2–5  ·  Bold = Run 1  ·  ✕ = commanded waypoints",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = out_dir / f"ND2_sensor_comparison_{orch}_{scen}_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Select input file ─────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        wf_path = pathlib.Path(sys.argv[1])
        if not wf_path.is_absolute():
            wf_path = RESULTS_DIR / wf_path
    else:
        wf_path = _find_latest_workflow_csv()

    if wf_path is None or not wf_path.exists():
        print("ERROR: No ND2_workflow_*.csv found in results/")
        print("       Run exp_ND2_agentic_vision_control.py first.")
        sys.exit(1)

    print(f"\nReading: {wf_path.name}")
    wf_rows = _load_csv(wf_path)
    print(f"  {len(wf_rows)} workflow rows loaded")

    # ── Output directory ──────────────────────────────────────────────────────
    ts_now  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"ND2_plots_{ts_now}"
    out_dir.mkdir(exist_ok=True)
    print(f"  Output: {out_dir}\n")

    # Quick data inventory
    orchs = sorted({r.get("orchestrator","?") for r in wf_rows})
    scens = sorted({r.get("scenario","?") for r in wf_rows})
    runs  = sorted({r.get("run","?") for r in wf_rows})
    print(f"  Orchestrators: {orchs}")
    print(f"  Scenarios    : {scens}")
    print(f"  Runs         : {runs}")
    print(f"\nGenerating plots:")

    # ── Generate all figures ──────────────────────────────────────────────────
    ts_file = wf_path.stem.replace("ND2_workflow_", "")   # original timestamp

    plot_xy_paths(wf_rows, out_dir, ts_file)
    plot_altitude(wf_rows, out_dir, ts_file)
    plot_drift_error(wf_rows, out_dir, ts_file)
    plot_combined_overlay(wf_rows, out_dir, ts_file)
    plot_altitude_mean_std(wf_rows, out_dir, ts_file)
    plot_waypoint_map(wf_rows, out_dir, ts_file)

    # Fig 7: sensor comparison panel for each orchestrator × scenario
    for orch in ORCHESTRATORS:
        for scen in SCENARIOS:
            orch_rows = [r for r in wf_rows
                         if r.get("orchestrator") == orch
                         and r.get("scenario") == scen]
            if orch_rows:
                plot_sensor_comparison(wf_rows, out_dir, ts_file,
                                       orch=orch, scen=scen)

    print(f"\n✓ All plots saved to: results/ND2_plots_{ts_now}/")
    print(f"  7 figure types:")
    print(f"    ND2_xy_path_*.png          — XY paths, 4 orch × 2 scenarios, 3 sensor traces")
    print(f"    ND2_altitude_*.png         — altitude vs step, 3 sensor traces")
    print(f"    ND2_drift_error_*.png      — drift magnitude: ToF+OF vs ToF-only vs IMU-only")
    print(f"    ND2_overlay_*.png          — all 4 orchestrators on one plot (run 1)")
    print(f"    ND2_altitude_mean_*.png    — mean ± 1σ altitude over 5 runs")
    print(f"    ND2_waypoints_*.png        — commanded waypoints annotated on actual path")
    print(f"    ND2_sensor_comparison_*.png— side-by-side 3-scenario panel per orch×scenario")
    print()

    # ── Drift model summary ───────────────────────────────────────────────────
    print("Sensor scenario drift models (sourced from drone_sim.py):")
    print()
    n_steps = 10  # typical 5-waypoint patrol ≈ 10 movement commands
    for key in DRIFT_ORDER:
        p = DRIFT_PROFILES[key]
        if p["sigma_xy"] == 0:
            print(f"  {p['label']:<45} — sim KF output (ToF 5mm RMS + flow 15mm/s RMS)")
        else:
            exp_xy  = np.sqrt(n_steps * p["sigma_xy"]**2 + (p["bias_xy"] * n_steps)**2)
            exp_z   = np.sqrt(n_steps * p["sigma_z"]**2  + (p["bias_z"]  * n_steps)**2)
            print(f"  {p['label']:<45}  σ_xy={p['sigma_xy']*100:.1f}cm  "
                  f"bias={p['bias_xy']*100:.1f}cm/cmd")
            print(f"    → expected drift after {n_steps} cmds: "
                  f"XY={exp_xy*100:.1f} cm  alt={exp_z*100:.1f} cm")


if __name__ == "__main__":
    main()
