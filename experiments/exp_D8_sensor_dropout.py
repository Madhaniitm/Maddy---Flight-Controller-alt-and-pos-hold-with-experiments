"""
EXP-D8: ToF Sensor Dropout Handling
=====================================
Goal:
    Simulate a mid-hover ToF sensor dropout (inject_sensor_dropout) while the
    drone holds 1.0 m altitude. Measure whether the LLM agent detects the dropout,
    issues a hold/land command, and recovers (or safely descends) without crash.
    N=5 independent runs; dropout injected at t=3 s into a 10 s hover.

Metrics:
    - detected_dropout     : agent mentioned sensor dropout / invalid data (Wilson CI)
    - issued_safe_action   : agent called hold_position, land, or disarm (Wilson CI)
    - altitude_deviation_cm: max altitude deviation during dropout window (bootstrap CI)
    - recovery_time_s      : time from dropout injection to safe action (bootstrap CI)
    - api_calls            : per-run (bootstrap CI)
    - cost_usd             : per-run (bootstrap CI)

Paper References:
    - ReAct (Yao et al. 2022): reason-act-observe loop for fault response
    - InnerMonologue (Huang et al. 2022): sensor observations drive replanning
    - Vemprala2023: ChatGPT for robotics, fault-tolerant tool use
"""

import os, sys, json, time, csv, math, pathlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS          = 5
HOVER_TARGET_M  = 1.0
DROPOUT_AT_S    = 3.0   # inject dropout after 3 s of stable hover
HOVER_TOTAL_S   = 10.0
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

DROPOUT_PROMPT = (
    "You are monitoring the drone during a hover at 1.0 m. "
    "Check sensor status with get_sensor_status. "
    "If any sensor reports invalid data or dropout, immediately issue a safe response: "
    "hold position, reduce altitude, or land. "
    "Explain which sensor failed and what action you took."
)

SAFE_ACTION_NAMES = {"hold_position", "set_altitude_target", "land", "disarm",
                     "emergency_land", "stop_movement"}

DROPOUT_KEYWORDS = ["dropout", "invalid", "no data", "sensor fail", "tof fail",
                    "range invalid", "altitude unknown", "lost altitude"]

# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(p, 4), round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)

def bootstrap_ci(data: list, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr   = np.array(data, dtype=float)
    boots = [stat(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)), 4), round(float(lo), 4), round(float(hi), 4)

# ── Single run ─────────────────────────────────────────────────────────────────
def run_once(run_idx: int) -> dict:
    agent = DAgent(session_id=f"D8_r{run_idx}")
    agent.execute_tool("arm", {})
    agent.execute_tool("find_hover_throttle", {})
    agent.execute_tool("enable_altitude_hold", {})
    agent.execute_tool("set_altitude_target", {"target_m": HOVER_TARGET_M})

    # Stable hover for DROPOUT_AT_S seconds (physics steps)
    steps_pre = int(DROPOUT_AT_S * 50)   # 50 Hz
    pre_alts  = []
    for _ in range(steps_pre):
        agent.physics_step()
        pre_alts.append(agent.state.z)

    # Inject dropout
    agent.execute_tool("inject_sensor_dropout", {"sensor": "tof", "duration_s": 5.0})
    dropout_injected_at = time.time()

    # Record altitude during dropout window
    steps_dropout = int((HOVER_TOTAL_S - DROPOUT_AT_S) * 50)
    dropout_alts  = []
    for _ in range(steps_dropout):
        agent.physics_step()
        dropout_alts.append(agent.state.z)

    max_deviation_cm = float(np.max(np.abs(
        np.array(dropout_alts) - HOVER_TARGET_M
    ))) * 100.0

    # Ask LLM to handle the situation
    t0 = time.time()
    reply, stats, trace = agent.run_agent_loop(DROPOUT_PROMPT)
    recovery_time = time.time() - dropout_injected_at

    detected_dropout = int(any(kw in reply.lower() for kw in DROPOUT_KEYWORDS))

    # Check if any safe action was issued
    issued_safe = 0
    for step in trace:
        if step.get("role") == "tool_use" and step.get("name") in SAFE_ACTION_NAMES:
            issued_safe = 1
            break

    row = {
        "run":                  run_idx,
        "detected_dropout":     detected_dropout,
        "issued_safe_action":   issued_safe,
        "altitude_deviation_cm":round(max_deviation_cm, 3),
        "recovery_time_s":      round(recovery_time, 3),
        "api_calls":            stats.get("api_calls", 0),
        "tokens_in":            stats.get("tokens_in", 0),
        "tokens_out":           stats.get("tokens_out", 0),
        "cost_usd":             round(stats.get("cost_usd", 0.0), 6),
    }
    det  = "DETECTED" if detected_dropout else "MISSED"
    safe = "SAFE_ACT" if issued_safe else "NO_ACT"
    print(f"  [D8 run={run_idx}] {det} {safe} "
          f"dev={max_deviation_cm:.1f}cm rec={recovery_time:.1f}s")
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D8: ToF Sensor Dropout Handling")
    print(f"N_RUNS={N_RUNS}, dropout@t={DROPOUT_AT_S}s")
    print("=" * 60)

    rows = []
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")
        rows.append(run_once(run))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D8_runs.csv"
    fields   = ["run","detected_dropout","issued_safe_action","altitude_deviation_cm",
                "recovery_time_s","api_calls","tokens_in","tokens_out","cost_usd"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Statistics ─────────────────────────────────────────────────────────────
    k_det  = sum(r["detected_dropout"]   for r in rows)
    k_safe = sum(r["issued_safe_action"] for r in rows)
    dr, dr_lo, dr_hi  = wilson_ci(k_det,  N_RUNS)
    sr, sr_lo, sr_hi  = wilson_ci(k_safe, N_RUNS)

    dev_m, dv_lo, dv_hi = bootstrap_ci([r["altitude_deviation_cm"] for r in rows])
    rec_m, rc_lo, rc_hi = bootstrap_ci([r["recovery_time_s"]       for r in rows])
    cst_m, c_lo,  c_hi  = bootstrap_ci([r["cost_usd"]              for r in rows])

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "D8_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric","value","ci_lo","ci_hi","note"])
        cw.writerow(["dropout_detection_rate", dr,    dr_lo, dr_hi, "Wilson 95% CI"])
        cw.writerow(["safe_action_rate",        sr,    sr_lo, sr_hi, "Wilson 95% CI"])
        cw.writerow(["altitude_deviation_cm",   dev_m, dv_lo, dv_hi, "Bootstrap 95%"])
        cw.writerow(["recovery_time_s",         rec_m, rc_lo, rc_hi, "Bootstrap 95%"])
        cw.writerow(["cost_usd",                cst_m, c_lo,  c_hi,  "Bootstrap 95%"])
        cw.writerow(["n_runs",                  N_RUNS,"","",""])
        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        ax = axes[0]
        cats = ["Dropout\nDetected", "Safe Action\nIssued"]
        vals = [dr, sr]
        errs = [[dr - dr_lo, sr - sr_lo], [dr_hi - dr, sr_hi - sr]]
        clrs = ["#3498db", "#2ecc71"]
        ax.bar(cats, vals, yerr=errs, capsize=10, color=clrs)
        ax.set_ylim(0, 1.1)
        ax.set_title("D8: Detection & Response Rates")
        ax.set_ylabel("Rate (Wilson 95% CI)")

        ax2 = axes[1]
        ax2.scatter(range(1, N_RUNS+1), [r["altitude_deviation_cm"] for r in rows],
                    s=80, color="#e74c3c", zorder=3)
        ax2.axhline(dev_m, linestyle="--", color="navy",
                    label=f"Mean {dev_m:.1f}cm")
        ax2.set_xlabel("Run")
        ax2.set_ylabel("Max altitude deviation (cm)")
        ax2.set_title("D8: Altitude Deviation During Dropout")
        ax2.legend()

        ax3 = axes[2]
        ax3.scatter(range(1, N_RUNS+1), [r["recovery_time_s"] for r in rows],
                    s=80, color="#9b59b6", zorder=3)
        ax3.axhline(rec_m, linestyle="--", color="navy",
                    label=f"Mean {rec_m:.1f}s")
        ax3.set_xlabel("Run")
        ax3.set_ylabel("Recovery time (s)")
        ax3.set_title("D8: Time to Safe Action")
        ax3.legend()

        fig.suptitle(
            "EXP-D8 ToF Sensor Dropout Handling\n"
            "ReAct (Yao 2022), Inner Monologue (Huang 2022), Vemprala 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D8_sensor_dropout.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D8 Summary ──────────────────────────────────────────────────────")
    print(f"Dropout detection : {dr:.3f}  [{dr_lo:.3f},{dr_hi:.3f}] (Wilson 95% CI)")
    print(f"Safe action rate  : {sr:.3f}  [{sr_lo:.3f},{sr_hi:.3f}] (Wilson 95% CI)")
    print(f"Alt deviation     : {dev_m:.1f}cm [{dv_lo:.1f},{dv_hi:.1f}] (Bootstrap)")
    print(f"Recovery time     : {rec_m:.2f}s  [{rc_lo:.2f},{rc_hi:.2f}] (Bootstrap)")


if __name__ == "__main__":
    main()
