"""
EXP-E2: Human-in-Loop vs Full-Auto Mission Time
================================================
Goal:
    Run the same 5-step mission (arm → 1m → hold 5s → rotate 90° → land) in
    two modes and measure total mission time, approval wait time, and trajectory
    quality. 3 independent runs per mode.

    Mode A — human_loop  : operator must approve every action (simulated 1–3 s delay)
    Mode B — full_auto   : no human, LLM executes immediately

Metrics:
    - total_mission_time_s  : wall-clock to mission complete (bootstrap CI)
    - approval_wait_s       : total time spent waiting for approval (human_loop only)
    - trajectory_rmse_cm    : altitude RMSE during 5s hold phase (bootstrap CI)
    - api_calls             : per run (bootstrap CI)
    - cost_usd              : per run (bootstrap CI)
    - mission_complete      : reached all 5 steps (Wilson CI)

Paper References:
    - ReAct (Yao et al. 2022): autonomy loop in both modes
    - InnerMonologue (Huang et al. 2022): human approval as observation
    - SayCan (Ahn et al. 2022): affordance-grounded action approval
"""

import os, sys, json, time, csv, math, pathlib, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS          = 5
APPROVAL_DELAY  = (1.0, 3.0)   # simulated human approval: uniform [lo, hi] seconds
MISSION_STEPS   = 5
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
    "SayCan":         "Ahn et al. 2022 — Do As I Can, Not As I Say: Grounding Language in Robotic Affordances",
}

MISSION_GOAL = (
    "Execute a 5-step mission: "
    "(1) arm the drone, "
    "(2) take off and hover at 1.0 m, "
    "(3) hold position for 5 seconds, "
    "(4) rotate 90 degrees clockwise, "
    "(5) land and disarm. "
    "Confirm each step before proceeding. Announce 'MISSION COMPLETE' when done."
)

MODES = [
    {"name": "human_loop", "mode": "human_loop"},
    {"name": "full_auto",  "mode": "full_auto"},
]

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
def run_once(run_idx: int, mode_cfg: dict) -> dict:
    agent = DAgent(session_id=f"E2_r{run_idx}_{mode_cfg['name']}")
    agent.scene_sim.set_scene("open_space")

    total_approval_wait = 0.0
    hold_altitudes      = []
    steps_completed     = 0
    mission_complete    = False

    def approve_callback(action_name: str, _args: dict) -> tuple[bool, str]:
        nonlocal total_approval_wait
        delay = random.uniform(*APPROVAL_DELAY)
        time.sleep(delay)   # simulate human review time
        total_approval_wait += delay
        return True, ""     # always approve in this experiment

    def on_iteration_end(iteration: int, last_reply: str, tool_calls: list):
        nonlocal steps_completed, mission_complete
        for tc in tool_calls:
            if tc.get("name") in ("arm","set_altitude_target","hold_position",
                                   "rotate_yaw","land","disarm"):
                steps_completed += 1
        # Record altitude during hold phase
        if agent.state.armed:
            hold_altitudes.append(agent.state.z)
        if "MISSION COMPLETE" in last_reply.upper():
            mission_complete = True
            return True
        return False

    t0 = time.time()
    reply, stats, trace = agent.autonomy_loop(
        goal             = MISSION_GOAL,
        mode             = mode_cfg["mode"],
        max_iterations   = 25,
        approve_callback = approve_callback if mode_cfg["mode"] == "human_loop" else None,
        iteration_hook   = on_iteration_end,
    )
    wall_time = time.time() - t0

    if "MISSION COMPLETE" in reply.upper():
        mission_complete = True

    # Altitude RMSE during hold phase (target 1.0 m)
    rmse_cm = 0.0
    if hold_altitudes:
        errs = [(z - 1.0) * 100.0 for z in hold_altitudes]
        rmse_cm = float(np.sqrt(np.mean(np.array(errs)**2)))

    row = {
        "run":                  run_idx,
        "mode":                 mode_cfg["name"],
        "mission_complete":     int(mission_complete),
        "total_time_s":         round(wall_time, 3),
        "approval_wait_s":      round(total_approval_wait, 3),
        "llm_time_s":           round(wall_time - total_approval_wait, 3),
        "trajectory_rmse_cm":   round(rmse_cm, 3),
        "steps_completed":      min(steps_completed, MISSION_STEPS),
        "api_calls":            stats.get("api_calls", 0),
        "tokens_in":            stats.get("tokens_in", 0),
        "tokens_out":           stats.get("tokens_out", 0),
        "cost_usd":             round(stats.get("cost_usd", 0.0), 6),
    }
    status = "DONE" if mission_complete else "FAIL"
    print(f"  [E2 run={run_idx} {mode_cfg['name']:12s}] {status} "
          f"total={wall_time:.1f}s wait={total_approval_wait:.1f}s "
          f"rmse={rmse_cm:.2f}cm")
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-E2: Human-in-Loop vs Full-Auto Mission Time")
    print(f"N_RUNS={N_RUNS} per mode, approval_delay={APPROVAL_DELAY}s")
    print("=" * 60)

    all_rows = []
    for mode_cfg in MODES:
        print(f"\n=== Mode: {mode_cfg['name']} ===")
        for run in range(1, N_RUNS + 1):
            print(f"--- Run {run}/{N_RUNS} ---")
            all_rows.append(run_once(run, mode_cfg))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "E2_runs.csv"
    fields   = ["run","mode","mission_complete","total_time_s","approval_wait_s",
                "llm_time_s","trajectory_rmse_cm","steps_completed",
                "api_calls","tokens_in","tokens_out","cost_usd"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Stats per mode ─────────────────────────────────────────────────────────
    mode_stats  = {}
    summary_csv = OUT_DIR / "E2_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["mode","metric","value","ci_lo","ci_hi","note"])
        for mc in MODES:
            mname = mc["name"]
            mr    = [r for r in all_rows if r["mode"] == mname]
            k_mc  = sum(r["mission_complete"] for r in mr)
            mc_r, mc_lo, mc_hi = wilson_ci(k_mc, len(mr))

            tt_m, tt_lo, tt_hi   = bootstrap_ci([r["total_time_s"]       for r in mr])
            aw_m, aw_lo, aw_hi   = bootstrap_ci([r["approval_wait_s"]    for r in mr])
            ll_m, ll_lo, ll_hi   = bootstrap_ci([r["llm_time_s"]         for r in mr])
            rms_m,rm_lo, rm_hi   = bootstrap_ci([r["trajectory_rmse_cm"] for r in mr])
            cost_m,c_lo, c_hi    = bootstrap_ci([r["cost_usd"]           for r in mr])

            mode_stats[mname] = {
                "mc":mc_r,"mc_lo":mc_lo,"mc_hi":mc_hi,
                "tt":tt_m,"tt_lo":tt_lo,"tt_hi":tt_hi,
                "aw":aw_m,"aw_lo":aw_lo,"aw_hi":aw_hi,
                "rms":rms_m,"rm_lo":rm_lo,"rm_hi":rm_hi,
            }

            cw.writerow([mname,"mission_complete_rate",mc_r, mc_lo, mc_hi, "Wilson 95%"])
            cw.writerow([mname,"total_time_s",         tt_m, tt_lo, tt_hi, "Bootstrap 95%"])
            cw.writerow([mname,"approval_wait_s",      aw_m, aw_lo, aw_hi, "Bootstrap 95%"])
            cw.writerow([mname,"llm_time_s",           ll_m, ll_lo, ll_hi, "Bootstrap 95%"])
            cw.writerow([mname,"trajectory_rmse_cm",   rms_m,rm_lo, rm_hi, "Bootstrap 95%"])
            cw.writerow([mname,"cost_usd",             cost_m,c_lo, c_hi,  "Bootstrap 95%"])

        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mnames = [m["name"] for m in MODES]
        labels = ["Human-in-Loop", "Full-Auto"]
        colors = ["#e67e22", "#2ecc71"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        ax = axes[0]
        tts  = [mode_stats[m]["tt"]    for m in mnames]
        aws  = [mode_stats[m]["aw"]    for m in mnames]
        llms = [mode_stats[m]["tt"] - mode_stats[m]["aw"] for m in mnames]
        x    = range(len(mnames))
        ax.bar(labels, llms, color="#3498db", label="LLM time")
        ax.bar(labels, aws, bottom=llms, color="#e74c3c", label="Approval wait")
        tt_err = [[mode_stats[m]["tt"] - mode_stats[m]["tt_lo"] for m in mnames],
                  [mode_stats[m]["tt_hi"] - mode_stats[m]["tt"] for m in mnames]]
        ax.errorbar(labels, tts, yerr=tt_err, fmt="none", capsize=6, color="black")
        ax.set_ylabel("Time (s)")
        ax.set_title("E2: Mission Time Breakdown")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        rms  = [mode_stats[m]["rms"]   for m in mnames]
        rms_err = [[mode_stats[m]["rms"] - mode_stats[m]["rm_lo"] for m in mnames],
                   [mode_stats[m]["rm_hi"] - mode_stats[m]["rms"] for m in mnames]]
        ax2.bar(labels, rms, color=colors, yerr=rms_err, capsize=8)
        ax2.set_ylabel("Altitude RMSE (cm)")
        ax2.set_title("E2: Trajectory Quality (Hold Phase)")

        ax3 = axes[2]
        slow = mode_stats[mnames[0]]["tt"] / max(mode_stats[mnames[1]]["tt"], 0.001)
        ax3.barh(["Slowdown\nFactor"], [slow], color="#e74c3c")
        ax3.axvline(1.0, color="green", linestyle="--")
        ax3.set_xlabel("human_loop time / full_auto time")
        ax3.set_title(f"E2: Human-Loop Slowdown\n{slow:.1f}× slower")

        fig.suptitle(
            "EXP-E2 Human-in-Loop vs Full-Auto Mission Time\n"
            "ReAct (Yao 2022), Inner Monologue (Huang 2022), SayCan (Ahn 2022)",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "E2_human_vs_auto_time.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── E2 Summary ──────────────────────────────────────────────────────")
    for mn in mnames:
        s = mode_stats[mn]
        print(f"  {mn:14s}: total={s['tt']:.1f}s [{s['tt_lo']:.1f},{s['tt_hi']:.1f}]"
              f"  wait={s['aw']:.1f}s  rmse={s['rms']:.2f}cm")


if __name__ == "__main__":
    main()
