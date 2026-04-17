"""
EXP-D6: Multi-LLM Anomaly Detection Comparison
===============================================
Goal:
    Inject 4 distinct faults into the simulated drone and test whether 4 LLM
    backends (Claude, GPT-4o, Gemini 1.5 Pro, LLaMA-3-70B) correctly detect
    each fault via detect_anomaly + telemetry analysis.
    N=3 independent runs per (fault × LLM) combination = 48 total runs.

Faults injected:
    1. motor_imbalance    — asymmetric motor output, roll drift
    2. battery_low        — voltage below safe threshold
    3. pid_oscillation    — high-frequency altitude oscillation (RMSE > 10 cm)
    4. gps_drift          — position estimate jumps (sim: random X/Y offset)

Metrics:
    - detection_rate      : fraction of correct fault identifications (Wilson CI)
    - latency_s           : time to first detect_anomaly result (bootstrap CI)
    - tokens_used         : tokens per run (bootstrap CI)
    - cost_usd            : per run (bootstrap CI)
    Broken down by: LLM backend × fault type

Paper References:
    - ReAct (Yao et al. 2022): reason-act-observe loop for anomaly diagnosis
    - Vemprala2023: ChatGPT for robotics, multi-LLM comparison
    - InnerMonologue (Huang et al. 2022): telemetry stream drives replanning
"""

import os, sys, json, time, csv, math, pathlib, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent, MultiLLMRunner, MULTI_LLM_MODELS

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS_PER_CELL = 5     # per (fault × LLM)
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
}

FAULT_TYPES = ["motor_imbalance", "battery_low", "pid_oscillation", "gps_drift"]
LLM_BACKENDS = list(MULTI_LLM_MODELS.keys())   # claude, gpt4o, gemini, llama3

# Keywords that indicate correct detection for each fault
FAULT_KEYWORDS = {
    "motor_imbalance": ["motor", "imbalance", "asymmetric", "roll drift"],
    "battery_low":     ["battery", "voltage", "low power", "power"],
    "pid_oscillation": ["oscillat", "pid", "vibrat", "unstable altitude"],
    "gps_drift":       ["gps", "position drift", "location jump", "navigation error"],
}

ANOMALY_PROMPT = (
    "Analyze the drone's current telemetry and camera data. Use detect_anomaly "
    "to identify any faults. Describe what fault you observe and its severity. "
    "Be specific about the type of fault."
)

# ── Fault injection helpers ────────────────────────────────────────────────────
def inject_fault(agent: DAgent, fault_type: str):
    """Modify agent state to simulate the fault."""
    if fault_type == "motor_imbalance":
        agent.state.roll_angle = random.uniform(8.0, 15.0)   # degrees
        agent.state.motor_speeds = [1400, 1400, 1600, 1600]  # asymmetric

    elif fault_type == "battery_low":
        agent.state.battery_v = random.uniform(3.3, 3.5)     # below 3.6V warning

    elif fault_type == "pid_oscillation":
        # Add oscillation to altitude buffer
        t = np.linspace(0, 2*math.pi, 50)
        noise = 0.12 * np.sin(8 * t)                          # 8 Hz, ±12 cm
        for dz in noise:
            agent._telemetry_buf.append(agent.state.z + dz)

    elif fault_type == "gps_drift":
        agent.state.x += random.uniform(0.8, 1.5)
        agent.state.y = getattr(agent.state, "y", 0.0) + random.uniform(0.8, 1.5)

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

def check_detection(reply: str, fault_type: str) -> bool:
    rl = reply.lower()
    return any(kw in rl for kw in FAULT_KEYWORDS[fault_type])

# ── Single run ─────────────────────────────────────────────────────────────────
def run_once(run_idx: int, fault: str, llm_key: str) -> dict:
    agent = DAgent(session_id=f"D6_r{run_idx}_{fault}_{llm_key}")
    agent.execute_tool("arm", {})
    agent.execute_tool("find_hover_throttle", {})
    agent.execute_tool("enable_altitude_hold", {})
    agent.execute_tool("set_altitude_target", {"target_m": 1.0})

    inject_fault(agent, fault)

    t0 = time.time()

    # Use multi-LLM runner if not claude, else standard agent loop
    if llm_key == "claude":
        reply, stats, trace = agent.run_agent_loop(ANOMALY_PROMPT)
    else:
        runner = MultiLLMRunner(llm_key=llm_key)
        reply, stats = runner.run(
            goal      = ANOMALY_PROMPT,
            tools     = agent.all_tools(),
            execute_fn= agent.execute_tool,
        )
        trace = []

    latency = time.time() - t0
    detected = int(check_detection(reply, fault))

    row = {
        "run":        run_idx,
        "fault":      fault,
        "llm":        llm_key,
        "detected":   detected,
        "latency_s":  round(latency, 3),
        "tokens_in":  stats.get("tokens_in", 0),
        "tokens_out": stats.get("tokens_out", 0),
        "cost_usd":   round(stats.get("cost_usd", 0.0), 6),
    }
    status = "DETECTED" if detected else "MISSED"
    print(f"  [D6 run={run_idx} {llm_key:8s} {fault:18s}] {status} lat={latency:.2f}s")
    return row

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-D6: Multi-LLM Anomaly Detection Comparison")
    print(f"N={N_RUNS_PER_CELL} runs/cell, {len(FAULT_TYPES)} faults × {len(LLM_BACKENDS)} LLMs")
    print("=" * 60)

    all_rows = []
    for fault in FAULT_TYPES:
        for llm in LLM_BACKENDS:
            print(f"\n=== fault={fault} llm={llm} ===")
            for run in range(1, N_RUNS_PER_CELL + 1):
                all_rows.append(run_once(run, fault, llm))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "D6_runs.csv"
    fields   = ["run","fault","llm","detected","latency_s","tokens_in","tokens_out","cost_usd"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Summary: detection rate per (fault × LLM) ────────────────────────────
    summary_csv = OUT_DIR / "D6_summary.csv"
    cell_stats  = {}
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["fault","llm","detection_rate","ci_lo","ci_hi",
                     "avg_latency_s","avg_cost_usd","note"])
        for fault in FAULT_TYPES:
            for llm in LLM_BACKENDS:
                cr = [r for r in all_rows if r["fault"] == fault and r["llm"] == llm]
                k  = sum(r["detected"] for r in cr)
                n  = len(cr)
                dr, lo, hi = wilson_ci(k, n)
                avg_lat    = round(np.mean([r["latency_s"] for r in cr]), 3)
                avg_cost   = round(np.mean([r["cost_usd"]  for r in cr]), 6)
                cell_stats[(fault, llm)] = {
                    "dr": dr, "lo": lo, "hi": hi,
                    "lat": avg_lat, "cost": avg_cost,
                }
                cw.writerow([fault, llm, dr, lo, hi, avg_lat, avg_cost,
                             f"Wilson 95%, N={n}"])
        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", "", "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_faults = len(FAULT_TYPES)
        n_llms   = len(LLM_BACKENDS)
        x        = np.arange(n_faults)
        width    = 0.2
        colors   = ["#3498db","#e67e22","#2ecc71","#9b59b6"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: detection rate heatmap-style grouped bar
        ax = axes[0]
        for i, llm in enumerate(LLM_BACKENDS):
            drs  = [cell_stats[(f, llm)]["dr"] for f in FAULT_TYPES]
            errs_lo = [cell_stats[(f, llm)]["dr"] - cell_stats[(f, llm)]["lo"] for f in FAULT_TYPES]
            errs_hi = [cell_stats[(f, llm)]["hi"] - cell_stats[(f, llm)]["dr"] for f in FAULT_TYPES]
            ax.bar(x + i*width, drs, width, yerr=[errs_lo, errs_hi],
                   label=llm, color=colors[i], capsize=4)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(FAULT_TYPES, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Detection Rate (Wilson 95% CI)")
        ax.set_title("D6: Detection Rate per Fault × LLM")
        ax.legend(fontsize=8)

        # Right: average latency per LLM
        ax2 = axes[1]
        avg_lat_per_llm = {llm: np.mean([cell_stats[(f, llm)]["lat"] for f in FAULT_TYPES])
                           for llm in LLM_BACKENDS}
        ax2.bar(LLM_BACKENDS, [avg_lat_per_llm[l] for l in LLM_BACKENDS], color=colors)
        ax2.set_xlabel("LLM Backend")
        ax2.set_ylabel("Avg latency (s)")
        ax2.set_title("D6: Average Latency per LLM")

        fig.suptitle(
            "EXP-D6 Multi-LLM Anomaly Detection\n"
            "ReAct (Yao 2022), Vemprala 2023, Inner Monologue (Huang 2022)",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "D6_anomaly_detection.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── D6 Summary ──────────────────────────────────────────────────────")
    print(f"{'Fault':20s} {'LLM':10s} {'Det.Rate':10s} {'95% CI':16s} {'Lat(s)'}")
    for fault in FAULT_TYPES:
        for llm in LLM_BACKENDS:
            s = cell_stats[(fault, llm)]
            print(f"  {fault:18s} {llm:10s} {s['dr']:.3f}      "
                  f"[{s['lo']:.3f},{s['hi']:.3f}]  {s['lat']:.2f}")


if __name__ == "__main__":
    main()
