"""
EXP-E3: Conversation Memory — Multi-Turn State Retention
=========================================================
Goal:
    Verify that the LLM correctly retains information from Turn 1 across 20
    filler messages injected between Turn 1 and Turn 3. This validates that
    the inner-monologue message history is sufficient for long-horizon tasks.

    Protocol (per run):
        Turn 1 → "Set altitude target to 1.35 m" (LLM executes, records target)
        Turns 2–21 → 20 neutral filler messages (status queries, unrelated tools)
        Turn 22 → "What was the altitude target I set at the start?"
        Turn 23 → "Now set it back to 0.8 m" (tests re-use of recalled value)

    N=5 independent runs. Each run: fresh agent, same sequence.

Metrics:
    - recall_correct      : LLM correctly reported 1.35 m in Turn 22 (Wilson CI)
    - replan_correct      : LLM set target to 0.8 m in Turn 23 (Wilson CI)
    - recall_response_s   : latency of Turn 22 (bootstrap CI)
    - context_tokens      : total tokens at Turn 22 (shows context growth)

Paper References:
    - InnerMonologue (Huang et al. 2022): message history as persistent memory
    - ReAct (Yao et al. 2022): multi-turn reason-act with accumulated context
    - Vemprala2023: multi-turn robotics conversations
"""

import os, sys, json, time, csv, math, pathlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

# ── Config ─────────────────────────────────────────────────────────────────────
N_RUNS          = 5
N_FILLER_MSGS   = 20
ORIGINAL_ALT_M  = 1.35
RECALL_ANSWER_M = 1.35
RESET_ALT_M     = 0.80
RECALL_TOL_M    = 0.05    # accept ±5 cm variation in stated value
OUT_DIR         = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "InnerMonologue": "Huang et al. 2022 — Inner Monologue: Embodied Reasoning through Planning with LMs",
    "ReAct":          "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "Vemprala2023":   "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

FILLER_MESSAGES = [
    "What is the current battery voltage?",
    "Report roll and pitch angles.",
    "How many API calls have been made so far?",
    "Is the altitude hold active?",
    "What is the current ToF reading?",
    "List the tools available to you.",
    "What is the estimated hover throttle?",
    "Report motor RPM if available.",
    "What is the current yaw heading?",
    "Is the drone currently armed?",
    "Describe the current flight mode.",
    "What is the EKF altitude estimate?",
    "How long has the drone been hovering?",
    "What is the ambient temperature from the IMU?",
    "Is there any active fault or anomaly?",
    "Report the current PID setpoint.",
    "What is the current control loop frequency?",
    "Describe the last tool you called.",
    "What is the battery state of charge estimate?",
    "Confirm the altitude hold target is active.",
]

assert len(FILLER_MESSAGES) >= N_FILLER_MSGS

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

def extract_altitude_mention(text: str, target: float) -> bool:
    """Check if the LLM mentioned a value close to target in its response."""
    import re
    nums = re.findall(r"\d+\.\d+", text)
    for n in nums:
        if abs(float(n) - target) <= RECALL_TOL_M:
            return True
    # Also check integer metres
    nums_int = re.findall(r"\b\d+\b", text)
    for n in nums_int:
        if abs(float(n) - target) <= RECALL_TOL_M:
            return True
    return False

# ── Single run ─────────────────────────────────────────────────────────────────
def run_once(run_idx: int) -> dict:
    agent = DAgent(session_id=f"E3_r{run_idx}")
    agent.execute_tool("arm", {})
    agent.execute_tool("find_hover_throttle", {})
    agent.execute_tool("enable_altitude_hold", {})

    total_tokens_in  = 0
    total_tokens_out = 0

    def _ask(prompt: str) -> tuple[str, dict]:
        reply, stats, _ = agent.run_agent_loop(prompt)
        return reply, stats

    # Turn 1: set altitude
    print(f"  [E3 run={run_idx}] Turn 1: set altitude to {ORIGINAL_ALT_M}m")
    t1_reply, t1_stats = _ask(f"Set the altitude target to {ORIGINAL_ALT_M} metres.")
    total_tokens_in  += t1_stats.get("tokens_in", 0)
    total_tokens_out += t1_stats.get("tokens_out", 0)

    # Turns 2–21: filler messages
    for i, filler in enumerate(FILLER_MESSAGES[:N_FILLER_MSGS], start=2):
        _, f_stats = _ask(filler)
        total_tokens_in  += f_stats.get("tokens_in", 0)
        total_tokens_out += f_stats.get("tokens_out", 0)
        if (i - 1) % 5 == 0:
            print(f"  [E3 run={run_idx}] Filler {i-1}/{N_FILLER_MSGS}")

    # Turn 22: recall query
    t0_recall = time.time()
    recall_reply, r_stats = _ask(
        "What was the altitude target I asked you to set at the very beginning "
        "of this conversation? State the exact value in metres."
    )
    recall_latency = time.time() - t0_recall
    total_tokens_in  += r_stats.get("tokens_in", 0)
    total_tokens_out += r_stats.get("tokens_out", 0)
    recall_correct = int(extract_altitude_mention(recall_reply, RECALL_ANSWER_M))

    print(f"  [E3 run={run_idx}] Recall reply: {recall_reply[:80]!r}")
    print(f"  [E3 run={run_idx}] Recall {'CORRECT' if recall_correct else 'WRONG'}")

    # Turn 23: replan with recalled value
    replan_reply, rp_stats = _ask(
        f"Good. Now set the altitude target to {RESET_ALT_M} metres."
    )
    total_tokens_in  += rp_stats.get("tokens_in", 0)
    total_tokens_out += rp_stats.get("tokens_out", 0)
    replan_correct = int(abs(agent.state.target_z - RESET_ALT_M) < 0.05)

    return {
        "run":              run_idx,
        "recall_correct":   recall_correct,
        "replan_correct":   replan_correct,
        "recall_latency_s": round(recall_latency, 3),
        "context_tokens":   total_tokens_in,
        "total_tokens_in":  total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "n_filler_msgs":    N_FILLER_MSGS,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-E3: Conversation Memory — Multi-Turn State Retention")
    print(f"N_RUNS={N_RUNS}, FILLER_MSGS={N_FILLER_MSGS}, original_alt={ORIGINAL_ALT_M}m")
    print("=" * 60)

    rows = []
    for run in range(1, N_RUNS + 1):
        print(f"\n--- Run {run}/{N_RUNS} ---")
        rows.append(run_once(run))

    # ── Save per-run CSV ───────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "E3_runs.csv"
    fields   = ["run","recall_correct","replan_correct","recall_latency_s",
                "context_tokens","total_tokens_in","total_tokens_out","n_filler_msgs"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-run data  → {runs_csv}")

    # ── Statistics ─────────────────────────────────────────────────────────────
    k_rc = sum(r["recall_correct"] for r in rows)
    k_rp = sum(r["replan_correct"] for r in rows)
    rc, rc_lo, rc_hi = wilson_ci(k_rc, N_RUNS)
    rp, rp_lo, rp_hi = wilson_ci(k_rp, N_RUNS)

    lat_m, l_lo, l_hi = bootstrap_ci([r["recall_latency_s"] for r in rows])
    tok_m, t_lo, t_hi = bootstrap_ci([r["context_tokens"]   for r in rows])

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "E3_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric","value","ci_lo","ci_hi","note"])
        cw.writerow(["recall_accuracy",  rc,    rc_lo, rc_hi,
                     f"Wilson 95% CI — recalled {ORIGINAL_ALT_M}m after {N_FILLER_MSGS} fillers"])
        cw.writerow(["replan_accuracy",  rp,    rp_lo, rp_hi,
                     f"Wilson 95% CI — set to {RESET_ALT_M}m after recall"])
        cw.writerow(["recall_latency_s", lat_m, l_lo,  l_hi,  "Bootstrap 95%"])
        cw.writerow(["context_tokens",   tok_m, t_lo,  t_hi,  "Bootstrap 95% — total input tokens at Turn 22"])
        cw.writerow(["n_filler_msgs",    N_FILLER_MSGS, "","", "messages between Turn 1 and recall"])
        cw.writerow(["n_runs",           N_RUNS, "", "", ""])
        for key, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{key}", ref, "", "", ""])
    print(f"Summary data  → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        ax = axes[0]
        cats = [f"Recall\n({N_FILLER_MSGS} filler msgs)", "Re-plan\n(set 0.8m)"]
        vals = [rc, rp]
        errs = [[rc - rc_lo, rp - rp_lo], [rc_hi - rc, rp_hi - rp]]
        clrs = ["#2ecc71" if v >= 0.8 else "#e67e22" for v in vals]
        ax.bar(cats, vals, yerr=errs, capsize=10, color=clrs)
        ax.set_ylim(0, 1.2)
        ax.set_title(f"E3: Memory Accuracy (N={N_RUNS})")
        ax.set_ylabel("Rate (Wilson 95% CI)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11)

        ax2 = axes[1]
        ax2.scatter(range(1, N_RUNS+1), [r["context_tokens"] for r in rows], s=80, color="#3498db")
        ax2.axhline(tok_m, linestyle="--", color="navy", label=f"Mean {tok_m:.0f} tokens")
        ax2.set_xlabel("Run")
        ax2.set_ylabel("Context tokens at recall turn")
        ax2.set_title("E3: Context Size at Recall Turn")
        ax2.legend()

        fig.suptitle(
            f"EXP-E3 Memory Retention — {N_FILLER_MSGS} filler messages between encoding and recall\n"
            "Inner Monologue (Huang 2022), ReAct (Yao 2022), Vemprala 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "E3_memory_retention.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── E3 Summary ──────────────────────────────────────────────────────")
    print(f"Recall accuracy  : {rc:.3f}  [{rc_lo:.3f},{rc_hi:.3f}] (Wilson 95% CI)")
    print(f"Replan accuracy  : {rp:.3f}  [{rp_lo:.3f},{rp_hi:.3f}] (Wilson 95% CI)")
    print(f"Recall latency   : {lat_m:.3f}s [{l_lo:.3f},{l_hi:.3f}] (Bootstrap)")
    print(f"Context tokens   : {tok_m:.0f}   [{t_lo:.0f},{t_hi:.0f}] at recall turn")
    print(f"Filler messages  : {N_FILLER_MSGS} between Turn 1 and recall")


if __name__ == "__main__":
    main()
