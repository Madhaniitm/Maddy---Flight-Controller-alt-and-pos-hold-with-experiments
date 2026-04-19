"""
EXP-C3: Multi-Turn Conversational Mission  (N=5 runs)
======================================================
5-turn conversation: arm → 1.5m → hold 5s → rotate 90° → land
Reports per-turn success rate with Wilson 95% CI over N=5 runs.

Outputs:
  results/C3_runs.csv        — per-run × per-turn results
  results/C3_summary.csv     — per-turn success rate + CI
  results/C3_multiturn.png   — success-rate heatmap + altitude traces
"""

import sys, os, csv, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
# ── Guardrail toggle (--guardrail on|off) ──────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--guardrail", choices=["on", "off"], default="on")
_args, _ = _parser.parse_known_args()
GUARDRAIL_ENABLED = _args.guardrail == "on"
GUARDRAIL_SUFFIX  = "guardrail_on" if GUARDRAIL_ENABLED else "guardrail_off"

OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", f"C3_runs_{GUARDRAIL_SUFFIX}.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", f"C3_summary_{GUARDRAIL_SUFFIX}.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", f"C3_multiturn_{GUARDRAIL_SUFFIX}.png")

N_RUNS = 5

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Multi-turn loop where each turn is a full reason→act→observe cycle."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "LLM uses accumulated conversation history as implicit state tracker across turns."
    ),
}

TURNS = [
    {"turn": 1, "user": "arm the drone",
     "expected_tools": ["arm"],       "should_NOT_contain": [],
     "description": "Arm motors"},
    {"turn": 2, "user": "go to 1.5 metres",
     "expected_tools": ["find_hover_throttle", "set_altitude_target"],
     "should_NOT_contain": ["arm"],
     "description": "Takeoff + climb to 1.5 m"},
    {"turn": 3, "user": "hold there for 5 seconds",
     "expected_tools": ["wait"],
     "should_NOT_contain": ["arm", "takeoff", "find_hover_throttle"],
     "description": "Wait 5 s at altitude"},
    {"turn": 4, "user": "rotate 90 degrees clockwise",
     "expected_tools": ["set_yaw"],
     "should_NOT_contain": ["arm", "land"],
     "description": "Yaw 90° CW"},
    {"turn": 5, "user": "land now",
     "expected_tools": ["land", "disarm"],
     "should_NOT_contain": ["arm", "takeoff"],
     "description": "Safe landing"},
]
N_TURNS = len(TURNS)

# ── Statistics helpers ─────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

# ── Single-run function ────────────────────────────────────────────────────────

def run_once(run_idx):
    print(f"\n[C3] ── Run {run_idx+1}/{N_RUNS} ─────────────────────────────────")
    agent   = SimAgent(session_id=f"C3_run{run_idx}", guardrail_enabled=GUARDRAIL_ENABLED)
    history = []
    turn_rows = []

    for turn_spec in TURNS:
        turn_idx = turn_spec["turn"]
        user_msg = turn_spec["user"]

        with agent.state.lock:
            z_before       = round(agent.state.ekf_z, 3)
            yaw_before     = round(agent.state.yaw,   2)
            armed_before   = agent.state.armed
            althold_before = agent.state.althold
            altsp_before   = round(agent.state.alt_sp, 2)

        state_ctx = (
            f"[Drone state: armed={armed_before}, "
            f"althold={'ON' if althold_before else 'OFF'}, "
            f"alt={z_before:.2f}m, setpoint={altsp_before:.2f}m] "
        )
        user_msg_with_ctx = state_ctx + user_msg

        text, api_stats, tool_trace, _ = agent.run_agent_loop(
            user_msg_with_ctx, history=list(history), max_turns=20,
        )

        history.append({"role": "user", "content": user_msg_with_ctx})
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": text if text.strip() else "Done."}],
        })

        agent.wait_sim(3.0)

        with agent.state.lock:
            z_after    = round(agent.state.ekf_z, 3)
            yaw_after  = round(agent.state.yaw,   2)
            armed_after= agent.state.armed

        tools_used = [t["name"] for t in tool_trace]
        tools_set  = set(tools_used)

        expected_found  = [et for et in turn_spec["expected_tools"] if et in tools_set]
        forbidden_found = [nt for nt in turn_spec["should_NOT_contain"] if nt in tools_set]

        state_ok = True
        if turn_idx == 1:   state_ok = armed_after
        elif turn_idx == 2: state_ok = z_after > 0.8
        elif turn_idx == 3: state_ok = abs(z_after - z_before) < 0.50
        elif turn_idx == 4:
            yaw_delta = abs(yaw_after - yaw_before)
            state_ok  = yaw_delta > 20
        elif turn_idx == 5: state_ok = not armed_after

        sequence_ok  = len(expected_found) > 0 and len(forbidden_found) == 0
        overall_pass = sequence_ok and state_ok

        n_api  = len(api_stats)
        in_tok = sum(s["input_tokens"]  for s in api_stats)
        out_tok= sum(s["output_tokens"] for s in api_stats)

        print(f"    T{turn_idx}: {turn_spec['description']:30s}  "
              f"pass={overall_pass}  z:{z_before:.2f}→{z_after:.2f}m")

        turn_rows.append({
            "run":            run_idx + 1,
            "turn":           turn_idx,
            "description":    turn_spec["description"],
            "z_before_m":     z_before,
            "z_after_m":      z_after,
            "yaw_before_deg": yaw_before,
            "yaw_after_deg":  yaw_after,
            "armed_after":    int(armed_after),
            "tools_used":     ";".join(tools_used[:10]),
            "expected_found": ";".join(expected_found),
            "forbidden_found":";".join(forbidden_found),
            "sequence_ok":    int(sequence_ok),
            "state_ok":       int(state_ok),
            "overall_pass":   int(overall_pass),
            "api_calls":      n_api,
            "input_tokens":   in_tok,
            "output_tokens":  out_tok,
        })

    n_pass = sum(r["overall_pass"] for r in turn_rows)
    print(f"  Run {run_idx+1}: {n_pass}/{N_TURNS} turns passed")
    return turn_rows, agent

# ── Run N times ────────────────────────────────────────────────────────────────

all_rows   = []
all_agents = []
for i in range(N_RUNS):
    rows, agent = run_once(i)
    all_rows.extend(rows)
    all_agents.append(agent)

# ── Aggregate per-turn success rates ──────────────────────────────────────────

print(f"\n[C3] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
summary_rows = []
turn_rates = []
turn_lo    = []
turn_hi    = []

for turn_idx in range(1, N_TURNS + 1):
    t_rows  = [r for r in all_rows if r["turn"] == turn_idx]
    n_ok    = sum(r["overall_pass"] for r in t_rows)
    n_tot   = len(t_rows)
    rate    = n_ok / n_tot
    lo, hi  = wilson_ci(n_ok, n_tot)
    desc    = t_rows[0]["description"]
    print(f"  T{turn_idx} [{desc}]: {n_ok}/{n_tot}  rate={rate:.2f}  CI=[{lo:.2f},{hi:.2f}]")
    summary_rows.append({
        "turn":          turn_idx,
        "description":   desc,
        "n_pass":        n_ok,
        "n_runs":        n_tot,
        "success_rate":  round(rate, 3),
        "wilson_ci_lo":  round(lo, 3),
        "wilson_ci_hi":  round(hi, 3),
    })
    turn_rates.append(rate)
    turn_lo.append(lo)
    turn_hi.append(hi)

per_run_totals = []
for run_i in range(1, N_RUNS + 1):
    rows_i = [r for r in all_rows if r["run"] == run_i]
    per_run_totals.append(sum(r["overall_pass"] for r in rows_i))
ci_tot = bootstrap_ci(per_run_totals)

overall_rate = float(np.mean(per_run_totals)) / N_TURNS
print(f"\n  Turns passed / run: {np.mean(per_run_totals):.2f}±{np.std(per_run_totals):.2f}  "
      f"(bootstrap CI: {ci_tot[0]:.2f}–{ci_tot[1]:.2f})")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    w.writerows(all_rows)
print(f"[C3] Per-run CSV: {OUT_RUNS}")

with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader()
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow({"turn": f"REF_{ref_key}", "description": ref_val,
                    "n_pass": "", "n_runs": "", "success_rate": "",
                    "wilson_ci_lo": "", "wilson_ci_hi": ""})
print(f"[C3] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: per-turn success rate with CI
x = np.arange(N_TURNS)
err_lo = [r - l for r, l in zip(turn_rates, turn_lo)]
err_hi = [h - r for r, h in zip(turn_rates, turn_hi)]
bar_colors = ["green" if r >= 0.8 else "orange" if r >= 0.6 else "red" for r in turn_rates]
bars = ax1.bar(x, turn_rates, color=bar_colors, alpha=0.75, edgecolor="black")
ax1.errorbar(x, turn_rates, yerr=[err_lo, err_hi],
             fmt="none", ecolor="black", capsize=5, lw=1.5)
ax1.set_xticks(x)
ax1.set_xticklabels(
    [f"T{r['turn']}\n{r['description'][:18]}" for r in summary_rows],
    fontsize=7
)
ax1.set_ylabel("Success rate (N=5 runs)")
ax1.set_ylim(0, 1.15)
ax1.set_title(f"C3: Per-turn success rate (N={N_RUNS})\nError bars = Wilson 95% CI")
ax1.grid(True, alpha=0.3, axis="y")
for bar, rate in zip(bars, turn_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.04,
             f"{rate:.2f}", ha="center", fontsize=9)

# Right: pass/fail heatmap (runs × turns)
matrix = np.zeros((N_RUNS, N_TURNS), dtype=int)
for r in all_rows:
    matrix[r["run"] - 1, r["turn"] - 1] = r["overall_pass"]

im = ax2.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(N_TURNS))
ax2.set_xticklabels([f"T{i+1}" for i in range(N_TURNS)])
ax2.set_yticks(range(N_RUNS))
ax2.set_yticklabels([f"Run {i+1}" for i in range(N_RUNS)])
for i in range(N_RUNS):
    for j in range(N_TURNS):
        ax2.text(j, i, "✓" if matrix[i, j] else "✗",
                 ha="center", va="center", fontsize=14,
                 color="black")
ax2.set_title(f"Pass/Fail heatmap (runs × turns)\n"
              f"Turns/run: {np.mean(per_run_totals):.2f}±{np.std(per_run_totals):.2f}  "
              f"CI=[{ci_tot[0]:.2f},{ci_tot[1]:.2f}]")

fig.suptitle(f"EXP-C3: Multi-Turn Mission  (N={N_RUNS} runs, temperature=0.2)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C3] Plot: {OUT_PNG}")

print(f"\n[C3] RESULT: {np.mean(per_run_totals):.2f}±{np.std(per_run_totals):.2f}/{N_TURNS} "
      f"turns passed per run  (CI: [{ci_tot[0]:.2f},{ci_tot[1]:.2f}])")
