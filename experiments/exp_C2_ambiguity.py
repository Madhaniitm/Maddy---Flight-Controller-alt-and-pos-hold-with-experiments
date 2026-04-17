"""
EXP-C2: Command Ambiguity Resolution  (N=5 runs)
=================================================
Send 6 progressively ambiguous commands. Measure how the LLM
interprets each across N=5 independent sessions.

Each run: fresh drone at 1.0 m, 6 commands sent in sequence.
Reports per-command success rate with Wilson 95% CI.

Outputs:
  results/C2_runs.csv        — per-run × per-command results
  results/C2_summary.csv     — per-command success rate + CI
  results/C2_ambiguity.png   — success-rate bar chart with CI error bars
"""

import sys, os, csv, json, re, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", "C2_runs.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", "C2_summary.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", "C2_ambiguity.png")

N_RUNS = 5

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Agent loop that processes ambiguous commands via reason→act→observe."
    ),
    "Vemprala2023": (
        "Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). "
        "ChatGPT for Robotics: Design Principles and Model Abilities. MSR-TR-2023-8. arXiv:2306.17582. "
        "Benchmark for NL command interpretation rates on UAV tasks."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "Explains how the LLM infers intent from conversation context without explicit clarification."
    ),
}

COMMANDS = [
    ("go to 2 metres",                    2.0,  (1.90, 2.10), "explicit"),
    ("climb to 2m",                       2.0,  (1.90, 2.10), "paraphrase"),
    ("go higher",                         None, (0.10, 1.50), "relative_no_num"),
    ("go up a bit",                       None, (0.05, 0.60), "vague_relative"),
    ("ascend slowly to a safe height",    None, (0.05, 2.00), "abstract"),
    ("I want it higher",                  None, (0.05, 2.00), "indirect"),
]
N_CMDS = len(COMMANDS)

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

# ── Helpers copied from original ───────────────────────────────────────────────

def extract_altitude_target(tool_trace, text_response):
    last_target = None
    for tr in tool_trace:
        if tr["name"] == "set_altitude_target":
            m = tr["args"].get("meters")
            if m is not None:
                last_target = float(m)
    if last_target is not None:
        return last_target, "set_altitude_target"
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:m(?:etre?s?)?|meter?s?)\b',
                         text_response, re.IGNORECASE)
    if numbers:
        return float(numbers[0]), "text_inference"
    return None, "unknown"

def asked_for_clarification(text_response):
    clarify_words = ["clarif", "specific", "how high", "what height",
                     "please specify", "did you mean", "could you clarify"]
    lower = text_response.lower()
    return any(w in lower for w in clarify_words)

# ── Single-run function ────────────────────────────────────────────────────────

def run_once(run_idx):
    print(f"\n[C2] ── Run {run_idx+1}/{N_RUNS} ─────────────────────────────────")

    agent_base = SimAgent(session_id=f"C2_run{run_idx}")

    # Pre-arm and hover at 1.0 m (direct sim, no LLM)
    with agent_base.state.lock:
        agent_base.state.armed = True
        agent_base.state.ch5   = 1000

    hover_pwm = agent_base._find_hover()
    hover_thr  = (hover_pwm - 1000) / 1000.0
    with agent_base.state.lock:
        s = agent_base.state
        s.hover_thr_locked = hover_thr
        s.althold  = True
        s.alt_sp   = s.z
        s.alt_sp_mm = s.z * 1000
    agent_base.physics.pid_alt_pos.reset()
    agent_base.physics.pid_alt_vel.reset()
    with agent_base.state.lock:
        agent_base.state.alt_sp    = 1.0
        agent_base.state.alt_sp_mm = 1000.0
    agent_base.wait_sim(8.0)

    with agent_base.state.lock:
        z_base = round(agent_base.state.ekf_z, 3)
    print(f"  Drone ready at {z_base:.3f} m")

    shared_history = [
        {
            "role": "user",
            "content": (
                "The drone is currently armed and hovering at approximately 1.0 m with "
                "altitude hold active. I will give you a series of flight commands."
            ),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text":
                "Understood. The drone is airborne at ~1.0 m with altitude hold active. "
                "Ready to receive your commands."}],
        },
    ]

    cmd_results = []
    for idx, (command, gt_exact, gt_range, cmd_type) in enumerate(COMMANDS, start=1):
        with agent_base.state.lock:
            z_before = round(agent_base.state.ekf_z, 3)

        turn_text, api_stats, tool_trace = agent_base.run_agent_loop(
            command,
            history=list(shared_history),
            max_turns=10,
        )

        shared_history.append({"role": "user", "content": command})
        shared_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": turn_text}],
        })

        agent_base.wait_sim(6.0)
        with agent_base.state.lock:
            z_after = round(agent_base.state.ekf_z, 3)
        increment = z_after - z_before

        target_m, target_src = extract_altitude_target(tool_trace, turn_text)
        clarified = asked_for_clarification(turn_text)

        if gt_exact is not None:
            correct = (target_m is not None and abs(target_m - gt_exact) <= 0.15)
        else:
            lo, hi = gt_range
            if clarified:
                correct = True
            else:
                correct = lo <= increment <= hi

        total_api  = len(api_stats)
        total_toks = sum(s["input_tokens"] + s["output_tokens"] for s in api_stats)

        print(f"    Cmd{idx} [{cmd_type}]: z {z_before:.3f}→{z_after:.3f}m "
              f"Δ={increment:+.3f}  target={target_m}  correct={correct}")

        cmd_results.append({
            "run":           run_idx + 1,
            "cmd_idx":       idx,
            "command":       command,
            "cmd_type":      cmd_type,
            "z_before_m":    z_before,
            "z_after_m":     z_after,
            "increment_m":   round(increment, 3),
            "target_m":      target_m,
            "target_source": target_src,
            "asked_clarify": int(clarified),
            "correct":       int(correct),
            "api_calls":     total_api,
            "tokens":        total_toks,
        })

    n_correct = sum(r["correct"] for r in cmd_results)
    print(f"  Run {run_idx+1}: {n_correct}/{N_CMDS} correct")
    return cmd_results

# ── Run N times ────────────────────────────────────────────────────────────────

all_run_rows = []   # flat list of all cmd-level rows
for i in range(N_RUNS):
    all_run_rows.extend(run_once(i))

# ── Aggregate per-command success rates ───────────────────────────────────────

print(f"\n[C2] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
summary_rows = []
cmd_rates = []   # for plot
cmd_lo    = []
cmd_hi    = []

for idx in range(1, N_CMDS + 1):
    cmd_rows = [r for r in all_run_rows if r["cmd_idx"] == idx]
    n_ok  = sum(r["correct"] for r in cmd_rows)
    n_tot = len(cmd_rows)     # = N_RUNS
    rate  = n_ok / n_tot
    lo, hi = wilson_ci(n_ok, n_tot)
    cmd_type = cmd_rows[0]["cmd_type"]
    command  = cmd_rows[0]["command"]
    increments = [r["increment_m"] for r in cmd_rows]
    print(f"  Cmd{idx} [{cmd_type}]: {n_ok}/{n_tot}  rate={rate:.2f}  "
          f"CI=[{lo:.2f},{hi:.2f}]  Δinc={np.mean(increments):.3f}±{np.std(increments):.3f}m")
    summary_rows.append({
        "cmd_idx":          idx,
        "command":          command,
        "cmd_type":         cmd_type,
        "n_correct":        n_ok,
        "n_runs":           n_tot,
        "success_rate":     round(rate, 3),
        "wilson_ci_lo":     round(lo, 3),
        "wilson_ci_hi":     round(hi, 3),
        "increment_mean_m": round(float(np.mean(increments)), 3),
        "increment_std_m":  round(float(np.std(increments)), 3),
    })
    cmd_rates.append(rate)
    cmd_lo.append(lo)
    cmd_hi.append(hi)

overall_correct = sum(r["correct"] for r in all_run_rows)
overall_total   = len(all_run_rows)
overall_rate    = overall_correct / overall_total
lo_ov, hi_ov    = wilson_ci(overall_correct, overall_total)

# Per-run total
per_run_totals  = []
for run_i in range(1, N_RUNS + 1):
    rows_i = [r for r in all_run_rows if r["run"] == run_i]
    per_run_totals.append(sum(r["correct"] for r in rows_i))
ci_total = bootstrap_ci(per_run_totals)

print(f"\n  Overall: {overall_correct}/{overall_total}  rate={overall_rate:.2f}  "
      f"Wilson CI=[{lo_ov:.2f},{hi_ov:.2f}]")
print(f"  Correct/run: {np.mean(per_run_totals):.2f} ± {np.std(per_run_totals):.2f}  "
      f"(bootstrap CI: {ci_total[0]:.2f}–{ci_total[1]:.2f})")

# ── Save per-run CSV ───────────────────────────────────────────────────────────
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_run_rows[0].keys())
    w.writeheader()
    w.writerows(all_run_rows)
print(f"[C2] Per-run CSV: {OUT_RUNS}")

# ── Save summary CSV ───────────────────────────────────────────────────────────
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader()
    w.writerows(summary_rows)

    # Append references
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow({"cmd_idx": f"REF_{ref_key}", "command": ref_val,
                    "cmd_type": "", "n_correct": "", "n_runs": "",
                    "success_rate": "", "wilson_ci_lo": "", "wilson_ci_hi": "",
                    "increment_mean_m": "", "increment_std_m": ""})

    # Append overall row
    w.writerow({
        "cmd_idx": "OVERALL",
        "command": "all commands",
        "cmd_type": "",
        "n_correct": overall_correct,
        "n_runs": overall_total,
        "success_rate": round(overall_rate, 3),
        "wilson_ci_lo": round(lo_ov, 3),
        "wilson_ci_hi": round(hi_ov, 3),
        "increment_mean_m": round(float(np.mean([r["increment_m"] for r in all_run_rows])), 3),
        "increment_std_m":  round(float(np.std([r["increment_m"] for r in all_run_rows])), 3),
    })
print(f"[C2] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: per-command success rate with CI error bars
x = np.arange(N_CMDS)
err_lo = [r - l for r, l in zip(cmd_rates, cmd_lo)]
err_hi = [h - r for r, h in zip(cmd_rates, cmd_hi)]
bar_colors = ["green" if r >= 0.6 else "orange" if r >= 0.4 else "red" for r in cmd_rates]
bars = ax1.bar(x, cmd_rates, color=bar_colors, alpha=0.75, edgecolor="black")
ax1.errorbar(x, cmd_rates,
             yerr=[err_lo, err_hi],
             fmt="none", ecolor="black", capsize=5, lw=1.5)
ax1.axhline(overall_rate, color="navy", ls="--", lw=1.5,
            label=f"Overall rate {overall_rate:.2f}")
ax1.set_xticks(x)
labels = [f'C{i+1}\n{COMMANDS[i][3]}\n"{COMMANDS[i][0][:20]}…"'
          if len(COMMANDS[i][0]) > 20
          else f'C{i+1}\n{COMMANDS[i][3]}\n"{COMMANDS[i][0]}"'
          for i in range(N_CMDS)]
ax1.set_xticklabels(labels, fontsize=7)
ax1.set_ylabel("Success rate (N=5 runs)")
ax1.set_ylim(0, 1.15)
ax1.set_title(f"C2: Per-command success rate (N={N_RUNS})\n"
              f"Error bars = Wilson 95% CI")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")
for bar, rate in zip(bars, cmd_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.05,
             f"{rate:.2f}", ha="center", fontsize=9)

# Right: distribution of total correct per run
ax2.bar(range(1, N_RUNS + 1), per_run_totals,
        color="steelblue", alpha=0.75, edgecolor="black")
ax2.axhline(np.mean(per_run_totals), color="red", ls="--", lw=1.5,
            label=f"Mean={np.mean(per_run_totals):.2f}")
ax2.fill_between(
    [0.5, N_RUNS + 0.5],
    np.mean(per_run_totals) - np.std(per_run_totals),
    np.mean(per_run_totals) + np.std(per_run_totals),
    alpha=0.12, color="red", label="±1σ"
)
ax2.set_xlabel("Run")
ax2.set_ylabel(f"Correct commands (out of {N_CMDS})")
ax2.set_ylim(0, N_CMDS + 0.5)
ax2.set_xticks(range(1, N_RUNS + 1))
ax2.set_title(f"Correct commands per run\n"
              f"Mean={np.mean(per_run_totals):.2f}±{np.std(per_run_totals):.2f}  "
              f"CI=[{ci_total[0]:.2f},{ci_total[1]:.2f}]")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(per_run_totals):
    ax2.text(i + 1, v + 0.1, str(v), ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"[C2] Plot: {OUT_PNG}")

print(f"\n[C2] RESULT: {np.mean(per_run_totals):.2f}±{np.std(per_run_totals):.2f}/{N_CMDS} "
      f"correct per run  (overall {overall_rate:.2f}, CI [{lo_ov:.2f},{hi_ov:.2f}])")
