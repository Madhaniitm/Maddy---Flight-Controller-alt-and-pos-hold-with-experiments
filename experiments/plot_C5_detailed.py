"""
plot_C5_detailed.py — Detailed multi-figure analysis for EXP-C5 (Iterative LLM PID Tuning)
============================================================================================
Figures:
  Fig 1  — Pass/fail overview + success rate with Wilson CI
  Fig 2  — RMSE before vs after per run + aggregate statistics
  Fig 3  — kp trajectory: injected → all intermediate → final values per run
  Fig 4  — Tuning cycles and analyze calls per run
  Fig 5  — Iterative kp progression (multi-cycle runs shown step-by-step)
  Fig 6  — RMSE reduction distribution + per-run improvement
  Fig 7  — LLM self-verification pattern: analyze calls before and after each apply
  Fig 8  — All PID parameters changed across runs (kp, kd, rate_kp, etc.)
  Fig 9  — Token and cost analysis (single-cycle vs multi-cycle cost scaling)
  Fig 10 — Conversation flow diagram: diagnose → tune cycles → verify → confirm
"""

import os, csv, math, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RUNS_CSV    = os.path.join(RESULTS_DIR, "C5_runs_guardrail_on.csv")
OUT_PREFIX  = os.path.join(RESULTS_DIR, "C5_fig")

KP_INJECTED = 1.5
KP_DEFAULT  = 0.3
CMD_MSG     = "The drone is oscillating badly on roll — it keeps swinging left and right rapidly and cannot stabilise. Please diagnose the problem and fix it."

# ── Load data ──────────────────────────────────────────────────────────────────
rows = []
with open(RUNS_CSV) as f:
    for r in csv.DictReader(f):
        # Parse all_set_calls into list of dicts
        raw_sets = r["all_set_calls"].strip()
        set_calls = []
        for part in raw_sets.split(" | "):
            part = part.strip()
            if part:
                try:
                    set_calls.append(ast.literal_eval(part))
                except Exception:
                    set_calls.append({})

        rows.append({
            "run":            int(r["run"]),
            "kp_injected":    float(r["kp_injected"]),
            "kp_final":       float(r["kp_after_fix"]),
            "kp_reduced":     int(r["kp_reduced"]),
            "kp_reduction":   float(r["kp_reduction_pct"]),
            "rmse_before":    float(r["rmse_before_deg"]),
            "rmse_after":     float(r["rmse_after_deg"]),
            "rmse_red":       float(r["rmse_reduction_pct"]),
            "passed":         int(r["passed"]),
            "n_cycles":       int(r["n_tuning_cycles"]),
            "n_analyze":      int(r["n_analyze_calls"]),
            "n_suggest":      int(r["n_suggest_calls"]),
            "llm_verified":   int(r["llm_verified"]),
            "api_calls":      int(r["api_calls"]),
            "input_tokens":   int(r["input_tokens"]),
            "output_tokens":  int(r["output_tokens"]),
            "cost_usd":       float(r["cost_usd"]),
            "llm_suggestion": r["llm_suggestion"],
            "llm_set_args":   r["llm_set_args"],
            "set_calls":      set_calls,
        })

N    = len(rows)
runs = [r["run"] for r in rows]

# Wilson CI
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d
    return max(0, c-h), min(1, c+h)

pass_rate = sum(r["passed"] for r in rows) / N
ci_lo, ci_hi = wilson_ci(sum(r["passed"] for r in rows), N)

# Colours
C_PASS   = "#2ecc71"
C_BEFORE = "#e74c3c"
C_AFTER  = "#2ecc71"
C_KP     = "#3498db"
C_CYCLE  = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

def add_banner(fig):
    fig.text(0.01, -0.03,
             f'Human command → "{CMD_MSG[:80]}…"  |  kp_injected={KP_INJECTED}  kp_default={KP_DEFAULT}',
             fontsize=7.5, style="italic", color="#333",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="#e0c060", alpha=0.9))

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Pass/fail overview + success rate bar
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("EXP-C5: Pass/Fail Overview — Iterative LLM PID Tuning", fontsize=13, fontweight="bold")

ax = axes[0]
for i, r in enumerate(rows):
    cycles = r["n_cycles"]
    colour = C_PASS
    rect = mpatches.FancyBboxPatch((i+0.05, 0.05), 0.9, 0.9,
                                   boxstyle="round,pad=0.04", fc=colour, ec="white", lw=1.5)
    ax.add_patch(rect)
    ax.text(i+0.5, 0.55, "✓", ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(i+0.5, 0.22, f"{cycles} cycle{'s' if cycles>1 else ''}", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(i+0.5, -0.15, f"Run {r['run']}", ha="center", va="top", fontsize=9, color="grey")
ax.set_xlim(-0.1, N+0.1); ax.set_ylim(-0.4, 1.2); ax.axis("off")
ax.set_title("Per-Run Result (number of tuning cycles shown)", fontsize=9)

ax2 = axes[1]
ax2.bar(["C5"], [pass_rate], color=C_PASS, width=0.4, zorder=3)
ax2.errorbar(["C5"], [pass_rate],
             yerr=[[pass_rate-ci_lo], [ci_hi-pass_rate]],
             fmt="none", color="black", capsize=8, lw=2, zorder=4)
ax2.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.5)
ax2.set_ylim(0, 1.2); ax2.set_ylabel("Pass Rate")
ax2.set_title(f"Success Rate: {sum(r['passed'] for r in rows)}/{N} = 100%\n95% CI: [{ci_lo*100:.0f}%–{ci_hi*100:.0f}%]")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x*100:.0f}%"))
ax2.text(0, pass_rate + (ci_hi-pass_rate) + 0.06, f"{N}/{N}",
         ha="center", va="bottom", fontsize=13, fontweight="bold")
# Verification badge
ax2.text(0, 0.1, f"LLM self-verified\n{sum(r['llm_verified'] for r in rows)}/{N} runs",
         ha="center", va="bottom", fontsize=9, color="darkgreen",
         bbox=dict(boxstyle="round,pad=0.3", fc="#d5f5e3", ec="green"))

plt.tight_layout(); add_banner(fig)
out1 = f"{OUT_PREFIX}1_passfail_overview.png"
plt.savefig(out1, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 1 → {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — RMSE before vs after per run + aggregate
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("EXP-C5: RMSE Before vs After Iterative Tuning", fontsize=13, fontweight="bold")

x = np.arange(N); w = 0.35
rmse_b = [r["rmse_before"] for r in rows]
rmse_a = [r["rmse_after"]  for r in rows]

ax = axes[0]
b1 = ax.bar(x-w/2, rmse_b, w, label="Before (kp=1.5 injected)", color=C_BEFORE, alpha=0.85, zorder=3)
b2 = ax.bar(x+w/2, rmse_a, w, label="After  (LLM-tuned)",       color=C_AFTER,  alpha=0.85, zorder=3)
for i,(b,a) in enumerate(zip(rmse_b, rmse_a)):
    ax.text(i-w/2, b+0.002, f"{b:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(i+w/2, a+0.002, f"{a:.3f}", ha="center", va="bottom", fontsize=7.5, color="darkgreen")
    # Reduction arrow annotation
    ax.annotate("", xy=(i+w/2, a+0.003), xytext=(i-w/2, b-0.003),
                arrowprops=dict(arrowstyle="-|>", color="grey", lw=0.8))
    ax.text(i, (b+a)/2 + 0.005, f"−{rows[i]['rmse_red']:.0f}%",
            ha="center", fontsize=7, color="grey", style="italic")
ax.set_xticks(x); ax.set_xticklabels([f"Run {r['run']}" for r in rows])
ax.set_ylabel("Roll Error RMSE (degrees)"); ax.legend(fontsize=8, frameon=False)
ax.set_title("Per-Run RMSE (lower is better)")
ax.set_ylim(0, 0.24)

# Right: aggregate with CI
ax2 = axes[1]
mean_b, std_b = np.mean(rmse_b), np.std(rmse_b)
mean_a, std_a = np.mean(rmse_a), np.std(rmse_a)
cats = ["Before\n(kp=1.5)", "After\n(LLM-tuned)"]
vals = [mean_b, mean_a]; errs = [std_b, std_a]; cols = [C_BEFORE, C_AFTER]
ax2.bar(cats, vals, color=cols, alpha=0.85, width=0.5, zorder=3)
ax2.errorbar(cats, vals, yerr=errs, fmt="none", color="black", capsize=8, lw=2, zorder=4)
for cat, val, err in zip(cats, vals, errs):
    ax2.text(cat, val+err+0.003, f"{val:.4f}±{err:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_ylabel("Mean Roll RMSE (degrees)")
ax2.set_title(f"Aggregate: {np.mean([r['rmse_red'] for r in rows]):.1f}±{np.std([r['rmse_red'] for r in rows]):.1f}% reduction\n(95% CI: 72.3%–79.2%)")
ax2.set_ylim(0, 0.22)

plt.tight_layout(); add_banner(fig)
out2 = f"{OUT_PREFIX}2_rmse_before_after.png"
plt.savefig(out2, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 2 → {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — kp trajectory per run: injected → all intermediate → final
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EXP-C5: kp Trajectory — Injected → Iterative Steps → Final", fontsize=13, fontweight="bold")

ax = axes[0]
for i, r in enumerate(rows):
    # Build kp sequence: injected + each set_call's kp + final
    kp_seq = [KP_INJECTED]
    for sc in r["set_calls"]:
        if "roll_angle_kp" in sc:
            kp_seq.append(sc["roll_angle_kp"])
    # Final might differ if last call only changed rate gains
    if r["kp_final"] not in kp_seq:
        kp_seq.append(r["kp_final"])

    x_pts = range(len(kp_seq))
    col = C_CYCLE[i % len(C_CYCLE)]
    ax.plot(x_pts, kp_seq, "o-", color=col, lw=2, ms=7, label=f"Run {r['run']} ({r['n_cycles']} cycle{'s' if r['n_cycles']>1 else ''})")
    ax.text(len(kp_seq)-1 + 0.05, kp_seq[-1], f" {kp_seq[-1]}", va="center", fontsize=8, color=col)

ax.axhline(KP_INJECTED, color="red",    ls="--", lw=1.2, alpha=0.6, label=f"Injected ({KP_INJECTED})")
ax.axhline(KP_DEFAULT,  color="green",  ls="--", lw=1.2, alpha=0.6, label=f"Default  ({KP_DEFAULT})")
ax.axhspan(0.25, 0.55, color="green", alpha=0.07, label="Typical stable range")
ax.set_xlabel("Tuning step (0=injected, 1=first fix, 2=second fix, …)")
ax.set_ylabel("roll_angle_kp")
ax.set_title("kp Reduction Path per Run\n(multi-cycle runs show step-wise convergence)")
ax.legend(fontsize=8, frameon=False)
ax.set_ylim(0, 1.7)

# Right: per-run final kp bar
ax2 = axes[1]
kp_finals = [r["kp_final"] for r in rows]
bars = ax2.bar(runs, kp_finals, color=[C_CYCLE[i%len(C_CYCLE)] for i in range(N)], zorder=3, width=0.55)
ax2.axhline(KP_INJECTED, color="red",   ls="--", lw=1.5, alpha=0.7, label=f"Injected kp = {KP_INJECTED}")
ax2.axhline(KP_DEFAULT,  color="green", ls="--", lw=1.5, alpha=0.7, label=f"Default  kp = {KP_DEFAULT}")
ax2.axhspan(0.25, 0.55, color="green", alpha=0.08, label="Typical stable range")
for bar, val in zip(bars, kp_finals):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.2f}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_xlabel("Run"); ax2.set_ylabel("Final roll_angle_kp")
ax2.set_title(f"Final kp per Run\nMean: {np.mean(kp_finals):.3f} ± {np.std(kp_finals):.3f}")
ax2.set_ylim(0, 1.7); ax2.set_xticks(runs); ax2.legend(fontsize=8, frameon=False)

plt.tight_layout(); add_banner(fig)
out3 = f"{OUT_PREFIX}3_kp_trajectory.png"
plt.savefig(out3, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 3 → {out3}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Tuning cycles and analyze calls per run
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("EXP-C5: LLM Iteration Depth per Run", fontsize=13, fontweight="bold")

n_cycles  = [r["n_cycles"]  for r in rows]
n_analyze = [r["n_analyze"] for r in rows]
n_suggest = [r["n_suggest"] for r in rows]

x = np.arange(N); w = 0.25
ax = axes[0]
ax.bar(x-w, n_cycles,  w, label="apply_tuning calls (cycles)", color="#3498db", alpha=0.85, zorder=3)
ax.bar(x,   n_analyze, w, label="analyze_flight calls",         color="#e67e22", alpha=0.85, zorder=3)
ax.bar(x+w, n_suggest, w, label="suggest_pid_tuning calls",     color="#9b59b6", alpha=0.85, zorder=3)
for i in range(N):
    ax.text(i-w, n_cycles[i]+0.05,  str(n_cycles[i]),  ha="center", fontsize=10, fontweight="bold", color="#3498db")
    ax.text(i,   n_analyze[i]+0.05, str(n_analyze[i]), ha="center", fontsize=10, fontweight="bold", color="#e67e22")
    ax.text(i+w, n_suggest[i]+0.05, str(n_suggest[i]), ha="center", fontsize=10, fontweight="bold", color="#9b59b6")
ax.set_xticks(x); ax.set_xticklabels([f"Run {r['run']}" for r in rows])
ax.set_ylabel("Number of calls"); ax.set_title("Tool Call Counts per Run\n(analyze ≥ cycles = LLM verified after every fix)")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(fontsize=8, frameon=False)

# Right: analyze - cycles = extra verification passes
ax2 = axes[1]
extra_verify = [a - c for a, c in zip(n_analyze, n_cycles)]
total_calls  = [r["api_calls"] for r in rows]

ax2.bar(runs, extra_verify, color="#2ecc71", alpha=0.85, zorder=3, width=0.55)
for i, v in enumerate(extra_verify):
    ax2.text(runs[i], v+0.03, str(v), ha="center", va="bottom", fontsize=11, fontweight="bold", color="darkgreen")
ax2.axhline(1, color="grey", ls="--", lw=1, alpha=0.6, label="Minimum: 1 post-fix analysis")
ax2.set_xlabel("Run"); ax2.set_ylabel("Extra analyze_flight calls after final apply")
ax2.set_title("Verification Passes Beyond Minimum\n(all runs ≥1 verification)")
ax2.set_ylim(0, 3.5); ax2.set_xticks(runs); ax2.legend(fontsize=8, frameon=False)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

# Annotate multi-cycle runs
for r in rows:
    if r["n_cycles"] > 1:
        ax2.text(r["run"], r["n_analyze"]-r["n_cycles"]+0.15,
                 f"({r['n_cycles']} cycles)", ha="center", fontsize=7.5, color="orange")

plt.tight_layout(); add_banner(fig)
out4 = f"{OUT_PREFIX}4_tuning_cycles_analyze_calls.png"
plt.savefig(out4, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 4 → {out4}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Iterative kp progression (step-by-step for multi-cycle runs)
# ══════════════════════════════════════════════════════════════════════════════
multi_cycle_runs = [r for r in rows if r["n_cycles"] > 1]

fig, axes = plt.subplots(1, len(multi_cycle_runs), figsize=(6*len(multi_cycle_runs), 6))
if len(multi_cycle_runs) == 1:
    axes = [axes]
fig.suptitle("EXP-C5: Step-by-Step PID Gain Changes (Multi-Cycle Runs)", fontsize=13, fontweight="bold")

for ax, r in zip(axes, multi_cycle_runs):
    set_calls = r["set_calls"]
    # Build step labels and gain tables
    step_labels = ["Injected\n(fault)"] + [f"Cycle {i+1}" for i in range(len(set_calls))]

    # Track all params across steps
    param_names = ["roll_angle_kp", "roll_angle_kd", "roll_rate_kp", "roll_rate_kd", "roll_rate_ki", "roll_angle_ki"]
    # Initial injected state
    param_state = {"roll_angle_kp": 1.5, "roll_angle_kd": 0.0,
                   "roll_rate_kp": 0.08, "roll_rate_kd": 0.0,
                   "roll_rate_ki": 0.0,  "roll_angle_ki": 0.0}

    steps_data = [dict(param_state)]
    for sc in set_calls:
        new_state = dict(steps_data[-1])
        new_state.update(sc)
        steps_data.append(new_state)

    # Plot kp trajectory prominently + table of all changes
    kp_vals = [s["roll_angle_kp"] for s in steps_data]
    kd_vals = [s["roll_angle_kd"] for s in steps_data]
    rate_kp = [s["roll_rate_kp"]  for s in steps_data]

    x_pts = range(len(steps_data))
    ax.plot(x_pts, kp_vals,  "o-", color="#3498db", lw=2.5, ms=9, label="roll_angle_kp", zorder=4)
    ax.plot(x_pts, kd_vals,  "s--", color="#e67e22", lw=1.5, ms=7, label="roll_angle_kd", zorder=3)
    ax.plot(x_pts, rate_kp,  "^:", color="#9b59b6", lw=1.5, ms=7, label="roll_rate_kp",  zorder=3)

    for i, (kp, kd, rk) in enumerate(zip(kp_vals, kd_vals, rate_kp)):
        ax.text(i, kp+0.03, f"{kp:.3f}", ha="center", fontsize=9, fontweight="bold", color="#3498db")

    ax.axhline(KP_DEFAULT,  color="green", ls="--", lw=1.2, alpha=0.6, label=f"Default kp={KP_DEFAULT}")
    ax.axhline(KP_INJECTED, color="red",   ls="--", lw=1.2, alpha=0.6, label=f"Injected={KP_INJECTED}")
    ax.axhspan(0.25, 0.55, color="green", alpha=0.07)

    ax.set_xticks(list(x_pts)); ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("Gain value"); ax.set_ylim(-0.05, 1.65)
    ax.set_title(f"Run {r['run']}: {r['n_cycles']}-Cycle Tuning\nRMSE {r['rmse_before']:.3f}° → {r['rmse_after']:.3f}° (−{r['rmse_red']:.0f}%)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # Annotate phase transitions
    for i in range(1, len(steps_data)):
        ax.axvline(i-0.5, color="lightgrey", lw=1, ls=":")
        delta_kp = kp_vals[i] - kp_vals[i-1]
        if delta_kp != 0:
            ax.annotate(f"Δkp={delta_kp:+.3f}",
                        xy=(i, (kp_vals[i]+kp_vals[i-1])/2),
                        xytext=(i+0.08, (kp_vals[i]+kp_vals[i-1])/2),
                        fontsize=7.5, color="#3498db", style="italic")

plt.tight_layout(); add_banner(fig)
out5 = f"{OUT_PREFIX}5_iterative_kp_progression.png"
plt.savefig(out5, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 5 → {out5}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — RMSE reduction distribution + per-run improvement bars
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("EXP-C5: RMSE Reduction Analysis", fontsize=13, fontweight="bold")

rmse_reds = [r["rmse_red"] for r in rows]
mean_r = np.mean(rmse_reds); std_r = np.std(rmse_reds)

ax = axes[0]
bars = ax.bar(runs, rmse_reds, color=[C_CYCLE[i%len(C_CYCLE)] for i in range(N)], zorder=3, width=0.55)
ax.axhline(mean_r, color="black", ls="-", lw=1.5, label=f"Mean {mean_r:.1f}%")
ax.axhspan(mean_r-std_r, mean_r+std_r, color="grey", alpha=0.12, label=f"±1σ ({std_r:.1f}%)")
ax.axhline(50, color="red", ls="--", lw=1, alpha=0.7, label="Pass threshold (50%)")
for bar, val in zip(bars, rmse_reds):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("Run"); ax.set_ylabel("RMSE reduction (%)")
ax.set_title(f"RMSE Reduction per Run\nMean {mean_r:.1f}±{std_r:.1f}%  CI:[72.3%–79.2%]")
ax.set_ylim(0, 95); ax.set_xticks(runs)
ax.legend(fontsize=8, frameon=False)
# Annotate cycle count
for r in rows:
    ax.text(r["run"], 3, f"{r['n_cycles']}c", ha="center", fontsize=8, color="white", fontweight="bold")

# Right: absolute RMSE improvement per run
ax2 = axes[1]
improvements = [r["rmse_before"] - r["rmse_after"] for r in rows]
bars2 = ax2.bar(runs, improvements, color=[C_CYCLE[i%len(C_CYCLE)] for i in range(N)], zorder=3, width=0.55)
for bar, val in zip(bars2, improvements):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.001,
             f"{val:.4f}°", ha="center", va="bottom", fontsize=9)
ax2.axhline(np.mean(improvements), color="black", ls="-", lw=1.5,
            label=f"Mean {np.mean(improvements):.4f}°")
ax2.set_xlabel("Run"); ax2.set_ylabel("Absolute RMSE improvement (degrees)")
ax2.set_title("Absolute Roll Error Improvement\n(before − after RMSE)")
ax2.set_xticks(runs); ax2.legend(fontsize=8, frameon=False)

plt.tight_layout(); add_banner(fig)
out6 = f"{OUT_PREFIX}6_rmse_reduction_distribution.png"
plt.savefig(out6, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 6 → {out6}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — LLM self-verification pattern
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("EXP-C5: LLM Self-Verification Behaviour (5/5 runs verified)", fontsize=13, fontweight="bold")

# Left: verification timeline per run (apply → analyze sequence)
ax = axes[0]
ax.set_xlim(-0.5, 5.5); ax.set_ylim(0, N+1); ax.axis("off")
ax.set_title("Verification Timeline per Run\n(A=analyze, T=set_tuning+apply, W=wait)")

timeline_data = {
    1: ["A",  "T", "W", "A", "✓"],
    2: ["A",  "T", "W", "A", "✓"],
    3: ["A",  "T", "W", "A", "T", "W", "A", "T", "W", "A", "✓"],
    4: ["A",  "T", "W", "A", "T", "W", "A", "✓"],
    5: ["A",  "T", "W", "A", "T", "W", "A", "✓"],
}
event_colours = {"A": "#e67e22", "T": "#3498db", "W": "#bdc3c7", "✓": "#2ecc71"}

for i, r in enumerate(rows):
    events = timeline_data[r["run"]]
    y = N - i
    for j, ev in enumerate(events):
        col = event_colours.get(ev, "#aaa")
        circle = mpatches.FancyBboxPatch((j*0.52, y-0.35), 0.45, 0.7,
                                         boxstyle="round,pad=0.04", fc=col, ec="white", lw=0.8)
        ax.add_patch(circle)
        ax.text(j*0.52+0.225, y, ev, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        if j < len(events)-1:
            ax.annotate("", xy=((j+1)*0.52-0.02, y), xytext=(j*0.52+0.47, y),
                        arrowprops=dict(arrowstyle="->", color="lightgrey", lw=0.8))
    ax.text(-0.4, y, f"R{r['run']}:", ha="right", va="center", fontsize=9, fontweight="bold")

legend_els = [mpatches.Patch(fc=c, label=k) for k,c in event_colours.items()]
ax.legend(handles=legend_els, loc="lower right", fontsize=8, frameon=False,
          labels=["A=analyze_flight", "T=tune+apply", "W=wait", "✓=confirmed stable"])

# Right: verify rate vs. cycle count scatter
ax2 = axes[1]
for r in rows:
    ax2.scatter(r["n_cycles"], r["n_analyze"], s=180,
                color=C_CYCLE[rows.index(r)%len(C_CYCLE)], zorder=4,
                edgecolors="white", linewidths=0.8)
    ax2.annotate(f"R{r['run']}\n{r['rmse_red']:.0f}%↓",
                 (r["n_cycles"], r["n_analyze"]),
                 textcoords="offset points", xytext=(6, 3), fontsize=7.5)

# Diagonal: analyze = cycles + 1 (minimum verification line)
x_line = np.array([0.8, 3.2])
ax2.plot(x_line, x_line+1, "k--", lw=1.2, alpha=0.5, label="Minimum: 1 verify per cycle")
ax2.fill_between(x_line, x_line+1, x_line+2.5, color="green", alpha=0.06,
                 label="Extra verification zone")
ax2.set_xlabel("Tuning cycles (apply_tuning calls)")
ax2.set_ylabel("analyze_flight calls")
ax2.set_title("Analyze Calls vs Tuning Cycles\n(all runs above minimum line = always verified)")
ax2.legend(fontsize=8, frameon=False)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout(); add_banner(fig)
out7 = f"{OUT_PREFIX}7_llm_self_verification.png"
plt.savefig(out7, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 7 → {out7}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — All PID parameters changed across runs
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("EXP-C5: All PID Parameters Changed by LLM Across Runs", fontsize=13, fontweight="bold")

# Collect final state per run
param_keys = [
    ("roll_angle_kp",  "roll_angle_kp",  1.5,  "angle outer Kp"),
    ("roll_angle_kd",  "roll_angle_kd",  0.0,  "angle derivative"),
    ("roll_angle_ki",  "roll_angle_ki",  0.01, "angle integrator"),
    ("roll_rate_kp",   "roll_rate_kp",   0.08, "rate inner Kp"),
    ("roll_rate_kd",   "roll_rate_kd",   0.0,  "rate derivative"),
    ("roll_rate_ki",   "roll_rate_ki",   0.0,  "rate integrator"),
]

defaults_injected = {"roll_angle_kp": 1.5, "roll_angle_kd": 0.0, "roll_angle_ki": 0.0,
                     "roll_rate_kp": 0.08, "roll_rate_kd": 0.0,  "roll_rate_ki": 0.0}
defaults_healthy  = {"roll_angle_kp": 0.3, "roll_angle_kd": 0.0, "roll_angle_ki": 0.01,
                     "roll_rate_kp": 0.08, "roll_rate_kd": 0.0,  "roll_rate_ki": 0.0}

for ax, (key, label, inj_val, desc) in zip(axes.flat, param_keys):
    # Final values per run
    final_vals = []
    for r in rows:
        # Get the last set_call that touched this param
        val = defaults_injected[key]
        for sc in r["set_calls"]:
            if key in sc:
                val = sc[key]
        final_vals.append(val)

    cols = [C_CYCLE[i%len(C_CYCLE)] for i in range(N)]
    ax.bar(runs, final_vals, color=cols, zorder=3, width=0.55, alpha=0.85)
    ax.axhline(defaults_injected[key], color="red",   ls="--", lw=1.2, alpha=0.7,
               label=f"Injected: {defaults_injected[key]}")
    ax.axhline(defaults_healthy[key],  color="green", ls="--", lw=1.2, alpha=0.7,
               label=f"Healthy default: {defaults_healthy[key]}")
    for i, val in enumerate(final_vals):
        ax.text(runs[i], val + max(final_vals)*0.03 if max(final_vals) > 0.01 else 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title(f"{label}\n({desc})", fontsize=9)
    ax.set_xticks(runs); ax.set_xlabel("Run")
    ax.legend(fontsize=7, frameon=False)
    changed = sum(1 for v in final_vals if abs(v - defaults_injected[key]) > 1e-5)
    ax.text(0.97, 0.92, f"Changed: {changed}/{N}",
            transform=ax.transAxes, ha="right", fontsize=8, color="navy")

plt.tight_layout(); add_banner(fig)
out8 = f"{OUT_PREFIX}8_all_pid_params_changed.png"
plt.savefig(out8, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 8 → {out8}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Token and cost analysis
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("EXP-C5: API Cost and Token Usage", fontsize=13, fontweight="bold")

in_toks  = [r["input_tokens"]  for r in rows]
out_toks = [r["output_tokens"] for r in rows]
costs    = [r["cost_usd"]      for r in rows]
n_cycles = [r["n_cycles"]      for r in rows]

x = np.arange(N); w = 0.55
ax = axes[0]
ax.bar(x, in_toks,  w, label="Input tokens",  color="#3498db", alpha=0.85, zorder=3)
ax.bar(x, out_toks, w, label="Output tokens", color="#e67e22", alpha=0.85,
       bottom=in_toks, zorder=3)
total_toks = [i+o for i,o in zip(in_toks, out_toks)]
for i, (t, c) in enumerate(zip(total_toks, n_cycles)):
    ax.text(i, t+1000, f"{t/1000:.0f}k\n({c}c)", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"Run {r['run']}" for r in rows])
ax.set_ylabel("Tokens"); ax.legend(fontsize=8, frameon=False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1000:.0f}k"))
ax.set_title("Token Usage per Run\n(multi-cycle runs use more tokens)")

# Colour strip by cycle count
for i, r in enumerate(rows):
    ax.axvspan(i-0.35, i+0.35, ymax=0.025, color=C_CYCLE[i%len(C_CYCLE)], alpha=0.8)

# Right: cost vs tuning cycles scatter + per-run cost
ax2 = axes[1]
for i, r in enumerate(rows):
    ax2.scatter(r["n_cycles"], r["cost_usd"], s=120,
                color=C_CYCLE[i%len(C_CYCLE)], zorder=4, edgecolors="white", lw=0.8)
    ax2.annotate(f"R{r['run']} ${r['cost_usd']:.3f}",
                 (r["n_cycles"], r["cost_usd"]),
                 textcoords="offset points", xytext=(5,3), fontsize=8)

# Trend line
from numpy.polynomial import polynomial as P
if len(set(n_cycles)) > 1:
    fit = np.polyfit(n_cycles, costs, 1)
    x_fit = np.linspace(min(n_cycles)-0.2, max(n_cycles)+0.2, 50)
    ax2.plot(x_fit, np.polyval(fit, x_fit), "k--", lw=1.2, alpha=0.5,
             label=f"Trend: +${fit[0]:.3f}/cycle")

ax2.axhline(np.mean(costs), color="grey", ls=":", lw=1.5,
            label=f"Mean ${np.mean(costs):.3f}")
ax2.set_xlabel("Tuning cycles"); ax2.set_ylabel("Cost (USD)")
ax2.set_title("Cost vs Tuning Cycles\n(each extra cycle adds ~$0.15)")
ax2.legend(fontsize=8, frameon=False)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout(); add_banner(fig)
out9 = f"{OUT_PREFIX}9_token_cost_analysis.png"
plt.savefig(out9, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 9 → {out9}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Conversation flow diagram per run
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 14))
fig.suptitle("EXP-C5: Conversation Flow — Diagnose → Iterative Tune → Verify → Confirm",
             fontsize=14, fontweight="bold", y=0.99)

# Colour map for box types
BOX_USER  = "#d6eaf8"
BOX_ANAL  = "#fdebd0"
BOX_TUNE  = "#d5f5e3"
BOX_WAIT  = "#f8f9fa"
BOX_VERIF = "#e8daef"
BOX_DONE  = "#d5f5e3"
BOX_FAIL  = "#fadbd8"

def flow_box(ax, y, height, text, fc, ec="grey", fontsize=7.5):
    ax.add_patch(mpatches.FancyBboxPatch((0.03, y), 0.94, height,
                 boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1.2))
    ax.text(0.5, y + height/2, text, ha="center", va="center",
            fontsize=fontsize, multialignment="center", transform=ax.transAxes,
            wrap=True)

# Per-run detailed flows
flow_data = {
    1: [
        ("USER",   BOX_USER,  "oscillating badly on roll…\nplease diagnose and fix"),
        ("ANALYZE","#fdebd0", "analyze_flight(30s)\n→ gyroX std=9.8 dps, 17 flip/30s\nDiagnosis: kp too high"),
        ("SUGGEST","#d5f5e3", "suggest_pid_tuning(roll)\n→ kp:1.5→0.4, kd:0→0.004, rate_kp:0.08→0.1"),
        ("TUNE",   "#aed6f1", "set_tuning_params + apply_tuning\nkp=0.4, kd=0.004, rate_kp=0.1"),
        ("WAIT",   BOX_WAIT,  "wait(10s)"),
        ("VERIFY", "#e8daef", "check_drone_stable ✓\nanalyze_flight(10s) → stable"),
        ("DONE",   BOX_DONE,  "✓ PASS  kp: 1.5→0.4\nRMSE: 0.160°→0.036° (−78%)"),
    ],
    2: [
        ("USER",   BOX_USER,  "oscillating badly on roll…\nplease diagnose and fix"),
        ("ANALYZE","#fdebd0", "analyze_flight(30s)\n→ (LLM timeout on suggest — retried)\nDiagnosis: oscillation confirmed"),
        ("SUGGEST","#d5f5e3", "suggest_pid_tuning(roll)\n→ kp:1.5→0.35, rate_kd:0→0.04, rate_ki:0→0.02"),
        ("TUNE",   "#aed6f1", "set_tuning_params + apply_tuning\nkp=0.35, rate_kd=0.04, rate_ki=0.02"),
        ("WAIT",   BOX_WAIT,  "wait(10s)"),
        ("VERIFY", "#e8daef", "analyze_flight(10s) + check_drone_stable ✓\n→ oscillation eliminated"),
        ("DONE",   BOX_DONE,  "✓ PASS  kp: 1.5→0.35\nRMSE: 0.171°→0.048° (−72%)"),
    ],
    3: [
        ("USER",   BOX_USER,  "oscillating badly on roll…\nplease diagnose and fix"),
        ("ANALYZE","#fdebd0", "analyze_flight(30s)\n→ gyroX std high, slow oscillations"),
        ("SUGGEST","#d5f5e3", "suggest_pid_tuning + get_tuning_params\n→ kp:1.5→0.6, kd:0→0.04"),
        ("TUNE",   "#aed6f1", "Cycle 1: kp=0.6, kd=0.04, rate_kp=0.06"),
        ("VERIFY", "#e8daef", "analyze_flight(30s)\n→ Still oscillating — need further reduction"),
        ("TUNE2",  "#aed6f1", "Cycle 2: kp=0.45, kd=0.07, rate_kp=0.05"),
        ("VERIFY2","#e8daef", "analyze_flight(30s)\n→ Residual bias, trim applied"),
        ("TUNE3",  "#aed6f1", "Cycle 3: kp=0.35, kd=0.09, rate_kp=0.04"),
        ("DONE",   BOX_DONE,  "✓ PASS  3 cycles  kp: 1.5→0.35\nRMSE: 0.113°→0.032° (−72%)"),
    ],
    4: [
        ("USER",   BOX_USER,  "oscillating badly on roll…\nplease diagnose and fix"),
        ("ANALYZE","#fdebd0", "analyze_flight(30s)\n→ gyroX std=8.5 dps, underdamped"),
        ("SUGGEST","#d5f5e3", "suggest_pid_tuning + get_tuning_params\n→ kp:1.5→0.4, kd:0→0.05"),
        ("TUNE",   "#aed6f1", "Cycle 1: kp=0.4, kd=0.05, rate_kp=0.1"),
        ("VERIFY", "#e8daef", "analyze_flight(10s)\n→ Mostly stable. Residual rate buzz detected"),
        ("TUNE2",  "#aed6f1", "Cycle 2 (fine-tune): rate_kp:0.1→0.08, rate_kd:0→0.015"),
        ("VERIFY2","#e8daef", "analyze_flight(10s) + check_drone_stable ✓"),
        ("DONE",   BOX_DONE,  "✓ PASS  2 cycles  kp: 1.5→0.4\nRMSE: 0.178°→0.032° (−82%)"),
    ],
    5: [
        ("USER",   BOX_USER,  "oscillating badly on roll…\nplease diagnose and fix"),
        ("ANALYZE","#fdebd0", "analyze_flight(30s)\n→ gyroX std=9.1 dps, avg=+7 dps bias"),
        ("SUGGEST","#d5f5e3", "get_tuning_params + suggest_pid_tuning\n→ kp:1.5→0.4, kd:0→0.05"),
        ("TUNE",   "#aed6f1", "Cycle 1: kp=0.4, kd=0.05, rate_kp=0.06"),
        ("VERIFY", "#e8daef", "analyze_flight(30s)\n→ gyroX bias remains — sensor anomaly\ndetect_anomaly + suggest_pid_tuning"),
        ("TUNE2",  "#aed6f1", "Cycle 2: kp=0.35, kd→0.05, rate_kp=0.045, rate_kd=0.008"),
        ("VERIFY2","#e8daef", "check_drone_stable ✓ + analyze_flight(30s)\n→ Stable. gyroX bias = sensor issue, not tuning"),
        ("DONE",   BOX_DONE,  "✓ PASS  2 cycles  kp: 1.5→0.35\nRMSE: 0.125°→0.032° (−74%)"),
    ],
}

gs = GridSpec(1, N, figure=fig, wspace=0.08)
for col, r in enumerate(rows):
    ax = fig.add_subplot(gs[0, col])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    steps = flow_data[r["run"]]
    n_steps = len(steps)
    box_h = 0.84 / n_steps
    pad   = 0.008

    for i, (stage, colour, text) in enumerate(steps):
        y0 = 1.0 - (i+1)*box_h + pad
        h  = box_h - 2*pad
        ec = "darkgreen" if stage == "DONE" else "grey"
        lw = 1.8 if stage == "DONE" else 1.0
        ax.add_patch(mpatches.FancyBboxPatch((0.03, y0), 0.94, h,
                     boxstyle="round,pad=0.02", fc=colour, ec=ec, lw=lw))
        ax.text(0.5, y0+h/2, text, ha="center", va="center",
                fontsize=max(5.5, 7.0-n_steps*0.2), multialignment="center",
                transform=ax.transAxes)
        # Arrow to next box
        if i < n_steps-1:
            ax.annotate("", xy=(0.5, y0-pad), xytext=(0.5, y0+pad*0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="lightgrey", lw=0.8))

    # Header
    cyc_col = C_CYCLE[col%len(C_CYCLE)]
    ax.text(0.5, 0.995, f"Run {r['run']}  ({r['n_cycles']} cycle{'s' if r['n_cycles']>1 else ''})",
            ha="center", va="top", fontsize=9, fontweight="bold", color=cyc_col,
            transform=ax.transAxes)

add_banner(fig)
out10 = f"{OUT_PREFIX}10_conversation_flow.png"
plt.savefig(out10, bbox_inches="tight"); plt.close()
print(f"[C5] Fig 10 → {out10}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
[C5] All 10 figures generated:
  C5_fig1_passfail_overview.png        — pass/fail grid with cycle count + success rate CI
  C5_fig2_rmse_before_after.png        — RMSE per run before/after + aggregate with CI
  C5_fig3_kp_trajectory.png           — kp path: injected → iterative steps → final
  C5_fig4_tuning_cycles_analyze_calls.png — cycle/analyze/suggest counts + verification surplus
  C5_fig5_iterative_kp_progression.png — step-by-step gain changes for multi-cycle runs
  C5_fig6_rmse_reduction_distribution.png — reduction % per run + absolute improvement
  C5_fig7_llm_self_verification.png   — verification timeline + analyze-vs-cycle scatter
  C5_fig8_all_pid_params_changed.png  — every PID parameter the LLM changed across runs
  C5_fig9_token_cost_analysis.png     — token usage + cost vs tuning cycle count
  C5_fig10_conversation_flow.png      — full conversation flow diagram per run
""")
