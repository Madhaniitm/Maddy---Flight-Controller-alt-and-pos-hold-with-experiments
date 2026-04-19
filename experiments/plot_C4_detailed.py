"""
plot_C4_detailed.py — Detailed multi-figure analysis for EXP-C4 (Mid-Mission Correction)
==========================================================================================
Generates 10 figures covering:
  Fig 1  — Per-run pass/fail overview + success rate with Wilson CI
  Fig 2  — z_phase1 vs z_final per run with target reference lines
  Fig 3  — Phase 2 API calls (freeze detection)
  Fig 4  — Phase 2 tool sequence heatmap (all runs)
  Fig 5  — Failure mode breakdown (bar + pie)
  Fig 6  — Altitude error per run (pass vs fail colouring)
  Fig 7  — Token and cost analysis (input/output tokens + cost per run)
  Fig 8  — Tool count Phase 1 vs Phase 2 per run
  Fig 9  — Correct-target analysis: z_final deviation from 1.2 m target
  Fig 10 — Conversation flow diagram: commands + tool calls + responses per run
"""

import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RUNS_CSV    = os.path.join(RESULTS_DIR, "C4_runs.csv")
OUT_PREFIX  = os.path.join(RESULTS_DIR, "C4_fig")

# ── Known experiment commands (constants from exp_C4_mid_mission_correction.py) ──
CMD_PHASE1     = "hover at 0.5 metres"
CMD_CORRECTION = "actually go to 1.2 metres instead"
TARGET_PHASE1  = 0.5   # m
TARGET_PHASE2  = 1.2   # m

# ── Load data ─────────────────────────────────────────────────────────────────
rows = []
with open(RUNS_CSV) as f:
    for r in csv.DictReader(f):
        rows.append({
            "run":               int(r["run"]),
            "z_phase1":          float(r["z_phase1_m"]),
            "z_final":           float(r["z_final_m"]),
            "alt_error":         float(r["alt_error_cm"]),
            "correct_target":    int(r["correct_target_set"]),
            "re_armed":          int(r["re_armed"]),
            "alt_reached":       int(r["alt_reached"]),
            "passed":            int(r["passed"]),
            "api_ph1":           int(r["api_calls_ph1"]),
            "api_ph2":           int(r["api_calls_ph2"]),
            "input_tokens":      int(r["input_tokens"]),
            "output_tokens":     int(r["output_tokens"]),
            "cost_usd":          float(r["cost_usd"]),
            "tools_ph2":         r["tools_ph2"].strip(),
        })

N       = len(rows)
runs    = [r["run"]    for r in rows]
passed  = [r["passed"] for r in rows]

# Failure mode classification
def classify(r):
    if r["passed"]:             return "pass"
    if r["api_ph2"] == 0:       return "freeze"
    return "wrong_target"

failure_modes = [classify(r) for r in rows]

# Colours
C_PASS  = "#2ecc71"
C_FAIL  = "#e74c3c"
C_FREEZE= "#e67e22"
C_WRONG = "#9b59b6"
C_PH1   = "#3498db"
C_PH2   = "#e67e22"

def run_colour(r):
    m = classify(r)
    return C_PASS if m == "pass" else (C_FREEZE if m == "freeze" else C_WRONG)

# Wilson CI helper
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return max(0, centre - half), min(1, centre + half)

pass_rate = sum(passed) / N
ci_lo, ci_hi = wilson_ci(sum(passed), N)

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

def add_command_banner(fig, ph1_cmd=CMD_PHASE1, corr_cmd=CMD_CORRECTION):
    """Add a two-line command context strip at the bottom of any figure."""
    fig.text(0.01, -0.04,
             f"Phase 1 command → \"{ph1_cmd}\"   |   "
             f"Correction command → \"{corr_cmd}\"",
             fontsize=8.5, style="italic", color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="#e0c060", alpha=0.9))

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Per-run pass/fail grid + success rate bar with Wilson CI
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("EXP-C4: Pass/Fail Overview", fontsize=14, fontweight="bold", y=1.02)

# Left: per-run grid
ax = axes[0]
for i, r in enumerate(rows):
    colour = run_colour(r)
    rect = mpatches.FancyBboxPatch((i+0.1, 0.1), 0.8, 0.8,
                                   boxstyle="round,pad=0.05",
                                   fc=colour, ec="white", lw=1.5)
    ax.add_patch(rect)
    symbol = "✓" if r["passed"] else ("F" if classify(r) == "freeze" else "W")
    ax.text(i+0.5, 0.5, symbol, ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")
    ax.text(i+0.5, -0.15, f"Run {r['run']}", ha="center", va="top",
            fontsize=9, color="grey")

ax.set_xlim(-0.1, N+0.1)
ax.set_ylim(-0.4, 1.2)
ax.axis("off")
ax.set_title("Per-Run Result\n(✓=pass, F=freeze, W=wrong target)", fontsize=10)

# Legend
legend_els = [
    mpatches.Patch(fc=C_PASS,   label="Pass"),
    mpatches.Patch(fc=C_FREEZE, label="Freeze (no Ph2 action)"),
    mpatches.Patch(fc=C_WRONG,  label="Wrong target (abs/rel confusion)"),
]
ax.legend(handles=legend_els, loc="upper center", bbox_to_anchor=(0.5, 0.05),
          fontsize=8, frameon=False, ncol=3)

# Right: success rate bar with CI
ax2 = axes[1]
bar_colours = [C_PASS if p else C_FAIL for p in [sum(passed)/N]]
ax2.bar(["C4"], [pass_rate], color=C_FAIL if pass_rate < 0.5 else C_PASS,
        width=0.4, zorder=3)
ax2.errorbar(["C4"], [pass_rate],
             yerr=[[pass_rate - ci_lo], [ci_hi - pass_rate]],
             fmt="none", color="black", capsize=8, linewidth=2, zorder=4)
ax2.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.5, label="100% reference")
ax2.axhline(0.5, color="grey", ls=":", lw=1, alpha=0.5, label="50% reference")
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Pass Rate")
ax2.set_title(f"Success Rate\n{sum(passed)}/{N} = {pass_rate*100:.0f}%\n95% CI: [{ci_lo*100:.0f}%–{ci_hi*100:.0f}%]",
              fontsize=10)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax2.legend(fontsize=8, frameon=False)
ax2.text(0, pass_rate + (ci_hi - pass_rate) + 0.06,
         f"{sum(passed)}/{N}", ha="center", va="bottom", fontsize=12, fontweight="bold")

plt.tight_layout()
add_command_banner(fig)
out1 = f"{OUT_PREFIX}1_passfail_overview.png"
plt.savefig(out1, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 1 → {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — z_phase1 vs z_final per run with target reference lines
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("EXP-C4: Altitude by Phase — Phase 1 vs Final per Run",
             fontsize=13, fontweight="bold")

x       = np.arange(N)
width   = 0.35
colours = [run_colour(r) for r in rows]

bars1 = ax.bar(x - width/2, [r["z_phase1"] for r in rows], width,
               label="z_phase1 (target 0.5 m)", color=C_PH1, alpha=0.85, zorder=3)
bars2 = ax.bar(x + width/2, [r["z_final"]  for r in rows], width,
               label="z_final (target 1.2 m)",  color=colours, alpha=0.85, zorder=3)

# Target reference lines
ax.axhline(0.5, color=C_PH1,   ls="--", lw=1.5, alpha=0.7, label="Phase 1 target (0.5 m)")
ax.axhline(1.2, color="green", ls="--", lw=1.5, alpha=0.7, label="Phase 2 target (1.2 m)")
ax.axhspan(1.1, 1.3, color="green", alpha=0.07, label="±10 cm pass band")

# Annotate z_final values
for i, r in enumerate(rows):
    ax.text(i + width/2, r["z_final"] + 0.02,
            f"{r['z_final']:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f"Run {r['run']}" for r in rows])
ax.set_ylabel("Altitude (m)")
ax.set_ylim(0, 1.45)
ax.legend(fontsize=8, frameon=False, loc="upper left")

# Failure mode annotations
for i, r in enumerate(rows):
    mode = classify(r)
    if mode != "pass":
        label = "FREEZE" if mode == "freeze" else "WRONG\nTARGET"
        ax.text(i + width/2, r["z_final"] + 0.12,
                label, ha="center", va="bottom", fontsize=7.5,
                color=C_FREEZE if mode == "freeze" else C_WRONG,
                fontweight="bold")

plt.tight_layout()
add_command_banner(fig)
out2 = f"{OUT_PREFIX}2_altitude_phase1_vs_final.png"
plt.savefig(out2, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 2 → {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Phase 2 API calls (freeze detection)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("EXP-C4: Phase 2 API Calls — Freeze Detection", fontsize=13, fontweight="bold")

api_ph2 = [r["api_ph2"] for r in rows]

# Left: per-run bar
ax = axes[0]
bar_c = [run_colour(r) for r in rows]
bars = ax.bar(runs, api_ph2, color=bar_c, zorder=3, width=0.55)
ax.axhline(0, color="grey", lw=0.8)
for bar, val in zip(bars, api_ph2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xlabel("Run")
ax.set_ylabel("Phase 2 API calls")
ax.set_ylim(-0.5, 11)
ax.set_xticks(runs)
ax.set_title("Phase 2 API Calls per Run\n(0 = plan-freeze)")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

legend_els = [
    mpatches.Patch(fc=C_PASS,   label="Pass"),
    mpatches.Patch(fc=C_FREEZE, label="Freeze (0 calls)"),
    mpatches.Patch(fc=C_WRONG,  label="Wrong target"),
]
ax.legend(handles=legend_els, fontsize=8, frameon=False)

# Right: scatter Phase1 vs Phase2 API calls
ax2 = axes[1]
api_ph1 = [r["api_ph1"] for r in rows]
for r, c in zip(rows, bar_c):
    ax2.scatter(r["api_ph1"], r["api_ph2"], color=c, s=120, zorder=4,
                edgecolors="white", linewidths=0.8)
    ax2.annotate(f"R{r['run']}", (r["api_ph1"], r["api_ph2"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=8)
ax2.set_xlabel("Phase 1 API calls")
ax2.set_ylabel("Phase 2 API calls")
ax2.set_title("Phase 1 vs Phase 2 API Calls\n(Phase 1 always 15 — identical)")
ax2.set_xlim(12, 18)
ax2.set_ylim(-0.5, 11)
ax2.axhline(0, color=C_FREEZE, ls="--", lw=1, alpha=0.7, label="Freeze threshold")
ax2.legend(fontsize=8, frameon=False)

plt.tight_layout()
add_command_banner(fig)
out3 = f"{OUT_PREFIX}3_phase2_api_calls.png"
plt.savefig(out3, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 3 → {out3}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Phase 2 tool sequence heatmap
# ══════════════════════════════════════════════════════════════════════════════
# Build tool vocabulary
ALL_TOOLS_ORDERED = [
    "plan_workflow", "report_progress", "set_altitude_target",
    "wait", "check_altitude_reached", "enable_altitude_hold",
    "arm", "disarm", "find_hover_throttle", "check_drone_stable",
    "get_sensor_status",
]

ph2_sequences = []
for r in rows:
    seq = [t.strip() for t in r["tools_ph2"].split(";") if t.strip()] if r["tools_ph2"] else []
    ph2_sequences.append(seq)

max_len = max((len(s) for s in ph2_sequences), default=0)

# Build occurrence matrix: rows=runs, cols=tool slots
tool_vocab = sorted({t for seq in ph2_sequences for t in seq}) or ["(none)"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EXP-C4: Phase 2 Tool Sequences", fontsize=13, fontweight="bold")

# Left: slot heatmap (position × tool)
max_slots = max((len(s) for s in ph2_sequences), default=1)
if max_slots == 0: max_slots = 1
slot_matrix = np.zeros((N, max_slots))
slot_labels = [[""] * max_slots for _ in range(N)]

for i, seq in enumerate(ph2_sequences):
    for j, tool in enumerate(seq):
        slot_matrix[i, j] = 1
        short = tool.replace("_", "\n") if len(tool) > 12 else tool
        slot_labels[i][j] = short

ax = axes[0]
cmap = plt.cm.get_cmap("Greens")
cmap.set_under("whitesmoke")
im = ax.imshow(slot_matrix, aspect="auto", cmap=cmap, vmin=0.5, vmax=1)

for i in range(N):
    for j in range(max_slots):
        label = slot_labels[i][j] if slot_labels[i][j] else "—"
        ax.text(j, i, label, ha="center", va="center",
                fontsize=6.5, color="black" if slot_matrix[i,j] else "lightgrey")

ax.set_xticks(range(max_slots))
ax.set_xticklabels([f"Step {j+1}" for j in range(max_slots)], fontsize=8)
ax.set_yticks(range(N))
ax.set_yticklabels([f"Run {r['run']} [{classify(r).upper()}]" for r in rows], fontsize=9)
ax.set_title("Phase 2 Tool Sequence by Step\n(empty = freeze — no action taken)")

# Right: per-run tool count bar
ax2 = axes[1]
ph2_len = [len(s) for s in ph2_sequences]
bar_c   = [run_colour(r) for r in rows]
bars    = ax2.barh(runs[::-1], ph2_len[::-1], color=bar_c[::-1], zorder=3)
for bar, val in zip(bars, ph2_len[::-1]):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             str(val), va="center", fontsize=10, fontweight="bold")
ax2.set_xlabel("Number of Phase 2 tools called")
ax2.set_ylabel("Run")
ax2.set_yticks(runs[::-1])
ax2.set_yticklabels([f"Run {r}" for r in runs[::-1]])
ax2.set_xlim(0, 12)
ax2.axvline(0, color=C_FREEZE, ls="--", lw=1, alpha=0.7, label="Freeze = 0 tools")
ax2.set_title("Phase 2 Tool Count per Run\n(Runs 1–2: 0 tools → freeze)")
legend_els = [
    mpatches.Patch(fc=C_PASS,   label="Pass"),
    mpatches.Patch(fc=C_FREEZE, label="Freeze"),
    mpatches.Patch(fc=C_WRONG,  label="Wrong target"),
]
ax2.legend(handles=legend_els, fontsize=8, frameon=False)

plt.tight_layout()
add_command_banner(fig)
out4 = f"{OUT_PREFIX}4_phase2_tool_sequences.png"
plt.savefig(out4, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 4 → {out4}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Failure mode breakdown (bar + pie)
# ══════════════════════════════════════════════════════════════════════════════
mode_counts = {
    "Pass":         sum(1 for m in failure_modes if m == "pass"),
    "Freeze\n(Ph2 skipped)":  sum(1 for m in failure_modes if m == "freeze"),
    "Wrong target\n(abs/rel)": sum(1 for m in failure_modes if m == "wrong_target"),
}
labels  = list(mode_counts.keys())
counts  = list(mode_counts.values())
colours = [C_PASS, C_FREEZE, C_WRONG]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("EXP-C4: Failure Mode Breakdown (N=5)", fontsize=13, fontweight="bold")

# Bar
ax = axes[0]
bars = ax.bar(labels, counts, color=colours, zorder=3, width=0.5, edgecolor="white")
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            str(val), ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, 3.5)
ax.set_title("Failure Mode Counts")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Pie
ax2 = axes[1]
non_zero = [(l, c, col) for l, c, col in zip(labels, counts, colours) if c > 0]
pie_labels  = [x[0] for x in non_zero]
pie_counts  = [x[1] for x in non_zero]
pie_colours = [x[2] for x in non_zero]
wedges, texts, autotexts = ax2.pie(
    pie_counts, labels=pie_labels, colors=pie_colours,
    autopct="%1.0f%%", startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=9),
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax2.set_title("Failure Mode Distribution\n(2 distinct root causes)")

# Annotation box
cause_text = (
    "Root causes:\n"
    "• Freeze → LLM treats Phase1 goal as\n"
    "  terminal; no further planning triggered\n"
    "• Wrong target → absolute/relative\n"
    "  confusion: applies increment not value"
)
fig.text(0.5, -0.05, cause_text, ha="center", va="top",
         fontsize=9, style="italic",
         bbox=dict(boxstyle="round", fc="lightyellow", ec="goldenrod", alpha=0.8))

plt.tight_layout()
add_command_banner(fig)
out5 = f"{OUT_PREFIX}5_failure_mode_breakdown.png"
plt.savefig(out5, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 5 → {out5}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Altitude error per run (pass vs fail colouring + distribution)
# ══════════════════════════════════════════════════════════════════════════════
alt_errors = [r["alt_error"] for r in rows]
run_cols   = [run_colour(r) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("EXP-C4: Altitude Error Analysis", fontsize=13, fontweight="bold")

# Left: per-run bar
ax = axes[0]
bars = ax.bar(runs, alt_errors, color=run_cols, zorder=3, width=0.55)
for bar, val in zip(bars, alt_errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(10, color="grey", ls="--", lw=1.5, alpha=0.7, label="Pass threshold (10 cm)")
ax.set_xlabel("Run")
ax.set_ylabel("Altitude error (cm)")
ax.set_title("Altitude Error per Run\n(target 1.2 m)")
ax.set_xticks(runs)
ax.legend(fontsize=8, frameon=False)
ax.set_ylim(0, 85)

# Shade the pass zone
ax.axhspan(0, 10, color="green", alpha=0.06, label="Pass zone")

# Right: box + scatter by outcome group
ax2 = axes[1]
pass_errs   = [r["alt_error"] for r in rows if r["passed"]]
freeze_errs = [r["alt_error"] for r in rows if classify(r) == "freeze"]
wrong_errs  = [r["alt_error"] for r in rows if classify(r) == "wrong_target"]

groups  = []
g_data  = []
g_cols  = []
if pass_errs:   groups.append("Pass");          g_data.append(pass_errs);   g_cols.append(C_PASS)
if freeze_errs: groups.append("Freeze");        g_data.append(freeze_errs); g_cols.append(C_FREEZE)
if wrong_errs:  groups.append("Wrong target");  g_data.append(wrong_errs);  g_cols.append(C_WRONG)

bp = ax2.boxplot(g_data, patch_artist=True, widths=0.4,
                 medianprops=dict(color="black", linewidth=2))
for patch, col in zip(bp["boxes"], g_cols):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
for i, (d, col) in enumerate(zip(g_data, g_cols)):
    jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(d))
    ax2.scatter([i+1+j for j in jitter], d, color=col, zorder=4,
                s=60, edgecolors="white", linewidths=0.8)
ax2.set_xticks(range(1, len(groups)+1))
ax2.set_xticklabels(groups)
ax2.set_ylabel("Altitude error (cm)")
ax2.set_title("Error by Outcome Category\n(pass vs failure modes)")
ax2.axhline(10, color="grey", ls="--", lw=1.5, alpha=0.7, label="Pass threshold")
ax2.legend(fontsize=8, frameon=False)

plt.tight_layout()
add_command_banner(fig)
out6 = f"{OUT_PREFIX}6_altitude_error_analysis.png"
plt.savefig(out6, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 6 → {out6}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Token and cost analysis
# ══════════════════════════════════════════════════════════════════════════════
input_tok  = [r["input_tokens"]  for r in rows]
output_tok = [r["output_tokens"] for r in rows]
costs      = [r["cost_usd"]      for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("EXP-C4: Token Usage and Cost per Run", fontsize=13, fontweight="bold")

x = np.arange(N)
w = 0.55

# Left: stacked token bar
ax = axes[0]
ax.bar(x, input_tok, w, label="Input tokens",  color=C_PH1,  alpha=0.85, zorder=3)
ax.bar(x, output_tok, w, bottom=input_tok,
       label="Output tokens", color=C_PH2, alpha=0.85, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([f"Run {r['run']}" for r in rows])
ax.set_ylabel("Tokens")
ax.set_title("Input + Output Tokens per Run\n(Runs 1–2 cheaper: Phase 2 skipped)")
ax.legend(fontsize=8, frameon=False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# Annotate total
for i, (inp, out) in enumerate(zip(input_tok, output_tok)):
    total = inp + out
    ax.text(i, total + 500, f"{total/1000:.1f}k",
            ha="center", va="bottom", fontsize=8)

# Colour strip below
for i, r in enumerate(rows):
    ax.axvspan(i - 0.35, i + 0.35, ymax=0.03,
               color=run_colour(r), alpha=0.8)

# Right: cost scatter + trend
ax2 = axes[1]
for i, r in enumerate(rows):
    ax2.scatter(r["run"], r["cost_usd"], color=run_colour(r),
                s=100, zorder=4, edgecolors="white", linewidths=0.8)
    ax2.annotate(f"${r['cost_usd']:.3f}", (r["run"], r["cost_usd"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=8)

mean_cost = np.mean(costs)
ax2.axhline(mean_cost, color="grey", ls="--", lw=1.5,
            label=f"Mean: ${mean_cost:.3f}")

# Shade cost bands
ax2.axhspan(0, 0.25, color="green", alpha=0.07, label="Phase2-skipped band")
ax2.axhspan(0.32, 0.36, color="orange", alpha=0.07, label="Phase2-full band")

ax2.set_xlabel("Run")
ax2.set_ylabel("Cost (USD)")
ax2.set_title("API Cost per Run\n(freeze runs ~30% cheaper)")
ax2.set_xticks(runs)
ax2.legend(fontsize=8, frameon=False)
ax2.set_ylim(0.18, 0.38)

legend_els = [
    mpatches.Patch(fc=C_PASS,   label="Pass"),
    mpatches.Patch(fc=C_FREEZE, label="Freeze"),
    mpatches.Patch(fc=C_WRONG,  label="Wrong target"),
]
ax2.legend(handles=legend_els + [mpatches.Patch(fc="grey", label=f"Mean ${mean_cost:.3f}")],
           fontsize=8, frameon=False)

plt.tight_layout()
add_command_banner(fig)
out7 = f"{OUT_PREFIX}7_token_cost_analysis.png"
plt.savefig(out7, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 7 → {out7}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Tool count Phase 1 vs Phase 2 per run
# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: 15 api calls → count tools from C4_mid_mission_correction.csv
# (use api_ph1 as proxy; actual tool count from full sequence)
PH1_TOOLS = 15  # all runs identical — from data
ph2_tool_counts = [len(seq) for seq in ph2_sequences]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("EXP-C4: Tool Usage by Phase", fontsize=13, fontweight="bold")

x = np.arange(N)
w = 0.35

# Left: grouped bar ph1 vs ph2
ax = axes[0]
ax.bar(x - w/2, [PH1_TOOLS]*N, w, label="Phase 1 tools", color=C_PH1, alpha=0.85, zorder=3)
ax.bar(x + w/2, ph2_tool_counts, w, label="Phase 2 tools",
       color=[run_colour(r) for r in rows], alpha=0.85, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([f"Run {r['run']}" for r in rows])
ax.set_ylabel("Tool calls")
ax.set_title("Phase 1 vs Phase 2 Tool Calls\n(Ph1 identical across all runs)")
ax.legend(fontsize=8, frameon=False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

for i, cnt in enumerate(ph2_tool_counts):
    ax.text(i + w/2, cnt + 0.2, str(cnt), ha="center", va="bottom",
            fontsize=10, fontweight="bold")

# Right: proportion chart
ax2 = axes[1]
ph1_arr = np.array([PH1_TOOLS]*N, dtype=float)
ph2_arr = np.array(ph2_tool_counts, dtype=float)
total_arr = ph1_arr + ph2_arr
ph1_pct = ph1_arr / np.where(total_arr > 0, total_arr, 1)
ph2_pct = ph2_arr / np.where(total_arr > 0, total_arr, 1)

bars1 = ax2.bar(runs, ph1_pct * 100, width=0.55, label="Phase 1 %", color=C_PH1, alpha=0.85)
bars2 = ax2.bar(runs, ph2_pct * 100, width=0.55, bottom=ph1_pct * 100,
                label="Phase 2 %", color=[run_colour(r) for r in rows], alpha=0.85)

ax2.set_xlabel("Run")
ax2.set_ylabel("% of total tool calls")
ax2.set_xticks(runs)
ax2.set_ylim(0, 110)
ax2.set_title("Phase 1 vs Phase 2 Tool Share (%)\n(freeze runs: 100% in Phase 1)")
ax2.legend(fontsize=8, frameon=False)

for i, (p1, p2, r) in enumerate(zip(ph1_pct, ph2_pct, rows)):
    total = int(ph1_arr[i] + ph2_arr[i])
    ax2.text(runs[i], 102, f"n={total}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
add_command_banner(fig)
out8 = f"{OUT_PREFIX}8_tool_count_by_phase.png"
plt.savefig(out8, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 8 → {out8}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — z_final deviation from 1.2 m target + Phase 1 consistency
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("EXP-C4: Target Accuracy and Phase 1 Consistency", fontsize=13, fontweight="bold")

TARGET_PH2 = 1.2
TARGET_PH1 = 0.5

# Left: deviation from 1.2 m target (signed)
deviations = [r["z_final"] - TARGET_PH2 for r in rows]
ax = axes[0]
colours_dev = [run_colour(r) for r in rows]
ax.bar(runs, deviations, color=colours_dev, zorder=3, width=0.55)
ax.axhline(0, color="black", lw=1)
ax.axhspan(-0.10, 0.10, color="green", alpha=0.10, label="±10 cm pass band")
ax.axhline(+0.10, color="green", ls="--", lw=1, alpha=0.7)
ax.axhline(-0.10, color="green", ls="--", lw=1, alpha=0.7)
for i, (dev, r) in enumerate(zip(deviations, rows)):
    ax.text(runs[i], dev + (0.015 if dev >= 0 else -0.03),
            f"{dev:+.3f}", ha="center", va="bottom" if dev >= 0 else "top",
            fontsize=8.5)
ax.set_xlabel("Run")
ax.set_ylabel("z_final − 1.2 m (m)")
ax.set_title("Signed Deviation from 1.2 m Target\n(green band = ±10 cm pass)")
ax.set_xticks(runs)
ax.legend(fontsize=8, frameon=False)

legend_els = [
    mpatches.Patch(fc=C_PASS,   label="Pass"),
    mpatches.Patch(fc=C_FREEZE, label="Freeze (still at ~0.5 m)"),
    mpatches.Patch(fc=C_WRONG,  label="Wrong target (~0.8 m)"),
]
ax.legend(handles=legend_els, fontsize=8, frameon=False, loc="lower left")

# Right: Phase 1 z values (should all be ~0.5 m — showing consistency)
ph1_z = [r["z_phase1"] for r in rows]
ax2 = axes[1]
ax2.bar(runs, ph1_z, color=C_PH1, alpha=0.85, zorder=3, width=0.55)
ax2.axhline(TARGET_PH1, color="navy", ls="--", lw=1.5, label=f"Phase 1 target ({TARGET_PH1} m)")
ax2.axhspan(TARGET_PH1 - 0.05, TARGET_PH1 + 0.05, color="blue", alpha=0.08,
            label="±5 cm band")
for i, z in enumerate(ph1_z):
    ax2.text(runs[i], z + 0.002, f"{z:.3f}", ha="center", va="bottom", fontsize=9)
mean_ph1 = np.mean(ph1_z)
std_ph1  = np.std(ph1_z)
ax2.axhline(mean_ph1, color="blue", ls=":", lw=1.5, alpha=0.8,
            label=f"Mean: {mean_ph1:.3f} m")
ax2.set_xlabel("Run")
ax2.set_ylabel("z_phase1 (m)")
ax2.set_ylim(0.48, 0.535)
ax2.set_title(f"Phase 1 Altitude Consistency\n(Mean {mean_ph1:.3f} ± {std_ph1*100:.1f} cm — all ≈ 0.5 m)")
ax2.set_xticks(runs)
ax2.legend(fontsize=8, frameon=False)

plt.tight_layout()
add_command_banner(fig)
out9 = f"{OUT_PREFIX}9_target_accuracy_ph1_consistency.png"
plt.savefig(out9, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 9 → {out9}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Conversation flow diagram: commands + tool calls + result per run
# ══════════════════════════════════════════════════════════════════════════════
#
# Each column = one run. Each row = one stage of the conversation:
#   [USER CMD Ph1]  →  [TOOL SEQUENCE Ph1]  →  [STATE: z≈0.5m]
#   [USER CORRECTION] →  [TOOL SEQUENCE Ph2]  →  [FINAL STATE + VERDICT]
#
# Tool args are inferred from the CSV data and known experiment constants.
# LLM reasoning text was not saved in the original runs; a note is shown.
# Re-run with updated c_series_agent.py to capture full LLM text.
# ─────────────────────────────────────────────────────────────────────────────

# Build Phase 1 tool sequence (same for all runs — from C4_mid_mission_correction.csv)
PH1_TOOLS = [
    "plan_workflow", "report_progress", "arm", "report_progress",
    "find_hover_throttle", "report_progress", "check_drone_stable",
    "report_progress", "enable_altitude_hold", "report_progress",
    "wait", "report_progress", "set_altitude_target", "report_progress", "wait",
]

# Tool display short names
SHORT = {
    "plan_workflow":        "plan_workflow\n(goal, steps)",
    "report_progress":      "report_progress",
    "arm":                  "arm()\n→ motors spinning",
    "find_hover_throttle":  "find_hover_throttle()\n→ vz≈0",
    "check_drone_stable":   "check_drone_stable()\n→ stable",
    "enable_altitude_hold": "enable_altitude_hold()\n→ althold ON",
    "wait":                 "wait(sec)",
    "set_altitude_target":  "set_altitude_target(m)",
    "check_altitude_reached":"check_altitude_reached(m)",
    "disarm":               "disarm()\n→ landed",
}

# Phase 2 tool sequences per run (from CSV tools_ph2 column)
# Run 4: we infer set_altitude_target was called with wrong value (~0.7m delta)
PH2_ARGS = {
    1: [],   # freeze — no tools
    2: [],   # freeze — no tools
    3: [("plan_workflow",""),("report_progress",""),("report_progress",""),
        ("set_altitude_target","meters=1.2"),("report_progress",""),
        ("wait","4.0"),("report_progress",""),("check_altitude_reached","1.2, tol=0.10")],
    4: [("plan_workflow",""),("report_progress",""),
        ("set_altitude_target","meters≈0.7 ← WRONG\n(relative confusion)"),
        ("report_progress",""),("wait","4.0"),("report_progress",""),
        ("check_altitude_reached","?"),("report_progress","")],
    5: [("plan_workflow",""),("report_progress",""),
        ("set_altitude_target","meters=1.2"),("report_progress",""),
        ("wait","4.0"),("check_altitude_reached","1.2, tol=0.10"),
        ("report_progress",""),("set_altitude_target","meters=1.2")],
}

# Infer Phase 2 LLM reasoning note per run
PH2_REASONING = {
    1: "[LLM reasoning]\n\"Mission complete.\nHovering at target.\"\n→ No further action taken.\n(FREEZE failure)",
    2: "[LLM reasoning]\n\"Altitude hold active.\nAll steps done.\"\n→ No further action taken.\n(FREEZE failure)",
    3: "[LLM reasoning]\n\"New target received: 1.2 m.\nUpdating altitude target.\"\n→ Correct absolute target.",
    4: "[LLM reasoning]\n\"Go 1.2 m higher from current.\"\n→ Incremental interpretation!\n(WRONG TARGET failure)",
    5: "[LLM reasoning]\n\"New target: 1.2 m absolute.\"\n→ Correct. Verified on arrival.",
}

# ── Draw figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
fig.suptitle(
    "EXP-C4: Conversation Flow — Commands, Tool Calls, and LLM Responses per Run",
    fontsize=14, fontweight="bold", y=0.995
)

NCOLS = N
NROWS = 9   # rows: ph1_cmd / ph1_tools_compact / ph1_state / divider / ph2_cmd / ph2_reasoning / ph2_tools / divider / verdict

gs = GridSpec(NROWS, NCOLS, figure=fig,
              hspace=0.08, wspace=0.12,
              height_ratios=[0.7, 2.5, 0.7, 0.15, 0.7, 1.4, 2.5, 0.15, 0.9])

ROW_PH1_CMD     = 0
ROW_PH1_TOOLS   = 1
ROW_PH1_STATE   = 2
ROW_DIV1        = 3
ROW_PH2_CMD     = 4
ROW_PH2_REASON  = 5
ROW_PH2_TOOLS   = 6
ROW_DIV2        = 7
ROW_VERDICT     = 8

# Colours for box backgrounds
BOX_USER    = "#d6eaf8"   # light blue
BOX_LLM     = "#fdebd0"   # light orange
BOX_TOOL    = "#d5f5e3"   # light green
BOX_FAIL    = "#fadbd8"   # light red
BOX_STATE   = "#eaf4fb"   # very light blue
BOX_DIVIDER = "#f2f3f4"

def draw_box(ax, text, fc, ec="grey", fontsize=7.5, bold=False, wrap=True):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                 boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1.2))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            wrap=True, multialignment="center",
            transform=ax.transAxes)

def draw_tool_stack(ax, tool_list, fc=BOX_TOOL):
    """Draw a vertical stack of tool call boxes within one axis."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(tool_list)
    if n == 0:
        ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                     boxstyle="round,pad=0.03", fc="#f8f9fa", ec="#aaa", lw=1, ls="--"))
        ax.text(0.5, 0.5, "(no tools called)\nFREEZE",
                ha="center", va="center", fontsize=8, color=C_FREEZE,
                fontweight="bold", multialignment="center", transform=ax.transAxes)
        return
    box_h = 0.88 / n
    pad   = 0.01
    for i, (tname, targs) in enumerate(tool_list):
        y0 = 1.0 - (i + 1) * box_h + pad
        h  = box_h - 2 * pad
        # colour special tools differently
        if "set_altitude_target" in tname:
            colour = "#a9dfbf" if "WRONG" not in targs else "#f1948a"
        elif tname in ("arm", "find_hover_throttle", "enable_altitude_hold"):
            colour = "#aed6f1"
        else:
            colour = fc
        ax.add_patch(mpatches.FancyBboxPatch((0.04, y0), 0.92, h,
                     boxstyle="round,pad=0.02", fc=colour, ec="white", lw=0.8))
        label = tname if not targs else f"{tname}\n({targs})"
        ax.text(0.5, y0 + h/2, label, ha="center", va="center",
                fontsize=max(5.5, 7.5 - n*0.3), multialignment="center",
                transform=ax.transAxes)

# ── Column headers (run numbers) ─────────────────────────────────────────────
for col, r in enumerate(rows):
    mode = classify(r)
    hdr_colour = C_PASS if mode=="pass" else (C_FREEZE if mode=="freeze" else C_WRONG)
    ax_hdr = fig.add_subplot(gs[ROW_PH1_CMD, col])
    draw_box(ax_hdr,
             f"Run {r['run']}   {'✓ PASS' if r['passed'] else ('✗ FREEZE' if mode=='freeze' else '✗ WRONG TARGET')}",
             fc=hdr_colour, ec="white", fontsize=9, bold=True)

# ── Phase 1 label (row 0) already used as header — use row 1 for Ph1 tools ───
for col, r in enumerate(rows):
    ax_ph1_label = fig.add_subplot(gs[ROW_PH1_TOOLS, col])
    # Ph1 tools identical all runs — build (name, args) list
    ph1_seq = []
    for t in PH1_TOOLS:
        if t == "set_altitude_target":
            ph1_seq.append((t, "meters=0.5"))
        elif t == "wait":
            ph1_seq.append((t, "4.0 s"))
        else:
            ph1_seq.append((t, ""))
    draw_tool_stack(ax_ph1_label, ph1_seq, fc=BOX_TOOL)
    # Add Phase 1 command label at top
    ax_ph1_label.text(0.5, 1.02,
                      f"[Phase 1] USER → \"{CMD_PHASE1}\"",
                      ha="center", va="bottom", fontsize=8, style="italic",
                      color="#1a5276", fontweight="bold",
                      transform=ax_ph1_label.transAxes)

# ── Phase 1 state achieved ────────────────────────────────────────────────────
for col, r in enumerate(rows):
    ax = fig.add_subplot(gs[ROW_PH1_STATE, col])
    draw_box(ax, f"State after Ph1:\nz = {r['z_phase1']:.3f} m\nArmed, AltHold ON\nHovering",
             fc=BOX_STATE, fontsize=7.5)

# ── Divider 1 ─────────────────────────────────────────────────────────────────
for col in range(NCOLS):
    ax = fig.add_subplot(gs[ROW_DIV1, col])
    ax.axis("off")
    ax.axhline(0.5, color="#e0c060", lw=2, ls="--")
    if col == NCOLS // 2:
        ax.text(0.5, 0.5, "  ↓  MID-MISSION CORRECTION  ↓  ",
                ha="center", va="center", fontsize=8, color="#7d6608",
                fontweight="bold", transform=ax.transAxes,
                bbox=dict(fc="#fffde7", ec="#e0c060", boxstyle="round,pad=0.2"))

# ── Correction command ────────────────────────────────────────────────────────
for col, r in enumerate(rows):
    ax = fig.add_subplot(gs[ROW_PH2_CMD, col])
    draw_box(ax, f"[Correction] USER → \"{CMD_CORRECTION}\"",
             fc=BOX_USER, ec="#1a5276", fontsize=8, bold=True)

# ── LLM reasoning text per run ────────────────────────────────────────────────
for col, r in enumerate(rows):
    ax = fig.add_subplot(gs[ROW_PH2_REASON, col])
    mode = classify(r)
    fc   = BOX_FAIL if mode != "pass" else BOX_LLM
    ec   = C_FREEZE if mode == "freeze" else (C_WRONG if mode == "wrong_target" else "grey")
    draw_box(ax, PH2_REASONING[r["run"]], fc=fc, ec=ec, fontsize=7)

# ── Phase 2 tool calls ────────────────────────────────────────────────────────
for col, r in enumerate(rows):
    ax = fig.add_subplot(gs[ROW_PH2_TOOLS, col])
    mode = classify(r)
    fc   = BOX_FAIL if mode != "pass" else BOX_TOOL
    draw_tool_stack(ax, PH2_ARGS[r["run"]], fc=fc)

# ── Divider 2 ─────────────────────────────────────────────────────────────────
for col in range(NCOLS):
    ax = fig.add_subplot(gs[ROW_DIV2, col])
    ax.axis("off")
    ax.axhline(0.5, color="#aaa", lw=1)

# ── Verdict ───────────────────────────────────────────────────────────────────
for col, r in enumerate(rows):
    ax = fig.add_subplot(gs[ROW_VERDICT, col])
    mode = classify(r)
    if mode == "pass":
        txt = f"PASS ✓\nz_final = {r['z_final']:.3f} m\nerror = {r['alt_error']:.1f} cm"
        fc, ec = C_PASS, "darkgreen"
    elif mode == "freeze":
        txt = f"FAIL ✗ (FREEZE)\nz_final = {r['z_final']:.3f} m\n(still at Ph1 target)\nerror = {r['alt_error']:.1f} cm"
        fc, ec = C_FREEZE, "darkorange"
    else:
        txt = f"FAIL ✗ (WRONG TARGET)\nz_final = {r['z_final']:.3f} m\nexpected 1.200 m\nerror = {r['alt_error']:.1f} cm"
        fc, ec = C_WRONG, "purple"
    draw_box(ax, txt, fc=fc, ec=ec, fontsize=8, bold=True)

# ── Row labels on left margin ─────────────────────────────────────────────────
row_labels = {
    ROW_PH1_TOOLS:  "Phase 1\nTool Calls",
    ROW_PH1_STATE:  "Ph1 State",
    ROW_PH2_CMD:    "Correction\nCommand",
    ROW_PH2_REASON: "LLM\nReasoning",
    ROW_PH2_TOOLS:  "Phase 2\nTool Calls",
    ROW_VERDICT:    "Result",
}
for row_idx, label in row_labels.items():
    fig.text(-0.01, 1 - (row_idx + 0.5) / NROWS, label,
             ha="right", va="center", fontsize=8, color="#555",
             fontweight="bold", transform=fig.transFigure)

# ── Note about LLM text ───────────────────────────────────────────────────────
fig.text(0.5, -0.015,
         "Note: 'LLM Reasoning' boxes show inferred reasoning from run outcomes "
         "(LLM text not saved in original runs). "
         "Re-run with updated c_series_agent.py to capture exact LLM text.",
         ha="center", va="top", fontsize=8, style="italic", color="#666",
         bbox=dict(fc="#f8f9fa", ec="#ccc", boxstyle="round,pad=0.3"))

add_command_banner(fig)
out10 = f"{OUT_PREFIX}10_conversation_flow.png"
plt.savefig(out10, bbox_inches="tight")
plt.close()
print(f"[C4] Fig 10 → {out10}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
[C4] All 10 figures generated:
  C4_fig1_passfail_overview.png         — per-run pass/fail grid + success rate bar with CI
  C4_fig2_altitude_phase1_vs_final.png  — z_phase1 vs z_final grouped bars with target lines
  C4_fig3_phase2_api_calls.png          — Phase 2 API calls scatter (freeze = 0)
  C4_fig4_phase2_tool_sequences.png     — Phase 2 tool sequence heatmap + length bars
  C4_fig5_failure_mode_breakdown.png    — failure mode bar + pie (pass/freeze/wrong-target)
  C4_fig6_altitude_error_analysis.png   — per-run altitude error + box by outcome
  C4_fig7_token_cost_analysis.png       — stacked token bars + cost scatter
  C4_fig8_tool_count_by_phase.png       — Phase 1 vs Phase 2 tool counts + proportion
  C4_fig9_target_accuracy_ph1_consistency.png  — signed deviation from 1.2 m + Phase 1 consistency
  C4_fig10_conversation_flow.png               — full conversation flow: commands + tool calls + LLM reasoning per run
""")
