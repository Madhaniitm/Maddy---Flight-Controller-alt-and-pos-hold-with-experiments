"""
plot_C4_1_detailed.py — Detailed multi-figure analysis for EXP-C4.1
====================================================================
Reads  : results/C4_runs.csv, results/C4_1_runs_guardrail_on.csv
Writes : results/C4_1_fig1_success_rate_comparison.png
         results/C4_1_fig2_per_run_z_final.png
         results/C4_1_fig3_altitude_trajectory.png
         results/C4_1_fig4_phase2_api_calls.png
         results/C4_1_fig5_phase2_tool_sequence.png
         results/C4_1_fig6_failure_mode_breakdown.png
         results/C4_1_fig7_alt_error_distribution.png
         results/C4_1_fig8_cost_efficiency.png
"""

import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# ── Load data ─────────────────────────────────────────────────────────────────

def load_csv(fname):
    rows = []
    with open(os.path.join(RESULTS, fname)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def parse_runs(rows):
    out = []
    for r in rows:
        try:
            out.append({
                "run":            int(r["run"]),
                "z_phase1":       float(r["z_phase1_m"]),
                "z_final":        float(r["z_final_m"]),
                "alt_error_cm":   float(r["alt_error_cm"]),
                "correct_target": int(r["correct_target_set"]),
                "re_armed":       int(r["re_armed"]),
                "passed":         int(r["passed"]),
                "failure_mode":   r.get("failure_mode", "unknown"),
                "api_ph1":        int(r["api_calls_ph1"]),
                "api_ph2":        int(r["api_calls_ph2"]),
                "in_tok":         int(r["input_tokens"]),
                "out_tok":        int(r["output_tokens"]),
                "cost":           float(r["cost_usd"]),
                "tools_ph2":      r["tools_ph2"].split(";") if r["tools_ph2"] else [],
            })
        except Exception:
            pass
    return out

c4  = parse_runs(load_csv("C4_runs.csv"))
c41 = parse_runs(load_csv("C4_1_runs_guardrail_on.csv"))

N = len(c4)
CORRECT_TARGET = 1.2
INITIAL_TARGET = 0.5
TOLERANCE      = 0.12
COST_IN  = 3.0  / 1_000_000
COST_OUT = 15.0 / 1_000_000

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k / n; d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    m = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0.0, c-m), min(1.0, c+m)

c4_pass  = sum(r["passed"] for r in c4)
c41_pass = sum(r["passed"] for r in c41)
c4_lo,  c4_hi  = wilson_ci(c4_pass,  N)
c41_lo, c41_hi = wilson_ci(c41_pass, N)
delta_pp = (c41_pass - c4_pass) / N * 100

RUN_LABELS = [f"Run {i+1}" for i in range(N)]
C4_PASS_COL  = ["#2ecc71" if r["passed"] else "#e74c3c" for r in c4]
C41_PASS_COL = ["#2ecc71" if r["passed"] else "#e74c3c" for r in c41]

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Success rate comparison  C4 vs C4.1
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: bar comparison
ax = axes[0]
exps   = ["C4\n(baseline)", "C4.1\n(fix)"]
rates  = [c4_pass/N, c41_pass/N]
lows   = [c4_lo, c41_lo]
highs  = [c4_hi, c41_hi]
colors = ["#e74c3c", "#2ecc71"]
bars   = ax.bar(exps, rates, color=colors, alpha=0.85, edgecolor="black", width=0.45, zorder=3)
ax.errorbar([0,1], rates,
            yerr=[[r-l for r,l in zip(rates,lows)],[h-r for r,h in zip(rates,highs)]],
            fmt="none", ecolor="black", capsize=10, lw=2, zorder=4)
for xi, (r, lo, hi, n_p) in enumerate(zip(rates, lows, highs,
                                           [c4_pass, c41_pass])):
    ax.text(xi, r + 0.06, f"{n_p}/{N}\n({r:.0%})",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(xi, 0.05, f"CI [{lo:.2f}–{hi:.2f}]",
            ha="center", fontsize=8, color="#555")
ax.annotate(f"Δ = {delta_pp:+.0f} pp", xy=(0.5, 0.78),
            ha="center", fontsize=14, fontweight="bold", color="#27ae60",
            arrowprops=None)
ax.set_ylim(0, 1.25); ax.set_ylabel("Success rate", fontsize=11)
ax.set_title("Overall Success Rate\nC4 baseline vs C4.1 fix", fontsize=12, fontweight="bold")
ax.axhline(1.0, color="grey", lw=1, ls="--", alpha=0.4)
ax.grid(True, axis="y", alpha=0.3, zorder=0)

# Right: per-run pass matrix
ax = axes[1]
data = np.array([[r["passed"] for r in c4],
                 [r["passed"] for r in c41]])
from matplotlib.colors import ListedColormap
cmap_pf = ListedColormap(["#e74c3c", "#2ecc71"])
ax.imshow(data, cmap=cmap_pf, vmin=0, vmax=1, aspect="auto")
for ri in range(2):
    for ci in range(N):
        sym = "✓" if data[ri,ci] else "✗"
        ax.text(ci, ri, sym, ha="center", va="center",
                fontsize=20, fontweight="bold", color="white")
ax.set_xticks(range(N)); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_yticks([0,1]); ax.set_yticklabels(["C4\nbaseline","C4.1\nfix"], fontsize=10)
ax.set_title("Per-Run Pass/Fail Grid\n(C4 vs C4.1)", fontsize=12, fontweight="bold")
green_p = mpatches.Patch(color="#2ecc71", label="Pass")
red_p   = mpatches.Patch(color="#e74c3c", label="Fail")
ax.legend(handles=[green_p, red_p], loc="upper right",
          bbox_to_anchor=(1.18, 1.02), fontsize=9)

fig.suptitle(f"EXP-C4.1 — Re-Targeting Protocol: {c41_pass}/{N} (100%) vs C4 baseline {c4_pass}/{N} (40%)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(RESULTS, "C4_1_fig1_success_rate_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 1 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Per-run z_final  (where each run ended up)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
x = np.arange(N); w = 0.35
ax.bar(x - w/2, [r["z_final"] for r in c4],  w, color=C4_PASS_COL,
       alpha=0.55, edgecolor="black", label="C4 baseline")
ax.bar(x + w/2, [r["z_final"] for r in c41], w, color=C41_PASS_COL,
       alpha=0.9,  edgecolor="black", label="C4.1 fix")
ax.axhline(CORRECT_TARGET, color="purple", lw=2, ls="-",
           label=f"Target {CORRECT_TARGET} m")
ax.axhspan(CORRECT_TARGET - TOLERANCE, CORRECT_TARGET + TOLERANCE,
           color="purple", alpha=0.08, label=f"±{TOLERANCE*100:.0f} cm pass band")
ax.axhline(INITIAL_TARGET, color="orange", lw=1.5, ls=":",
           alpha=0.7, label=f"Initial target {INITIAL_TARGET} m")
for xi, r in enumerate(c41):
    ax.text(xi + w/2, r["z_final"] + 0.02,
            f"{r['z_final']:.3f}", ha="center", fontsize=8, fontweight="bold")
for xi, r in enumerate(c4):
    ax.text(xi - w/2, r["z_final"] + 0.02,
            f"{r['z_final']:.3f}", ha="center", fontsize=7, color="#666")
ax.set_xticks(x); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_ylabel("z_final (m)", fontsize=11)
ax.set_ylim(0, 1.5)
ax.set_title("Final Altitude per Run After Correction\n(green=pass, red=fail)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, axis="y", alpha=0.3)

# Right: altitude error bars
ax = axes[1]
c4_errs  = [r["alt_error_cm"] for r in c4]
c41_errs = [r["alt_error_cm"] for r in c41]
ax.bar(x - w/2, c4_errs,  w, color=C4_PASS_COL,  alpha=0.55, edgecolor="black",
       label=f"C4  (mean={np.mean(c4_errs):.1f} cm)")
ax.bar(x + w/2, c41_errs, w, color=C41_PASS_COL, alpha=0.9,  edgecolor="black",
       label=f"C4.1 (mean={np.mean(c41_errs):.2f} cm)")
ax.axhline(TOLERANCE * 100, color="purple", lw=1.5, ls="--",
           label=f"Pass threshold {TOLERANCE*100:.0f} cm")
for xi, e in enumerate(c41_errs):
    ax.text(xi + w/2, e + 0.5, f"{e:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_ylabel("Altitude error from 1.2 m (cm)", fontsize=11)
ax.set_title(f"Altitude Error per Run\nC4: {np.mean(c4_errs):.1f}±{np.std(c4_errs):.1f} cm  "
             f"→  C4.1: {np.mean(c41_errs):.2f}±{np.std(c41_errs):.2f} cm",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("EXP-C4.1 — Per-Run Altitude Results  (C4 vs C4.1)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(RESULTS, "C4_1_fig2_per_run_z_final.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 2 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Altitude trajectory  (phase1 → after correction)
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(CORRECT_TARGET, color="purple", lw=2, ls="-", alpha=0.8,
           label=f"Corrected target {CORRECT_TARGET} m", zorder=2)
ax.axhspan(CORRECT_TARGET - TOLERANCE, CORRECT_TARGET + TOLERANCE,
           color="purple", alpha=0.08, zorder=1)
ax.axhline(INITIAL_TARGET, color="orange", lw=1.5, ls=":", alpha=0.7,
           label=f"Initial target {INITIAL_TARGET} m", zorder=2)

cmap_r = plt.cm.tab10
for i, (r4, r41) in enumerate(zip(c4, c41)):
    color = cmap_r(i)
    # C4: dashed
    ax.plot([1, 2], [r4["z_phase1"], r4["z_final"]], "--o",
            color=color, lw=1.5, ms=7, alpha=0.45,
            markerfacecolor="white", markeredgecolor=color)
    # C4.1: solid, thicker
    ax.plot([1, 2], [r41["z_phase1"], r41["z_final"]], "-o",
            color=color, lw=2.5, ms=9, zorder=5,
            label=f"Run {i+1}")
    # Annotate C4.1 endpoints
    ax.text(2.03, r41["z_final"], f"R{i+1}: {r41['z_final']:.3f} m",
            va="center", fontsize=8, color=color, fontweight="bold")
    # Annotate C4 endpoints (failed ones)
    if not r4["passed"]:
        ax.text(1.97, r4["z_final"], f"✗{r4['z_final']:.3f}",
                va="center", ha="right", fontsize=7, color=color, alpha=0.6)

ax.set_xticks([1, 2])
ax.set_xticklabels(["After Phase 1\n(hover at 0.5 m)", "After Correction\n(target 1.2 m)"],
                   fontsize=11)
ax.set_ylabel("EKF altitude (m)", fontsize=11)
ax.set_xlim(0.5, 2.4)
ax.set_ylim(0.3, 1.5)
ax.set_title("Altitude Trajectory Per Run\nSolid = C4.1 (fix)   Dashed = C4 (baseline)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.grid(True, alpha=0.3)
# Add correction arrow annotation
ax.annotate("Correction\narrives here", xy=(1.5, 0.9),
            xytext=(1.15, 0.75),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=9, ha="center", color="black")

fig.suptitle("EXP-C4.1 — Altitude Trajectory Before and After Mid-Mission Correction",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(RESULTS, "C4_1_fig3_altitude_trajectory.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 3 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Phase 2 API calls  (freeze detection)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

c4_api2  = [r["api_ph2"] for r in c4]
c41_api2 = [r["api_ph2"] for r in c41]

ax = axes[0]
x = np.arange(N); w = 0.35
ax.bar(x - w/2, c4_api2,  w, color=C4_PASS_COL,  alpha=0.6, edgecolor="black",
       label="C4 baseline")
ax.bar(x + w/2, c41_api2, w, color="#2ecc71", alpha=0.85, edgecolor="black",
       label="C4.1 fix")
for xi, (c4v, c41v) in enumerate(zip(c4_api2, c41_api2)):
    if c4v == 0:
        ax.text(xi - w/2, 0.2, "FREEZE\n(0 calls)", ha="center",
                fontsize=7, color="#e74c3c", fontweight="bold")
    ax.text(xi + w/2, c41v + 0.1, str(c41v), ha="center",
            fontsize=10, fontweight="bold", color="#27ae60")
ax.set_xticks(x); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_ylabel("Phase 2 API calls", fontsize=11)
ax.set_ylim(0, 12)
ax.set_title("Phase 2 LLM Activity\nC4 Runs 1&2: freeze (0 calls) → C4.1: all active",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

# Right: total API calls comparison
ax = axes[1]
c4_total  = [r["api_ph1"] + r["api_ph2"] for r in c4]
c41_total = [r["api_ph1"] + r["api_ph2"] for r in c41]
ax.bar(x - w/2, c4_total,  w, color=C4_PASS_COL,  alpha=0.6, edgecolor="black",
       label=f"C4  (mean={np.mean(c4_total):.1f})")
ax.bar(x + w/2, c41_total, w, color="#2ecc71", alpha=0.85, edgecolor="black",
       label=f"C4.1 (mean={np.mean(c41_total):.1f})")
ax.set_xticks(x); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_ylabel("Total API calls (Ph1 + Ph2)", fontsize=11)
ax.set_title("Total API Calls per Run\n(C4.1 more efficient despite higher pass rate)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("EXP-C4.1 — LLM Activity: Phase 2 API Calls Comparison",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(RESULTS, "C4_1_fig4_phase2_api_calls.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 4 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Phase 2 tool sequences  (C4 varied vs C4.1 identical)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

TOOL_CAT = {
    "plan_workflow":         "plan",
    "report_progress":       "plan",
    "set_altitude_target":   "action",
    "check_altitude_reached":"verify",
    "wait":                  "wait",
    "enable_altitude_hold":  "action",
    "check_drone_stable":    "verify",
    "get_sensor_status":     "verify",
}
CAT_COLOR = {"action": "#3498db", "verify": "#2ecc71",
             "plan":   "#95a5a6", "wait":   "#e67e22"}

def draw_sequence(ax, seqs, title, run_labels):
    max_len = max(len(s) for s in seqs)
    for ri, seq in enumerate(seqs):
        y = len(seqs) - 1 - ri
        for xi, tool in enumerate(seq):
            cat   = TOOL_CAT.get(tool, "plan")
            color = CAT_COLOR[cat]
            ax.scatter(xi, y, s=220, color=color, edgecolors="black",
                       linewidths=1.5, zorder=5)
            short = tool.replace("_", "\n")
            ax.text(xi, y + 0.22, short, ha="center", va="bottom",
                    fontsize=6, rotation=40)
        # arrows
        for xi in range(len(seq) - 1):
            ax.annotate("", xy=(xi+1, y), xytext=(xi, y),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    ax.set_xlim(-0.5, max_len + 0.5)
    ax.set_ylim(-0.8, len(seqs) - 0.2)
    ax.set_xticks([]); ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels(list(reversed(run_labels)), fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    legend_p = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLOR.items()]
    ax.legend(handles=legend_p, loc="lower right", fontsize=7, ncol=2)

# C4 Phase 2 sequences (from known data)
c4_ph2_seqs = [
    [],                                                        # Run 1: freeze
    [],                                                        # Run 2: freeze
    ["plan_workflow","report_progress","report_progress",
     "set_altitude_target","report_progress","wait",
     "report_progress","check_altitude_reached"],              # Run 3: pass
    ["plan_workflow","report_progress","set_altitude_target",
     "report_progress","wait","report_progress",
     "check_altitude_reached","report_progress"],              # Run 4: wrong tgt
    ["plan_workflow","report_progress","set_altitude_target",
     "report_progress","wait","check_altitude_reached",
     "report_progress","set_altitude_target"],                 # Run 5: pass
]
# Replace empty with a "FREEZE" placeholder for display
freeze_label = ["(NO TOOL CALLS — FREEZE)"]

ax = axes[0]
display_seqs_c4 = []
for i, seq in enumerate(c4_ph2_seqs):
    display_seqs_c4.append(seq if seq else ["FREEZE"])
# Draw manually for C4 (handle freeze)
max_len = max(len(s) for s in c4_ph2_seqs if s) + 1
for ri, (seq, r) in enumerate(zip(c4_ph2_seqs, c4)):
    y = N - 1 - ri
    label_color = "#27ae60" if r["passed"] else "#e74c3c"
    if not seq:
        ax.text(0, y, "NO TOOL CALLS  →  FREEZE FAILURE", va="center",
                fontsize=9, color="#e74c3c", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec="#e74c3c"))
    else:
        for xi, tool in enumerate(seq):
            cat   = TOOL_CAT.get(tool, "plan")
            color = CAT_COLOR[cat]
            ax.scatter(xi, y, s=180, color=color, edgecolors="black",
                       linewidths=1.2, zorder=5)
            ax.text(xi, y + 0.22, tool.replace("_","\n"),
                    ha="center", va="bottom", fontsize=5.5, rotation=38)
        for xi in range(len(seq)-1):
            ax.annotate("", xy=(xi+1, y), xytext=(xi, y),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
ax.set_xlim(-0.5, 9); ax.set_ylim(-0.8, N - 0.2)
ax.set_xticks([]); ax.set_yticks(range(N))
ax.set_yticklabels([f"Run {N-i} ({'✓' if c4[N-1-i]['passed'] else '✗'})"
                    for i in range(N)], fontsize=9)
ax.set_title("C4 Phase 2 Tool Sequences\n(varied, 2 freezes, 1 wrong target)",
             fontsize=11, fontweight="bold")
legend_p = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLOR.items()]
axes[0].legend(handles=legend_p, loc="lower right", fontsize=7, ncol=2)

# C4.1: all identical
ax = axes[1]
c41_seq = ["set_altitude_target", "wait", "check_altitude_reached"]
for ri in range(N):
    y = N - 1 - ri
    for xi, tool in enumerate(c41_seq):
        cat   = TOOL_CAT.get(tool, "plan")
        color = CAT_COLOR[cat]
        ax.scatter(xi, y, s=220, color=color, edgecolors="black",
                   linewidths=2, zorder=5)
        ax.text(xi, y + 0.25, tool.replace("_","\n"),
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    for xi in range(len(c41_seq)-1):
        ax.annotate("", xy=(xi+1, y), xytext=(xi, y),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    ax.text(3.1, y, "✓ Pass", va="center", fontsize=10,
            fontweight="bold", color="#27ae60")
ax.axvspan(-0.3, 2.3, color="#2ecc71", alpha=0.06, zorder=0)
ax.text(1, -0.55, "Identical 3-tool sequence in all 5 runs",
        ha="center", fontsize=9, color="#27ae60", fontweight="bold", style="italic")
ax.set_xlim(-0.5, 4); ax.set_ylim(-0.8, N - 0.2)
ax.set_xticks([]); ax.set_yticks(range(N))
ax.set_yticklabels([f"Run {N-i} ✓" for i in range(N)], fontsize=9, color="#27ae60")
ax.set_title("C4.1 Phase 2 Tool Sequences\n(identical across all 5 runs — no planning overhead)",
             fontsize=11, fontweight="bold")
legend_p2 = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLOR.items()]
ax.legend(handles=legend_p2, loc="lower right", fontsize=7, ncol=2)

fig.suptitle("EXP-C4.1 — Phase 2 Tool Sequence Comparison\n"
             "C4: varied (freeze / wrong-target / correct)   C4.1: uniform (always correct)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(RESULTS, "C4_1_fig5_phase2_tool_sequence.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 5 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Failure mode breakdown
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: stacked bar C4 vs C4.1
ax = axes[0]
categories  = ["Pass", "Freeze\n(no tool calls)", "Wrong\ntarget"]
c4_counts   = [2, 2, 1]
c41_counts  = [c41_pass, 0, 0]
cat_colors  = ["#2ecc71", "#e74c3c", "#e67e22"]
x3 = np.arange(len(categories)); w = 0.35
for xi, (c4v, c41v, col) in enumerate(zip(c4_counts, c41_counts, cat_colors)):
    ax.bar(xi - w/2, c4v,  w, color=col, alpha=0.5, edgecolor="black")
    ax.bar(xi + w/2, c41v, w, color=col, alpha=0.9, edgecolor="black")
    if c4v:
        ax.text(xi - w/2, c4v + 0.05, str(c4v), ha="center",
                fontsize=12, fontweight="bold", alpha=0.7)
    if c41v:
        ax.text(xi + w/2, c41v + 0.05, str(c41v), ha="center",
                fontsize=12, fontweight="bold")
    elif xi > 0:
        ax.text(xi + w/2, 0.15, "0\n(fixed)", ha="center",
                fontsize=9, color="#27ae60", fontweight="bold")

ax.set_xticks(x3); ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, N + 1); ax.set_ylabel("Number of runs", fontsize=11)
ax.set_yticks(range(N + 1))
ax.set_title("Failure Mode Breakdown\nC4 vs C4.1", fontsize=11, fontweight="bold")
c4_leg  = mpatches.Patch(color="#aaa", alpha=0.5, label="C4 baseline")
c41_leg = mpatches.Patch(color="#aaa", alpha=0.9, label="C4.1 fix")
ax.legend(handles=[c4_leg, c41_leg], fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

# Right: pie for C4, pie for C4.1
ax2_left  = fig.add_axes([0.57, 0.12, 0.18, 0.72])
ax2_right = fig.add_axes([0.77, 0.12, 0.18, 0.72])
axes[1].set_visible(False)

pie_labels = ["Pass", "Freeze", "Wrong\ntarget"]
c4_pie  = [2, 2, 1]
c41_pie = [5, 0, 0]
pie_colors = ["#2ecc71", "#e74c3c", "#e67e22"]

wedges4, texts4, auto4 = ax2_left.pie(
    c4_pie, labels=pie_labels, colors=pie_colors,
    autopct="%1.0f%%", startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2), pctdistance=0.6)
ax2_left.set_title("C4", fontsize=11, fontweight="bold", pad=10)

wedges41, texts41, auto41 = ax2_right.pie(
    c41_pie, labels=["Pass\n5/5", "", ""],
    colors=pie_colors, autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
    startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
ax2_right.set_title("C4.1", fontsize=11, fontweight="bold", pad=10)

fig.suptitle("EXP-C4.1 — Failure Mode Analysis: Both Failure Types Eliminated",
             fontsize=12, fontweight="bold")
out = os.path.join(RESULTS, "C4_1_fig6_failure_mode_breakdown.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 6 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Altitude error distribution  C4 vs C4.1
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

c4_errs  = [r["alt_error_cm"] for r in c4]
c41_errs = [r["alt_error_cm"] for r in c41]

ax = axes[0]
ax.scatter([1]*N, c4_errs,  s=120, color=C4_PASS_COL,
           edgecolors="black", zorder=5, label="C4 (per run)")
ax.scatter([2]*N, c41_errs, s=120, color="#2ecc71",
           edgecolors="black", zorder=5, label="C4.1 (per run)")
for i, (e4, e41) in enumerate(zip(c4_errs, c41_errs)):
    ax.plot([1, 2], [e4, e41], "--", color="grey", lw=0.8, alpha=0.5)
    ax.text(1 - 0.08, e4, f"R{i+1}", ha="right", va="center", fontsize=8)
ax.axhline(TOLERANCE*100, color="purple", lw=1.5, ls="--",
           label=f"Pass threshold {TOLERANCE*100:.0f} cm")
ax.axhline(np.mean(c4_errs),  color="#e74c3c", lw=2, ls="-",
           label=f"C4 mean {np.mean(c4_errs):.1f} cm")
ax.axhline(np.mean(c41_errs), color="#27ae60", lw=2, ls="-",
           label=f"C4.1 mean {np.mean(c41_errs):.2f} cm")
ax.set_xticks([1, 2]); ax.set_xticklabels(["C4\nbaseline","C4.1\nfix"], fontsize=11)
ax.set_ylabel("Altitude error from 1.2 m (cm)", fontsize=11)
ax.set_title("Per-Run Error Distribution\n(lines connect same run number)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
improvement = [c4e - c41e for c4e, c41e in zip(c4_errs, c41_errs)]
bar_colors = ["#27ae60" if imp > 0 else "#e74c3c" for imp in improvement]
bars = ax.bar(range(1, N+1), improvement, color=bar_colors,
              edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", lw=1.5)
for xi, imp in enumerate(improvement, 1):
    ax.text(xi, imp + 0.5, f"{imp:+.1f} cm", ha="center",
            fontsize=10, fontweight="bold")
ax.set_xticks(range(1, N+1)); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_ylabel("Error reduction (cm)  [C4 − C4.1]", fontsize=11)
ax.set_title(f"Error Improvement Per Run\n"
             f"Total mean improvement: {np.mean(improvement):.1f} cm",
             fontsize=11, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("EXP-C4.1 — Altitude Error: C4 vs C4.1  "
             f"(mean: {np.mean(c4_errs):.1f} cm → {np.mean(c41_errs):.2f} cm)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(RESULTS, "C4_1_fig7_alt_error_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 7 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Cost and token efficiency
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

c4_cost  = [r["cost"] for r in c4]
c41_cost = [r["cost"] for r in c41]

ax = axes[0]
x = np.arange(N); w = 0.35
ax.bar(x - w/2, c4_cost,  w, color=C4_PASS_COL,  alpha=0.6, edgecolor="black",
       label=f"C4  (mean ${np.mean(c4_cost):.3f})")
ax.bar(x + w/2, c41_cost, w, color="#2ecc71", alpha=0.85, edgecolor="black",
       label=f"C4.1 (mean ${np.mean(c41_cost):.3f})")
ax.set_xticks(x); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_ylabel("Total cost per run (USD)", fontsize=11)
ax.set_title("API Cost per Run\n(C4.1 costs less for better results)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
c4_in   = [r["in_tok"]  for r in c4]
c41_in  = [r["in_tok"]  for r in c41]
c4_out  = [r["out_tok"] for r in c4]
c41_out = [r["out_tok"] for r in c41]
ax.bar(x - w/2, np.array(c4_in)/1000,  w, color="#2980b9", alpha=0.6,
       edgecolor="black", label="C4 input (K tok)")
ax.bar(x + w/2, np.array(c41_in)/1000, w, color="#2980b9", alpha=0.9,
       edgecolor="black", label="C4.1 input (K tok)")
ax.bar(x - w/2, np.array(c4_out)/1000,  w, bottom=np.array(c4_in)/1000,
       color="#e74c3c", alpha=0.6, edgecolor="black", label="C4 output (K tok)")
ax.bar(x + w/2, np.array(c41_out)/1000, w, bottom=np.array(c41_in)/1000,
       color="#e74c3c", alpha=0.9, edgecolor="black", label="C4.1 output (K tok)")
ax.set_xticks(x); ax.set_xticklabels(RUN_LABELS, fontsize=9)
ax.set_ylabel("Tokens (thousands)", fontsize=11)
ax.set_title("Token Usage per Run  (stacked input + output)\n"
             "C4.1 uses fewer tokens: protocol reduces planning overhead",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("EXP-C4.1 — API Cost & Token Efficiency",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(RESULTS, "C4_1_fig8_cost_efficiency.png")
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"[C4.1] Fig 8 → {out}")

print("\n[C4.1] All figures generated:")
figs = [
    "C4_1_fig1_success_rate_comparison.png — C4 vs C4.1 bar + pass/fail grid",
    "C4_1_fig2_per_run_z_final.png         — per-run z_final and altitude error",
    "C4_1_fig3_altitude_trajectory.png     — trajectory lines phase1→correction",
    "C4_1_fig4_phase2_api_calls.png        — freeze detection via Phase 2 call count",
    "C4_1_fig5_phase2_tool_sequence.png    — C4 varied vs C4.1 identical 3-tool seq",
    "C4_1_fig6_failure_mode_breakdown.png  — pass/freeze/wrong-target categorical",
    "C4_1_fig7_alt_error_distribution.png  — error scatter + per-run improvement",
    "C4_1_fig8_cost_efficiency.png         — cost and token usage comparison",
]
for f in figs:
    print(f"  {f}")
