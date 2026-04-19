"""
plot_C6_detailed.py  —  10 diagnostic figures for EXP-C6 (Mission Planning)
=============================================================================
Reads:  results/C6_runs.csv, results/C6_summary.csv
Writes: results/C6_fig1_passfail_overview.png
        results/C6_fig2_xy_coverage_footprints.png
        results/C6_fig3_squareness_analysis.png
        results/C6_fig4_path_length_analysis.png
        results/C6_fig5_plan_steps_analysis.png
        results/C6_fig6_xy_range_scatter.png
        results/C6_fig7_direction_changes.png
        results/C6_fig8_token_cost_analysis.png
        results/C6_fig9_drift_efficiency.png
        results/C6_fig10_conversation_flow.png
"""

import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle

RESULTS     = os.path.join(os.path.dirname(__file__), "results")
RUNS_CSV    = os.path.join(RESULTS, "C6_runs.csv")
SUMMARY_CSV = os.path.join(RESULTS, "C6_summary.csv")
COMMAND     = '"do a square pattern at 1 metre height"'
IDEAL_SIDE  = 1.0

# ── Load data ─────────────────────────────────────────────────────────────────
rows = []
with open(RUNS_CSV) as f:
    for r in csv.DictReader(f):
        rows.append({
            "run":         int(r["run"]),
            "plan_steps":  int(r["n_plan_steps"]),
            "x_range":     float(r["x_range_m"]),
            "y_range":     float(r["y_range_m"]),
            "squareness":  float(r["squareness_ratio"]),
            "path_m":      float(r["total_path_m"]),
            "dir_changes": int(r["dir_changes"]),
            "passed":      int(r["passed"]),
            "api_calls":   int(r["api_calls"]),
            "in_tok":      int(r["input_tokens"]),
            "out_tok":     int(r["output_tokens"]),
            "cost":        float(r["cost_usd"]),
        })

summary = {}
with open(SUMMARY_CSV) as f:
    for r in csv.DictReader(f):
        try:    summary[r["metric"]] = float(r["value"])
        except: summary[r["metric"]] = r["value"]

N          = len(rows)
runs       = [r["run"]        for r in rows]
plan_steps = [r["plan_steps"] for r in rows]
x_ranges   = [r["x_range"]   for r in rows]
y_ranges   = [r["y_range"]   for r in rows]
squareness = [r["squareness"] for r in rows]
path_m     = [r["path_m"]    for r in rows]
dir_chgs   = [r["dir_changes"]for r in rows]
passed     = [r["passed"]    for r in rows]
costs      = [r["cost"]      for r in rows]
in_toks    = [r["in_tok"]    for r in rows]
out_toks   = [r["out_tok"]   for r in rows]

PASS_COL   = "#2ecc71"
FAIL_COL   = "#e74c3c"
RUN_COLS   = ["#3498db","#e67e22","#9b59b6","#1abc9c","#e74c3c"]
sq_mean    = float(summary.get("squareness_mean",  np.mean(squareness)))
sq_ci_lo   = float(summary.get("squareness_ci_lo", 0.0))
sq_ci_hi   = float(summary.get("squareness_ci_hi", 1.0))

def add_banner(fig):
    fig.text(0.5, 0.005,
             f'EXP-C6: Mission Planning  |  Command: {COMMAND}  |  N=5  |  5/5 passed',
             ha='center', va='bottom', fontsize=8, style='italic', color='#555555',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f4f8', edgecolor='#cccccc'))

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    m = z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return max(0,c-m), min(1,c+m)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Pass/fail overview
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 6))
fig.suptitle("EXP-C6: Mission Planning — Pass/Fail Overview (N=5)", fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# Left: pass/fail tiles
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, N); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title("Run outcomes", fontsize=11, fontweight='bold')
tile_w, tile_h = 0.85, 0.82
for i, r in enumerate(rows):
    cx = i + 0.5
    col = PASS_COL if r["passed"] else FAIL_COL
    rect = FancyBboxPatch((cx - tile_w/2, 0.05), tile_w, tile_h,
                          boxstyle="round,pad=0.03", facecolor=col, edgecolor='white', lw=2)
    ax.add_patch(rect)
    ax.text(cx, 0.80, f"Run {r['run']}", ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(cx, 0.62, "PASS" if r["passed"] else "FAIL",
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(cx, 0.42, f"sq={r['squareness']:.3f}", ha='center', va='center', fontsize=9, color='white')
    ax.text(cx, 0.25, f"path={r['path_m']:.1f}m",  ha='center', va='center', fontsize=9, color='white')
    ax.text(cx, 0.12, f"{r['plan_steps']} steps",   ha='center', va='center', fontsize=8, color='white')

# Middle: success rate
ax = fig.add_subplot(gs[0, 1])
sr = sum(passed)/N
lo, hi = wilson_ci(sum(passed), N)
ax.bar([0], [sr], color=PASS_COL, alpha=0.85, edgecolor='black', width=0.5)
ax.errorbar([0], [sr], yerr=[[sr-lo],[hi-sr]], fmt='none', color='black',
            capsize=12, capthick=2, elinewidth=2)
ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 1.3)
ax.set_xticks([0]); ax.set_xticklabels(['C6'])
ax.set_ylabel("Success rate", fontsize=11)
ax.set_title(f"Success rate\n5/5 — CI: {lo:.3f}–{hi:.3f}", fontsize=11, fontweight='bold')
ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
ax.text(0, hi+0.05, f"100%\nCI: {lo:.3f}–{hi:.3f}", ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Right: squareness per run (bars only, clean)
ax = fig.add_subplot(gs[0, 2])
x = np.arange(1, N+1)
cols = [RUN_COLS[i] for i in range(N)]
ax.bar(x, squareness, color=cols, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(sq_mean, color='navy', ls='--', lw=2, label=f"Mean={sq_mean:.3f}")
ax.fill_between([0.5,N+0.5], sq_ci_lo, sq_ci_hi, alpha=0.12, color='navy',
                label=f"95% CI [{sq_ci_lo:.3f},{sq_ci_hi:.3f}]")
ax.axhline(1.0, color='gold', ls=':', lw=2, label="Perfect = 1.0")
for xi, sq in zip(x, squareness):
    ax.text(xi, sq+0.025, f"{sq:.3f}", ha='center', fontsize=10, fontweight='bold')
# plan steps as text annotations above bars
for xi, ps in zip(x, plan_steps):
    ax.text(xi, -0.06, f"{ps} steps", ha='center', fontsize=8, color='gray')
ax.set_ylim(-0.12, 1.3); ax.set_xticks(x)
ax.set_xlabel("Run"); ax.set_ylabel("Squareness ratio")
ax.set_title("Squareness per run\n(plan steps shown below axis)", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig1_passfail_overview.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 1 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — XY coverage footprints (5 runs + overlay)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("EXP-C6: EKF XY Coverage Footprints — Measured x_range × y_range vs Ideal 1×1 m",
             fontsize=12, fontweight='bold')

for i, r in enumerate(rows):
    ax = axes[i//3][i%3]
    xr, yr = r["x_range"], r["y_range"]
    ideal = Rectangle((-IDEAL_SIDE/2, -IDEAL_SIDE/2), IDEAL_SIDE, IDEAL_SIDE,
                      fill=False, edgecolor='dimgray', lw=2.5, ls='--', label="Ideal 1×1m")
    act_col = RUN_COLS[i]
    actual = Rectangle((-xr/2, -yr/2), xr, yr,
                       fill=True, facecolor=act_col, edgecolor='black', lw=2, alpha=0.35,
                       label=f"Actual {xr:.2f}×{yr:.2f}m")
    ax.add_patch(ideal); ax.add_patch(actual)
    ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=5)  # origin
    lim = max(IDEAL_SIDE, xr, yr)*0.6 + 0.35
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    verdict = "✓ PASS" if r["passed"] else "✗ FAIL"
    ax.set_title(f"Run {r['run']}  {verdict}\nsq={r['squareness']:.3f}  path={r['path_m']:.2f}m  {r['plan_steps']} steps",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, alpha=0.2); ax.axvline(0, color='k', lw=0.5, alpha=0.2)
    ax.text(0.03, 0.06, f"min/max={min(xr,yr):.3f}/{max(xr,yr):.3f}\n→ sq={r['squareness']:.3f}",
            transform=ax.transAxes, fontsize=8, color='navy',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Bottom-right: all overlaid
ax = axes[1][2]
for i, r in enumerate(rows):
    xr, yr = r["x_range"], r["y_range"]
    rect = Rectangle((-xr/2, -yr/2), xr, yr,
                     fill=False, edgecolor=RUN_COLS[i], lw=2,
                     label=f"R{r['run']} sq={r['squareness']:.2f}")
    ax.add_patch(rect)
ideal = Rectangle((-IDEAL_SIDE/2, -IDEAL_SIDE/2), IDEAL_SIDE, IDEAL_SIDE,
                  fill=False, edgecolor='black', lw=2.5, ls='--', label="Ideal 1×1m")
ax.add_patch(ideal)
ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=5)
mx = max(max(x_ranges), max(y_ranges))/2+0.5
ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx)
ax.set_aspect('equal')
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_title("All runs overlaid (centred at takeoff)", fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5, alpha=0.2); ax.axvline(0, color='k', lw=0.5, alpha=0.2)

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig2_xy_coverage_footprints.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 2 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Squareness deep-dive
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EXP-C6: Squareness Analysis — LLM Waypoint Geometry Quality", fontsize=12, fontweight='bold')

ax = axes[0]
x = np.arange(1, N+1)
ax.bar(x, squareness, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(sq_mean, color='navy', ls='--', lw=2, label=f"Mean={sq_mean:.3f}")
ax.fill_between([0.5,N+0.5], sq_ci_lo, sq_ci_hi, alpha=0.15, color='navy',
                label=f"95% CI [{sq_ci_lo:.3f},{sq_ci_hi:.3f}]")
ax.axhline(1.0, color='gold', ls=':', lw=2, label="Perfect = 1.0")
for xi, sq in zip(x, squareness):
    ax.text(xi, sq+0.02, f"{sq:.3f}", ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.25); ax.set_xticks(x)
ax.set_xlabel("Run"); ax.set_ylabel("Squareness (min/max range)")
ax.set_title("Squareness per run\n(1.0 = perfect square)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
w = 0.35
ax.bar(x-w/2, x_ranges, w, color='steelblue', alpha=0.85, edgecolor='black', label='X range (m)')
ax.bar(x+w/2, y_ranges, w, color='coral',     alpha=0.85, edgecolor='black', label='Y range (m)')
ax.axhline(IDEAL_SIDE, color='gray', ls='--', lw=1.5, label=f"Ideal = {IDEAL_SIDE}m")
for xi, xr, yr in zip(x, x_ranges, y_ranges):
    ax.text(xi-w/2, xr+0.08, f"{xr:.2f}", ha='center', fontsize=8, color='steelblue', fontweight='bold')
    ax.text(xi+w/2, yr+0.08, f"{yr:.2f}", ha='center', fontsize=8, color='coral', fontweight='bold')
ax.set_xticks(x); ax.set_xlabel("Run"); ax.set_ylabel("Range (m)")
ax.set_title("X range vs Y range per run\n(equal → square; unequal → elongated)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

ax = axes[2]
sq_arr = np.array(squareness)
ax.hist(sq_arr, bins=np.linspace(0,1,11), color='steelblue', alpha=0.75, edgecolor='black', rwidth=0.85)
ax.axvline(sq_mean,  color='navy', ls='--', lw=2, label=f"Mean={sq_mean:.3f}")
ax.axvline(sq_ci_lo, color='navy', ls=':',  lw=1.2, label=f"CI lo={sq_ci_lo:.3f}")
ax.axvline(sq_ci_hi, color='navy', ls=':',  lw=1.2, label=f"CI hi={sq_ci_hi:.3f}")
ax.axvline(1.0, color='gold', ls='-', lw=2, alpha=0.8, label="Perfect=1.0")
ax.set_xlabel("Squareness ratio"); ax.set_ylabel("Count")
ax.set_title("Squareness distribution\n(large spread = strategy variance + sensor drift)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
ax.text(0.05, 0.78, "Variance has 2 sources:\n① LLM waypoint spacing\n② Optical flow dead-reckoning drift",
        transform=ax.transAxes, fontsize=8.5, va='top',
        bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.85))

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig3_squareness_analysis.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 3 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Path length analysis
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EXP-C6: Total Path Length Analysis — LLM Strategy vs Optical Flow Drift", fontsize=12, fontweight='bold')

ax = axes[0]
ax.bar(np.arange(1,N+1), path_m, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ideal_path = 4 * IDEAL_SIDE
ax.axhline(ideal_path, color='gray', ls='--', lw=2, label=f"Ideal = {ideal_path}m (4×1m sides)")
ax.axhline(np.mean(path_m), color='navy', ls='-', lw=1.5,
           label=f"Mean={np.mean(path_m):.2f}±{np.std(path_m):.2f}m")
for xi, p in zip(np.arange(1,N+1), path_m):
    ax.text(xi, p+0.2, f"{p:.2f}m", ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(np.arange(1,N+1)); ax.set_xlabel("Run"); ax.set_ylabel("Total EKF path (m)")
ax.set_title("EKF total path per run\n(ideal 4×1m square = 4.0m)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
for i, (p, sq) in enumerate(zip(path_m, squareness)):
    ax.scatter(p, sq, s=150, color=RUN_COLS[i], edgecolors='black', zorder=5)
    ax.annotate(f"R{i+1}", (p, sq), textcoords='offset points', xytext=(7,4), fontsize=9)
ax.axvline(ideal_path, color='gray', ls='--', lw=1.5, label=f"Ideal={ideal_path}m", alpha=0.6)
ax.axhline(1.0, color='gold', ls=':', lw=1.5, label="Perfect sq=1.0")
z = np.polyfit(path_m, squareness, 1)
xfit = np.linspace(min(path_m)*0.9, max(path_m)*1.05, 50)
ax.plot(xfit, np.polyval(z, xfit), 'k--', lw=1, alpha=0.5, label="Trend")
ax.set_xlabel("Total path (m)"); ax.set_ylabel("Squareness ratio")
ax.set_title("Path length vs squareness\n(longer path ≠ better square)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
efficiency = [p/ideal_path for p in path_m]
ax.bar(np.arange(1,N+1), efficiency, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(1.0, color='gray', ls='--', lw=2, label="Ideal = 1.0×")
for xi, e, p in zip(np.arange(1,N+1), efficiency, path_m):
    ax.text(xi, e+0.04, f"{e:.2f}×\n({p:.1f}m)", ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(np.arange(1,N+1)); ax.set_xlabel("Run")
ax.set_ylabel("Path efficiency (actual / ideal 4m)")
ax.set_title("Path efficiency per run\n(1.0× = flew exact 4m; higher = extra distance)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
ax.text(0.02, 0.88, "Run 5: 10.8m = 2.7× ideal\nLLM generated ~2.7m legs\n(no coordinate constraint)",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.85))

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig4_path_length_analysis.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 4 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Plan steps analysis
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EXP-C6: Plan Complexity — LLM step count does not predict trajectory quality", fontsize=12, fontweight='bold')

strategy = {1:"Compact\n(direct)", 2:"Compact\n(direct)",
            3:"Verbose\n(+ verif.)", 4:"Compact\n(direct)", 5:"Medium\n(+ some verif.)"}
ax = axes[0]
bars = ax.bar(np.arange(1,N+1), plan_steps, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(np.mean(plan_steps), color='navy', ls='--', lw=1.5,
           label=f"Mean={np.mean(plan_steps):.1f}±{np.std(plan_steps):.1f}")
for xi, ps, st in zip(np.arange(1,N+1), plan_steps, strategy.values()):
    ax.text(xi, ps+0.4, str(ps), ha='center', fontsize=12, fontweight='bold')
    ax.text(xi, ps/2,   st,     ha='center', va='center', fontsize=8, color='white', fontweight='bold')
ax.set_xticks(np.arange(1,N+1)); ax.set_xlabel("Run"); ax.set_ylabel("plan_workflow steps")
ax.set_title("LLM plan_workflow step count per run\n(Run 3 most verbose: 34 steps)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
for i, (ps, sq) in enumerate(zip(plan_steps, squareness)):
    ax.scatter(ps, sq, s=150, color=RUN_COLS[i], edgecolors='black', zorder=5)
    ax.annotate(f"R{i+1}", (ps, sq), textcoords='offset points', xytext=(6,4), fontsize=9)
ax.axhline(1.0, color='gold', ls=':', lw=2, label="Perfect sq=1.0")
ax.set_xlabel("Plan steps"); ax.set_ylabel("Squareness ratio")
ax.set_title("Plan steps vs squareness\n(more steps ≠ better geometry)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.text(0.05, 0.12,
        "Run 3 (34 steps) → sq=0.300\nRun 2 (15 steps) → sq=0.647\nPlan verbosity ≠ trajectory quality",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.85))

ax = axes[2]
for i, (ps, p) in enumerate(zip(plan_steps, path_m)):
    ax.scatter(ps, p, s=150, color=RUN_COLS[i], edgecolors='black', zorder=5)
    ax.annotate(f"R{i+1}\n{p:.1f}m", (ps, p), textcoords='offset points', xytext=(6,4), fontsize=8)
ax.axhline(4.0, color='gray', ls='--', lw=1.5, label="Ideal = 4m", alpha=0.6)
ax.set_xlabel("Plan steps"); ax.set_ylabel("Total path (m)")
ax.set_title("Plan steps vs path length\n(compact plans ≠ shorter flights)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig5_plan_steps_analysis.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 5 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — XY range scatter
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("EXP-C6: EKF XY Range — Aspect Ratio Fingerprint per Run", fontsize=12, fontweight='bold')

ax = axes[0]
max_r = max(max(x_ranges), max(y_ranges)) * 1.1
for i, (xr, yr) in enumerate(zip(x_ranges, y_ranges)):
    ax.scatter(xr, yr, s=200, color=RUN_COLS[i], edgecolors='black', zorder=5)
    ax.annotate(f"Run {i+1}\nsq={squareness[i]:.2f}", (xr, yr),
                textcoords='offset points', xytext=(8, 6), fontsize=8.5)
diag = np.linspace(0, max_r, 50)
ax.plot(diag, diag, 'k--', lw=1.5, alpha=0.4, label="X = Y (perfect square)")
ax.axvline(IDEAL_SIDE, color='silver', ls=':', lw=1, alpha=0.7)
ax.axhline(IDEAL_SIDE, color='silver', ls=':', lw=1, alpha=0.7)
ax.set_xlim(0, max_r); ax.set_ylim(0, max_r); ax.set_aspect('equal')
ax.set_xlabel("X range (m)"); ax.set_ylabel("Y range (m)")
ax.set_title("X range vs Y range\n(on X=Y line → perfect square aspect ratio)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.text(max_r*0.55, max_r*0.08, "X dominates\n(wider than tall)", fontsize=8.5, color='gray')
ax.text(max_r*0.05, max_r*0.60, "Y dominates\n(taller than wide)", fontsize=8.5, color='gray')

ax = axes[1]
xy_ratio = [max(xr,yr)/min(xr,yr) if min(xr,yr)>0.01 else 25 for xr,yr in zip(x_ranges,y_ranges)]
xy_clipped = [min(r,20) for r in xy_ratio]
ax.bar(np.arange(1,N+1), xy_clipped, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(1.0, color='gold', ls='--', lw=2, label="Ideal ratio = 1.0")
for xi, r_c, r_r in zip(np.arange(1,N+1), xy_clipped, xy_ratio):
    lbl = f"{r_r:.1f}×" if r_r<20 else f"≈{r_r:.0f}×"
    ax.text(xi, r_c+0.2, lbl, ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(np.arange(1,N+1)); ax.set_xlabel("Run")
ax.set_ylabel("max/min of (X range, Y range)")
ax.set_title("Aspect ratio per run\n(1.0 = square; high = elongated rectangle)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
ax.text(0.02, 0.82,
        f"Run 1: X={x_ranges[0]:.2f}m, Y={y_ranges[0]:.2f}m\n→ ratio ≈ {xy_ratio[0]:.0f}× — near 1D motion\nDrone barely moved in Y axis",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.85))

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig6_xy_range_scatter.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 6 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Direction changes
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EXP-C6: Direction Changes — Trajectory Turning Behaviour", fontsize=12, fontweight='bold')

ideal_dc = 4
ax = axes[0]
ax.bar(np.arange(1,N+1), dir_chgs, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(ideal_dc, color='gold', ls='--', lw=2, label=f"Ideal = {ideal_dc} corners")
for xi, dc in zip(np.arange(1,N+1), dir_chgs):
    ax.text(xi, dc+0.2, str(dc), ha='center', fontsize=13, fontweight='bold')
ax.set_xticks(np.arange(1,N+1)); ax.set_xlabel("Run"); ax.set_ylabel("Direction changes (>45°)")
ax.set_title("Direction changes per run\n(square → 4 turns; deviations = drift/overshot)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
notes = {1:"11 changes → oscillating\n(position hold fighting drift)",
         5:"1 change → nearly straight\n(flew in one direction)"}
for run_i, note in notes.items():
    ax.text(run_i, dir_chgs[run_i-1]+0.8, note, ha='center', fontsize=7.5, color='#333',
            bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.8, boxstyle='round'))

ax = axes[1]
for i, (dc, sq) in enumerate(zip(dir_chgs, squareness)):
    ax.scatter(dc, sq, s=150, color=RUN_COLS[i], edgecolors='black', zorder=5)
    ax.annotate(f"R{i+1}", (dc, sq), textcoords='offset points', xytext=(6,4), fontsize=9)
ax.axvline(ideal_dc, color='gold', ls='--', lw=1.5, label=f"Ideal = {ideal_dc} turns")
ax.axhline(1.0, color='gray', ls=':', lw=1.5, label="Perfect sq=1.0")
ax.set_xlabel("Direction changes"); ax.set_ylabel("Squareness ratio")
ax.set_title("Direction changes vs squareness\n(~4 turns = cleanest square)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
for i, (dc, p) in enumerate(zip(dir_chgs, path_m)):
    ax.scatter(dc, p, s=150, color=RUN_COLS[i], edgecolors='black', zorder=5)
    ax.annotate(f"R{i+1}\n{p:.1f}m", (dc, p), textcoords='offset points', xytext=(6,4), fontsize=8)
ax.axhline(4.0, color='gray', ls='--', lw=1, label="Ideal=4m", alpha=0.6)
ax.set_xlabel("Direction changes"); ax.set_ylabel("Total path (m)")
ax.set_title("Direction changes vs path length\n(oscillation = many turns, short path)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig7_direction_changes.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 7 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Token & cost analysis
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EXP-C6: API Token Usage & Cost — All runs use exactly 30 API calls", fontsize=12, fontweight='bold')

ax = axes[0]
x = np.arange(1, N+1)
ax.bar(x, in_toks,  color='steelblue', alpha=0.85, edgecolor='black', label='Input tokens')
ax.bar(x, out_toks, color='coral',     alpha=0.85, edgecolor='black', label='Output tokens', bottom=in_toks)
for xi, it, ot in zip(x, in_toks, out_toks):
    ax.text(xi, it+ot+500, f"{(it+ot)//1000}K", ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xlabel("Run"); ax.set_ylabel("Tokens")
ax.set_title("Token usage per run\n(dominated by input — long tool result context)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1000:.0f}K"))

ax = axes[1]
ax.bar(x, costs, color='mediumpurple', alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(np.mean(costs), color='navy', ls='--', lw=2, label=f"Mean=${np.mean(costs):.3f}")
for xi, c in zip(x, costs):
    ax.text(xi, c+0.003, f"${c:.3f}", ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xlabel("Run"); ax.set_ylabel("Cost (USD)")
ax.set_title(f"Cost per run  (Total = ${sum(costs):.2f})")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
ax.text(0.05, 0.15,
        f"Total: ${sum(costs):.3f}\nMean: ${np.mean(costs):.4f}/run\nStd:  ${np.std(costs):.4f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor='lavender', edgecolor='purple', alpha=0.85))

ax = axes[2]
api_calls = [r["api_calls"] for r in rows]
ax.bar(x, api_calls, color='teal', alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(30, color='navy', ls='--', lw=2, label="30 calls (constant)")
for xi, ac in zip(x, api_calls):
    ax.text(xi, ac+0.2, str(ac), ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, 40); ax.set_xticks(x); ax.set_xlabel("Run"); ax.set_ylabel("API calls")
ax.set_title("API calls per run\n(identical across all runs — 30 calls each)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
ax.text(0.05, 0.78,
        "Zero variance in API call count.\nCost variance entirely driven by\ntoken count in context window.",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='lavender', edgecolor='purple', alpha=0.85))

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig8_token_cost_analysis.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 8 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Drift & sensor limitation analysis
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EXP-C6: Sensor Drift & Coverage — Optical Flow Position Uncertainty", fontsize=12, fontweight='bold')

ax = axes[0]
areas = [xr*yr for xr,yr in zip(x_ranges, y_ranges)]
ideal_area = IDEAL_SIDE**2
ax.bar(np.arange(1,N+1), areas, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(ideal_area, color='gold', ls='--', lw=2, label=f"Ideal = {ideal_area:.1f} m²")
for xi, a in zip(np.arange(1,N+1), areas):
    ax.text(xi, a+0.05, f"{a:.3f}m²", ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(np.arange(1,N+1)); ax.set_xlabel("Run"); ax.set_ylabel("Coverage area (m²)")
ax.set_title("EKF coverage area per run (X×Y)\n(ideal = 1.0 m² for 1×1m square)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
# Isoperimetric ratio: 4A/L² — square = 0.25, larger = more compact
shape_eff = [4*a/(p**2) if p>0.01 else 0 for a,p in zip(areas, path_m)]
ideal_se = 4*ideal_area/(4.0**2)  # = 0.25
ax.bar(np.arange(1,N+1), shape_eff, color=RUN_COLS, alpha=0.85, edgecolor='black', width=0.6)
ax.axhline(ideal_se, color='gold', ls='--', lw=2, label=f"Ideal (square) = {ideal_se:.3f}")
for xi, se in zip(np.arange(1,N+1), shape_eff):
    ax.text(xi, se+0.003, f"{se:.3f}", ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(np.arange(1,N+1)); ax.set_xlabel("Run")
ax.set_ylabel("Shape efficiency  4A / L²")
ax.set_title("Shape efficiency (isoperimetric ratio)\n(0.25 = perfect square; low = wasted path)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
ax.text(0.05, 0.72, "4A/L² measures how efficiently\nthe path enclosed its area.\nRun 1: large turns, tiny area → near 0",
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='lightyellow', edgecolor='orange', alpha=0.85))

ax = axes[2]
w = 0.35; x = np.arange(1, N+1)
ax.bar(x-w/2, x_ranges, w, color='steelblue', alpha=0.85, edgecolor='black', label='X range')
ax.bar(x+w/2, y_ranges, w, color='coral',     alpha=0.85, edgecolor='black', label='Y range')
ax.axhline(IDEAL_SIDE, color='gold', ls='--', lw=2, label=f"Expected side = {IDEAL_SIDE}m")
for xi, xr, yr in zip(x, x_ranges, y_ranges):
    ax.text(xi-w/2, xr+0.08, f"{xr:.2f}", ha='center', fontsize=8, color='steelblue', fontweight='bold')
    ax.text(xi+w/2, yr+0.08, f"{yr:.2f}", ha='center', fontsize=8, color='coral',     fontweight='bold')
ax.set_xticks(x); ax.set_xlabel("Run"); ax.set_ylabel("Range (m)")
ax.set_title("X/Y measured range vs expected 1m\nDrift causes over/under-travel per axis")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
notes2 = ["R1 Y=0.04m\n(barely moved)", "", "", "R4 X=2.65m\n(2.7× expected)", "R5 X=8.46m\n(8.5× expected)"]
for xi, nt, xr in zip(x, notes2, x_ranges):
    if nt:
        ax.text(xi, xr+0.5, nt, ha='center', fontsize=7, color='navy')

plt.tight_layout(rect=[0,0.04,1,1])
add_banner(fig)
out = os.path.join(RESULTS, "C6_fig9_drift_efficiency.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 9 → {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Conversation flow (clean table layout)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor('#f5f6fa')
fig.suptitle("EXP-C6: Conversation Flow per Run\n"
             f"Command: {COMMAND}  —  All 5/5 PASS",
             fontsize=13, fontweight='bold', y=0.99)

# Use GridSpec: 4 rows × 6 cols (col 0 = row labels, cols 1-5 = runs)
gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.12, wspace=0.08,
                       left=0.07, right=0.99, top=0.92, bottom=0.06)

ROW_LABELS  = ["Human\nCommand", "LLM\nPlan\n(plan_workflow)", "Execution\nMetrics", "Outcome"]
ROW_HEIGHTS = [0.12, 0.38, 0.32, 0.12]
ROW_BG      = ['#dce8f5', '#d4edda', '#fff3cd', '#e8d5f5']

# Column headers (run summaries)
run_headers = [
    f"Run 1\nsq=0.147  path=0.51m\n15 steps",
    f"Run 2\nsq=0.647  path=4.10m\n15 steps",
    f"Run 3\nsq=0.300  path=1.72m\n34 steps",
    f"Run 4\nsq=0.647  path=4.85m\n15 steps",
    f"Run 5\nsq=0.424  path=10.82m\n21 steps",
]

# Row content
row_contents = [
    # Row 0: command (same for all)
    ['"do a square pattern\nat 1 metre height"']*5,

    # Row 1: LLM plan strategy
    [
        "plan_workflow(15 steps)\n\ntakeoff\n→ corner NE\n→ corner SE\n→ corner SW\n→ corner NW\n→ return\n→ land\n\nCompact, direct",
        "plan_workflow(15 steps)\n\ntakeoff\n→ corner NE\n→ corner SE\n→ corner SW\n→ corner NW\n→ return\n→ land\n\nCompact, direct",
        "plan_workflow(34 steps)\n\ntakeoff\n→ stabilise check\n→ corner NE + verify\n→ corner SE + verify\n→ corner SW + verify\n→ corner NW + verify\n→ return\n→ land\n\nVerbose + verifications",
        "plan_workflow(15 steps)\n\ntakeoff\n→ corner NE\n→ corner SE\n→ corner SW\n→ corner NW\n→ return\n→ land\n\nCompact, direct",
        "plan_workflow(21 steps)\n\ntakeoff\n→ corner NE\n→ corner SE\n→ corner SW\n→ corner NW\n→ return\n→ land\n\nMedium + some checks",
    ],

    # Row 2: execution metrics
    [
        f"30 API calls\n{in_toks[0]//1000}K/{out_toks[0]//1000}K tokens\n${costs[0]:.3f}\n\nX={x_ranges[0]:.2f}m  Y={y_ranges[0]:.2f}m\nsq={squareness[0]:.3f}\npath={path_m[0]:.2f}m\n{dir_chgs[0]} dir_changes\n\nNote: barely moved in Y\nDrone oscillated in place\nPosition hold dominated",
        f"30 API calls\n{in_toks[1]//1000}K/{out_toks[1]//1000}K tokens\n${costs[1]:.3f}\n\nX={x_ranges[1]:.2f}m  Y={y_ranges[1]:.2f}m\nsq={squareness[1]:.3f}\npath={path_m[1]:.2f}m\n{dir_chgs[1]} dir_changes\n\nBest near-square result\nX≈1.6m, Y≈1.0m\nClosest to 1:1 aspect",
        f"30 API calls\n{in_toks[2]//1000}K/{out_toks[2]//1000}K tokens\n${costs[2]:.3f}\n\nX={x_ranges[2]:.2f}m  Y={y_ranges[2]:.2f}m\nsq={squareness[2]:.3f}\npath={path_m[2]:.2f}m\n{dir_chgs[2]} dir_changes\n\nMost verbose plan (34 steps)\nbut mid-range squareness\nX>>Y: elongated in X",
        f"30 API calls\n{in_toks[3]//1000}K/{out_toks[3]//1000}K tokens\n${costs[3]:.3f}\n\nX={x_ranges[3]:.2f}m  Y={y_ranges[3]:.2f}m\nsq={squareness[3]:.3f}\npath={path_m[3]:.2f}m\n{dir_chgs[3]} dir_changes\n\nLargest absolute range\nX=2.65m, Y=1.71m\nLLM sent ~2.7m leg targets",
        f"30 API calls\n{in_toks[4]//1000}K/{out_toks[4]//1000}K tokens\n${costs[4]:.3f}\n\nX={x_ranges[4]:.2f}m  Y={y_ranges[4]:.2f}m\nsq={squareness[4]:.3f}\npath={path_m[4]:.2f}m\n{dir_chgs[4]} dir_change\n\nLongest total path (10.8m)\nOnly 1 heading change\nAlmost flew in 1 direction",
    ],

    # Row 3: outcome
    ["✓ PASS\nplan_ok=✓ alt_ok=✓",]*5,
]

# Draw label column (col 0)
for row_i, (label, bg) in enumerate(zip(ROW_LABELS, ROW_BG)):
    ax = fig.add_subplot(gs[row_i, 0])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    rect = FancyBboxPatch((0.02,0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.03", facecolor='#e8e8e8',
                          edgecolor='#888888', lw=1.5)
    ax.add_patch(rect)
    ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color='#222222')

# Draw run header row + data rows
for col_i in range(N):
    # Column header
    ax_hdr = fig.add_subplot(gs[0, col_i+1])
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1); ax_hdr.axis('off')
    bg = PASS_COL if rows[col_i]["passed"] else FAIL_COL
    rect = FancyBboxPatch((0.02,0.02), 0.96, 0.96,
                          boxstyle="round,pad=0.03", facecolor=RUN_COLS[col_i],
                          edgecolor='black', lw=1.5)
    ax_hdr.add_patch(rect)
    ax_hdr.text(0.5, 0.70, f"Run {col_i+1}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    ax_hdr.text(0.5, 0.30, row_contents[0][col_i], ha='center', va='center',
                fontsize=7.5, color='white')

    for row_i in range(1, 4):
        ax = fig.add_subplot(gs[row_i, col_i+1])
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
        cell_bg = ROW_BG[row_i]
        if row_i == 3:
            cell_bg = '#d4edda' if rows[col_i]["passed"] else '#f8d7da'
        rect = FancyBboxPatch((0.02,0.02), 0.96, 0.96,
                              boxstyle="round,pad=0.03", facecolor=cell_bg,
                              edgecolor='#aaaaaa', lw=1)
        ax.add_patch(rect)
        fs = 8 if row_i == 2 else 9.5
        fw = 'bold' if row_i == 3 else 'normal'
        ax.text(0.5, 0.5, row_contents[row_i][col_i], ha='center', va='center',
                fontsize=fs, fontweight=fw, color='#1a1a1a', linespacing=1.4)

add_banner(fig)
out = os.path.join(RESULTS, "C6_fig10_conversation_flow.png")
plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
print(f"[C6] Fig 10 → {out}")

print(f"""
[C6] All 10 figures generated:
  C6_fig1_passfail_overview.png        — pass/fail tiles + success rate CI + squareness bars
  C6_fig2_xy_coverage_footprints.png   — per-run x_range×y_range rectangles vs ideal 1×1m
  C6_fig3_squareness_analysis.png      — squareness bars, X vs Y range, distribution
  C6_fig4_path_length_analysis.png     — path per run, path vs squareness, efficiency ratio
  C6_fig5_plan_steps_analysis.png      — step count, steps vs squareness, steps vs path
  C6_fig6_xy_range_scatter.png         — X vs Y scatter + aspect ratio bar
  C6_fig7_direction_changes.png        — dir_changes per run, vs squareness, vs path
  C6_fig8_token_cost_analysis.png      — stacked tokens, cost, API calls (all 30)
  C6_fig9_drift_efficiency.png         — coverage area, shape efficiency, X/Y vs expected
  C6_fig10_conversation_flow.png       — full per-run table: command→plan→metrics→outcome
""")
