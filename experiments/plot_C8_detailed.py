"""
plot_C8_detailed.py
===================
Comprehensive 12-figure diagnostic plot suite for EXP-C8 v3.

Figures generated:
  C8_fig1_overall_rmse_comparison.png     — 3-mode bar + CI + pass rate overlay
  C8_fig2_per_run_rmse_B_and_C.png        — per-run bars for B and C with A reference
  C8_fig3_per_waypoint_heatmap.png        — RMSE heatmap (mode × waypoint)
  C8_fig4_per_waypoint_grouped_bars.png   — grouped bar per WP, all 3 modes
  C8_fig5_wp_radar.png                    — radar/spider per mode across 4 WPs
  C8_fig6_rmse_distribution.png           — box + strip + violin for B and C
  C8_fig7_rmse_vs_cost_scatter.png        — RMSE vs cost per run (each run a point)
  C8_fig8_token_usage.png                 — input/output tokens per run (B vs C)
  C8_fig9_api_and_cost_breakdown.png      — API calls and cost side-by-side
  C8_fig10_improvement_factor.png         — how many times each LLM mode beats A
  C8_fig11_B_vs_C_head_to_head.png        — run-matched scatter B RMSE vs C RMSE
  C8_fig12_summary_table.png              — publication-ready summary table figure
"""

import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
RES    = os.path.join(BASE, "results")
RUNS   = os.path.join(RES, "C8_runs_guardrail_on.csv")
SUMM   = os.path.join(RES, "C8_summary_guardrail_on.csv")
OUT    = RES   # output directory

# ── Load data ─────────────────────────────────────────────────────────────────
def load_runs():
    rows = []
    with open(RUNS) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("mode",) else v) for k, v in r.items()})
    return rows

def load_summary():
    d = {}
    with open(SUMM) as f:
        for row in csv.DictReader(f):
            d[row["metric"]] = row["value"]
    return d

runs = load_runs()
summ = load_summary()

B_runs = [r for r in runs if r["mode"] == "B"]
C_runs = [r for r in runs if r["mode"] == "C"]

WAYPOINTS    = [0.8, 1.2, 1.5, 1.0]
WP_LABELS    = ["WP1\n0.8 m", "WP2\n1.2 m", "WP3\n1.5 m", "WP4\n1.0 m"]
N_RUNS       = 5
PASS_RMSE_CM = 15.0

# Mode A from summary
A_overall  = float(summ["mode_A_overall_rmse_cm"])
A_wp       = [float(summ[f"mode_A_wp{i+1}_rmse_cm"]) for i in range(4)]

# Aggregates
B_rmse     = [r["rmse_cm"] for r in B_runs]
C_rmse     = [r["rmse_cm"] for r in C_runs]
B_api      = [r["api_calls"] for r in B_runs]
C_api      = [r["api_calls"] for r in C_runs]
B_cost     = [r["cost_usd"] for r in B_runs]
C_cost     = [r["cost_usd"] for r in C_runs]
B_in_tok   = [r["input_tokens"] for r in B_runs]
C_in_tok   = [r["input_tokens"] for r in C_runs]
B_out_tok  = [r["output_tokens"] for r in B_runs]
C_out_tok  = [r["output_tokens"] for r in C_runs]

def wp_col(mode_runs, wp_i):
    return [r[f"wp{wp_i+1}_rmse_cm"] for r in mode_runs]

def bootstrap_ci(vals, n=2000, alpha=0.05):
    arr = np.array(vals); boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n)]
    return np.percentile(boots, 100*alpha/2), np.percentile(boots, 100*(1-alpha/2))

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0., 1.
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    m = z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))/d
    return max(0., c-m), min(1., c+m)

# ── Palette ────────────────────────────────────────────────────────────────────
CA = "#95a5a6"   # Mode A grey
CB = "#e67e22"   # Mode B orange
CC = "#2ecc71"   # Mode C green
RUNS_PAL = ["#3498db","#e67e22","#9b59b6","#1abc9c","#e74c3c"]

DPI = 150

print("[C8 plots] Starting 12-figure suite …")

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Overall RMSE comparison (bar + CI + pass rate panel)
# ═══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios":[2,1]})

mode_labels = ["A\n(scripted)", "B\n(supervisor)", "C\n(full-auto)"]
rmse_means  = [A_overall, np.mean(B_rmse), np.mean(C_rmse)]
rmse_stds   = [0, np.std(B_rmse), np.std(C_rmse)]
B_ci = bootstrap_ci(B_rmse); C_ci = bootstrap_ci(C_rmse)
yerr_lo = [0, np.mean(B_rmse)-B_ci[0], np.mean(C_rmse)-C_ci[0]]
yerr_hi = [0, B_ci[1]-np.mean(B_rmse), C_ci[1]-np.mean(C_rmse)]

bars = ax1.bar(mode_labels, rmse_means, color=[CA,CB,CC], alpha=0.85,
               edgecolor="black", width=0.5, zorder=3)
ax1.errorbar(mode_labels, rmse_means, yerr=[yerr_lo, yerr_hi],
             fmt="none", ecolor="black", capsize=10, lw=2, zorder=4)
for i, (m, lo, hi) in enumerate(zip(rmse_means, yerr_lo, yerr_hi)):
    ax1.text(i, m + hi + 0.04, f"{m:.3f} cm", ha="center", fontsize=10, fontweight="bold")
ax1.axhline(PASS_RMSE_CM, color="red", ls=":", lw=1.5, label=f"Pass ≤{PASS_RMSE_CM:.0f} cm", alpha=0.7)
ax1.set_ylabel("Overall RMSE (cm)", fontsize=11)
ax1.set_title("Overall RMSE — Three-Mode Comparison\n(error bars = 95% bootstrap CI, N=5 for B/C)", fontsize=10)
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis="y", zorder=0)
ax1.set_ylim(0, max(rmse_means)*1.4)

# Pass rates with Wilson CI
pass_rates = [1.0, 1.0, 1.0]
pass_cis   = [(1.0,1.0), wilson_ci(5,5), wilson_ci(5,5)]
err_lo2 = [r-lo for r,(lo,hi) in zip(pass_rates, pass_cis)]
err_hi2 = [hi-r for r,(lo,hi) in zip(pass_rates, pass_cis)]
ax2.bar(mode_labels, pass_rates, color=[CA,CB,CC], alpha=0.85, edgecolor="black", width=0.5, zorder=3)
ax2.errorbar(mode_labels, pass_rates, yerr=[err_lo2, err_hi2],
             fmt="none", ecolor="black", capsize=10, lw=2, zorder=4)
ax2.set_ylim(0, 1.35); ax2.set_ylabel("Pass rate", fontsize=11)
ax2.set_title("Pass rate (RMSE≤15 cm,\nall WP, disarmed)", fontsize=10)
ax2.grid(True, alpha=0.3, axis="y", zorder=0)
for i, (r,(lo,hi)) in enumerate(zip(pass_rates, pass_cis)):
    ax2.text(i, r+0.07, f"{r:.0%}\n[{lo:.2f},{hi:.2f}]", ha="center", fontsize=8)

fig.suptitle("EXP-C8 v3: Three-Mode RMSE and Pass Rate Summary", fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig1_overall_rmse_comparison.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 1: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Per-run RMSE bars for B and C with Mode A reference
# ═══════════════════════════════════════════════════════════════════════════════
fig, (axB, axC) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
run_ids = np.arange(1, N_RUNS+1)

for ax, rlist, vals, mode, col in [
    (axB, B_runs, B_rmse, "B — NL Supervisor", CB),
    (axC, C_runs, C_rmse, "C — Full-Auto", CC),
]:
    ax.bar(run_ids, vals, color=[RUNS_PAL[i] for i in range(N_RUNS)],
           edgecolor="black", alpha=0.85, width=0.6, zorder=3)
    ax.axhline(np.mean(vals), color="navy", ls="--", lw=1.8,
               label=f"Mean = {np.mean(vals):.3f} cm")
    ax.axhline(A_overall, color=CA, ls=":", lw=2,
               label=f"Mode A = {A_overall:.3f} cm")
    for xi, v in zip(run_ids, vals):
        ax.text(xi, v+0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ci = bootstrap_ci(vals)
    ax.fill_between([-0.3, N_RUNS+0.3], ci[0], ci[1], color=col, alpha=0.12,
                    label=f"95% CI [{ci[0]:.3f},{ci[1]:.3f}]")
    ax.set_xticks(run_ids)
    ax.set_xticklabels([f"Run {i}" for i in run_ids])
    ax.set_xlabel("Run"); ax.set_ylabel("Overall RMSE (cm)")
    ax.set_title(f"Mode {mode[0]}: {mode[4:]}\nRMSE = {np.mean(vals):.3f} ± {np.std(vals):.3f} cm")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y", zorder=0)

axB.set_ylabel("Overall RMSE (cm)", fontsize=11)
fig.suptitle("EXP-C8 v3: Per-Run Overall RMSE — Mode B vs Mode C", fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig2_per_run_rmse_B_and_C.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 2: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Per-waypoint RMSE heatmap (mode × waypoint)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Matrix: rows = runs (B1-B5, C1-C5), cols = waypoints
B_mat = np.array([[r[f"wp{j+1}_rmse_cm"] for j in range(4)] for r in B_runs])
C_mat = np.array([[r[f"wp{j+1}_rmse_cm"] for j in range(4)] for r in C_runs])

for ax, mat, mode_label, col in [(axes[0], B_mat, "Mode B (NL Supervisor)", CB),
                                   (axes[1], C_mat, "Mode C (Full-Auto)", CC)]:
    cmap = LinearSegmentedColormap.from_list("custom", ["#ffffff", col], N=256)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(WP_LABELS, fontsize=9)
    ax.set_yticks(range(N_RUNS)); ax.set_yticklabels([f"Run {i+1}" for i in range(N_RUNS)])
    for i in range(N_RUNS):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="black" if mat[i,j] < 0.9 else "white")
    plt.colorbar(im, ax=ax, label="RMSE (cm)")
    ax.set_title(f"{mode_label}\nPer-run × Per-waypoint RMSE (cm)", fontsize=10)

fig.suptitle("EXP-C8 v3: RMSE Heatmap — Runs × Waypoints", fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig3_per_waypoint_heatmap.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 3: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Per-waypoint grouped bars (all 3 modes)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(4); w = 0.25

B_wp_mean = [np.mean(wp_col(B_runs, j)) for j in range(4)]
B_wp_std  = [np.std(wp_col(B_runs, j))  for j in range(4)]
C_wp_mean = [np.mean(wp_col(C_runs, j)) for j in range(4)]
C_wp_std  = [np.std(wp_col(C_runs, j))  for j in range(4)]

b1 = ax.bar(x-w, A_wp,       w, color=CA, alpha=0.85, edgecolor="black", label="Mode A (scripted, 1 run)")
b2 = ax.bar(x,   B_wp_mean,  w, color=CB, alpha=0.85, edgecolor="black", label="Mode B (supervisor, N=5)")
b3 = ax.bar(x+w, C_wp_mean,  w, color=CC, alpha=0.85, edgecolor="black", label="Mode C (full-auto, N=5)")

ax.errorbar(x,   B_wp_mean, yerr=B_wp_std, fmt="none", ecolor="black", capsize=6, lw=1.5)
ax.errorbar(x+w, C_wp_mean, yerr=C_wp_std, fmt="none", ecolor="black", capsize=6, lw=1.5)

for xi, av, bv, cv in zip(x, A_wp, B_wp_mean, C_wp_mean):
    ax.text(xi-w, av+0.015, f"{av:.3f}", ha="center", fontsize=8)
    ax.text(xi,   bv+0.015, f"{bv:.3f}", ha="center", fontsize=8)
    ax.text(xi+w, cv+0.015, f"{cv:.3f}", ha="center", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(WP_LABELS, fontsize=10)
ax.set_ylabel("Mean RMSE (cm)", fontsize=11)
ax.set_title("EXP-C8 v3: Per-Waypoint RMSE — All Three Modes\n"
             "(error bars = std, N=5 for B/C; A is deterministic 1 run)", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
p = os.path.join(OUT, "C8_fig4_per_waypoint_grouped_bars.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 4: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Radar / spider chart (per-waypoint RMSE, all 3 modes)
# ═══════════════════════════════════════════════════════════════════════════════
categories  = [f"WP{i+1}\n{WAYPOINTS[i]}m" for i in range(4)]
N_cat       = len(categories)
angles      = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
angles     += angles[:1]

A_vals = A_wp + A_wp[:1]
B_vals = B_wp_mean + B_wp_mean[:1]
C_vals = C_wp_mean + C_wp_mean[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

ax.plot(angles, A_vals, "o-", lw=2, color=CA, label=f"Mode A  {A_overall:.3f} cm")
ax.fill(angles, A_vals, alpha=0.12, color=CA)
ax.plot(angles, B_vals, "s-", lw=2, color=CB, label=f"Mode B  {np.mean(B_rmse):.3f} cm")
ax.fill(angles, B_vals, alpha=0.15, color=CB)
ax.plot(angles, C_vals, "^-", lw=2, color=CC, label=f"Mode C  {np.mean(C_rmse):.3f} cm")
ax.fill(angles, C_vals, alpha=0.15, color=CC)

ax.set_ylim(0, max(A_wp)*1.2)
ax.set_title("EXP-C8 v3: Per-Waypoint RMSE — Radar\n(outer = larger error)", fontsize=11, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
fig.tight_layout()
p = os.path.join(OUT, "C8_fig5_wp_radar.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 5: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — RMSE distribution: box + strip + violin for B and C
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: box plots
ax = axes[0]
bp = ax.boxplot([B_rmse, C_rmse], labels=["Mode B\n(supervisor)", "Mode C\n(full-auto)"],
                patch_artist=True, widths=0.4,
                medianprops=dict(color="black", lw=2))
for patch, col in zip(bp["boxes"], [CB, CC]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
for xi, vals, col in [(1, B_rmse, CB), (2, C_rmse, CC)]:
    ax.scatter([xi]*len(vals), vals, color=col, s=60, zorder=5, edgecolors="black", lw=0.8)
ax.axhline(A_overall, color=CA, ls=":", lw=2, label=f"Mode A = {A_overall:.3f} cm")
ax.set_ylabel("Overall RMSE (cm)"); ax.set_title("Box + Strip Plot"); ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Panel 2: violin
ax = axes[1]
parts = ax.violinplot([B_rmse, C_rmse], positions=[1,2], showmeans=True, showextrema=True)
for pc, col in zip(parts["bodies"], [CB, CC]):
    pc.set_facecolor(col); pc.set_alpha(0.6)
ax.scatter([1]*len(B_rmse), B_rmse, color=CB, s=60, zorder=5, edgecolors="black", lw=0.8)
ax.scatter([2]*len(C_rmse), C_rmse, color=CC, s=60, zorder=5, edgecolors="black", lw=0.8)
ax.axhline(A_overall, color=CA, ls=":", lw=2, label=f"Mode A = {A_overall:.3f} cm")
ax.set_xticks([1,2]); ax.set_xticklabels(["Mode B", "Mode C"])
ax.set_title("Violin Plot"); ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: CDF
ax = axes[2]
for vals, col, lbl in [(B_rmse, CB, "Mode B"), (C_rmse, CC, "Mode C")]:
    sorted_v = np.sort(vals)
    cdf = np.arange(1, len(sorted_v)+1) / len(sorted_v)
    ax.step(sorted_v, cdf, color=col, lw=2, label=lbl, where="post")
    ax.scatter(sorted_v, cdf, color=col, s=50, zorder=5)
ax.axvline(A_overall, color=CA, ls=":", lw=2, label=f"Mode A = {A_overall:.3f} cm")
ax.set_xlabel("RMSE (cm)"); ax.set_ylabel("Cumulative probability")
ax.set_title("Empirical CDF"); ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("EXP-C8 v3: RMSE Distribution — Mode B vs Mode C", fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig6_rmse_distribution.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 6: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — RMSE vs cost scatter (each run a point)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for runs_list, vals, costs, col, lbl in [
    (B_runs, B_rmse, B_cost, CB, "Mode B"),
    (C_runs, C_rmse, C_cost, CC, "Mode C"),
]:
    ax.scatter(costs, vals, color=col, s=100, zorder=5, edgecolors="black",
               lw=0.8, label=lbl)
    for i, (c, v) in enumerate(zip(costs, vals)):
        ax.annotate(f"  R{i+1}", (c, v), fontsize=8, color=col)

# Mean crosses
ax.scatter([np.mean(B_cost)], [np.mean(B_rmse)], marker="X", s=200,
           color=CB, edgecolors="black", lw=1.2, zorder=6,
           label=f"B mean ({np.mean(B_cost):.3f} USD, {np.mean(B_rmse):.3f} cm)")
ax.scatter([np.mean(C_cost)], [np.mean(C_rmse)], marker="X", s=200,
           color=CC, edgecolors="black", lw=1.2, zorder=6,
           label=f"C mean ({np.mean(C_cost):.3f} USD, {np.mean(C_rmse):.3f} cm)")

ax.set_xlabel("Cost per run (USD)", fontsize=11)
ax.set_ylabel("Overall RMSE (cm)", fontsize=11)
ax.set_title("EXP-C8 v3: RMSE vs Cost — Each Run\n"
             "(lower-left = better accuracy AND lower cost)", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
p = os.path.join(OUT, "C8_fig7_rmse_vs_cost_scatter.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 7: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Token usage per run (input vs output, B and C)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
run_ids = np.arange(1, N_RUNS+1)

for ax, in_toks, out_toks, costs, col, lbl in [
    (axes[0], B_in_tok, B_out_tok, B_cost, CB, "Mode B (NL Supervisor)"),
    (axes[1], C_in_tok, C_out_tok, C_cost, CC, "Mode C (Full-Auto)"),
]:
    x = run_ids
    b_in  = ax.bar(x-0.2, [t/1000 for t in in_toks],  0.38, label="Input tokens (k)",
                   color=col, alpha=0.7, edgecolor="black")
    b_out = ax.bar(x+0.2, [t/1000 for t in out_toks], 0.38, label="Output tokens (k)",
                   color=col, alpha=0.35, edgecolor="black", hatch="//")
    ax2 = ax.twinx()
    ax2.plot(x, costs, "o--", color="red", lw=1.5, ms=6, label="Cost (USD)")
    ax2.set_ylabel("Cost (USD)", color="red", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="red")
    for xi, c in zip(x, costs):
        ax2.annotate(f"${c:.2f}", (xi, c), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="red")
    ax.set_xticks(x); ax.set_xticklabels([f"Run {i}" for i in x])
    ax.set_ylabel("Tokens (thousands)", fontsize=10)
    ax.set_title(f"{lbl}\nToken usage and cost per run", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("EXP-C8 v3: Token Usage per Run — Mode B vs Mode C", fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig8_token_usage.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 8: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9 — API calls and cost side-by-side
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: API calls grouped
ax = axes[0]
x = run_ids
ax.bar(x-0.2, B_api, 0.38, color=CB, alpha=0.8, edgecolor="black", label="Mode B")
ax.bar(x+0.2, C_api, 0.38, color=CC, alpha=0.8, edgecolor="black", label="Mode C")
ax.axhline(np.mean(B_api), color=CB, ls="--", lw=1.5, label=f"B mean={np.mean(B_api):.1f}")
ax.axhline(np.mean(C_api), color=CC, ls="--", lw=1.5, label=f"C mean={np.mean(C_api):.1f}")
for xi, b, c in zip(x, B_api, C_api):
    ax.text(xi-0.2, b+0.3, str(int(b)), ha="center", fontsize=9)
    ax.text(xi+0.2, c+0.3, str(int(c)), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"Run {i}" for i in x])
ax.set_ylabel("API calls"); ax.set_title("API Calls per Run")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# Panel 2: cost per run
ax = axes[1]
total_B = sum(B_cost); total_C = sum(C_cost)
ax.bar(x-0.2, B_cost, 0.38, color=CB, alpha=0.8, edgecolor="black", label=f"Mode B (total ${total_B:.2f})")
ax.bar(x+0.2, C_cost, 0.38, color=CC, alpha=0.8, edgecolor="black", label=f"Mode C (total ${total_C:.2f})")
for xi, b, c in zip(x, B_cost, C_cost):
    ax.text(xi-0.2, b+0.01, f"${b:.2f}", ha="center", fontsize=8)
    ax.text(xi+0.2, c+0.01, f"${c:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"Run {i}" for i in x])
ax.set_ylabel("Cost (USD)"); ax.set_title("Cost per Run (USD)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("EXP-C8 v3: API Calls and Cost — Mode B vs Mode C", fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig9_api_and_cost_breakdown.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 9: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Improvement factor: how many times B and C beat Mode A
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: overall improvement factor per run
ax = axes[0]
B_factor = [A_overall / v for v in B_rmse]
C_factor = [A_overall / v for v in C_rmse]
ax.bar(x-0.2, B_factor, 0.38, color=CB, alpha=0.8, edgecolor="black", label="Mode B")
ax.bar(x+0.2, C_factor, 0.38, color=CC, alpha=0.8, edgecolor="black", label="Mode C")
ax.axhline(1.0, color="gray", ls="--", lw=1.5, label="Parity (= Mode A)")
ax.axhline(np.mean(B_factor), color=CB, ls=":", lw=1.5, label=f"B mean={np.mean(B_factor):.2f}×")
ax.axhline(np.mean(C_factor), color=CC, ls=":", lw=1.5, label=f"C mean={np.mean(C_factor):.2f}×")
for xi, bf, cf in zip(x, B_factor, C_factor):
    ax.text(xi-0.2, bf+0.03, f"{bf:.2f}×", ha="center", fontsize=9)
    ax.text(xi+0.2, cf+0.03, f"{cf:.2f}×", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"Run {i}" for i in x])
ax.set_ylabel("Improvement factor (A RMSE / Mode RMSE)")
ax.set_title("Overall Improvement Factor vs Mode A")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# Panel 2: per-waypoint improvement factor (mean over runs)
ax = axes[1]
B_wp_factor = [A_wp[j] / np.mean(wp_col(B_runs, j)) for j in range(4)]
C_wp_factor = [A_wp[j] / np.mean(wp_col(C_runs, j)) for j in range(4)]
xw = np.arange(4)
ax.bar(xw-0.2, B_wp_factor, 0.38, color=CB, alpha=0.8, edgecolor="black", label="Mode B")
ax.bar(xw+0.2, C_wp_factor, 0.38, color=CC, alpha=0.8, edgecolor="black", label="Mode C")
ax.axhline(1.0, color="gray", ls="--", lw=1.5, label="Parity")
for xi, bf, cf in zip(xw, B_wp_factor, C_wp_factor):
    ax.text(xi-0.2, bf+0.03, f"{bf:.2f}×", ha="center", fontsize=9)
    ax.text(xi+0.2, cf+0.03, f"{cf:.2f}×", ha="center", fontsize=9)
ax.set_xticks(xw); ax.set_xticklabels(WP_LABELS)
ax.set_ylabel("Improvement factor")
ax.set_title("Per-Waypoint Improvement Factor vs Mode A")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("EXP-C8 v3: Improvement Factor — LLM Modes vs Scripted Baseline",
             fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig10_improvement_factor.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 10: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Mode B vs Mode C head-to-head scatter (run-matched)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: overall RMSE scatter
ax = axes[0]
ax.scatter(C_rmse, B_rmse, s=120, c=RUNS_PAL[:N_RUNS], edgecolors="black",
           lw=0.8, zorder=5)
for i, (c, b) in enumerate(zip(C_rmse, B_rmse)):
    ax.annotate(f"  R{i+1}", (c, b), fontsize=9)
lim = [0.78, 0.95]
ax.plot(lim, lim, "k--", lw=1.2, label="B = C (parity line)")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Mode C RMSE (cm)"); ax.set_ylabel("Mode B RMSE (cm)")
ax.set_title("Head-to-Head: Overall RMSE\n(run-matched, each point = one run pair)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 2: per-waypoint B vs C (all runs together)
ax = axes[1]
wp_colors = ["#3498db","#e67e22","#9b59b6","#1abc9c"]
for j in range(4):
    bv = wp_col(B_runs, j); cv = wp_col(C_runs, j)
    ax.scatter(cv, bv, s=80, color=wp_colors[j], edgecolors="black",
               lw=0.5, label=WP_LABELS[j], zorder=5)
lim2 = [0.4, 1.4]
ax.plot(lim2, lim2, "k--", lw=1.2, label="B = C")
ax.set_xlim(lim2); ax.set_ylim(lim2)
ax.set_xlabel("Mode C per-WP RMSE (cm)"); ax.set_ylabel("Mode B per-WP RMSE (cm)")
ax.set_title("Head-to-Head: Per-Waypoint RMSE\n(all 5 runs × 4 WPs = 20 points)")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

fig.suptitle("EXP-C8 v3: Mode B vs Mode C Head-to-Head (Run-Matched)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(OUT, "C8_fig11_B_vs_C_head_to_head.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 11: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Publication-ready summary table
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

col_headers = ["Mode", "RMSE (cm)\nmean ± std", "95% CI\n(cm)", "Pass\nrate",
               "API calls\nmean ± std", "Cost/run\n(USD)", "Sim time\n(s)",
               "WP1 (cm)", "WP2 (cm)", "WP3 (cm)", "WP4 (cm)"]

B_rmse_ci = bootstrap_ci(B_rmse)
C_rmse_ci = bootstrap_ci(C_rmse)

rows = [
    ["A — Scripted\n(1 run, det.)",
     f"{A_overall:.3f}",
     "—",
     "1/1",
     "0",
     "—",
     "73.1",
     f"{A_wp[0]:.3f}", f"{A_wp[1]:.3f}", f"{A_wp[2]:.3f}", f"{A_wp[3]:.3f}"],
    ["B — NL Supervisor\n(5 turns/mission)",
     f"{np.mean(B_rmse):.3f} ± {np.std(B_rmse):.3f}",
     f"[{B_rmse_ci[0]:.3f}, {B_rmse_ci[1]:.3f}]",
     "5/5",
     f"{np.mean(B_api):.1f} ± {np.std(B_api):.1f}",
     f"${np.mean(B_cost):.3f}",
     "59.9",
     f"{np.mean(wp_col(B_runs,0)):.3f}±{np.std(wp_col(B_runs,0)):.3f}",
     f"{np.mean(wp_col(B_runs,1)):.3f}±{np.std(wp_col(B_runs,1)):.3f}",
     f"{np.mean(wp_col(B_runs,2)):.3f}±{np.std(wp_col(B_runs,2)):.3f}",
     f"{np.mean(wp_col(B_runs,3)):.3f}±{np.std(wp_col(B_runs,3)):.3f}"],
    ["C — Full-Auto\n(single command)",
     f"{np.mean(C_rmse):.3f} ± {np.std(C_rmse):.3f}",
     f"[{C_rmse_ci[0]:.3f}, {C_rmse_ci[1]:.3f}]",
     "5/5",
     f"{np.mean(C_api):.1f} ± {np.std(C_api):.1f}",
     f"${np.mean(C_cost):.3f}",
     "59.9",
     f"{np.mean(wp_col(C_runs,0)):.3f}±{np.std(wp_col(C_runs,0)):.3f}",
     f"{np.mean(wp_col(C_runs,1)):.3f}±{np.std(wp_col(C_runs,1)):.3f}",
     f"{np.mean(wp_col(C_runs,2)):.3f}±{np.std(wp_col(C_runs,2)):.3f}",
     f"{np.mean(wp_col(C_runs,3)):.3f}±{np.std(wp_col(C_runs,3)):.3f}"],
]

tbl = ax.table(cellText=rows, colLabels=col_headers,
               cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.90])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)

# Header style
for j in range(len(col_headers)):
    tbl[0, j].set_facecolor("#2c3e50"); tbl[0, j].set_text_props(color="white", fontweight="bold")

# Row colours
row_cols = [CA, CB, CC]
for i, col in enumerate(row_cols, start=1):
    for j in range(len(col_headers)):
        tbl[i, j].set_facecolor(col); tbl[i, j].set_alpha(0.18)
    tbl[i, 0].set_facecolor(col); tbl[i, 0].set_alpha(0.55)

ax.set_title(
    "EXP-C8 v3: Three-Mode Comparison — Publication Summary Table\n"
    "Mission: [0.8, 1.2, 1.5, 1.0] m waypoints · 8 s hold each · "
    "RMSE = backward confirmed-arrival window · N=5 independent runs (B, C)",
    fontsize=10, fontweight="bold", pad=8
)
fig.tight_layout()
p = os.path.join(OUT, "C8_fig12_summary_table.png")
plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close()
print(f"  Fig 12: {p}")


print("\n[C8 plots] All 12 figures saved to:", OUT)
print("Files:")
for i in range(1, 13):
    names = {1:"overall_rmse_comparison", 2:"per_run_rmse_B_and_C",
             3:"per_waypoint_heatmap", 4:"per_waypoint_grouped_bars",
             5:"wp_radar", 6:"rmse_distribution", 7:"rmse_vs_cost_scatter",
             8:"token_usage", 9:"api_and_cost_breakdown",
             10:"improvement_factor", 11:"B_vs_C_head_to_head",
             12:"summary_table"}
    print(f"  C8_fig{i}_{names[i]}.png")
