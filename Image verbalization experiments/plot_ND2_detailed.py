"""
plot_ND2_detailed.py — ND2 Thesis Plots (Agentic Patrol Mission)
================================================================
Three thesis-quality plots for ND2:

  Fig 1 — Patrol completion & correct classification per model × scenario
           Shows asymmetric failure: Claude fails clear / GPT-4o-mini fails alert
  Fig 2 — GPT-4o-mini 3-stage progression: Original → M1–M3 → M1–M6
           Patrol 0% → 0% → 100%  |  Cost $0.615 → $0.215 → $0.027
  Fig 3 — Cost vs quality scatter: all model × scenario combinations
           Gemini: cheapest; Claude alert: most thorough but $2.63/run

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_ND2_detailed.py"
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────
VIZ_DIR  = pathlib.Path(__file__).parent
RES_DIR  = VIZ_DIR.parent / "experiments" / "results"
PLOT_DIR = RES_DIR / "ND2_plots_thesis"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
runs = pd.read_csv(RES_DIR / "ND2_runs_20260528_224548.csv")   # all 4 models
fix  = pd.read_csv(RES_DIR / "ND2_fix_litm_gpt4omini_alert_20260529_175238.csv")  # M1-M6
wf   = pd.read_csv(RES_DIR / "ND2_workflow_20260528_224548.csv")

# ── Derive patrol_active: did the drone navigate to at least 1 waypoint? ──────
# This separates *patrol behaviour* from *mission termination* (landing).
# Claude clear_room: navigated 1–5 waypoints in 4/5 runs → patrol_active, just no land()
# GPT-4o-mini alert_room: 0 navigate calls, 16–23 hover() calls → NOT patrolling
nav_counts = (
    wf[wf["tool_name"] == "navigate_to_waypoint"]
    .groupby(["orchestrator", "scenario", "run"])
    .size()
    .reset_index(name="nav_calls")
)
runs = runs.rename(columns={"global_run": "run"})
runs = runs.merge(nav_counts, how="left",
                  left_on=["orchestrator", "scenario", "run"],
                  right_on=["orchestrator", "scenario", "run"])
runs["nav_calls"] = runs["nav_calls"].fillna(0).astype(int)
# patrol_active = drone moved to at least 1 waypoint (left its takeoff position)
runs["patrol_active"] = (runs["nav_calls"] >= 1).astype(int)

MODELS        = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS  = ["Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS  = ["#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]
SCENARIOS     = ["clear_room", "alert_room"]
SCEN_LABELS   = ["Clear Room\n(truth=CLEAR)", "Alert Room\n(truth=ALERT)"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Patrol completion & correct classification per model × scenario
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig.suptitle(
    "Fig 1 — ND2 Mission Outcomes: Patrol Active vs Mission Complete vs Correct Classification\n"
    "Patrol Active = drone navigated waypoints · Mission Complete = patrol + land · "
    "Claude patrolled but never landed · GPT-4o-mini never left takeoff position",
    fontsize=11)

x     = np.arange(len(MODELS))
width = 0.22

# Three metrics: patrol active (drone moved), mission complete (patrol+land), correct classification
metric_triples = [
    ("patrol_active",      "Patrol Active\n(navigated waypoints)",   "#2ecc71"),
    ("patrol_complete",    "Mission Complete\n(patrol + land)",       "#f39c12"),
    ("correct_room_report","Correct Report\n(right classification)",  "#3498db"),
]

for ax, (scen, scen_label) in zip(axes, zip(SCENARIOS, SCEN_LABELS)):
    for si, (metric, mlabel, mcolor) in enumerate(metric_triples):
        vals = [
            runs[(runs["orchestrator"] == m) & (runs["scenario"] == scen)][metric].mean() * 100
            for m in MODELS
        ]
        offset = (si - 1) * width
        bars = ax.bar(x + offset, vals, width, label=mlabel,
                      color=mcolor, alpha=0.87, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="#c0392b" if (val < 99 and metric != "patrol_complete") else
                          "#e67e22" if (val < 99 and metric == "patrol_complete") else
                          "dimgray")

    # Annotate key story per scenario
    if scen == "clear_room":
        claude_idx = MODELS.index("claude")
        ax.annotate("Patrolled but\nnever landed",
                    xy=(x[claude_idx], 2),
                    xytext=(x[claude_idx] + 0.55, 35),
                    arrowprops=dict(arrowstyle="->", color="#e07b54", lw=1.3),
                    fontsize=8, color="#e07b54", fontweight="bold")
    else:
        mini_idx = MODELS.index("gpt4o_mini")
        ax.annotate("87 hover()\n0 nav calls",
                    xy=(x[mini_idx] - width, 2),
                    xytext=(x[mini_idx] - width + 0.55, 28),
                    arrowprops=dict(arrowstyle="->", color="#6cbf6c", lw=1.3),
                    fontsize=8, color="#6cbf6c", fontweight="bold")

    ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_title(scen_label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylim(0, 125)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

axes[0].set_ylabel("Rate (%)")

plt.tight_layout()
out1 = PLOT_DIR / "ND2_fig1_patrol_vs_correct.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — GPT-4o-mini 3-stage progression: Original → M1–M3 → M1–M6
# ─────────────────────────────────────────────────────────────────────────────

# Stage data (alert_room only — the failure scenario)
# Original ND2: from ND2_runs_20260528_224548.csv (5 runs)
orig = runs[(runs["orchestrator"] == "gpt4o_mini") & (runs["scenario"] == "alert_room")].copy()

# M1–M3: single exploratory rerun (hardcoded — no saved CSV for intermediate)
# Values from ND2_observations_20260529.md Finding 10 — Expected Impact table
m13_data = {
    "patrol_complete":    0.0,
    "quality_score":      0.0,
    "n_scene_calls":     10.0,
    "total_cost_usd":     0.215,
    "wall_time_s":       617.0,
    "word_count":          0.0,
    "truncated":           1.0,
}

# M1–M6: from ND2_fix_litm_gpt4omini_alert_20260529_175238.csv (5 runs)
fix_means = fix.mean(numeric_only=True)

stages       = ["Original\nND2", "After\nM1–M3", "After\nM1–M6"]
stage_colors = ["#e74c3c", "#e67e22", "#27ae60"]

metrics_to_plot = [
    ("patrol_complete",  "Patrol Complete (%)",     True,   False),  # (field, label, is_pct, is_log)
    ("quality_score",    "Quality Score (0–5)",     False,  False),
    ("n_scene_calls",    "Scene Calls per Run",     False,  False),
    ("total_cost_usd",   "Cost per Run (USD)",      False,  True),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    "Fig 2 — GPT-4o-mini Alert-Room: 3-Stage Fix Progression\n"
    "M1–M3 broke the infinite loop but didn't complete mission; M1–M6 achieves 100% patrol + $0.027/run",
    fontsize=12)

orig_vals = {
    "patrol_complete":  orig["patrol_complete"].mean(),
    "quality_score":    orig["quality_score"].mean(),
    "n_scene_calls":    orig["n_scene_calls"].mean(),
    "total_cost_usd":   orig["total_cost_usd"].mean(),
    "word_count":       orig["word_count"].mean(),
    "truncated":        orig["truncated"].mean(),
    "wall_time_s":      orig["wall_time_s"].mean(),
}
fix_vals = {
    "patrol_complete":  fix_means["patrol_complete"],
    "quality_score":    fix_means["quality_score"],
    "n_scene_calls":    fix_means["n_scene_calls"],
    "total_cost_usd":   fix_means["total_cost_usd"],
    "word_count":       fix_means["word_count"],
    "truncated":        fix_means["truncated"],
    "wall_time_s":      fix_means["wall_time_s"],
}

for ax, (field, ylabel, is_pct, is_log) in zip(axes, metrics_to_plot):
    stage_vals = [
        orig_vals[field] * (100 if is_pct else 1),
        m13_data[field]  * (100 if is_pct else 1),
        fix_vals[field]  * (100 if is_pct else 1),
    ]
    bars = ax.bar(range(3), stage_vals, color=stage_colors, alpha=0.87,
                  edgecolor="white", width=0.55)

    for bar, val in zip(bars, stage_vals):
        label = (f"{val:.0f}%" if is_pct else
                 f"${val:.3f}" if field == "total_cost_usd" else
                 f"{val:.1f}")
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * (1.08 if is_log else 1) + (0 if is_log else max(stage_vals) * 0.02),
                label,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="dimgray")

    if is_log and all(v > 0 for v in stage_vals):
        ax.set_yscale("log")

    ax.set_title(ylabel, fontsize=10.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(stages, fontsize=9.5)
    ax.set_ylim(bottom=0 if not is_log else None)
    if is_pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    elif field == "total_cost_usd":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.3f}"))

# Highlight the M1-M6 bars with a green outline
for ax in axes:
    ax.patches[2].set_edgecolor("#27ae60")
    ax.patches[2].set_linewidth(2.5)

plt.tight_layout()
out2 = PLOT_DIR / "ND2_fig2_gpt4omini_fix_progression.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Cost vs quality scatter: all model × scenario combinations
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))

model_color = dict(zip(MODELS, MODEL_COLORS))
model_label_map = dict(zip(MODELS, MODEL_LABELS))
scen_markers = {"clear_room": "o", "alert_room": "s"}
scen_label_map = {"clear_room": "clear_room", "alert_room": "alert_room"}

for model in MODELS:
    for scen in SCENARIOS:
        sub = runs[(runs["orchestrator"] == model) & (runs["scenario"] == scen)]
        if len(sub) == 0:
            continue
        cost    = sub["total_cost_usd"].mean()
        quality = sub["quality_score"].mean()
        patrol  = sub["patrol_active"].mean() * 100

        color  = model_color[model]
        marker = scen_markers[scen]
        size   = 220 + patrol * 2.5   # bigger circle = higher patrol rate

        ax.scatter(cost, quality,
                   s=size, c=color, marker=marker,
                   alpha=0.87, edgecolors="white", linewidth=1.5, zorder=5)

        # Label offset to avoid overlap
        offsets = {
            ("claude",     "clear_room"):  (-0.10,  0.05),
            ("claude",     "alert_room"):  ( 0.04,  0.05),
            ("gpt4o",      "clear_room"):  (-0.06, -0.12),
            ("gpt4o",      "alert_room"):  ( 0.04,  0.05),
            ("gpt4o_mini", "clear_room"):  (-0.01, -0.13),
            ("gpt4o_mini", "alert_room"):  ( 0.04,  0.05),
            ("gemini",     "clear_room"):  (-0.04, -0.13),
            ("gemini",     "alert_room"):  ( 0.04,  0.05),
        }
        dx, dy = offsets.get((model, scen), (0.03, 0.03))
        ax.annotate(
            f"{model_label_map[model]}\n{scen_label_map[scen]}\n({patrol:.0f}% patrol)",
            (cost, quality),
            xytext=(cost + dx, quality + dy),
            fontsize=7.5, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.5),
        )

# Add M1-M6 fix point for GPT-4o-mini alert_room
fix_cost    = fix_means["total_cost_usd"]
fix_quality = fix_means["quality_score"]
ax.scatter(fix_cost, fix_quality, s=370, c="#6cbf6c", marker="*",
           alpha=0.95, edgecolors="white", linewidth=1.5, zorder=6)
ax.annotate("GPT-4o-mini\nalert M1–M6\n(100% patrol)",
            (fix_cost, fix_quality),
            xytext=(fix_cost + 0.06, fix_quality + 0.06),
            fontsize=8, color="#6cbf6c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#6cbf6c", lw=1.2))

# Legend
legend_elements = [
    mpatches.Patch(color=MODEL_COLORS[i], label=MODEL_LABELS[i])
    for i in range(len(MODELS))
]
legend_elements += [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
           markersize=10, label="clear_room"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="gray",
           markersize=10, label="alert_room"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#6cbf6c",
           markersize=12, label="GPT-4o-mini after M1–M6"),
]
ax.legend(handles=legend_elements, loc="upper left",
          framealpha=0.9, fontsize=9, ncol=2)

# Cost zones
ax.axvspan(0,    0.05, alpha=0.04, color="#27ae60")
ax.axvspan(0.05, 0.50, alpha=0.04, color="#f39c12")
ax.axvspan(0.50, 3.5,  alpha=0.04, color="#e74c3c")
ax.text(0.025, 0.15, "Budget\n(<$0.05)", fontsize=8, color="#27ae60", alpha=0.8, ha="center")
ax.text(0.27,  0.15, "Moderate\n($0.05–$0.50)", fontsize=8, color="#f39c12", alpha=0.8, ha="center")
ax.text(1.8,   0.15, "Expensive\n(>$0.50)", fontsize=8, color="#e74c3c", alpha=0.8, ha="center")

ax.set_xlabel("Mean Cost per Run (USD)")
ax.set_ylabel("Mean Quality Score (0–5)")
ax.set_title(
    "Fig 3 — Cost vs Quality: All Model × Scenario Combinations\n"
    "Size = patrol active rate (drone navigated) · ○ clear_room · □ alert_room · ★ GPT-4o-mini after M1–M6",
    fontsize=12)
ax.set_xlim(-0.05, 3.2)
ax.set_ylim(0, 5.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))

plt.tight_layout()
out3 = PLOT_DIR / "ND2_fig3_cost_vs_quality.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
