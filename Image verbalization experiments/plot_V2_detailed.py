"""
plot_V2_detailed.py — V2 Thesis Plots (Reasoning Quality Focus)
================================================================
Three thesis-quality plots for V2 (Prompt Technique Comparison).
Focus: how well each technique and model understands the scene
       and produces correct pilot actions — not label accuracy.
ReAct excluded (one-shot unsuitable; validated separately in C-series).

  Fig 1 — Reasoning quality by technique × model
           DescAcc + RsnRisk + RsnAct grouped bars per technique
  Fig 2 — Reasoning profile per model (all 4 techniques combined)
           DescAcc, RsnRisk, RsnAct, ActSafe — which model reasons best?
  Fig 3 — Heatmap: model × technique, composite reasoning score
           Best model + technique combination at a glance

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_V2_detailed.py"
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

VIZ_DIR  = pathlib.Path(__file__).parent
RES_DIR  = VIZ_DIR / "results"
PLOT_DIR = RES_DIR / "V2_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
v2 = pd.read_csv(RES_DIR / "V2_runs_20260525_035756.csv")
v2 = v2[v2["error"].isna() | (v2["error"] == "")].copy()

# Drop ReAct — not suited for one-shot; validated separately in C-series
v2 = v2[v2["technique"] != "react"].copy()

# ── Fix s5_pilot_action scoring ───────────────────────────────────────────────
# When detected_action is NaN the scorer sets s5_pilot_action=0 because it
# cannot find a parseable action keyword — this is a FORMAT failure (Claude
# writes narrative without a clean action line), not a reasoning contradiction.
# We NaN-out s5_pilot_action wherever detected_action is missing so means are
# computed only over trials that actually produced a detectable action.
v2.loc[v2["detected_action"].isna(), "s5_pilot_action"] = np.nan

MODELS       = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS = ["Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS = ["#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]

TECHNIQUES   = ["zero_shot", "few_shot_3", "cot", "structured"]
TECH_LABELS  = ["Zero-Shot", "Few-Shot (3)", "CoT", "Structured"]

# Reasoning metrics and their CSV column names
# s1_scene   = scene description accuracy (DescAcc)
# s2_proximity = reasoning–risk alignment (RsnRisk)
# s5_pilot_action = reason–action alignment (RsnAct)
# action_safe = action safety (ActSafe)
REASON_COLS   = ["s1_scene", "s2_proximity", "s5_pilot_action", "action_safe"]
REASON_LABELS = ["Scene\nUnderstanding", "Risk\nReasoning", "Action\nConsistency", "Action\nSafety"]
REASON_COLORS = ["#4a90d9", "#e07b54", "#6cbf6c", "#b07ed9"]

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
# FIG 1 — Reasoning quality by technique × model
# For each technique, show DescAcc / RsnRisk / RsnAct as grouped bars.
# Each cluster = one technique, bars split by reasoning metric.
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(
    "Fig 1 — LLM Reasoning Quality by Prompt Technique and Model\n"
    "(Scene Understanding · Risk Reasoning · Action Consistency†)\n"
    "†Action Consistency computed only over trials that produced a parseable action",
    fontsize=11, y=1.04
)

metric_pairs = [
    ("s1_scene",         "Scene Understanding (DescAcc)"),
    ("s2_proximity",     "Risk Reasoning (RsnRisk)"),
    ("s5_pilot_action",  "Action Consistency (RsnAct)"),
]

for ax, (col, title) in zip(axes, metric_pairs):
    x      = np.arange(len(TECHNIQUES))
    width  = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODELS))

    for i, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLORS)):
        means = []
        for tech in TECHNIQUES:
            sub = v2[(v2["model"] == model) & (v2["technique"] == tech)]
            means.append(sub[col].mean() * 100 if len(sub) > 0 else 0)
        ax.bar(x + offsets[i], means, width,
               label=label, color=color, alpha=0.87,
               edgecolor="white", linewidth=0.5)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(TECH_LABELS, fontsize=9.5)
    ax.set_ylim(60, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

axes[0].set_ylabel("Score (%)")
axes[0].legend(loc="lower left", framealpha=0.9, fontsize=8.5)

plt.tight_layout()
out1 = PLOT_DIR / "V2_fig1_reasoning_by_technique_model.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Reasoning profile per model (all 4 techniques combined)
# Shows which model has the best scene understanding and action quality overall.
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x      = np.arange(len(MODELS))
width  = 0.18
offsets = np.linspace(-1.5 * width, 1.5 * width, len(REASON_COLS))

for j, (col, label, color) in enumerate(zip(REASON_COLS, REASON_LABELS, REASON_COLORS)):
    means = []
    for model in MODELS:
        sub = v2[v2["model"] == model]
        means.append(sub[col].mean() * 100 if len(sub) > 0 else 0)
    bars = ax.bar(x + offsets[j], means, width,
                  label=label.replace("\n", " "), color=color,
                  alpha=0.87, edgecolor="white", linewidth=0.5)
    # Value labels on top of each bar
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=7.5, color="dimgray")

ax.set_xlabel("Model")
ax.set_ylabel("Score (%)")
ax.set_title("Fig 2 — LLM Reasoning Profile per Model\n"
             "Scene Understanding · Risk Reasoning · Action Consistency† · Action Safety\n"
             "(averaged across Zero-Shot, Few-Shot, CoT, Structured)\n"
             "†Action Consistency: only trials with a parseable action counted")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS)
ax.set_ylim(60, 110)
ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

plt.tight_layout()
out2 = PLOT_DIR / "V2_fig2_reasoning_profile_per_model.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Heatmap: model × technique, composite reasoning score
# Composite = mean(DescAcc, RsnRisk, RsnAct) — excludes label accuracy
# ─────────────────────────────────────────────────────────────────────────────
reasoning_cols = ["s1_scene", "s2_proximity", "s5_pilot_action"]

matrix = np.zeros((len(MODELS), len(TECHNIQUES)))
for i, model in enumerate(MODELS):
    for j, tech in enumerate(TECHNIQUES):
        sub = v2[(v2["model"] == model) & (v2["technique"] == tech)]
        if len(sub) > 0:
            matrix[i, j] = sub[reasoning_cols].mean(axis=1).mean() * 100

fig, ax = plt.subplots(figsize=(9, 4))
cmap = LinearSegmentedColormap.from_list("rg", ["#d73027", "#ffffbf", "#1a9850"])
im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=75, vmax=100)

ax.set_xticks(range(len(TECH_LABELS)))
ax.set_xticklabels(TECH_LABELS, fontsize=11)
ax.set_yticks(range(len(MODEL_LABELS)))
ax.set_yticklabels(MODEL_LABELS, fontsize=11)
ax.set_title("Fig 3 — Composite Reasoning Score: Model × Technique (%)\n"
             "Mean of Scene Understanding + Risk Reasoning + Action Consistency")

for i in range(len(MODELS)):
    for j in range(len(TECHNIQUES)):
        val   = matrix[i, j]
        color = "white" if val < 87 else "black"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=11.5, color=color, fontweight="bold")

# Highlight the best cell
best_i, best_j = np.unravel_index(np.argmax(matrix), matrix.shape)
ax.add_patch(plt.Rectangle(
    (best_j - 0.5, best_i - 0.5), 1, 1,
    fill=False, edgecolor="gold", linewidth=3, zorder=10
))
ax.text(best_j, best_i + 0.42, "★", ha="center", va="center",
        fontsize=9, color="gold", zorder=11)

cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Composite Reasoning Score (%)", fontsize=9)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
out3 = PLOT_DIR / "V2_fig3_reasoning_heatmap_model_technique.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
