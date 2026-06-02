"""
plot_G1_detailed.py — G1 Thesis Plots (Tier Comparison)
=========================================================
Three thesis-quality plots for G1:

  Fig 1 — Action safety by condition × model (grouped bar)
           Shows tier2_only blind spot; LLM conditions mostly safe
  Fig 2 — Dangerous cases heatmap: condition × scene
           Shows WHERE each architecture fails (blocked_lens vs wall_close)
  Fig 3 — Label accuracy vs latency scatter
           Tradeoff: tier1.5 fast/blunt, GPT-4o LLM-only best accuracy

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_G1_detailed.py"
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
PLOT_DIR = RES_DIR / "G1_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RES_DIR / "G1_runs_20260525_021111.csv")
df = df[df["error"].isna() | (df["error"] == "")].copy()

MODELS        = ["rule_based", "claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS  = ["Rule-Based", "Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS  = ["#95a5a6", "#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]

CONDITIONS      = ["tier1_5_only", "tier2_only", "tier3_only", "tier1_5_tier2_tier3"]
COND_LABELS     = ["Tier 1.5 Only\n(MediaPipe\n14ms)", "Tier 2 Only\n(YOLO+Depth\n275ms)",
                   "Tier 3 Only\n(LLM, no metadata)", "Full Pipeline\n(Tier1.5+2+3)"]
COND_LABELS_SHORT = ["Tier 1.5\nOnly", "Tier 2\nOnly", "Tier 3\nOnly\n(LLM)", "Full\nPipeline"]

SCENES = ["person_near", "wall_close", "blocked_lens", "person_far",
          "dim_light", "cluttered", "object_table", "door_open"]
SCENE_LABELS = ["person\nnear", "wall\nclose", "blocked\nlens", "person\nfar",
                "dim\nlight", "cluttered", "object\ntable", "door\nopen"]

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
# FIG 1 — Action safety by condition × model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

# Which models appear in each condition
cond_models = {
    "tier1_5_only":          ["rule_based"],
    "tier2_only":            ["rule_based"],
    "tier3_only":            ["claude", "gpt4o", "gpt4o_mini", "gemini"],
    "tier1_5_tier2_tier3":   ["claude", "gpt4o", "gpt4o_mini", "gemini"],
}

x      = np.arange(len(CONDITIONS))
width  = 0.18
offsets = np.linspace(-1.5 * width, 1.5 * width, 4)  # max 4 model bars per condition
model_color = dict(zip(MODELS, MODEL_COLORS))
model_label = dict(zip(MODELS, MODEL_LABELS))

plotted = set()
for ci, (cond, clabel) in enumerate(zip(CONDITIONS, COND_LABELS)):
    models_in = cond_models[cond]
    n = len(models_in)
    local_offsets = np.linspace(-(n-1)/2 * width, (n-1)/2 * width, n)
    for mi, model in enumerate(models_in):
        sub = df[(df["condition"] == cond) & (df["model"] == model)]
        val = sub["action_safe"].mean() * 100 if len(sub) > 0 else 0
        label = model_label[model] if model not in plotted else "_nolegend_"
        bar = ax.bar(x[ci] + local_offsets[mi], val, width,
                     color=model_color[model], alpha=0.87,
                     edgecolor="white", linewidth=0.5, label=label)
        plotted.add(model)
        # Annotate non-100% values
        if val < 99.9:
            ax.text(x[ci] + local_offsets[mi],
                    val + 0.5, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    color="#c0392b", fontweight="bold")

# 100% reference line
ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.5)

# Annotate tier2 blind spot
ax.annotate("Blocked lens\nblind spot",
            xy=(x[1], 87.5), xytext=(x[1] + 0.35, 80),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3),
            fontsize=8.5, color="#c0392b")

ax.set_xlabel("System Condition")
ax.set_ylabel("Action Safety (%)")
ax.set_title("Fig 1 — Action Safety by System Condition\n"
             "Tier 2 alone has a fatal blind spot (blocked lens) — LLMs eliminate it")
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, fontsize=9)
ax.set_ylim(70, 107)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

plt.tight_layout()
out1 = PLOT_DIR / "G1_fig1_action_safety_by_condition.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Dangerous cases heatmap: condition × scene
# ─────────────────────────────────────────────────────────────────────────────
# Aggregate across all models per condition
danger_matrix = np.zeros((len(CONDITIONS), len(SCENES)))
for i, cond in enumerate(CONDITIONS):
    for j, scene in enumerate(SCENES):
        sub = df[(df["condition"] == cond) & (df["scene_label"] == scene)]
        danger_matrix[i, j] = sub["action_dangerous"].sum()

fig, ax = plt.subplots(figsize=(12, 4))
cmap = LinearSegmentedColormap.from_list("safe_danger",
       ["#eafaf1", "#fef9e7", "#fadbd8", "#e74c3c"])
im = ax.imshow(danger_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=5)

ax.set_xticks(range(len(SCENE_LABELS)))
ax.set_xticklabels(SCENE_LABELS, fontsize=10.5)
ax.set_yticks(range(len(COND_LABELS_SHORT)))
ax.set_yticklabels(COND_LABELS_SHORT, fontsize=10)
ax.set_title("Fig 2 — Dangerous Recommendations: Condition × Scene\n"
             "(truth=hazard, action=forward — aggregated across all models)")

for i in range(len(CONDITIONS)):
    for j in range(len(SCENES)):
        val = int(danger_matrix[i, j])
        if val == 0:
            ax.text(j, i, "✓", ha="center", va="center",
                    fontsize=13, color="#27ae60", fontweight="bold")
        else:
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=13, color="white", fontweight="bold")

# Highlight the two failure zones
# blocked_lens column (index 2) for tier2_only (row 1)
ax.add_patch(plt.Rectangle((1.5, 0.5), 1, 1,
             fill=False, edgecolor="#c0392b", linewidth=2.5, zorder=10))
ax.text(2, -0.6, "Sensor\nblind spot", ha="center", fontsize=8.5,
        color="#c0392b", fontweight="bold")

# wall_close column (index 1) for full pipeline (row 3)
ax.add_patch(plt.Rectangle((0.5, 2.5), 1, 1,
             fill=False, edgecolor="#e67e22", linewidth=2.5, zorder=10))
ax.text(1, 3.6, "Depth sensor\nerror", ha="center", fontsize=8.5,
        color="#e67e22", fontweight="bold")

cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Dangerous cases", fontsize=9)
cb.set_ticks([0, 1, 2, 3, 4, 5])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
out2 = PLOT_DIR / "G1_fig2_dangerous_heatmap.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Label accuracy vs latency scatter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Points: (condition, model) pairs
points = [
    # (label, x=latency_ms, y=accuracy, color, size, marker)
    ("Tier 1.5\n(MediaPipe)",
     df[(df["condition"]=="tier1_5_only")]["total_ms"].mean(),
     df[(df["condition"]=="tier1_5_only")]["risk_correct"].mean()*100,
     "#95a5a6", 200, "s"),

    ("Tier 2\n(YOLO+Depth)",
     df[(df["condition"]=="tier2_only")]["total_ms"].mean(),
     df[(df["condition"]=="tier2_only")]["risk_correct"].mean()*100,
     "#95a5a6", 200, "D"),
]

# LLM-only conditions per model
llm_markers = {"claude":"o", "gpt4o":"s", "gpt4o_mini":"^", "gemini":"D"}
for model, label, color in zip(
        ["claude","gpt4o","gpt4o_mini","gemini"],
        ["Claude\n(LLM only)","GPT-4o\n(LLM only)","GPT-4o-mini\n(LLM only)","Gemini\n(LLM only)"],
        MODEL_COLORS[1:]):
    sub = df[(df["condition"]=="tier3_only") & (df["model"]==model)]
    if len(sub):
        points.append((label,
                       sub["total_ms"].mean(),
                       sub["risk_correct"].mean()*100,
                       color, 180, llm_markers[model]))

# Full pipeline per model
for model, label, color in zip(
        ["claude","gpt4o","gpt4o_mini","gemini"],
        ["Claude\n(Full)", "GPT-4o\n(Full)", "GPT-4o-mini\n(Full)", "Gemini\n(Full)"],
        MODEL_COLORS[1:]):
    sub = df[(df["condition"]=="tier1_5_tier2_tier3") & (df["model"]==model)]
    if len(sub):
        points.append((label,
                       sub["total_ms"].mean(),
                       sub["risk_correct"].mean()*100,
                       color, 180, llm_markers[model]))

for label, x_val, y_val, color, size, marker in points:
    ax.scatter(x_val, y_val, s=size, color=color, marker=marker,
               zorder=5, edgecolors="white", linewidth=1.2, alpha=0.9)
    # Offset labels to avoid overlap
    xytext = (8, 5)
    if "GPT-4o-mini" in label and "Full" in label:
        xytext = (8, -14)
    if "GPT-4o\n(LLM" in label:
        xytext = (8, 8)
    ax.annotate(label, (x_val, y_val),
                textcoords="offset points", xytext=xytext,
                fontsize=8, color=color)

# Latency zones
ax.axvspan(0, 100, alpha=0.04, color="#27ae60")
ax.axvspan(100, 1000, alpha=0.04, color="#f39c12")
ax.axvspan(1000, 10000, alpha=0.04, color="#e74c3c")
ax.text(14, 95, "Real-time\n(<100ms)", fontsize=8, color="#27ae60", alpha=0.8)
ax.text(110, 95, "Near-real-time", fontsize=8, color="#f39c12", alpha=0.8)
ax.text(1100, 95, "Deliberative\n(>1s)", fontsize=8, color="#e74c3c", alpha=0.8)

ax.set_xscale("log")
ax.set_xlabel("Mean Total Latency (ms, log scale)")
ax.set_ylabel("Label Accuracy (%)")
ax.set_title("Fig 3 — Label Accuracy vs Latency: Tier Tradeoff\n"
             "Tier 1.5 is fastest but bluntest; GPT-4o LLM-only is most accurate")
ax.set_xlim(8, 12000)
ax.set_ylim(40, 100)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# Legend for marker shapes
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#95a5a6",
           markersize=9, label="Rule-based"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="dimgray",
           markersize=9, label="Claude"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="dimgray",
           markersize=9, label="GPT-4o"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="dimgray",
           markersize=9, label="GPT-4o-mini"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="dimgray",
           markersize=9, label="Gemini"),
]
ax.legend(handles=legend_elements, loc="lower right",
          framealpha=0.9, fontsize=9)

plt.tight_layout()
out3 = PLOT_DIR / "G1_fig3_accuracy_vs_latency.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
