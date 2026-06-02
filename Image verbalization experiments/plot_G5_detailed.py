"""
plot_G5_detailed.py — G5 Thesis Plots (Real Vision Pipeline)
=============================================================
Three thesis-quality plots for G5:

  Fig 1 — Label accuracy vs action safety per model (paired bar)
           Shows why label accuracy is wrong: GPT-4o-mini 80%/100% vs GPT-4o 52%/92%
  Fig 2 — Before vs after sensor fix (wall-fill proximity warning)
           Dangerous cases: 10 → 3 (-70%); Claude: 13% → 0%
  Fig 3 — Per-scene action safety heatmap: scene × model
           7/8 scenes 100% safe everywhere; wall_close fails only GPT-4o

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_G5_detailed.py"
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
PLOT_DIR = RES_DIR / "G5_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
after  = pd.read_csv(RES_DIR / "G5_runs_20260525_011427.csv")   # sensor fix applied
before = pd.read_csv(RES_DIR / "G5_runs_20260525_000536.csv")   # before fix

for df in (after, before):
    df.drop(df[df["error"].notna() & (df["error"] != "")].index, inplace=True)
    # action_dangerous: truth=hazard AND detected=safe → PITCH_FORWARD
    df["action_dangerous"] = ((df["truth"] == "hazard") &
                               (df["detected_risk"] == "safe")).astype(int)
    # action_safe: not dangerous
    df["action_safe"] = (1 - df["action_dangerous"])

MODELS       = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS = ["Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS = ["#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]

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
# FIG 1 — Label accuracy vs action safety per model (paired bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

x     = np.arange(len(MODELS))
width = 0.32

label_acc   = [after[after["model"] == m]["risk_correct"].mean() * 100 for m in MODELS]
action_safe = [after[after["model"] == m]["action_safe"].mean()  * 100 for m in MODELS]

b1 = ax.bar(x - width/2, label_acc,   width,
            label="Label Accuracy\n(detected risk = ground truth)",
            color="#bdc3c7", alpha=0.87, edgecolor="white")
b2 = ax.bar(x + width/2, action_safe, width,
            label="Action Safety\n(pilot action is safe for actual scene)",
            color=MODEL_COLORS, alpha=0.87, edgecolor="white")

# Value labels
for bar, val in zip(b1, label_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9.5,
            color="dimgray")

for bar, val, color in zip(b2, action_safe, MODEL_COLORS):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9.5,
            color=color, fontweight="bold")

# Annotate the gap for GPT-4o-mini (biggest story)
mini_idx = MODELS.index("gpt4o_mini")
ax.annotate("Best label acc\n+ 100% safe",
            xy=(x[mini_idx] + width/2, action_safe[mini_idx]),
            xytext=(x[mini_idx] + 0.6, 93),
            arrowprops=dict(arrowstyle="->", color="#6cbf6c", lw=1.3),
            fontsize=8.5, color="#6cbf6c", fontweight="bold")

# Annotate GPT-4o gap
gpt4o_idx = MODELS.index("gpt4o")
ax.annotate("Low label acc\nbut 3 dangerous\nactions",
            xy=(x[gpt4o_idx] + width/2, action_safe[gpt4o_idx]),
            xytext=(x[gpt4o_idx] + 0.55, 80),
            arrowprops=dict(arrowstyle="->", color="#4a90d9", lw=1.3),
            fontsize=8.5, color="#4a90d9")

ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.4)
ax.set_xlabel("Model")
ax.set_ylabel("Score (%)")
ax.set_title("Fig 1 — Label Accuracy vs Action Safety per Model\n"
             "Label accuracy penalises over-caution equally as genuine failure — "
             "action safety is the correct metric")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_ylim(40, 112)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

plt.tight_layout()
out1 = PLOT_DIR / "G5_fig1_label_acc_vs_action_safety.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Before vs after sensor fix
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 2 — Before vs After Sensor Fix (Wall-Fill Proximity Warning)\n"
             "Adding 'depth unreliable' metadata: dangerous cases 10 → 3 (−70%)",
             fontsize=13)

# Left: dangerous cases per model
ax = axes[0]
x     = np.arange(len(MODELS))
width = 0.32

before_danger = [before[before["model"] == m]["action_dangerous"].sum() for m in MODELS]
after_danger  = [after[after["model"]  == m]["action_dangerous"].sum() for m in MODELS]

b1 = ax.bar(x - width/2, before_danger, width,
            label="Before fix", color="#e74c3c", alpha=0.87, edgecolor="white")
b2 = ax.bar(x + width/2, after_danger,  width,
            label="After fix",  color="#27ae60", alpha=0.87, edgecolor="white")

for bar, val in zip(b1, before_danger):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(val)), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#c0392b")

for bar, val in zip(b2, after_danger):
    label = str(int(val)) if val > 0 else "✓"
    color = "#c0392b" if val > 0 else "#27ae60"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            label, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color)

ax.set_title("Dangerous Actions per Model", fontsize=11)
ax.set_xlabel("Model")
ax.set_ylabel("Count of Dangerous Recommendations")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=10)
ax.set_ylim(0, 13)
ax.legend(framealpha=0.9, fontsize=10)

# Right: action safety %
ax = axes[1]
before_safe = [before[before["model"] == m]["action_safe"].mean() * 100 for m in MODELS]
after_safe  = [after[after["model"]  == m]["action_safe"].mean() * 100 for m in MODELS]

b1 = ax.bar(x - width/2, before_safe, width,
            label="Before fix", color="#e74c3c", alpha=0.87, edgecolor="white")
b2 = ax.bar(x + width/2, after_safe,  width,
            label="After fix",  color="#27ae60", alpha=0.87, edgecolor="white")

for bar, val in zip(b1, before_safe):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9.5,
            color="#c0392b", fontweight="bold")

for bar, val in zip(b2, after_safe):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=9.5,
            color="#27ae60", fontweight="bold")

ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.4)
ax.set_title("Action Safety %", fontsize=11)
ax.set_xlabel("Model")
ax.set_ylabel("Action Safety (%)")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=10)
ax.set_ylim(75, 108)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
out2 = PLOT_DIR / "G5_fig2_before_after_sensor_fix.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Per-scene action safety heatmap: scene × model
# ─────────────────────────────────────────────────────────────────────────────
safety_matrix = np.zeros((len(SCENES), len(MODELS)))
danger_matrix = np.zeros((len(SCENES), len(MODELS)))

for i, scene in enumerate(SCENES):
    for j, model in enumerate(MODELS):
        sub = after[(after["scene_label"] == scene) & (after["model"] == model)]
        if len(sub) > 0:
            safety_matrix[i, j] = sub["action_safe"].mean() * 100
            danger_matrix[i, j] = sub["action_dangerous"].sum()

fig, ax = plt.subplots(figsize=(10, 6))
cmap = LinearSegmentedColormap.from_list("safe_danger",
       ["#e74c3c", "#f39c12", "#eafaf1", "#27ae60"])
im = ax.imshow(safety_matrix, aspect="auto", cmap=cmap, vmin=60, vmax=100)

ax.set_xticks(range(len(MODEL_LABELS)))
ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_yticks(range(len(SCENE_LABELS)))
ax.set_yticklabels(SCENE_LABELS, fontsize=10)
ax.set_title("Fig 3 — Action Safety per Scene × Model (%)\n"
             "7 of 8 scenes: 100% safe across all models · wall_close fails only GPT-4o")

for i in range(len(SCENES)):
    for j in range(len(MODELS)):
        val = safety_matrix[i, j]
        d   = int(danger_matrix[i, j])
        if val >= 99.9:
            ax.text(j, i, "✓", ha="center", va="center",
                    fontsize=14, color="#27ae60", fontweight="bold")
        else:
            txt = f"{val:.0f}%\n({d} ✗)" if d > 0 else f"{val:.0f}%"
            color = "white" if val < 85 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9.5, color=color, fontweight="bold")

# Highlight wall_close row
wall_idx = SCENES.index("wall_close")
ax.add_patch(plt.Rectangle(
    (-0.5, wall_idx - 0.5), len(MODELS), 1,
    fill=False, edgecolor="#c0392b", linewidth=2.5, zorder=10
))
ax.text(len(MODELS) - 0.4, wall_idx,
        "← DA v2 depth\nerror (20cm→2m)",
        va="center", fontsize=8.5, color="#c0392b")

# Over-caution note for person_far
pf_idx = SCENES.index("person_far")
ax.text(len(MODELS) - 0.4, pf_idx,
        "← 89% over-hazard\nbut all HOVER ✓",
        va="center", fontsize=8, color="#27ae60", style="italic")

cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.12)
cb.set_label("Action Safety (%)", fontsize=9)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
out3 = PLOT_DIR / "G5_fig3_per_scene_safety_heatmap.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
