"""
plot_V8_detailed.py — V8 Thesis Plots (Temperature Sweep)
==========================================================
Three thesis-quality plots for V8:

  Fig 1 — Risk accuracy vs temperature per model (line plot)
           Shows t=0.0 optimal; GPT-4o-mini peak at t=0.0; Gemini flat
  Fig 2 — Label flip rate vs temperature per model (line plot)
           Consistency argument: flip rate climbs with temperature
  Fig 3 — Dangerous cases heatmap: model × temperature
           All 5 dangerous cases cluster at GPT-4o-mini × t≥0.5

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_V8_detailed.py"
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
PLOT_DIR = RES_DIR / "V8_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df  = pd.read_csv(RES_DIR / "V8_runs_20260526_121412.csv")
df  = df[df["error"].isna() | (df["error"] == "")].copy()

# Derive action_dangerous: truth=hazard AND action=PITCH_FORWARD
df["action_dangerous"] = (
    (df["truth"] == "hazard") & (df["detected_risk"] == "safe")
).astype(int)

MODELS        = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS  = ["Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS  = ["#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]
MODEL_MARKERS = ["o", "s", "^", "D"]

TEMPS       = [0.0, 0.2, 0.5, 0.8, 1.0]
TEMP_LABELS = ["0.0\n(deterministic)", "0.2", "0.5", "0.8", "1.0\n(creative)"]

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
# FIG 1 — Risk accuracy vs temperature per model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for model, label, color, marker in zip(MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS):
    accs = []
    sems = []
    for t in TEMPS:
        sub = df[(df["model"] == model) & (df["temperature"] == t)]
        accs.append(sub["s3_risk"].mean() * 100 if len(sub) > 0 else 0)
        sems.append(sub["s3_risk"].sem() * 100  if len(sub) > 0 else 0)
    ax.plot(TEMPS, accs, color=color, marker=marker,
            linewidth=2.2, markersize=9, label=label)
    ax.fill_between(TEMPS,
                    [a - s for a, s in zip(accs, sems)],
                    [a + s for a, s in zip(accs, sems)],
                    color=color, alpha=0.1)

# Annotate GPT-4o-mini peak at t=0.0
mini_t0 = df[(df["model"] == "gpt4o_mini") & (df["temperature"] == 0.0)]["s3_risk"].mean() * 100
ax.annotate(f"GPT-4o-mini\n{mini_t0:.0f}% at t=0.0",
            xy=(0.0, mini_t0),
            xytext=(0.15, mini_t0 - 12),
            arrowprops=dict(arrowstyle="->", color="#6cbf6c", lw=1.3),
            fontsize=8.5, color="#6cbf6c")

# Annotate Claude anomaly at t=0.8
claude_t08 = df[(df["model"] == "claude") & (df["temperature"] == 0.8)]["s3_risk"].mean() * 100
ax.annotate(f"Claude anomaly\n{claude_t08:.0f}% at t=0.8",
            xy=(0.8, claude_t08),
            xytext=(0.6, claude_t08 + 8),
            arrowprops=dict(arrowstyle="->", color="#e07b54", lw=1.3),
            fontsize=8.5, color="#e07b54")

# Shade unsafe zone
ax.axvspan(0.45, 1.05, alpha=0.05, color="#e74c3c")
ax.text(0.72, 20, "Dangerous cases\nappear here (t≥0.5)",
        ha="center", fontsize=8.5, color="#c0392b", style="italic")

ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Temperature")
ax.set_ylabel("Risk Accuracy (%)")
ax.set_title("Fig 1 — Risk Accuracy vs Temperature per Model\n"
             "t=0.0 is optimal overall; dangerous cases appear at t≥0.5")
ax.set_xticks(TEMPS)
ax.set_xticklabels(TEMP_LABELS)
ax.set_ylim(15, 95)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
out1 = PLOT_DIR / "V8_fig1_accuracy_vs_temperature.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Label flip rate vs temperature per model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for model, label, color, marker in zip(MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS):
    flips = []
    for t in TEMPS:
        sub = df[(df["model"] == model) & (df["temperature"] == t)]
        if len(sub) < 2:
            flips.append(0)
            continue
        # Flip rate: fraction of scene pairs where label differs across runs
        flip_count = 0
        total_pairs = 0
        for scene in sub["scene_label"].unique():
            labels = sub[sub["scene_label"] == scene]["detected_risk"].dropna().tolist()
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    total_pairs += 1
                    if labels[i] != labels[j]:
                        flip_count += 1
        flips.append((flip_count / total_pairs * 100) if total_pairs > 0 else 0)

    ax.plot(TEMPS, flips, color=color, marker=marker,
            linewidth=2.2, markersize=9, label=label)

# Annotate Gemini 0% at t=0.0
ax.annotate("Gemini: 0%\nat t=0.0",
            xy=(0.0, 0), xytext=(0.12, 8),
            arrowprops=dict(arrowstyle="->", color="#b07ed9", lw=1.3),
            fontsize=8.5, color="#b07ed9")

# Annotate Claude high even at t=0.0
claude_flip_t0 = None
sub = df[(df["model"] == "claude") & (df["temperature"] == 0.0)]
flip_count = 0; total_pairs = 0
for scene in sub["scene_label"].unique():
    labels = sub[sub["scene_label"] == scene]["detected_risk"].dropna().tolist()
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            total_pairs += 1
            if labels[i] != labels[j]:
                flip_count += 1
claude_flip_t0 = (flip_count / total_pairs * 100) if total_pairs > 0 else 0
ax.annotate(f"Claude: {claude_flip_t0:.0f}%\neven at t=0.0\n(non-deterministic API)",
            xy=(0.0, claude_flip_t0), xytext=(0.15, claude_flip_t0 + 6),
            arrowprops=dict(arrowstyle="->", color="#e07b54", lw=1.3),
            fontsize=8.5, color="#e07b54")

ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Temperature")
ax.set_ylabel("Label Flip Rate (%)")
ax.set_title("Fig 2 — Label Flip Rate vs Temperature per Model\n"
             "Lower = more consistent decisions on identical inputs · t=0.0 minimises flipping")
ax.set_xticks(TEMPS)
ax.set_xticklabels(TEMP_LABELS)
ax.set_ylim(-3, 65)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
out2 = PLOT_DIR / "V8_fig2_flip_rate_vs_temperature.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Dangerous cases heatmap: model × temperature
# ─────────────────────────────────────────────────────────────────────────────
# Dangerous = truth is hazard AND model classifies as safe (PITCH_FORWARD)
danger_matrix = np.zeros((len(MODELS), len(TEMPS)))
for i, model in enumerate(MODELS):
    for j, t in enumerate(TEMPS):
        sub = df[(df["model"] == model) & (df["temperature"] == t)]
        hazard_sub = sub[sub["truth"] == "hazard"]
        danger_matrix[i, j] = (hazard_sub["detected_risk"] == "safe").sum()

fig, ax = plt.subplots(figsize=(10, 4))
cmap = LinearSegmentedColormap.from_list("safe_danger", ["#f0f9f0", "#fff3cd", "#e74c3c"])
im = ax.imshow(danger_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=3)

ax.set_xticks(range(len(TEMPS)))
ax.set_xticklabels(TEMP_LABELS, fontsize=10.5)
ax.set_yticks(range(len(MODEL_LABELS)))
ax.set_yticklabels(MODEL_LABELS, fontsize=11)
ax.set_title("Fig 3 — Dangerous Cases: Model × Temperature\n"
             "(truth=hazard, model classified as safe → PITCH_FORWARD into hazard)")

for i in range(len(MODELS)):
    for j in range(len(TEMPS)):
        val = int(danger_matrix[i, j])
        if val == 0:
            ax.text(j, i, "✓", ha="center", va="center",
                    fontsize=14, color="#27ae60", fontweight="bold")
        else:
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=14, color="white", fontweight="bold")

# Highlight danger zone
for j, t in enumerate(TEMPS):
    if t >= 0.5:
        ax.add_patch(plt.Rectangle(
            (j - 0.5, -0.5), 1, len(MODELS),
            fill=False, edgecolor="#c0392b",
            linewidth=1.5, linestyle="--", zorder=10
        ))

ax.text(3.0, len(MODELS) - 0.1,
        "Unsafe zone (t≥0.5)",
        ha="center", va="bottom", fontsize=9,
        color="#c0392b", style="italic")

cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Dangerous recommendations", fontsize=9)
cb.set_ticks([0, 1, 2, 3])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
out3 = PLOT_DIR / "V8_fig3_dangerous_cases_heatmap.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
