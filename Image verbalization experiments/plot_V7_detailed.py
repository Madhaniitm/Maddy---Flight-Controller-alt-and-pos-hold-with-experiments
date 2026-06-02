"""
plot_V7_detailed.py — V7 Thesis Plots (Scene Context History)
=============================================================
Three thesis-quality plots for V7:

  Fig 1 — Risk accuracy by history mode × model (grouped bar)
           Main finding: short wins; GPT-4o-mini stateless collapses to 4%
  Fig 2 — Per-frame risk accuracy across 5-frame sequence (line plot)
           Shows spike at frame 3 (person enters) and drop at frame 5 (leaves)
  Fig 3 — Hazard onset vs clearance detection per model (grouped bar)
           Asymmetry: 100% onset detection vs 13–47% clearance detection

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_V7_detailed.py"
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ_DIR  = pathlib.Path(__file__).parent
RES_DIR  = VIZ_DIR / "results"
PLOT_DIR = RES_DIR / "V7_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RES_DIR / "V7_runs_20260525_221631.csv")
df = df[df["error"].isna() | (df["error"] == "")].copy()

MODELS        = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS  = ["Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS  = ["#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]
MODEL_MARKERS = ["o", "s", "^", "D"]

MODES       = ["stateless", "short", "full"]
MODE_LABELS = ["Stateless", "Short (2 frames)", "Full (all frames)"]
MODE_COLORS = ["#95a5a6", "#2ecc71", "#e74c3c"]

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
# FIG 1 — Risk accuracy by history mode × model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

x       = np.arange(len(MODES))
width   = 0.18
offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODELS))

for i, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLORS)):
    means = []
    for mode in MODES:
        sub = df[(df["model"] == model) & (df["history_mode"] == mode)]
        means.append(sub["risk_correct"].mean() * 100 if len(sub) > 0 else 0)
    bars = ax.bar(x + offsets[i], means, width,
                  label=label, color=color, alpha=0.87,
                  edgecolor="white", linewidth=0.5)
    # Annotate GPT-4o-mini stateless (dramatic 4% result)
    if model == "gpt4o_mini":
        bar = bars[0]  # stateless bar
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                "4%!", ha="center", fontsize=8.5,
                color="#c0392b", fontweight="bold")

# Short history reference line (overall best)
short_overall = df[df["history_mode"] == "short"]["risk_correct"].mean() * 100
ax.axhline(short_overall, color="#27ae60", linestyle="--",
           linewidth=1.2, alpha=0.6,
           label=f"Short overall ({short_overall:.1f}%)")

ax.set_xlabel("History Mode")
ax.set_ylabel("Risk Accuracy (%)")
ax.set_title("Fig 1 — Risk Accuracy by History Mode and Model\n"
             "Short history (last 2 frames) maximises accuracy across the model portfolio")
ax.set_xticks(x)
ax.set_xticklabels(MODE_LABELS)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

plt.tight_layout()
out1 = PLOT_DIR / "V7_fig1_risk_acc_by_mode_model.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Per-frame risk accuracy across 5-frame sequence
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

FRAMES       = [1, 2, 3, 4, 5]
FRAME_LABELS = ["F1\ndoor_open\n(safe)", "F2\ndoor_open\n(safe)",
                "F3\nperson_near\n(hazard)", "F4\nperson_near\n(hazard)",
                "F5\ndoor_open\n(safe)"]

for model, label, color, marker in zip(MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS):
    accs = []
    for f in FRAMES:
        sub = df[(df["model"] == model) & (df["frame_num"] == f)]
        accs.append(sub["risk_correct"].mean() * 100 if len(sub) > 0 else 0)
    ax.plot(FRAMES, accs, color=color, marker=marker,
            linewidth=2.2, markersize=9, label=label)

# Shade change events
ax.axvspan(2.5, 3.5, alpha=0.08, color="#e74c3c", label="Change: person enters")
ax.axvspan(4.5, 5.5, alpha=0.08, color="#3498db", label="Change: person leaves")

# Annotations
ax.text(3, 96, "Person enters\n100% detected", ha="center",
        fontsize=9, color="#c0392b", fontweight="bold")
ax.text(5, 10, "Person leaves\n26.7% detected", ha="center",
        fontsize=9, color="#2980b9", fontweight="bold")

ax.set_xlabel("Frame in Sequence")
ax.set_ylabel("Risk Accuracy (%)")
ax.set_title("Fig 2 — Per-Frame Risk Accuracy Across 5-Frame Sequence\n"
             "Hazard onset (F3) detected 100% · Hazard clearance (F5) detected 26.7%")
ax.set_xticks(FRAMES)
ax.set_xticklabels(FRAME_LABELS, fontsize=9)
ax.set_ylim(-5, 110)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(loc="center left", framealpha=0.9, fontsize=9.5)

plt.tight_layout()
out2 = PLOT_DIR / "V7_fig2_per_frame_accuracy.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Hazard onset vs clearance detection per model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

x     = np.arange(len(MODELS))
width = 0.32

# Frame 3 = hazard onset (safe→hazard change)
# Frame 5 = hazard clearance (hazard→safe change)
onset     = []
clearance = []
for model in MODELS:
    f3 = df[(df["model"] == model) & (df["frame_num"] == 3)]
    f5 = df[(df["model"] == model) & (df["frame_num"] == 5)]
    onset.append(f3["risk_correct"].mean() * 100 if len(f3) > 0 else 0)
    clearance.append(f5["risk_correct"].mean() * 100 if len(f5) > 0 else 0)

b1 = ax.bar(x - width/2, onset,     width,
            label="Hazard Onset — F3 (person enters)",
            color="#e74c3c", alpha=0.87, edgecolor="white")
b2 = ax.bar(x + width/2, clearance, width,
            label="Hazard Clearance — F5 (person leaves)",
            color="#3498db", alpha=0.87, edgecolor="white")

# Value labels
for bar, val in zip(b1, onset):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#c0392b")

for bar, val in zip(b2, clearance):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#2980b9")

# Safety note
ax.text(0.5, 0.04,
        "✓  Fail-safe asymmetry: system never misses a hazard appearing,\n"
        "    may over-linger in HOVER after hazard clears — acceptable for safety",
        transform=ax.transAxes, ha="center", fontsize=9,
        color="dimgray", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#dee2e6"))

ax.set_xlabel("Model")
ax.set_ylabel("Detection Rate (%)")
ax.set_title("Fig 3 — Asymmetric Change Detection: Hazard Onset vs Clearance\n"
             "All models detect person entering (100%) · Few detect person leaving")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS)
ax.set_ylim(0, 120)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
out3 = PLOT_DIR / "V7_fig3_onset_vs_clearance.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
