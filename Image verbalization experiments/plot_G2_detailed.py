"""
plot_G2_detailed.py — G2 Thesis Plots (Event-Driven vs Periodic LLM Calling)
==============================================================================
Three thesis-quality plots for G2:

  Fig 1 — Detection timeline: 10-frame sequence diagram
           Shows hazard zone, scheduled fires at F6, hybrid fires at F3,
           3-frame undetected exposure window highlighted
  Fig 2 — Before vs After MediaPipe upgrade: catch rate per model
           GPT-4o-mini: 40%/60% (YOLO trigger) → 100%/100% (MediaPipe trigger)
  Fig 3 — Frames late + LLM calls tradeoff
           Hybrid: 0 frames late for +1 LLM call — makes the cost argument concrete

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_G2_detailed.py"
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

VIZ_DIR  = pathlib.Path(__file__).parent
RES_DIR  = VIZ_DIR / "results"
PLOT_DIR = RES_DIR / "G2_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
new_df = pd.read_csv(RES_DIR / "G2_runs_20260524_232234.csv")   # MediaPipe trigger
old_df = pd.read_csv(RES_DIR / "G2_runs_20260521_104125.csv")   # YOLO trigger

MODELS       = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS = ["Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS = ["#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]

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
# FIG 1 — Detection timeline: 10-frame sequence diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4.5))

FRAMES = list(range(1, 11))

# Scene labels per frame
SCENE = {1:"door_open", 2:"door_open",
         3:"person_near", 4:"person_near", 5:"person_near",
         6:"person_near", 7:"person_near",
         8:"door_open", 9:"door_open", 10:"door_open"}
SAFE_FRAMES   = [f for f in FRAMES if SCENE[f] == "door_open"]
HAZARD_FRAMES = [f for f in FRAMES if SCENE[f] == "person_near"]

# Draw frame boxes
for f in FRAMES:
    is_hazard = SCENE[f] == "person_near"
    color = "#fadbd8" if is_hazard else "#eafaf1"
    ax.add_patch(mpatches.FancyBboxPatch(
        (f - 0.45, 0.1), 0.9, 0.8,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#bdc3c7", linewidth=1.2
    ))
    ax.text(f, 0.5, f"F{f}", ha="center", va="center",
            fontsize=10, fontweight="bold",
            color="#c0392b" if is_hazard else "#27ae60")

# Scene label row below
ax.text(1.5, -0.05, "Safe (door open)", ha="center", va="top",
        fontsize=9, color="#27ae60")
ax.text(5.0, -0.05, "HAZARD — person enters", ha="center", va="top",
        fontsize=9, color="#c0392b", fontweight="bold")
ax.text(9.0, -0.05, "Safe (door open)", ha="center", va="top",
        fontsize=9, color="#27ae60")

# Scheduled call markers (F1, F6)
for f, label in [(1, "Scheduled\ncall"), (6, "Scheduled\ncall")]:
    ax.annotate("", xy=(f, 1.0), xytext=(f, 1.35),
                arrowprops=dict(arrowstyle="-|>", color="#4a90d9", lw=2))
    ax.text(f, 1.42, label, ha="center", va="bottom",
            fontsize=8.5, color="#4a90d9", fontweight="bold")

# Hybrid trigger at F3
ax.annotate("", xy=(3, 1.0), xytext=(3, 1.65),
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.5))
ax.text(3, 1.72, "Hybrid trigger\n(MediaPipe fires)", ha="center", va="bottom",
        fontsize=8.5, color="#e74c3c", fontweight="bold")

# 3-frame exposure window (F3 to F6)
ax.add_patch(mpatches.FancyArrowPatch(
    (3.05, 0.93), (5.95, 0.93),
    arrowstyle="<->", color="#e67e22", lw=2,
    mutation_scale=12
))
ax.text(4.5, 0.97, "3 frames undetected\n= 100ms = 3–5cm travel",
        ha="center", va="bottom", fontsize=8.5,
        color="#e67e22", fontweight="bold")

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#eafaf1", edgecolor="#bdc3c7", label="Safe frame"),
    mpatches.Patch(facecolor="#fadbd8", edgecolor="#bdc3c7", label="Hazard frame"),
    Line2D([0],[0], color="#4a90d9", lw=2, label="Scheduled call (F1, F6)"),
    Line2D([0],[0], color="#e74c3c", lw=2.5, label="Hybrid trigger (F3)"),
]
ax.legend(handles=legend_elements, loc="upper right",
          framealpha=0.9, fontsize=9)

ax.set_xlim(0.3, 10.7)
ax.set_ylim(-0.25, 2.1)
ax.axis("off")
ax.set_title("Fig 1 — Detection Timeline: Scheduled vs Hybrid Strategy\n"
             "Hybrid eliminates 3-frame (100ms) exposure window at cost of +1 LLM call",
             fontsize=13, pad=10)

plt.tight_layout()
out1 = PLOT_DIR / "G2_fig1_detection_timeline.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Before vs After MediaPipe upgrade: catch rate per model × strategy
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle("Fig 2 — Catch Rate: YOLO Trigger (old) vs MediaPipe Trigger (new)\n"
             "MediaPipe upgrade fixes GPT-4o-mini — closes kneeling person blind spot",
             fontsize=13)

for ax, (df, run_label, color_bar) in zip(
        axes,
        [(old_df, "YOLO Trigger (Old Run)", "#e0e0e0"),
         (new_df, "MediaPipe Trigger (New Run)", "#4a90d9")]):

    x     = np.arange(len(MODELS))
    width = 0.32
    strats = ["scheduled", "hybrid"]
    s_colors = ["#95a5a6", "#27ae60"]
    s_labels = ["Scheduled", "Hybrid"]

    for si, (strat, sc, sl) in enumerate(zip(strats, s_colors, s_labels)):
        catch = [df[(df["model"] == m) & (df["strategy"] == strat)]["hazard_caught"].mean() * 100
                 for m in MODELS]
        offset = (si - 0.5) * width
        bars = ax.bar(x + offset, catch, width,
                      label=sl, color=sc, alpha=0.87,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, catch):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#c0392b" if val < 99 else "dimgray")

    ax.set_title(run_label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylim(0, 120)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(framealpha=0.9, fontsize=9.5)

axes[0].set_ylabel("Hazard Catch Rate (%)")

# Highlight GPT-4o-mini improvement with arrow between panels
fig.text(0.5, 0.18,
         "GPT-4o-mini: 40% → 100% (scheduled)  ·  60% → 100% (hybrid)",
         ha="center", fontsize=10, color="#c0392b", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7",
                   edgecolor="#f39c12"))

plt.tight_layout(rect=[0, 0.08, 1, 1])
out2 = PLOT_DIR / "G2_fig2_before_after_mediapipe.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Frames late + LLM calls tradeoff (new run only)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fig 3 — Cost of Hybrid: +1 LLM Call Eliminates 3-Frame Detection Lag",
             fontsize=13)

x     = np.arange(len(MODELS))
width = 0.32
strats       = ["scheduled", "hybrid"]
strat_labels = ["Scheduled", "Hybrid"]
strat_colors = ["#95a5a6", "#27ae60"]

# Left panel: frames late
for si, (strat, sc, sl) in enumerate(zip(strats, strat_colors, strat_labels)):
    vals = [new_df[(new_df["model"] == m) & (new_df["strategy"] == strat)]["frames_late"].mean()
            for m in MODELS]
    offset = (si - 0.5) * width
    bars = ax1.bar(x + offset, vals, width, label=sl,
                   color=sc, alpha=0.87, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{val:.0f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

ax1.set_title("Frames Late to Detect Hazard", fontsize=11)
ax1.set_xlabel("Model")
ax1.set_ylabel("Mean Frames Late")
ax1.set_xticks(x)
ax1.set_xticklabels(MODEL_LABELS, fontsize=10)
ax1.set_ylim(0, 5)
ax1.legend(framealpha=0.9, fontsize=10)
ax1.text(0.5, -0.18,
         "3 frames = 100ms at 30fps = 3–5cm travel at 0.3–0.5 m/s",
         ha="center", transform=ax1.transAxes,
         fontsize=8.5, color="#e67e22", style="italic")

# Right panel: LLM calls per sequence
for si, (strat, sc, sl) in enumerate(zip(strats, strat_colors, strat_labels)):
    vals = [new_df[(new_df["model"] == m) & (new_df["strategy"] == strat)]["llm_calls"].mean()
            for m in MODELS]
    offset = (si - 0.5) * width
    bars = ax2.bar(x + offset, vals, width, label=sl,
                   color=sc, alpha=0.87, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{val:.0f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

ax2.set_title("LLM Calls per Sequence", fontsize=11)
ax2.set_xlabel("Model")
ax2.set_ylabel("Mean LLM Calls")
ax2.set_xticks(x)
ax2.set_xticklabels(MODEL_LABELS, fontsize=10)
ax2.set_ylim(0, 5)
ax2.legend(framealpha=0.9, fontsize=10)
ax2.text(0.5, -0.18,
         "+1 LLM call per hazard event — Gemini cost: +$0.00012 per trigger",
         ha="center", transform=ax2.transAxes,
         fontsize=8.5, color="dimgray", style="italic")

plt.tight_layout()
out3 = PLOT_DIR / "G2_fig3_frames_late_vs_llm_calls.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
