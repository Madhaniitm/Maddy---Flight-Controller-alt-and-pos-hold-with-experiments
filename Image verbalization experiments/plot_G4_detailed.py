"""
plot_G4_detailed.py — G4 Thesis Plots (Three-Tier Timescale Validation)
========================================================================
Three thesis-quality plots for G4:

  Fig 1 — Latency cascade: all tiers on log scale
           4 orders of magnitude from PID (0.001ms) to Claude (7,218ms)
  Fig 2 — LLM latency comparison with 95% CI
           Claude 3× slower than Gemini — directly informs model selection
  Fig 3 — Per-run consistency line plot
           Shows YOLO warm-up spike at run 1, stable production latency runs 2–5

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_G4_detailed.py"
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VIZ_DIR  = pathlib.Path(__file__).parent
RES_DIR  = VIZ_DIR / "results"
PLOT_DIR = RES_DIR / "G4_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
runs  = pd.read_csv(RES_DIR / "G4_runs_20260524_234054.csv")
calls = pd.read_csv(RES_DIR / "G4_calls_20260524_234054.csv")
calls = calls[calls["error"].isna() | (calls["error"] == "")].copy()

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

# ── Compute summary stats from runs ──────────────────────────────────────────
pid_mean    = runs["pid_mean_ms"].mean()
local_mean  = runs["local_mean_ms"].mean()
yolo_mean   = runs["yolo_mean_ms"].mean()
yolo_stable = runs[runs["run"] > 1]["yolo_mean_ms"].mean()  # exclude warm-up

llm_means = {m: calls[calls["model"] == m]["latency_ms"].mean() for m in MODELS}
llm_cis   = {m: 1.96 * calls[calls["model"] == m]["latency_ms"].sem() for m in MODELS}

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Latency cascade: all tiers on log scale
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Tiers in order
tier_labels  = ["Tier 1\nPID\nController",
                "Tier 1.5\nMediaPipe\nEfficientDet",
                "Tier 2\nYOLO-World\n+ Depth",
                "Tier 3\nGemini",
                "Tier 3\nGPT-4o",
                "Tier 3\nGPT-4o-mini",
                "Tier 3\nClaude"]
tier_values  = [pid_mean, local_mean, yolo_stable,
                llm_means["gemini"], llm_means["gpt4o"],
                llm_means["gpt4o_mini"], llm_means["claude"]]
tier_colors  = ["#2ecc71", "#27ae60", "#f39c12",
                "#b07ed9", "#4a90d9", "#6cbf6c", "#e07b54"]
tier_freqs   = ["~4,000×/s", "~60×/s", "~3–4×/s",
                "~0.4×/s", "~0.4×/s", "~0.2×/s", "~0.1×/s"]

x = np.arange(len(tier_labels))
bars = ax.bar(x, tier_values, color=tier_colors, alpha=0.87,
              edgecolor="white", linewidth=0.8, width=0.65)

# Value labels on top of each bar
for bar, val, freq in zip(bars, tier_values, tier_freqs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.35,
            f"{val:.3f}ms" if val < 1 else f"{val:.0f}ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.4,
            freq, ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

# Tier separators
ax.axvline(1.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)
ax.axvline(2.5, color="gray", linestyle="--", linewidth=1, alpha=0.4)

# Order of magnitude annotations
for i in range(len(tier_values) - 1):
    ratio = tier_values[i+1] / tier_values[i]
    if ratio > 5:
        mid_x = (x[i] + x[i+1]) / 2
        ax.text(mid_x, tier_values[i+1] * 3,
                f"×{ratio:.0f}", ha="center", fontsize=9,
                color="gray", style="italic")

ax.set_yscale("log")
ax.set_ylabel("Latency (ms, log scale)")
ax.set_title("Fig 1 — Four-Tier Architecture: Latency Cascade\n"
             "4 orders of magnitude from PID (0.001ms) to LLM (2,333–7,218ms) — "
             "separation is real, not assumed")
ax.set_xticks(x)
ax.set_xticklabels(tier_labels, fontsize=9.5)
ax.set_ylim(0.0005, 30000)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.3f}" if v < 1 else f"{v:.0f}ms"))

# Tier labels at top
for tx, tlabel, tc in [(0.5, "Tier 1\n(Reactive)", "#2ecc71"),
                        (2.0, "Tier 1.5 + 2\n(Perceptive)", "#f39c12"),
                        (5.0, "Tier 3\n(Cognitive)", "#e07b54")]:
    ax.text(tx, 15000, tlabel, ha="center", fontsize=9,
            color=tc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=tc, alpha=0.8))

plt.tight_layout()
out1 = PLOT_DIR / "G4_fig1_latency_cascade.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — LLM latency comparison with 95% CI
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

x     = np.arange(len(MODELS))
means = [llm_means[m] for m in MODELS]
cis   = [llm_cis[m]   for m in MODELS]

bars = ax.bar(x, means, color=MODEL_COLORS, alpha=0.87,
              edgecolor="white", linewidth=0.5, width=0.5,
              yerr=cis, capsize=6,
              error_kw=dict(elinewidth=1.5, ecolor="dimgray", capthick=1.5))

# Value + CI labels
for bar, val, ci in zip(bars, means, cis):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + ci + 80,
            f"{val:.0f}ms\n±{ci:.0f}ms",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# Claude vs Gemini ratio annotation
ratio = llm_means["claude"] / llm_means["gemini"]
ax.annotate("",
            xy=(x[0], llm_means["claude"]),
            xytext=(x[3], llm_means["gemini"]),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
ax.text((x[0] + x[3]) / 2, (llm_means["claude"] + llm_means["gemini"]) / 2 + 200,
        f"Claude is {ratio:.1f}× slower than Gemini",
        ha="center", fontsize=9.5, color="gray", style="italic")

# Practical threshold line
ax.axhline(2000, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(3.3, 2100, "2s threshold\n(practical indoor drone)", fontsize=8.5,
        color="#e67e22", ha="right")

ax.set_ylabel("Mean Latency (ms)")
ax.set_title("Fig 2 — Tier 3 LLM Latency Comparison (95% CI)\n"
             "Claude is 3× slower than Gemini — latency directly informs model selection")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_ylim(0, 10000)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v/1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"))

plt.tight_layout()
out2 = PLOT_DIR / "G4_fig2_llm_latency_comparison.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Per-run consistency line plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fig 3 — Per-Run Latency Consistency (5 Runs)\n"
             "YOLO warm-up spike at run 1 (319ms → 263ms stable); all other tiers stable",
             fontsize=13)

RUN_NOS = runs["run"].tolist()

# Left panel: Tier 1.5 + Tier 2 (ms scale)
ax = axes[0]
ax.plot(RUN_NOS, runs["local_mean_ms"], color="#27ae60", marker="o",
        linewidth=2, markersize=8, label="Tier 1.5 — MediaPipe")
ax.plot(RUN_NOS, runs["yolo_mean_ms"], color="#f39c12", marker="s",
        linewidth=2, markersize=8, label="Tier 2 — YOLO+Depth")

# Annotate warm-up
ax.annotate("Warm-up\n(model load)",
            xy=(1, runs[runs["run"]==1]["yolo_mean_ms"].values[0]),
            xytext=(1.4, 305),
            arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.3),
            fontsize=8.5, color="#f39c12")
ax.annotate("Stable\n(cached)",
            xy=(2, runs[runs["run"]==2]["yolo_mean_ms"].values[0]),
            xytext=(2.3, 235),
            arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.3),
            fontsize=8.5, color="#f39c12")

ax.set_title("Tier 1.5 and Tier 2", fontsize=11)
ax.set_xlabel("Run Number")
ax.set_ylabel("Latency (ms)")
ax.set_xticks(RUN_NOS)
ax.set_ylim(0, 380)
ax.legend(framealpha=0.9, fontsize=9.5)

# Right panel: Tier 3 LLMs
ax = axes[1]
llm_cols = {"claude": "claude_mean_ms", "gpt4o": "gpt4o_mean_ms",
            "gpt4o_mini": "gpt4o_mini_mean_ms", "gemini": "gemini_mean_ms"}
for model, label, color, marker in zip(MODELS, MODEL_LABELS, MODEL_COLORS,
                                        ["o","s","^","D"]):
    col = llm_cols[model]
    ax.plot(RUN_NOS, runs[col], color=color, marker=marker,
            linewidth=2, markersize=8, label=label)

ax.set_title("Tier 3 — LLM Models", fontsize=11)
ax.set_xlabel("Run Number")
ax.set_ylabel("Latency (ms)")
ax.set_xticks(RUN_NOS)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v/1000:.1f}s"))
ax.legend(framealpha=0.9, fontsize=9.5)

plt.tight_layout()
out3 = PLOT_DIR / "G4_fig3_per_run_consistency.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
