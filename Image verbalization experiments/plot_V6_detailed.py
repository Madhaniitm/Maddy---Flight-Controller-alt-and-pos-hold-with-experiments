"""
plot_V6_detailed.py — V6 Thesis Plots (Verbosity vs Quality)
=============================================================
Three thesis-quality plots for V6:

  Fig 1 — Quality vs token budget per model (line plot)
           Core finding: plateau at 256 tokens; GPT-4o/mini flatten at 128
  Fig 2 — With CLIP vs Without CLIP (quality lines overlaid)
           Ablation finding: CLIP adds nothing; lines nearly identical
  Fig 3 — Quality/Cost efficiency vs token budget per model
           Deployment decision: 256 is the knee; 512 buys 0.01 more quality

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_V6_detailed.py"
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ_DIR  = pathlib.Path(__file__).parent
RES_DIR  = VIZ_DIR / "results"
PLOT_DIR = RES_DIR / "V6_plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
clip    = pd.read_csv(RES_DIR / "V6_runs_20260525_185304.csv")
noclip  = pd.read_csv(RES_DIR / "V6_runs_noclip_20260525_201620.csv")

for df in (clip, noclip):
    df.drop(df[df["error"].notna() & (df["error"] != "")].index, inplace=True)

TOKEN_LEVELS  = [64, 128, 256, 512]
MODELS        = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MODEL_LABELS  = ["Claude", "GPT-4o", "GPT-4o-mini", "Gemini"]
MODEL_COLORS  = ["#e07b54", "#4a90d9", "#6cbf6c", "#b07ed9"]
MODEL_MARKERS = ["o", "s", "^", "D"]

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
# FIG 1 — Quality vs Token Budget per model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for model, label, color, marker in zip(MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS):
    means = [clip[(clip["model"] == model) & (clip["max_tokens"] == t)]["quality_score"].mean()
             for t in TOKEN_LEVELS]
    sems  = [clip[(clip["model"] == model) & (clip["max_tokens"] == t)]["quality_score"].sem()
             for t in TOKEN_LEVELS]
    ax.plot(TOKEN_LEVELS, means, color=color, marker=marker,
            linewidth=2, markersize=8, label=label)
    ax.fill_between(TOKEN_LEVELS,
                    [m - s for m, s in zip(means, sems)],
                    [m + s for m, s in zip(means, sems)],
                    color=color, alpha=0.12)

# Plateau annotation
ax.axvline(256, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(258, 3.35, "Plateau\n(256 tok)", fontsize=9, color="gray", va="bottom")

# GPT-4o/mini early plateau annotation
ax.axvline(128, color="#b0b0b0", linestyle=":", linewidth=1, alpha=0.6)
ax.text(130, 3.2, "GPT models\nplateau here", fontsize=8.5, color="#b0b0b0", va="bottom")

ax.set_xlabel("max_tokens (output token budget)")
ax.set_ylabel("Quality Score / 5")
ax.set_title("Fig 1 — Verbalization Quality vs Output Token Budget\n"
             "GPT-4o and GPT-4o-mini plateau at 128; Claude and Gemini need 256")
ax.set_xticks(TOKEN_LEVELS)
ax.set_xticklabels([str(t) for t in TOKEN_LEVELS])
ax.set_ylim(2.8, 5.2)
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
out1 = PLOT_DIR / "V6_fig1_quality_vs_tokens.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 1 → {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — With CLIP vs Without CLIP (averaged across all models)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fig 2 — CLIP Ablation: With vs Without CLIP\n"
             "Solid = with CLIP · Dashed = without CLIP",
             fontsize=13)

# Left panel: overall average
ax = axes[0]
clip_avg   = [clip[clip["max_tokens"] == t]["quality_score"].mean()   for t in TOKEN_LEVELS]
noclip_avg = [noclip[noclip["max_tokens"] == t]["quality_score"].mean() for t in TOKEN_LEVELS]
clip_sem   = [clip[clip["max_tokens"] == t]["quality_score"].sem()    for t in TOKEN_LEVELS]
noclip_sem = [noclip[noclip["max_tokens"] == t]["quality_score"].sem() for t in TOKEN_LEVELS]

ax.plot(TOKEN_LEVELS, clip_avg,   color="#4a90d9", marker="o", linewidth=2.5,
        markersize=9, label="With CLIP", zorder=5)
ax.plot(TOKEN_LEVELS, noclip_avg, color="#e07b54", marker="s", linewidth=2.5,
        markersize=9, linestyle="--", label="Without CLIP", zorder=5)
ax.fill_between(TOKEN_LEVELS,
                [m - s for m, s in zip(clip_avg, clip_sem)],
                [m + s for m, s in zip(clip_avg, clip_sem)],
                color="#4a90d9", alpha=0.12)
ax.fill_between(TOKEN_LEVELS,
                [m - s for m, s in zip(noclip_avg, noclip_sem)],
                [m + s for m, s in zip(noclip_avg, noclip_sem)],
                color="#e07b54", alpha=0.12)

# Max delta annotation
deltas = [abs(c - n) for c, n in zip(clip_avg, noclip_avg)]
max_d_idx = int(np.argmax(deltas))
ax.annotate(f"Max ΔQ = {deltas[max_d_idx]:.2f}",
            xy=(TOKEN_LEVELS[max_d_idx], max(clip_avg[max_d_idx], noclip_avg[max_d_idx])),
            xytext=(TOKEN_LEVELS[max_d_idx] + 30, max(clip_avg[max_d_idx], noclip_avg[max_d_idx]) + 0.15),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            fontsize=9, color="gray")

ax.set_title("All Models — Average Quality", fontsize=11)
ax.set_xlabel("max_tokens")
ax.set_ylabel("Quality Score / 5")
ax.set_xticks(TOKEN_LEVELS)
ax.set_ylim(2.8, 5.2)
ax.legend(framealpha=0.9, fontsize=10)

# Right panel: per-model at 128 tokens (where CLIP hurts Gemini most)
ax = axes[1]
x     = np.arange(len(MODELS))
width = 0.35
t     = 128   # token level where CLIP effect is largest

clip_vals   = [clip[(clip["model"] == m) & (clip["max_tokens"] == t)]["quality_score"].mean()
               for m in MODELS]
noclip_vals = [noclip[(noclip["model"] == m) & (noclip["max_tokens"] == t)]["quality_score"].mean()
               for m in MODELS]

b1 = ax.bar(x - width/2, clip_vals,   width, label="With CLIP",    color="#4a90d9", alpha=0.85, edgecolor="white")
b2 = ax.bar(x + width/2, noclip_vals, width, label="Without CLIP", color="#e07b54", alpha=0.85,
            edgecolor="white", hatch="//", linewidth=0)

# Annotate delta on each pair
for i, (c, n) in enumerate(zip(clip_vals, noclip_vals)):
    delta = n - c
    sign  = "+" if delta >= 0 else ""
    color = "#27ae60" if delta > 0.05 else ("#c0392b" if delta < -0.05 else "gray")
    ax.text(x[i], max(c, n) + 0.06, f"Δ{sign}{delta:.2f}",
            ha="center", fontsize=9, color=color, fontweight="bold")

ax.set_title("Per Model at 128 tokens\n(where CLIP effect is largest)", fontsize=11)
ax.set_xlabel("Model")
ax.set_ylabel("Quality Score / 5")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS)
ax.set_ylim(2.8, 5.5)
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
out2 = PLOT_DIR / "V6_fig2_clip_ablation.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 2 → {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Quality/Cost Efficiency vs Token Budget per model
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for model, label, color, marker in zip(MODELS, MODEL_LABELS, MODEL_COLORS, MODEL_MARKERS):
    effs = []
    for t in TOKEN_LEVELS:
        sub = clip[(clip["model"] == model) & (clip["max_tokens"] == t)]
        q   = sub["quality_score"].mean()
        c   = sub["cost_usd"].mean()
        effs.append(q / c if c > 0 else 0)
    # Normalise per model to 0–1 so all models are on same scale
    effs = np.array(effs, dtype=float)
    effs_norm = (effs - effs.min()) / (effs.max() - effs.min() + 1e-9)
    ax.plot(TOKEN_LEVELS, effs_norm, color=color, marker=marker,
            linewidth=2, markersize=8, label=label)

ax.axvline(256, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(258, 0.05, "256 tokens\n(optimal)", fontsize=9, color="gray")

ax.set_xlabel("max_tokens (output token budget)")
ax.set_ylabel("Quality/Cost Efficiency (normalised per model)")
ax.set_title("Fig 3 — Quality/Cost Efficiency vs Token Budget\n"
             "256 tokens is the knee — beyond this, cost rises with no quality gain")
ax.set_xticks(TOKEN_LEVELS)
ax.set_xticklabels([str(t) for t in TOKEN_LEVELS])
ax.set_ylim(-0.05, 1.15)
ax.legend(framealpha=0.9, fontsize=10)

plt.tight_layout()
out3 = PLOT_DIR / "V6_fig3_efficiency_vs_tokens.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Fig 3 → {out3.name}")

print(f"\nAll 3 plots saved to:\n  {PLOT_DIR}")
