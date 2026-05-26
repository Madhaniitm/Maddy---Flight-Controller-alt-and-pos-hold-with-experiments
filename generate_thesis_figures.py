#!/usr/bin/env python3
"""
generate_thesis_figures.py
==========================
Generates all Chapter 3 figure PDFs for the thesis.

Output directory: Thesis/figures/  (created automatically)

Figures produced:
  fig3_2_v2_prompt_techniques.pdf   -- V2 prompt technique comparison
  fig3_3_v6_token_budget.pdf        -- V6 token budget + CLIP ablation
  fig3_4_v7_history_modes.pdf       -- V7 scene context history
  fig3_5_g4_timescales.pdf          -- G4 four-tier latency (log scale)
  fig3_6_g1_tier_isolation.pdf      -- G1 tier isolation safety
  fig3_7_g2_trigger_timeline.pdf    -- G2 hybrid vs scheduled trigger
  fig3_8_g5_scene_heatmap.pdf       -- G5 safety heatmap (model x scene)
  fig3_9_g5_latency_cost.pdf        -- G5 latency and cost per model
  fig3_10_g5_reasoning_radar.pdf    -- G5 reasoning quality radar chart

NOT generated (must be created manually):
  fig3_1_pipeline_architecture.pdf  -- TikZ diagram or draw.io export

Run from the project root:
  python generate_thesis_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "Image verbalization experiments", "results")
OUT_DIR    = os.path.join(SCRIPT_DIR, "Thesis", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style — clean academic look
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.30,
    "grid.linestyle":     "--",
    "grid.zorder":        0,
})

MODEL_COLORS = {
    "claude":     "#4C72B0",
    "gpt4o":      "#DD8452",
    "gpt4o_mini": "#55A868",
    "gemini":     "#C44E52",
}
MODEL_LABELS = {
    "claude":     "Claude",
    "gpt4o":      "GPT-4o",
    "gpt4o_mini": "GPT-4o-mini",
    "gemini":     "Gemini",
}

def _save(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  [OK] {name}")

def _path(filename):
    return os.path.join(DATA_DIR, filename)


# ===========================================================================
# FIG 3.2  V2 — Prompt Technique Comparison
# ===========================================================================
def fig_v2():
    print("Generating fig3_2_v2_prompt_techniques ...")

    df     = pd.read_csv(_path("V2_runs_20260525_035756.csv"))
    df_cot = pd.read_csv(_path("V2_cot_rerun_20260525_175408.csv"))

    # Normalise strings
    df["technique"]    = df["technique"].str.strip().str.lower()
    df["detected_risk"]= df["detected_risk"].str.strip().str.lower().fillna("")
    df["truth"]        = df["truth"].str.strip().str.lower()

    # Exclude ReAct (not reported in chapter)
    df = df[df["technique"] != "react"]

    TECHS = ["zero_shot", "few_shot_3", "cot", "structured"]
    TECH_LABELS = {
        "zero_shot":  "Zero-shot",
        "few_shot_3": "Few-shot-3",
        "cot":        "CoT\n(300 tok)",
        "cot_600":    "CoT\n(600 tok)",
        "structured": "Structured",
    }

    rows = []
    for t in TECHS:
        sub = df[df["technique"] == t].dropna(subset=["detected_risk"])
        sub = sub[sub["detected_risk"] != ""]
        acc  = (sub["detected_risk"] == sub["truth"]).mean() * 100
        dang = int(sub["action_dangerous"].sum())
        rows.append(dict(key=t, label=TECH_LABELS[t], acc=acc, dang=dang, n=len(sub)))

    # CoT 600 rerun — different CSV, only has detected_risk / truth / action_dangerous
    df_cot["detected_risk"] = df_cot["detected_risk"].str.strip().str.lower().fillna("")
    df_cot["truth"]         = df_cot["truth"].str.strip().str.lower()
    sub600 = df_cot[df_cot["detected_risk"] != ""]
    rows.insert(3, dict(
        key="cot_600",
        label=TECH_LABELS["cot_600"],
        acc=(sub600["detected_risk"] == sub600["truth"]).mean() * 100,
        dang=int(sub600["action_dangerous"].sum()),
        n=len(sub600),
    ))

    labels = [r["label"] for r in rows]
    acc    = [r["acc"]   for r in rows]
    dang   = [r["dang"]  for r in rows]
    colors = ["#6baed6", "#74c476", "#fdae6b", "#fb6a4a", "#9e9ac8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    # --- (a) Label Accuracy ---
    bars = ax1.bar(labels, acc, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax1.set_ylabel("Label Accuracy (%)")
    ax1.set_title("(a) Label Accuracy by Prompt Technique")
    ax1.set_ylim(0, 70)
    for bar, v in zip(bars, acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.6,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
    # Highlight winner
    ax1.get_children()[4].set_edgecolor("#333")
    ax1.get_children()[4].set_linewidth(2)

    # --- (b) Dangerous count ---
    bars2 = ax2.bar(labels, dang, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax2.set_ylabel("Dangerous Recommendations (count)")
    ax2.set_title("(b) Dangerous Recommendations by Technique")
    ax2.set_ylim(0, max(dang) + 3)
    ax2.axhline(0, color="green", linestyle="--", linewidth=1, alpha=0.6)
    for bar, v in zip(bars2, dang):
        label = str(v) if v > 0 else "0 ✓"
        color = "black" if v > 0 else "#1a7a1a"
        ax2.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.1,
                 label, ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")

    plt.tight_layout(pad=1.5)
    _save("fig3_2_v2_prompt_techniques.pdf")


# ===========================================================================
# FIG 3.3  V6 — Token Budget and CLIP Ablation
# ===========================================================================
def fig_v6():
    print("Generating fig3_3_v6_token_budget ...")

    df_clip   = pd.read_csv(_path("V6_runs_20260525_185304.csv"))
    df_noclip = pd.read_csv(_path("V6_runs_noclip_20260525_201620.csv"))

    TOKENS  = [64, 128, 256, 512]
    models  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
    markers = ["o", "s", "^", "D"]
    lstyles = ["-", "--", "-.", ":"]

    def tok_quality(df, tok):
        return df[df["max_tokens"] == tok]["quality_score"].mean()

    clip_avg   = [tok_quality(df_clip,   t) for t in TOKENS]
    noclip_avg = [tok_quality(df_noclip, t) for t in TOKENS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- (a) Overall: with vs without CLIP ---
    ax1.plot(TOKENS, clip_avg,   "o-",  color="#2171b5", lw=2, ms=7, label="With CLIP",    zorder=4)
    ax1.plot(TOKENS, noclip_avg, "s--", color="#cb181d", lw=2, ms=7, label="Without CLIP", zorder=4)
    ax1.set_xlabel("max\_tokens")
    ax1.set_ylabel("Quality Score (/ 5)")
    ax1.set_title("(a) Quality vs Token Budget\n(all models averaged)")
    ax1.set_xticks(TOKENS)
    ax1.set_ylim(3.5, 5.1)
    ax1.axvline(256, color="grey", ls=":", lw=1.1, alpha=0.7)
    ax1.text(256 + 8, 3.62, "plateau\n@256", fontsize=7.5, color="grey", va="bottom")
    ax1.legend()

    # --- (b) Per model, no CLIP ---
    for i, model in enumerate(models):
        sub  = df_noclip[df_noclip["model"] == model]
        means= [sub[sub["max_tokens"] == t]["quality_score"].mean() for t in TOKENS]
        ax2.plot(TOKENS, means,
                 marker=markers[i], ls=lstyles[i],
                 color=MODEL_COLORS[model], lw=1.8, ms=6,
                 label=MODEL_LABELS[model])

    ax2.set_xlabel("max\_tokens")
    ax2.set_ylabel("Quality Score (/ 5)")
    ax2.set_title("(b) Per-Model Quality vs Token Budget\n(without CLIP)")
    ax2.set_xticks(TOKENS)
    ax2.set_ylim(3.5, 5.1)
    ax2.axvline(256, color="grey", ls=":", lw=1.1, alpha=0.7)
    ax2.text(256 + 8, 3.62, "plateau\n@256", fontsize=7.5, color="grey", va="bottom")
    ax2.legend(loc="lower right")

    plt.tight_layout(pad=1.5)
    _save("fig3_3_v6_token_budget.pdf")


# ===========================================================================
# FIG 3.4  V7 — Scene Context History
# ===========================================================================
def fig_v7():
    print("Generating fig3_4_v7_history_modes ...")

    df = pd.read_csv(_path("V7_runs_20260525_221631.csv"))

    modes   = ["stateless", "short", "full"]
    models  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
    M_LABS  = {"stateless": "Stateless", "short": "Short\n(2 fr.)", "full": "Full\n(all fr.)"}

    # Per-model × mode risk accuracy
    per_model = {
        m: [df[(df["model"] == m) & (df["history_mode"] == mo)]["risk_correct"].mean() * 100
            for mo in modes]
        for m in models
    }
    # Overall per mode
    overall = [df[df["history_mode"] == mo]["risk_correct"].mean() * 100 for mo in modes]

    x = np.arange(len(modes))
    width = 0.18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- (a) Grouped bars per model ---
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for i, model in enumerate(models):
        ax1.bar(x + offsets[i] * width, per_model[model], width,
                label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                edgecolor="white", linewidth=0.6, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([M_LABS[m] for m in modes])
    ax1.set_ylabel("Risk Accuracy (%)")
    ax1.set_title("(a) Risk Accuracy by Model × History Mode")
    ax1.set_ylim(0, 105)
    ax1.legend(ncol=2, fontsize=8)

    # --- (b) Overall with best highlighted ---
    bar_colors = ["#deebf7", "#08519c", "#deebf7"]   # short = dark
    bars = ax2.bar([M_LABS[m] for m in modes], overall,
                   color=bar_colors, edgecolor=["#3182bd"] * 3,
                   linewidth=1.2, zorder=3)
    for i, v in enumerate(overall):
        ax2.text(i, v + 0.8, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=10,
                 fontweight="bold" if i == 1 else "normal",
                 color="#08519c" if i == 1 else "black")
    ax2.set_ylabel("Risk Accuracy (%)")
    ax2.set_title("(b) Overall Risk Accuracy by History Mode\n(all models)")
    ax2.set_ylim(0, 75)
    # Annotation for short history advantage
    delta = overall[1] - overall[0]
    ax2.annotate(f"+{delta:.1f}pp\nover stateless",
                 xy=(1, overall[1]), xytext=(1.6, overall[1] + 5),
                 arrowprops=dict(arrowstyle="->", color="#08519c", lw=1.2),
                 fontsize=8, color="#08519c")

    plt.tight_layout(pad=1.5)
    _save("fig3_4_v7_history_modes.pdf")


# ===========================================================================
# FIG 3.5  G4 — Four-Tier Timescales (log scale)
# ===========================================================================
def fig_g4():
    print("Generating fig3_5_g4_timescales ...")

    df = pd.read_csv(_path("G4_runs_20260524_234054.csv"))

    # Compute means across runs
    pid_ms   = df["pid_mean_ms"].mean()
    local_ms = df["local_mean_ms"].mean()
    yolo_ms  = df["yolo_mean_ms"].mean()
    claude_ms= df["claude_mean_ms"].mean()
    gpt4o_ms = df["gpt4o_mean_ms"].mean()
    mini_ms  = df["gpt4o_mini_mean_ms"].mean()
    gem_ms   = df["gemini_mean_ms"].mean()

    labels = [
        "PID\n(Tier 1)",
        "MediaPipe\n(Tier 1.5)",
        "YOLO-World\n(Tier 2)",
        "Claude\n(Tier 3)",
        "GPT-4o\n(Tier 3)",
        "GPT-4o-mini\n(Tier 3)",
        "Gemini\n(Tier 3)",
    ]
    values = [pid_ms, local_ms, yolo_ms, claude_ms, gpt4o_ms, mini_ms, gem_ms]
    colors = [
        "#e6550d",                      # Tier 1 — red-orange
        "#fd8d3c",                      # Tier 1.5 — orange
        "#3182bd",                      # Tier 2 — blue
        MODEL_COLORS["claude"],         # Tier 3 LLMs
        MODEL_COLORS["gpt4o"],
        MODEL_COLORS["gpt4o_mini"],
        MODEL_COLORS["gemini"],
    ]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Mean Latency (ms, log scale)")
    ax.set_title("G4: Per-Tier Processing Latency — Four Orders of Magnitude")

    # Value labels above each bar
    for bar, v in zip(bars, values):
        text = f"{v:.4f}" if v < 0.01 else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                v * 1.6,
                f"{text} ms",
                ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    # Vertical dividers between tiers
    for xpos in [0.5, 1.5, 2.5]:
        ax.axvline(xpos, color="grey", ls="--", lw=0.7, alpha=0.45)

    # Tier labels at top
    ymax_label = max(values) * 15
    ax.text(0.0, ymax_label, "Tier 1",   ha="center", fontsize=8.5, color="#e6550d",  style="italic")
    ax.text(1.0, ymax_label, "Tier 1.5", ha="center", fontsize=8.5, color="#fd8d3c",  style="italic")
    ax.text(2.0, ymax_label, "Tier 2",   ha="center", fontsize=8.5, color="#3182bd",  style="italic")
    ax.text(4.5, ymax_label, "Tier 3 (LLM APIs)", ha="center", fontsize=8.5, color="#555", style="italic")

    # Orders-of-magnitude annotation
    ax.annotate("",
                xy=(0, pid_ms), xytext=(0, yolo_ms),
                arrowprops=dict(arrowstyle="<->", color="grey", lw=1.0))
    ax.text(-0.45, np.sqrt(pid_ms * yolo_ms),
            "~6 orders\nof magnitude",
            fontsize=7.5, color="grey", va="center", ha="right")

    plt.tight_layout(pad=1.5)
    _save("fig3_5_g4_timescales.pdf")


# ===========================================================================
# FIG 3.6  G1 — Tier Isolation: Action Safety
# ===========================================================================
def fig_g1():
    print("Generating fig3_6_g1_tier_isolation ...")

    df = pd.read_csv(_path("G1_runs_20260525_021111.csv"))

    COND_ORDER  = ["tier1_5_only", "tier2_only", "tier3_only", "tier1_5_tier2_tier3"]
    COND_LABELS = {
        "tier1_5_only":          "Tier 1.5\nOnly",
        "tier2_only":            "Tier 2\nOnly",
        "tier3_only":            "Tier 3\nOnly",
        "tier1_5_tier2_tier3":   "Full\nPipeline",
    }

    conds_present = [c for c in COND_ORDER if c in df["condition"].unique()]

    def metrics(cond):
        sub = df[df["condition"] == cond]
        return {
            "safe_rate": sub["action_safe"].mean() * 100,
            "dangerous": int(sub["action_dangerous"].sum()),
            "n":         len(sub),
        }

    data = {c: metrics(c) for c in conds_present}

    labels     = [COND_LABELS[c] for c in conds_present]
    safe_rates = [data[c]["safe_rate"] for c in conds_present]
    dang       = [data[c]["dangerous"] for c in conds_present]
    colors     = ["#4dac26" if s == 100.0 else "#d01c8b" if s < 90 else "#f1b6da"
                  for s in safe_rates]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    # --- (a) Action safety % ---
    bars1 = ax1.bar(labels, safe_rates, color=colors, edgecolor="white", lw=0.8, zorder=3)
    ax1.set_ylabel("Action Safety (%)")
    ax1.set_title("(a) Action Safety by Pipeline Configuration")
    ax1.set_ylim(0, 115)
    ax1.axhline(100, color="#1a7a1a", ls="--", lw=1.0, alpha=0.55)
    for bar, v in zip(bars1, safe_rates):
        marker = "✓" if v == 100 else f"{v:.1f}%"
        col    = "#1a7a1a" if v == 100 else "#d01c8b"
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                 marker, ha="center", va="bottom", fontsize=11, fontweight="bold", color=col)

    # --- (b) Dangerous count ---
    dang_colors = ["#4dac26" if d == 0 else "#d01c8b" for d in dang]
    bars2 = ax2.bar(labels, dang, color=dang_colors, edgecolor="white", lw=0.8, zorder=3)
    ax2.set_ylabel("Dangerous Recommendations (count)")
    ax2.set_title("(b) Dangerous Recommendations per Configuration")
    ax2.set_ylim(0, max(dang + [1]) + 3)
    ax2.axhline(0, color="#1a7a1a", ls="--", lw=1.0, alpha=0.55)
    for bar, v in zip(bars2, dang):
        label = "0 ✓" if v == 0 else str(v)
        col   = "#1a7a1a" if v == 0 else "#d01c8b"
        ax2.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.1,
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold", color=col)

    # Note about blocked_lens
    fig.text(0.5, -0.02,
             "Note: dangerous recommendations in Tier 2 Only arise solely from the blocked-lens scene "
             "(depth sensor blind spot). The full pipeline LLM detects sensor failure and overrides.",
             ha="center", fontsize=7.5, style="italic", color="#555")

    plt.tight_layout(pad=1.5)
    _save("fig3_6_g1_tier_isolation.pdf")


# ===========================================================================
# FIG 3.7  G2 — Hybrid vs Scheduled Trigger
# ===========================================================================
def fig_g2():
    print("Generating fig3_7_g2_trigger_timeline ...")

    df_runs  = pd.read_csv(_path("G2_runs_20260524_232234.csv"))
    df_calls = pd.read_csv(_path("G2_calls_20260524_232234.csv"))

    models = sorted(df_runs["model"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("G2: Hybrid vs Scheduled Trigger Strategy", fontsize=12, y=1.01)

    # ---- (a) frames_late histogram ----
    ax = axes[0, 0]
    sched_late = df_runs[df_runs["strategy"] == "scheduled"]["frames_late"]
    hyb_late   = df_runs[df_runs["strategy"] == "hybrid"]["frames_late"]
    all_vals   = pd.concat([sched_late, hyb_late])
    bins = np.arange(all_vals.min() - 0.5, all_vals.max() + 1.5, 1)
    ax.hist(sched_late, bins=bins, alpha=0.72, color="#d62728", label="Scheduled", edgecolor="white")
    ax.hist(hyb_late,   bins=bins, alpha=0.72, color="#2ca02c", label="Hybrid",    edgecolor="white")
    ax.set_xlabel("Frames Late (hazard detection delay)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Hazard Detection Delay Distribution")
    ax.legend()
    # Mean lines
    ax.axvline(sched_late.mean(), color="#d62728", ls="--", lw=1.2,
               label=f"Sched. mean={sched_late.mean():.1f}")
    ax.axvline(hyb_late.mean(),   color="#2ca02c", ls="--", lw=1.2,
               label=f"Hybrid mean={hyb_late.mean():.1f}")
    ax.legend(fontsize=8)

    # ---- (b) cost per sequence ----
    ax = axes[0, 1]
    sched_cost = df_runs[df_runs["strategy"] == "scheduled"]["cost_usd"] * 1000
    hyb_cost   = df_runs[df_runs["strategy"] == "hybrid"]["cost_usd"] * 1000
    ax.hist(sched_cost, bins=20, alpha=0.72, color="#d62728", label="Scheduled", edgecolor="white")
    ax.hist(hyb_cost,   bins=20, alpha=0.72, color="#2ca02c", label="Hybrid",    edgecolor="white")
    ax.set_xlabel("Cost per Sequence (USD × 10⁻³)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Cost per Sequence Distribution")
    ax.legend()

    # ---- (c) call timeline — one representative sequence ----
    ax = axes[1, 0]
    SHOW_MODEL = "claude"
    SHOW_SEQ   = 1
    for s_idx, (strat, color, y_pos, label) in enumerate([
        ("scheduled", "#d62728", 1.0, "Scheduled"),
        ("hybrid",    "#2ca02c", 0.0, "Hybrid"),
    ]):
        sub = df_calls[
            (df_calls["strategy"] == strat) &
            (df_calls["model"] == SHOW_MODEL) &
            (df_calls["seq"] == SHOW_SEQ)
        ].copy().sort_values("frame_num")

        if len(sub) == 0:
            continue

        max_frame = sub["frame_num"].max()

        # Timeline baseline
        ax.hlines(y_pos, 1, max_frame, color=color, lw=1.2, alpha=0.35)
        ax.text(-0.2, y_pos, label, va="center", ha="right", fontsize=9,
                color=color, fontweight="bold")

        # Plot calls
        for _, row in sub.iterrows():
            is_interrupt = str(row.get("call_source", "")).strip().lower() == "local"
            marker = "^" if is_interrupt else "o"
            ms     = 12  if is_interrupt else 9
            ax.plot(row["frame_num"], y_pos, marker=marker,
                    color=color, ms=ms, zorder=5, markeredgecolor="white", mew=0.8)

        # Shade hazard frames
        if "in_hazard" in sub.columns:
            haz = sub[sub["in_hazard"] == 1]["frame_num"]
            if len(haz) > 0:
                ax.axvspan(haz.min() - 0.45, haz.max() + 0.45,
                           alpha=0.12, color="red", zorder=0)

    # Legend patches
    leg_patches = [
        mpatches.Patch(facecolor="red", alpha=0.2, label="Hazard frames"),
        plt.Line2D([0], [0], marker="o", color="grey", ms=8, lw=0, label="Scheduled call"),
        plt.Line2D([0], [0], marker="^", color="grey", ms=10, lw=0, label="Interrupt call"),
    ]
    ax.legend(handles=leg_patches, fontsize=7.5, loc="upper right")
    ax.set_xlabel("Frame Number")
    ax.set_title(f"(c) Call Timeline — {MODEL_LABELS[SHOW_MODEL]}, Seq {SHOW_SEQ}")
    ax.set_xlim(0, df_calls["frame_num"].max() + 1)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([])

    # ---- (d) frames_late by model ----
    ax = axes[1, 1]
    x  = np.arange(len(models))
    w  = 0.33
    for j, (strat, color, label) in enumerate([
        ("scheduled", "#d62728", "Scheduled"),
        ("hybrid",    "#2ca02c", "Hybrid"),
    ]):
        means = [
            df_runs[(df_runs["strategy"] == strat) & (df_runs["model"] == m)]["frames_late"].mean()
            for m in models
        ]
        bars = ax.bar(x + (j - 0.5) * w, means, w,
                      label=label, color=color, edgecolor="white", lw=0.7, zorder=3)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=15)
    ax.set_ylabel("Mean Frames Late")
    ax.set_title("(d) Mean Detection Delay by Model")
    ax.legend()

    plt.tight_layout(pad=1.5)
    _save("fig3_7_g2_trigger_timeline.pdf")


# ===========================================================================
# FIG 3.8  G5 — Scene Safety Heatmap
# ===========================================================================
def fig_g5_heatmap():
    print("Generating fig3_8_g5_scene_heatmap ...")

    df = pd.read_csv(_path("G5_runs_20260525_011427_reasoned.csv"))

    models = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
    scenes = sorted(df["scene_label"].unique())

    # Dangerous rate (%) per model × scene
    mat = np.zeros((len(models), len(scenes)))
    ann = np.empty((len(models), len(scenes)), dtype=object)

    for i, m in enumerate(models):
        for j, s in enumerate(scenes):
            sub = df[(df["model"] == m) & (df["scene_label"] == s)]
            if len(sub) == 0:
                mat[i, j]  = np.nan
                ann[i, j]  = "–"
            else:
                rate         = sub["action_dangerous"].mean() * 100
                mat[i, j]   = rate
                ann[i, j]   = f"{rate:.0f}%"

    scene_clean = [s.replace("_", "\n") for s in scenes]
    model_clean = [MODEL_LABELS[m] for m in models]

    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scene_clean, fontsize=8.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_clean)
    ax.set_title("G5 Full Pipeline: Dangerous Action Rate (%) by Model × Scene\n"
                 "(green = 0 % dangerous; red = high dangerous rate)")

    for i in range(len(models)):
        for j in range(len(scenes)):
            v     = mat[i, j]
            label = ann[i, j]
            tc    = "white" if (not np.isnan(v) and v > 55) else "black"
            fw    = "bold" if (not np.isnan(v) and v > 0) else "normal"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    fontweight=fw, color=tc)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("Dangerous Rate (%)", fontsize=9)

    plt.tight_layout(pad=1.2)
    _save("fig3_8_g5_scene_heatmap.pdf")


# ===========================================================================
# FIG 3.9  G5 — Latency and Cost per Model
# ===========================================================================
def fig_g5_latency_cost():
    print("Generating fig3_9_g5_latency_cost ...")

    df = pd.read_csv(_path("G5_runs_20260525_011427_reasoned.csv"))

    models = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

    lat_m, lat_s, cost_m, cost_s = [], [], [], []
    for m in models:
        sub = df[df["model"] == m]
        lat_m.append(sub["llm_ms"].mean())
        lat_s.append(sub["llm_ms"].std())
        cost_m.append(sub["cost_usd"].mean() * 1000)
        cost_s.append(sub["cost_usd"].std()  * 1000)

    mlabs  = [MODEL_LABELS[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    x      = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    # --- (a) Latency ---
    bars1 = ax1.bar(mlabs, lat_m, yerr=lat_s, color=colors,
                    edgecolor="white", lw=0.8, capsize=5, zorder=3,
                    error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    ax1.set_ylabel("LLM Latency (ms)")
    ax1.set_title("(a) Mean LLM API Latency per Model\n(G5 full pipeline, ±1 SD)")
    for i, v in enumerate(lat_m):
        ax1.text(i, v + lat_s[i] + 60,
                 f"{v:.0f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    # Dashed line at Gemini (production choice)
    ax1.axhline(lat_m[3], color=MODEL_COLORS["gemini"], ls=":", lw=1, alpha=0.6)
    ax1.text(3.5, lat_m[3] + 50, "production\nchoice", fontsize=7.5,
             color=MODEL_COLORS["gemini"], ha="left")

    # --- (b) Cost ---
    bars2 = ax2.bar(mlabs, cost_m, yerr=cost_s, color=colors,
                    edgecolor="white", lw=0.8, capsize=5, zorder=3,
                    error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    ax2.set_ylabel("Cost per Call (USD × 10⁻³)")
    ax2.set_title("(b) Mean API Cost per LLM Call\n(×10⁻³ USD, ±1 SD)")
    for i, v in enumerate(cost_m):
        ax2.text(i, v + cost_s[i] + 0.008,
                 f"${v:.3f}m", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    # Hourly rate annotation for Gemini (0.4 Hz = 1440/hr... actually 0.4 Hz = 1440 calls/hr)
    # 0.4 Hz = 0.4 calls/s = 1440 calls/hr
    # At cost_m[3] milli-USD/call: hourly = cost_m[3] * 1440 / 1000 USD
    gem_hourly = cost_m[3] * 1440 / 1000  # USD/hr (0.4 Hz continuous)
    ax2.text(0.02, 0.97,
             f"Gemini @ 0.4 Hz:\n~${gem_hourly:.2f}/hr",
             transform=ax2.transAxes, fontsize=8, va="top",
             color=MODEL_COLORS["gemini"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=MODEL_COLORS["gemini"], lw=0.8))

    plt.tight_layout(pad=1.5)
    _save("fig3_9_g5_latency_cost.pdf")


# ===========================================================================
# FIG 3.10  G5 — Reasoning Quality Radar Chart
# ===========================================================================
def fig_g5_radar():
    print("Generating fig3_10_g5_reasoning_radar ...")

    df = pd.read_csv(_path("G5_runs_20260525_011427_reasoned.csv"))
    # Derive action_safe from action_dangerous
    df["action_safe_flag"] = 1 - df["action_dangerous"]

    models  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
    dims    = ["desc_accurate", "reason_risk_aligned", "reason_action_aligned", "action_safe_flag"]
    dim_labels = [
        "Desc. Accuracy\n(DA)",
        "Risk Alignment\n(RRA)",
        "Action Alignment\n(RAA)",
        "Action Safety\n(AS)",
    ]

    per_model = {}
    for m in models:
        sub = df[df["model"] == m]
        per_model[m] = [sub[d].mean() * 100 for d in dims]

    N      = len(dims)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})

    for model in models:
        vals  = per_model[model] + [per_model[model][0]]
        ax.plot(angles, vals,
                lw=2.2, ls="-", marker="o", ms=7,
                color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]}  "
                      f"({'/'.join(f'{v:.0f}' for v in per_model[model])}%)")
        ax.fill(angles, vals, alpha=0.07, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=9)
    ax.set_ylim(0, 108)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7.5)
    ax.set_title("G5 Full Pipeline: Reasoning Quality\n(DA / RRA / RAA / AS)", pad=22, size=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), fontsize=8.5)

    plt.tight_layout()
    _save("fig3_10_g5_reasoning_radar.pdf")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print(f"Data  : {DATA_DIR}")
    print(f"Output: {OUT_DIR}")
    print()

    errors = []
    for fn in [fig_v2, fig_v6, fig_v7, fig_g4, fig_g1, fig_g2,
               fig_g5_heatmap, fig_g5_latency_cost, fig_g5_radar]:
        try:
            fn()
        except Exception as e:
            errors.append((fn.__name__, str(e)))
            print(f"  [FAIL] {fn.__name__}: {e}")

    print()
    produced = [f for f in os.listdir(OUT_DIR) if f.endswith(".pdf")]
    print(f"Files in output dir: {len(produced)}")
    for f in sorted(produced):
        print(f"  {f}")

    if errors:
        print(f"\n{len(errors)} figure(s) failed:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print("\nAll figures generated successfully.")
        print("\nStill needed (manual / TikZ):")
        print("  fig3_1_pipeline_architecture.pdf")
