"""
plot_V_series.py  —  Generate PNG plots for all V-series experiments
=====================================================================
Reads the CSV files produced by exp_V1 … exp_V9 and saves one PNG per
experiment to the same results/ directory.  Experiments whose CSVs are
missing are silently skipped so this script can be run incrementally.

Usage:
    python plot_V_series.py              # plot all available experiments
    python plot_V_series.py V1 V5 V8    # plot specific experiments only

Output files (all in results/):
    V1_plot.png   Multi-model comparison   (4-panel: accuracy, quality, latency, cost)
    V2_plot.png   Prompt technique         (4-panel: accuracy, quality, tokens, cost)
    V3_plot.png   Multilingual input       (3-panel: relevance/lang_match/accuracy, latency)
    V4_plot.png   Model × Prompt matrix    (3-panel: quality heatmap, delta gains, acc heatmap)
    V5_plot.png   YOLO threshold sweep     (3-panel: P/R/F1 curves, accuracy/FAR/MR, confusion)
    V6_plot.png   Verbosity vs quality     (4-panel: quality, truncation, cost, efficiency)
    V7_plot.png   Scene context history    (4-panel: risk_acc, change_det, tokens, per-frame)
    V8_plot.png   Temperature sweep        (5-panel: quality, accuracy, consistency, flip, cost)
    V9_plot.png   Model × Params matrix    (4-panel: quality heatmaps ×3, interaction deltas)
"""

import sys, pathlib, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

SCRIPT_DIR  = pathlib.Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Colour palette (colour-blind friendly) ────────────────────────────────────
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
           "#D55E00", "#F0E442", "#999999"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flt(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def save(fig: plt.Figure, name: str) -> None:
    out = RESULTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")


def err_bars(rows: list[dict], mean_key: str, lo_key: str, hi_key: str):
    means = [flt(r, mean_key) for r in rows]
    lo    = [flt(r, mean_key) - flt(r, lo_key)  for r in rows]
    hi    = [flt(r, hi_key)  - flt(r, mean_key) for r in rows]
    return means, [lo, hi]


def bar_with_ci(ax, labels, means, errs, colors=None, title="", ylabel="", xlabel=""):
    x = np.arange(len(labels))
    col = colors if colors else PALETTE[:len(labels)]
    bars = ax.bar(x, means, color=col, width=0.55,
                  yerr=errs, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#555"})
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    return bars


def line_with_ci(ax, x_vals, means, lo, hi, label="", color=PALETTE[0],
                 marker="o", title="", xlabel="", ylabel=""):
    ax.plot(x_vals, means, marker=marker, color=color, label=label, linewidth=1.8)
    ax.fill_between(x_vals, lo, hi, alpha=0.18, color=color)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)


def heatmap(ax, data: np.ndarray, row_labels, col_labels, title="", fmt=".2f", vmin=None, vmax=None):
    cmap = cm.get_cmap("YlGnBu")
    norm = Normalize(vmin=vmin or np.nanmin(data), vmax=vmax or np.nanmax(data))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=8, color="white" if norm(v) > 0.6 else "black")
    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


# ─────────────────────────────────────────────────────────────────────────────
# V1 — Multi-Model Vision Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_V1():
    rows = read_csv(RESULTS_DIR / "V1_summary.csv")
    if not rows:
        return
    print("Plotting V1…")

    models = [r["model"] for r in rows]
    cols   = PALETTE[:len(models)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("V1 — Multi-Model Vision Comparison", fontsize=13, fontweight="bold")

    # Accuracy
    acc, acc_e = err_bars(rows, "accuracy", "acc_lo", "acc_hi")
    bar_with_ci(axes[0,0], models, acc, acc_e, cols,
                title="Classification Accuracy", ylabel="Accuracy (Wilson CI)")
    axes[0,0].axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    # Quality
    q, q_e = err_bars(rows, "quality", "q_lo", "q_hi")
    bar_with_ci(axes[0,1], models, q, q_e, cols,
                title="Quality Score (0–4)", ylabel="Mean quality (Bootstrap CI)")
    axes[0,1].set_ylim(0, 4.2)

    # Latency
    lat  = [flt(r, "latency_ms") for r in rows]
    lat_e = [[flt(r, "latency_ms") - flt(r, "lat_lo") for r in rows],
              [flt(r, "lat_hi") - flt(r, "latency_ms") for r in rows]]
    bar_with_ci(axes[1,0], models, lat, lat_e, cols,
                title="API Latency", ylabel="Latency (ms)")

    # Cost
    cost, cost_e = err_bars(rows, "cost_usd", "cost_lo", "cost_hi")
    bar_with_ci(axes[1,1], models, cost, cost_e, cols,
                title="Cost per Call", ylabel="Cost (USD)")

    plt.tight_layout()
    save(fig, "V1_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V2 — Prompt Technique Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_V2():
    rows = read_csv(RESULTS_DIR / "V2_summary.csv")
    if not rows:
        return
    print("Plotting V2…")

    techs = [r["technique"] for r in rows]
    cols  = PALETTE[:len(techs)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("V2 — Prompt Technique Comparison (Claude)", fontsize=13, fontweight="bold")

    acc, acc_e = err_bars(rows, "accuracy", "acc_lo", "acc_hi")
    bar_with_ci(axes[0,0], techs, acc, acc_e, cols,
                title="Classification Accuracy", ylabel="Accuracy (Wilson CI)")

    q, q_e = err_bars(rows, "quality", "q_lo", "q_hi")
    bar_with_ci(axes[0,1], techs, q, q_e, cols,
                title="Quality Score (0–4)", ylabel="Mean quality (Bootstrap CI)")
    axes[0,1].set_ylim(0, 4.2)

    tokens = [flt(r, "input_tokens") for r in rows]
    bar_with_ci(axes[1,0], techs, tokens, [[0]*len(techs), [0]*len(techs)], cols,
                title="Mean Input Tokens", ylabel="Input tokens")

    cost = [flt(r, "cost_usd") for r in rows]
    bar_with_ci(axes[1,1], techs, cost, [[0]*len(techs), [0]*len(techs)], cols,
                title="Cost per Call", ylabel="Cost (USD)")

    plt.tight_layout()
    save(fig, "V2_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V3 — Multilingual Input
# ─────────────────────────────────────────────────────────────────────────────

def plot_V3():
    rows = read_csv(RESULTS_DIR / "V3_summary.csv")
    if not rows:
        return
    print("Plotting V3…")

    langs = [r["language"] for r in rows]
    x     = np.arange(len(langs))
    w     = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("V3 — Multilingual Input Comparison (Claude)", fontsize=13, fontweight="bold")

    ax = axes[0]
    metrics = [
        ("relevance",  "rel_lo",  "rel_hi",  "Relevance",   PALETTE[0]),
        ("lang_match", "lm_lo",   "lm_hi",   "Lang Match",  PALETTE[1]),
        ("accuracy",   "acc_lo",  "acc_hi",  "Accuracy",    PALETTE[2]),
    ]
    for i, (mk, lo_k, hi_k, lbl, col) in enumerate(metrics):
        vals = [flt(r, mk) for r in rows]
        lo   = [flt(r, mk) - flt(r, lo_k) for r in rows]
        hi   = [flt(r, hi_k) - flt(r, mk) for r in rows]
        ax.bar(x + i*w, vals, w, label=lbl, color=col,
               yerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1.1})
    ax.set_xticks(x + w); ax.set_xticklabels(langs)
    ax.set_ylim(0, 1.15); ax.set_title("Relevance / Language Match / Accuracy")
    ax.set_ylabel("Rate (Wilson CI)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    lat = [flt(r, "latency_ms") for r in rows]
    bar_with_ci(ax2, langs, lat, [[0]*len(langs), [0]*len(langs)],
                colors=PALETTE[:len(langs)],
                title="API Latency per Language", ylabel="Latency (ms)")

    plt.tight_layout()
    save(fig, "V3_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V4 — Model × Prompt Interaction
# ─────────────────────────────────────────────────────────────────────────────

def plot_V4():
    runs = read_csv(RESULTS_DIR / "V4_runs.csv")
    if not runs:
        return
    print("Plotting V4…")

    models     = ["claude", "gpt4o", "gemini"]
    techniques = ["zero_shot", "cot", "structured"]

    # Build quality and accuracy matrices
    q_mat  = np.full((len(models), len(techniques)), np.nan)
    ac_mat = np.full((len(models), len(techniques)), np.nan)
    for i, m in enumerate(models):
        for j, t in enumerate(techniques):
            cell = [r for r in runs if r["model"] == m and r["technique"] == t and not r["error"]]
            if cell:
                q_mat[i, j]  = np.mean([flt(r, "quality_score") for r in cell])
                ac_mat[i, j] = np.mean([flt(r, "s3_risk") for r in cell])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("V4 — Model × Prompt Technique Interaction (3×3 Factorial)",
                 fontsize=13, fontweight="bold")

    heatmap(axes[0], q_mat, models, techniques,
            title="Quality Score (0–4)", vmin=0, vmax=4)
    heatmap(axes[1], ac_mat, models, techniques,
            title="Accuracy (risk correct)", vmin=0, vmax=1)

    # Δ gains over zero_shot per model
    ax = axes[2]
    x  = np.arange(len(models))
    w  = 0.3
    for ki, (tech, col) in enumerate(zip(["cot", "structured"], [PALETTE[1], PALETTE[2]])):
        deltas = []
        for m in models:
            z = [flt(r, "quality_score") for r in runs
                 if r["model"] == m and r["technique"] == "zero_shot" and not r["error"]]
            c = [flt(r, "quality_score") for r in runs
                 if r["model"] == m and r["technique"] == tech and not r["error"]]
            deltas.append((np.mean(c) - np.mean(z)) if z and c else 0)
        ax.bar(x + ki*w, deltas, w, label=f"Δ {tech}", color=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + w/2); ax.set_xticklabels(models)
    ax.set_title("Quality Δ vs zero_shot per Model")
    ax.set_ylabel("Δ quality score"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "V4_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V5 — YOLO Threshold Sweep
# ─────────────────────────────────────────────────────────────────────────────

def plot_V5():
    roc = read_csv(RESULTS_DIR / "V5_roc.csv")
    if not roc:
        return
    print("Plotting V5…")

    thresholds = [flt(r, "threshold") for r in roc]
    precision  = [flt(r, "precision")         for r in roc]
    recall     = [flt(r, "recall")            for r in roc]
    f1         = [flt(r, "f1")                for r in roc]
    accuracy   = [flt(r, "accuracy")          for r in roc]
    far        = [flt(r, "false_alarm_rate")  for r in roc]
    mr         = [flt(r, "miss_rate")         for r in roc]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("V5 — YOLO Confidence Threshold Sweep", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(thresholds, precision, marker="o", color=PALETTE[0], label="Precision")
    ax.plot(thresholds, recall,    marker="s", color=PALETTE[1], label="Recall")
    ax.plot(thresholds, f1,        marker="^", color=PALETTE[2], label="F1", linewidth=2)
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(thresholds, accuracy, marker="o", color=PALETTE[0], label="Accuracy")
    ax.plot(thresholds, far,      marker="s", color=PALETTE[3], label="False Alarm Rate")
    ax.plot(thresholds, mr,       marker="^", color=PALETTE[4], label="Miss Rate")
    ax.set_title("Accuracy / FAR / Miss Rate vs Threshold")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # PR curve (recall on x, precision on y)
    ax = axes[2]
    ax.plot(recall, precision, marker="o", color=PALETTE[2])
    for r, p, t in zip(recall, precision, thresholds):
        ax.annotate(f"{t:.2f}", (r, p), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color="#333")
    best_f1_idx = int(np.argmax(f1))
    ax.scatter([recall[best_f1_idx]], [precision[best_f1_idx]],
               s=100, color="red", zorder=5, label=f"Best F1={f1[best_f1_idx]:.3f}")
    ax.set_title("Precision–Recall Curve"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "V5_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V6 — Verbosity vs Quality
# ─────────────────────────────────────────────────────────────────────────────

def plot_V6():
    rows = read_csv(RESULTS_DIR / "V6_summary.csv")
    if not rows:
        return
    print("Plotting V6…")

    max_toks = [int(flt(r, "max_tokens")) for r in rows]
    q        = [flt(r, "quality")   for r in rows]
    q_lo     = [flt(r, "q_lo")      for r in rows]
    q_hi     = [flt(r, "q_hi")      for r in rows]
    cost     = [flt(r, "cost_usd")  for r in rows]
    cost_lo  = [flt(r, "cost_lo")   for r in rows]
    cost_hi  = [flt(r, "cost_hi")   for r in rows]
    eff      = [flt(r, "efficiency")      for r in rows]
    trunc    = [flt(r, "truncation_rate") for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("V6 — Verbosity vs Quality Tradeoff (Claude)", fontsize=13, fontweight="bold")

    line_with_ci(axes[0,0], max_toks, q, q_lo, q_hi,
                 title="Quality Score vs max_tokens",
                 xlabel="max_tokens", ylabel="Quality (0–4)")
    axes[0,0].set_ylim(0, 4.2)

    axes[0,1].bar(range(len(max_toks)), trunc, color=PALETTE[3], width=0.55)
    axes[0,1].set_xticks(range(len(max_toks)))
    axes[0,1].set_xticklabels([str(t) for t in max_toks])
    axes[0,1].set_title("Truncation Rate vs max_tokens")
    axes[0,1].set_ylabel("Truncation rate"); axes[0,1].set_ylim(0, 1)
    axes[0,1].grid(axis="y", alpha=0.3)

    line_with_ci(axes[1,0], max_toks, cost, cost_lo, cost_hi,
                 color=PALETTE[1],
                 title="Cost vs max_tokens",
                 xlabel="max_tokens", ylabel="Cost (USD)")

    axes[1,1].plot(max_toks, eff, marker="o", color=PALETTE[2], linewidth=1.8)
    best_i = int(np.argmax(eff))
    axes[1,1].scatter([max_toks[best_i]], [eff[best_i]], s=100, color="red",
                      zorder=5, label=f"Sweet spot: {max_toks[best_i]}")
    axes[1,1].set_title("Efficiency (quality / USD) vs max_tokens")
    axes[1,1].set_xlabel("max_tokens"); axes[1,1].set_ylabel("quality / USD")
    axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "V6_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V7 — Scene Context History
# ─────────────────────────────────────────────────────────────────────────────

def plot_V7():
    summary = read_csv(RESULTS_DIR / "V7_summary.csv")
    runs    = read_csv(RESULTS_DIR / "V7_runs.csv")
    if not summary:
        return
    print("Plotting V7…")

    modes = [r["history_mode"] for r in summary]
    cols  = PALETTE[:len(modes)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("V7 — Scene Context History Effect (Claude)", fontsize=13, fontweight="bold")

    # Risk accuracy
    acc, acc_e = err_bars(summary, "risk_accuracy", "acc_lo", "acc_hi")
    bar_with_ci(axes[0,0], modes, acc, acc_e, cols,
                title="Overall Risk Accuracy", ylabel="Accuracy (Wilson CI)")
    axes[0,0].set_ylim(0, 1.15)

    # Change detection
    cd, cd_e = err_bars(summary, "change_detect", "cd_lo", "cd_hi")
    bar_with_ci(axes[0,1], modes, cd, cd_e, cols,
                title="Change Detection Rate (frames 3 & 5)", ylabel="Detection rate (Wilson CI)")
    axes[0,1].set_ylim(0, 1.15)

    # Mean input tokens per mode
    toks = [flt(r, "mean_tokens") for r in summary]
    bar_with_ci(axes[1,0], modes, toks, [[0]*len(modes), [0]*len(modes)], cols,
                title="Mean Input Tokens per Call", ylabel="Input tokens")

    # Per-frame risk accuracy across modes
    ax = axes[1,1]
    frame_nums = [1, 2, 3, 4, 5]
    for i, mode in enumerate(modes):
        accs = []
        for fn in frame_nums:
            fr = [r for r in runs if r["history_mode"] == mode
                  and int(flt(r, "frame_num")) == fn and not r["error"]]
            if fr:
                accs.append(sum(flt(r, "risk_correct") for r in fr) / len(fr))
            else:
                accs.append(float("nan"))
        ax.plot(frame_nums, accs, marker="o", color=cols[i], label=mode, linewidth=1.8)
    ax.axvline(2.5, color="grey", linestyle="--", linewidth=0.9, alpha=0.7, label="change events")
    ax.axvline(4.5, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("Per-Frame Risk Accuracy by History Mode")
    ax.set_xlabel("Frame number"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15); ax.set_xticks(frame_nums)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "V7_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V8 — Temperature Sweep
# ─────────────────────────────────────────────────────────────────────────────

def plot_V8():
    rows = read_csv(RESULTS_DIR / "V8_summary.csv")
    if not rows:
        return
    print("Plotting V8…")

    temps   = [flt(r, "temperature")    for r in rows]
    q       = [flt(r, "quality_mean")   for r in rows]
    q_lo    = [flt(r, "quality_lo")     for r in rows]
    q_hi    = [flt(r, "quality_hi")     for r in rows]
    acc     = [flt(r, "accuracy")       for r in rows]
    acc_lo  = [flt(r, "accuracy_lo")    for r in rows]
    acc_hi  = [flt(r, "accuracy_hi")    for r in rows]
    cons    = [flt(r, "consistency_std")for r in rows]
    flips   = [flt(r, "label_flip_rate")for r in rows]
    cost    = [flt(r, "cost_usd_mean")  for r in rows]
    cost_lo = [flt(r, "cost_usd_lo")    for r in rows]
    cost_hi = [flt(r, "cost_usd_hi")    for r in rows]

    # Best temperature
    scores = [a - f for a, f in zip(acc, flips)]
    best_t = temps[int(np.argmax(scores))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("V8 — Temperature Sweep (Claude, zero-shot, max_tokens=200)",
                 fontsize=13, fontweight="bold")

    line_with_ci(axes[0,0], temps, q, q_lo, q_hi,
                 color=PALETTE[0],
                 title="Quality Score vs Temperature",
                 xlabel="temperature", ylabel="Quality (0–4)")
    axes[0,0].set_ylim(0, 4.2)

    line_with_ci(axes[0,1], temps, acc, acc_lo, acc_hi,
                 color=PALETTE[1],
                 title="Classification Accuracy vs Temperature",
                 xlabel="temperature", ylabel="Accuracy (Wilson CI)")
    axes[0,1].set_ylim(0, 1.1)

    axes[0,2].plot(temps, cons, marker="o", color=PALETTE[2], linewidth=1.8)
    axes[0,2].fill_between(temps, cons, 0, alpha=0.15, color=PALETTE[2])
    axes[0,2].set_title("Consistency Std (↓ = better)")
    axes[0,2].set_xlabel("temperature"); axes[0,2].set_ylabel("σ quality across runs")
    axes[0,2].grid(alpha=0.3)

    axes[1,0].plot(temps, [f*100 for f in flips], marker="s", color=PALETTE[3], linewidth=1.8)
    axes[1,0].fill_between(temps, [f*100 for f in flips], 0, alpha=0.15, color=PALETTE[3])
    axes[1,0].set_title("Label Flip Rate (↓ = more stable)")
    axes[1,0].set_xlabel("temperature"); axes[1,0].set_ylabel("Flip rate (%)"); axes[1,0].grid(alpha=0.3)

    line_with_ci(axes[1,1], temps, cost, cost_lo, cost_hi,
                 color=PALETTE[4],
                 title="Cost per Call vs Temperature",
                 xlabel="temperature", ylabel="Cost (USD)")

    # Combined score (accuracy − flip rate)
    ax = axes[1,2]
    ax.bar(temps, scores, width=0.12, color=[
        "green" if t == best_t else PALETTE[0] for t in temps])
    ax.set_title(f"Combined Score (accuracy − flip rate)\nRecommended: t={best_t:.1f}")
    ax.set_xlabel("temperature"); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "V8_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# V9 — Model × Temperature × Max-Tokens Full Factorial
# ─────────────────────────────────────────────────────────────────────────────

def plot_V9():
    rows = read_csv(RESULTS_DIR / "V9_summary.csv")
    if not rows:
        return
    print("Plotting V9…")

    models     = ["claude", "gpt4o", "gemini"]
    temps      = [0.0, 0.5, 1.0]
    max_toks   = [128, 256, 512]

    # ── Figure 1: Quality heatmaps (one per model) ──────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle("V9 — Quality Score Heatmap: Temperature × Max-tokens per Model",
                  fontsize=13, fontweight="bold")

    all_q = []
    cell_q = {}
    for r in rows:
        key = (r["model"], flt(r, "temperature"), int(flt(r, "max_tokens")))
        cell_q[key] = flt(r, "quality_mean")
        all_q.append(flt(r, "quality_mean"))
    vmin, vmax = min(all_q) if all_q else 0, max(all_q) if all_q else 4

    for ax, model in zip(axes1, models):
        mat = np.full((len(temps), len(max_toks)), np.nan)
        for i, t in enumerate(temps):
            for j, mt in enumerate(max_toks):
                v = cell_q.get((model, t, mt))
                if v is not None:
                    mat[i, j] = v
        im = heatmap(ax, mat,
                     [f"t={t}" for t in temps],
                     [f"tok={mt}" for mt in max_toks],
                     title=model.upper(), vmin=vmin, vmax=vmax)
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Quality (0–4)")

    plt.tight_layout()
    save(fig1, "V9_plot_quality_heatmaps.png")

    # ── Figure 2: Accuracy heatmaps ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("V9 — Accuracy Heatmap: Temperature × Max-tokens per Model",
                  fontsize=13, fontweight="bold")

    cell_acc = {}
    all_acc  = []
    for r in rows:
        key = (r["model"], flt(r, "temperature"), int(flt(r, "max_tokens")))
        cell_acc[key] = flt(r, "accuracy")
        all_acc.append(flt(r, "accuracy"))
    vmin_a, vmax_a = min(all_acc) if all_acc else 0, 1.0

    for ax, model in zip(axes2, models):
        mat = np.full((len(temps), len(max_toks)), np.nan)
        for i, t in enumerate(temps):
            for j, mt in enumerate(max_toks):
                v = cell_acc.get((model, t, mt))
                if v is not None:
                    mat[i, j] = v
        im = heatmap(ax, mat,
                     [f"t={t}" for t in temps],
                     [f"tok={mt}" for mt in max_toks],
                     title=model.upper(), vmin=vmin_a, vmax=vmax_a)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")

    plt.tight_layout()
    save(fig2, "V9_plot_accuracy_heatmaps.png")

    # ── Figure 3: Interaction delta bars ────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle("V9 — Interaction Effects", fontsize=13, fontweight="bold")

    # Δ_temp(0→1) per model at max_tokens=256
    ax = axes3[0]
    delta_temp = []
    for model in models:
        q0 = cell_q.get((model, 0.0, 256), np.nan)
        q1 = cell_q.get((model, 1.0, 256), np.nan)
        delta_temp.append(q1 - q0 if not np.isnan(q0) and not np.isnan(q1) else 0)
    bars = ax.bar(models, delta_temp,
                  color=[PALETTE[0] if d >= 0 else PALETTE[3] for d in delta_temp],
                  width=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Δ Quality: t=1.0 vs t=0.0\n(max_tokens=256 fixed)")
    ax.set_ylabel("Δ quality score (negative = temp hurts)"); ax.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, delta_temp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{d:+.2f}", ha="center", va="bottom", fontsize=9)

    # Δ_tokens(128→512) per model at temp=0.5
    ax = axes3[1]
    delta_tok = []
    for model in models:
        q128 = cell_q.get((model, 0.5, 128), np.nan)
        q512 = cell_q.get((model, 0.5, 512), np.nan)
        delta_tok.append(q512 - q128 if not np.isnan(q128) and not np.isnan(q512) else 0)
    bars = ax.bar(models, delta_tok,
                  color=[PALETTE[2] if d >= 0 else PALETTE[3] for d in delta_tok],
                  width=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Δ Quality: max_tokens=512 vs 128\n(temperature=0.5 fixed)")
    ax.set_ylabel("Δ quality score"); ax.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, delta_tok):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{d:+.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save(fig3, "V9_plot_interactions.png")

    # ── Figure 4: Model gap vs temperature ──────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    fig4.suptitle("V9 — Model Quality Gap vs Temperature (max_tokens=256)",
                  fontsize=13, fontweight="bold")

    for model, col in zip(models, PALETTE):
        qs = []
        for t in temps:
            q = cell_q.get((model, t, 256), np.nan)
            qs.append(q)
        ax4.plot(temps, qs, marker="o", color=col, label=model, linewidth=1.8)

    ax4.set_xlabel("Temperature"); ax4.set_ylabel("Mean quality score")
    ax4.set_ylim(0, 4.2); ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    plt.tight_layout()
    save(fig4, "V9_plot_model_gap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

PLOTTERS = {
    "V1": plot_V1,
    "V2": plot_V2,
    "V3": plot_V3,
    "V4": plot_V4,
    "V5": plot_V5,
    "V6": plot_V6,
    "V7": plot_V7,
    "V8": plot_V8,
    "V9": plot_V9,
}

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(PLOTTERS)
    targets = [t.upper() for t in targets]
    invalid = [t for t in targets if t not in PLOTTERS]
    if invalid:
        print(f"Unknown experiment(s): {invalid}. Valid: {list(PLOTTERS)}")
        sys.exit(1)

    print(f"Generating plots for: {targets}")
    print(f"Output directory    : {RESULTS_DIR}")
    print()
    for key in targets:
        PLOTTERS[key]()
    print("\nDone.")
