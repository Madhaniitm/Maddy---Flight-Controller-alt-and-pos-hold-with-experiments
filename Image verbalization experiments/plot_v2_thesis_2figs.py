"""
plot_v2_thesis_2figs.py
=======================
Two crisp thesis-quality figures for V2 prompt technique comparison.

Fig A — Why Structured JSON is best
        Grouped bars: LblAcc | RsnAct | ActSafe + Danger marker per technique

Fig B — CoT-600 cost: output tokens ≈ 2.25× structured, cost ≈ 1.45×
        Twin-axis: output tokens (bars) + cost per 1000 calls (line)

Run:
    /opt/homebrew/bin/python3.11 "Image verbalization experiments/plot_v2_thesis_2figs.py"
"""

import pathlib, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RES  = pathlib.Path(__file__).parent / "results"
OUT  = RES / "V2_plots"
OUT.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
rows = []
with open(RES / "V2_runs_20260525_035756.csv") as f:
    rows = list(csv.DictReader(f))

cot600 = []
with open(RES / "V2_cot_rerun_20260525_175408.csv") as f:
    cot600 = list(csv.DictReader(f))

# ── Helpers ───────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = k / n
    lo = (p + z**2/(2*n) - z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1+z**2/n)
    hi = (p + z**2/(2*n) + z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1+z**2/n)
    return p, max(0, lo), min(1, hi)

def mean_ci(vals):
    n = len(vals)
    mu = sum(vals)/n
    lo, hi = wilson_ci(int(round(mu*n)), n)[1:3]
    return mu, lo, hi

# ── Aggregate per technique ───────────────────────────────────────────────────
TECHS = ["zero_shot", "few_shot_3", "cot", "cot_600", "structured"]
LABELS = ["Zero-Shot", "Few-Shot-3", "CoT-300\n(67% trunc.)", "CoT-600", "Structured\nJSON"]
COLORS = ["#7fb3d3", "#a8d5a2", "#f0b27a", "#e88989", "#5dade2"]
HIGHLIGHT = [False, False, False, False, True]

data = {}
for tech in TECHS:
    if tech == "cot_600":
        tr = cot600
    else:
        tr = [r for r in rows if r["technique"] == tech]

    valid = [r for r in tr if r.get("reply","").strip()]
    n = len(valid)

    # Quality metrics
    s1 = [int(r["s1_scene"]) for r in valid] if "s1_scene" in valid[0] else []
    s2 = [int(r["s2_proximity"]) for r in valid] if "s2_proximity" in valid[0] else []
    s3 = [int(r["s3_risk"]) for r in valid]
    s5 = [int(r["s5_pilot_action"]) for r in valid] if "s5_pilot_action" in valid[0] else []
    sa = [int(r["action_safe"]) for r in valid]
    dg = sum(int(r["action_dangerous"]) for r in valid)

    # Compute s1/s5 from reply for cot_600 (not stored in CSV)
    if tech == "cot_600":
        SCENE_KW = {"see","observe","wall","obstacle","object","floor","ceiling",
                    "surface","dark","bright","blurry","clear","colour","color",
                    "person","table","door","lamp","light","room","outdoor","window",
                    "laptop","chair","bag","box","covered","hand","partial"}
        PROX_KW  = {"cm","mm","metre","meter","distance","close","near","far",
                    "proxim","within","away","behind","front","side","left","right",
                    "above","below","approximately","roughly","about","adjacent"}
        PILOT_ACT = ("pitch_forward","pitch forward","pitch_back","pitch back",
                     "hover","ascend","descend","land","stop","slow_down","hold","proceed")
        s1 = [1 if any(k in r["reply"].lower() for k in SCENE_KW) else 0 for r in valid]
        s2 = [1 if any(k in r["reply"].lower() for k in PROX_KW)  else 0 for r in valid]
        s5 = [1 if any(k in r["reply"].lower() for k in PILOT_ACT) else 0 for r in valid]

    lbl_mu, lbl_lo, lbl_hi = mean_ci(s3)
    act_mu, act_lo, act_hi = mean_ci(s5) if s5 else (0,0,0)
    saf_mu, saf_lo, saf_hi = mean_ci(sa)

    # Cost / latency / tokens
    lat  = [float(r["latency_ms"]) for r in valid if r.get("latency_ms")]
    cost = [float(r["cost_usd"])   for r in valid if r.get("cost_usd")]
    otok = [int(r["output_tokens"]) for r in valid if r.get("output_tokens")]

    data[tech] = dict(
        n=n, dg=dg,
        lbl=(lbl_mu, lbl_lo, lbl_hi),
        act=(act_mu, act_lo, act_hi),
        saf=(saf_mu, saf_lo, saf_hi),
        lat=sum(lat)/len(lat) if lat else 0,
        cost=sum(cost)/len(cost)*1000 if cost else 0,
        otok=sum(otok)/len(otok) if otok else 0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIG A — Why Structured JSON is best
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))

x   = np.arange(len(TECHS))
w   = 0.22
off = [-w, 0, w]

METRICS = [
    ("lbl", "LblAcc",  "#4a90d9"),
    ("act", "RsnAct",  "#e07b54"),
    ("saf", "ActSafe", "#6cbf6c"),
]

for i, (key, label, col) in enumerate(METRICS):
    vals  = [data[t][key][0]*100 for t in TECHS]
    lo_err = [data[t][key][0]*100 - data[t][key][1]*100 for t in TECHS]
    hi_err = [data[t][key][2]*100 - data[t][key][0]*100 for t in TECHS]
    bars = ax.bar(x + off[i], vals, w,
                  label=label, color=col, alpha=0.88,
                  yerr=[lo_err, hi_err], capsize=3,
                  error_kw=dict(elinewidth=0.8, capthick=0.8))

# Danger markers above each group
for j, tech in enumerate(TECHS):
    dg = data[tech]["dg"]
    if dg > 0:
        ax.annotate(f"⚠ {dg} danger",
                    xy=(x[j], 103), ha="center", fontsize=7.5,
                    color="#c0392b", fontweight="bold")

# Highlight Structured
ax.axvspan(x[-1]-0.38, x[-1]+0.38, alpha=0.08, color="#5dade2", zorder=0)
ax.annotate("Best", xy=(x[-1], 97), ha="center", fontsize=8,
            color="#1a6fa0", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_ylim(0, 112)
ax.set_title("V2 Prompt Technique Comparison — Quality Metrics\n"
             "(4 models × 8 scenes × 5 runs; error bars = 95% Wilson CI)",
             fontsize=10)
ax.legend(fontsize=9, loc="lower right")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fig_v2_A_quality.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_v2_A_quality.png", dpi=150, bbox_inches="tight")
print("Saved fig_v2_A_quality")


# ══════════════════════════════════════════════════════════════════════════════
# FIG B — CoT-600 output tokens and cost vs other techniques
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 4))

otoks = [data[t]["otok"] for t in TECHS]
costs = [data[t]["cost"] for t in TECHS]

bar_colors = COLORS[:]
# Bold red outline for CoT-600
bars = ax1.bar(x, otoks, 0.55, color=bar_colors, alpha=0.85,
               edgecolor=["none"]*4+["none"],
               linewidth=1.5, zorder=2)
# Red edge on CoT-600 bar
bars[3].set_edgecolor("#c0392b")
bars[3].set_linewidth(2.0)

ax1.set_ylabel("Mean Output Tokens", fontsize=10, color="#333333")
ax1.set_ylim(0, max(otoks)*1.35)
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=9)
ax1.yaxis.grid(True, linestyle="--", alpha=0.35)
ax1.set_axisbelow(True)
ax1.spines[["top","right"]].set_visible(False)

# Secondary axis — cost
ax2 = ax1.twinx()
ax2.plot(x, costs, "o-", color="#c0392b", linewidth=2, markersize=6,
         label="Cost / 1000 calls ($)", zorder=3)
ax2.set_ylabel("Cost per 1000 calls (USD)", fontsize=10, color="#c0392b")
ax2.tick_params(axis="y", colors="#c0392b")
ax2.spines["right"].set_edgecolor("#c0392b")
ax2.set_ylim(0, max(costs)*1.45)
ax2.spines[["top","left"]].set_visible(False)

# Annotations: ratio vs structured
struct_tok  = data["structured"]["otok"]
struct_cost = data["structured"]["cost"]
for j, tech in enumerate(TECHS):
    tok_r  = data[tech]["otok"]  / struct_tok
    cost_r = data[tech]["cost"]  / struct_cost
    ax1.text(x[j], otoks[j] + 6,
             f"{otoks[j]:.0f} tok\n({tok_r:.1f}×)",
             ha="center", va="bottom", fontsize=7.5,
             color="#c0392b" if tech=="cot_600" else "#444444",
             fontweight="bold" if tech=="cot_600" else "normal")

ax1.set_title("V2 Output Tokens and Cost per Technique\n"
              "(CoT-600 uses 2.25× more tokens than Structured JSON → 1.45× higher cost)",
              fontsize=10)

# Legend
tok_patch = mpatches.Patch(color="#a0c4e8", label="Output tokens (bars)")
cost_line = plt.Line2D([0],[0], color="#c0392b", linewidth=2,
                       marker="o", label="Cost / 1000 calls (line)")
ax1.legend(handles=[tok_patch, cost_line], fontsize=9, loc="upper left")

plt.tight_layout()
fig.savefig(OUT / "fig_v2_B_cost.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_v2_B_cost.png", dpi=150, bbox_inches="tight")
print("Saved fig_v2_B_cost")
print(f"\nBoth figures saved to: {OUT}")
