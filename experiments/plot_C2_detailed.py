"""
plot_C2_detailed.py  —  Comprehensive diagnostic plots for EXP-C2
==================================================================
Generates 9 figures covering every measurable aspect of the C2 experiment
(Ambiguity Resolution — 6 commands from explicit to indirect).

Figures saved to experiments/results/:
  C2_fig1_accuracy_degradation.png   Accuracy + Wilson CI per command (the headline result)
  C2_fig2_outcome_heatmap.png        Pass/Fail/TextInference heatmap: run × command
  C2_fig3_altitude_trajectory.png    Per-run altitude state evolution across all 6 commands
  C2_fig4_increment_analysis.png     Actual altitude increments per command — pass vs fail coloured
  C2_fig5_tool_source_split.png      set_altitude_target vs text_inference usage per command
  C2_fig6_run_divergence.png         How each run's drone state diverges after Cmd3/Cmd5 failures
  C2_fig7_token_api_analysis.png     Token usage and API calls per command — shows text_inference collapse
  C2_fig8_cmd3_failure_deep_dive.png z_before vs z_after scatter for Cmd3: LLM sets target ≈ current alt
  C2_fig9_cmd5_interpretation.png    Per-run "safe height" targets chosen by LLM, with context
"""

import pathlib, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS = pathlib.Path(__file__).parent / "results"

def rcsv(name):
    p = RESULTS / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save(fig, name):
    out = RESULTS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.name}")

def flt(row, key, default=0.0):
    try: return float(row[key])
    except: return default

def intv(row, key, default=0):
    try: return int(row[key])
    except: return default

# ── Palette ───────────────────────────────────────────────────────────────────
C_PASS     = "#2ECC71"   # green
C_FAIL     = "#E74C3C"   # red
C_TEXTINF  = "#F39C12"   # amber  — text_inference (no flight action)
C_NEUTRAL  = "#BDC3C7"   # grey
RUN_COLS   = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9"]

# ── Constants ─────────────────────────────────────────────────────────────────
CMD_LABELS  = ["Cmd1\nexplicit", "Cmd2\nparaphrase", "Cmd3\nno-number",
               "Cmd4\nvague", "Cmd5\nabstract", "Cmd6\nindirect"]
CMD_SHORT   = ["Cmd1","Cmd2","Cmd3","Cmd4","Cmd5","Cmd6"]
CMD_TYPES   = ["explicit","paraphrase","relative_no_num","vague_relative","abstract","indirect"]
N_CMDS      = 6
N_RUNS      = 5

# ── Load data ─────────────────────────────────────────────────────────────────
run_rows = rcsv("C2_runs.csv")
sum_rows = [r for r in rcsv("C2_summary.csv")
            if r.get("cmd_idx","").isdigit()]

if not run_rows:
    print("ERROR: C2_runs.csv not found — run the experiment first.")
    raise SystemExit(1)

# Organise into grid[run][cmd_idx] — 1-indexed
grid = {}
for r in run_rows:
    try:
        run = int(r["run"]); cmd = int(r["cmd_idx"])
        grid[(run, cmd)] = r
    except: pass

def get(run, cmd, key, default=0.0):
    return flt(grid.get((run, cmd), {}), key, default)

def gets(run, cmd, key, default=""):
    return grid.get((run, cmd), {}).get(key, default)

# Summary arrays
acc   = [flt(r,"success_rate")  for r in sum_rows]
acc_lo= [flt(r,"wilson_ci_lo")  for r in sum_rows]
acc_hi= [flt(r,"wilson_ci_hi")  for r in sum_rows]


# =============================================================================
# Figure 1 — Accuracy Degradation Curve
# =============================================================================
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("C2 — Ambiguity Resolution: Accuracy Degradation",
                 fontsize=13, fontweight="bold")

    # Bar chart with Wilson CI error bars
    ax = axes[0]
    x  = np.arange(N_CMDS)
    cols = [C_PASS if a == 1.0 else C_FAIL if a == 0.0 else "#E67E22" for a in acc]
    bars = ax.bar(x, acc, color=cols, width=0.55,
                  yerr=[[a-lo for a,lo in zip(acc,acc_lo)],
                        [hi-a for a,hi in zip(acc,acc_hi)]],
                  capsize=5, error_kw={"elinewidth":1.4,"ecolor":"#333"})
    ax.set_xticks(x); ax.set_xticklabels(CMD_LABELS, fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Success rate (Wilson 95% CI)")
    ax.set_title("Success Rate per Command Type", fontweight="bold")
    ax.axhline(0.5, color="#555", linewidth=0.8, linestyle="--", alpha=0.5,
               label="50% chance baseline")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    for bar, a, lo, hi in zip(bars, acc, acc_lo, acc_hi):
        ax.text(bar.get_x()+bar.get_width()/2, a+0.05,
                f"{a:.0%}\n[{lo:.2f},{hi:.2f}]",
                ha="center", va="bottom", fontsize=8)

    # n_correct raw counts
    ax2 = axes[1]
    n_correct = [int(flt(r,"n_correct")) for r in sum_rows]
    n_wrong   = [N_RUNS - n for n in n_correct]
    ax2.bar(x, n_correct, color=C_PASS, width=0.55, label="Correct")
    ax2.bar(x, n_wrong,   bottom=n_correct, color=C_FAIL, width=0.55, label="Incorrect")
    ax2.set_xticks(x); ax2.set_xticklabels(CMD_LABELS, fontsize=9)
    ax2.set_ylabel("Count (N=5 runs)"); ax2.set_yticks(range(6))
    ax2.set_title("Raw Pass / Fail Counts per Command", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    for xi, (nc, nw) in enumerate(zip(n_correct, n_wrong)):
        ax2.text(xi, nc/2, f"{nc}", ha="center", va="center",
                 color="white", fontweight="bold", fontsize=11)
        if nw:
            ax2.text(xi, nc + nw/2, f"{nw}", ha="center", va="center",
                     color="white", fontweight="bold", fontsize=11)

    plt.tight_layout()
    save(fig, "C2_fig1_accuracy_degradation.png")


# =============================================================================
# Figure 2 — Outcome Heatmap (run × command)
# =============================================================================
def fig2():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("C2 — Per-Run × Per-Command Outcome Heatmap",
                 fontsize=13, fontweight="bold")

    # Encode: 2=pass, 1=set_altitude_target but wrong, 0=text_inference
    mat = np.zeros((N_RUNS, N_CMDS))
    label_mat = [[""]*N_CMDS for _ in range(N_RUNS)]
    for run in range(1, N_RUNS+1):
        for cmd in range(1, N_CMDS+1):
            correct = intv(grid.get((run,cmd),{}), "correct")
            source  = gets(run, cmd, "target_source")
            if correct == 1:
                mat[run-1, cmd-1] = 2
                label_mat[run-1][cmd-1] = "PASS"
            elif source == "text_inference":
                mat[run-1, cmd-1] = 0
                label_mat[run-1][cmd-1] = "TEXT\nINF"
            else:
                mat[run-1, cmd-1] = 1
                label_mat[run-1][cmd-1] = "FAIL\n(set_alt)"

    cmap = matplotlib.colors.ListedColormap([C_TEXTINF, C_FAIL, C_PASS])
    im   = ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(range(N_CMDS))
    ax.set_xticklabels(CMD_LABELS, fontsize=9)
    ax.set_yticks(range(N_RUNS))
    ax.set_yticklabels([f"Run {r}" for r in range(1, N_RUNS+1)], fontsize=9)

    for i in range(N_RUNS):
        for j in range(N_CMDS):
            ax.text(j, i, label_mat[i][j], ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    patches = [mpatches.Patch(color=C_PASS,    label="Pass (correct target)"),
               mpatches.Patch(color=C_FAIL,    label="Fail — wrong set_altitude_target"),
               mpatches.Patch(color=C_TEXTINF, label="Fail — text_inference (no flight action)")]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              bbox_to_anchor=(1.0, -0.18), ncol=3)

    ax.set_title("Each cell: run × command outcome. Amber=text_inference, Red=wrong target, Green=pass",
                 fontsize=9)
    plt.tight_layout()
    save(fig, "C2_fig2_outcome_heatmap.png")


# =============================================================================
# Figure 3 — Altitude Trajectory Across All 6 Commands (per run)
# =============================================================================
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("C2 — Altitude State Evolution Across 6 Commands (per run)",
                 fontsize=13, fontweight="bold")

    # Left: all runs together — spaghetti showing divergence
    ax = axes[0]
    for run in range(1, N_RUNS+1):
        zs = [get(run, 1, "z_before_m")]  # starting altitude
        for cmd in range(1, N_CMDS+1):
            zs.append(get(run, cmd, "z_after_m"))
        x_pts = list(range(N_CMDS+1))
        ax.plot(x_pts, zs, marker="o", color=RUN_COLS[run-1],
                linewidth=1.8, label=f"Run {run}")
        # Mark failures with X
        for cmd in range(1, N_CMDS+1):
            correct = intv(grid.get((run,cmd),{}), "correct")
            if not correct:
                ax.scatter([cmd], [get(run,cmd,"z_after_m")],
                           marker="x", color=RUN_COLS[run-1], s=80, zorder=5, linewidths=2)

    ax.set_xticks(range(N_CMDS+1))
    ax.set_xticklabels(["Start"]+CMD_SHORT, fontsize=9)
    ax.set_ylabel("Altitude (m)"); ax.set_xlabel("Command sequence")
    ax.set_title("All Runs — × marks failed commands")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axhline(2.0, color="#555", linewidth=0.7, linestyle="--", alpha=0.5, label="2 m ref")
    ax.axhline(2.5, color="#999", linewidth=0.7, linestyle="--", alpha=0.5, label="2.5 m ref")

    # Right: delta (z_after - z_before) per command per run as grouped dots
    ax2 = axes[1]
    x = np.arange(N_CMDS)
    for run in range(1, N_RUNS+1):
        deltas = [get(run, cmd, "increment_m") for cmd in range(1, N_CMDS+1)]
        correct_flags = [intv(grid.get((run,cmd),{}), "correct") for cmd in range(1, N_CMDS+1)]
        for xi, (d, ok) in enumerate(zip(deltas, correct_flags)):
            marker = "o" if ok else "x"
            ax2.scatter(xi + (run-3)*0.1, d, marker=marker,
                        color=RUN_COLS[run-1], s=60, zorder=4,
                        linewidths=1.5 if not ok else 0)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(CMD_LABELS, fontsize=9)
    ax2.set_ylabel("Altitude increment (m)"); ax2.set_xlabel("Command")
    ax2.set_title("Altitude Increment per Command\n(dot=pass, ×=fail, colour=run)")
    ax2.grid(alpha=0.3)
    for run in range(1, N_RUNS+1):
        ax2.scatter([], [], color=RUN_COLS[run-1], marker="o",
                    label=f"Run {run}")
    ax2.legend(fontsize=8, ncol=5)

    plt.tight_layout()
    save(fig, "C2_fig3_altitude_trajectory.png")


# =============================================================================
# Figure 4 — Increment Analysis (box + scatter per command)
# =============================================================================
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("C2 — Altitude Increment Applied per Command",
                 fontsize=13, fontweight="bold")

    increments_by_cmd = {}
    correct_by_cmd    = {}
    for cmd in range(1, N_CMDS+1):
        increments_by_cmd[cmd] = []
        correct_by_cmd[cmd]    = []
        for run in range(1, N_RUNS+1):
            row = grid.get((run, cmd))
            if row:
                increments_by_cmd[cmd].append(flt(row,"increment_m"))
                correct_by_cmd[cmd].append(intv(row,"correct"))

    # Box plot of increments per command
    ax = axes[0]
    data = [increments_by_cmd[c] for c in range(1, N_CMDS+1)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops={"color":"black","linewidth":2})
    box_cols = []
    for cmd in range(1, N_CMDS+1):
        n_c = sum(correct_by_cmd[cmd])
        col = C_PASS if n_c == 5 else (C_FAIL if n_c == 0 else "#E67E22")
        box_cols.append(col)
    for patch, col in zip(bp["boxes"], box_cols):
        patch.set_facecolor(col); patch.set_alpha(0.6)

    # Scatter individual runs over box
    for cmd_i, cmd in enumerate(range(1, N_CMDS+1)):
        for run in range(1, N_RUNS+1):
            row = grid.get((run, cmd))
            if row:
                y   = flt(row,"increment_m")
                ok  = intv(row,"correct")
                marker = "o" if ok else "x"
                ax.scatter(cmd_i+1 + (run-3)*0.08, y,
                           marker=marker, color=RUN_COLS[run-1],
                           s=50, zorder=5, linewidths=1.5 if not ok else 0)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(1, N_CMDS+1))
    ax.set_xticklabels(CMD_LABELS, fontsize=9)
    ax.set_ylabel("Altitude increment (m)")
    ax.set_title("Increment Distribution per Command\n(box=all runs, dot=pass, ×=fail)")
    ax.grid(axis="y", alpha=0.3)

    # Mean ± std per command — pass vs fail separately
    ax2 = axes[1]
    x   = np.arange(N_CMDS)
    means_pass = []; means_fail = []; std_pass = []; std_fail = []
    for cmd in range(1, N_CMDS+1):
        pass_inc = [increments_by_cmd[cmd][i] for i,ok in enumerate(correct_by_cmd[cmd]) if ok]
        fail_inc = [increments_by_cmd[cmd][i] for i,ok in enumerate(correct_by_cmd[cmd]) if not ok]
        means_pass.append(np.mean(pass_inc) if pass_inc else np.nan)
        means_fail.append(np.mean(fail_inc) if fail_inc else np.nan)
        std_pass.append(np.std(pass_inc) if len(pass_inc)>1 else 0)
        std_fail.append(np.std(fail_inc) if len(fail_inc)>1 else 0)

    w = 0.3
    ax2.bar(x-w/2, means_pass, w, color=C_PASS, label="Mean increment (pass)",
            yerr=std_pass, capsize=4, error_kw={"elinewidth":1.2})
    ax2.bar(x+w/2, means_fail, w, color=C_FAIL, label="Mean increment (fail)",
            yerr=std_fail, capsize=4, error_kw={"elinewidth":1.2})
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(CMD_LABELS, fontsize=9)
    ax2.set_ylabel("Mean increment (m)")
    ax2.set_title("Mean Increment: Pass vs Fail Runs per Command\n(NaN bar = no data in that category)")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "C2_fig4_increment_analysis.png")


# =============================================================================
# Figure 5 — Tool Source Split per Command
# =============================================================================
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C2 — Tool Source: set_altitude_target vs text_inference",
                 fontsize=13, fontweight="bold")

    n_set = []; n_txt = []; n_set_wrong = []
    for cmd in range(1, N_CMDS+1):
        s = t = sw = 0
        for run in range(1, N_RUNS+1):
            row = grid.get((run, cmd))
            if row:
                source  = row.get("target_source","")
                correct = intv(row,"correct")
                if source == "text_inference":
                    t += 1
                elif correct:
                    s += 1
                else:
                    sw += 1
        n_set.append(s); n_set_wrong.append(sw); n_txt.append(t)

    x = np.arange(N_CMDS)
    ax = axes[0]
    ax.bar(x, n_set,       color=C_PASS,    width=0.55, label="set_alt_target (correct)")
    ax.bar(x, n_set_wrong, bottom=n_set,    color=C_FAIL,    width=0.55, label="set_alt_target (wrong)")
    ax.bar(x, n_txt,       bottom=[a+b for a,b in zip(n_set,n_set_wrong)],
           color=C_TEXTINF, width=0.55, label="text_inference (no flight action)")
    ax.set_xticks(x); ax.set_xticklabels(CMD_LABELS, fontsize=9)
    ax.set_ylabel("Count (N=5)"); ax.set_yticks(range(6))
    ax.set_title("How LLM Responded per Command\n(stacked by tool choice)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Rate of text_inference as line
    ax2 = axes[1]
    txt_rate = [t/N_RUNS for t in n_txt]
    set_rate = [(s+sw)/N_RUNS for s,sw in zip(n_set,n_set_wrong)]
    ax2.fill_between(range(N_CMDS), txt_rate, 0, alpha=0.35, color=C_TEXTINF,
                     label="text_inference rate")
    ax2.fill_between(range(N_CMDS), set_rate, 0, alpha=0.25, color=C_FAIL,
                     label="set_altitude_target rate (all)")
    ax2.plot(range(N_CMDS), txt_rate, marker="s", color=C_TEXTINF, linewidth=2.0)
    ax2.plot(range(N_CMDS), [a/N_RUNS for a in acc], marker="o",
             color=C_PASS, linewidth=2.0, label="accuracy (correct rate)")
    ax2.set_xticks(range(N_CMDS)); ax2.set_xticklabels(CMD_SHORT, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Rate"); ax2.set_title("text_inference Rate vs Accuracy")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "C2_fig5_tool_source_split.png")


# =============================================================================
# Figure 6 — Run Divergence: How States Fork After Failures
# =============================================================================
def fig6():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("C2 — Per-Run Altitude Profile (one panel per run)",
                 fontsize=13, fontweight="bold")

    for run in range(1, N_RUNS+1):
        ax  = axes[(run-1)//3][(run-1)%3]
        zs  = [get(run, 1, "z_before_m")]
        for cmd in range(1, N_CMDS+1):
            zs.append(get(run, cmd, "z_after_m"))

        # Draw segments coloured by outcome
        for cmd in range(1, N_CMDS+1):
            ok  = intv(grid.get((run,cmd),{}), "correct")
            src = gets(run, cmd, "target_source")
            if ok:
                col = C_PASS
            elif src == "text_inference":
                col = C_TEXTINF
            else:
                col = C_FAIL
            ax.plot([cmd-1, cmd], [zs[cmd-1], zs[cmd]],
                    color=col, linewidth=2.5, solid_capstyle="round")
            ax.scatter(cmd, zs[cmd], color=col, s=60, zorder=5)

        ax.scatter(0, zs[0], color="#555", s=60, zorder=5)
        ax.set_xticks(range(N_CMDS+1))
        ax.set_xticklabels(["Start"]+CMD_SHORT, fontsize=8)
        ax.set_ylabel("Alt (m)", fontsize=8)
        ax.set_title(f"Run {run}", fontsize=10, fontweight="bold")
        ax.set_ylim(0.8, 2.8)
        ax.axhline(2.0, color="#ccc", linewidth=0.8, linestyle="--")
        ax.axhline(2.5, color="#ddd", linewidth=0.8, linestyle="--")
        ax.grid(alpha=0.25)

        # Annotate the z value at each command step
        for cmd in range(1, N_CMDS+1):
            ax.text(cmd, zs[cmd]+0.05, f"{zs[cmd]:.2f}",
                    ha="center", va="bottom", fontsize=6.5)

    # Legend in 6th panel
    ax6 = axes[1][2]
    patches = [mpatches.Patch(color=C_PASS,    label="Correct (pass)"),
               mpatches.Patch(color=C_FAIL,    label="Wrong set_altitude_target"),
               mpatches.Patch(color=C_TEXTINF, label="text_inference (no action)")]
    ax6.legend(handles=patches, loc="center", fontsize=10)
    ax6.axis("off")
    ax6.set_title("Legend", fontsize=10)

    plt.tight_layout()
    save(fig, "C2_fig6_run_divergence.png")


# =============================================================================
# Figure 7 — Token Usage & API Calls per Command
# =============================================================================
def fig7():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("C2 — Token Usage & API Calls per Command",
                 fontsize=13, fontweight="bold")

    # Per command: mean tokens across runs
    mean_tokens = []; std_tokens = []; mean_api = []; std_api = []
    for cmd in range(1, N_CMDS+1):
        toks = [flt(grid.get((run,cmd),{}),"tokens") for run in range(1,N_RUNS+1)]
        apis = [flt(grid.get((run,cmd),{}),"api_calls") for run in range(1,N_RUNS+1)]
        mean_tokens.append(np.mean(toks)); std_tokens.append(np.std(toks))
        mean_api.append(np.mean(apis));    std_api.append(np.std(apis))

    x = np.arange(N_CMDS)
    cols_by_acc = [C_PASS if a==1.0 else C_FAIL if a==0.0 else "#E67E22" for a in acc]

    # Mean tokens per command
    ax = axes[0,0]
    ax.bar(x, mean_tokens, color=cols_by_acc, width=0.55,
           yerr=std_tokens, capsize=4, error_kw={"elinewidth":1.2})
    ax.set_xticks(x); ax.set_xticklabels(CMD_LABELS, fontsize=9)
    ax.set_ylabel("Mean tokens (input)"); ax.set_title("Mean Token Count per Command")
    ax.grid(axis="y", alpha=0.3)
    for xi, (m, s) in enumerate(zip(mean_tokens, std_tokens)):
        ax.text(xi, m+s+500, f"{m/1000:.0f}k", ha="center", fontsize=8)

    # Scatter of all run tokens per command
    ax2 = axes[0,1]
    for cmd_i, cmd in enumerate(range(1, N_CMDS+1)):
        for run in range(1, N_RUNS+1):
            row = grid.get((run,cmd))
            if row:
                tok = flt(row,"tokens")
                src = row.get("target_source","")
                marker = "o" if src != "text_inference" else "s"
                ax2.scatter(cmd_i + (run-3)*0.12, tok,
                            marker=marker, color=RUN_COLS[run-1], s=55, zorder=4)
    ax2.set_xticks(x); ax2.set_xticklabels(CMD_LABELS, fontsize=9)
    ax2.set_ylabel("Input tokens")
    ax2.set_title("Tokens per Run × Command\n(circle=set_alt_target, square=text_inference)")
    ax2.grid(alpha=0.3)
    for run in range(1,N_RUNS+1):
        ax2.scatter([],[], color=RUN_COLS[run-1], marker="o", label=f"Run {run}")
    ax2.legend(fontsize=7, ncol=5)

    # API calls per command — dramatic collapse at Cmd5/6
    ax3 = axes[1,0]
    ax3.bar(x, mean_api, color=cols_by_acc, width=0.55,
            yerr=std_api, capsize=4, error_kw={"elinewidth":1.2})
    ax3.set_xticks(x); ax3.set_xticklabels(CMD_LABELS, fontsize=9)
    ax3.set_ylabel("Mean API calls"); ax3.set_title("Mean API Calls per Command")
    ax3.grid(axis="y", alpha=0.3)
    for xi, m in enumerate(mean_api):
        ax3.text(xi, m+0.1, f"{m:.1f}", ha="center", fontsize=9)

    # Cumulative tokens across the session (how context grows)
    ax4 = axes[1,1]
    for run in range(1, N_RUNS+1):
        cum = []
        for cmd in range(1, N_CMDS+1):
            tok = flt(grid.get((run,cmd),{}),"tokens")
            cum.append(tok)
        ax4.plot(range(1, N_CMDS+1), cum, marker="o",
                 color=RUN_COLS[run-1], linewidth=1.8, label=f"Run {run}")
    ax4.set_xticks(range(1, N_CMDS+1)); ax4.set_xticklabels(CMD_SHORT, fontsize=9)
    ax4.set_ylabel("Input tokens this call")
    ax4.set_title("Token Count per Call Within Session\n(shows context growth + text_inference collapse)")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "C2_fig7_token_api_analysis.png")


# =============================================================================
# Figure 8 — Cmd3 "go higher" Failure Deep Dive
# =============================================================================
def fig8():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("C2 — Cmd3 'go higher' Failure Deep Dive (0/5 runs)",
                 fontsize=13, fontweight="bold")

    z_bef = [get(r, 3, "z_before_m") for r in range(1, N_RUNS+1)]
    z_aft = [get(r, 3, "z_after_m")  for r in range(1, N_RUNS+1)]
    tgts  = [get(r, 3, "target_m")   for r in range(1, N_RUNS+1)]
    delts = [get(r, 3, "increment_m")for r in range(1, N_RUNS+1)]

    # Scatter: z_before vs z_after — should be above diagonal for "go higher"
    ax = axes[0]
    lo = min(z_bef+z_aft) - 0.05
    hi = max(z_bef+z_aft) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="z_after = z_before (no movement)")
    ax.plot([lo, hi], [v+0.3 for v in [lo,hi]], "g:", linewidth=0.8, alpha=0.5,
            label="+0.3 m (expected minimum 'higher')")
    for run, (zb, za) in enumerate(zip(z_bef, z_aft)):
        ax.scatter(zb, za, color=RUN_COLS[run], s=100, zorder=5)
        ax.annotate(f"Run {run+1}", (zb, za),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color=RUN_COLS[run])
    ax.set_xlabel("z_before (m)"); ax.set_ylabel("z_after (m)")
    ax.set_title("z_before vs z_after — all 5 runs\nAll points hug the diagonal")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Bar: increment per run
    ax2 = axes[1]
    bars = ax2.bar(range(1, N_RUNS+1), delts,
                   color=[C_FAIL]*N_RUNS, width=0.55)
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.axhline(0.3, color="green", linewidth=0.8, linestyle="--",
                label="Minimum expected increment")
    ax2.set_xlabel("Run"); ax2.set_ylabel("Increment (m)")
    ax2.set_title("Altitude Increment per Run\n(all near-zero or negative)")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(range(1, N_RUNS+1))
    for bar, d in zip(bars, delts):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height() + (0.002 if d >= 0 else -0.004),
                 f"{d:+.3f} m", ha="center",
                 va="bottom" if d >= 0 else "top", fontsize=9)

    # z_before vs LLM target — shows LLM sets target ≈ current altitude
    ax3 = axes[2]
    ax3.scatter(z_bef, tgts, color=C_FAIL, s=100, zorder=5)
    lo2 = min(z_bef+tgts) - 0.05; hi2 = max(z_bef+tgts) + 0.05
    ax3.plot([lo2, hi2], [lo2, hi2], "k--", linewidth=1.0,
             label="target = current altitude")
    for run, (zb, tg) in enumerate(zip(z_bef, tgts)):
        ax3.annotate(f"Run {run+1}", (zb, tg),
                     textcoords="offset points", xytext=(6,4),
                     fontsize=8, color=RUN_COLS[run])
    ax3.set_xlabel("z_before (m)"); ax3.set_ylabel("LLM target (m)")
    ax3.set_title("LLM Target vs Current Altitude\nTarget ≈ current → no movement")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    plt.tight_layout()
    save(fig, "C2_fig8_cmd3_failure_deep_dive.png")


# =============================================================================
# Figure 9 — Cmd5 "safe height" Interpretation Variance
# =============================================================================
def fig9():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("C2 — Cmd5 'ascend slowly to a safe height' Interpretation Variance",
                 fontsize=13, fontweight="bold")

    z_bef  = [get(r, 5, "z_before_m") for r in range(1, N_RUNS+1)]
    z_aft  = [get(r, 5, "z_after_m")  for r in range(1, N_RUNS+1)]
    tgts   = [get(r, 5, "target_m")   for r in range(1, N_RUNS+1)]
    delts  = [get(r, 5, "increment_m")for r in range(1, N_RUNS+1)]
    srcs   = [gets(r, 5, "target_source") for r in range(1, N_RUNS+1)]
    correct= [intv(grid.get((r,5),{}),"correct") for r in range(1, N_RUNS+1)]

    outcome_labels = []
    for ok, src in zip(correct, srcs):
        if ok: outcome_labels.append("PASS")
        elif src == "text_inference": outcome_labels.append("TEXT INF")
        else: outcome_labels.append("FAIL (wrong tgt)")

    outcome_cols = [C_PASS if ok else (C_TEXTINF if s=="text_inference" else C_FAIL)
                    for ok, s in zip(correct, srcs)]

    # Bar: what altitude did the LLM choose as "safe height"
    ax = axes[0]
    bars = ax.bar(range(1, N_RUNS+1), tgts, color=outcome_cols, width=0.55)
    ax.axhline(np.mean(z_bef), color="#555", linewidth=1.0, linestyle="--",
               label=f"Mean z_before ({np.mean(z_bef):.2f} m)")
    ax.set_xlabel("Run"); ax.set_ylabel("LLM 'safe height' target (m)")
    ax.set_title("What altitude did LLM pick\nas 'safe height' per run?")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); ax.set_xticks(range(1,N_RUNS+1))
    for bar, tg, ol in zip(bars, tgts, outcome_labels):
        ax.text(bar.get_x()+bar.get_width()/2, tg+0.03,
                f"{tg:.2f} m\n({ol})", ha="center", va="bottom", fontsize=8)

    # z_before vs z_after — shows the Run 3 descent
    ax2 = axes[1]
    lo = min(z_bef+z_aft) - 0.1; hi = max(z_bef+z_aft) + 0.1
    ax2.plot([lo,hi],[lo,hi], "k--", linewidth=0.9, label="no movement")
    ax2.axhline(2.5, color="green", linewidth=0.8, linestyle=":",
                label="2.5 m ceiling")
    ax2.axhline(1.5, color="red", linewidth=0.8, linestyle=":",
                label="Run 3 descent target")
    for run, (zb, za, col) in enumerate(zip(z_bef, z_aft, outcome_cols)):
        ax2.scatter(zb, za, color=col, s=100, zorder=5)
        ax2.annotate(f"R{run+1} ({outcome_labels[run]})", (zb, za),
                     textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax2.set_xlabel("z_before (m)"); ax2.set_ylabel("z_after (m)")
    ax2.set_title("z_before vs z_after per Run\n(Run 3 descends — 'safe height'=1.5 m)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Increment bar with annotation of each run's reasoning
    ax3 = axes[2]
    bars = ax3.bar(range(1, N_RUNS+1), delts, color=outcome_cols, width=0.55)
    ax3.axhline(0, color="black", linewidth=0.9)
    ax3.set_xlabel("Run"); ax3.set_ylabel("Increment (m)")
    ax3.set_title("Altitude Change per Run\n(negative = descent)")
    ax3.set_xticks(range(1, N_RUNS+1)); ax3.grid(axis="y", alpha=0.3)
    notes = ["no action\n(text_inf)", "+0.20 m\n(PASS)", "−0.51 m\n(DESCENT!)",
             "+0.20 m\n(PASS)", "no action\n(text_inf)"]
    for bar, d, note in zip(bars, delts, notes):
        ypos = d + (0.01 if d >= 0 else -0.01)
        va   = "bottom" if d >= 0 else "top"
        ax3.text(bar.get_x()+bar.get_width()/2, ypos, note,
                 ha="center", va=va, fontsize=8)

    patches = [mpatches.Patch(color=C_PASS,    label="Pass"),
               mpatches.Patch(color=C_FAIL,    label="Wrong target"),
               mpatches.Patch(color=C_TEXTINF, label="text_inference")]
    axes[0].legend(handles=patches + axes[0].get_legend_handles_labels()[0],
                   fontsize=7, loc="upper left")

    plt.tight_layout()
    save(fig, "C2_fig9_cmd5_interpretation.png")


# =============================================================================
# Run all figures
# =============================================================================
if __name__ == "__main__":
    import sys
    targets = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else \
              ["FIG1","FIG2","FIG3","FIG4","FIG5","FIG6","FIG7","FIG8","FIG9"]
    fns = {"FIG1":fig1,"FIG2":fig2,"FIG3":fig3,"FIG4":fig4,"FIG5":fig5,
           "FIG6":fig6,"FIG7":fig7,"FIG8":fig8,"FIG9":fig9}

    print(f"Generating C2 detailed plots → {RESULTS}")
    for key in targets:
        if key in fns: fns[key]()
        else: print(f"  unknown: {key}")
    print("\nDone.")
