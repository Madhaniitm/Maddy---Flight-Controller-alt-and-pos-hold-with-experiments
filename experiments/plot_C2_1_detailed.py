"""
plot_C2_1_detailed.py  —  Comparative diagnostic plots for EXP-C2 vs EXP-C2.1
===============================================================================
Generates 7 figures that show:
  1. Overall accuracy improvement: C2 57% → C2.1 87%
  2. Cmd3 fix proof (0/5 → 5/5, +0.1 m policy applied consistently)
  3. Side-by-side outcome heatmaps showing distribution shift
  4. Policy altitude progression (clean 2.0→2.1→2.2→2.3→2.4 m staircase)
  5. Per-run altitude trajectory comparison C2 vs C2.1
  6. Increment distribution shift across all 6 commands
  7. Tool source shift (text_inference rate before vs after fix)

Figures saved to experiments/results/:
  C2_1_fig1_comparison_accuracy.png    Side-by-side accuracy bars: C2 vs C2.1 per command
  C2_1_fig2_cmd3_fix_proof.png         Cmd3 increment & trajectory: before vs after (+0.1 m)
  C2_1_fig3_outcome_heatmap_21.png     C2 vs C2.1 outcome heatmaps side-by-side
  C2_1_fig4_policy_progression.png     Altitude staircase: +0.1 m applied per ambiguous command
  C2_1_fig5_trajectory_comparison.png  Per-run altitude traces C2 vs C2.1 side-by-side
  C2_1_fig6_increment_shift.png        Increment distribution shift for all commands C2 vs C2.1
  C2_1_fig7_tool_source_comparison.png text_inference rate before vs after fix
"""

import pathlib, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# ── Colours ───────────────────────────────────────────────────────────────────
C_PASS    = "#2ECC71"
C_FAIL    = "#E74C3C"
C_TEXTINF = "#F39C12"
C_C2      = "#2980B9"   # blue  — C2 (baseline)
C_C21     = "#8E44AD"   # purple — C2.1 (fixed)
RUN_COLS  = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9"]

CMD_LABELS = ["Cmd1\nexplicit", "Cmd2\nparaphrase", "Cmd3\nno-number",
              "Cmd4\nvague", "Cmd5\nabstract", "Cmd6\nindirect"]
CMD_SHORT  = ["Cmd1","Cmd2","Cmd3","Cmd4","Cmd5","Cmd6"]
N_CMDS = 6; N_RUNS = 5

# ── Load data ─────────────────────────────────────────────────────────────────
c2_runs   = rcsv("C2_runs.csv")
c21_runs  = rcsv("C2_1_runs.csv")
c2_sum    = [r for r in rcsv("C2_summary.csv")   if r.get("cmd_idx","").isdigit()]
c21_sum   = [r for r in rcsv("C2_1_summary.csv") if r.get("cmd_idx","").isdigit()]

if not c21_runs:
    print("ERROR: C2_1_runs.csv not found — run exp_C2_1_prompt_fix.py first.")
    raise SystemExit(1)

def build_grid(rows):
    g = {}
    for r in rows:
        try: g[(int(r["run"]), int(r["cmd_idx"]))] = r
        except: pass
    return g

g2  = build_grid(c2_runs)
g21 = build_grid(c21_runs)

def gets(g, run, cmd, key, default=""):
    return g.get((run,cmd),{}).get(key, default)

def getf(g, run, cmd, key, default=0.0):
    return flt(g.get((run,cmd),{}), key, default)

acc2  = [flt(r,"success_rate") for r in c2_sum]
acc21 = [flt(r,"success_rate") for r in c21_sum]
lo2   = [flt(r,"wilson_ci_lo") for r in c2_sum]
hi2   = [flt(r,"wilson_ci_hi") for r in c2_sum]
lo21  = [flt(r,"wilson_ci_lo") for r in c21_sum]
hi21  = [flt(r,"wilson_ci_hi") for r in c21_sum]


# =============================================================================
# Figure 1 — Side-by-Side Accuracy Comparison
# =============================================================================
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("C2 vs C2.1 — Accuracy per Command: Prompt Engineering Fix Effect",
                 fontsize=13, fontweight="bold")

    x = np.arange(N_CMDS); w = 0.35

    ax = axes[0]
    b2  = ax.bar(x-w/2, acc2,  w, color=C_C2,  label="C2 (no fix)",  alpha=0.8)
    b21 = ax.bar(x+w/2, acc21, w, color=C_C21, label="C2.1 (fixed)", alpha=0.8)
    ax.errorbar(x-w/2, acc2,  yerr=[[a-l for a,l in zip(acc2,lo2)],
                                     [h-a for a,h in zip(acc2,hi2)]],
                fmt="none", ecolor="#333", capsize=4, lw=1.2)
    ax.errorbar(x+w/2, acc21, yerr=[[a-l for a,l in zip(acc21,lo21)],
                                     [h-a for a,h in zip(acc21,hi21)]],
                fmt="none", ecolor="#333", capsize=4, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(CMD_LABELS, fontsize=9)
    ax.set_ylim(0, 1.25); ax.set_ylabel("Success rate (Wilson 95% CI)")
    ax.set_title("Per-command accuracy: C2 vs C2.1")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for xi, (a, b) in enumerate(zip(acc2, acc21)):
        delta = b - a
        col   = C_PASS if delta > 0 else (C_FAIL if delta < 0 else "#888")
        ax.annotate(f"Δ{delta:+.2f}", (xi, max(a,b)+0.07),
                    ha="center", fontsize=8.5, color=col, fontweight="bold")

    # Delta bar chart
    ax2 = axes[1]
    deltas = [b - a for a, b in zip(acc2, acc21)]
    cols   = [C_PASS if d > 0 else (C_FAIL if d < 0 else "#aaa") for d in deltas]
    bars   = ax2.bar(x, deltas, color=cols, width=0.55, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(CMD_LABELS, fontsize=9)
    ax2.set_ylabel("Accuracy change (C2.1 − C2)")
    ax2.set_title("Δ Accuracy per Command\n(green=improved, red=degraded)")
    ax2.grid(axis="y", alpha=0.3)
    for bar, d in zip(bars, deltas):
        ypos = d + (0.02 if d >= 0 else -0.02)
        ax2.text(bar.get_x()+bar.get_width()/2, ypos,
                 f"{d:+.2f}", ha="center", va="bottom" if d>=0 else "top",
                 fontsize=10, fontweight="bold")

    total_c2  = sum(acc2)  / N_CMDS
    total_c21 = sum(acc21) / N_CMDS
    ax.axhline(total_c2,  color=C_C2,  linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"C2 mean {total_c2:.2f}")
    ax.axhline(total_c21, color=C_C21, linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"C2.1 mean {total_c21:.2f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "C2_1_fig1_comparison_accuracy.png")


# =============================================================================
# Figure 2 — Cmd3 Fix Proof
# =============================================================================
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("C2.1 — Cmd3 'go higher' Fix: 0/5 → 5/5  (+0.1 m conservative default policy)",
                 fontsize=13, fontweight="bold")

    inc2  = [getf(g2,  r, 3, "increment_m") for r in range(1, N_RUNS+1)]
    inc21 = [getf(g21, r, 3, "increment_m") for r in range(1, N_RUNS+1)]

    ax = axes[0]
    x  = np.arange(N_RUNS); w = 0.35
    ax.bar(x-w/2, inc2,  w, color=C_C2,  label="C2 (no fix)", alpha=0.85)
    ax.bar(x+w/2, inc21, w, color=C_C21, label="C2.1 (fixed)", alpha=0.85)
    ax.axhline(0,   color="black", linewidth=0.8)
    ax.axhline(0.1, color=C_PASS, linewidth=1.2, linestyle="--",
               label="Policy default +0.1 m")
    ax.set_xticks(x); ax.set_xticklabels([f"Run {r}" for r in range(1, N_RUNS+1)])
    ax.set_ylabel("Altitude increment (m)")
    ax.set_title("Cmd3 increment per run\n(C2 ≈ 0 m, C2.1 ≈ +0.1 m)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # z_before vs z_after scatter — both experiments
    ax2 = axes[1]
    zb2  = [getf(g2,  r, 3, "z_before_m") for r in range(1, N_RUNS+1)]
    za2  = [getf(g2,  r, 3, "z_after_m")  for r in range(1, N_RUNS+1)]
    zb21 = [getf(g21, r, 3, "z_before_m") for r in range(1, N_RUNS+1)]
    za21 = [getf(g21, r, 3, "z_after_m")  for r in range(1, N_RUNS+1)]

    lo = min(zb2+za2+zb21+za21) - 0.05
    hi = max(zb2+za2+zb21+za21) + 0.05
    ax2.plot([lo,hi],[lo,hi],     "k--", linewidth=0.9, label="no movement (y=x)")
    ax2.plot([lo,hi],[v+0.1 for v in [lo,hi]], color=C_PASS, linewidth=1.2, linestyle=":",
             label="+0.1 m policy line (y=x+0.1)")
    ax2.scatter(zb2,  za2,  color=C_C2,  s=80, marker="o", label="C2", zorder=5)
    ax2.scatter(zb21, za21, color=C_C21, s=80, marker="^", label="C2.1", zorder=5)
    ax2.set_xlabel("z_before (m)"); ax2.set_ylabel("z_after (m)")
    ax2.set_title("z_before vs z_after — Cmd3\n(C2: on y=x diagonal  |  C2.1: on y=x+0.1)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Target chosen per run
    ax3 = axes[2]
    tgt2  = [getf(g2,  r, 3, "target_m") for r in range(1, N_RUNS+1)]
    tgt21 = [getf(g21, r, 3, "target_m") for r in range(1, N_RUNS+1)]
    x = np.arange(N_RUNS)
    ax3.bar(x-w/2, tgt2,  w, color=C_C2,  label="C2", alpha=0.85)
    ax3.bar(x+w/2, tgt21, w, color=C_C21, label="C2.1", alpha=0.85)
    ax3.axhline(2.1, color=C_PASS, linewidth=1.2, linestyle="--",
                label="Expected target (z_before+0.1 ≈ 2.1)")
    ax3.set_xticks(x); ax3.set_xticklabels([f"Run {r}" for r in range(1, N_RUNS+1)])
    ax3.set_ylabel("LLM target altitude (m)")
    ax3.set_title("Cmd3 LLM target per run\n(C2 ≈ 2.0 m, C2.1 ≈ 2.1 m)")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "C2_1_fig2_cmd3_fix_proof.png")


# =============================================================================
# Figure 3 — C2.1 Outcome Heatmap
# =============================================================================
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("C2 vs C2.1 — Outcome Heatmaps: How the Distribution Shifted",
                 fontsize=13, fontweight="bold")

    def draw_heatmap(ax, grid, title):
        mat = np.zeros((N_RUNS, N_CMDS))
        lbl = [[""]*N_CMDS for _ in range(N_RUNS)]
        for run in range(1, N_RUNS+1):
            for cmd in range(1, N_CMDS+1):
                ok  = intv(grid.get((run,cmd),{}), "correct")
                src = grid.get((run,cmd),{}).get("target_source","")
                if ok == 1:
                    mat[run-1,cmd-1] = 2; lbl[run-1][cmd-1] = "PASS"
                elif src == "text_inference":
                    mat[run-1,cmd-1] = 0; lbl[run-1][cmd-1] = "TEXT\nINF"
                else:
                    mat[run-1,cmd-1] = 1; lbl[run-1][cmd-1] = "FAIL"
        cmap = matplotlib.colors.ListedColormap([C_TEXTINF, C_FAIL, C_PASS])
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(N_CMDS)); ax.set_xticklabels(CMD_LABELS, fontsize=9)
        ax.set_yticks(range(N_RUNS)); ax.set_yticklabels([f"Run {r}" for r in range(1,N_RUNS+1)])
        for i in range(N_RUNS):
            for j in range(N_CMDS):
                ax.text(j, i, lbl[i][j], ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")

    draw_heatmap(axes[0], g2,  "C2 — Baseline (no prompt fix)")
    draw_heatmap(axes[1], g21, "C2.1 — With Prompt Fix")

    patches = [mpatches.Patch(color=C_PASS,    label="PASS"),
               mpatches.Patch(color=C_FAIL,    label="FAIL (wrong target)"),
               mpatches.Patch(color=C_TEXTINF, label="text_inference (no action)")]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    save(fig, "C2_1_fig3_outcome_heatmap_21.png")


# =============================================================================
# Figure 4 — Policy Altitude Progression
# =============================================================================
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "C2.1 — Conservative +0.1 m Policy: Altitude Staircase and Remaining Failures",
        fontsize=13, fontweight="bold")

    # Panel 1: Mean z_after per command in C2 vs C2.1
    ax = axes[0]
    expected_c21 = [2.0, 2.0, 2.1, 2.2, 2.3, 2.4]
    mean_za_c2   = [np.mean([getf(g2,  r, cmd, "z_after_m") for r in range(1, N_RUNS+1)])
                    for cmd in range(1, N_CMDS+1)]
    mean_za_c21  = [np.mean([getf(g21, r, cmd, "z_after_m") for r in range(1, N_RUNS+1)])
                    for cmd in range(1, N_CMDS+1)]
    x = np.arange(N_CMDS)
    ax.plot(x, expected_c21, "g--", linewidth=1.5, marker="s", markersize=6,
            label="Expected (policy: +0.1 m per cmd)")
    ax.plot(x, mean_za_c2,  color=C_C2,  linewidth=2, marker="o", markersize=7,
            label="C2 mean z_after")
    ax.plot(x, mean_za_c21, color=C_C21, linewidth=2, marker="^", markersize=7,
            label="C2.1 mean z_after")
    ax.axhline(2.4, color="#999", linewidth=1.0, linestyle=":", label="Operational ceiling 2.4 m")
    ax.axhline(2.5, color=C_FAIL, linewidth=1.0, linestyle="--", label="Sim ceiling 2.5 m")
    ax.set_xticks(x); ax.set_xticklabels(CMD_LABELS, fontsize=8)
    ax.set_ylabel("Mean altitude after command (m)")
    ax.set_title("Mean altitude per command\n(C2.1 follows +0.1 m staircase)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 2: Per-run increment for all 4 ambiguous commands in C2.1
    ax2 = axes[1]
    w = 0.18
    run_cols = RUN_COLS
    for ri, run in enumerate(range(1, N_RUNS+1)):
        incs = [getf(g21, run, cmd, "increment_m") for cmd in range(3, N_CMDS+1)]
        xi = np.arange(4) + (ri - 2) * w
        ok_cols = [C_PASS if intv(g21.get((run, cmd), {}), "correct") else C_FAIL
                   for cmd in range(3, N_CMDS+1)]
        bars = ax2.bar(xi, incs, w, color=run_cols[ri], alpha=0.8,
                       edgecolor=[c for c in ok_cols], linewidth=1.5,
                       label=f"Run {run}")
    ax2.axhline(0.1, color=C_PASS, linewidth=1.5, linestyle="--",
                label="Policy: +0.1 m")
    ax2.axhline(0,   color="black", linewidth=0.8)
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(["Cmd3\n(no-num)", "Cmd4\n(vague)", "Cmd5\n(abstract)", "Cmd6\n(indirect)"],
                        fontsize=8)
    ax2.set_ylabel("Altitude increment (m)")
    ax2.set_title("C2.1 per-run increments — ambiguous commands\n(green edge=pass, red edge=fail)")
    ax2.legend(fontsize=7, ncol=2); ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Pass count per command C2 vs C2.1 — full picture
    ax3 = axes[2]
    passes_c2  = [sum(intv(g2.get((r, cmd),{}),  "correct") for r in range(1, N_RUNS+1))
                  for cmd in range(1, N_CMDS+1)]
    passes_c21 = [sum(intv(g21.get((r, cmd),{}), "correct") for r in range(1, N_RUNS+1))
                  for cmd in range(1, N_CMDS+1)]
    x = np.arange(N_CMDS); w = 0.35
    ax3.bar(x-w/2, passes_c2,  w, color=C_C2,  label="C2",  alpha=0.85)
    ax3.bar(x+w/2, passes_c21, w, color=C_C21, label="C2.1", alpha=0.85)
    for xi, (a, b) in enumerate(zip(passes_c2, passes_c21)):
        delta = b - a
        col = C_PASS if delta > 0 else (C_FAIL if delta < 0 else "#888")
        ax3.annotate(f"{delta:+d}", (xi, max(a, b) + 0.15),
                     ha="center", fontsize=9, color=col, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(CMD_LABELS, fontsize=8)
    ax3.set_yticks(range(6)); ax3.set_ylabel("N correct (out of 5)")
    ax3.set_title("Pass count per command: C2 vs C2.1\n(all ambiguous commands improved or equal)")
    ax3.legend(fontsize=9); ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "C2_1_fig4_policy_progression.png")


# =============================================================================
# Figure 5 — Per-Run Altitude Trajectory Comparison
# =============================================================================
def fig5():
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("C2 vs C2.1 — Per-Run Altitude Trajectories (top=C2, bottom=C2.1)",
                 fontsize=13, fontweight="bold")

    def draw_run(ax, grid, run, title_prefix):
        zs = [getf(grid, run, 1, "z_before_m")]
        for cmd in range(1, N_CMDS+1):
            zs.append(getf(grid, run, cmd, "z_after_m"))
        for cmd in range(1, N_CMDS+1):
            ok  = intv(grid.get((run,cmd),{}), "correct")
            src = grid.get((run,cmd),{}).get("target_source","")
            col = C_PASS if ok else (C_TEXTINF if src=="text_inference" else C_FAIL)
            ax.plot([cmd-1, cmd], [zs[cmd-1], zs[cmd]], color=col, linewidth=2.2)
            ax.scatter(cmd, zs[cmd], color=col, s=50, zorder=5)
        ax.scatter(0, zs[0], color="#555", s=50, zorder=5)
        ax.axhline(2.5, color="#ccc", linewidth=0.8, linestyle="--", label="2.5 m ceiling")
        ax.axhline(2.0, color="#eee", linewidth=0.6, linestyle="--")
        ax.set_xticks(range(N_CMDS+1))
        ax.set_xticklabels(["S"]+[f"C{i}" for i in range(1,N_CMDS+1)], fontsize=7)
        ax.set_ylim(0.8, 2.75)
        ax.set_title(f"{title_prefix} R{run}", fontsize=9, fontweight="bold")
        ax.grid(alpha=0.2)
        for cmd in range(1, N_CMDS+1):
            ax.text(cmd, zs[cmd]+0.04, f"{zs[cmd]:.2f}",
                    ha="center", va="bottom", fontsize=5.5)

    for run in range(1, N_RUNS+1):
        draw_run(axes[0][run-1], g2,  run, "C2")
        draw_run(axes[1][run-1], g21, run, "C2.1")

    patches = [mpatches.Patch(color=C_PASS,    label="Pass"),
               mpatches.Patch(color=C_FAIL,    label="Fail (wrong target)"),
               mpatches.Patch(color=C_TEXTINF, label="text_inference")]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    save(fig, "C2_1_fig5_trajectory_comparison.png")


# =============================================================================
# Figure 6 — Increment Distribution Shift
# =============================================================================
def fig6():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("C2 vs C2.1 — Altitude Increment Distribution per Command",
                 fontsize=13, fontweight="bold")

    for ci, cmd in enumerate(range(1, N_CMDS+1)):
        ax = axes[ci//3][ci%3]
        inc2  = [getf(g2,  r, cmd, "increment_m") for r in range(1, N_RUNS+1)]
        inc21 = [getf(g21, r, cmd, "increment_m") for r in range(1, N_RUNS+1)]
        ok2   = [intv(g2.get((r,cmd),{}),  "correct") for r in range(1, N_RUNS+1)]
        ok21  = [intv(g21.get((r,cmd),{}), "correct") for r in range(1, N_RUNS+1)]

        x = np.arange(N_RUNS); w = 0.35
        for xi, (v, ok) in enumerate(zip(inc2, ok2)):
            ax.bar(xi-w/2, v, w, color=C_PASS if ok else C_FAIL,
                   alpha=0.75, edgecolor=C_C2, linewidth=1.5)
        for xi, (v, ok) in enumerate(zip(inc21, ok21)):
            ax.bar(xi+w/2, v, w, color=C_PASS if ok else "#BB8FCE",
                   alpha=0.75, edgecolor=C_C21, linewidth=1.5, linestyle="--")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"R{r}" for r in range(1,N_RUNS+1)], fontsize=8)
        acc_c2  = sum(ok2)/N_RUNS; acc_c21 = sum(ok21)/N_RUNS
        ax.set_title(f"{CMD_SHORT[ci]} — C2: {acc_c2:.0%} | C2.1: {acc_c21:.0%}",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Increment (m)"); ax.grid(axis="y", alpha=0.3)

        # Mean lines
        ax.axhline(np.mean(inc2),  color=C_C2,  linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"C2 mean={np.mean(inc2):.3f}")
        ax.axhline(np.mean(inc21), color=C_C21, linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"C2.1 mean={np.mean(inc21):.3f}")
        ax.legend(fontsize=7)

    # Patch legend
    patches = [mpatches.Patch(color=C_PASS, label="Correct", alpha=0.75),
               mpatches.Patch(color=C_FAIL, label="C2 fail", alpha=0.75),
               mpatches.Patch(color="#BB8FCE", label="C2.1 fail", alpha=0.75),
               mpatches.Patch(edgecolor=C_C2,   facecolor="none", linewidth=2, label="C2 border"),
               mpatches.Patch(edgecolor=C_C21,  facecolor="none", linewidth=2, label="C2.1 border")]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout()
    save(fig, "C2_1_fig6_increment_shift.png")


# =============================================================================
# Figure 7 — Tool Source Comparison
# =============================================================================
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C2 vs C2.1 — text_inference Rate per Command",
                 fontsize=13, fontweight="bold")

    def count_sources(grid):
        n_set_ok, n_set_wrong, n_txt = [], [], []
        for cmd in range(1, N_CMDS+1):
            s = w_ = t = 0
            for run in range(1, N_RUNS+1):
                row = grid.get((run,cmd),{})
                src = row.get("target_source","")
                ok  = intv(row,"correct")
                if src == "text_inference": t += 1
                elif ok: s += 1
                else:    w_ += 1
            n_set_ok.append(s); n_set_wrong.append(w_); n_txt.append(t)
        return n_set_ok, n_set_wrong, n_txt

    so2, sw2, st2   = count_sources(g2)
    so21, sw21, st21= count_sources(g21)

    x = np.arange(N_CMDS)
    for ax, so, sw, st, title, col in [
        (axes[0], so2,  sw2,  st2,  "C2 — Baseline",     C_C2),
        (axes[1], so21, sw21, st21, "C2.1 — Prompt Fix",  C_C21),
    ]:
        ax.bar(x, so, color=C_PASS,    width=0.55, label="set_alt_target (correct)")
        ax.bar(x, sw, bottom=so,       color=C_FAIL,    width=0.55, label="set_alt_target (wrong)")
        ax.bar(x, st, bottom=[a+b for a,b in zip(so,sw)],
               color=C_TEXTINF, width=0.55, label="text_inference")
        ax.set_xticks(x); ax.set_xticklabels(CMD_LABELS, fontsize=9)
        ax.set_ylabel("Count (N=5)"); ax.set_yticks(range(6))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Annotate the key change: Cmd3 shift from all-wrong to all-correct
    axes[0].annotate("0/5\nwrong", (2, 0.3), ha="center", fontsize=8,
                     color="white", fontweight="bold")
    axes[1].annotate("5/5\ncorrect!", (2, 2.3), ha="center", fontsize=8,
                     color="white", fontweight="bold")

    plt.tight_layout()
    save(fig, "C2_1_fig7_tool_source_comparison.png")


# =============================================================================
# Run all
# =============================================================================
if __name__ == "__main__":
    import sys
    targets = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else \
              ["FIG1","FIG2","FIG3","FIG4","FIG5","FIG6","FIG7"]
    fns = {"FIG1":fig1,"FIG2":fig2,"FIG3":fig3,"FIG4":fig4,
           "FIG5":fig5,"FIG6":fig6,"FIG7":fig7}

    print(f"Generating C2.1 comparative plots → {RESULTS}")
    for key in targets:
        if key in fns: fns[key]()
        else: print(f"  unknown: {key}")
    print("\nDone.")
