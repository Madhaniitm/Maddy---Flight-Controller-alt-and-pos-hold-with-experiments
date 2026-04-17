"""
EXP-F1: LLM Benchmark — Task Success Rate Across All Capabilities
==================================================================
Goal:
    Compile results from D6, D7, E5, C2, C5 into a single 4-model × 5-capability
    benchmark matrix. No new API calls — this is a pure analysis script that reads
    previously saved summary CSVs.

Capability dimensions:
    1. anomaly_detection    — from D6_summary.csv  (detection_rate per LLM, avg over 4 faults)
    2. pid_adaptation       — from D7_summary.csv  (rmse_reduction_pct normalised 0-1)
    3. fault_supervision    — from E5_summary.csv  (correct_rate avg over S2+S3, non-trivial scenarios)
    4. ambiguity_handling   — from C2_summary.csv  (success rate on ambiguous commands 3-6)
    5. diagnosis_accuracy   — from C5_summary.csv  (rmse_reduction_pct normalised 0-1)

Models scored: claude, gpt4o, gemini, llama3
    (C2/C5 are Claude-only; other models get NaN → shown as "not tested" in matrix)

Outputs:
    - F1_benchmark_matrix.csv  : 4×5 score matrix
    - F1_benchmark_summary.csv : overall rank + per-capability scores
    - F1_benchmark_radar.png   : radar chart (4 models, 5 axes)
    - F1_benchmark_heatmap.png : annotated heat-map of the matrix

Paper References:
    - ReAct (Yao et al. 2022): common prompt backbone across all benchmarked tasks
    - Vemprala2023: multi-LLM benchmark methodology for robotics
"""

import os, sys, csv, math, pathlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR     = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

PAPER_REFS = {
    "ReAct":      "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "Vemprala2023":"Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

MODELS      = ["claude", "gpt4o", "gemini", "llama3"]
CAPABILITIES= ["anomaly_detection", "pid_adaptation", "fault_supervision",
               "ambiguity_handling", "diagnosis_accuracy"]
CAP_LABELS  = ["Anomaly\nDetection", "PID\nAdaptation", "Fault\nSupervision",
               "Ambiguity\nHandling", "Diagnosis\nAccuracy"]

# ── CSV reader helpers ─────────────────────────────────────────────────────────
def read_csv_dict(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def find_val(rows: list[dict], col_filter: dict, value_col: str) -> float | None:
    """Find first row where all col_filter k:v match, return float(value_col)."""
    for row in rows:
        if all(row.get(k,"").strip() == v for k, v in col_filter.items()):
            try:
                return float(row[value_col])
            except (ValueError, KeyError):
                return None
    return None

# ── Source file loaders ────────────────────────────────────────────────────────
def load_d6(llm: str) -> float | None:
    """Mean detection_rate across 4 faults for this LLM."""
    rows = read_csv_dict(OUT_DIR / "D6_summary.csv")
    vals = []
    for row in rows:
        if row.get("llm","").strip() == llm and row.get("metric","") == "":
            # D6_summary.csv columns: fault, llm, detection_rate, ci_lo, ci_hi, ...
            try:
                vals.append(float(row.get("detection_rate", row.get("value", ""))))
            except (ValueError, TypeError):
                pass
    # Alternate: rows where first column is fault, second is llm
    if not vals:
        for row in rows:
            cols = list(row.values())
            # Try: [fault, llm, detection_rate, ...]
            if len(cols) >= 3 and cols[1].strip() == llm:
                try:
                    vals.append(float(cols[2]))
                except (ValueError, TypeError):
                    pass
    return round(float(np.mean(vals)), 4) if vals else None

def load_d7(llm: str) -> float | None:
    """RMSE reduction % for this LLM (normalised 0-1 by dividing by 100)."""
    rows = read_csv_dict(OUT_DIR / "D7_summary.csv")
    for row in rows:
        if (row.get("llm","").strip() == llm and
                "rmse_reduction" in row.get("metric","").lower()):
            try:
                return min(float(row.get("value","")) / 100.0, 1.0)
            except (ValueError, TypeError):
                pass
    return None

def load_e5(llm: str) -> float | None:
    """Avg correct_rate on S2+S3 (non-trivial scenarios) for this LLM."""
    rows = read_csv_dict(OUT_DIR / "E5_summary.csv")
    vals = []
    for row in rows:
        if (row.get("supervisor","").strip() == llm and
                row.get("scenario","").strip() in ("S2", "S3")):
            try:
                vals.append(float(row.get("correct_rate",
                                  row.get("value", ""))))
            except (ValueError, TypeError):
                pass
    return round(float(np.mean(vals)), 4) if vals else None

def load_c2() -> float | None:
    """Claude ambiguity success rate on commands 3-6 (ambiguous subset)."""
    rows = read_csv_dict(OUT_DIR / "C2_summary.csv")
    # C2 stores per-command accuracy; commands 3-6 are the ambiguous ones
    ambiguous_keys = [f"accuracy_cmd{i}" for i in range(3, 7)]
    vals = []
    for row in rows:
        metric = row.get("metric", row.get("command","")).strip()
        if any(metric.endswith(k) or k in metric for k in ambiguous_keys):
            try:
                vals.append(float(row.get("value", row.get("accuracy",""))))
            except (ValueError, TypeError):
                pass
    # Fallback: use overall_accuracy if per-command not available
    if not vals:
        for row in rows:
            if "overall" in row.get("metric","").lower():
                try:
                    return float(row.get("value",""))
                except (ValueError, TypeError):
                    pass
    return round(float(np.mean(vals)), 4) if vals else None

def load_c5() -> float | None:
    """Claude diagnosis: rmse_reduction_pct normalised 0-1."""
    rows = read_csv_dict(OUT_DIR / "C5_summary.csv")
    for row in rows:
        if "rmse_reduction" in row.get("metric","").lower():
            try:
                return min(float(row.get("value","")) / 100.0, 1.0)
            except (ValueError, TypeError):
                pass
    return None

# ── Build score matrix ─────────────────────────────────────────────────────────
def build_matrix() -> dict:
    """Returns {model: {capability: score_or_None}}."""
    matrix = {m: {c: None for c in CAPABILITIES} for m in MODELS}

    for m in MODELS:
        matrix[m]["anomaly_detection"] = load_d6(m)
        matrix[m]["pid_adaptation"]    = load_d7(m)
        matrix[m]["fault_supervision"] = load_e5(m)

    # C2 and C5 are Claude-only experiments
    matrix["claude"]["ambiguity_handling"] = load_c2()
    matrix["claude"]["diagnosis_accuracy"] = load_c5()

    # If any score is still None (CSV not yet generated), use placeholder NaN
    for m in MODELS:
        for c in CAPABILITIES:
            if matrix[m][c] is None:
                matrix[m][c] = float("nan")

    return matrix

def overall_score(scores: dict) -> float:
    valid = [v for v in scores.values() if not math.isnan(v)]
    return round(float(np.mean(valid)), 4) if valid else float("nan")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-F1: LLM Benchmark — Capability Matrix")
    print("(Analysis only — reads D6/D7/E5/C2/C5 result CSVs)")
    print("=" * 60)

    matrix = build_matrix()

    # ── Print matrix ───────────────────────────────────────────────────────────
    header = f"{'Model':10s}" + "".join(f"  {c[:14]:14s}" for c in CAPABILITIES) + "  Overall"
    print("\n" + header)
    print("-" * len(header))
    for m in MODELS:
        row_str = f"  {m:10s}"
        for c in CAPABILITIES:
            v = matrix[m][c]
            row_str += f"  {v:.3f}{'':9s}" if not math.isnan(v) else f"  {'N/A':13s}"
        ov = overall_score(matrix[m])
        row_str += f"  {ov:.3f}" if not math.isnan(ov) else "  N/A"
        print(row_str)

    # ── Save benchmark matrix CSV ──────────────────────────────────────────────
    matrix_csv = OUT_DIR / "F1_benchmark_matrix.csv"
    with open(matrix_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + CAPABILITIES + ["overall"])
        for m in MODELS:
            scores = [matrix[m][c] for c in CAPABILITIES]
            ov     = overall_score(matrix[m])
            w.writerow([m] + [f"{v:.4f}" if not math.isnan(v) else "N/A"
                               for v in scores] + [f"{ov:.4f}" if not math.isnan(ov) else "N/A"])
    print(f"\nMatrix CSV    → {matrix_csv}")

    # ── Save summary with rank ─────────────────────────────────────────────────
    ranks = sorted(MODELS, key=lambda m: -overall_score(matrix[m])
                   if not math.isnan(overall_score(matrix[m])) else -999)

    summary_csv = OUT_DIR / "F1_benchmark_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["rank","model","overall_score"] + CAPABILITIES + ["notes"])
        for rank, m in enumerate(ranks, 1):
            ov     = overall_score(matrix[m])
            scores = [matrix[m][c] for c in CAPABILITIES]
            note   = ("Claude-only tasks (C2/C5) excluded from other models' averages"
                      if m != "claude" else "All 5 capabilities tested")
            cw.writerow([rank, m,
                         f"{ov:.4f}" if not math.isnan(ov) else "N/A"] +
                        [f"{v:.4f}" if not math.isnan(v) else "N/A" for v in scores] +
                        [note])
        for key, ref in PAPER_REFS.items():
            cw.writerow(["", f"ref_{key}", ref] + [""] * (len(CAPABILITIES) + 1))
    print(f"Summary CSV   → {summary_csv}")

    # ── Radar chart ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        N    = len(CAPABILITIES)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]   # close the polygon
        colors = {"claude":"#3498db","gpt4o":"#e67e22","gemini":"#2ecc71","llama3":"#9b59b6"}

        fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                                 subplot_kw={"polar": False})

        # Left: radar chart
        ax_r = plt.subplot(121, polar=True)
        ax_r.set_theta_offset(math.pi / 2)
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(CAP_LABELS, size=9)
        ax_r.set_ylim(0, 1)

        for m in MODELS:
            vals = [matrix[m][c] if not math.isnan(matrix[m][c]) else 0.0
                    for c in CAPABILITIES]
            vals += vals[:1]
            ax_r.plot(angles, vals, "o-", linewidth=2, label=m, color=colors[m])
            ax_r.fill(angles, vals, alpha=0.08, color=colors[m])
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
        ax_r.set_title("F1: LLM Capability Radar", size=11, pad=20)

        # Right: heatmap
        ax_h = axes[1]
        mat_vals = np.array([[matrix[m][c] if not math.isnan(matrix[m][c]) else 0.0
                               for c in CAPABILITIES] for m in MODELS])
        im = ax_h.imshow(mat_vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax_h.set_xticks(range(N))
        ax_h.set_xticklabels(CAP_LABELS, fontsize=8, rotation=20, ha="right")
        ax_h.set_yticks(range(len(MODELS)))
        ax_h.set_yticklabels(MODELS)
        plt.colorbar(im, ax=ax_h, label="Score (0–1)")
        for i, m in enumerate(MODELS):
            for j, c in enumerate(CAPABILITIES):
                v = matrix[m][c]
                txt = f"{v:.2f}" if not math.isnan(v) else "N/A"
                ax_h.text(j, i, txt, ha="center", va="center", fontsize=9,
                          color="black" if not math.isnan(v) else "gray")
        ax_h.set_title("F1: Capability Score Heatmap", size=11)

        fig.suptitle(
            "EXP-F1 Multi-LLM Benchmark — 4 Models × 5 Capability Dimensions\n"
            "ReAct (Yao 2022), Vemprala 2023 | Sources: D6, D7, E5, C2, C5",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "F1_benchmark_radar.png"
        fig.savefig(png, dpi=150)
        print(f"Plot saved    → {png}")
        plt.close(fig)
    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console rank table ─────────────────────────────────────────────────────
    print("\n── F1 Overall Rankings ─────────────────────────────────────────────")
    for rank, m in enumerate(ranks, 1):
        ov = overall_score(matrix[m])
        print(f"  #{rank}  {m:10s}  overall={ov:.3f}")


if __name__ == "__main__":
    main()
