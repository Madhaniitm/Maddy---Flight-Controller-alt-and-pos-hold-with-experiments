"""
Sensitivity Analysis — C2 and C2.1 Scoring Window
===================================================
Re-scores C2 and C2.1 runs across upper-bound thresholds for the
ambiguous commands (0.15, 0.175, 0.20, 0.225, 0.25 m) and confirms
that model rankings are stable.

For explicit commands (Cmd1/Cmd2) the scoring is target-based (fixed
±0.15 m tolerance on the absolute target), so the window sweep only
applies to the ambiguous commands (Cmd3–Cmd6 in C2.1, Cmd3–Cmd6 in C2).

Outputs:
  results/sensitivity_C2_window.csv    — per-(model, threshold) overall rate
  results/sensitivity_C2_1_window.csv  — same for C2.1
"""

import csv, os, math, numpy as np
from itertools import product

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# ── C2 Scoring ────────────────────────────────────────────────────────────────
# explicit/paraphrase: correct if abs(target_m - 2.0) <= 0.15  (fixed, not swept)
# relative commands:  correct if lo <= increment <= hi  OR asked_clarify
#   C2 original ranges: Cmd3 (0.10,1.50), Cmd4 (0.05,0.60), Cmd5/6 (0.05,2.00)
#   For sweep we vary the upper bound of Cmd3 and Cmd4 (Cmd5/6 are very wide anyway)

C2_EXPLICIT_TYPES = {"explicit", "paraphrase"}
C2_FIXED_LO = {
    "relative_no_num": 0.10,
    "vague_relative":  0.05,
    "abstract":        0.05,
    "indirect":        0.05,
}
# C2 original upper bounds (used at baseline threshold=original)
C2_ORIG_HI = {
    "relative_no_num": 1.50,
    "vague_relative":  0.60,
    "abstract":        2.00,
    "indirect":        2.00,
}

# ── C2.1 Scoring ──────────────────────────────────────────────────────────────
# All ambiguous commands use (0.05, upper) where upper is swept
C21_FIXED_LO = 0.05

THRESHOLDS = [0.15, 0.175, 0.20, 0.225, 0.25]  # upper bound for C2.1 sweep
# For C2, Cmd3/4 sweepable; Cmd5/6 are already wide (keep originals)
C2_SWEEP_TYPES = {"relative_no_num", "vague_relative"}


def score_c2_row(row, upper_bound):
    cmd_type = row["cmd_type"]
    if cmd_type in C2_EXPLICIT_TYPES:
        try:
            t = float(row["target_m"])
            return abs(t - 2.0) <= 0.15
        except (ValueError, TypeError):
            return False
    lo = C2_FIXED_LO.get(cmd_type, 0.05)
    if cmd_type in C2_SWEEP_TYPES:
        hi = upper_bound
    else:
        hi = C2_ORIG_HI.get(cmd_type, 2.00)
    try:
        inc = float(row["increment_m"])
    except (ValueError, TypeError):
        return False
    clarify = str(row.get("asked_clarify", "0")).strip() in {"1", "True", "true"}
    return clarify or (lo <= inc <= hi)


def score_c21_row(row, upper_bound):
    cmd_type = row["cmd_type"]
    if cmd_type in C2_EXPLICIT_TYPES:
        try:
            t = float(row["target_m"])
            return abs(t - 2.0) <= 0.15
        except (ValueError, TypeError):
            return False
    lo = C21_FIXED_LO
    hi = upper_bound
    try:
        inc = float(row["increment_m"])
    except (ValueError, TypeError):
        return False
    clarify = str(row.get("asked_clarify", "0")).strip() in {"1", "True", "true"}
    return clarify or (lo <= inc <= hi)


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("cmd_idx", "").isdigit():
                rows.append(r)
    return rows


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── C2 analysis ───────────────────────────────────────────────────────────────
C2_FILES = {
    "Claude":  os.path.join(RESULTS, "C2_runs.csv"),
    "GPT-4o":  os.path.join(RESULTS, "C2_runs_gpt4o_guardrail_on.csv"),
    "mini":    os.path.join(RESULTS, "C2_runs_gpt4omini_guardrail_on.csv"),
}

C21_FILES = {
    "Claude":  os.path.join(RESULTS, "C2_1_runs.csv"),
    "GPT-4o":  os.path.join(RESULTS, "C2_1_runs_gpt4o_guardrail_on.csv"),
    "mini":    os.path.join(RESULTS, "C2_1_runs_gpt4omini_guardrail_on.csv"),
}

print("\n══ C2 Sensitivity Analysis (upper bound sweep on Cmd3/4) ══")
print(f"{'Model':<12}" + "".join(f"  thr={t:.3f}" for t in THRESHOLDS))

c2_results = {}
for model, path in C2_FILES.items():
    rows = load_csv(path)
    row_scores = {}
    for thr in THRESHOLDS:
        scored = [score_c2_row(r, thr) for r in rows]
        n_ok = sum(scored)
        n_tot = len(scored)
        rate = n_ok / n_tot if n_tot else 0
        lo, hi = wilson_ci(n_ok, n_tot)
        row_scores[thr] = (rate, lo, hi, n_ok, n_tot)
    c2_results[model] = row_scores
    line = f"{model:<12}" + "".join(f"  {row_scores[t][0]:.3f}     " for t in THRESHOLDS)
    print(line)

print("\n══ C2.1 Sensitivity Analysis (upper bound sweep on all ambiguous cmds) ══")
print(f"{'Model':<12}" + "".join(f"  thr={t:.3f}" for t in THRESHOLDS))

c21_results = {}
for model, path in C21_FILES.items():
    rows = load_csv(path)
    row_scores = {}
    for thr in THRESHOLDS:
        scored = [score_c21_row(r, thr) for r in rows]
        n_ok = sum(scored)
        n_tot = len(scored)
        rate = n_ok / n_tot if n_tot else 0
        lo, hi = wilson_ci(n_ok, n_tot)
        row_scores[thr] = (rate, lo, hi, n_ok, n_tot)
    c21_results[model] = row_scores
    line = f"{model:<12}" + "".join(f"  {row_scores[t][0]:.3f}     " for t in THRESHOLDS)
    print(line)

# ── Per-command breakdown at each threshold (C2.1) ────────────────────────────
print("\n══ C2.1 Per-command rates at each threshold ══")
CMD_TYPES = ["explicit", "paraphrase", "relative_no_num", "vague_relative", "abstract", "indirect"]
CMD_LABELS = ["Cmd1 explicit", "Cmd2 paraphrase", "Cmd3 rel_no_num", "Cmd4 vague_rel", "Cmd5 abstract", "Cmd6 indirect"]

for model, path in C21_FILES.items():
    rows = load_csv(path)
    print(f"\n  {model}:")
    print(f"    {'Command':<20}" + "".join(f"  {t:.3f}" for t in THRESHOLDS))
    for ctype, clabel in zip(CMD_TYPES, CMD_LABELS):
        cmd_rows = [r for r in rows if r["cmd_type"] == ctype]
        rates = []
        for thr in THRESHOLDS:
            scored = [score_c21_row(r, thr) for r in cmd_rows]
            rates.append(sum(scored) / len(scored) if scored else 0.0)
        print(f"    {clabel:<20}" + "".join(f"  {r:.2f}  " for r in rates))

# ── Save CSVs ─────────────────────────────────────────────────────────────────
out_c2 = os.path.join(RESULTS, "sensitivity_C2_window.csv")
with open(out_c2, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "threshold", "n_correct", "n_total", "rate", "wilson_lo", "wilson_hi"])
    for model, scores in c2_results.items():
        for thr, (rate, lo, hi, nok, ntot) in scores.items():
            w.writerow([model, thr, nok, ntot, round(rate, 4), round(lo, 4), round(hi, 4)])
print(f"\nSaved: {out_c2}")

out_c21 = os.path.join(RESULTS, "sensitivity_C2_1_window.csv")
with open(out_c21, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "threshold", "n_correct", "n_total", "rate", "wilson_lo", "wilson_hi"])
    for model, scores in c21_results.items():
        for thr, (rate, lo, hi, nok, ntot) in scores.items():
            w.writerow([model, thr, nok, ntot, round(rate, 4), round(lo, 4), round(hi, 4)])
print(f"Saved: {out_c21}")

# ── Ranking stability check ────────────────────────────────────────────────────
print("\n══ Ranking Stability ══")
print("C2.1 ranking at each threshold (should be Claude > GPT-4o > mini throughout):")
for thr in THRESHOLDS:
    ranked = sorted(c21_results.keys(), key=lambda m: c21_results[m][thr][0], reverse=True)
    rates  = [f"{m}={c21_results[m][thr][0]:.3f}" for m in ranked]
    stable = (ranked == ["Claude", "GPT-4o", "mini"])
    print(f"  thr={thr:.3f}: {' > '.join(rates)}  {'✓ stable' if stable else '✗ CHANGED'}")

print("\nC2 ranking at each threshold:")
for thr in THRESHOLDS:
    ranked = sorted(c2_results.keys(), key=lambda m: c2_results[m][thr][0], reverse=True)
    rates  = [f"{m}={c2_results[m][thr][0]:.3f}" for m in ranked]
    print(f"  thr={thr:.3f}: {' > '.join(rates)}")
