"""
EXP-C2-BASELINE: Non-LLM Baselines for EXP-C2
===============================================
Two deterministic baselines, both requiring zero API calls:

  1. Regex-only    — extract explicit metre value via regex; no action if no number found.
     Score: 2/6 = 0.33 (Cmd1+Cmd2 only). Represents zero semantic understanding.

  2. Keyword+regex — regex first, then fixed priors for relative terms:
       "higher" / "ascend" → +0.5 m,  "a bit" → +0.3 m,  "safe height" → 1.5 m fixed.
     Score: 4/6 = 0.67 on C2 (wide window) — coincidental wins on Cmd3/6.
            2/6 = 0.33 on C2.1 (tight 0.05–0.20 m policy window) — identical to regex-only.

Key finding: on C2.1 (the policy-testing experiment) both baselines score 0.33.
LLMs with the policy score 0.53–0.87. The keyword baseline's C2 advantage is an
artefact of the wide acceptance window, not semantic capability.

Rule logic:
  Cmd1 "go to 2 metres"               → regex extracts 2.0 m              → target=2.0
  Cmd2 "climb to 2m"                  → regex extracts 2.0 m              → target=2.0
  Cmd3 "go higher"                    → "higher" keyword, no number       → +0.5 m (fixed prior)
  Cmd4 "go up a bit"                  → "up"+"bit" keywords, no number    → +0.3 m (fixed prior)
  Cmd5 "ascend slowly to a safe height"→ "safe height" keyword            → 1.5 m (fixed safe alt)
  Cmd6 "I want it higher"             → "higher" keyword, no number       → +0.5 m (fixed prior)

The rule system is deterministic — N=1 run is sufficient.
Scoring uses the same acceptance ranges as exp_C2_ambiguity.py.

Outputs:
  results/C2_runs_rule_baseline.csv
  results/C2_summary_rule_baseline.csv
"""

import re, csv, os, math

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Same commands + scoring windows as exp_C2_ambiguity.py ───────────────────
COMMANDS = [
    ("go to 2 metres",                    2.0,  (1.90, 2.10), "explicit"),
    ("climb to 2m",                        2.0,  (1.90, 2.10), "paraphrase"),
    ("go higher",                          None, (0.10, 1.50), "relative_no_num"),
    ("go up a bit",                        None, (0.05, 0.60), "vague_relative"),
    ("ascend slowly to a safe height",     None, (0.05, 2.00), "abstract"),
    ("I want it higher",                   None, (0.05, 2.00), "indirect"),
]

# ── Rule definitions ──────────────────────────────────────────────────────────
# Each rule returns (target_absolute, increment_over_current)
# We simulate starting altitude of ~2.0m (same as C2 post-Cmd1/2 state)
STARTING_Z = 1.0  # drone starts at 1.0m, rises to 2.0m after Cmd1

def _extract_number_metres(text):
    """Extract explicit metre value from text."""
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:metres?|meters?|m)\b',
        r'\bto\s+(\d+(?:\.\d+)?)\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def rule_interpret(command, z_current):
    """
    Deterministic rule matcher.
    Returns (target_m, increment_m, method_str)
    """
    lower = command.lower()

    # 1. Try to extract explicit metre value
    explicit = _extract_number_metres(command)
    if explicit is not None:
        return explicit, explicit - z_current, "regex_explicit"

    # 2. "safe height" / "safe altitude" → fixed 1.5 m
    if "safe height" in lower or "safe altitude" in lower:
        target = 1.5
        return target, target - z_current, "keyword_safe_height"

    # 3. "higher" or "up" with no number → fixed +0.5 m
    if any(w in lower for w in ["higher", "go up", "ascend", "climb up"]):
        if "bit" in lower or "little" in lower or "slightly" in lower:
            # "a bit" → smaller prior: +0.3 m
            target = z_current + 0.3
            return target, 0.3, "keyword_up_bit"
        else:
            # generic "higher" → +0.5 m
            target = z_current + 0.5
            return target, 0.5, "keyword_higher"

    # 4. No rule matched → no action
    return None, 0.0, "no_match"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── Single deterministic run ──────────────────────────────────────────────────
print("\n[C2-BASELINE] Rule-based interpreter (1 deterministic run)")
print("="*60)

z = STARTING_Z
run_rows = []

for idx, (command, gt_exact, gt_range, cmd_type) in enumerate(COMMANDS, start=1):
    z_before = round(z, 3)
    target, increment, method = rule_interpret(command, z_before)

    # Clamp to sim ceiling
    if target is not None:
        target = min(target, 2.5)
        increment = target - z_before
    z_after = round(z_before + increment, 3)
    z = z_after

    # Score
    if gt_exact is not None:
        correct = (target is not None and abs(target - gt_exact) <= 0.15)
    else:
        lo, hi = gt_range
        correct = lo <= increment <= hi

    print(f"  Cmd{idx} [{cmd_type:>16s}]: {command}")
    print(f"    rule={method}, target={target}, inc={increment:+.3f}m → {'' if correct else 'FAIL '}correct={correct}")

    run_rows.append({
        "run":         1,
        "cmd_idx":     idx,
        "command":     command,
        "cmd_type":    cmd_type,
        "z_before_m":  z_before,
        "z_after_m":   z_after,
        "increment_m": round(increment, 3),
        "target_m":    round(target, 3) if target is not None else "",
        "target_source": method,
        "asked_clarify": 0,
        "correct":     int(correct),
        "api_calls":   0,
        "tokens":      0,
    })

n_correct = sum(r["correct"] for r in run_rows)
print(f"\n  Overall: {n_correct}/{len(COMMANDS)} = {n_correct/len(COMMANDS):.2f}")

# ── Aggregated summary ────────────────────────────────────────────────────────
print("\n  Per-command summary:")
print(f"  {'Cmd':<6} {'Type':<20} {'Correct':<10} {'Rate':<8} {'95% CI'}")
summary_rows = []
for r in run_rows:
    lo, hi = wilson_ci(r["correct"], 1)
    print(f"  Cmd{r['cmd_idx']:<3} {r['cmd_type']:<20} {r['correct']}/1    "
          f"{r['correct']:.2f}     [{lo:.2f},{hi:.2f}]")
    summary_rows.append({
        "cmd_idx":        r["cmd_idx"],
        "command":        r["command"],
        "cmd_type":       r["cmd_type"],
        "n_correct":      r["correct"],
        "n_runs":         1,
        "success_rate":   r["correct"],
        "wilson_ci_lo":   round(lo, 3),
        "wilson_ci_hi":   round(hi, 3),
        "increment_mean_m": r["increment_m"],
        "increment_std_m":  0.0,
    })

# Overall
overall_ok = sum(r["correct"] for r in run_rows)
overall_n  = len(run_rows)
lo_all, hi_all = wilson_ci(overall_ok, overall_n)
summary_rows.append({
    "cmd_idx": "OVERALL", "command": "all commands", "cmd_type": "",
    "n_correct": overall_ok, "n_runs": overall_n,
    "success_rate": round(overall_ok / overall_n, 4),
    "wilson_ci_lo": round(lo_all, 4), "wilson_ci_hi": round(hi_all, 4),
    "increment_mean_m": "", "increment_std_m": "",
})

# ── Save ──────────────────────────────────────────────────────────────────────
out_runs = os.path.join(RESULTS, "C2_runs_rule_baseline.csv")
with open(out_runs, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
    w.writeheader()
    w.writerows(run_rows)
print(f"\nSaved runs   : {out_runs}")

out_sum = os.path.join(RESULTS, "C2_summary_rule_baseline.csv")
with open(out_sum, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)
print(f"Saved summary: {out_sum}")

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON: Rule-Baseline vs LLM models (C2, N=15)")
print("="*60)
print(f"{'Command':<35} {'Rule':>6} {'Claude':>8} {'GPT-4o':>8} {'Mini':>6}")
print("-"*70)

# Load LLM C2 results for comparison
def load_c2_rates(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("cmd_idx", "").isdigit():
                rows.append(r)
    rates = {}
    for idx in range(1, 7):
        cmd_rows = [r for r in rows if int(r["cmd_idx"]) == idx]
        if cmd_rows:
            rates[idx] = sum(int(r["correct"]) for r in cmd_rows) / len(cmd_rows)
    return rates

claude_rates = load_c2_rates(os.path.join(RESULTS, "C2_runs.csv"))
gpt4o_rates  = load_c2_rates(os.path.join(RESULTS, "C2_runs_gpt4o_guardrail_on.csv"))
mini_rates   = load_c2_rates(os.path.join(RESULTS, "C2_runs_gpt4omini_guardrail_on.csv"))

for r in run_rows:
    idx = r["cmd_idx"]
    rule_r  = r["correct"]
    cl_r    = claude_rates.get(idx, 0)
    gpt_r   = gpt4o_rates.get(idx, 0)
    mn_r    = mini_rates.get(idx, 0)
    print(f"Cmd{idx} {r['cmd_type']:<30} {rule_r:>6.2f} {cl_r:>8.2f} {gpt_r:>8.2f} {mn_r:>6.2f}")

rule_overall  = overall_ok / overall_n
cl_overall    = sum(claude_rates.values()) / 6
gpt_overall   = sum(gpt4o_rates.values()) / 6
mini_overall  = sum(mini_rates.values()) / 6
print("-"*70)
print(f"{'OVERALL':<35} {rule_overall:>6.2f} {cl_overall:>8.2f} {gpt_overall:>8.2f} {mini_overall:>6.2f}")
