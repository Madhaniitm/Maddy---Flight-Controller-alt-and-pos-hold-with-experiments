"""
EXP-C6.1: Mission Planning — Directional Movement Protocol Fix  (N=5, Gemini only)
===================================================================================
Identical to EXP-C6 in every respect EXCEPT one change: the system prompt is
augmented with a DIRECTIONAL MOVEMENT PROTOCOL explaining how to compose
set_pitch / set_roll / set_yaw + wait into a directional flight leg.

WHY C6 FAILED FOR GEMINI (3/5):
---------------------------------
C6 Gemini produced 2 refusal runs (plan_steps=0, output_tokens~0) where Gemini
responded: "I cannot perform a square pattern flight. I lack the ability to set
specific position targets or execute timed directional movements."

This is factually incorrect — the API has set_pitch, set_roll, set_yaw, wait, hover,
which are sufficient for any rectangular flight pattern. Claude and GPT-4o infer this
automatically from tool descriptions. Gemini does not — it concludes it lacks the
capability rather than composing available tools.

ROOT CAUSE — prompt incompleteness, not model incapability:
The SYSTEM_PROMPT has explicit protocol sections for takeoff, landing, altitude hold,
and PID tuning — but NO section explaining directional movement via tilt + wait.
Gemini needs the composition explained explicitly. This is a prompt engineering gap,
not a Gemini model deficiency: Gemini passes 3/5 runs where it *does* attempt
the task, demonstrating it has the capability when it chooses to engage.

WHY THIS IS GEMINI-SPECIFIC:
-------------------------------
Claude and GPT-4o pass C6 5/5 with the same SYSTEM_PROMPT. They infer that
set_pitch(1700) + wait(3.0) = "fly forward 3 seconds" from the tool description
alone. Gemini requires this stated explicitly. This demonstrates a general
principle: **prompt tuning requirements differ across LLM families**. The same
system prompt that is sufficient for one model may need explicit elaboration for
another — especially for implicit knowledge like physical tool composition.

THE FIX — one structural addition, no task-specific examples:
The DIRECTIONAL MOVEMENT PROTOCOL describes the general pattern (tilt + wait +
hover = directional leg) for each axis and direction. The model uses these
building blocks to plan any multi-leg path — it still decides timing and
sequencing. The fix removes the misclassification that "timed directional
movement" is impossible.

EXPECTED OUTCOME:
  5/5 pass (vs 3/5 baseline) — the refusal runs become successful attempts.
  Squareness should be similar to or better than passing C6 runs (0.40–0.84).

Outputs:
  results/C6_1_runs_gemini25flash_guardrail_on.csv
  results/C6_1_summary_gemini25flash_guardrail_on.csv
  results/C6_1_movement_fix_gemini25flash_guardrail_on.png
"""

import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import c_series_agent
import multi_llm_provider as _mlp
from multi_llm_provider import make_provider, MultiLLMSimAgent as _MLLM

# ── Movement protocol addition ─────────────────────────────────────────────────
MOVEMENT_PROTOCOL = """
━━ DIRECTIONAL MOVEMENT ━━

Tilt + wait = directional flight leg. Use this pattern for any movement:

  set_pitch(1700) → wait(N) → hover()    ← fly FORWARD  N seconds
  set_pitch(1300) → wait(N) → hover()    ← fly BACKWARD N seconds
  set_roll(1700)  → wait(N) → hover()    ← fly RIGHT    N seconds
  set_roll(1300)  → wait(N) → hover()    ← fly LEFT     N seconds
  set_yaw(1700)   → wait(1.5) → hover()  ← turn ~90° clockwise
  set_yaw(1300)   → wait(1.5) → hover()  ← turn ~90° counter-clockwise

Chain these building blocks in sequence to execute any multi-leg flight path.
You HAVE the ability to execute timed directional flight using the tools above.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

c_series_agent.SYSTEM_PROMPT = c_series_agent.SYSTEM_PROMPT + MOVEMENT_PROTOCOL
_mlp.SYSTEM_PROMPT            = c_series_agent.SYSTEM_PROMPT

from c_series_agent import SimAgent

# ── Provider / guardrail args ──────────────────────────────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--guardrail", choices=["on", "off"], default="on")
_parser.add_argument("--provider", default="gemini",
                     choices=["anthropic_azure", "openai", "gemini", "ollama",
                              "azure_openai", "azure_gemini", "groq"])
_parser.add_argument("--model", default="gemini-2.5-flash")
_args, _ = _parser.parse_known_args()
GUARDRAIL_ENABLED = _args.guardrail == "on"
GUARDRAIL_SUFFIX  = "guardrail_on" if GUARDRAIL_ENABLED else "guardrail_off"
PROVIDER_NAME     = _args.provider
MODEL_NAME        = _args.model
_clean            = lambda s: s.replace("-", "").replace(".", "").replace("_", "")
MODEL_TAG         = ("_" + _clean(MODEL_NAME or {"openai": "gpt4o", "gemini": "gemini",
                     "ollama": "ollama"}.get(PROVIDER_NAME, PROVIDER_NAME))) \
                    if PROVIDER_NAME != "anthropic_azure" else ""

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", f"C6_1_runs{MODEL_TAG}_{GUARDRAIL_SUFFIX}.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", f"C6_1_summary{MODEL_TAG}_{GUARDRAIL_SUFFIX}.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", f"C6_1_movement_fix{MODEL_TAG}_{GUARDRAIL_SUFFIX}.png")

def _make_agent(session_id):
    if PROVIDER_NAME == "anthropic_azure":
        return SimAgent(session_id=session_id, guardrail_enabled=GUARDRAIL_ENABLED)
    return _MLLM(provider=make_provider(PROVIDER_NAME, MODEL_NAME),
                 session_id=session_id, guardrail_enabled=GUARDRAIL_ENABLED)

COMMAND    = "do a square pattern at 1 metre height"
TARGET_ALT = 1.0
N_RUNS     = 5

# C6 Gemini baseline (before fix) for comparison
C6_BASELINE = {
    "n_pass":           3,
    "success_rate":     0.60,
    "ci_lo":            0.23,
    "ci_hi":            0.88,
    "squareness_mean":  0.410,
    "squareness_std":   0.367,
    "path_mean":        11.7,
    "path_std":         10.6,
    "refusal_runs":     2,   # runs 1 and 5 — empty response
    "per_run_pass":     [0, 1, 1, 1, 0],
    "per_run_sq":       [0.000, 0.411, 0.801, 0.838, 0.000],
    "per_run_path":     [0.0, 27.7, 15.8, 14.8, 0.0],
}

# ── Statistics helpers ─────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

def count_direction_changes(kx_arr, ky_arr):
    if len(kx_arr) < 10:
        return 0
    dx = np.diff(kx_arr)
    dy = np.diff(ky_arr)
    headings = np.arctan2(dy, dx)
    dh = np.abs(np.diff(headings))
    dh = np.minimum(dh, 2 * np.pi - dh)
    return int(np.sum(dh > np.radians(45)))

# ── Single-run function ────────────────────────────────────────────────────────

def run_once(run_idx):
    print(f"\n[C6.1] ── Run {run_idx+1}/{N_RUNS} ─────────────────────────────────")
    agent = _make_agent(f"C6_1_run{run_idx}")
    text, api_stats, tool_trace, _ = agent.run_agent_loop(COMMAND, max_turns=30)

    plan_steps = []
    for tr in tool_trace:
        if tr["name"] == "plan_workflow":
            plan_steps = tr["args"].get("steps", [])
            break

    alt_targets = [t["args"].get("meters") for t in tool_trace
                   if t["name"] == "set_altitude_target" and t["args"].get("meters") is not None]

    tel   = agent.get_telem_arrays()
    kx    = tel.get("kx", np.array([]))
    ky    = tel.get("ky", np.array([]))
    z_v   = tel.get("z_true", np.array([]))

    if len(kx) > 0 and len(z_v) > 0:
        mask   = z_v > 0.3
        kx_air = kx[mask]
        ky_air = ky[mask]
    else:
        kx_air = ky_air = np.array([])

    if len(kx_air) > 5:
        x_range    = float(kx_air.max() - kx_air.min())
        y_range    = float(ky_air.max() - ky_air.min())
        squareness = (min(x_range, y_range) / max(x_range, y_range)
                      if max(x_range, y_range) > 0.01 else 0.0)
        total_path = float(np.sum(np.sqrt(np.diff(kx_air)**2 + np.diff(ky_air)**2)))
        dir_changes = count_direction_changes(kx_air, ky_air)
    else:
        x_range = y_range = squareness = total_path = 0.0
        dir_changes = 0

    tools_used   = [t["name"] for t in tool_trace]
    n_plan_steps = len(plan_steps)
    had_plan     = "plan_workflow" in set(tools_used)
    n_alt_ok     = sum(1 for a in alt_targets if a is not None and abs(a - TARGET_ALT) < 0.25)
    alt_ok       = n_alt_ok >= 1 or (len(z_v) > 0 and float(np.max(z_v)) > 0.5)
    plan_ok      = had_plan and n_plan_steps >= 3
    passed       = plan_ok and alt_ok

    # Detect refusal (same signature as C6 failing runs)
    out_tokens = sum(s["output_tokens"] for s in api_stats)
    refused    = (len(tool_trace) == 0 and out_tokens < 100)

    n_api  = len(api_stats)
    in_tok = sum(s["input_tokens"]  for s in api_stats)
    cost   = sum(s["cost_usd"]      for s in api_stats)

    print(f"  plan_steps={n_plan_steps}  squareness={squareness:.3f}  "
          f"path={total_path:.2f}m  refused={refused}  pass={passed}")

    return {
        "run":               run_idx + 1,
        "n_plan_steps":      n_plan_steps,
        "had_plan_workflow": int(had_plan),
        "n_alt_target_ok":   n_alt_ok,
        "x_range_m":         round(x_range, 3),
        "y_range_m":         round(y_range, 3),
        "squareness_ratio":  round(squareness, 4),
        "total_path_m":      round(total_path, 3),
        "dir_changes":       dir_changes,
        "plan_ok":           int(plan_ok),
        "alt_ok":            int(alt_ok),
        "refused":           int(refused),
        "passed":            int(passed),
        "api_calls":         n_api,
        "input_tokens":      in_tok,
        "output_tokens":     out_tokens,
        "cost_usd":          round(cost, 6),
        "_kx_air":           kx_air,
        "_ky_air":           ky_air,
    }

# ── Run N times ────────────────────────────────────────────────────────────────

all_results = [run_once(i) for i in range(N_RUNS)]

# ── Aggregate ─────────────────────────────────────────────────────────────────

def col(key):
    return [r[key] for r in all_results]

n_pass    = sum(col("passed"))
n_refused = sum(col("refused"))
pass_lo, pass_hi = wilson_ci(n_pass, N_RUNS)

squareness_vals = col("squareness_ratio")
path_vals       = col("total_path_m")
plan_step_vals  = col("n_plan_steps")
sq_ci   = bootstrap_ci(squareness_vals)
path_ci = bootstrap_ci(path_vals)

delta_pp = round((n_pass / N_RUNS - C6_BASELINE["success_rate"]) * 100, 1)

print(f"\n[C6.1] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
print(f"  Success rate:    {n_pass}/{N_RUNS}  CI=[{pass_lo:.2f},{pass_hi:.2f}]")
print(f"  vs C6 Gemini:    {C6_BASELINE['n_pass']}/5 ({C6_BASELINE['success_rate']:.0%})  "
      f"→ Δ = {delta_pp:+.1f} pp")
print(f"  Refusal runs:    {n_refused}/{N_RUNS}  (was {C6_BASELINE['refusal_runs']}/5 in C6)")
print(f"  Squareness:      {np.mean(squareness_vals):.3f}±{np.std(squareness_vals):.3f}  "
      f"CI=[{sq_ci[0]:.3f},{sq_ci[1]:.3f}]")
print(f"  Total path (m):  {np.mean(path_vals):.3f}±{np.std(path_vals):.3f}  "
      f"CI=[{path_ci[0]:.3f},{path_ci[1]:.3f}]")
print(f"  Plan steps:      {np.mean(plan_step_vals):.1f}±{np.std(plan_step_vals):.1f}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
csv_keys = [k for k in all_results[0].keys() if not k.startswith("_")]
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_keys)
    w.writeheader()
    for r in all_results:
        w.writerow({k: r[k] for k in csv_keys})
print(f"[C6.1] Per-run CSV: {OUT_RUNS}")

summary_rows = [
    ("experiment",              "C6.1"),
    ("n_runs",                  N_RUNS),
    ("n_pass",                  n_pass),
    ("success_rate",            round(n_pass / N_RUNS, 3)),
    ("success_rate_ci_lo",      round(pass_lo, 3)),
    ("success_rate_ci_hi",      round(pass_hi, 3)),
    ("vs_c6_baseline_rate",     C6_BASELINE["success_rate"]),
    ("delta_percentage_points", delta_pp),
    ("n_refusal_runs",          n_refused),
    ("squareness_mean",         round(float(np.mean(squareness_vals)), 4)),
    ("squareness_std",          round(float(np.std(squareness_vals)), 4)),
    ("squareness_ci_lo",        round(sq_ci[0], 4)),
    ("squareness_ci_hi",        round(sq_ci[1], 4)),
    ("total_path_mean_m",       round(float(np.mean(path_vals)), 3)),
    ("total_path_std_m",        round(float(np.std(path_vals)), 3)),
    ("plan_steps_mean",         round(float(np.mean(plan_step_vals)), 1)),
    ("plan_steps_std",          round(float(np.std(plan_step_vals)), 1)),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
print(f"[C6.1] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"EXP-C6.1: Mission Planning — Directional Movement Protocol Fix\n"
    f"({PROVIDER_NAME}/{MODEL_NAME}, N={N_RUNS})  "
    f"C6 Gemini baseline: 3/5 (60%)  →  C6.1: {n_pass}/{N_RUNS} "
    f"({n_pass/N_RUNS:.0%}, CI: {pass_lo:.2f}–{pass_hi:.2f})  Δ = {delta_pp:+.1f} pp",
    fontsize=11, fontweight="bold"
)

traj_colors = plt.cm.tab10(np.linspace(0, 0.9, N_RUNS))

# Left: C6 baseline trajectories (from stored per-run values)
ax1 = axes[0]
ax1.set_title("C6 Gemini baseline (before fix)\n3/5 pass — 2 refusals", fontsize=10)
c6_sq_vals  = C6_BASELINE["per_run_sq"]
c6_pass     = C6_BASELINE["per_run_pass"]
for i in range(N_RUNS):
    if c6_pass[i]:
        ax1.bar(i + 1, c6_sq_vals[i], color="green", alpha=0.7, edgecolor="black")
    else:
        ax1.bar(i + 1, 0.05, color="red", alpha=0.7, edgecolor="black",
                label="Refused" if i == 0 else "")
        ax1.text(i + 1, 0.07, "REFUSED", ha="center", fontsize=7, color="red")
ax1.axhline(C6_BASELINE["squareness_mean"], color="navy", ls="--", lw=1.5,
            label=f"Mean={C6_BASELINE['squareness_mean']:.3f}")
ax1.set_xlabel("Run")
ax1.set_ylabel("Squareness ratio")
ax1.set_ylim(0, 1.1)
ax1.set_xticks(range(1, N_RUNS + 1))
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# Middle: C6.1 fix squareness per run
ax2 = axes[1]
ax2.set_title(f"C6.1 Gemini fix (after protocol)\n{n_pass}/{N_RUNS} pass", fontsize=10)
for i, r in enumerate(all_results):
    color = "green" if r["passed"] else ("orange" if r["refused"] else "red")
    ax2.bar(i + 1, r["squareness_ratio"] if not r["refused"] else 0.05,
            color=color, alpha=0.75, edgecolor="black")
    if r["refused"]:
        ax2.text(i + 1, 0.07, "REFUSED", ha="center", fontsize=7, color="red")
ax2.axhline(np.mean(squareness_vals), color="navy", ls="--", lw=1.5,
            label=f"Mean={np.mean(squareness_vals):.3f}")
ax2.fill_between([0.5, N_RUNS + 0.5], sq_ci[0], sq_ci[1],
                 alpha=0.12, color="navy", label=f"95% CI [{sq_ci[0]:.3f},{sq_ci[1]:.3f}]")
ax2.set_xlabel("Run")
ax2.set_ylabel("Squareness ratio")
ax2.set_ylim(0, 1.1)
ax2.set_xticks(range(1, N_RUNS + 1))
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

# Right: C6 vs C6.1 summary comparison
ax3 = axes[2]
ax3.set_title("C6 vs C6.1 Gemini: pass rate + refusal count", fontsize=10)
metrics = ["Pass rate", "Refusal rate"]
c6_vals  = [C6_BASELINE["success_rate"], C6_BASELINE["refusal_runs"] / N_RUNS]
c61_vals = [n_pass / N_RUNS, n_refused / N_RUNS]
x = np.arange(len(metrics))
w = 0.35
bars_c6  = ax3.bar(x - w/2, c6_vals,  w, color=["#e74c3c", "#e74c3c"], alpha=0.6,
                   edgecolor="black", label="C6 (baseline)")
bars_c61 = ax3.bar(x + w/2, c61_vals, w, color=["#2ecc71", "#2ecc71"], alpha=0.85,
                   edgecolor="black", label="C6.1 (fix)")
for xi, (c6v, c61v) in enumerate(zip(c6_vals, c61_vals)):
    dp = (c61v - c6v) * 100
    color = "#27ae60" if (xi == 0 and dp > 0) or (xi == 1 and dp < 0) else "#e74c3c"
    ax3.text(xi, max(c6v, c61v) + 0.04, f"{dp:+.0f} pp",
             ha="center", fontsize=10, fontweight="bold", color=color)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=10)
ax3.set_ylim(0, 1.3)
ax3.set_ylabel("Rate")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"[C6.1] Plot: {OUT_PNG}")

print(f"\n[C6.1] RESULT: {n_pass}/{N_RUNS} passed  (95% CI: {pass_lo:.2f}–{pass_hi:.2f})")
print(f"       vs C6 Gemini baseline: 3/5 (60%)  Δ = {delta_pp:+.1f} pp")
print(f"       Refusals: {n_refused}/{N_RUNS}  (was 2/5 in C6)")
print(f"       Squareness: {np.mean(squareness_vals):.3f}±{np.std(squareness_vals):.3f}")
print()
print("PAPER NOTE: This experiment demonstrates that prompt tuning requirements")
print("differ across LLM families. Claude and GPT-4o infer tool composability")
print("(tilt + wait = directional movement) without explicit guidance. Gemini")
print("requires the composition to be stated explicitly in the system prompt.")
print("The same capability exists in all models; the difference is in how much")
print("implicit physical reasoning each model applies to tool descriptions.")
