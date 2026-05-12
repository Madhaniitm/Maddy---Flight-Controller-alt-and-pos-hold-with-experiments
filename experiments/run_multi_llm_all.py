"""
run_multi_llm_all.py
====================
Master runner — executes all C-series experiments (C1–C8) for each
non-Claude LLM provider and generates a cross-model summary CSV.

Claude results are NOT re-run (use existing results/C*_*_guardrail_on.csv).
This script only runs: GPT-4o, GPT-4o-mini, Gemini-2.0-Flash, LLaMA-3.1 (Ollama).

Usage
-----
  # All models, all experiments
  python run_multi_llm_all.py

  # Specific models only
  python run_multi_llm_all.py --providers openai/gpt-4o-mini gemini/gemini-2.0-flash

  # Specific experiments only
  python run_multi_llm_all.py --experiments C1 C5 C7

  # Quick smoke-test (1 run each)
  python run_multi_llm_all.py --n_runs 1 --providers ollama/llama3.1

Required env vars
-----------------
  OPENAI_API_KEY   — for openai provider
  GEMINI_API_KEY   — for gemini provider
  (Ollama needs local server running on localhost:11434, no key)

Outputs
-------
  results/C*_runs_<model>_guardrail_on.csv     — per experiment per model
  results/mllm_all_summary.csv                 — cross-model comparison table
"""

import os, sys, subprocess, csv, argparse, time, math

EXPERIMENTS_DIR = os.path.dirname(__file__)
RESULTS_DIR     = os.path.join(EXPERIMENTS_DIR, "results")

# (provider, model, short_label)
DEFAULT_PROVIDERS = [
    ("azure_openai", "gpt-4o",                 "gpt4o"),
    ("azure_openai", "gpt-4o-mini",             "gpt4omini"),
    ("azure_openai", "grok-4-1-fast-reasoning",     "grok41fast"),
    ("azure_openai", "grok-4-20-reasoning",         "grok420"),
    ("groq",         "llama-3.3-70b-versatile", "llama33"),
]

EXPERIMENT_SCRIPTS = [
    ("C1",   "exp_C1_nl_to_toolchain.py"),
    ("C2",   "exp_C2_ambiguity.py"),
    ("C3",   "exp_C3_multiturn.py"),
    ("C4",   "exp_C4_mid_mission_correction.py"),
    ("C5",   "exp_C5_human_describes_problem.py"),
    ("C6",   "exp_C6_mission_planning.py"),
    ("C7",   "exp_C7_safety_override.py"),
    ("C8",   "exp_C8_three_mode_comparison.py"),
    ("C7_2", "exp_C7_2_adversarial_disarm.py"),
    ("GV",   "exp_guardrail_validation.py"),
]

# Experiments that need guardrail OFF run in addition to ON:
#   C5  — PID gain clipping fires in C5
#   C7  — safety override; compare LLM behaviour without guardrail backstop
#   C8  — non-invasiveness check
#   C7_2 — adversarial disarm; OFF reveals whether model attempts disarm without guardrail
# GV always runs guardrail ON (it IS the guardrail validation — OFF makes no sense)
NEEDS_GUARDRAIL_OFF = {"C5", "C7", "C8", "C7_2"}
SKIP_GUARDRAIL_ON   = set()   # all experiments run guardrail ON


def run_experiment(script, provider, model, n_runs, guardrail="on"):
    cmd = [
        sys.executable,
        os.path.join(EXPERIMENTS_DIR, script),
        "--provider", provider,
        "--model",    model,
        "--guardrail", guardrail,
    ]
    if n_runs is not None:
        cmd += ["--n_runs", str(n_runs)]

    print(f"\n{'='*60}")
    print(f"  {script}  |  {provider}/{model}  |  guardrail={guardrail}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=EXPERIMENTS_DIR)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  → {status}  ({elapsed:.0f}s)")
    return result.returncode == 0


def read_summary_value(path, key):
    """Read a value from a summary CSV (metric,value format)."""
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("metric") == key:
                    return row.get("value", "")
    except Exception:
        pass
    return ""


def read_overall_pass_rate(path):
    """Read overall pass rate from a summary CSV (handles all formats)."""
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        # metric,value format (C4, C8, C7_2 etc.)
        for row in rows:
            if row.get("metric") in ("success_rate", "pass_rate", "safe_rate"):
                return float(row["value"])
        # guardrail validation: report guardrail_fired rate over total LLM runs
        total, fired = None, None
        for row in rows:
            if row.get("metric") == "part_B_llm_runs_total":
                total = float(row["value"])
            if row.get("metric") == "part_B_guardrail_fired":
                fired = float(row["value"])
        if total is not None and fired is not None and total > 0:
            return fired / total
        # cmd_idx format (C2 etc.) — look for OVERALL row
        for row in rows:
            if str(row.get("cmd_idx", "")).upper() == "OVERALL":
                return float(row.get("success_rate", "nan"))
        # first numeric success_rate column
        for row in rows:
            v = row.get("success_rate", "")
            try:
                return float(v)
            except ValueError:
                continue
    except Exception:
        pass
    return float("nan")


def build_summary(providers_run, experiments_run):
    """Read all result files and build cross-model summary."""
    # Claude baseline row (existing results)
    claude_label = "claude-sonnet-4-6"
    rows = []

    all_labels = [claude_label] + [label for _, _, label in providers_run]

    for label in all_labels:
        row = {"model": label}
        for exp_id, _ in experiments_run:
            if label == claude_label:
                if exp_id == "GV":
                    # Guardrail validation uses different file naming
                    summary_path = os.path.join(
                        RESULTS_DIR, "guardrail_validation_summary.csv")
                elif exp_id == "C7_2":
                    summary_path = os.path.join(
                        RESULTS_DIR, "C7_2_summary_guardrail_on.csv")
                else:
                    summary_path = os.path.join(
                        RESULTS_DIR, f"{exp_id}_summary_guardrail_on.csv")
            else:
                if exp_id == "GV":
                    summary_path = os.path.join(
                        RESULTS_DIR, f"guardrail_validation_summary_{label}.csv")
                elif exp_id == "C7_2":
                    summary_path = os.path.join(
                        RESULTS_DIR, f"C7_2_summary_{label}_guardrail_on.csv")
                else:
                    summary_path = os.path.join(
                        RESULTS_DIR, f"{exp_id}_summary_{label}_guardrail_on.csv")
            rate = read_overall_pass_rate(summary_path)
            row[f"{exp_id}_pass"] = round(rate, 3) if not math.isnan(rate) else "N/A"
        rows.append(row)

    out = os.path.join(RESULTS_DIR, "mllm_all_summary.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[Summary] {out}")

    # Print table
    print("\n" + "="*90)
    exp_ids = [e for e, _ in experiments_run]
    header = f"{'Model':<22}" + "".join(f"{e:>9}" for e in exp_ids)
    print(header)
    print("-"*90)
    for row in rows:
        line = f"{row['model']:<22}"
        for e in exp_ids:
            v = row.get(f"{e}_pass", "N/A")
            line += f"{str(v):>9}"
        print(line)
    print("="*90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=None,
                    help="provider/model pairs e.g. openai/gpt-4o-mini gemini/gemini-2.0-flash")
    ap.add_argument("--experiments", nargs="+", default=None,
                    help="Experiment IDs to run e.g. C1 C5 C7")
    ap.add_argument("--n_runs", type=int, default=None,
                    help="Override N_RUNS in each script (default: use script default)")
    ap.add_argument("--rest_s", type=float, default=5.0,
                    help="Seconds to rest between experiments (default 5)")
    args = ap.parse_args()

    # Parse providers
    if args.providers:
        providers_run = []
        for p in args.providers:
            if "/" in p:
                pname, model = p.split("/", 1)
            else:
                pname, model = p, None
            label = (model or pname).replace("-", "").replace(".", "").replace("_", "")
            providers_run.append((pname, model or "", label))
    else:
        providers_run = DEFAULT_PROVIDERS

    # Parse experiments
    exp_ids = set(args.experiments) if args.experiments else None
    experiments_run = [(eid, s) for eid, s in EXPERIMENT_SCRIPTS
                       if exp_ids is None or eid in exp_ids]

    print(f"Models   : {[l for _,_,l in providers_run]}")
    print(f"Exps     : {[e for e,_ in experiments_run]}")
    print(f"N_RUNS   : {args.n_runs or 'script default'}")

    failed = []
    for pname, model, label in providers_run:
        for exp_id, script in experiments_run:
            # guardrail ON — all experiments
            ok = run_experiment(script, pname, model, args.n_runs, guardrail="on")
            if not ok:
                failed.append(f"{exp_id}/{label}/on")
            if args.rest_s > 0:
                time.sleep(args.rest_s)

            # guardrail OFF — only C5, C7, C8 (guardrail fires meaningfully there)
            if exp_id in NEEDS_GUARDRAIL_OFF:
                ok = run_experiment(script, pname, model, args.n_runs, guardrail="off")
                if not ok:
                    failed.append(f"{exp_id}/{label}/off")
                if args.rest_s > 0:
                    time.sleep(args.rest_s)

    build_summary(providers_run, experiments_run)

    if failed:
        print(f"\n[WARN] Failed runs: {failed}")
    else:
        print("\n[Done] All runs completed successfully.")


if __name__ == "__main__":
    main()
