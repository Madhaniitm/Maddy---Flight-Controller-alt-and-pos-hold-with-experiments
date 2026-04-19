"""
run_all_comparison.py — Guardrail comparison runner
====================================================
Runs every C-series experiment twice: once with guardrails ON, once OFF.
Outputs go to results/C*_runs_guardrail_on.csv  (and off), etc.

Usage:
    # Full comparison (guardrail on then off) for all experiments:
    python run_all_comparison.py

    # Single experiment both ways:
    python run_all_comparison.py --exp C1

    # One mode only:
    python run_all_comparison.py --guardrail on
    python run_all_comparison.py --guardrail off

    # Dry-run (print commands without executing):
    python run_all_comparison.py --dry-run
"""

import sys, os, subprocess, argparse, time

EXPERIMENTS_DIR = os.path.dirname(__file__)

# Ordered list: (script, label)
ALL_EXPERIMENTS = [
    ("exp_C1_nl_to_toolchain.py",        "C1"),
    ("exp_C2_ambiguity.py",              "C2"),
    ("exp_C2_1_prompt_fix.py",           "C2.1"),
    ("exp_C3_multiturn.py",              "C3"),
    ("exp_C4_mid_mission_correction.py", "C4"),
    ("exp_C4_1_retarget_fix.py",        "C4.1"),
    ("exp_C5_human_describes_problem.py","C5"),
    ("exp_C6_mission_planning.py",       "C6"),
    ("exp_C7_safety_override.py",        "C7"),
    ("exp_C8_three_mode_comparison.py",  "C8"),
]

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run C-series experiments with/without guardrails")
parser.add_argument("--exp",       default="all",
                    help='Experiment label(s) to run (e.g. "C1" or "C1,C2,C3"). Default: all')
parser.add_argument("--guardrail", choices=["on", "off", "both"], default="both",
                    help='Which guardrail mode(s) to run. Default: both')
parser.add_argument("--dry-run",   action="store_true",
                    help="Print commands without executing")
args = parser.parse_args()

# ── Filter experiments ────────────────────────────────────────────────────────
if args.exp.lower() == "all":
    experiments = ALL_EXPERIMENTS
else:
    labels = {l.strip().upper() for l in args.exp.split(",")}
    experiments = [(s, l) for s, l in ALL_EXPERIMENTS if l.upper() in labels]
    if not experiments:
        print(f"[ERROR] No matching experiments found for --exp {args.exp!r}")
        sys.exit(1)

# ── Guard modes ───────────────────────────────────────────────────────────────
modes = ["on", "off"] if args.guardrail == "both" else [args.guardrail]

# ── Runner ────────────────────────────────────────────────────────────────────
results = []
total = len(experiments) * len(modes)
run_n = 0

print(f"\n{'='*70}")
print(f"  C-Series Guardrail Comparison")
print(f"  Experiments : {', '.join(l for _, l in experiments)}")
print(f"  Modes       : {', '.join(modes)}")
print(f"  Total runs  : {total}")
print(f"{'='*70}\n")

for script, label in experiments:
    for mode in modes:
        run_n += 1
        cmd = [sys.executable,
               os.path.join(EXPERIMENTS_DIR, script),
               "--guardrail", mode]
        tag = f"[{run_n}/{total}] {label} guardrail={mode}"
        print(f"\n{'-'*60}")
        print(f"  {tag}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'-'*60}")

        if args.dry_run:
            results.append({"label": label, "mode": mode, "status": "dry-run", "elapsed_s": 0})
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=EXPERIMENTS_DIR,
                timeout=1800,   # 30 min hard cap per experiment
            )
            elapsed = round(time.time() - t0, 1)
            status = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - t0, 1)
            status = "timeout"
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            status = f"error: {e}"

        results.append({"label": label, "mode": mode, "status": status, "elapsed_s": elapsed})
        print(f"  → {status}  ({elapsed}s)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"  {'Exp':<8} {'Mode':<12} {'Status':<20} {'Elapsed':>10}")
print(f"  {'-'*50}")
for r in results:
    mark = "✓" if r["status"] in ("ok", "dry-run") else "✗"
    print(f"  {r['label']:<8} {r['mode']:<12} {mark} {r['status']:<18} {r['elapsed_s']:>8.1f}s")
print(f"{'='*70}\n")

ok_count = sum(1 for r in results if r["status"] in ("ok", "dry-run"))
print(f"  {ok_count}/{total} completed successfully.")
print(f"  Results saved to experiments/results/ with _guardrail_on / _guardrail_off suffixes.\n")
