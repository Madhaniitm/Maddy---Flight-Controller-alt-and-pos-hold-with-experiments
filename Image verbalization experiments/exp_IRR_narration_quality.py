"""
EXP-IRR: Inter-Rater Reliability for Narration Quality
=======================================================
Goal:
    Establish inter-rater weighted Cohen's κ for the 0–5 narration
    quality rubric used in G5. Required for RAL reviewer credibility.
    Target: weighted κ ≥ 0.61 (substantial agreement).

Rubric (0–5):
    5 — Accurate scene description, correct risk identification,
        specific spatial details, clear action rationale
    4 — Mostly accurate, one minor omission
    3 — Correct action recommended, but vague/generic description
    2 — Partially correct; at least one misleading spatial claim
    1 — Significant scene description error, but not completely wrong
    0 — Hallucinated content or completely incorrect response

Usage:
    STEP 1 — Rater 1 scores (run this yourself):
        python3.11 exp_IRR_narration_quality.py --rater 1

    STEP 2 — Rater 2 scores (give to a second person, same command):
        python3.11 exp_IRR_narration_quality.py --rater 2

    STEP 3 — Compute kappa:
        python3.11 exp_IRR_narration_quality.py --kappa

    Each rater session takes ~20 minutes for 48 narrations.
    The narrations are presented WITHOUT the scene label (blind scoring).

Input:
    Most recent G5_runs_*.csv from results/
    (or specify with --csv path/to/file.csv)

Output:
    results/IRR_rater1_scores_<ts>.csv
    results/IRR_rater2_scores_<ts>.csv
    results/IRR_kappa_result.txt
"""

import argparse, csv, datetime, glob, os, pathlib, random, sys, textwrap

VIZ_DIR    = pathlib.Path(__file__).parent
RESULTS    = VIZ_DIR / "results"
RUBRIC     = textwrap.dedent("""
    ┌────────────────────────────────────────────────────────────────┐
    │  NARRATION QUALITY RUBRIC  (0–5)                               │
    │                                                                │
    │  5 — Accurate scene, correct risk, specific spatial details,  │
    │      clear action rationale                                    │
    │  4 — Mostly accurate, one minor omission                      │
    │  3 — Correct action, but vague or generic description         │
    │  2 — Partially correct; at least one misleading claim         │
    │  1 — Significant scene error, not completely wrong            │
    │  0 — Hallucinated / completely incorrect                      │
    │                                                                │
    │  Score based on the NARRATION TEXT ONLY.                      │
    │  The scene label is hidden — score on content quality alone.  │
    └────────────────────────────────────────────────────────────────┘
""").strip()

SEPARATOR = "─" * 68


def find_latest_g5_csv() -> pathlib.Path:
    pattern = str(RESULTS / "G5_runs_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No G5_runs_*.csv found in {RESULTS}. Run exp_G5 first.")
    return pathlib.Path(files[-1])


def load_sample(csv_path: pathlib.Path, n_per_scene: int = 6) -> list[dict]:
    """
    Load stratified sample from G5 runs CSV.
    Returns up to n_per_scene rows per scene, balanced across models.
    Only includes rows with non-empty reply and no error.
    """
    rows_by_scene: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error") or not row.get("reply", "").strip():
                continue
            scene = row["scene_label"]
            rows_by_scene.setdefault(scene, []).append(row)

    sample = []
    for scene, rows in rows_by_scene.items():
        # Pick models evenly (round-robin across models)
        by_model: dict = {}
        for r in rows:
            by_model.setdefault(r["model"], []).append(r)
        picked = []
        model_queues = list(by_model.values())
        i = 0
        while len(picked) < n_per_scene and any(model_queues):
            q = model_queues[i % len(model_queues)]
            if q:
                picked.append(q.pop(0))
            i += 1
        sample.extend(picked)

    # Shuffle with a fixed seed so both raters see the same order
    rng = random.Random(42)
    rng.shuffle(sample)
    return sample


def get_seed_file() -> pathlib.Path:
    """Seed file stores the fixed sample so both raters score identical narrations."""
    return RESULTS / "IRR_sample_seed.csv"


def save_seed(sample: list[dict]):
    seed_path = get_seed_file()
    fields = ["idx", "model", "scene_label", "truth", "run", "reply", "quality_score"]
    rows = [{**{k: row.get(k, "") for k in fields}, "idx": i+1}
            for i, row in enumerate(sample)]
    with open(seed_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return seed_path


def load_seed() -> list[dict]:
    seed_path = get_seed_file()
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Seed file not found: {seed_path}\n"
            "Run --rater 1 first to generate the seed.")
    with open(seed_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_rating_session(rater_id: int, sample: list[dict]) -> pathlib.Path:
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path   = RESULTS / f"IRR_rater{rater_id}_scores_{ts}.csv"
    scored     = []

    print(f"\n{'='*68}")
    print(f"  NARRATION QUALITY RATING — RATER {rater_id}")
    print(f"  {len(sample)} narrations to score")
    print(f"{'='*68}")
    print(RUBRIC)
    print(f"\nPress ENTER to begin. Type a score (0–5) for each narration.")
    input("> ")

    for i, row in enumerate(sample, 1):
        print(f"\n{SEPARATOR}")
        print(f"  Narration {i:02d} / {len(sample)}")
        print(SEPARATOR)

        # Show narration WITHOUT scene label (blind scoring)
        narration = row.get("reply", "").strip()
        wrapped   = textwrap.fill(narration, width=68,
                                  initial_indent="  ", subsequent_indent="  ")
        print(f"\n{wrapped}\n")
        print(f"[{RUBRIC.splitlines()[2]}]")

        while True:
            raw = input(f"  Score (0–5): ").strip()
            if raw in ("0","1","2","3","4","5"):
                score = int(raw)
                break
            if raw.lower() in ("q","quit","exit"):
                print(f"\n[IRR] Session interrupted at narration {i}. Partial results not saved.")
                sys.exit(0)
            print("  Please enter a number 0–5  (or 'q' to quit)")

        scored.append({
            "idx":             row.get("idx", i),
            "rater":           rater_id,
            "scene_label":     row.get("scene_label", ""),
            "truth":           row.get("truth", ""),
            "model":           row.get("model", ""),
            "run":             row.get("run", ""),
            "human_score":     score,
            "auto_score":      row.get("quality_score", ""),
        })
        print(f"  ✓ Recorded: {score}/5")

    # Save results
    fields = ["idx","rater","scene_label","truth","model","run",
              "human_score","auto_score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(scored)

    print(f"\n{SEPARATOR}")
    print(f"  Rating complete — {len(scored)} narrations scored.")
    print(f"  Saved → {out_path}")
    print(f"{SEPARATOR}\n")
    return out_path


def compute_kappa():
    """Find the two most recent rater files and compute weighted Cohen's κ."""
    r1_files = sorted(glob.glob(str(RESULTS / "IRR_rater1_scores_*.csv")))
    r2_files = sorted(glob.glob(str(RESULTS / "IRR_rater2_scores_*.csv")))

    if not r1_files:
        print("[IRR] No Rater 1 file found. Run --rater 1 first.")
        sys.exit(1)
    if not r2_files:
        print("[IRR] No Rater 2 file found. Run --rater 2 first.")
        sys.exit(1)

    r1_path = pathlib.Path(r1_files[-1])
    r2_path = pathlib.Path(r2_files[-1])
    print(f"[IRR] Rater 1 file: {r1_path.name}")
    print(f"[IRR] Rater 2 file: {r2_path.name}")

    def load_scores(path):
        scores = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores[int(row["idx"])] = int(row["human_score"])
        return scores

    s1 = load_scores(r1_path)
    s2 = load_scores(r2_path)

    common = sorted(set(s1) & set(s2))
    if len(common) < 10:
        print(f"[IRR] Only {len(common)} matching narration IDs — need at least 10.")
        sys.exit(1)

    y1 = [s1[i] for i in common]
    y2 = [s2[i] for i in common]

    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(y1, y2, weights="quadratic")
    except ImportError:
        print("[IRR] sklearn not available — computing kappa manually.")
        kappa = _manual_weighted_kappa(y1, y2)

    # Interpretation
    if kappa >= 0.80:
        interp = "almost perfect agreement"
    elif kappa >= 0.61:
        interp = "substantial agreement"
    elif kappa >= 0.41:
        interp = "moderate agreement — consider revising rubric"
    else:
        interp = "fair/poor agreement — rubric needs clarification"

    # Per-scene breakdown
    seed = {int(r["idx"]): r for r in load_seed()}
    scene_kappas: dict = {}
    for i in common:
        sc = seed[i]["scene_label"] if i in seed else "unknown"
        scene_kappas.setdefault(sc, ([], []))
        scene_kappas[sc][0].append(s1[i])
        scene_kappas[sc][1].append(s2[i])

    # Absolute agreement rate
    agree     = sum(1 for a, b in zip(y1, y2) if a == b)
    within1   = sum(1 for a, b in zip(y1, y2) if abs(a-b) <= 1)
    mean_diff = sum(abs(a-b) for a, b in zip(y1, y2)) / len(y1)

    result_lines = [
        f"Inter-Rater Reliability — Narration Quality (0–5)",
        f"",
        f"  Rater 1 file : {r1_path.name}",
        f"  Rater 2 file : {r2_path.name}",
        f"  N (matched)  : {len(common)}",
        f"",
        f"  Weighted κ   : {kappa:.4f}  ({interp})",
        f"  Exact agree  : {agree}/{len(common)} = {agree/len(common)*100:.1f}%",
        f"  Within-1 agree: {within1}/{len(common)} = {within1/len(common)*100:.1f}%",
        f"  Mean |diff|  : {mean_diff:.3f}",
        f"",
        f"  Per-scene (Rater 1 mean / Rater 2 mean):",
    ]
    for sc, (a, b) in sorted(scene_kappas.items()):
        m1, m2 = sum(a)/len(a), sum(b)/len(b)
        result_lines.append(f"    {sc:20s}  {m1:.2f} / {m2:.2f}  (n={len(a)})")

    result_text = "\n".join(result_lines)
    print("\n" + result_text)

    out = RESULTS / "IRR_kappa_result.txt"
    out.write_text(result_text + "\n", encoding="utf-8")
    print(f"\n[IRR] Full result saved → {out}")

    # One-liner for paper
    print(f"\n  LaTeX sentence:")
    print(f'  "Two independent evaluators scored {len(common)} narrations using a 0--5')
    print(f'   rubric (inter-rater weighted \\kappa = {kappa:.2f}), indicating {interp}."')


def _manual_weighted_kappa(y1, y2, max_score=5):
    """Quadratic-weighted Cohen's kappa without sklearn."""
    import numpy as np
    n    = len(y1)
    cats = list(range(max_score + 1))
    k    = len(cats)
    O    = np.zeros((k, k))
    for a, b in zip(y1, y2):
        O[a][b] += 1
    O /= n
    W = np.array([[(i-j)**2 / (k-1)**2 for j in cats] for i in cats])
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    kappa = 1 - (W * O).sum() / (W * E).sum()
    return float(kappa)


def main():
    parser = argparse.ArgumentParser(
        description="Inter-rater reliability for narration quality scoring")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rater", type=int, choices=[1, 2],
                       help="Run a rating session for rater 1 or 2")
    group.add_argument("--kappa", action="store_true",
                       help="Compute kappa from existing rater files")
    parser.add_argument("--csv", default=None,
                        help="Path to G5_runs CSV (default: latest in results/)")
    parser.add_argument("--n-per-scene", type=int, default=6,
                        help="Narrations per scene in stratified sample (default 6)")
    args = parser.parse_args()

    if args.kappa:
        compute_kappa()
        return

    # Rating session
    rater_id = args.rater
    if rater_id == 1:
        # Rater 1 generates the fixed seed
        csv_path = pathlib.Path(args.csv) if args.csv else find_latest_g5_csv()
        print(f"[IRR] Loading narrations from: {csv_path.name}")
        sample = load_sample(csv_path, n_per_scene=args.n_per_scene)
        seed_path = save_seed(sample)
        print(f"[IRR] Seed file saved → {seed_path}")
        print(f"[IRR] Give {seed_path} to Rater 2, or they can run --rater 2 on the same machine.")
    else:
        # Rater 2 loads the fixed seed from Rater 1
        sample = load_seed()
        print(f"[IRR] Loaded {len(sample)} narrations from seed file.")

    run_rating_session(rater_id, sample)


if __name__ == "__main__":
    main()
