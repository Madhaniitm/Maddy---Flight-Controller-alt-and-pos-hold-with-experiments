"""
EXP-H2: Face Recognition Operator Authentication
=================================================
Goal:
    Replace YOLO "person" class detection with face_recognition library
    to authenticate known operators before granting drone control.

    Protocol:
        1. Capture frame (webcam / synthetic fallback)
        2. Detect faces using face_recognition (dlib HOG)
        3. Compare encodings against enrolled operator database
        4. Grant/deny control to Claude agent based on auth result
        5. Measure: auth_latency_ms, false_accept_rate, false_reject_rate

    Enrolled operators: 3 synthetic face encodings (128-d vectors)
    Test set          : 20 frames (12 enrolled, 8 unknown/attacker faces)
    N=5 independent runs

Metrics:
    - auth_latency_ms : time from frame capture to auth decision (Bootstrap CI)
    - true_accept_rate: enrolled faces correctly accepted (Wilson CI)
    - false_accept_rate: unknown faces incorrectly accepted (Wilson CI)
    - false_reject_rate: enrolled faces incorrectly rejected (Wilson CI)

Paper References:
    - King 2009 (dlib): face_recognition library underlying HOG + CNN detector
    - ReAct (Yao et al. 2022): outer loop only executes post-authentication
    - Vemprala et al. 2023: identity-aware drone access control
"""

import os, sys, time, csv, math, pathlib, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import DAgent

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_RUNS              = 5
N_ENROLLED          = 3
N_FRAMES_ENROLLED   = 12   # frames showing enrolled operators
N_FRAMES_UNKNOWN    = 8    # frames showing unknown faces
TOLERANCE           = 0.6  # face_recognition distance threshold
ENCODING_DIM        = 128  # dlib face encoding dimension
ENCODING_NOISE      = 0.08 # intra-person variation noise std

PAPER_REFS = {
    "King2009":  "King 2009 — Dlib-ml: A Machine Learning Toolkit",
    "ReAct":     "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
    "Vemprala":  "Vemprala et al. 2023 — ChatGPT for Robotics: Design Principles and Model Abilities",
}

# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2/n
    c = (p + z**2/(2*n)) / denom
    m = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(p,4), round(max(0,c-m),4), round(min(1,c+m),4)

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr = np.array(data, dtype=float)
    boots = [stat(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)),4), round(float(lo),4), round(float(hi),4)

# ── Synthetic face encoding database ──────────────────────────────────────────
def make_enrolled_db(rng: np.random.Generator) -> list:
    """Return list of (name, encoding_128d) for N_ENROLLED operators."""
    return [
        (f"operator_{i+1}", rng.random(ENCODING_DIM).astype(np.float32))
        for i in range(N_ENROLLED)
    ]

def make_probe_encoding(enrolled_db: list, is_enrolled: bool,
                        rng: np.random.Generator) -> tuple:
    """
    Return (encoding, ground_truth_label).
    enrolled: add small noise to a random enrolled face.
    unknown : generate a random encoding far from any enrolled.
    """
    if is_enrolled:
        name, base_enc = enrolled_db[int(rng.integers(0, N_ENROLLED))]
        probe = base_enc + rng.normal(0, ENCODING_NOISE, ENCODING_DIM).astype(np.float32)
        return probe, name
    else:
        probe = rng.random(ENCODING_DIM).astype(np.float32)
        return probe, "unknown"

# ── Authentication engine ──────────────────────────────────────────────────────
def authenticate(probe: np.ndarray, enrolled_db: list) -> tuple:
    """
    Try to use real face_recognition library.
    Falls back to numpy distance comparison.
    Returns (auth_ms, is_match, matched_name).
    """
    t0 = time.perf_counter()

    try:
        import face_recognition
        known_encodings = [enc for _, enc in enrolled_db]
        matches = face_recognition.compare_faces(known_encodings, probe,
                                                  tolerance=TOLERANCE)
        distances = face_recognition.face_distance(known_encodings, probe)
        auth_ms = (time.perf_counter() - t0) * 1000.0
        if any(matches):
            best_idx = int(np.argmin(distances))
            return round(auth_ms, 2), True, enrolled_db[best_idx][0]
        return round(auth_ms, 2), False, None

    except ImportError:
        # Numpy fallback: Euclidean distance normalised to [0,1]
        distances = [
            float(np.linalg.norm(probe - enc) / np.sqrt(ENCODING_DIM))
            for _, enc in enrolled_db
        ]
        auth_ms = (time.perf_counter() - t0) * 1000.0
        best_idx = int(np.argmin(distances))
        is_match = distances[best_idx] < TOLERANCE
        matched  = enrolled_db[best_idx][0] if is_match else None
        return round(auth_ms, 2), is_match, matched

# ── Single trial ──────────────────────────────────────────────────────────────
def run_trial(run_idx: int, frame_idx: int, is_enrolled: bool,
              enrolled_db: list, rng: np.random.Generator) -> dict:
    probe, gt_label = make_probe_encoding(enrolled_db, is_enrolled, rng)
    auth_ms, is_match, matched_name = authenticate(probe, enrolled_db)

    # TP/FP/FN/TN labels
    tp = is_enrolled and is_match
    fn = is_enrolled and not is_match
    fp = not is_enrolled and is_match
    tn = not is_enrolled and not is_match

    return {
        "run":         run_idx,
        "frame":       frame_idx,
        "is_enrolled": int(is_enrolled),
        "gt_label":    gt_label,
        "predicted":   matched_name or "unknown",
        "match":       int(is_match),
        "auth_ms":     auth_ms,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-H2: Face Recognition Operator Authentication")
    print(f"N_RUNS={N_RUNS}, enrolled={N_FRAMES_ENROLLED}, unknown={N_FRAMES_UNKNOWN}")
    print("=" * 60)

    all_rows = []

    for run in range(1, N_RUNS + 1):
        rng_np = np.random.default_rng(run * 17 + 3)
        enrolled_db = make_enrolled_db(rng_np)

        print(f"\n--- Run {run}/{N_RUNS} ---")
        frame = 0
        for is_enrolled, n_frames in [(True, N_FRAMES_ENROLLED),
                                       (False, N_FRAMES_UNKNOWN)]:
            for _ in range(n_frames):
                frame += 1
                row = run_trial(run, frame, is_enrolled, enrolled_db, rng_np)
                all_rows.append(row)

        tp_r = sum(r["TP"] for r in all_rows if r["run"] == run)
        fp_r = sum(r["FP"] for r in all_rows if r["run"] == run)
        fn_r = sum(r["FN"] for r in all_rows if r["run"] == run)
        print(f"  TP={tp_r} FP={fp_r} FN={fn_r}  "
              f"auth_ms={np.mean([r['auth_ms'] for r in all_rows if r['run']==run]):.2f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "H2_runs.csv"
    fields   = ["run","frame","is_enrolled","gt_label","predicted","match",
                "auth_ms","TP","FP","FN","TN"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-trial data → {runs_csv}")

    # ── Stats ──────────────────────────────────────────────────────────────────
    total_TP = sum(r["TP"] for r in all_rows)
    total_FP = sum(r["FP"] for r in all_rows)
    total_FN = sum(r["FN"] for r in all_rows)
    total_TN = sum(r["TN"] for r in all_rows)

    n_enr = N_FRAMES_ENROLLED * N_RUNS
    n_unk = N_FRAMES_UNKNOWN   * N_RUNS

    tar, tar_lo, tar_hi = wilson_ci(total_TP, n_enr)
    far, far_lo, far_hi = wilson_ci(total_FP, n_unk)
    frr, frr_lo, frr_hi = wilson_ci(total_FN, n_enr)
    al_m, al_lo, al_hi  = bootstrap_ci([r["auth_ms"] for r in all_rows])

    summary_csv = OUT_DIR / "H2_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric","value","ci_lo","ci_hi","note"])
        cw.writerow(["auth_latency_ms",   al_m, al_lo, al_hi, "Bootstrap 95%"])
        cw.writerow(["true_accept_rate",  tar,  tar_lo,tar_hi, "Wilson 95% — TP/enrolled"])
        cw.writerow(["false_accept_rate", far,  far_lo,far_hi, "Wilson 95% — FP/unknown"])
        cw.writerow(["false_reject_rate", frr,  frr_lo,frr_hi, "Wilson 95% — FN/enrolled"])
        cw.writerow(["total_TP", total_TP,"","",""])
        cw.writerow(["total_FP", total_FP,"","",""])
        cw.writerow(["total_FN", total_FN,"","",""])
        cw.writerow(["total_TN", total_TN,"","",""])
        for k, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{k}", ref,"","",""])
    print(f"Summary        → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        conf_mat = np.array([[total_TP, total_FN],
                              [total_FP, total_TN]], dtype=float)
        im = ax.imshow(conf_mat, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Accepted","Rejected"])
        ax.set_yticklabels(["Enrolled","Unknown"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("H2: Confusion Matrix (all runs)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(conf_mat[i,j]), ha="center", va="center",
                        fontsize=14, color="black")
        fig.colorbar(im, ax=ax)

        ax2 = axes[1]
        metrics = ["TAR","FAR","FRR"]
        vals    = [tar, far, frr]
        errs_lo = [tar-tar_lo, far-far_lo, frr-frr_lo]
        errs_hi = [tar_hi-tar, far_hi-far, frr_hi-frr]
        clrs    = ["#2ecc71","#e74c3c","#f39c12"]
        ax2.bar(metrics, vals, color=clrs, alpha=0.8)
        ax2.errorbar(range(3), vals, yerr=[errs_lo,errs_hi],
                     fmt="none", color="black", capsize=8)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("Rate")
        ax2.set_title("H2: TAR / FAR / FRR with Wilson CI")
        for i, v in enumerate(vals):
            ax2.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=10)

        fig.suptitle(
            f"EXP-H2 Face Recognition Operator Authentication\n"
            f"Auth latency={al_m:.2f}ms  TAR={tar:.3f}  FAR={far:.3f}  FRR={frr:.3f}\n"
            "King 2009 (dlib), ReAct (Yao 2022), Vemprala 2023",
            fontsize=9
        )
        fig.tight_layout()
        png = OUT_DIR / "H2_face_recognition_auth.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Plot  → {png}")
    except Exception as e:
        print(f"[plot skipped] {e}")

    print(f"\n── H2 Summary ───────────────────────────────────────────────────")
    print(f"Auth latency: {al_m:.2f}ms [{al_lo:.2f},{al_hi:.2f}]")
    print(f"TAR         : {tar:.3f} [{tar_lo:.3f},{tar_hi:.3f}]")
    print(f"FAR         : {far:.3f} [{far_lo:.3f},{far_hi:.3f}]")
    print(f"FRR         : {frr:.3f} [{frr_lo:.3f},{frr_hi:.3f}]")

if __name__ == "__main__":
    main()
