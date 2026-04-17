"""
EXP-I1: Multi-Model Vision Benchmark for Drone Scene Understanding
==================================================================
Goal:
    Benchmark four vision-language models on identical drone camera frames.
    Each model receives the SAME JPEG (synthetic or webcam) + a classification
    prompt and must label the scene from a fixed vocabulary.

    Models tested:
        claude  — Claude 3.5 Sonnet (Anthropic Vision)
        gpt4o   — GPT-4o            (OpenAI Vision)
        gemini  — Gemini 1.5 Flash  (Google Vision)
        ollama  — LLaVA-13B         (open-source, Ollama)

    10 scene classes × N=5 frames per class = 50 frames per model = 200 trials.

Metrics:
    - accuracy       : fraction of frames correctly classified (Wilson CI)
    - latency_ms     : per-call round-trip (Bootstrap CI)
    - cost_usd       : per-call cost (Bootstrap CI)
    - confusion_mat  : 10×10 confusion matrix per model
    - top_error      : most confused scene pair per model

    All four models are compared on the same frames — direct apples-to-apples.
    This is the primary multi-model vision credibility experiment.

Paper References:
    - Achiam et al. 2023 (GPT-4V): OpenAI vision benchmark
    - Reid et al. 2024 (Gemini 1.5): Google multimodal evaluation
    - Liu et al. 2023 (LLaVA): open-source visual instruction tuning
    - ReAct (Yao et al. 2022): vision → reasoning → action loop
"""

import os, sys, csv, math, pathlib, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from d_series_agent import (
    DAgent, MultiLLMRunner, SceneSimulator,
    SCENE_TYPES, SCENE_LABELS,
)

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_FRAMES_PER_SCENE = 5
MODELS = ["claude", "gpt4o", "gemini", "ollama"]

# All 10 scene classes and their correct label
SCENES = [
    ("open_space",          "open_space"),
    ("wall_close",          "wall_close"),
    ("wall_far",            "wall_far"),
    ("floor_pattern",       "floor_or_downward"),
    ("ceiling",             "ceiling_or_upward"),
    ("obstacle_left",       "obstacle_left"),
    ("obstacle_right",      "obstacle_right"),
    ("dark_room",           "low_visibility"),
    ("textured_floor",      "floor_or_downward"),
    ("bright_overexposure", "low_visibility"),
]

VALID_LABELS = sorted(set(lbl for _, lbl in SCENES))

CLASSIFICATION_PROMPT = f"""You are analysing a drone camera frame.
Classify the scene into EXACTLY ONE of these labels:
  {', '.join(VALID_LABELS)}

Rules:
  - Reply with ONLY the label. No explanation, no punctuation.
  - If unsure, pick the closest match.
  - wall_close means obstacle < 30 cm.
  - low_visibility means dark or overexposed.
"""

PAPER_REFS = {
    "GPT4V":   "Achiam et al. 2023 — GPT-4 Technical Report",
    "Gemini15":"Reid et al. 2024 — Gemini 1.5: Unlocking multimodal understanding",
    "LLaVA":   "Liu et al. 2023 — Visual Instruction Tuning (LLaVA)",
    "ReAct":   "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in Language Models",
}

# ── Statistics helpers ─────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / denom
    m = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(p,4), round(max(0,c-m),4), round(min(1,c+m),4)

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr = np.array(data, dtype=float)
    boots = [stat(np.random.choice(arr, size=len(arr), replace=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)),4), round(float(lo),4), round(float(hi),4)

# ── Label extractor ────────────────────────────────────────────────────────────
def extract_label(reply: str) -> str:
    """Pull the first valid label token out of model reply."""
    cleaned = reply.strip().lower().replace("-", "_").replace(" ", "_")
    for lbl in VALID_LABELS:
        if lbl in cleaned:
            return lbl
    # fallback: first word
    first = cleaned.split()[0] if cleaned.split() else ""
    return first if first in VALID_LABELS else "unknown"

# ── Generate JPEG frames ───────────────────────────────────────────────────────
def make_frames(scene_type: str, n: int) -> list:
    """Return n JPEG byte strings for the given scene type."""
    sim = SceneSimulator()
    sim.set_scene(scene_type)
    frames = []
    for seed in range(n):
        # add slight distance variation per frame for realism
        sim._obs_dist_m = None
        if scene_type == "wall_close":
            sim.set_obstacle_distance(0.15 + seed * 0.02)
        elif scene_type == "wall_far":
            sim.set_obstacle_distance(0.80 + seed * 0.1)
        frames.append(sim.capture_jpeg())
    return frames

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXP-I1: Multi-Model Vision Benchmark")
    print(f"Models: {MODELS}")
    print(f"Scenes: {len(SCENES)} × N={N_FRAMES_PER_SCENE} frames = "
          f"{len(SCENES)*N_FRAMES_PER_SCENE} trials per model")
    print("=" * 60)

    # Pre-generate ALL frames once — same frames for every model
    print("\nGenerating frames…")
    frame_bank = {}   # (scene_type, frame_idx) -> jpeg_bytes
    for scene_type, _ in SCENES:
        jpegs = make_frames(scene_type, N_FRAMES_PER_SCENE)
        for i, j in enumerate(jpegs):
            frame_bank[(scene_type, i)] = j
    print(f"  {len(frame_bank)} frames ready.")

    all_rows = []

    for model_key in MODELS:
        print(f"\n{'='*40}")
        print(f"Model: {model_key}")
        print(f"{'='*40}")

        # Build a minimal DAgent + MultiLLMRunner (no physics needed for vision-only)
        agent  = DAgent(session_id=f"I1_{model_key}")
        runner = MultiLLMRunner(agent, model_key=model_key)

        for scene_type, gt_label in SCENES:
            for frame_idx in range(N_FRAMES_PER_SCENE):
                jpeg = frame_bank[(scene_type, frame_idx)]
                result = runner.run_vision_call(jpeg, CLASSIFICATION_PROMPT)

                predicted = extract_label(result["reply"])
                correct   = int(predicted == gt_label)

                row = {
                    "model":       model_key,
                    "scene_type":  scene_type,
                    "gt_label":    gt_label,
                    "predicted":   predicted,
                    "correct":     correct,
                    "latency_ms":  result["latency_ms"],
                    "tokens_in":   result["input_tokens"],
                    "tokens_out":  result["output_tokens"],
                    "cost_usd":    result["cost_usd"],
                    "error":       result["error"] or "",
                    "frame_idx":   frame_idx,
                    "raw_reply":   result["reply"][:80],
                }
                all_rows.append(row)
                status = "✓" if correct else "✗"
                print(f"  [{model_key:8s}] {scene_type:22s} → {predicted:22s} {status}"
                      f"  {result['latency_ms']:.0f}ms")

    # ── Save per-trial CSV ─────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "I1_runs.csv"
    fields   = ["model","scene_type","gt_label","predicted","correct",
                "latency_ms","tokens_in","tokens_out","cost_usd","error",
                "frame_idx","raw_reply"]
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-trial data → {runs_csv}")

    # ── Per-model stats ────────────────────────────────────────────────────────
    summary_csv = OUT_DIR / "I1_summary.csv"
    model_stats = {}
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["model","metric","value","ci_lo","ci_hi","note"])

        for mk in MODELS:
            mr = [r for r in all_rows if r["model"] == mk]
            kc = sum(r["correct"]     for r in mr)
            n  = len(mr)
            ac, ac_lo, ac_hi = wilson_ci(kc, n)
            lm, lm_lo, lm_hi = bootstrap_ci([r["latency_ms"] for r in mr])
            co, co_lo, co_hi = bootstrap_ci([r["cost_usd"]   for r in mr])
            ti, _,     _     = bootstrap_ci([r["tokens_in"]  for r in mr])

            model_stats[mk] = {"acc": ac, "lat": lm, "cost": co, "tok": ti}

            cw.writerow([mk,"accuracy",   ac, ac_lo, ac_hi, "Wilson 95%"])
            cw.writerow([mk,"latency_ms", lm, lm_lo, lm_hi, "Bootstrap 95%"])
            cw.writerow([mk,"cost_usd",   co, co_lo, co_hi, "Bootstrap 95%"])
            cw.writerow([mk,"tokens_in",  ti, "",    "",    "Bootstrap mean"])

        for k, ref in PAPER_REFS.items():
            cw.writerow(["", f"ref_{k}", ref,"","",""])
    print(f"Summary        → {summary_csv}")

    # ── Per-model confusion matrix ─────────────────────────────────────────────
    conf_csv = OUT_DIR / "I1_confusion.csv"
    unique_labels = VALID_LABELS
    with open(conf_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["model","gt_label","predicted","count"])
        for mk in MODELS:
            mr = [r for r in all_rows if r["model"] == mk]
            from collections import Counter
            counts = Counter((r["gt_label"], r["predicted"]) for r in mr)
            for (gt, pred), cnt in sorted(counts.items()):
                cw.writerow([mk, gt, pred, cnt])
    print(f"Confusion data → {conf_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import Counter

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        colors = {"claude":"#3498db","gpt4o":"#2ecc71",
                  "gemini":"#e67e22","ollama":"#9b59b6"}

        # ── Accuracy bar chart ────────────────────────────────────────────────
        ax = axes[0, 0]
        accs    = [model_stats[mk]["acc"] for mk in MODELS]
        acc_bars= ax.bar(MODELS, accs,
                         color=[colors[mk] for mk in MODELS], alpha=0.85)
        for mk_i, (mk, acc) in enumerate(zip(MODELS, accs)):
            mr   = [r for r in all_rows if r["model"] == mk]
            _, lo, hi = wilson_ci(sum(r["correct"] for r in mr), len(mr))
            ax.errorbar(mk_i, acc, yerr=[[acc-lo],[hi-acc]],
                        fmt="none", color="black", capsize=7)
            ax.text(mk_i, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=9,
                    fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Scene classification accuracy")
        ax.set_title("I1: Accuracy by Vision Model")
        ax.axhline(1.0, color="green", linestyle="--", alpha=0.4, label="Perfect")
        ax.legend(fontsize=8)

        # ── Latency box plots ──────────────────────────────────────────────────
        ax2 = axes[0, 1]
        lat_data = [[r["latency_ms"] for r in all_rows if r["model"]==mk]
                    for mk in MODELS]
        bp = ax2.boxplot(lat_data, labels=MODELS, patch_artist=True)
        for patch, mk in zip(bp["boxes"], MODELS):
            patch.set_facecolor(colors[mk])
            patch.set_alpha(0.7)
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("I1: Response Latency Distribution")
        ax2.set_yscale("log")

        # ── Cost per call ──────────────────────────────────────────────────────
        ax3 = axes[0, 2]
        cost_means = [model_stats[mk]["cost"] * 1000 for mk in MODELS]  # → millidollars
        ax3.bar(MODELS, cost_means,
                color=[colors[mk] for mk in MODELS], alpha=0.85)
        ax3.set_ylabel("Cost per call (milli-USD)")
        ax3.set_title("I1: Cost per Vision Call")
        for i, v in enumerate(cost_means):
            ax3.text(i, v + max(cost_means)*0.02,
                     f"${v:.4f}m", ha="center", fontsize=8)

        # ── Per-scene accuracy heatmap ─────────────────────────────────────────
        ax4 = axes[1, 0]
        scene_types = [s for s, _ in SCENES]
        heat = np.zeros((len(MODELS), len(scene_types)))
        for mi, mk in enumerate(MODELS):
            for si, (st, _) in enumerate(SCENES):
                mr = [r for r in all_rows if r["model"]==mk and r["scene_type"]==st]
                heat[mi, si] = sum(r["correct"] for r in mr) / len(mr) if mr else 0
        im = ax4.imshow(heat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax4.set_yticks(range(len(MODELS)));   ax4.set_yticklabels(MODELS, fontsize=8)
        ax4.set_xticks(range(len(scene_types)))
        ax4.set_xticklabels([s[:10] for s in scene_types],
                             rotation=45, ha="right", fontsize=7)
        ax4.set_title("I1: Per-Scene Accuracy Heatmap")
        fig.colorbar(im, ax=ax4, fraction=0.046)
        for mi in range(len(MODELS)):
            for si in range(len(scene_types)):
                ax4.text(si, mi, f"{heat[mi,si]:.1f}",
                         ha="center", va="center", fontsize=7)

        # ── Radar chart (accuracy + speed + cost) ─────────────────────────────
        ax5 = axes[1, 1]
        ax5.remove()
        ax5 = fig.add_subplot(2, 3, 5, polar=True)

        radar_cats = ["Accuracy", "Speed\n(inv lat)", "Cost\n(efficiency)",
                      "Open\nsource", "Tok\nefficiency"]
        N_cats = len(radar_cats)
        angles = [n / float(N_cats) * 2 * np.pi for n in range(N_cats)]
        angles += angles[:1]

        max_lat  = max(model_stats[mk]["lat"]  for mk in MODELS) or 1
        max_cost = max(model_stats[mk]["cost"] for mk in MODELS) or 1
        max_tok  = max(model_stats[mk]["tok"]  for mk in MODELS) or 1
        open_src = {"claude":0,"gpt4o":0,"gemini":0,"ollama":1}  # 1=open-source

        for mk in MODELS:
            s = model_stats[mk]
            vals = [
                s["acc"],
                1.0 - s["lat"] / max_lat,      # higher = faster
                1.0 - s["cost"] / max_cost,    # higher = cheaper
                float(open_src[mk]),
                1.0 - s["tok"] / max_tok,      # higher = fewer tokens
            ]
            vals += vals[:1]
            ax5.plot(angles, vals, color=colors[mk], linewidth=2, label=mk)
            ax5.fill(angles, vals, color=colors[mk], alpha=0.1)

        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(radar_cats, fontsize=8)
        ax5.set_ylim(0, 1)
        ax5.set_title("I1: Multi-Criteria Radar", pad=20)
        ax5.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

        # ── Confusion matrix for best model ───────────────────────────────────
        best_model = max(MODELS, key=lambda mk: model_stats[mk]["acc"])
        ax6 = axes[1, 2]
        lbl_short = [lbl[:12] for lbl in VALID_LABELS]
        conf_mat = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=int)
        mr = [r for r in all_rows if r["model"] == best_model]
        for r in mr:
            gi = VALID_LABELS.index(r["gt_label"])    if r["gt_label"]  in VALID_LABELS else -1
            pi = VALID_LABELS.index(r["predicted"])   if r["predicted"] in VALID_LABELS else -1
            if gi >= 0 and pi >= 0:
                conf_mat[gi, pi] += 1
        im2 = ax6.imshow(conf_mat, cmap="Blues")
        ax6.set_xticks(range(len(VALID_LABELS)))
        ax6.set_yticks(range(len(VALID_LABELS)))
        ax6.set_xticklabels(lbl_short, rotation=45, ha="right", fontsize=6)
        ax6.set_yticklabels(lbl_short, fontsize=6)
        ax6.set_xlabel("Predicted")
        ax6.set_ylabel("Ground truth")
        ax6.set_title(f"I1: Confusion Matrix — {best_model} (best)")
        for i in range(len(VALID_LABELS)):
            for j in range(len(VALID_LABELS)):
                if conf_mat[i,j] > 0:
                    ax6.text(j, i, conf_mat[i,j],
                             ha="center", va="center", fontsize=7)
        fig.colorbar(im2, ax=ax6, fraction=0.046)

        fig.suptitle(
            "EXP-I1: Multi-Model Vision Benchmark — Drone Scene Classification\n"
            "Claude Vision · GPT-4o · Gemini 1.5 Flash · LLaVA-13B\n"
            "GPT-4V (Achiam 2023), Gemini 1.5 (Reid 2024), LLaVA (Liu 2023), ReAct (Yao 2022)",
            fontsize=10
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        png = OUT_DIR / "I1_multimodal_vision_benchmark.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Plot  → {png}")

    except Exception as e:
        print(f"[plot skipped] {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n── I1 Summary ───────────────────────────────────────────────────")
    print(f"{'Model':10s} {'Accuracy':10s} {'Latency(ms)':13s} {'Cost/call':12s}")
    for mk in MODELS:
        s = model_stats[mk]
        mr = [r for r in all_rows if r["model"] == mk]
        _, lo, hi = wilson_ci(sum(r["correct"] for r in mr), len(mr))
        print(f"  {mk:10s} {s['acc']:.3f} [{lo:.3f},{hi:.3f}]  "
              f"{s['lat']:8.0f}ms    ${s['cost']:.7f}")

    best = max(MODELS, key=lambda mk: model_stats[mk]["acc"])
    print(f"\nBest accuracy: {best} ({model_stats[best]['acc']:.3f})")

if __name__ == "__main__":
    main()
