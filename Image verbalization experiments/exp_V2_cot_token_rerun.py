"""
EXP-V2-CoT Token Rerun
======================
Goal:
    Re-run CoT technique only with max_tokens=600 (vs 300 in V2).
    Tests whether CoT's low label accuracy (19.2%) in V2 was due to
    token cutoff — models reasoning correctly but running out of tokens
    before writing the final Risk: X line.

    Hypothesis: CoT with 600 tokens will match or beat structured (56.6%)
    because its reasoning-risk alignment (96%) is already the highest.

Pipeline: CLAHE → YOLO-World + CLIP → CoT prompt → LLM (600 tokens)
Models: claude, gpt4o, gpt4o_mini, gemini
Scenes: 8 canonical scenes (same as V2)
Runs: 5 per scene per model → 160 trials

Output: results/V2_cot_rerun_<timestamp>.csv
"""

import sys, time, pathlib, datetime
import numpy as np
import cv2

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = pathlib.Path(__file__).parent
EXP_DIR   = REPO_ROOT / "experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    SCENES, get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_clip, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer
)
from robust_local_detector import load_mediapipe_detector, detect_hazard

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
MAX_TOKENS = 600   # V2 used 300 — doubling to let CoT reach its conclusion

COGNITIVE_PREAMBLE = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "flying at ~1m altitude indoors.\n"
    "YOUR PRIMARY INPUT IS THE CAMERA IMAGE — look at it directly and reason from it.\n"
    "The YOLO-World and CLIP data above are supplementary sensor hints: YOLO measures "
    "distances and detects specific classes; CLIP labels the overall scene. "
    "Both can be wrong or incomplete. If you see something in the image that YOLO "
    "did not detect, it still exists — use it. If the image contradicts sensor data, "
    "trust the image and note the discrepancy.\n\n"
    "Risk definitions (judge from what YOU see, not just sensor output):\n"
    "  hazard : Person very close (<1m) OR obstacle directly in ~1m flight path "
    "within 2m OR lens fully covered. Stop immediately.\n"
    "  caution: Dim/dark scene (not fully black), OR background clutter at floor/"
    "table level not blocking the 1m corridor. Slow down or hover.\n"
    "  safe   : Flight path at ~1m is clear — open space, navigable door/window, "
    "objects on tables/shelves (not at drone altitude), persons visually >3m away.\n\n"
)

COT_PROMPT = (
    "Analyse this image step by step using your visual reasoning:\n"
    "Step 1: What do YOU see in the image? Describe objects, people, lighting, space.\n"
    "Step 2: How close are any objects or people — judge from visual size and position "
    "in the frame, not just YOLO est_dist.\n"
    "Step 3: Does YOLO/CLIP confirm or contradict what you see? Note any discrepancy.\n"
    "Step 4: Is the flight path at ~1m altitude actually clear based on what you see?\n"
    "Step 5: What is the overall risk level and why?\n\n"
    "Conclude with:\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <HOVER|PITCH_FORWARD|PITCH_BACK|ROLL_LEFT|ROLL_RIGHT|ASCEND|DESCEND|LAND>"
)


def build_prompt(tier2: dict) -> str:
    sensor_header = (
        f"Supplementary sensor data (advisory — verify against the image):\n"
        f"  YOLO-World: {tier2['yolo_meta']}\n"
        f"  CLIP: {tier2['clip_label']} "
        f"(conf={tier2['clip_conf']:.3f}, risk={tier2['clip_risk']})\n\n"
    )
    return COGNITIVE_PREAMBLE + sensor_header + COT_PROMPT


def parse_risk(reply: str) -> str | None:
    low = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if f"risk: {lvl}" in low:
            return lvl
    for lvl in ("hazard", "caution", "safe"):
        if lvl in low:
            return lvl
    return None


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(SCENES) * N_RUNS * len(MODELS)
    print("=" * 65)
    print("EXP-V2-CoT Token Rerun (max_tokens=600 vs 300 in V2)")
    print(f"Models={MODELS}")
    print(f"Scenes={len(SCENES)}  N={N_RUNS}  → {total} trials")
    print("Hypothesis: CoT LblAcc will rise from 19.2% → near structured (56.6%)")
    print("=" * 65)

    print("\nLoading MediaPipe…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2…")
    depth_pipe, _ = load_depth_anything()
    print("Loading CLIP…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")

        for run in range(1, N_RUNS + 1):
            jpeg    = get_saved_frame(scene["label"])
            img_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            t_local = time.perf_counter()
            local_r = detect_hazard(img_bgr, depth_map=None,
                                    mp_detector=mp_detector, mp_type=mp_type)
            local_ms = round((time.perf_counter() - t_local) * 1000.0, 2)

            tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                        clip_model, clip_preprocess,
                                        clip_tokenizer, jpeg,
                                        coco_model=coco_model,
                                        depth_pipe=depth_pipe)
            tier2["yolo_meta"] += (
                f"\n  Local detector (Tier 1.5 — MediaPipe EfficientDet-Lite0, advisory): "
                f"{local_r['metadata']}"
            )

            prompt = build_prompt(tier2)

            for model in MODELS:
                res      = call_vision_llm(jpeg, prompt, model=model,
                                           max_tokens=MAX_TOKENS, temperature=0.0)
                detected = parse_risk(res["reply"])
                scores   = score_verbalization(res["reply"], scene["truth"])
                s3       = int(detected == scene["truth"]) if detected else 0

                _action = (scores["detected_action"] or "").upper()
                _truth  = scene["truth"]
                _danger = int(_truth == "hazard" and _action == "PITCH_FORWARD")
                _safe   = int(not _danger)

                # Check if reply was truncated (no Risk: line found)
                truncated = int(detected is None)

                row = {
                    "scene_id":         scene["id"],
                    "scene_label":      scene["label"],
                    "truth":            scene["truth"],
                    "model":            model,
                    "run":              run,
                    "local_ms":         local_ms,
                    "local_risk":       local_r["risk"],
                    "detected_risk":    detected or "",
                    "detected_action":  scores["detected_action"] or "",
                    "s3_risk":          s3,
                    "truncated":        truncated,
                    "action_safe":      _safe,
                    "action_dangerous": _danger,
                    "word_count":       scores["word_count"],
                    "latency_ms":       res["latency_ms"],
                    "input_tokens":     res["input_tokens"],
                    "output_tokens":    res["output_tokens"],
                    "cost_usd":         res["cost_usd"],
                    "reply":            res["reply"],
                    "error":            res["error"][:80] if res["error"] else "",
                }
                all_rows.append(row)
                print(f"   run={run}  {model:12s}  risk={detected or '?':8s}  "
                      f"correct={s3}  trunc={truncated}  "
                      f"tokens={res['output_tokens']}  lat={res['latency_ms']:.0f}ms")

            time.sleep(0.3)

    # ── Save CSV
    fields = ["scene_id","scene_label","truth","model","run",
              "local_ms","local_risk","detected_risk","detected_action",
              "s3_risk","truncated","action_safe","action_dangerous","word_count",
              "latency_ms","input_tokens","output_tokens","cost_usd","reply","error"]
    runs_csv = RESULTS_DIR / f"V2_cot_rerun_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary
    print(f"\n── CoT Rerun Summary (max_tokens={MAX_TOKENS}) ──────────────────")
    print(f"  {'model':12s}  {'LblAcc':>7s}  {'Truncated':>9s}  {'AvgTokens':>9s}  {'Dangerous':>9s}")
    print(f"  {'(V2 CoT)':12s}  {'19.2%':>7s}  {'67%':>9s}  {'~180':>9s}  {'0':>9s}")
    print("-" * 60)

    for model in MODELS:
        tr = [r for r in all_rows if r["model"] == model and not r["error"]]
        if not tr: continue
        acc, _, _ = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
        trunc_rate = sum(r["truncated"] for r in tr) / len(tr)
        avg_tokens = sum(r["output_tokens"] for r in tr) / len(tr)
        dangerous  = sum(r["action_dangerous"] for r in tr)
        print(f"  {model:12s}  {acc:7.1%}  {trunc_rate:9.1%}  {avg_tokens:9.0f}  {dangerous:9d}")

    overall = [r for r in all_rows if not r["error"]]
    acc_all, _, _ = wilson_ci(sum(r["s3_risk"] for r in overall), len(overall))
    trunc_all = sum(r["truncated"] for r in overall) / len(overall)
    print(f"\n  Overall LblAcc: {acc_all:.1%}  (V2 CoT was 19.2%,  Structured was 56.6%)")
    print(f"  Truncation rate: {trunc_all:.1%}  (V2 CoT was ~67%)")
    print(f"\nResults → {runs_csv}")


if __name__ == "__main__":
    main()
