"""
EXP-V2: Prompt Technique Comparison (Enhanced YOLO-World Pipeline)
===================================================================
Goal:
    Compare 5 prompting strategies across all 4 models on the same
    8 canonical scenes × N=5 runs = 800 trials total.

    Each technique receives the same YOLO-World + CLIP metadata header,
    then applies its own reasoning structure on top.

    Techniques:
        zero_shot   : bare instruction, no examples, no structure
        few_shot_3  : 3 worked examples prepended before the question
        cot         : explicit step-by-step reasoning chain
        structured  : request JSON output with defined fields
        react       : Reason → Observe → Act loop

    Pipeline: CLAHE → YOLO-World → CLIP → [technique prompt] → LLM
    One saved run03 frame per scene. All models × techniques see identical JPEG + metadata.

Metrics:
    - classification_accuracy   Wilson CI
    - quality_score (0-5 rubric) Bootstrap CI
    - latency_ms / cost_usd      Bootstrap CI

Output: results/V2_runs_<timestamp>.csv, results/V2_summary_<timestamp>.csv
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
    SCENES, get_saved_frame, call_vision_llm, score_verbalization, extract_json_risk,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_clip, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer
)
from robust_local_detector import load_mediapipe_detector, detect_hazard

N_RUNS  = 5
MODELS  = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

# ── Shared cognitive framing — prepended to every technique ──────────────────
# LLM is the cognitive reasoning layer. Image is primary. YOLO/CLIP are supplementary.
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

# ── Prompt technique bodies (injected after cognitive preamble + sensor header) ─
PROMPTS = {

"zero_shot": (
    "Look at this image. Describe what you see and classify the risk.\n"
    "End with:\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <HOVER|PITCH_FORWARD|PITCH_BACK|ROLL_LEFT|ROLL_RIGHT|ASCEND|DESCEND|LAND>"
),

"few_shot_3": (
    "Here are 3 examples of correct cognitive analysis:\n\n"
    "Example 1:\n"
    "Image: empty corridor ahead, YOLO: none detected\n"
    "Analysis: I see a clear open corridor with no obstacles at drone altitude. "
    "YOLO found nothing — consistent with what I see. "
    "Risk: safe. Pilot suggested action: PITCH_FORWARD\n\n"
    "Example 2:\n"
    "Image: wall fills the frame, YOLO: wall (conf=0.82, est_dist~0.2m)\n"
    "Analysis: Solid wall extremely close, occupies the full frame. "
    "YOLO distance estimate is consistent with what I see — immediate collision risk. "
    "Risk: hazard. Pilot suggested action: PITCH_BACK\n\n"
    "Example 3:\n"
    "Image: dim room, YOLO: none detected\n"
    "Analysis: Room is dark with poor visibility. YOLO found nothing but the dim "
    "lighting itself is a caution condition regardless of detections. "
    "Risk: caution. Pilot suggested action: HOVER\n\n"
    "Now analyse this image. Use your visual judgment first, then cross-check sensor data.\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <HOVER|PITCH_FORWARD|PITCH_BACK|ROLL_LEFT|ROLL_RIGHT|ASCEND|DESCEND|LAND>"
),

"cot": (
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
),

"structured": (
    "Analyse this image using your visual reasoning, then output ONLY valid JSON:\n"
    "{\n"
    '  "what_i_see": "your direct visual description of the image",\n'
    '  "objects_detected_visually": ["list from YOUR observation"],\n'
    '  "sensor_agreement": "consistent|yolo_missed_X|image_contradicts_sensor",\n'
    '  "proximity_visual_estimate": "your distance judgment from image",\n'
    '  "lighting_quality": "good|dim|dark",\n'
    '  "flight_path_clear": true or false,\n'
    '  "risk_level": "safe|caution|hazard",\n'
    '  "recommended_action": "HOVER|PITCH_FORWARD|PITCH_BACK|ROLL_LEFT|ROLL_RIGHT|ASCEND|DESCEND|LAND"\n'
    "}\n"
    "JSON only, no other text."
),

"react": (
    "Use the Reason-Observe-Act framework as a cognitive agent:\n\n"
    "REASON: I need to determine if this scene is safe for drone navigation at ~1m altitude.\n\n"
    "OBSERVE: Look at the image directly. Describe what you see — space, lighting, people, "
    "obstacles, and how the sensor data (YOLO/CLIP above) compares to your visual analysis. "
    "If YOLO missed something visible, note it.\n\n"
    "ACT: Based on YOUR visual observation (with sensor data as a reference), "
    "give your final risk classification and pilot action.\n"
    "Risk: <safe|caution|hazard>\n"
    "Pilot suggested action: <HOVER|PITCH_FORWARD|PITCH_BACK|ROLL_LEFT|ROLL_RIGHT|ASCEND|DESCEND|LAND>"
),
}

TECHNIQUES = list(PROMPTS.keys())


def build_enhanced_prompt(tier2: dict, technique_body: str) -> str:
    """Build prompt: cognitive preamble → sensor metadata → technique body.
    Cognitive framing comes FIRST so LLM reads its role before seeing sensor data."""
    sensor_header = (
        f"Supplementary sensor data (advisory — verify against the image):\n"
        f"  YOLO-World: {tier2['yolo_meta']}\n"
        f"  CLIP: {tier2['clip_label']} "
        f"(conf={tier2['clip_conf']:.3f}, risk={tier2['clip_risk']})\n\n"
    )
    return COGNITIVE_PREAMBLE + sensor_header + technique_body


def parse_risk(reply: str, technique: str) -> str | None:
    """Extract risk level from reply, technique-aware."""
    if technique == "structured":
        r = extract_json_risk(reply)
        if r:
            return r
    low = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if f"risk: {lvl}" in low or f'"risk_level": "{lvl}"' in low:
            return lvl
    for lvl in ("hazard", "caution", "safe"):
        if lvl in low:
            return lvl
    return None


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(SCENES) * N_RUNS * len(TECHNIQUES) * len(MODELS)
    print("=" * 65)
    print("EXP-V2: Prompt Technique Comparison (YOLO-World + CLIP)")
    print(f"Models={MODELS}")
    print(f"Techniques={TECHNIQUES}")
    print(f"Scenes={len(SCENES)}  N={N_RUNS}  → {total} trials")
    print("=" * 65)

    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5 — local emergency detector)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (structural hazards)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO (person + 80 classes)…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print("Loading CLIP scene screener (V-series — retained for thesis evidence)…")
    clip_model, clip_preprocess, clip_tokenizer = load_clip()

    all_rows = []

    for scene in SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")

        for run in range(1, N_RUNS + 1):
            jpeg     = get_saved_frame(scene["label"])
            img_bgr  = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            t_local  = time.perf_counter()
            local_r  = detect_hazard(img_bgr, depth_map=None,
                                     mp_detector=mp_detector, mp_type=mp_type)
            local_ms = round((time.perf_counter() - t_local) * 1000.0, 2)

            tier2 = enhanced_yolo_infer(yolo_model, yolo_type,
                                        clip_model, clip_preprocess,
                                        clip_tokenizer, jpeg,
                                        coco_model=coco_model,
                                        depth_pipe=depth_pipe)
            tier2["yolo_meta"] += (
                f"\n  Local detector (Tier 1.5 — MediaPipe EfficientDet-Lite0, advisory — image overrides): "
                f"{local_r['metadata']}"
            )
            print(f"   run={run}  local={local_ms:.0f}ms({local_r['risk']})  yolo={tier2['yolo_meta'][:50]}  clip={tier2['clip_label']}")

            for tech, technique_body in PROMPTS.items():
                prompt = build_enhanced_prompt(tier2, technique_body)

                for model in MODELS:
                    res      = call_vision_llm(jpeg, prompt, model=model,
                                               max_tokens=300, temperature=0.0)
                    detected = parse_risk(res["reply"], tech)
                    scores   = score_verbalization(res["reply"], scene["truth"])
                    s3       = int(detected == scene["truth"]) if detected else 0

                    _action  = (scores["detected_action"] or "").upper()
                    _truth   = scene["truth"]
                    _danger  = int(_truth == "hazard" and _action == "PITCH_FORWARD")
                    _safe    = int(not _danger)

                    row = {
                        "scene_id":        scene["id"],
                        "scene_label":     scene["label"],
                        "truth":           scene["truth"],
                        "model":           model,
                        "technique":       tech,
                        "run":             run,
                        "local_ms":        local_ms,
                        "local_risk":      local_r["risk"],
                        "quality_score":   (scores["s1_scene"] + scores["s2_proximity"]
                                            + s3 + scores["s4_length"] + scores["s5_pilot_action"]),
                        "s1_scene":        scores["s1_scene"],
                        "s2_proximity":    scores["s2_proximity"],
                        "s3_risk":         s3,
                        "s4_length":       scores["s4_length"],
                        "s5_pilot_action": scores["s5_pilot_action"],
                        "detected_risk":   detected or "",
                        "detected_action": scores["detected_action"] or "",
                        "action_safe":     _safe,
                        "action_dangerous": _danger,
                        "word_count":      scores["word_count"],
                        "yolo_meta":       tier2["yolo_meta"],
                        "clip_label":      tier2["clip_label"],
                        "clip_risk":       tier2["clip_risk"],
                        "latency_ms":      res["latency_ms"],
                        "input_tokens":    res["input_tokens"],
                        "output_tokens":   res["output_tokens"],
                        "cost_usd":        res["cost_usd"],
                        "reply":           res["reply"],
                        "error":           res["error"][:80] if res["error"] else "",
                    }
                    all_rows.append(row)
                    print(f"     {tech:12s}  {model:12s}  q={row['quality_score']}/5  "
                          f"risk={detected or '?':8s}  lat={res['latency_ms']:.0f}ms")

            time.sleep(0.3)

    # ── Save runs CSV
    fields = ["scene_id","scene_label","truth","model","technique","run",
              "local_ms","local_risk",
              "quality_score","s1_scene","s2_proximity","s3_risk","s4_length",
              "s5_pilot_action","detected_risk","detected_action",
              "action_safe","action_dangerous","word_count",
              "yolo_meta","clip_label","clip_risk",
              "latency_ms","input_tokens","output_tokens","cost_usd","reply","error"]
    runs_csv = RESULTS_DIR / f"V2_runs_{ts}.csv"
    write_csv(runs_csv, all_rows, fields)

    # ── Summary per technique (all models)
    print(f"\n── V2 Summary by Technique (all models) ───────────────────────")
    print(f"  {'technique':14s}  {'acc':>6s}  [lo,  hi ]  {'quality':>7s}  {'lat_ms':>7s}  {'$/call':>8s}")
    summary_rows = []
    for tech in TECHNIQUES:
        tr = [r for r in all_rows if r["technique"] == tech and not r["error"]]
        if not tr:
            continue
        acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
        qm,  qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
        lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
        cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])
        print(f"  {tech:14s}  {acc:.3f}  [{alo:.3f},{ahi:.3f}]  {qm:.2f}/5  {lm:.0f}ms  ${cm:.5f}")
        summary_rows.append({
            "model": "all", "technique": tech, "n_trials": len(tr),
            "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
            "quality": qm, "q_lo": qlo, "q_hi": qhi,
            "latency_ms": lm, "cost_usd": cm,
        })

    # ── Summary per model × technique
    print(f"\n── V2 Summary by Model × Technique ────────────────────────────")
    print(f"  {'model':12s}  {'technique':14s}  {'acc':>6s}  {'quality':>7s}  {'lat_ms':>7s}")
    for model in MODELS:
        for tech in TECHNIQUES:
            tr = [r for r in all_rows
                  if r["model"] == model and r["technique"] == tech and not r["error"]]
            if not tr:
                continue
            acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
            qm,  qlo, qhi = bootstrap_ci([r["quality_score"] for r in tr])
            lm,  _,   _   = bootstrap_ci([r["latency_ms"]    for r in tr])
            cm,  _,   _   = bootstrap_ci([r["cost_usd"]      for r in tr])
            print(f"  {model:12s}  {tech:14s}  {acc:.3f}  {qm:.2f}/5  {lm:.0f}ms")
            summary_rows.append({
                "model": model, "technique": tech, "n_trials": len(tr),
                "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                "quality": qm, "q_lo": qlo, "q_hi": qhi,
                "latency_ms": lm, "cost_usd": cm,
            })

    summary_csv = RESULTS_DIR / f"V2_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["model","technique","n_trials","accuracy","acc_lo","acc_hi",
               "quality","q_lo","q_hi","latency_ms","cost_usd"])

    print(f"\nData    → {runs_csv}")
    print(f"Summary → {summary_csv}")
    print(f"[V2] Done — {len(all_rows)} trials recorded.")


if __name__ == "__main__":
    main()
