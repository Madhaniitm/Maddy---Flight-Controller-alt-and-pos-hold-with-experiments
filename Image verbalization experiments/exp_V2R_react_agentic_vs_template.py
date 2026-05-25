"""
EXP-V2R: ReAct Agentic vs ReAct Template  (YOLO-World + CLIP pipeline)
========================================================================
Goal:
    Isolate the contribution of the feedback loop in ReAct prompting for
    vision classification. Compares two conditions on identical frames:

    Condition A — react_template (single-pass, NO feedback):
        Loaded from V2 results — technique="react" rows.
        [YOLO-World+CLIP metadata + image] → model writes REASON/OBSERVE/ACT → classify
        (already collected in V2_runs_20260523_233838.csv, N=5 per scene per model)

    Condition B — react_agentic (2-call feedback loop):
        Run fresh in this experiment.
        Call 1: [image only] → model observes without sensor data
        Call 2: [model's observation + YOLO-World+CLIP confirmation] → final classification

    The only variable that differs is WHEN the YOLO-World+CLIP feedback arrives —
    before (template) or after (agentic) the model's initial visual reasoning.
    This isolates exactly the feedback loop contribution.

Motivation:
    New V2 (YOLO-World+CLIP) shows react_template underperforms on:
        dim_light=25%, person_near=35% (worst react_template failures)
        door_open=55% (included for continuity with old V2R run)
    Agentic loop hypothesis: delayed YOLO-World feedback lets models
    observe neutrally first, then update — breaking the template priming artifact.

Scenes:  dim_light (25%), person_near (35%), door_open (55%) — worst react failures
Models:  claude, gpt4o, gpt4o_mini, gemini
N runs:  5
Trials:  3 × 4 × 5 = 60 new trials (Condition B only, ~15 minutes)

Output:  results/V2R_runs_<timestamp>.csv   (Condition B rows)
         results/V2R_summary_<timestamp>.csv (A vs B comparison)
"""

import sys, os, time, pathlib, datetime, json, glob, csv, base64, urllib.request
import numpy as np
import cv2

REPO_ROOT = pathlib.Path(__file__).parent.parent
VIZ_DIR   = pathlib.Path(__file__).parent
EXP_DIR   = REPO_ROOT / "experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    SCENES, get_saved_frame, call_vision_llm, score_verbalization,
    bootstrap_ci, wilson_ci, write_csv, RESULTS_DIR,
    ANTHROPIC_API_KEY, AZURE_CLAUDE_ENDPOINT, AZURE_CLAUDE_VERSION,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MINI_KEY, OPENAI_MINI_URL,
    GEMINI_API_KEY, GEMINI_MODEL, CLAUDE_MODEL,
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_clip, load_coco_yolo, load_depth_anything,
    enhanced_yolo_infer
)
from robust_local_detector import load_mediapipe_detector, detect_hazard

N_RUNS = 5
MODELS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

# Three worst react_template scenes in new V2 (YOLO-World+CLIP run):
#   dim_light=25%, person_near=35%, door_open=55%
#   door_open kept for continuity with old V2R; dim_light + person_near are new failures
import verbalization_utils as _vu
TARGET_SCENES = [
    s for s in _vu.SCENES
    if s["label"] in {"dim_light", "person_near", "door_open"}
]

# ── Load Condition A from V2 ──────────────────────────────────────────────────

TARGET_LABELS = {s["label"] for s in TARGET_SCENES}

def load_condition_a() -> list[dict]:
    """Load react_template rows for the 3 target scenes from the most recent V2 runs CSV."""
    files = sorted(glob.glob(str(RESULTS_DIR / "V2_runs_*.csv")))
    if not files:
        raise FileNotFoundError("No V2_runs_*.csv found. Run exp_V2_prompt_techniques.py first.")
    src = files[-1]
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f):
            if r["technique"] == "react" and r["scene_label"] in TARGET_LABELS:
                rows.append({
                    "scene_id":      r["scene_id"],
                    "scene_label":   r["scene_label"],
                    "truth":         r["truth"],
                    "model":         r["model"],
                    "run":           r["run"],
                    "condition":     "react_template",
                    "s3_risk":       int(r["s3_risk"]),
                    "detected_risk": r["detected_risk"],
                    "quality_score": int(r["quality_score"]),
                    "word_count":    int(r["word_count"]),
                    "latency_ms":    float(r["latency_ms"]),
                    "input_tokens":  int(r["input_tokens"]),
                    "output_tokens": int(r["output_tokens"]),
                    "cost_usd":      float(r["cost_usd"]),
                    "call2_latency_ms": 0,
                    "call2_tokens":  0,
                    "error":         r["error"],
                })
    print(f"[V2R] Loaded {len(rows)} react_template rows from {pathlib.Path(src).name}")
    return rows

# ── Prompts ───────────────────────────────────────────────────────────────────

AGENTIC_OBSERVE_PROMPT = (
    "You are the cognitive reasoning layer of a three-tier drone safety system "
    "flying at ~1m altitude indoors.\n"
    "YOUR PRIMARY INPUT IS THE CAMERA IMAGE. No sensor data yet — use your own vision.\n\n"
    "OBSERVE: Look carefully at the image. Describe exactly what you see:\n"
    "- What objects, people, or surfaces are visible?\n"
    "- How close do they appear to the camera based on visual evidence?\n"
    "- What is the lighting like?\n"
    "- Is the flight path at ~1m altitude clear or blocked?\n\n"
    "Be specific. This is your raw visual observation before sensor cross-check."
)

def build_act_prompt(observation: str, tier2: dict) -> str:
    return (
        f"Your visual observation of the image:\n{observation}\n\n"
        f"Supplementary sensor data from Tier 2 (cross-check against your observation):\n"
        f"  YOLO-World: {tier2['yolo_meta']}\n"
        f"  CLIP: {tier2['clip_label']} (conf={tier2['clip_conf']:.3f}, risk={tier2['clip_risk']})\n\n"
        "ACT: You are the cognitive layer — your visual observation takes precedence.\n"
        "If sensor data confirms what you saw, state that. "
        "If sensor data missed something you saw, or contradicts your observation, "
        "trust what you saw in the image and note the discrepancy.\n\n"
        "Final classification:\n"
        "Risk: <safe|caution|hazard>\n"
        "Pilot suggested action: <HOVER|PITCH_FORWARD|PITCH_BACK|ROLL_LEFT|ROLL_RIGHT|ASCEND|DESCEND|LAND>"
    )

# ── Text-only call (no image — Condition B, Call 2) ───────────────────────────

def call_text_llm(prompt: str, model: str, max_tokens: int = 100) -> dict:
    t0 = time.time()
    try:
        if model == "claude":
            body = {"model": CLAUDE_MODEL, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            req = urllib.request.Request(
                AZURE_CLAUDE_ENDPOINT, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {ANTHROPIC_API_KEY}",
                         "anthropic-version": AZURE_CLAUDE_VERSION},
                method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = json.loads(r.read().decode())
            reply = raw["content"][0]["text"] if raw.get("content") else ""
            i = raw.get("usage", {}).get("input_tokens", 0)
            o = raw.get("usage", {}).get("output_tokens", 0)
            cost = round(i*3e-6 + o*15e-6, 6)

        elif model in ("gpt4o", "gpt4o_mini"):
            import openai
            if model == "gpt4o":
                client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
                m, ci, co = "gpt-4o", 5e-6, 15e-6
            else:
                client = openai.OpenAI(api_key=OPENAI_MINI_KEY, base_url=OPENAI_MINI_URL)
                m, ci, co = "gpt-4o-mini", 0.15e-6, 0.60e-6
            resp  = client.chat.completions.create(
                model=m, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            reply = resp.choices[0].message.content or ""
            i, o  = resp.usage.prompt_tokens, resp.usage.completion_tokens
            cost  = round(i*ci + o*co, 6)

        elif model == "gemini":
            import json as _json
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}")
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.0,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            data = _json.dumps(body).encode("utf-8")
            req2 = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req2, timeout=60) as r:
                raw = _json.loads(r.read().decode("utf-8"))
            candidate = raw.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            reply = "".join(p.get("text", "") for p in parts if "text" in p)
            usage = raw.get("usageMetadata", {})
            i = usage.get("promptTokenCount", 0)
            o = usage.get("candidatesTokenCount", 0)
            cost = round(i*0.075e-6 + o*0.30e-6, 6)
        else:
            return {"reply":"","latency_ms":0,"input_tokens":0,
                    "output_tokens":0,"cost_usd":0,"error":f"unknown model {model}"}

        return {"reply": reply, "latency_ms": round((time.time()-t0)*1000,1),
                "input_tokens": i, "output_tokens": o, "cost_usd": cost, "error": ""}
    except Exception as e:
        return {"reply":"","latency_ms":round((time.time()-t0)*1000,1),
                "input_tokens":0,"output_tokens":0,"cost_usd":0,"error":str(e)[:120]}


def parse_risk(reply: str):
    low = reply.lower()
    for lvl in ("hazard", "caution", "safe"):
        if f"risk: {lvl}" in low: return lvl
    for lvl in ("hazard", "caution", "safe"):
        if lvl in low: return lvl
    return None


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("="*65)
    print("EXP-V2R: ReAct Agentic vs ReAct Template")
    print(f"Condition A: loaded from V2 (react technique, N=5)")
    print(f"Condition B: running now  (react_agentic, N={N_RUNS})")
    print(f"Models={MODELS}  Scenes={len(SCENES)}")
    print(f"New trials={len(TARGET_SCENES)*len(MODELS)*N_RUNS}  (Scenes={[s['label'] for s in TARGET_SCENES]})")
    print("="*65)

    # Load Condition A
    cond_a_rows = load_condition_a()

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

    # Run Condition B
    cond_b_rows = []

    for scene in TARGET_SCENES:
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
            print(f"   run={run}  local={local_ms:.0f}ms({local_r['risk']})  yolo={tier2['yolo_meta'][:50]}")

            for model in MODELS:
                # Call 1: image only — model observes
                res1 = call_vision_llm(jpeg, AGENTIC_OBSERVE_PROMPT,
                                       model=model, max_tokens=300)
                # Call 2: observation + YOLO-World + CLIP → final classification
                res2 = call_text_llm(build_act_prompt(res1["reply"], tier2),
                                     model=model, max_tokens=150)

                det   = parse_risk(res2["reply"])
                sc    = score_verbalization(res2["reply"], scene["truth"])
                s3    = int(det == scene["truth"]) if det else 0

                _action  = (sc["detected_action"] or "").upper()
                _truth   = scene["truth"]
                _danger  = int(_truth == "hazard" and _action == "PITCH_FORWARD")
                _safe    = int(not _danger)

                row = {
                    "scene_id":          scene["id"],
                    "scene_label":       scene["label"],
                    "truth":             scene["truth"],
                    "model":             model,
                    "run":               run,
                    "condition":         "react_agentic",
                    "local_ms":          local_ms,
                    "local_risk":        local_r["risk"],
                    "s3_risk":           s3,
                    "detected_risk":     det or "",
                    "detected_action":   sc["detected_action"] or "",
                    "action_safe":       _safe,
                    "action_dangerous":  _danger,
                    "quality_score":     sc["s1_scene"]+sc["s2_proximity"]+s3+sc["s4_length"]+sc["s5_pilot_action"],
                    "word_count":        sc["word_count"],
                    "latency_ms":        round(res1["latency_ms"]+res2["latency_ms"],1),
                    "input_tokens":      res1["input_tokens"]+res2["input_tokens"],
                    "output_tokens":     res1["output_tokens"]+res2["output_tokens"],
                    "cost_usd":          round(res1["cost_usd"]+res2["cost_usd"],6),
                    "call2_latency_ms":  res2["latency_ms"],
                    "call2_tokens":      res2["input_tokens"],
                    "observe_reply":     res1["reply"],
                    "act_reply":         res2["reply"],
                    "error":             (res1["error"] or res2["error"])[:80],
                }
                cond_b_rows.append(row)
                print(f"     {model:12s}  risk={det or '?':8s}  "
                      f"correct={s3}  "
                      f"lat={row['latency_ms']:.0f}ms "
                      f"(obs={res1['latency_ms']:.0f}+act={res2['latency_ms']:.0f}ms)")

            time.sleep(1)

    # ── Save Condition B runs
    fields = ["scene_id","scene_label","truth","model","condition","run",
              "local_ms","local_risk",
              "quality_score","s3_risk","detected_risk","detected_action",
              "action_safe","action_dangerous","word_count",
              "latency_ms","call2_latency_ms","input_tokens","call2_tokens",
              "output_tokens","cost_usd","observe_reply","act_reply","error"]
    runs_csv = RESULTS_DIR / f"V2R_runs_{ts}.csv"
    write_csv(runs_csv, cond_b_rows, fields)

    # ── Combined summary: A vs B
    all_rows = cond_a_rows + cond_b_rows

    print(f"\n── V2R Summary — A vs B per model ─────────────────────────────────")
    print(f"  {'condition':16s}  {'model':12s}  {'acc':>6s}  [lo,  hi ]  {'lat_ms':>8s}")
    print("-"*70)
    summary_rows = []
    for cond in ("react_template", "react_agentic"):
        for model in MODELS:
            tr = [r for r in all_rows if r["condition"]==cond
                  and r["model"]==model and not r["error"]]
            if not tr: continue
            acc, alo, ahi = wilson_ci(sum(r["s3_risk"] for r in tr), len(tr))
            lm,  _,   _   = bootstrap_ci([r["latency_ms"] for r in tr])
            cm,  _,   _   = bootstrap_ci([r["cost_usd"]   for r in tr])
            qm,  _,   _   = bootstrap_ci([r["quality_score"] for r in tr])
            print(f"  {cond:16s}  {model:12s}  {acc:.3f}  [{alo:.3f},{ahi:.3f}]  {lm:.0f}ms")
            summary_rows.append({
                "condition": cond, "model": model, "n_trials": len(tr),
                "accuracy": acc, "acc_lo": alo, "acc_hi": ahi,
                "quality": qm, "latency_ms": lm, "cost_usd": cm,
            })

    print(f"\n── V2R Per-Scene Delta: agentic − template (all models avg) ────────")
    print(f"  {'scene':20s}  {'truth':7s}  {'template':>8s}  {'agentic':>8s}  {'delta':>7s}")
    for scene in TARGET_SCENES:
        sl = scene["label"]
        t  = [r for r in all_rows if r["scene_label"]==sl and r["condition"]=="react_template"]
        a  = [r for r in all_rows if r["scene_label"]==sl and r["condition"]=="react_agentic"]
        ta = sum(r["s3_risk"] for r in t)/len(t) if t else 0
        aa = sum(r["s3_risk"] for r in a)/len(a) if a else 0
        delta = aa - ta
        sign  = "+" if delta >= 0 else ""
        print(f"  {sl:20s}  {scene['truth']:7s}  {ta:8.2f}  {aa:8.2f}  {sign}{delta:.2f}")

    summary_csv = RESULTS_DIR / f"V2R_summary_{ts}.csv"
    write_csv(summary_csv, summary_rows,
              ["condition","model","n_trials","accuracy","acc_lo","acc_hi",
               "quality","latency_ms","cost_usd"])

    print(f"\nCondition B data → {runs_csv}")
    print(f"Summary (A+B)    → {summary_csv}")

if __name__ == "__main__":
    main()
