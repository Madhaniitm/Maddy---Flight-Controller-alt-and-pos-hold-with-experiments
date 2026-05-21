"""
EXP-V2R: ReAct Agentic vs ReAct Template
==========================================
Goal:
    Isolate the contribution of the feedback loop in ReAct prompting for
    vision classification. Compares two conditions on identical frames:

    Condition A — react_template (single-pass, NO feedback):
        Loaded from V2 results — technique="react" rows.
        [YOLO metadata + image] → model writes REASON/OBSERVE/ACT → classify
        (already collected, N=5 per scene per model, 160 rows total)

    Condition B — react_agentic (2-call feedback loop):
        Run fresh in this experiment.
        Call 1: [image only] → model observes without sensor data
        Call 2: [model's observation + YOLO confirmation] → final classification

    The only variable that differs is WHEN the YOLO feedback arrives —
    before (template) or after (agentic) the model's initial visual reasoning.
    This isolates exactly the feedback loop contribution.

Motivation:
    V2 shows react_template underperforms on several scenes:
        door_open=5%, person_far=0%, wall_close=40%
    C-series shows agentic ReAct with real tool feedback works well for
    drone control. This experiment tests whether the feedback loop —
    not just the ReAct framing — drives that performance.

Scene:   door_open — react_template collapses to 5% (worst failure across all 8 scenes)
           Selected because it shows the largest gap between template and agentic,
           isolating the feedback loop contribution cleanly.
Models:  claude, gpt4o, gpt4o_mini, gemini
N runs:  5
Trials:  1 × 4 × 5 = 20 new trials (Condition B only, ~5 minutes)

Output:  results/V2R_runs_<timestamp>.csv   (Condition B rows)
         results/V2R_summary_<timestamp>.csv (A vs B comparison)
"""

import sys, os, time, pathlib, datetime, json, glob, csv, base64, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verbalization_utils import (
    SCENES, get_frame, call_vision_llm, score_verbalization,
    run_yolo_on_frame, bootstrap_ci, wilson_ci, write_csv, preflight, RESULTS_DIR,
    ANTHROPIC_API_KEY, AZURE_CLAUDE_ENDPOINT, AZURE_CLAUDE_VERSION,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MINI_KEY, OPENAI_MINI_URL,
    GEMINI_API_KEY, GEMINI_MODEL, CLAUDE_MODEL,
)

N_RUNS = 5
MODELS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]

# door_open: worst react_template failure (5%) — maximum contrast for feedback loop isolation
TARGET_SCENES = [
    s for s in __import__('verbalization_utils').SCENES
    if s["label"] == "door_open"
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
    "You are a drone vision agent.\n"
    "REASON: I need to determine whether this scene is safe for drone navigation.\n"
    "OBSERVE: Look carefully at the camera image. Describe exactly what you see — "
    "objects, people, proximity to camera, lighting conditions, any obstructions. "
    "Be specific about distances and what fills the frame."
)

def build_act_prompt(observation: str, yolo_meta: str) -> str:
    return (
        f"Your previous visual observation:\n{observation}\n\n"
        f"YOLO SENSOR CONFIRMATION:\n{yolo_meta}\n\n"
        "ACT: Based on your observation and the sensor confirmation, "
        "give your final classification.\n"
        "Risk: <safe|caution|hazard>\n"
        "Pilot suggested action: <PROCEED|SLOW_DOWN|STOP|LAND|HOLD>"
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
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GEMINI_API_KEY)
            resp   = client.models.generate_content(
                model=GEMINI_MODEL, contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens, temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            reply = resp.text or ""
            i = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
            o = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0
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
    print(f"New trials={len(SCENES)*len(MODELS)*N_RUNS}")
    print("="*65)

    # Load Condition A
    cond_a_rows = load_condition_a()

    if not preflight():
        ans = input("ESP32 not reachable. Use synthetic frames? [y/N]: ")
        if ans.strip().lower() != "y":
            return

    # Run Condition B
    cond_b_rows = []

    for scene in TARGET_SCENES:
        print(f"\n── Scene {scene['id']:02d}: {scene['label']}  (truth={scene['truth']}) ──")
        print(f"   Setup: {scene['setup']}")
        input("   [READY] Press Enter when scene is set up…")

        for run in range(1, N_RUNS + 1):
            jpeg      = get_frame(scene["label"])
            yolo_meta = run_yolo_on_frame(jpeg)
            print(f"   run={run}  frame={len(jpeg)}B  yolo={yolo_meta[:60]}")

            for model in MODELS:
                # Call 1: image only — model observes
                res1 = call_vision_llm(jpeg, AGENTIC_OBSERVE_PROMPT,
                                       model=model, max_tokens=200)
                # Call 2: observation + YOLO → final classification
                res2 = call_text_llm(build_act_prompt(res1["reply"], yolo_meta),
                                     model=model, max_tokens=100)

                det   = parse_risk(res2["reply"])
                sc    = score_verbalization(res2["reply"], scene["truth"])
                s3    = int(det == scene["truth"]) if det else 0

                row = {
                    "scene_id":          scene["id"],
                    "scene_label":       scene["label"],
                    "truth":             scene["truth"],
                    "model":             model,
                    "run":               run,
                    "condition":         "react_agentic",
                    "s3_risk":           s3,
                    "detected_risk":     det or "",
                    "quality_score":     sc["s1_scene"]+sc["s2_proximity"]+s3+sc["s4_length"]+sc["s5_pilot_action"],
                    "word_count":        sc["word_count"],
                    "latency_ms":        round(res1["latency_ms"]+res2["latency_ms"],1),
                    "input_tokens":      res1["input_tokens"]+res2["input_tokens"],
                    "output_tokens":     res1["output_tokens"]+res2["output_tokens"],
                    "cost_usd":          round(res1["cost_usd"]+res2["cost_usd"],6),
                    "call2_latency_ms":  res2["latency_ms"],
                    "call2_tokens":      res2["input_tokens"],
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
              "quality_score","s3_risk","detected_risk","word_count",
              "latency_ms","call2_latency_ms","input_tokens","call2_tokens",
              "output_tokens","cost_usd","error"]
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
