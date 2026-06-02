"""
EXP-ND1: Camera Pipeline as Tool Call — Multi-Orchestrator Comparison
======================================================================
Goal:
    Compare four LLM orchestrators (Claude, GPT-4o, GPT-4o-mini, Gemini) on
    their ability to use analyze_scene as a tool call and produce a high-quality
    structured room safety report.

    Analogous to V-series (which model gives the best reasoning?) but now the
    model acts as an ORCHESTRATOR that calls a tool rather than doing raw vision
    directly. The orchestrator sees the camera image directly and uses
    analyze_scene only to get raw sensor metadata (MediaPipe + YOLO + Depth).

Architecture per run:
    Orchestrator LLM  →  calls analyze_scene (native tool-call API)
    analyze_scene     →  MediaPipe + YOLO-World + DepthAnything (no inner LLM)
    Orchestrator LLM  →  sees image + sensor JSON, replies in V-series format

Orchestrators : claude, gpt4o, gpt4o_mini, gemini
Scenes        : 8 canonical V-series scenes
N runs        : 5 per scene per orchestrator
Total         : 4 × 8 × 5 = 160 runs

Scoring (same 5-point rubric as V/G-series):
    s1  +1  scene content described (what the camera shows)
    s2  +1  proximity / spatial info mentioned
    s3  +1  correct risk classification (EMERGENCY/ALERT/CLEAR vs truth)
    s4  +1  reply length 10-150 words
    s5  +1  pilot action recommended (ND action vocabulary)
    quality_score = s1+s2+s3+s4+s5  (max 5)

Output:
    results/ND1_runs_<ts>.csv         — per run: full reply, scores, cost
    results/ND1_summary_<ts>.csv      — per orchestrator × scene: quality + latency + cost
    results/ND1_orchestrator_<ts>.csv — per orchestrator overall

Run:
    export GLOG_minloglevel=3
    /opt/homebrew/bin/python3.11 experiments/exp_ND1_camera_as_tool.py
"""

import sys, os, pathlib, datetime, json, re
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
EXP_DIR   = pathlib.Path(__file__).parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    SCENES, get_saved_frame, bootstrap_ci, write_csv,
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
)
from robust_local_detector import load_mediapipe_detector
from nd_series_agent import NDAgent, CAMERA_CSV_FIELDS

# ── Config ────────────────────────────────────────────────────────────────────
ND_RESULTS        = EXP_DIR / "results"
ND_RESULTS.mkdir(exist_ok=True)

ORCHESTRATORS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
N_RUNS        = 5

# V-series truth → ND risk label mapping (for s3 scoring)
TRUTH_TO_ND = {"safe": "CLEAR", "caution": "ALERT", "hazard": "EMERGENCY"}

# ── Prompt (same for all orchestrators) ───────────────────────────────────────
ND1_PROMPT = (
    "You are a drone safety inspector. "
    "Use analyze_scene to get sensor metadata from the current camera frame. "
    "Then look at the camera image in your context, cross-reference with the sensor data, "
    "and reply in the REQUIRED FORMAT from your system prompt: "
    "Description / Sensor note / Proximity / Risk / Pilot suggested action / Confidence. "
    "Do NOT issue any drone control commands — just analyse and report."
)

# ── ND-adapted 5-point scoring rubric (same structure as V/G-series) ─────────

_ND_SCENE_KW = {
    "see", "observe", "wall", "obstacle", "object", "floor", "ceiling",
    "surface", "dark", "bright", "blurry", "clear", "colour", "color",
    "person", "table", "door", "lamp", "light", "room", "window",
    "laptop", "chair", "bag", "box", "covered", "hand", "partial",
    "camera", "lens", "blocked", "cluttered",
}
_ND_PROX_KW = {
    "cm", "mm", "metre", "meter", "distance", "close", "near", "far",
    "proxim", "within", "away", "behind", "front", "side", "left", "right",
    "above", "below", "approximately", "roughly", "about", "adjacent",
    "depth", "metres", "meters",
}
_ND_ACTION_KW = {
    "investigate_closer", "investigate closer", "investigate",
    "hover_monitor", "hover monitor", "hover",
    "alert_operator", "alert operator",
    "continue_patrol", "continue patrol", "patrol",
    "return_to_base", "return to base", "return",
}


def score_verbalization_nd(reply: str, true_risk_vs: str) -> dict:
    """
    ND-adapted 5-point rubric — same structure as V/G-series score_verbalization().

    s1  +1  scene content described
    s2  +1  proximity / spatial info mentioned
    s3  +1  correct risk classification (EMERGENCY/ALERT/CLEAR)
    s4  +1  reply length 10-150 words
    s5  +1  ND pilot action recommended
    quality_score = s1+s2+s3+s4+s5 (max 5)

    Args:
        reply        : full orchestrator reply text
        true_risk_vs : V-series ground truth label ('safe' | 'caution' | 'hazard')
    """
    r       = reply.lower()
    words   = reply.split()
    n_words = len(words)

    s1 = int(any(kw in r for kw in _ND_SCENE_KW))
    s2 = int(any(kw in r for kw in _ND_PROX_KW))
    s4 = int(10 <= n_words <= 150)
    s5 = int(any(a in r for a in _ND_ACTION_KW))

    # s3 — risk classification in ND vocabulary
    true_nd = TRUTH_TO_ND.get(true_risk_vs, true_risk_vs.upper())
    detected_risk = None
    for lvl in ("emergency", "alert", "clear"):
        if lvl in r:
            detected_risk = lvl.upper()
            break
    # fallback keyword scan
    if detected_risk is None:
        if any(w in r for w in ("danger", "collapsed", "fire", "critical", "hazard")):
            detected_risk = "EMERGENCY"
        elif any(w in r for w in ("warning", "concern", "unusual", "caution", "investigate")):
            detected_risk = "ALERT"
        elif any(w in r for w in ("okay", "fine", "no issue", "normal", "safe")):
            detected_risk = "CLEAR"

    s3 = int(detected_risk == true_nd) if detected_risk else 0

    # detected action (ND vocabulary)
    detected_action = None
    for a in ("investigate_closer", "investigate closer",
              "hover_monitor", "hover monitor",
              "alert_operator", "alert operator",
              "continue_patrol", "continue patrol",
              "return_to_base", "return to base"):
        if a in r:
            detected_action = a.replace(" ", "_").upper()
            break

    return {
        "s1_scene":        s1,
        "s2_proximity":    s2,
        "s3_risk":         s3,
        "s4_length":       s4,
        "s5_pilot_action": s5,
        "quality_score":   s1 + s2 + s3 + s4 + s5,
        "detected_risk":   detected_risk or "unknown",
        "detected_action": detected_action or "unknown",
        "word_count":      n_words,
    }


# ── CSV fields ────────────────────────────────────────────────────────────────
RUN_FIELDS = [
    "orchestrator", "run", "scene_label", "truth",
    "tool_called",          # 1 if orchestrator called analyze_scene
    "mp_risk",              # MediaPipe risk 
    from sensor tool
    "pipeline_ms",          # sensor pipeline latency
    # V/G-series 5-point rubric scores
    "quality_score",        # s1+s2+s3+s4+s5 (max 5)
    "s1_scene",             # scene content described
    "s2_proximity",         # proximity / spatial info
    "s3_risk",              # correct risk classification
    "s4_length",            # reply length 10-150 words
    "s5_pilot_action",      # ND pilot action recommended
    "detected_risk",        # risk label extracted from reply
    "detected_action",      # action extracted from reply
    "word_count",
    # Structured reply fields (V-series format)
    "risk",                 # Risk: field (EMERGENCY | ALERT | CLEAR)
    "pilot_action",         # Pilot suggested action: field
    "confidence",           # Confidence: field (0.0–1.0, -1 if absent)
    "description",          # Description: field
    "sensor_note",          # Sensor note: field
    "proximity",            # Proximity: field
    # Full reply logging
    "all_turns_text",       # every orchestrator turn joined by |TURN|
    "final_text",           # final reply (full, untruncated)
    "loop_cost_usd",
    "total_cost_usd",
    "wall_s",
    "error",
]

SUMMARY_FIELDS = [
    "orchestrator", "scene_label", "truth", "n_runs",
    "tool_call_rate",
    "mean_quality", "ci_quality_lo", "ci_quality_hi",
    "risk_acc",                        # s3 accuracy (fraction correct)
    "mean_pipeline_ms", "ci_pipeline_lo", "ci_pipeline_hi",
    "mean_total_cost_usd",
]

ORCH_SUMMARY_FIELDS = [
    "orchestrator",
    "mean_quality", "q_lo", "q_hi",
    "risk_acc",
    "tool_call_rate",
    "mean_pipeline_ms", "mean_total_cost_usd", "total_cost_usd", "n_runs",
]


def _extract_field(text: str, field: str) -> str:
    """Extract a named field from V-series style structured reply."""
    for line in text.splitlines():
        if line.strip().lower().startswith(field.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_risk(text: str) -> str:
    """Extract Risk: field; fall back to keyword scan."""
    val = _extract_field(text, "Risk").upper()
    for lvl in ("EMERGENCY", "ALERT", "CLEAR"):
        if lvl in val:
            return lvl
    up = text.upper()
    for lvl in ("EMERGENCY", "ALERT", "CLEAR"):
        if lvl in up:
            return lvl
    return "unknown"


def _extract_confidence(text: str) -> float:
    """Extract Confidence: field (0.0–1.0); return -1.0 if absent."""
    val = _extract_field(text, "Confidence")
    if not val:
        return -1.0
    m = re.search(r"\d+\.?\d*", val)
    if m:
        f = float(m.group())
        return round(f if f <= 1.0 else f / 100.0, 3)
    return -1.0


def _extract_action(text: str) -> str:
    """Extract Pilot suggested action: field; fall back to keyword scan."""
    val = _extract_field(text, "Pilot suggested action").upper()
    for a in ["INVESTIGATE_CLOSER", "HOVER_MONITOR", "ALERT_OPERATOR",
              "CONTINUE_PATROL", "RETURN_TO_BASE"]:
        if a in val:
            return a
    up = text.upper()
    for a in ["INVESTIGATE_CLOSER", "HOVER_MONITOR", "ALERT_OPERATOR",
              "CONTINUE_PATROL", "RETURN_TO_BASE"]:
        if a in up:
            return a
    return "unknown"


def main():
    import time
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_runs  = ND_RESULTS / f"ND1_runs_{ts}.csv"
    out_summ  = ND_RESULTS / f"ND1_summary_{ts}.csv"
    out_orch  = ND_RESULTS / f"ND1_orchestrator_{ts}.csv"

    total_runs = len(ORCHESTRATORS) * len(SCENES) * N_RUNS
    print("=" * 65)
    print("EXP-ND1: Camera as Tool — Multi-Orchestrator Comparison")
    print(f"Orchestrators : {ORCHESTRATORS}")
    print(f"Scenes        : {len(SCENES)}   N={N_RUNS}")
    print(f"Total runs    : {total_runs}  (4 × 8 × 5)")
    print(f"Scoring       : 5-point rubric (s1 scene, s2 prox, s3 risk, s4 len, s5 action)")
    print("=" * 65)

    # ── Load camera infrastructure once ──────────────────────────────────────
    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (Tier 2)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print()

    all_run_rows    = []
    all_camera_rows = []
    all_summ_rows   = []
    global_run      = 0

    for orchestrator in ORCHESTRATORS:
        print(f"\n{'═'*65}")
        print(f"  ORCHESTRATOR: {orchestrator}")
        print(f"{'═'*65}")

        agent = NDAgent(
            session_id    = f"ND1_{orchestrator}_{ts}",
            mp_detector   = mp_detector,
            mp_type       = mp_type,
            yolo_model    = yolo_model,
            yolo_type     = yolo_type,
            coco_model    = coco_model,
            depth_pipe    = depth_pipe,
            frame_source  = "saved",
        )

        orch_qualities = []
        orch_s3        = []
        orch_tools     = []
        orch_lats      = []
        orch_costs     = []

        for scene in SCENES:
            label = scene["label"]
            truth = scene["truth"]
            print(f"\n  ── Scene: {label}  (truth={truth} → ND={TRUTH_TO_ND.get(truth, truth)}) ──")

            sc_qualities = []
            sc_s3        = []
            sc_tools     = []
            sc_lats      = []
            sc_costs     = []

            for run in range(1, N_RUNS + 1):
                global_run += 1
                print(f"    Run {run}/{N_RUNS}  [{global_run}/{total_runs}]",
                      end="  ", flush=True)

                try:
                    jpeg = get_saved_frame(label)
                except FileNotFoundError as e:
                    print(f"SKIP — {e}")
                    all_run_rows.append({
                        "orchestrator": orchestrator, "run": run,
                        "scene_label": label, "truth": truth,
                        "tool_called": 0, "mp_risk": "error", "pipeline_ms": 0,
                        "quality_score": 0, "s1_scene": 0, "s2_proximity": 0,
                        "s3_risk": 0, "s4_length": 0, "s5_pilot_action": 0,
                        "detected_risk": "error", "detected_action": "error",
                        "word_count": 0,
                        "risk": "error", "pilot_action": "error",
                        "confidence": -1.0, "description": "",
                        "sensor_note": "", "proximity": "",
                        "all_turns_text": "", "final_text": "",
                        "loop_cost_usd": 0, "total_cost_usd": 0,
                        "wall_s": 0, "error": str(e),
                    })
                    continue

                agent.reset_run(run)
                agent.set_frame(jpeg)

                t_wall    = time.time()
                error_msg = ""
                try:
                    final_text, api_stats, _ = agent.run_orchestrator_loop(
                        orchestrator = orchestrator,
                        user_prompt  = ND1_PROMPT,
                        max_turns    = 5,
                        max_tokens   = 1024,
                    )
                except Exception as e:
                    final_text, api_stats = "", []
                    error_msg = str(e)
                    print(f"ERROR — {e}")

                wall_s = round(time.time() - t_wall, 2)

                # ── Sensor results ────────────────────────────────────────────
                cam_rows    = list(agent._camera_rows)
                tool_called = int(len(cam_rows) > 0)
                mp_risk     = cam_rows[0]["mp_risk"]           if cam_rows else "unknown"
                pipeline_ms = cam_rows[0]["total_pipeline_ms"] if cam_rows else 0

                # ── All turn texts ────────────────────────────────────────────
                all_turns      = [s.get("text", "") for s in api_stats if s.get("text")]
                all_turns_text = " |TURN| ".join(all_turns)

                # ── V/G-series 5-point rubric on orchestrator reply ───────────
                scores       = score_verbalization_nd(final_text, truth)
                quality      = scores["quality_score"]
                s3           = scores["s3_risk"]

                # ── Structured reply field extraction ─────────────────────────
                risk         = _extract_risk(final_text)
                pilot_action = _extract_action(final_text)
                confidence   = _extract_confidence(final_text)
                description  = _extract_field(final_text, "Description")
                sensor_note  = _extract_field(final_text, "Sensor note")
                proximity    = _extract_field(final_text, "Proximity")

                loop_cost  = sum(s.get("cost_usd", 0) for s in api_stats)
                total_cost = round(loop_cost, 6)

                print(f"tool={tool_called}  quality={quality}/5  "
                      f"s3_risk={'✓' if s3 else '✗'}  "
                      f"risk={risk}  action={pilot_action}  "
                      f"conf={confidence:.2f}  pipe={pipeline_ms:.0f}ms  "
                      f"${total_cost:.5f}")

                all_run_rows.append({
                    "orchestrator":   orchestrator,
                    "run":            run,
                    "scene_label":    label,
                    "truth":          truth,
                    "tool_called":    tool_called,
                    "mp_risk":        mp_risk,
                    "pipeline_ms":    pipeline_ms,
                    "quality_score":  quality,
                    "s1_scene":       scores["s1_scene"],
                    "s2_proximity":   scores["s2_proximity"],
                    "s3_risk":        s3,
                    "s4_length":      scores["s4_length"],
                    "s5_pilot_action":scores["s5_pilot_action"],
                    "detected_risk":  scores["detected_risk"],
                    "detected_action":scores["detected_action"],
                    "word_count":     scores["word_count"],
                    "risk":           risk,
                    "pilot_action":   pilot_action,
                    "confidence":     confidence,
                    "description":    description,
                    "sensor_note":    sensor_note,
                    "proximity":      proximity,
                    "all_turns_text": all_turns_text,
                    "final_text":     final_text,
                    "loop_cost_usd":  round(loop_cost, 6),
                    "total_cost_usd": total_cost,
                    "wall_s":         wall_s,
                    "error":          error_msg,
                })

                for cr in agent._camera_rows:
                    cr["orchestrator"] = orchestrator
                    cr["scene_label"]  = label
                    cr["truth"]        = truth
                all_camera_rows.extend(agent._camera_rows)

                if error_msg == "":
                    sc_qualities.append(quality)
                    sc_s3.append(s3)
                    sc_tools.append(tool_called)
                    sc_lats.append(pipeline_ms)
                    sc_costs.append(total_cost)

                    orch_qualities.append(quality)
                    orch_s3.append(s3)
                    orch_tools.append(tool_called)
                    orch_lats.append(pipeline_ms)
                    orch_costs.append(total_cost)

            # ── Scene-level summary ───────────────────────────────────────────
            n = len(sc_qualities)
            if n:
                qm, qlo, qhi  = bootstrap_ci(sc_qualities)
                lm, llo, lhi  = bootstrap_ci(sc_lats)
                cm, _, _      = bootstrap_ci(sc_costs)
                risk_acc      = round(sum(sc_s3) / n, 3)
                tool_rate     = round(sum(sc_tools) / n, 3)
                print(f"    → quality={qm:.2f}/5 [{qlo:.2f}–{qhi:.2f}]  "
                      f"risk_acc={risk_acc:.0%}  tool_rate={tool_rate:.0%}")
                all_summ_rows.append({
                    "orchestrator":      orchestrator,
                    "scene_label":       label,
                    "truth":             truth,
                    "n_runs":            n,
                    "tool_call_rate":    tool_rate,
                    "mean_quality":      qm,
                    "ci_quality_lo":     qlo,
                    "ci_quality_hi":     qhi,
                    "risk_acc":          risk_acc,
                    "mean_pipeline_ms":  lm,
                    "ci_pipeline_lo":    llo,
                    "ci_pipeline_hi":    lhi,
                    "mean_total_cost_usd": cm,
                })

        # ── Orchestrator-level overall ────────────────────────────────────────
        n_o = len(orch_qualities)
        if n_o:
            o_pipe  = round(float(np.mean([x for x in orch_lats if x > 0])), 1)
            o_cost  = round(sum(orch_costs), 4)
            o_q, o_qlo, o_qhi = bootstrap_ci(orch_qualities)
            o_racc  = round(sum(orch_s3) / n_o, 3)
            print(f"\n  {orchestrator} OVERALL: quality={o_q:.2f}/5 "
                  f"[{o_qlo:.2f}–{o_qhi:.2f}]  risk_acc={o_racc:.0%}  "
                  f"tool={sum(orch_tools)/n_o:.0%}  "
                  f"pipe={o_pipe:.0f}ms  cost=${o_cost:.4f}")

        agent.stop_emergency_monitor()

    # ── Write CSVs ────────────────────────────────────────────────────────────
    write_csv(out_runs, all_run_rows, RUN_FIELDS)
    print(f"\nSaved runs    → {out_runs}")

    if all_camera_rows:
        out_cam = ND_RESULTS / f"ND1_camera_{ts}.csv"
        cam_fields_ext = CAMERA_CSV_FIELDS + ["orchestrator", "scene_label", "truth"]
        write_csv(out_cam, all_camera_rows, cam_fields_ext)
        print(f"Saved camera  → {out_cam}")

    if all_summ_rows:
        write_csv(out_summ, all_summ_rows, SUMMARY_FIELDS)
        print(f"Saved summary → {out_summ}")

    # ── Per-orchestrator aggregate ────────────────────────────────────────────
    orch_rows = []
    for orch in ORCHESTRATORS:
        rows = [r for r in all_run_rows
                if r["orchestrator"] == orch and r["error"] == ""]
        n    = len(rows)
        if not n:
            continue
        qualities = [r["quality_score"]   for r in rows]
        s3s       = [r["s3_risk"]         for r in rows]
        tools     = [r["tool_called"]     for r in rows]
        lats      = [r["pipeline_ms"]     for r in rows if r["pipeline_ms"] > 0]
        costs     = [r["total_cost_usd"]  for r in rows]
        qm, qlo, qhi = bootstrap_ci(qualities)
        orch_rows.append({
            "orchestrator":        orch,
            "mean_quality":        round(qm, 3),
            "q_lo":                round(qlo, 3),
            "q_hi":                round(qhi, 3),
            "risk_acc":            round(sum(s3s) / n, 3),
            "tool_call_rate":      round(sum(tools) / n, 3),
            "mean_pipeline_ms":    round(float(np.mean(lats)), 1) if lats else 0,
            "mean_total_cost_usd": round(float(np.mean(costs)), 6),
            "total_cost_usd":      round(sum(costs), 4),
            "n_runs":              n,
        })

    if orch_rows:
        write_csv(out_orch, orch_rows, ORCH_SUMMARY_FIELDS)
        print(f"Saved orch    → {out_orch}")

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ND1 ORCHESTRATOR RESULTS")
    print(f"{'Orch':<14} {'Quality':>9} {'[lo–hi]':^13} {'RiskAcc':>8} "
          f"{'Tool%':>6} {'Pipe(ms)':>9} {'Cost/run':>10}")
    print("─" * 65)
    for r in orch_rows:
        print(f"  {r['orchestrator']:<12} {r['mean_quality']:>7.2f}/5  "
              f"[{r['q_lo']:.2f}–{r['q_hi']:.2f}]  "
              f"{r['risk_acc']:>7.0%}  "
              f"{r['tool_call_rate']:>5.0%}  "
              f"{r['mean_pipeline_ms']:>7.0f}ms  "
              f"${r['mean_total_cost_usd']:.5f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
