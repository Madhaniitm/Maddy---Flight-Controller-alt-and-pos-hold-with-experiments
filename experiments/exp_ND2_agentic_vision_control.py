"""
EXP-ND2: Agentic Vision-Guided Room Emergency Surveillance
===========================================================
Goal:
    Compare four LLM orchestrators (Claude, GPT-4o, GPT-4o-mini, Gemini) on
    their ability to autonomously conduct a room emergency surveillance mission
    by combining camera analysis (analyze_scene) with drone control tool calls.

    The drone is NOT protecting itself — it is monitoring the room for
    emergencies (person in distress, fire, unusual conditions).

    The LLM orchestrates:
      1. plan_workflow           — declare the full patrol plan
      2. navigate_to_waypoint()  — move to next patrol position
      3. analyze_scene(patrol)   — survey the room from this position
      4. get_sensor_status()     — read altitude + position
      5. Decide based on room_status:
           EMERGENCY → hover + investigate_closer + alert_operator
           ALERT     → fly toward subject + analyze_scene(investigate)
           CLEAR     → continue patrol to next waypoint
      6. report_progress(step, N, findings)
      7. Repeat until patrol complete

    Two scenarios per run set:
      clear_room  (door_open)    — normal room, patrol should complete cleanly
      alert_room  (person_near)  — person visible, drone should investigate

    N=5 runs per scenario per orchestrator.
    Total: 4 orchestrators × 2 scenarios × 5 runs = 40 runs.

    Background room-event monitor runs MediaPipe every 1s.
    Person detected → stores trigger frame → advisory injected → LLM decides
    to accept or ignore based on history.

    Every drone command logs telemetry_before + telemetry_after.
    Every LLM reasoning turn is saved in full (no truncation).

Scoring:
    Same 5-point V-series rubric applied to the final LLM reply per run:
    s1 scene + s2 proximity + s3 risk + s4 length + s5 action = quality_score /5

Output:
    results/ND2_runs_<ts>.csv     — per run: orchestrator, scenario, full reply, scores
    results/ND2_camera_<ts>.csv   — per analyze_scene call (sensor pipeline outputs)
    results/ND2_workflow_<ts>.csv — per drone command with telemetry
    results/ND2_summary_<ts>.csv  — per orchestrator × scenario

Run:
    export GLOG_minloglevel=3
    /opt/homebrew/bin/python3.11 experiments/exp_ND2_agentic_vision_control.py
"""

import sys, os, pathlib, datetime, time, json, re
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
EXP_DIR   = pathlib.Path(__file__).parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    get_saved_frame, wilson_ci, bootstrap_ci, write_csv,
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
)
from robust_local_detector import load_mediapipe_detector
from nd_series_agent import (
    NDAgent, CAMERA_CSV_FIELDS, WORKFLOW_CSV_FIELDS, LLM_STATS_CSV_FIELDS,
)

# ── Output paths ──────────────────────────────────────────────────────────────
ND_RESULTS = EXP_DIR / "results"
ND_RESULTS.mkdir(exist_ok=True)

ORCHESTRATORS = ["claude", "gpt4o", "gpt4o_mini", "gemini"]
N_RUNS        = 5

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name":        "clear_room",
        "scene_label": "door_open",
        "truth":       "CLEAR",
        "description": "Normal room with open doorway. Drone should complete patrol and report CLEAR.",
    },
    {
        "name":        "alert_room",
        "scene_label": "person_near",
        "truth":       "ALERT",
        "description": "Person close to drone camera. Drone should investigate and report findings.",
    },
]

# ── Mission prompt ────────────────────────────────────────────────────────────
MISSION_PROMPT = """\
Execute an indoor room emergency surveillance mission.

MISSION: You are a surveillance drone deployed to monitor this room for emergencies.
Take off to 1 metre, conduct a patrol sweep of the room using analyze_scene at each
position, investigate anything suspicious, and produce a final structured room report.

MANDATORY PROTOCOL:
  1. plan_workflow("room_surveillance", steps)  — write the full patrol plan first
  2. arm()
     → check_drone_stable()                     — verify stable after arm
     → find_hover_throttle()
     → enable_altitude_hold()
     → set_altitude_target(1.0)
     → check_altitude_reached(target_m=1.0, tolerance_m=0.10)
       If ✗: wait(2.0) → re-check. After 3 failures → land() and abort.

  3. At each patrol position:
       a. navigate_to_waypoint(...)
       b. check_drone_stable()                  — verify stable after movement
          If ✗: wait(1.0) → re-check (max 3×)
       c. analyze_scene(context="patrol")       — survey the room from here
       d. get_sensor_status()                   — log altitude and position
       e. Reply in REQUIRED FORMAT, then decide:
            EMERGENCY → hover()
                        check_drone_stable()
                        analyze_scene(context="investigate")
                        report_progress with EMERGENCY findings + ALERT_OPERATOR
            ALERT     → navigate toward subject
                        check_drone_stable()
                        analyze_scene(context="investigate")
                        report_progress with findings
            CLEAR     → continue to next patrol waypoint

  4. After full patrol: land() and produce a final room safety report with:
       - Overall room status (EMERGENCY / ALERT / CLEAR)
       - Findings from each waypoint
       - Recommended actions for human operator

VERIFICATION RULE: Never proceed to the next step without confirming the current
command succeeded. If check_altitude_reached or check_drone_stable returns ✗,
wait and retry. Abort only after 3 consecutive failures.

Do NOT ignore ALERT or EMERGENCY detections.
Do NOT land until the full patrol is complete or an emergency requires immediate response.
"""

# ── V-series ND scoring rubric ────────────────────────────────────────────────
TRUTH_TO_ND = {"safe": "CLEAR", "caution": "ALERT", "hazard": "EMERGENCY",
               "CLEAR": "CLEAR", "ALERT": "ALERT", "EMERGENCY": "EMERGENCY"}

_ND_SCENE_KW = {
    "see","observe","wall","obstacle","object","floor","ceiling","surface",
    "dark","bright","blurry","clear","colour","color","person","table","door",
    "lamp","light","room","window","chair","covered","camera","lens","blocked","cluttered",
}
_ND_PROX_KW = {
    "cm","mm","metre","meter","distance","close","near","far","proxim",
    "within","away","behind","front","side","left","right","above","below",
    "approximately","roughly","about","adjacent","depth","metres","meters",
}
_ND_ACTION_KW = {
    "investigate_closer","investigate closer","investigate",
    "hover_monitor","hover monitor","hover",
    "alert_operator","alert operator",
    "continue_patrol","continue patrol","patrol",
    "return_to_base","return to base","return",
}


def score_verbalization_nd(reply: str, true_risk: str) -> dict:
    """5-point V-series rubric adapted for ND risk vocabulary."""
    r      = reply.lower()
    words  = reply.split()
    n      = len(words)
    s1 = int(any(kw in r for kw in _ND_SCENE_KW))
    s2 = int(any(kw in r for kw in _ND_PROX_KW))
    s4 = int(10 <= n <= 150)
    s5 = int(any(a in r for a in _ND_ACTION_KW))
    true_nd = TRUTH_TO_ND.get(true_risk, true_risk.upper())
    detected = None
    for lvl in ("emergency", "alert", "clear"):
        if lvl in r:
            detected = lvl.upper(); break
    if detected is None:
        if any(w in r for w in ("danger","collapsed","fire","critical","hazard")): detected = "EMERGENCY"
        elif any(w in r for w in ("warning","concern","unusual","caution","investigate")): detected = "ALERT"
        elif any(w in r for w in ("okay","fine","no issue","normal","safe")): detected = "CLEAR"
    s3 = int(detected == true_nd) if detected else 0
    detected_action = None
    for a in ("investigate_closer","investigate closer","hover_monitor","hover monitor",
              "alert_operator","alert operator","continue_patrol","continue patrol",
              "return_to_base","return to base"):
        if a in r:
            detected_action = a.replace(" ","_").upper(); break
    return {
        "s1_scene":        s1,
        "s2_proximity":    s2,
        "s3_risk":         s3,
        "s4_length":       s4,
        "s5_pilot_action": s5,
        "quality_score":   s1 + s2 + s3 + s4 + s5,
        "detected_risk":   detected or "unknown",
        "detected_action": detected_action or "unknown",
        "word_count":      n,
    }


def _extract_field(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith(field.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_risk(text: str) -> str:
    val = _extract_field(text, "Risk").upper()
    for lvl in ("EMERGENCY", "ALERT", "CLEAR"):
        if lvl in val: return lvl
    up = text.upper()
    for lvl in ("EMERGENCY", "ALERT", "CLEAR"):
        if lvl in up: return lvl
    return "unknown"


def _extract_confidence(text: str) -> float:
    val = _extract_field(text, "Confidence")
    if not val: return -1.0
    m = re.search(r"\d+\.?\d*", val)
    if m:
        f = float(m.group())
        return round(f if f <= 1.0 else f / 100.0, 3)
    return -1.0


def is_truncated(reply: str) -> int:
    s = reply.strip()
    if not s: return 1
    return int(s[-1] not in ".!?:\"'")


def _check_patrol_complete(workflow_rows: list) -> bool:
    return any(r["tool_name"] in ("land", "disarm") for r in workflow_rows)


def _check_correct_response(scenario: dict, workflow_rows: list, camera_rows: list,
                              final_text: str) -> bool:
    """
    clear_room → patrol complete (landed).
    alert_room → investigated (called analyze_scene(investigate)) OR
                 reported ALERT/EMERGENCY in final text.
    """
    if scenario["truth"] == "CLEAR":
        return _check_patrol_complete(workflow_rows)
    else:
        investigated   = any(r.get("context") == "investigate" for r in camera_rows)
        reported_alert = any(lvl in final_text.upper() for lvl in ("ALERT", "EMERGENCY"))
        mp_triggered   = any(r.get("mp_risk") in ("medium", "high") for r in camera_rows)
        return investigated or reported_alert or mp_triggered


# ── CSV fields ────────────────────────────────────────────────────────────────
RUN_FIELDS = [
    "orchestrator", "global_run", "scenario", "scene_label", "truth",
    "patrol_complete", "correct_room_report",
    "n_scene_calls", "n_drone_commands", "n_room_events",
    # V-series 5-point rubric on final reply
    "quality_score", "s1_scene", "s2_proximity", "s3_risk", "s4_length", "s5_pilot_action",
    "detected_risk", "detected_action", "word_count", "truncated",
    # Structured reply fields
    "risk", "confidence", "description", "sensor_note", "proximity",
    # Full reply logging (no truncation)
    "all_turns_text",
    "final_text",
    # Performance
    "total_pipeline_ms", "mean_pipeline_ms",
    "loop_cost_usd", "total_cost_usd",
    "wall_time_s", "error",
]

SUMMARY_FIELDS = [
    "orchestrator", "scenario", "scene_label", "truth", "n_runs",
    "patrol_rate", "correct_rate", "correct_lo", "correct_hi",
    "mean_quality", "mean_s3_risk",
    "mean_scene_calls", "mean_drone_cmds",
    "mean_pipeline_ms", "mean_total_cost_usd",
]


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_runs      = ND_RESULTS / f"ND2_runs_{ts}.csv"
    out_camera    = ND_RESULTS / f"ND2_camera_{ts}.csv"
    out_workflow  = ND_RESULTS / f"ND2_workflow_{ts}.csv"
    out_summary   = ND_RESULTS / f"ND2_summary_{ts}.csv"
    out_api_stats = ND_RESULTS / f"ND2_api_stats_{ts}.csv"  # per-turn LLM metrics

    total_runs = len(ORCHESTRATORS) * len(SCENARIOS) * N_RUNS
    print("=" * 65)
    print("EXP-ND2: Agentic Vision-Guided Room Emergency Surveillance")
    print(f"Orchestrators : {ORCHESTRATORS}")
    print(f"Scenarios     : {len(SCENARIOS)}   N={N_RUNS}   Total: {total_runs}")
    print("=" * 65)

    # ── Load camera models once ───────────────────────────────────────────────
    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (Tier 2)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print()

    all_run_rows      = []
    all_camera_rows   = []
    all_workflow_rows = []
    all_summary_rows  = []
    all_api_stats_rows = []   # per-turn LLM metrics (C-series style)
    global_run        = 0

    for orchestrator in ORCHESTRATORS:
        print(f"\n{'═'*65}")
        print(f"  ORCHESTRATOR: {orchestrator}")
        print(f"{'═'*65}")

        for scenario in SCENARIOS:
            s_name = scenario["name"]
            label  = scenario["scene_label"]
            truth  = scenario["truth"]
            print(f"\n  {'─'*60}")
            print(f"  Scenario: {s_name}  scene={label}  truth={truth}")
            print(f"  {scenario['description']}")
            print(f"  {'─'*60}")

            try:
                jpeg = get_saved_frame(label)
            except FileNotFoundError as e:
                print(f"  SKIP — {e}")
                continue

            sc_correct   = []
            sc_patrol    = []
            sc_qualities = []
            sc_s3s       = []
            sc_costs     = []
            sc_lats      = []

            for run in range(1, N_RUNS + 1):
                global_run += 1
                print(f"\n    Run {run}/{N_RUNS}  [global {global_run}/{total_runs}]")

                agent = NDAgent(
                    session_id   = f"ND2_{orchestrator}_{s_name}_r{run}_{ts}",
                    mp_detector  = mp_detector,
                    mp_type      = mp_type,
                    yolo_model   = yolo_model,
                    yolo_type    = yolo_type,
                    coco_model   = coco_model,
                    depth_pipe   = depth_pipe,
                    frame_source = "saved",
                )
                agent.reset_run(run)
                agent.set_frame(jpeg)
                agent.start_emergency_monitor()

                t_wall    = time.time()
                error_msg = ""
                try:
                    final_text, api_stats, tool_trace = agent.run_orchestrator_loop(
                        orchestrator = orchestrator,
                        user_prompt  = MISSION_PROMPT,
                        max_turns    = 50,   # increased from 30 — Claude needs ~35 turns for full patrol+land
                        max_tokens   = 2048,
                    )
                except Exception as e:
                    final_text = ""
                    api_stats  = []
                    tool_trace = []
                    error_msg  = str(e)
                    print(f"    ERROR: {e}")
                finally:
                    agent.stop_emergency_monitor()

                wall_s = round(time.time() - t_wall, 2)

                # ── Per-turn LLM metrics (C-series style) ────────────────────
                for stat in api_stats:
                    all_api_stats_rows.append({
                        "orchestrator": orchestrator,
                        "global_run":   global_run,
                        "scenario":     s_name,
                        "scene_label":  label,
                        "turn":         stat.get("turn"),
                        "latency_s":    stat.get("latency_s"),
                        "input_tokens": stat.get("input_tokens"),
                        "output_tokens":stat.get("output_tokens"),
                        "cost_usd":     stat.get("cost_usd"),
                        "text":         stat.get("text", ""),
                    })

                # ── Collect sensor + workflow rows ────────────────────────────
                cam_rows  = list(agent._camera_rows)
                wkf_rows  = list(agent._workflow_rows)
                for r in cam_rows:
                    r["orchestrator"] = orchestrator
                    r["scenario"]     = s_name
                    r["scene_label"]  = label
                for r in wkf_rows:
                    r["orchestrator"] = orchestrator
                    r["scenario"]     = s_name
                    r["scene_label"]  = label
                all_camera_rows.extend(cam_rows)
                all_workflow_rows.extend(wkf_rows)

                # ── All LLM turn texts — no truncation ───────────────────────
                all_turns      = [s.get("text", "") for s in api_stats if s.get("text")]
                all_turns_text = " |TURN| ".join(all_turns)

                # ── Mission metrics ───────────────────────────────────────────
                patrol_done  = _check_patrol_complete(wkf_rows)
                correct_resp = _check_correct_response(scenario, wkf_rows, cam_rows, final_text)
                n_cam        = len(cam_rows)
                n_drone      = len([r for r in wkf_rows if r.get("is_drone_command")])
                pipe_times   = [r["total_pipeline_ms"] for r in cam_rows]
                total_pipe   = round(sum(pipe_times), 1)
                mean_pipe    = round(float(np.mean(pipe_times)), 1) if pipe_times else 0.0
                loop_cost    = round(sum(s.get("cost_usd", 0) for s in api_stats), 6)
                total_cost   = round(loop_cost, 6)

                # ── V-series 5-point rubric on final reply ────────────────────
                scores     = score_verbalization_nd(final_text, truth)
                quality    = scores["quality_score"]
                risk       = _extract_risk(final_text)
                confidence = _extract_confidence(final_text)
                desc       = _extract_field(final_text, "Description")
                snote      = _extract_field(final_text, "Sensor note")
                prox       = _extract_field(final_text, "Proximity")
                trunc      = is_truncated(final_text)

                sc_correct.append(int(correct_resp))
                sc_patrol.append(int(patrol_done))
                sc_qualities.append(quality)
                sc_s3s.append(scores["s3_risk"])
                sc_costs.append(total_cost)
                sc_lats.append(mean_pipe)

                print(f"    patrol={patrol_done}  correct={correct_resp}  "
                      f"quality={quality}/5  risk={risk}  "
                      f"scene_calls={n_cam}  drone_cmds={n_drone}")
                print(f"    pipeline={total_pipe:.0f}ms  cost=${total_cost:.5f}  wall={wall_s}s")

                all_run_rows.append({
                    "orchestrator":       orchestrator,
                    "global_run":         global_run,
                    "scenario":           s_name,
                    "scene_label":        label,
                    "truth":              truth,
                    "patrol_complete":    int(patrol_done),
                    "correct_room_report":int(correct_resp),
                    "n_scene_calls":      n_cam,
                    "n_drone_commands":   n_drone,
                    "n_room_events":      int(agent._emergency_flag.is_set()),
                    "quality_score":      quality,
                    "s1_scene":           scores["s1_scene"],
                    "s2_proximity":       scores["s2_proximity"],
                    "s3_risk":            scores["s3_risk"],
                    "s4_length":          scores["s4_length"],
                    "s5_pilot_action":    scores["s5_pilot_action"],
                    "detected_risk":      scores["detected_risk"],
                    "detected_action":    scores["detected_action"],
                    "word_count":         scores["word_count"],
                    "truncated":          trunc,
                    "risk":               risk,
                    "confidence":         confidence,
                    "description":        desc,
                    "sensor_note":        snote,
                    "proximity":          prox,
                    "all_turns_text":     all_turns_text,
                    "final_text":         final_text,
                    "total_pipeline_ms":  total_pipe,
                    "mean_pipeline_ms":   mean_pipe,
                    "loop_cost_usd":      loop_cost,
                    "total_cost_usd":     total_cost,
                    "wall_time_s":        wall_s,
                    "error":              error_msg,
                })

            # ── Scenario × orchestrator summary ──────────────────────────────
            n = len(sc_correct)
            if n:
                corr_rate, c_lo, c_hi = wilson_ci(sum(sc_correct), n)
                patr_rate, _, _       = wilson_ci(sum(sc_patrol), n)
                q_mean, _, _          = bootstrap_ci(sc_qualities)
                s3_mean               = round(sum(sc_s3s) / n, 3)
                cost_mean, _, _       = bootstrap_ci(sc_costs)
                lat_mean, _, _        = bootstrap_ci(sc_lats)
                print(f"\n    {orchestrator}/{s_name}: "
                      f"correct={corr_rate*100:.1f}% [{c_lo*100:.1f}–{c_hi*100:.1f}%]  "
                      f"patrol={patr_rate*100:.1f}%  quality={q_mean:.2f}/5")
                all_summary_rows.append({
                    "orchestrator":        orchestrator,
                    "scenario":            s_name,
                    "scene_label":         label,
                    "truth":               truth,
                    "n_runs":              n,
                    "patrol_rate":         round(patr_rate, 3),
                    "correct_rate":        round(corr_rate, 3),
                    "correct_lo":          round(c_lo, 3),
                    "correct_hi":          round(c_hi, 3),
                    "mean_quality":        round(q_mean, 3),
                    "mean_s3_risk":        s3_mean,
                    "mean_scene_calls":    round(np.mean([r["n_scene_calls"]   for r in all_run_rows
                                                          if r["orchestrator"]==orchestrator
                                                          and r["scenario"]==s_name]), 2),
                    "mean_drone_cmds":     round(np.mean([r["n_drone_commands"] for r in all_run_rows
                                                          if r["orchestrator"]==orchestrator
                                                          and r["scenario"]==s_name]), 2),
                    "mean_pipeline_ms":    round(lat_mean, 1),
                    "mean_total_cost_usd": round(cost_mean, 6),
                })

    # ── Write all CSVs ────────────────────────────────────────────────────────
    write_csv(out_runs, all_run_rows, RUN_FIELDS)
    print(f"\nSaved runs      → {out_runs}")

    cam_fields_ext = CAMERA_CSV_FIELDS + ["orchestrator", "scenario", "scene_label"]
    wkf_fields_ext = WORKFLOW_CSV_FIELDS + ["orchestrator", "scenario", "scene_label"]

    if all_camera_rows:
        write_csv(out_camera, all_camera_rows, cam_fields_ext)
        print(f"Saved camera    → {out_camera}")
    if all_workflow_rows:
        write_csv(out_workflow, all_workflow_rows, wkf_fields_ext)
        print(f"Saved workflow  → {out_workflow}")
    if all_summary_rows:
        write_csv(out_summary, all_summary_rows, SUMMARY_FIELDS)
        print(f"Saved summary   → {out_summary}")
    if all_api_stats_rows:
        write_csv(out_api_stats, all_api_stats_rows, LLM_STATS_CSV_FIELDS)
        print(f"Saved api_stats → {out_api_stats}  "
              f"({len(all_api_stats_rows)} turns × "
              f"{len(set(r['orchestrator'] for r in all_api_stats_rows))} orchestrators)")

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ND2 RESULTS")
    print(f"{'Orch':<14} {'Scenario':<12} {'Correct':>8} {'Patrol':>7} "
          f"{'Quality':>8} {'SceneCalls':>11} {'Cost/run':>10}")
    print("─" * 70)
    for r in all_summary_rows:
        print(f"  {r['orchestrator']:<12} {r['scenario']:<12} "
              f"{r['correct_rate']*100:>7.1f}%  "
              f"{r['patrol_rate']*100:>6.1f}%  "
              f"{r['mean_quality']:>6.2f}/5  "
              f"{r['mean_scene_calls']:>9.1f}    "
              f"${r['mean_total_cost_usd']:.5f}")
    print("=" * 70)

    # ── Overall per orchestrator ──────────────────────────────────────────────
    print("\nOverall per orchestrator:")
    for orch in ORCHESTRATORS:
        rows = [r for r in all_run_rows if r["orchestrator"] == orch and r["error"] == ""]
        if not rows: continue
        corr  = sum(r["correct_room_report"] for r in rows)
        qual  = np.mean([r["quality_score"] for r in rows])
        cost  = sum(r["total_cost_usd"] for r in rows)
        acc, lo, hi = wilson_ci(corr, len(rows))
        print(f"  {orch:<14} correct={acc*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}%]  "
              f"quality={qual:.2f}/5  total_cost=${cost:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
