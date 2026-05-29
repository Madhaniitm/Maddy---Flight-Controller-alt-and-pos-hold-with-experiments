"""
EXP-ND2 Targeted Re-run: GPT-4o-mini × alert_room × 5 runs
============================================================
Investigates non-deterministic hover-loop behaviour seen in the
original ND2 experiment (20260528_224548).

Output:
    results/ND2_rerun_gpt4omini_alert_<ts>.csv   — per-run detail
    results/ND2_rerun_gpt4omini_alert_summary_<ts>.csv — summary row

Run:
    /opt/homebrew/bin/python3.11 experiments/exp_ND2_gpt4omini_alert_rerun.py
"""

import sys, os, pathlib, datetime, time, json, re
import numpy as np

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

# ── Import scoring helpers from main experiment ──────────────────────────────
from exp_ND2_agentic_vision_control import (
    MISSION_PROMPT,
    score_verbalization_nd,
    _extract_risk, _extract_confidence, _extract_field,
    is_truncated, _check_patrol_complete, _check_correct_response,
    RUN_FIELDS, SUMMARY_FIELDS,
)

ND_RESULTS = EXP_DIR / "results"
ND_RESULTS.mkdir(exist_ok=True)

ORCHESTRATOR = "gpt4o_mini"
N_RUNS       = 5

SCENARIO = {
    "name":        "alert_room",
    "scene_label": "person_near",
    "truth":       "ALERT",
    "description": "Person close to drone camera. Drone should investigate and report findings.",
}


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_runs      = ND_RESULTS / f"ND2_rerun_gpt4omini_alert_{ts}.csv"
    out_summary   = ND_RESULTS / f"ND2_rerun_gpt4omini_alert_summary_{ts}.csv"
    out_api_stats = ND_RESULTS / f"ND2_rerun_gpt4omini_alert_apistats_{ts}.csv"

    print("=" * 65)
    print("EXP-ND2 Re-run: gpt4o_mini × alert_room × 5 runs")
    print(f"Investigating non-deterministic hover-loop from ND2 original")
    print("=" * 65)

    print("\nLoading MediaPipe…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print()

    try:
        jpeg = get_saved_frame(SCENARIO["scene_label"])
    except FileNotFoundError as e:
        print(f"FATAL — {e}")
        return

    all_run_rows      = []
    all_api_stats_rows = []

    sc_correct   = []
    sc_patrol    = []
    sc_qualities = []
    sc_s3s       = []
    sc_costs     = []
    sc_lats      = []

    for run in range(1, N_RUNS + 1):
        print(f"\n{'─'*60}")
        print(f"  Run {run}/{N_RUNS}  [gpt4o_mini / alert_room]")
        print(f"{'─'*60}")

        agent = NDAgent(
            session_id   = f"ND2_rerun_gpt4omini_alert_r{run}_{ts}",
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
                orchestrator = ORCHESTRATOR,
                user_prompt  = MISSION_PROMPT,
                max_turns    = 50,
                max_tokens   = 2048,
            )
        except Exception as e:
            final_text = ""
            api_stats  = []
            tool_trace = []
            error_msg  = str(e)
            print(f"  ERROR: {e}")
        finally:
            agent.stop_emergency_monitor()

        wall_s = round(time.time() - t_wall, 2)

        # ── per-turn LLM stats ────────────────────────────────────────────────
        for stat in api_stats:
            all_api_stats_rows.append({
                "orchestrator": ORCHESTRATOR,
                "global_run":   run,
                "scenario":     SCENARIO["name"],
                "scene_label":  SCENARIO["scene_label"],
                "turn":         stat.get("turn"),
                "latency_s":    stat.get("latency_s"),
                "input_tokens": stat.get("input_tokens"),
                "output_tokens":stat.get("output_tokens"),
                "cost_usd":     stat.get("cost_usd"),
                "text":         stat.get("text", ""),
            })

        cam_rows  = list(agent._camera_rows)
        wkf_rows  = list(agent._workflow_rows)

        all_turns      = [s.get("text", "") for s in api_stats if s.get("text")]
        all_turns_text = " |TURN| ".join(all_turns)

        patrol_done  = _check_patrol_complete(wkf_rows)
        correct_resp = _check_correct_response(SCENARIO, wkf_rows, cam_rows, final_text)
        n_cam        = len(cam_rows)
        n_drone      = len([r for r in wkf_rows if r.get("is_drone_command")])
        pipe_times   = [r["total_pipeline_ms"] for r in cam_rows]
        total_pipe   = round(sum(pipe_times), 1)
        mean_pipe    = round(float(np.mean(pipe_times)), 1) if pipe_times else 0.0
        loop_cost    = round(sum(s.get("cost_usd", 0) for s in api_stats), 6)
        total_cost   = round(loop_cost, 6)

        scores     = score_verbalization_nd(final_text, SCENARIO["truth"])
        quality    = scores["quality_score"]
        risk       = _extract_risk(final_text)
        confidence = _extract_confidence(final_text)
        desc       = _extract_field(final_text, "Description")
        snote      = _extract_field(final_text, "Sensor note")
        prox       = _extract_field(final_text, "Proximity")
        trunc      = is_truncated(final_text)

        # ── hover-loop detection ──────────────────────────────────────────────
        tool_names = [r.get("tool_name", "") for r in wkf_rows]
        hover_count = tool_names.count("hover")
        used_land   = "land" in tool_names
        used_plan   = any("plan_workflow" in (r.get("tool_name","")) for r in wkf_rows)

        sc_correct.append(int(correct_resp))
        sc_patrol.append(int(patrol_done))
        sc_qualities.append(quality)
        sc_s3s.append(scores["s3_risk"])
        sc_costs.append(total_cost)
        sc_lats.append(mean_pipe)

        print(f"  patrol={patrol_done}  correct={correct_resp}  quality={quality}/5  risk={risk}")
        print(f"  scene_calls={n_cam}  drone_cmds={n_drone}  hover_count={hover_count}")
        print(f"  used_land={used_land}  used_plan_workflow={used_plan}")
        print(f"  pipeline={total_pipe:.0f}ms  cost=${total_cost:.5f}  wall={wall_s}s")
        print(f"  word_count={scores['word_count']}  truncated={trunc}")
        print(f"  --- Final text ({scores['word_count']} words) ---")
        print(f"  {final_text[:500]!r}" if final_text else "  [EMPTY]")

        all_run_rows.append({
            "orchestrator":       ORCHESTRATOR,
            "global_run":         run,
            "scenario":           SCENARIO["name"],
            "scene_label":        SCENARIO["scene_label"],
            "truth":              SCENARIO["truth"],
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

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(sc_correct)
    corr_rate, c_lo, c_hi = wilson_ci(sum(sc_correct), n)
    patr_rate, _, _       = wilson_ci(sum(sc_patrol), n)
    q_mean, _, _          = bootstrap_ci(sc_qualities)
    s3_mean               = round(sum(sc_s3s) / n, 3)
    cost_mean, _, _       = bootstrap_ci(sc_costs)
    lat_mean, _, _        = bootstrap_ci(sc_lats)

    summary_row = {
        "orchestrator":        ORCHESTRATOR,
        "scenario":            SCENARIO["name"],
        "scene_label":         SCENARIO["scene_label"],
        "truth":               SCENARIO["truth"],
        "n_runs":              n,
        "patrol_rate":         round(patr_rate, 3),
        "correct_rate":        round(corr_rate, 3),
        "correct_lo":          round(c_lo, 3),
        "correct_hi":          round(c_hi, 3),
        "mean_quality":        round(q_mean, 3),
        "mean_s3_risk":        s3_mean,
        "mean_scene_calls":    round(np.mean([r["n_scene_calls"]   for r in all_run_rows]), 2),
        "mean_drone_cmds":     round(np.mean([r["n_drone_commands"] for r in all_run_rows]), 2),
        "mean_pipeline_ms":    round(lat_mean, 1),
        "mean_total_cost_usd": round(cost_mean, 6),
    }

    write_csv(out_runs, all_run_rows, RUN_FIELDS)
    write_csv(out_summary, [summary_row], SUMMARY_FIELDS)
    if all_api_stats_rows:
        write_csv(out_api_stats, all_api_stats_rows, LLM_STATS_CSV_FIELDS)

    print("\n" + "=" * 65)
    print("RERUN SUMMARY — gpt4o_mini / alert_room")
    print("=" * 65)
    print(f"  correct_rate : {corr_rate*100:.1f}%  [{c_lo*100:.1f}–{c_hi*100:.1f}%]")
    print(f"  patrol_rate  : {patr_rate*100:.1f}%")
    print(f"  quality      : {q_mean:.2f}/5")
    print(f"  s3_risk      : {s3_mean:.3f}")
    print(f"  mean_cost    : ${cost_mean:.5f}/run")
    print(f"  mean_latency : {lat_mean:.1f}ms")
    print(f"\n  Saved runs    → {out_runs}")
    print(f"  Saved summary → {out_summary}")
    print("=" * 65)

    # ── Hover-loop analysis ───────────────────────────────────────────────────
    print("\nPer-run hover-loop analysis:")
    print(f"  {'Run':<5} {'Hover#':>7} {'Land?':>6} {'Plan?':>6} {'Words':>6} {'Correct':>8}")
    print("  " + "─" * 45)
    for row in all_run_rows:
        at   = row["all_turns_text"]
        hc   = at.count("hover(") + at.count('"hover"')
        land = "yes" if "land(" in at or '"land"' in at else "no"
        plan = "yes" if "plan_workflow" in at else "no"
        print(f"  {row['global_run']:<5} {hc:>7} {land:>6} {plan:>6} "
              f"{row['word_count']:>6} {row['correct_room_report']:>8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
