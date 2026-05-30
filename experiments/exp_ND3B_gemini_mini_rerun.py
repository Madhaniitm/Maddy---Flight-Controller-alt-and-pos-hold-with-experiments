"""
EXP-ND3B: HITL Re-run — Gemini + GPT-4o-mini only
====================================================
Identical to EXP-ND3 in every way EXCEPT:
  • Only runs the two orchestrators that had code bugs in the original ND3 run:
        gemini      — broke after 1 turn (finishReason="STOP" bug, now fixed)
        gpt4o_mini  — clear_room silently failed due to Azure 500 errors
  • Output prefix: ND3B_

Bug fixes applied before this re-run (in nd_series_agent.py):
  1. Gemini break condition: split into two separate ifs so finishReason="STOP"
     alongside function calls no longer exits the loop prematurely.
  2. Inner API-error handlers: changed `break` → `raise` so errors propagate
     to the outer run-level handler and set error_msg properly.

Run:
    export GLOG_minloglevel=3
    /opt/homebrew/bin/python3.11 experiments/exp_ND3B_gemini_mini_rerun.py
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

# ── Import everything from ND2 / ND3 ─────────────────────────────────────────
from exp_ND2_agentic_vision_control import (
    MISSION_PROMPT,
    SCENARIOS,
    score_verbalization_nd,
    _extract_risk, _extract_confidence, _extract_field,
    is_truncated,
    _check_patrol_complete, _check_correct_response,
    RUN_FIELDS, SUMMARY_FIELDS,
)
from exp_ND3_human_in_loop_vision_control import (
    make_approve_callback,
    make_auto_approve_callback,
    APPROVAL_FIELDS,
    ND3_RUN_FIELDS,
    ND3_SUMMARY_FIELDS,
)

# ── Output paths ──────────────────────────────────────────────────────────────
ND_RESULTS = EXP_DIR / "results"
ND_RESULTS.mkdir(exist_ok=True)

# ── ND3B config — only the two broken orchestrators ──────────────────────────
ORCHESTRATORS = ["gpt4o_mini", "gemini"]   # only the two that had bugs
N_RUNS        = 5

# ── Approval mode ─────────────────────────────────────────────────────────────
AUTO_APPROVE  = True


# ── Main (parameterised so sub-experiments can call it directly) ──────────────

def main(orchestrators=None, prefix="ND3B"):
    """
    orchestrators : list of orchestrator names to run (default = ORCHESTRATORS)
    prefix        : output CSV file prefix, e.g. "ND3B_gemini" or "ND3B_mini"
    """
    if orchestrators is None:
        orchestrators = ORCHESTRATORS

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_runs      = ND_RESULTS / f"{prefix}_runs_{ts}.csv"
    out_camera    = ND_RESULTS / f"{prefix}_camera_{ts}.csv"
    out_workflow  = ND_RESULTS / f"{prefix}_workflow_{ts}.csv"
    out_approvals = ND_RESULTS / f"{prefix}_approvals_{ts}.csv"
    out_summary   = ND_RESULTS / f"{prefix}_summary_{ts}.csv"
    out_api_stats = ND_RESULTS / f"{prefix}_api_stats_{ts}.csv"

    total_runs = len(orchestrators) * len(SCENARIOS) * N_RUNS
    print("=" * 65)
    print(f"EXP-{prefix}: HITL Re-run — {', '.join(orchestrators)}")
    print(f"Orchestrators : {orchestrators}")
    print(f"Scenarios     : {len(SCENARIOS)}   N={N_RUNS}   Total: {total_runs}")
    print()
    if AUTO_APPROVE:
        print("MODE: AUTO (simulated operator — safety-bounded approvals)")
        print("  Approves: startup, navigation within ±2.5m / 0.5–2.0m altitude")
        print("  Rejects : out-of-bounds, excessive hover>3×, large steps")
    else:
        print("MODE: INTERACTIVE (terminal prompt per drone command)")
    print()
    print("Bug fixes applied vs original ND3:")
    print("  [1] Gemini finishReason='STOP' no longer breaks when fn_calls exist")
    print("  [2] API errors propagate to run-level handler (error_msg populated)")
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

    all_run_rows       = []
    all_camera_rows    = []
    all_workflow_rows  = []
    all_approval_rows  = []
    all_summary_rows   = []
    all_api_stats_rows = []
    global_run         = 0

    for orchestrator in orchestrators:
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

            sc_correct    = []
            sc_patrol     = []
            sc_qualities  = []
            sc_s3s        = []
            sc_costs      = []
            sc_lats       = []
            sc_apr_rates  = []
            sc_op_times   = []
            sc_n_proposed = []

            for run in range(1, N_RUNS + 1):
                global_run += 1
                run_id = f"ND3B_{orchestrator}_{s_name}_r{run}"
                print(f"\n    Run {run}/{N_RUNS}  [global {global_run}/{total_runs}]  id={run_id}")

                run_approval_rows: list = []

                agent = NDAgent(
                    session_id   = f"{run_id}_{ts}",
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

                if AUTO_APPROVE:
                    approve_cb = make_auto_approve_callback(
                        orchestrator, run_id, s_name, label, run_approval_rows
                    )
                else:
                    approve_cb = make_approve_callback(
                        orchestrator, run_id, s_name, label, run_approval_rows
                    )

                agent.start_emergency_monitor()

                t_wall    = time.time()
                error_msg = ""
                try:
                    final_text, api_stats, tool_trace = agent.run_orchestrator_loop(
                        orchestrator     = orchestrator,
                        user_prompt      = MISSION_PROMPT,
                        max_turns        = 50,
                        max_tokens       = 2048,
                        approve_callback = approve_cb,
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

                # ── Per-turn LLM stats ────────────────────────────────────────
                for stat in api_stats:
                    all_api_stats_rows.append({
                        "orchestrator":  orchestrator,
                        "global_run":    global_run,
                        "scenario":      s_name,
                        "scene_label":   label,
                        "turn":          stat.get("turn"),
                        "latency_s":     stat.get("latency_s"),
                        "input_tokens":  stat.get("input_tokens"),
                        "output_tokens": stat.get("output_tokens"),
                        "cost_usd":      stat.get("cost_usd"),
                        "text":          stat.get("text", ""),
                    })

                # ── Collect sensor + workflow rows ────────────────────────────
                cam_rows = list(agent._camera_rows)
                wkf_rows = list(agent._workflow_rows)
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
                all_approval_rows.extend(run_approval_rows)

                # ── All LLM turn texts ────────────────────────────────────────
                all_turns      = [s.get("text", "") for s in api_stats if s.get("text")]
                all_turns_text = " |TURN| ".join(all_turns)

                # ── Mission metrics (identical to ND2/ND3) ────────────────────
                patrol_done  = _check_patrol_complete(wkf_rows)
                correct_resp = _check_correct_response(scenario, wkf_rows, cam_rows, final_text)
                n_cam        = len(cam_rows)
                n_drone      = len([r for r in wkf_rows if r.get("is_drone_command")])
                pipe_times   = [r["total_pipeline_ms"] for r in cam_rows]
                total_pipe   = round(sum(pipe_times), 1)
                mean_pipe    = round(float(np.mean(pipe_times)), 1) if pipe_times else 0.0
                loop_cost    = round(sum(s.get("cost_usd", 0) for s in api_stats), 6)
                total_cost   = round(loop_cost, 6)

                scores     = score_verbalization_nd(final_text, truth)
                quality    = scores["quality_score"]
                risk       = _extract_risk(final_text)
                confidence = _extract_confidence(final_text)
                desc       = _extract_field(final_text, "Description")
                snote      = _extract_field(final_text, "Sensor note")
                prox       = _extract_field(final_text, "Proximity")
                trunc      = is_truncated(final_text)

                # ── Approval metrics ──────────────────────────────────────────
                n_proposed  = len(run_approval_rows)
                n_approved  = sum(r["approved"] for r in run_approval_rows)
                n_denied    = n_proposed - n_approved
                apr_rate    = round(n_approved / n_proposed, 4) if n_proposed else 1.0
                op_times    = [r["response_time_s"] for r in run_approval_rows]
                mean_op_s   = round(float(np.mean(op_times)), 2) if op_times else 0.0
                total_op_s  = round(sum(op_times), 2)

                sc_correct.append(int(correct_resp))
                sc_patrol.append(int(patrol_done))
                sc_qualities.append(quality)
                sc_s3s.append(scores["s3_risk"])
                sc_costs.append(total_cost)
                sc_lats.append(mean_pipe)
                sc_apr_rates.append(apr_rate)
                sc_op_times.append(mean_op_s)
                sc_n_proposed.append(n_proposed)

                print(f"    patrol={patrol_done}  correct={correct_resp}  "
                      f"quality={quality}/5  risk={risk}  "
                      f"scene_calls={n_cam}  drone_cmds={n_drone}")
                print(f"    approvals={n_approved}/{n_proposed}  denied={n_denied}  "
                      f"op_time={total_op_s:.1f}s")
                print(f"    pipeline={total_pipe:.0f}ms  cost=${total_cost:.5f}  wall={wall_s}s")
                if error_msg:
                    print(f"    ⚠️  ERROR: {error_msg[:120]}")

                all_run_rows.append({
                    # ND2-identical fields
                    "orchestrator":        orchestrator,
                    "global_run":          global_run,
                    "scenario":            s_name,
                    "scene_label":         label,
                    "truth":               truth,
                    "patrol_complete":     int(patrol_done),
                    "correct_room_report": int(correct_resp),
                    "n_scene_calls":       n_cam,
                    "n_drone_commands":    n_drone,
                    "n_room_events":       int(agent._emergency_flag.is_set()),
                    "quality_score":       quality,
                    "s1_scene":            scores["s1_scene"],
                    "s2_proximity":        scores["s2_proximity"],
                    "s3_risk":             scores["s3_risk"],
                    "s4_length":           scores["s4_length"],
                    "s5_pilot_action":     scores["s5_pilot_action"],
                    "detected_risk":       scores["detected_risk"],
                    "detected_action":     scores["detected_action"],
                    "word_count":          scores["word_count"],
                    "truncated":           trunc,
                    "risk":                risk,
                    "confidence":          confidence,
                    "description":         desc,
                    "sensor_note":         snote,
                    "proximity":           prox,
                    "all_turns_text":      all_turns_text,
                    "final_text":          final_text,
                    "total_pipeline_ms":   total_pipe,
                    "mean_pipeline_ms":    mean_pipe,
                    "loop_cost_usd":       loop_cost,
                    "total_cost_usd":      total_cost,
                    "wall_time_s":         wall_s,
                    "error":               error_msg,
                    # ND3-specific fields
                    "n_drone_commands_proposed": n_proposed,
                    "n_approved":                n_approved,
                    "n_denied":                  n_denied,
                    "approval_rate":             apr_rate,
                    "mean_operator_response_s":  mean_op_s,
                    "total_operator_time_s":     total_op_s,
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
                apr_mean              = round(sum(sc_apr_rates) / n, 4)
                op_mean               = round(sum(sc_op_times) / n, 2)
                tot_approved          = sum(r["n_approved"] for r in all_run_rows
                                            if r["orchestrator"] == orchestrator
                                            and r["scenario"] == s_name)
                tot_denied            = sum(r["n_denied"] for r in all_run_rows
                                            if r["orchestrator"] == orchestrator
                                            and r["scenario"] == s_name)

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
                    "mean_scene_calls":    round(np.mean([r["n_scene_calls"]    for r in all_run_rows
                                                          if r["orchestrator"] == orchestrator
                                                          and r["scenario"] == s_name]), 2),
                    "mean_drone_cmds":     round(np.mean([r["n_drone_commands"] for r in all_run_rows
                                                          if r["orchestrator"] == orchestrator
                                                          and r["scenario"] == s_name]), 2),
                    "mean_pipeline_ms":    round(lat_mean, 1),
                    "mean_total_cost_usd": round(cost_mean, 6),
                    "mean_n_proposed":     round(sum(sc_n_proposed) / n, 2),
                    "mean_approval_rate":  apr_mean,
                    "mean_operator_response_s": op_mean,
                    "total_approvals":     tot_approved,
                    "total_denials":       tot_denied,
                })

                print(f"\n  [{orchestrator} / {s_name}] "
                      f"correct={corr_rate*100:.1f}%  patrol={patr_rate*100:.1f}%  "
                      f"quality={q_mean:.2f}/5  approval_rate={apr_mean*100:.1f}%")

    # ── Write all CSVs ────────────────────────────────────────────────────────
    cam_fields_ext = CAMERA_CSV_FIELDS  + ["orchestrator", "scenario", "scene_label"]
    wkf_fields_ext = WORKFLOW_CSV_FIELDS + ["orchestrator", "scenario", "scene_label"]

    if all_run_rows:
        write_csv(out_runs, all_run_rows, ND3_RUN_FIELDS)
        print(f"\nSaved runs      → {out_runs}")
    if all_camera_rows:
        write_csv(out_camera, all_camera_rows, cam_fields_ext)
        print(f"Saved camera    → {out_camera}")
    if all_workflow_rows:
        write_csv(out_workflow, all_workflow_rows, wkf_fields_ext)
        print(f"Saved workflow  → {out_workflow}")
    if all_approval_rows:
        write_csv(out_approvals, all_approval_rows, APPROVAL_FIELDS)
        print(f"Saved approvals → {out_approvals}")
    if all_summary_rows:
        write_csv(out_summary, all_summary_rows, ND3_SUMMARY_FIELDS)
        print(f"Saved summary   → {out_summary}")
    if all_api_stats_rows:
        write_csv(out_api_stats, all_api_stats_rows, LLM_STATS_CSV_FIELDS)
        print(f"Saved api_stats → {out_api_stats}")

    # ── Overall summary ───────────────────────────────────────────────────────
    valid = [r for r in all_run_rows if not r["error"]]
    n_tot = len(valid)
    if n_tot:
        correct_all = sum(r["correct_room_report"] for r in valid)
        acc, lo, hi = wilson_ci(correct_all, n_tot)
        total_cost  = sum(r["total_cost_usd"] for r in valid)
        pipe_vals   = [r["mean_pipeline_ms"] for r in valid if r["mean_pipeline_ms"]]
        mean_pipe   = float(np.mean(pipe_vals)) if pipe_vals else 0.0
        n_approved  = sum(r["n_approved"] for r in valid)
        n_proposed  = sum(r["n_drone_commands_proposed"] for r in valid)
        overall_apr = round(n_approved / n_proposed, 3) if n_proposed else 1.0
        op_vals     = [r["mean_operator_response_s"] for r in valid]
        mean_op     = float(np.mean(op_vals)) if op_vals else 0.0

        errors = [r for r in all_run_rows if r["error"]]
        print("\n" + "=" * 65)
        print(f"{prefix} OVERALL RESULTS  ({', '.join(orchestrators)}, bug-fixed)")
        print(f"  Valid runs               : {n_tot}/{len(all_run_rows)}"
              + (f"  ({len(errors)} errors)" if errors else ""))
        print(f"  Correct mission response : {acc*100:.1f}%  [{lo*100:.1f}–{hi*100:.1f}%]")
        print(f"  Overall approval rate    : {overall_apr*100:.1f}%  ({n_approved}/{n_proposed})")
        print(f"  Mean operator response   : {mean_op:.1f}s per prompt")
        print(f"  Mean pipeline latency    : {mean_pipe:.0f}ms")
        print(f"  Total cost               : ${total_cost:.4f}")
        print("=" * 65)


if __name__ == "__main__":
    main(orchestrators=ORCHESTRATORS, prefix="ND3B")
