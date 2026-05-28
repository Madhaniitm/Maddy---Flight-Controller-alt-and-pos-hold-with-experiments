"""
EXP-ND3: Human-in-the-Loop Vision-Guided Drone Control
========================================================
Goal:
    Identical to ND2 but with a mandatory human approval gate before
    every drone control command. Camera reads (analyze_scene) and
    telemetry reads (get_sensor_status) are NEVER gated — they always
    run automatically.

    Only DRONE_CONTROL_TOOLS require operator approval:
        arm, disarm, find_hover_throttle,
        enable/disable_altitude_hold, set_altitude_target,
        hover, set_throttle, wait,
        move_forward/backward/left/right,
        stop_movement, navigate_to_waypoint, return_to_home,
        rotate_yaw, set_home_position, land, emergency_land

    The approve_callback is a terminal prompt. In live operation, this
    would be a UI button; for the experiment it simulates a human operator
    who can approve or deny each proposed action.

    Operator interaction is logged to ND3_approvals.csv with:
      - tool proposed (name + args)
      - telemetry context shown to operator
      - operator decision (approved / denied)
      - deny reason (free text)
      - response time (seconds)

    Additionally records same camera and workflow CSVs as ND2 for
    direct comparison.

Scenarios : 2 (safe_scene, hazard_scene), N=3 runs each (shorter — human in loop)
Output:
    results/ND3_camera_<ts>.csv    — per analyze_scene call
    results/ND3_workflow_<ts>.csv  — per drone command + approval flag
    results/ND3_approvals_<ts>.csv — per human approval decision
    results/ND3_summary_<ts>.csv   — per scenario metrics

Run:
    export GLOG_minloglevel=3
    /opt/homebrew/bin/python3.11 experiments/exp_ND3_human_in_loop_vision_control.py

    Interactive mode — the script will pause and prompt you at every
    drone command. Type 'y' (approve) or 'n <reason>' (deny).
    Example:
        Approve arm({})? [y/n reason]: y
        Approve move_forward({"distance_m":0.5})? [y/n reason]: n too close to wall
"""

import sys, os, pathlib, datetime, time, json
import numpy as np
import cv2

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
EXP_DIR   = pathlib.Path(__file__).parent
VIZ_DIR   = REPO_ROOT / "Image verbalization experiments"
sys.path.insert(0, str(VIZ_DIR))
sys.path.insert(0, str(EXP_DIR))

from verbalization_utils import (
    get_saved_frame, wilson_ci, bootstrap_ci, write_csv, RESULTS_DIR as _VIZ_RESULTS
)
from enhanced_yolo_pipeline import (
    load_enhanced_yolo, load_coco_yolo, load_depth_anything,
)
from robust_local_detector import load_mediapipe_detector

from nd_series_agent import (
    NDAgent, CAMERA_CSV_FIELDS, WORKFLOW_CSV_FIELDS,
)

# ── Output paths ──────────────────────────────────────────────────────────────
ND_RESULTS = EXP_DIR / "results"
ND_RESULTS.mkdir(exist_ok=True)

# ND3 is interactive → fewer runs (3 per scenario = 6 total operator sessions)
N_RUNS       = 3
VISION_MODEL = "gpt4o"

# ── Scenarios (room emergency surveillance, same framing as ND2) ──────────────
SCENARIOS = [
    {
        "name":        "clear_room",
        "scene_label": "door_open",
        "truth":       "CLEAR",
        "description": "Normal room. Patrol expected to complete. Operator approves all movement.",
    },
    {
        "name":        "alert_room",
        "scene_label": "person_near",
        "truth":       "ALERT",
        "description": "Person visible. Drone should investigate. Operator may redirect if needed.",
    },
]

# ── Mission prompt (room surveillance with human approval) ────────────────────
MISSION_PROMPT = """\
Execute an indoor room emergency surveillance mission with human operator oversight.

MISSION: You are a surveillance drone deployed to monitor this room for emergencies.
Take off to 1 metre, conduct a patrol sweep, investigate anything suspicious, and
produce a final structured room report.

MANDATORY PROTOCOL:
  1. plan_workflow("room_surveillance", steps)  — write the full patrol plan first
  2. arm() → find_hover_throttle() → enable_altitude_hold() → set_altitude_target(1.0)
  3. At each patrol position:
       a. analyze_scene(context="patrol")       — survey the room (auto, no approval)
       b. get_sensor_status()                   — log altitude and position (auto)
       c. Decide based on room_status:
            EMERGENCY → hover() + analyze_scene(context="investigate") + ALERT_OPERATOR
            ALERT     → navigate toward subject + analyze_scene(context="investigate")
            CLEAR     → continue to next patrol waypoint
  4. After full patrol: land() and produce final room safety report

Note: A HUMAN OPERATOR must approve every drone control command before it executes.
      Camera scans (analyze_scene) and sensor reads run automatically without approval.
      If the operator denies a command, adapt your plan — do not repeat the denied command.
      The operator may redirect you toward specific areas of the room.
"""

# ── Approval log fields ───────────────────────────────────────────────────────
APPROVAL_FIELDS = [
    "run", "scenario", "scene_label",
    "approval_num",
    "tool_name", "tool_args",
    "telemetry_context",
    "approved",           # 1 = approved, 0 = denied
    "deny_reason",        # operator's reason if denied
    "response_time_s",    # time operator took to respond
]

SUMMARY_FIELDS = [
    "run", "scenario",
    "scene_label", "truth",
    "patrol_complete",
    "correct_room_report",
    "n_scene_calls",
    "n_drone_commands_proposed",
    "n_approved",
    "n_denied",
    "approval_rate",
    "total_pipeline_ms",
    "mean_pipeline_ms",
    "total_camera_cost_usd",
    "total_loop_cost_usd",
    "total_cost_usd",
    "mean_operator_response_s",
    "total_operator_time_s",
    "wall_time_s",
    "final_room_status",
    "error",
]


def _check_patrol_complete(workflow_rows: list[dict]) -> bool:
    for r in workflow_rows:
        if r["tool_name"] in ("land", "disarm"):
            return True
    return False


def _check_correct_response(scenario: dict, workflow_rows: list[dict],
                             camera_rows: list[dict] = None) -> bool:
    if scenario["truth"] == "CLEAR":
        return _check_patrol_complete(workflow_rows)
    else:  # ALERT — drone should have investigated the person
        cam = camera_rows or []
        investigated = any(r.get("context") == "investigate" for r in cam)
        reported_alert = any(r.get("llm_risk") in ("ALERT", "EMERGENCY") for r in cam)
        return investigated or reported_alert


def make_approve_callback(run_id: str, scenario_name: str,
                          scene_label: str, approval_rows: list[dict]):
    """
    Returns a callable approve_callback(tool_name, tool_args, context_str) -> (bool, str).

    Prompts the operator in the terminal. Logs decision to approval_rows.
    """
    approval_counter = [0]

    def _callback(tool_name: str, tool_args: dict, context_str: str) -> tuple:
        approval_counter[0] += 1
        num = approval_counter[0]

        print()
        print(f"  ┌─ HUMAN APPROVAL REQUIRED ─────────────────────────────┐")
        print(f"  │  Run: {run_id}  [{scenario_name}]  approval #{num}")
        print(f"  │  Proposed: {tool_name}({json.dumps(tool_args)[:60]})")
        print(f"  │  Context:  {context_str[:120]}")
        print(f"  └────────────────────────────────────────────────────────┘")
        print(f"  Approve {tool_name}? [y / n <reason>]: ", end="", flush=True)

        t_start = time.time()
        try:
            answer = input().strip()
        except (EOFError, KeyboardInterrupt):
            answer = "n experiment_interrupted"
        response_s = round(time.time() - t_start, 2)

        if answer.lower().startswith("y"):
            approved    = True
            deny_reason = ""
            print(f"  ✅ APPROVED  (response: {response_s:.1f}s)")
        else:
            approved    = False
            deny_reason = answer[1:].strip() if len(answer) > 1 else "operator denied"
            print(f"  ❌ DENIED: {deny_reason}  (response: {response_s:.1f}s)")

        approval_rows.append({
            "run":              run_id,
            "scenario":         scenario_name,
            "scene_label":      scene_label,
            "approval_num":     num,
            "tool_name":        tool_name,
            "tool_args":        json.dumps(tool_args),
            "telemetry_context": context_str[:300],
            "approved":         int(approved),
            "deny_reason":      deny_reason,
            "response_time_s":  response_s,
        })

        return approved, deny_reason

    return _callback


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_camera    = ND_RESULTS / f"ND3_camera_{ts}.csv"
    out_workflow  = ND_RESULTS / f"ND3_workflow_{ts}.csv"
    out_approvals = ND_RESULTS / f"ND3_approvals_{ts}.csv"
    out_summary   = ND_RESULTS / f"ND3_summary_{ts}.csv"

    total_runs = len(SCENARIOS) * N_RUNS
    print("=" * 65)
    print("EXP-ND3: Human-in-the-Loop Vision-Guided Drone Control")
    print(f"Model     : {VISION_MODEL}")
    print(f"Scenarios : {len(SCENARIOS)}   N={N_RUNS}   Total runs: {total_runs}")
    print()
    print("OPERATOR INSTRUCTIONS:")
    print("  • Type 'y' to approve a command")
    print("  • Type 'n <reason>' to deny (e.g. 'n too close to wall')")
    print("  • Camera checks and sensor reads run automatically (no prompt)")
    print("=" * 65)

    # ── Load camera models ────────────────────────────────────────────────────
    print("\nLoading MediaPipe EfficientDet-Lite0 (Tier 1.5)…")
    mp_detector, mp_type = load_mediapipe_detector()
    print("Loading YOLO-World (Tier 2)…")
    yolo_model, yolo_type = load_enhanced_yolo()
    print("Loading YOLOv11n COCO…")
    coco_model, _ = load_coco_yolo()
    print("Loading DepthAnything v2 Metric Indoor…")
    depth_pipe, _ = load_depth_anything()
    print()

    all_camera_rows   = []
    all_workflow_rows = []
    all_approval_rows = []
    all_summary_rows  = []

    global_run = 0

    for scenario in SCENARIOS:
        s_name  = scenario["name"]
        label   = scenario["scene_label"]
        truth   = scenario["truth"]
        print(f"\n{'─'*65}")
        print(f"Scenario: {s_name}  scene={label}  truth={truth}")
        print(f"  {scenario['description']}")
        print(f"{'─'*65}")

        # Load scene frame
        try:
            jpeg = get_saved_frame(label)
        except FileNotFoundError as e:
            print(f"SKIP — {e}")
            continue

        for run in range(1, N_RUNS + 1):
            global_run += 1
            run_id = f"ND3_{s_name}_r{run}"
            print(f"\n  Run {run}/{N_RUNS}  [global {global_run}/{total_runs}]  id={run_id}")
            print(f"  {'─'*60}")

            # Approval log for this run
            run_approval_rows: list[dict] = []

            # ── Build fresh agent ─────────────────────────────────────────────
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

            # ── Build human approval callback ─────────────────────────────────
            approve_cb = make_approve_callback(
                run_id, s_name, label, run_approval_rows
            )

            # Start background emergency monitor
            agent.start_emergency_monitor()

            t_wall    = time.time()
            error_msg = ""
            try:
                final_text, api_stats, tool_trace = agent.run_nd_agent_loop(
                    user_prompt      = MISSION_PROMPT,
                    max_turns        = 30,
                    max_tokens       = 2048,
                    approve_callback = approve_cb,
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

            # ── Collect rows ──────────────────────────────────────────────────
            cam_rows = list(agent._camera_rows)
            wkf_rows = list(agent._workflow_rows)
            for r in cam_rows:
                r["scenario"]   = s_name
                r["scene_label"] = label
            for r in wkf_rows:
                r["scenario"]   = s_name
                r["scene_label"] = label

            all_camera_rows.extend(cam_rows)
            all_workflow_rows.extend(wkf_rows)
            all_approval_rows.extend(run_approval_rows)

            # ── Metrics ───────────────────────────────────────────────────────
            patrol_done   = _check_patrol_complete(wkf_rows)
            correct_resp  = _check_correct_response(scenario, wkf_rows, cam_rows)
            n_cam         = len(cam_rows)
            n_drone_prop  = len(run_approval_rows)   # proposed drone commands
            n_approved    = sum(r["approved"] for r in run_approval_rows)
            n_denied      = n_drone_prop - n_approved
            approval_rate = round(n_approved / n_drone_prop, 4) if n_drone_prop else 0.0
            pipe_times    = [r["total_pipeline_ms"] for r in cam_rows]
            total_pipe    = round(sum(pipe_times), 1)
            mean_pipe     = round(float(np.mean(pipe_times)), 1) if pipe_times else 0.0
            cam_cost      = round(sum(r["llm_cost_usd"] for r in cam_rows), 6)
            loop_cost     = round(sum(s.get("cost_usd", 0) for s in api_stats), 6)
            total_cost    = round(cam_cost + loop_cost, 6)
            last_status   = cam_rows[-1]["llm_risk"] if cam_rows else "UNKNOWN"

            op_times      = [r["response_time_s"] for r in run_approval_rows]
            mean_op_s     = round(float(np.mean(op_times)), 2) if op_times else 0.0
            total_op_s    = round(sum(op_times), 2)

            print(f"  patrol_done={patrol_done}  correct_resp={correct_resp}  "
                  f"final_room={last_status}")
            print(f"  approvals={n_approved}/{n_drone_prop}  denied={n_denied}")
            print(f"  pipeline={total_pipe:.0f}ms total  cost=${total_cost:.5f}")
            print(f"  operator_total={total_op_s:.1f}s  mean_per_prompt={mean_op_s:.1f}s")

            all_summary_rows.append({
                "run":                         global_run,
                "scenario":                    s_name,
                "scene_label":                 label,
                "truth":                       truth,
                "patrol_complete":             int(patrol_done),
                "correct_room_report":         int(correct_resp),
                "n_scene_calls":               n_cam,
                "n_drone_commands_proposed":   n_drone_prop,
                "n_approved":                  n_approved,
                "n_denied":                    n_denied,
                "approval_rate":               approval_rate,
                "total_pipeline_ms":           total_pipe,
                "mean_pipeline_ms":            mean_pipe,
                "total_camera_cost_usd":       cam_cost,
                "total_loop_cost_usd":         loop_cost,
                "total_cost_usd":              total_cost,
                "mean_operator_response_s":    mean_op_s,
                "total_operator_time_s":       total_op_s,
                "wall_time_s":                 wall_s,
                "final_room_status":           last_status,
                "error":                       error_msg,
            })

    # ── Write CSVs ────────────────────────────────────────────────────────────
    cam_fields_ext = CAMERA_CSV_FIELDS  + ["scenario", "scene_label"]
    wkf_fields_ext = WORKFLOW_CSV_FIELDS + ["scenario", "scene_label"]

    if all_camera_rows:
        write_csv(out_camera, all_camera_rows, cam_fields_ext)
        print(f"\nSaved camera    → {out_camera}")
    if all_workflow_rows:
        write_csv(out_workflow, all_workflow_rows, wkf_fields_ext)
        print(f"Saved workflow  → {out_workflow}")
    if all_approval_rows:
        write_csv(out_approvals, all_approval_rows, APPROVAL_FIELDS)
        print(f"Saved approvals → {out_approvals}")
    if all_summary_rows:
        write_csv(out_summary, all_summary_rows, SUMMARY_FIELDS)
        print(f"Saved summary   → {out_summary}")

    # ── Overall stats ─────────────────────────────────────────────────────────
    valid = [r for r in all_summary_rows if r["error"] == ""]
    n_tot = len(valid)
    if n_tot:
        correct_all  = sum(r["correct_room_report"] for r in valid)
        acc, lo, hi  = wilson_ci(correct_all, n_tot)
        total_cost   = sum(r["total_cost_usd"] for r in valid)
        mean_pipe    = float(np.mean([r["mean_pipeline_ms"] for r in valid]))
        all_op_times = [r for r in all_approval_rows]
        mean_op      = float(np.mean([r["response_time_s"] for r in all_op_times])) if all_op_times else 0.0
        n_approved   = sum(r["n_approved"]            for r in valid)
        n_proposed   = sum(r["n_drone_commands_proposed"] for r in valid)
        overall_rate = round(n_approved / n_proposed, 3) if n_proposed else 0.0

        print("\n" + "=" * 65)
        print("ND3 OVERALL RESULTS")
        print(f"  Correct mission response : {acc*100:.1f}% [{lo*100:.1f}–{hi*100:.1f}%]  N={n_tot}")
        print(f"  Overall approval rate    : {overall_rate*100:.1f}%  ({n_approved}/{n_proposed})")
        print(f"  Mean operator response   : {mean_op:.1f} s per prompt")
        print(f"  Mean analyze_scene lat   : {mean_pipe:.0f} ms")
        print(f"  Total cost               : ${total_cost:.4f}")
        print("=" * 65)


if __name__ == "__main__":
    main()
