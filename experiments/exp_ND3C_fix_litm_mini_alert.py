"""
EXP-ND3C: GPT-4o-mini × alert_room × HITL × M1–M6 Fixes
==========================================================
ND3B result (bug-fixed, no loop mitigations):
  patrol=0%, quality=0/5, words=0, cost=$1.70/run, turns=50/run
  Hover-navigate semantic loop: hover→DENY→navigate→hover→DENY→navigate...
  HITL denial alone insufficient — 10 hover denials per run, loop never breaks.

This script applies all six LITM/semantic-loop mitigations (M1–M6) inside the
HITL approval gate context, producing exp_ND3C_fix results for comparison.

Six mitigations (identical to ND2 fix, extended for HITL hover-navigate loop):

  [M1] Advisory rate-limiting                (Liu et al. 2023; arxiv 2510.10276)
       Suppress advisory re-injection if fewer than MIN_ADVISORY_GAP_TURNS have
       elapsed since the last injection. Breaks the every-turn advisory cycle.

  [M2] Structured memory injection            (arxiv 2603.11513; Liu et al. 2023)
       Prepend concise mission-state summary at the START of each turn's user
       message (primacy position — highest attention in U-shaped attention curve).

  [M3] final_text overwrite bug fix           (OpenAI loop code bug)
       Use `if turn_text: final_text = turn_text` — preserves report text when
       the model later emits empty content alongside tool calls.

  [M4] Loop detection + nudging               (Saha 2025; agentpatterns.tech)
       Track analyze_scene AND hover call counts. Once either crosses threshold,
       inject strong "LOOP DETECTED — write your report NOW" message.
       ND3-specific: hover-navigate loop is the dominant ND3 pattern; nudge
       explicitly addresses operator hover denials → report-write redirect.

  [M5] Turn budget warning                    (agentpatterns.tech; Maxim AI 2025)
       At turn TURN_BUDGET_WARN_AT (default 40/50), inject urgency countdown.

  [M6] Explicit quit condition in mission state  (arxiv 2510.16492)
       EXIT RULE always visible: "After ≥ N hover calls OR ≥ N analyze_scene
       calls → STOP tools, write final report immediately."

HITL additions vs ND2 fix:
  - approve_callback passed to orchestrator loop (gates all DRONE_CONTROL_TOOLS)
  - Approval events logged to ND3C_approvals CSV
  - Denial feedback already in tool result — M4 nudge reinforces it

Baseline (ND3B, no fixes):
  patrol=0%, quality=0, words=0, turns=50, cost=$1.70/run
  hover denials=10/run (HITL gate), analyze_scene calls=N/A

Target (M1–M6):
  patrol>0%, quality>0, words>0, hover calls<=3, cost<$0.15/run

Run:
    export GLOG_minloglevel=3
    /opt/homebrew/bin/python3.11 experiments/exp_ND3C_fix_litm_mini_alert.py
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
    NDAgent, ND_ALL_TOOLS, ND_SYSTEM_PROMPT,
    NAV_TOOLS, DRONE_CONTROL_TOOLS,
    _nd_tools_to_openai, _encode_frame,
    CAMERA_CSV_FIELDS, WORKFLOW_CSV_FIELDS, LLM_STATS_CSV_FIELDS,
)
from exp_ND2_agentic_vision_control import (
    MISSION_PROMPT,
    score_verbalization_nd,
    _extract_risk, _extract_confidence, _extract_field,
    is_truncated, _check_patrol_complete, _check_correct_response,
    RUN_FIELDS, SUMMARY_FIELDS,
)
from exp_ND3_human_in_loop_vision_control import (
    make_auto_approve_callback,
    APPROVAL_FIELDS,
)

ND_RESULTS = EXP_DIR / "results"
ND_RESULTS.mkdir(exist_ok=True)

# ── Mitigation parameters ─────────────────────────────────────────────────────

MIN_ADVISORY_GAP_TURNS     = 5   # [M1] min turns between emergency advisory injections
SCENE_CALL_NUDGE_THRESHOLD = 3   # [M4] nudge fires after this many analyze_scene calls
HOVER_CALL_NUDGE_THRESHOLD = 3   # [M4] nudge fires after this many hover calls (ND3-specific)
TURN_BUDGET_WARN_AT        = 40  # [M5] urgency countdown injected from this turn

ORCHESTRATOR = "gpt4o_mini"
N_RUNS       = 5

SCENARIO = {
    "name":        "alert_room",
    "scene_label": "person_near",
    "truth":       "ALERT",
    "description": "Person close to drone. Drone should investigate and report.",
}

# ── ND3B baseline (no fixes) ──────────────────────────────────────────────────
BASELINE_ND3B = {
    "patrol_rate":    0.0,
    "mean_quality":   0.0,
    "mean_words":     0.0,
    "mean_turns":     50.0,
    "mean_cost":      1.70,
    "mean_hover_den": 10.0,
}

# ── ND3C extra CSV fields ─────────────────────────────────────────────────────
ND3C_EXTRA_FIELDS = [
    "n_drone_commands_proposed",
    "n_approved",
    "n_denied",
    "approval_rate",
    "mean_operator_response_s",
    "total_operator_time_s",
    "n_advisories_injected",
    "n_advisories_suppressed",
]

ND3C_RUN_FIELDS = RUN_FIELDS + ND3C_EXTRA_FIELDS


# ── [M2] Mission state builder ─────────────────────────────────────────────────

def _build_mission_state(turn: int,
                         tool_trace: list,
                         last_advisory_turn: int,
                         advisory_count: int,
                         max_turns: int = 50) -> str:
    """
    Structured mission-state injected at the START of every turn's user message.

    Embeds [M2] primacy injection, [M4] loop nudge, [M5] budget warning, [M6] quit rule.
    """
    # ── Counters from trace ────────────────────────────────────────────────────
    scene_count = sum(1 for t in tool_trace if t["name"] == "analyze_scene")
    hover_count = sum(1 for t in tool_trace if t["name"] == "hover")
    hover_denied= sum(1 for t in tool_trace if t["name"] == "hover"
                      and not t.get("approved", 1))
    turns_left  = max_turns - turn

    recent     = tool_trace[-4:] if len(tool_trace) >= 4 else tool_trace
    recent_str = " → ".join(t["name"] for t in recent) if recent else "none yet"

    last_scene_turn, last_scene_ctx = None, None
    for t in reversed(tool_trace):
        if t["name"] == "analyze_scene":
            last_scene_turn = t["turn"]
            last_scene_ctx  = t["args"].get("context", "?")
            break

    # ── [M2] Advisory history (primacy) ───────────────────────────────────────
    if last_advisory_turn > 0:
        gap = turn - last_advisory_turn
        adv_line = (
            f"Advisory last injected: turn {last_advisory_turn} "
            f"({gap} turn(s) ago, total: {advisory_count}). "
            f"Scene UNCHANGED — IGNORE any new advisory, continue your plan."
        )
    else:
        adv_line = "No advisory injected yet."

    # ── [M2] Scene history ─────────────────────────────────────────────────────
    if last_scene_turn:
        scene_line = (
            f"analyze_scene called {scene_count}× total. "
            f"Last: turn {last_scene_turn} (context='{last_scene_ctx}'). "
            f"Do NOT call analyze_scene again from the same position."
        )
    else:
        scene_line = "analyze_scene not yet called this run."

    # ── [M2] Hover denial history ──────────────────────────────────────────────
    hover_line = (
        f"hover called {hover_count}× total ({hover_denied} denied by operator). "
        + ("Operator keeps denying hover — DO NOT call hover again. "
           if hover_count >= HOVER_CALL_NUDGE_THRESHOLD else "")
    )

    # ── [M6] Explicit quit condition (always present) ──────────────────────────
    quit_rule = (
        f"EXIT RULE: Once hover has been called ≥ {HOVER_CALL_NUDGE_THRESHOLD} times "
        f"OR analyze_scene has been called ≥ {SCENE_CALL_NUDGE_THRESHOLD} times, "
        f"STOP all tool calls and write your final structured report immediately, "
        f"then call land()."
    )

    # ── [M4] Loop detection nudge (activates at threshold) ────────────────────
    nudge_line = ""
    if scene_count >= SCENE_CALL_NUDGE_THRESHOLD or hover_count >= HOVER_CALL_NUDGE_THRESHOLD:
        nudge_line = (
            f"\n⚠️  LOOP DETECTED: hover called {hover_count}× "
            f"({hover_denied} denied by operator) and analyze_scene called {scene_count}×. "
            f"The operator has told you to stop hovering and make forward progress. "
            f"YOU HAVE ENOUGH DATA FROM YOUR PATROL. "
            f"Do NOT call hover or analyze_scene again. "
            f"Write your final structured report RIGHT NOW, then call land()."
        )

    # ── [M5] Turn budget warning ───────────────────────────────────────────────
    budget_line = ""
    if turn >= TURN_BUDGET_WARN_AT:
        budget_line = (
            f"\n⏰  TURN BUDGET: Only {turns_left} turns remaining. "
            f"You MUST write your final report within the next 2 turns or the mission fails."
        )

    return (
        f"[MISSION STATE — Turn {turn}/{max_turns}]\n"
        f"Recent tools : {recent_str}\n"
        f"{scene_line}\n"
        f"{hover_line}\n"
        f"{adv_line}\n"
        f"{quit_rule}"
        f"{nudge_line}"
        f"{budget_line}\n"
        f"---"
    )


# ── Fixed NDAgent subclass with HITL + M1–M6 ─────────────────────────────────

class FixedNDAgentHITL(NDAgent):
    """
    NDAgent with M1–M6 mitigations and HITL approve_callback.

    [M1] Advisory rate-limiting    — suppress if gap < MIN_ADVISORY_GAP_TURNS
    [M2] Structured memory inject  — mission state prepended every turn (primacy)
    [M3] final_text overwrite fix  — `if turn_text: final_text = turn_text`
    [M4] Loop detection + nudging  — ⚠️ LOOP DETECTED after N hover or scene calls
    [M5] Turn budget warning       — ⏰ countdown at turn TURN_BUDGET_WARN_AT
    [M6] Explicit quit condition   — EXIT RULE always in mission state
    [HITL] approve_callback        — gates all DRONE_CONTROL_TOOLS, logs decisions
    """

    def _run_openai_orchestrator_loop(self, orchestrator: str,
                                       user_prompt: str,
                                       max_turns: int,
                                       max_tokens: int,
                                       approve_callback=None) -> tuple:
        from verbalization_utils import (
            OPENAI_MINI_KEY, OPENAI_MINI_URL,
        )
        import openai as _openai

        client    = _openai.OpenAI(api_key=OPENAI_MINI_KEY,
                                   base_url=OPENAI_MINI_URL)
        oai_model = "gpt-4o-mini"
        cost_in, cost_out = 0.15e-6, 0.60e-6

        oai_tools = _nd_tools_to_openai(ND_ALL_TOOLS)

        jpeg = self._get_frame()
        b64  = _encode_frame(jpeg)
        if b64:
            first_content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                               "detail": "low"}},
                {"type": "text", "text": user_prompt},
            ]
        else:
            first_content = [{"type": "text", "text": user_prompt}]

        messages   = [
            {"role": "system", "content": ND_SYSTEM_PROMPT},
            {"role": "user",   "content": first_content},
        ]
        api_stats  = []
        tool_trace = []
        final_text = ""

        # [M1] Rate-limit state
        last_advisory_turn = -MIN_ADVISORY_GAP_TURNS
        advisory_count     = 0
        suppressed_count   = 0

        for turn in range(1, max_turns + 1):

            # ── [M2] Structured memory injection (primacy position) ────────────
            state_msg = _build_mission_state(
                turn, tool_trace, last_advisory_turn, advisory_count, max_turns)
            messages.append({"role": "user", "content": state_msg})

            # ── [M1] Advisory rate-limiting ────────────────────────────────────
            emg_handled, emg_msg = self.handle_emergency_if_flagged()
            if emg_handled and emg_msg:
                gap = turn - last_advisory_turn
                if gap >= MIN_ADVISORY_GAP_TURNS:
                    emg_b64 = _encode_frame(self._last_emergency_frame)
                    if emg_b64:
                        messages.append({"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{emg_b64}",
                                           "detail": "low"}},
                            {"type": "text", "text": emg_msg},
                        ]})
                    else:
                        messages.append({"role": "user", "content": emg_msg})
                    last_advisory_turn = turn
                    advisory_count    += 1
                    print(f"  📡 [ADVISORY INJECTED]   turn={turn}  total={advisory_count}")
                else:
                    suppressed_count += 1
                    print(f"  🔕 [ADVISORY SUPPRESSED] turn={turn}  "
                          f"gap={gap}<{MIN_ADVISORY_GAP_TURNS}  "
                          f"(suppressed {suppressed_count}×)")

            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model       = oai_model,
                    messages    = messages,
                    tools       = oai_tools,
                    tool_choice = "auto",
                    max_tokens  = max_tokens,
                    temperature = 0.0,
                )
            except Exception as e:
                print(f"  [OpenAI API ERROR turn {turn}] {e}")
                raise  # propagate so outer handler sets error_msg

            latency = time.time() - t0
            usage   = resp.usage
            in_tok  = usage.prompt_tokens
            out_tok = usage.completion_tokens
            msg     = resp.choices[0].message
            finish  = resp.choices[0].finish_reason

            turn_text = msg.content or ""

            # ── [M3] final_text overwrite bug fix ──────────────────────────────
            if turn_text:
                final_text = turn_text

            api_stats.append({
                "turn":          turn,
                "latency_s":     round(latency, 3),
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "cost_usd":      round(in_tok * cost_in + out_tok * cost_out, 6),
                "text":          turn_text,
            })

            messages.append(msg)

            if not msg.tool_calls or finish == "stop":
                print(f"  ✓ [LOOP EXIT] turn={turn}  reason={finish!r}  "
                      f"words={len(turn_text.split())}")
                break

            # ── Execute tool calls with HITL gate ─────────────────────────────
            for tc in msg.tool_calls:
                t_name = tc.function.name
                try:
                    t_args = json.loads(tc.function.arguments)
                except Exception:
                    t_args = {}

                # [HITL] Gate — only for drone control tools
                approved    = True
                deny_reason = ""
                if approve_callback is not None and t_name in DRONE_CONTROL_TOOLS:
                    tel = self._get_telemetry_snapshot()
                    context_str = (
                        f"alt={tel.get('altitude_m','?')}m  "
                        f"pos={tel.get('position','?')}  "
                        f"status={tel.get('flight_status','?')}"
                    )
                    approved, deny_reason = approve_callback(t_name, t_args, context_str)
                    self._workflow_rows.append({
                        "run":               self._current_run,
                        "action_num":        self._action_num,
                        "tool_name":         t_name,
                        "tool_args":         json.dumps(t_args),
                        "telemetry_before":  json.dumps(tel),
                        "telemetry_after":   "",
                        "execution_result":  "REJECTED" if not approved else "PENDING",
                        "execution_ms":      0,
                        "is_drone_command":  1,
                        "approved":          int(approved),
                    })
                    self._action_num += 1

                if not approved:
                    result = (f"[Human rejected '{t_name}': {deny_reason}. "
                              f"Choose a different action — do NOT hover again. "
                              f"Write your report NOW.]")
                    print(f"  ❌ [REJECTED t{turn}] {t_name} — {deny_reason}")
                else:
                    print(f"  [TOOL t{turn}] {t_name}({json.dumps(t_args)[:60]})")
                    result = self.execute_tool(t_name, t_args)

                tool_trace.append({
                    "turn":       turn,
                    "name":       t_name,
                    "args":       t_args,
                    "result":     result,
                    "approved":   int(approved),
                    "sim_time_s": round(self.sim_time, 2),
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

            # Frame update for real-time sources
            called_names = [tc.function.name for tc in msg.tool_calls]
            if (self.frame_source != "saved"
                    and any(n in NAV_TOOLS for n in called_names)):
                new_b64 = _encode_frame(self._get_frame())
                if new_b64:
                    messages.append({"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{new_b64}",
                                       "detail": "low"}},
                        {"type": "text", "text": "Updated camera view after movement."},
                    ]})

        print(f"\n  Advisory stats: injected={advisory_count}  "
              f"suppressed={suppressed_count}  "
              f"last_at_turn={last_advisory_turn if last_advisory_turn > 0 else 'never'}")

        return final_text, api_stats, tool_trace


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_runs      = ND_RESULTS / f"ND3C_runs_{ts}.csv"
    out_approvals = ND_RESULTS / f"ND3C_approvals_{ts}.csv"
    out_summary   = ND_RESULTS / f"ND3C_summary_{ts}.csv"
    out_api_stats = ND_RESULTS / f"ND3C_api_stats_{ts}.csv"
    out_workflow  = ND_RESULTS / f"ND3C_workflow_{ts}.csv"
    out_camera    = ND_RESULTS / f"ND3C_camera_{ts}.csv"

    print("=" * 65)
    print("EXP-ND3C: GPT-4o-mini × alert_room × HITL × M1–M6")
    print("LITM + Hover-Navigate Semantic Loop Mitigations")
    print(f"  [M1] Advisory rate-limit   : gap >= {MIN_ADVISORY_GAP_TURNS} turns")
    print(f"  [M2] Memory injection      : mission state prepended every turn")
    print(f"  [M3] final_text bug fix    : preserve non-empty text across turns")
    print(f"  [M4] Loop nudge            : fires after hover>={HOVER_CALL_NUDGE_THRESHOLD}x or scene>={SCENE_CALL_NUDGE_THRESHOLD}x")
    print(f"  [M5] Turn budget warning   : countdown from turn {TURN_BUDGET_WARN_AT}/50")
    print(f"  [M6] Explicit quit rule    : EXIT RULE in every mission state")
    print(f"  [HITL] Approval gate       : auto-approve (safety-bounded operator)")
    print()
    print("Baseline (ND3B, no fixes):")
    print(f"  patrol={BASELINE_ND3B['patrol_rate']*100:.0f}%  "
          f"quality={BASELINE_ND3B['mean_quality']:.1f}/5  "
          f"words={BASELINE_ND3B['mean_words']:.0f}  "
          f"turns={BASELINE_ND3B['mean_turns']:.0f}  "
          f"cost=${BASELINE_ND3B['mean_cost']:.2f}/run  "
          f"hover_denied={BASELINE_ND3B['mean_hover_den']:.0f}/run")
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

    all_run_rows       = []
    all_api_stats_rows = []
    all_approval_rows  = []
    all_camera_rows    = []
    all_workflow_rows  = []

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
        run_id = f"ND3C_gpt4o_mini_alert_room_r{run}"
        print(f"\n{'═'*65}")
        print(f"  Run {run}/{N_RUNS}  [gpt4o_mini / alert_room / HITL+M1-M6]  id={run_id}")
        print(f"{'═'*65}")

        run_approval_rows: list = []
        approve_cb = make_auto_approve_callback(
            "gpt4o_mini", run_id, "alert_room", SCENARIO["scene_label"],
            run_approval_rows,
        )

        agent = FixedNDAgentHITL(
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
        agent.start_emergency_monitor()

        t_wall    = time.time()
        error_msg = ""
        try:
            final_text, api_stats, tool_trace = agent._run_openai_orchestrator_loop(
                orchestrator     = "gpt4o_mini",
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

        # ── API stats ─────────────────────────────────────────────────────────
        for stat in api_stats:
            all_api_stats_rows.append({
                "orchestrator":  "gpt4o_mini",
                "global_run":    run,
                "scenario":      "alert_room",
                "scene_label":   SCENARIO["scene_label"],
                "turn":          stat.get("turn"),
                "latency_s":     stat.get("latency_s"),
                "input_tokens":  stat.get("input_tokens"),
                "output_tokens": stat.get("output_tokens"),
                "cost_usd":      stat.get("cost_usd"),
                "text":          stat.get("text", ""),
            })

        # ── Camera + workflow rows ────────────────────────────────────────────
        cam_rows = list(agent._camera_rows)
        wkf_rows = list(agent._workflow_rows)
        for r in cam_rows:
            r["orchestrator"] = "gpt4o_mini"
            r["scenario"]     = "alert_room"
            r["scene_label"]  = SCENARIO["scene_label"]
        for r in wkf_rows:
            r["orchestrator"] = "gpt4o_mini"
            r["scenario"]     = "alert_room"
            r["scene_label"]  = SCENARIO["scene_label"]
        all_camera_rows.extend(cam_rows)
        all_workflow_rows.extend(wkf_rows)
        all_approval_rows.extend(run_approval_rows)

        # ── Mission metrics ───────────────────────────────────────────────────
        all_turns_text = " |TURN| ".join(
            s.get("text", "") for s in api_stats if s.get("text"))
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

        # ── Approval metrics ──────────────────────────────────────────────────
        n_proposed = len(run_approval_rows)
        n_approved = sum(r["approved"] for r in run_approval_rows)
        n_denied   = n_proposed - n_approved
        apr_rate   = round(n_approved / n_proposed, 4) if n_proposed else 1.0
        op_times   = [r["response_time_s"] for r in run_approval_rows]
        mean_op_s  = round(float(np.mean(op_times)), 2) if op_times else 0.0
        total_op_s = round(sum(op_times), 2)

        # Advisory stats from tool_trace (injected/suppressed tracked inside loop)
        n_adv_inj  = sum(1 for t in tool_trace if t.get("name") == "__advisory__")
        # (advisory counts printed to stdout; not in tool_trace — read from api_stats length proxy)

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
              f"quality={quality}/5  risk={risk}")
        print(f"    scene_calls={n_cam}  drone_cmds={n_drone}  "
              f"words={scores['word_count']}")
        print(f"    approvals={n_approved}/{n_proposed}  denied={n_denied}  "
              f"op_time={total_op_s:.1f}s")
        print(f"    pipeline={total_pipe:.0f}ms  cost=${total_cost:.5f}  wall={wall_s}s")
        if error_msg:
            print(f"    ⚠️  ERROR: {error_msg[:120]}")

        all_run_rows.append({
            # ND2-identical fields
            "orchestrator":        "gpt4o_mini",
            "global_run":          run,
            "scenario":            "alert_room",
            "scene_label":         SCENARIO["scene_label"],
            "truth":               SCENARIO["truth"],
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
            # ND3C-specific fields
            "n_drone_commands_proposed": n_proposed,
            "n_approved":                n_approved,
            "n_denied":                  n_denied,
            "approval_rate":             apr_rate,
            "mean_operator_response_s":  mean_op_s,
            "total_operator_time_s":     total_op_s,
            "n_advisories_injected":     0,   # printed to stdout; placeholder
            "n_advisories_suppressed":   0,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(sc_correct)
    if n:
        corr_rate, c_lo, c_hi = wilson_ci(sum(sc_correct), n)
        patr_rate, _, _       = wilson_ci(sum(sc_patrol),  n)
        q_mean, _, _          = bootstrap_ci(sc_qualities)
        cost_mean, _, _       = bootstrap_ci(sc_costs)
        apr_mean              = round(sum(sc_apr_rates) / n, 4)

        summary_row = {
            "orchestrator":        "gpt4o_mini",
            "scenario":            "alert_room",
            "scene_label":         SCENARIO["scene_label"],
            "truth":               SCENARIO["truth"],
            "n_runs":              n,
            "patrol_rate":         round(patr_rate, 3),
            "correct_rate":        round(corr_rate, 3),
            "correct_lo":          round(c_lo, 3),
            "correct_hi":          round(c_hi, 3),
            "mean_quality":        round(q_mean, 3),
            "mean_s3_risk":        round(sum(sc_s3s) / n, 3),
            "mean_scene_calls":    round(np.mean([r["n_scene_calls"]   for r in all_run_rows]), 2),
            "mean_drone_cmds":     round(np.mean([r["n_drone_commands"] for r in all_run_rows]), 2),
            "mean_pipeline_ms":    round(float(np.mean(sc_lats)), 1) if sc_lats else 0,
            "mean_total_cost_usd": round(cost_mean, 6),
            "mean_n_proposed":     round(sum(sc_n_proposed) / n, 2),
            "mean_approval_rate":  apr_mean,
            "mean_operator_response_s": round(sum(sc_op_times) / n, 2),
            "total_approvals":     sum(r["n_approved"] for r in all_run_rows),
            "total_denials":       sum(r["n_denied"]   for r in all_run_rows),
        }

    # ── Write CSVs ────────────────────────────────────────────────────────────
    cam_fields_ext = CAMERA_CSV_FIELDS  + ["orchestrator", "scenario", "scene_label"]
    wkf_fields_ext = WORKFLOW_CSV_FIELDS + ["orchestrator", "scenario", "scene_label"]

    if all_run_rows:
        write_csv(out_runs, all_run_rows, ND3C_RUN_FIELDS)
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
    if n:
        write_csv(out_summary, [summary_row], list(summary_row.keys()))
        print(f"Saved summary   → {out_summary}")
    if all_api_stats_rows:
        write_csv(out_api_stats, all_api_stats_rows, LLM_STATS_CSV_FIELDS)
        print(f"Saved api_stats → {out_api_stats}")

    # ── Final report ──────────────────────────────────────────────────────────
    valid = [r for r in all_run_rows if not r["error"]]
    n_tot = len(valid)
    if n_tot:
        correct_all = sum(r["correct_room_report"] for r in valid)
        acc, lo, hi = wilson_ci(correct_all, n_tot)
        tot_cost    = sum(r["total_cost_usd"] for r in valid)
        n_approved  = sum(r["n_approved"] for r in valid)
        n_proposed  = sum(r["n_drone_commands_proposed"] for r in valid)
        overall_apr = round(n_approved / n_proposed, 3) if n_proposed else 1.0

        print("\n" + "=" * 65)
        print("ND3C RESULTS  (GPT-4o-mini / alert_room / HITL+M1-M6)")
        print(f"  Valid runs               : {n_tot}/{N_RUNS}")
        print(f"  Patrol rate              : {patr_rate*100:.1f}%")
        print(f"  Correct response         : {acc*100:.1f}%  [{lo*100:.1f}–{hi*100:.1f}%]")
        print(f"  Mean quality             : {q_mean:.2f}/5")
        print(f"  Mean word count          : {np.mean([r['word_count'] for r in valid]):.0f}")
        print(f"  Overall approval rate    : {overall_apr*100:.1f}%  ({n_approved}/{n_proposed})")
        print(f"  Total cost               : ${tot_cost:.4f}  (${tot_cost/n_tot:.4f}/run)")
        print()
        print("Baseline comparison (ND3B, no fixes):")
        print(f"  patrol: {BASELINE_ND3B['patrol_rate']*100:.0f}% → {patr_rate*100:.1f}%  "
              f"quality: {BASELINE_ND3B['mean_quality']:.1f} → {q_mean:.2f}  "
              f"cost: ${BASELINE_ND3B['mean_cost']:.2f} → ${tot_cost/n_tot:.4f}/run")
        print("=" * 65)


if __name__ == "__main__":
    main()
