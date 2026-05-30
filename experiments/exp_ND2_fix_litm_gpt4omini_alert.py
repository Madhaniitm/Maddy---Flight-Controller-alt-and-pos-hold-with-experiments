"""
EXP-ND2-FIX: GPT-4o-mini × alert_room × 5 runs
Lost-in-the-Middle + Semantic Loop corrections (Findings 9 & 10)
============================================================
Baseline failure (ND2, 20260528_224548):
  GPT-4o-mini entered an infinite analyze_scene(room_event) loop on alert_room.
  Root cause: background advisory injected EVERY turn (monitor fires every 1s,
  LLM turns take 2-3s) → model attends to fresh high-salience advisory each turn
  and ignores its own prior history ("Lost in the Middle", Liu et al. 2023).

Six mitigations applied here — all from literature (see ND2 Findings 9 & 10):

  [M1] Advisory rate-limiting                (Liu et al. 2023; arxiv 2510.10276)
       Suppress advisory re-injection if fewer than MIN_ADVISORY_GAP_TURNS have
       elapsed since the last injection. Breaks the every-turn advisory cycle.

  [M2] Structured memory injection            (arxiv 2603.11513; Liu et al. 2023)
       Prepend a concise mission-state summary at the START of each turn's user
       message. "Already handled advisory at turn N" is placed in the PRIMACY
       position (highest attention in U-shaped attention curve), not the middle.

  [M3] final_text overwrite bug fix           (OpenAI loop code bug)
       Use `if turn_text: final_text = turn_text` instead of
       `final_text = turn_text` — preserves earlier report text when the model
       later emits an empty content alongside tool calls.

  [M4] Loop detection + nudging               (Saha 2025; agentpatterns.tech)
       Track analyze_scene and hover call counts. Once either crosses
       SCENE_CALL_NUDGE_THRESHOLD, inject a strong "LOOP DETECTED — write
       your report NOW" message in the mission state. Breaks the semantic loop
       by naming the behaviour and redirecting to a concrete alternative action.

  [M5] Turn budget warning                    (agentpatterns.tech; Maxim AI 2025)
       At turn TURN_BUDGET_WARN_AT (default 40/50), inject a countdown:
       "Only N turns remain — you MUST write your report in the next 2 turns."
       Creates explicit deadline awareness; small models respond to countdowns.

  [M6] Explicit quit condition in mission state  (arxiv 2510.16492 — Liu et al.)
       Always include a fixed exit rule in the mission state:
       "After ≥ N analyze_scene calls → STOP tools, write report immediately."
       Gives the model a self-monitoring rule before the loop forms.

Comparison:
  Original ND2:   patrol=0%, correct=100%, quality=0/5, words=0, scene_calls=17-23/run
  M1-M3 only:     patrol=0%, correct=0%,   quality=0/5, words=0, scene_calls=10/run (semantic loop shifted)
  Target (M1-M6): patrol>0%, correct>=80%, quality>0,   words>0, scene_calls<=6/run

Run:
    export GLOG_minloglevel=3
    /opt/homebrew/bin/python3.11 experiments/exp_ND2_fix_litm_gpt4omini_alert.py
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

ND_RESULTS = EXP_DIR / "results"
ND_RESULTS.mkdir(exist_ok=True)

# ── Mitigation parameters ─────────────────────────────────────────────────────

MIN_ADVISORY_GAP_TURNS     = 5   # [M1] minimum turns between advisory injections
                                  # Liu et al. 2023: recency bias dominates within 5 turns
                                  # → suppress re-injection to break the loop cycle

SCENE_CALL_NUDGE_THRESHOLD = 3   # [M4] loop nudge fires after this many analyze_scene calls
                                  # Saha 2025: "You already tried this, try something different"

HOVER_CALL_NUDGE_THRESHOLD = 3   # [M4] loop nudge fires after this many hover calls
                                  # agentpatterns.tech: deduplication on repeated tool+args

TURN_BUDGET_WARN_AT        = 40  # [M5] inject urgency countdown from this turn onward
                                  # agentpatterns.tech: hard limits force exit
                                  # Maxim AI 2025: deterministic stop rules

ORCHESTRATOR = "gpt4o_mini"
N_RUNS       = 5

SCENARIO = {
    "name":        "alert_room",
    "scene_label": "person_near",
    "truth":       "ALERT",
    "description": "Person close to drone. Drone should investigate and report.",
}

# ── Original ND2 baseline for comparison (from 20260528_224548) ───────────────
BASELINE = {
    "patrol_rate":  0.0,
    "correct_rate": 1.0,
    "mean_quality": 0.0,
    "mean_words":   0.0,
    "mean_scenes":  17.4,
    "mean_cost":    0.615,
}


# ── [M2] Mission state builder ─────────────────────────────────────────────────

def _build_mission_state(turn: int,
                         tool_trace: list,
                         last_advisory_turn: int,
                         advisory_count: int,
                         max_turns: int = 50) -> str:
    """
    Build a structured mission-state summary for injection at the START of
    each turn's user message.

    Mitigations embedded here:
      [M2] Primacy-position memory injection   (Liu et al. 2023; arxiv 2603.11513)
      [M4] Loop detection + nudging            (Saha 2025; agentpatterns.tech)
      [M5] Turn budget warning                 (agentpatterns.tech; Maxim AI 2025)
      [M6] Explicit quit condition             (arxiv 2510.16492)
    """
    # ── Counters from trace ────────────────────────────────────────────────────
    scene_count = sum(1 for t in tool_trace if t["name"] == "analyze_scene")
    hover_count = sum(1 for t in tool_trace if t["name"] == "hover")
    turns_left  = max_turns - turn

    # Last 4 tool calls (recency context)
    recent      = tool_trace[-4:] if len(tool_trace) >= 4 else tool_trace
    recent_str  = " → ".join(t["name"] for t in recent) if recent else "none yet"

    # Last analyze_scene call details
    last_scene_turn, last_scene_ctx = None, None
    for t in reversed(tool_trace):
        if t["name"] == "analyze_scene":
            last_scene_turn = t["turn"]
            last_scene_ctx  = t["args"].get("context", "?")
            break

    # ── [M2] Advisory history (primacy injection) ──────────────────────────────
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

    # ── [M6] Explicit quit condition (always present) ──────────────────────────
    quit_rule = (
        f"EXIT RULE: Once analyze_scene has been called ≥ {SCENE_CALL_NUDGE_THRESHOLD} times, "
        f"STOP all tool calls and write your final structured report immediately."
    )

    # ── [M4] Loop detection nudge (activates at threshold) ────────────────────
    nudge_line = ""
    if scene_count >= SCENE_CALL_NUDGE_THRESHOLD or hover_count >= HOVER_CALL_NUDGE_THRESHOLD:
        nudge_line = (
            f"\n⚠️  LOOP DETECTED: analyze_scene called {scene_count}× and "
            f"hover called {hover_count}× this run. "
            f"YOU HAVE MORE THAN ENOUGH DATA. "
            f"Do NOT call analyze_scene or hover again. "
            f"Write your final structured report RIGHT NOW, then call land()."
        )

    # ── [M5] Turn budget warning (activates near end) ─────────────────────────
    budget_line = ""
    if turn >= TURN_BUDGET_WARN_AT:
        budget_line = (
            f"\n⏰  TURN BUDGET: Only {turns_left} turns remaining. "
            f"You MUST write your final report within the next 2 turns or the mission fails."
        )

    return (
        f"[MISSION STATE — Turn {turn}/{max_turns}]\n"
        f"Recent tools: {recent_str}\n"
        f"{scene_line}\n"
        f"{adv_line}\n"
        f"{quit_rule}"
        f"{nudge_line}"
        f"{budget_line}\n"
        f"---"
    )


# ── Fixed NDAgent subclass ─────────────────────────────────────────────────────

class FixedNDAgent(NDAgent):
    """
    NDAgent with six mitigations applied to the OpenAI loop (Findings 9 & 10).

    [M1] Advisory rate-limiting    — suppress advisory if gap < MIN_ADVISORY_GAP_TURNS
    [M2] Structured memory inject  — mission state prepended every turn (primacy)
    [M3] final_text overwrite fix  — `if turn_text: final_text = turn_text`
    [M4] Loop detection + nudging  — ⚠️ LOOP DETECTED message after N repeated calls
    [M5] Turn budget warning       — ⏰ countdown injected at turn TURN_BUDGET_WARN_AT
    [M6] Explicit quit condition   — EXIT RULE always in mission state
    """

    def _run_openai_orchestrator_loop(self, orchestrator: str,
                                       user_prompt: str,
                                       max_turns: int,
                                       max_tokens: int) -> tuple:
        from verbalization_utils import (
            OPENAI_API_KEY, OPENAI_BASE_URL,
            OPENAI_MINI_KEY, OPENAI_MINI_URL,
            GPT4O_MODEL,
        )
        import openai as _openai

        # GPT-4o-mini only in this experiment
        client    = _openai.OpenAI(api_key=OPENAI_MINI_KEY,
                                   base_url=OPENAI_MINI_URL)
        oai_model = "gpt-4o-mini"
        cost_in, cost_out = 0.15e-6, 0.60e-6

        oai_tools = _nd_tools_to_openai(ND_ALL_TOOLS)

        # Initial message with camera frame
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
        api_stats      = []
        tool_trace     = []
        final_text     = ""

        # [M1] Rate-limit state
        last_advisory_turn = -MIN_ADVISORY_GAP_TURNS   # allow first injection at turn 1
        advisory_count     = 0
        suppressed_count   = 0

        for turn in range(1, max_turns + 1):

            # ── [M2] Structured memory injection ──────────────────────────────
            # Inject mission state at the START of this turn's user message block.
            # Position: immediately before the API call → primacy of recent context.
            state_msg = _build_mission_state(
                turn, tool_trace, last_advisory_turn, advisory_count, max_turns)
            messages.append({"role": "user", "content": state_msg})

            # ── [M1] Advisory rate-limiting ────────────────────────────────────
            emg_handled, emg_msg = self.handle_emergency_if_flagged()
            if emg_handled and emg_msg:
                gap = turn - last_advisory_turn
                if gap >= MIN_ADVISORY_GAP_TURNS:
                    # Inject advisory — gap is large enough
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
                    print(f"  📡 [ADVISORY INJECTED]   turn={turn}  "
                          f"total_injections={advisory_count}")
                else:
                    # Suppress — too soon since last injection
                    suppressed_count += 1
                    print(f"  🔕 [ADVISORY SUPPRESSED] turn={turn}  "
                          f"gap={gap} < {MIN_ADVISORY_GAP_TURNS}  "
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
                break

            latency = time.time() - t0
            usage   = resp.usage
            in_tok  = usage.prompt_tokens
            out_tok = usage.completion_tokens
            msg     = resp.choices[0].message
            finish  = resp.choices[0].finish_reason

            turn_text = msg.content or ""

            # ── [M3] final_text overwrite bug fix ──────────────────────────────
            # Original: `final_text = turn_text` — overwrites with "" on tool turns
            # Fixed:    preserve last non-empty text so report isn't lost
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
                      f"text_words={len(turn_text.split())}")
                break

            # Execute tool calls
            for tc in msg.tool_calls:
                t_name = tc.function.name
                try:
                    t_args = json.loads(tc.function.arguments)
                except Exception:
                    t_args = {}
                print(f"  [TOOL OAI t{turn}] {t_name}({json.dumps(t_args)[:60]})")
                result = self.execute_tool(t_name, t_args)
                tool_trace.append({
                    "turn":       turn,
                    "name":       t_name,
                    "args":       t_args,
                    "result":     result,
                    "approved":   1,
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

        # ── Advisory suppression report ────────────────────────────────────────
        print(f"\n  Advisory stats: injected={advisory_count}  "
              f"suppressed={suppressed_count}  "
              f"last_at_turn={last_advisory_turn}")

        return final_text, api_stats, tool_trace


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_runs    = ND_RESULTS / f"ND2_fix_litm_gpt4omini_alert_{ts}.csv"
    out_summary = ND_RESULTS / f"ND2_fix_litm_gpt4omini_alert_summary_{ts}.csv"
    out_stats   = ND_RESULTS / f"ND2_fix_litm_gpt4omini_alert_apistats_{ts}.csv"

    print("=" * 65)
    print("EXP-ND2-FIX: GPT-4o-mini × alert_room × 5 runs")
    print("Lost-in-the-Middle + Semantic Loop mitigations (Findings 9 & 10)")
    print(f"  [M1] Advisory rate-limit   : gap >= {MIN_ADVISORY_GAP_TURNS} turns")
    print(f"  [M2] Memory injection      : mission state prepended every turn")
    print(f"  [M3] final_text bug fix    : preserve non-empty text across turns")
    print(f"  [M4] Loop detection nudge  : fire after analyze_scene >= {SCENE_CALL_NUDGE_THRESHOLD}x or hover >= {HOVER_CALL_NUDGE_THRESHOLD}x")
    print(f"  [M5] Turn budget warning   : countdown injected from turn {TURN_BUDGET_WARN_AT}/50")
    print(f"  [M6] Explicit quit rule    : EXIT RULE in every mission state")
    print()
    print("Baseline (original ND2):")
    print(f"  patrol={BASELINE['patrol_rate']*100:.0f}%  "
          f"correct={BASELINE['correct_rate']*100:.0f}%  "
          f"quality={BASELINE['mean_quality']:.1f}/5  "
          f"words={BASELINE['mean_words']:.0f}  "
          f"scene_calls={BASELINE['mean_scenes']:.1f}/run")
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
        print(f"\n{'═'*65}")
        print(f"  Run {run}/{N_RUNS}  [gpt4o_mini / alert_room / LITM-fixed]")
        print(f"{'═'*65}")

        agent = FixedNDAgent(
            session_id   = f"ND2_fix_litm_r{run}_{ts}",
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

        # Per-turn LLM stats
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

        # Count advisory events from tool trace
        tool_names_in_trace = [t["name"] for t in tool_trace]
        analyze_room_event  = tool_names_in_trace.count("analyze_scene")
        hover_count         = tool_names_in_trace.count("hover")
        used_land           = "land" in tool_names_in_trace

        sc_correct.append(int(correct_resp))
        sc_patrol.append(int(patrol_done))
        sc_qualities.append(quality)
        sc_s3s.append(scores["s3_risk"])
        sc_costs.append(total_cost)
        sc_lats.append(mean_pipe)

        print(f"\n  Results:")
        print(f"  patrol={patrol_done}  correct={correct_resp}  "
              f"quality={quality}/5  risk={risk}")
        print(f"  scene_calls={n_cam}  hover_count={hover_count}  "
              f"used_land={used_land}  drone_cmds={n_drone}")
        print(f"  word_count={scores['word_count']}  truncated={trunc}")
        print(f"  pipeline={total_pipe:.0f}ms  cost=${total_cost:.5f}  wall={wall_s}s")
        print(f"\n  vs baseline:")
        print(f"  scene_calls: {n_cam} (baseline: ~{BASELINE['mean_scenes']:.0f})")
        print(f"  words:       {scores['word_count']} (baseline: 0)")
        print(f"  quality:     {quality}/5 (baseline: 0/5)")
        print(f"\n  Final text ({scores['word_count']} words):")
        preview = final_text[:400] if final_text else "[EMPTY — loop did not produce report]"
        print(f"  {preview!r}")

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

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(sc_correct)
    corr_rate, c_lo, c_hi = wilson_ci(sum(sc_correct), n)
    patr_rate, _, _       = wilson_ci(sum(sc_patrol), n)
    q_mean, _, _          = bootstrap_ci(sc_qualities)
    s3_mean               = round(sum(sc_s3s) / n, 3)
    cost_mean, _, _       = bootstrap_ci(sc_costs)
    lat_mean, _, _        = bootstrap_ci(sc_lats)
    mean_words            = np.mean([r["word_count"] for r in all_run_rows])
    mean_scenes           = np.mean([r["n_scene_calls"] for r in all_run_rows])

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
        "mean_scene_calls":    round(mean_scenes, 2),
        "mean_drone_cmds":     round(np.mean([r["n_drone_commands"] for r in all_run_rows]), 2),
        "mean_pipeline_ms":    round(lat_mean, 1),
        "mean_total_cost_usd": round(cost_mean, 6),
    }

    write_csv(out_runs, all_run_rows, RUN_FIELDS)
    write_csv(out_summary, [summary_row], SUMMARY_FIELDS)
    if all_api_stats_rows:
        write_csv(out_stats, all_api_stats_rows, LLM_STATS_CSV_FIELDS)

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FIXED vs BASELINE — gpt4o_mini / alert_room")
    print("=" * 65)
    print(f"{'Metric':<22} {'Baseline (ND2)':>16} {'Fixed (LITM)':>14} {'Δ':>6}")
    print("─" * 60)

    def _delta(new, old, fmt=".1f", pct=False):
        d = new - old
        suffix = "%" if pct else ""
        sign   = "+" if d > 0 else ""
        return f"{sign}{d:{fmt}}{suffix}"

    print(f"  {'patrol_rate':<20} {BASELINE['patrol_rate']*100:>15.0f}%"
          f"  {patr_rate*100:>12.0f}%"
          f"  {_delta(patr_rate*100, BASELINE['patrol_rate']*100, '.0f', True):>6}")
    print(f"  {'correct_rate':<20} {BASELINE['correct_rate']*100:>15.0f}%"
          f"  {corr_rate*100:>12.0f}%"
          f"  {_delta(corr_rate*100, BASELINE['correct_rate']*100, '.0f', True):>6}")
    print(f"  {'quality /5':<20} {BASELINE['mean_quality']:>15.1f}"
          f"  {q_mean:>12.1f}"
          f"  {_delta(q_mean, BASELINE['mean_quality']):>6}")
    print(f"  {'mean words':<20} {BASELINE['mean_words']:>15.0f}"
          f"  {mean_words:>12.0f}"
          f"  {_delta(mean_words, BASELINE['mean_words'], '.0f'):>6}")
    print(f"  {'mean scene_calls':<20} {BASELINE['mean_scenes']:>15.1f}"
          f"  {mean_scenes:>12.1f}"
          f"  {_delta(mean_scenes, BASELINE['mean_scenes']):>6}")
    print(f"  {'cost/run ($)':<20} {BASELINE['mean_cost']:>15.3f}"
          f"  {cost_mean:>12.3f}"
          f"  {_delta(cost_mean, BASELINE['mean_cost']):>6}")
    print("─" * 60)
    print(f"\n  Mitigations applied:")
    print(f"    [M1] Advisory gap : >= {MIN_ADVISORY_GAP_TURNS} turns between injections")
    print(f"    [M2] Memory inject: mission state prepended every turn (primacy position)")
    print(f"    [M3] Bug fix      : final_text preserved across tool-call turns")
    print()
    print(f"  Saved runs    → {out_runs}")
    print(f"  Saved summary → {out_summary}")
    print("=" * 65)

    # ── Per-run loop analysis ─────────────────────────────────────────────────
    print("\nPer-run analysis:")
    print(f"  {'Run':<5} {'Patrol':>7} {'Correct':>8} {'Quality':>8} "
          f"{'Words':>6} {'Scenes':>7} {'Land?':>6}")
    print("  " + "─" * 50)
    for row in all_run_rows:
        land_yn = "yes" if row["patrol_complete"] else "no"
        print(f"  {row['global_run']:<5} "
              f"{'✓' if row['patrol_complete'] else '✗':>7} "
              f"{'✓' if row['correct_room_report'] else '✗':>8} "
              f"{row['quality_score']:>8} "
              f"{row['word_count']:>6} "
              f"{row['n_scene_calls']:>7} "
              f"{land_yn:>6}")
    print("=" * 65)


if __name__ == "__main__":
    main()
