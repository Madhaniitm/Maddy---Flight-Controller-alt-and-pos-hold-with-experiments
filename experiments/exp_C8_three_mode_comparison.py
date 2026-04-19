"""
EXP-C8 v2: Three-Mode Comparison — Scripted vs NL-Commanded vs Full-Auto
=========================================================================
Mission: takeoff → waypoints [0.8m, 1.2m, 1.5m, 1.0m] → land
         Hold each waypoint for 8 s. RMSE window starts only after the drone
         arrives within ARRIVAL_TOL (0.15 m) of each target — same rule for
         all three modes so the comparison is fair.

Mode A — Scripted baseline (no LLM, deterministic, 1 run):
         Uses althold PID. Loops through waypoints, waits for arrival,
         measures RMSE, then moves on. No fixed-timer assumptions.

Mode B — NL-commanded (human issues each waypoint as a separate turn):
         5 conversational turns, one per mission phase. LLM executes each.
         N = 5 independent runs.

Mode C — Full-auto (single command, LLM plans + executes entire mission):
         No further human input after the initial command.
         N = 5 independent runs.

Outputs:
  results/C8_runs_guardrail_on.csv
  results/C8_summary_guardrail_on.csv
  results/C8_three_mode_comparison_guardrail_on.png
"""

import sys, os, csv, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import PhysicsLoop, DroneState
from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--guardrail", choices=["on", "off"], default="on")
_args, _ = _parser.parse_known_args()
GUARDRAIL_ENABLED = _args.guardrail == "on"
GUARDRAIL_SUFFIX  = "guardrail_on" if GUARDRAIL_ENABLED else "guardrail_off"

OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", f"C8_runs_{GUARDRAIL_SUFFIX}.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", f"C8_summary_{GUARDRAIL_SUFFIX}.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", f"C8_three_mode_comparison_{GUARDRAIL_SUFFIX}.png")

# ── Mission definition ─────────────────────────────────────────────────────────
WAYPOINTS    = [0.8, 1.2, 1.5, 1.0]   # metres, in order
HOLD_TIME    = 8.0                      # seconds per waypoint hold measurement
ARRIVAL_TOL  = 0.15                     # metres — start measuring once within this
ARRIVAL_WAIT = 25.0                     # seconds max to wait for arrival before timeout
PASS_RMSE_CM = 15.0                     # overall RMSE pass threshold (cm)
N_RUNS       = 5
SIM_HZ       = 200
DT           = 1.0 / SIM_HZ

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Modes B and C use the ReAct loop; Mode A is the non-LLM scripted baseline."
    ),
    "Vemprala2023": (
        "Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). "
        "ChatGPT for Robotics: Design Principles and Model Abilities. MSR-TR-2023-8. arXiv:2306.17582. "
        "Three-mode comparison follows Vemprala's manual vs LLM evaluation protocol."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "Full-auto Mode C relies on inner monologue tool feedback with no human in the loop."
    ),
}

# ── Statistics helpers ─────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n_boot)]
    return float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    m = z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))/d
    return max(0.0, c-m), min(1.0, c+m)

# ── RMSE extraction: same function for all modes ───────────────────────────────

def extract_waypoint_rmse(tel_buf, waypoints, hold_time=HOLD_TIME,
                          arrival_tol=ARRIVAL_TOL):
    """
    For each waypoint target (in sequence), scan telemetry for the first
    sample within arrival_tol, then collect hold_time seconds of samples,
    compute RMSE. Used for Mode A (scripted) only — Mode A measures hold
    samples immediately after its own arrival check, so this is correct.

    tel_buf entries must have 'lw_z' in mm.
    """
    rmses       = []
    search_from = 0
    hold_n      = int(hold_time * 10)

    for target in waypoints:
        arrival_idx = None
        for i in range(search_from, len(tel_buf)):
            lw_z = tel_buf[i].get("lw_z")
            if lw_z is not None and abs(lw_z / 1000.0 - target) < arrival_tol:
                arrival_idx = i
                break

        if arrival_idx is None:
            rmses.append(float("nan"))
            search_from = len(tel_buf)
            continue

        end_idx   = min(arrival_idx + hold_n, len(tel_buf))
        hold_samp = [tel_buf[j]["lw_z"]/1000.0
                     for j in range(arrival_idx, end_idx)
                     if tel_buf[j].get("lw_z") is not None]

        if len(hold_samp) < 5:
            rmses.append(float("nan"))
        else:
            rmses.append(math.sqrt(sum((z-target)**2 for z in hold_samp) / len(hold_samp)))

        search_from = arrival_idx + hold_n

    return rmses


def overall_rmse_from_wp(wp_rmses):
    """RMS of valid per-waypoint RMSEs (nan-safe)."""
    valid = [r for r in wp_rmses if not math.isnan(r)]
    if not valid:
        return float("nan")
    return math.sqrt(sum(r**2 for r in valid) / len(valid))


def extract_rmse_from_confirmed(agent, waypoints, hold_time=HOLD_TIME, debug=False):
    """
    Measure per-waypoint RMSE using agent.wp_confirmed_tel_idx — the telemetry
    index stamped by check_altitude_reached() the moment it returns ✓.

    Measurement window is the hold_time seconds BEFORE the confirmation index.
    This works because the prompts call wait(8.0) first, then check_altitude_reached()
    to confirm — so the hold period always precedes the confirmation stamp.
    Window: [conf_idx - hold_n, conf_idx]
    """
    tel_buf  = agent.tel_buf
    hold_n   = int(hold_time * 10)
    conf_idx = agent.wp_confirmed_tel_idx
    rmses    = []

    for wp_i, target in enumerate(waypoints):
        if wp_i >= len(conf_idx):
            if debug:
                print(f"  [RMSE DEBUG] WP{wp_i+1} target={target}m  no confirmation recorded")
            rmses.append(float("nan"))
            continue

        end   = conf_idx[wp_i]
        start = max(0, end - hold_n)

        if debug:
            z_at = round(tel_buf[end-1]["lw_z"]/1000.0, 3) if end > 0 and end <= len(tel_buf) else None
            print(f"  [RMSE DEBUG] WP{wp_i+1} target={target}m  "
                  f"confirmed_at_idx={end}  z_before_confirm={z_at}m  "
                  f"window=[{start},{end}]  n_samp={end-start}  tel_len={len(tel_buf)}")

        hold_samp = [tel_buf[j]["lw_z"]/1000.0
                     for j in range(start, end) if tel_buf[j].get("lw_z") is not None]

        if len(hold_samp) < 5:
            if debug:
                print(f"  [RMSE DEBUG] WP{wp_i+1} — insufficient samples ({len(hold_samp)}), returning NaN")
            rmses.append(float("nan"))
        else:
            rmse = math.sqrt(sum((z - target)**2 for z in hold_samp) / len(hold_samp))
            rmses.append(rmse)

    return rmses


# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — Scripted baseline (no LLM, 1 deterministic run)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[C8] ── Mode A: Scripted baseline (1 run, no LLM) ─────────────")

state_A   = DroneState()
physics_A = PhysicsLoop(state_A)
tel_A     = []   # same format as agent.tel_buf
t_A       = 0.0
_last_tel_t = -1.0

def tick_A(n=1):
    global t_A, _last_tel_t
    for _ in range(n):
        physics_A.tick()
        t_A += DT
        # Record at ~10 Hz
        if t_A - _last_tel_t >= 0.099:
            _last_tel_t = t_A
            with state_A.lock:
                tel_A.append({
                    "t":       round(t_A, 3),
                    "lw_z":    round(state_A.ekf_z * 1000.0, 1),   # mm
                    "vz":      round(state_A.vz, 4),
                    "althold": int(state_A.althold),
                    "alt_sp":  round(state_A.alt_sp, 3),
                })

# ── Arm and find hover throttle ────────────────────────────────────────────────
with state_A.lock:
    state_A.armed = True
    state_A.ch5   = 1000

# Ramp up slowly
for pwm in range(1000, 1560, 5):
    with state_A.lock:
        state_A.ch1 = pwm
    tick_A(4)

with state_A.lock:
    state_A.ch1 = 1560
tick_A(int(8 * SIM_HZ))

# Hunt for hover (vz → 0)
pwm_now = 1550
with state_A.lock:
    state_A.ch1 = pwm_now
for _ in range(200):
    tick_A(int(0.2 * SIM_HZ))
    with state_A.lock:
        vz_now = state_A.vz
    if abs(vz_now) < 0.010:
        break
    pwm_now += 1 if vz_now < 0 else -1
    pwm_now = max(1400, min(1700, pwm_now))
    with state_A.lock:
        state_A.ch1 = pwm_now

HOVER_PWM_A = pwm_now
HOVER_THR_A = (HOVER_PWM_A - 1000) / 1000.0
print(f"  Hover found: {HOVER_PWM_A} PWM  thr={HOVER_THR_A:.3f}")

# Enable althold
with state_A.lock:
    state_A.althold          = True
    state_A.alt_sp           = WAYPOINTS[0]
    state_A.alt_sp_mm        = WAYPOINTS[0] * 1000.0
    state_A.hover_thr_locked = HOVER_THR_A
physics_A.pid_alt_pos.reset()
physics_A.pid_alt_vel.reset()

# ── Visit each waypoint ────────────────────────────────────────────────────────
wp_rmses_A    = []
wp_arrived_A  = []

for wp_idx, target in enumerate(WAYPOINTS):
    with state_A.lock:
        state_A.alt_sp    = target
        state_A.alt_sp_mm = target * 1000.0

    # Wait for arrival (ARRIVAL_TOL) — same rule as Modes B/C
    arrived = False
    for _ in range(int(ARRIVAL_WAIT * SIM_HZ)):
        tick_A(1)
        with state_A.lock:
            z_now = state_A.ekf_z
        if abs(z_now - target) < ARRIVAL_TOL:
            arrived = True
            break

    wp_arrived_A.append(arrived)
    if not arrived:
        print(f"  WP{wp_idx+1} ({target}m): TIMEOUT — not reached")
        wp_rmses_A.append(float("nan"))
        # still advance hold timer so telemetry exists
        tick_A(int(HOLD_TIME * SIM_HZ))
        continue

    # Measure RMSE during HOLD_TIME seconds
    hold_samples = []
    for _ in range(int(HOLD_TIME * SIM_HZ)):
        tick_A(1)
        with state_A.lock:
            hold_samples.append(state_A.ekf_z)

    rmse_wp = math.sqrt(sum((z - target)**2 for z in hold_samples) / len(hold_samples))
    wp_rmses_A.append(rmse_wp)
    print(f"  WP{wp_idx+1} ({target}m): arrived={arrived}  RMSE={rmse_wp*100:.3f}cm")

# ── Land scripted ──────────────────────────────────────────────────────────────
with state_A.lock:
    state_A.althold = False
    state_A.ch2 = 1500; state_A.ch3 = 1500
for pwm in [1400, 1300, 1200, 1100, 1000]:
    with state_A.lock:
        state_A.ch1 = pwm
    tick_A(int(0.5 * SIM_HZ))
# Poll until landed
for _ in range(int(10 * SIM_HZ)):
    tick_A(1)
    with state_A.lock:
        if state_A.ekf_z < 0.05:
            break
with state_A.lock:
    state_A.ch5   = 2000
    state_A.armed = False

rmse_A_overall = overall_rmse_from_wp(wp_rmses_A)
print(f"  Mode A overall RMSE: {rmse_A_overall*100:.3f}cm  mission_time={t_A:.1f}s")
print(f"  Mode A per-WP RMSE (cm): {[round(r*100,3) if not math.isnan(r) else 'NaN' for r in wp_rmses_A]}")

# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — NL-commanded (multi-turn, one turn per waypoint)
# ══════════════════════════════════════════════════════════════════════════════

# Redesigned Mode B: human-in-loop as a supervisor, not a step-by-step commander.
#
# Turn 1: LLM does full setup (arm/hover/enable_althold) and flies WP1 autonomously,
#         then pauses and reports status.
# Turns 2-4: script injects real-time simulator state at the top of the human message
#            so the LLM knows althold is already ACTIVE and must not re-enable it.
#            Human message is a deviation-capable approval: "proceed / modify / abort".
# Turn 5: land.
#
# State context is injected by get_state_context(agent) just before each approval turn.

MODE_B_TURNS = [
    # Turn 1 — full setup + WP1 (LLM establishes althold once)
    ("Complete mission setup, then fly to the first waypoint. "
     "Execute these steps IN ORDER, calling each as a tool:\n"
     "1. arm()\n"
     "2. find_hover_throttle()\n"
     "3. check_drone_stable()\n"
     "4. enable_altitude_hold()  ← do this ONCE; it stays active for the whole mission\n"
     "5. set_altitude_target(0.8)\n"
     "6. wait(4.0)  ← wait for the drone to climb to 0.8 m\n"
     "7. wait(8.0)  ← hold at 0.8 m for exactly 8 seconds; do NOT skip\n"
     "8. check_altitude_reached(0.8, 0.10)  ← confirm altitude after the hold\n"
     "9. get_sensor_status()  ← report final altitude and status\n"
     "Do not end this turn before step 8 is complete."),

    # Turns 2-4: human approval — state context will be prepended by run_mode_B()
    ("Approved. Proceed to 1.2 m. "
     "Use only set_altitude_target(1.2) — altitude hold is already active. "
     "Then wait(4.0) for the drone to reach 1.2 m, "
     "then wait(8.0) to hold for 8 seconds, "
     "then check_altitude_reached(1.2, 0.10) to confirm. Report status."),

    ("Approved. Proceed to 1.5 m. "
     "Use only set_altitude_target(1.5) — altitude hold is already active. "
     "Then wait(4.0) for the drone to reach 1.5 m, "
     "then wait(8.0) to hold for 8 seconds, "
     "then check_altitude_reached(1.5, 0.10) to confirm. Report status."),

    ("Approved. Descend to 1.0 m. "
     "Use only set_altitude_target(1.0) — altitude hold is already active. "
     "Then wait(4.0) for the drone to reach 1.0 m, "
     "then wait(8.0) to hold for 8 seconds, "
     "then check_altitude_reached(1.0, 0.10) to confirm. Report status."),

    # Turn 5 — land
    ("Mission complete. Land the drone and disarm safely."),
]


def get_state_context(agent):
    """Read real simulator state and return a context string to prepend to human turns 2+."""
    with agent.state.lock:
        z       = round(agent.state.z, 3)
        althold = agent.state.althold
        armed   = agent.state.armed
        alt_sp  = round(agent.state.alt_sp, 3)
    althold_str = ("ACTIVE — do NOT call enable_altitude_hold() again"
                   if althold else "INACTIVE")
    return (
        f"[CURRENT FLIGHT STATE — verified from sensors]\n"
        f"  armed          : {armed}\n"
        f"  altitude_hold  : {althold_str}\n"
        f"  current_altitude: {z} m\n"
        f"  current_target : {alt_sp} m\n"
        f"Continue from this state. Do not re-arm, do not re-enable altitude hold.\n\n"
    )

MODE_C_COMMAND = (
    "Autonomous survey mission — complete entirely without further human input:\n"
    "1. Arm, find hover throttle, check stability.\n"
    "2. Enable altitude hold, set target to 0.8 m. wait(4.0) to climb, "
    "then wait(8.0) to hold for 8 seconds, then check_altitude_reached(0.8, 0.10) to confirm.\n"
    "3. set_altitude_target(1.2). wait(4.0) to transition, "
    "then wait(8.0) to hold, then check_altitude_reached(1.2, 0.10).\n"
    "4. set_altitude_target(1.5). wait(4.0) to climb, "
    "then wait(8.0) to hold, then check_altitude_reached(1.5, 0.10).\n"
    "5. set_altitude_target(1.0). wait(4.0) to descend, "
    "then wait(8.0) to hold, then check_altitude_reached(1.0, 0.10).\n"
    "6. Land and disarm safely.\n"
    "The order is always: set_target → wait(4.0) → wait(8.0) → check_altitude_reached. "
    "The hold (wait 8s) must come BEFORE the check call."
)


def run_mode_B(run_idx):
    print(f"\n[C8] ── Mode B Run {run_idx+1}/{N_RUNS} ─────────────────────────")
    agent = SimAgent(session_id=f"C8_B_run{run_idx}", guardrail_enabled=GUARDRAIL_ENABLED)
    t_wall_start = time.time()
    history      = []
    all_api      = []
    all_trace    = []

    for turn_i, cmd in enumerate(MODE_B_TURNS):
        # Turns 2+ (approval turns): prepend real-time simulator state so the LLM
        # knows althold is already active and must not re-initialise it.
        if turn_i > 0:
            full_cmd = get_state_context(agent) + cmd
        else:
            full_cmd = cmd
        print(f"  [Turn {turn_i+1}] {cmd[:60]}…")
        text, api_stats, trace, history = agent.run_agent_loop(
            full_cmd, history=list(history), max_turns=30
        )
        all_api.extend(api_stats)
        all_trace.extend(trace)

    t_wall = time.time() - t_wall_start

    wp_rmses = extract_rmse_from_confirmed(agent, WAYPOINTS, debug=(run_idx == 0))
    rmse_overall = overall_rmse_from_wp(wp_rmses)

    with agent.state.lock:
        landed   = agent.state.z < 0.15
        disarmed = not agent.state.armed

    n_api  = len(all_api)
    in_tok = sum(s["input_tokens"]  for s in all_api)
    out_tok= sum(s["output_tokens"] for s in all_api)
    cost   = sum(s["cost_usd"]      for s in all_api)

    rmse_cm = rmse_overall * 100 if not math.isnan(rmse_overall) else float("nan")
    n_arrived = sum(1 for r in wp_rmses if not math.isnan(r))
    passed = (not math.isnan(rmse_cm) and rmse_cm <= PASS_RMSE_CM
              and n_arrived == len(WAYPOINTS) and disarmed)

    wp_cm = [round(r*100, 3) if not math.isnan(r) else float("nan") for r in wp_rmses]
    print(f"  RMSE={rmse_cm:.2f}cm  wp={wp_cm}  api={n_api}  pass={passed}")

    return {
        "mode":          "B",
        "run":           run_idx + 1,
        "rmse_cm":       round(rmse_cm, 3),
        "wp1_rmse_cm":   wp_cm[0], "wp2_rmse_cm": wp_cm[1],
        "wp3_rmse_cm":   wp_cm[2], "wp4_rmse_cm": wp_cm[3],
        "n_wp_reached":  n_arrived,
        "mission_time_s":round(agent.sim_time, 1),
        "landed":        int(landed),
        "disarmed":      int(disarmed),
        "passed":        int(passed),
        "api_calls":     n_api,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      round(cost, 6),
        "wall_time_s":   round(t_wall, 1),
    }


def run_mode_C(run_idx):
    print(f"\n[C8] ── Mode C Run {run_idx+1}/{N_RUNS} ─────────────────────────")
    agent = SimAgent(session_id=f"C8_C_run{run_idx}", guardrail_enabled=GUARDRAIL_ENABLED)
    t_wall_start = time.time()

    text, api_stats, trace, _ = agent.run_agent_loop(MODE_C_COMMAND, max_turns=40)
    t_wall = time.time() - t_wall_start

    wp_rmses = extract_rmse_from_confirmed(agent, WAYPOINTS, debug=(run_idx == 0))
    rmse_overall = overall_rmse_from_wp(wp_rmses)

    with agent.state.lock:
        landed   = agent.state.z < 0.15
        disarmed = not agent.state.armed

    n_api  = len(api_stats)
    in_tok = sum(s["input_tokens"]  for s in api_stats)
    out_tok= sum(s["output_tokens"] for s in api_stats)
    cost   = sum(s["cost_usd"]      for s in api_stats)

    rmse_cm = rmse_overall * 100 if not math.isnan(rmse_overall) else float("nan")
    n_arrived = sum(1 for r in wp_rmses if not math.isnan(r))
    passed = (not math.isnan(rmse_cm) and rmse_cm <= PASS_RMSE_CM
              and n_arrived == len(WAYPOINTS) and disarmed)

    wp_cm = [round(r*100, 3) if not math.isnan(r) else float("nan") for r in wp_rmses]
    print(f"  RMSE={rmse_cm:.2f}cm  wp={wp_cm}  api={n_api}  pass={passed}")

    return {
        "mode":          "C",
        "run":           run_idx + 1,
        "rmse_cm":       round(rmse_cm, 3),
        "wp1_rmse_cm":   wp_cm[0], "wp2_rmse_cm": wp_cm[1],
        "wp3_rmse_cm":   wp_cm[2], "wp4_rmse_cm": wp_cm[3],
        "n_wp_reached":  n_arrived,
        "mission_time_s":round(agent.sim_time, 1),
        "landed":        int(landed),
        "disarmed":      int(disarmed),
        "passed":        int(passed),
        "api_calls":     n_api,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      round(cost, 6),
        "wall_time_s":   round(t_wall, 1),
    }


# ── Run modes B and C ──────────────────────────────────────────────────────────
print("\n[C8] Running Mode B (NL-commanded, multi-turn) × 5 …")
B_results = [run_mode_B(i) for i in range(N_RUNS)]

print("\n[C8] Running Mode C (Full-auto, single command) × 5 …")
C_results = [run_mode_C(i) for i in range(N_RUNS)]

# ── Aggregate ──────────────────────────────────────────────────────────────────

def agg(results, key):
    return [r[key] for r in results if not math.isnan(r.get(key, float("nan")))]

B_rmse = agg(B_results, "rmse_cm");  C_rmse = agg(C_results, "rmse_cm")
B_time = agg(B_results, "mission_time_s"); C_time = agg(C_results, "mission_time_s")
B_api  = agg(B_results, "api_calls");      C_api  = agg(C_results, "api_calls")
B_cost = agg(B_results, "cost_usd");       C_cost = agg(C_results, "cost_usd")

B_rmse_ci = bootstrap_ci(B_rmse);  C_rmse_ci = bootstrap_ci(C_rmse)
B_time_ci = bootstrap_ci(B_time);  C_time_ci = bootstrap_ci(C_time)

n_B_pass = sum(r["passed"] for r in B_results)
n_C_pass = sum(r["passed"] for r in C_results)
B_pass_lo, B_pass_hi = wilson_ci(n_B_pass, N_RUNS)
C_pass_lo, C_pass_hi = wilson_ci(n_C_pass, N_RUNS)

print(f"\n[C8] ── AGGREGATE ({N_RUNS} runs) ──────────────────────────────────")
print(f"  Mode A (scripted, 1 run): overall RMSE={rmse_A_overall*100:.3f}cm")
print(f"  Mode A per-WP:  {[round(r*100,3) if not math.isnan(r) else 'NaN' for r in wp_rmses_A]} cm")
print(f"  Mode B: RMSE={np.mean(B_rmse):.3f}±{np.std(B_rmse):.3f}cm  "
      f"CI=[{B_rmse_ci[0]:.3f},{B_rmse_ci[1]:.3f}]  pass={n_B_pass}/{N_RUNS}")
print(f"  Mode C: RMSE={np.mean(C_rmse):.3f}±{np.std(C_rmse):.3f}cm  "
      f"CI=[{C_rmse_ci[0]:.3f},{C_rmse_ci[1]:.3f}]  pass={n_C_pass}/{N_RUNS}")
print(f"  Mode B API: {np.mean(B_api):.1f}±{np.std(B_api):.1f}")
print(f"  Mode C API: {np.mean(C_api):.1f}±{np.std(C_api):.1f}")
print(f"  A vs B ratio: {rmse_A_overall*100/np.mean(B_rmse):.1f}x")
print(f"  A vs C ratio: {rmse_A_overall*100/np.mean(C_rmse):.1f}x")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
csv_keys = [k for k in B_results[0].keys()]
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_keys)
    w.writeheader()
    for r in B_results + C_results:
        w.writerow(r)
print(f"[C8] Per-run CSV: {OUT_RUNS}")

# Mode A per-WP summary
mode_A_wp_summary = {
    f"mode_A_wp{i+1}_rmse_cm": round(wp_rmses_A[i]*100, 3) if not math.isnan(wp_rmses_A[i]) else "NaN"
    for i in range(len(WAYPOINTS))
}

summary_rows = [
    ("mission",                  f"waypoints={WAYPOINTS} hold={HOLD_TIME}s arrival_tol={ARRIVAL_TOL}m"),
    ("n_runs",                   N_RUNS),
    ("mode_A_overall_rmse_cm",   round(rmse_A_overall*100, 3)),
    *[(f"mode_A_wp{i+1}_rmse_cm", round(wp_rmses_A[i]*100,3) if not math.isnan(wp_rmses_A[i]) else "NaN")
      for i in range(len(WAYPOINTS))],
    ("mode_A_note",              "deterministic scripted, 1 run, arrival-triggered RMSE window"),
    ("mode_B_rmse_mean_cm",      round(float(np.mean(B_rmse)), 3)),
    ("mode_B_rmse_std_cm",       round(float(np.std(B_rmse)), 3)),
    ("mode_B_rmse_ci_lo",        round(B_rmse_ci[0], 3)),
    ("mode_B_rmse_ci_hi",        round(B_rmse_ci[1], 3)),
    ("mode_B_pass_rate",         round(n_B_pass/N_RUNS, 3)),
    ("mode_B_pass_ci_lo",        round(B_pass_lo, 3)),
    ("mode_B_pass_ci_hi",        round(B_pass_hi, 3)),
    ("mode_B_api_mean",          round(float(np.mean(B_api)), 1)),
    ("mode_B_api_std",           round(float(np.std(B_api)), 1)),
    ("mode_B_time_mean_s",       round(float(np.mean(B_time)), 1)),
    ("mode_B_cost_mean_usd",     round(float(np.mean(B_cost)), 4)),
    ("mode_C_rmse_mean_cm",      round(float(np.mean(C_rmse)), 3)),
    ("mode_C_rmse_std_cm",       round(float(np.std(C_rmse)), 3)),
    ("mode_C_rmse_ci_lo",        round(C_rmse_ci[0], 3)),
    ("mode_C_rmse_ci_hi",        round(C_rmse_ci[1], 3)),
    ("mode_C_pass_rate",         round(n_C_pass/N_RUNS, 3)),
    ("mode_C_pass_ci_lo",        round(C_pass_lo, 3)),
    ("mode_C_pass_ci_hi",        round(C_pass_hi, 3)),
    ("mode_C_api_mean",          round(float(np.mean(C_api)), 1)),
    ("mode_C_api_std",           round(float(np.std(C_api)), 1)),
    ("mode_C_time_mean_s",       round(float(np.mean(C_time)), 1)),
    ("mode_C_cost_mean_usd",     round(float(np.mean(C_cost)), 4)),
    ("ratio_A_vs_B",             round(rmse_A_overall*100/np.mean(B_rmse), 1) if B_rmse else "NaN"),
    ("ratio_A_vs_C",             round(rmse_A_overall*100/np.mean(C_rmse), 1) if C_rmse else "NaN"),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow([f"ref_{ref_key}", ref_val])
print(f"[C8] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(3, 4, hspace=0.50, wspace=0.38)

ax_B   = fig.add_subplot(gs[0, 0:2])
ax_C   = fig.add_subplot(gs[0, 2:4])
ax_wp  = fig.add_subplot(gs[1, 0:2])
ax_cmp = fig.add_subplot(gs[1, 2])
ax_api = fig.add_subplot(gs[1, 3])
ax_tme = fig.add_subplot(gs[2, 0])
ax_pas = fig.add_subplot(gs[2, 1])
ax_cst = fig.add_subplot(gs[2, 2])
ax_txt = fig.add_subplot(gs[2, 3])

RUN_COLS   = ["#3498db","#e67e22","#9b59b6","#1abc9c","#e74c3c"]
MODE_COLS  = ["#95a5a6", "#e67e22", "#2ecc71"]
WP_COLS    = ["#3498db","#e67e22","#9b59b6","#1abc9c"]

# ── Top row: per-run RMSE for B and C ─────────────────────────────────────────
for ax, results, label, color in [
    (ax_B, B_results, "Mode B: NL-Commanded (multi-turn)", "#e67e22"),
    (ax_C, C_results, "Mode C: Full-Auto (single command)", "#2ecc71"),
]:
    rmse_vals = [r["rmse_cm"] for r in results]
    run_ids   = np.arange(1, N_RUNS+1)
    bar_cols  = ["green" if r["passed"] else "red" for r in results]
    ax.bar(run_ids, rmse_vals, color=bar_cols, alpha=0.75, edgecolor="black", width=0.6)
    ax.axhline(np.mean(rmse_vals), color="navy", ls="--", lw=1.5,
               label=f"Mean={np.mean(rmse_vals):.2f}cm")
    ax.axhline(rmse_A_overall*100, color="grey", ls=":", lw=1.5,
               label=f"Mode A ref={rmse_A_overall*100:.2f}cm")
    ax.axhline(PASS_RMSE_CM, color="red", ls=":", lw=1, label=f"Pass ≤{PASS_RMSE_CM:.0f}cm", alpha=0.6)
    for xi, v in zip(run_ids, rmse_vals):
        ax.text(xi, v+0.05, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(run_ids)
    ax.set_xticklabels([f"Run {i}\n({'✓' if r['passed'] else '✗'})"
                        for i, r in enumerate(results, 1)], fontsize=8)
    ax.set_ylabel("Overall RMSE (cm)")
    rmse_ci_local = bootstrap_ci(rmse_vals)
    ax.set_title(f"{label}\nRMSE={np.mean(rmse_vals):.3f}±{np.std(rmse_vals):.3f}cm  "
                 f"CI=[{rmse_ci_local[0]:.3f},{rmse_ci_local[1]:.3f}]", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

# ── Mid-left: per-waypoint RMSE breakdown ─────────────────────────────────────
wp_labels = [f"WP{i+1}\n{WAYPOINTS[i]}m" for i in range(len(WAYPOINTS))]
x_wp      = np.arange(len(WAYPOINTS))
w_bar     = 0.25

# Mode A (single run)
a_wp_cm = [wp_rmses_A[i]*100 if not math.isnan(wp_rmses_A[i]) else 0
           for i in range(len(WAYPOINTS))]
b_wp_cm = [float(np.mean([r[f"wp{i+1}_rmse_cm"] for r in B_results
                           if not math.isnan(r[f"wp{i+1}_rmse_cm"])]))
           for i in range(len(WAYPOINTS))]
c_wp_cm = [float(np.mean([r[f"wp{i+1}_rmse_cm"] for r in C_results
                           if not math.isnan(r[f"wp{i+1}_rmse_cm"])]))
           for i in range(len(WAYPOINTS))]

ax_wp.bar(x_wp - w_bar, a_wp_cm, w_bar, color=MODE_COLS[0], alpha=0.85,
          edgecolor="black", label="Mode A (scripted)")
ax_wp.bar(x_wp,          b_wp_cm, w_bar, color=MODE_COLS[1], alpha=0.85,
          edgecolor="black", label="Mode B (NL-cmd)")
ax_wp.bar(x_wp + w_bar, c_wp_cm, w_bar, color=MODE_COLS[2], alpha=0.85,
          edgecolor="black", label="Mode C (auto)")
for xi, av, bv, cv in zip(x_wp, a_wp_cm, b_wp_cm, c_wp_cm):
    ax_wp.text(xi-w_bar, av+0.02, f"{av:.2f}", ha="center", fontsize=7)
    ax_wp.text(xi,       bv+0.02, f"{bv:.2f}", ha="center", fontsize=7)
    ax_wp.text(xi+w_bar, cv+0.02, f"{cv:.2f}", ha="center", fontsize=7)
ax_wp.set_xticks(x_wp); ax_wp.set_xticklabels(wp_labels)
ax_wp.set_ylabel("Mean RMSE at waypoint (cm)")
ax_wp.set_title("Per-waypoint RMSE comparison\n(all 3 modes, same arrival-triggered window)")
ax_wp.legend(fontsize=8); ax_wp.grid(True, alpha=0.3, axis="y")

# ── Mid: overall RMSE comparison ─────────────────────────────────────────────
mode_labels = ["A\n(scripted)", "B\n(NL-cmd)", "C\n(auto)"]
rmse_means  = [rmse_A_overall*100, float(np.mean(B_rmse)), float(np.mean(C_rmse))]
rmse_stds   = [0, float(np.std(B_rmse)), float(np.std(C_rmse))]
ax_cmp.bar(mode_labels, rmse_means, color=MODE_COLS, alpha=0.85, edgecolor="black")
ax_cmp.errorbar(mode_labels, rmse_means, yerr=rmse_stds,
                fmt="none", ecolor="black", capsize=8, lw=2)
ax_cmp.set_ylabel("Overall RMSE (cm)")
ax_cmp.set_title("Overall RMSE\n(error bars=std, N=5 for B/C)")
ax_cmp.grid(True, alpha=0.3, axis="y")
for i, (m, s) in enumerate(zip(rmse_means, rmse_stds)):
    ax_cmp.text(i, m+s+0.05, f"{m:.2f}cm", ha="center", fontsize=9, fontweight="bold")

# ── Mid: API calls ─────────────────────────────────────────────────────────────
api_means = [0, float(np.mean(B_api)), float(np.mean(C_api))]
api_stds  = [0, float(np.std(B_api)),  float(np.std(C_api))]
ax_api.bar(mode_labels, api_means, color=MODE_COLS, alpha=0.85, edgecolor="black")
ax_api.errorbar(mode_labels, api_means, yerr=api_stds,
                fmt="none", ecolor="black", capsize=8, lw=2)
ax_api.set_ylabel("API calls"); ax_api.set_title("API calls per mission")
ax_api.grid(True, alpha=0.3, axis="y")
for i, (m, s) in enumerate(zip(api_means, api_stds)):
    if m > 0:
        ax_api.text(i, m+s+0.3, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")

# ── Bottom: mission time, pass rate, cost, summary box ────────────────────────
time_means = [t_A, float(np.mean(B_time)), float(np.mean(C_time))]
time_stds  = [0,   float(np.std(B_time)),  float(np.std(C_time))]
ax_tme.bar(mode_labels, time_means, color=MODE_COLS, alpha=0.85, edgecolor="black")
ax_tme.errorbar(mode_labels, time_means, yerr=time_stds,
                fmt="none", ecolor="black", capsize=8, lw=2)
ax_tme.set_ylabel("Mission time (sim s)"); ax_tme.set_title("Mission time")
ax_tme.grid(True, alpha=0.3, axis="y")

pass_rates = [1.0, n_B_pass/N_RUNS, n_C_pass/N_RUNS]
pass_cis   = [(1.0, 1.0), wilson_ci(n_B_pass, N_RUNS), wilson_ci(n_C_pass, N_RUNS)]
err_lo = [r-lo for r,(lo,hi) in zip(pass_rates, pass_cis)]
err_hi = [hi-r for r,(lo,hi) in zip(pass_rates, pass_cis)]
ax_pas.bar(mode_labels, pass_rates, color=MODE_COLS, alpha=0.85, edgecolor="black")
ax_pas.errorbar(mode_labels, pass_rates, yerr=[err_lo, err_hi],
                fmt="none", ecolor="black", capsize=8, lw=2)
ax_pas.set_ylim(0, 1.3); ax_pas.set_ylabel("Pass rate")
ax_pas.set_title(f"Pass rate (Wilson 95% CI)\n(RMSE≤{PASS_RMSE_CM:.0f}cm, all WP reached, disarmed)")
ax_pas.grid(True, alpha=0.3, axis="y")
for i, (r, (lo, hi)) in enumerate(zip(pass_rates, pass_cis)):
    ax_pas.text(i, r+0.06, f"{r:.2f}\n[{lo:.2f},{hi:.2f}]", ha="center", fontsize=7)

cost_means = [0, float(np.mean(B_cost)), float(np.mean(C_cost))]
ax_cst.bar(mode_labels, cost_means, color=MODE_COLS, alpha=0.85, edgecolor="black")
ax_cst.set_ylabel("Cost per run (USD)"); ax_cst.set_title("API cost per run")
ax_cst.grid(True, alpha=0.3, axis="y")
for i, c in enumerate(cost_means):
    if c > 0:
        ax_cst.text(i, c*1.04, f"${c:.3f}", ha="center", fontsize=8, fontweight="bold")

ax_txt.axis("off")
ratio_AB = rmse_A_overall*100/np.mean(B_rmse) if B_rmse else float("nan")
ratio_AC = rmse_A_overall*100/np.mean(C_rmse) if C_rmse else float("nan")
summary = (
    f"Mission: {len(WAYPOINTS)} waypoints\n"
    f"  {WAYPOINTS} m\n"
    f"  Hold {HOLD_TIME}s per WP\n"
    f"  Arrival window ±{ARRIVAL_TOL}m\n\n"
    f"Mode A (scripted):\n"
    f"  RMSE = {rmse_A_overall*100:.3f} cm\n"
    f"  All WPs: althold PID\n\n"
    f"Mode B (NL-cmd, N=5):\n"
    f"  RMSE = {np.mean(B_rmse):.3f}±{np.std(B_rmse):.3f} cm\n"
    f"  Pass: {n_B_pass}/{N_RUNS}\n"
    f"  API: {np.mean(B_api):.1f} calls\n\n"
    f"Mode C (auto, N=5):\n"
    f"  RMSE = {np.mean(C_rmse):.3f}±{np.std(C_rmse):.3f} cm\n"
    f"  Pass: {n_C_pass}/{N_RUNS}\n"
    f"  API: {np.mean(C_api):.1f} calls\n\n"
    f"A vs B: {ratio_AB:.1f}×\n"
    f"A vs C: {ratio_AC:.1f}×\n"
    f"B vs C: {np.mean(B_rmse)/np.mean(C_rmse):.2f}× (CI overlap)"
)
ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
            fontsize=8.5, va="top", family="monospace",
            bbox=dict(facecolor="#f0f4f8", edgecolor="#aaa",
                      boxstyle="round,pad=0.4"))

fig.suptitle(
    f"EXP-C8 v2: Three-Mode Comparison — Scripted vs NL-Commanded vs Full-Auto\n"
    f"Mission: {WAYPOINTS} m waypoints, {HOLD_TIME}s hold each, arrival-triggered RMSE  |  "
    f"Mode A={rmse_A_overall*100:.2f}cm  B={np.mean(B_rmse):.2f}cm  C={np.mean(C_rmse):.2f}cm",
    fontsize=11
)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"[C8] Plot: {OUT_PNG}")

print(f"\n[C8 v2] FINAL COMPARISON (fair: arrival-triggered RMSE window, same mission all modes):")
print(f"  Mode A (scripted, 1 run): RMSE={rmse_A_overall*100:.3f}cm")
print(f"  Mode B (NL-cmd, N={N_RUNS}):  RMSE={np.mean(B_rmse):.3f}±{np.std(B_rmse):.3f}cm  "
      f"pass={n_B_pass}/{N_RUNS}  CI=[{B_pass_lo:.2f},{B_pass_hi:.2f}]")
print(f"  Mode C (auto, N={N_RUNS}):    RMSE={np.mean(C_rmse):.3f}±{np.std(C_rmse):.3f}cm  "
      f"pass={n_C_pass}/{N_RUNS}  CI=[{C_pass_lo:.2f},{C_pass_hi:.2f}]")
print(f"  A vs B: {ratio_AB:.1f}×  |  A vs C: {ratio_AC:.1f}×")
