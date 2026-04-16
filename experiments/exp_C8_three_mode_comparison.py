"""
EXP-C8: Three-Mode Comparison — Manual vs NL-Commanded vs Full-Auto
====================================================================
Same mission (takeoff → 1m → hold 10s → land) in 3 modes:

  Mode A — Manual: scripted fixed PWM, no LLM (baseline, like B-series)
  Mode B — NL-commanded: human says "take off, hover at 1m for 10 seconds, then land"
  Mode C — Full-auto: LLM + task description "take off, hold at 1m for 10s, then land"
            (same prompt as B but no "human" framing — LLM acts fully autonomously)

Measures per mode:
  - Altitude RMSE during hold phase
  - Mission time (simulated seconds)
  - Human inputs (0 for A and C, 1 for B)
  - API calls (0 for A)
  - Tokens per mode
  - Completion success

Outputs:
  results/C8_three_mode_comparison.csv  — per-mode metrics
  results/C8_three_mode_comparison.png  — side-by-side altitude plots + bar charts
"""

import sys, os, csv, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c_series_agent import SimAgent

os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "C8_three_mode_comparison.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "results", "C8_three_mode_comparison.png")

TARGET_ALT = 1.0
HOLD_TIME  = 10.0   # seconds
TOLERANCE  = 0.10   # m for RMSE

# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — Manual scripted (no LLM)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C8] ── Mode A: Manual scripted ──────────────────────────────")

SIM_HZ = 200
DT     = 1.0 / SIM_HZ

from drone_sim import PhysicsLoop, DroneState

state_A   = DroneState()
physics_A = PhysicsLoop(state_A)

def ticks_A(s):
    return int(s * SIM_HZ)

tel_A = []
t_A   = 0.0

def tick_A(n=1):
    global t_A
    for _ in range(n):
        physics_A.tick()
        t_A += DT
        if int(t_A * 10) % 1 == 0 and len(tel_A) < int(t_A * 10):
            with state_A.lock:
                tel_A.append({"t": t_A, "z": state_A.z, "ekf_z": state_A.ekf_z,
                              "alt_sp": state_A.alt_sp})

t_A_start = time.time()

# Arm
with state_A.lock:
    state_A.armed = True
    state_A.ch5   = 1000

# Ramp up
for pwm in range(1000, 1560, 5):
    with state_A.lock:
        state_A.ch1 = pwm
    tick_A(4)

# Climb to ~1m
with state_A.lock:
    state_A.ch1 = 1560
tick_A(ticks_A(8))

# Find hover (simple binary search)
pwm_now = 1550
with state_A.lock:
    state_A.ch1 = pwm_now
for _ in range(150):
    tick_A(ticks_A(0.2))
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

# Enable althold at ~1m
with state_A.lock:
    state_A.althold          = True
    state_A.alt_sp           = 1.0
    state_A.alt_sp_mm        = 1000.0
    state_A.hover_thr_locked = HOVER_THR_A
physics_A.pid_alt_pos.reset()
physics_A.pid_alt_vel.reset()

# Wait for drone to settle at 1m before measuring hold phase (mirrors LLM wait)
tick_A(ticks_A(8.0))

# Hold for HOLD_TIME
t_hold_start_A = t_A
tick_A(ticks_A(HOLD_TIME))
t_hold_end_A   = t_A

# Land
with state_A.lock:
    state_A.althold = False
    state_A.ch2 = 1500; state_A.ch3 = 1500
for pwm in [1400, 1200, 1100, 1000]:
    with state_A.lock:
        state_A.ch1 = pwm
    tick_A(ticks_A(1.0))
with state_A.lock:
    state_A.ch5   = 2000
    state_A.armed = False

t_A_mission = t_A
t_A_wall    = time.time() - t_A_start

# Telemetry during hold
hold_tel_A = [s for s in tel_A if t_hold_start_A <= s["t"] < t_hold_end_A]
z_hold_A   = [s["ekf_z"] for s in hold_tel_A]
rmse_A     = math.sqrt(sum((z - TARGET_ALT)**2 for z in z_hold_A) / len(z_hold_A)) if z_hold_A else float("nan")

print(f"  Hover PWM:     {HOVER_PWM_A}")
print(f"  Mission time:  {t_A_mission:.1f} s (sim)")
print(f"  RMSE (hold):   {rmse_A*100:.2f} cm")

# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — NL-commanded (human gives command, LLM executes)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C8] ── Mode B: NL-commanded ──────────────────────────────────")

NL_COMMAND = (
    "Please take off, hover at exactly 1 metre for 10 seconds, "
    "then land and disarm safely."
)

agent_B = SimAgent(session_id="C8_B")
t_B_wall_start = time.time()

text_B, stats_B, trace_B = agent_B.run_agent_loop(NL_COMMAND, max_turns=35)

t_B_wall = time.time() - t_B_wall_start
t_B_mission = agent_B.sim_time

tel_B_all = agent_B.tel_buf
# Altitude hold phase: find when althold became active at ~1m
hold_tel_B = []
in_hold    = False
for s in tel_B_all:
    if s["althold"] == 1 and abs(s["lw_z"]/1000.0 - TARGET_ALT) < 0.30:
        in_hold = True
    if in_hold:
        hold_tel_B.append(s)
z_hold_B = [s["lw_z"]/1000.0 for s in hold_tel_B[:int(HOLD_TIME * 10)]]
rmse_B   = (math.sqrt(sum((z - TARGET_ALT)**2 for z in z_hold_B) / len(z_hold_B))
            if z_hold_B else float("nan"))

n_api_B  = len(stats_B)
in_tok_B = sum(s["input_tokens"]  for s in stats_B)
out_tok_B= sum(s["output_tokens"] for s in stats_B)
cost_B   = sum(s["cost_usd"]      for s in stats_B)

with agent_B.state.lock:
    landed_B  = agent_B.state.z < 0.15
    disarmed_B= not agent_B.state.armed

print(f"  Mission time:  {t_B_mission:.1f} s (sim)")
print(f"  RMSE (hold):   {rmse_B*100:.2f} cm")
print(f"  API calls:     {n_api_B}")
print(f"  Tokens in/out: {in_tok_B}/{out_tok_B}")
print(f"  Landed:        {landed_B}, Disarmed: {disarmed_B}")

# ══════════════════════════════════════════════════════════════════════════════
#  MODE C — Full-auto (same as B but framed as autonomous task, no "human" framing)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C8] ── Mode C: Full-auto ─────────────────────────────────────")

AUTO_COMMAND = (
    "Autonomous mission: take off, achieve and hold 1.0 m altitude for 10 seconds, "
    "then execute a full safe landing. No human in the loop — complete the full mission autonomously."
)

agent_C = SimAgent(session_id="C8_C")
t_C_wall_start = time.time()

text_C, stats_C, trace_C = agent_C.run_agent_loop(AUTO_COMMAND, max_turns=35)

t_C_wall = time.time() - t_C_wall_start
t_C_mission = agent_C.sim_time

tel_C_all = agent_C.tel_buf
hold_tel_C = []
in_hold_C  = False
for s in tel_C_all:
    if s["althold"] == 1 and abs(s["lw_z"]/1000.0 - TARGET_ALT) < 0.30:
        in_hold_C = True
    if in_hold_C:
        hold_tel_C.append(s)
z_hold_C = [s["lw_z"]/1000.0 for s in hold_tel_C[:int(HOLD_TIME * 10)]]
rmse_C   = (math.sqrt(sum((z - TARGET_ALT)**2 for z in z_hold_C) / len(z_hold_C))
            if z_hold_C else float("nan"))

n_api_C  = len(stats_C)
in_tok_C = sum(s["input_tokens"]  for s in stats_C)
out_tok_C= sum(s["output_tokens"] for s in stats_C)
cost_C   = sum(s["cost_usd"]      for s in stats_C)

with agent_C.state.lock:
    landed_C  = agent_C.state.z < 0.15
    disarmed_C= not agent_C.state.armed

print(f"  Mission time:  {t_C_mission:.1f} s (sim)")
print(f"  RMSE (hold):   {rmse_C*100:.2f} cm")
print(f"  API calls:     {n_api_C}")
print(f"  Tokens in/out: {in_tok_C}/{out_tok_C}")
print(f"  Landed:        {landed_C}, Disarmed: {disarmed_C}")

# ══════════════════════════════════════════════════════════════════════════════
#  Summary comparison
# ══════════════════════════════════════════════════════════════════════════════
modes = [
    {
        "mode":          "A_manual",
        "description":   "Manual scripted (no LLM)",
        "rmse_cm":       round(rmse_A * 100, 2),
        "mission_time_s":round(t_A_mission, 1),
        "human_inputs":  0,
        "api_calls":     0,
        "input_tokens":  0,
        "output_tokens": 0,
        "cost_usd":      0.0,
        "landed":        True,
        "disarmed":      True,
    },
    {
        "mode":          "B_nl_commanded",
        "description":   "NL-commanded (human prompt → LLM)",
        "rmse_cm":       round(rmse_B * 100, 2) if not math.isnan(rmse_B) else -1,
        "mission_time_s":round(t_B_mission, 1),
        "human_inputs":  1,
        "api_calls":     n_api_B,
        "input_tokens":  in_tok_B,
        "output_tokens": out_tok_B,
        "cost_usd":      round(cost_B, 6),
        "landed":        landed_B,
        "disarmed":      disarmed_B,
    },
    {
        "mode":          "C_full_auto",
        "description":   "Full-auto (autonomous LLM)",
        "rmse_cm":       round(rmse_C * 100, 2) if not math.isnan(rmse_C) else -1,
        "mission_time_s":round(t_C_mission, 1),
        "human_inputs":  0,
        "api_calls":     n_api_C,
        "input_tokens":  in_tok_C,
        "output_tokens": out_tok_C,
        "cost_usd":      round(cost_C, 6),
        "landed":        landed_C,
        "disarmed":      disarmed_C,
    },
]

print(f"\n[C8] ── COMPARISON TABLE ──────────────────────────────────────")
print(f"{'Mode':20s} {'RMSE (cm)':>10} {'Time (s)':>10} {'Inputs':>8} "
      f"{'API calls':>10} {'Tokens':>10} {'Landed':>8}")
print("-" * 80)
for m in modes:
    print(f"{m['mode']:20s} {m['rmse_cm']:>10.2f} {m['mission_time_s']:>10.1f} "
          f"{m['human_inputs']:>8} {m['api_calls']:>10} "
          f"{m['input_tokens']+m['output_tokens']:>10} {str(m['landed']):>8}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=modes[0].keys())
    w.writeheader()
    w.writerows(modes)
print(f"\n[C8] CSV: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

ax_alt_A = fig.add_subplot(gs[0, 0])
ax_alt_B = fig.add_subplot(gs[0, 1])
ax_alt_C = fig.add_subplot(gs[0, 2])
ax_rmse  = fig.add_subplot(gs[1, 0])
ax_time  = fig.add_subplot(gs[1, 1])
ax_api   = fig.add_subplot(gs[1, 2])

mode_colors = ["#3498db", "#e67e22", "#2ecc71"]

# Altitude plots
for ax, tel_data, title, color, hold_samps in [
    (ax_alt_A, tel_A, "Mode A: Manual", mode_colors[0], hold_tel_A),
    (ax_alt_B, agent_B.tel_buf, "Mode B: NL-Commanded", mode_colors[1], hold_tel_B),
    (ax_alt_C, agent_C.tel_buf, "Mode C: Full-Auto", mode_colors[2], hold_tel_C),
]:
    if tel_data:
        if isinstance(tel_data[0], dict) and "t" in tel_data[0]:
            t_v = [s["t"] for s in tel_data]
            if "z" in tel_data[0]:
                z_v = [s["z"]       for s in tel_data]
                ze_v= [s["ekf_z"]   for s in tel_data]
            else:
                z_v = [s.get("z_true", s.get("lw_z", 0)/1000.0) for s in tel_data]
                ze_v= [s.get("lw_z", 0)/1000.0                   for s in tel_data]
        else:
            t_v = z_v = ze_v = []

        if t_v:
            ax.plot(t_v, z_v,  color=color, lw=1.5, label="True z")
            ax.plot(t_v, ze_v, color="gray", lw=1.0, alpha=0.7, label="EKF z")
        ax.axhline(TARGET_ALT, color="red", ls="--", lw=1, label=f"{TARGET_ALT}m target")
        ax.axhspan(TARGET_ALT-TOLERANCE, TARGET_ALT+TOLERANCE, alpha=0.08, color="red")
    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Sim time (s)")
    rmse_v = modes[["A_manual","B_nl_commanded","C_full_auto"].index(
        next(m["mode"] for m in modes if title.split(":")[0].strip().split()[-1][0] in m["mode"]))]["rmse_cm"]
    ax.set_title(f"{title}\nRMSE={rmse_v:.2f}cm")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Fix alt axis for Mode A (different telemetry format)
if tel_A:
    ax_alt_A.clear()
    t_v_A  = [s["t"] for s in tel_A]
    z_v_A  = [s["z"] for s in tel_A]
    ze_v_A = [s["ekf_z"] for s in tel_A]
    ax_alt_A.plot(t_v_A, z_v_A,  color=mode_colors[0], lw=1.5, label="True z")
    ax_alt_A.plot(t_v_A, ze_v_A, color="gray", lw=1.0, alpha=0.7, label="EKF z")
    ax_alt_A.axhline(TARGET_ALT, color="red", ls="--", lw=1, label=f"{TARGET_ALT}m target")
    ax_alt_A.axhspan(TARGET_ALT-TOLERANCE, TARGET_ALT+TOLERANCE, alpha=0.08, color="red")
    ax_alt_A.set_title(f"Mode A: Manual\nRMSE={rmse_A*100:.2f}cm")
    ax_alt_A.set_ylabel("Altitude (m)"); ax_alt_A.set_xlabel("Sim time (s)")
    ax_alt_A.legend(fontsize=7); ax_alt_A.grid(True, alpha=0.3)

# Bar charts
mode_labels = ["A: Manual", "B: NL-Cmd", "C: Auto"]
rmse_vals   = [m["rmse_cm"] for m in modes]
time_vals   = [m["mission_time_s"] for m in modes]
api_vals    = [m["api_calls"] for m in modes]
tok_vals    = [(m["input_tokens"]+m["output_tokens"])/1000.0 for m in modes]

ax_rmse.bar(mode_labels, rmse_vals, color=mode_colors, alpha=0.75, edgecolor="black")
ax_rmse.axhline(TOLERANCE*100, color="red", ls="--", lw=1, label=f"Target ±{TOLERANCE*100:.0f}cm")
ax_rmse.set_ylabel("Hold phase RMSE (cm)")
ax_rmse.set_title("Altitude RMSE (hold phase)")
ax_rmse.legend(fontsize=8); ax_rmse.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(rmse_vals):
    ax_rmse.text(i, v + 0.3, f"{v:.2f}cm", ha="center", fontsize=9)

ax_time.bar(mode_labels, time_vals, color=mode_colors, alpha=0.75, edgecolor="black")
ax_time.set_ylabel("Mission time (sim seconds)")
ax_time.set_title("Mission time comparison")
ax_time.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(time_vals):
    ax_time.text(i, v + 1, f"{v:.0f}s", ha="center", fontsize=9)

ax2_twin = ax_api.twinx()
bars1 = ax_api.bar([0], [api_vals[0]], color=mode_colors[0], alpha=0.75, label="API calls")
bars2 = ax_api.bar([1], [api_vals[1]], color=mode_colors[1], alpha=0.75)
bars3 = ax_api.bar([2], [api_vals[2]], color=mode_colors[2], alpha=0.75)
ax2_twin.bar([0,1,2], tok_vals, color=mode_colors, alpha=0.25, label="Tokens (k)")
ax_api.set_xticks([0,1,2])
ax_api.set_xticklabels(mode_labels)
ax_api.set_ylabel("API calls")
ax2_twin.set_ylabel("Tokens (thousands)")
ax_api.set_title("API calls and token usage")
ax_api.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    "EXP-C8: Three-Mode Comparison — Manual vs NL-Commanded vs Full-Auto\n"
    f"Mission: takeoff → hold at {TARGET_ALT}m for {HOLD_TIME:.0f}s → land",
    fontsize=12
)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"[C8] Plot: {OUT_PNG}")

print(f"\n[C8] FINAL COMPARISON:")
for m in modes:
    print(f"  {m['mode']:20s}: RMSE={m['rmse_cm']:.2f}cm, "
          f"time={m['mission_time_s']:.0f}s, "
          f"api={m['api_calls']}, "
          f"landed={m['landed']}")
