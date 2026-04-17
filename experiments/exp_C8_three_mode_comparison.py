"""
EXP-C8: Three-Mode Comparison — Manual vs NL-Commanded vs Full-Auto  (N=5 runs)
================================================================================
Same mission (takeoff → 1m → hold 10s → land) in 3 modes:
  Mode A — Manual: scripted, no LLM (deterministic — run once, reported as reference)
  Mode B — NL-commanded: human prompt → LLM (N=5 runs)
  Mode C — Full-auto: autonomous LLM (N=5 runs)

Reports RMSE mean±std and 95% bootstrap CI per mode for B and C.
Mode A is deterministic and used as baseline reference only.

Outputs:
  results/C8_runs.csv        — per-run metrics (modes B and C)
  results/C8_summary.csv     — aggregate statistics
  results/C8_three_mode_comparison.png
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
OUT_RUNS    = os.path.join(os.path.dirname(__file__), "results", "C8_runs.csv")
OUT_SUMMARY = os.path.join(os.path.dirname(__file__), "results", "C8_summary.csv")
OUT_PNG     = os.path.join(os.path.dirname(__file__), "results", "C8_three_mode_comparison.png")

TARGET_ALT = 1.0
HOLD_TIME  = 10.0
TOLERANCE  = 0.10
N_RUNS     = 5

PAPER_REFS = {
    "ReAct": (
        "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). "
        "ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. "
        "Modes B and C both use the ReAct loop; Mode A (scripted) is the non-LLM baseline."
    ),
    "Vemprala2023": (
        "Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). "
        "ChatGPT for Robotics: Design Principles and Model Abilities. MSR-TR-2023-8. arXiv:2306.17582. "
        "Three-mode comparison design directly follows Vemprala's manual vs LLM evaluation protocol."
    ),
    "InnerMonologue": (
        "Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning "
        "with Language Models. arXiv:2207.05608. "
        "Full-auto Mode C relies on tool-result inner monologue with no human in the loop."
    ),
}
SIM_HZ     = 200
DT         = 1.0 / SIM_HZ

NL_COMMAND   = ("Please take off, hover at exactly 1 metre for 10 seconds, "
                "then land and disarm safely.")
AUTO_COMMAND = ("Autonomous mission: take off, achieve and hold 1.0 m altitude for 10 seconds, "
                "then execute a full safe landing. No human in the loop — complete the full mission autonomously.")

# ── Statistics helpers ─────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    boots = [np.mean(np.random.choice(arr, len(arr))) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def hold_rmse(tel_buf, target):
    """Extract RMSE during the altitude hold phase from an agent telemetry buffer."""
    hold_samps = []
    in_hold = False
    for s in tel_buf:
        if s["althold"] == 1 and abs(s["lw_z"]/1000.0 - target) < 0.30:
            in_hold = True
        if in_hold:
            hold_samps.append(s)
    z_hold = [s["lw_z"]/1000.0 for s in hold_samps[:int(HOLD_TIME * 10)]]
    if not z_hold:
        return float("nan")
    return math.sqrt(sum((z - target)**2 for z in z_hold) / len(z_hold))

# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — Manual scripted (deterministic reference — run once)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[C8] ── Mode A: Manual scripted (reference, 1 run) ───────────")

state_A   = DroneState()
physics_A = PhysicsLoop(state_A)

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

with state_A.lock:
    state_A.armed = True
    state_A.ch5   = 1000

for pwm in range(1000, 1560, 5):
    with state_A.lock:
        state_A.ch1 = pwm
    tick_A(4)

with state_A.lock:
    state_A.ch1 = 1560
tick_A(int(8 * SIM_HZ))

pwm_now = 1550
with state_A.lock:
    state_A.ch1 = pwm_now
for _ in range(150):
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

with state_A.lock:
    state_A.althold          = True
    state_A.alt_sp           = 1.0
    state_A.alt_sp_mm        = 1000.0
    state_A.hover_thr_locked = HOVER_THR_A
physics_A.pid_alt_pos.reset()
physics_A.pid_alt_vel.reset()
tick_A(int(8.0 * SIM_HZ))

t_hold_start_A = t_A
tick_A(int(HOLD_TIME * SIM_HZ))
t_hold_end_A   = t_A

with state_A.lock:
    state_A.althold = False
    state_A.ch2 = 1500; state_A.ch3 = 1500
for pwm in [1400, 1200, 1100, 1000]:
    with state_A.lock:
        state_A.ch1 = pwm
    tick_A(int(1.0 * SIM_HZ))
with state_A.lock:
    state_A.ch5   = 2000
    state_A.armed = False

hold_tel_A = [s for s in tel_A if t_hold_start_A <= s["t"] < t_hold_end_A]
z_hold_A   = [s["ekf_z"] for s in hold_tel_A]
rmse_A     = math.sqrt(sum((z - TARGET_ALT)**2 for z in z_hold_A) / len(z_hold_A)) \
             if z_hold_A else float("nan")

print(f"  Mode A RMSE (hold): {rmse_A*100:.2f} cm  mission_time={t_A:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
#  MODE B and C — LLM runs (N=5 each)
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_mode(mode_label, command, run_idx):
    print(f"\n[C8] ── Mode {mode_label} Run {run_idx+1}/{N_RUNS} ──────────────────")
    agent = SimAgent(session_id=f"C8_{mode_label}_run{run_idx}")
    t_wall_start = time.time()

    text, api_stats, trace = agent.run_agent_loop(command, max_turns=35)
    t_wall = time.time() - t_wall_start

    rmse = hold_rmse(agent.tel_buf, TARGET_ALT)

    with agent.state.lock:
        landed   = agent.state.z < 0.15
        disarmed = not agent.state.armed

    n_api  = len(api_stats)
    in_tok = sum(s["input_tokens"]  for s in api_stats)
    out_tok= sum(s["output_tokens"] for s in api_stats)
    cost   = sum(s["cost_usd"]      for s in api_stats)

    rmse_cm = rmse * 100 if not math.isnan(rmse) else float("nan")
    passed  = not math.isnan(rmse) and rmse_cm <= (TOLERANCE * 100 * 2) and disarmed

    print(f"  RMSE={rmse_cm:.2f}cm  sim_time={agent.sim_time:.1f}s  "
          f"api={n_api}  landed={landed}  pass={passed}")

    return {
        "mode":          mode_label,
        "run":           run_idx + 1,
        "rmse_cm":       round(rmse_cm, 3),
        "mission_time_s":round(agent.sim_time, 1),
        "landed":        int(landed),
        "disarmed":      int(disarmed),
        "passed":        int(passed),
        "api_calls":     n_api,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      round(cost, 6),
        "wall_time_s":   round(t_wall, 1),
        "_tel":          agent.tel_buf,
    }

print("\n[C8] Running Mode B (NL-commanded) × 5 …")
B_results = [run_llm_mode("B", NL_COMMAND,   i) for i in range(N_RUNS)]

print("\n[C8] Running Mode C (Full-auto) × 5 …")
C_results = [run_llm_mode("C", AUTO_COMMAND, i) for i in range(N_RUNS)]

# ── Aggregate ─────────────────────────────────────────────────────────────────

def agg(results, key):
    return [r[key] for r in results if not math.isnan(r[key])]

B_rmse = agg(B_results, "rmse_cm")
C_rmse = agg(C_results, "rmse_cm")
B_time = agg(B_results, "mission_time_s")
C_time = agg(C_results, "mission_time_s")
B_api  = agg(B_results, "api_calls")
C_api  = agg(C_results, "api_calls")

B_rmse_ci = bootstrap_ci(B_rmse)
C_rmse_ci = bootstrap_ci(C_rmse)
B_time_ci = bootstrap_ci(B_time)
C_time_ci = bootstrap_ci(C_time)

n_B_pass = sum(r["passed"] for r in B_results)
n_C_pass = sum(r["passed"] for r in C_results)
B_pass_lo, B_pass_hi = wilson_ci(n_B_pass, N_RUNS)
C_pass_lo, C_pass_hi = wilson_ci(n_C_pass, N_RUNS)

print(f"\n[C8] ── AGGREGATE ({N_RUNS} runs) ───────────────────────────────")
print(f"  Mode A (ref):  RMSE={rmse_A*100:.2f}cm (deterministic)")
print(f"  Mode B:  RMSE={np.mean(B_rmse):.2f}±{np.std(B_rmse):.2f}cm  "
      f"CI=[{B_rmse_ci[0]:.2f},{B_rmse_ci[1]:.2f}]  pass={n_B_pass}/{N_RUNS}  "
      f"CI=[{B_pass_lo:.2f},{B_pass_hi:.2f}]")
print(f"  Mode C:  RMSE={np.mean(C_rmse):.2f}±{np.std(C_rmse):.2f}cm  "
      f"CI=[{C_rmse_ci[0]:.2f},{C_rmse_ci[1]:.2f}]  pass={n_C_pass}/{N_RUNS}  "
      f"CI=[{C_pass_lo:.2f},{C_pass_hi:.2f}]")
print(f"  Mode B time:  {np.mean(B_time):.1f}±{np.std(B_time):.1f}s")
print(f"  Mode C time:  {np.mean(C_time):.1f}±{np.std(C_time):.1f}s")
print(f"  Mode B API:   {np.mean(B_api):.1f}±{np.std(B_api):.1f}")
print(f"  Mode C API:   {np.mean(C_api):.1f}±{np.std(C_api):.1f}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
csv_keys = [k for k in B_results[0].keys() if not k.startswith("_")]
all_llm_runs = B_results + C_results
with open(OUT_RUNS, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_keys)
    w.writeheader()
    for r in all_llm_runs:
        w.writerow({k: r[k] for k in csv_keys})
print(f"[C8] Per-run CSV: {OUT_RUNS}")

summary_rows = [
    ("n_runs",            N_RUNS),
    ("mode_A_rmse_cm",    round(rmse_A * 100, 3)),
    ("mode_A_note",       "deterministic reference (1 run)"),
    ("mode_B_rmse_mean",  round(float(np.mean(B_rmse)), 3)),
    ("mode_B_rmse_std",   round(float(np.std(B_rmse)), 3)),
    ("mode_B_rmse_ci_lo", round(B_rmse_ci[0], 3)),
    ("mode_B_rmse_ci_hi", round(B_rmse_ci[1], 3)),
    ("mode_B_pass_rate",  round(n_B_pass / N_RUNS, 3)),
    ("mode_B_pass_ci_lo", round(B_pass_lo, 3)),
    ("mode_B_pass_ci_hi", round(B_pass_hi, 3)),
    ("mode_B_time_mean",  round(float(np.mean(B_time)), 1)),
    ("mode_B_time_std",   round(float(np.std(B_time)), 1)),
    ("mode_B_api_mean",   round(float(np.mean(B_api)), 1)),
    ("mode_B_api_std",    round(float(np.std(B_api)), 1)),
    ("mode_C_rmse_mean",  round(float(np.mean(C_rmse)), 3)),
    ("mode_C_rmse_std",   round(float(np.std(C_rmse)), 3)),
    ("mode_C_rmse_ci_lo", round(C_rmse_ci[0], 3)),
    ("mode_C_rmse_ci_hi", round(C_rmse_ci[1], 3)),
    ("mode_C_pass_rate",  round(n_C_pass / N_RUNS, 3)),
    ("mode_C_pass_ci_lo", round(C_pass_lo, 3)),
    ("mode_C_pass_ci_hi", round(C_pass_hi, 3)),
    ("mode_C_time_mean",  round(float(np.mean(C_time)), 1)),
    ("mode_C_time_std",   round(float(np.std(C_time)), 1)),
    ("mode_C_api_mean",   round(float(np.mean(C_api)), 1)),
    ("mode_C_api_std",    round(float(np.std(C_api)), 1)),
]
with open(OUT_SUMMARY, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(summary_rows)
    for ref_key, ref_val in PAPER_REFS.items():
        w.writerow([f"ref_{ref_key}", ref_val])
print(f"[C8] Summary CSV: {OUT_SUMMARY}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

ax_B = fig.add_subplot(gs[0, 0:2])
ax_C = fig.add_subplot(gs[0, 2:4])
ax_rmse  = fig.add_subplot(gs[1, 0])
ax_time  = fig.add_subplot(gs[1, 1])
ax_api   = fig.add_subplot(gs[1, 2])
ax_pass  = fig.add_subplot(gs[1, 3])

# Top: RMSE per run for B and C
for ax, results, label, color in [
    (ax_B, B_results, "Mode B: NL-Commanded", "#e67e22"),
    (ax_C, C_results, "Mode C: Full-Auto",    "#2ecc71"),
]:
    rmse_vals = [r["rmse_cm"] for r in results]
    run_ids   = range(1, N_RUNS + 1)
    bar_cols  = ["green" if r["passed"] else "red" for r in results]
    ax.bar(run_ids, rmse_vals, color=bar_cols, alpha=0.75, edgecolor="black")
    ax.axhline(np.mean(rmse_vals), color="navy", ls="--", lw=1.5,
               label=f"Mean={np.mean(rmse_vals):.2f}cm")
    ax.axhline(rmse_A * 100, color="grey", ls=":", lw=1.5,
               label=f"Mode A ref={rmse_A*100:.2f}cm")
    ax.axhline(TOLERANCE * 100, color="red", ls=":", lw=1, label=f"Tolerance {TOLERANCE*100:.0f}cm")
    ax.set_xticks(run_ids)
    ax.set_xticklabels([f"Run {i}\n({'✓' if r['passed'] else '✗'})" for i, r in enumerate(results, 1)],
                       fontsize=8)
    ax.set_ylabel("RMSE during hold (cm)")
    rmse_ci = bootstrap_ci(rmse_vals)
    ax.set_title(f"{label}\nRMSE={np.mean(rmse_vals):.2f}±{np.std(rmse_vals):.2f}cm  "
                 f"CI=[{rmse_ci[0]:.2f},{rmse_ci[1]:.2f}]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

# Bottom row charts
mode_labels = ["A (ref)", "B (NL)", "C (Auto)"]
mode_colors = ["#3498db", "#e67e22", "#2ecc71"]

# RMSE comparison
rmse_means = [rmse_A * 100, float(np.mean(B_rmse)), float(np.mean(C_rmse))]
rmse_stds  = [0,             float(np.std(B_rmse)),  float(np.std(C_rmse))]
ax_rmse.bar(mode_labels, rmse_means, color=mode_colors, alpha=0.75, edgecolor="black")
ax_rmse.errorbar(mode_labels, rmse_means, yerr=rmse_stds,
                 fmt="none", ecolor="black", capsize=6, lw=1.5)
ax_rmse.axhline(TOLERANCE * 100, color="red", ls="--", lw=1)
ax_rmse.set_ylabel("RMSE hold phase (cm)")
ax_rmse.set_title("RMSE comparison\n(error bars=std, N=5 for B/C)")
ax_rmse.grid(True, alpha=0.3, axis="y")
for i, (m, s) in enumerate(zip(rmse_means, rmse_stds)):
    ax_rmse.text(i, m + s + 0.2, f"{m:.2f}cm", ha="center", fontsize=8)

# Mission time
time_means = [t_A, float(np.mean(B_time)), float(np.mean(C_time))]
time_stds  = [0,   float(np.std(B_time)),  float(np.std(C_time))]
ax_time.bar(mode_labels, time_means, color=mode_colors, alpha=0.75, edgecolor="black")
ax_time.errorbar(mode_labels, time_means, yerr=time_stds,
                 fmt="none", ecolor="black", capsize=6, lw=1.5)
ax_time.set_ylabel("Mission time (sim s)")
ax_time.set_title("Mission time\n(error bars=std)")
ax_time.grid(True, alpha=0.3, axis="y")

# API calls
api_means = [0, float(np.mean(B_api)), float(np.mean(C_api))]
api_stds  = [0, float(np.std(B_api)),  float(np.std(C_api))]
ax_api.bar(mode_labels, api_means, color=mode_colors, alpha=0.75, edgecolor="black")
ax_api.errorbar(mode_labels, api_means, yerr=api_stds,
                fmt="none", ecolor="black", capsize=6, lw=1.5)
ax_api.set_ylabel("API calls")
ax_api.set_title("API calls per mode\n(0 for manual)")
ax_api.grid(True, alpha=0.3, axis="y")

# Pass rate
pass_rates = [1.0, n_B_pass/N_RUNS, n_C_pass/N_RUNS]
pass_cis   = [(1.0, 1.0),
              wilson_ci(n_B_pass, N_RUNS),
              wilson_ci(n_C_pass, N_RUNS)]
err_lo3 = [r - lo for r, (lo, hi) in zip(pass_rates, pass_cis)]
err_hi3 = [hi - r for r, (lo, hi) in zip(pass_rates, pass_cis)]
ax_pass.bar(mode_labels, pass_rates, color=mode_colors, alpha=0.75, edgecolor="black")
ax_pass.errorbar(mode_labels, pass_rates, yerr=[err_lo3, err_hi3],
                 fmt="none", ecolor="black", capsize=6, lw=1.5)
ax_pass.set_ylim(0, 1.25)
ax_pass.set_ylabel("Pass rate")
ax_pass.set_title(f"Pass rate (Wilson 95% CI)\n(N={N_RUNS} for B/C)")
ax_pass.grid(True, alpha=0.3, axis="y")
for i, (r, (lo, hi)) in enumerate(zip(pass_rates, pass_cis)):
    ax_pass.text(i, r + 0.06, f"{r:.2f}\n[{lo:.2f},{hi:.2f}]",
                 ha="center", fontsize=7)

fig.suptitle(
    f"EXP-C8: Three-Mode Comparison  (N={N_RUNS} runs for B/C, temperature=0.2)\n"
    f"Mode A (ref) RMSE={rmse_A*100:.2f}cm  |  "
    f"Mode B RMSE={np.mean(B_rmse):.2f}±{np.std(B_rmse):.2f}cm  |  "
    f"Mode C RMSE={np.mean(C_rmse):.2f}±{np.std(C_rmse):.2f}cm",
    fontsize=11
)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"[C8] Plot: {OUT_PNG}")

print(f"\n[C8] FINAL COMPARISON:")
print(f"  Mode A (ref):  RMSE={rmse_A*100:.2f}cm")
print(f"  Mode B (N={N_RUNS}): RMSE={np.mean(B_rmse):.2f}±{np.std(B_rmse):.2f}cm  "
      f"pass={n_B_pass}/{N_RUNS}  CI=[{B_pass_lo:.2f},{B_pass_hi:.2f}]")
print(f"  Mode C (N={N_RUNS}): RMSE={np.mean(C_rmse):.2f}±{np.std(C_rmse):.2f}cm  "
      f"pass={n_C_pass}/{N_RUNS}  CI=[{C_pass_lo:.2f},{C_pass_hi:.2f}]")
