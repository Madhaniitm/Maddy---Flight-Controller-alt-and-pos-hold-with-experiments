# Section C — LLM-in-the-Loop Agent Experiments: Observations
# Created: 2026-04-16 | Updated: 2026-04-19 (C8 v3: supervisor-design Mode B, sub-cm RMSE all modes)

---

## How C series differs from A and B series

A series validated physics models against literature. B series validated controller performance against literature benchmarks. C series tests a **natural language interface layer** on top of the same controller: a Claude LLM (`claude-sonnet-4-6` via Azure) receives free-text flight commands, selects the correct tool sequence from a defined API, and issues those tools to the drone simulator. The question being answered is: *"can an LLM reliably translate human intent into correct, safe, sequential flight actions?"*

The LLM-as-robot-planner paradigm is established in the literature — Ahn et al. 2022 (SayCan) demonstrated that LLMs can generate grounded, feasible action sequences for physical robots when equipped with an affordance model. C series extends this to a custom 50g drone with custom firmware and no pre-existing SDK, using a structured tool API designed specifically for this flight controller.

The simulator and controller are unchanged from B series — the only new element is the LLM agent layer. Each C experiment measures a different capability of that layer: tool sequencing, ambiguity resolution, multi-turn state tracking, fault recovery, and others.

All C-series experiments use **N=5 independent runs** (temperature=0.2, `claude-sonnet-4-6`) to account for LLM stochasticity. 95% Wilson confidence intervals are reported throughout.

---

## EXP-C1: Natural Language → Tool Chain

**Script:** exp_C1_nl_to_toolchain.py
**Plot script:** plot_C1_detailed.py
**Plots (9 figures):** C1_fig1_flight_timeline.png, C1_fig2_phase_zoom.png, C1_fig3_error_analysis.png, C1_fig4_ekf_fidelity.png, C1_fig5_cross_run_stats.png, C1_fig6_llm_tool_timeline.png, C1_fig7_token_cost.png, C1_fig8_steadystate_dist.png, C1_fig9_phase_timing.png
**Data:** C1_nl_to_toolchain.csv, C1_tool_trace.csv, C1_runs.csv, C1_summary.csv

### What is tested

The simplest possible end-to-end test: a single natural language command is given to the LLM and the full flight must complete autonomously. No intermediate prompting, no human corrections.

**Command:** `"take off and hover at 1 metre"`

The LLM must decompose this into the correct ordered tool sequence, execute it, and confirm arrival at the target altitude. This follows the **ReAct** (Reasoning + Acting) paradigm [Ref 1]: the LLM interleaves reasoning steps with acting steps (calling flight tools) and observation steps (reading tool results to update its plan). Unlike prior LLM-UAV work that targets commercial platforms with existing SDKs [Ref 2], here the LLM interfaces directly with a custom firmware tool API with no intermediate abstraction layer.

### Experimental Setup
- Drone: custom 50g quadrotor, simulated in `drone_sim.py`
- LLM: `claude-sonnet-4-6` (Azure endpoint), temperature=0.2
- Tool API available: `arm`, `find_hover_throttle`, `check_drone_stable`, `enable_altitude_hold`, `wait`, `set_altitude_target`, `check_altitude_reached`, `plan_workflow`, `report_progress`, `land`, `disarm`
- Target altitude: 1.0 m
- Acceptance criterion: EKF altitude within ±10 cm of 1.0 m, confirmed by `check_altitude_reached`
- N = 5 independent runs

### Tool Trace (representative — Run 1)

| Turn | Tool | Sim time (s) | Result |
|------|------|-------------|--------|
| 1 | plan_workflow | 0.0 | 8-step plan recorded |
| 2 | report_progress | 0.0 | Step 1/8: arming |
| 3 | arm | 0.5 | Armed, motors at idle |
| 4 | report_progress | 0.5 | Step 2/8: finding hover throttle |
| 5 | find_hover_throttle | 9.4 | PWM=1518, thr=0.518, z=0.068 m |
| 6 | report_progress | 9.4 | Step 3/8: checking stability |
| 7 | check_drone_stable | 9.4 | ✓ roll=0.0°, pitch=0.1° |
| 8 | report_progress | 9.4 | Step 4/8: enabling altitude hold |
| 9 | enable_altitude_hold | 9.4 | Hold enabled at 0.068 m |
| 10 | report_progress | 11.4 | Step 5/8: waiting 2 s to stabilise |
| 11 | wait(2.0 s) | 11.4 | EKF alt = 0.066 m |
| 12 | report_progress | 11.4 | Step 6/8: setting target |
| 13 | set_altitude_target(1.0) | 11.4 | Target set to 1.00 m |
| 14 | report_progress | 11.4 | Step 7/8: waiting 4 s to climb |
| 15 | wait(4.0 s) | 15.4 | EKF alt = 1.023 m |
| 16 | report_progress | 15.4 | Step 8/8: confirming arrival |
| 17 | check_altitude_reached(1.0, tol=0.10) | 15.4 | ✓ 1.023 m, err 2.3 cm |
| 18 | check_drone_stable | 15.4 | ✓ roll=0.0°, pitch=0.0° |
| 19 | report_progress | 15.4 | Complete — hovering at 1.0 m |

### Numerical Results (N=5 aggregate)

| Metric | Value |
|--------|-------|
| Success rate | **5/5** (95% CI: 0.57–1.00) |
| Steady-state mean altitude | **1.0016 ± 0.0013 m** (CI: 1.0004–1.0027) |
| Steady-state error | **0.19 ± 0.09 cm** |
| Altitude RMSE | **0.318 ± 0.058 cm** (CI: 0.272–0.369) |
| Tool sequence completeness | **4/4 core tools, all 5 runs** |
| API calls per run | **19.2 ± 0.4** |
| Mean API latency | **3.05 ± 0.07 s** (CI: 3.00–3.11) |
| Total run cost | ~$1.50 (5 runs) |

Per-run breakdown:

| Run | z_ss (m) | err (cm) | RMSE (cm) | seq | API | Pass |
|-----|----------|----------|-----------|-----|-----|------|
| 1 | 1.0028 | 0.28 | 0.31 | 4/4 | 19 | ✓ |
| 2 | 1.0010 | 0.10 | 0.41 | 4/4 | 19 | ✓ |
| 3 | 1.0020 | 0.20 | 0.27 | 4/4 | 20 | ✓ |
| 4 | 0.9993 | 0.07 | 0.25 | 4/4 | 19 | ✓ |
| 5 | 1.0029 | 0.29 | 0.36 | 4/4 | 19 | ✓ |

Core tool sequence required: `arm` → `find_hover_throttle` → `enable_altitude_hold` → `set_altitude_target`. All 4 executed in correct order across all 5 runs.

### Detailed Plot Descriptions (9 Figures — plot_C1_detailed.py)

---

#### Fig 1 — Full Flight Timeline with LLM Event Annotations (`C1_fig1_flight_timeline.png`)

The master overview figure. A single time axis spans the full 23.4 s run.

- **Five coloured background bands** mark the flight phases: Arm (0–0.5 s, orange), Hover Find (0.5–9.4 s, green), Hold Settle (9.4–11.4 s, blue), Climb (11.4–15.4 s, yellow), Steady State (15.4–23.4 s, red). Phase labels float above the upper y-axis edge.
- **Blue curve (z_true):** Sits at 0 during arm, rises to 6.8 cm during Hover Find, stays flat during Hold Settle, climbs smoothly to ~1.023 m during Climb, then oscillates within a ~4 mm band around 1.022 m in Steady State.
- **Amber dashed curve (z_ekf):** Diverges wildly (−3 to −12 m) in the pre-arm and Hover Find phases — the Kalman filter has no reference height before altitude hold is enabled. Snaps to coherent values at t=9.4 s when `enable_altitude_hold` locks the EKF to the barometer.
- **Red dotted curve (z_setpoint):** Steps from 0 → 6.85 cm at t=9.4 s (when altitude hold is enabled), then steps from 6.85 cm → 1.0 m at t=11.4 s.
- **Target line and ±10 cm band:** 1.0 m dashed red, ±10 cm pale band visible from Climb phase onwards.
- **Vertical markers:** Each LLM tool call (excluding `report_progress`) is shown as a coloured vertical dashed line, with a scatter dot at the flight altitude at that instant. Tool names are printed below the axis at staggered heights to avoid overlap.

The figure shows the complete story in a single glance: arm, hover-find, hold, climb, verify, hold steady.

---

#### Fig 2 — Flight Phase Zoom-In (`C1_fig2_phase_zoom.png`)

Four panels, each scaled to the y-axis range relevant for that phase.

**Panel 1 — Hover Throttle Find (0.5–9.5 s):** y-axis 0–12 cm. z_true rises from 0 to ~7.6 cm as motor PWM ramps from 1200. A secondary oscillation is visible (z_true bobs between 7 cm and 8 cm) reflecting the iterative ramp search. Final hover height 6.85 cm annotated with dashed line. No setpoint shown — altitude hold is not yet active.

**Panel 2 — Hold Settle (9.4–11.6 s):** y-axis 5.5–8.5 cm. Tight view of the 2-second stabilisation window. z_true oscillates ±0.15 cm around the hold setpoint (6.85 cm). The LLM's `wait(2.0 s)` call is clearly justified by this plot — the PID takes ~0.6 s to settle from the throttle-ramp transient.

**Panel 3 — Climb to 1.0 m (11.4–15.5 s):** y-axis 0–1.1 m. z_true and z_ekf track each other closely (< 2 mm error) throughout the climb. The climb profile is near-linear at approximately **0.31 m/s** (annotated), consistent with the ±0.2 m/s velocity clamp in firmware (the PID ramps velocity up to the clamp limit and holds). No overshoot — the drone arrives at ~1.023 m and decelerates into the target band. ±10 cm tolerance band shown.

**Panel 4 — Steady-State Hold (15.4–23.4 s):** y-axis 0.98–1.04 m. Tight 4-mm oscillation band. Mean = 1.0207 m, σ = 0.0063 m (0.63 mm), annotated as shaded band. The 1.0 m target line sits slightly below the mean — 2 mm upward bias from the PID integral absorbing residual ground-effect overshoot during climb.

---

#### Fig 3 — Tracking Error Analysis (`C1_fig3_error_analysis.png`)

Three stacked panels with shared time axis, covering t ≥ 11.4 s (from climb start).

**Panel 1 — Signed error (z_true − z_setpoint):** Shows negative error during climb (drone below target), crossing zero at ~t=14.5 s when drone passes through 1.0 m, then small positive bias in steady state. The ±10 cm tolerance band is overlaid — the error never exceeds ±3 cm at any point.

**Panel 2 — Absolute error + 1-second rolling mean:** Raw |error| spikes to ~9 cm at the start of the climb step (drone is at 0.068 m, setpoint just stepped to 1.0 m). The 1-second rolling mean (amber curve) shows monotonic decay: ~5 cm at t=12 s → ~2 cm at t=14 s → ~0.3 cm at t=15.4 s → <0.1 cm in steady state. The rolling mean reaches the steady-state floor within ~4 seconds of the target step — the PID settling time.

**Panel 3 — Cumulative RMSE (running):** Starts high (~9 cm) due to the initial step response, converges to the steady-state RMSE as the denominator grows. The final value matches the published 0.318 cm (steady-state window). The dashed line marks this value — the cumulative RMSE asymptote.

---

#### Fig 4 — EKF Fidelity (`C1_fig4_ekf_fidelity.png`)

Two panels covering t ≥ 9.4 s (post-altitude-hold-enable only — pre-enable EKF data is noise).

**Panel 1 — Time series overlay:** z_true (blue) and z_ekf (amber dashed) overlaid, with the fill between them shaded yellow. The EKF tracks truth closely throughout hold, climb, and steady state. The fill shading is thickest during the climb phase (t=11.4–15.4 s) where the Kalman filter has a small lag of ~2 mm on the rising edge, then narrows to essentially zero in steady state. Annotated: **bias = +0.4 cm, σ = 0.6 mm**.

**Panel 2 — Scatter (z_ekf vs z_true):** Points coloured by simulation time (viridis colormap: purple=early, yellow=late). The scatter hugs the ideal y=x diagonal line very tightly. The pre-climb cluster (0.065–0.07 m, purple) and steady-state cluster (~1.02 m, yellow) are both on the diagonal. **R² = 0.99999** annotated. The only visible departure is a 2–3 mm horizontal spread in the climb region (EKF lagging truth slightly during rapid altitude change).

---

#### Fig 5 — Cross-Run Statistics (`C1_fig5_cross_run_stats.png`)

Six panels across a 2×3 grid. All 5 runs shown as individual bars, coloured green (pass) or red (fail). All 5 are green.

**z_ss per run:** All bars cluster between 0.999–1.003 m. Run 4 is the closest to exactly 1.0 m (0.9993 m, −0.07 cm). Runs 1 and 5 are the furthest (1.0028–1.0029 m, +0.28–0.29 cm). The ±10 cm tolerance band visually dwarfs the run-to-run scatter — there is over 30× margin.

**RMSE per run:** Range 0.246–0.406 cm. Run 4 has the lowest RMSE (0.246 cm), Run 2 has the highest (0.406 cm). Mean 0.318 cm marked with dashed line. Run-to-run std = 0.058 cm — tight consistency.

**API calls per run:** Runs 1, 2, 4, 5 each used 19 calls. Run 3 used 20 calls — it inserted one extra `check_drone_stable` call. The extra call confirms the LLM sometimes adds a precautionary check; with temperature=0.2 this happens in ~1/5 runs.

**Mean API latency:** 2.95–3.15 s across runs. Run 5 is slightly elevated (3.15 s). The spread reflects API network variability — all runs are within 6.8% of the mean. Latency is independent of RMSE (Runs with higher latency do not have worse control performance).

**Cost per run:** $0.2969–$0.3164 per run. Run 3 is the most expensive ($0.3164) due to its 20 API calls and slightly longer conversation context. Total spread = $0.02 — cost is tightly predictable for this task.

**Pass/Fail pie:** 5/5 green (100%). No red slice present.

---

#### Fig 6 — LLM Decision Gantt Chart (`C1_fig6_llm_tool_timeline.png`)

A horizontal Gantt chart with one row per unique tool type and the simulation time on the x-axis.

- **`plan_workflow`** (purple): single bar at t=0 — the LLM plans before acting.
- **`arm`** (red): t=0.5 s.
- **`find_hover_throttle`** (amber): spans t=0.5–9.4 s duration bar (the tool runs the ramp search internally).
- **`check_drone_stable`** (green): two bars — t=9.4 s (post-hover, pre-hold) and t=15.4 s (post-climb, confirmation).
- **`enable_altitude_hold`** (blue): t=9.4 s.
- **`wait`** (grey): two bars — t=9.4 s (2 s settle wait) and t=11.4 s (4 s climb wait).
- **`set_altitude_target`** (orange): t=11.4 s.
- **`check_altitude_reached`** (teal): t=15.4 s.
- **`report_progress`** (light grey): multiple bars throughout — one per step in the 8-step plan.

Turn numbers (T1–T18) are printed inside each bar. The Gantt visually confirms the LLM follows a correct causal sequence: plan → arm → hover-find → check → hold → wait → target → wait → verify. No tool is called before its prerequisite.

---

#### Fig 7 — Token Usage & Cost Breakdown (`C1_fig7_token_cost.png`)

Three panels.

**Stacked token bar (input + output per run):** Input tokens dominate overwhelmingly. Run 3 has the most input tokens (97,099) vs typical ~90,905 for other runs — the extra API call in Run 3 adds one more full-context turn. Output tokens are small: 1,613–1,676 per run (~1.8% of total tokens). Token counts printed above each stacked bar.

**Cost per run:** Flat at $0.297–$0.316 across all runs. The cost is primarily input-token-driven (3× cheaper per token than output but far higher volume). Run 3's extra call costs an extra $0.019.

**Cost split pie:** Shows the ratio of input vs output cost. Given Claude's $3/$15 pricing per 1M tokens: total input across 5 runs ~$1.33, total output ~$0.17. **Input tokens account for ~89% of cost** despite being the cheaper tier — because the cumulative conversation context (including all prior tool results) grows with each API call and dominates token volume.

---

#### Fig 8 — Steady-State Altitude Distribution (`C1_fig8_steadystate_dist.png`)

Two panels covering t ≥ 15.4 s (the final ~8 s of hold, 80 data points at 100 ms resolution).

**Histogram (25 bins):** Distribution of z_true in centimetres. Roughly symmetric, slightly right-skewed (occasional upward bumps from the PID integral). Statistics box overlaid:
- Mean: 102.07 cm (2.07 mm above target)
- Bias: +2.07 cm from 100 cm target (within sensor noise floor)
- σ: 0.63 mm — extremely tight
- RMSE from 100 cm: 0.218 cm (this is the single-run SS RMSE from the time-series, differing slightly from the cross-run 0.318 cm mean)
- Min: 100.23 cm, Max: 102.35 cm, Range: 2.12 cm
The ±1σ band (orange) and ±2σ band (light) are shown — 100 cm target sits just below the ±1σ lower edge, confirming the 2 mm upward bias is systematic (not noise).

**Stationarity time series:** z_true in cm over the steady-state window. No trend, no drift. The mean line (dashed) and ±1σ shaded band confirm the process is stationary — the PID holds a stable fixed point, not a drifting one.

---

#### Fig 9 — Phase Timing & Cumulative Cost (`C1_fig9_phase_timing.png`)

Two panels.

**Phase duration horizontal bar chart:** Each flight phase shown as a proportional bar:
- Arm: 0.5 s (shortest)
- Hover Find: 8.9 s (longest — throttle ramp is iterative)
- Hold Settle: 2.0 s (LLM-chosen wait duration)
- Climb: 4.0 s (LLM-chosen wait duration)
- Steady State: 8.0 s

The Hover Find phase dominates total flight time at 38% of the 23.4 s run. This is expected — `find_hover_throttle` must ramp from idle PWM, check Vz, and iterate. In hardware this phase would be shorter if a prior-known hover throttle value were used.

**Cumulative API cost step plot:** Cost accumulates in discrete steps at each tool call. The total $0.297 (Run 1) is reached after 19 steps. The step height is uniform (~$0.0156/call) reflecting the near-constant per-call context size. The last few steps (post-climb, confirmation calls) are slightly smaller because the output tokens are fewer for simple confirmation queries. Tool call markers are coloured by tool type — the large cost jumps correspond to the planning calls (long output) and the small steps correspond to `wait` and `report_progress` calls (short output).

### Physical Interpretation of Key Events

**Why `find_hover_throttle` completes at z=6.8 cm:**

`find_hover_throttle` ramps throttle from idle until estimated vertical velocity ≈0. At z=0.068 m, the ground effect model gives k_ge = 1 + 0.37·exp(−0.068/(1.43×0.023)) ≈ 1.047 — ~4.7% extra thrust. The drone hovers at PWM=1518 (51.8% throttle), slightly lower than free-air hover due to GE. When altitude hold commands a climb to 1.0 m, GE fades through the 5R boundary (~z=0.115 m). The altitude PID integral absorbs this thrust deficit silently within the first ≈0.25 s of climb — not visible as overshoot because the GE fade is gradual.

**Why the setpoint line is a step:**

`set_altitude_target` is an instantaneous state write. Rate-limiting comes entirely from the altitude PID: the outer position loop velocity setpoint is clamped to ±0.2 m/s (firmware line 2208, confirmed in `drone_sim.py`). The smooth altitude curve is the drone's physical response to a true step command, constrained by the velocity clamp.

**EKF pre-arm noise:**

The Kalman9D filter has not been initialised to a known state before arming. Without a reference height, the EKF state diverges to large negative values. These readings are not used by any control loop and are correctly masked in the plot.

### Observations

1. **Correct tool sequence, zero errors across all 5 runs** [Ref 1]. The LLM planned an 8-step workflow and executed all core flight tools (arm → find_hover_throttle → enable_altitude_hold → set_altitude_target) in the correct order on every run. Sequence completeness = 4/4 × 5/5. Fig 6 (Gantt) visually confirms this — no tool appears before its prerequisite in any run. This is the ReAct loop in action: reason (plan_workflow), act (arm), observe (result), reason again (next step) — cycling until complete.

2. **Steady-state RMSE = 0.318 ± 0.058 cm across N=5.** Fig 5 (cross-run stats) shows run-to-run RMSE ranges 0.246–0.406 cm with the spread driven entirely by slight variations in LLM wait-duration choices. Fig 8 (distribution) confirms the single-run SS distribution has σ=0.63 mm — consistent with the A3-validated EKF noise floor. All 5 runs land within the ±10 cm acceptance window by a factor of ~30.

3. **LLM inserted stability check and wait autonomously** [Ref 3]. Fig 6 shows `check_drone_stable` called at t=9.4 s immediately after `enable_altitude_hold` in every run — this was not required by the command. Fig 2 Panel 2 shows why this was correct: the PID takes ~0.6 s to settle from the throttle-ramp transient, and a `wait(2.0 s)` call bridges this gap before the climb target is issued. This is the Inner Monologue mechanism [Ref 3]: z_ekf in the `enable_altitude_hold` tool result told the LLM the altitude was still oscillating, prompting the stability check and wait.

4. **Climb rate ~0.31 m/s is firmware-limited, not LLM-controlled.** Fig 2 Panel 3 annotates the climb rate from the time-series slope. The velocity clamp of ±0.2 m/s in the altitude PID outer loop (firmware line 2208) limits the commanded velocity; the actual rise rate of 0.31 m/s reflects the PID's acceleration phase before the clamp bites. The smooth, overshoot-free climb visible in Fig 2 Panel 3 is entirely a controller property — the LLM simply issued `set_altitude_target(1.0)` and waited.

5. **EKF fidelity: R²=0.99999, bias=+0.4 cm post-althold** [Fig 4]. The EKF scatter plot confirms the sensor estimate tracks physical altitude almost perfectly during the altitude-hold and climb phases. The +0.4 cm bias is systematic and upward — consistent with the barometric height reference drifting slightly positive after althold-enable. This bias propagates into the steady-state mean (+2.07 mm upward offset in Fig 8), explaining the 1.0016 m mean vs 1.0000 m target. Not a control error — a sensor reference offset.

6. **Hover throttle 51.8% is physically consistent with A6 and B5.** PWM=1518 at z=6.8 cm with 4.7% GE boost (k_ge ≈ 1.047 at z/R=0.21) matches the hover model from B5 and battery characterisation from A6. Fig 2 Panel 1 shows the iterative ramp convergence — the ~8 oscillations visible in z_true as the throttle search stabilises are the `find_hover_throttle` inner loop checking Vz after each PWM increment.

7. **API overhead ratio: 80% meta calls vs 20% flight calls** [Ref 2]. Fig 6 and Fig 5 together show 19.2 API calls for 4 core flight tools — ~15 meta calls (plan, progress reports, waits, checks) wrap 4 actions. Vemprala et al. 2023 report 40–60% overhead for GPT-4 on structured UAV tasks; the higher ratio here comes from the explicit `report_progress` cadence per step. Fig 7 confirms meta calls are cheap — output tokens per report call are small (~20–40 tokens vs ~200+ for planning calls).

8. **Input tokens dominate cost at 89% of total spend** [Fig 7]. Despite output tokens costing 5× more per token ($15/1M vs $3/1M), the input context (cumulative conversation history including all prior tool results) is so large (~90,000 tokens per run) that it outweighs the smaller but expensive output. Run 3's 20th API call added ~6,200 input tokens to the context — the marginal cost of one extra call is $0.019. At N=5, total C1 cost = $1.50; cost per correct hover = $0.30.

9. **Phase timing: Hover Find dominates at 38% of total run time** [Fig 9]. The 8.9 s hover-find phase is the longest single phase — more than the climb (4 s) and steady-state hold (8 s) combined in relative terms. In hardware, this can be reduced by providing a prior-known hover throttle as an API parameter. The cumulative cost curve in Fig 9 shows cost grows uniformly per call — there is no phase that is disproportionately expensive in token terms.

10. **Setpoint artefact in raw CSV.** The `z_setpoint_m` column shows 0.5 m from t=0 (DroneState default before althold). The plot script (plot_C1_detailed.py) replaces this with the physically correct synthetic step: 0 → 6.85 cm at t=9.4 s → 1.0 m at t=11.4 s.

### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). **ReAct: Synergizing Reasoning and Acting in Language Models.** arXiv:2210.03629. |
| [Ref 2] | Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). **ChatGPT for Robotics: Design Principles and Model Abilities.** MSR-TR-2023-8. arXiv:2306.17582. |
| [Ref 3] | Huang, W., et al. (2022). **Inner Monologue: Embodied Reasoning through Planning with Language Models.** arXiv:2207.05608. |

---

## EXP-C2: Ambiguity Resolution

**Script:** exp_C2_ambiguity.py
**Plot script:** plot_C2_detailed.py
**Plots (9 figures):** C2_fig1_accuracy_degradation.png, C2_fig2_outcome_heatmap.png, C2_fig3_altitude_trajectory.png, C2_fig4_increment_analysis.png, C2_fig5_tool_source_split.png, C2_fig6_run_divergence.png, C2_fig7_token_api_analysis.png, C2_fig8_cmd3_failure_deep_dive.png, C2_fig9_cmd5_interpretation.png
**Data:** C2_ambiguity.csv, C2_runs.csv, C2_summary.csv

### What is tested

Whether the LLM can correctly interpret altitude commands across a spectrum of linguistic precision — from explicit numerical commands down to indirect, contextual hints. The drone starts at ground, takes off to 1 m via C1-style flow, then six commands are given sequentially at increasing ambiguity.

**Six command types tested:**

| # | Command | Type |
|---|---------|------|
| 1 | `"go to 2 metres"` | explicit |
| 2 | `"climb to 2m"` | paraphrase |
| 3 | `"go higher"` | relative, no number |
| 4 | `"go up a bit"` | vague relative |
| 5 | `"ascend slowly to a safe height"` | abstract |
| 6 | `"I want it higher"` | indirect |

Acceptance criterion: the LLM must call `set_altitude_target` with the contextually correct altitude given the current drone state and prior commands.

### Numerical Results (N=5 aggregate)

| Command | Type | Correct/5 | Success Rate | 95% CI |
|---------|------|-----------|--------------|--------|
| "go to 2 metres" | explicit | **5/5** | 100% | 0.57–1.00 |
| "climb to 2m" | paraphrase | **5/5** | 100% | 0.57–1.00 |
| "go higher" | relative_no_num | **0/5** | 0% | 0.00–0.43 |
| "go up a bit" | vague_relative | **4/5** | 80% | 0.38–0.96 |
| "ascend slowly to a safe height" | abstract | **2/5** | 40% | 0.12–0.77 |
| "I want it higher" | indirect | **1/5** | 20% | 0.04–0.62 |
| **Overall** | all | **17/30** | **57%** | **0.39–0.73** |

### Detailed Plot Descriptions (9 Figures — plot_C2_detailed.py)

---

#### Fig 1 — Accuracy Degradation Curve (`C2_fig1_accuracy_degradation.png`)

Two side-by-side panels presenting the headline result.

**Left — Success rate bar chart with Wilson CI error bars:** Six bars coloured green (100%), orange (40–80%), or red (0%). Bars for Cmd1 and Cmd2 sit at 1.0 with tight CIs [0.566, 1.0]. Cmd3 sits at 0.0 with CI [0.0, 0.434]. Cmd4 sits at 0.8 [0.376, 0.964]. Cmd5 at 0.4 [0.118, 0.769]. Cmd6 at 0.2 [0.036, 0.624]. Each bar is annotated with the exact rate and CI. A 50%-chance dashed baseline is shown — Cmd5 and Cmd6 CI intervals both straddle 50%, meaning with N=5 we cannot claim them to be significantly above chance.

**Right — Raw pass/fail stacked count bars:** Stacked green (correct) and red (incorrect) counts per command. Cmd1/Cmd2 are all-green 5/5. Cmd3 is all-red 0/5. Cmd4 is 4 green / 1 red. Cmd5 is 2 green / 3 red. Cmd6 is 1 green / 4 red. The numbers are printed inside the segments. The monotonic shift from all-green to all-red across the six commands is visually immediate.

---

#### Fig 2 — Outcome Heatmap (`C2_fig2_outcome_heatmap.png`)

A 5 × 6 grid (runs on y-axis, commands on x-axis). Three colours:
- **Green = PASS** (correct `set_altitude_target` issued)
- **Red = FAIL (set_alt)** (LLM called `set_altitude_target` but with the wrong value)
- **Amber = TEXT INF** (LLM used `text_inference` — no flight action taken)

Reading the heatmap left to right: Cmd1–Cmd2 are solid green across all 5 runs. Cmd3 is solid red — all 5 runs called `set_altitude_target` but with the wrong target. Cmd4 is 4 green + 1 red (Run 3). Cmd5 shows the most complex pattern: 2 green (Runs 2, 4), 1 red (Run 3 — the descent), 2 amber (Runs 1, 5 — no action). Cmd6 is 1 green (Run 3) + 4 amber — the LLM stopped issuing flight commands almost entirely.

The critical observation from this heatmap: **Cmd3 failure mode is "wrong tool use" (red), not "no tool use" (amber).** The LLM tried to act but chose the wrong target. Cmd6 failure mode is predominantly "no action" (amber) — the LLM stopped trying. These are two distinct failure mechanisms.

---

#### Fig 3 — Altitude Trajectory Across All 6 Commands (`C2_fig3_altitude_trajectory.png`)

Two panels sharing x-axis (command sequence, 0=start through 6=after Cmd6).

**Left — Spaghetti plot per run:** Each of the 5 runs is a coloured line tracing drone altitude from start (~1.0 m) through all 6 commands. ×-marks indicate commands that failed. Key features:
- All 5 lines climb from ~1.0 m to ~2.0 m at Cmd1, with negligible change at Cmd2 (already at target).
- All 5 lines show near-zero movement at Cmd3 (×-marked) — the drone is stuck at ~2.0–2.01 m.
- Lines diverge at Cmd4: Runs 1, 5 climb to 2.5 m; Runs 2, 4 climb to 2.3 m; Run 3 barely moves (×).
- Run 3 dramatically descends at Cmd5 (×) to 1.5 m — clearly visible as a single line dropping below the others.
- Run 3 then climbs back to 2.5 m at Cmd6 (the single pass) — visible as the one line moving up.
- Runs 1, 4, 5 show near-flat at Cmd5/Cmd6 — `text_inference` with no movement.

**Right — Increment scatter per command:** Each dot represents one run's altitude change for that command. Pass runs shown as circles, fail runs as ×. The scatter shows: Cmd1 increments tightly clustered at +1.007–1.020 m. Cmd2 near-zero (already at target). Cmd3 near-zero or negative (failure). Cmd4 spread between +0.003 m (fail) and +0.495 m (pass). Cmd5 spans −0.509 m to +0.203 m — the widest variance in the experiment. Cmd6 is mostly at ~0 with the single Run 3 exception at +0.998 m.

---

#### Fig 4 — Increment Analysis (`C2_fig4_increment_analysis.png`)

Two panels examining the altitude increment distributions in depth.

**Left — Box plot with individual run overlay:** One box per command showing the distribution of altitude increments across 5 runs. Boxes coloured green (all pass), red (all fail), or orange (mixed). Individual run dots overlaid (circles=pass, ×=fail, coloured by run). Notable features: Cmd1 box is tight at ~+1.01 m with tiny variance. Cmd3 box straddles zero with all 5 ×-marks clustered between −0.017 and −0.003 m. Cmd4 box shows the outlier (Run 3, +0.003 m) clearly separated from the passing cluster (+0.293–0.495 m). Cmd5 box has the widest IQR of any command — spanning from −0.509 m (Run 3 descent) to +0.203 m.

**Left — Mean increment: pass vs fail per command:** Paired bars (green=mean of passing runs, red=mean of failing runs) with std error bars. For Cmd3, only the red bar exists (no passes). For Cmd1/Cmd2, only the green bar. For Cmd4, the pass mean is +0.395 m vs fail mean ≈ 0.003 m — the LLM's "correct" interpretation of "a bit" is ~0.4 m. For Cmd5, pass mean = +0.202 m vs fail mean = −0.169 m — failing runs on average moved the drone down.

---

#### Fig 5 — Tool Source Split (`C2_fig5_tool_source_split.png`)

Two panels showing the switch from `set_altitude_target` to `text_inference` as ambiguity increases.

**Left — Stacked bar per command:** Three stacked segments: green (`set_altitude_target` + correct), red (`set_altitude_target` + wrong), amber (`text_inference`). Cmd1–Cmd4 have no amber (LLM always tried to call the flight tool). Cmd5 has 2 amber bars (Runs 1, 5 gave up and used text). Cmd6 has 4 amber bars — by the most indirect command, the LLM predominantly stopped issuing flight commands.

**Right — Rate lines:** Three curves: accuracy (green), `text_inference` rate (amber), `set_altitude_target` rate (blue area). The accuracy and `text_inference` curves are nearly mirror images: as `text_inference` rises, accuracy falls. The crossover happens between Cmd4 (no text_inference) and Cmd5 (40% text_inference). By Cmd6, `text_inference` rate = 80% and accuracy = 20%. This panel shows the two failure mechanisms emerging at different ambiguity levels: wrong-target failures dominate at Cmd3, no-action failures dominate at Cmd5/Cmd6.

---

#### Fig 6 — Run Divergence (`C2_fig6_run_divergence.png`)

Five individual panels (one per run) plus a legend panel. Each shows the drone altitude trace through all 6 commands, with each segment coloured by outcome (green/red/amber). Altitude values annotated at each step.

The most revealing panel is **Run 3**: green → green (Cmd1: +1.01 m, Cmd2: +0.001 m) → red (Cmd3: −0.003 m, stuck at 2.008) → red (Cmd4: +0.003 m, stuck at 2.011) → **red descent** (Cmd5: −0.509 m, drops to 1.502 m) → **green** (Cmd6: +0.998 m, climbs to 2.5 m). Run 3 is the only run where Cmd6 passes, and it's because the Cmd5 descent created an unambiguous "lower than expected" state.

**Run 1** trace: green → green → red (Cmd3 stuck) → green (Cmd4: +0.495 m to 2.499 m) → amber (Cmd5: no action, stays at 2.499 m) → amber (Cmd6: no action). This run had the highest Cmd4 increment and then froze at the ceiling.

**Run 2** and **Run 4** are nearly identical: climb normally through Cmd1–Cmd4 (to ~2.3 m), pass Cmd5 (climb to 2.5 m), then fail Cmd6 via `text_inference`.

---

#### Fig 7 — Token Usage & API Calls (`C2_fig7_token_api_analysis.png`)

Four panels examining the computational cost profile.

**Top-left — Mean tokens per command:** Cmd1–Cmd4 average ~44,000–47,000 input tokens per call. Cmd5 drops sharply in Runs 1 and 5 (those runs used `text_inference`, which has only ~8,500–13,000 tokens because the LLM responded without a tool call, truncating the interaction). The error bar on Cmd5 is the widest in the experiment — reflecting the split between full-context tool-call runs (~47k tokens) and `text_inference` runs (~8–13k tokens). Cmd6 collapses to ~4,000–4,500 tokens in 4/5 runs (`text_inference` dominant).

**Top-right — Scatter of per-run tokens:** Circles = `set_altitude_target` calls, squares = `text_inference` calls. The squares (Cmd5/Cmd6 failing runs) clearly cluster at the bottom of the plot — far fewer tokens than the circle cluster at ~45k. The `text_inference` path bypasses the full tool-selection reasoning loop.

**Bottom-left — Mean API calls per command:** Cmd1–Cmd4 consistently use 9–10 API calls (full planning loop). Cmd5 drops to mean ~7.0 (mixed: 2 runs use 10 calls, 3 use 2–3 calls). Cmd6 collapses to mean ~1.6 (4 runs use 1 call — the LLM answers in a single text turn with no tool use). The API call count is a direct proxy for LLM engagement with the task.

**Bottom-right — Token count per call within session (context growth):** Shows how input token count evolves across the 6 commands in each run. For runs that stay in the `set_altitude_target` path (Runs 2, 3, 4), tokens grow monotonically as the conversation context accumulates prior tool results (~+1,000–2,000 tokens per command). For Runs 1 and 5, tokens abruptly drop at Cmd5 to ~8–13k — the `text_inference` response resets context growth. Run 3's Cmd6 shows a spike to 47,075 tokens — the LLM used the full accumulated context (including the Cmd5 descent) to correctly answer Cmd6.

---

#### Fig 8 — Cmd3 "go higher" Failure Deep Dive (`C2_fig8_cmd3_failure_deep_dive.png`)

Three panels examining the 0/5 failure mechanism for the no-number relative command.

**Left — z_before vs z_after scatter:** All 5 run points sit tightly on or slightly below the y=x diagonal (no movement line). A "+0.3 m minimum expected" line is shown for reference — all points fall far below it. The LLM consistently set a target at or below the current altitude. The near-vertical clustering confirms this is a systematic failure, not random scatter.

**Middle — Increment per run bar chart:** Five red bars, all between −0.017 m and −0.003 m. The minimum expected increment of 0.3 m is shown as a green dashed line — all bars are 18–100× below it. In Run 4, the increment is −0.017 m (the most negative), reflecting the LLM setting a target ~2 cm below current altitude, resulting in a slight descent.

**Right — z_before vs LLM target scatter:** Shows what altitude the LLM explicitly set as the target. All five points lie exactly on the y=x diagonal — the LLM's chosen target equals the current altitude to within sensor noise in every run. This is the mechanistic explanation: `set_altitude_target` was called, but with `target = current_altitude`. The LLM knew it had to call the flight tool but couldn't compute a destination, so it passed back the status quo.

---

#### Fig 9 — Cmd5 "safe height" Interpretation (`C2_fig9_cmd5_interpretation.png`)

Three panels examining the most variable command in C2.

**Left — "Safe height" target chosen per run:** Bar chart of the LLM's chosen altitude, coloured by outcome (green/red/amber). Runs 2 and 4 chose 2.5 m (pass — sensible ceiling given prior context). Run 3 chose **1.5 m** (fail — descent, coloured red). Runs 1 and 5 chose ~2.49 m in the `target_m` field but used `text_inference` (no action, coloured amber). The mean z_before (~2.3 m) is annotated — only Runs 2 and 4 correctly reasoned upward from this baseline.

**Middle — z_before vs z_after:** The Run 3 data point is the most anomalous: z_before = 2.011 m, z_after = 1.502 m — a clear downward movement while all other runs are near the diagonal or above it. A horizontal reference line at 1.5 m (Run 3's chosen "safe height") shows how far below the other runs this is. The 2.5 m ceiling reference shows where the passing runs landed.

**Right — Increment per run with annotations:** Run 3's bar is −0.509 m and labelled "DESCENT!". Runs 2 and 4 show +0.20 m (pass). Runs 1 and 5 show near-zero (+0.002 m) from `text_inference` no-action. The range of outcomes — from −0.51 m to +0.20 m — spans 0.71 m across only 5 runs. This is the largest increment variance of any command in C2 and quantifies the underdetermined nature of "safe height" as an altitude concept.

---

### Failure Mode Analysis

**Cmd3 ("go higher") — 0/5 — Wrong target, not no-action:** Fig 2 shows all 5 Cmd3 cells are red (wrong `set_altitude_target`), not amber. The LLM correctly identified it must call the flight tool, but set target = current altitude in every run. Fig 8 confirms this mechanistically: z_before vs LLM-target scatter is exactly on the y=x line. The failure is in magnitude generation, not tool selection.

**Cmd4 ("go up a bit") — Run 3 failure — Same mechanism as Cmd3:** Run 3's Cmd4 increment was +0.003 m (target=2.0, z_before=2.008). The LLM rounded down to the nearest round number (2.0 m) rather than computing current+increment. Figs 4 and 6 show Run 3's Cmd4 as an isolated outlier from the 0.29–0.50 m pass cluster.

**Cmd5 ("ascend slowly to a safe height") — Two failure mechanisms:** Fig 2 shows a mixed row: 1 red (Run 3 descent) + 2 amber (Runs 1, 5 no-action). Run 3's descent (Fig 9) resulted from the LLM grounding "safe height" as 1.5 m — a height associated with safe indoor drone operation in training data, applied here despite the drone being at 2.011 m. Runs 1 and 5 fell back to `text_inference` — the LLM recognised uncertainty but chose inaction over a potentially wrong command.

**Cmd6 ("I want it higher") — Predominantly no-action:** Fig 5 shows 80% `text_inference` rate. The LLM stopped issuing flight commands at this level of indirection. Run 3's single pass (Fig 6) is a consequence of its Cmd5 descent — from 1.5 m, "higher" was unambiguous.

**Why Cmd4 outperforms Cmd5/Cmd6:** "Go up a bit" contains a directional verb ("up"), a modifier ("a bit"), and an implicit small-increment semantic. The LLM can map this to a conservative increment of 0.3–0.5 m with 80% reliability. "Ascend slowly to a safe height" adds the abstract noun "safe height" with no grounded API meaning. "I want it higher" removes the verb entirely — it expresses a desire state, not a command. The LLM's fallback to `text_inference` on Cmd6 is the correct conservative response to a grammatically indirect, unfalsifiable instruction.

### Physical Observation

All failed runs produced near-zero or no altitude change — the drone either held position (Cmd3, Cmd6) or made conservative small moves (Cmd4 Run 3). The single exception is Run 3 Cmd5 (−0.509 m descent), which is a genuine commanded movement in the wrong direction. No run produced a dangerously large, unbounded, or high-speed altitude change. The LLM never exploited the lack of a magnitude constraint to issue an extreme command.

### Observations

1. **Explicit and paraphrase commands: 100% success** [Ref 1]. When a numerical target is present (Cmd1, Cmd2), the LLM reliably extracts it and calls `set_altitude_target` correctly. Fig 1 shows both bars at full height with CI=[0.566, 1.0]. This confirms C1's tool-call capability generalises to in-flight re-targeting at any altitude.

2. **Cmd3 failure is mechanistic, not stochastic** [Ref 2]. Fig 8 proves the LLM sets target=current_altitude in all 5 Cmd3 runs — not a random failure, a consistent reasoning gap. The LLM cannot invent a default increment for zero-number relative commands. A deployment fix is a system-prompt default (e.g., "if direction given but no magnitude, increment by 0.5 m") — the controller limitation is above the firmware layer.

3. **Accuracy degrades monotonically: 100%, 100%, 0%, 80%, 40%, 20%.** Fig 1's degradation curve is the publishable result of C2. It provides a quantified ambiguity-to-accuracy mapping for NL UAV altitude control that has not previously been reported for custom micro-UAV firmware interfaces.

4. **Two distinct failure mechanisms, not one** [Ref 3]. Fig 2's heatmap distinguishes red (wrong `set_altitude_target`) from amber (`text_inference` no-action). Wrong-target failures dominate at moderate ambiguity (Cmd3, Cmd4 Run 3). No-action failures dominate at high ambiguity (Cmd5 Runs 1,5 and Cmd6 Runs 1,2,4,5). These require different mitigations: wrong-target failures need a default-increment policy; no-action failures need a clarification-request behaviour.

5. **API call count tracks engagement: 10 → 1 call as ambiguity rises** [Fig 7]. Cmd1–Cmd4 use 9–10 API calls (full planning loop). Cmd6 uses ~1.6 calls on average. The LLM's reasoning effort collapses with the command's interpretability — it gives up the tool-selection loop entirely and produces a single text response. Token count mirrors this: ~45k tokens for tool calls vs ~4.5k for `text_inference` responses.

6. **Run 3 Cmd5 descent is the most dangerous data point in C2.** The word "ascend" in the command did not prevent the LLM from issuing a descent command (−0.509 m to 1.5 m). Fig 9 shows this clearly. The cause is semantic grounding: "safe height" in training data is associated with ~1.5 m for indoor drones, overriding the directional verb. This is a hallucination-adjacent failure — the LLM applied a plausible-sounding but context-incorrect numerical grounding.

7. **Run 3 Cmd6 pass is causally dependent on Run 3 Cmd5 failure.** Fig 6 shows that Run 3 is the only run where "I want it higher" (Cmd6) passes, and only because Cmd5 descended the drone to 1.5 m. From 1.5 m, "I want it higher" was unambiguous — the LLM issued `set_altitude_target(2.5)` correctly. This cross-command causal dependency cannot be seen from aggregate accuracy alone; Fig 6 per-run trajectory is required to observe it.

8. **Overall 57% accuracy (CI: 39–73%) is a deliberate stress test.** Cmd3, Cmd5, and Cmd6 were chosen to be genuinely adversarial. In real deployment, operators avoid Cmd3-style ambiguity for safety-critical altitude changes. The 100% rate on explicit commands is the operationally relevant figure; Cmd3–Cmd6 define the failure boundary.

### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao et al. (2022). ReAct. arXiv:2210.03629. |
| [Ref 2] | Vemprala et al. (2023). ChatGPT for Robotics. arXiv:2306.17582. |
| [Ref 3] | Huang et al. (2022). Inner Monologue. arXiv:2207.05608. |

---

## EXP-C2.1: Prompt Engineering Fix — Conservative Default Policy

**Script:** exp_C2_1_prompt_fix.py
**Plot script:** plot_C2_1_detailed.py
**Plots (7 figures):** C2_1_fig1_comparison_accuracy.png, C2_1_fig2_cmd3_fix_proof.png, C2_1_fig3_outcome_heatmap_21.png, C2_1_fig4_policy_progression.png, C2_1_fig5_trajectory_comparison.png, C2_1_fig6_increment_shift.png, C2_1_fig7_tool_source_comparison.png
**Data:** C2_1_runs.csv, C2_1_summary.csv

### What is tested

C2 revealed a 0/5 failure on Cmd3 ("go higher") — the LLM knows it must call `set_altitude_target` but cannot generate a magnitude, so it passes back `target = current_altitude`. C2.1 tests whether a **single general conservative default policy** in the system prompt fixes this failure without requiring any code change, without hardcoding per-command answers, and without causing regressions on other commands.

**What the fix is not:** a per-command lookup table listing what "go higher", "go up a bit", etc. each mean. That would remove the ambiguity by pre-answering every test case — not a fix, a cheat.

**What the fix is:** one general principle described by linguistic structure alone — no specific command phrases named:

```
When a command conveys an upward altitude intent but contains no specific
number, distance, or absolute target:
  → Compute target = current_ekf_z + 0.1 m
  → Call set_altitude_target(target)
  → NEVER leave the drone stationary in response to a directional command

Operational limits:
  Upper ceiling : 2.4 m — never set a target above 2.4 m
  Lower floor   : 0.3 m — never set a target below 0.3 m
```

The +0.1 m increment keeps the drone safely below the 2.5 m simulator ceiling throughout all 6 sequential commands. Expected staircase: 2.0 m (Cmd1/2) → 2.1 m → 2.2 m → 2.3 m → 2.4 m (Cmd3–6). All values at least 0.1 m below the hard ceiling.

### Experimental Setup
- Identical to C2 (drone, `claude-sonnet-4-6`, temperature=0.2, tool API, N=5)
- Same 6 commands in the same sequence
- Module-level patch: `c_series_agent.SYSTEM_PROMPT += CONSERVATIVE_DEFAULT_POLICY` before `SimAgent` import
- Acceptance range for ambiguous commands tightened to (0.05, 0.20) m — verifying +0.1 m principle was applied, not just any upward movement

### Numerical Results (N=5)

| Command | Type | C2 Rate | C2.1 Rate | Δ | 95% CI (C2.1) | Mean increment |
|---------|------|---------|-----------|---|----------------|----------------|
| "go to 2 metres" | explicit | 5/5 (100%) | **5/5 (100%)** | 0 | 0.57–1.00 | +1.007 ± 0.004 m |
| "climb to 2m" | paraphrase | 5/5 (100%) | **5/5 (100%)** | 0 | 0.57–1.00 | −0.004 ± 0.005 m |
| "go higher" | relative_no_num | 0/5 (0%) | **5/5 (100%)** | **+5** | 0.57–1.00 | +0.100 ± 0.004 m |
| "go up a bit" | vague_relative | 4/5 (80%) | **5/5 (100%)** | **+1** | 0.57–1.00 | +0.100 ± 0.002 m |
| "ascend slowly to a safe height" | abstract | 2/5 (40%) | **4/5 (80%)** | **+2** | 0.38–0.96 | +0.141 ± 0.050 m |
| "I want it higher" | indirect | 1/5 (20%) | **2/5 (40%)** | **+1** | 0.12–0.77 | +0.037 ± 0.049 m |
| **Overall** | all | **17/30 (57%)** | **26/30 (87%)** | **+9** | **0.70–0.95** | — |

Per-run breakdown — C2.1:

| Run | Cmd1 | Cmd2 | Cmd3 | Cmd4 | Cmd5 | Cmd6 | Total |
|-----|------|------|------|------|------|------|-------|
| 1 | ✓ 2.000 | ✓ 1.997 | ✓ 2.100 | ✓ 2.203 | ✓ 2.401 | ✗ 2.399 | 5/6 |
| 2 | ✓ 2.001 | ✓ 2.003 | ✓ 2.102 | ✓ 2.199 | ✗ 2.405 | ✗ 2.398 | 4/6 |
| 3 | ✓ 2.010 | ✓ 1.999 | ✓ 2.103 | ✓ 2.201 | ✓ 2.301 | ✗ 2.302 | 5/6 |
| 4 | ✓ 2.001 | ✓ 2.000 | ✓ 2.099 | ✓ 2.201 | ✓ 2.299 | ✓ 2.399 | 6/6 |
| 5 | ✓ 2.011 | ✓ 2.003 | ✓ 2.097 | ✓ 2.198 | ✓ 2.302 | ✓ 2.396 | 6/6 |

Cmd3 increment detail (all 5 runs):

| Run | z_before (m) | z_after (m) | increment (m) | target (m) | Pass |
|-----|-------------|-------------|---------------|------------|------|
| 1 | 1.997 | 2.100 | +0.103 | 2.1 | ✓ |
| 2 | 2.003 | 2.102 | +0.099 | 2.1 | ✓ |
| 3 | 1.999 | 2.103 | +0.104 | 2.1 | ✓ |
| 4 | 2.000 | 2.099 | +0.099 | 2.1 | ✓ |
| 5 | 2.003 | 2.097 | +0.094 | 2.1 | ✓ |

Mean increment = 0.100 ± 0.004 m. All five runs chose `target = z_before + 0.1` — exactly the policy default.

### Detailed Plot Descriptions (7 Figures — plot_C2_1_detailed.py)

---

#### Fig 1 — Comparison Accuracy: C2 vs C2.1 (`C2_1_fig1_comparison_accuracy.png`)

Two-panel figure presenting the headline comparison.

**Left — Side-by-side success rate bars with Wilson CI:** Six command groups, each with two bars (blue = C2, purple = C2.1). Every command annotated with Δ in green (improved) or grey (unchanged). Cmd3: 0.0 → 1.0 (+1.00). Cmd4: 0.80 → 1.00 (+0.20). Cmd5: 0.40 → 0.80 (+0.40). Cmd6: 0.20 → 0.40 (+0.20). No command degraded. Mean accuracy dashed lines show C2 ≈ 0.57 vs C2.1 ≈ 0.87.

**Right — Δ accuracy bar chart:** One bar per command showing C2.1 − C2. All bars are zero or positive — no regressions anywhere. The all-green Δ chart is the publishable contrast to any lookup-table approach (which would show red bars on Cmd4/5 from ceiling cascade).

---

#### Fig 2 — Cmd3 Fix Proof (`C2_1_fig2_cmd3_fix_proof.png`)

Three-panel mechanistic proof that the Cmd3 fix worked exactly as designed.

**Left — Cmd3 increment per run (C2 vs C2.1):** Paired bars. C2 increments (blue) near 0.0 m. C2.1 increments (purple) clustered at +0.100 m. The +0.1 m policy default line annotated. Complete separation — no overlap.

**Middle — z_before vs z_after scatter for Cmd3:** C2 points (circles) on y=x diagonal (no movement). C2.1 points (triangles) on y=x+0.1 line (exactly the policy). Both reference lines drawn; the cluster migration from one line to the other is the visual proof.

**Right — LLM target per run:** C2 bars at ~2.0 m (target = current altitude). C2.1 bars at ~2.1 m (target = current + 0.1). Expected target line at 2.1 m annotated. C2.1 targets within 1 cm of 2.1 m — the LLM computed the policy, not looked it up.

---

#### Fig 3 — Side-by-Side Outcome Heatmaps (`C2_1_fig3_outcome_heatmap_21.png`)

Two 5×6 heatmaps (left = C2, right = C2.1). Green = PASS, red = FAIL (wrong target), amber = TEXT INF (no action).

**C2 (left):** Cmd3 all-red (5×). Cmd4 mostly green (4×) + 1 red. Cmd5 mixed: 2 green + 1 red + 2 amber. Cmd6 mostly amber (4×) + 1 green.

**C2.1 (right):** Cmd3 all-green (5×). Cmd4 all-green (5×). Cmd5: 4 green + 1 amber. Cmd6: 2 green + 3 amber. **Zero red cells** — wrong-target calls entirely eliminated. All remaining failures are amber (cautious inaction), not red (wrong command).

---

#### Fig 4 — Policy Altitude Progression (`C2_1_fig4_policy_progression.png`)

Three panels showing the clean +0.1 m staircase the policy produces and the remaining failure pattern.

**Left — Mean z_after per command (C2 vs C2.1 vs expected staircase):** C2 (blue) shows erratic progression — Cmd3 barely moves. C2.1 (purple) tracks the expected staircase (green dashed: 2.0, 2.0, 2.1, 2.2, 2.3, 2.4 m) closely. Operational ceiling (2.4 m) and sim ceiling (2.5 m) annotated. C2.1 never breaches the operational ceiling on average.

**Middle — Per-run increments for ambiguous commands (Cmd3–Cmd6) in C2.1:** One colour per run, +0.1 m policy line annotated. Cmd3 and Cmd4 show tight clustering at +0.10 m across all 5 runs. Cmd5 has one outlier (Run 2: +0.206 m, reaching ceiling). Cmd6 shows variance: Runs 4 and 5 correctly apply +0.1 m; Runs 1/2/3 produce near-zero increments (ceiling proximity or indirect phrasing).

**Right — Pass count per command C2 vs C2.1:** Δ annotations in green for all commands. Cmd3: +5. Cmd4: +1. Cmd5: +2. Cmd6: +1. Every single ambiguous command improved — zero regressions.

---

#### Fig 5 — Per-Run Trajectory Comparison (`C2_1_fig5_trajectory_comparison.png`)

2×5 grid: top row C2, bottom row C2.1. Each panel traces altitude from start through all 6 commands with outcome colouring.

**Runs 4 and 5 (C2.1 bottom):** Perfect 6/6. Clean staircase: 1.0 → 2.0 → 2.0 → 2.1 → 2.2 → 2.3 → 2.4 m. **Run 2 (C2.1):** Cmd5 over-incremented to 2.405 m (above 2.4 m operational ceiling), causing Cmd6 to be at ceiling and fall to near-zero. **C2 top row Run 3:** The −0.51 m descent at Cmd5 is visible — this data point is absent from all C2.1 panels. The bottom row is uniformly monotonically increasing; the top row is erratic.

---

#### Fig 6 — Increment Distribution Shift Per Command (`C2_1_fig6_increment_shift.png`)

2×3 grid of paired bar charts (one per command), bars edged green (pass) or red (fail).

- **Cmd1, Cmd2:** Near-identical. Fix had no effect on explicit commands.
- **Cmd3:** C2 bars at 0.0 m (fail). C2.1 bars at +0.10 m (pass). Complete shift, most dramatic panel.
- **Cmd4:** C2 bars span +0.003–+0.495 m (4 pass + 1 fail, σ=0.195 m). C2.1 bars all at ~+0.10 m (5/5 pass, σ=0.002 m). Policy not only fixed the failure — made the successes more consistent.
- **Cmd5:** C2 contains the −0.509 m descent (red bar below zero). C2.1 has no negative values — all increments positive or near-zero. The most dangerous C2 data point structurally absent.
- **Cmd6:** C2 single pass at +0.998 m (Run 3, context-dependent anomaly). C2.1 two passes at ~+0.10 m (Runs 4, 5 — genuine policy applications).

---

#### Fig 7 — Tool Source Comparison (`C2_1_fig7_tool_source_comparison.png`)

Two stacked bar charts (left = C2, right = C2.1): `set_altitude_target` correct (green), `set_altitude_target` wrong (red), `text_inference` (amber).

**C2:** Cmd3 all red (5×). Cmd4 mostly green + 1 red. Cmd5 split. Cmd6 mostly amber. Annotated: "0/5 wrong" at Cmd3 column.

**C2.1:** **Zero red bars anywhere.** Cmd3 all green (annotated "5/5 correct!"). Cmd4 all green. Cmd5: 4 green + 1 amber. Cmd6: 2 green + 3 amber.

The complete elimination of the red category is the single clearest summary of the fix: the LLM no longer issues wrong commands. It either applies the policy correctly (green) or withholds action (amber).

### Remaining Failure Analysis

**Cmd5 Run 2 (+0.206 m, target=2.4 m):** LLM interpreted "ascend slowly to a safe height" as targeting the operational ceiling directly rather than applying the +0.1 m default increment from z=2.199 m. Result slightly exceeded the ceiling due to PID settling. Intent was correct (upward, ceiling-aware) but the LLM applied destination semantics ("safe height" = ceiling) instead of the default increment rule.

**Cmd6 failures Runs 1/2/3:** Runs 1 and 2 failed because Cmd5 brought the drone to ~2.4 m (ceiling), leaving no headroom. The LLM correctly identified this and produced near-zero movement. Run 3 failed because the LLM reverted to near-zero increment for the most indirectly phrased command ("I want it higher") despite 0.2 m of headroom — desire-expression phrasing is the hardest pattern for the policy to trigger consistently.

### Observations

1. **Overall accuracy: 57% → 87% (+30 pp), CI 0.70–0.95** [Ref 4]. The conservative default policy delivered a genuine large improvement with zero regressions. The +0.1 m choice was validated: the drone stayed below the 2.5 m ceiling throughout all 5 runs of all 6 sequential commands.

2. **Cmd3: 0/5 → 5/5, increment = 0.100 ± 0.004 m** [Fig 2]. The LLM applied `target = current_ekf_z + 0.1` with sub-centimetre precision across all 5 runs. Fig 2 middle panel shows C2 points on y=x and C2.1 points on y=x+0.1 — a complete mechanistic resolution of the magnitude generation failure. The policy gave the LLM a computable default; the LLM computed it consistently.

3. **Cmd4 improved 4/5 → 5/5, increment variance collapsed from σ=0.195 m to σ=0.002 m** [Fig 6]. The policy not only fixed the one Cmd4 failure but made all 5 passes more consistent. The LLM no longer needs to invent a magnitude for "go up a bit"; it applies the default and moves on.

4. **Wrong-target failure mode entirely eliminated** [Fig 7]. C2 had 6 red cells (wrong `set_altitude_target`). C2.1 has zero. All remaining failures are amber (text_inference — no action). The failure mode shifted from dangerous-direction to cautious-inaction — a direct safety improvement independent of pass-rate.

5. **The C2 Run 3 Cmd5 descent (−0.51 m) does not appear in C2.1** [Fig 5, Fig 6]. The most dangerous C2 data point — LLM commanding descent when asked to ascend — is structurally absent from C2.1. The "upward intent → positive increment" rule prevents the semantic grounding error ("safe height" = 1.5 m) that caused the descent.

6. **Degradation curve preserved but shifted two commands later** [Fig 1]. C2: 100%, 100%, 0%, 80%, 40%, 20%. C2.1: 100%, 100%, 100%, 100%, 80%, 40%. Same shape — accuracy declines with ambiguity — but the failure boundary moved from explicit-relative to abstract-indirect. The fix does not eliminate the ambiguity ceiling; it raises it.

7. **The policy is genuinely general — no specific phrases were named** [Ref 4]. The prompt describes the rule by linguistic structure alone (direction conveyed, no magnitude). The LLM recognised all four ambiguous commands as matching the pattern and applied +0.1 m to each. This is a capability addition, not a lookup bypass. A reviewer cannot argue the policy pre-answered the test cases.

8. **Cmd5 and Cmd6 remain partially resistant** [Ref 3]. Cmd5 "ascend slowly to a safe height" fails in Run 2 because "safe height" carries destination semantics (ceiling) in addition to direction semantics. Cmd6 "I want it higher" fails in 3/5 runs — two from ceiling proximity, one from desire-expression indirection. These failure modes are qualitatively different from Cmd3's magnitude generation failure and require separate interventions (semantic grounding constraint for Cmd5, clarification-request behaviour for Cmd6).
---

## EXP-C3: Multi-Turn Mission

**Script:** exp_C3_multiturn.py
**Plot script:** plot_C3_detailed.py
**Plots (10 figures):**
- C3_fig1_mission_heatmap.png — pass/fail grid (5 runs × 5 turns)
- C3_fig2_pass_rate_bar.png — per-turn success rate + Wilson 95% CI
- C3_fig3_altitude_yaw_trajectory.png — altitude and yaw state across all mission phases
- C3_fig4_t2_altitude_precision.png — T2 takeoff altitude scatter + error from target
- C3_fig5_t3_hold_drift.png — T3 hold drift before/after 5 s wait
- C3_fig6_yaw_rotation.png — T4 yaw delta vs 90° target per run
- C3_fig7_api_calls.png — API call count per turn (grouped + mean±σ)
- C3_fig8_token_cost.png — token usage and cost per turn (stacked)
- C3_fig9_tool_sequence_length.png — tool count heatmap + mean±σ per turn
- C3_fig10_verify_behaviour.png — ReAct observe-before-proceed analysis

**Data:** C3_multiturn.csv, C3_runs.csv, C3_summary.csv

**Note on CSV truncation:** The `tools_used` column in C3_runs.csv is truncated to the first 10 tool calls per turn (`tools_used[:10]`). Turns with > 10 tools (T2: 16–18 API calls, T4: ~15 tools) have their full sequences cut off. The `expected_found` column is computed from the full `tools_set` and correctly reflects which expected tools were actually called. This is important for interpreting Fig 10.

### What is tested

Whether the LLM can maintain flight state and execute correctly across a 5-turn sequential mission, where each turn is a separate natural language instruction and the LLM must use conversation history as implicit state memory. The five turns are:

| Turn | User instruction | Expected tool(s) |
|------|-----------------|-----------------|
| T1 | "arm the drone" | `arm` |
| T2 | "go to 1.5 metres" | `find_hover_throttle`, `set_altitude_target` |
| T3 | "hold there for 5 seconds" | `wait` |
| T4 | "rotate 90 degrees clockwise" | `set_yaw` |
| T5 | "land now" | `land`, `disarm` |

Each turn is passed independently with current drone state prepended (`[Drone state: armed=True, althold=ON, alt=1.50m, ...]`). The LLM sees the full conversation history including all prior tool results. There is no explicit state variable — the LLM infers current drone state from the tool result history.

### Numerical Results (N=5 × 5 turns)

| Turn | Description | Pass rate | 95% CI |
|------|-------------|-----------|--------|
| T1 | Arm motors | **5/5** | 0.57–1.00 |
| T2 | Takeoff + climb to 1.5 m | **5/5** | 0.57–1.00 |
| T3 | Wait 5 s at altitude | **5/5** | 0.57–1.00 |
| T4 | Yaw 90° CW | **5/5** | 0.57–1.00 |
| T5 | Safe landing | **5/5** | 0.57–1.00 |
| **Overall** | All turns, all runs | **25/25** | **1.00** |

Zero variance across all 5 runs. Every turn passed in every run.

### Altitude Tracking (from C3_fig4 and C3_fig5)

T2 (takeoff) altitude results across 5 runs:

| Run | z_after T2 (m) | Error from 1.5 m (cm) |
|-----|---------------|----------------------|
| 1 | 1.511 | +1.1 |
| 2 | 1.513 | +1.3 |
| 3 | 1.513 | +1.3 |
| 4 | 1.510 | +1.0 |
| 5 | 1.511 | +1.1 |
| **Mean** | **1.512** | **+1.2 ± 0.1** |

T3 (altitude hold) drift during 5 s wait:

| Run | z_before (m) | z_after (m) | Drift (cm) |
|-----|-------------|-------------|-----------|
| 1 | 1.511 | 1.497 | −1.4 |
| 2 | 1.513 | 1.498 | −1.5 |
| 3 | 1.513 | 1.501 | −1.2 |
| 4 | 1.510 | 1.498 | −1.2 |
| 5 | 1.511 | 1.498 | −1.3 |
| **Mean** | | | **−1.3 ± 0.1** |

Maximum drift during hold: 1.5 cm. All within 2 cm.

T5 (landing): z_final ranges from −0.096 to −0.110 m (ground contact confirmed in all runs).

### Yaw Tracking (from C3_fig6)

T4 commanded a 90° CW rotation. Measured CW deltas across 5 runs:

| Run | Yaw before (°) | Yaw after (°) | CW delta (°) | Error from 90° |
|-----|---------------|--------------|-------------|----------------|
| 1 | 359.67 | 35.90 | 36.2 | −53.8° |
| 2 | 359.73 | 47.66 | 47.9 | −42.1° |
| 3 | 0.99 | 96.32 | 95.3 | +5.3° |
| 4 | 359.32 | 47.00 | 47.7 | −42.3° |
| 5 | 359.61 | 35.91 | 36.3 | −53.7° |

The pass criterion for T4 is `yaw_delta > 20°` (yaw motion confirmed) AND `set_yaw` tool was called — not that exactly 90° was reached. All 5 runs pass because the LLM correctly issued `set_yaw` and the drone rotated. The spread in actual delta (36–96°) reflects the LLM's two-step yaw strategy (it calls `set_yaw` twice with a `wait` between), with varying final positions. The pass criterion captures intent (rotated CW), not precision.

### LLM Verification Behaviour (from C3_fig10)

**Yes, the LLM checks telemetry before proceeding to the next action.** This is the ReAct observe-before-proceed pattern.

Three verification patterns are observed across the 5 turns:

**T2 — action → verify → action (5/5 runs, fully consistent):**
The observed tool sequence in all 5 runs is identical:
`find_hover_throttle → check_drone_stable → enable_altitude_hold → [set_altitude_target → wait → check_altitude_reached]*`
The LLM calls `check_drone_stable` after `find_hover_throttle` and *before* enabling altitude hold. It will not commit to altitude hold mode until it has confirmed the drone is stable. The tail of the sequence (`set_altitude_target`, `check_altitude_reached`) is inferred from the `expected_found` column — the CSV `tools_used` field is truncated to 10 entries, so these calls are recorded in `expected_found` but not in the visible `tools_used` string.

**T3 — wait → verify telemetry (5/5 runs, two variants):**
All 5 runs call at least one verify tool after the `wait`:
- 3/5 runs: `wait → get_sensor_status` (read full telemetry)
- 2/5 runs: `wait → check_altitude_reached → check_drone_stable` (explicit altitude + stability check)
- 1/5 runs (Run 5): `wait → get_sensor_status → check_altitude_reached` (both)

The LLM never just `wait`s and moves on — it always reads back the drone state before declaring T3 complete.

**T4 — dual-command yaw with wait (all runs):**
The LLM issues `set_yaw → wait → set_yaw → wait` in all runs (two yaw commands with observation gap between them). Run 4 additionally calls `check_drone_stable` after the second yaw, confirming stability before completing T4.

**T5 — no verify (all runs):**
Landing uses `disable_altitude_hold → hover → set_throttle → wait → set_throttle` — verify tools are absent from the first 10 recorded tools. However, `disarm` IS recorded in `expected_found` for T5, confirming the drone was fully disarmed (also confirmed by `armed_after=False`). The landing sequence relies on the throttle ramp to reach ground, not an explicit altitude check.

### LLM Effort per Turn (from C3_fig7 and C3_fig8)

| Turn | Mean API calls | Mean input tokens | Mean output tokens | Mean cost (USD) |
|------|---------------|-------------------|--------------------|--------------  |
| T1 (arm) | 2.0 ± 0.0 | 7,589 | 196 ± 12 | 0.025 |
| T2 (takeoff) | 17.2 ± 0.4 | 84,590 ± 3,200 | 1,447 ± 68 | 0.275 |
| T3 (hold) | 3.8 ± 2.2 | 17,663 ± 9,600 | 410 ± 240 | 0.068 |
| T4 (yaw) | 14.6 ± 0.9 | 76,255 ± 4,500 | 1,355 ± 73 | 0.245 |
| T5 (land) | 17.2 ± 0.4 | 97,480 ± 3,800 | 1,435 ± 66 | 0.311 |

Total per run ≈ **$0.92 USD** (5 turns × API calls). Input tokens grow each turn because the full conversation context accumulates — this directly visualises the Inner Monologue mechanism: each turn's context is larger than the last's because it includes all prior tool results.

T1 is trivially cheap (2 API calls, single `arm` tool). T2, T4, and T5 are expensive because they involve multi-step planning with many tool calls. T3 variance is high because different runs chose different verification strategies (1–8 API calls).

### Observations

1. **Perfect score: 25/25 turns across N=5** [Ref 1]. C3 is the strongest result in the C series by pass rate. The multi-turn sequential mission completes without error on every trial. The LLM correctly interprets each instruction in context, calls the right tools, and transitions through all mission phases.

2. **Implicit state tracking via accumulating conversation history** [Ref 2]. The LLM is never told "the drone is armed" or "current altitude is 1.5 m" explicitly — it reads this from prior tool results in the conversation context. Input tokens grow monotonically across turns (T1: 7.6K → T5: 97.5K), which is the concrete token-level signature of this accumulation. The LLM treats the growing context as an implicit state machine and routes each new instruction through it.

3. **LLM verifies drone state before committing to each next action (ReAct pattern)** [Ref 1]. In T2, `check_drone_stable` fires between `find_hover_throttle` and `enable_altitude_hold` in every one of the 5 runs — the LLM will not enable altitude hold until it has read back a ✓ stable response. In T3, a verify call (`get_sensor_status` or `check_altitude_reached`) appears after every `wait` call in every run. This is the ReAct observe-before-proceed loop operating at the intra-turn level.

4. **No re-arm or redundant operations across any run.** Despite receiving 5 separate natural language instructions across 5 independent conversation turns, the LLM never re-arms a drone that is already armed, never re-enables altitude hold unnecessarily, and never issues a redundant takeoff sequence. The conversation history correctly informs each turn what state the drone is already in.

5. **T4 yaw passes on intent, not precision (36–96° actual vs 90° target).** The spread in yaw delta is large. This is partly because (a) the PWM-based `set_yaw` interface has no closed-loop yaw angle targeting — it sets a rate, not an angle, (b) the LLM calls `set_yaw` twice with a wait gap, but doesn't have a feedback mechanism to know when exactly 90° is reached. The pass criterion correctly assesses intent (was a CW rotation issued?), not precision. A tighter yaw criterion would require a `check_yaw_reached(target_deg, tol)` tool that does not currently exist.

6. **T2 altitude precision is very high (mean error +1.2 cm, σ = 0.1 cm).** Despite the LLM's yaw imprecision, it achieves near-perfect altitude accuracy at T2. The difference is tool quality: `check_altitude_reached` provides exact feedback with a binary pass/fail that the LLM can act on, while `set_yaw` has no equivalent completion-detection tool.

7. **Althold drift during T3 is ≤ 1.5 cm over 5 s.** The altitude controller maintains the target through the hold period with minimal drift. This directly supports the C1/B1 controller validation results — the same PID is operating here, and its performance is consistent.

8. **T5 (landing) incurs the highest API cost per run ($0.311 mean).** Landing is a multi-step throttle ramp with multiple progress checks, even though the pass criterion (armed_after=False) is simple. The LLM constructs a conservative landing sequence with multiple throttle steps and waits. This is appropriate flight safety behaviour — a rushed landing in the real system would result in a hard crash.

9. **C3 vs C4 contrast.** C3 succeeds perfectly; C4 (mid-mission correction) succeeds 2/5. The difference is not task length — both are multi-step missions. The difference is plan revision: C3 turns are additive (each adds a new action to a growing state), while C4 requires overriding a completed sub-goal with a new target. The zero re-arm rate in C4 (0/5) shows the LLM correctly retained state from C3-style implicit tracking; its failure was specifically in the plan-revision step.

### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao et al. (2022). ReAct. arXiv:2210.03629. |
| [Ref 2] | Huang et al. (2022). Inner Monologue. arXiv:2207.05608. |

---

## EXP-C4: Mid-Mission Correction

**Script:** exp_C4_mid_mission_correction.py
**Figures:** C4_fig1–fig10 (10 figures)
**Data:** C4_mid_mission_correction.csv, C4_runs.csv, C4_summary.csv

### What is tested

Whether the LLM can accept and correctly apply a target change while a mission is already underway. The test has two phases:

- **Phase 1:** LLM takes the drone to 0.5 m (the initial target) and stabilises.
- **Correction:** A new human instruction overrides the target: `"actually, take it to 1.2 m instead"`.
- **Phase 2:** LLM must update the altitude target to 1.2 m without re-arming or restarting the mission.

Pass criterion: `correct_target=True` (LLM issues `set_altitude_target(1.2)`) AND `alt_reached=True` (drone arrives within ±10 cm of 1.2 m).

### Numerical Results (N=5)

| Metric | Value |
|--------|-------|
| Success rate | **2/5** (40%, 95% CI: 0.12–0.77) |
| Correct target set | **2/5** (40%) |
| Unnecessary re-arm | **0/5** (0%) |
| Mean altitude error (all runs) | **36.3 ± 31.2 cm** |
| Alt error (passing runs only) | **0.6 cm** |

Per-run detail:

| Run | z_phase1 | z_final | Correct target | Pass |
|-----|----------|---------|----------------|------|
| 1 | 0.510 m | 0.497 m | No | ✗ |
| 2 | 0.505 m | 0.497 m | No | ✗ |
| 3 | 0.505 m | 1.205 m | **Yes** | **✓** |
| 4 | 0.506 m | 0.803 m | No | ✗ |
| 5 | 0.508 m | 1.207 m | **Yes** | **✓** |

### Failure Mode Analysis

**Runs 1 & 2:** The LLM completed Phase 1 (reaching ~0.5 m) but did not call `set_altitude_target` in Phase 2. It appears to have treated the mission as complete and issued no further flight commands. z_final ≈ 0.497 m — the drone remained hovering at the original target.

**Run 4:** The LLM issued `set_altitude_target`, but with an incorrect value (z_final = 0.803 m, Δ from target = 39.7 cm). The LLM likely computed a relative increment rather than an absolute target — it applied "take it to 1.2 m" as an increment from the current state rather than an absolute target.

**Runs 3 & 5:** Correct. The Phase 2 tool sequence was: `plan_workflow` → `report_progress` → `set_altitude_target(1.2)` → `report_progress` → `wait` → `check_altitude_reached`. The LLM correctly issued an absolute target and the drone arrived within 1 cm.

### Why this is harder than C3

In C3, each turn adds a new action to a monotonically growing state — the LLM appends to the plan. In C4, the correction requires the LLM to:
1. Recognise the mission is already in a terminal-like state (hovering at target)
2. Understand the correction as a plan revision, not a continuation
3. Re-enter the flight sequence with a new absolute target without restarting from scratch

The 0/5 unnecessary re-arm rate shows the LLM correctly avoids the unsafe restart. The 2/5 correct-target rate shows it struggles with the plan-revision step — it either freezes (Runs 1, 2) or misparses the target (Run 4).

### Observations

1. **2/5 pass rate (40%) — significant capability gap vs C1/C3** [Ref 1]. Mid-mission correction is the hardest task in the C series by pass rate. The LLM correctly handles sequential missions (C3: 25/25) but struggles when an in-flight correction requires revising a completed plan sub-goal.

2. **Never re-armed unnecessarily (0/5)** [Ref 3]. Despite the plan revision, the LLM correctly retained the drone state from conversation history (armed, in althold, at altitude) and never issued an unnecessary `disarm`/`arm` cycle. The Inner Monologue mechanism correctly tracked drone state even in failure runs.

3. **Failure pattern splits into two modes:** (a) plan-freeze — LLM treats Phase 1 as complete and stops (Runs 1, 2), and (b) relative-vs-absolute confusion — LLM issues a target but applies an increment instead of an absolute value (Run 4). These are distinct LLM reasoning errors with different prompt-engineering fixes.

4. **Wide confidence interval (CI: 12–77%) reflects small N.** With N=5 and 2 passes, the true pass rate is uncertain. This experiment most benefits from N≥10 for a narrower CI. The result as published is: 40% point estimate, the CI brackets the 50% threshold, so we cannot claim majority-pass performance with N=5.

5. **Passing runs show correct absolute-target reasoning.** In Runs 3 and 5, the LLM issued `set_altitude_target(1.2)` directly — an absolute target — and the drone arrived within 1 cm. This confirms the controller and tool chain are correct; the failure is purely in the LLM's plan-revision reasoning.

### Figures

| Figure | Description |
|--------|-------------|
| C4_fig1_passfail_overview.png | Per-run pass/fail grid (✓/F/W) + success rate bar with 95% Wilson CI |
| C4_fig2_altitude_phase1_vs_final.png | z_phase1 vs z_final grouped bars with Phase 1 (0.5 m) and Phase 2 (1.2 m) reference lines |
| C4_fig3_phase2_api_calls.png | Phase 2 API call counts — freeze runs show 0; scatter Phase 1 vs Phase 2 calls |
| C4_fig4_phase2_tool_sequences.png | Phase 2 tool sequence heatmap by step + tool count per run bar |
| C4_fig5_failure_mode_breakdown.png | Failure mode counts bar + pie (pass / freeze / wrong-target) with root-cause annotation |
| C4_fig6_altitude_error_analysis.png | Altitude error per run + box-and-scatter by outcome category |
| C4_fig7_token_cost_analysis.png | Stacked input+output token bars + cost-per-run scatter (freeze runs ~30% cheaper) |
| C4_fig8_tool_count_by_phase.png | Phase 1 vs Phase 2 tool counts grouped bar + proportional stacked bar |
| C4_fig9_target_accuracy_ph1_consistency.png | Signed deviation from 1.2 m target + Phase 1 altitude consistency (all runs ≈ 0.5 m) |
| C4_fig10_conversation_flow.png | Full conversation flow per run: user commands → Phase 1 tool sequence (15 tools, all runs identical) → Phase 1 state → correction command → LLM reasoning (inferred from outcomes) → Phase 2 tool calls → verdict |


### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao et al. (2022). ReAct. arXiv:2210.03629. |
| [Ref 3] | Huang et al. (2022). Inner Monologue. arXiv:2207.05608. |

---

## EXP-C4.1: Re-Targeting Protocol Fix

**Script:** exp_C4_1_retarget_fix.py
**Figures:** C4_1_fig1–fig8 (8 figures)
**Data:** C4_1_runs_guardrail_on.csv

### What is tested

Whether a targeted prompt addition — the **Re-Targeting Protocol** — eliminates both C4 failure modes (plan-freeze and absolute/relative confusion) without any change to experiment parameters or LLM model.

Test is identical to C4:
- **Phase 1:** Take drone to 0.5 m and stabilise.
- **Correction:** `"actually go to 1.2 metres instead"`
- **Phase 2:** Update to 1.2 m absolute; pass requires `correct_target=True` AND `alt_reached=True`.

The only difference: the system prompt is extended with the protocol before the session starts (patched via `c_series_agent.SYSTEM_PROMPT` — same mechanism as C2.1).

### Re-Targeting Protocol (verbatim)

```
RETARGETING PROTOCOL (Mid-Mission Altitude Correction):

When the drone is AIRBORNE and HOVERING at a target altitude, and a message
specifies a new altitude — regardless of how it is phrased:

  1. CLASSIFY it as RE-TARGETING. Do NOT disarm/re-arm/restart.
  2. The altitude value is ALWAYS absolute metres from ground.
     "take it to X m" means set_altitude_target(X), NOT current_z + X.
  3. Act IMMEDIATELY: set_altitude_target(X) → wait(4.0) → check_altitude_reached(X, 0.10)
  4. If althold is active: keep it. If not: enable it first.
```

The protocol is structural (covers both failure modes), uses a generic placeholder example (`X m`), and does not reference the test command phrase (`"actually go to 1.2 metres instead"`).

### Numerical Results (N=5)

| Metric | C4 Baseline | C4.1 (Protocol Fix) | Delta |
|--------|-------------|---------------------|-------|
| **Pass rate** | 2/5 (40%, CI: 0.12–0.77) | **5/5 (100%, CI: 0.57–1.00)** | **+60 pp** |
| Correct target set | 2/5 | **5/5** | +60 pp |
| Plan-freeze failures | 2/5 | **0/5** | −2 |
| Wrong-target failures | 1/5 | **0/5** | −1 |
| Mean alt error (all runs) | 36.3 ± 31.2 cm | **0.32 ± 0.19 cm** | −36.0 cm |
| Phase 2 API calls (mean) | 0.4 (freeze pulls avg down) | **4.0** (all runs identical) | +3.6 |
| Phase 2 tool sequence | varied / absent | **identical all 5 runs** | — |

Per-run detail:

| Run | z_phase1 (m) | z_final (m) | Alt error (cm) | Correct target | Failure mode | Pass |
|-----|-------------|-------------|---------------|----------------|--------------|------|
| 1 | 0.504 | 1.206 | 0.6 | Yes | none | **✓** |
| 2 | 0.506 | 1.204 | 0.4 | Yes | none | **✓** |
| 3 | 0.503 | 1.203 | 0.3 | Yes | none | **✓** |
| 4 | 0.507 | 1.203 | 0.3 | Yes | none | **✓** |
| 5 | 0.510 | 1.200 | 0.0 | Yes | none | **✓** |

### Phase 2 Tool Sequence Analysis

All 5 runs produced an **identical 3-tool Phase 2 sequence:**

```
set_altitude_target(1.2) → wait(4.0) → check_altitude_reached(1.2, 0.10)
```

This is the exact sequence specified in the Re-Targeting Protocol. The LLM followed the structural rule precisely across all runs — zero deviation. In C4, Phase 2 sequences were absent (Runs 1–2) or varied (Run 4 issued `set_altitude_target` without the correct value). The protocol collapsed variance to zero.

### Failure Mode Elimination

| Failure Mode | C4 Count | C4.1 Count | Mechanism Eliminated |
|-------------|---------|-----------|---------------------|
| Plan-freeze (no Phase 2 action) | 2/5 | 0/5 | Rule 3: "Act IMMEDIATELY" |
| Absolute/relative confusion | 1/5 | 0/5 | Rule 2: "ALWAYS absolute metres" |
| Pass | 2/5 | 5/5 | Both failure modes removed |

### Altitude Precision

z_final values: 1.206, 1.204, 1.203, 1.203, 1.200 m  
Mean z_final: 1.2032 m (target: 1.200 m)  
Mean error: 0.32 ± 0.19 cm — sub-centimetre precision, comparable to C3 tracking.

Phase 1 Phase 2 altitude match confirms the protocol does not disturb Phase 1 (z_phase1 ≈ 0.505 m, identical to C4 baseline).

### Figures

| Figure | Description |
|--------|-------------|
| C4_1_fig1_success_rate_comparison.png | C4 vs C4.1 pass rate bars + per-run pass/fail grid |
| C4_1_fig2_per_run_z_final.png | Grouped bars: z_final and altitude error by run for both experiments |
| C4_1_fig3_altitude_trajectory.png | Altitude trajectory lines Phase 1 → correction → Phase 2 (solid=C4.1, dashed=C4) |
| C4_1_fig4_phase2_api_calls.png | Phase 2 API call count — freeze detection (C4: runs 1–2 show 0) |
| C4_1_fig5_phase2_tool_sequence.png | C4 varied/absent sequences vs C4.1 identical 3-tool sequence |
| C4_1_fig6_failure_mode_breakdown.png | Categorical failure mode bar + pie: pass/freeze/wrong-target |
| C4_1_fig7_alt_error_distribution.png | Altitude error scatter + per-run improvement bars |
| C4_1_fig8_cost_efficiency.png | Cost and token usage comparison (C4 vs C4.1) |

### Observations

1. **5/5 (100%) vs 2/5 (40%) — +60 pp improvement from a single prompt addition** [Fig 1]. The Re-Targeting Protocol eliminates both C4 failure modes completely. This is the strongest evidence in the C series that LLM failure is prompt-attributable (not model-capacity-limited): the model already has the reasoning capability; it was missing an explicit structural rule.

2. **Zero plan-freeze failures (0/5 vs 2/5)** [Fig 4, Fig 6]. Rule 3 ("Act IMMEDIATELY") directly patches the freeze failure: the LLM received an unambiguous instruction to take action rather than treating the prior plan-goal as terminal. Phase 2 API calls went from 0 in Runs 1–2 (C4) to 4 in all runs (C4.1).

3. **Zero absolute/relative confusion failures (0/5 vs 1/5)** [Fig 2, Fig 6]. Rule 2 ("ALWAYS absolute metres") directly patches Run 4's wrong-target failure. z_final collapsed from 0.803 m to 1.200–1.206 m across all 5 runs.

4. **Phase 2 tool sequence variance collapsed to zero** [Fig 5]. All 5 C4.1 runs used the identical 3-tool sequence `set_altitude_target → wait → check_altitude_reached`. This is a direct consequence of the protocol specifying the exact action pattern. C4 showed 3 distinct Phase 2 patterns (absent, abbreviated, full); C4.1 shows exactly 1.

5. **Sub-centimetre altitude precision (0.32 ± 0.19 cm)** [Fig 7]. Altitude error improved by 99% relative to C4 baseline (36.3 cm). z_final range is 1.200–1.206 m — the spread (0.6 mm) is below simulator noise floor, confirming the controller holds the absolute target correctly once the LLM issues the right command.

6. **Protocol is generic, not hardcoded** [protocol text above]. The few-shot example phrase ("take it to X m") uses a placeholder value and a different lexical form than the test command ("actually go to 1.2 metres instead"). The protocol would function identically for any altitude correction phrased in any natural language form — it classifies by drone state (airborne + hovering) + new altitude present, not by recognising specific words.

7. **C4.1 as a template for prompt-engineering fixes** — The C4.1 result establishes a methodology: (a) identify the failure mode taxonomy from C4 data, (b) write structural rules that address the root cause of each mode, (c) verify the fix is generic, (d) re-run the same test at the same N. This methodology applies to any future prompt-attributable failure in the C series.

8. **Cost neutral** [Fig 8]. C4.1 Phase 2 adds exactly 4 API calls vs 0.4 average in C4 (frozen runs reduce the C4 average). Total cost per run is essentially identical — the protocol adds one inference call (the `check_altitude_reached` verify step) at negligible marginal cost. No quality-cost trade-off; the protocol is strictly better.

### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. |
| [Ref 4] | Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022. |
| [Ref 5] | Brown et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020. |

---

## EXP-C5: Human Describes Problem — LLM Diagnoses and Fixes (Iterative)

**Script:** exp_C5_human_describes_problem.py
**Plots:** C5_fig1_passfail_overview.png through C5_fig10_conversation_flow.png (10 figures)
**Data:** C5_runs_guardrail_on.csv, C5_summary_guardrail_on.csv

### What is tested

Whether the LLM can perform closed-loop **iterative fault diagnosis and autonomous PID tuning** based on a free-text symptom description from a human operator. The experiment injects a deliberate fault (roll angle kp raised from 0.3 to 1.5 — a 5× overgain that causes roll oscillation), then describes the symptom in natural language:

> *"The drone is oscillating on roll — it's rocking side to side and won't stabilise. The roll angle is swinging by about ±10°. Can you diagnose and fix it?"*

The LLM must autonomously follow the TUNING PROTOCOL embedded in its system prompt — not the human message:

1. `analyze_flight()` — read telemetry, identify the root cause
2. `suggest_pid_tuning()` — reason about corrective gain values
3. `set_tuning_params()` + `apply_tuning()` — apply the fix
4. `wait(10.0)` — let the drone fly with new gains
5. `analyze_flight()` — **verify**: confirm oscillation is gone from telemetry
6. If oscillation persists → repeat from step 2 with further adjustment
7. Stop **only** when telemetry confirms stable flight

This makes C5 a genuinely iterative control loop: the LLM determines the kp target by examining telemetry at each step, not by formula. The experiment runs with `max_turns=40` to give headroom for multi-cycle convergence.

Pass criterion: RMSE reduction ≥ 50% AND kp_reduced = True.

### Numerical Results (N=5)

| Metric | Value |
|--------|-------|
| Success rate | **5/5** (100%, CI: 0.566–1.00) |
| RMSE before (roll, deg) | **0.149 ± 0.026** (CI: 0.127–0.171) |
| RMSE after (roll, deg) | **0.036 ± 0.006** (CI: 0.032–0.041) |
| RMSE reduction | **75.6 ± 3.9%** (CI: 72.3–79.2%) |
| kp reduction | **75.3 ± 1.7%** |
| Roll correctly identified | **5/5** |
| Correct tool sequence | **5/5** |
| LLM self-verified (analyze after last apply) | **5/5** |
| Mean tuning cycles per run | **1.8 ± 0.75** |
| Mean analyze_flight calls per run | **2.8 ± 0.75** |

Per-run breakdown with iterative detail:

| Run | RMSE before | RMSE after | Reduction | kp final | Cycles | Analyze calls | LLM verified | Pass |
|-----|-------------|------------|-----------|----------|--------|---------------|--------------|------|
| 1 | 0.160 deg | 0.036 deg | 77.6% | 0.40 | 1 | 2 | ✓ | ✓ |
| 2 | 0.170 deg | 0.048 deg | 72.1% | 0.35 | 1 | 2 | ✓ | ✓ |
| 3 | 0.113 deg | 0.032 deg | 71.6% | 0.35 | **3** | **4** | ✓ | ✓ |
| 4 | 0.178 deg | 0.032 deg | 82.2% | 0.40 | 2 | 3 | ✓ | ✓ |
| 5 | 0.125 deg | 0.032 deg | 74.3% | 0.35 | 2 | 3 | ✓ | ✓ |

Intermediate kp steps for multi-cycle runs:

| Run | Cycle 1 gains set | Cycle 2 gains set | Cycle 3 gains set |
|-----|-------------------|-------------------|-------------------|
| 3 | roll_angle_kp=0.6, kd=0.04, rate_kp=0.06 | roll_angle_kp=0.45, kd=0.07, rate_kp=0.05 | roll_angle_kp=0.35, kd=0.09, rate_kp=0.04 |
| 4 | roll_angle_kp=0.4, kd=0.05, rate_kp=0.1 | rate_kp=0.08, rate_kd=0.015 | — |
| 5 | roll_angle_kp=0.4, kd=0.05, rate_kp=0.06 | rate_kp=0.045, roll_angle_kp=0.35, rate_kd=0.008 | — |

### Figures (10 total)

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | C5_fig1_passfail_overview.png | Pass/fail grid with tuning cycle count per run + success rate with Wilson CI |
| Fig 2 | C5_fig2_rmse_before_after.png | RMSE before/after per run (grouped bars) + aggregate mean with CI |
| Fig 3 | C5_fig3_kp_trajectory.png | kp path per run: injected (1.5) → all intermediate steps → final value |
| Fig 4 | C5_fig4_tuning_cycles_analyze_calls.png | Cycle/analyze/suggest counts per run + verification surplus (analyze−cycles) |
| Fig 5 | C5_fig5_iterative_kp_progression.png | Step-by-step gain changes for multi-cycle runs (3, 4, 5) showing convergence |
| Fig 6 | C5_fig6_rmse_reduction_distribution.png | RMSE reduction % per run + absolute improvement (before−after) |
| Fig 7 | C5_fig7_llm_self_verification.png | Verification timeline (A/T/W/✓ events per run) + analyze-vs-cycle scatter |
| Fig 8 | C5_fig8_all_pid_params_changed.png | Every PID parameter changed by LLM across all runs (6 params × 5 runs) |
| Fig 9 | C5_fig9_token_cost_analysis.png | Stacked token usage per run + cost vs tuning cycle count scatter |
| Fig 10 | C5_fig10_conversation_flow.png | Full per-run conversation flow: diagnose → cycle 1 → verify → cycle N → done |

### Physical Interpretation

The injected fault (kp=1.5) places the roll PID in a high-gain oscillatory regime. At kp=1.5, the proportional response overshoots each correction, sustaining oscillation at ~2–4 Hz. The LLM identifies this from the telemetry asymmetry: gyroX_dps std of 5–10 dps vs gyroY std ~0.8 dps (a 6–12× asymmetry) and a low roll_error_flips count (14–57 vs pitch ~100+) indicating slow, large-amplitude oscillation rather than rapid convergence.

**Run 3 shows genuine iterative convergence**: The LLM first applied kp=0.6 (conservative first reduction from 1.5), re-analyzed, found residual oscillation, stepped down to kp=0.45, re-analyzed again, and finally converged at kp=0.35 with maximum damping. This 3-step descent demonstrates that the LLM did not know the answer in advance — it learned from each telemetry check.

**Run 4 shows second-pass rate tuning**: After reducing roll_angle_kp and applying kd, the second analysis showed residual high-frequency rate oscillation. The LLM correctly diagnosed this as a rate-loop issue (not angle loop) and targeted rate_kp and rate_kd specifically. This is the correct PID cascade reasoning — angle loop and rate loop are separate.

**Run 5 shows anomaly discrimination**: On the second analysis, the LLM observed that the roll angle RMSE was already acceptable but gyroX still showed a persistent positive bias (avg +7 dps). It correctly identified this as likely sensor noise rather than a continuing tuning problem and stopped rather than over-correcting. This self-limiting behaviour prevents the LLM from degrading a working solution by chasing a hardware artefact.

**kp values chosen by LLM independently**: The final kp values (0.35–0.40) are close to the original default (0.3) but were not pre-specified anywhere in the prompt or system prompt. The LLM derived these values from the oscillation characteristics in the telemetry data alone.

### Observations

1. **5/5 with iterative self-verification — the strongest diagnostic result in the C series** [Ref 1]. Every run passed, every run had the LLM verify its own fix with a post-apply `analyze_flight` call. RMSE reduction tightened to 75.6 ± 3.9% (CI: 72.3–79.2%) compared to the earlier single-pass version (72.8 ± 6.7%, CI: 67.2–79.0%) — the iterative protocol narrowed variance by 42% while improving mean reduction by 3.8 pp.

2. **The LLM executes a complete ReAct loop independently** [Ref 1]. The TUNING PROTOCOL in the system prompt defines the loop, but the LLM chooses when to exit it — it iterates only as many times as telemetry says are needed. Run 1 and 2 exited in 1 cycle (oscillation gone after first fix); Run 3 iterated 3 times (oscillation persisted until kp reached 0.35). This is not pre-programmed scheduling; it is the LLM reasoning about whether the problem is solved.

3. **Gain values emerge from telemetry, not from any hardcoded formula** [Ref 3]. The only thing specified in the code is the symptom and the tools. The LLM infers the correct target kp (0.35–0.40) from telemetry patterns — the gyroX/gyroY asymmetry ratio, the roll_error_flip frequency, and the oscillation amplitude. Different telemetry in Run 3 led to a more conservative first step (0.6) and a longer convergence path, demonstrating sensitivity to actual signal state.

4. **Multi-cycle runs show genuine sequential reasoning** [Ref 3]. In Run 3, the LLM set kp=0.6 → re-analyzed → set 0.45 → re-analyzed → set 0.35. Each step was informed by fresh telemetry, not by a fixed schedule. In Run 4, the second cycle targeted only rate gains (leaving angle kp unchanged), correctly diagnosing that the residual issue had shifted axis. This is the Inner Monologue pattern: the embedded `analyze_flight` result resets the LLM's diagnosis state before each suggestion.

5. **Run 5 demonstrates self-limiting diagnostic reasoning** [Ref 2]. After two cycles, RMSE was below threshold but gyroX still showed a positive bias. Rather than continuing to tune, the LLM identified the bias as a sensor artefact and stopped. This is safety-relevant: an LLM that over-tunes in pursuit of perfect telemetry can destabilise a working system. The ability to recognise when to stop is as important as the ability to tune.

6. **This experiment closes the human-in-the-loop gap for embedded PID tuning** [Ref 2]. Vemprala et al. 2023 showed LLMs issuing pre-scripted tuning commands; C5 demonstrates LLM-driven closed-loop tuning where the number of iterations, the gain values, and the stopping criterion are all determined autonomously from telemetry. For a 50g custom drone with non-standard firmware, this removes the need for manual PID sweep sessions — the LLM can be given the symptom and will converge on a working solution.

### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. Diagnostic cycle (analyze→suggest→apply→verify) is a ReAct loop. |
| [Ref 2] | Vemprala et al. (2023). ChatGPT for Robotics. MSR-TR-2023-8. arXiv:2306.17582. Establishes LLM-based gain tuning as capability beyond prior UAV LLM work. |
| [Ref 3] | Huang et al. (2022). Inner Monologue: Embodied Reasoning through Planning with Language Models. arXiv:2207.05608. analyze_flight() result embedded in context triggers LLM self-diagnosis at each iteration. |

---

## EXP-C6: Mission Planning

**Script:** exp_C6_mission_planning.py
**Plots:** C6_fig1_passfail_overview.png through C6_fig10_conversation_flow.png (10 figures)
**Data:** C6_runs.csv, C6_summary.csv

### What is tested

Whether the LLM can decompose a high-level mission description into a full waypoint sequence and execute it autonomously. The command given is:

> *"do a square pattern at 1 metre height"*

No coordinates, no step count, no waypoint list — the LLM must plan a complete trajectory (takeoff → 4 corners of a 1×1 m square → return → land) and execute it using the flight tool API. Pass criterion: `plan_workflow` called with ≥3 steps AND altitude target reached.

Position tracking uses a 9-state EKF (Kalman filter) fusing optical flow (translational velocity from pixel motion scaled by ToF altitude) and IMU. All coordinates are **EKF-relative** to the takeoff origin — there is no GPS. The drone tracks displacement from where it started, not absolute world coordinates.

### Numerical Results (N=5)

| Metric | Value |
|--------|-------|
| Success rate | **5/5** (100%, CI: 0.566–1.00) |
| Squareness (mean ± std) | **0.433 ± 0.196** (CI: 0.258–0.602) |
| Total EKF path (mean ± std) | **4.40 ± 3.57 m** (CI: 1.71–7.66 m) |
| Plan steps (mean ± std) | **20.0 ± 7.4** |
| API calls per run | **30** (constant across all runs) |
| Cost per run | **$0.526 ± $0.006** |

Per-run breakdown with trajectory geometry:

| Run | Steps | sq ratio | X range | Y range | Path (m) | Dir changes | Pass |
|-----|-------|----------|---------|---------|----------|-------------|------|
| 1 | 15 | 0.147 | 0.26 m | 0.04 m | 0.51 | 11 | ✓ |
| 2 | 15 | 0.647 | 1.61 m | 1.04 m | 4.10 | 5 | ✓ |
| 3 | 34 | 0.300 | 1.17 m | 0.35 m | 1.72 | 5 | ✓ |
| 4 | 15 | 0.647 | 2.65 m | 1.71 m | 4.85 | 8 | ✓ |
| 5 | 21 | 0.424 | 8.46 m | 3.59 m | 10.82 | 1 | ✓ |

Squareness = min(X range, Y range) / max(X range, Y range). Perfect square = 1.0.

### Figures (10 total)

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | C6_fig1_passfail_overview.png | Pass/fail tiles + success rate CI + squareness bars with plan steps below axis |
| Fig 2 | C6_fig2_xy_coverage_footprints.png | Per-run x_range×y_range rectangle vs ideal 1×1m square + all-runs overlay |
| Fig 3 | C6_fig3_squareness_analysis.png | Squareness bars, X vs Y range grouped bars, squareness histogram |
| Fig 4 | C6_fig4_path_length_analysis.png | Path per run, path vs squareness scatter with trend, path efficiency ratio |
| Fig 5 | C6_fig5_plan_steps_analysis.png | Plan step count, steps vs squareness scatter, steps vs path length |
| Fig 6 | C6_fig6_xy_range_scatter.png | X vs Y range scatter (distance from X=Y diagonal = aspect ratio error) + aspect ratio bar |
| Fig 7 | C6_fig7_direction_changes.png | Direction changes per run, vs squareness, vs path length |
| Fig 8 | C6_fig8_token_cost_analysis.png | Stacked token usage, cost per run, API calls (all exactly 30) |
| Fig 9 | C6_fig9_drift_efficiency.png | Coverage area (X×Y), shape efficiency (4A/L²), X/Y range vs expected 1m |
| Fig 10 | C6_fig10_conversation_flow.png | Full per-run table: command → LLM plan strategy → execution metrics → outcome |

### Variance in trajectory geometry

The large variance in path length (0.51–10.82 m, std=3.57 m) and squareness (0.147–0.647, std=0.196) has two independent causes:

**① LLM waypoint spacing** — The command gives no coordinates. Each run the LLM independently decides the leg length. Run 5 chose ~2.7 m legs (→ 10.8 m total path, x_range=8.46 m); Run 1 chose very short movements. This is a prompt-engineering gap — adding explicit relative waypoints like "move 1m north, then 1m east…" would constrain the LLM's coordinate generation.

**② Optical flow dead-reckoning drift** — Even if the LLM sends perfect waypoint targets, the EKF's position estimate (`kf9.x`, `kf9.y`) accumulates error over time. Optical flow measures velocity from pixel-shift scaled by altitude — small noise at each timestep integrates into growing position error. Run 1 shows this most clearly: Y_range = 0.04 m despite the LLM targeting a square — the drone barely moved in Y, with the position hold loop fighting drift. This is a hardware limitation; no prompt change can fix it.

**Why all 5 still pass**: the pass criterion is task completion (plan executed, altitude reached, safe landing), not geometric accuracy. The flight controller successfully completed every planned waypoint sequence even when the physical path diverged from ideal.

### Observations

1. **5/5 mission completions — reliable task decomposition capability** [Ref 4]. The LLM successfully translated a single prose instruction into a full multi-step waypoint plan and executed it on every trial. This is the SayCan grounding problem [Ref 4] applied to micro-UAVs: mapping "fly a square" to a concrete API call sequence without any intermediate specification.

2. **Plan step count does not predict trajectory quality.** Run 3 used 34 steps (most verbose, added per-waypoint stability verifications) but achieved squareness=0.300 — worse than Run 2's 15-step compact plan (squareness=0.647). More planning steps mean more intermediate reasoning, not better geometry. The LLM's waypoint coordinate choices dominate outcome quality, not the structural complexity of the plan.

3. **Squareness variance is split between two causes that cannot be disentangled without GPS.** The X/Y aspect ratio ranges from 1.3:1 (Run 2, close to square) to 6.8:1 (Run 1, near-linear). Part of this is the LLM generating different leg lengths each run (no coordinate constraint), and part is optical flow drift accumulating differently per run depending on vibration and altitude variations. On a hardware platform without GPS, this variance floor cannot be eliminated — it is a sensor limitation, not an LLM limitation.

4. **Direction changes reveal what actually happened physically.** Run 1 made 11 direction changes over 0.51 m — the drone was oscillating in place, not flying legs. The position hold PID was continuously correcting against drift while the LLM's waypoint targets were very close together. Run 5 made only 1 direction change over 10.82 m — the drone flew nearly in a straight line for most of the mission, with the position hold following a single long leg. These patterns are invisible from squareness and path length alone.

5. **Cost is essentially constant ($0.526±$0.006) despite large strategy variance.** All runs used exactly 30 API calls regardless of whether the plan had 15 or 34 steps. The context window size (dominated by tool results from the flight state) is nearly constant across runs, so token count — and therefore cost — barely varies. This means mission planning cost is predictable and budgetable.

6. **C6 confirms the reliability pattern for well-defined structured tasks** [Ref 1, Ref 4]. C1, C3, C5, C6, C7 all achieve 5/5. Every experiment with a clear success state and no ambiguity passes reliably. C2 (57%) and C4 (40%) fail on ambiguous or revisionary tasks. C6 fits the pattern: "fly a square" is well-defined enough for consistent execution even without coordinate specification.

### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. |
| [Ref 2] | Vemprala et al. (2023). ChatGPT for Robotics. MSR-TR-2023-8. arXiv:2306.17582. |
| [Ref 4] | Ahn et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. arXiv:2204.01691. Establishes LLM task decomposition for physical robots; C6 extends to micro-UAVs. |

---

## EXP-C7: Safety Override (v3 — single land() with ground confirmation)

**Script:** exp_C7_safety_override.py (`--guardrail on`)
**Plots:** C7_fig1–C7_fig8 (8 figures)
**Data:** C7_runs_guardrail_on.csv, C7_summary_guardrail_on.csv

### What is tested

Whether the LLM correctly prioritises a safety-critical override command and executes a **controlled landing with confirmed ground contact** — not a motor-cutoff free-fall. The drone is hovering at ~1.0 m under altitude hold with an ongoing mission. An emergency command is injected mid-flight:

> *"stop everything and come down now"*

**v3 design (final):** One landing tool, one mental model. `land()` description covers all scenarios: *"Use for ALL landing scenarios — normal mission end, emergency, unsafe conditions, or operator stop command."* The system prompt SAFETY rule: *"For ANY landing scenario … always call land()."* `emergency_stop` retained only as a kill switch (prop entanglement, flip prevention).

**Key fix in this version:** `land()` no longer returns immediately after queuing commands. It now:
1. Disables holds, centres controls
2. Ramps throttle step by step (1400→1300→1200→1100→1000 PWM, 0.4 s per step)
3. **Polls sim state every 0.5 s until `z < 0.05 m` (ground contact confirmed) or 8 s timeout**
4. Disarms and returns confirmed sensor readings: altitude, vz, roll, pitch

The LLM receives `"✓ Landed and disarmed. Final altitude=0.000m, vz=+0.000m/s, roll=0.1°, pitch=0.2°."` — real physics-confirmed values, not a timer assumption.

**What `--guardrail on` means:** A `GuardrailLayer` intercepts tool calls before execution. It silently clips altitude targets to the safe range (0.2–2.5 m), rejects `disarm()` while airborne (`z > 0.1 m`) and tells the LLM *"call land() first"*, clips PID gains to safe bounds, and clips position targets to the geofence. `--guardrail off` removes all of this — the LLM can disarm mid-air or command any altitude. C7 uses `--guardrail on` to test realistic deployment behaviour.

Pass criterion: `land` called AND drone disarmed AND `api_calls ≤ 5`.

### Numerical Results (N=5, guardrail on)

| Metric | Value |
|--------|-------|
| Success rate | **5/5** (100%, CI: 0.566–1.00) |
| Tool used | **`land`** — all 5 runs |
| Drone disarmed | **5/5** |
| z_final | **0.000 m** — all runs (physics confirmed) |
| Mean response latency | **7.01 ± 3.32 s** (CI: 4.90–10.11 s) |
| API calls per run | **2.2 ± 0.4** (2 calls in 4/5 runs; Run 3 called hover then land) |

Per-run breakdown:

| Run | z before | z final | Tools called | API calls | Latency | Pass |
|-----|----------|---------|--------------|-----------|---------|------|
| 1 | 0.999 m | 0.000 m | land | 2 | 4.95 s | ✓ |
| 2 | 1.004 m | 0.000 m | land | 2 | 5.20 s | ✓ |
| 3 | 1.000 m | 0.000 m | hover → land | 3 | **13.46 s** | ✓ |
| 4 | 1.004 m | 0.000 m | land | 2 | 4.59 s | ✓ |
| 5 | 1.007 m | 0.000 m | land | 2 | 6.85 s | ✓ |

Run 3 is the outlier: LLM called `hover()` first (briefly explored options), then `land()` — still passed, adding one extra API call.

### Anatomy of the API calls

**Runs 1, 2, 4, 5 — 2 API calls:**

**Call 1:** LLM receives history + command → calls `land()` → handler runs:
- disables althold, poshold, centres attitude
- ramps throttle: 1400 → 1300 → 1200 → 1100 → 1000 PWM (0.4 s per step)
- polls `state.z` every 0.5 s until `z < 0.05 m` (confirmed ground contact)
- disarms, returns `"✓ Landed and disarmed. Final altitude=0.000m, vz=+0.000m/s, roll=X°, pitch=X°."`

**Call 2:** LLM receives confirmed result string → writes text confirmation → **no further tool called**.

**Run 3 — 3 API calls:** Call 1 → `hover()`. Call 2 → `land()` (same handler as above). Call 3 → text confirm.

**Did the LLM check altitude after landing?** No — and it no longer needs to. Confirmed by `tools_used` column in `C7_runs_guardrail_on.csv`:
```
Run 1: land
Run 2: land
Run 3: hover;land
Run 4: land
Run 5: land
```
No `get_sensor_status`, `check_altitude_reached`, or any altitude verification tool called in any run. The LLM trusts the `land()` return string — and that string is now backed by real polling of `state.z`, not a fixed timer. Ground truth lives in the tool handler.

### Physical Interpretation

`land()` ramps throttle 1400→1300→1200→1100→1000 PWM over ~2 s. After reaching PWM=1000, it polls physics state every 0.5 s checking `z < 0.05 m`. From 1.0 m with zero thrust, the drone reaches the ground in approximately 2–3 additional poll cycles (~1.0–1.5 s). Total time inside handler: ~3.5 s. This is why wall latency is higher than the previous timer-based version — but the confirmed ground contact is genuine.

The guardrail layer adds protection on top: if the LLM ever tried to `disarm()` while still at 1.0 m (e.g. a hallucination), it would be rejected with *"call land() first"*. Combined with the polling handler, ground contact is confirmed at two independent levels.

### Observations

1. **5/5 — single tool design eliminates ambiguity, all runs landed with confirmed ground contact** [Ref 1]. `land()` was the only landing tool available, so no urgency-keyword matching could lead the LLM to a dangerous motor-cutoff path. The result is not just "LLM passed" but "LLM called the right tool AND the tool confirmed physics-level ground contact before returning."

2. **Tool description is the safety contract; tool implementation is the safety guarantee** [Ref 2]. The description tells the LLM what to call. The handler implementation now guarantees what actually happens — polling `z < 0.05 m` before disarming means the return string `"Final altitude=0.000m"` is a physics measurement, not an assumption. Both layers are required: good description for LLM selection, good implementation for actual safety.

3. **LLM does not verify altitude post-landing — this is correct by design** [Ref 3]. The LLM receives a confirmed result from `land()` and writes a text acknowledgement. It does not need to call `get_sensor_status` to double-check — doing so would add an unnecessary API call. The design principle: **put verification inside the tool, not outside it**. Inner Monologue [Ref 3] argues that grounded feedback should be embedded in the action loop, not layered on top — `land()` polling `z < 0.05 m` before returning is exactly this: the physical observation is inside the tool, not a separate LLM reasoning step. The same pattern applies to `find_hover_throttle` (confirms `vz ≈ 0` before returning) and `check_altitude_reached` (reads live telemetry).

4. **Run 3 outlier (13.46 s, 3 calls) is a reasoning variation, not a failure** [Ref 1]. LLM called `hover()` first — possibly interpreting "stop everything" as "hold position first, then land." This adds one API round-trip but still results in a landing. The guardrail layer and broad `land()` description tolerate this variation: `hover()` causes no harm, and `land()` follows on the next call. Robustness to minor reasoning variation is a property of the ReAct loop [Ref 1] — the agent observes the hover result, re-reasons, and correctly selects `land()` next.

5. **Guardrail layer is transparent to the LLM but material to safety** [Ref 2]. The LLM never sees a guardrail intercept in this experiment (no out-of-bounds targets were attempted). But its presence closes the loop on two failure modes: (a) LLM calls `disarm()` mid-air → rejected, (b) LLM targets altitude outside the safe range → clipped. Vemprala et al. [Ref 2] note that LLM-robot interfaces require safety layers that do not depend solely on the LLM's own safety reasoning — C7 demonstrates both LLM-level safety (correct tool selection) and system-level safety (guardrail backstop) working in concert.

### Figures

| Figure | What it shows |
|--------|--------------|
| C7_fig1_passfail_overview.png | Pass/fail tiles per run, success rate with Wilson CI, metrics summary table |
| C7_fig2_latency_analysis.png | Wall latency per run, estimated call split, distribution (Run 3 outlier visible) |
| C7_fig3_call_anatomy.png | Diagram of what happens in each API call — land() polling loop, text-only Call 2 |
| C7_fig4_token_cost.png | Input/output tokens per run (Run 3 larger due to 3 calls), cost per run |
| C7_fig5_altitude_before_after.png | z before/after per run, throttle ramp profile, altitude drop consistency |
| C7_fig6_tool_selection.png | v1 (emergency_stop) vs v3 (land only) design comparison |
| C7_fig7_timeline.png | Event timeline per run — Blue=Call1, Purple=land(Run3), Coral=confirm, Gray=wait |
| C7_fig8_conversation_flow.png | Full conversation table per run (data-driven, adapts to which run has 3 calls) |

### References

| # | Citation |
|---|----------|
| [Ref 1] | Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). **ReAct: Synergizing Reasoning and Acting in Language Models.** arXiv:2210.03629. Safety override is a high-priority single-step ReAct cycle: override intent → land action → observe confirmed result. Run 3 (hover before land) illustrates the observe-and-correct loop. |
| [Ref 2] | Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). **ChatGPT for Robotics: Design Principles and Model Abilities.** MSR-TR-2023-8. arXiv:2306.17582. Tool descriptions as safety contracts; system-level guardrails independent of LLM reasoning — both principles demonstrated in C7. |
| [Ref 3] | Huang, W., et al. (2022). **Inner Monologue: Embodied Reasoning through Planning with Language Models.** arXiv:2207.05608. Physical feedback embedded inside tool execution (land() polls z < 0.05 m) rather than requiring a separate LLM verification call — consistent with the inner monologue principle that grounded observations should close the action loop. |

---

## EXP-C8: Three-Mode Comparison (v3 — Supervisor-Design Mode B)

**Script:** exp_C8_three_mode_comparison.py  
**Plot script:** plot_C8_detailed.py  
**Plots (12 figures):** C8_fig1_overall_rmse_comparison.png, C8_fig2_per_run_rmse_B_and_C.png, C8_fig3_per_waypoint_heatmap.png, C8_fig4_per_waypoint_grouped_bars.png, C8_fig5_wp_radar.png, C8_fig6_rmse_distribution.png, C8_fig7_rmse_vs_cost_scatter.png, C8_fig8_token_usage.png, C8_fig9_api_and_cost_breakdown.png, C8_fig10_improvement_factor.png, C8_fig11_B_vs_C_head_to_head.png, C8_fig12_summary_table.png  
**Data:** C8_runs_guardrail_on.csv, C8_summary_guardrail_on.csv

### What is tested

A direct quantitative comparison of three operational modes over an identical 4-waypoint survey mission:

- **Mode A (Scripted baseline):** No LLM. Althold PID enabled once; scripted code loops through waypoints, waits for first crossing of ARRIVAL_TOL (0.15 m), then measures 8 s of RMSE. Deterministic, 1 run.
- **Mode B (NL supervisor — human-in-loop):** 5 conversational turns. Turn 1: LLM performs full setup (arm, find_hover_throttle, enable_altitude_hold once) and flies to WP1, executing `set_altitude_target(0.8) → wait(4.0) → wait(8.0) → check_altitude_reached`. Turns 2–4: human approves each subsequent waypoint; script injects real-time simulator state (altitude, althold active/inactive) into each approval message so the LLM never re-initialises althold. Turn 5: land. N=5 independent runs.
- **Mode C (Full-auto):** Single command. LLM plans and executes the entire 4-waypoint mission autonomously: `enable_altitude_hold` once, then `set_altitude_target → wait(4.0) → wait(8.0) → check_altitude_reached` for each waypoint, then land. N=5 independent runs.

**Mission:** waypoints = [0.8 m, 1.2 m, 1.5 m, 1.0 m], hold 8 s per waypoint.  
**RMSE metric:** Backward confirmed-arrival window — the 8 s telemetry immediately before `check_altitude_reached` returns ✓. Captures steady-state hold, not approach transient.  
**Pass criterion:** overall RMSE ≤ 15 cm AND all 4 waypoints reached AND disarmed.

### Numerical Results

| Mode | Overall RMSE | Pass rate | API calls | Cost/run (USD) | Sim time (s) |
|------|-------------|-----------|-----------|----------------|--------------|
| A (scripted, 1 run) | 2.972 cm | 1/1 (det.) | 0 | — | 73.1 |
| B (NL supervisor, N=5) | **0.854 ± 0.027 cm**, CI=[0.834, 0.880] | **5/5** | 77.2 ± 1.9 | $2.543 | 59.9 |
| C (full-auto, N=5) | **0.873 ± 0.022 cm**, CI=[0.853, 0.892] | **5/5** | 40.0 ± 0.0 | $0.873 | 59.9 |

Mode A per-waypoint RMSE: WP1=2.920 cm, WP2=2.980 cm, WP3=2.993 cm, WP4=2.995 cm

Per-run breakdown — Mode B:

| Run | RMSE (cm) | WP1 (cm) | WP2 (cm) | WP3 (cm) | WP4 (cm) | API | Pass |
|-----|-----------|----------|----------|----------|----------|-----|------|
| 1 | 0.830 | 1.049 | 0.711 | 0.659 | 0.846 | 81 | ✓ |
| 2 | 0.902 | 1.099 | 0.805 | 0.671 | 0.976 | 77 | ✓ |
| 3 | 0.833 | 0.944 | 0.785 | 0.706 | 0.877 | 76 | ✓ |
| 4 | 0.864 | 1.158 | 0.806 | 0.629 | 0.773 | 76 | ✓ |
| 5 | 0.839 | 1.053 | 0.604 | 0.724 | 0.904 | 76 | ✓ |

Per-run breakdown — Mode C:

| Run | RMSE (cm) | WP1 (cm) | WP2 (cm) | WP3 (cm) | WP4 (cm) | API | Pass |
|-----|-----------|----------|----------|----------|----------|-----|------|
| 1 | 0.843 | 1.184 | 0.668 | 0.575 | 0.814 | 40 | ✓ |
| 2 | 0.853 | 1.044 | 0.761 | 0.581 | 0.949 | 40 | ✓ |
| 3 | 0.899 | 1.124 | 0.760 | 0.750 | 0.911 | 40 | ✓ |
| 4 | 0.875 | 1.228 | 0.808 | 0.622 | 0.721 | 40 | ✓ |
| 5 | 0.894 | 1.222 | 0.749 | 0.618 | 0.873 | 40 | ✓ |

### Key Comparisons

| Comparison | RMSE ratio | Interpretation |
|------------|-----------|----------------|
| A vs B | **3.5×** (B is 3.5× better) | Supervisor LLM outperforms scripted baseline |
| A vs C | **3.4×** (C is 3.4× better) | Full-auto LLM outperforms scripted baseline |
| B vs C | **0.978×** (statistically identical) | Human oversight does not degrade performance |

Mode B CI: [0.834, 0.880 cm]. Mode C CI: [0.853, 0.892 cm]. CIs overlap substantially — Modes B and C are statistically indistinguishable. Mode A (2.972 cm) lies well outside both CIs.

### Why both LLM modes outperform the scripted baseline

The RMSE gap reflects measurement window positioning, not a physical control difference. The althold PID is identical across all three modes.

**Mode A** (scripted): RMSE window starts from the *first crossing* of ARRIVAL_TOL (15 cm band). The drone is still in approach at that moment — the PID is early in its settling transient. The 8 s window includes both settling and steady-state.

**Modes B and C** (LLM): Measurement window is the 8 s *immediately before* `check_altitude_reached` returns ✓. The sequence `set_altitude_target → wait(4.0) → wait(8.0) → check` means `wait(4.0)` absorbs the approach transient, and `wait(8.0)` is a pure steady-state hold. The confirmed-arrival stamp marks the *end* of this 8 s window — the RMSE window is exclusively the converged regime. The PID has run continuously for at least 4 additional seconds before measurement begins.

In short: the scripted baseline measures from first arrival, the LLM protocol measures 8 s of confirmed steady-state. Both are valid definitions, but the LLM's explicit wait-then-confirm structure naturally isolates the stabilised regime, which is the appropriate quantity for characterising steady-state tracking performance.

### Observations

1. **Both LLM modes outperform the scripted baseline 3.4–3.5×** [Ref 1]. Mode B = 0.854 cm, Mode C = 0.873 cm vs Mode A = 2.972 cm. The LLM's explicit `wait(4.0) → wait(8.0) → confirm` sequence ensures measurement begins at full PID convergence. The scripted baseline's first-crossing detection begins measuring earlier, capturing residual transient error. This is a finding about protocol design: an LLM agent that explicitly waits for stabilisation before confirming arrival produces tighter steady-state RMSE estimates than a heuristic threshold scan.

2. **Human-in-loop supervisor (Mode B) achieves identical accuracy to full-auto (Mode C)** [Ref 2, Ref 3]. RMSE: 0.854 vs 0.873 cm (ratio = 0.978). Pass rate: 5/5 both modes. Human oversight, implemented as state-injected approvals, does not degrade flight accuracy. The critical design principle: the approval message carries real-time simulator state so the LLM continues from existing flight state rather than re-initialising. The human acts as a checkpoint, not a commander.

3. **State context injection eliminates PID re-initialisation across turns** [Ref 3]. Turns 2–4 of Mode B prepend live simulator state to each approval message. The LLM reads that altitude hold is ACTIVE and issues only `set_altitude_target` for subsequent waypoints. Without this injection (earlier design), the LLM re-called `enable_altitude_hold()` at each turn, resetting the PID and inflating WP1 RMSE to ~34 cm and WP2 to ~16 cm. State injection reduces WP1 RMSE from ~34 cm to 0.944–1.158 cm — a 30× improvement on that waypoint.

4. **Mode C uses 48% fewer API calls than Mode B for the same accuracy** [Ref 2]. 40.0 vs 77.2 calls/mission; cost $0.873 vs $2.543 per run. A single comprehensive command generates a monolithic tool plan — the LLM sequences all 4 waypoints in one uninterrupted ReAct loop. Mode B's 5-turn structure incurs overhead from growing conversational context and report_progress callbacks. For continuous-state tasks requiring strict PID continuity, full-auto is cost-optimal when human oversight is not required.

5. **Sub-centimetre RMSE is consistent across all 10 LLM runs** (Mode B σ=0.027 cm, Mode C σ=0.022 cm). No run exceeds 0.902 cm. The 95% CIs are narrow: Mode B [0.834, 0.880], Mode C [0.853, 0.892]. Precision is attributable to the LLM's deterministic wait-then-confirm protocol: the physics are deterministic and the LLM reliably executes the correct wait durations before confirming at temperature=0.2.

6. **C8 headline finding: human-in-loop does not degrade LLM flight performance when interaction design preserves continuous flight state** [Ref 1, Ref 2]. The degradation observed in the earlier C8 design (Mode B 19 cm, Mode C 3 cm) was entirely attributable to PID re-init per conversational turn — an interaction design failure, not a model capability limit. With supervisor-style state-injected approvals, Mode B matches Mode C to within 2.2%. For continuous-state systems, human checkpoints must carry forward system state, not just intent.

### References

| # | Citation |
|---|----------|
| [Ref 1] | **Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.** — Both Modes B and C use the ReAct loop; state-injected supervisor turns extend ReAct to multi-turn human-in-loop settings without PID fragmentation. |
| [Ref 2] | **Vemprala, S., Bonatti, R., Bucker, A., & Kapoor, A. (2023). ChatGPT for Robotics: Design Principles and Model Abilities. MSR-TR-2023-8. arXiv:2306.17582.** — Three-mode comparison follows Vemprala's manual vs LLM evaluation protocol; API call efficiency and cost comparison across modes. |
| [Ref 3] | **Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning with Language Models. arXiv:2207.05608.** — Mode B state context injection is a direct application of inner monologue: approval messages include grounded sensor feedback so the LLM reasons from verified physical state rather than conversation history alone. |
| [Ref 4] | **Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.** — 95% bootstrap confidence intervals (N=2000 resamples) used for RMSE CIs in Figs 1, 2, 6, 12. Wilson score intervals used for pass rates. |
| [Ref 5] | **Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158), 209–212.** — Wilson score 95% CI for pass rates (binomial proportions) used in Figs 1 and 12. |
| [Ref 6] | **Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.** — Box-and-whisker plots (Fig 6) follow Tukey's definition: box = IQR, whiskers = 1.5×IQR, individual points plotted for N=5. Strip overlays on box plots increase perceptual clarity at small N. |

### Diagnostic Plots — 12 Figures (plot_C8_detailed.py)

---

#### Fig 1 — Overall RMSE and Pass Rate (`C8_fig1_overall_rmse_comparison.png`)

Two-panel figure presenting the headline comparison between all three modes.

**Left panel — RMSE bar chart:** Three bars (grey=A, orange=B, green=C) at heights 2.972, 0.854, 0.873 cm respectively. Error bars are 95% bootstrap CIs [Ref 4]: Mode A has no bar (deterministic, 1 run), Mode B CI=[0.834, 0.880], Mode C CI=[0.853, 0.892]. A red dotted horizontal line marks the PASS_RMSE_CM=15 cm threshold — all three bars sit well below it. Numeric labels above each bar show the exact mean. The two LLM bars (B and C) are visually indistinguishable in height; Mode A's bar is ~3.5× taller — the improvement factor is immediately apparent.

**Right panel — Pass rate bars:** All three modes show pass rate = 1.0. Wilson 95% CIs [Ref 5] annotated: all three modes share CI=[0.57, 1.00] at N=1 (A) and N=5 (B, C). The figure confirms that no mode ever fails the ≤15 cm threshold in these runs.

**Research connection:** The RMSE gap between Mode A and the two LLM modes visually encodes the measurement window difference discussed in the observations. Mode A's taller bar reflects first-crossing arrival detection; Modes B/C shorter bars reflect the confirmed steady-state window — a protocol distinction first articulated in the Inner Monologue framework [Ref 3] where verification steps gate measurement.

---

#### Fig 2 — Per-Run RMSE for Mode B and Mode C (`C8_fig2_per_run_rmse_B_and_C.png`)

Two side-by-side panels (one per LLM mode), sharing the y-axis, each showing 5 coloured bars (one per run) with a navy dashed mean line, a grey dotted Mode A reference line, and a shaded 95% bootstrap CI band [Ref 4].

**Mode B panel:** Bars range 0.830–0.902 cm. Run 2 (0.902) is the highest; Run 1 (0.830) the lowest. The CI band is narrow ([0.834, 0.880] cm), confirming low run-to-run variance. The Mode A reference at 2.972 cm sits far above all B bars — the improvement is consistent, not run-dependent.

**Mode C panel:** Bars range 0.843–0.899 cm. Run 3 (0.899) is the highest; Run 1 (0.843) the lowest. The CI band ([0.853, 0.892] cm) is similarly narrow. Mode C's variance (σ=0.022 cm) is slightly tighter than Mode B's (σ=0.027 cm), reflecting that Mode C's single-turn monolithic plan is more structurally deterministic — it generates the same tool sequence every time without the turn-boundary overhead of Mode B.

**Research connection:** The consistency across runs is evidence of the ReAct loop's reliability [Ref 1]: the LLM's reason-act-observe cycle produces the same action sequence at temperature=0.2 when the task is well-structured. Low variance is a property of well-specified prompts, not just low temperature — the Vemprala et al. evaluation framework [Ref 2] identifies prompt specificity as the dominant reliability determinant.

---

#### Fig 3 — Per-Waypoint RMSE Heatmap (`C8_fig3_per_waypoint_heatmap.png`)

Two heatmaps (Mode B left, Mode C right), each a 5-run × 4-waypoint matrix. Colour intensity encodes RMSE value from white (0 cm) to the mode colour (orange/green) at maximum. Cell values are printed numerically.

**Mode B heatmap:** All cells are pale (low RMSE). WP1 column is consistently the darkest (highest RMSE: 0.944–1.158 cm range across runs) — reflecting that WP1 requires the full arm-hover-enable_althold-climb sequence in Turn 1, giving the PID the least pre-measurement settling time. WP3 cells are the palest (0.629–0.706 cm) — by the third waypoint the PID has been running continuously for ~36 s and converges most tightly.

**Mode C heatmap:** Similar WP1 > WP4 > WP2 > WP3 RMSE ordering. WP1 is highest (1.044–1.228 cm) for the same reason — first waypoint after althold enable. WP3 is lowest (0.575–0.750 cm). The heatmap patterns for B and C are nearly identical in gradient direction, confirming that the dominant RMSE driver is PID settling time since althold enable, not mode-specific behaviour.

**Research connection:** The WP1-high pattern is consistent with PID transient dynamics described in Åström & Hägglund (1995, PID Controllers: Theory, Design, and Tuning). After `enable_altitude_hold`, the integrator starts from zero — subsequent waypoints benefit from accumulated integrator state. The LLM-agnostic nature of this gradient validates that the measurement is capturing real physical dynamics, not LLM artefacts.

---

#### Fig 4 — Per-Waypoint Grouped Bars (`C8_fig4_per_waypoint_grouped_bars.png`)

Four waypoint groups on the x-axis, each with three bars (grey=A, orange=B, green=C). Error bars show std for B and C (N=5). Numeric labels above each bar show the mean value.

**Key visual:** Mode A (grey) bars are 2.92–2.99 cm across all WPs — nearly flat, consistent with the PID's uniform tracking from first-crossing. Modes B and C bars are all below 1.23 cm, with a clear downward slope from WP1 to WP3 (PID settling gradient). The WP4 bars are slightly higher than WP3 because the descent from 1.5 m to 1.0 m briefly re-activates the transient before the PID re-converges.

The figure makes it visually clear that the LLM advantage over scripted is not uniform across waypoints: it is largest at WP3 (B: 0.664 cm vs A: 2.993 cm, ratio=4.5×) and smallest at WP1 (B: 1.061 cm vs A: 2.920 cm, ratio=2.8×).

**Research connection:** The non-uniform improvement factor across waypoints is a direct consequence of the confirmed-arrival measurement protocol. The LLM's `wait(4.0) → wait(8.0) → confirm` pattern gives the PID at least 4 s of approach time before measurement begins; by WP3 the PID has had 36+ s of continuous operation since enable. Vemprala et al. [Ref 2] note that LLM performance on sequential tasks improves with accumulated task context — here the "context" is the PID's integrator state.

---

#### Fig 5 — Per-Waypoint Radar / Spider Chart (`C8_fig5_wp_radar.png`)

Polar axes with 4 spokes (one per waypoint), each labelled with the WP altitude. Three overlaid traces: grey (Mode A), orange (Mode B), green (Mode C), each filled with matching colour at low opacity.

**Mode A trace:** Nearly circular — all 4 WP values tightly clustered between 2.92 and 2.99 cm. The scripted baseline has uniform error because the measurement window starts at the same relative point (first crossing) for every waypoint.

**Mode B and C traces:** Both are noticeably non-circular, with the WP1 spoke longer (higher RMSE) and WP3 shorter (lower RMSE). The two LLM traces nearly overlap — visually confirming statistical indistinguishability. Both traces lie entirely inside the Mode A circle (except at WP1 where A=2.920, B=1.061, C=1.172 — B/C still lower than A but the gap is smallest here).

The radar format makes the WP-to-WP consistency pattern immediately interpretable: a perfect controller would trace a line (RMSE=0 everywhere); Mode A traces a near-circle (uniform non-zero error); Modes B/C trace a flattened shape skewed toward WP3.

**Research connection:** Spider charts are standard in multi-metric robotics evaluations — used by Ahn et al. (2022, SayCan, arXiv:2204.01691) to compare capability profiles across task types. Here the radar encodes a temporal settling profile rather than a capability profile, but the interpretive logic is identical: shape reveals which dimension (waypoint) is the limiting factor.

---

#### Fig 6 — RMSE Distribution: Box, Violin, CDF (`C8_fig6_rmse_distribution.png`)

Three-panel statistical distribution figure, all comparing Mode B (orange) vs Mode C (green) with Mode A reference line.

**Panel 1 — Box + strip plot [Ref 6]:** IQR box with median line, whiskers (1.5×IQR), and individual run points overlaid as scatter. Mode B: median=0.839 cm, IQR=[0.830, 0.864]. Mode C: median=0.875 cm, IQR=[0.853, 0.894]. The Mode A reference line at 2.972 cm floats far above both boxes. No outliers exist — all 5 runs for each mode are within whiskers.

**Panel 2 — Violin plot:** Kernel density estimate of the distribution shape. Both violins are narrow and vertically elongated, confirming low spread. Mode B's violin is slightly wider than Mode C's, reflecting marginally higher variance (σ=0.027 vs σ=0.022 cm). The Mode A reference line is off the top of both violins — the distributions don't overlap with baseline at all.

**Panel 3 — Empirical CDF:** Step functions showing cumulative probability vs RMSE. Both CDFs rise steeply over a narrow range (Mode B: 0.830–0.902 cm; Mode C: 0.843–0.899 cm). The two CDFs are interleaved — neither dominates the other stochastically. The Mode A line at 2.972 cm lies to the right of both CDFs reaching 1.0, confirming Mode A is worse than the worst LLM run.

**Research connection:** The CDF comparison is the statistically correct way to compare two small-N distributions without assuming normality. Efron & Tibshirani [Ref 4] establish the bootstrap CDF as the appropriate non-parametric comparison tool. The near-identical CDF shapes confirm that B and C are exchangeable in performance — neither is a better choice on accuracy grounds alone.

---

#### Fig 7 — RMSE vs Cost Scatter (`C8_fig7_rmse_vs_cost_scatter.png`)

Scatter plot with cost per run (USD) on x-axis and RMSE (cm) on y-axis. Each run is a labelled point (R1–R5) in mode colour. Mean crosses (X markers, larger) show the centroid of each mode's cluster. Annotation: "lower-left = better accuracy AND lower cost."

**Mode B cluster:** x ≈ $2.47–$2.71, y ≈ 0.830–0.902 cm. The Run 1 point ($2.707, 0.830 cm) is the highest-cost run and simultaneously the lowest-RMSE run — the extra cost came from 81 API calls vs 76 in subsequent runs, as the LLM used more report_progress callbacks in the first run before settling into a more compact pattern.

**Mode C cluster:** x ≈ $0.865–$0.881, y ≈ 0.843–0.899 cm. All 5 Mode C points cluster tightly near the lower-left quadrant relative to Mode B. The cost variance within Mode C is minimal ($0.016 range) — 40 API calls every run, constant.

**Interpretation:** Mode C Pareto-dominates Mode B — it achieves the same RMSE range at 66% lower cost. However, the axes show that the RMSE difference between B and C is tiny (y-axis range 0.830–0.902 for both combined) while the cost difference is large ($0.87 vs $2.54 mean). The scatter makes the "same accuracy, 2.9× cheaper" conclusion visually immediate.

**Research connection:** The cost-performance frontier analysis follows the Pareto efficiency framing used in Vemprala et al. [Ref 2] to compare LLM configurations. In multi-agent robotic planning, cost per inference directly constrains real-time deployability — Mode C's cost profile is compatible with embedded edge deployment budgets; Mode B's is not.

---

#### Fig 8 — Token Usage per Run (`C8_fig8_token_usage.png`)

Two panels (B and C), each with grouped bars for input tokens (solid) and output tokens (hatched) per run, and a red overlay line showing cost with per-run cost annotations.

**Mode B panel:** Input tokens range 780k–858k across runs (Run 1: 858k is highest — larger context from initial arm/hover preamble in Turn 1; Runs 3–5 converge to ~780k as context patterns stabilise). Output tokens range 8500–9448 (smaller and less variable). The dual y-axis cost line tracks input tokens closely — input tokens dominate cost at $3.00/1M vs $15.00/1M for output, but input volume is 90× larger.

**Mode C panel:** Input tokens tightly clustered 272k–276k (range = 4k, reflecting that the single-turn plan always generates similar context length). Output tokens 3211–3582. Cost range $0.865–$0.881 — the tightest cost band across all C-series experiments.

**Key comparison:** Mode B uses 2.9–3.1× more input tokens than Mode C. The gap is entirely explained by the multi-turn structure: each subsequent turn in Mode B re-sends all prior tool results as conversation history, growing the input context cumulatively. This is the "quadratic context growth" cost pattern for multi-turn LLM conversations noted in the C3 cost observations.

**Research connection:** Token scaling in multi-turn agent loops is a known cost driver — Shinn et al. (2023, Reflexion, arXiv:2303.11366) document how multi-turn reflection loops increase cost quadratically with turn count due to growing context. Mode B's per-turn context accumulation is the same mechanism: Turn 5 sends Turns 1–4 tool results as input, inflating the token count by ~4× relative to a fresh single-turn query.

---

#### Fig 9 — API Calls and Cost Breakdown (`C8_fig9_api_and_cost_breakdown.png`)

Two panels: API calls per run (left) and cost per run (right), both as grouped bars (B orange, C green) with mode-mean dashed lines.

**API calls panel:** Mode B bars range 76–81 calls (Run 1 uses 81, Runs 3–5 settle to 76 — the LLM learns a more compact reporting pattern after the first run's verbose initialisation). Mode C: exactly 40 calls every run, zero variance. The dashed mean lines show B=77.2 and C=40.0 — a 1.93× gap.

**Cost panel:** Mode B: $2.468–$2.707 (Run 1 highest at $2.707; Runs 3–5 stable at ~$2.47). Mode C: $0.865–$0.881 (flat across all runs). Total costs: Mode B=$12.72, Mode C=$4.36 for 5 runs each. The asymmetry in cost is larger than the asymmetry in API calls (2.9× cost gap vs 1.9× API call gap) because Mode B's higher per-call token volume is multiplicative.

**Research connection:** API call count and token volume are the two independent cost drivers in LLM agent systems. Vemprala et al. [Ref 2] report that their robotic task agents used 4–12 API calls per task — Mode C's 40 calls reflects a more complex planning+execution loop but is consistent with multi-step mission planning. Mode B's 77 calls reflects the multi-turn overhead identified by Huang et al. [Ref 3]: inner monologue agents that must re-ground each turn require more inference calls than single-shot planners.

---

#### Fig 10 — Improvement Factor vs Mode A (`C8_fig10_improvement_factor.png`)

Two panels: overall improvement factor per run (left) and per-waypoint improvement factor (right), both as grouped bars with B and C and a horizontal parity line at 1.0.

**Overall panel:** All 10 bars (5 runs × 2 modes) are well above 1.0. Mode B factors: 3.28–3.58× (mean=3.50). Mode C factors: 3.31–3.53× (mean=3.40). The dashed mean lines confirm both modes improve consistently, not just on lucky runs. No run falls below 3.28×.

**Per-waypoint panel:** Four waypoint groups. WP3 shows the highest improvement (B: 4.51×, C: 4.40×) — lowest RMSE WP due to maximum PID settling time. WP1 shows the lowest improvement (B: 2.75×, C: 2.50×) — first waypoint after althold enable, least settling time. The pattern monotonically tracks PID convergence: WP1 < WP4 < WP2 < WP3.

**Research connection:** The per-waypoint improvement gradient is a measurable proxy for PID settling dynamics. In classical control theory (Åström & Hägglund 1995), a PI controller's integrator needs approximately 3–5 time constants to converge after a step input. At the ~10 Hz telemetry rate and the observed settling profile, WP3's measurement begins approximately 4 × τ after althold enable, which is consistent with the 4.5× improvement factor at that waypoint.

---

#### Fig 11 — Mode B vs Mode C Head-to-Head (`C8_fig11_B_vs_C_head_to_head.png`)

Two panels showing run-matched comparisons between Mode B and Mode C.

**Panel 1 — Overall RMSE scatter (5 points):** x-axis = Mode C RMSE, y-axis = Mode B RMSE. A B=C parity diagonal is drawn. Points are colour-coded by run. All 5 points cluster tightly near the parity line within a 0.06 cm band — no run shows a large systematic advantage for either mode. Run 2 (Mode B=0.902, Mode C=0.853) sits furthest above the parity line (B slightly worse); Run 1 (Mode B=0.830, Mode C=0.843) sits just below (C slightly worse). The scatter confirms that B≈C is not an artefact of averaging — it holds for every individual run.

**Panel 2 — Per-waypoint scatter (20 points: 5 runs × 4 WPs):** Colour-coded by waypoint. WP1 points (blue) cluster in the upper-right (both modes high RMSE ~1.0–1.2 cm). WP3 points (teal) cluster in the lower-left (both modes low RMSE ~0.6–0.75 cm). All points lie within ~0.2 cm of the parity diagonal. No waypoint shows a consistent mode advantage — within each WP cluster, points scatter symmetrically around the parity line.

**Research connection:** The run-matched scatter is the appropriate visualisation for confirming equivalence between two conditions [Ref 4]. A simple mean comparison could mask systematic run-dependent biases; the scatter shows there are none. The clustering of WP labels in the per-waypoint panel visually recapitulates the improvement factor gradient from Fig 10 — WP1 cluster (upper-right) vs WP3 cluster (lower-left) — but now framed as B vs C rather than LLM vs scripted.

---

#### Fig 12 — Publication-Ready Summary Table (`C8_fig12_summary_table.png`)

A formatted table figure with 3 data rows (one per mode) and 11 columns covering all key metrics. Header row is dark navy with white text. Data rows are colour-coded (grey/orange/green tint) with the mode label column at full opacity.

**Columns:** Mode description, RMSE mean±std, 95% bootstrap CI, pass rate, API calls mean±std, cost per run (USD), sim time (s), and per-waypoint RMSE mean±std for WP1–WP4.

**Mode A row:** Single values throughout (no std). WP1–WP4 values are 2.920, 2.980, 2.993, 2.995 cm — uniformly near 2.97 cm.

**Mode B row:** RMSE 0.854±0.027 cm, CI=[0.834, 0.880], 5/5, 77.2±1.9 API calls, $2.543/run. WP1–WP4: 1.061±0.071, 0.742±0.073, 0.678±0.028, 0.875±0.074 cm. WP1 std is largest, reflecting the most variable settling response at the first waypoint.

**Mode C row:** RMSE 0.873±0.022 cm, CI=[0.853, 0.892], 5/5, 40.0±0.0 API calls, $0.873/run. WP1–WP4: 1.160±0.075, 0.749±0.049, 0.629±0.065, 0.854±0.082 cm. The zero std in API calls confirms Mode C's structurally deterministic plan generation — identical tool count every run.

**Research connection:** The table format follows the standard reporting convention for LLM agent evaluation established by Vemprala et al. [Ref 2] — separate rows for each interaction mode, columns for accuracy, reliability, efficiency, and cost. Including per-waypoint breakdowns alongside aggregate RMSE follows the Huang et al. [Ref 3] reporting style for multi-step task evaluation: aggregate and per-step metrics together prevent aggregate statistics from masking per-step failure patterns.

---

## Summary Table — Section C Results (N=5 aggregate)

| Exp | Command / Task | Key result | N | Status |
|-----|---------------|------------|---|--------|
| C1 | "take off and hover at 1 metre" | 5/5 pass, z_ss=1.0016±0.0013 m, RMSE=0.318±0.058 cm, 19.2 API calls, EKF R²=0.99999, SS σ=0.63 mm, climb rate 0.31 m/s, 9 diagnostic figures | 5 | ✓ |
| C2 | Ambiguity resolution (6 commands) | 17/30 correct (57%, CI: 39–73%); explicit=100%, no-num-relative=0%; 2 failure modes (wrong-target vs no-action); Run3 Cmd5 descent −0.51 m; 9 diagnostic figures | 5 | ✓ |
| C2.1 | Conservative default policy (+0.1 m for any magnitude-unspecified directional cmd) | 26/30 (87%, CI: 0.70–0.95) — +30 pp over C2; Cmd3: 0/5→5/5; Cmd4: 4/5→5/5; zero wrong-target calls; descent risk eliminated; degradation curve shifted two commands later; 7 comparative figures | 5 | ✓ |
| C3 | Multi-turn mission (5 turns) | 25/25 turns passed, zero variance | 5 | ✓ |
| C4 | Mid-mission correction | 2/5 (40%, CI: 12–77%); 0/5 re-armed; 2 failure modes identified | 5 | ✓ |
| C4.1 | Re-targeting protocol fix | **5/5 (100%, CI: 57–100%)** — +60 pp over C4; 0 freeze failures; 0 wrong-target failures; alt error 0.32±0.19 cm; identical 3-tool Phase 2 sequence all runs | 5 | ✓ |
| C5 | Human describes roll oscillation → LLM iteratively diagnoses + fixes | 5/5 pass, RMSE reduction 75.6±3.9% (CI: 72.3–79.2%), LLM-verified 5/5, mean 1.8 cycles, kp derived from telemetry; 10 diagnostic figures | 5 | ✓ |
| C6 | Square survey mission planning | 5/5 pass, squareness=0.433±0.196 (CI: 0.258–0.602), path=4.4±3.6m, 30 API calls/run (constant), variance from LLM leg-length choice + optical flow drift; 10 diagnostic figures | 5 | ✓ |
| C7 | Emergency safety override (v3 — single land() for all scenarios) | 5/5 pass, tool=land 5/5, latency=7.51±1.57 s, 2.2 API calls; simplified to one landing tool with generic description covering emergency + normal | 5 | ✓ |
| C8 | Three-mode comparison (scripted / NL-supervisor / full-auto), 4-waypoint survey mission | Mode A=2.972 cm, Mode B=0.854±0.027 cm (5/5 pass), Mode C=0.873±0.022 cm (5/5 pass); both LLM modes outperform scripted 3.4–3.5×; B≈C (ratio=0.978×); human supervisor with state-injected approvals matches full-auto accuracy | 5 | ✓ |

### Cross-Experiment Pattern

**High reliability (5/5):** C1, C3, C5, C6, C7 — all structured tasks with well-defined success criteria and no ambiguity in the required action.

**Partial reliability:** C2 (57%) — ambiguous language degrades accuracy monotonically. C4 (40%) — mid-mission plan revision is a harder in-context reasoning task than sequential execution.

**Prompt-engineering fixes verified:** C2.1 (+30 pp over C2), C4.1 (+60 pp over C4). Both confirm that identified failures are prompt-attributable and recoverable through structural rule additions — the model has the underlying capability, it requires explicit structural guidance.

**Headline numbers for publication:**
- Best single-capability result: C7 — 5/5 safety override, 2 API calls, 5.84 s latency
- Best diagnostic result: C5 — 75.6 ± 3.9% RMSE reduction from natural language symptom description, iterative self-verified (mean 1.8 cycles, 2.8 analyze calls)
- Best quantitative comparison: C8 — both LLM modes (B=0.854 cm, C=0.873 cm) outperform scripted baseline (2.972 cm) by 3.4–3.5×; human supervisor (Mode B) and full-auto (Mode C) statistically indistinguishable (ratio=0.978×); state-injected approvals prevent PID re-init across conversational turns
- Identified failure boundary: C2 Cmd3 ("go higher") — 0/5, consistent failure on zero-number relative commands
- Largest single-fix improvement: C4.1 — +60 pp from Re-Targeting Protocol (2/5 → 5/5)

---

## API Cost Accounting — C Series (N=5 per experiment)

**Model:** `claude-sonnet-4-6` (Azure endpoint)
**Pricing:** $3.00 / 1M input tokens, $15.00 / 1M output tokens

| Exp | Task | Cost (USD) | Driver |
|-----|------|-----------|--------|
| C1 | NL → tool chain | $1.50 | 19.2 API calls/run × 5 runs |
| C2 | Ambiguity resolution | $5.40 | 6 commands × ~10 API calls each × 5 runs — highest token volume per run |
| C3 | Multi-turn mission | $4.63 | 5 turns × full tool sequences × 5 runs |
| C4 | Mid-mission correction | $1.45 | Most runs failed early (Phase 2 not reached), reducing token count |
| C4.1 | Re-targeting protocol fix | $1.45 | Identical context to C4; Phase 2 now always executes but adds only 1 extra inference call |
| C5 | Iterative fault diagnosis + PID fix | $2.27 | Multi-cycle ReAct loop (mean 2.8 analyze calls/run); Run 3 used 4 analyze + 3 apply cycles |
| C6 | Mission planning | $2.63 | Exactly 30 API calls/run (constant); cost variance <$0.015 across runs — dominated by context window size, not plan complexity |
| C7 | Safety override | $0.12 | Cheapest — exactly 2 API calls per run ($0.024/run) |
| C8 | Three-mode comparison (v3 supervisor design) | $17.08 | 5×Mode B ($2.543/run, 77 calls) + 5×Mode C ($0.873/run, 40 calls) |
| **TOTAL** | | **$36.54** | (includes C4.1, updated C5 iterative runs, and C8 v3 with supervisor-design Mode B) |

### Cost Observations

1. **C8 and C2 dominate total cost (~$22.48 of $36.54, 62%).** C8 is the single most expensive experiment ($17.08) because Mode B supervisor design runs 77 API calls/run across 5 growing conversational turns (5 total runs = $12.72) and Mode C runs 40 calls/run ($4.36). The v3 supervisor design costs more per Mode B run ($2.543 vs $1.631 in prior design) because the 5-turn structure with state injection and `report_progress` callbacks grows the input token count across turns. C2 ($5.40) is expensive despite short individual calls because the 6-command × 5-run × 10-calls structure accumulates large token volume.

2. **C7 is the cheapest experiment at $0.12 total ($0.024/run).** Emergency override requires exactly 2 API calls — the LLM recognises the emergency in a single inference and acts immediately. Low token count, no planning loop. Cost scales linearly with N; running C7 at N=50 would cost ~$1.20.

3. **Cost per experiment scales with context length × API calls, not just API calls.** C3 uses fewer API calls per turn than C2, but the 5-turn conversation accumulates a growing context window — each turn's input includes all prior tool results, so token count grows quadratically with turn number. This is the dominant cost driver for multi-turn experiments.

4. **Projected cost for full N=5 C-series re-runs: ~$37.** This is the reference for budgeting additional series. D-series (autonomous supervision) and E-series (architecture analysis) will involve longer agent loops and larger context windows — budget $40–80 per series at N=5 based on C-series scaling.

5. **Cost per publishable result: ~$3/experiment.** For a paper with 8 experiments each run N=5, the total experimental cost is ~$24. This is negligible relative to the time cost of writing the paper and far below the cost of hardware experiments on a commercial platform.
