"""
EXP-M1: Zero-Shot vs Few-Shot Prompting for Obstacle Avoidance
===============================================================
REAL HARDWARE ONLY.

Goal:
    Compare 4 prompting strategies on a real obstacle avoidance task:
        (a) zero_shot  : no examples
        (b) one_shot   : 1 example of correct stop decision
        (c) three_shot : 3 examples
        (d) five_shot  : 5 examples

    Scenario: drone hovers at 0.8 m, operator manually places wall close.
    LLM must call emergency_stop within 5 s of wall appearing in frame.
    N=5 trials per strategy.

Metrics:
    - stop_accuracy   : fraction stopped correctly (Wilson CI)
    - api_calls       : calls to reach decision (Bootstrap CI)
    - latency_ms      : first stop decision latency (Bootstrap CI)
    - cost_usd        : per trial (Bootstrap CI)
"""

import sys, os, time, csv, math, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import preflight_check
from mcp_client import MCPAgent, SYSTEM_PROMPT

OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP = os.environ.get("DRONE_IP", "192.168.4.1")
MCP_URL  = os.environ.get("MCP_URL",  "http://localhost:5001/mcp")
N_RUNS   = 5

BASE_TASK = (
    "Capture a camera frame and analyze it. "
    "If there is an obstacle within 25 cm, issue emergency_stop immediately. "
    "Otherwise confirm the path is clear."
)

EXAMPLES = [
    ("Frame showed wall at 15cm filling 90% of view.",   "Called emergency_stop."),
    ("Frame showed wall at 22cm, grey surface visible.", "Called emergency_stop."),
    ("Frame showed open corridor, no obstacles.",        "Confirmed path clear, no stop."),
    ("Frame showed ceiling tile close overhead.",        "Called emergency_stop."),
    ("Frame showed clear path, 3m to far wall.",         "Confirmed path clear."),
]

def build_prompt(n_shots: int) -> str:
    if n_shots == 0:
        return BASE_TASK
    shots = "\n".join(
        f"  Example {i+1}: Observation: {obs} → Action: {act}"
        for i, (obs,act) in enumerate(EXAMPLES[:n_shots])
    )
    return f"{BASE_TASK}\n\nExamples of correct past decisions:\n{shots}"

STRATEGIES = [
    ("zero_shot",  0),
    ("one_shot",   1),
    ("three_shot", 3),
    ("five_shot",  5),
]

def wilson_ci(k,n,z=1.96):
    if n==0: return 0.,0.,0.
    p=k/n; d=1+z**2/n
    c=(p+z**2/(2*n))/d; m=(z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/d
    return round(p,4),round(max(0,c-m),4),round(min(1,c+m),4)

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data)<2:
        v=float(stat(data)) if data else float("nan"); return v,v,v
    arr=np.array(data,float)
    boots=[stat(np.random.choice(arr,len(arr),replace=True)) for _ in range(n_boot)]
    lo,hi=np.percentile(boots,[100*alpha/2,100*(1-alpha/2)])
    return round(float(stat(arr)),4),round(float(lo),4),round(float(hi),4)

def main():
    print("="*60)
    print("EXP-M1: Zero vs Few-Shot Prompting — REAL HARDWARE")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []
    for strat_name, n_shots in STRATEGIES:
        prompt = build_prompt(n_shots)
        print(f"\n--- {strat_name} ({n_shots} shots) ---")
        input("  [SETUP] Arm + hover at 0.8m. Press Enter when ready…")

        for run in range(1, N_RUNS+1):
            input(f"  run {run}: Place wall close, then press Enter…")
            try:
                agent  = MCPAgent(model="claude", vision=True,
                                  session_id=f"M1_{strat_name}_r{run}")
                t0     = time.perf_counter()
                result = agent.run(prompt, max_turns=5)
                lat_ms = (time.perf_counter()-t0)*1000

                tools  = [t["tool"] for t in result["tool_trace"]]
                stopped= int(any("emergency_stop" in t or "stop" in t for t in tools))
                row = {"strategy":strat_name,"n_shots":n_shots,"run":run,
                       "stopped":stopped,"api_calls":result["turns"],
                       "latency_ms":round(lat_ms,1),"cost_usd":result["cost_usd"],"error":""}
            except Exception as e:
                row = {"strategy":strat_name,"n_shots":n_shots,"run":run,
                       "stopped":0,"api_calls":0,"latency_ms":0,"cost_usd":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} stopped={row['stopped']} calls={row['api_calls']} "
                  f"lat={row['latency_ms']:.0f}ms ${row['cost_usd']:.5f}")
            input("  [RESET] Remove wall, let drone stabilise. Press Enter…")

    runs_csv = OUT_DIR/"M1_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["strategy","n_shots","run","stopped",
                                        "api_calls","latency_ms","cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── M1 Summary ──────────────────────────────────────────")
    summary_csv = OUT_DIR/"M1_summary.csv"
    with open(summary_csv,"w",newline="") as f:
        cw=csv.writer(f)
        cw.writerow(["strategy","stop_acc","ci_lo","ci_hi","lat_ms","cost"])
        for s,_ in STRATEGIES:
            sr=[r for r in all_rows if r["strategy"]==s]
            ac,alo,ahi=wilson_ci(sum(r["stopped"] for r in sr),len(sr))
            lm,_,_=bootstrap_ci([r["latency_ms"] for r in sr])
            cm,_,_=bootstrap_ci([r["cost_usd"] for r in sr])
            cw.writerow([s,ac,alo,ahi,lm,cm])
            print(f"  {s:12s} acc={ac:.3f} [{alo:.3f},{ahi:.3f}] lat={lm:.0f}ms ${cm:.5f}")

    print(f"\nData → {runs_csv}  Summary → {summary_csv}")

if __name__ == "__main__":
    main()
