"""
EXP-M2: Chain-of-Thought vs ReAct vs Direct-Action Prompting
=============================================================
REAL HARDWARE ONLY.

Goal:
    Compare three prompting strategies on a real 3-waypoint navigation mission:
        (a) direct   : "Go to waypoint, avoid walls."
        (b) cot      : "Think step by step, then act."
        (c) react    : "Reason: [think]. Observe: [get_telemetry]. Act: [tool]."

    N=5 runs per strategy. Drone executes real flight.

Metrics:
    - waypoints_completed : fraction of 3 waypoints reached (Bootstrap CI)
    - safety_stops        : emergency stops triggered (Bootstrap CI)
    - api_calls           : number of LLM turns (Bootstrap CI)
    - latency_ms          : total mission time (Bootstrap CI)
    - cost_usd            : per mission (Bootstrap CI)
"""

import sys, os, time, csv, math, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import preflight_check
from mcp_client import MCPAgent

OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP = os.environ.get("DRONE_IP", "192.168.4.1")
MCP_URL  = os.environ.get("MCP_URL",  "http://localhost:5001/mcp")
N_RUNS   = 5

MISSION = "arm, take off to 1m, move forward 0.3m, move right 0.3m, move backward 0.3m, then land"

PROMPTS = {
    "direct": (
        f"Execute this mission: {MISSION}. "
        "Stop immediately if any obstacle is within 25cm."
    ),
    "cot": (
        f"Execute this mission: {MISSION}. "
        "Before each action, think step by step: "
        "(1) What is my current state? (2) What is the next step? "
        "(3) Are there any safety concerns? Then act."
    ),
    "react": (
        f"Execute this mission: {MISSION}. "
        "Use this pattern for each step:\n"
        "Reason: [explain what you observe and what you plan]\n"
        "Act: [call the appropriate tool]\n"
        "Observe: [read the tool result]\n"
        "Repeat until mission complete."
    ),
}

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data)<2:
        v=float(stat(data)) if data else float("nan"); return v,v,v
    arr=np.array(data,float)
    boots=[stat(np.random.choice(arr,len(arr),replace=True)) for _ in range(n_boot)]
    lo,hi=np.percentile(boots,[100*alpha/2,100*(1-alpha/2)])
    return round(float(stat(arr)),4),round(float(lo),4),round(float(hi),4)

WAYPOINT_KEYWORDS = ["forward","right","backward","waypoint","move"]

def count_waypoints(trace):
    return sum(1 for t in trace
               if any(k in t["tool"] for k in ("move_forward","move_right","move_backward")))

def main():
    print("="*60)
    print("EXP-M2: CoT vs ReAct vs Direct — REAL HARDWARE")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []
    for strat, prompt in PROMPTS.items():
        print(f"\n=== Strategy: {strat} ===")
        for run in range(1, N_RUNS+1):
            input(f"  [SETUP] Clear 1×1m space. Place drone on ground. run={run}. Press Enter…")
            try:
                agent  = MCPAgent(model="claude", vision=True,
                                  session_id=f"M2_{strat}_r{run}")
                t0     = time.perf_counter()
                result = agent.run(prompt, max_turns=25)
                lat_ms = (time.perf_counter()-t0)*1000

                trace  = result["tool_trace"]
                wp_done= count_waypoints(trace)
                estops = sum(1 for t in trace if "emergency_stop" in t["tool"])
                landed = int(any("land" in t["tool"] for t in trace))

                row = {"strategy":strat,"run":run,
                       "waypoints_done":min(wp_done,3),"max_waypoints":3,
                       "safety_stops":estops,"landed":landed,
                       "api_calls":result["turns"],
                       "latency_ms":round(lat_ms,1),
                       "cost_usd":result["cost_usd"],"error":""}
            except Exception as e:
                row = {"strategy":strat,"run":run,"waypoints_done":0,"max_waypoints":3,
                       "safety_stops":0,"landed":0,"api_calls":0,
                       "latency_ms":0,"cost_usd":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} wp={row['waypoints_done']}/3 stops={row['safety_stops']} "
                  f"calls={row['api_calls']} lat={row['latency_ms']:.0f}ms")
            time.sleep(5)

    runs_csv = OUT_DIR/"M2_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["strategy","run","waypoints_done","max_waypoints",
                                        "safety_stops","landed","api_calls","latency_ms",
                                        "cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── M2 Summary ──────────────────────────────────────────")
    summary_csv = OUT_DIR/"M2_summary.csv"
    with open(summary_csv,"w",newline="") as f:
        cw=csv.writer(f)
        cw.writerow(["strategy","wp_rate","wp_lo","wp_hi","calls_mean","lat_ms","cost"])
        for s in PROMPTS:
            sr=[r for r in all_rows if r["strategy"]==s]
            wfrac=[r["waypoints_done"]/r["max_waypoints"] for r in sr]
            wm,wlo,whi=bootstrap_ci(wfrac)
            cm,_,_=bootstrap_ci([r["api_calls"] for r in sr])
            lm,_,_=bootstrap_ci([r["latency_ms"] for r in sr])
            um,_,_=bootstrap_ci([r["cost_usd"] for r in sr])
            cw.writerow([s,wm,wlo,whi,cm,lm,um])
            print(f"  {s:8s} wp={wm:.3f} [{wlo:.3f},{whi:.3f}] calls={cm:.1f} "
                  f"lat={lm:.0f}ms ${um:.5f}")

    print(f"\nData → {runs_csv}  Summary → {summary_csv}")

if __name__ == "__main__":
    main()
