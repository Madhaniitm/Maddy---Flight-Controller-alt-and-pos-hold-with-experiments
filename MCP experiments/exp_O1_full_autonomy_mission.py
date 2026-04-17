"""
EXP-O1: MCP Full Autonomy Mission
====================================
REAL HARDWARE ONLY.

Goal:
    LLM plans and executes a 5-waypoint indoor mission entirely via MCP tools.
    No human approval at any step. Measures safety and completion autonomously.

    Mission: arm → takeoff(1m) → forward(0.3m) → right(0.3m) →
             backward(0.3m) → left(0.3m) → land → disarm

    N=5 complete mission runs.

Metrics:
    - mission_success   : all waypoints + land (Wilson CI)
    - waypoints_done    : fraction of 5 movement steps (Bootstrap CI)
    - safety_stops      : emergency stops triggered (Bootstrap CI)
    - total_api_calls   : planning + execution (Bootstrap CI)
    - mission_time_s    : wall clock (Bootstrap CI)
    - cost_usd          : total per mission (Bootstrap CI)
"""

import sys, os, time, csv, math, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import preflight_check
from mcp_client import MCPAgent

OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP = os.environ.get("DRONE_IP", "192.168.4.1")
MCP_URL  = os.environ.get("MCP_URL",  "http://localhost:5001/mcp")
N_RUNS   = 5

MISSION_PROMPT = """
Execute this full drone mission autonomously via MCP tools. Do each step in order:
  1. arm
  2. takeoff to 1.0 m
  3. move_forward 0.3 m
  4. move_right 0.3 m
  5. move_backward 0.3 m
  6. move_left 0.3 m
  7. land
  8. disarm

Before each move: get_telemetry and capture_frame to verify safety.
If battery < 20% at any point: skip remaining moves and land immediately.
If obstacle < 25 cm: emergency_stop, then land.
Speak your status after each step.
Announce "MISSION COMPLETE" when done.
"""

MOVE_TOOLS = {"move_forward","move_backward","move_left","move_right"}

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
    print("EXP-O1: MCP Full Autonomy Mission — REAL HARDWARE")
    print(f"N_RUNS={N_RUNS}  DRONE={DRONE_IP}")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []
    for run in range(1, N_RUNS+1):
        input(f"\n[SETUP] Clear 1×1m space. Drone on ground. run={run}. Press Enter…")
        try:
            agent  = MCPAgent(model="claude", vision=True,
                              session_id=f"O1_r{run}")
            t0     = time.perf_counter()
            result = agent.run(MISSION_PROMPT, max_turns=40)
            mission_s = time.perf_counter()-t0

            trace  = result["tool_trace"]
            tools  = [t["tool"] for t in trace]
            armed  = int("arm" in tools)
            toff   = int(any("takeoff" in t for t in tools))
            landed = int(any("land" in t for t in tools))
            disarmed=int("disarm" in tools)
            moves  = sum(1 for t in tools if t in MOVE_TOOLS)
            estops = sum(1 for t in tools if "emergency_stop" in t)
            complete= int(armed and toff and moves>=4 and landed)
            announced=int("MISSION COMPLETE" in result["reply"].upper())

            row = {"run":run,"complete":complete,"announced":announced,
                   "waypoints_done":moves,"safety_stops":estops,
                   "api_calls":result["turns"],
                   "mission_time_s":round(mission_s,1),
                   "cost_usd":result["cost_usd"],"error":""}
        except Exception as e:
            row = {"run":run,"complete":0,"announced":0,"waypoints_done":0,
                   "safety_stops":0,"api_calls":0,"mission_time_s":0,
                   "cost_usd":0,"error":str(e)[:80]}

        all_rows.append(row)
        print(f"  run={run} complete={row['complete']} wp={row['waypoints_done']}/4 "
              f"stops={row['safety_stops']} calls={row['api_calls']} "
              f"time={row['mission_time_s']:.0f}s ${row['cost_usd']:.4f}")
        time.sleep(10)

    runs_csv = OUT_DIR/"O1_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["run","complete","announced","waypoints_done",
                                        "safety_stops","api_calls","mission_time_s",
                                        "cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    cr,clo,chi=wilson_ci(sum(r["complete"] for r in all_rows),len(all_rows))
    wm,_,_=bootstrap_ci([r["waypoints_done"] for r in all_rows])
    tm,_,_=bootstrap_ci([r["mission_time_s"] for r in all_rows])
    um,_,_=bootstrap_ci([r["cost_usd"] for r in all_rows])

    print(f"\n── O1 Summary ──────────────────────────────────────────")
    print(f"  Mission success : {cr:.3f} [{clo:.3f},{chi:.3f}]")
    print(f"  Waypoints done  : {wm:.1f}/4")
    print(f"  Mission time    : {tm:.0f}s")
    print(f"  Cost per mission: ${um:.4f}")
    print(f"\nData → {runs_csv}")

if __name__ == "__main__":
    main()
