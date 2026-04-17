"""
EXP-P1: Human-in-the-Loop Approval Latency
============================================
REAL HARDWARE ONLY.

Goal:
    Compare total mission time between full-auto and HITL modes on the same
    5-waypoint mission. In HITL mode, human approves each action via terminal.
    Auto-approve after 10s timeout (to prevent blocking).

    N=5 full missions per mode (10 total runs).

Metrics:
    - mission_time_s    : total wall clock (Bootstrap CI)
    - approval_time_s   : mean human decision time per action (Bootstrap CI)
    - hitl_overhead_pct : (hitl_time - auto_time) / auto_time * 100
    - safety_catches    : actions rejected by human that LLM proposed (Wilson CI)
    - waypoints_done    : task completion rate (Bootstrap CI)
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
AUTO_APPROVE_TIMEOUT_S = 10.0

MISSION = (
    "Arm, take off to 1m, move forward 0.3m, move right 0.3m, "
    "move backward 0.3m, move left 0.3m, then land and disarm."
)

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

def timed_input(prompt_text: str, timeout: float) -> tuple:
    """Non-blocking input with timeout. Returns (answer, decision_time_s)."""
    import threading
    result = [None]
    t0     = time.perf_counter()

    def _read():
        try:
            result[0] = input(prompt_text)
        except Exception:
            result[0] = ""

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=timeout)
    dt = time.perf_counter()-t0
    return (result[0] or "y"), round(dt, 2)

def main():
    print("="*60)
    print("EXP-P1: HITL Approval Latency — REAL HARDWARE")
    print(f"N_RUNS={N_RUNS} per mode")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []

    for mode in ("full_auto", "hitl"):
        print(f"\n=== Mode: {mode} ===")
        for run in range(1, N_RUNS+1):
            input(f"  [SETUP] Drone on ground. run={run}. Press Enter…")
            approval_times = []
            rejections     = [0]

            def hitl_hook(tool_name: str, args: dict) -> bool:
                if tool_name in ("get_telemetry","capture_frame","speak","chat_reply"):
                    return True
                ans, dt = timed_input(
                    f"  [HITL] Approve '{tool_name}'? [Y/n]: ", AUTO_APPROVE_TIMEOUT_S
                )
                approval_times.append(dt)
                if ans.strip().lower() in ("n","no"):
                    rejections[0] += 1
                    return False
                return True

            try:
                agent = MCPAgent(model="claude",
                                 hitl=(mode=="hitl"),
                                 vision=True,
                                 session_id=f"P1_{mode}_r{run}")
                if mode == "hitl":
                    # Monkey-patch the hitl approve method
                    agent._hitl_approve = lambda name, args: hitl_hook(name, args)

                t0     = time.perf_counter()
                result = agent.run(MISSION, max_turns=30)
                mission_s = time.perf_counter()-t0

                trace = result["tool_trace"]
                tools = [t["tool"] for t in trace]
                wp    = sum(1 for t in tools if t in
                            {"move_forward","move_backward","move_left","move_right"})
                landed= int(any("land" in t for t in tools))

                row = {"mode":mode,"run":run,
                       "mission_time_s":round(mission_s,1),
                       "mean_approval_s":round(np.mean(approval_times),2) if approval_times else 0,
                       "rejections":rejections[0],
                       "waypoints_done":wp,"landed":landed,
                       "api_calls":result["turns"],
                       "cost_usd":result["cost_usd"],"error":""}
            except Exception as e:
                row = {"mode":mode,"run":run,"mission_time_s":0,
                       "mean_approval_s":0,"rejections":0,"waypoints_done":0,
                       "landed":0,"api_calls":0,"cost_usd":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} time={row['mission_time_s']:.0f}s "
                  f"approvals={row['mean_approval_s']:.1f}s "
                  f"rejects={row['rejections']} wp={row['waypoints_done']}/4")
            time.sleep(10)

    runs_csv = OUT_DIR/"P1_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["mode","run","mission_time_s","mean_approval_s",
                                        "rejections","waypoints_done","landed",
                                        "api_calls","cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── P1 Summary ──────────────────────────────────────────")
    auto_t = [r["mission_time_s"] for r in all_rows if r["mode"]=="full_auto"]
    hitl_t = [r["mission_time_s"] for r in all_rows if r["mode"]=="hitl"]
    at_m,_,_=bootstrap_ci(auto_t); ht_m,_,_=bootstrap_ci(hitl_t)
    overhead = round((ht_m-at_m)/at_m*100,1) if at_m>0 else 0
    rej_total = sum(r["rejections"] for r in all_rows if r["mode"]=="hitl")
    print(f"  Full-auto time : {at_m:.0f}s")
    print(f"  HITL time      : {ht_m:.0f}s")
    print(f"  HITL overhead  : {overhead}%")
    print(f"  Rejections     : {rej_total}")
    print(f"\nData → {runs_csv}")

if __name__ == "__main__":
    main()
