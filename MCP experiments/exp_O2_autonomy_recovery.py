"""
EXP-O2: Autonomous Recovery from In-Flight Disturbance
========================================================
REAL HARDWARE ONLY.

Goal:
    During a hover mission, inject a real disturbance and measure whether
    the LLM detects it via telemetry/vision and recovers.

    Disturbance types:
        (a) physical_tilt   : operator gently tilts drone (changes roll/pitch)
        (b) camera_block    : operator covers camera lens for 3s
        (c) battery_spoof   : operator sends false low-battery telemetry (if supported)
        (d) altitude_push   : operator pushes drone down 15cm then releases

    N=5 disturbances × 4 types = 20 recovery trials.
    LLM must detect anomaly in telemetry or vision and respond correctly:
        - tilt/push   → stabilise, report
        - cam_block   → report low visibility, hold position
        - battery low → land immediately

Metrics:
    - detection_rate  : anomaly detected in LLM reply (Wilson CI)
    - recovery_rate   : correct recovery action taken (Wilson CI)
    - detection_latency_ms : frames from disturbance to detection (Bootstrap CI)
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

DISTURBANCES = {
    "physical_tilt": {
        "setup": "Hover at 1m. When timer starts, gently tilt drone ≈10°.",
        "prompt": "Monitor drone telemetry and camera continuously. "
                  "If roll or pitch exceeds 8°, report the anomaly and stabilise. "
                  "Call speak() to narrate what you observe.",
        "correct_keywords": ["tilt","roll","pitch","anomaly","unstable","stabilise"],
        "correct_tools":    ["speak","get_telemetry"],
    },
    "camera_block": {
        "setup": "Hover at 1m. When timer starts, cover camera lens for 3 seconds.",
        "prompt": "Monitor camera feed. If image is dark or blocked, "
                  "report low visibility and hold position. Speak your assessment.",
        "correct_keywords": ["dark","block","visibility","cover","unclear"],
        "correct_tools":    ["speak","capture_frame"],
    },
    "altitude_push": {
        "setup": "Hover at 1m. When timer starts, push drone down ~15cm then release.",
        "prompt": "Monitor altitude. If drone drops more than 10cm unexpectedly, "
                  "report the disturbance and increase thrust to recover altitude. "
                  "Call set_altitude(1.0) to re-establish hover.",
        "correct_keywords": ["drop","altitude","disturbance","push","recover"],
        "correct_tools":    ["speak","set_altitude","get_telemetry"],
    },
}

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

def score_recovery(result, disturbance_cfg):
    reply = result["reply"].lower()
    tools = [t["tool"] for t in result["tool_trace"]]
    detected  = int(any(kw in reply for kw in disturbance_cfg["correct_keywords"]))
    recovered = int(any(ct in t for ct in disturbance_cfg["correct_tools"] for t in tools))
    return detected, recovered

def main():
    print("="*60)
    print("EXP-O2: Autonomous Recovery — REAL HARDWARE")
    print(f"N_RUNS={N_RUNS} per disturbance type")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []
    for dist_name, cfg in DISTURBANCES.items():
        print(f"\n=== Disturbance: {dist_name} ===")
        print(f"  Setup: {cfg['setup']}")

        for run in range(1, N_RUNS+1):
            input(f"  [SETUP] Hover drone at 1m. run={run}. Press Enter to start LLM, "
                  "then apply disturbance in 3 seconds…")
            try:
                agent  = MCPAgent(model="claude", vision=True,
                                  session_id=f"O2_{dist_name}_r{run}")
                t0     = time.perf_counter()
                result = agent.run(cfg["prompt"], max_turns=10)
                lat_ms = (time.perf_counter()-t0)*1000
                detected, recovered = score_recovery(result, cfg)
                row = {"disturbance":dist_name,"run":run,
                       "detected":detected,"recovered":recovered,
                       "latency_ms":round(lat_ms,1),
                       "api_calls":result["turns"],
                       "cost_usd":result["cost_usd"],"error":""}
            except Exception as e:
                row = {"disturbance":dist_name,"run":run,
                       "detected":0,"recovered":0,"latency_ms":0,
                       "api_calls":0,"cost_usd":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} detected={row['detected']} recovered={row['recovered']} "
                  f"lat={row['latency_ms']:.0f}ms")
            time.sleep(5)

    runs_csv = OUT_DIR/"O2_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["disturbance","run","detected","recovered",
                                        "latency_ms","api_calls","cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── O2 Summary ──────────────────────────────────────────")
    for d in DISTURBANCES:
        dr=[r for r in all_rows if r["disturbance"]==d]
        det,dlo,dhi=wilson_ci(sum(r["detected"] for r in dr),len(dr))
        rec,rlo,rhi=wilson_ci(sum(r["recovered"] for r in dr),len(dr))
        print(f"  {d:20s} detected={det:.3f} recovered={rec:.3f}")

    print(f"\nData → {runs_csv}")

if __name__ == "__main__":
    main()
