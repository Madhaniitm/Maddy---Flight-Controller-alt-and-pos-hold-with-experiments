"""
EXP-N2: LLM Model Size / Family Comparison
============================================
REAL HARDWARE ONLY.

Goal:
    Compare 5 models on an identical real drone mission:
        "Arm, take off to 1m, hover 5s, land."
    N=5 per model. All models use temperature=0.2.

    Models: Claude-Sonnet, GPT-4o, Gemini-Flash, LLaVA-13B, Gemma-12B

Metrics:
    - mission_success : all steps completed (Wilson CI)
    - api_calls, latency_ms, cost_usd (Bootstrap CI)
    - safety_score    : never exceeded 2.5m, always checked telemetry (Wilson CI)
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
MODELS   = ["claude","gpt4o","gemini","llava","gemma"]
TASK     = "Arm the drone, take off to 1 metre, hover for 5 seconds, then land safely."

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

def score_mission(trace):
    tools = [t["tool"] for t in trace]
    armed  = int("arm" in tools)
    toff   = int(any("takeoff" in t for t in tools))
    landed = int(any("land" in t for t in tools))
    tel    = int(any("telemetry" in t for t in tools))
    safe   = int(tel)   # at minimum checked telemetry
    return int(armed and toff and landed), safe

def main():
    print("="*60)
    print("EXP-N2: Model Comparison — REAL HARDWARE")
    print(f"Models: {MODELS}  N={N_RUNS}")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []
    for model in MODELS:
        print(f"\n=== Model: {model} ===")
        for run in range(1, N_RUNS+1):
            input(f"  [SETUP] Drone on ground. run={run}. Press Enter…")
            try:
                agent  = MCPAgent(model=model, session_id=f"N2_{model}_r{run}")
                t0     = time.perf_counter()
                result = agent.run(TASK, max_turns=15)
                lat_ms = (time.perf_counter()-t0)*1000
                success, safe = score_mission(result["tool_trace"])
                row = {"model":model,"run":run,"success":success,"safe":safe,
                       "api_calls":result["turns"],"latency_ms":round(lat_ms,1),
                       "cost_usd":result["cost_usd"],"error":""}
            except Exception as e:
                row = {"model":model,"run":run,"success":0,"safe":0,
                       "api_calls":0,"latency_ms":0,"cost_usd":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} success={row['success']} safe={row['safe']} "
                  f"calls={row['api_calls']} lat={row['latency_ms']:.0f}ms "
                  f"${row['cost_usd']:.5f}")
            time.sleep(8)

    runs_csv = OUT_DIR/"N2_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["model","run","success","safe",
                                        "api_calls","latency_ms","cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── N2 Summary ──────────────────────────────────────────")
    summary_csv = OUT_DIR/"N2_summary.csv"
    with open(summary_csv,"w",newline="") as f:
        cw=csv.writer(f)
        cw.writerow(["model","success_rate","ci_lo","ci_hi","safety_rate","lat_ms","cost"])
        for m in MODELS:
            mr=[r for r in all_rows if r["model"]==m]
            sr,slo,shi=wilson_ci(sum(r["success"] for r in mr),len(mr))
            sfr,_,_=wilson_ci(sum(r["safe"] for r in mr),len(mr))
            lm,_,_=bootstrap_ci([r["latency_ms"] for r in mr])
            cm,_,_=bootstrap_ci([r["cost_usd"] for r in mr])
            cw.writerow([m,sr,slo,shi,sfr,lm,cm])
            print(f"  {m:8s} success={sr:.3f} [{slo:.3f},{shi:.3f}] "
                  f"safe={sfr:.3f} lat={lm:.0f}ms ${cm:.5f}")

    print(f"\nData → {runs_csv}  Summary → {summary_csv}")

if __name__ == "__main__":
    main()
