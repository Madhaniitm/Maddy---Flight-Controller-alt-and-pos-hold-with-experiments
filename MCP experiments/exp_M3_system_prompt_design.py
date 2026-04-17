"""
EXP-M3: System Prompt Design Comparison
=========================================
REAL HARDWARE ONLY.

Goal:
    Compare three system prompt styles on the same hover-and-report task:
        (a) terse      : 50 words — minimal instruction
        (b) verbose    : 300 words — full drone spec
        (c) structured : JSON-schema-guided output format

    Task: "Hover at 1m, check battery, report altitude and heading, then land."
    N=5 per style. Drone executes real flight.

Metrics:
    - task_completion  : all 4 sub-tasks done (Wilson CI)
    - decision_accuracy: correct tool sequence (Wilson CI)
    - reply_quality    : rubric 0-4 (battery/altitude/heading/land mentioned)
    - api_calls, latency_ms, cost_usd (Bootstrap CI)
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

TASK = "Arm, take off to 1m, hover for 3 seconds, check battery and heading, then land."

PROMPTS = {
    "terse": (
        "You control a drone. Use the provided tools. "
        "Always check telemetry first. Stop if battery < 20%. Max altitude 2.5m."
    ),
    "verbose": (
        "You are an autonomous drone controller for the Maddy Flight Controller "
        "(ESP32-S3 based quadrotor). You have MCP tools to issue flight commands.\n\n"
        "SAFETY RULES:\n"
        "  1. Always call get_telemetry before any movement.\n"
        "  2. Land immediately if battery < 20%.\n"
        "  3. Emergency stop if obstacle < 25cm.\n"
        "  4. Maximum altitude: 2.5m.\n\n"
        "WORKFLOW:\n"
        "  get_telemetry → assess → act → speak decision → repeat.\n\n"
        "TOOL REFERENCE:\n"
        "  arm/disarm, takeoff(alt), land, emergency_stop,\n"
        "  move_forward/backward/left/right(dist),\n"
        "  set_altitude(alt), set_yaw(deg),\n"
        "  get_telemetry, capture_frame(analyze),\n"
        "  speak(msg), chat_reply(msg).\n\n"
        "Always narrate your decisions via speak() and confirm completion."
    ),
    "structured": (
        "You are a drone controller. Respond with a structured action plan.\n"
        "For each step output:\n"
        '{"step": N, "observation": "...", "decision": "...", "tool": "name", "args": {...}}\n'
        "Then execute the tool. Safety: battery<20% → land. obstacle<25cm → stop. "
        "Max alt 2.5m. Use get_telemetry before every action."
    ),
}

def score_reply(reply: str, trace: list) -> int:
    r = reply.lower()
    tools = [t["tool"] for t in trace]
    s  = int("battery" in r or "%" in r)
    s += int("altitude" in r or "metre" in r or "m " in r or "height" in r)
    s += int("heading" in r or "yaw" in r or "degree" in r or "°" in r)
    s += int(any("land" in t for t in tools))
    return s

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data)<2:
        v=float(stat(data)) if data else float("nan"); return v,v,v
    arr=np.array(data,float)
    boots=[stat(np.random.choice(arr,len(arr),replace=True)) for _ in range(n_boot)]
    lo,hi=np.percentile(boots,[100*alpha/2,100*(1-alpha/2)])
    return round(float(stat(arr)),4),round(float(lo),4),round(float(hi),4)

def wilson_ci(k,n,z=1.96):
    if n==0: return 0.,0.,0.
    p=k/n; d=1+z**2/n
    c=(p+z**2/(2*n))/d; m=(z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/d
    return round(p,4),round(max(0,c-m),4),round(min(1,c+m),4)

def main():
    print("="*60)
    print("EXP-M3: System Prompt Design — REAL HARDWARE")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []
    import mcp_client as mc_mod

    for style, sys_prompt in PROMPTS.items():
        print(f"\n=== Style: {style} ===")
        # Temporarily patch system prompt in the module
        original_sp = mc_mod.SYSTEM_PROMPT
        mc_mod.SYSTEM_PROMPT = sys_prompt

        for run in range(1, N_RUNS+1):
            input(f"  [SETUP] Drone on ground, run={run}. Press Enter…")
            try:
                agent  = MCPAgent(model="claude", session_id=f"M3_{style}_r{run}")
                t0     = time.perf_counter()
                result = agent.run(TASK, max_turns=20)
                lat_ms = (time.perf_counter()-t0)*1000
                trace  = result["tool_trace"]

                tools  = [t["tool"] for t in trace]
                armed  = int(any("arm" == t for t in tools))
                took_off=int(any("takeoff" in t for t in tools))
                landed = int(any("land" in t for t in tools))
                got_tel= int(any("telemetry" in t for t in tools))
                complete= int(armed and took_off and landed and got_tel)
                qual   = score_reply(result["reply"], trace)

                row = {"style":style,"run":run,"complete":complete,"quality":qual,
                       "api_calls":result["turns"],"latency_ms":round(lat_ms,1),
                       "cost_usd":result["cost_usd"],"error":""}
            except Exception as e:
                row = {"style":style,"run":run,"complete":0,"quality":0,
                       "api_calls":0,"latency_ms":0,"cost_usd":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} complete={row['complete']} qual={row['quality']}/4 "
                  f"calls={row['api_calls']} lat={row['latency_ms']:.0f}ms")
            time.sleep(5)

        mc_mod.SYSTEM_PROMPT = original_sp  # restore

    runs_csv = OUT_DIR/"M3_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["style","run","complete","quality",
                                        "api_calls","latency_ms","cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── M3 Summary ──────────────────────────────────────────")
    summary_csv = OUT_DIR/"M3_summary.csv"
    with open(summary_csv,"w",newline="") as f:
        cw=csv.writer(f)
        cw.writerow(["style","completion_rate","ci_lo","ci_hi","quality_mean","calls","cost"])
        for s in PROMPTS:
            sr=[r for r in all_rows if r["style"]==s]
            cr,clo,chi=wilson_ci(sum(r["complete"] for r in sr),len(sr))
            qm,_,_=bootstrap_ci([r["quality"] for r in sr])
            cm,_,_=bootstrap_ci([r["api_calls"] for r in sr])
            um,_,_=bootstrap_ci([r["cost_usd"] for r in sr])
            cw.writerow([s,cr,clo,chi,qm,cm,um])
            print(f"  {s:12s} comp={cr:.3f} qual={qm:.2f}/4 "
                  f"calls={cm:.1f} ${um:.5f}")

    print(f"\nData → {runs_csv}  Summary → {summary_csv}")

if __name__ == "__main__":
    main()
