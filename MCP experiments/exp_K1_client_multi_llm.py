"""
EXP-K1: MCP Client Multi-LLM Validation
==========================================
REAL HARDWARE ONLY.

Goal:
    Validate that all 5 LLM backends (Claude, GPT-4o, Gemini, LLaVA, Gemma)
    can successfully connect to the MCP server, discover tools, issue the
    correct drone commands, and read telemetry for a simple mission:
        "Arm the drone and report battery and altitude."

    N=5 runs per model. Measures:
        - connect_success   : MCP tools/list succeeds (Wilson CI)
        - tool_call_accuracy: correct tool called (arm + get_telemetry) (Wilson CI)
        - reply_quality     : rubric 0–3 (mentions arm/battery/altitude)
        - latency_ms        : end-to-end per run (Bootstrap CI)
        - cost_usd          : per run (Bootstrap CI)
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
MODELS   = ["claude", "gpt4o", "gemini", "llava", "gemma"]
TASK     = "Arm the drone and report current battery percentage and altitude."

PAPER_REFS = {
    "MCP":    "Anthropic 2024 — Model Context Protocol Specification",
    "GPT4V":  "Achiam et al. 2023 — GPT-4 Technical Report",
    "Gemini": "Reid et al. 2024 — Gemini 1.5: Unlocking multimodal understanding",
    "LLaVA":  "Liu et al. 2023 — Visual Instruction Tuning (LLaVA)",
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

def score_reply(reply: str, tools_called: list) -> int:
    r = reply.lower()
    s  = int("arm" in r or any("arm" in t for t in tools_called))
    s += int("battery" in r or "%" in r)
    s += int("altitude" in r or "metre" in r or "meter" in r or "m " in r)
    return s

def main():
    print("="*60)
    print("EXP-K1: MCP Client Multi-LLM Validation — REAL HARDWARE")
    print(f"Models: {MODELS}  N={N_RUNS}  DRONE={DRONE_IP}")
    print("="*60)

    if not preflight_check(DRONE_IP, MCP_URL):
        print("ABORT: preflight failed."); return

    all_rows = []
    for model in MODELS:
        print(f"\n--- Model: {model} ---")
        for run in range(1, N_RUNS+1):
            try:
                agent = MCPAgent(model=model, session_id=f"K1_{model}_r{run}")
                t0    = time.perf_counter()
                result= agent.run(TASK, max_turns=6)
                elapsed = (time.perf_counter()-t0)*1000

                tools = [t["tool"] for t in result["tool_trace"]]
                arm_ok = int(any("arm" in t for t in tools))
                tel_ok = int(any("telemetry" in t for t in tools))
                qual   = score_reply(result["reply"], tools)

                row = {
                    "model": model, "run": run,
                    "arm_called": arm_ok, "telemetry_called": tel_ok,
                    "quality": qual, "latency_ms": round(elapsed,1),
                    "cost_usd": result["cost_usd"],
                    "error": "",
                }
            except Exception as e:
                row = {"model":model,"run":run,"arm_called":0,"telemetry_called":0,
                       "quality":0,"latency_ms":0,"cost_usd":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} arm={row['arm_called']} tel={row['telemetry_called']} "
                  f"qual={row['quality']}/3 lat={row['latency_ms']:.0f}ms")
            time.sleep(2)  # safety gap between runs

    runs_csv = OUT_DIR/"K1_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["model","run","arm_called","telemetry_called",
                                        "quality","latency_ms","cost_usd","error"])
        w.writeheader(); w.writerows(all_rows)

    summary_csv = OUT_DIR/"K1_summary.csv"
    with open(summary_csv,"w",newline="") as f:
        cw=csv.writer(f)
        cw.writerow(["model","metric","value","ci_lo","ci_hi","note"])
        for m in MODELS:
            mr=[r for r in all_rows if r["model"]==m]
            arm_r,alo,ahi=wilson_ci(sum(r["arm_called"] for r in mr),len(mr))
            tel_r,tlo,thi=wilson_ci(sum(r["telemetry_called"] for r in mr),len(mr))
            lm,llo,lhi=bootstrap_ci([r["latency_ms"] for r in mr])
            qm,qlo,qhi=bootstrap_ci([r["quality"] for r in mr])
            for row in [(m,"arm_success",arm_r,alo,ahi,"Wilson 95%"),
                        (m,"telemetry_success",tel_r,tlo,thi,"Wilson 95%"),
                        (m,"reply_quality",qm,qlo,qhi,"Bootstrap 95%"),
                        (m,"latency_ms",lm,llo,lhi,"Bootstrap 95%")]:
                cw.writerow(row)
            print(f"  {m:8s} arm={arm_r:.3f} tel={tel_r:.3f} "
                  f"qual={qm:.2f}/3 lat={lm:.0f}ms")
        for k,ref in PAPER_REFS.items():
            cw.writerow([f"ref_{k}",ref,"","","",""])

    print(f"\nData    → {runs_csv}")
    print(f"Summary → {summary_csv}")

if __name__ == "__main__":
    main()
