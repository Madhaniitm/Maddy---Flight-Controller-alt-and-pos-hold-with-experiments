"""
EXP-N3: Context Length / Conversation History Effect
======================================================
REAL HARDWARE ONLY.

Goal:
    Test whether maintaining conversation history helps the LLM remember
    previous decisions (e.g., initial altitude) when asked to "return to
    initial altitude" after several moves.

    History variants:
        (a) stateless  : each API call gets only the current prompt (no history)
        (b) short      : last 3 turns of history
        (c) full       : full mission history (all turns)

    Scenario:
        1. Arm + takeoff to 1.5m (initial altitude)
        2. Change altitude to 0.8m (via set_altitude)
        3. Ask: "Return to the altitude you started at."
        → Correct: set_altitude(1.5). Wrong: any other value.

    N=5 per variant.

Metrics:
    - correct_recall    : returned to 1.5m (Wilson CI)
    - altitude_error_m  : |returned_alt - 1.5| (Bootstrap CI)
    - tokens_in         : context size (Bootstrap CI)
"""

import sys, os, time, csv, math, json, pathlib, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import preflight_check, MCPClient
from mcp_client import (MCPAgent, call_anthropic, mcp_to_anthropic,
                         SYSTEM_PROMPT)

OUT_DIR   = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP  = os.environ.get("DRONE_IP",  "192.168.4.1")
MCP_URL   = os.environ.get("MCP_URL",   "http://localhost:5001/mcp")
N_RUNS    = 5
INIT_ALT  = 1.5
CHANGED_ALT = 0.8
RECALL_PROMPT = (
    "Return the drone to the altitude you started at in this mission. "
    "Use set_altitude with the correct value."
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

def extract_set_altitude(trace):
    for t in trace:
        if "set_altitude" in t.get("tool",""):
            try:
                return float(t["args"].get("altitude_m", 0))
            except: pass
    return None

def run_mission_with_history(agent, history_mode: str) -> dict:
    mcp = MCPClient(MCP_URL)
    tools = mcp_to_anthropic(mcp.list_tools())
    full_history = []

    # Step 1: arm + takeoff
    def _llm_turn(prompt, history):
        msgs = list(history) + [{"role":"user","content":prompt}]
        reply, tool_uses, in_tok, out_tok, lat = call_anthropic(
            msgs, tools, temperature=0.2
        )
        # Execute first tool use if any
        trace = []
        if tool_uses:
            tu   = tool_uses[0]
            name = tu["name"]
            args = tu.get("input",{})
            r    = mcp.call_tool(name, args)
            out  = r["content"][0]["text"]
            trace.append({"tool":name,"args":args,"result":out[:100]})
        return reply, trace, in_tok, msgs

    step1_prompt = f"Arm the drone and take off to {INIT_ALT}m altitude."
    reply1, tr1, tok1, h1 = _llm_turn(step1_prompt, [])
    full_history.extend(h1)
    full_history.append({"role":"assistant","content":reply1 or "done"})
    time.sleep(4)

    # Step 2: change altitude
    step2_prompt = f"Change the altitude to {CHANGED_ALT}m."
    h_for_s2 = [] if history_mode=="stateless" else \
               full_history[-6:] if history_mode=="short" else full_history
    reply2, tr2, tok2, h2 = _llm_turn(step2_prompt, h_for_s2)
    full_history.extend(h2[-2:])
    full_history.append({"role":"assistant","content":reply2 or "done"})
    time.sleep(3)

    # Step 3: recall
    h_for_s3 = [] if history_mode=="stateless" else \
               full_history[-6:] if history_mode=="short" else full_history
    reply3, tr3, tok3, h3 = _llm_turn(RECALL_PROMPT, h_for_s3)

    returned_alt = extract_set_altitude(tr3)
    correct = int(returned_alt is not None and abs(returned_alt - INIT_ALT) < 0.05)
    alt_err = abs(returned_alt - INIT_ALT) if returned_alt else float("nan")

    return {
        "correct": correct,
        "returned_alt": returned_alt,
        "alt_error_m":  round(alt_err, 3) if not math.isnan(alt_err) else None,
        "tokens_total": tok1+tok2+tok3,
        "trace3": tr3,
    }

def main():
    print("="*60)
    print("EXP-N3: Context Length Effect — REAL HARDWARE")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    all_rows = []
    for hist_mode in ("stateless","short","full"):
        print(f"\n=== History mode: {hist_mode} ===")
        for run in range(1, N_RUNS+1):
            input(f"  [SETUP] Drone on ground. run={run}. Press Enter…")
            try:
                agent  = MCPAgent(model="claude", session_id=f"N3_{hist_mode}_r{run}")
                t0     = time.perf_counter()
                res    = run_mission_with_history(agent, hist_mode)
                lat_ms = (time.perf_counter()-t0)*1000
                row = {"history_mode":hist_mode,"run":run,
                       "correct":res["correct"],
                       "returned_alt":res["returned_alt"],
                       "alt_error_m":res["alt_error_m"],
                       "tokens_total":res["tokens_total"],
                       "latency_ms":round(lat_ms,1),"error":""}
            except Exception as e:
                row = {"history_mode":hist_mode,"run":run,"correct":0,
                       "returned_alt":None,"alt_error_m":None,
                       "tokens_total":0,"latency_ms":0,"error":str(e)[:80]}

            all_rows.append(row)
            print(f"  run={run} correct={row['correct']} "
                  f"ret_alt={row['returned_alt']} err={row['alt_error_m']}")
            time.sleep(6)

    runs_csv = OUT_DIR/"N3_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["history_mode","run","correct","returned_alt",
                                        "alt_error_m","tokens_total","latency_ms","error"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── N3 Summary ──────────────────────────────────────────")
    for hm in ("stateless","short","full"):
        hr=[r for r in all_rows if r["history_mode"]==hm]
        cr,clo,chi=wilson_ci(sum(r["correct"] for r in hr),len(hr))
        errs=[r["alt_error_m"] for r in hr if r["alt_error_m"] is not None]
        em,_,_=bootstrap_ci(errs) if errs else (float("nan"),0,0)
        print(f"  {hm:12s} correct={cr:.3f} [{clo:.3f},{chi:.3f}] "
              f"alt_err={em:.3f}m")

    print(f"\nData → {runs_csv}")

if __name__ == "__main__":
    main()
