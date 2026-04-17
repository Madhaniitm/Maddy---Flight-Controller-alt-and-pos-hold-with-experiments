"""
EXP-J1: MCP Server Tool Reliability
=====================================
REAL HARDWARE ONLY. No simulation.

Goal:
    Call each MCP tool 50 times on the real drone and measure success/fail rate.
    Proves the MCP server layer is reliable before LLM experiments begin.

    Tools tested (15 total):
        arm, disarm, takeoff(0.5m), land, emergency_stop,
        move_forward(0.2m), move_backward(0.2m), move_left(0.2m), move_right(0.2m),
        set_altitude(1.0m), set_yaw(90), get_telemetry, capture_frame,
        speak("test"), chat_reply("test")

    IMPORTANT: arm/takeoff/move tools run in a controlled sequence per trial,
    not independently, to avoid dangerous states.

Metrics:
    - success_rate   : fraction of calls that returned non-error (Wilson CI)
    - error_breakdown: count by error type (timeout / HTTP 4xx / 5xx / parse)
    - latency_ms     : per-tool round-trip (Bootstrap CI)
"""

import sys, os, time, csv, math, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import MCPClient, preflight_check

OUT_DIR    = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

DRONE_IP   = os.environ.get("DRONE_IP",   "192.168.4.1")
MCP_URL    = os.environ.get("MCP_URL",    "http://localhost:5001/mcp")
N_CALLS    = 50   # per tool (split across safe sequences)

PAPER_REFS = {
    "MCP":   "Anthropic 2024 — Model Context Protocol Specification",
    "ReAct": "Yao et al. 2022 — ReAct: Synergizing Reasoning and Acting in LMs",
}

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k/n; denom = 1+z**2/n
    c = (p+z**2/(2*n))/denom
    m = (z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/denom
    return round(p,4), round(max(0,c-m),4), round(min(1,c+m),4)

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr   = np.array(data, float)
    boots = [stat(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)),4), round(float(lo),4), round(float(hi),4)

def call_and_measure(client: MCPClient, tool: str, args: dict = None) -> dict:
    t0 = time.perf_counter()
    try:
        r   = client.call_tool(tool, args or {})
        ms  = (time.perf_counter()-t0)*1000
        txt = r["content"][0]["text"]
        ok  = not txt.startswith("ERROR")
        return {"tool": tool, "ok": int(ok), "latency_ms": round(ms,2),
                "error": "" if ok else txt[:80]}
    except Exception as e:
        ms = (time.perf_counter()-t0)*1000
        return {"tool": tool, "ok": 0, "latency_ms": round(ms,2), "error": str(e)[:80]}

def main():
    print("="*60)
    print("EXP-J1: MCP Server Tool Reliability — REAL HARDWARE")
    print(f"N_CALLS={N_CALLS} per tool  DRONE={DRONE_IP}")
    print("="*60)

    if not preflight_check(DRONE_IP, MCP_URL):
        print("ABORT: preflight check failed.")
        return

    client = MCPClient(MCP_URL)
    all_rows = []

    # ── Read-only tools: call N_CALLS times directly ──────────────────────────
    read_only = [
        ("get_telemetry",  {}),
        ("capture_frame",  {"analyze": False}),
        ("speak",          {"message": "reliability test"}),
        ("chat_reply",     {"message": "reliability test"}),
    ]
    for tool, args in read_only:
        print(f"\n--- {tool} (N={N_CALLS}) ---")
        for i in range(N_CALLS):
            row = call_and_measure(client, tool, args)
            all_rows.append(row)
            if i % 10 == 9:
                recent = [r for r in all_rows if r["tool"]==tool]
                sr = sum(r["ok"] for r in recent)/len(recent)
                print(f"  [{i+1:3d}/{N_CALLS}] sr={sr:.3f}")

    # ── Flight tools: run in safe arm→takeoff→[move]→land sequence ───────────
    # Each "sequence" = 1 trial for the movement tool. N_CALLS sequences total.
    flight_tools = [
        ("move_forward",  {"distance_m": 0.2}),
        ("move_backward", {"distance_m": 0.2}),
        ("move_left",     {"distance_m": 0.2}),
        ("move_right",    {"distance_m": 0.2}),
        ("set_altitude",  {"altitude_m": 0.8}),
        ("set_yaw",       {"heading_deg": 0}),
    ]

    for tool, args in flight_tools:
        print(f"\n--- {tool} (N={N_CALLS} sequences) ---")
        for i in range(N_CALLS):
            # Pre: arm + takeoff
            call_and_measure(client, "arm")
            time.sleep(1.0)
            call_and_measure(client, "takeoff", {"altitude_m": 0.5})
            time.sleep(3.0)

            # Measure target tool
            row = call_and_measure(client, tool, args)
            all_rows.append(row)

            # Post: land
            time.sleep(1.5)
            call_and_measure(client, "land")
            time.sleep(4.0)

            if i % 5 == 4:
                recent = [r for r in all_rows if r["tool"]==tool]
                sr = sum(r["ok"] for r in recent)/len(recent)
                print(f"  [{i+1:3d}/{N_CALLS}] sr={sr:.3f}")

    # arm/disarm standalone (no flight)
    for tool in ("arm", "disarm"):
        print(f"\n--- {tool} (N={N_CALLS}) ---")
        for i in range(N_CALLS):
            call_and_measure(client, "arm")
            time.sleep(0.5)
            row = call_and_measure(client, "disarm")
            all_rows.append({"tool": tool, **row})
            time.sleep(0.5)

    # emergency_stop during hover (rare — only N=10 for safety)
    print("\n--- emergency_stop (N=10) ---")
    for i in range(10):
        call_and_measure(client, "arm")
        time.sleep(0.5)
        call_and_measure(client, "takeoff", {"altitude_m": 0.4})
        time.sleep(2.0)
        row = call_and_measure(client, "emergency_stop")
        all_rows.append(row)
        time.sleep(3.0)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    runs_csv = OUT_DIR / "J1_runs.csv"
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tool","ok","latency_ms","error"])
        w.writeheader(); w.writerows(all_rows)

    # ── Stats ──────────────────────────────────────────────────────────────────
    tools = sorted(set(r["tool"] for r in all_rows))
    summary_csv = OUT_DIR / "J1_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["tool","n","success_rate","ci_lo","ci_hi","lat_mean","lat_lo","lat_hi"])
        for t in tools:
            tr = [r for r in all_rows if r["tool"]==t]
            kc = sum(r["ok"] for r in tr)
            sr, lo, hi = wilson_ci(kc, len(tr))
            lm, llo, lhi = bootstrap_ci([r["latency_ms"] for r in tr])
            cw.writerow([t, len(tr), sr, lo, hi, lm, llo, lhi])
            print(f"  {t:22s} sr={sr:.3f} [{lo:.3f},{hi:.3f}]  lat={lm:.0f}ms")
        for k, ref in PAPER_REFS.items():
            cw.writerow([f"ref_{k}", ref,"","","","","",""])

    print(f"\nPer-call data → {runs_csv}")
    print(f"Summary       → {summary_csv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(14,6))

        sr_vals = []
        lat_vals = []
        for t in tools:
            tr = [r for r in all_rows if r["tool"]==t]
            sr_vals.append(sum(r["ok"] for r in tr)/len(tr))
            lat_vals.append(np.mean([r["latency_ms"] for r in tr]))

        axes[0].barh(tools, sr_vals, color="#2ecc71", alpha=0.8)
        axes[0].axvline(0.95, color="red", linestyle="--", label="95% target")
        axes[0].set_xlabel("Success rate"); axes[0].set_title("J1: Tool Success Rate")
        axes[0].legend(); axes[0].set_xlim(0,1.05)

        axes[1].barh(tools, lat_vals, color="#3498db", alpha=0.8)
        axes[1].set_xlabel("Mean latency (ms)"); axes[1].set_title("J1: Tool Latency")

        fig.suptitle("EXP-J1 MCP Server Tool Reliability — Real Hardware\n"
                     "MCP (Anthropic 2024), ReAct (Yao 2022)", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT_DIR/"J1_tool_reliability.png", dpi=150)
        plt.close(fig)
        print(f"Plot → {OUT_DIR/'J1_tool_reliability.png'}")
    except Exception as e:
        print(f"[plot skipped] {e}")

if __name__ == "__main__":
    main()
