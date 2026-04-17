"""
EXP-J2: MCP Server Latency Benchmark
======================================
REAL HARDWARE ONLY.

Goal:
    Decompose the round-trip latency into three segments for every MCP tool call:
        segment_1: Python → MCP server socket (JSON-RPC overhead)
        segment_2: MCP server → Drone HTTP command (network + FC processing)
        segment_3: Drone telemetry confirmation read-back (loop-back verify)

    N=50 per tool for read-only tools (get_telemetry, capture_frame).
    N=20 per flight tool (arm/disarm only — no full flight sequences here).

Metrics:
    - seg1_ms  : client→server overhead (Bootstrap CI)
    - seg2_ms  : server→drone HTTP time (Bootstrap CI, from server log)
    - seg3_ms  : drone→telemetry confirm (Bootstrap CI)
    - total_ms : end-to-end (Bootstrap CI)
"""

import sys, os, time, csv, math, json, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import MCPClient, drone_get, preflight_check

OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP = os.environ.get("DRONE_IP", "192.168.4.1")
MCP_URL  = os.environ.get("MCP_URL",  "http://localhost:5001/mcp")
N_READ   = 50
N_FLIGHT = 20

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan")
        return v, v, v
    arr = np.array(data, float)
    boots = [stat(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)),4), round(float(lo),4), round(float(hi),4)

def measure_trip(client: MCPClient, tool: str, args: dict) -> dict:
    # seg1: client round-trip to MCP server (ping-only via get_log to avoid drone call)
    t_s1 = time.perf_counter()
    client._call("ping")
    seg1 = (time.perf_counter()-t_s1)*1000

    # seg2+seg3: full tool call (MCP server → drone → back)
    t_full = time.perf_counter()
    r = client.call_tool(tool, args)
    full_ms = (time.perf_counter()-t_full)*1000
    server_lat = r.get("latency_ms", 0)   # server-measured drone HTTP time

    seg2 = server_lat
    seg3 = max(0, full_ms - seg1 - seg2)

    return {
        "tool":    tool,
        "seg1_ms": round(seg1, 2),
        "seg2_ms": round(seg2, 2),
        "seg3_ms": round(seg3, 2),
        "total_ms":round(full_ms, 2),
        "ok": int(not r["content"][0]["text"].startswith("ERROR")),
    }

def main():
    print("="*60)
    print("EXP-J2: MCP Server Latency Benchmark — REAL HARDWARE")
    print(f"DRONE={DRONE_IP}  N_READ={N_READ}  N_FLIGHT={N_FLIGHT}")
    print("="*60)

    if not preflight_check(DRONE_IP, MCP_URL):
        print("ABORT: preflight failed."); return

    client   = MCPClient(MCP_URL)
    all_rows = []

    for tool, args, n in [
        ("get_telemetry",  {}, N_READ),
        ("capture_frame",  {"analyze":False}, N_READ),
        ("speak",          {"message":"latency test"}, N_READ),
    ]:
        print(f"\n--- {tool} (N={n}) ---")
        for i in range(n):
            row = measure_trip(client, tool, args)
            all_rows.append(row)
            if i % 10 == 9:
                recent = [r for r in all_rows if r["tool"]==tool]
                print(f"  [{i+1:3d}] mean_total={np.mean([r['total_ms'] for r in recent]):.0f}ms")

    # arm/disarm latency only (N_FLIGHT pairs)
    print(f"\n--- arm/disarm (N={N_FLIGHT} pairs) ---")
    for i in range(N_FLIGHT):
        row_arm = measure_trip(client, "arm", {})
        time.sleep(0.3)
        row_dis = measure_trip(client, "disarm", {})
        time.sleep(0.3)
        all_rows.extend([row_arm, row_dis])

    # ── Save + stats ───────────────────────────────────────────────────────────
    runs_csv = OUT_DIR/"J2_runs.csv"
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tool","seg1_ms","seg2_ms","seg3_ms","total_ms","ok"])
        w.writeheader(); w.writerows(all_rows)

    summary_csv = OUT_DIR/"J2_summary.csv"
    tools = sorted(set(r["tool"] for r in all_rows))
    with open(summary_csv, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["tool","seg","mean","ci_lo","ci_hi"])
        for t in tools:
            tr = [r for r in all_rows if r["tool"]==t]
            for seg in ("seg1_ms","seg2_ms","seg3_ms","total_ms"):
                m,lo,hi = bootstrap_ci([r[seg] for r in tr])
                cw.writerow([t, seg, m, lo, hi])
            print(f"  {t:20s} total={np.mean([r['total_ms'] for r in tr]):.0f}ms "
                  f"(s1={np.mean([r['seg1_ms'] for r in tr]):.0f} "
                  f"s2={np.mean([r['seg2_ms'] for r in tr]):.0f} "
                  f"s3={np.mean([r['seg3_ms'] for r in tr]):.0f})")

    print(f"\nPer-call data → {runs_csv}")
    print(f"Summary       → {summary_csv}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12,6))
        bottom = np.zeros(len(tools))
        colours = ["#3498db","#e67e22","#2ecc71"]
        for seg, colour in zip(("seg1_ms","seg2_ms","seg3_ms"), colours):
            vals = [np.mean([r[seg] for r in all_rows if r["tool"]==t]) for t in tools]
            ax.bar(tools, vals, bottom=bottom, color=colour, alpha=0.85,
                   label={"seg1_ms":"Client→Server","seg2_ms":"Server→Drone","seg3_ms":"Confirm"}[seg])
            bottom += np.array(vals)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("J2: MCP Round-Trip Latency by Segment")
        ax.legend()
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(OUT_DIR/"J2_latency_breakdown.png", dpi=150)
        plt.close(fig)
        print(f"Plot → {OUT_DIR/'J2_latency_breakdown.png'}")
    except Exception as e:
        print(f"[plot skipped] {e}")

if __name__ == "__main__":
    main()
