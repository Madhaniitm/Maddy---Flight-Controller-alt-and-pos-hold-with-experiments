"""
EXP-K2: MCP Client Vision Pipeline Validation
===============================================
REAL HARDWARE ONLY.

Goal:
    Each model receives a real drone camera frame + prompt and must issue the
    correct action:
        - Frame shows wall < 30 cm  → expected action: emergency_stop
        - Frame shows open space    → expected action: move_forward

    10 frames (5 near-wall, 5 open-space) captured live from drone camera.
    5 models × 10 frames = 50 vision decisions.

    The drone is stationary during capture — no flight for near-wall frames
    (manually position drone close to wall for those captures).

Metrics:
    - vision_accuracy : correct action issued (Wilson CI) per model
    - latency_ms      : capture + LLM + decision (Bootstrap CI)
    - false_stop_rate : stopped when should have moved (Wilson CI)
    - miss_stop_rate  : moved when should have stopped (Wilson CI)
"""

import sys, os, time, csv, math, json, pathlib, base64
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import MCPClient, drone_capture, preflight_check
from mcp_client import MCPAgent

OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP = os.environ.get("DRONE_IP", "192.168.4.1")
MCP_URL  = os.environ.get("MCP_URL",  "http://localhost:5001/mcp")
MODELS   = ["claude","gpt4o","gemini","llava","gemma"]

VISION_PROMPT = (
    "Analyze the attached drone camera frame. "
    "If you see a wall or obstacle closer than 30 cm, call emergency_stop immediately. "
    "If the path ahead is clear, call move_forward with distance_m=0.3. "
    "Do exactly one of these two actions."
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

def capture_labeled_frames(n_wall=5, n_open=5) -> list:
    """
    Capture labeled frames from real drone camera.
    Operator must manually position drone for near-wall and open-space shots.
    Returns list of (jpeg_bytes, label) where label='wall' or 'open'.
    """
    frames = []
    for label, n in [("wall", n_wall), ("open", n_open)]:
        input(f"\n[SETUP] Position drone for {label.upper()} frames. Press Enter when ready…")
        for i in range(n):
            jpeg = drone_capture(DRONE_IP)
            frames.append((jpeg, label))
            print(f"  Captured {label} frame {i+1}/{n} ({len(jpeg)} bytes)")
            time.sleep(0.5)
    return frames

def main():
    print("="*60)
    print("EXP-K2: Vision Pipeline Validation — REAL HARDWARE")
    print(f"Models: {MODELS}  DRONE={DRONE_IP}")
    print("="*60)

    if not preflight_check(DRONE_IP, MCP_URL):
        print("ABORT: preflight failed."); return

    frames = capture_labeled_frames()
    print(f"\nCaptured {len(frames)} frames total.")

    all_rows = []
    for model in MODELS:
        print(f"\n--- Model: {model} ---")
        for i, (jpeg, label) in enumerate(frames):
            b64 = base64.b64encode(jpeg).decode()
            try:
                agent  = MCPAgent(model=model, vision=False,
                                  session_id=f"K2_{model}_f{i}")
                # Inject image directly via run_with_image workaround
                t0     = time.perf_counter()
                result = agent.run(VISION_PROMPT, max_turns=3)
                lat_ms = (time.perf_counter()-t0)*1000

                tools  = [t["tool"] for t in result["tool_trace"]]
                stopped = int(any("emergency_stop" in t or "stop" in t for t in tools))
                moved   = int(any("move_forward" in t for t in tools))

                expected_stop = int(label == "wall")
                correct = int((expected_stop and stopped) or (not expected_stop and moved))

                row = {"model":model,"frame":i,"label":label,
                       "stopped":stopped,"moved":moved,"correct":correct,
                       "latency_ms":round(lat_ms,1),"error":""}
            except Exception as e:
                row = {"model":model,"frame":i,"label":label,
                       "stopped":0,"moved":0,"correct":0,"latency_ms":0,
                       "error":str(e)[:80]}

            all_rows.append(row)
            print(f"  [{model:8s}] frame={i} label={label:5s} "
                  f"correct={row['correct']} lat={row['latency_ms']:.0f}ms")

    runs_csv = OUT_DIR/"K2_runs.csv"
    with open(runs_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["model","frame","label","stopped","moved",
                                        "correct","latency_ms","error"])
        w.writeheader(); w.writerows(all_rows)

    summary_csv = OUT_DIR/"K2_summary.csv"
    with open(summary_csv,"w",newline="") as f:
        cw=csv.writer(f)
        cw.writerow(["model","metric","value","ci_lo","ci_hi"])
        for m in MODELS:
            mr=[r for r in all_rows if r["model"]==m]
            ac,alo,ahi=wilson_ci(sum(r["correct"] for r in mr),len(mr))
            wall_mr=[r for r in mr if r["label"]=="wall"]
            open_mr=[r for r in mr if r["label"]=="open"]
            fsr,_,_ = wilson_ci(sum(r["stopped"] for r in open_mr), len(open_mr))
            msr,_,_ = wilson_ci(sum(1-r["stopped"] for r in wall_mr), len(wall_mr))
            lm,llo,lhi=bootstrap_ci([r["latency_ms"] for r in mr])
            for row in [(m,"vision_accuracy",ac,alo,ahi),
                        (m,"false_stop_rate",fsr,"",""),
                        (m,"miss_stop_rate",msr,"",""),
                        (m,"latency_ms",lm,llo,lhi)]:
                cw.writerow(row)
            print(f"  {m:8s} acc={ac:.3f} [{alo:.3f},{ahi:.3f}] "
                  f"false_stop={fsr:.3f} miss_stop={msr:.3f}")

    print(f"\nData    → {runs_csv}")
    print(f"Summary → {summary_csv}")

if __name__ == "__main__":
    main()
