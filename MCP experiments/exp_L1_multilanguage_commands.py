"""
EXP-L1: Multilanguage Drone Command Understanding
==================================================
REAL HARDWARE ONLY.

Goal:
    Send drone commands in 5 languages to Claude (best multilingual model)
    via MCP client and verify the CORRECT MCP tool is called.

    Languages: English, Hindi, Tamil, Spanish, French
    Commands per language: arm, takeoff_1m, land, move_forward, emergency_stop

    5 commands × 5 languages × N=5 = 125 real hardware trials.

Metrics:
    - command_accuracy  : correct tool called (Wilson CI) per language
    - latency_ms        : per call (Bootstrap CI) per language
    - translation_notes : any model refusals or misroutes logged
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

COMMANDS = {
    "arm": {
        "expected_tool": "arm",
        "en": "Arm the drone.",
        "hi": "ड्रोन को आर्म करो।",
        "ta": "டிரோனை ஆர்ம் செய்யுங்கள்.",
        "es": "Arma el dron.",
        "fr": "Armez le drone.",
    },
    "takeoff": {
        "expected_tool": "takeoff",
        "en": "Take off to 1 metre altitude.",
        "hi": "1 मीटर ऊंचाई पर उड़ान भरो।",
        "ta": "1 மீட்டர் உயரத்தில் பறக்கவும்.",
        "es": "Despega a 1 metro de altitud.",
        "fr": "Décollez à 1 mètre d'altitude.",
    },
    "land": {
        "expected_tool": "land",
        "en": "Land the drone now.",
        "hi": "अभी ड्रोन को लैंड करो।",
        "ta": "இப்போது டிரோனை தரையிறக்குங்கள்.",
        "es": "Aterriza el dron ahora.",
        "fr": "Faites atterrir le drone maintenant.",
    },
    "move_forward": {
        "expected_tool": "move_forward",
        "en": "Move forward 30 centimetres.",
        "hi": "30 सेंटीमीटर आगे बढ़ो।",
        "ta": "30 சென்டிமீட்டர் முன்னே செல்லுங்கள்.",
        "es": "Avanza 30 centímetros.",
        "fr": "Avancez de 30 centimètres.",
    },
    "emergency_stop": {
        "expected_tool": "emergency_stop",
        "en": "Emergency stop! Stop all motors now!",
        "hi": "आपातकालीन स्टॉप! अभी सभी मोटरें बंद करो!",
        "ta": "அவசர நிறுத்தம்! இப்போது அனைத்து மோட்டார்களையும் நிறுத்துங்கள்!",
        "es": "¡Parada de emergencia! ¡Detén todos los motores ahora!",
        "fr": "Arrêt d'urgence! Arrêtez tous les moteurs maintenant!",
    },
}

LANG_NAMES = {"en":"English","hi":"Hindi","ta":"Tamil","es":"Spanish","fr":"French"}

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
    print("EXP-L1: Multilanguage Command Understanding — REAL HARDWARE")
    print(f"Commands: {list(COMMANDS)}  Languages: {list(LANG_NAMES)}")
    print("="*60)

    if not preflight_check(DRONE_IP, MCP_URL):
        print("ABORT: preflight failed."); return

    all_rows = []
    for cmd_name, cmd_data in COMMANDS.items():
        expected = cmd_data["expected_tool"]
        for lang_code, lang_name in LANG_NAMES.items():
            prompt = cmd_data[lang_code]
            print(f"\n--- {cmd_name} | {lang_name} ---")
            print(f"  Prompt: {prompt}")

            # Safety: arm before move commands, land before exit
            if cmd_name == "move_forward":
                input("  [SETUP] Arm + hover drone manually. Press Enter…")
            elif cmd_name in ("land","emergency_stop"):
                input("  [SETUP] Ensure drone is hovering. Press Enter…")

            for run in range(1, N_RUNS+1):
                try:
                    agent  = MCPAgent(model="claude", session_id=f"L1_{lang_code}_{cmd_name}_r{run}")
                    t0     = time.perf_counter()
                    result = agent.run(prompt, max_turns=4)
                    lat_ms = (time.perf_counter()-t0)*1000

                    tools  = [t["tool"] for t in result["tool_trace"]]
                    correct= int(any(expected in t for t in tools))

                    row = {"lang":lang_code,"lang_name":lang_name,
                           "command":cmd_name,"expected":expected,
                           "tools_called":"|".join(tools),"correct":correct,
                           "latency_ms":round(lat_ms,1),"run":run,"error":""}
                except Exception as e:
                    row = {"lang":lang_code,"lang_name":lang_name,
                           "command":cmd_name,"expected":expected,
                           "tools_called":"","correct":0,
                           "latency_ms":0,"run":run,"error":str(e)[:80]}

                all_rows.append(row)
                status = "✓" if row["correct"] else "✗"
                print(f"  run={run} {status} tools={row['tools_called'][:40]}")
                time.sleep(1.5)

    runs_csv = OUT_DIR/"L1_runs.csv"
    with open(runs_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["lang","lang_name","command","expected",
                                        "tools_called","correct","latency_ms","run","error"])
        w.writeheader(); w.writerows(all_rows)

    summary_csv = OUT_DIR/"L1_summary.csv"
    with open(summary_csv,"w",newline="",encoding="utf-8") as f:
        cw=csv.writer(f)
        cw.writerow(["lang","command","accuracy","ci_lo","ci_hi","lat_mean"])
        print(f"\n── L1 Summary ──────────────────────────────────────────")
        for lang_code in LANG_NAMES:
            for cmd_name in COMMANDS:
                mr=[r for r in all_rows if r["lang"]==lang_code and r["command"]==cmd_name]
                ac,alo,ahi=wilson_ci(sum(r["correct"] for r in mr),len(mr))
                lm,_,_=bootstrap_ci([r["latency_ms"] for r in mr])
                cw.writerow([lang_code,cmd_name,ac,alo,ahi,lm])
            lang_all=[r for r in all_rows if r["lang"]==lang_code]
            ov,olo,ohi=wilson_ci(sum(r["correct"] for r in lang_all),len(lang_all))
            print(f"  {LANG_NAMES[lang_code]:10s} overall={ov:.3f} [{olo:.3f},{ohi:.3f}]")

    print(f"\nData    → {runs_csv}")
    print(f"Summary → {summary_csv}")

if __name__ == "__main__":
    main()
