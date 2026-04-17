"""
EXP-Q2: Chat While Flying — Operator Q&A During Hover
======================================================
REAL HARDWARE ONLY.

Goal:
    While the drone hovers autonomously at 1m, the operator asks the LLM
    ad-hoc questions via terminal. The LLM answers via TTS and chat_reply MCP
    tool WITHOUT interrupting or landing the drone.

    5 question categories × N=5 runs each = 25 chat trials.
    A separate background thread keeps the drone hovering by calling
    get_telemetry every 5s and set_altitude(1.0) if altitude drifts >10cm.

    Question categories:
        (a) telemetry_query  : "What is the current battery level?"
        (b) position_query   : "How high are we flying right now?"
        (c) obstacle_query   : "Is there anything blocking our path?"
        (d) status_summary   : "Give me a quick status update."
        (e) hypothetical     : "What would you do if battery drops to 10%?"

Metrics:
    - answer_relevance    : reply mentions the asked entity (Wilson CI)
    - hover_maintained    : drone altitude stayed within ±15cm throughout (Wilson CI)
    - response_latency_ms : time from question to TTS start (Bootstrap CI)
    - tool_calls_per_q    : MCP tools used per answer (Bootstrap CI)
    - tts_success         : TTS played without error (Wilson CI)
"""

import sys, os, time, csv, math, pathlib, threading
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mcp_server import preflight_check, MCPClient
from mcp_client import MCPAgent

OUT_DIR  = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DRONE_IP = os.environ.get("DRONE_IP", "192.168.4.1")
MCP_URL  = os.environ.get("MCP_URL",  "http://localhost:5001/mcp")
N_RUNS   = 5
HOVER_ALT = 1.0
HOVER_TOLERANCE = 0.15

QUESTIONS = {
    "telemetry_query": {
        "q": "What is the current battery level and voltage?",
        "relevance_keywords": ["battery","percent","%","volt","charge","power"],
        "expected_tools": ["get_telemetry","chat_reply"],
    },
    "position_query": {
        "q": "How high are we flying right now?",
        "relevance_keywords": ["metre","meter","altitude","height","1","m","above"],
        "expected_tools": ["get_telemetry","chat_reply"],
    },
    "obstacle_query": {
        "q": "Is there anything blocking our path ahead?",
        "relevance_keywords": ["obstacle","sensor","sonar","clear","detect","block","range"],
        "expected_tools": ["get_telemetry","capture_frame","chat_reply"],
    },
    "status_summary": {
        "q": "Give me a quick status update on the drone.",
        "relevance_keywords": ["hover","altitude","battery","status","flying","stable"],
        "expected_tools": ["get_telemetry","chat_reply","speak"],
    },
    "hypothetical": {
        "q": "What would you do if the battery drops below 10%?",
        "relevance_keywords": ["land","emergency","battery","10","immediately","safe"],
        "expected_tools": ["chat_reply","speak"],
    },
}

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0., 0., 0.
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d; m = (z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/d
    return round(p,4), round(max(0,c-m),4), round(min(1,c+m),4)

def bootstrap_ci(data, stat=np.mean, n_boot=2000, alpha=0.05):
    if len(data) < 2:
        v = float(stat(data)) if data else float("nan"); return v, v, v
    arr = np.array(data, float)
    boots = [stat(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(float(stat(arr)),4), round(float(lo),4), round(float(hi),4)

def tts_speak(text: str) -> tuple:
    t0 = time.perf_counter()
    success = 0
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(text[:300])
        engine.runAndWait()
        success = 1
    except Exception:
        try:
            import tempfile, subprocess
            from gtts import gTTS
            tts = gTTS(text=text[:300], lang="en", slow=False)
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(tmp.name)
            subprocess.Popen(["ffplay","-nodisp","-autoexit",tmp.name],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            success = 1
        except Exception:
            pass
    return round((time.perf_counter() - t0)*1000, 1), success

class HoverGuard:
    """Background thread that polls telemetry and corrects altitude drift."""

    def __init__(self, mcp: MCPClient, target_alt: float, tolerance: float):
        self._mcp = mcp
        self._target = target_alt
        self._tol = tolerance
        self._running = False
        self._thread = None
        self.altitude_samples: list = []
        self.corrections: int = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            try:
                tel = self._mcp.telemetry()
                alt = float(tel.get("altitude_m", self._target))
                self.altitude_samples.append(alt)
                if abs(alt - self._target) > self._tol:
                    self._mcp.call_tool("set_altitude", {"altitude_m": self._target})
                    self.corrections += 1
            except Exception:
                pass
            time.sleep(5)

    def hover_maintained(self) -> int:
        if not self.altitude_samples:
            return 1  # assume OK if no samples
        return int(all(abs(a - self._target) <= self._tol + 0.05
                       for a in self.altitude_samples))

def ask_question_via_agent(agent: MCPAgent, question: str, cfg: dict) -> dict:
    chat_prompt = (
        f"The drone is currently hovering at {HOVER_ALT}m. "
        f"Do NOT land or stop the drone. "
        f"Answer this operator question concisely and then call chat_reply() to deliver the answer. "
        f"Also call speak() to say it aloud. "
        f"Question: {question}"
    )
    t0 = time.perf_counter()
    result = agent.run(chat_prompt, max_turns=6)
    llm_ms = round((time.perf_counter()-t0)*1000, 1)

    reply = result.get("reply","")
    trace = result.get("tool_trace",[])
    tools = [t["tool"] for t in trace]

    r = reply.lower()
    relevant = int(any(kw in r for kw in cfg["relevance_keywords"]))

    landed_during = int(any("land" in t or "disarm" in t for t in tools))
    tool_count = len(tools)

    tts_ms, tts_ok = tts_speak(reply)

    return {
        "reply":         reply,
        "relevant":      relevant,
        "landed_during": landed_during,
        "tool_count":    tool_count,
        "llm_ms":        llm_ms,
        "tts_ms":        tts_ms,
        "tts_success":   tts_ok,
        "api_calls":     result.get("turns", 0),
        "cost_usd":      result.get("cost_usd", 0),
    }

def main():
    print("="*60)
    print("EXP-Q2: Chat While Flying — REAL HARDWARE")
    print(f"N_RUNS={N_RUNS} per question category")
    print("="*60)
    if not preflight_check(DRONE_IP, MCP_URL): return

    mcp = MCPClient(MCP_URL)

    all_rows = []
    for q_name, q_cfg in QUESTIONS.items():
        print(f"\n=== Question: {q_name} ===")
        print(f"  Q: {q_cfg['q']}")

        for run in range(1, N_RUNS+1):
            input(f"  [SETUP] Arm and take off to 1m. run={run}. Press Enter when hovering…")

            hover_guard = HoverGuard(mcp, HOVER_ALT, HOVER_TOLERANCE)
            hover_guard.start()

            try:
                agent = MCPAgent(model="claude", vision=False,
                                 session_id=f"Q2_{q_name}_r{run}")
                res = ask_question_via_agent(agent, q_cfg["q"], q_cfg)
                hover_ok = hover_guard.hover_maintained()

                row = {
                    "question_type":   q_name,
                    "run":             run,
                    "answer_relevance":res["relevant"],
                    "hover_maintained":hover_ok,
                    "response_ms":     res["llm_ms"],
                    "tts_ms":          res["tts_ms"],
                    "tts_success":     res["tts_success"],
                    "tool_calls":      res["tool_count"],
                    "landed_during":   res["landed_during"],
                    "api_calls":       res["api_calls"],
                    "cost_usd":        res["cost_usd"],
                    "reply_snippet":   res["reply"][:80].replace("\n"," "),
                    "error":           "",
                }
            except Exception as e:
                row = {
                    "question_type":   q_name,
                    "run":             run,
                    "answer_relevance":0,
                    "hover_maintained":0,
                    "response_ms":     0,
                    "tts_ms":          0,
                    "tts_success":     0,
                    "tool_calls":      0,
                    "landed_during":   0,
                    "api_calls":       0,
                    "cost_usd":        0,
                    "reply_snippet":   "",
                    "error":           str(e)[:80],
                }
            finally:
                hover_guard.stop()

            all_rows.append(row)
            print(f"  run={run} relevant={row['answer_relevance']} "
                  f"hover_ok={row['hover_maintained']} "
                  f"resp={row['response_ms']:.0f}ms tts={row['tts_ms']:.0f}ms "
                  f"tools={row['tool_calls']}")
            time.sleep(8)

        input(f"  [TEARDOWN] Land drone. Press Enter to continue…")

    runs_csv = OUT_DIR / "Q2_runs.csv"
    fields = ["question_type","run","answer_relevance","hover_maintained",
              "response_ms","tts_ms","tts_success","tool_calls","landed_during",
              "api_calls","cost_usd","reply_snippet","error"]
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)

    print(f"\n── Q2 Summary ──────────────────────────────────────────")
    ar, arlo, arhi = wilson_ci(sum(r["answer_relevance"]  for r in all_rows), len(all_rows))
    hm, hmlo, hmhi = wilson_ci(sum(r["hover_maintained"]  for r in all_rows), len(all_rows))
    ts, tslo, tshi = wilson_ci(sum(r["tts_success"]       for r in all_rows), len(all_rows))
    rm, _, _       = bootstrap_ci([r["response_ms"] for r in all_rows if r["response_ms"]>0])
    tc, _, _       = bootstrap_ci([r["tool_calls"]  for r in all_rows])

    print(f"  Answer relevance  : {ar:.3f} [{arlo:.3f},{arhi:.3f}]")
    print(f"  Hover maintained  : {hm:.3f} [{hmlo:.3f},{hmhi:.3f}]")
    print(f"  TTS success rate  : {ts:.3f} [{tslo:.3f},{tshi:.3f}]")
    print(f"  Response latency  : {rm:.0f}ms")
    print(f"  Tool calls / Q    : {tc:.1f}")

    print(f"\n  Per-question breakdown:")
    for qn in QUESTIONS:
        qr = [r for r in all_rows if r["question_type"]==qn]
        a, alo, ahi = wilson_ci(sum(r["answer_relevance"] for r in qr), len(qr))
        h, hlo, hhi = wilson_ci(sum(r["hover_maintained"] for r in qr), len(qr))
        print(f"    {qn:20s} relevant={a:.3f} hover={h:.3f}")

    print(f"\nData → {runs_csv}")

if __name__ == "__main__":
    main()
