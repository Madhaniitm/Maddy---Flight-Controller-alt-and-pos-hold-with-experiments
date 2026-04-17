"""
MCP Client — Multi-LLM Drone Controller
=========================================
Connects to the MCP server, discovers tools automatically, and routes them
to the configured LLM. Supports:
    - Claude   (Anthropic — direct or Azure)
    - GPT-4o   (OpenAI)
    - Gemini   (Google Generative Language)
    - LLaVA    (Ollama — open source vision)
    - Gemma-3  (Ollama — open source text)

Features:
    - Vision: captures real drone frame (ESP32-S3) and passes to LLM
    - TTS: speaks agent decisions via pyttsx3 / gTTS
    - Chat: terminal keyboard loop for human ↔ agent conversation
    - HITL mode: human approves/rejects each tool call before execution
    - Decision log: every tool call + reply saved to results/

Usage:
    # Interactive chat
    python mcp_client.py --model claude

    # Single command (non-interactive)
    python mcp_client.py --model gpt4o --cmd "arm and take off to 1 metre"

    # HITL mode
    python mcp_client.py --model claude --hitl

    # Vision mode (capture frame before each LLM call)
    python mcp_client.py --model gemini --vision

Run the MCP server first:
    python mcp_server.py --drone-ip 192.168.4.1
"""

import argparse
import base64
import json
import os
import sys
import time
import threading
import urllib.request
import urllib.error
import pathlib

# ── API configuration ──────────────────────────────────────────────────────────
# Read from environment variables first, fall back to inline defaults.

ANTHROPIC_ENDPOINT = os.environ.get(
    "ANTHROPIC_ENDPOINT",
    "https://claude-test-madhan-resource.services.ai.azure.com"
    "/anthropic/v1/messages?api-version=2025-01-01-preview",
)
ANTHROPIC_KEY  = os.environ.get(
    "ANTHROPIC_API_KEY",
    "EpilO2YT1tLIiwwKoIqCv9oodffWWedT4R7gJdocTTrSVwCC2GEUJQQJ99CCACfhMk5XJ3w3AAAAACOGGVL2",
)
ANTHROPIC_MODEL   = os.environ.get("ANTHROPIC_MODEL",   "claude-sonnet-4-6")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")

OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_KEY      = os.environ.get("OPENAI_API_KEY", "")

GEMINI_KEY      = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-1.5-flash"

OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")

COST_IN  = 3.0  / 1_000_000
COST_OUT = 15.0 / 1_000_000

MCP_SERVER_URL = "http://localhost:5001/mcp"
LOG_DIR        = pathlib.Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are an autonomous drone controller for the Maddy Flight Controller.
You have access to MCP tools to control a real drone via ESP32-S3.

SAFETY RULES (absolute — never override):
  1. Always call get_telemetry before any movement command.
  2. If battery < 20% → land immediately.
  3. If obstacle detected < 25 cm in camera frame → emergency_stop.
  4. Maximum altitude: 2.5 m.
  5. Never arm indoors without confirming clear overhead space.

WORKFLOW:
  1. get_telemetry() — check state
  2. capture_frame(analyze=true) — assess environment (if vision available)
  3. Decide action
  4. Execute one tool
  5. speak() — narrate decision to operator
  6. Repeat until goal achieved

When answering human questions, use chat_reply() to send the response.
Always speak() your key decisions aloud.
"""

# ── MCP Client (talks to mcp_server.py) ───────────────────────────────────────

class MCPClient:
    def __init__(self, url: str = MCP_SERVER_URL):
        self.url     = url
        self._req_id = 0

    def _call(self, method: str, params: dict = None) -> dict:
        self._req_id += 1
        body = json.dumps({
            "jsonrpc": "2.0",
            "id":      self._req_id,
            "method":  method,
            "params":  params or {},
        }).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp.get("result", {})

    def initialize(self):       return self._call("initialize")
    def list_tools(self):       return self._call("tools/list").get("tools", [])
    def call_tool(self, name, args=None):
        return self._call("tools/call", {"name": name, "arguments": args or {}})
    def ping(self):             return self._call("ping")


# ── Tool format converters ─────────────────────────────────────────────────────

def mcp_to_anthropic(tools: list) -> list:
    return [
        {
            "name":         t["name"],
            "description":  t["description"],
            "input_schema": t["inputSchema"],
        }
        for t in tools
    ]

def mcp_to_openai(tools: list) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["inputSchema"],
            },
        }
        for t in tools
    ]

def mcp_to_gemini(tools: list) -> list:
    decls = []
    for t in tools:
        schema = dict(t["inputSchema"])
        schema.pop("required", None)
        decls.append({
            "name":        t["name"],
            "description": t["description"],
            "parameters":  schema,
        })
    return [{"function_declarations": decls}]


# ── LLM backends ──────────────────────────────────────────────────────────────

def _http_post(url, body, headers) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def call_anthropic(messages, tools, image_b64=None, max_tokens=2048, temperature=0.2):
    """Returns (reply_text, tool_uses, in_tok, out_tok, latency_ms)"""
    # Prepend vision image to last user message if provided
    if image_b64:
        last = messages[-1]
        if last["role"] == "user":
            text_content = last["content"] if isinstance(last["content"], str) \
                           else last["content"]
            messages[-1] = {
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg",
                                "data": image_b64}},
                    {"type": "text", "text": text_content
                     if isinstance(text_content, str) else str(text_content)},
                ],
            }

    t0 = time.time()
    resp = _http_post(
        ANTHROPIC_ENDPOINT,
        {
            "model":       ANTHROPIC_MODEL,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "system":      SYSTEM_PROMPT,
            "messages":    messages,
            "tools":       tools,
        },
        {
            "Content-Type":      "application/json",
            "Authorization":     f"Bearer {ANTHROPIC_KEY}",
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    latency = (time.time() - t0) * 1000
    content = resp.get("content", [])
    usage   = resp.get("usage", {})
    reply   = " ".join(b.get("text","") for b in content if b.get("type")=="text")
    uses    = [b for b in content if b.get("type") == "tool_use"]
    return reply, uses, usage.get("input_tokens",0), usage.get("output_tokens",0), latency


def call_openai(messages, tools, model="gpt-4o", image_b64=None,
                max_tokens=2048, temperature=0.2):
    """Returns (reply_text, tool_calls, in_tok, out_tok, latency_ms)"""
    oai_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        role    = m["role"]
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text","") for b in content if "text" in b)
        oai_msgs.append({"role": role, "content": content})

    if image_b64:
        oai_msgs[-1]["content"] = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}",
                           "detail": "low"}},
            {"type": "text", "text": oai_msgs[-1]["content"]},
        ]

    t0 = time.time()
    resp = _http_post(
        OPENAI_ENDPOINT,
        {"model": model, "max_tokens": max_tokens, "temperature": temperature,
         "messages": oai_msgs, "tools": tools},
        {"Content-Type": "application/json",
         "Authorization": f"Bearer {OPENAI_KEY}"},
    )
    latency = (time.time() - t0) * 1000
    choice  = resp["choices"][0]
    msg     = choice["message"]
    usage   = resp.get("usage", {})
    return (msg.get("content",""), msg.get("tool_calls",[]),
            usage.get("prompt_tokens",0), usage.get("completion_tokens",0), latency)


def call_gemini(messages, tools, image_b64=None, max_tokens=2048, temperature=0.2):
    """Returns (reply_text, fn_calls, in_tok, out_tok, latency_ms)"""
    endpoint = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}")
    contents = []
    for m in messages:
        role    = "user" if m["role"] == "user" else "model"
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text","") for b in content if "text" in b)
        contents.append({"role": role, "parts": [{"text": content}]})

    if image_b64 and contents:
        contents[-1]["parts"].insert(
            0, {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}}
        )

    t0 = time.time()
    resp = _http_post(
        endpoint,
        {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents":           contents,
            "tools":              tools,
            "generationConfig":   {"maxOutputTokens": max_tokens,
                                   "temperature": temperature},
        },
        {"Content-Type": "application/json"},
    )
    latency   = (time.time() - t0) * 1000
    candidate = resp.get("candidates", [{}])[0]
    parts     = candidate.get("content", {}).get("parts", [])
    reply     = " ".join(p.get("text","") for p in parts if "text" in p)
    fn_calls  = [p["functionCall"] for p in parts if "functionCall" in p]
    usage     = resp.get("usageMetadata", {})
    return (reply, fn_calls,
            usage.get("promptTokenCount",0), usage.get("candidatesTokenCount",0), latency)


def call_ollama(messages, model="llava:13b", image_b64=None,
                max_tokens=2048, temperature=0.2):
    """Ollama OpenAI-compatible endpoint — returns same tuple."""
    return call_openai(messages, [], model=model, image_b64=image_b64,
                       max_tokens=max_tokens, temperature=temperature)


# ── TTS ────────────────────────────────────────────────────────────────────────

def speak(text: str):
    def _do():
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty("rate", 155)
            e.say(text)
            e.runAndWait()
            return
        except Exception:
            pass
        try:
            import io, subprocess
            from gtts import gTTS
            buf = io.BytesIO()
            gTTS(text=text, lang="en").write_to_fp(buf)
            buf.seek(0)
            subprocess.Popen(["ffplay","-nodisp","-autoexit","-"],
                             stdin=buf, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()


# ── Decision logger ────────────────────────────────────────────────────────────

class DecisionLogger:
    def __init__(self, session_id: str, model: str):
        self.session_id = session_id
        self.model      = model
        self.entries    = []
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"{session_id}_{model}_{ts}.jsonl"

    def log(self, entry: dict):
        entry["ts"]    = time.strftime("%H:%M:%S")
        entry["model"] = self.model
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def summary(self):
        tools_called = [e.get("tool") for e in self.entries if e.get("tool")]
        total_cost   = sum(
            (e.get("in_tok",0)*COST_IN + e.get("out_tok",0)*COST_OUT)
            for e in self.entries
        )
        return {
            "session":      self.session_id,
            "model":        self.model,
            "turns":        len(self.entries),
            "tools_called": tools_called,
            "cost_usd":     round(total_cost, 6),
            "log_path":     str(self.path),
        }


# ── Core agent loop ────────────────────────────────────────────────────────────

class MCPAgent:
    """
    Multi-turn agent loop that:
      1. Calls the chosen LLM with available MCP tools
      2. Executes tool calls via MCPClient
      3. Loops until LLM stops calling tools
      4. Speaks decisions and logs everything
    """
    def __init__(self, model: str = "claude",
                 hitl: bool = False,
                 vision: bool = False,
                 temperature: float = 0.2,
                 max_tokens: int = 2048,
                 session_id: str = None):
        self.model       = model
        self.hitl        = hitl
        self.vision      = vision
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.mcp         = MCPClient()
        self.logger      = DecisionLogger(session_id or f"session_{int(time.time())}",
                                          model)

        # Discover tools from MCP server
        raw_tools        = self.mcp.list_tools()
        self.tools_raw   = raw_tools
        self.tools_claude  = mcp_to_anthropic(raw_tools)
        self.tools_openai  = mcp_to_openai(raw_tools)
        self.tools_gemini  = mcp_to_gemini(raw_tools)

    def _capture_vision(self) -> str | None:
        """Return base64 JPEG from drone camera, or None."""
        try:
            result = self.mcp.call_tool("capture_frame", {"analyze": False})
            text   = result["content"][0]["text"]
            data   = json.loads(text)
            return data.get("jpeg_b64")
        except Exception:
            return None

    def _hitl_approve(self, tool_name: str, args: dict) -> bool:
        print(f"\n  [HITL] LLM wants to call: {tool_name}({json.dumps(args)[:60]})")
        ans = input("  Approve? [Y/n/reason]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return True
        print(f"  [HITL] REJECTED: {ans}")
        return False

    def run(self, user_prompt: str, max_turns: int = 20) -> dict:
        """
        Run one full agent loop. Returns summary dict with reply, stats, trace.
        """
        messages   = [{"role": "user", "content": user_prompt}]
        tool_trace = []
        final_text = ""
        total_in   = 0
        total_out  = 0
        total_lat  = 0.0

        image_b64 = self._capture_vision() if self.vision else None

        for turn in range(1, max_turns + 1):
            # ── LLM call ──────────────────────────────────────────────────────
            try:
                if self.model == "claude":
                    reply, tool_uses, in_tok, out_tok, lat = call_anthropic(
                        messages, self.tools_claude, image_b64,
                        self.max_tokens, self.temperature,
                    )
                    # tool_uses: list of {type:"tool_use", id, name, input}
                    tool_calls_normalized = [
                        {"name": tu["name"], "args": tu.get("input",{}),
                         "id": tu.get("id", f"tu_{turn}")}
                        for tu in tool_uses
                    ]
                elif self.model == "gpt4o":
                    reply, oai_calls, in_tok, out_tok, lat = call_openai(
                        messages, self.tools_openai, image_b64=image_b64,
                        max_tokens=self.max_tokens, temperature=self.temperature,
                    )
                    tool_calls_normalized = [
                        {"name": tc["function"]["name"],
                         "args": json.loads(tc["function"].get("arguments","{}") or "{}"),
                         "id":   tc.get("id","tc")}
                        for tc in (oai_calls or [])
                    ]
                elif self.model == "gemini":
                    reply, fn_calls, in_tok, out_tok, lat = call_gemini(
                        messages, self.tools_gemini, image_b64=image_b64,
                        max_tokens=self.max_tokens, temperature=self.temperature,
                    )
                    tool_calls_normalized = [
                        {"name": fc["name"], "args": fc.get("args",{}),
                         "id":   f"fc_{turn}_{i}"}
                        for i, fc in enumerate(fn_calls)
                    ]
                elif self.model in ("llava", "gemma"):
                    ollama_model = "llava:13b" if self.model == "llava" else "gemma3:12b"
                    reply, _, in_tok, out_tok, lat = call_ollama(
                        messages, model=ollama_model, image_b64=image_b64,
                        max_tokens=self.max_tokens, temperature=self.temperature,
                    )
                    tool_calls_normalized = []
                else:
                    raise ValueError(f"Unknown model: {self.model}")

            except Exception as e:
                print(f"  [LLM ERROR t{turn}] {e}")
                break

            image_b64    = None  # only on first turn
            total_in    += in_tok
            total_out   += out_tok
            total_lat   += lat
            final_text   = reply or final_text

            self.logger.log({
                "turn":       turn,
                "prompt":     user_prompt if turn == 1 else "[continued]",
                "reply":      reply[:200],
                "in_tok":     in_tok,
                "out_tok":    out_tok,
                "latency_ms": round(lat, 1),
            })

            if not tool_calls_normalized:
                break

            # ── Execute tool calls ─────────────────────────────────────────────
            results = []
            for tc in tool_calls_normalized:
                t_name = tc["name"]
                t_args = tc["args"]

                # HITL gate
                if self.hitl and t_name not in ("get_telemetry","capture_frame",
                                                 "chat_reply","speak"):
                    approved = self._hitl_approve(t_name, t_args)
                    if not approved:
                        results.append((tc["id"], f"Action '{t_name}' rejected by operator."))
                        continue

                t0     = time.perf_counter()
                result = self.mcp.call_tool(t_name, t_args)
                exec_ms = (time.perf_counter() - t0) * 1000
                output  = result["content"][0]["text"]

                tool_trace.append({
                    "turn":    turn,
                    "tool":    t_name,
                    "args":    t_args,
                    "result":  output[:200],
                    "exec_ms": round(exec_ms, 1),
                })
                self.logger.log({
                    "tool":    t_name,
                    "args":    t_args,
                    "result":  output[:200],
                    "exec_ms": round(exec_ms, 1),
                })
                results.append((tc["id"], output))

                # Auto-speak significant actions
                if t_name in ("takeoff","land","emergency_stop","arm","disarm"):
                    speak(f"{t_name}: {output[:60]}")

            # Append tool results to message history (model-specific format)
            if self.model == "claude":
                messages.append({"role": "assistant", "content": [
                    b for b in [{"type":"text","text":reply}] + [
                        {"type":"tool_use","id":tc["id"],"name":tc["name"],
                         "input":tc["args"]}
                        for tc in tool_calls_normalized
                    ] if b
                ]})
                messages.append({"role": "user", "content": [
                    {"type":"tool_result","tool_use_id":tid,"content":out}
                    for tid, out in results
                ]})
            else:
                # OpenAI / Gemini / Ollama: append assistant + tool results as user
                if reply:
                    messages.append({"role": "assistant", "content": reply})
                for _tid, out in results:
                    messages.append({"role": "user",
                                     "content": f"[Tool result] {out}"})

        summary = {
            "reply":        final_text,
            "turns":        turn,
            "tool_trace":   tool_trace,
            "tokens_in":    total_in,
            "tokens_out":   total_out,
            "cost_usd":     round(total_in*COST_IN + total_out*COST_OUT, 6),
            "latency_ms":   round(total_lat, 1),
        }
        return summary


# ── Interactive chat loop ──────────────────────────────────────────────────────

def chat_loop(agent: MCPAgent):
    print(f"\n[MCP CHAT] Model={agent.model}  HITL={agent.hitl}  Vision={agent.vision}")
    print("[MCP CHAT] Type your command. 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit","exit","q"):
            break

        print("  [thinking…]")
        result = agent.run(user_input)
        reply  = result["reply"]

        if reply:
            print(f"\nAgent: {reply}")
            speak(reply)

        summary = agent.logger.summary()
        print(f"  [{summary['turns']} turns | "
              f"{result['tokens_in']}+{result['tokens_out']} tok | "
              f"${result['cost_usd']:.5f}]\n")

    print("\n[MCP CHAT] Session ended.")
    s = agent.logger.summary()
    print(f"  Log: {s['log_path']}")
    print(f"  Total cost: ${s['cost_usd']:.5f}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Maddy Drone MCP Client")
    parser.add_argument("--model", default="claude",
                        choices=["claude","gpt4o","gemini","llava","gemma"])
    parser.add_argument("--hitl",        action="store_true",
                        help="Human-in-the-loop: approve each action")
    parser.add_argument("--vision",      action="store_true",
                        help="Capture drone frame before each LLM call")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens",  type=int,   default=2048)
    parser.add_argument("--cmd",         type=str,   default=None,
                        help="Single command (non-interactive)")
    parser.add_argument("--mcp-url",     default=MCP_SERVER_URL)
    parser.add_argument("--session-id",  default=None)
    args = parser.parse_args()

    # Verify MCP server is up
    try:
        client = MCPClient(args.mcp_url)
        client.initialize()
        print(f"[MCP CLIENT] Connected to {args.mcp_url}")
    except Exception as e:
        print(f"[MCP CLIENT] ERROR: Cannot connect to MCP server: {e}")
        print("  Start the server first: python mcp_server.py")
        sys.exit(1)

    agent = MCPAgent(
        model       = args.model,
        hitl        = args.hitl,
        vision      = args.vision,
        temperature = args.temperature,
        max_tokens  = args.max_tokens,
        session_id  = args.session_id,
    )
    agent.mcp.url = args.mcp_url

    if args.cmd:
        print(f"[MCP CLIENT] Running: {args.cmd}")
        result = agent.run(args.cmd)
        print(f"\nAgent: {result['reply']}")
        if result["reply"]:
            speak(result["reply"])
    else:
        chat_loop(agent)


if __name__ == "__main__":
    main()
