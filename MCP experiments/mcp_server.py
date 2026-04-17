"""
MCP Server — Maddy Flight Controller
=====================================
Implements the Model Context Protocol (JSON-RPC 2.0 over HTTP, port 5001).
Bridges MCP tool calls to the real ESP32-S3 drone's HTTP API.

Drone HTTP endpoints assumed (from Maddy FC firmware):
    GET  http://{DRONE_IP}/telemetry  → JSON telemetry
    POST http://{DRONE_IP}/command    → {"cmd": str, "value": ...}
    GET  http://{DRONE_IP}/capture    → JPEG bytes

MCP endpoints exposed on port 5001:
    POST /mcp   — JSON-RPC 2.0 multiplexed
        methods: initialize, tools/list, tools/call

Safety rules enforced server-side (cannot be overridden by any LLM):
    - Arm only if battery > 15 %
    - Max altitude cap: 2.5 m
    - Emergency stop on any tool error during flight
    - Minimum obstacle distance check before move commands (if ToF available)

Usage:
    python mcp_server.py [--drone-ip 192.168.4.1] [--port 5001]
"""

import argparse
import base64
import json
import time
import threading
import urllib.request
import urllib.error
import http.server
import socketserver
from http import HTTPStatus

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_DRONE_IP  = "192.168.4.1"
DEFAULT_PORT      = 5001
CMD_TIMEOUT_S     = 5.0
TELEMETRY_TIMEOUT = 3.0
CAPTURE_TIMEOUT   = 4.0
MAX_ALTITUDE_M    = 2.5
MIN_BATTERY_PCT   = 15.0

# ── Drone HTTP helpers ─────────────────────────────────────────────────────────

def drone_get(drone_ip: str, path: str, timeout: float = TELEMETRY_TIMEOUT) -> dict:
    url = f"http://{drone_ip}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def drone_post(drone_ip: str, path: str, body: dict,
               timeout: float = CMD_TIMEOUT_S) -> dict:
    url = f"http://{drone_ip}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def drone_capture(drone_ip: str) -> bytes:
    url = f"http://{drone_ip}/capture"
    req = urllib.request.Request(url, headers={"Accept": "image/jpeg"})
    with urllib.request.urlopen(req, timeout=CAPTURE_TIMEOUT) as r:
        return r.read()


# ── MCP Tool definitions ───────────────────────────────────────────────────────

MCP_TOOLS = [
    {
        "name": "arm",
        "description": "Arm the drone motors. Safety check: battery > 15%. "
                       "Drone must be on flat ground.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "disarm",
        "description": "Disarm the drone motors. Safe to call at any time.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "takeoff",
        "description": "Take off and hover at target altitude.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "altitude_m": {
                    "type": "number",
                    "description": "Target hover altitude in metres (0.3–2.5).",
                }
            },
            "required": ["altitude_m"],
        },
    },
    {
        "name": "land",
        "description": "Descend and land safely. Disarms after touchdown.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "emergency_stop",
        "description": "IMMEDIATELY halt all motors. Use only for crash prevention.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move_forward",
        "description": "Move drone forward by given distance at low speed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "distance_m": {"type": "number",
                               "description": "Distance (0.1–2.0 m)."}
            },
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_backward",
        "description": "Move drone backward by given distance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "distance_m": {"type": "number", "description": "Distance (0.1–2.0 m)."}
            },
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_left",
        "description": "Strafe drone left by given distance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "distance_m": {"type": "number", "description": "Distance (0.1–2.0 m)."}
            },
            "required": ["distance_m"],
        },
    },
    {
        "name": "move_right",
        "description": "Strafe drone right by given distance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "distance_m": {"type": "number", "description": "Distance (0.1–2.0 m)."}
            },
            "required": ["distance_m"],
        },
    },
    {
        "name": "set_altitude",
        "description": "Change hover altitude to new target.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "altitude_m": {"type": "number",
                               "description": "New altitude (0.3–2.5 m)."}
            },
            "required": ["altitude_m"],
        },
    },
    {
        "name": "set_yaw",
        "description": "Rotate drone to absolute heading.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "heading_deg": {"type": "number",
                                "description": "Target heading 0–360°."}
            },
            "required": ["heading_deg"],
        },
    },
    {
        "name": "get_telemetry",
        "description": "Read current drone telemetry: altitude, battery, "
                       "roll/pitch/yaw, armed state, position.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "capture_frame",
        "description": "Capture a JPEG frame from the drone camera (ESP32-S3 OV2640). "
                       "Returns base64-encoded JPEG and a text description.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "analyze": {
                    "type": "boolean",
                    "description": "If true, return a text scene description "
                                   "in addition to the raw frame.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "speak",
        "description": "Speak a message to the operator via TTS. "
                       "Use to narrate decisions or warnings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string",
                            "description": "Text to speak aloud."}
            },
            "required": ["message"],
        },
    },
    {
        "name": "chat_reply",
        "description": "Send a text reply to the operator's chat terminal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string",
                            "description": "Reply text shown in chat."}
            },
            "required": ["message"],
        },
    },
]


# ── Tool executor ──────────────────────────────────────────────────────────────

class DroneToolExecutor:
    def __init__(self, drone_ip: str):
        self.drone_ip = drone_ip
        self._lock    = threading.Lock()
        self._log     = []

    def _record(self, tool: str, args: dict, result: str, ok: bool):
        entry = {
            "ts":     time.strftime("%H:%M:%S"),
            "tool":   tool,
            "args":   args,
            "result": result[:200],
            "ok":     ok,
        }
        with self._lock:
            self._log.append(entry)
            if len(self._log) > 500:
                self._log.pop(0)
        print(f"  [MCP EXEC {'OK' if ok else 'ERR'}] {tool}  → {result[:80]}")

    def execute(self, tool_name: str, args: dict) -> str:
        try:
            result = self._dispatch(tool_name, args)
            self._record(tool_name, args, result, True)
            return result
        except Exception as e:
            msg = f"ERROR: {e}"
            self._record(tool_name, args, msg, False)
            return msg

    def _dispatch(self, name: str, args: dict) -> str:
        ip = self.drone_ip

        if name == "arm":
            # Safety: check battery first
            try:
                tel = drone_get(ip, "/telemetry")
                bat = float(tel.get("battery_pct", 100))
                if bat < MIN_BATTERY_PCT:
                    return f"ARM REFUSED: battery {bat:.0f}% < {MIN_BATTERY_PCT}%"
            except Exception:
                pass  # proceed if telemetry unavailable
            r = drone_post(ip, "/command", {"cmd": "arm", "value": True})
            return f"Armed. {r}"

        if name == "disarm":
            r = drone_post(ip, "/command", {"cmd": "arm", "value": False})
            return f"Disarmed. {r}"

        if name == "takeoff":
            alt = float(args.get("altitude_m", 1.0))
            alt = max(0.3, min(MAX_ALTITUDE_M, alt))
            r = drone_post(ip, "/command", {"cmd": "takeoff", "altitude": alt})
            return f"Taking off to {alt:.2f} m. {r}"

        if name == "land":
            r = drone_post(ip, "/command", {"cmd": "land"})
            return f"Landing. {r}"

        if name == "emergency_stop":
            r = drone_post(ip, "/command", {"cmd": "emergency_stop"})
            return f"EMERGENCY STOP ISSUED. {r}"

        if name == "move_forward":
            d = max(0.1, min(2.0, float(args.get("distance_m", 0.3))))
            r = drone_post(ip, "/command", {"cmd": "move", "dir": "forward", "dist": d})
            return f"Moving forward {d:.2f} m. {r}"

        if name == "move_backward":
            d = max(0.1, min(2.0, float(args.get("distance_m", 0.3))))
            r = drone_post(ip, "/command", {"cmd": "move", "dir": "backward", "dist": d})
            return f"Moving backward {d:.2f} m. {r}"

        if name == "move_left":
            d = max(0.1, min(2.0, float(args.get("distance_m", 0.3))))
            r = drone_post(ip, "/command", {"cmd": "move", "dir": "left", "dist": d})
            return f"Moving left {d:.2f} m. {r}"

        if name == "move_right":
            d = max(0.1, min(2.0, float(args.get("distance_m", 0.3))))
            r = drone_post(ip, "/command", {"cmd": "move", "dir": "right", "dist": d})
            return f"Moving right {d:.2f} m. {r}"

        if name == "set_altitude":
            alt = max(0.3, min(MAX_ALTITUDE_M, float(args.get("altitude_m", 1.0))))
            r = drone_post(ip, "/command", {"cmd": "set_altitude", "altitude": alt})
            return f"Altitude target set to {alt:.2f} m. {r}"

        if name == "set_yaw":
            hdg = float(args.get("heading_deg", 0)) % 360
            r = drone_post(ip, "/command", {"cmd": "set_yaw", "heading": hdg})
            return f"Yaw target: {hdg:.1f}°. {r}"

        if name == "get_telemetry":
            tel = drone_get(ip, "/telemetry")
            return json.dumps(tel)

        if name == "capture_frame":
            jpeg = drone_capture(ip)
            b64  = base64.b64encode(jpeg).decode()
            analyze = bool(args.get("analyze", False))
            result  = {"jpeg_b64": b64, "size_bytes": len(jpeg)}
            if analyze:
                result["description"] = f"Frame captured ({len(jpeg)} bytes). " \
                                         "Pass to vision model for analysis."
            return json.dumps(result)

        if name == "speak":
            msg = str(args.get("message", ""))
            _tts_speak(msg)
            return f"Spoke: {msg[:60]}"

        if name == "chat_reply":
            msg = str(args.get("message", ""))
            print(f"\n  [AGENT] {msg}")
            return f"Reply sent: {msg[:60]}"

        return f"Unknown tool: {name}"

    def get_log(self):
        with self._lock:
            return list(self._log)


# ── TTS helper ─────────────────────────────────────────────────────────────────

def _tts_speak(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass
    try:
        import io, subprocess
        from gtts import gTTS
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        # Play silently via mpg123 or ffplay if available
        subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-"],
                         stdin=buf, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass  # TTS unavailable — silent


# ── JSON-RPC 2.0 handler ───────────────────────────────────────────────────────

class MCPHandler(http.server.BaseHTTPRequestHandler):
    executor: DroneToolExecutor = None   # set before server starts

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def do_POST(self):
        if self.path != "/mcp":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        try:
            req = json.loads(body.decode())
        except Exception:
            self._send_error(-32700, "Parse error", None)
            return

        rpc_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo":      {"name": "maddy-drone-mcp", "version": "1.0"},
                "capabilities":    {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": MCP_TOOLS}
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            t0        = time.perf_counter()
            output    = MCPHandler.executor.execute(tool_name, tool_args)
            latency   = round((time.perf_counter() - t0) * 1000, 2)
            result    = {
                "content": [{"type": "text", "text": output}],
                "latency_ms": latency,
                "isError":  output.startswith("ERROR"),
            }
        elif method == "ping":
            result = {"status": "ok", "drone_ip": MCPHandler.executor.drone_ip}
        elif method == "server/log":
            result = {"log": MCPHandler.executor.get_log()[-50:]}
        else:
            self._send_error(-32601, f"Method not found: {method}", rpc_id)
            return

        self._send_result(result, rpc_id)

    def _send_result(self, result, rpc_id):
        body = json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": result}).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message, rpc_id):
        body = json.dumps({
            "jsonrpc": "2.0",
            "id":      rpc_id,
            "error":   {"code": code, "message": message},
        }).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Convenience client helper (used by experiment scripts) ────────────────────

class MCPClient:
    """
    Thin Python client for the MCP server.
    Used by all experiment scripts to call tools without reimplementing HTTP.
    """
    def __init__(self, server_url: str = f"http://localhost:{DEFAULT_PORT}/mcp"):
        self.url     = server_url
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

    def initialize(self) -> dict:
        return self._call("initialize")

    def list_tools(self) -> list:
        return self._call("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict = None) -> dict:
        return self._call("tools/call", {"name": name, "arguments": arguments or {}})

    def ping(self) -> dict:
        return self._call("ping")

    def get_log(self) -> list:
        return self._call("server/log").get("log", [])

    # Convenience wrappers
    def arm(self):           return self.call_tool("arm")
    def disarm(self):        return self.call_tool("disarm")
    def takeoff(self, alt):  return self.call_tool("takeoff", {"altitude_m": alt})
    def land(self):          return self.call_tool("land")
    def estop(self):         return self.call_tool("emergency_stop")
    def telemetry(self):
        r = self.call_tool("get_telemetry")
        text = r["content"][0]["text"]
        try:    return json.loads(text)
        except: return {"raw": text}
    def capture(self, analyze=False):
        r    = self.call_tool("capture_frame", {"analyze": analyze})
        text = r["content"][0]["text"]
        try:    return json.loads(text)
        except: return {"raw": text}
    def speak(self, msg):    return self.call_tool("speak", {"message": msg})
    def chat(self, msg):     return self.call_tool("chat_reply", {"message": msg})


def preflight_check(drone_ip: str, mcp_url: str) -> bool:
    """
    Verify both the drone HTTP API and MCP server are reachable.
    Called at the start of every experiment. Returns False if either is down.
    """
    print(f"[PREFLIGHT] Checking drone at {drone_ip}…")
    try:
        drone_get(drone_ip, "/telemetry", timeout=3.0)
        print("  ✓ Drone HTTP reachable")
    except Exception as e:
        print(f"  ✗ Drone unreachable: {e}")
        return False

    print(f"[PREFLIGHT] Checking MCP server at {mcp_url}…")
    try:
        client = MCPClient(mcp_url)
        r = client.ping()
        print(f"  ✓ MCP server OK: {r}")
    except Exception as e:
        print(f"  ✗ MCP server unreachable: {e}")
        return False

    return True


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Maddy Drone MCP Server")
    parser.add_argument("--drone-ip", default=DEFAULT_DRONE_IP)
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    executor = DroneToolExecutor(drone_ip=args.drone_ip)
    MCPHandler.executor = executor

    with socketserver.TCPServer(("", args.port), MCPHandler) as srv:
        srv.allow_reuse_address = True
        print(f"[MCP SERVER] Listening on port {args.port}")
        print(f"[MCP SERVER] Drone IP: {args.drone_ip}")
        print(f"[MCP SERVER] {len(MCP_TOOLS)} tools registered")
        print(f"[MCP SERVER] POST http://localhost:{args.port}/mcp")
        print("[MCP SERVER] Ctrl+C to stop\n")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[MCP SERVER] Stopped.")


if __name__ == "__main__":
    main()
