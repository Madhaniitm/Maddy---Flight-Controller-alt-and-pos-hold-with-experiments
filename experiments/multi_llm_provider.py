"""
multi_llm_provider.py
=====================
Provider-agnostic LLM adapter for the C-series drone experiments.

Supported providers
-------------------
  anthropic_azure  — Azure-hosted Anthropic API (existing baseline)
  openai           — OpenAI API  (gpt-4o, gpt-4o-mini, …)
  gemini           — Google Gemini API  (gemini-1.5-pro, gemini-2.0-flash, …)
  ollama           — Local Ollama server  (llama3, mistral, …)

Usage
-----
    from multi_llm_provider import make_provider, MultiLLMSimAgent

    provider = make_provider("openai", "gpt-4o",
                             api_key=os.environ["OPENAI_API_KEY"])
    agent = MultiLLMSimAgent(provider=provider, guardrail_enabled=True)
    text, stats, trace, _ = agent.run_agent_loop("take off to 1 m")

Normalized history item types
------------------------------
  {"role": "user",         "content": str}
  {"role": "assistant",    "text": str,   "tool_calls": [{id, name, args}]}
  {"role": "tool_results", "results":     [{id, name, content}]}
"""

import json, os, time, math, urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import Any

# ── Load credentials (sets env vars if not already set) ───────────────────────
try:
    import credentials  # noqa: F401
except ImportError:
    pass

# ── Import parent agent ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from c_series_agent import (
    SimAgent, SIM_TOOLS, SYSTEM_PROMPT,
    MAX_TURNS, COST_IN, COST_OUT,
    _ENDPOINT as _AZURE_ENDPOINT,
    _API_KEY  as _AZURE_KEY,
    _MODEL    as _AZURE_MODEL,
    _VERSION  as _AZURE_VERSION,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Normalized response
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NormResp:
    text:         str  = ""
    tool_calls:   list = field(default_factory=list)  # [{id,name,args}]
    stop_reason:  str  = "end_turn"                   # "end_turn" | "tool_use"
    input_tokens: int  = 0
    output_tokens:int  = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Base provider
# ═══════════════════════════════════════════════════════════════════════════════

class LLMProvider:
    name:  str = "base"
    model: str = ""

    def __init__(self, model: str, api_key: str = ""):
        self.model   = model
        self.api_key = api_key

    # ── Override in subclasses ──────────────────────────────────────────────

    def _tools_native(self, tools: list) -> Any:
        """Convert Anthropic-format tool defs to provider-native format."""
        raise NotImplementedError

    def _history_native(self, history: list, system: str) -> list:
        """Convert normalized history to provider-native message list."""
        raise NotImplementedError

    def _raw_call(self, native_messages: list, native_tools: Any,
                  system: str, max_tokens: int) -> dict:
        """HTTP call to provider API. Returns raw response dict."""
        raise NotImplementedError

    def _parse(self, raw: dict) -> NormResp:
        """Parse raw response to NormResp."""
        raise NotImplementedError

    # ── Public API (called by MultiLLMSimAgent) ─────────────────────────────

    def call(self, history: list, tools: list,
             system: str, max_tokens: int) -> NormResp:
        import re as _re
        native_msgs  = self._history_native(history, system)
        native_tools = self._tools_native(tools)
        for attempt in range(6):
            try:
                raw = self._raw_call(native_msgs, native_tools, system, max_tokens)
                return self._parse(raw)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8")
                if e.code == 429:
                    m = _re.search(r'try again in (\d+\.?\d*)s', body)
                    wait_s = float(m.group(1)) + 3 if m else 35
                    print(f"[{self.name}] Rate limited, waiting {wait_s:.0f}s…")
                    time.sleep(wait_s)
                    continue
                print(f"[{self.name}] HTTP {e.code}: {body[:400]}")
                raise
            except Exception as exc:
                if attempt < 2:
                    print(f"[{self.name}] attempt {attempt+1} failed ({exc}), retry…")
                    time.sleep(3)
                else:
                    raise

    def cost_per_token(self) -> tuple:
        """Return (input_cost, output_cost) per token in USD."""
        return COST_IN, COST_OUT   # default: Anthropic sonnet pricing

    def _http_post(self, url: str, headers: dict, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        headers.setdefault("User-Agent", "drone-llm-experiment/1.0")
        req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Anthropic / Azure provider  (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class AnthropicAzureProvider(LLMProvider):
    name = "anthropic_azure"

    def __init__(self, model: str = _AZURE_MODEL,
                 api_key: str = _AZURE_KEY,
                 endpoint: str = _AZURE_ENDPOINT,
                 version: str  = _AZURE_VERSION):
        super().__init__(model, api_key)
        self.endpoint = endpoint
        self.version  = version

    def _tools_native(self, tools):
        return tools   # already in Anthropic format

    def _history_native(self, history: list, system: str) -> list:
        msgs = []
        for h in history:
            role = h["role"]
            if role == "user":
                msgs.append({"role": "user", "content": h["content"]})
            elif role == "assistant":
                content = []
                if h.get("text", "").strip():
                    content.append({"type": "text", "text": h["text"]})
                for tc in h.get("tool_calls", []):
                    content.append({
                        "type":  "tool_use",
                        "id":    tc["id"],
                        "name":  tc["name"],
                        "input": tc["args"],
                    })
                if content:
                    msgs.append({"role": "assistant", "content": content})
            elif role == "tool_results":
                results = []
                for r in h["results"]:
                    results.append({
                        "type":        "tool_result",
                        "tool_use_id": r["id"],
                        "content":     r["content"],
                    })
                msgs.append({"role": "user", "content": results})
        return msgs

    def _raw_call(self, native_messages, native_tools, system, max_tokens):
        body = {
            "model":      self.model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "system":     system,
            "messages":   native_messages,
            "tools":      native_tools,
        }
        return self._http_post(
            self.endpoint,
            headers={
                "Content-Type":      "application/json",
                "Authorization":     f"Bearer {self.api_key}",
                "anthropic-version": self.version,
            },
            body=body,
        )

    def _parse(self, raw: dict) -> NormResp:
        content      = raw.get("content", [])
        stop_reason  = raw.get("stop_reason", "end_turn")
        usage        = raw.get("usage", {})
        text         = ""
        tool_calls   = []
        for block in content:
            if block.get("type") == "text":
                text = block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id":   block["id"],
                    "name": block["name"],
                    "args": block.get("input", {}),
                })
        return NormResp(
            text         = text,
            tool_calls   = tool_calls,
            stop_reason  = "tool_use" if tool_calls else "end_turn",
            input_tokens = usage.get("input_tokens",  0),
            output_tokens= usage.get("output_tokens", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenAI provider  (gpt-4o, gpt-4o-mini, …)
# ═══════════════════════════════════════════════════════════════════════════════

class OpenAIProvider(LLMProvider):
    name = "openai"
    _ENDPOINT = "https://api.openai.com/v1/chat/completions"

    # Approximate pricing (USD / token) — update if needed
    _COST = {
        "gpt-4o":               (2.5e-6,  10e-6),
        "gpt-4o-mini":          (0.15e-6,  0.6e-6),
        "gpt-4o-2024-11-20":    (2.5e-6,  10e-6),
        "gpt-4.1":              (2.0e-6,   8e-6),
        "gpt-4.1-mini":         (0.4e-6,   1.6e-6),
        "o4-mini":              (1.1e-6,   4.4e-6),
    }

    def __init__(self, model: str = "gpt-4o",
                 api_key: str = None):
        super().__init__(model, api_key or os.environ.get("OPENAI_API_KEY", ""))

    def cost_per_token(self):
        return self._COST.get(self.model, (2.5e-6, 10e-6))

    def _tools_native(self, tools: list) -> list:
        """Convert Anthropic tool defs → OpenAI function-tool format."""
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return result

    def _history_native(self, history: list, system: str) -> list:
        msgs = [{"role": "system", "content": system}]
        for h in history:
            role = h["role"]
            if role == "user":
                msgs.append({"role": "user", "content": h["content"]})
            elif role == "assistant":
                msg = {"role": "assistant", "content": h.get("text", "") or ""}
                tcs = h.get("tool_calls", [])
                if tcs:
                    msg["tool_calls"] = [
                        {
                            "id":       tc["id"],
                            "type":     "function",
                            "function": {
                                "name":      tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in tcs
                    ]
                msgs.append(msg)
            elif role == "tool_results":
                for r in h["results"]:
                    msgs.append({
                        "role":         "tool",
                        "tool_call_id": r["id"],
                        "content":      r["content"],
                    })
        return msgs

    def _raw_call(self, native_messages, native_tools, system, max_tokens):
        body = {
            "model":      self.model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages":   native_messages,
            "tools":      native_tools,
            "tool_choice": "auto",
        }
        return self._http_post(
            self._ENDPOINT,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            body=body,
        )

    def _parse(self, raw: dict) -> NormResp:
        choice  = raw.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage   = raw.get("usage", {})
        text    = message.get("content", "") or ""
        tcs_raw = message.get("tool_calls") or []
        tool_calls = []
        for tc in tcs_raw:
            fn   = tc.get("function", {})
            args = fn.get("arguments", "{}")
            try:
                args = json.loads(args) or {}
            except Exception:
                args = {}
            tool_calls.append({"id": tc["id"], "name": fn["name"], "args": args})
        return NormResp(
            text          = text,
            tool_calls    = tool_calls,
            stop_reason   = "tool_use" if tool_calls else "end_turn",
            input_tokens  = usage.get("prompt_tokens",     0),
            output_tokens = usage.get("completion_tokens", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Google Gemini provider
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    name = "gemini"
    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    _COST = {
        "gemini-2.0-flash":         (0.1e-6, 0.4e-6),
        "gemini-2.0-flash-lite":    (0.075e-6, 0.3e-6),
        "gemini-1.5-pro":           (1.25e-6, 5.0e-6),
        "gemini-1.5-flash":         (0.075e-6, 0.3e-6),
        "gemini-2.5-flash-preview-04-17": (0.15e-6, 0.6e-6),
        "gemini-2.5-flash":         (0.15e-6, 0.6e-6),
        "gemini-2.5-pro":           (1.25e-6, 10.0e-6),
    }

    def __init__(self, model: str = "gemini-2.5-flash",
                 api_key: str = None):
        super().__init__(model, api_key or os.environ.get("GEMINI_API_KEY", ""))

    @property
    def _endpoint(self):
        return f"{self._BASE}/{self.model}:generateContent?key={self.api_key}"

    def cost_per_token(self):
        return self._COST.get(self.model, (0.1e-6, 0.4e-6))

    def _tools_native(self, tools: list) -> list:
        """Convert Anthropic tool defs → Gemini function_declarations format."""
        decls = []
        for t in tools:
            schema = t.get("input_schema", {"type": "object", "properties": {}})
            # Gemini doesn't support "required" at top level for empty-property tools
            decls.append({
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  schema,
            })
        return [{"function_declarations": decls}]

    def _raw_call(self, native_messages, native_tools, system, max_tokens):
        body = {
            "contents":           native_messages,
            "tools":              native_tools,
            "systemInstruction":  {"parts": [{"text": system}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature":     0.2,
            },
        }
        return self._http_post(
            self._endpoint,
            headers={"Content-Type": "application/json"},
            body=body,
        )

    def _history_native(self, history: list, system: str):
        """Override: return native contents list (system instruction handled separately in body)."""
        contents = []
        for h in history:
            role = h["role"]
            if role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": h["content"]}],
                })
            elif role == "assistant":
                parts = []
                if h.get("text", "").strip():
                    parts.append({"text": h["text"]})
                for tc in h.get("tool_calls", []):
                    parts.append({
                        "function_call": {
                            "name": tc["name"],
                            "args": tc["args"],
                        }
                    })
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool_results":
                parts = []
                for r in h["results"]:
                    parts.append({
                        "function_response": {
                            "name":     r["name"],
                            "response": {"content": r["content"]},
                        }
                    })
                contents.append({"role": "user", "parts": parts})
        return contents

    def _parse(self, raw: dict) -> NormResp:
        try:
            candidate = raw.get("candidates", [{}])[0]
            parts     = candidate.get("content", {}).get("parts", [])
            usage     = raw.get("usageMetadata", {})
        except (IndexError, KeyError):
            return NormResp()

        text = ""
        tool_calls = []
        tc_counter = 0
        for part in parts:
            if "text" in part:
                text += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tc_counter += 1
                tool_calls.append({
                    "id":   f"gemini_tc_{tc_counter}",
                    "name": fc["name"],
                    "args": fc.get("args", {}),
                })

        return NormResp(
            text          = text,
            tool_calls    = tool_calls,
            stop_reason   = "tool_use" if tool_calls else "end_turn",
            input_tokens  = usage.get("promptTokenCount",     0),
            output_tokens = usage.get("candidatesTokenCount", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Azure OpenAI provider  (gpt-4o, gpt-4o-mini via Azure OpenAI Service)
# ═══════════════════════════════════════════════════════════════════════════════

class AzureOpenAIProvider(OpenAIProvider):
    """
    Azure OpenAI Service — same wire format as OpenAI but uses api-key header.

    Each model can use a different Azure resource (different endpoint + key).
    Per-model env vars take priority over generic fallbacks:

      gpt-4o      → AZURE_GPT4O_ENDPOINT      + AZURE_GPT4O_KEY
      gpt-4o-mini → AZURE_GPT4OMINI_ENDPOINT   + AZURE_GPT4OMINI_KEY
      (others)    → AZURE_OPENAI_ENDPOINT       + AZURE_OPENAI_API_KEY

    Endpoint should be the full chat/completions URL, e.g.:
      https://<resource>.services.ai.azure.com/openai/v1/chat/completions
    OR the traditional deployment URL:
      https://<resource>.openai.azure.com/openai/deployments/<model>/chat/completions?api-version=2024-02-01
    """
    name = "azure_openai"
    _API_VERSION = "2024-02-01"

    _COST = {
        "gpt-4o":      (2.5e-6,  10e-6),
        "gpt-4o-mini": (0.15e-6,  0.6e-6),
    }

    # Per-model env var names: (endpoint_var, key_var)
    _MODEL_ENVS = {
        "gpt-4o":                  ("AZURE_GPT4O_ENDPOINT",     "AZURE_GPT4O_KEY"),
        "gpt-4o-mini":             ("AZURE_GPT4OMINI_ENDPOINT",  "AZURE_GPT4OMINI_KEY"),
        "grok-4-1-fast-reasoning": ("AZURE_GROK_ENDPOINT",       "AZURE_GROK_KEY"),
        "grok-4-20-reasoning":     ("AZURE_GROK_ENDPOINT",       "AZURE_GROK_KEY"),
    }

    def __init__(self, model: str = "gpt-4o",
                 api_key: str = None,
                 endpoint: str = None):
        ep_env, key_env = self._MODEL_ENVS.get(
            model, ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"))
        resolved_key = (api_key
                        or os.environ.get(key_env)
                        or os.environ.get("AZURE_OPENAI_API_KEY", ""))
        super().__init__(model, resolved_key)
        base = os.environ.get("AZURE_OPENAI_BASE", "").rstrip("/")
        self._ENDPOINT = (
            endpoint
            or os.environ.get(ep_env)
            or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            or f"{base}/openai/deployments/{model}/chat/completions"
               f"?api-version={self._API_VERSION}"
        )

    def _raw_call(self, native_messages, native_tools, system, max_tokens):
        body = {
            "model":       self.model,
            "max_tokens":  max_tokens,
            "temperature": 0.2,
            "messages":    native_messages,
            "tools":       native_tools,
            "tool_choice": "auto",
        }
        return self._http_post(
            self._ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "api-key":      self.api_key,
            },
            body=body,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Azure Gemini provider  (Gemini via Azure AI Foundry / AI Services)
# ═══════════════════════════════════════════════════════════════════════════════

class AzureGeminiProvider(AzureOpenAIProvider):
    """
    Google Gemini hosted on Azure AI Foundry — uses the Azure AI Inference
    OpenAI-compatible endpoint.

    Env vars:
      AZURE_GEMINI_ENDPOINT — full URL
                              e.g. https://<project>.services.ai.azure.com/
                                   models/chat/completions?api-version=2024-05-01-preview
      AZURE_GEMINI_API_KEY
    """
    name = "azure_gemini"

    _COST = {
        "gemini-2.0-flash": (0.1e-6, 0.4e-6),
        "gemini-1.5-pro":   (1.25e-6, 5.0e-6),
    }

    def __init__(self, model: str = "gemini-2.0-flash",
                 api_key: str = None,
                 endpoint: str = None):
        # Call LLMProvider.__init__ directly to avoid OpenAI key lookup
        LLMProvider.__init__(self, model,
                             api_key or os.environ.get("AZURE_GEMINI_API_KEY", ""))
        self._ENDPOINT = (
            endpoint
            or os.environ.get("AZURE_GEMINI_ENDPOINT", "")
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Groq provider  (OpenAI-compatible — LLaMA, Mixtral, Gemma via Groq Cloud)
# ═══════════════════════════════════════════════════════════════════════════════

class GroqProvider(OpenAIProvider):
    """
    Groq Cloud — OpenAI-compatible API with fast LLaMA/Mixtral inference.

    Env var: GROQ_API_KEY
    Common models:
      llama-3.1-70b-versatile   (recommended — best capability)
      llama-3.3-70b-versatile
      llama-3.1-8b-instant      (fastest / cheapest)
      mixtral-8x7b-32768
    """
    name = "groq"
    _ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    _COST = {
        "llama-3.1-70b-versatile": (0.59e-6, 0.79e-6),
        "llama-3.3-70b-versatile": (0.59e-6, 0.79e-6),
        "llama-3.1-8b-instant":    (0.05e-6, 0.08e-6),
        "llama3-70b-8192":         (0.59e-6, 0.79e-6),
        "mixtral-8x7b-32768":      (0.24e-6, 0.24e-6),
    }

    def __init__(self, model: str = "llama-3.3-70b-versatile",
                 api_key: str = None):
        super().__init__(model, api_key or os.environ.get("GROQ_API_KEY", ""))

    def cost_per_token(self):
        return self._COST.get(self.model, (0.59e-6, 0.79e-6))


# ═══════════════════════════════════════════════════════════════════════════════
#  Ollama provider  (local server, OpenAI-compatible endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaProvider(OpenAIProvider):
    """
    Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint.
    No API key required. Default host: http://localhost:11434
    """
    name = "ollama"
    _COST = {}   # local inference — no monetary cost

    def __init__(self, model: str = "llama3.1",
                 host: str = "http://localhost:11434"):
        super().__init__(model, api_key="ollama")
        self._ENDPOINT = f"{host}/v1/chat/completions"

    def cost_per_token(self):
        return (0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_provider(provider_name: str, model: str = None, **kwargs) -> LLMProvider:
    """
    Create a provider by name.

    provider_name : "anthropic_azure" | "openai" | "gemini" | "ollama"
                  | "azure_openai"   | "azure_gemini" | "groq"
    model         : model string (defaults per provider if None)
    kwargs        : api_key, endpoint, host, …

    Required env vars per provider:
      azure_openai  — AZURE_OPENAI_API_KEY + (AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_BASE)
      azure_gemini  — AZURE_GEMINI_API_KEY + AZURE_GEMINI_ENDPOINT
      groq          — GROQ_API_KEY
      openai        — OPENAI_API_KEY
      gemini        — GEMINI_API_KEY
    """
    p = provider_name.lower().replace("-", "_")
    if p in ("anthropic_azure", "anthropic", "azure"):
        return AnthropicAzureProvider(
            model   = model or _AZURE_MODEL,
            api_key = kwargs.get("api_key", _AZURE_KEY),
            endpoint= kwargs.get("endpoint", _AZURE_ENDPOINT),
        )
    if p == "openai":
        return OpenAIProvider(
            model   = model or "gpt-4o",
            api_key = kwargs.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
        )
    if p == "gemini":
        return GeminiProvider(
            model   = model or "gemini-2.0-flash",
            api_key = kwargs.get("api_key", os.environ.get("GEMINI_API_KEY", "")),
        )
    if p == "azure_openai":
        return AzureOpenAIProvider(
            model    = model or "gpt-4o",
            api_key  = kwargs.get("api_key"),
            endpoint = kwargs.get("endpoint"),
        )
    if p == "azure_gemini":
        return AzureGeminiProvider(
            model    = model or "gemini-2.0-flash",
            api_key  = kwargs.get("api_key"),
            endpoint = kwargs.get("endpoint"),
        )
    if p == "groq":
        return GroqProvider(
            model   = model or "llama-3.1-70b-versatile",
            api_key = kwargs.get("api_key"),
        )
    if p == "ollama":
        return OllamaProvider(
            model = model or "llama3.1",
            host  = kwargs.get("host", "http://localhost:11434"),
        )
    raise ValueError(f"Unknown provider: {provider_name!r}. "
                     "Choose from: anthropic_azure, openai, gemini, ollama, "
                     "azure_openai, azure_gemini, groq")


# ═══════════════════════════════════════════════════════════════════════════════
#  MultiLLMSimAgent  — drop-in SimAgent replacement with pluggable provider
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLLMSimAgent(SimAgent):
    """
    Same as SimAgent but uses a pluggable LLMProvider instead of the
    hard-coded Anthropic/Azure endpoint.

    Returns the same (final_text, api_stats, tool_trace, messages) tuple
    so all existing C-series experiment scripts work unchanged.

    Usage:
        provider = make_provider("openai", "gpt-4o")
        agent = MultiLLMSimAgent(provider=provider)
        text, stats, trace, _ = agent.run_agent_loop("take off to 1 m")
    """

    def __init__(self, provider: LLMProvider, **kwargs):
        """
        provider : LLMProvider instance (from make_provider())
        kwargs   : passed to SimAgent.__init__ (session_id, guardrail_enabled)
        """
        super().__init__(**kwargs)
        self._llm = provider

    # ─────────────────────────────────────────────────────────────────────────
    #  Override the agent loop with provider-agnostic history management
    # ─────────────────────────────────────────────────────────────────────────

    def run_agent_loop(self, user_prompt: str,
                       history: list = None,
                       max_turns: int = MAX_TURNS,
                       max_tokens: int = 2048) -> tuple:
        """
        Provider-agnostic agent loop.

        history: list of normalized items (same format as this method builds).
                 Pass None to start fresh. If called with existing Anthropic-format
                 history from another SimAgent, convert first.

        Returns (final_text, api_stats, tool_trace, norm_history)
          norm_history can be passed back in future calls to continue a session.
        """
        norm_history = list(history or [])
        norm_history.append({"role": "user", "content": user_prompt})

        api_stats  = []
        tool_trace = []
        final_text = ""
        cost_in, cost_out = self._llm.cost_per_token()

        for turn in range(1, max_turns + 1):
            t0 = time.time()
            try:
                resp = self._llm.call(
                    history   = norm_history,
                    tools     = SIM_TOOLS,
                    system    = SYSTEM_PROMPT,
                    max_tokens= max_tokens,
                )
            except Exception as exc:
                print(f"[API ERROR turn {turn}] {exc}")
                break

            latency = time.time() - t0
            cost    = resp.input_tokens * cost_in + resp.output_tokens * cost_out
            api_stats.append({
                "turn":          turn,
                "latency_s":     round(latency, 3),
                "input_tokens":  resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd":      round(cost, 6),
                "provider":      self._llm.name,
                "model":         self._llm.model,
            })

            if resp.text:
                final_text = resp.text

            # Append assistant turn to normalized history
            norm_history.append({
                "role":       "assistant",
                "text":       resp.text,
                "tool_calls": resp.tool_calls,
            })

            if not resp.tool_calls or resp.stop_reason == "end_turn":
                break

            # Execute tools
            tool_results = []
            for tc in resp.tool_calls:
                t_name = tc["name"]
                t_args = tc["args"]
                t_id   = tc["id"]
                print(f"  [TOOL t{turn}] {t_name}({json.dumps(t_args)[:80]})")

                allowed, t_args, feedback = self.guardrail.check(
                    t_name, t_args, self.state)
                if not allowed:
                    result = feedback
                    print(f"  [GUARDRAIL BLOCK] {feedback}")
                else:
                    result = self.execute_tool(t_name, t_args)
                    if feedback:
                        print(f"  [GUARDRAIL CLIP] {feedback}")
                        result = feedback + " " + result

                tool_trace.append({
                    "turn":              turn,
                    "name":              t_name,
                    "args":              t_args,
                    "result":            result[:600],
                    "sim_time_s":        round(self.sim_time, 2),
                    "guardrail_fired":   feedback is not None,
                    "guardrail_allowed": allowed,
                })
                tool_results.append({
                    "id":      t_id,
                    "name":    t_name,
                    "content": result,
                })

            norm_history.append({"role": "tool_results", "results": tool_results})

        return final_text, api_stats, tool_trace, norm_history


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI smoke-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic_azure",
                    choices=["anthropic_azure", "openai", "gemini", "ollama",
                             "azure_openai", "azure_gemini", "groq"])
    ap.add_argument("--model",    default=None)
    ap.add_argument("--prompt",   default="take off to 1 m, hover 3 s, land")
    args = ap.parse_args()

    prov  = make_provider(args.provider, args.model)
    agent = MultiLLMSimAgent(provider=prov, guardrail_enabled=True)
    print(f"Provider: {prov.name} / {prov.model}")
    text, stats, trace, _ = agent.run_agent_loop(args.prompt)
    print(f"\nFinal text: {text}")
    total_cost = sum(s["cost_usd"] for s in stats)
    total_tok  = sum(s["input_tokens"] + s["output_tokens"] for s in stats)
    print(f"Tokens: {total_tok}, Cost: ${total_cost:.4f}, "
          f"Turns: {len(stats)}, Tool calls: {len(trace)}")
