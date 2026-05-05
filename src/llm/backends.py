"""LLM backends — local (Ollama) and cloud (Claude, Gemini).

All backends implement the same streaming interface.
"""

import json
import logging
import time
import urllib.request
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    thought_signature: bytes | None = None  # Gemini thinking models only


@dataclass
class LLMChunk:
    """One piece of a streaming response."""

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    model: str = ""


class LLMBackend(Protocol):
    """Interface that all LLM backends must satisfy."""

    @property
    def name(self) -> str: ...

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Generator[LLMChunk, None, None]:
        """Yield chunks of the response as they arrive."""
        ...


# ---------------------------------------------------------------------------
# Connectivity helper
# ---------------------------------------------------------------------------
def check_ollama_connectivity(base_url: str, timeout: float = 3.0) -> bool:
    """Returns True if Ollama HTTP API is reachable at base_url."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def get_running_remote_model(base_url: str, timeout: float = 3.0) -> str | None:
    """Return the name of the largest model currently loaded in RAM on a remote Ollama, or None."""
    import re

    def _size_b(name: str) -> float:
        m = re.search(r":?(\d+(?:\.\d+)?)b", name.lower())
        return float(m.group(1)) if m else 0.0

    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=timeout) as r:
            models = json.loads(r.read()).get("models", [])
        names = [m["model"] for m in models if m.get("model")]
        if names:
            return max(names, key=_size_b)
    except Exception:
        pass
    return None


def check_llama_cpp_connectivity(base_url: str, timeout: float = 3.0) -> bool:
    """Returns True if llama.cpp server /health endpoint is reachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def check_claude_availability(api_key: str, timeout: float = 3.0) -> tuple[bool, str]:
    """Returns (available, reason). GET /v1/models — zero token cost."""
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        urllib.request.urlopen(req, timeout=timeout)
        return True, "ok"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def check_gemini_availability(api_key: str, timeout: float = 3.0) -> tuple[bool, str]:
    """Returns (available, reason). GET /v1beta/models — zero token cost."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        urllib.request.urlopen(url, timeout=timeout)
        return True, "ok"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------
class OllamaBackend:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_ctx: int = 4096,
        num_predict: int | None = 384,
        num_thread: int | None = None,
        system_prompt: str = "",
        think: bool = False,
        label: str = "local",
    ):
        import httpx
        import ollama

        self._client = ollama.Client(
            host=base_url,
            timeout=httpx.Timeout(connect=2.0, read=300.0, write=30.0, pool=10.0),
        )
        self._base_url = base_url.rstrip("/")  # stored for is_loaded() / diagnostics
        self._model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._num_thread = num_thread
        self._system_prompt = system_prompt
        self._think = think
        self._label = label

    @property
    def name(self) -> str:
        return f"{self._label}/{self._model}"

    def warm(self):
        """Preload model into memory and prime the KV cache.

        Uses the real num_ctx AND the same think/tools flags as stream() so
        Ollama doesn't reload the model on the first real request due to a
        parameter mismatch.  Sets keep_alive=-1 so the model stays loaded
        indefinitely (no 5-min timeout).
        """
        try:
            log.info(f"Warming up {self._model}...")
            start = time.perf_counter()
            options: dict[str, Any] = {
                "num_predict": 1,
                "num_ctx": self._num_ctx,
            }
            if self._num_thread is not None:
                options["num_thread"] = self._num_thread
            # Dummy tool — ensures Ollama allocates the tool-calling model variant
            # in the same configuration used by stream(), preventing a silent reload.
            _dummy_tool = {
                "type": "function",
                "function": {
                    "name": "_warmup",
                    "description": "warmup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            chat_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_dummy_tool],
                "options": options,
                "keep_alive": -1,
            }
            # `think` is supported in ollama-python ≥ 0.4.7 (qwen3 hybrid-think).
            # Try with it first; fall back silently if this version doesn't support it.
            try:
                self._client.chat(**chat_kwargs, think=self._think)
            except TypeError:
                log.debug("ollama client doesn't support think= kwarg; retrying without it")
                self._client.chat(**chat_kwargs)
            elapsed = time.perf_counter() - start
            log.info(f"Model {self._model} warm in {elapsed:.1f}s")
        except Exception as e:
            err = str(e).lower()
            if "404" in err or "not found" in err:
                # Model not installed on this server — retrying won't help
                log.warning(
                    f"Warm-up skipped for {self._model} on {self._base_url}: model not found (404)"
                )
                raise  # propagate so the caller can skip retries immediately
            log.warning(f"Warm-up failed for {self._model}: {e}")

    def is_loaded(self) -> bool:
        """Return True if Ollama currently has this model loaded in RAM.

        Queries GET /api/ps — the same endpoint used by `ollama ps`.
        """
        try:
            import json as _json
            import urllib.request as _req

            with _req.urlopen(f"{self._base_url}/api/ps", timeout=3) as r:
                data = _json.loads(r.read())
            return any(m.get("model", "").startswith(self._model) for m in data.get("models", []))
        except Exception:
            return False

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Generator[LLMChunk, None, None]:
        sys_prompt = system or self._system_prompt

        full_messages = []
        if sys_prompt:
            full_messages.append({"role": "system", "content": sys_prompt})
        full_messages.extend(messages)

        # qwen3 respects /no_think in the last user message to suppress thinking.
        # The Ollama `think` param only works if the template handles it; this is
        # a reliable fallback that works regardless of the modelfile template.
        if not self._think:
            for msg in reversed(full_messages):
                if msg["role"] == "user":
                    if "/no_think" not in msg["content"]:
                        msg["content"] += " /no_think"
                    break

        options: dict[str, Any] = {
            "temperature": self._temperature,
            "num_ctx": self._num_ctx,
        }
        if self._num_predict is not None:
            options["num_predict"] = self._num_predict
        if self._num_thread is not None:
            options["num_thread"] = self._num_thread

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "stream": True,
            "options": options,
            "think": self._think,
            "keep_alive": -1,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.chat(**kwargs)
            in_think = False
            buf = ""

            for chunk in response:
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                buf += content

                # Buffered state machine: split <think>...</think> from visible text.
                # Tags may arrive split across chunks, so we buffer and scan.
                visible, thinking = "", ""
                while True:
                    if not in_think:
                        idx = buf.find("<think>")
                        if idx == -1:
                            # No tag — emit all but last 6 chars (partial "<think" guard)
                            safe = max(0, len(buf) - 6)
                            visible += buf[:safe]
                            buf = buf[safe:]
                            break
                        visible += buf[:idx]
                        buf = buf[idx + 7 :]  # skip "<think>"
                        in_think = True
                    else:
                        idx = buf.find("</think>")
                        if idx == -1:
                            # Still thinking — emit all but last 8 chars (partial guard)
                            safe = max(0, len(buf) - 8)
                            thinking += buf[:safe]
                            buf = buf[safe:]
                            break
                        thinking += buf[:idx]
                        buf = buf[idx + 8 :]  # skip "</think>"
                        in_think = False

                tool_calls = []
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        tool_calls.append(
                            ToolCall(
                                name=fn.get("name", ""),
                                arguments=fn.get("arguments", {}),
                            )
                        )

                yield LLMChunk(
                    text=visible,
                    thinking=thinking,
                    tool_calls=tool_calls,
                    done=chunk.get("done", False),
                    model=self.name,
                )

            # Flush remaining buffer after stream ends
            if buf.strip():
                if in_think:
                    yield LLMChunk(thinking=buf, model=self.name)
                else:
                    yield LLMChunk(text=buf, model=self.name)

        except Exception as e:
            log.error(f"Ollama stream error: {e}")
            raise  # Let state_machine handle fallback silently; don't pollute TTS with error text


# ---------------------------------------------------------------------------
# llama.cpp server (local RPi)
# ---------------------------------------------------------------------------
class LlamaCppBackend:
    """llama.cpp server via its OpenAI-compatible /v1/chat/completions API.

    No warm-up or keep-alive needed — the model is always loaded when the
    server is running.  Tool calling uses the standard OpenAI streaming
    format (arguments are streamed per-character and accumulated here).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "",
        temperature: float = 0.7,
        num_predict: int | None = 500,
        system_prompt: str = "",
        label: str = "local",
    ):
        import httpx

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._num_predict = num_predict
        self._system_prompt = system_prompt
        self._label = label
        self._timeout = httpx.Timeout(connect=3.0, read=300.0, write=30.0, pool=10.0)

    @property
    def name(self) -> str:
        return f"{self._label}/{self._model}" if self._model else self._label

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        """Convert internal Ollama-style message format to strict OpenAI format.

        - tool_calls entries get id, type="function", arguments as JSON string
        - tool result messages get tool_call_id instead of tool_name
        """
        normalized: list[dict] = []
        # (name, id) pairs from the most recent assistant tool_calls block
        pending: list[tuple[str, str]] = []

        for msg in messages:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                pending = []
                new_tcs = []
                for i, tc in enumerate(msg["tool_calls"]):
                    fn = tc.get("function", tc)
                    name = fn["name"]
                    args = fn.get("arguments", {})
                    call_id = tc.get("id") or f"call_{name}_{i}"
                    new_tcs.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else args,
                            },
                        }
                    )
                    pending.append((name, call_id))
                normalized.append({**msg, "tool_calls": new_tcs})
            elif msg["role"] == "tool":
                tool_name = msg.get("tool_name", "")
                call_id = f"call_{tool_name}_0"
                for i, (name, cid) in enumerate(pending):
                    if name == tool_name:
                        call_id = cid
                        pending.pop(i)
                        break
                normalized.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(msg.get("content", "")),
                    }
                )
            else:
                normalized.append(msg)

        return normalized

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Generator[LLMChunk, None, None]:
        import httpx

        sys_prompt = system or self._system_prompt
        full_messages: list[dict] = []
        if sys_prompt:
            full_messages.append({"role": "system", "content": sys_prompt})
        full_messages.extend(self._normalize_messages(messages))

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "stream": True,
            "temperature": self._temperature,
            # Disable thinking mode for Gemma 4 (and other thinking-capable models).
            # The Jinja template checks `enable_thinking` to inject <|think|>;
            # setting it false prevents that token and keeps inference fast.
            # Also set thinking_budget_tokens=0 as belt-and-suspenders for builds
            # that support the per-request reasoning budget API.
            # Most reliable method: start llama-server with --reasoning-budget 0.
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_budget_tokens": 0,
            # Reuse KV cache from the previous request when the prompt prefix is
            # identical (system prompt + tools + earlier conversation turns).
            # After the first request, only new messages need prefill — this cuts
            # first-token latency from ~70s to <5s on subsequent queries.
            "cache_prompt": True,
        }
        if self._num_predict is not None:
            payload["max_tokens"] = self._num_predict
        if tools:
            payload["tools"] = tools

        # index → {name, args_str} — accumulate streamed tool-call fragments
        tool_acc: dict[int, dict] = {}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream(
                    "POST",
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        finish_reason = choices[0].get("finish_reason")

                        # Text tokens — emit immediately
                        content = delta.get("content")
                        if content:
                            yield LLMChunk(text=content, model=self.name)

                        # Tool-call argument fragments — accumulate across chunks
                        for tcd in delta.get("tool_calls", []):
                            idx = tcd["index"]
                            if idx not in tool_acc:
                                tool_acc[idx] = {"name": "", "args_str": ""}
                            fn = tcd.get("function", {})
                            if fn.get("name"):
                                tool_acc[idx]["name"] += fn["name"]
                            tool_acc[idx]["args_str"] += fn.get("arguments", "")

                        # Flush complete tool calls when server signals finish.
                        # Some llama.cpp builds send "stop" instead of "tool_calls";
                        # flush on any terminal signal if we accumulated tool data.
                        if finish_reason and tool_acc:
                            calls: list[ToolCall] = []
                            for acc in (tool_acc[i] for i in sorted(tool_acc)):
                                try:
                                    args = json.loads(acc["args_str"]) if acc["args_str"] else {}
                                except json.JSONDecodeError:
                                    args = {}
                                calls.append(ToolCall(name=acc["name"], arguments=args))
                            yield LLMChunk(tool_calls=calls, model=self.name)
                            tool_acc = {}

            yield LLMChunk(done=True, model=self.name)
        except Exception as e:
            log.error(f"LlamaCpp stream error: {e}")
            raise


# ---------------------------------------------------------------------------
# Claude (cloud)
# ---------------------------------------------------------------------------
class ClaudeBackend:
    def __init__(self, model: str = "claude-sonnet-4-5-20250929", max_tokens: int = 1024):
        import anthropic

        self._client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"claude/{self._model}"

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Generator[LLMChunk, None, None]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": self._normalize_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            # Convert Ollama-style tools to Anthropic format
            kwargs["tools"] = self._convert_tools(tools)

        try:
            with self._client.messages.stream(**kwargs) as stream:
                current_tool_name = ""
                current_tool_json = ""

                for event in stream:
                    if not hasattr(event, "type"):
                        continue

                    if event.type == "content_block_start":
                        cb = getattr(event, "content_block", None)
                        if cb and getattr(cb, "type", None) == "tool_use":
                            current_tool_name = getattr(cb, "name", "")
                            current_tool_json = ""

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            yield LLMChunk(text=delta.text, model=self.name)
                        elif hasattr(delta, "partial_json"):
                            current_tool_json += delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool_name:
                            try:
                                args = json.loads(current_tool_json) if current_tool_json else {}
                            except json.JSONDecodeError:
                                args = {}
                            yield LLMChunk(
                                tool_calls=[ToolCall(name=current_tool_name, arguments=args)],
                                model=self.name,
                            )
                            current_tool_name = ""
                            current_tool_json = ""

                    elif event.type == "message_stop":
                        yield LLMChunk(done=True, model=self.name)
        except Exception as e:
            log.error(f"Claude error: {e}")
            yield LLMChunk(
                text=f"Cloud model unavailable: {e}",
                done=True,
                model=self.name,
            )

    @staticmethod
    def _normalize_messages(messages: list[dict]) -> list[dict]:
        """Convert Ollama-style messages to Anthropic API format.

        Anthropic requires tool calls as content blocks with stable IDs, and
        tool results as user-role messages referencing those IDs.
        """
        normalized: list[dict] = []
        # track (name, id) pairs from the most recent assistant tool_calls block
        pending: list[tuple[str, str]] = []

        for msg in messages:
            role = msg["role"]

            if role == "assistant" and msg.get("tool_calls"):
                pending = []
                content: list[dict] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for i, tc in enumerate(msg["tool_calls"]):
                    fn = tc.get("function", tc)
                    name = fn["name"]
                    args = fn.get("arguments", {})
                    call_id = tc.get("id") or f"call_{name}_{i}"
                    content.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
                    pending.append((name, call_id))
                normalized.append({"role": "assistant", "content": content})

            elif role == "tool":
                tool_name = msg.get("tool_name", "")
                call_id = f"call_{tool_name}_0"
                for i, (name, cid) in enumerate(pending):
                    if name == tool_name:
                        call_id = cid
                        pending.pop(i)
                        break
                # Anthropic groups tool results under role "user"
                # Merge consecutive tool results into the previous user block if present
                if normalized and normalized[-1]["role"] == "user" and isinstance(normalized[-1]["content"], list):
                    normalized[-1]["content"].append(
                        {"type": "tool_result", "tool_use_id": call_id, "content": str(msg.get("content", ""))}
                    )
                else:
                    normalized.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": call_id, "content": str(msg.get("content", ""))}],
                    })

            else:
                normalized.append({"role": role, "content": msg.get("content", "")})

        return normalized

    @staticmethod
    def _convert_tools(ollama_tools: list[dict]) -> list[dict]:
        """Convert Ollama/OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for t in ollama_tools:
            fn = t.get("function", t)
            anthropic_tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                }
            )
        return anthropic_tools


# ---------------------------------------------------------------------------
# Gemini (cloud)
# ---------------------------------------------------------------------------
class GeminiBackend:
    def __init__(self, model: str = "gemini-2.0-flash", max_tokens: int = 1024):
        from google import genai

        self._client = genai.Client()  # Uses GOOGLE_API_KEY env var
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"gemini/{self._model}"

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> Generator[LLMChunk, None, None]:
        from google.genai import types

        # Convert messages to Gemini format (handles text, tool calls, tool results)
        contents = self._convert_messages(messages)

        config = types.GenerateContentConfig(
            system_instruction=system or "",
            max_output_tokens=self._max_tokens,
            tools=self._convert_tools(tools) if tools else None,
            # Disable thinking tokens so function_call turns satisfy Gemini's strict
            # turn-ordering constraint on thinking models (e.g. gemini-3.x-flash-*).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        try:
            response = self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
            for chunk in response:
                # Access raw parts to preserve thought_signature for thinking models
                parts = []
                try:
                    parts = chunk.candidates[0].content.parts or []
                except (AttributeError, IndexError):
                    pass

                fc_parts = [p for p in parts if p.function_call is not None]
                if fc_parts:
                    tool_calls = []
                    for part in fc_parts:
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                                thought_signature=part.thought_signature or None,
                            )
                        )
                    yield LLMChunk(tool_calls=tool_calls, model=self.name)
                else:
                    text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
                    if text:
                        yield LLMChunk(text=text, model=self.name)
            yield LLMChunk(done=True, model=self.name)
        except Exception as e:
            log.error(f"Gemini error: {e}")
            yield LLMChunk(
                text=f"Cloud model unavailable: {e}",
                done=True,
                model=self.name,
            )

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list:
        """Convert internal message format to Gemini Content objects.

        Handles:
          - user/assistant text messages
          - assistant messages with tool_calls → model Content with FunctionCall parts
          - tool result messages → user Content with FunctionResponse parts
            (consecutive tool results are grouped into a single Content)
        """
        from google.genai import types

        contents: list[types.Content] = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "tool":
                # Group consecutive tool results into a single user Content
                parts = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tmsg = messages[i]
                    parts.append(
                        types.Part.from_function_response(
                            name=tmsg.get("tool_name", "unknown"),
                            response={"result": tmsg.get("content", "")},
                        )
                    )
                    i += 1
                contents.append(types.Content(role="user", parts=parts))
                continue

            if msg["role"] == "assistant":
                parts = []
                # Text part (if any)
                if msg.get("content"):
                    parts.append(types.Part(text=msg["content"]))
                # Function call parts from tool-loop metadata
                # Format: {"function": {"name": ..., "arguments": ...}} (Ollama-compatible)
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", tc)  # support both nested and flat
                        thought_sig = tc.get("thought_signature") or fn.get("thought_signature")
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=fn["name"],
                                    args=fn.get("arguments", {}),
                                ),
                                thought_signature=thought_sig,
                            )
                        )
                if not parts:
                    parts.append(types.Part(text=""))
                contents.append(types.Content(role="model", parts=parts))
            else:
                # user message
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg.get("content", ""))],
                    )
                )

            i += 1

        # Gemini requires the conversation to start with a user turn.
        # When the 6-message context window starts mid-history, the first
        # entry may be a model turn — strip it to keep the format valid.
        while contents and contents[0].role != "user":
            contents.pop(0)

        return contents

    @staticmethod
    def _convert_tools(ollama_tools: list[dict]) -> list:
        """Convert Ollama/OpenAI tool format to Gemini format."""
        from google.genai import types

        declarations = []
        for t in ollama_tools:
            fn = t.get("function", t)
            declarations.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters", {}),
                )
            )
        return [types.Tool(function_declarations=declarations)]
