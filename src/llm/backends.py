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
        num_thread: int | None = None,
        system_prompt: str = "",
        think: bool = False,
        label: str = "local",
    ):
        import ollama
        self._client = ollama.Client(host=base_url)
        self._base_url = base_url.rstrip("/")   # stored for is_loaded() / diagnostics
        self._model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
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
                log.warning(f"Warm-up skipped for {self._model} on {self._base_url}: model not found (404)")
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
            return any(
                m.get("model", "").startswith(self._model)
                for m in data.get("models", [])
            )
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

        options: dict[str, Any] = {
            "temperature": self._temperature,
            "num_ctx": self._num_ctx,
        }
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
                        buf = buf[idx + 7:]  # skip "<think>"
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
                        buf = buf[idx + 8:]  # skip "</think>"
                        in_think = False

                tool_calls = []
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        tool_calls.append(ToolCall(
                            name=fn.get("name", ""),
                            arguments=fn.get("arguments", {}),
                        ))

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
            "messages": messages,
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
    def _convert_tools(ollama_tools: list[dict]) -> list[dict]:
        """Convert Ollama/OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for t in ollama_tools:
            fn = t.get("function", t)
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
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
        )

        try:
            response = self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
            for chunk in response:
                # Check for function calls first
                fc_list = getattr(chunk, "function_calls", None)
                if fc_list:
                    tool_calls = []
                    for fc in fc_list:
                        tool_calls.append(ToolCall(
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        ))
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
                    parts.append(types.Part.from_function_response(
                        name=tmsg.get("tool_name", "unknown"),
                        response={"result": tmsg.get("content", "")},
                    ))
                    i += 1
                contents.append(types.Content(role="user", parts=parts))
                continue

            if msg["role"] == "assistant":
                parts = []
                # Text part (if any)
                if msg.get("content"):
                    parts.append(types.Part(text=msg["content"]))
                # Function call parts from tool-loop metadata
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        parts.append(types.Part(function_call=types.FunctionCall(
                            name=tc["name"],
                            args=tc.get("arguments", {}),
                        )))
                if not parts:
                    parts.append(types.Part(text=""))
                contents.append(types.Content(role="model", parts=parts))
            else:
                # user message
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg.get("content", ""))],
                ))

            i += 1

        return contents

    @staticmethod
    def _convert_tools(ollama_tools: list[dict]) -> list:
        """Convert Ollama/OpenAI tool format to Gemini format."""
        from google.genai import types

        declarations = []
        for t in ollama_tools:
            fn = t.get("function", t)
            declarations.append(types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            ))
        return [types.Tool(function_declarations=declarations)]