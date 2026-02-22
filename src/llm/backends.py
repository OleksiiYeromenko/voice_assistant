"""LLM backends — local (Ollama) and cloud (Claude, Gemini).

All backends implement the same streaming interface.
"""

import json
import logging
import time
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
# Ollama (local)
# ---------------------------------------------------------------------------
class OllamaBackend:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        num_ctx: int = 4096,
        system_prompt: str = "",
        think: bool = False,
    ):
        import ollama
        self._client = ollama.Client(host=base_url)
        self._model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._system_prompt = system_prompt
        self._think = think

    @property
    def name(self) -> str:
        return f"local/{self._model}"

    def warm(self):
        """Send a minimal request to preload the model into memory."""
        try:
            log.info(f"Warming up {self._model}...")
            start = time.perf_counter()
            self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1, "num_ctx": 32},
            )
            elapsed = time.perf_counter() - start
            log.info(f"Model {self._model} warm in {elapsed:.1f}s")
        except Exception as e:
            log.warning(f"Warm-up failed for {self._model}: {e}")

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

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
            },
            "think": self._think,
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
            log.error(f"Ollama error: {e}")
            yield LLMChunk(
                text=f"Sorry, local model error: {e}",
                done=True,
                model=self.name,
            )


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

        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))

        config = types.GenerateContentConfig(
            system_instruction=system or "",
            max_output_tokens=self._max_tokens,
        )

        try:
            response = self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
            for chunk in response:
                text = chunk.text if hasattr(chunk, "text") and chunk.text else ""
                yield LLMChunk(text=text, model=self.name)
            yield LLMChunk(done=True, model=self.name)
        except Exception as e:
            log.error(f"Gemini error: {e}")
            yield LLMChunk(
                text=f"Cloud model unavailable: {e}",
                done=True,
                model=self.name,
            )