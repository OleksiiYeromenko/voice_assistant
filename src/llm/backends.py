"""LLM backends — local (Ollama) and cloud (Claude, Gemini).

All backends implement the same streaming interface.
"""

import json
import logging
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
            for chunk in response:
                msg = chunk.get("message", {})
                tool_calls = []
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        tool_calls.append(ToolCall(
                            name=fn.get("name", ""),
                            arguments=fn.get("arguments", {}),
                        ))

                yield LLMChunk(
                    text=msg.get("content", ""),
                    tool_calls=tool_calls,
                    done=chunk.get("done", False),
                    model=self.name,
                )
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
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                yield LLMChunk(text=event.delta.text, model=self.name)
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