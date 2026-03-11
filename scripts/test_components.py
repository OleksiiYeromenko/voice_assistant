#!/usr/bin/env python3
"""Component tests — run individual pieces to verify they work.

Usage:
  uv run scripts/test_components.py stt          # Test microphone + transcription
  uv run scripts/test_components.py tts           # Test text-to-speech
  uv run scripts/test_components.py llm           # Test Ollama chat
  uv run scripts/test_components.py llm-tools     # Test Ollama with tool calling
  uv run scripts/test_components.py gemini        # Test Gemini streaming
  uv run scripts/test_components.py gemini-tools  # Test Gemini with tool calling (full round-trip)
  uv run scripts/test_components.py weather       # Test weather tool
  uv run scripts/test_components.py search        # Test web search tool
  uv run scripts/test_components.py router        # Test model routing
  uv run scripts/test_components.py all           # Run all tests
"""

import sys
import time

sys.path.insert(0, ".")


def test_stt():
    print("=" * 50)
    print("Testing STT (speak for up to 5 seconds)")
    print("=" * 50)
    from src.stt.engine import STTEngine
    from src.config import load_config

    cfg = load_config()
    alsa_dev = cfg["stt"].get("alsa_device")
    stt = STTEngine(model_size="base.en", alsa_device=alsa_dev)
    stt.load()
    text, rec_time, trans_time = stt.record_and_transcribe(max_duration_s=5.0)
    print(f"  Text: {text}")
    print(f"  Record: {rec_time:.2f}s | Transcribe: {trans_time:.2f}s")
    return bool(text)


def test_tts():
    print("\n" + "=" * 50)
    print("Testing TTS")
    print("=" * 50)
    from src.tts.engine import TTSEngine

    tts = TTSEngine()
    text = "Hello! This is a test of the text to speech system."
    print(f"  Speaking: '{text}'")
    tts.speak(text)
    print("  Done!")
    return True


def test_llm():
    print("\n" + "=" * 50)
    print("Testing LLM (Ollama)")
    print("=" * 50)
    from src.llm.backends import OllamaBackend

    llm = OllamaBackend()
    messages = [{"role": "user", "content": "What is 2 + 2? Answer in one sentence."}]

    print("  Response: ", end="", flush=True)
    full = ""
    for chunk in llm.stream(messages):
        print(chunk.text, end="", flush=True)
        full += chunk.text
    print()
    return bool(full.strip())


def test_llm_tools():
    print("\n" + "=" * 50)
    print("Testing LLM with tool calling")
    print("=" * 50)
    from src.llm.backends import OllamaBackend
    from src.tools.executor import ALL_TOOLS, execute_tool

    llm = OllamaBackend()
    messages = [{"role": "user", "content": "What's the weather in London?"}]

    print("  Sending query with tools...")
    text = ""
    tool_calls = []
    for chunk in llm.stream(messages, tools=ALL_TOOLS):
        text += chunk.text
        tool_calls.extend(chunk.tool_calls)

    if tool_calls:
        tc = tool_calls[0]
        print(f"  Tool call: {tc.name}({tc.arguments})")
        result = execute_tool(tc.name, tc.arguments)
        print(f"  Result: {result}")
        return True
    else:
        print(f"  No tool call — direct response: {text[:100]}")
        return bool(text)


def test_gemini():
    print("\n" + "=" * 50)
    print("Testing Gemini (streaming text)")
    print("=" * 50)
    from src.config import load_config
    from src.llm.backends import GeminiBackend

    cfg = load_config()
    model = cfg["llm"]["cloud"]["gemini"]["model"]
    backend = GeminiBackend(model=model)
    print(f"  Model: {backend.name}")

    messages = [{"role": "user", "content": "What is 2 + 2? Answer in one sentence."}]
    print("  Response: ", end="", flush=True)
    full = ""
    for chunk in backend.stream(messages, system="Be brief."):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full += chunk.text
    print()

    if "unavailable" in full.lower():
        print(f"  ✗ API error — check GOOGLE_API_KEY and quota")
        return False
    return bool(full.strip())


def test_gemini_tools():
    print("\n" + "=" * 50)
    print("Testing Gemini with tool calling (full round-trip)")
    print("=" * 50)
    from src.config import load_config
    from src.llm.backends import GeminiBackend
    from src.tools.executor import ALL_TOOLS, execute_tool

    cfg = load_config()
    model = cfg["llm"]["cloud"]["gemini"]["model"]
    backend = GeminiBackend(model=model)
    system = "Use tools when needed. Be brief."

    messages = [{"role": "user", "content": "What time is it right now?"}]
    print(f"  Model: {backend.name}")
    print(f"  Query: '{messages[0]['content']}'")

    # Round 1: expect a tool call
    print("  Round 1 (tool call)...")
    text = ""
    tool_calls = []
    for chunk in backend.stream(messages, tools=ALL_TOOLS, system=system):
        text += chunk.text
        tool_calls.extend(chunk.tool_calls)

    if not tool_calls:
        print(f"  ✗ No tool call — direct response: {text[:100]}")
        return False

    tc = tool_calls[0]
    print(f"    Tool: {tc.name}({tc.arguments})")

    # Execute tool and build round-2 messages
    result = execute_tool(tc.name, tc.arguments)
    print(f"    Result: {result}")

    messages.append({
        "role": "assistant", "content": text,
        "tool_calls": [{"function": {"name": tc.name, "arguments": tc.arguments}}],
    })
    messages.append({"role": "tool", "tool_name": tc.name, "content": result})

    # Round 2: expect a text response using tool result
    print("  Round 2 (final response)...")
    final = ""
    for chunk in backend.stream(messages, tools=ALL_TOOLS, system=system):
        if chunk.text:
            final += chunk.text
    print(f"    Response: {final[:120]}")
    return bool(final.strip())


def test_weather():
    print("\n" + "=" * 50)
    print("Testing weather tool directly")
    print("=" * 50)
    from src.tools.executor import get_weather

    result = get_weather("London")
    print(f"  {result}")
    return "°C" in result


def test_search():
    print("\n" + "=" * 50)
    print("Testing web search tool directly")
    print("=" * 50)
    from src.tools.executor import web_search

    result = web_search("Raspberry Pi 5 specs")
    print(f"  {result[:200]}")
    return bool(result) and "No results" not in result


def test_router():
    print("\n" + "=" * 50)
    print("Testing model router")
    print("=" * 50)
    from src.router.router import ModelRouter

    # Mock backends
    class MockBackend:
        def __init__(self, n):
            self._name = n
        @property
        def name(self): return self._name

    backends = {
        "local": MockBackend("local"),
        "claude": MockBackend("claude"),
        "gemini": MockBackend("gemini"),
    }
    router = ModelRouter(backends=backends)

    tests = [
        ("What time is it?", "local"),
        ("Use Claude to analyze this data", "claude"),
        ("Ask Gemini about quantum computing", "gemini"),
        ("Switch to Claude", "claude"),
        ("What's the weather?", "claude"),  # Session preference
        ("Go back to automatic", "local"),
        ("What's 2+2?", "local"),  # Reset
    ]

    all_pass = True
    for text, expected in tests:
        decision = router.route(text)
        status = "✓" if decision.backend_key == expected else "✗"
        if decision.backend_key != expected:
            all_pass = False
        print(f"  {status} '{text}' → {decision.backend_key} ({decision.reason})")

    return all_pass


TESTS = {
    "stt": test_stt,
    "tts": test_tts,
    "llm": test_llm,
    "llm-tools": test_llm_tools,
    "gemini": test_gemini,
    "gemini-tools": test_gemini_tools,
    "weather": test_weather,
    "search": test_search,
    "router": test_router,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TESTS and sys.argv[1] != "all":
        print(__doc__)
        sys.exit(1)

    to_run = TESTS.keys() if sys.argv[1] == "all" else [sys.argv[1]]
    results = {}

    for name in to_run:
        try:
            results[name] = TESTS[name]()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results.items():
        print(f"  {'✓' if passed else '✗'} {name}")


if __name__ == "__main__":
    main()