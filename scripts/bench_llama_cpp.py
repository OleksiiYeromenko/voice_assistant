#!/usr/bin/env python3
"""Benchmark llama.cpp server inference speed.

Measures time-to-first-token (TTFT), tokens/second, and total latency
for a set of prompts representative of a voice assistant workload.

Thinking suppression (for qwen3 thinking models):
  - Per-request: chat_template_kwargs={"enable_thinking": false} + thinking_budget_tokens=0
  - Note: the most reliable way is to start the server with --reasoning-budget 0
    e.g.  llama-server -m model.gguf --reasoning-budget 0 -ngl 99
  - The /no_think suffix does NOT work with llama.cpp (it's Ollama-only)

Usage:
    uv run scripts/bench_llama_cpp.py                     # bench with defaults
    uv run scripts/bench_llama_cpp.py --rounds 3          # average over 3 rounds
    uv run scripts/bench_llama_cpp.py --max-tokens 150    # cap output length
    uv run scripts/bench_llama_cpp.py --url http://localhost:8080
    uv run scripts/bench_llama_cpp.py -o results.json     # save to JSON
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import httpx

# ---------------------------------------------------------------------------
# Test prompts — short, voice-assistant-style queries
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is the capital of France?",
    "Explain what causes thunder in one sentence.",
    "Name three planets in our solar system.",
    "What's two plus two?",
    "Write one sentence about autumn leaves.",
]

SYSTEM_PROMPT = "You are a helpful home voice assistant. Keep responses concise — 1-3 sentences."

DEFAULT_URL = "http://localhost:8080"
DEFAULT_MAX_TOKENS = 200


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Run:
    prompt: str
    ttft_s: float
    total_s: float
    tokens: int
    tok_per_s: float


@dataclass
class BenchResult:
    label: str
    runs: list[Run] = field(default_factory=list)
    error: str = ""

    @property
    def avg_ttft(self) -> float:
        return sum(r.ttft_s for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def avg_tok_s(self) -> float:
        return sum(r.tok_per_s for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def avg_total(self) -> float:
        return sum(r.total_s for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.runs)


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------
def check_server(base_url: str) -> str | None:
    """Return server model name if reachable, None otherwise."""
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        return models[0]["id"] if models else "<unknown>"
    except Exception as e:
        return None


def bench_prompt(
    client: httpx.Client,
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> Run:
    """Stream a single prompt and measure TTFT, tok/s, total time."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "messages": messages,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Disable thinking for qwen3 thinking models.
        # Most reliable method is --reasoning-budget 0 at server startup.
        # These per-request params are the documented API but have known bugs
        # with some model versions (github.com/ggml-org/llama.cpp/issues/20182).
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_budget_tokens": 0,
        # Ask llama.cpp to include usage stats in the final SSE chunk
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first: float | None = None
    completion_tokens = 0
    generated_tokens = 0  # fallback: count non-empty content chunks

    with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
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

            # Usage stats arrive in the final chunk (stream_options.include_usage)
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                generated_tokens += 1  # fallback counter

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first is not None else total

    # Prefer server-reported token count; fall back to chunk count
    tokens = completion_tokens if completion_tokens > 0 else generated_tokens

    return Run(
        prompt=prompt,
        ttft_s=ttft,
        total_s=total,
        tokens=tokens,
        tok_per_s=tokens / total if total > 0 else 0.0,
    )


def run_bench(
    base_url: str,
    prompts: list[str],
    rounds: int,
    max_tokens: int,
    label: str,
) -> BenchResult:
    result = BenchResult(label=label)

    with httpx.Client() as client:
        for r in range(rounds):
            if rounds > 1:
                print(f"  Round {r + 1}/{rounds}")
            for prompt in prompts:
                short = prompt[:45] + ("..." if len(prompt) > 45 else "")
                print(f"    \"{short}\"", end=" ", flush=True)
                try:
                    run = bench_prompt(client, base_url, prompt, max_tokens)
                    result.runs.append(run)
                    print(
                        f"TTFT {run.ttft_s:.2f}s  "
                        f"{run.tok_per_s:.1f} tok/s  "
                        f"{run.tokens} tok  "
                        f"total {run.total_s:.2f}s"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_summary(result: BenchResult) -> None:
    if result.error and not result.runs:
        print(f"\nBenchmark failed: {result.error}")
        return

    print("\n" + "=" * 70)
    print(f"  Model/server: {result.label}")
    print(f"  Prompts run:  {len(result.runs)}")
    print("-" * 70)
    print(f"  Avg TTFT:     {result.avg_ttft:.2f}s")
    print(f"  Avg tok/s:    {result.avg_tok_s:.1f}")
    print(f"  Avg total:    {result.avg_total:.2f}s")
    print(f"  Total tokens: {result.total_tokens}")
    print("=" * 70)

    if result.runs:
        best = min(result.runs, key=lambda r: r.ttft_s)
        worst = max(result.runs, key=lambda r: r.ttft_s)
        print(f"\n  Best TTFT:  {best.ttft_s:.2f}s  ({best.prompt[:40]}...)")
        print(f"  Worst TTFT: {worst.ttft_s:.2f}s  ({worst.prompt[:40]}...)")


def save_json(result: BenchResult, path: str) -> None:
    data = {
        "label": result.label,
        "avg_ttft_s": round(result.avg_ttft, 3),
        "avg_tok_s": round(result.avg_tok_s, 1),
        "avg_total_s": round(result.avg_total, 3),
        "total_tokens": result.total_tokens,
        "error": result.error,
        "runs": [
            {
                "prompt": r.prompt,
                "ttft_s": round(r.ttft_s, 3),
                "total_s": round(r.total_s, 3),
                "tokens": r.tokens,
                "tok_per_s": round(r.tok_per_s, 1),
            }
            for r in result.runs
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp server inference speed")
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"llama.cpp server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Rounds per prompt, results are averaged (default: 1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens per prompt (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Check server reachability
    print(f"llama.cpp server: {args.url}")
    model_name = check_server(args.url)
    if model_name is None:
        print(f"ERROR: Cannot reach server at {args.url}")
        print("Make sure llama.cpp server is running (e.g. llama-server -m model.gguf)")
        sys.exit(1)

    label = f"{model_name} @ {args.url}"
    print(f"Model:  {model_name}")
    print(f"Prompts: {len(PROMPTS)}  Rounds: {args.rounds}  max_tokens: {args.max_tokens}")
    print("Thinking: disabled (chat_template_kwargs + thinking_budget_tokens=0)")
    print("          For reliable suppression start server with: --reasoning-budget 0\n")

    result = run_bench(
        base_url=args.url,
        prompts=PROMPTS,
        rounds=args.rounds,
        max_tokens=args.max_tokens,
        label=label,
    )

    print_summary(result)

    if args.output:
        save_json(result, args.output)


if __name__ == "__main__":
    main()
