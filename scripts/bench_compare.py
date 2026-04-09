#!/usr/bin/env python3
"""Compare inference speed of specific Ollama models on local hardware.

Measures time-to-first-token (TTFT), tokens/second, and total latency.
Models are tested on the local Ollama instance (localhost:11434).

Usage:
    uv run scripts/bench_compare.py                  # compare default models
    uv run scripts/bench_compare.py --rounds 3       # average over 3 rounds
    uv run scripts/bench_compare.py --num-predict 100 # cap output length
    uv run scripts/bench_compare.py -o results.json  # save to JSON
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import ollama

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    "qwen3:4b-instruct",
    "qwen3.5:4b",
    "gemma4:e2b",
    "gemma4:e4b",
]

# ---------------------------------------------------------------------------
# Test prompts — varied workload
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is the capital of France?",
    "Explain what causes thunder in one sentence.",
    "Name three planets in our solar system.",
    "What's two plus two?",
    "Write one sentence about autumn leaves.",
]

SYSTEM_PROMPT = "You are a helpful assistant. Keep responses brief."

NUM_CTX = 2048


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
class ModelResult:
    model: str
    load_s: float = 0.0
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
# Benchmark helpers
# ---------------------------------------------------------------------------
def unload_model(client: ollama.Client, model: str) -> None:
    """Tell Ollama to evict the model from RAM (keep_alive=0)."""
    try:
        client.chat(model=model, messages=[{"role": "user", "content": ""}], keep_alive=0)
    except Exception:
        pass


def _chat_no_think(client: ollama.Client, **kwargs) -> any:
    """Call client.chat with think=False; fall back if the kwarg isn't supported."""
    try:
        return client.chat(**kwargs, think=False)
    except TypeError:
        return client.chat(**kwargs)


def warm(client: ollama.Client, model: str) -> float:
    """Load model into RAM, return load time in seconds."""
    t0 = time.perf_counter()
    _chat_no_think(
        client,
        model=model,
        messages=[{"role": "user", "content": "hi /no_think"}],
        options={"num_predict": 1, "num_ctx": 32},
    )
    return time.perf_counter() - t0


def bench_prompt(
    client: ollama.Client,
    model: str,
    prompt: str,
    num_predict: int | None,
) -> Run:
    options: dict = {"temperature": 0.0, "num_ctx": NUM_CTX}
    if num_predict is not None:
        options["num_predict"] = num_predict

    t_start = time.perf_counter()
    t_first: float | None = None
    tokens = 0

    for chunk in _chat_no_think(
        client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + " /no_think"},
        ],
        stream=True,
        options=options,
    ):
        content = chunk.get("message", {}).get("content", "")
        if content and t_first is None:
            t_first = time.perf_counter()
        if chunk.get("done"):
            tokens = chunk.get("eval_count", tokens)

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else total

    return Run(
        prompt=prompt,
        ttft_s=ttft,
        total_s=total,
        tokens=tokens,
        tok_per_s=tokens / total if total > 0 else 0.0,
    )


def bench_model(
    client: ollama.Client,
    model: str,
    rounds: int,
    num_predict: int | None,
) -> ModelResult:
    result = ModelResult(model=model)

    print(f"  Warming up...", end=" ", flush=True)
    try:
        result.load_s = warm(client, model)
    except Exception as e:
        result.error = str(e)
        print(f"FAILED: {e}")
        return result
    print(f"{result.load_s:.1f}s")

    for r in range(rounds):
        if rounds > 1:
            print(f"  Round {r + 1}/{rounds}")
        for prompt in PROMPTS:
            label = prompt[:45] + ("..." if len(prompt) > 45 else "")
            print(f"    \"{label}\"", end=" ", flush=True)
            try:
                run = bench_prompt(client, model, prompt, num_predict)
                result.runs.append(run)
                print(
                    f"TTFT {run.ttft_s:.2f}s  "
                    f"{run.tok_per_s:.1f} tok/s  "
                    f"{run.tokens} tok  "
                    f"total {run.total_s:.2f}s"
                )
            except Exception as e:
                print(f"ERROR: {e}")

    print(f"  Unloading...", end=" ", flush=True)
    unload_model(client, model)
    print("done")

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_summary(results: list[ModelResult]) -> None:
    ok = [r for r in results if not r.error and r.runs]
    failed = [r for r in results if r.error]

    if not ok:
        print("\nNo successful results.")
        return

    ranked = sorted(ok, key=lambda r: r.avg_tok_s, reverse=True)
    best_tok = ranked[0].avg_tok_s

    print("\n" + "=" * 80)
    print(f"  {'MODEL':<28} {'LOAD':>6} {'TTFT':>7} {'TOK/S':>7} {'AVG_T':>7} {'TOKENS':>7}  SPEED")
    print("-" * 80)
    for r in ranked:
        pct = r.avg_tok_s / best_tok * 100 if best_tok > 0 else 0
        bar = "#" * int(pct / 5)
        print(
            f"  {r.model:<28} "
            f"{r.load_s:>5.1f}s "
            f"{r.avg_ttft:>6.2f}s "
            f"{r.avg_tok_s:>6.1f} "
            f"{r.avg_total:>6.2f}s "
            f"{r.total_tokens:>7}  "
            f"{bar:<20} {pct:.0f}%"
        )
    print("=" * 80)

    best = ranked[0]
    print(f"\nFastest: {best.model}")
    print(f"  {best.avg_tok_s:.1f} tok/s  TTFT {best.avg_ttft:.2f}s  avg response {best.avg_total:.2f}s")

    if len(ranked) > 1:
        print("\nComparison vs fastest:")
        for r in ranked[1:]:
            ratio = r.avg_tok_s / best.avg_tok_s if best.avg_tok_s > 0 else 0
            print(f"  {r.model}: {ratio:.2f}x  ({r.avg_tok_s:.1f} tok/s)")

    if failed:
        print(f"\nFailed models: {', '.join(r.model for r in failed)}")


def save_json(results: list[ModelResult], path: str) -> None:
    data = [
        {
            "model": r.model,
            "load_s": round(r.load_s, 3),
            "avg_ttft_s": round(r.avg_ttft, 3),
            "avg_tok_s": round(r.avg_tok_s, 1),
            "avg_total_s": round(r.avg_total, 3),
            "total_tokens": r.total_tokens,
            "error": r.error,
            "runs": [
                {
                    "prompt": run.prompt,
                    "ttft_s": round(run.ttft_s, 3),
                    "total_s": round(run.total_s, 3),
                    "tokens": run.tokens,
                    "tok_per_s": round(run.tok_per_s, 1),
                }
                for run in r.runs
            ],
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Ollama model inference speed")
    parser.add_argument(
        "models", nargs="*",
        help=f"Models to compare (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Rounds per prompt, results are averaged (default: 1)",
    )
    parser.add_argument(
        "--num-predict", type=int, default=None,
        help="Cap output tokens per prompt (default: no limit)",
    )
    parser.add_argument(
        "--url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    client = ollama.Client(host=args.url)

    print(f"Ollama: {args.url}")
    print(f"Models: {', '.join(models)}")
    print(f"Prompts: {len(PROMPTS)}  Rounds: {args.rounds}", end="")
    if args.num_predict:
        print(f"  num_predict cap: {args.num_predict}", end="")
    print("\n")

    results: list[ModelResult] = []
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model}")
        result = bench_model(client, model, args.rounds, args.num_predict)
        results.append(result)
        print()

    print_summary(results)

    if args.output:
        save_json(results, args.output)


if __name__ == "__main__":
    main()
