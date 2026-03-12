#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming benchmark.

Measures first-token latency, tokens/sec, and chunk-boundary overhead
across different chunk_size / context_size / stream_first settings.

Usage:
    python -m benchmarks.run_streaming [--output out.json]
"""

import argparse
import asyncio
import time

from benchmarks.conftest import PerfResult, compute_stats, write_results
from benchmarks.fake_provider import FakeStreamingProvider
from benchmarks.fake_rails import cpu_bound_scan
from benchmarks.scenarios import STREAMING_SWEEP, BenchmarkScenario


async def stream_with_checks(
    provider: FakeStreamingProvider,
    prompt: str,
    chunk_size: int,
    context_size: int,
    stream_first: bool,
    rail_rounds: int = 200,
) -> dict:
    """Simulate a streaming request with chunk-boundary rail checks.

    Returns timing info: first_token_ms, total_ms, tokens, chunks_checked.
    """
    t0 = time.perf_counter()
    first_token_time = None
    buf: list[str] = []
    recent = ""
    total_tokens = 0
    chunks_checked = 0

    async for token in provider.stream(prompt):
        total_tokens += 1
        if first_token_time is None:
            first_token_time = time.perf_counter()

        buf.append(token)
        recent = (recent + token)[-context_size:]

        buf_len = sum(len(t) for t in buf)
        if buf_len >= chunk_size:
            chunk = "".join(buf)
            # Run a quick CPU check on the chunk
            cpu_bound_scan(chunk, rounds=rail_rounds)
            chunks_checked += 1
            buf.clear()

    # Final partial chunk
    if buf:
        chunk = "".join(buf)
        cpu_bound_scan(chunk, rounds=rail_rounds)
        chunks_checked += 1

    total_ms = (time.perf_counter() - t0) * 1000
    ftl_ms = ((first_token_time - t0) * 1000) if first_token_time else total_ms

    return {
        "first_token_ms": ftl_ms,
        "total_ms": total_ms,
        "tokens": total_tokens,
        "chunks_checked": chunks_checked,
    }


async def run_scenario(scenario: BenchmarkScenario) -> PerfResult:
    """Run streaming scenario multiple times and collect stats."""
    provider = FakeStreamingProvider(
        latency_ms=scenario.provider_latency_ms,
        inter_token_ms=3,
    )

    # Warmup
    for _ in range(3):
        await stream_with_checks(
            provider, "warmup", scenario.chunk_size, scenario.context_size,
            scenario.stream_first, rail_rounds=50,
        )

    ftl_list: list[float] = []
    total_list: list[float] = []
    token_counts: list[int] = []

    for _ in range(scenario.iterations):
        info = await stream_with_checks(
            provider, "benchmark prompt",
            scenario.chunk_size, scenario.context_size,
            scenario.stream_first, rail_rounds=scenario.cpu_work_units,
        )
        ftl_list.append(info["first_token_ms"])
        total_list.append(info["total_ms"])
        token_counts.append(info["tokens"])

    stats = compute_stats(total_list)
    ftl_stats = compute_stats(ftl_list)
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    avg_total_s = stats["mean_ms"] / 1000 if stats.get("mean_ms") else 1
    tps = avg_tokens / avg_total_s if avg_total_s > 0 else 0

    return PerfResult(
        scenario=scenario.name,
        count=stats["count"],
        p50_ms=stats["p50_ms"],
        p95_ms=stats["p95_ms"],
        p99_ms=stats["p99_ms"],
        mean_ms=stats["mean_ms"],
        stdev_ms=stats["stdev_ms"],
        first_token_ms=ftl_stats.get("p50_ms", 0),
        tokens_per_sec=round(tps, 1),
    )


async def main():
    parser = argparse.ArgumentParser(description="Streaming benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--max-scenarios", type=int, default=24,
                        help="Cap scenario count for quick runs")
    args = parser.parse_args()

    scenarios = list(STREAMING_SWEEP)[: args.max_scenarios]

    print(f"Running {len(scenarios)} streaming scenarios...\n")
    results = []
    for scenario in scenarios:
        print(f"  {scenario.name} ... ", end="", flush=True)
        result = await run_scenario(scenario)
        print(f"ftl={result.first_token_ms:.1f}ms  tps={result.tokens_per_sec}  p95={result.p95_ms:.1f}ms")
        results.append(result)

    print()
    write_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
