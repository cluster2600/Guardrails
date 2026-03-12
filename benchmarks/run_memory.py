#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory profiling benchmark.

Measures RSS growth and (optionally) per-request allocation counts over
many iterations to detect memory leaks and excessive allocation pressure.

Uses the MEMORY_LONGRUN scenarios from scenarios.py and follows the same
runner pattern as run_latency.py and run_streaming.py.

Usage:
    python -m benchmarks.run_memory [--output out.json] [--tracemalloc]

Options:
    --output, -o     Write results JSON to this file
    --tracemalloc    Enable tracemalloc for detailed allocation tracking
    --snapshot-interval  How often (iterations) to sample RSS (default: 1000)

Exit codes:
    0 — benchmark completed successfully
"""

import argparse
import asyncio
import gc
import sys
import time
import tracemalloc as _tracemalloc_mod

from benchmarks.conftest import (
    SAMPLE_PAYLOADS,
    PerfResult,
    compute_stats,
    get_rss_mb,
    write_results,
)
from benchmarks.fake_provider import FakeProvider
from benchmarks.fake_rails import build_rail_set, run_rails_parallel, run_rails_serial
from benchmarks.scenarios import MEMORY_LONGRUN, BenchmarkScenario


async def run_scenario(
    scenario: BenchmarkScenario,
    use_tracemalloc: bool = False,
    snapshot_interval: int = 1000,
) -> PerfResult:
    """Execute a single memory-profiling scenario.

    1.  Force a GC and record baseline RSS.
    2.  Optionally start tracemalloc.
    3.  Run ``scenario.iterations`` request cycles.
    4.  Sample RSS periodically at ``snapshot_interval`` boundaries.
    5.  After the loop, force GC again and record final RSS.
    6.  Compute per-request allocation stats if tracemalloc is on.
    """
    provider = FakeProvider(latency_ms=scenario.provider_latency_ms)
    rails = build_rail_set(
        num_rails=scenario.num_rails,
        cpu_work_units=scenario.cpu_work_units,
        io_latency_ms=scenario.provider_latency_ms,
        parallel=scenario.parallel,
    )
    run_fn = run_rails_parallel if scenario.parallel else run_rails_serial

    # ---- Warmup ----
    payload = SAMPLE_PAYLOADS[0]
    for _ in range(5):
        await provider.generate(payload)
        await run_fn(payload, rails)

    # ---- Stabilise memory ----
    gc.collect()
    gc.collect()

    rss_before = get_rss_mb()
    rss_snapshots: list[tuple[int, float]] = [(0, rss_before)]

    # ---- Optional tracemalloc ----
    tracemalloc_active = False
    snapshot_before = None
    if use_tracemalloc:
        _tracemalloc_mod.start()
        snapshot_before = _tracemalloc_mod.take_snapshot()
        tracemalloc_active = True

    # ---- Main loop ----
    latencies: list[float] = []
    wall_start = time.perf_counter()

    for i in range(scenario.iterations):
        payload = SAMPLE_PAYLOADS[i % len(SAMPLE_PAYLOADS)]
        t0 = time.perf_counter()

        response = await provider.generate(payload)
        await run_fn(payload, rails)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        # Periodic RSS snapshot
        if snapshot_interval > 0 and (i + 1) % snapshot_interval == 0:
            rss_snapshots.append((i + 1, get_rss_mb()))

    wall_s = time.perf_counter() - wall_start

    # ---- Final memory measurement ----
    gc.collect()
    gc.collect()

    rss_after = get_rss_mb()
    rss_snapshots.append((scenario.iterations, rss_after))
    rss_delta = rss_after - rss_before

    # ---- Tracemalloc analysis ----
    allocations_per_req = 0
    top_allocators: list[dict] = []

    if tracemalloc_active:
        snapshot_after = _tracemalloc_mod.take_snapshot()
        _tracemalloc_mod.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        total_new_blocks = sum(s.count_diff for s in stats if s.count_diff > 0)
        allocations_per_req = (
            total_new_blocks // scenario.iterations
            if scenario.iterations > 0
            else 0
        )

        # Capture top allocators for diagnostic detail
        for stat in stats[:10]:
            top_allocators.append({
                "file": str(stat.traceback),
                "count_diff": stat.count_diff,
                "size_diff_kb": round(stat.size_diff / 1024, 1),
            })

    # ---- Compute latency stats ----
    stats_dict = compute_stats(latencies)

    # ---- Build extra diagnostics ----
    extra: dict = {
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(rss_after, 2),
        "rss_snapshots": [
            {"iteration": it, "rss_mb": round(rss, 2)}
            for it, rss in rss_snapshots
        ],
        "gc_counts": list(gc.get_count()),
    }
    if top_allocators:
        extra["top_allocators"] = top_allocators
    if tracemalloc_active:
        extra["tracemalloc_enabled"] = True

    return PerfResult(
        scenario=scenario.name,
        count=stats_dict.get("count", 0),
        p50_ms=stats_dict.get("p50_ms", 0.0),
        p95_ms=stats_dict.get("p95_ms", 0.0),
        p99_ms=stats_dict.get("p99_ms", 0.0),
        mean_ms=stats_dict.get("mean_ms", 0.0),
        stdev_ms=stats_dict.get("stdev_ms", 0.0),
        rps=stats_dict.get("count", 0) / wall_s if wall_s > 0 else 0,
        rss_mb_delta=round(rss_delta, 2),
        allocations_per_req=allocations_per_req,
        wall_time_s=round(wall_s, 3),
        extra=extra,
    )


async def main():
    parser = argparse.ArgumentParser(description="Memory profiling benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument(
        "--tracemalloc",
        action="store_true",
        default=False,
        help="Enable tracemalloc for per-request allocation tracking",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=1000,
        help="RSS snapshot interval in iterations (default: 1000)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Run only named scenarios (default: all MEMORY_LONGRUN)",
    )
    args = parser.parse_args()

    scenarios = list(MEMORY_LONGRUN)

    # Optional scenario filter
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s.name in wanted]
        if not scenarios:
            print(f"No matching scenarios found. Available: "
                  f"{[s.name for s in MEMORY_LONGRUN]}", file=sys.stderr)
            sys.exit(1)

    print(f"Running {len(scenarios)} memory scenarios "
          f"(tracemalloc={'on' if args.tracemalloc else 'off'})...\n")

    results: list[PerfResult] = []
    for scenario in scenarios:
        print(f"  {scenario.name} ({scenario.iterations} iterations) ... ",
              end="", flush=True)

        result = await run_scenario(
            scenario,
            use_tracemalloc=args.tracemalloc,
            snapshot_interval=args.snapshot_interval,
        )

        rss_info = f"RSS delta={result.rss_mb_delta:+.1f}MiB"
        alloc_info = (
            f"  allocs/req={result.allocations_per_req}"
            if args.tracemalloc
            else ""
        )
        print(f"p95={result.p95_ms:.1f}ms  {rss_info}{alloc_info}  "
              f"wall={result.wall_time_s:.1f}s")

        results.append(result)

    print()
    write_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
