#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Throughput benchmark — CPU-bound and I/O-bound workloads.

Measures requests/sec under concurrency for CPU-heavy and I/O-heavy
rail configurations.

Usage:
    python -m benchmarks.run_throughput [--output out.json] [--suite cpu|io|all]
"""

import argparse
import asyncio
import time

from benchmarks.conftest import (
    SAMPLE_PAYLOADS,
    PerfResult,
    compute_stats,
    get_rss_mb,
    write_results,
)
from benchmarks.fake_provider import FakeProvider
from benchmarks.fake_rails import build_rail_set, run_rails_parallel, run_rails_serial
from benchmarks.scenarios import THROUGHPUT_CPU, THROUGHPUT_IO, BenchmarkScenario


async def run_scenario(scenario: BenchmarkScenario) -> PerfResult:
    """Execute a throughput scenario under concurrency."""
    provider = FakeProvider(latency_ms=scenario.provider_latency_ms)
    rails = build_rail_set(
        num_rails=scenario.num_rails,
        cpu_work_units=scenario.cpu_work_units,
        io_latency_ms=scenario.provider_latency_ms,
        parallel=scenario.parallel,
    )
    run_fn = run_rails_parallel if scenario.parallel else run_rails_serial

    sem = asyncio.Semaphore(scenario.concurrency)
    latencies: list[float] = []

    async def one_request(idx: int):
        payload = SAMPLE_PAYLOADS[idx % len(SAMPLE_PAYLOADS)]
        async with sem:
            t0 = time.perf_counter()
            await provider.generate(payload)
            await run_fn(payload, rails)
            latencies.append((time.perf_counter() - t0) * 1000)

    rss_before = get_rss_mb()
    wall_start = time.perf_counter()
    await asyncio.gather(*[one_request(i) for i in range(scenario.iterations)])
    wall_s = time.perf_counter() - wall_start
    rss_after = get_rss_mb()

    stats = compute_stats(latencies)

    return PerfResult(
        scenario=scenario.name,
        count=stats["count"],
        p50_ms=stats["p50_ms"],
        p95_ms=stats["p95_ms"],
        p99_ms=stats["p99_ms"],
        mean_ms=stats["mean_ms"],
        stdev_ms=stats["stdev_ms"],
        rps=stats["count"] / wall_s if wall_s > 0 else 0,
        wall_time_s=wall_s,
        rss_mb_delta=round(rss_after - rss_before, 2),
    )


async def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--suite", choices=["cpu", "io", "all"], default="all")
    args = parser.parse_args()

    scenarios: list[BenchmarkScenario] = []
    if args.suite in ("cpu", "all"):
        scenarios.extend(THROUGHPUT_CPU)
    if args.suite in ("io", "all"):
        scenarios.extend(THROUGHPUT_IO)

    print(f"Running {len(scenarios)} throughput scenarios...\n")
    results = []
    for scenario in scenarios:
        print(f"  {scenario.name} (c={scenario.concurrency}, n={scenario.iterations}) ... ", end="", flush=True)
        result = await run_scenario(scenario)
        print(f"rps={result.rps:.0f}  p95={result.p95_ms:.1f}ms  rss_delta={result.rss_mb_delta}MB")
        results.append(result)

    print()
    write_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
