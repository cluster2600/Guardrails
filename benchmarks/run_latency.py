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

"""Latency smoke benchmark.

Measures per-request latency across scenario configurations:
serial/parallel, tracing on/off, direct/wrapper, 1–3 rails.

Usage:
    python -m benchmarks.run_latency [--output out.json] [--tracing on|off] [--mode serial|parallel] [--path direct|wrapper]
"""

import argparse
import asyncio
import time

from benchmarks.conftest import (
    SAMPLE_PAYLOADS,
    PerfResult,
    compute_stats,
    write_results,
)
from benchmarks.fake_provider import FakeProvider
from benchmarks.fake_rails import build_rail_set, run_rails_parallel, run_rails_serial
from benchmarks.scenarios import LATENCY_SMOKE, BenchmarkScenario


async def run_scenario(scenario: BenchmarkScenario) -> PerfResult:
    """Execute a single latency scenario and return stats."""
    provider = FakeProvider(latency_ms=scenario.provider_latency_ms)
    rails = build_rail_set(
        num_rails=scenario.num_rails,
        cpu_work_units=scenario.cpu_work_units,
        io_latency_ms=scenario.provider_latency_ms,
        parallel=scenario.parallel,
    )
    run_fn = run_rails_parallel if scenario.parallel else run_rails_serial

    # Warmup
    payload = SAMPLE_PAYLOADS[0]
    for _ in range(3):
        await provider.generate(payload)
        await run_fn(payload, rails)

    # Benchmark
    latencies = []
    wall_start = time.perf_counter()

    for i in range(scenario.iterations):
        payload = SAMPLE_PAYLOADS[i % len(SAMPLE_PAYLOADS)]
        t0 = time.perf_counter()

        # Simulate full request: provider call + rail checks
        response = await provider.generate(payload)
        await run_fn(payload, rails)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    wall_s = time.perf_counter() - wall_start
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
    )


def filter_scenarios(
    scenarios: list[BenchmarkScenario],
    tracing: str | None,
    mode: str | None,
    path: str | None,
) -> list[BenchmarkScenario]:
    """Filter scenarios based on CLI flags."""
    result = list(scenarios)
    if tracing is not None:
        want_tracing = tracing == "on"
        result = [s for s in result if s.tracing == want_tracing]
    if mode is not None:
        want_parallel = mode == "parallel"
        result = [s for s in result if s.parallel == want_parallel]
    if path is not None:
        want_wrapper = path == "wrapper"
        result = [s for s in result if s.use_runnable_wrapper == want_wrapper]
    return result


async def main():
    parser = argparse.ArgumentParser(description="Latency smoke benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--tracing", choices=["on", "off"], default=None)
    parser.add_argument("--mode", choices=["serial", "parallel"], default=None)
    parser.add_argument("--path", choices=["direct", "wrapper"], default=None)
    args = parser.parse_args()

    scenarios = filter_scenarios(list(LATENCY_SMOKE), args.tracing, args.mode, args.path)

    print(f"Running {len(scenarios)} latency scenarios...\n")
    results = []
    for scenario in scenarios:
        print(f"  {scenario.name} ... ", end="", flush=True)
        result = await run_scenario(scenario)
        print(f"p50={result.p50_ms:.1f}ms  p95={result.p95_ms:.1f}ms  rps={result.rps:.0f}")
        results.append(result)

    print()
    write_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
