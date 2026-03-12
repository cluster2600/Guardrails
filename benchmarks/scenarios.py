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

"""Benchmark scenario definitions.

Each scenario is an immutable descriptor that tells the runner *what*
to measure without prescribing *how* to wire it up.  Runners import
scenarios and iterate over them.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BenchmarkScenario:
    """One benchmark configuration."""

    name: str
    num_rails: int = 1
    parallel: bool = False
    streaming: bool = False
    tracing: bool = False
    concurrency: int = 1
    provider_latency_ms: int = 10
    cpu_work_units: int = 500
    use_runnable_wrapper: bool = False
    chunk_size: int = 200
    context_size: int = 50
    stream_first: bool = True
    iterations: int = 200


# ---------------------------------------------------------------------------
# Pre-built scenario sets used by the runners
# ---------------------------------------------------------------------------

LATENCY_SMOKE: Sequence[BenchmarkScenario] = (
    BenchmarkScenario(name="1rail_serial_notrace", num_rails=1),
    BenchmarkScenario(name="3rail_serial_notrace", num_rails=3),
    BenchmarkScenario(name="3rail_parallel_notrace", num_rails=3, parallel=True),
    BenchmarkScenario(name="1rail_serial_trace", num_rails=1, tracing=True),
    BenchmarkScenario(name="3rail_serial_trace", num_rails=3, tracing=True),
    BenchmarkScenario(name="3rail_parallel_trace", num_rails=3, parallel=True, tracing=True),
    BenchmarkScenario(name="1rail_wrapper", num_rails=1, use_runnable_wrapper=True),
    BenchmarkScenario(name="3rail_wrapper", num_rails=3, use_runnable_wrapper=True),
)

THROUGHPUT_CPU: Sequence[BenchmarkScenario] = tuple(
    BenchmarkScenario(
        name=f"cpu_{n}rail_c{c}",
        num_rails=n,
        parallel=n > 1,
        concurrency=c,
        cpu_work_units=2000,
        provider_latency_ms=0,
        iterations=500,
    )
    for n in (1, 3, 5)
    for c in (8, 32, 64)
)

THROUGHPUT_IO: Sequence[BenchmarkScenario] = tuple(
    BenchmarkScenario(
        name=f"io_3rail_{mode}_lat{lat}",
        num_rails=3,
        parallel=(mode == "parallel"),
        concurrency=16,
        provider_latency_ms=lat,
        cpu_work_units=100,
        iterations=300,
    )
    for mode in ("serial", "parallel")
    for lat in (20, 100, 300)
)

STREAMING_SWEEP: Sequence[BenchmarkScenario] = tuple(
    BenchmarkScenario(
        name=f"stream_cs{cs}_ctx{ctx}_sf{int(sf)}_r{nr}",
        num_rails=nr,
        parallel=nr > 1,
        streaming=True,
        chunk_size=cs,
        context_size=ctx,
        stream_first=sf,
        iterations=150,
    )
    for cs in (50, 100, 200, 400)
    for ctx in (25, 50, 100)
    for sf in (True, False)
    for nr in (1, 3)
)

ADAPTER_COST: Sequence[BenchmarkScenario] = (
    BenchmarkScenario(name="direct_engine", use_runnable_wrapper=False),
    BenchmarkScenario(name="wrapper_invoke", use_runnable_wrapper=True),
    BenchmarkScenario(name="wrapper_streaming", use_runnable_wrapper=True, streaming=True),
    BenchmarkScenario(
        name="wrapper_batch",
        use_runnable_wrapper=True,
        concurrency=8,
        iterations=400,
    ),
)

MEMORY_LONGRUN: Sequence[BenchmarkScenario] = (
    BenchmarkScenario(
        name="memory_10k",
        num_rails=3,
        parallel=True,
        iterations=10_000,
        concurrency=8,
    ),
)

THROUGHPUT_THREADPOOL: Sequence[BenchmarkScenario] = tuple(
    BenchmarkScenario(
        name=f"tpool_{n}rail_r{r}",
        num_rails=n,
        parallel=n > 1,
        cpu_work_units=r,
        provider_latency_ms=0,
        iterations=100,
    )
    for n in (1, 2, 4, 8)
    for r in (100, 500, 2000)
)
