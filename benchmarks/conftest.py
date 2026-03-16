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

"""Shared helpers for benchmark runners."""

import json
import platform
import statistics
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Standardised result container — every benchmark emits one or more of these
# so that CI regression tooling can compare runs across Python versions and
# build flavours (standard GIL vs free-threaded) in a uniform JSON schema.
# ---------------------------------------------------------------------------


@dataclass
class PerfResult:
    """Standardised benchmark result.

    JSON-serialisable for CI regression checks.
    """

    scenario: str
    # Capture the exact CPython micro-version so results are traceable
    python: str = field(
        default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    # Detect free-threaded build at runtime via the Py_GIL_DISABLED sysconfig
    # flag — this lets downstream analysis automatically partition results by
    # GIL vs no-GIL without manual labelling.
    mode: str = field(
        default_factory=lambda: "free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "standard"
    )
    platform: str = field(default_factory=lambda: f"{platform.system()}-{platform.machine()}")
    count: int = 0
    # Percentile latencies in milliseconds — p50 (median), p95, p99
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    stdev_ms: float = 0.0
    rps: float = 0.0  # Requests per second (throughput metric)
    first_token_ms: float = 0.0  # Time-to-first-token for streaming scenarios
    tokens_per_sec: float = 0.0
    rss_mb_delta: float = 0.0  # Resident set size growth (memory leak indicator)
    allocations_per_req: int = 0  # Heap allocation count per request
    wall_time_s: float = 0.0
    extra: dict = field(default_factory=dict)  # Escape hatch for section-specific data

    def to_dict(self) -> dict:
        d = asdict(self)
        # Round floats to four decimal places for human readability in JSON
        # whilst retaining enough precision for regression detection.
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def compute_stats(latencies_ms: Sequence[float]) -> dict:
    """Compute p50/p95/p99/mean/stdev from a list of latencies in ms."""
    if not latencies_ms:
        return {}
    s = sorted(latencies_ms)  # Sort once; percentile look-ups are O(1) by index
    n = len(s)
    return {
        "count": n,
        # Percentiles via nearest-rank method — simple and deterministic,
        # unlike interpolation-based approaches which add overhead without
        # meaningful accuracy improvement at n >= 30 samples.
        "p50_ms": s[n // 2],
        "p95_ms": s[int(n * 0.95)],
        "p99_ms": s[int(n * 0.99)],
        "mean_ms": statistics.mean(s),
        # Guard against single-sample input where stdev is undefined
        "stdev_ms": statistics.stdev(s) if n > 1 else 0.0,
    }


def write_result(result: PerfResult, output: str | None = None):
    """Write a PerfResult to stdout and optionally to a JSON file."""
    d = result.to_dict()
    blob = json.dumps(d, indent=2)
    print(blob)
    if output:
        # Ensure parent directories exist so callers need not pre-create them
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(blob + "\n")


def write_results(results: Sequence[PerfResult], output: str | None = None):
    """Write multiple PerfResults."""
    data = [r.to_dict() for r in results]
    blob = json.dumps(data, indent=2)
    print(blob)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(blob + "\n")


def get_rss_mb() -> float:
    """Return current RSS in MiB (Linux/macOS)."""
    try:
        import resource

        # ru_maxrss units differ by platform: KiB on Linux, bytes on macOS.
        # We normalise to MiB for consistent cross-platform comparison.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)  # bytes -> MiB on macOS
        return rss / 1024  # KiB -> MiB on Linux
    except Exception:
        return 0.0  # Graceful fallback on unsupported platforms (e.g. Windows)


# ---------------------------------------------------------------------------
# Payload corpus
# ---------------------------------------------------------------------------
# A representative mix of user inputs for benchmarking guardrail evaluation.
# Deliberately includes PII patterns (SSN, credit card, e-mail) so that
# content-safety rails have non-trivial work to do during benchmarks.
# The variety of lengths and topics ensures template rendering, tokenisation,
# and pattern-matching benchmarks exercise realistic code paths.
# ---------------------------------------------------------------------------

SAMPLE_PAYLOADS = [
    "What is the capital of France?",
    "Can you help me write a cover letter for a software engineering position?",
    "My social security number is 123-45-6789 and my email is user@example.com",  # PII-laden
    "Tell me how to make a simple pasta dish with tomatoes and basil.",
    (
        "I need advice on investing in cryptocurrency. Should I put all my "
        "savings into Bitcoin? My credit card number is 4111-1111-1111-1111."
    ),  # PII-laden, longest payload — used by GC pressure benchmarks
    "Explain quantum computing in simple terms for a 10-year-old.",
    "What are the best practices for writing unit tests in Python?",
    "Please summarise the main themes of Shakespeare's Hamlet.",
]
