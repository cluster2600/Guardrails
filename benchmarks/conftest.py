# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for benchmark runners."""

import json
import os
import platform
import statistics
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class PerfResult:
    """Standardised benchmark result.

    JSON-serialisable for CI regression checks.
    """

    scenario: str
    python: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    mode: str = field(default_factory=lambda: "free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "standard")
    platform: str = field(default_factory=lambda: f"{platform.system()}-{platform.machine()}")
    count: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    stdev_ms: float = 0.0
    rps: float = 0.0
    first_token_ms: float = 0.0
    tokens_per_sec: float = 0.0
    rss_mb_delta: float = 0.0
    allocations_per_req: int = 0
    wall_time_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Round floats for readability
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def compute_stats(latencies_ms: Sequence[float]) -> dict:
    """Compute p50/p95/p99/mean/stdev from a list of latencies in ms."""
    if not latencies_ms:
        return {}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "count": n,
        "p50_ms": s[n // 2],
        "p95_ms": s[int(n * 0.95)],
        "p99_ms": s[int(n * 0.99)],
        "mean_ms": statistics.mean(s),
        "stdev_ms": statistics.stdev(s) if n > 1 else 0.0,
    }


def write_result(result: PerfResult, output: str | None = None):
    """Write a PerfResult to stdout and optionally to a JSON file."""
    d = result.to_dict()
    blob = json.dumps(d, indent=2)
    print(blob)
    if output:
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
        # ru_maxrss is in KiB on Linux, bytes on macOS
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Payload corpus
# ---------------------------------------------------------------------------

SAMPLE_PAYLOADS = [
    "What is the capital of France?",
    "Can you help me write a cover letter for a software engineering position?",
    "My social security number is 123-45-6789 and my email is user@example.com",
    "Tell me how to make a simple pasta dish with tomatoes and basil.",
    (
        "I need advice on investing in cryptocurrency. Should I put all my "
        "savings into Bitcoin? My credit card number is 4111-1111-1111-1111."
    ),
    "Explain quantum computing in simple terms for a 10-year-old.",
    "What are the best practices for writing unit tests in Python?",
    "Please summarise the main themes of Shakespeare's Hamlet.",
]
