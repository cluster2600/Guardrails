# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic guardrail implementations for benchmarking.

Two rail archetypes:

* **I/O-bound** — simulates an external API call (content moderation
  service, embedding lookup, etc.).  Dominated by network wait time.
* **CPU-bound** — simulates regex scanning, hashing, PII detection,
  schema validation.  Dominated by compute time.

Each rail returns a standardised result dict so runners can assert
correctness without coupling to NeMo internals.
"""

import asyncio
import hashlib
import re

# Pre-compiled PII patterns (representative of real content-safety rails)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CREDIT_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

_PATTERNS = [_SSN_RE, _EMAIL_RE, _PHONE_RE, _CREDIT_RE]


# ---------------------------------------------------------------------------
# I/O-bound rail
# ---------------------------------------------------------------------------


async def io_bound_rail(text: str, latency_ms: int = 50) -> dict:
    """Simulate an external moderation API call."""
    await asyncio.sleep(latency_ms / 1000)
    return {"ok": True, "kind": "io", "length": len(text)}


def io_bound_rail_sync(text: str, latency_ms: int = 50) -> dict:
    """Sync variant for thread-pool benchmarks."""
    import time

    time.sleep(latency_ms / 1000)
    return {"ok": True, "kind": "io", "length": len(text)}


# ---------------------------------------------------------------------------
# CPU-bound rail
# ---------------------------------------------------------------------------


def cpu_bound_scan(text: str, rounds: int = 500) -> dict:
    """CPU-intensive content scan.

    Performs pattern matching, token counting, and iterative hashing
    to simulate real CPU-bound guardrail work.
    """
    found_pii = False
    for pat in _PATTERNS:
        if pat.search(text):
            found_pii = True
            break

    # Token-level work
    tokens = text.split()
    token_count = len(tokens)

    # Iterative hashing (simulates classification / embedding surrogate)
    h = text.encode("utf-8")
    for _ in range(rounds):
        h = hashlib.sha256(h).digest()

    return {
        "ok": not found_pii,
        "kind": "cpu",
        "tokens": token_count,
        "digest": h.hex()[:16],
    }


async def cpu_bound_rail(text: str, rounds: int = 500) -> dict:
    """Async wrapper over cpu_bound_scan.

    In production you would dispatch to a thread pool; here the
    sync call is inlined so we measure raw CPU cost.
    """
    return cpu_bound_scan(text, rounds)


async def cpu_bound_rail_threaded(text: str, rounds: int = 500) -> dict:
    """Dispatch CPU work to the default executor (thread pool)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, cpu_bound_scan, text, rounds)


# ---------------------------------------------------------------------------
# Mixed rail set builder
# ---------------------------------------------------------------------------


def build_rail_set(
    num_rails: int,
    cpu_work_units: int = 500,
    io_latency_ms: int = 50,
    parallel: bool = False,
) -> list[dict]:
    """Build a list of rail descriptors for the benchmark engine.

    Alternates between CPU-bound and I/O-bound rails.

    Returns a list of dicts:
        [{"name": "rail_0", "kind": "cpu", "fn": <coroutine>, ...}, ...]
    """
    rails = []
    for i in range(num_rails):
        if i % 2 == 0:
            rails.append(
                {
                    "name": f"cpu_rail_{i}",
                    "kind": "cpu",
                    "fn": cpu_bound_rail,
                    "kwargs": {"rounds": cpu_work_units},
                    "stream_safe": True,
                    "mutates_input": False,
                }
            )
        else:
            rails.append(
                {
                    "name": f"io_rail_{i}",
                    "kind": "io",
                    "fn": io_bound_rail,
                    "kwargs": {"latency_ms": io_latency_ms},
                    "stream_safe": True,
                    "mutates_input": False,
                }
            )
    return rails


# ---------------------------------------------------------------------------
# Minimal execution helpers (used by runners, not by NeMo itself)
# ---------------------------------------------------------------------------


async def run_rails_serial(text: str, rails: list[dict]) -> list[dict]:
    """Execute rails one-by-one."""
    results = []
    for rail in rails:
        r = await rail["fn"](text, **rail["kwargs"])
        r["rail_name"] = rail["name"]
        results.append(r)
        if not r.get("ok", True):
            break
    return results


async def run_rails_parallel(text: str, rails: list[dict]) -> list[dict]:
    """Execute independent rails concurrently."""
    coros = [rail["fn"](text, **rail["kwargs"]) for rail in rails]
    raw = await asyncio.gather(*coros, return_exceptions=True)
    results = []
    for rail, r in zip(rails, raw):
        if isinstance(r, Exception):
            results.append({"ok": False, "error": str(r), "rail_name": rail["name"]})
        else:
            r["rail_name"] = rail["name"]
            results.append(r)
    return results
