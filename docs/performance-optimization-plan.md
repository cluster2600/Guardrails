<!--
  SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# NeMo Guardrails Performance Optimisation Plan

> **Audience:** Contributors and maintainers of the NeMo Guardrails library.
> **Status:** Living document — updated as optimisations land.

---

## Executive Summary

A guardrails framework sits on the **critical path between the user and
the LLM**.  Every millisecond of orchestration overhead is latency the
user feels before the first token arrives.  The performance goal is
simple: **the framework should be invisible**.

Users report latencies ranging from **50 ms** (lightweight classifiers)
to **10+ seconds** (unoptimised configurations with vector DB + dialog
rails).  This document catalogues every known bottleneck, ranks them by
impact, and proposes concrete fixes with expected gains.

### Tier summary

| Tier | Description | Expected aggregate gain |
|------|-------------|------------------------|
| **P1 — Critical** | High ROI, low risk | 15–30 % latency reduction |
| **P2 — High value** | Medium effort, measurable impact | 10–20 % additional |
| **P3 — Medium value** | Smaller wins or higher risk | 5–10 % additional |

---

## Table of Contents

1. [Current State](#1-current-state)
2. [P1 — Critical Optimisations](#2-p1--critical-optimisations)
3. [P2 — High-Value Optimisations](#3-p2--high-value-optimisations)
4. [P3 — Medium-Value Optimisations](#4-p3--medium-value-optimisations)
5. [Architecture-Level Strategies](#5-architecture-level-strategies)
6. [Streaming Optimisations](#6-streaming-optimisations)
7. [Cold Start & Import Time](#7-cold-start--import-time)
8. [Completed Milestones](#8-completed-milestones)
9. [Milestone Roadmap](#9-milestone-roadmap)
10. [Measurement & Regression Gates](#10-measurement--regression-gates)
11. [References](#11-references)

---

## 1. Current State

### 1.1 Completed Optimisations

| Milestone | Status | Impact | File |
|-----------|--------|--------|------|
| M2 — Zero-overhead tracing | Done | < 1 % overhead when disabled | `llmrails.py:270-280` |
| M3 — Jinja2 template caching | Done | 46–123× speedup on hot templates | `taskmanager.py:101-120` |
| M4 — DAG rail scheduler | Done | 4.12× parallelism (fan-out) | `dag_scheduler.py` |
| M6 — Wrapper overhead | Done | < 2 % overhead vs direct calls | `runnable_rails.py:110` |
| M7 — Free-threaded Python | Partial | Infrastructure in place | `_thread_safety.py` |
| PEP 649 — Deferred annotations | Done | Faster import on Python 3.14+ | Multiple files |

### 1.2 Bottleneck Taxonomy

| Category | Where it lives | Typical magnitude | Optimisation lever |
|----------|---------------|-------------------|-------------------|
| **I/O-bound** | LLM API calls, embedding lookups, network round-trips | 100–2000 ms | Concurrency (asyncio, parallel rails) |
| **CPU-bound** | Regex matching, YARA scanning, perplexity computation | 5–200 ms | Thread-pool dispatch, free-threaded Python |
| **Orchestration** | Config resolution, flow parsing, dependency graph walks | 0.5–5 ms | Caching, DAG scheduling, hot-path optimisation |
| **Wrapper** | RunnableRails adapter, message format conversions | 0.1–2 ms | Eliminate copies, lazy conversion |
| **Tracing** | Span creation, attribute serialisation, callback dispatch | 0.1–3 ms | Pay-for-play (zero cost when disabled) |

### 1.3 Known Pain Points (from GitHub Issues)

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| [#154](https://github.com/NVIDIA/NeMo-Guardrails/issues/154) | 3.5 s minimum latency | Embeddings regenerated per request |
| [#255](https://github.com/NVIDIA/NeMo-Guardrails/issues/255) | 10–11 s with local LLMs | Sequential rail execution + vector DB overhead |
| [#200](https://github.com/NVIDIA/NeMo-Guardrails/issues/200) | Repeated embedding computation | No durable embedding cache |
| [Discussion #587](https://github.com/NVIDIA-NeMo/Guardrails/discussions/587) | 5–10 min in extreme cases | Repeated embedding regeneration |

---

## 2. P1 — Critical Optimisations

### 2.1 Remove the Global Semaphore

**File:** `nemoguardrails/rails/llm/llmrails.py:123, 1392`

```python
# Line 123 — module-level lock
process_events_semaphore = asyncio.Semaphore(1)

# Line 1392 — acquired on every process_events_async() call
async with process_events_semaphore:
    ...
```

**Problem:** Every call to `process_events_async()` (Colang 2.x path)
is serialised behind a single global semaphore.  With 10 concurrent
requests, 9 queue behind the first.  The code contains a TODO comment
("Why is this?"), suggesting it may be overly conservative.

**Proposed fix:**
1. Audit whether the Colang 2.x runtime is actually thread-unsafe.
2. If so, replace with a **per-instance** semaphore (different config
   instances must not block each other).
3. If thread-safe, remove entirely.

**Expected gain:** 2–5× throughput improvement for concurrent requests.

---

### 2.2 Cache Embedding Computations Persistently

**Files:** `nemoguardrails/embeddings/basic.py:138-154`,
`nemoguardrails/embeddings/cache.py`

**Problem:** Policy embeddings (knowledge base, canonical forms) are
recomputed on every request in some configurations.  This is the single
most-reported performance issue on GitHub.

**Proposed fix:**
1. Compute embeddings once at `LLMRails.__init__()` time and store
   in a persistent cache (filesystem or Redis).
2. Invalidate only when source documents change (hash-based check).
3. Ensure the existing `@cache_embeddings` decorator is applied
   consistently to all embedding paths.

**Expected gain:** Eliminates 0.5–3 s per request for knowledge-base
configurations.

---

### 2.3 Extend DAG Scheduler to Colang v2.x

**File:** `nemoguardrails/colang/v2_x/runtime/statemachine.py`

**Problem:** The DAG scheduler is integrated into the Colang v1.0
runtime but **not** into v2.x.  Users on v2.x miss out on
dependency-aware parallel rail execution.

**Proposed fix:**
1. Add `_run_flows_with_dag_scheduler()` to the v2.x runtime,
   analogous to `colang/v1_0/runtime/runtime.py:491`.
2. Use `has_dependencies()` as a fast-path check.
3. Fall back to sequential execution when no dependencies are declared.

**Expected gain:** 2–4× speedup for v2.x with 3+ parallel input rails.

---

### 2.4 Cache Action Name Normalisation

**File:** `nemoguardrails/actions/action_dispatcher.py:199-205`

```python
def _normalize_action_name(self, name: str) -> str:
    if name not in self.registered_actions:
        if name.endswith("Action"):
            name = name.replace("Action", "")
        name = utils.camelcase_to_snakecase(name)
    return name
```

**Problem:** String operations run on every `execute_action()` call.

**Proposed fix:** Build a `_normalised_names: Dict[str, str]` cache at
registration time.  Look up in O(1) without string manipulation.

**Expected gain:** Micro-optimisation — negligible for low-volume
workloads but measurable at 1000+ actions/s.

---

## 3. P2 — High-Value Optimisations

### 3.1 Copy-on-Write Event Lists

**File:** `nemoguardrails/colang/v1_0/runtime/runtime.py:312, 365`

```python
_events = events.copy()           # Line 312 — per parallel rail
stopped_task_processing_logs[flow_id].copy()  # Line 365 — per task
```

**Problem:** Events list copied for every parallel flow.  With 200
events and 5 parallel rails, that is 1000 dict copies.

**Proposed fix:**
- Use `tuple(events)` when flows are read-only.
- For mutating flows, use a copy-on-write wrapper.

**Expected gain:** 5–15 % reduction in multi-rail overhead.

---

### 3.2 Lazy Initialisation of Library Flows

**File:** `nemoguardrails/rails/llm/llmrails.py:166-210`

**Problem:** `LLMRails.__init__()` loads and parses all library flows
at construction time, even if never used.

**Proposed fix:** Defer flow loading until first `generate()` call.

**Expected gain:** 100–300 ms reduction in cold-start latency.

---

### 3.3 Lock-Free LFU Cache Fast Path

**File:** `nemoguardrails/llm/cache/lfu.py:107, 154-173`

**Problem:** Every cache lookup acquires `threading.RLock()`.  At
1000+ req/s, lock contention becomes measurable.

**Proposed fix:**
- Use `threading.Lock()` instead of `RLock()`.
- Double-checked locking for cache hits.
- On Python 3.14t, use per-shard locks.

**Expected gain:** 5–10 % throughput at high concurrency.

---

### 3.4 Bounded Events History Cache

**File:** `nemoguardrails/rails/llm/llmrails.py:166`

```python
self.events_history_cache = {}
```

**Problem:** Never evicted — grows without bound over long-running
instances.

**Proposed fix:** LRU cache with configurable max size (e.g. 1024).

**Expected gain:** Prevents OOM in long-running deployments.

---

## 4. P3 — Medium-Value Optimisations

### 4.1 Pre-Compute Prompt Context

**File:** `nemoguardrails/llm/taskmanager.py:170-180`

Every `_render_string()` call iterates `prompt_context` to check for
callable values.  Pre-compute once when context is set.

### 4.2 Cache Message Format Conversions

**File:** `nemoguardrails/integrations/langchain/runnable_rails.py:569-600`

Multiple format conversions iterate message lists.  For a 20-message
conversation with 5 conversions, that is 100 iterations.  Cache the
converted form keyed by message list hash.

### 4.3 Pre-Normalise Action Names at Registration

Normalise once at `register_action()` time and store under both the
original and normalised names.

### 4.4 Flow Matching Result Caching (Colang v1.0)

**File:** `nemoguardrails/colang/v1_0/runtime/runtime.py:205-235`

`_compute_next_steps()` walks the flow graph for every event.  Cache
the result keyed by `(event_type, frozenset(state.items()))`.

---

## 5. Architecture-Level Strategies

### 5.1 Run Input Rails in Parallel with LLM Call

Instead of the sequential pipeline:

```
[input rails] ──► [LLM call] ──► [output rails]
```

Overlap input rails with the LLM call:

```
[input rails] ──┐
                 ├──► [output rails]
[LLM call]   ──┘
```

If input rails block, discard the LLM response.  Otherwise, the
guardrail latency is hidden behind LLM inference time.

**Implementation:** `asyncio.gather()` with `return_exceptions=True`
to run the LLM call and input rails concurrently.

---

### 5.2 Tiered Filtering

Use a multi-stage pipeline — skip expensive stages if cheap ones block:

1. **Fast pass** (< 5 ms): regex, keyword lists, blocklists.
2. **ML pass** (50–500 ms): classifier-based checks (content safety,
   jailbreak detection).
3. **LLM pass** (1–8 s): LLM-as-Judge for nuanced checks.

---

### 5.3 Use Classifiers Instead of LLM-as-Judge

A fine-tuned BERT classifier (~50 ms) is **100× faster** than an
LLM-based judge (~5–8 s).  Reserve LLM-as-Judge for checks requiring
full conversation reasoning.

| Method | Latency | Accuracy | When to use |
|--------|---------|----------|-------------|
| Regex / keyword | < 1 ms | Low–medium | Known bad patterns |
| BERT classifier | 50 ms | High | Toxicity, content safety |
| NeMo NIM | 100–500 ms | Very high | Jailbreak, topic control |
| LLM-as-Judge | 1–8 s | Highest | Nuanced policy checks |

---

### 5.4 Selective Rail Execution

Not every request needs every rail:

- **Trust-level routing:** Authenticated internal users skip PII checks.
- **Content-type routing:** Code-only responses skip toxicity checks.
- **Length-based routing:** Short responses (< 50 tokens) skip
  hallucination checks.

---

### 5.5 Decouple Non-Critical Rails from the Request Path

For lower-risk applications, run output rails asynchronously:

```python
# Return response immediately
response = await llm.generate(prompt)
yield response

# Run output rails in background — log violations, don't block
asyncio.create_task(run_output_rails(response))
```

---

## 6. Streaming Optimisations

### 6.1 Current Configuration

```yaml
streaming:
  enabled: True
  chunk_size: 200      # tokens per validation window
  context_size: 50     # sliding context overlap
  stream_first: True   # send tokens before validation
```

### 6.2 Optimisation Opportunities

| Area | Current | Proposed | Impact |
|------|---------|----------|--------|
| Context allocation | New dict per chunk | Pre-allocate + reuse | -5 ms/chunk |
| Event list creation | New list per chunk | Reuse template | -2 ms/chunk |
| Chunk size | Fixed 200 | Adaptive based on rail cost | Variable |
| First-token latency | Waits for chunk | Send first token immediately | Perceived -200 ms |

### 6.3 Adaptive Chunk Sizing

Adapt chunk size based on output rail cost:

- **Cheap rails** (regex, blocklist): Large chunks (512 tokens) —
  fewer validations.
- **Expensive rails** (ML classifier): Small chunks (128 tokens) —
  faster intervention.
- **LLM-as-Judge**: Largest chunks (1024 tokens) — amortise the high
  per-call cost.

### 6.4 Streaming Context Reuse

**File:** `nemoguardrails/rails/llm/llmrails.py:1640-1660`

```python
# Current: new context dict per chunk
context = self._prepare_context_for_parallel_rails(...)
events = self._create_events_for_chunk(...)

# Proposed: pre-allocate and update in place
self._chunk_context["text"] = chunk_text  # reuse dict
```

---

## 7. Cold Start & Import Time

### 7.1 Current Import Chain

```
nemoguardrails/__init__.py
  → patch_asyncio (unconditional)
  → rails.RailsConfig
    → config.py (pydantic models, yaml, re, logging)
    → colang parser (lark grammar)
    → embeddings (annoy, numpy — heavy C extensions)
```

### 7.2 Optimisation Strategies

| Strategy | Effort | Impact |
|----------|--------|--------|
| Lazy-import embeddings until first use | Low | -200 ms |
| Lazy-import Colang parser | Low | -100 ms |
| Pre-warm `LLMRails` at app startup | Config | Eliminates per-request init |
| `from __future__ import annotations` | Done | -30 ms on Python 3.14 |
| Only import configured rail modules | Medium | -50–150 ms |

### 7.3 Pre-Warming Pattern

```python
# At application startup — NOT per request
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("config/")
rails = LLMRails(config)

# Per request — reuse the rails instance
result = await rails.generate_async(messages=user_messages)
```

---

## 8. Completed Milestones

### M2 — Zero-Overhead Tracing

Tracing configuration cached as instance attributes.  When tracing is
disabled, the overhead is < 1 % — no span creation, no context
propagation.

### M3 — Jinja2 Template Caching

Templates and variable sets cached in dictionaries keyed by template
string.  Measured improvement: **46–123× speedup** on cached renders.

### M4 — DAG Rail Scheduler

Kahn's algorithm computes execution groups.  Independent rails run
concurrently via `asyncio.gather()`.  Measured: **4.12× speedup** on
fan-out topology with 8 rails.

### M6 — Wrapper Overhead

`GenerationOptions` pre-created to avoid repeated Pydantic
instantiation.  Measured overhead: **< 2 %** vs direct engine calls.

---

## 9. Milestone Roadmap

### Remaining Milestones

| # | Milestone | Status | Risk | ROI |
|---|-----------|--------|------|-----|
| M4+ | DAG scheduler → Colang v2.x | Not started | Medium | High |
| M5 | Streaming first-token optimisation | Partial | Medium | High |
| M7 | Free-threaded Python evidence | Partial | Medium | Medium |
| M8 | Subinterpreters (PEP 734) | Deferred | High | Low |
| M9 | CI regression gates (nightly expansion) | Partial | Low | High |

### Priority Order

1. **M9** — Expand nightly regression gates (prevents future regressions)
2. **§2.1** — Remove global semaphore (highest throughput gain)
3. **§2.2** — Persistent embedding cache (most-requested fix)
4. **M4+** — DAG scheduler for v2.x (unlocks parallelism for v2 users)
5. **§3.1** — Copy-on-write events (reduces allocations on hot path)
6. **M5** — Streaming first-token (improves perceived latency)
7. **M7** — Free-threaded benchmarks (evidence for claims)
8. **§5.1** — Parallel input rails + LLM call (hides rail latency)

---

## 10. Measurement & Regression Gates

### 10.1 Benchmark Harness

The `benchmarks/` directory contains scenario-based benchmarks:

| Runner | What it measures |
|--------|-----------------|
| `run_latency` | End-to-end latency (p50, p95, p99) |
| `run_throughput` | Concurrent request throughput |
| `run_streaming` | Time-to-first-token, chunk latency |
| `run_import` | Cold-start / import time |
| `bench_py314_advantages` | Python 3.14 specific gains |

### 10.2 CI Regression Gates

`.github/workflows/perf.yml` runs:

- **PR gate** (~2 min): latency smoke + import benchmarks
- **Nightly** (~8 min): throughput, streaming, memory, CPU-heavy

Regressions detected by comparing against `perf_baselines/*.json`:

| Metric | Max regression |
|--------|---------------|
| p95 latency | 5 % |
| First-token latency | 20 ms |
| Mean latency | 10 % |

### 10.3 SLO-Style Targets

| Metric | Target | Gate |
|--------|--------|------|
| p95 orchestration (1-rail, no tracing) | ≤ 12 ms | PR blocks |
| p95 orchestration (3-rail parallel) | ≤ 25 ms | PR blocks |
| First-token latency regression | ≤ 20 ms | PR blocks |
| Tracing-disabled overhead | ≤ 2 % | PR blocks |
| Wrapper overhead vs direct | ≤ 5 % | PR blocks |
| Parallel speedup (3 independent rails) | ≥ 2.0× | Nightly |
| Cold-start import time | ≤ 250 ms | Nightly warning |
| RSS drift over 10k requests | ≤ 50 MB | Nightly |

### 10.4 How to Profile Locally

```bash
# Latency smoke test
python -m benchmarks.run_latency --output /tmp/perf.json

# Compare against baseline
python -m benchmarks.compare \
  --baseline perf_baselines/linux-py312.json \
  --current /tmp/perf.json \
  --max-p95-regression 5

# Python 3.14 advantages
python -m benchmarks.bench_py314_advantages --output /tmp/py314.json
```

---

## 11. References

### NVIDIA Resources

- [Measuring AI Guardrails Performance](https://developer.nvidia.com/blog/measuring-the-effectiveness-and-performance-of-ai-guardrails-in-generative-ai-applications/) — NVIDIA blog
- [Streaming with NeMo Guardrails](https://developer.nvidia.com/blog/stream-smarter-and-safer-learn-how-nvidia-nemo-guardrails-enhance-llm-output-streaming/) — NVIDIA blog

### GitHub Issues & Discussions

- [#154 — Performance issues](https://github.com/NVIDIA/NeMo-Guardrails/issues/154)
- [#200 — Durable embeddings](https://github.com/NVIDIA/NeMo-Guardrails/issues/200)
- [#255 — Slow with local LLMs](https://github.com/NVIDIA/NeMo-Guardrails/issues/255)
- [Discussion #587 — Guardrails is very slow](https://github.com/NVIDIA-NeMo/Guardrails/discussions/587)

### External Research

- [Modelmetry — Latency of LLM Guardrails](https://modelmetry.com/blog/latency-of-llm-guardrails) — Benchmark comparison
- [Fiddler AI — Enterprise Guardrails Benchmarks 2025](https://www.fiddler.ai/guardrails-benchmarks) — Industry benchmarks
- [BudEcosystem — Reinventing Guardrails](https://blog.budecosystem.com/reinventing-guardrails-part-1-why-performance-latency-and-safety-need-a-new-equation/) — Performance/safety trade-offs
- [Apple — Disentangled Safety Adapters](https://machinelearning.apple.com/research/disentangled-safety) — LoRA-based safety injection

### Internal Documentation

- [ADR 001 — DAG Rail Scheduler](/adr/001-dag-rail-scheduler.md)
- [Performance Improvements & Benchmarks](/performance-improvements.md)
