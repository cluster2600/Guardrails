# NeMo Guardrails Performance Roadmap

A comprehensive analysis of performance optimisation opportunities in the
NeMo Guardrails framework, based on a deep audit of the execution hot path,
async/threading patterns, caching infrastructure, and serialisation overhead.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Where Time Is Spent Today](#where-time-is-spent-today)
3. [Tier 1 — High-Impact Optimisations](#tier-1--high-impact-optimisations)
4. [Tier 2 — Medium-Impact Optimisations](#tier-2--medium-impact-optimisations)
5. [Tier 3 — Low-Impact / Long-Term](#tier-3--low-impact--long-term)
6. [Bugs and Correctness Issues Found](#bugs-and-correctness-issues-found)
7. [Architecture Observations](#architecture-observations)
8. [Measurement Plan](#measurement-plan)

---

## Executive Summary

NeMo Guardrails spends **80–95% of its request latency** on LLM provider
calls (I/O-bound), with the remaining **5–20%** split across rail
evaluation (regex, PII, YARA), orchestration overhead (DAG scheduling,
event routing), template rendering, embedding computation, and
serialisation.

The framework already has several strong optimisations in place:

| Already Optimised | Status |
|---|---|
| Jinja2 template caching (LRU-512) | 45–120x speedup on hot path |
| DAG scheduler with topological parallelism | 3–4x for independent rails |
| Eager task factory (Python 3.12+) | 2.0–2.2x for cache-hit rails |
| Free-threaded CPU pool (Python 3.14t) | 4.06x at 8 parallel CPU rails |
| `@cpu_bound` thread pool offloading | Prevents event loop starvation |
| Embedding cache (in-memory / filesystem / Redis) | Avoids recomputation |
| LFU model cache with statistics | Reduces duplicate LLM calls |

The opportunities below target the **remaining 5–20%** of request latency
plus startup time, cold-start costs, and throughput under concurrency.

---

## Where Time Is Spent Today

| Component | Typical Share | Nature | Key File(s) |
|---|---|---|---|
| LLM provider call | 80–95% | I/O | `actions/llm/generation.py` |
| Rail evaluation (regex, PII, YARA) | 2–15% | CPU | `library/*/actions.py` |
| Embedding computation | 5–30% | CPU/I/O | `embeddings/basic.py` |
| Orchestration (DAG, event routing) | 1–5% | CPU | `dag_scheduler.py`, `runtime.py` |
| Template rendering | 1–3% | CPU | `llm/taskmanager.py` |
| Config parsing / Colang compilation | 0% runtime, startup | CPU | `rails/llm/config.py`, `colang/` |
| JSON / state serialisation | 1–2% | CPU | `server/api.py`, `serialization.py` |
| Tracing / observability | 0–5% | Mixed | `tracing/` |

---

## Tier 1 — High-Impact Optimisations

### 1.1 Prompt-Model Lookup Index

**Problem:** `_get_prompt()` in `llm/prompts.py` iterates through **all**
registered prompts (potentially hundreds) for every task/model combination,
computing a score for each.  This runs on every single LLM call.

**Current code path:**
```
_get_prompt(task, model) → O(n) linear scan with score computation
```

**Proposed fix:** Build a `Dict[Tuple[str, str], TaskPrompt]` index at
prompt load time, keyed by `(task_name, model_type)`.  Fall back to
scoring only when no exact match exists.

**Expected impact:** O(1) prompt lookup for the common case.  On a
typical config with 50–100 prompts, this saves ~0.1–0.5ms per LLM call.

**Files:** `nemoguardrails/llm/prompts.py`

---

### 1.2 Colang Parse Caching

**Problem:** Colang files are re-parsed on every `RailsConfig`
instantiation.  For server deployments that reload configs, this adds
unnecessary startup latency.  The parser runs regex-heavy version
detection (`_is_colang_v2()`) and full grammar parsing without
memoisation.

**Proposed fix:** Content-hash-based cache for parsed Colang ASTs.
Key = SHA-256 of file content; value = parsed flow config.  Invalidate
on content change.

**Expected impact:** Eliminate 50–200ms of parsing on warm reload.

**Files:** `nemoguardrails/colang/__init__.py`, parser modules

---

### 1.3 Config Validation Deduplication

**Problem:** Every `LLMRails()` instantiation re-validates the entire
`RailsConfig` through Pydantic v2 validators, even when the same config
object is reused.  `model_validate()` during state deserialisation also
re-validates the full config.

**Proposed fix:**
- Add a `_validated: bool` flag to `RailsConfig` that skips validation
  on subsequent uses of the same instance.
- For state deserialisation, use `model_construct()` (skips validation)
  when the config was already validated at load time.

**Expected impact:** Save 5–20ms per request on complex configs (10+
validators, nested Pydantic models).

**Files:** `nemoguardrails/rails/llm/config.py`,
`nemoguardrails/colang/v2_x/runtime/serialization.py`

---

### 1.4 Embedding Cache Binary Format

**Problem:** The filesystem embedding cache (`FilesystemCacheStore`)
serialises float32 vectors to JSON text files — one file per text input.
JSON encoding of a 1536-dimensional vector (OpenAI `text-embedding-3`)
is ~12KB of text vs 6KB binary.  Per-file I/O compounds the overhead.

**Proposed fix:**
- Use NumPy `.npy` or `struct.pack()` for binary vector serialisation
  (2x smaller, 5–10x faster read/write).
- Batch writes: group embeddings into shard files (e.g. 100 vectors per
  shard) to reduce filesystem syscalls.

**Expected impact:** 5–10x faster cache reads for large knowledge bases.

**Files:** `nemoguardrails/embeddings/cache.py`

---

### 1.5 Lazy Prompt Loading

**Problem:** `llm/prompts.py` loads **all** YAML prompt files (~80KB)
at module import time, regardless of which tasks the current config
actually uses.  This adds to cold-start latency for every process.

**Proposed fix:** Load prompts lazily on first access per task, or use a
frozen module-level singleton that's only populated once per process.

**Expected impact:** Reduce import time by 10–30ms; reduce memory
footprint for minimal configs.

**Files:** `nemoguardrails/llm/prompts.py`

---

### 1.6 History Cache Bounds

**Problem:** The event history cache (`self.events_history_cache`) in
`llmrails.py` is an unbounded `dict` with no size limit or eviction
policy.  In long-running server deployments, this grows indefinitely.

**Proposed fix:** Replace with a bounded LRU (e.g. `ThreadSafeCache` or
`functools.lru_cache` with `maxsize=1024`).

**Expected impact:** Prevent OOM in long-running services; stable memory
profile.

**Files:** `nemoguardrails/rails/llm/llmrails.py`

---

### 1.7 Double Serialisation Elimination

**Problem:** The server API performs double serialisation in several
places:

```python
# server/api.py line 406
json.dumps(processed_chunk.model_dump())  # Pydantic → dict → JSON
```

`model_dump()` creates a Python dict, then `json.dumps()` converts it to
a string.  Pydantic v2 has `model_dump_json()` which goes directly to
JSON bytes, bypassing the intermediate dict.

**Proposed fix:** Replace `json.dumps(x.model_dump())` with
`x.model_dump_json()` throughout the server layer.

**Expected impact:** 20–40% faster serialisation for API responses
(eliminates dict allocation + GC pressure).

**Files:** `nemoguardrails/server/api.py`,
`nemoguardrails/server/schemas/utils.py`

---

## Tier 2 — Medium-Impact Optimisations

### 2.1 Knowledge Base Initialisation

**Problem:** KB initialisation in `llmrails.py` uses a hardcoded
threading workaround:

```python
if True or check_sync_call_from_async_loop():
    t = threading.Thread(target=asyncio.run, args=(self._init_kb(),))
    t.start()
    t.join()  # Blocks calling thread
```

The `True or` bypasses the actual check, always creating a new thread
and event loop just to join immediately.

**Proposed fix:** If already in an event loop, `await` the init directly.
Otherwise, use `loop.run_until_complete()`.  Remove the dead `True or`.

**Expected impact:** Eliminate ~1–5ms of thread creation/join overhead;
cleaner async lifecycle.

**Files:** `nemoguardrails/rails/llm/llmrails.py` (~line 307)

---

### 2.2 Cache Statistics Off Hot Path

**Problem:** The LFU cache checks whether to log statistics on **every
`get()` and `put()` call**, inside the lock:

```python
with self._lock:
    now = time.time()
    if now - self._last_log_time >= self._log_interval:
        self._log_stats()
        self._last_log_time = now
```

**Proposed fix:** Move statistics logging to a background timer or only
log on explicit request.  The time check inside the lock adds contention.

**Expected impact:** Reduce lock hold time by ~0.01ms per cache
operation (measurable at >1000 RPS).

**Files:** `nemoguardrails/llm/cache/lfu.py`

---

### 2.3 Unified Free-Threading Detection

**Problem:** Three different modules detect free-threaded Python using
three different methods:

| Module | Method |
|---|---|
| `_thread_safety.py` | `sysconfig.get_config_var("Py_GIL_DISABLED")` |
| `dag_scheduler.py` | `sys._is_gil_enabled()` (non-public API) |
| `thread_pool.py` | Duplicate of `_thread_safety.py` |

**Proposed fix:** Consolidate to a single `is_free_threaded()` function
in `_thread_safety.py` and import everywhere.  Remove
`sys._is_gil_enabled()` usage (non-public, may change).

**Expected impact:** Correctness and maintainability.

**Files:** `_thread_safety.py`, `dag_scheduler.py`, `thread_pool.py`

---

### 2.4 Per-Rail Timeout Support

**Problem:** The DAG scheduler only supports a group-wide timeout, not
per-rail timeouts.  A slow rail in one group blocks all other groups.

**Proposed fix:** Add optional `timeout` field to `FlowWithDeps` config
model and honour it in `_execute_with_timeout()`.

**Expected impact:** Better tail-latency control for mixed-speed rails.

**Files:** `nemoguardrails/rails/llm/dag_scheduler.py`,
`nemoguardrails/rails/llm/config.py`

---

### 2.5 Batch Embedding Cache I/O

**Problem:** `FilesystemCacheStore.set()` writes one file per embedding
vector.  For a batch of 100 texts, this is 100 file creates + 100
writes + 100 fsyncs.

**Proposed fix:** Batch writes into shard files (e.g. 100 vectors per
shard file using a simple binary format).  Read back with memory-mapped
I/O.

**Expected impact:** 10–50x fewer syscalls for large batch caches.

**Files:** `nemoguardrails/embeddings/cache.py`

---

### 2.6 Streaming Fire-and-Forget Fix

**Problem:** `streaming.py` line 231 creates an untracked task:

```python
asyncio.create_task(self.pipe_to.push_chunk(chunk))
```

Exceptions in the piped handler are silently lost.

**Proposed fix:** Store task references and check for exceptions on
stream completion, or use `asyncio.TaskGroup` (Python 3.11+) with a
fallback.

**Expected impact:** Correctness — prevents silent data loss.

**Files:** `nemoguardrails/streaming.py`

---

### 2.7 Precompute Colang Version Detection

**Problem:** `_is_colang_v2()` runs regex-based heuristics on every
Colang file without memoisation.

**Proposed fix:** Cache the result keyed by file content hash.

**Expected impact:** Eliminate redundant regex on reload.

**Files:** `nemoguardrails/colang/__init__.py`

---

## Tier 3 — Low-Impact / Long-Term

### 3.1 TaskGroup Migration (Python 3.11+)

Replace `asyncio.gather()` + manual cancellation with
`asyncio.TaskGroup` for structured concurrency and automatic exception
propagation.  Version-gate behind `sys.version_info >= (3, 11)`.

### 3.2 Module-Level Pool Cleanup

The `_CPU_POOL` `ThreadPoolExecutor` in `dag_scheduler.py` is created at
module import time and never cleaned up.  Add an `atexit` handler to shut
it down gracefully.

### 3.3 Event Processing Log Truncation

The processing log (`processing_log`) accumulates linearly with events.
For long conversations this can grow large.  Add a configurable max
length with FIFO eviction.

### 3.4 Parallel LLM Calls

When multiple rails need LLM calls (e.g. self-check-input +
content-safety), these are currently serialised within a single action.
DAG-level parallelism handles this when rails are independent, but within
a single rail's action, multiple LLM calls could be batched.

### 3.5 Connection Pooling for External Rails

Rails that call external APIs (ActiveFence, Pangea, CrowdStrike) create
new HTTP sessions per call.  Share an `aiohttp.ClientSession` across
calls for connection reuse.

### 3.6 Frozen Dataclasses for Hot-Path Objects

Replace frequently-created `dict` objects (events, results) with
`@dataclass(frozen=True, slots=True)` for lower allocation cost and
attribute access time.

### 3.7 orjson for JSON Serialisation

Replace `json.dumps()` / `json.loads()` with `orjson` throughout the
server layer.  orjson is 3–10x faster for JSON encoding/decoding and
produces bytes directly.

---

## Bugs and Correctness Issues Found

| Issue | Severity | File | Line |
|---|---|---|---|
| `True or check_sync_call_from_async_loop()` bypasses check | Medium | `llmrails.py` | ~307 |
| Streaming `create_task()` not tracked — lost exceptions | Medium | `streaming.py` | 231 |
| `streaming_handler_var` defined twice (second overwrites) | Low | `context.py` | 23, 32 |
| HuggingFace streamer puts `text` instead of `stop_signal` | Medium | `huggingface/streamers.py` | 59 |
| `busy_count` in `AsyncWorkQueue` not atomic on free-threaded | Low | `async_work_queue.py` | — |
| `process_events_semaphore` is module-global, not per-instance | Medium | `llmrails.py` | 123 |
| History cache grows unbounded | Medium | `llmrails.py` | 166 |

---

## Architecture Observations

### Strengths

1. **Well-structured async pipeline** — contextvar-based request
   isolation, proper `asyncio.gather()` with `return_exceptions=True`.

2. **DAG scheduler** — topological sort ensures correct dependency
   order while maximising parallelism.  Early exit on block saves work.

3. **Template caching** — the bounded LRU cache with thread-safe
   wrapper is a model implementation.  45–120x speedup is significant.

4. **Pluggable cache stores** — embedding cache supports in-memory,
   filesystem, and Redis with a clean interface.

5. **`@cpu_bound` decorator** — clean separation of CPU-bound work with
   automatic thread-pool dispatch.

### Weaknesses

1. **No request-level caching layer** — each request re-validates
   configs, re-resolves prompts, and re-creates transient objects.

2. **Sync/async impedance** — several places use `threading.Thread` +
   `join()` or `nest_asyncio` to bridge sync and async code, adding
   overhead and complexity.

3. **Per-file embedding cache** — filesystem store uses one file per
   vector, creating excessive I/O for large knowledge bases.

4. **Three detection methods for free-threading** — inconsistent and
   fragile.

5. **No structured concurrency** — manual task tracking and cancellation
   instead of `TaskGroup`.

---

## Measurement Plan

To validate these optimisations, extend the existing benchmark suite
(`benchmarks/bench_py314_advantages.py`) with:

| Benchmark | What It Measures | Target |
|---|---|---|
| `prompt_lookup` | `_get_prompt()` latency with/without index | <0.01ms |
| `colang_parse` | Cold vs cached Colang parse time | 10x speedup |
| `config_validate` | First vs subsequent config validation | Skip on reuse |
| `embedding_cache_io` | JSON vs binary vector serialisation | 5x faster |
| `server_serialisation` | `model_dump` + `json.dumps` vs `model_dump_json` | 30% faster |
| `history_cache_memory` | Memory growth over 10K requests | Bounded |
| `kb_init` | Thread + join vs direct await | Eliminate overhead |

All benchmarks should be runnable with:

```bash
python -m benchmarks.bench_performance_roadmap
```

And produce cross-version comparable JSON output.
