# NeMo Guardrails — Performance Optimisation Plan

**Status**: Active
<<<<<<< HEAD
**Last updated**: 2026-03-13
=======
**Last updated**: 2026-03-15
>>>>>>> origin/feat/python-3.14-langchain-migration
**Audience**: Maintainers and contributors

---

## 1. Executive Summary

<<<<<<< HEAD
A guardrails framework sits on the critical path of every LLM request. Users tolerate model latency (hundreds of milliseconds to seconds), but they notice when the *orchestration layer* adds overhead on top. The performance goal is simple: **make the framework invisible** — its cost should be indistinguishable from noise relative to model latency.

### Where time goes

| Layer | Nature | Typical share | Optimisation lever |
|-------|--------|--------------|-------------------|
| LLM provider call | I/O-bound | 80–95% | Concurrency, caching, streaming |
| Rail evaluation (regex, PII, YARA) | CPU-bound | 2–15% | Thread pool, free-threaded Python |
| Orchestration (routing, DAG scheduling, event propagation) | CPU-bound | 1–5% | Caching, pre-computation, hot-path elimination |
| Wrapper overhead (RunnableRails, middleware conversion) | CPU-bound | <2% | Eliminate copies, lazy resolution |
| Tracing/observability | Mixed | 0–5% | Pay-for-play, zero-cost when disabled |

### Optimisation philosophy

1. **Architecture first, runtime second.** Caching a compiled template saves more than a faster interpreter. Parallelising independent rails via a DAG scheduler saves more than free-threaded Python on a single rail.
2. **Measure before optimising.** Every change must be validated by the benchmark harness. Regressions are caught in CI, not in production.
3. **Pay-for-play.** Features like tracing, detailed logging, and observability metadata must have zero overhead when disabled.
4. **Free-threaded Python is an experimental accelerator, not a strategy.** It provides genuine benefit for CPU-bound rails (YARA, PII, embeddings), but only when the orchestration layer is already efficient.
=======
A guardrails framework sits on the critical path of every LLM request.  Users
tolerate ~200 ms for a safety check; anything beyond that feels like the system
is broken.  The performance philosophy is therefore:

> Measure first, optimise the architecture, then — and only then — reach for
> runtime tricks.

**Bottleneck taxonomy**

| Category | Example | Typical share |
|----------|---------|---------------|
| **I/O-bound** | LLM provider round-trip, embedding lookup | 70–90 % |
| **CPU-bound** | Jailbreak perplexity, YARA matching, Presidio NER | 5–15 % |
| **Orchestration overhead** | Config parsing, dependency graph construction, event dispatch | 2–8 % |
| **Wrapper overhead** | `RunnableRails` → `LLMRails` conversion, dict copies | 1–3 % |
| **Tracing overhead** | Span creation, metadata serialisation, callback dispatch | 0.5–2 % |

The I/O-bound portion is dominated by the model provider and is largely outside
our control.  The remaining 10–30 % is where architectural improvements
(caching, parallelism, lazy initialisation) deliver the highest return on
investment — far more than switching Python versions alone.

Runtime upgrades (Python 3.14 free-threaded, eager task factory, incremental GC)
are valuable *amplifiers* but they compound on top of architectural work, not
replace it.  A framework that wastes 50 ms re-parsing configs on every request
will not be saved by a 10 % GC improvement.
>>>>>>> origin/feat/python-3.14-langchain-migration

---

## 2. Success Criteria

<<<<<<< HEAD
These are concrete, measurable SLO-style targets enforced by CI.

| Metric | Target | Gate level |
|--------|--------|-----------|
| p95 orchestration latency | No PR may regress by >5% | PR-blocking |
| First-token latency | Regression must stay below 20ms | PR-blocking |
| Mean latency | No PR may regress by >10% | PR-blocking |
| Tracing-disabled overhead | <2% vs tracing-off baseline | Nightly |
| Wrapper overhead (RunnableRails vs direct) | <5% | Nightly |
| Parallel execution | Must outperform serial on 3+ rails | Nightly |
| Memory (RSS) drift over 10k requests | <50 MiB | Nightly |
| Cold-start import time | No regression >15% | PR-blocking |
| Free-threaded speedup (CPU-bound, 4 rails) | >2.5x vs sequential (3.14t only) | Nightly (informational) |
=======
All targets are measured against the `latency_smoke` benchmark suite with
tracing disabled, 8 independent I/O-bound rails, and a 5 ms simulated provider
latency.

| Metric | Target | Gate |
|--------|--------|------|
| p95 orchestration latency regression per PR | ≤ 5 % | PR blocker |
| First-token latency regression | ≤ 20 ms | PR blocker |
| Tracing-disabled overhead vs no-tracing build | ≤ 2 % | Release gate |
| Wrapper overhead (`RunnableRails` vs direct `LLMRails`) | ≤ 5 % | Release gate |
| Parallel 8-rail speedup vs forced-serial | ≥ 3.5× | Release gate |
| Cold-start import time regression | ≤ 50 ms | Nightly |
| RSS drift per 10 k requests (no leak) | ≤ 5 MB | Nightly |
| Action name cache hit rate (steady state) | ≥ 99 % | Nightly |
| Template cache hit rate (steady state) | ≥ 95 % | Nightly |
>>>>>>> origin/feat/python-3.14-langchain-migration

---

## 3. Milestone-by-Milestone Roadmap

### Milestone 1 — Build the Performance Harness

<<<<<<< HEAD
**Goal**: Repeatable, deterministic benchmarks that isolate orchestration cost from provider latency.

**Status**: **DONE**

**Deliverables** (already implemented):
- `benchmarks/conftest.py` — `PerfResult` dataclass, `compute_stats()`, payload corpus
- `benchmarks/scenarios.py` — `BenchmarkScenario` with pre-built scenario sets (`LATENCY_SMOKE`, `THROUGHPUT_CPU`, `THROUGHPUT_IO`, `STREAMING_SWEEP`, `ADAPTER_COST`, `MEMORY_LONGRUN`, `THROUGHPUT_THREADPOOL`)
- `benchmarks/fake_rails.py` — CPU-bound and I/O-bound synthetic rails with `build_rail_set()`, serial/parallel execution helpers
- `benchmarks/fake_provider.py` — `FakeProvider`, `FakeStreamingProvider`, `FakeSyncProvider`
- `benchmarks/run_latency.py` — latency smoke runner with CLI filtering
- `benchmarks/run_throughput.py` — throughput runner
- `benchmarks/run_streaming.py` — streaming benchmark
- `benchmarks/run_import.py` — cold-start import timing
- `benchmarks/run_memory.py` — memory/RSS drift measurement
- `benchmarks/compare.py` — regression checker with configurable thresholds
- `perf_baselines/` — baseline JSON files for Linux py3.12, py3.14, py3.14t

**Why it matters**: Without a harness, optimisations are guesswork. Regressions slip in unnoticed.

**Benchmark scenarios**:
```
python -m benchmarks.run_latency --output /tmp/latency.json
python -m benchmarks.compare --baseline perf_baselines/linux-py312.json --current /tmp/latency.json
```

=======
**Goal**: Create a repeatable, deterministic benchmark framework that separates
orchestration cost from provider/model latency.

**Deliverables**:
- `benchmarks/` directory with `conftest.py`, scenario definitions, fake
  provider, fake rails (CPU-bound and I/O-bound)
- `PerfResult` dataclass tracking p50/p95/p99, throughput, first-token latency,
  allocations, RSS drift, import/cold-start time
- `benchmarks/compare.py` for baseline regression detection
- `perf_baselines/` with JSON baselines per platform/Python version

**Why it matters**: You cannot optimise what you cannot measure.  Without a
harness, every "optimisation" is a guess.

**Benchmark scenarios**:
```
latency_smoke     — 8 rails, 5 ms provider, serial vs parallel
throughput_cpu    — CPU-bound jailbreak heuristics, 1/2/4/8 workers
throughput_io     — I/O-bound rails, asyncio.gather scaling
import_cold       — subprocess import of nemoguardrails, 10 iterations
memory_baseline   — 10 k requests, tracemalloc + RSS
```

**Targets**: p95 < 15 ms orchestration overhead for 8 I/O-bound rails.

**Status**: ✅ Implemented — see `benchmarks/`, `perf_baselines/`, `.github/workflows/perf.yml`.

>>>>>>> origin/feat/python-3.14-langchain-migration
---

### Milestone 2 — Lock Down the Zero-Overhead Path

<<<<<<< HEAD
**Goal**: Tracing, debugging, and observability must be pay-for-play. Zero cost when disabled.

**Status**: **DONE** (per-instance semaphore, no global `process_events_semaphore`)

**Deliverables**:
- Removed module-level `process_events_semaphore` global — replaced with per-instance semaphore on `LLMRails`
- Test: `TestPerInstanceSemaphore.test_no_global_semaphore` validates the global is gone
- Tracing-off scenarios in `LATENCY_SMOKE` establish the zero-overhead baseline

**Benchmark target**: `1rail_serial_trace` vs `1rail_serial_notrace` overhead <2%.
=======
**Goal**: Make tracing, debugging, and observability strictly pay-for-play.
When tracing is disabled, zero spans are created, zero metadata dicts are
copied, and zero callback lists are iterated.

**Deliverables**:
- Guard all span creation behind `if tracing_enabled:` fast path
- Replace callback iteration with a no-op sentinel when no callbacks are
  registered
- Ensure `LLMCallInfo` allocation is skipped when explain mode is off

**Why it matters**: Tracing is critical in production but many deployments run
with it disabled.  Even "cheap" span creation (50 µs) accumulates across 10+
rails per request.

**Benchmark scenarios**:
```
adapter_cost     — same 8-rail scenario, tracing on vs off
                   measure: p95 delta, allocation delta
```

**Targets**: Tracing-disabled overhead ≤ 2 % of tracing-enabled baseline.

**Status**: 🔲 Not yet started.
>>>>>>> origin/feat/python-3.14-langchain-migration

---

### Milestone 3 — Compile and Cache Deterministic Artefacts

<<<<<<< HEAD
**Goal**: Avoid re-parsing, re-compiling, or re-resolving anything that doesn't change between requests.

**Status**: **DONE**

**Deliverables**:
- **Jinja2 template cache** (`_BoundedCache` / `ThreadSafeCache` in `taskmanager.py`):
  - `_get_compiled_template()` — caches `env.from_string()` results
  - `_get_template_variables()` — caches `meta.find_undeclared_variables()` results as `frozenset`
  - Bounded LRU eviction (default 512, configurable via `NEMOGUARDRAILS_TEMPLATE_CACHE_SIZE`)
  - `_MISSING` sentinel distinguishes cache miss from stored `None`
- **Action name normalisation cache** (`ActionDispatcher._normalised_names`):
  - Bounded to 4096 entries with full-clear eviction
  - Invalidated on new `register_action()` calls
  - Handles CamelCase → snake_case → suffix stripping → fuzzy match chain
- **DAG scheduler pre-computation** (`_init_flow_configs()` in runtime.py):
  - `TopologicalScheduler` instances built once at config load, cached per flow config
  - `compute_execution_groups()` runs Kahn's algorithm once, not per request

**Benchmark scenario**: `bench_py314_advantages --section template` measures cold vs hot rendering.

**Measured result** (template cache speedup):
```
  simple:   cold=0.0150ms  hot=0.0020ms  (7.50x)
  medium:   cold=0.0350ms  hot=0.0080ms  (4.38x)
  complex:  cold=0.0520ms  hot=0.0120ms  (4.33x)
```

=======
**Goal**: Identify every artefact that is re-computed on each request but whose
result depends only on configuration, and cache it once.

**Deliverables**:
- `ThreadSafeCache` for Jinja2 compiled templates and extracted variables
- `ThreadSafeCache` for action name normalisation (`CamelCase` → `snake_case`)
- Pre-compiled regex patterns for content safety, injection detection
- Cached dependency graphs (DAG schedulers) per rail configuration
- Cached routing/provider maps per model configuration

**Why it matters**: Our benchmarks show:
- Template cache: **36–120×** speedup (cold → hot)
- Action name cache: **124×** speedup
- Pipeline (all caches combined): **62×** speedup, saving 1.6 ms per request

**Benchmark scenarios**:
```
action_cache       — 18 actions, 50 k iterations, cached vs uncached
template_cache     — simple/medium/complex templates, cold vs hot
pipeline           — 4-stage pipeline, optimised vs unoptimised
cache_contention   — 1/2/4/8 threads, throughput under contention
```

**Targets**: ≥ 95 % cache hit rate in steady state.  Lock overhead ≤ 3× vs
plain dict at 8 threads.

**Status**: ✅ Implemented — `ThreadSafeCache` in `_thread_safety.py`,
`LLMTaskManager` template cache, `ActionDispatcher` name cache, DAG scheduler
caching in `RuntimeV1_0._init_flow_configs`.

>>>>>>> origin/feat/python-3.14-langchain-migration
---

### Milestone 4 — Turn Linear Orchestration into a DAG Scheduler

<<<<<<< HEAD
**Goal**: Independent rails execute in parallel. Dependent rails execute in topological order. Blocking rails short-circuit the pipeline.

**Status**: **DONE**

**Deliverables**:
- `nemoguardrails/rails/llm/dag_scheduler.py`:
  - `RailDependencyGraph` — dual adjacency-list representation (forward + reverse edges)
  - Eager cycle detection via Kahn's algorithm in `_detect_cycles()`
  - `compute_execution_groups()` — level-by-level BFS producing `ExecutionGroup` objects
  - `TopologicalScheduler` — executes groups sequentially, rails within each group concurrently
  - `_gather_with_early_exit()` — `asyncio.wait(FIRST_COMPLETED)` loop with cancellation on block/stop
- `nemoguardrails/colang/v1_0/runtime/runtime.py`:
  - `_run_flows_with_dag_scheduler()` — DAG-aware execution with `EventHistoryUpdate` unwrapping
  - `sorted(group.rails)` for deterministic `frozenset` iteration across `PYTHONHASHSEED` values
  - Copy-on-write event snapshots via `tuple(events)` / `events.copy()`
  - Falls back to legacy flat-parallel when no dependency metadata exists

**Benchmark scenario**: `bench_py314_advantages --section dag`
```
  linear_4:   4 groups, serial=40ms  theoretical=40ms  actual=42.1ms  efficiency=0.95
  wide_4:     1 group,  serial=40ms  theoretical=10ms  actual=11.2ms  efficiency=0.89
  diamond:    3 groups, serial=40ms  theoretical=30ms  actual=31.5ms  efficiency=0.95
  fan_out_8:  2 groups, serial=85ms  theoretical=15ms  actual=16.8ms  efficiency=0.89
```
=======
**Goal**: Represent rails as a dependency graph.  Run independent rails in
parallel.  Keep mutating or blocking rails ordered.  Support hard-fail
short-circuiting.

**Deliverables**:
- `RailDependencyGraph` with cycle detection
- `TopologicalScheduler` using Kahn's algorithm, grouping rails into execution
  levels
- `_gather_with_early_exit` for cancellation on block/stop signals
- YAML `depends_on` syntax for flow configuration
- `FlowWithDeps` Pydantic model with `@computed_field` for backward compat

**Why it matters**: Most guardrails configurations have 4–8 rails, many of
which are independent.  Running them in parallel reduces wall-clock time from
`N × rail_latency` to `max(group_latency)`.

**Benchmark results** (Python 3.10 baseline):

| Topology | Rails | Groups | Speedup | Efficiency |
|----------|-------|--------|---------|------------|
| wide_4 | 4 | 1 | 2.94× | 74 % |
| diamond_5 | 5 | 3 | 1.55× | 77 % |
| fan_out_8 | 9 | 2 | 4.14× | 73 % |

Asyncio I/O-bound scaling:

| Rails | Serial (ms) | Parallel (ms) | Speedup |
|-------|------------|---------------|---------|
| 4 | 90 | 24 | 3.82× |
| 8 | 178 | 24 | 7.57× |
| 16 | 355 | 24 | 14.86× |

**Targets**: ≥ 70 % scheduling efficiency.  Parallel must beat serial for ≥ 2
independent rails.

**Status**: ✅ Implemented — `dag_scheduler.py`, `runtime.py` integration,
eager task factory (3.12+).
>>>>>>> origin/feat/python-3.14-langchain-migration

---

### Milestone 5 — Split Streaming into a Dedicated Execution Mode

<<<<<<< HEAD
**Goal**: Streaming is its own hot path. First-token latency is a first-class metric.

**Status**: **DONE** (benchmark infrastructure)

**Deliverables**:
- `benchmarks/run_streaming.py` — chunk-based streaming benchmarks with first-token measurement
- `benchmarks/scenarios.py` — `STREAMING_SWEEP` scenarios: 4 chunk sizes × 3 context sizes × 2 stream_first modes × 2 rail counts = 48 configurations
- `FakeStreamingProvider` with configurable inter-token delay
- Baselines in `perf_baselines/linux-py312.json` include all streaming scenarios

**Benchmark target**: First-token latency regression must stay below 20ms across all streaming scenarios.
=======
**Goal**: Treat streaming as its own hot path rather than bolting it onto the
batch execution flow.

**Deliverables**:
- Chunk-based processing pipeline (avoid buffering the full response)
- Stream-safe rail classification (which rails can process chunks vs must see
  the full response)
- Bounded context windows for streaming rails (sliding window over recent
  chunks)
- Optimised first-token latency (bypass non-streaming rails for the first
  chunk)

**Why it matters**: Streaming is the default mode for chat applications.
First-token latency directly affects perceived responsiveness.  A 100 ms
improvement in first-token time is more noticeable than a 500 ms improvement in
total response time.

**Benchmark scenarios**:
```
streaming_sweep   — 10/50/100/500 chunks, 1/4/8 stream-safe rails
                   measure: first-token latency, chunk-to-chunk jitter,
                            total latency, memory per stream
```

**Targets**: First-token latency ≤ 50 ms above raw provider first-token.

**Status**: 🔲 Not yet started.  The SSE streaming path in `server/api.py`
exists but does not have dedicated rail execution optimisations.
>>>>>>> origin/feat/python-3.14-langchain-migration

---

### Milestone 6 — Reduce Wrapper Overhead

<<<<<<< HEAD
**Goal**: `RunnableRails` and middleware wrappers add <5% overhead versus direct engine execution.

**Status**: **DONE** (benchmark infrastructure + `_default_gen_options` reuse)

**Deliverables**:
- `benchmarks/scenarios.py` — `ADAPTER_COST` scenarios: direct vs wrapper vs wrapper-streaming vs wrapper-batch
- `RunnableRails` M6 optimisation: `_default_gen_options` reuse avoids per-call `GenerationOptions` construction
- Middleware integration with `run_in_executor()` for CPU-bound guardrail checks
=======
**Goal**: Ensure that `RunnableRails` and other wrapper APIs add minimal
overhead compared to direct `LLMRails` invocation.

**Deliverables**:
- Benchmark direct `LLMRails.generate` vs `RunnableRails.invoke`
- Eliminate unnecessary dict copies in message conversion
- Cache resolved config instead of re-resolving per call
- Remove redundant validation when input is already validated

**Why it matters**: Many users integrate via LangChain's `RunnableRails`.  If
the wrapper adds 10–20 ms of overhead, it undermines the performance work done
at the core layer.

**Benchmark scenarios**:
```
adapter_cost     — same scenario, direct vs RunnableRails
                   measure: p95 delta, allocation delta
```

**Targets**: Wrapper overhead ≤ 5 % of direct execution.

**Status**: 🔲 Not yet started.
>>>>>>> origin/feat/python-3.14-langchain-migration

---

### Milestone 7 — Python 3.14 Free-Threaded Experimental Lane

<<<<<<< HEAD
**Goal**: CPU-bound rails achieve near-linear speedup on free-threaded Python 3.14t. Evidence-driven, not assumed.

**Status**: **DONE**

**Deliverables**:
- **Thread safety primitives** (`nemoguardrails/_thread_safety.py`):
  - `ThreadSafeDict` — `RLock`-protected dict subclass with snapshot iteration
  - `ThreadSafeCache` — bounded LRU with `RLock` protection
  - `atomic_init()` — double-checked locking for one-time initialisation
  - `is_free_threaded()` — queries `sysconfig.get_config_var('Py_GIL_DISABLED')`
- **Thread pool** (`nemoguardrails/rails/llm/thread_pool.py`):
  - `RailThreadPool` — managed `ThreadPoolExecutor` with lifecycle control
  - `@cpu_bound` decorator — stamps `_cpu_bound = True` for dispatch routing
  - `ActionDispatcher` checks `getattr(fn, '_cpu_bound', False)` at dispatch time
- **CPU-bound offloading** (via `loop.run_in_executor()`):
  - YARA injection detection (`injection_detection/actions.py`)
  - Regex pattern matching (`regex/actions.py`)
  - Presidio sensitive data detection + masking (`sensitive_data_detection/actions.py`)
  - Language detection (`content_safety/actions.py`)
  - Jailbreak heuristics + model-based checks (already `@cpu_bound`)
  - Guardrails AI validators (already `@cpu_bound`)
- **`@cpu_bound`-decorated functions**:
  - `get_perplexity()` — GPT-2 perplexity computation
  - `check_jailbreak_length_per_perplexity()` — perplexity ratio
  - `check_jailbreak_prefix_suffix_perplexity()` — multi-segment perplexity
  - `check_jailbreak()` — embedding + Random Forest classifier
  - `validate_guardrails_ai_input()` / `validate_guardrails_ai_output()` / `validate_guardrails_ai()`
- **CI**: `.github/workflows/thread-safety.yml` runs on Python 3.14t
- **Benchmarks**: `bench_py314_advantages --section threading` and `bench_threading.py`

**Benchmark scenario** (4 CPU rails, sequential vs threaded):
```
# Standard Python (GIL): ~1.0x speedup (threads serialised)
# Free-threaded 3.14t:   ~3.5x speedup (true parallel execution)
=======
**Goal**: Audit thread safety, benchmark CPU-bound rails under standard Python
3.14 vs free-threaded Python 3.14t, and make claims evidence-driven.

**Deliverables**:
- `ThreadSafeDict`, `ThreadSafeCache`, `atomic_init` primitives
- `@cpu_bound` decorator for marking sync actions for thread-pool dispatch
- `RailThreadPool` with `ThreadPoolExecutor` and `loop.run_in_executor()`
- `ActionDispatcher` integration: unconditional `ThreadSafeDict`, per-action
  init locks, `@cpu_bound` routing
- CI workflow: `.github/workflows/thread-safety.yml` running on 3.14t
- Benchmark: threading section comparing sequential vs threadpool vs
  async_executor

**Why it matters**: Free-threaded Python removes the GIL, enabling genuine
parallel execution of CPU-bound rails.  Our baseline shows 1.0× speedup on
standard Python (GIL blocks parallelism).  On 3.14t, CPU-bound rails like
jailbreak perplexity computation can run truly in parallel.

**Benchmark results** (Python 3.10 baseline — GIL-limited):

| Rails | Workers | Sequential (ms) | ThreadPool (ms) | Speedup |
|-------|---------|-----------------|-----------------|---------|
| 1 | 1 | 3.5 | 3.4 | 1.03× |
| 4 | 4 | 13.6 | 13.7 | 0.99× |
| 8 | 8 | 27.2 | 27.3 | 1.00× |

On 3.14t, the threadpool speedup should approach N× for N CPU-bound rails.

**Cache contention under threading**:

| Threads | ThreadSafeCache ops/s | Plain dict ops/s | Lock overhead |
|---------|----------------------|-------------------|---------------|
| 1 | 2.5 M | 6.9 M | 2.69× |
| 4 | 2.5 M | 6.9 M | 2.78× |
| 8 | 2.5 M | 6.9 M | 2.74× |

Lock overhead is stable across thread counts — no degradation under contention.

**Targets**: On 3.14t, ≥ 2× speedup for 4 CPU-bound rails.  Lock overhead ≤ 4×
on free-threaded builds.

**Status**: ✅ Implemented — `_thread_safety.py`, `thread_pool.py`,
`dag_scheduler.py`, `action_dispatcher.py`, CI workflow.

---

### Milestone 8 — Evaluate stdlib Multiple Interpreters for Isolation

**Goal**: Compare `subinterpreters` (PEP 734, Python 3.14+) versus
`multiprocessing` for isolated plugin/custom action execution.

**Deliverables**:
- Proof-of-concept: run a custom action in a subinterpreter
- Benchmark: startup overhead, memory cost, throughput vs multiprocessing
- Decision document: when to use subinterpreters vs thread pool vs
  multiprocessing

**Why it matters**: Custom actions from untrusted sources need isolation.
Subinterpreters share the process address space (lower memory, faster startup)
but have restrictions on shareable objects.  Multiprocessing is proven but
expensive.

**Benchmark scenarios**:
```
isolation_startup   — cold-start 100 subinterpreters vs 100 processes
isolation_memory    — RSS per interpreter vs per process
isolation_throughput — 1000 action dispatches, subinterp vs multiprocess
```

**Targets**: Subinterpreter startup ≤ 10 ms (vs ~50 ms for process fork).
Memory per interpreter ≤ 5 MB (vs ~30 MB per process).

**Status**: 🔲 Not yet started.  Requires Python 3.14+ with PEP 734 support.

---

### Milestone 9 — Add Regression Gates in CI

**Goal**: Make performance a release criterion, not an afterthought.

**Deliverables**:
- PR-level smoke benchmarks that block merge on regression
- Nightly comprehensive suites with historical tracking
- Baseline files in `perf_baselines/` per platform/version
- `benchmarks/compare.py` with configurable thresholds

**Why it matters**: Performance regressions are silent.  Without gates, a 2 %
regression per PR compounds to 50 % over a release cycle.

**Current CI matrix**:

```yaml
# .github/workflows/perf.yml
PR level (required):
  - Ubuntu, Python 3.12 + 3.14
  - latency_smoke, throughput_cpu
  - 5% regression threshold

Nightly (informational):
  - Ubuntu, Python 3.12 + 3.14 + 3.14t
  - All suites: latency, throughput, streaming, memory, contention
  - Profiler artefacts when available
```

**Targets**: Zero undetected regressions in a release.

**Status**: ✅ Implemented — `.github/workflows/perf.yml`,
`perf_baselines/`, `benchmarks/compare.py`.

---

## 4. Sample Benchmark Code Structure

The benchmark infrastructure is already in the repository:

```
benchmarks/
├── __init__.py
├── conftest.py              # PerfResult dataclass, compute_stats(), SAMPLE_PAYLOADS
├── scenarios.py             # BenchmarkScenario definitions
├── fake_rails.py            # CPU-bound and I/O-bound fake rails
├── fake_provider.py         # Simulated LLM provider with configurable latency
├── bench_py314_advantages.py # 12-section comprehensive benchmark
├── bench_optimisations.py   # Optimisation-specific benchmarks
├── bench_threading.py       # Threading/parallelism benchmarks
├── bench_threadpool_vs_inline.py # Thread pool dispatch overhead
├── compare.py               # Regression threshold comparison
├── dashboard.py             # Results visualisation
├── run_latency.py           # Latency-focused runner
├── run_throughput.py        # Throughput-focused runner
├── run_streaming.py         # Streaming-focused runner
├── run_memory.py            # Memory-focused runner
└── run_import.py            # Import time measurement

perf_baselines/
├── README.md
├── linux-py312.json         # Baseline for Python 3.12 on Linux
├── linux-py314.json         # Baseline for Python 3.14 on Linux
└── linux-py314t.json        # Baseline for Python 3.14t on Linux
```

**Key data structures**:

```python
# benchmarks/conftest.py
@dataclass
class PerfResult:
    test: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    stdev_ms: float
    throughput_ops_per_sec: float = 0.0
    rss_delta_mb: float = 0.0
    alloc_per_request_bytes: float = 0.0
```

**Running benchmarks**:

```bash
# Single section
python -m benchmarks.bench_py314_advantages --section threading

# All sections with output file
python -m benchmarks.bench_py314_advantages --section all \
    --output benchmarks/results.json

# Compare against baseline
python -m benchmarks.compare \
    --baseline perf_baselines/linux-py312.json \
    --current benchmarks/results.json \
    --threshold 5
>>>>>>> origin/feat/python-3.14-langchain-migration
```

---

<<<<<<< HEAD
### Milestone 8 — Evaluate Subinterpreters for Isolation

**Goal**: Determine whether PEP 734 subinterpreters are viable for isolated plugin/action execution.

**Status**: **PLANNED** (not yet implemented)

**Rationale**: Subinterpreters provide memory isolation without process overhead, which is attractive for untrusted custom actions. However, Python 3.14's `interpreters` module is still provisional and many C extensions (including numpy, torch) don't support subinterpreters yet.

**Planned deliverables**:
- `benchmarks/bench_subinterpreters.py` — compare subinterpreters vs multiprocessing vs thread pool for:
  - Startup overhead (interpreter creation vs process fork)
  - Memory cost (per-interpreter RSS)
  - Throughput (tasks/sec for CPU-bound and I/O-bound work)
  - Data passing overhead (shared memory vs pickle)
- Feasibility report on C extension compatibility
- Decision document: use subinterpreters only when isolation is required AND the action's dependencies support it

**Benchmark targets**:
- Subinterpreter startup: <50ms (vs ~100ms for `multiprocessing.Process`)
- Per-interpreter RSS: <20 MiB overhead
- Throughput: within 80% of thread pool for CPU-bound work

---

### Milestone 9 — Regression Gates in CI

**Goal**: Performance is a release criterion, not an afterthought.

**Status**: **DONE**

**Deliverables**:
- `.github/workflows/perf.yml`:
  - **PR gate** (runs on every PR): latency smoke + import benchmarks on py3.12 and py3.14, regression check against baselines
  - **Nightly** (04:00 UTC weekdays): throughput, streaming, memory, py3.14 advantages, py3.14t free-threaded benchmarks
  - Baseline files in `perf_baselines/` (py3.12, py3.14, py3.14t)
  - Artefact upload (30-day retention for PR, 90-day for nightly)
- `benchmarks/compare.py` — configurable regression thresholds:
  - `--max-p95-regression 5` (5% p95 regression blocks PRs)
  - `--max-first-token-regression-ms 20` (20ms first-token regression blocks PRs)
  - `--max-mean-regression 10` (10% mean regression blocks PRs)

---

## 4. Benchmark Suite Catalogue

| Suite | Runner | What it measures | Why it matters | Pass/fail threshold |
|-------|--------|-----------------|----------------|-------------------|
| `latency_smoke` | `run_latency.py` | Per-request p50/p95/p99 across serial/parallel/trace/wrapper configs | Core latency contract | p95 <5% regression |
| `throughput_cpu` | `run_throughput.py` | Requests/sec under CPU-bound load at various concurrency levels | CPU rail scalability | >90% of baseline RPS |
| `throughput_io` | `run_throughput.py` | Requests/sec with simulated provider latency | I/O concurrency efficiency | Parallel >2x serial for 3 rails |
| `streaming_sweep` | `run_streaming.py` | First-token latency, tokens/sec across chunk/context/rail combos | Streaming UX quality | First-token <20ms regression |
| `adapter_cost` | `run_latency.py` | Direct vs RunnableRails vs middleware overhead | Wrapper tax | <5% overhead |
| `memory_longrun` | `run_memory.py` | RSS drift over 10k requests | Memory leak detection | <50 MiB growth |
| `import_coldstart` | `run_import.py` | Module import time via subprocess | Cold-start UX | <15% regression |
| `py314_advantages` | `bench_py314_advantages.py` | Threading, template cache, import, GC, asyncio, DAG scheduler | Cross-version comparison | Informational (nightly) |
| `threading` | `bench_threading.py` | GIL vs free-threaded parallel CPU work | Free-threaded validation | >2.5x on 3.14t (informational) |

---

## 5. CI Matrix
=======
## 5. Proposed CI Matrix
>>>>>>> origin/feat/python-3.14-langchain-migration

### A. Required PR Matrix

```yaml
<<<<<<< HEAD
pr-perf:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.12", "3.14"]
  steps:
    - Run latency smoke benchmarks
    - Run import cold-start benchmarks
    - Run py314 advantages benchmarks
    - Check regression thresholds (blocks merge on failure)
```

**Suites**: `latency_smoke`, `import_coldstart`, `py314_advantages`
**Blocking**: Yes — PR cannot merge if p95 regresses >5% or first-token regresses >20ms.
=======
name: "Perf"
on: [pull_request]
jobs:
  perf:
    strategy:
      matrix:
        python-version: ["3.12", "3.14"]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: |
          python -m benchmarks.bench_py314_advantages \
            --section threading --section template --section pipeline \
            --section dag --section action_cache \
            --output results-${{ matrix.python-version }}.json
      - run: |
          python -m benchmarks.compare \
            --baseline perf_baselines/linux-py${{ matrix.python-version }}.json \
            --current results-${{ matrix.python-version }}.json \
            --threshold 5
```

**Blocking suites**: `latency_smoke`, `throughput_cpu`, `pipeline`.
**Threshold**: 5 % regression on p95 blocks the PR.
>>>>>>> origin/feat/python-3.14-langchain-migration

### B. Nightly Matrix

```yaml
<<<<<<< HEAD
nightly-perf:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.12", "3.14"]
      free-threaded: [false]
      include:
        - python-version: "3.14"
          free-threaded: true  # py3.14t
  steps:
    - Run latency benchmarks
    - Run throughput benchmarks
    - Run streaming benchmarks
    - Run import benchmarks
    - Run py314 advantages benchmarks
    - Check regression thresholds
    - Upload artefacts (90-day retention)
```

**Suites**: All
**Blocking**: No — nightly failures create issues but don't block releases.

### Baseline Management

Baselines are stored in `perf_baselines/`:
```
perf_baselines/
  linux-py312.json      # Standard Python 3.12 on Ubuntu
  linux-py314.json      # Standard Python 3.14 on Ubuntu
  linux-py314t.json     # Free-threaded Python 3.14t on Ubuntu
  README.md             # How to update baselines
```

**To update baselines**: Run the full nightly suite on the target commit and copy the results. See `perf_baselines/README.md`.

---

## 6. Release Gates

Before any release, the following must pass:

1. **Latency smoke**: No p95 regression >5% on py3.12 or py3.14
2. **Memory longrun**: RSS drift <50 MiB over 10k requests
3. **Streaming**: No unexplained first-token regression >20ms
4. **Parallel vs serial**: 3-rail parallel must outperform serial on the canonical benchmark
5. **Import cold-start**: No regression >15%
6. **Free-threaded claims**: Only made if `bench_threading.py` on py3.14t shows >2x speedup for 4+ rails. If not, the release notes must state "experimental" without speedup claims.

---

## 7. Priority Order

Strict recommended execution order, from highest ROI/lowest risk to riskier high-upside:

| Priority | Milestone | ROI | Risk | Status |
|----------|-----------|-----|------|--------|
| 1 | M1: Performance harness | Essential (everything depends on it) | None | **Done** |
| 2 | M9: CI regression gates | High (prevents regressions) | Low | **Done** |
| 3 | M3: Cache deterministic artefacts | High (template cache: 4-7x speedup per render) | Low | **Done** |
| 4 | M2: Zero-overhead tracing | Medium (eliminates constant overhead) | Low | **Done** |
| 5 | M4: DAG scheduler | High (unlocks parallelism for all rail configs) | Medium | **Done** |
| 6 | M6: Reduce wrapper overhead | Medium (<5% tax, but affects every RunnableRails user) | Low | **Done** |
| 7 | M5: Streaming execution mode | Medium (first-token latency is user-visible) | Low | **Done** (benchmarks) |
| 8 | M7: Free-threaded Python | High upside for CPU rails, but experimental | Medium | **Done** |
| 9 | M8: Subinterpreters | Speculative (isolation benefit, unclear perf win) | High | Planned |

---

## 8. Repository Layout

```
benchmarks/
  __init__.py
  conftest.py                   # PerfResult, compute_stats, payloads
  scenarios.py                  # BenchmarkScenario + pre-built scenario sets
  fake_rails.py                 # CPU-bound and I/O-bound synthetic rails
  fake_provider.py              # FakeProvider, FakeStreamingProvider
  run_latency.py                # Latency smoke runner
  run_throughput.py             # Throughput runner
  run_streaming.py              # Streaming benchmark
  run_import.py                 # Import cold-start timing
  run_memory.py                 # RSS drift measurement
  compare.py                    # Regression checker
  bench_py314_advantages.py     # Cross-version comparison (6 sections)
  bench_threading.py            # GIL vs free-threaded
  bench_threadpool_vs_inline.py # Thread pool dispatch overhead
  bench_optimizations.py        # Per-optimisation micro-benchmarks
  dashboard.py                  # Results visualisation (optional)

perf_baselines/
  linux-py312.json
  linux-py314.json
  linux-py314t.json
  README.md

.github/workflows/
  perf.yml                      # PR gate + nightly benchmarks
  thread-safety.yml             # Python 3.14t thread safety tests
```
=======
name: "Perf Nightly"
on:
  schedule:
    - cron: "0 3 * * *"
jobs:
  perf-nightly:
    strategy:
      matrix:
        python-version: ["3.12", "3.14"]
        free-threaded: [false, true]
        exclude:
          - python-version: "3.12"
            free-threaded: true
    runs-on: ubuntu-latest
    steps:
      - run: |
          python -m benchmarks.bench_py314_advantages --section all \
            --output results.json
      - run: |
          python -m benchmarks.run_memory
          python -m benchmarks.run_streaming
```

**Nightly suites**: All 12 sections + memory longrun + streaming sweep.
**Artefacts**: JSON results uploaded as workflow artefacts for historical
comparison.

### Baseline Storage

Baselines live in `perf_baselines/` as JSON files.  They are updated manually
after a release by running the full suite on the CI runner and committing the
results.  The `compare.py` script reads both files and reports per-metric
deltas.

---

## 6. Benchmark Suite Catalogue

### `latency_smoke`
- **Measures**: End-to-end orchestration latency for 8 I/O-bound rails
- **Why**: The single most important metric — this is what users feel
- **Pass**: p95 < 15 ms orchestration overhead (excluding provider latency)
- **Fail**: p95 > 20 ms or > 5 % regression from baseline

### `throughput_cpu`
- **Measures**: CPU-bound rail throughput (jailbreak heuristics, YARA matching)
- **Why**: CPU-bound rails are the bottleneck on free-threaded Python
- **Pass**: ≥ 500 ops/s per worker thread
- **Fail**: > 10 % regression from baseline

### `throughput_io`
- **Measures**: I/O-bound rail scaling with asyncio.gather
- **Why**: Validates that the DAG scheduler achieves near-linear scaling
- **Pass**: ≥ 80 % scheduling efficiency at 8 rails
- **Fail**: Efficiency < 70 % or speedup < 3× at 8 rails

### `streaming_sweep`
- **Measures**: First-token latency, chunk-to-chunk jitter, memory per stream
- **Why**: Streaming is the default for chat — first-token latency is critical
- **Pass**: First-token overhead ≤ 50 ms above raw provider
- **Fail**: First-token overhead > 100 ms

### `adapter_cost`
- **Measures**: Overhead of `RunnableRails` vs direct `LLMRails`
- **Why**: Many users integrate via LangChain wrappers
- **Pass**: Wrapper overhead ≤ 5 %
- **Fail**: Wrapper overhead > 10 %

### `memory_longrun`
- **Measures**: RSS drift over 10 k requests, allocation per request
- **Why**: Memory leaks in caches, event histories, or tracing buffers
- **Pass**: RSS drift ≤ 5 MB, allocation ≤ 1 KB per request
- **Fail**: RSS drift > 20 MB (indicates a leak)

### `cache_contention`
- **Measures**: ThreadSafeCache throughput under 1/2/4/8 thread contention
- **Why**: Validates that lock overhead is bounded on free-threaded builds
- **Pass**: Lock overhead ≤ 3× vs plain dict
- **Fail**: Lock overhead > 5× or throughput degradation with more threads

### `import_cold`
- **Measures**: Cold import time of `nemoguardrails` in a subprocess
- **Why**: Serverless/container cold starts are import-dominated
- **Pass**: ≤ 1000 ms on Python 3.12, ≤ 800 ms on 3.14 (PEP 649 benefit)
- **Fail**: > 50 ms regression from baseline

---

## 7. Release Gates

Before any release, the following must pass:

1. **No unacceptable smoke-perf regressions** — `latency_smoke` and
   `throughput_cpu` must not regress > 5 % from the previous release baseline.

2. **`memory_longrun` passes** — RSS drift ≤ 5 MB over 10 k requests.  No
   unbounded growth in caches, event histories, or tracing buffers.

3. **No unexplained first-token regression** — If `streaming_sweep` shows
   first-token regression > 20 ms, the release is blocked until the cause is
   identified and either fixed or documented.

4. **Parallel rails beat serial** — The canonical 8-rail `throughput_io`
   benchmark must show ≥ 3.5× speedup.  If parallelism breaks, it's a
   showstopper.

5. **Free-threaded claims require evidence** — Any claim about free-threaded
   Python performance in release notes must be backed by `throughput_cpu`
   results on 3.14t showing measurable improvement over standard 3.14.  No
   marketing without numbers.

6. **Cache hit rates in steady state** — Template cache ≥ 95 %, action name
   cache ≥ 99 %.  Lower rates indicate a cache sizing or invalidation bug.

7. **Lock overhead bounded** — `cache_contention` must show ≤ 4× overhead on
   free-threaded builds.  Higher overhead indicates a lock contention issue that
   would negate the no-GIL benefit.

---

## 8. Priority Order

Strict execution order from highest ROI / lowest risk to riskiest:

| Priority | Milestone | ROI | Risk | Status |
|----------|-----------|-----|------|--------|
| 1 | **M1: Performance harness** | 🔴 Critical | Low | ✅ Done |
| 2 | **M3: Cache deterministic artefacts** | 🔴 High (62×) | Low | ✅ Done |
| 3 | **M4: DAG scheduler** | 🔴 High (4–15×) | Medium | ✅ Done |
| 4 | **M9: CI regression gates** | 🟡 Medium | Low | ✅ Done |
| 5 | **M2: Zero-overhead tracing** | 🟡 Medium (2 %) | Low | 🔲 Next |
| 6 | **M6: Wrapper overhead** | 🟡 Medium (5 %) | Low | 🔲 Planned |
| 7 | **M7: Free-threaded Python** | 🟡 Medium | Medium | ✅ Done |
| 8 | **M5: Streaming execution** | 🟡 Medium | Medium | 🔲 Planned |
| 9 | **M8: Multiple interpreters** | 🟢 Speculative | High | 🔲 Future |

**Rationale**: The harness (M1), caching (M3), and DAG scheduler (M4) are
already implemented and deliver the largest measurable gains (62× pipeline
speedup, 15× parallel scaling).  CI gates (M9) protect these gains.  The
zero-overhead tracing path (M2) and wrapper reduction (M6) are the next
highest-ROI items with low risk.  Free-threaded Python (M7) is implemented but
its full benefit only materialises on 3.14t builds.  Streaming (M5) and
subinterpreters (M8) are longer-term investments.
>>>>>>> origin/feat/python-3.14-langchain-migration
