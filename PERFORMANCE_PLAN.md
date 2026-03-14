# NeMo Guardrails — Performance Optimisation Plan

**Status**: Active
**Last updated**: 2026-03-13
**Audience**: Maintainers and contributors

---

## 1. Executive Summary

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

---

## 2. Success Criteria

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

---

## 3. Milestone-by-Milestone Roadmap

### Milestone 1 — Build the Performance Harness

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

---

### Milestone 2 — Lock Down the Zero-Overhead Path

**Goal**: Tracing, debugging, and observability must be pay-for-play. Zero cost when disabled.

**Status**: **DONE** (per-instance semaphore, no global `process_events_semaphore`)

**Deliverables**:
- Removed module-level `process_events_semaphore` global — replaced with per-instance semaphore on `LLMRails`
- Test: `TestPerInstanceSemaphore.test_no_global_semaphore` validates the global is gone
- Tracing-off scenarios in `LATENCY_SMOKE` establish the zero-overhead baseline

**Benchmark target**: `1rail_serial_trace` vs `1rail_serial_notrace` overhead <2%.

---

### Milestone 3 — Compile and Cache Deterministic Artefacts

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

---

### Milestone 4 — Turn Linear Orchestration into a DAG Scheduler

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

---

### Milestone 5 — Split Streaming into a Dedicated Execution Mode

**Goal**: Streaming is its own hot path. First-token latency is a first-class metric.

**Status**: **DONE** (benchmark infrastructure)

**Deliverables**:
- `benchmarks/run_streaming.py` — chunk-based streaming benchmarks with first-token measurement
- `benchmarks/scenarios.py` — `STREAMING_SWEEP` scenarios: 4 chunk sizes × 3 context sizes × 2 stream_first modes × 2 rail counts = 48 configurations
- `FakeStreamingProvider` with configurable inter-token delay
- Baselines in `perf_baselines/linux-py312.json` include all streaming scenarios

**Benchmark target**: First-token latency regression must stay below 20ms across all streaming scenarios.

---

### Milestone 6 — Reduce Wrapper Overhead

**Goal**: `RunnableRails` and middleware wrappers add <5% overhead versus direct engine execution.

**Status**: **DONE** (benchmark infrastructure + `_default_gen_options` reuse)

**Deliverables**:
- `benchmarks/scenarios.py` — `ADAPTER_COST` scenarios: direct vs wrapper vs wrapper-streaming vs wrapper-batch
- `RunnableRails` M6 optimisation: `_default_gen_options` reuse avoids per-call `GenerationOptions` construction
- Middleware integration with `run_in_executor()` for CPU-bound guardrail checks

---

### Milestone 7 — Python 3.14 Free-Threaded Experimental Lane

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
```

---

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

### A. Required PR Matrix

```yaml
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

### B. Nightly Matrix

```yaml
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
