# ADR-001: DAG-Based Rail Scheduler

| Field       | Value                          |
|-------------|--------------------------------|
| **Status**  | Accepted                       |
| **Date**    | 2026-03-12                     |
| **Authors** | NeMo Guardrails team           |
| **Ticket**  | GARDR-34                       |

## Context

NeMo Guardrails executes safety rails (input rails, output rails, tool rails)
as part of every request lifecycle.  The current execution model offers two
modes controlled by the `parallel: bool` flag on `InputRails`, `OutputRails`,
`ToolInputRails`, and `ToolOutputRails` in `config.yml`:

1. **Sequential** (`parallel: false`, the default) -- rails run one after
   another in declared order.  Simple and deterministic, but total latency
   equals the sum of every rail's latency.

2. **All-parallel** (`parallel: true`) -- every rail in the section is
   dispatched concurrently via `asyncio.create_task()` and results are
   collected with `asyncio.as_completed()`.  Latency drops to approximately
   the slowest rail, but every rail must be fully independent: no rail may
   read a value that another concurrent rail writes, or the result is
   non-deterministic.

In practice many configurations fall between these extremes.  For example:

* A content-safety rail and a topic-control rail are independent and can run
  concurrently.
* A PII-scrubbing rail **mutates** the user message, and a downstream
  jailbreak-detection rail must see the scrubbed text.  These two have an
  ordering dependency.
* A custom embedding-lookup rail populates a context variable that a
  classification rail later reads.

There is no way today to express "run A and B in parallel, then run C after
both finish."  Users must choose between full parallelism (risking race
conditions) or full sequentiality (sacrificing latency).

## Decision

Introduce a **DAG-based rail scheduler** that resolves execution order from
explicit dependency declarations.  Rails that share no dependency edges run
concurrently; rails whose inputs depend on the outputs of other rails wait
until those predecessors complete.

### Core algorithm

The scheduler uses **Kahn's algorithm** for topological sorting of the rail
dependency graph:

1. Build an adjacency list from the `depends_on` declarations in `config.yml`.
2. Compute the in-degree of every rail node.
3. Seed an initial ready-set with all rails whose in-degree is zero.
4. While the ready-set is non-empty:
   a. Dispatch every rail in the ready-set concurrently
      (`asyncio.gather()`).
   b. On completion, decrement the in-degree of each successor.
   c. Any successor whose in-degree reaches zero enters the next ready-set.
5. If all rails have been scheduled, the execution is complete.  If not, a
   cycle exists and scheduling fails.

### Graph representation

The dependency graph is stored as a `dict[str, list[str]]` adjacency list
(node -> list of successors) plus a parallel `dict[str, int]` in-degree map.
This gives O(V+E) construction and traversal where V is the number of rails
and E is the number of dependency edges -- efficient even for large
configurations.

### Cycle detection

Cycles are detected **at configuration load time** (inside `RailsConfig`
validation).  If a cycle is found the application fails fast with a clear
error message naming the rails involved.  This prevents silent deadlocks at
request time.

### Backward compatibility

When no `depends_on` fields are declared:

* If `parallel: true`, every rail has in-degree zero and they all land in the
  first ready-set -- equivalent to today's all-parallel behaviour.
* If `parallel: false` (or omitted), the scheduler is bypassed and rails
  execute sequentially in declaration order, preserving the current default.

This means existing configurations require zero changes.

### Execution groups

Each iteration of the scheduler loop emits a **group** -- a set of rails that
can safely run at the same time.  The runtime dispatches each group with
`asyncio.gather()`.  This is a natural fit for the existing event-driven
architecture: each group produces events that subsequent groups may depend on.

### Thread-pool dispatch for CPU-bound rails

I/O-bound rails (external API calls to LLM endpoints, moderation services)
already benefit from `asyncio` concurrency.  CPU-bound rails (regex scanning,
PII detection, hashing) block the event loop under standard Python because of
the GIL.

The design introduces a `@cpu_bound` decorator.  A function decorated with
`@cpu_bound` is automatically dispatched to a `ThreadPoolExecutor` via
`loop.run_in_executor()`.  On free-threaded Python 3.14t (no-GIL builds) this
achieves true parallel CPU execution across cores.  On standard GIL builds the
thread pool still unblocks the event loop so that I/O-bound rails in the same
group are not starved.

## Alternatives Considered

### 1. Manual priority ordering

Each rail would receive a numeric priority; lower numbers run first, ties run
in parallel.

**Rejected** because:

* Priorities are a flat ordering and cannot express diamond dependencies
  (A -> B, A -> C, B -> D, C -> D).
* Renumbering is fragile when rails are added or removed.
* Users must manually compute a valid ordering, which is error-prone.

### 2. Event-driven dependency resolution

Rails would publish and subscribe to named events.  A rail starts when all
events it subscribes to have been published.

**Rejected** because:

* The indirection makes the execution order hard to reason about and debug.
* Colang already has an event system; layering a second event mechanism on top
  adds complexity with marginal benefit.
* There is no static validation -- a missing publisher is only detected at
  runtime.

### 3. Simple tier / layer system

Rails would be assigned to numbered tiers (0, 1, 2, ...).  All rails in tier
N run before tier N+1.

**Rejected** because:

* Tiers cannot represent arbitrary DAG structures.  If rail D depends on both
  B (tier 1) and C (tier 2), D must be in tier 3 even if B finished long ago.
* It degenerates to sequential execution when every rail is in a different
  tier, offering no incremental benefit over the current model.

## Consequences

### Positive

* **Reduced latency**: Independent rails execute concurrently while
  dependent rails respect ordering.  For a typical 4-rail configuration with
  two independent pairs, wall-clock latency is approximately halved.
* **Explicit dependencies**: The `depends_on` field makes data-flow
  relationships visible in configuration, aiding code review and debugging.
* **Fail-fast cycle detection**: Configuration errors are caught at load time,
  not at request time.
* **Backward compatible**: Existing `parallel: true/false` configurations
  continue to work unchanged.
* **CPU-bound rail support**: The `@cpu_bound` decorator plus ThreadPoolExecutor
  unblocks the event loop and enables true parallelism on free-threaded Python.

### Negative

* **Configuration complexity**: Users who need dependency-aware scheduling must
  learn the `depends_on` syntax and understand DAG semantics.
* **Debugging concurrency**: Parallel execution is inherently harder to debug
  than sequential.  Logging must clearly label which group and which rail
  produced each log line.
* **Thread-pool sizing**: The default thread-pool size must be tuned.  Too few
  threads under-utilise cores; too many waste memory.

### Risks

* **State leakage between rails**: If two rails in the same group both write
  the same context variable, the result is non-deterministic.  The
  documentation must clearly state that rails in the same execution group must
  not share mutable state.
* **Executor overhead on GIL builds**: On standard CPython, dispatching to
  threads adds scheduling overhead without true parallelism for CPU work.
  Benchmarks (`benchmarks/bench_threading.py`) confirm this is small (~5%)
  and acceptable.
