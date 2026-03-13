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

"""DAG-based rail scheduler for dependency-aware parallel execution.

Overall Architecture
--------------------
This module implements a three-stage pipeline for executing guardrail flows:

    1. **Dependency graph construction** — each rail (guardrail flow) is added as
       a node in a directed acyclic graph (DAG).  Edges encode "must run after"
       relationships (e.g. ``content_safety`` depends on ``pii_detection``).

    2. **Topological sort via Kahn's algorithm** — the DAG is partitioned into
       *execution groups* (also called *topological levels*).  All rails within
       the same group are independent of one another, so they may be executed
       concurrently.  Groups themselves are processed sequentially, guaranteeing
       that every rail's dependencies have completed before it starts.

    3. **Parallel execution with early cancellation** — each group's rails are
       dispatched as ``asyncio`` tasks.  If any rail signals a block (i.e. the
       input should be refused), the scheduler cancels the remaining tasks and
       short-circuits out, avoiding unnecessary work.

Key components:
    - RailDependencyGraph: DAG data structure with eager cycle detection.
    - TopologicalScheduler: Kahn's algorithm for execution-group generation,
      plus the async executor that honours parallelisation and early exit.
    - schedule_rails / build_scheduler_from_config: High-level API for
      integrating the scheduler with the NeMo Guardrails configuration system.

Example YAML configuration::

    rails:
      input:
        flows:
          - name: pii_detection
          - name: jailbreak_check
          - name: content_safety
            depends_on: [pii_detection]
          - name: final_gate
            depends_on: [jailbreak_check, content_safety]

In this example:
    - Group 0: pii_detection, jailbreak_check (run in parallel)
    - Group 1: content_safety (depends on pii_detection)
    - Group 2: final_gate (depends on jailbreak_check, content_safety)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, FrozenSet, List, Optional, Set

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Python version feature flags — used to gate optimisations that require
# specific interpreter versions.
# ---------------------------------------------------------------------------

_PY_VERSION = sys.version_info[:2]

# Python 3.12+ provides ``asyncio.eager_task_factory`` which lets coroutines
# that complete synchronously (e.g. cache hits) bypass the event-loop
# scheduling round-trip entirely.
_HAS_EAGER_TASK_FACTORY: bool = _PY_VERSION >= (3, 12)

# Free-threaded Python 3.13t / 3.14t builds disable the GIL, allowing true
# thread-level parallelism for CPU-bound rails.  We detect this via the
# ``Py_GIL_DISABLED`` sysconfig flag (PEP 703).
_IS_FREE_THREADED: bool = bool(getattr(sys, "_is_gil_enabled", lambda: True)() is False)

# Shared thread-pool for CPU-bound rail work.  On free-threaded builds we
# size it to the number of cores so that CPU-bound guardrail checks (regex,
# PII, YARA) can run in true parallel.  On standard (GIL) builds we still
# use a pool to avoid blocking the event loop, but the GIL limits actual
# CPU parallelism.
_CPU_POOL: Optional[ThreadPoolExecutor] = None

if _IS_FREE_THREADED:
    _cpu_count = os.cpu_count() or 4
    _CPU_POOL = ThreadPoolExecutor(
        max_workers=_cpu_count,
        thread_name_prefix="rail-cpu",
    )
    log.info(
        "Free-threaded Python detected — using %d-thread pool for CPU-bound rails",
        _cpu_count,
    )


# ---------------------------------------------------------------------------
# Exception types — raised at configuration time so that invalid dependency
# graphs are caught early, before any rail execution takes place.
# ---------------------------------------------------------------------------


class CyclicDependencyError(ValueError):
    """Raised when the rail dependency graph contains a cycle.

    A cycle means there is no valid execution order — rail A depends on B
    which depends on A (directly or transitively).  The offending node
    names are stored in ``cycle_nodes`` for diagnostic purposes.
    """

    def __init__(self, cycle_nodes: Set[str]):
        self.cycle_nodes = cycle_nodes
        # Sort for deterministic error messages in logs and tests.
        cycle_str = " -> ".join(sorted(cycle_nodes))
        super().__init__(
            f"Cyclic dependency detected among rails: {cycle_str}. "
            f"Rail dependencies must form a DAG (directed acyclic graph)."
        )


class UnknownRailError(ValueError):
    """Raised when a dependency references a rail not in the graph.

    This catches typos and misconfiguration at graph-building time,
    rather than letting the scheduler silently skip a missing rail.
    """

    def __init__(self, rail: str, dependency: str):
        self.rail = rail
        self.dependency = dependency
        super().__init__(
            f"Rail '{rail}' depends on unknown rail '{dependency}'. "
            f"Ensure all referenced rails are defined in the flows list."
        )


# ---------------------------------------------------------------------------
# Graph node — lightweight value object representing a single rail.
# ---------------------------------------------------------------------------


@dataclass
class RailNode:
    """A node in the rail dependency graph.

    Each node corresponds to a single guardrail flow.  It records the
    flow's name, the set of rails it must wait for (``depends_on``), and
    any optional metadata carried over from the YAML configuration.

    Attributes:
        name: Unique identifier for this rail (flow name).
        depends_on: Immutable set of rail names this rail depends on.
        metadata: Optional metadata dict (e.g., cpu_bound, timeout).
    """

    name: str
    depends_on: FrozenSet[str] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hashing and equality are based solely on the rail name, so that
    # nodes can be stored in sets and used as dictionary keys.
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RailNode):
            return self.name == other.name
        return NotImplemented


# ---------------------------------------------------------------------------
# ExecutionGroup — an immutable bundle of rails that share the same
# topological level and can therefore be executed concurrently.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionGroup:
    """A group of rails that can be executed concurrently.

    All rails in a group have their dependencies satisfied by
    previously completed groups.  The ``index`` field records the
    topological level: group 0 contains rails with no dependencies,
    group 1 contains rails whose dependencies are all in group 0,
    and so on.

    The scheduler iterates over groups in index order, executing all
    rails within a group as parallel ``asyncio`` tasks before moving
    on to the next group.

    Attributes:
        index: Topological level (0 = no dependencies).
        rails: Frozenset of rail names in this group.
    """

    index: int
    rails: FrozenSet[str]

    def __len__(self) -> int:
        """Return the number of rails in this group (supports ``len()``)."""
        return len(self.rails)

    def __iter__(self):
        """Iterate over rail names in this group."""
        return iter(self.rails)


# ---------------------------------------------------------------------------
# RailDependencyGraph — the core DAG data structure.
# ---------------------------------------------------------------------------


class RailDependencyGraph:
    """Directed acyclic graph representing rail dependencies.

    The graph uses a *dual adjacency-list* representation:
      - ``_forward[A]`` = set of rails that depend on A  (A's dependents).
      - ``_reverse[A]`` = set of rails that A depends on (A's prerequisites).

    This dual representation allows O(1) look-ups in both directions and
    enables Kahn's algorithm to run in O(V + E) time.

    Cycle detection is performed eagerly on every edge addition so that
    invalid configurations are caught immediately at graph-building time.

    Usage::

        graph = RailDependencyGraph()
        graph.add_rail("pii_detection")
        graph.add_rail("jailbreak_check")
        graph.add_rail("content_safety", depends_on=["pii_detection"])
        graph.add_rail("final_gate", depends_on=["jailbreak_check", "content_safety"])

        groups = graph.compute_execution_groups()
        # [ExecutionGroup(0, {"pii_detection", "jailbreak_check"}),
        #  ExecutionGroup(1, {"content_safety"}),
        #  ExecutionGroup(2, {"final_gate"})]
    """

    def __init__(self) -> None:
        # Forward adjacency: node -> set of nodes that depend on it.
        # Used during Kahn's algorithm to propagate "this node is done"
        # signals to its dependents.
        self._forward: Dict[str, Set[str]] = defaultdict(set)

        # Reverse adjacency: node -> set of nodes it depends on.
        # The length of each set gives the node's in-degree, which is the
        # key metric in Kahn's algorithm.
        self._reverse: Dict[str, Set[str]] = defaultdict(set)

        # Canonical registry of all rail nodes, keyed by name.
        self._nodes: Dict[str, RailNode] = {}

    @property
    def nodes(self) -> Dict[str, RailNode]:
        """Return a shallow copy of all registered rail nodes."""
        return dict(self._nodes)

    @property
    def node_count(self) -> int:
        """Number of rails in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of dependency edges (total across all nodes)."""
        return sum(len(deps) for deps in self._forward.values())

    def add_rail(
        self,
        name: str,
        depends_on: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RailNode:
        """Add a rail to the dependency graph.

        Args:
            name: Unique rail identifier (flow name).
            depends_on: List of rail names this rail depends on.
            metadata: Optional metadata (e.g., cpu_bound, timeout_ms).

        Returns:
            The created RailNode.

        Raises:
            UnknownRailError: If a dependency references an unregistered rail.
            CyclicDependencyError: If adding this rail creates a cycle.
        """
        deps = frozenset(depends_on or [])

        # Validate that every dependency has already been registered.
        # This enforces a declaration-before-use discipline.
        for dep in deps:
            if dep not in self._nodes and dep != name:
                raise UnknownRailError(name, dep)

        # A rail depending on itself is the simplest possible cycle.
        if name in deps:
            raise CyclicDependencyError({name})

        node = RailNode(name=name, depends_on=deps, metadata=metadata or {})
        self._nodes[name] = node

        # Ensure the node has entries in both adjacency maps even if it
        # has no edges yet — this simplifies iteration in Kahn's algorithm.
        if name not in self._forward:
            self._forward[name] = set()
        if name not in self._reverse:
            self._reverse[name] = set()

        # Add directed edges: for each dependency ``dep``, create an edge
        # dep -> name (meaning ``name`` depends on ``dep``).  The forward
        # map records dependents; the reverse map records prerequisites.
        for dep in deps:
            self._forward[dep].add(name)
            self._reverse[name].add(dep)

        # If any edges were added, re-run cycle detection across the
        # entire graph.  This is intentionally eager so that invalid
        # configurations fail fast.
        if deps:
            self._detect_cycles()

        return node

    def _detect_cycles(self) -> None:
        """Detect cycles using Kahn's algorithm.

        Kahn's algorithm works by repeatedly removing nodes with in-degree
        zero (i.e. no unsatisfied dependencies).  If the algorithm
        terminates before visiting every node, the unvisited nodes must
        be part of one or more cycles — because each of those nodes still
        has at least one predecessor that also could not be removed.

        Raises:
            CyclicDependencyError: If the graph contains a cycle.  The
                ``cycle_nodes`` attribute lists all nodes involved.
        """
        # Initialise in-degree counts for every node.
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node, deps in self._reverse.items():
            if node in in_degree:
                in_degree[node] = len(deps)

        # Seed the queue with all nodes that have no prerequisites.
        queue = deque(n for n, d in in_degree.items() if d == 0)
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            # For every dependent of the current node, decrement its
            # in-degree.  When in-degree reaches zero, all of its
            # prerequisites have been processed and it can be enqueued.
            for neighbor in self._forward.get(node, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # If we could not visit all nodes, the remainder form cycles.
        if visited < len(self._nodes):
            cycle_nodes = {n for n, d in in_degree.items() if d > 0}
            raise CyclicDependencyError(cycle_nodes)

    def compute_execution_groups(self) -> List[ExecutionGroup]:
        """Compute execution groups via topological sort (Kahn's algorithm).

        This method performs a *level-by-level* (breadth-first) variant of
        Kahn's algorithm.  Instead of emitting one node at a time, it
        collects all nodes whose in-degree reaches zero at the same level
        into a single ``ExecutionGroup``.

        The result is the minimum number of sequential steps needed to
        execute every rail whilst respecting all dependency constraints.
        Rails within each group can be run concurrently because they are
        guaranteed to have no mutual dependencies.

        Returns:
            Ordered list of ExecutionGroups.  Group 0 contains rails with
            no dependencies; group N+1 contains rails whose dependencies
            are all satisfied by groups 0..N.
        """
        if not self._nodes:
            return []

        # Build the initial in-degree map from the reverse adjacency list.
        in_degree: Dict[str, int] = {}
        for name in self._nodes:
            in_degree[name] = len(self._reverse.get(name, set()))

        # Seed level 0 with all zero-in-degree nodes (no prerequisites).
        current_level = [n for n, d in in_degree.items() if d == 0]
        groups: List[ExecutionGroup] = []
        level = 0

        while current_level:
            # Freeze the current level into an ExecutionGroup.
            group = ExecutionGroup(index=level, rails=frozenset(current_level))
            groups.append(group)

            # Determine the next level: for every node we are "removing"
            # in this level, decrement the in-degree of its dependents.
            # Any dependent whose in-degree drops to zero is ready to run.
            next_level = []
            for node in current_level:
                for neighbor in self._forward.get(node, set()):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)

            current_level = next_level
            level += 1

        return groups

    def get_dependencies(self, rail_name: str) -> FrozenSet[str]:
        """Get direct dependencies (prerequisites) of a rail."""
        if rail_name not in self._nodes:
            return frozenset()
        return self._nodes[rail_name].depends_on

    def get_dependents(self, rail_name: str) -> FrozenSet[str]:
        """Get rails that directly depend on the given rail."""
        return frozenset(self._forward.get(rail_name, set()))

    @classmethod
    def from_flow_config(cls, flows: List[Any]) -> "RailDependencyGraph":
        """Build a dependency graph from a rails flow configuration.

        This factory method supports the three flow-configuration formats
        used across NeMo Guardrails:

        1. **Simple string**: ``"flow_name"`` — a rail with no dependencies.
        2. **Dictionary**: ``{"name": "flow_name", "depends_on": [...]}``
           — explicit dependency list.
        3. **Pydantic model**: any object with a ``.name`` attribute and an
           optional ``.depends_on`` attribute (e.g. ``FlowWithDeps``).

        The method uses a two-pass approach:
          * **First pass** — register every rail name so that dependency
            validation in the second pass can verify that all referenced
            rails actually exist.
          * **Second pass** — wire up dependency edges and attach metadata.

        Args:
            flows: List of flow names (str) or flow config dicts.

        Returns:
            Populated RailDependencyGraph.
        """
        graph = cls()

        # ------------------------------------------------------------------
        # First pass: collect all rail names and normalise each flow entry
        # into a uniform dict format.  This ensures that every rail is
        # registered before we try to validate dependencies.
        # ------------------------------------------------------------------
        rail_names = []
        rail_configs = []

        for flow in flows:
            if isinstance(flow, str):
                # Format 1: bare string — no dependencies.
                rail_names.append(flow)
                rail_configs.append({"name": flow})
            elif isinstance(flow, dict):
                # Format 2: configuration dictionary.
                name = flow.get("name", flow.get("flow_name", ""))
                rail_names.append(name)
                rail_configs.append(flow)
            elif hasattr(flow, "name"):
                # Format 3: Pydantic model (e.g. FlowWithDeps).
                name = flow.name
                deps = list(getattr(flow, "depends_on", []))
                rail_names.append(name)
                rail_configs.append({"name": name, "depends_on": deps})
            else:
                log.warning("Unexpected flow config type: %s", type(flow))

        # Register all rails *without* edges first.  This populates both
        # adjacency maps with empty sets and creates placeholder RailNode
        # objects so that the second pass can reference any rail by name.
        for name in rail_names:
            if name not in graph._nodes:
                graph._nodes[name] = RailNode(name=name)
                graph._forward[name] = set()
                graph._reverse[name] = set()

        # ------------------------------------------------------------------
        # Second pass: add dependency edges and attach metadata.
        # ------------------------------------------------------------------
        for config in rail_configs:
            name = config.get("name", config.get("flow_name", ""))
            depends_on = config.get("depends_on", [])
            # Collect any extra keys as metadata (everything that is not
            # the rail's name or its dependency list).
            metadata = {k: v for k, v in config.items() if k not in ("name", "flow_name", "depends_on")}

            if depends_on:
                node = graph._nodes[name]
                deps = frozenset(depends_on)

                # Validate that every dependency references a known rail.
                for dep in deps:
                    if dep not in graph._nodes:
                        raise UnknownRailError(name, dep)

                # Replace the placeholder node with the fully populated one.
                graph._nodes[name] = RailNode(name=name, depends_on=deps, metadata=metadata)

                # Wire up forward and reverse adjacency edges.
                for dep in deps:
                    graph._forward[dep].add(name)
                    graph._reverse[name].add(dep)

            elif metadata:
                # No dependencies, but metadata was supplied — preserve it
                # on the existing node without altering its edges.
                graph._nodes[name] = RailNode(
                    name=name,
                    depends_on=graph._nodes[name].depends_on,
                    metadata=metadata,
                )

        # Run a final cycle check across the whole graph.  This catches
        # cycles that span multiple rails added in the second pass (the
        # incremental check in ``add_rail`` is not used here because we
        # batch-insert edges for efficiency).
        if graph.edge_count > 0:
            graph._detect_cycles()

        return graph

    def __repr__(self) -> str:
        return f"RailDependencyGraph(nodes={self.node_count}, edges={self.edge_count})"


# ---------------------------------------------------------------------------
# TopologicalScheduler — the async executor that walks the execution groups
# and dispatches rails as asyncio tasks with optional early cancellation.
# ---------------------------------------------------------------------------


class TopologicalScheduler:
    """Scheduler that executes rails in topological order.

    The scheduler pre-computes execution groups from the dependency graph
    at initialisation time.  At execution time it iterates over groups
    sequentially; within each group, independent rails are dispatched as
    concurrent ``asyncio`` tasks via ``asyncio.gather`` (or a custom
    early-exit loop).

    This design maximises parallelisation whilst guaranteeing that every
    rail's prerequisites have finished before it begins.

    Usage::

        graph = RailDependencyGraph.from_flow_config(config.rails.input.flows)
        scheduler = TopologicalScheduler(graph)

        results = await scheduler.execute(
            rail_executor=my_rail_runner,
            context=rail_context,
        )

    Args:
        graph: The rail dependency graph.
        timeout_per_group: Optional timeout in seconds for each execution group.
    """

    def __init__(
        self,
        graph: RailDependencyGraph,
        timeout_per_group: Optional[float] = None,
    ) -> None:
        self._graph = graph
        self._timeout = timeout_per_group
        # Pre-compute execution groups once at construction time so that
        # repeated calls to ``execute`` do not re-run the topological sort.
        self._groups = graph.compute_execution_groups()

    @property
    def groups(self) -> List[ExecutionGroup]:
        """Return a copy of the computed execution groups."""
        return list(self._groups)

    @property
    def num_groups(self) -> int:
        """Number of sequential execution steps (topological levels)."""
        return len(self._groups)

    @property
    def max_parallelism(self) -> int:
        """Maximum number of rails that can run concurrently.

        This equals the size of the largest execution group and gives an
        upper bound on the degree of parallelisation.
        """
        if not self._groups:
            return 0
        return max(len(g) for g in self._groups)

    async def execute(
        self,
        rail_executor: Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]],
        context: Optional[Dict[str, Any]] = None,
        early_exit_on_block: bool = True,
    ) -> Dict[str, Any]:
        """Execute all rails in topological order.

        The method walks through execution groups sequentially.  Within
        each group it either:
          * runs a single rail directly (avoiding ``gather`` overhead), or
          * dispatches multiple rails concurrently as ``asyncio`` tasks.

        When ``early_exit_on_block`` is enabled, the scheduler monitors
        task results as they complete and cancels all remaining tasks in
        the current group (and skips subsequent groups) as soon as any
        rail returns a blocking result.  This avoids wasted computation
        when a guardrail has already decided to refuse the input.

        Args:
            rail_executor: Async callable(rail_name, context) -> result.
                Must return a dict with at least {"action": "continue"|"block"}.
            context: Shared context dict passed to each rail.
            early_exit_on_block: If True, stop execution when any rail
                returns action="block". Remaining rails are cancelled.

        Returns:
            Dict with:
                - "results": Dict[str, Any] mapping rail names to results
                - "blocked_by": Optional rail name that triggered a block
                - "execution_order": List of lists showing actual execution
                - "elapsed_ms": Total execution time in milliseconds
        """
        ctx = context or {}
        results: Dict[str, Any] = {}
        execution_order: List[List[str]] = []
        blocked_by: Optional[str] = None
        start = time.monotonic()

        # -----------------------------------------------------------------
        # Python 3.12+ optimisation: install the eager task factory for the
        # duration of this execute() call.  Eager tasks start executing
        # immediately when created (before the next event-loop tick), so
        # coroutines that complete synchronously — e.g. cache hits or
        # trivially fast checks — never pay the scheduling round-trip.
        # -----------------------------------------------------------------
        loop = asyncio.get_running_loop()
        previous_task_factory = None
        if _HAS_EAGER_TASK_FACTORY:
            previous_task_factory = loop.get_task_factory()  # type: ignore[attr-defined]
            loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]

        try:
            for group in self._groups:
                group_rails = list(group.rails)
                # Record which rails were attempted in this group, useful for
                # debugging and observability.
                execution_order.append(group_rails)

                if len(group_rails) == 1:
                    # -------------------------------------------------------
                    # Fast path: single rail in the group — execute it
                    # directly without the overhead of creating asyncio tasks
                    # and calling gather/wait.
                    # -------------------------------------------------------
                    rail_name = group_rails[0]
                    try:
                        result = await self._execute_with_timeout(rail_executor, rail_name, ctx)
                        results[rail_name] = result

                        # Check whether this rail's result blocks further
                        # execution (e.g. the input was refused).
                        if early_exit_on_block and self._is_block(result):
                            blocked_by = rail_name
                            log.info("Rail '%s' blocked — stopping execution", rail_name)
                            break
                    except Exception as e:
                        log.error("Rail '%s' failed: %s", rail_name, e)
                        results[rail_name] = {
                            "action": "error",
                            "error": str(e),
                        }
                else:
                    # -------------------------------------------------------
                    # Parallel path: multiple independent rails in this group.
                    # Create an asyncio.Task for each rail so they run
                    # concurrently on the event loop.
                    # -------------------------------------------------------
                    tasks = {
                        rail_name: asyncio.create_task(
                            self._execute_with_timeout(rail_executor, rail_name, ctx),
                            name=f"rail-{rail_name}",
                        )
                        for rail_name in group_rails
                    }

                    try:
                        if early_exit_on_block:
                            # Use the custom early-exit gatherer which watches
                            # tasks as they complete and cancels the rest if
                            # a blocking result is observed.
                            blocked_by = await self._gather_with_early_exit(tasks, results)
                            if blocked_by:
                                break
                        else:
                            # No early exit — simply wait for all tasks to
                            # finish, collecting exceptions as values rather
                            # than raising them (return_exceptions=True).
                            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
                            for rail_name, result in zip(tasks.keys(), gathered):
                                if isinstance(result, Exception):
                                    results[rail_name] = {
                                        "action": "error",
                                        "error": str(result),
                                    }
                                else:
                                    results[rail_name] = result
                    finally:
                        # Defensive cleanup: cancel any tasks that are still
                        # pending (e.g. due to an unexpected exception in the
                        # loop above).  We await each cancelled task to ensure
                        # it has fully terminated before moving on.
                        for task in tasks.values():
                            if not task.done():
                                task.cancel()
                                try:
                                    await task
                                except (asyncio.CancelledError, Exception):
                                    pass
        finally:
            # Restore the previous task factory so we don't leak state.
            if _HAS_EAGER_TASK_FACTORY:
                loop.set_task_factory(previous_task_factory)  # type: ignore[attr-defined]

        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "results": results,
            "blocked_by": blocked_by,
            "execution_order": execution_order,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    async def _execute_with_timeout(
        self,
        rail_executor: Callable,
        rail_name: str,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single rail, optionally enforcing a timeout.

        If ``self._timeout`` is set, the coroutine is wrapped in
        ``asyncio.wait_for`` which raises ``asyncio.TimeoutError`` if the
        rail does not complete in time.
        """
        if self._timeout:
            return await asyncio.wait_for(
                rail_executor(rail_name, context),
                timeout=self._timeout,
            )
        return await rail_executor(rail_name, context)

    async def _gather_with_early_exit(
        self,
        tasks: Dict[str, asyncio.Task],
        results: Dict[str, Any],
    ) -> Optional[str]:
        """Run tasks concurrently, stopping on first block.

        Unlike ``asyncio.gather``, this method processes tasks *as they
        complete* (using ``asyncio.wait`` with ``FIRST_COMPLETED``).
        After each completion it inspects the result; if the result is a
        blocking action, it cancels all still-pending tasks and returns
        immediately.

        The ``task_to_rail`` reverse mapping is necessary because
        ``asyncio.wait`` returns generic ``Task`` objects — we need a way
        to map each finished task back to the rail name that spawned it
        so we can store the result under the correct key in ``results``
        and report which rail caused the block.

        Args:
            tasks: Mapping of rail name -> asyncio.Task.
            results: Mutable dict that receives rail_name -> result entries.

        Returns:
            The name of the blocking rail, or ``None`` if all tasks
            completed without blocking.
        """
        # Build a reverse lookup: Task object -> rail name.  This is
        # required because ``asyncio.wait`` returns a set of Task objects
        # with no reference to the rail names we originally associated
        # with them.
        task_to_rail = {task: name for name, task in tasks.items()}
        pending = set(tasks.values())

        while pending:
            # Wait until at least one task finishes.  ``FIRST_COMPLETED``
            # allows us to inspect results incrementally rather than
            # waiting for the entire group.
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                # Look up which rail this task corresponds to.
                rail_name = task_to_rail[task]

                try:
                    result = task.result()
                    results[rail_name] = result

                    # If this rail blocked, cancel every remaining task
                    # in the group to avoid unnecessary work.
                    if self._is_block(result):
                        for p in pending:
                            p.cancel()
                        return rail_name

                except Exception as e:
                    log.error("Rail '%s' failed: %s", rail_name, e)
                    results[rail_name] = {
                        "action": "error",
                        "error": str(e),
                    }

        # All tasks completed without any blocking result.
        return None

    @staticmethod
    def _is_block(result: Any) -> bool:
        """Check whether a rail result indicates a blocking outcome.

        A blocking result means the guardrail has decided to refuse or
        stop processing.  The method recognises three action values:
          - ``"block"``   — the input is explicitly blocked.
          - ``"stop"``    — processing should be halted.
          - ``"refused"`` — the request has been refused.

        Any other action value (including ``"continue"``) is considered
        non-blocking.

        Args:
            result: The value returned by the rail executor.

        Returns:
            ``True`` if the result signals that execution should be
            halted; ``False`` otherwise.
        """
        if isinstance(result, dict):
            action = result.get("action", "")
            return action in ("block", "stop", "refused")
        return False


# ---------------------------------------------------------------------------
# Public convenience functions — thin wrappers that build the graph and
# scheduler in one step, suitable for use by the NeMo Guardrails runtime.
# ---------------------------------------------------------------------------


def build_scheduler_from_config(
    flows: List[Any],
    timeout_per_group: Optional[float] = None,
) -> TopologicalScheduler:
    """Build a TopologicalScheduler from a rails flow configuration.

    This is the main entry point for integrating the DAG scheduler
    with the existing NeMo Guardrails configuration system.

    Args:
        flows: List of flow names (str) or flow config dicts with
            optional "depends_on" fields.
        timeout_per_group: Optional timeout per execution group.

    Returns:
        A configured TopologicalScheduler.

    Raises:
        CyclicDependencyError: If flows contain circular dependencies.
        UnknownRailError: If a dependency references an undefined flow.
    """
    graph = RailDependencyGraph.from_flow_config(flows)
    return TopologicalScheduler(graph, timeout_per_group=timeout_per_group)


def get_cpu_executor() -> Optional[ThreadPoolExecutor]:
    """Return the shared CPU-bound thread pool, if available.

    On free-threaded Python (3.13t / 3.14t) this returns a
    ``ThreadPoolExecutor`` sized to the CPU count, enabling true
    parallel execution of CPU-bound guardrail checks (regex matching,
    PII detection, YARA scanning, etc.).

    On standard GIL-enabled Python this returns ``None`` — callers
    should fall back to ``loop.run_in_executor(None, ...)`` which uses
    the default thread pool.
    """
    return _CPU_POOL


def has_dependencies(flows: List[Any]) -> bool:
    """Check if any flow in the config declares dependencies.

    This serves as a fast-path check so that the runtime can skip DAG
    construction entirely when no dependencies are declared.  This
    preserves backward compatibility with configurations that predate
    the dependency-aware scheduler — those flows are simply executed
    in their original declaration order.

    Args:
        flows: List of flow names (str) or flow config dicts.

    Returns:
        True if any flow has a "depends_on" field with a non-empty value.
    """
    for flow in flows:
        if isinstance(flow, dict) and flow.get("depends_on"):
            return True
        if getattr(flow, "depends_on", None):
            return True
    return False
