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

This module implements a directed acyclic graph (DAG) scheduler that enables
rails to declare dependencies on other rails and be executed in topological
order. Rails within the same execution group (no mutual dependencies) run
concurrently via asyncio, while respecting the ordering constraints defined
by the dependency graph.

Key components:
    - RailDependencyGraph: DAG data structure with cycle detection
    - TopologicalScheduler: Kahn's algorithm for execution group generation
    - schedule_rails: High-level API for scheduling rail execution

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
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, FrozenSet, List, Optional, Set

log = logging.getLogger(__name__)


class CyclicDependencyError(ValueError):
    """Raised when the rail dependency graph contains a cycle."""

    def __init__(self, cycle_nodes: Set[str]):
        self.cycle_nodes = cycle_nodes
        cycle_str = " -> ".join(sorted(cycle_nodes))
        super().__init__(
            f"Cyclic dependency detected among rails: {cycle_str}. "
            f"Rail dependencies must form a DAG (directed acyclic graph)."
        )


class UnknownRailError(ValueError):
    """Raised when a dependency references a rail not in the graph."""

    def __init__(self, rail: str, dependency: str):
        self.rail = rail
        self.dependency = dependency
        super().__init__(
            f"Rail '{rail}' depends on unknown rail '{dependency}'. "
            f"Ensure all referenced rails are defined in the flows list."
        )


@dataclass
class RailNode:
    """A node in the rail dependency graph.

    Attributes:
        name: Unique identifier for this rail (flow name).
        depends_on: Set of rail names this rail depends on.
        metadata: Optional metadata dict (e.g., cpu_bound, timeout).
    """

    name: str
    depends_on: FrozenSet[str] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RailNode):
            return self.name == other.name
        return NotImplemented


@dataclass(frozen=True)
class ExecutionGroup:
    """A group of rails that can be executed concurrently.

    All rails in a group have their dependencies satisfied by
    previously completed groups.

    Attributes:
        index: Topological level (0 = no dependencies).
        rails: Frozenset of rail names in this group.
    """

    index: int
    rails: FrozenSet[str]

    def __len__(self) -> int:
        return len(self.rails)

    def __iter__(self):
        return iter(self.rails)


class RailDependencyGraph:
    """Directed acyclic graph representing rail dependencies.

    The graph uses an adjacency-list representation for O(V+E) traversal.
    Cycle detection is performed eagerly on every edge addition to fail
    fast at configuration time.

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
        # Forward adjacency: node -> set of nodes that depend on it
        self._forward: Dict[str, Set[str]] = defaultdict(set)
        # Reverse adjacency: node -> set of nodes it depends on
        self._reverse: Dict[str, Set[str]] = defaultdict(set)
        # All registered rail nodes
        self._nodes: Dict[str, RailNode] = {}

    @property
    def nodes(self) -> Dict[str, RailNode]:
        """All registered rail nodes."""
        return dict(self._nodes)

    @property
    def node_count(self) -> int:
        """Number of rails in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of dependency edges."""
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

        # Validate all dependencies exist
        for dep in deps:
            if dep not in self._nodes and dep != name:
                raise UnknownRailError(name, dep)

        # Self-dependency check
        if name in deps:
            raise CyclicDependencyError({name})

        node = RailNode(name=name, depends_on=deps, metadata=metadata or {})
        self._nodes[name] = node

        # Ensure the node appears in adjacency lists
        if name not in self._forward:
            self._forward[name] = set()
        if name not in self._reverse:
            self._reverse[name] = set()

        # Add edges: for each dependency, add forward edge dep -> name
        for dep in deps:
            self._forward[dep].add(name)
            self._reverse[name].add(dep)

        # Check for cycles after adding edges
        if deps:
            self._detect_cycles()

        return node

    def _detect_cycles(self) -> None:
        """Detect cycles using Kahn's algorithm.

        If the topological sort cannot visit all nodes, the remaining
        nodes form one or more cycles.

        Raises:
            CyclicDependencyError: If the graph contains a cycle.
        """
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node, deps in self._reverse.items():
            if node in in_degree:
                in_degree[node] = len(deps)

        queue = deque(n for n, d in in_degree.items() if d == 0)
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in self._forward.get(node, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        if visited < len(self._nodes):
            cycle_nodes = {n for n, d in in_degree.items() if d > 0}
            raise CyclicDependencyError(cycle_nodes)

    def compute_execution_groups(self) -> List[ExecutionGroup]:
        """Compute execution groups via topological sort (Kahn's algorithm).

        Rails with no dependencies form group 0. Rails whose dependencies
        are all in group N form group N+1. This is a level-based BFS that
        produces the minimum number of sequential steps.

        Returns:
            Ordered list of ExecutionGroups. Rails within each group
            can be executed concurrently.
        """
        if not self._nodes:
            return []

        in_degree: Dict[str, int] = {}
        for name in self._nodes:
            in_degree[name] = len(self._reverse.get(name, set()))

        # Start with all zero-degree nodes
        current_level = [n for n, d in in_degree.items() if d == 0]
        groups: List[ExecutionGroup] = []
        level = 0

        while current_level:
            group = ExecutionGroup(index=level, rails=frozenset(current_level))
            groups.append(group)

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
        """Get direct dependencies of a rail."""
        if rail_name not in self._nodes:
            return frozenset()
        return self._nodes[rail_name].depends_on

    def get_dependents(self, rail_name: str) -> FrozenSet[str]:
        """Get rails that directly depend on the given rail."""
        return frozenset(self._forward.get(rail_name, set()))

    @classmethod
    def from_flow_config(cls, flows: List[Any]) -> "RailDependencyGraph":
        """Build a dependency graph from a rails flow configuration.

        Supports two flow config formats:
        1. Simple string: "flow_name" (no dependencies)
        2. Dict with depends_on: {"name": "flow_name", "depends_on": [...]}

        Args:
            flows: List of flow names (str) or flow config dicts.

        Returns:
            Populated RailDependencyGraph.
        """
        graph = cls()

        # First pass: register all rails (needed for dependency validation)
        rail_names = []
        rail_configs = []

        for flow in flows:
            if isinstance(flow, str):
                rail_names.append(flow)
                rail_configs.append({"name": flow})
            elif isinstance(flow, dict):
                name = flow.get("name", flow.get("flow_name", ""))
                rail_names.append(name)
                rail_configs.append(flow)
            elif hasattr(flow, "name"):
                # FlowWithDeps or similar pydantic model
                name = flow.name
                deps = list(getattr(flow, "depends_on", []))
                rail_names.append(name)
                rail_configs.append({"name": name, "depends_on": deps})
            else:
                log.warning("Unexpected flow config type: %s", type(flow))

        # Register all rails without dependencies first
        for name in rail_names:
            if name not in graph._nodes:
                graph._nodes[name] = RailNode(name=name)
                graph._forward[name] = set()
                graph._reverse[name] = set()

        # Second pass: add dependency edges
        for config in rail_configs:
            name = config.get("name", config.get("flow_name", ""))
            depends_on = config.get("depends_on", [])
            metadata = {k: v for k, v in config.items() if k not in ("name", "flow_name", "depends_on")}

            if depends_on:
                node = graph._nodes[name]
                deps = frozenset(depends_on)

                # Validate dependencies
                for dep in deps:
                    if dep not in graph._nodes:
                        raise UnknownRailError(name, dep)

                # Update node
                graph._nodes[name] = RailNode(name=name, depends_on=deps, metadata=metadata)

                # Add edges
                for dep in deps:
                    graph._forward[dep].add(name)
                    graph._reverse[name].add(dep)

            elif metadata:
                graph._nodes[name] = RailNode(
                    name=name,
                    depends_on=graph._nodes[name].depends_on,
                    metadata=metadata,
                )

        # Final cycle check
        if graph.edge_count > 0:
            graph._detect_cycles()

        return graph

    def __repr__(self) -> str:
        return f"RailDependencyGraph(nodes={self.node_count}, edges={self.edge_count})"


class TopologicalScheduler:
    """Scheduler that executes rails in topological order.

    Groups of independent rails run concurrently via asyncio.gather().
    Groups are executed sequentially in dependency order.

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
        self._groups = graph.compute_execution_groups()

    @property
    def groups(self) -> List[ExecutionGroup]:
        """The computed execution groups."""
        return list(self._groups)

    @property
    def num_groups(self) -> int:
        """Number of sequential execution steps."""
        return len(self._groups)

    @property
    def max_parallelism(self) -> int:
        """Maximum number of rails that can run concurrently."""
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

        for group in self._groups:
            group_rails = list(group.rails)
            execution_order.append(group_rails)

            if len(group_rails) == 1:
                # Single rail — no need for gather overhead
                rail_name = group_rails[0]
                try:
                    result = await self._execute_with_timeout(rail_executor, rail_name, ctx)
                    results[rail_name] = result

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
                # Multiple rails — run in parallel
                tasks = {
                    rail_name: asyncio.create_task(
                        self._execute_with_timeout(rail_executor, rail_name, ctx),
                        name=f"rail-{rail_name}",
                    )
                    for rail_name in group_rails
                }

                try:
                    if early_exit_on_block:
                        # Process as they complete for early exit
                        blocked_by = await self._gather_with_early_exit(tasks, results)
                        if blocked_by:
                            break
                    else:
                        # Wait for all to complete
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
                    # Cancel any pending tasks
                    for task in tasks.values():
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, Exception):
                                pass

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
        """Execute a single rail with optional timeout."""
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

        Returns the name of the blocking rail, or None.
        """
        task_to_rail = {task: name for name, task in tasks.items()}
        pending = set(tasks.values())

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                rail_name = task_to_rail[task]

                try:
                    result = task.result()
                    results[rail_name] = result

                    if self._is_block(result):
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        return rail_name

                except Exception as e:
                    log.error("Rail '%s' failed: %s", rail_name, e)
                    results[rail_name] = {
                        "action": "error",
                        "error": str(e),
                    }

        return None

    @staticmethod
    def _is_block(result: Any) -> bool:
        """Check if a rail result indicates blocking."""
        if isinstance(result, dict):
            action = result.get("action", "")
            return action in ("block", "stop", "refused")
        return False


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


def has_dependencies(flows: List[Any]) -> bool:
    """Check if any flow in the config declares dependencies.

    This is used as a fast path to skip DAG construction when no
    dependencies are declared (backward compatibility).

    Args:
        flows: List of flow names (str) or flow config dicts.

    Returns:
        True if any flow has a "depends_on" field.
    """
    for flow in flows:
        if isinstance(flow, dict) and flow.get("depends_on"):
            return True
        if getattr(flow, "depends_on", None):
            return True
    return False
