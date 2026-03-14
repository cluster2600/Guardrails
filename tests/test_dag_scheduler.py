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

"""Tests for nemoguardrails.rails.llm.dag_scheduler.

Covers:
  - RailDependencyGraph: construction, cycle detection, edge cases
  - TopologicalScheduler: execution group computation and execution
  - build_scheduler_from_config: factory function
  - has_dependencies: fast-path check
  - Integration with FlowWithDeps config model
"""

import asyncio

import pytest

from nemoguardrails.rails.llm.dag_scheduler import (
    CyclicDependencyError,
    RailDependencyGraph,
    UnknownRailError,
    build_scheduler_from_config,
    has_dependencies,
)

# ---------------------------------------------------------------------------
# RailDependencyGraph tests
# ---------------------------------------------------------------------------


class TestRailDependencyGraph:
    """Tests for the DAG data structure."""

    def test_empty_graph(self):
        g = RailDependencyGraph()
        assert g.node_count == 0
        assert g.edge_count == 0
        assert g.compute_execution_groups() == []

    def test_single_node(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        assert g.node_count == 1
        assert g.edge_count == 0
        groups = g.compute_execution_groups()
        assert len(groups) == 1
        assert "a" in groups[0].rails

    def test_two_independent_nodes(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("b")
        groups = g.compute_execution_groups()
        assert len(groups) == 1
        assert groups[0].rails == frozenset({"a", "b"})

    def test_linear_chain(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("b", depends_on=["a"])
        g.add_rail("c", depends_on=["b"])
        groups = g.compute_execution_groups()
        assert len(groups) == 3
        assert groups[0].rails == frozenset({"a"})
        assert groups[1].rails == frozenset({"b"})
        assert groups[2].rails == frozenset({"c"})

    def test_diamond_dependency(self):
        """A -> B, A -> C, B -> D, C -> D."""
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("b", depends_on=["a"])
        g.add_rail("c", depends_on=["a"])
        g.add_rail("d", depends_on=["b", "c"])
        groups = g.compute_execution_groups()
        assert len(groups) == 3
        assert groups[0].rails == frozenset({"a"})
        assert groups[1].rails == frozenset({"b", "c"})
        assert groups[2].rails == frozenset({"d"})

    def test_cycle_detection_direct(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        with pytest.raises(CyclicDependencyError):
            g.add_rail("b", depends_on=["a"])
            g.add_rail("a", depends_on=["b"])

    def test_cycle_detection_indirect(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("b", depends_on=["a"])
        with pytest.raises(CyclicDependencyError):
            g.add_rail("c", depends_on=["b"])
            g.add_rail("a", depends_on=["c"])

    def test_unknown_dependency(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        with pytest.raises(UnknownRailError):
            g.add_rail("b", depends_on=["nonexistent"])

    def test_get_dependencies(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("b", depends_on=["a"])
        assert g.get_dependencies("b") == frozenset({"a"})
        assert g.get_dependencies("a") == frozenset()

    def test_get_dependents(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("b", depends_on=["a"])
        g.add_rail("c", depends_on=["a"])
        assert g.get_dependents("a") == frozenset({"b", "c"})

    def test_duplicate_add_rail(self):
        """Adding the same rail twice without changing it should be ok."""
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("a")  # no-op
        assert g.node_count == 1

    def test_from_flow_config_strings(self):
        """Build graph from plain string flow list."""
        flows = ["flow_a", "flow_b", "flow_c"]
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.node_count == 3
        assert g.edge_count == 0
        groups = g.compute_execution_groups()
        assert len(groups) == 1

    def test_from_flow_config_dicts(self):
        """Build graph from dict flow configs with depends_on."""
        flows = [
            {"name": "flow_a"},
            {"name": "flow_b", "depends_on": ["flow_a"]},
            {"name": "flow_c", "depends_on": []},
        ]
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.node_count == 3
        assert g.edge_count == 1
        groups = g.compute_execution_groups()
        assert len(groups) == 2
        assert "flow_a" in groups[0].rails
        assert "flow_c" in groups[0].rails
        assert "flow_b" in groups[1].rails

    def test_from_flow_config_mixed(self):
        """Build graph from mixed string and dict flows."""
        flows = [
            "flow_a",
            {"name": "flow_b", "depends_on": ["flow_a"]},
        ]
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.node_count == 2
        assert g.edge_count == 1

    def test_from_flow_config_flowwithdeps(self):
        """Build graph from FlowWithDeps objects."""
        from nemoguardrails.rails.llm.config import FlowWithDeps

        flows = [
            FlowWithDeps(name="flow_a"),
            FlowWithDeps(name="flow_b", depends_on=["flow_a"]),
        ]
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.node_count == 2
        assert g.edge_count == 1
        groups = g.compute_execution_groups()
        assert len(groups) == 2

    def test_from_flow_config_cycle_raises(self):
        """Cycle in flow config should raise at construction time."""
        flows = [
            {"name": "a", "depends_on": ["b"]},
            {"name": "b", "depends_on": ["a"]},
        ]
        with pytest.raises(CyclicDependencyError):
            RailDependencyGraph.from_flow_config(flows)

    def test_repr(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        g.add_rail("b", depends_on=["a"])
        assert "nodes=2" in repr(g)
        assert "edges=1" in repr(g)


# ---------------------------------------------------------------------------
# TopologicalScheduler tests
# ---------------------------------------------------------------------------


class TestTopologicalScheduler:
    """Tests for the scheduler execution engine."""

    def _make_scheduler(self, flows):
        return build_scheduler_from_config(flows)

    def test_single_flow(self):
        s = self._make_scheduler(["flow_a"])
        assert s.num_groups == 1
        assert s.max_parallelism == 1

    def test_all_independent(self):
        s = self._make_scheduler(["a", "b", "c"])
        assert s.num_groups == 1
        assert s.max_parallelism == 3

    def test_linear_chain_groups(self):
        s = self._make_scheduler(
            [
                {"name": "a"},
                {"name": "b", "depends_on": ["a"]},
                {"name": "c", "depends_on": ["b"]},
            ]
        )
        assert s.num_groups == 3
        assert s.max_parallelism == 1

    @pytest.mark.asyncio
    async def test_execute_all_continue(self):
        """All rails return continue — no blocking."""
        s = self._make_scheduler(["a", "b"])

        execution_log = []

        async def executor(rail_name, ctx):
            execution_log.append(rail_name)
            return {"action": "continue"}

        result = await s.execute(executor)
        assert result["blocked_by"] is None
        assert set(execution_log) == {"a", "b"}
        assert "elapsed_ms" in result

    @pytest.mark.asyncio
    async def test_execute_with_block(self):
        """A rail that returns block should stop execution."""
        s = self._make_scheduler(
            [
                {"name": "a"},
                {"name": "b", "depends_on": ["a"]},
            ]
        )

        async def executor(rail_name, ctx):
            if rail_name == "a":
                return {"action": "block", "reason": "unsafe"}
            return {"action": "continue"}

        result = await s.execute(executor)
        assert result["blocked_by"] == "a"
        # b should not have been executed
        assert "b" not in result["results"]

    @pytest.mark.asyncio
    async def test_execute_dependency_order(self):
        """Rails must execute in dependency order."""
        s = self._make_scheduler(
            [
                {"name": "a"},
                {"name": "b", "depends_on": ["a"]},
                {"name": "c", "depends_on": ["b"]},
            ]
        )

        execution_order = []

        async def executor(rail_name, ctx):
            execution_order.append(rail_name)
            return {"action": "continue"}

        await s.execute(executor)
        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_execute_parallel_group(self):
        """Independent rails in the same group run concurrently."""
        s = self._make_scheduler(
            [
                {"name": "a"},
                {"name": "b"},
                {"name": "c", "depends_on": ["a", "b"]},
            ]
        )

        started = []
        finished = []

        async def executor(rail_name, ctx):
            started.append(rail_name)
            await asyncio.sleep(0.01)  # Simulate work
            finished.append(rail_name)
            return {"action": "continue"}

        await s.execute(executor)
        # a and b should both start before c
        assert "c" not in started[:2] or set(started[:2]) == {"a", "b"}
        assert "c" in finished

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        """A rail that raises should be recorded as an error."""
        s = self._make_scheduler(["a"])

        async def executor(rail_name, ctx):
            raise ValueError("test error")

        result = await s.execute(executor)
        assert result["results"]["a"]["action"] == "error"
        assert "test error" in result["results"]["a"]["error"]

    @pytest.mark.asyncio
    async def test_execute_no_early_exit(self):
        """With early_exit_on_block=False, all rails complete even with blocks."""
        s = self._make_scheduler(["a", "b"])

        async def executor(rail_name, ctx):
            if rail_name == "a":
                return {"action": "block"}
            return {"action": "continue"}

        result = await s.execute(executor, early_exit_on_block=False)
        # Both should have results
        assert "a" in result["results"]
        assert "b" in result["results"]


# ---------------------------------------------------------------------------
# has_dependencies tests
# ---------------------------------------------------------------------------


class TestHasDependencies:
    """Tests for the fast-path dependency check."""

    def test_no_dependencies_strings(self):
        assert has_dependencies(["a", "b"]) is False

    def test_no_dependencies_dicts(self):
        assert has_dependencies([{"name": "a"}, {"name": "b"}]) is False

    def test_has_dependencies_dict(self):
        assert (
            has_dependencies(
                [
                    {"name": "a"},
                    {"name": "b", "depends_on": ["a"]},
                ]
            )
            is True
        )

    def test_has_dependencies_flowwithdeps(self):
        from nemoguardrails.rails.llm.config import FlowWithDeps

        flows = [
            FlowWithDeps(name="a"),
            FlowWithDeps(name="b", depends_on=["a"]),
        ]
        assert has_dependencies(flows) is True

    def test_empty_depends_on_not_counted(self):
        assert has_dependencies([{"name": "a", "depends_on": []}]) is False


# ---------------------------------------------------------------------------
# Integration with config models
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Tests for DAG scheduler integration with InputRails/OutputRails."""

    def test_input_rails_backward_compat(self):
        """Plain string flows should work without dependencies."""
        from nemoguardrails.rails.llm.config import InputRails

        ir = InputRails(**{"flows": ["flow_a", "flow_b"]})
        assert ir.flows == ["flow_a", "flow_b"]
        assert ir.has_dependencies is False

    def test_input_rails_with_dependencies(self):
        """Dict flows with depends_on should be parsed correctly."""
        from nemoguardrails.rails.llm.config import InputRails

        ir = InputRails(
            **{
                "parallel": True,
                "flows": [
                    "flow_a",
                    {"name": "flow_b", "depends_on": ["flow_a"]},
                ],
            }
        )
        assert ir.flows == ["flow_a", "flow_b"]
        assert ir.has_dependencies is True

    def test_output_rails_with_dependencies(self):
        from nemoguardrails.rails.llm.config import OutputRails

        ors = OutputRails(
            **{
                "parallel": True,
                "flows": [
                    {"name": "check_a"},
                    {"name": "check_b", "depends_on": ["check_a"]},
                ],
            }
        )
        assert ors.has_dependencies is True
        assert len(ors.flow_configs) == 2

    def test_scheduler_from_input_rails(self):
        """Build a scheduler from InputRails config."""
        from nemoguardrails.rails.llm.config import InputRails

        ir = InputRails(
            **{
                "parallel": True,
                "flows": [
                    {"name": "a"},
                    {"name": "b", "depends_on": ["a"]},
                    {"name": "c"},
                ],
            }
        )
        scheduler = build_scheduler_from_config(ir.flow_configs)
        assert scheduler.num_groups == 2
        groups = scheduler.groups
        assert "a" in groups[0].rails and "c" in groups[0].rails
        assert "b" in groups[1].rails

    def test_empty_rails(self):
        from nemoguardrails.rails.llm.config import InputRails

        ir = InputRails()
        assert ir.flows == []
        assert ir.has_dependencies is False
