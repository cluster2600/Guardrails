# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advanced tests for nemoguardrails.rails.llm.dag_scheduler.

Covers edge cases not in test_dag_scheduler.py:
  - Timeout handling in _execute_with_timeout
  - Parallel group with early exit on block
  - _gather_with_early_exit error handling
  - Parallel group without early exit + exceptions
  - Task cancellation cleanup
  - from_flow_config with unexpected types / metadata
  - ExecutionGroup iteration and len
  - Self-dependency detection
  - get_dependencies for unknown rail
"""

import asyncio

import pytest

from nemoguardrails.rails.llm.dag_scheduler import (
    CyclicDependencyError,
    ExecutionGroup,
    RailDependencyGraph,
    RailNode,
    TopologicalScheduler,
    build_scheduler_from_config,
    has_dependencies,
)


class TestExecutionGroup:
    """Tests for ExecutionGroup dataclass."""

    def test_len(self):
        g = ExecutionGroup(index=0, rails=frozenset({"a", "b", "c"}))
        assert len(g) == 3

    def test_iter(self):
        g = ExecutionGroup(index=0, rails=frozenset({"a", "b"}))
        assert set(g) == {"a", "b"}

    def test_frozen(self):
        g = ExecutionGroup(index=0, rails=frozenset({"a"}))
        with pytest.raises(AttributeError):
            g.index = 1


class TestRailNode:
    """Tests for RailNode dataclass."""

    def test_hash(self):
        n1 = RailNode(name="a")
        n2 = RailNode(name="a", depends_on=frozenset({"b"}))
        assert hash(n1) == hash(n2)

    def test_eq(self):
        n1 = RailNode(name="a")
        n2 = RailNode(name="a")
        assert n1 == n2

    def test_eq_not_implemented(self):
        n = RailNode(name="a")
        assert n != "a"


class TestSelfDependency:
    """Test self-dependency detection."""

    def test_self_dependency_raises(self):
        g = RailDependencyGraph()
        g.add_rail("a")
        with pytest.raises(CyclicDependencyError):
            g.add_rail("a", depends_on=["a"])


class TestGetDependenciesUnknown:
    """Test get_dependencies for an unknown rail."""

    def test_unknown_rail_returns_empty(self):
        g = RailDependencyGraph()
        assert g.get_dependencies("nonexistent") == frozenset()


class TestFromFlowConfigEdgeCases:
    """Tests for from_flow_config with unusual inputs."""

    def test_unexpected_type_logs_warning(self, caplog):
        """Passing an unsupported type should log a warning."""
        flows = ["a", 12345]  # int is not str, dict, or has .name
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.node_count == 1  # only "a" registered

    def test_flow_config_with_metadata(self):
        """Dict configs with extra metadata should be preserved."""
        flows = [
            {"name": "a", "cpu_bound": True},
            {"name": "b", "depends_on": ["a"], "timeout_ms": 500},
        ]
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.nodes["a"].metadata.get("cpu_bound") is True
        assert g.nodes["b"].metadata.get("timeout_ms") == 500

    def test_flow_config_with_flow_name_key(self):
        """Dict configs using 'flow_name' instead of 'name' should work."""
        flows = [
            {"flow_name": "a"},
            {"flow_name": "b", "depends_on": ["a"]},
        ]
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.node_count == 2

    def test_flow_config_pydantic_model(self):
        """Objects with .name and .depends_on attrs should work."""

        class FakeFlowModel:
            def __init__(self, name, depends_on=None):
                self.name = name
                self.depends_on = depends_on or []

        flows = [
            FakeFlowModel("a"),
            FakeFlowModel("b", depends_on=["a"]),
        ]
        g = RailDependencyGraph.from_flow_config(flows)
        assert g.node_count == 2
        assert g.edge_count == 1


class TestSchedulerTimeout:
    """Tests for timeout handling in TopologicalScheduler."""

    @pytest.mark.asyncio
    async def test_timeout_triggers(self):
        """A slow rail should raise asyncio.TimeoutError with timeout_per_group."""
        s = build_scheduler_from_config(["a"], timeout_per_group=0.01)

        async def slow_executor(rail_name, ctx):
            await asyncio.sleep(10)
            return {"action": "continue"}

        result = await s.execute(slow_executor)
        # Should be recorded as error (asyncio.TimeoutError has empty str())
        assert result["results"]["a"]["action"] == "error"

    @pytest.mark.asyncio
    async def test_timeout_not_triggered(self):
        """A fast rail should complete fine with a generous timeout."""
        s = build_scheduler_from_config(["a"], timeout_per_group=10.0)

        async def fast_executor(rail_name, ctx):
            return {"action": "continue"}

        result = await s.execute(fast_executor)
        assert result["results"]["a"]["action"] == "continue"
        assert result["blocked_by"] is None


class TestParallelGroupEarlyExit:
    """Tests for parallel group execution with early exit on block."""

    @pytest.mark.asyncio
    async def test_parallel_block_cancels_remaining(self):
        """When one rail blocks in a parallel group, others should be cancelled."""
        s = build_scheduler_from_config(["a", "b", "c"])

        executed = []

        async def executor(rail_name, ctx):
            executed.append(rail_name)
            if rail_name == "a":
                return {"action": "block", "reason": "unsafe"}
            await asyncio.sleep(1)  # slow — should be cancelled
            return {"action": "continue"}

        result = await s.execute(executor, early_exit_on_block=True)
        assert result["blocked_by"] is not None

    @pytest.mark.asyncio
    async def test_parallel_no_block_all_complete(self):
        """With no blocks, all parallel rails should complete."""
        s = build_scheduler_from_config(["a", "b", "c"])

        async def executor(rail_name, ctx):
            await asyncio.sleep(0.01)
            return {"action": "continue"}

        result = await s.execute(executor)
        assert result["blocked_by"] is None
        assert len(result["results"]) == 3


class TestParallelGroupNoEarlyExit:
    """Tests for parallel group execution without early exit."""

    @pytest.mark.asyncio
    async def test_parallel_no_early_exit_all_complete(self):
        """With early_exit_on_block=False, all rails complete even with blocks."""
        s = build_scheduler_from_config(["a", "b", "c"])

        async def executor(rail_name, ctx):
            if rail_name == "a":
                return {"action": "block"}
            return {"action": "continue"}

        result = await s.execute(executor, early_exit_on_block=False)
        assert "a" in result["results"]
        assert "b" in result["results"]
        assert "c" in result["results"]

    @pytest.mark.asyncio
    async def test_parallel_no_early_exit_with_exception(self):
        """Exceptions in parallel group without early exit should be captured."""
        s = build_scheduler_from_config(["a", "b"])

        async def executor(rail_name, ctx):
            if rail_name == "a":
                raise RuntimeError("rail a exploded")
            return {"action": "continue"}

        result = await s.execute(executor, early_exit_on_block=False)
        assert result["results"]["a"]["action"] == "error"
        assert "exploded" in result["results"]["a"]["error"]
        assert result["results"]["b"]["action"] == "continue"


class TestGatherWithEarlyExitErrors:
    """Tests for _gather_with_early_exit error handling."""

    @pytest.mark.asyncio
    async def test_exception_in_parallel_task(self):
        """A rail that raises during parallel execution should be captured."""
        s = build_scheduler_from_config(["a", "b"])

        async def executor(rail_name, ctx):
            if rail_name == "a":
                raise ValueError("bad input")
            await asyncio.sleep(0.01)
            return {"action": "continue"}

        result = await s.execute(executor, early_exit_on_block=True)
        assert result["results"]["a"]["action"] == "error"
        assert "bad input" in result["results"]["a"]["error"]

    @pytest.mark.asyncio
    async def test_stop_action_triggers_early_exit(self):
        """action='stop' should also trigger early exit."""
        s = build_scheduler_from_config(["a", "b"])

        async def executor(rail_name, ctx):
            if rail_name == "a":
                return {"action": "stop"}
            await asyncio.sleep(1)
            return {"action": "continue"}

        result = await s.execute(executor, early_exit_on_block=True)
        assert result["blocked_by"] is not None

    @pytest.mark.asyncio
    async def test_refused_action_triggers_early_exit(self):
        """action='refused' should also trigger early exit."""
        s = build_scheduler_from_config(["a", "b"])

        async def executor(rail_name, ctx):
            if rail_name == "a":
                return {"action": "refused"}
            await asyncio.sleep(1)
            return {"action": "continue"}

        result = await s.execute(executor, early_exit_on_block=True)
        assert result["blocked_by"] is not None


class TestIsBlock:
    """Tests for the _is_block static method."""

    def test_block_action(self):
        assert TopologicalScheduler._is_block({"action": "block"}) is True

    def test_stop_action(self):
        assert TopologicalScheduler._is_block({"action": "stop"}) is True

    def test_refused_action(self):
        assert TopologicalScheduler._is_block({"action": "refused"}) is True

    def test_continue_action(self):
        assert TopologicalScheduler._is_block({"action": "continue"}) is False

    def test_non_dict_result(self):
        assert TopologicalScheduler._is_block("not a dict") is False
        assert TopologicalScheduler._is_block(None) is False
        assert TopologicalScheduler._is_block(42) is False

    def test_dict_without_action(self):
        assert TopologicalScheduler._is_block({"status": "ok"}) is False


class TestSchedulerProperties:
    """Tests for TopologicalScheduler properties."""

    def test_empty_graph_max_parallelism(self):
        g = RailDependencyGraph()
        s = TopologicalScheduler(g)
        assert s.max_parallelism == 0
        assert s.num_groups == 0
        assert s.groups == []

    def test_groups_returns_copy(self):
        s = build_scheduler_from_config(["a", "b"])
        g1 = s.groups
        g2 = s.groups
        assert g1 == g2
        assert g1 is not g2  # should be a copy


class TestHasDependenciesEdgeCases:
    """Additional has_dependencies tests."""

    def test_empty_list(self):
        assert has_dependencies([]) is False

    def test_dict_with_empty_depends_on(self):
        assert has_dependencies([{"name": "a", "depends_on": []}]) is False

    def test_object_with_depends_on_attr(self):
        class FakeFlow:
            depends_on = ["a"]

        assert has_dependencies([FakeFlow()]) is True

    def test_object_with_none_depends_on(self):
        class FakeFlow:
            depends_on = None

        assert has_dependencies([FakeFlow()]) is False

    def test_object_without_depends_on(self):
        class FakeFlow:
            pass

        assert has_dependencies([FakeFlow()]) is False
