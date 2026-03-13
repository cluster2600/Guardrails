# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advanced tests for nemoguardrails.rails.llm.config.

Covers:
  - FlowWithDeps validation
  - _coerce_flow_list edge cases (invalid entries)
  - _RailSectionMixin properties
  - InputRails / OutputRails normalisation
"""

import pytest

from nemoguardrails.rails.llm.config import (
    FlowWithDeps,
    InputRails,
    OutputRails,
    _coerce_flow_list,
)


class TestFlowWithDeps:
    """Tests for the FlowWithDeps model."""

    def test_plain_name(self):
        f = FlowWithDeps(name="flow_a")
        assert f.name == "flow_a"
        assert f.depends_on == []

    def test_with_dependencies(self):
        f = FlowWithDeps(name="flow_b", depends_on=["flow_a"])
        assert f.depends_on == ["flow_a"]

    def test_extra_fields_allowed(self):
        f = FlowWithDeps(name="flow_a", cpu_bound=True, timeout_ms=500)
        assert f.name == "flow_a"


class TestCoerceFlowList:
    """Tests for _coerce_flow_list."""

    def test_strings(self):
        result = _coerce_flow_list(["a", "b"])
        assert len(result) == 2
        assert all(isinstance(r, FlowWithDeps) for r in result)
        assert result[0].name == "a"

    def test_dicts(self):
        result = _coerce_flow_list([{"name": "a"}, {"name": "b", "depends_on": ["a"]}])
        assert len(result) == 2
        assert result[1].depends_on == ["a"]

    def test_flowwithdeps_passthrough(self):
        f = FlowWithDeps(name="a")
        result = _coerce_flow_list([f])
        assert result[0] is f

    def test_mixed(self):
        result = _coerce_flow_list(["a", {"name": "b"}, FlowWithDeps(name="c")])
        assert len(result) == 3

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid flow entry"):
            _coerce_flow_list([123])

    def test_none_entry_raises(self):
        with pytest.raises(ValueError, match="Invalid flow entry"):
            _coerce_flow_list([None])

    def test_empty_list(self):
        result = _coerce_flow_list([])
        assert result == []


class TestInputRailsNormalisation:
    """Tests for InputRails flow normalisation."""

    def test_legacy_flows_key(self):
        ir = InputRails(**{"flows": ["a", "b"]})
        assert ir.flows == ["a", "b"]
        assert len(ir.flow_configs) == 2

    def test_flow_configs_key(self):
        ir = InputRails(**{"flow_configs": [FlowWithDeps(name="a")]})
        assert ir.flows == ["a"]

    def test_dict_flows_with_depends_on(self):
        ir = InputRails(
            **{
                "flows": [
                    {"name": "a"},
                    {"name": "b", "depends_on": ["a"]},
                ]
            }
        )
        assert ir.has_dependencies is True
        assert ir.flows == ["a", "b"]

    def test_no_dependencies(self):
        ir = InputRails(**{"flows": ["a", "b"]})
        assert ir.has_dependencies is False

    def test_empty(self):
        ir = InputRails()
        assert ir.flows == []
        assert ir.has_dependencies is False
        assert ir.flow_configs == []


class TestOutputRailsNormalisation:
    """Tests for OutputRails flow normalisation."""

    def test_with_dependencies(self):
        ors = OutputRails(
            **{
                "flows": [
                    {"name": "a"},
                    {"name": "b", "depends_on": ["a"]},
                ]
            }
        )
        assert ors.has_dependencies is True
        assert len(ors.flow_configs) == 2
