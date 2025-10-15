# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails as _LLMRails
from nemoguardrails.rails.llm.options import RailResult, RailsStatus


class TestRailsStatusDataStructures:
    """Test the RailResult and RailsStatus data structures."""

    def test_rail_result_creation(self):
        result = RailResult(
            rail_type="input",
            rail_name="test_rail",
            passed=False,
            stopped=True,
            message="Test failed",
            decisions=["stop"],
            executed_actions=["action1"],
        )

        assert result.rail_type == "input"
        assert result.rail_name == "test_rail"
        assert result.passed is False
        assert result.stopped is True
        assert result.message == "Test failed"
        assert result.decisions == ["stop"]
        assert result.executed_actions == ["action1"]

    def test_rails_status_all_passed_true(self):
        status = RailsStatus(
            input_rails=[
                RailResult(
                    rail_type="input", rail_name="rail1", passed=True, stopped=False
                )
            ],
            output_rails=[
                RailResult(
                    rail_type="output", rail_name="rail2", passed=True, stopped=False
                )
            ],
        )

        assert status.all_passed is True
        assert len(status.failed_rails) == 0

    def test_rails_status_all_passed_false(self):
        status = RailsStatus(
            input_rails=[
                RailResult(
                    rail_type="input",
                    rail_name="rail1",
                    passed=False,
                    stopped=True,
                    message="Failed",
                )
            ],
            output_rails=[
                RailResult(
                    rail_type="output", rail_name="rail2", passed=True, stopped=False
                )
            ],
        )

        assert status.all_passed is False
        assert len(status.failed_rails) == 1
        assert status.failed_rails[0].rail_name == "rail1"

    def test_rails_status_multiple_failures(self):
        status = RailsStatus(
            input_rails=[
                RailResult(
                    rail_type="input",
                    rail_name="rail1",
                    passed=False,
                    stopped=True,
                ),
                RailResult(
                    rail_type="input",
                    rail_name="rail2",
                    passed=False,
                    stopped=True,
                ),
            ],
            output_rails=[
                RailResult(
                    rail_type="output",
                    rail_name="rail3",
                    passed=False,
                    stopped=True,
                )
            ],
        )

        assert status.all_passed is False
        assert len(status.failed_rails) == 3
        failed_names = [r.rail_name for r in status.failed_rails]
        assert "rail1" in failed_names
        assert "rail2" in failed_names
        assert "rail3" in failed_names


class TestRailsStatusPopulation:
    """Test that rails_status is populated correctly in generate()."""

    def test_rails_status_populated_for_input_rails_only(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        response = rails.generate(
            messages=[{"role": "user", "content": "hello"}],
            options={"rails": {"input": True, "output": False, "dialog": False}},
        )

        assert hasattr(response, "rails_status")
        assert response.rails_status is not None
        assert isinstance(response.rails_status, RailsStatus)

    def test_rails_status_not_populated_when_dialog_enabled(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        response = rails.generate(
            messages=[{"role": "user", "content": "hello"}],
            options={
                "rails": {"input": True, "output": False, "dialog": True}
            },  # dialog=True
        )

        # rails_status should not be populated when dialog is enabled
        assert response.rails_status is None

    def test_rails_status_contains_input_rail_results(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        response = rails.generate(
            messages=[{"role": "user", "content": "hello"}],
            options={"rails": {"input": True, "output": False, "dialog": False}},
        )

        assert response.rails_status is not None
        # Should have at least one input rail (self check input)
        assert len(response.rails_status.input_rails) > 0

        # Check structure of rail result
        rail_result = response.rails_status.input_rails[0]
        assert hasattr(rail_result, "rail_type")
        assert hasattr(rail_result, "rail_name")
        assert hasattr(rail_result, "passed")
        assert hasattr(rail_result, "stopped")
        assert hasattr(rail_result, "decisions")
        assert hasattr(rail_result, "executed_actions")

    def test_rails_status_input_rail_passes(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        response = rails.generate(
            messages=[{"role": "user", "content": "hello"}],
            options={"rails": {"input": True, "output": False, "dialog": False}},
        )

        assert response.rails_status is not None
        # Safe input should pass
        assert response.rails_status.all_passed is True
        assert len(response.rails_status.failed_rails) == 0

    def test_rails_status_input_rail_fails(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        response = rails.generate(
            messages=[{"role": "user", "content": "you are stupid"}],
            options={"rails": {"input": True, "output": False, "dialog": False}},
        )

        assert response.rails_status is not None
        # Bad input should fail
        assert response.rails_status.all_passed is False
        assert len(response.rails_status.failed_rails) > 0

        # Check the failed rail details
        failed_rail = response.rails_status.failed_rails[0]
        assert failed_rail.passed is False
        assert failed_rail.stopped is True
        assert failed_rail.rail_type == "input"
        # Should have a message explaining why it failed
        assert failed_rail.message is not None
        assert len(failed_rail.message) > 0

    def test_rails_status_decisions_populated(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        response = rails.generate(
            messages=[{"role": "user", "content": "you are stupid"}],
            options={"rails": {"input": True, "output": False, "dialog": False}},
        )

        assert response.rails_status is not None
        failed_rail = response.rails_status.failed_rails[0]

        # Decisions should be populated with the flow steps
        assert len(failed_rail.decisions) > 0
        # Should contain "stop" since the rail stopped
        assert "stop" in failed_rail.decisions

    def test_rails_status_executed_actions_populated(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        response = rails.generate(
            messages=[{"role": "user", "content": "you are stupid"}],
            options={"rails": {"input": True, "output": False, "dialog": False}},
        )

        assert response.rails_status is not None
        # Check that executed_actions is populated
        for rail in response.rails_status.input_rails:
            assert isinstance(rail.executed_actions, list)
            # At least one action should have been executed (e.g., self_check_input)
            if not rail.passed:
                assert len(rail.executed_actions) > 0


class TestExtractRailsStatus:
    """Test the _extract_rails_status() helper method."""

    def test_extract_rails_status_with_stopped_rail(self):
        from nemoguardrails.rails.llm.options import ActivatedRail, ExecutedAction

        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        # Create mock activated rail with stop=True
        activated_rails = [
            ActivatedRail(
                type="input",
                name="test_rail",
                stop=True,
                decisions=["refuse to respond", "stop"],
                executed_actions=[
                    ExecutedAction(action_name="test_action", action_params={})
                ],
            )
        ]

        status = rails._extract_rails_status(activated_rails)

        assert len(status.input_rails) == 1
        result = status.input_rails[0]
        assert result.rail_name == "test_rail"
        assert result.passed is False  # stop=True means not passed
        assert result.stopped is True
        assert result.decisions == ["refuse to respond", "stop"]
        assert result.executed_actions == ["test_action"]
        # Message should be formatted from decisions
        assert "refuse to respond" in result.message
        assert "stop" in result.message

    def test_extract_rails_status_with_passing_rail(self):
        from nemoguardrails.rails.llm.options import ActivatedRail, ExecutedAction

        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        # Create mock activated rail with stop=False
        activated_rails = [
            ActivatedRail(
                type="input",
                name="test_rail",
                stop=False,
                decisions=["continue"],
                executed_actions=[
                    ExecutedAction(action_name="test_action", action_params={})
                ],
            )
        ]

        status = rails._extract_rails_status(activated_rails)

        assert len(status.input_rails) == 1
        result = status.input_rails[0]
        assert result.passed is True  # stop=False means passed
        assert result.stopped is False
        assert result.message is None  # No error message for passing rails

    def test_extract_rails_status_separates_input_output(self):
        from nemoguardrails.rails.llm.options import ActivatedRail

        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        activated_rails = [
            ActivatedRail(type="input", name="input_rail", stop=False),
            ActivatedRail(type="output", name="output_rail", stop=True),
            ActivatedRail(type="input", name="input_rail2", stop=True),
        ]

        status = rails._extract_rails_status(activated_rails)

        assert len(status.input_rails) == 2
        assert len(status.output_rails) == 1
        assert status.input_rails[0].rail_name == "input_rail"
        assert status.input_rails[1].rail_name == "input_rail2"
        assert status.output_rails[0].rail_name == "output_rail"

    def test_extract_rails_status_handles_empty_decisions(self):
        from nemoguardrails.rails.llm.options import ActivatedRail

        config = RailsConfig.from_path("./examples/bots/abc")
        rails = _LLMRails(config)

        activated_rails = [
            ActivatedRail(
                type="input", name="test_rail", stop=True, decisions=[]  # Empty
            )
        ]

        status = rails._extract_rails_status(activated_rails)

        result = status.input_rails[0]
        assert result.message == "Rail stopped"  # Default message
