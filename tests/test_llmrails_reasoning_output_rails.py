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

from typing import Any, Dict, List, Optional, Union

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat


@pytest.fixture
def rails_config_with_output_rails():
    """Create a RailsConfig with output rails enabled."""
    config = RailsConfig.from_content(
        colang_content="""
        define flow check sensitive info
            $not_allowed = execute check_sensitive_info
            if $not_allowed
                bot provide sanitized response
                stop
        define bot provide sanitized response
            "I cannot share sensitive information."
        """,
        yaml_content="""
        models:
          - type: main
            engine: fake
            model: fake
            reasoning_config:
              remove_thinking_traces: True
        colang_version: "1.0"
        rails:
          output:
            flows:
              - check sensitive info
            guardrail_reasoning_traces: True
        """,
    )
    return config


def test_basic_output_rail_config(rails_config_with_output_rails):
    """Test basic output rail configuration."""
    assert rails_config_with_output_rails.rails.output.flows == ["check sensitive info"]
    assert (
        rails_config_with_output_rails.rails.output.guardrail_reasoning_traces is True
    )


@pytest.mark.asyncio
async def test_output_rails_with_reasoning_traces_1(rails_config_with_output_rails):
    """Test that output rails properly handle reasoning traces in the response."""
    chat = TestChat(
        rails_config_with_output_rails,
        llm_completions=[
            '<think> I should not share sensitive info </think>\n  "Here is my credit card: 1234-5678-9012-3456"',
        ],
    )

    async def check_sensitive_info(context: Dict[str, Any]) -> bool:
        response = context.get("bot_message", "")

        result = "credit card" in response.lower() or any(
            c.isdigit() for c in response if c.isdigit() or c == "-"
        )
        return result

    chat.app.runtime.register_action(check_sensitive_info)

    # Test sensitive information is blocked and reasoning is preserved
    messages = [{"role": "user", "content": "What's your credit card number?"}]
    response = await chat.app.generate_async(messages=messages)

    # Verify the response contains reasoning traces
    assert "<think>" in response["content"]
    assert "I should not share sensitive info" in response["content"]
    assert "credit card" not in response["content"].lower()


@pytest.mark.asyncio
async def test_output_rails_without_reasoning_traces():
    """Test output rails without reasoning traces."""
    config = RailsConfig.from_content(
        colang_content="""
        define flow check sensitive info
            $not_allowed = execute check_sensitive_info
            if $not_allowed
                bot provide sanitized response
                stop
        define flow check think tag
            $not_allowed = execute check_think_tag_present
            if $not_allowed
                bot says tag not allowed
                stop

        define bot says tag not allowed
            "<think> tag is not allowed it must be removed"

        define bot provide sanitized response
            "I cannot share sensitive information."
        """,
        yaml_content="""
        models:
          - type: main
            engine: fake
            model: fake
            reasoning_config:
              remove_thinking_traces: True
        colang_version: "1.0"
        rails:
          input:
            flows:
              - check sensitive info
          output:
            flows:
              - check sensitive info
              - check think tag
            guardrail_reasoning_traces: false
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "<think> I should think more </think> Your credit card number is 1234-5678-9012-3456",
        ],
    )

    async def check_sensitive_info(context: Dict[str, Any]) -> bool:
        response = context.get("bot_message", "")
        prompt = context.get("user_message", "")

        input = response or prompt
        return "credit card" in input.lower() or any(
            c.isdigit() for c in input if c.isdigit() or c == "-"
        )

    async def check_think_tag_present(context: Dict[str, Any]) -> bool:
        response = context.get("bot_message", "")
        return "<think>" in response

    chat.app.runtime.register_action(check_sensitive_info)
    chat.app.runtime.register_action(check_think_tag_present)

    # Test sensitive information is blocked and reasoning is removed
    messages = [{"role": "user", "content": "What's your credit card number?"}]
    response = await chat.app.generate_async(messages=messages)

    info = chat.app.explain()
    # Verify the response does not contain reasoning traces
    # because it get blocked by the input rail

    assert "<think>" not in response["content"]
    assert "I should not share sensitive info" not in response["content"]
    assert response["content"] == "I cannot share sensitive information."

    messages = [{"role": "user", "content": "Tell me some numbers"}]
    response = await chat.app.generate_async(messages=messages)

    assert "<think>" in response["content"]
    assert "I should not share sensitive info" not in response["content"]
    assert (
        response["content"]
        == "<think> I should think more </think>I cannot share sensitive information."
    )


@pytest.mark.asyncio
async def test_output_rails_with_reasoning_traces():
    """Test output rails without reasoning traces."""
    config = RailsConfig.from_content(
        colang_content="""
        define flow check sensitive info
            $not_allowed = execute check_sensitive_info
            if $not_allowed
                bot provide sanitized response
                stop

        define flow check think tag
            $not_allowed = execute check_think_tag_present
            if $not_allowed
                bot informs tag not allowed
                stop

        define bot informs tag not allowed
            "think tag is not allowed it must be removed"

        """,
        yaml_content="""
        models:
          - type: main
            engine: fake
            model: fake
            reasoning_config:
              remove_thinking_traces: False
        colang_version: "1.0"
        rails:
          output:
            flows:
              - check think tag
            guardrail_reasoning_traces: True
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "<think> I should think more </think> Your kindness is appreciated"
        ],
    )

    async def check_think_tag_present(context: Dict[str, Any]) -> bool:
        response = context.get("bot_message", "")
        return "<think>" in response

    chat.app.runtime.register_action(check_think_tag_present)

    # Test sensitive information is blocked and reasoning is removed
    messages = [{"role": "user", "content": "you are nice"}]
    response = await chat.app.generate_async(messages=messages)

    # Verify the response does not contain reasoning traces
    assert "<think>" in response["content"]
    assert "think tag is not allowed" in response["content"]
