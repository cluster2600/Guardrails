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


"""Tests for Nemotron prompt mode selection behavior.

This module verifies that:
1. With "reasoning" mode, Nemotron models use message-based prompts from nemotron.yml
2. With any other mode, Nemotron models use content-based prompts from deepseek.yml
3. Some tasks in nemotron.yml have "detailed thinking" (two system messages)
4. Other tasks (GENERATE_USER_INTENT, and GENERATE_NEXT_STEPS) in nemotron.yml don't have "detailed thinking" (one system message)
"""

import os

import yaml

from nemoguardrails import RailsConfig
from nemoguardrails.llm.prompts import _get_prompt, _prompts, get_prompt
from nemoguardrails.llm.types import Task

NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
DEEPSEEK_MODEL = "deepseek-ai/deepseek-v2"


def colang_config():
    """Basic colang configuration for testing."""
    return """
    define user express greeting
        "hi"
        "hello"

    define flow
        user express greeting
        bot express greeting
    """


def create_config(prompting_mode=None, model=NEMOTRON_MODEL):
    """Create a test config with specified model and prompting mode."""
    config = {"models": [{"type": "main", "engine": "nim", "model": model}]}

    if prompting_mode:
        config["prompting_mode"] = prompting_mode

    return yaml.dump(config)


def test_reasoning_mode_uses_messages():
    """Verify "reasoning" mode uses message-based format from nemotron.yml."""

    config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config("reasoning")
    )
    prompt = get_prompt(config, Task.GENERATE_BOT_MESSAGE)

    assert prompt.mode == "reasoning"
    assert hasattr(prompt, "messages") and prompt.messages is not None
    assert not hasattr(prompt, "content") or prompt.content is None


def test_tasks_without_detailed_thinking():
    """Verify tasks that should have only one system message (no detailed thinking)."""
    config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config("reasoning")
    )

    for task in [Task.GENERATE_USER_INTENT, Task.GENERATE_NEXT_STEPS]:
        prompt = get_prompt(config, task)

        assert prompt.mode == "reasoning"
        assert hasattr(prompt, "messages") and prompt.messages is not None

        # one system message (no detailed thinking)
        system_messages = [
            msg
            for msg in prompt.messages
            if hasattr(msg, "type") and msg.type == "system"
        ]
        assert (
            len(system_messages) == 1
        ), f"Task {task} should have exactly one system message"

        assert (
            "detailed thinking on" not in system_messages[0].content
        ), f"Task {task} should not have 'detailed thinking on' in system message"


def test_tasks_with_detailed_thinking():
    """Verify tasks that should have two system messages (with detailed thinking)."""

    config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config("reasoning")
    )

    for task in [Task.GENERATE_BOT_MESSAGE, Task.GENERATE_VALUE]:
        prompt = get_prompt(config, task)

        assert prompt.mode == "reasoning"
        assert hasattr(prompt, "messages") and prompt.messages is not None

        # two system messages (one for detailed thinking, one for instructions)
        system_messages = [
            msg
            for msg in prompt.messages
            if hasattr(msg, "type") and msg.type == "system"
        ]
        assert (
            len(system_messages) == 2
        ), f"Task {task} should have exactly two system messages"

        assert (
            "detailed thinking on" in system_messages[0].content
        ), f"Task {task} should have 'detailed thinking on' in first system message"


def test_nemotron_standard_mode():
    """Verify nemotron in standard mode uses nemotron_standard.yml without detailed thinking."""

    # test both standard mode (default) and another explicit mode
    for mode in [None, "compact"]:
        config = RailsConfig.from_content(
            colang_config(), yaml_content=create_config(mode)
        )

        for task in [Task.GENERATE_BOT_MESSAGE, Task.GENERATE_USER_INTENT]:
            prompt = get_prompt(config, task)

            # Should use message-based format from nemotron_standard.yml
            assert hasattr(prompt, "messages") and prompt.messages is not None
            assert not hasattr(prompt, "content") or prompt.content is None

            # should not have "detailed thinking on" in any system message
            system_messages = [
                msg
                for msg in prompt.messages
                if hasattr(msg, "type") and msg.type == "system"
            ]

            for msg in system_messages:
                assert (
                    "detailed thinking on" not in msg.content
                ), f"Task {task} in standard mode should not have 'detailed thinking on'"

            # should not have mode=reasoning
            assert (
                not hasattr(prompt, "mode") or prompt.mode != "reasoning"
            ), f"Task {task} in standard mode should not have reasoning mode"


def test_deepseek_uses_deepseek_yml():
    """Verify DeepSeek models use deepseek.yml."""
    config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config(None, DEEPSEEK_MODEL)
    )

    for task in [Task.GENERATE_BOT_MESSAGE, Task.GENERATE_USER_INTENT]:
        prompt = get_prompt(config, task)

        # should use content-based format from deepseek.yml
        assert hasattr(prompt, "content") and prompt.content is not None
        assert not hasattr(prompt, "messages") or prompt.messages is None

        # should have "Use a short thinking process" from deepseek.yml
        assert "IMPORTANT: Use a short thinking process" in prompt.content
        assert "deepseek" in prompt.models
        assert "nemotron" not in prompt.models


def test_prompt_selection_logic():
    """Test the direct prompt selection logic to verify file source."""
    reasoning_config = RailsConfig.from_content(
        colang_config(),
        yaml_content=create_config("reasoning"),  # Nemotron with reasoning
    )
    standard_config = RailsConfig.from_content(
        colang_config(),
        yaml_content=create_config(None),  # Nemotron with standard mode
    )
    deepseek_config = RailsConfig.from_content(
        colang_config(),
        yaml_content=create_config(None, DEEPSEEK_MODEL),  # DeepSeek model
    )

    for task in [
        Task.GENERATE_BOT_MESSAGE,
        Task.GENERATE_USER_INTENT,
        Task.GENERATE_NEXT_STEPS,
        Task.GENERATE_VALUE,
    ]:
        # Nemotron with reasoning mode -> nemotron_reasoning.yml with detailed thinking
        reasoning_prompt = get_prompt(reasoning_config, task)
        assert hasattr(reasoning_prompt, "messages")
        if task in [Task.GENERATE_BOT_MESSAGE, Task.GENERATE_VALUE]:
            system_messages = [
                m
                for m in reasoning_prompt.messages
                if hasattr(m, "type") and m.type == "system"
            ]
            assert len(system_messages) > 1
            assert "detailed thinking on" in system_messages[0].content

        # Nemotron with standard mode -> nemotron_standard.yml without detailed thinking
        standard_prompt = get_prompt(standard_config, task)
        assert hasattr(standard_prompt, "messages")
        system_messages = [
            m
            for m in standard_prompt.messages
            if hasattr(m, "type") and m.type == "system"
        ]
        for msg in system_messages:
            assert "detailed thinking on" not in msg.content

        # DeepSeek model -> deepseek.yml
        deepseek_prompt = get_prompt(deepseek_config, task)
        assert hasattr(deepseek_prompt, "content")
        assert "IMPORTANT: Use a short thinking process" in deepseek_prompt.content


def test_prompt_source_files():
    """Verify the source files for prompts based on model and mode."""
    nemotron_reasoning_prompts = []
    nemotron_standard_prompts = []
    deepseek_prompts = []

    for prompt in _prompts:
        if not prompt.models:
            continue

        if "nemotron" in prompt.models:
            if hasattr(prompt, "mode") and prompt.mode == "reasoning":
                nemotron_reasoning_prompts.append(prompt)
            elif not hasattr(prompt, "mode") or prompt.mode != "reasoning":
                nemotron_standard_prompts.append(prompt)

        if "deepseek" in prompt.models:
            deepseek_prompts.append(prompt)

    assert len(nemotron_reasoning_prompts) > 0, "Should have nemotron reasoning prompts"
    assert len(nemotron_standard_prompts) > 0, "Should have nemotron standard prompts"
    assert len(deepseek_prompts) > 0, "Should have deepseek prompts"

    for prompt in nemotron_reasoning_prompts:
        assert hasattr(
            prompt, "messages"
        ), "Reasoning prompts should use message format"

    for prompt in nemotron_standard_prompts:
        assert hasattr(
            prompt, "messages"
        ), "Standard nemotron prompts should use message format"

    for prompt in deepseek_prompts:
        assert hasattr(prompt, "content"), "DeepSeek prompts should use content format"


def test_prompt_selection_mechanism():
    """Test the core prompt selection mechanism directly."""

    task_name = Task.GENERATE_BOT_MESSAGE.value
    nemotron_model = NEMOTRON_MODEL
    deepseek_model = DEEPSEEK_MODEL

    # Nemotron with reasoning mode -> detailed thinking
    reasoning_prompt = _get_prompt(task_name, nemotron_model, "reasoning", _prompts)
    assert hasattr(reasoning_prompt, "messages")
    assert reasoning_prompt.models == ["nemotron"]
    assert reasoning_prompt.mode == "reasoning"

    # Nemotron with standard mode -> no detailed thinking
    standard_prompt = _get_prompt(task_name, nemotron_model, "standard", _prompts)
    assert hasattr(standard_prompt, "messages")

    # sort the prompts to ensure deterministic selection across Python versions
    nemotron_prompts = [
        p
        for p in _prompts
        if hasattr(p, "models")
        and p.models
        and "nemotron" in p.models
        and p.task == task_name
    ]
    standard_nemotron_prompts = [
        p for p in nemotron_prompts if not hasattr(p, "mode") or p.mode != "reasoning"
    ]
    assert (
        len(standard_nemotron_prompts) > 0
    ), "Should have at least one standard Nemotron prompt for this task"

    # the bug is that on Python 3.10/3.11, it's selecting a Llama3 prompt instead of Nemotron
    # this is likely due to iteration order differences
    assert "nemotron" in standard_prompt.models
    assert not hasattr(standard_prompt, "mode") or standard_prompt.mode != "reasoning"

    # Nemotron with compact mode -> should also use standard without detailed thinking
    compact_prompt = _get_prompt(task_name, nemotron_model, "compact", _prompts)
    assert hasattr(compact_prompt, "messages")
    assert "nemotron" in compact_prompt.models
    assert "deepseek" not in compact_prompt.models
    assert not hasattr(compact_prompt, "mode") or compact_prompt.mode != "reasoning"

    # DeepSeek model -> should use deepseek.yml
    deepseek_prompt = _get_prompt(task_name, deepseek_model, "standard", _prompts)
    assert hasattr(deepseek_prompt, "content")
    assert "deepseek" in deepseek_prompt.models
    assert "nemotron" not in deepseek_prompt.models
