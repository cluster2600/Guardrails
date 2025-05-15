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


def create_config(prompting_mode=None):
    """Create a test config with nemotron model and specified prompting mode."""
    config = {"models": [{"type": "main", "engine": "nim", "model": NEMOTRON_MODEL}]}

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


def test_other_modes_use_deepseek():
    """Verify non-reasoning modes use content-based format from deepseek.yml."""
    # test both standard mode (default) and another explicit mode
    for mode in [None, "compact"]:
        config = RailsConfig.from_content(
            colang_config(), yaml_content=create_config(mode)
        )
        prompt = get_prompt(config, Task.GENERATE_USER_INTENT)

        # content-based format
        assert hasattr(prompt, "content") and prompt.content is not None
        assert not hasattr(prompt, "messages") or prompt.messages is None

        # verify it's the deepseek.yml content
        assert "IMPORTANT: Use a short thinking process" in prompt.content


def test_prompt_selection_logic():
    """Test the direct prompt selection logic to verify file source."""
    # Create both configs
    reasoning_config = RailsConfig.from_content(
        colang_config(), yaml_content=create_config("reasoning")
    )
    standard_config = RailsConfig.from_content(
        colang_config(),
        yaml_content=create_config(None),  # Standard mode
    )

    # Test various tasks in both modes
    for task in [
        Task.GENERATE_BOT_MESSAGE,
        Task.GENERATE_USER_INTENT,
        Task.GENERATE_NEXT_STEPS,
        Task.GENERATE_VALUE,
    ]:
        # Check reasoning mode - should select from nemotron.yml
        reasoning_prompt = get_prompt(reasoning_config, task)
        assert (
            reasoning_prompt.mode == "reasoning"
        ), f"Task {task} should use reasoning mode"
        assert reasoning_prompt.models == [
            "nemotron"
        ], f"Task {task} should use nemotron model template"

        # Check standard mode - should select from deepseek.yml
        standard_prompt = get_prompt(standard_config, task)
        assert (
            not hasattr(standard_prompt, "mode") or standard_prompt.mode != "reasoning"
        ), f"Task {task} in standard mode should not use reasoning mode"
        assert (
            "nemotron" in standard_prompt.models
        ), f"Task {task} in standard mode should include nemotron in models list"

        # Verify content vs messages format
        assert (
            hasattr(reasoning_prompt, "messages")
            and reasoning_prompt.messages is not None
        ), f"Task {task} in reasoning mode should use message-based format"
        assert (
            hasattr(standard_prompt, "content") and standard_prompt.content is not None
        ), f"Task {task} in standard mode should use content-based format"


def test_prompt_source_files():
    """Verify the source files for prompts based on model and mode."""
    # Get a map of all prompts to their source files
    prompts_by_source = {}

    # Get the prompts directory path
    prompts_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../nemoguardrails/llm/prompts"
    )

    # Index all prompts by their source file, task and mode
    for prompt in _prompts:
        if prompt.models and "nemotron" in prompt.models:
            key = (prompt.task, getattr(prompt, "mode", "standard"))
            prompts_by_source[key] = getattr(prompt, "messages", None) is not None

    # Check that for each task, we have both message-based and content-based versions
    for task in [
        Task.GENERATE_BOT_MESSAGE.value,
        Task.GENERATE_USER_INTENT.value,
        Task.GENERATE_NEXT_STEPS.value,
        Task.GENERATE_VALUE.value,
    ]:
        # Check reasoning mode uses messages format (from nemotron.yml)
        assert prompts_by_source.get(
            (task, "reasoning"), False
        ), f"Task {task} should have a message-based prompt for reasoning mode"

        # Check standard mode uses content format (from deepseek.yml)
        reasoning_key = (task, "reasoning")
        standard_key = (task, "standard")

        # Either standard explicitly exists or there's a version without mode specified
        has_standard = standard_key in prompts_by_source or any(
            k[0] == task and (k[1] != "reasoning" or k[1] is None)
            for k in prompts_by_source
        )

        assert (
            has_standard
        ), f"Task {task} should have a content-based prompt for standard mode"


def test_prompt_selection_mechanism():
    """Test the core prompt selection mechanism directly."""
    task_name = Task.GENERATE_BOT_MESSAGE.value
    model_name = NEMOTRON_MODEL

    reasoning_prompt = _get_prompt(task_name, model_name, "reasoning", _prompts)
    assert reasoning_prompt.mode == "reasoning"
    assert hasattr(reasoning_prompt, "messages")
    assert reasoning_prompt.models == ["nemotron"]

    # should select a content-based prompt from deepseek.yml
    standard_prompt = _get_prompt(task_name, model_name, "standard", _prompts)
    assert not hasattr(standard_prompt, "mode") or standard_prompt.mode != "reasoning"
    assert hasattr(standard_prompt, "content")
    assert "nemotron" in standard_prompt.models
    assert "deepseek" in standard_prompt.models

    compact_prompt = _get_prompt(task_name, model_name, "compact", _prompts)
    assert not hasattr(compact_prompt, "mode") or compact_prompt.mode != "reasoning"
    assert hasattr(compact_prompt, "content")
    assert "nemotron" in compact_prompt.models
    assert "deepseek" in compact_prompt.models

    model_name = "deepseek-r1"

    compact_prompt = _get_prompt(task_name, model_name, "compact", _prompts)
    assert not hasattr(compact_prompt, "mode") or compact_prompt.mode != "reasoning"
    assert hasattr(compact_prompt, "content")
    assert "nemotron" in compact_prompt.models
    assert "deepseek" in compact_prompt.models
