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

"""Tests for the LLMTaskManager with multimodal content."""

import textwrap
from typing import Any, Dict, List

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task


def test_history_integration_with_filters():
    """Test the integration of filters with history processing."""
    from nemoguardrails.llm.filters import to_chat_messages, user_assistant_sequence

    # Create events with multimodal content
    multimodal_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    events = [
        {"type": "UserMessage", "text": multimodal_message},
        {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
    ]

    # Mock a template rendering environment similar to what the task manager would use
    import jinja2

    env = jinja2.Environment()
    env.filters["user_assistant_sequence"] = user_assistant_sequence
    env.filters["to_chat_messages"] = to_chat_messages

    # Test with user_assistant_sequence filter
    template1 = env.from_string("{{ events | user_assistant_sequence }}")
    result1 = template1.render(events=events)

    # Test with to_chat_messages filter
    template2 = env.from_string("{{ events | to_chat_messages | tojson }}")
    result2 = template2.render(events=events)

    # Verify the multimodal content is correctly formatted by our filters
    assert "User: What's in this image? [+ image]" in result1
    assert "Assistant: I see a cat in the image." in result1

    # Verify that to_chat_messages preserves the structure
    import json

    chat_messages = json.loads(result2)
    assert len(chat_messages) == 2
    assert chat_messages[0]["role"] == "user"
    assert isinstance(chat_messages[0]["content"], list)
    assert chat_messages[0]["content"][0]["type"] == "text"
    assert chat_messages[0]["content"][1]["type"] == "image_url"


def test_user_assistant_sequence_with_multimodal():
    """Test that the user_assistant_sequence filter correctly formats multimodal content."""
    from nemoguardrails.llm.filters import user_assistant_sequence

    # Create events with multimodal content
    multimodal_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    events = [
        {"type": "UserMessage", "text": "Hello!"},
        {"type": "StartUtteranceBotAction", "script": "Hi there! How can I help you?"},
        {"type": "UserMessage", "text": multimodal_message},
        {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
    ]

    # Apply the user_assistant_sequence filter
    formatted_history = user_assistant_sequence(events)

    # Verify the multimodal content is correctly formatted
    assert "User: Hello!" in formatted_history
    assert "Assistant: Hi there! How can I help you?" in formatted_history
    assert "User: What's in this image? [+ image]" in formatted_history
    assert "Assistant: I see a cat in the image." in formatted_history


def test_to_chat_messages_multimodal_integration():
    """Test integration of to_chat_messages with multimodal content."""
    from nemoguardrails.llm.filters import to_chat_messages

    # Create events with multimodal content
    multimodal_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    events = [
        {"type": "UserMessage", "text": multimodal_message},
        {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
    ]

    # Apply the to_chat_messages filter
    chat_messages = to_chat_messages(events)

    # Verify that the structure is preserved correctly
    assert len(chat_messages) == 2
    assert chat_messages[0]["role"] == "user"
    assert isinstance(chat_messages[0]["content"], list)
    assert len(chat_messages[0]["content"]) == 2
    assert chat_messages[0]["content"][0]["type"] == "text"
    assert chat_messages[0]["content"][0]["text"] == "What's in this image?"
    assert chat_messages[0]["content"][1]["type"] == "image_url"
    assert (
        chat_messages[0]["content"][1]["image_url"]["url"]
        == "https://example.com/image.jpg"
    )

    assert chat_messages[1]["role"] == "assistant"
    assert chat_messages[1]["content"] == "I see a cat in the image."
