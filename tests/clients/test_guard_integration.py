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
from nemoguardrails.clients.guard import ClientRails, GuardrailViolation
from nemoguardrails.rails.llm.llmrails import LLMRails


class MockOpenAIClient:
    def __init__(self):
        self.chat = MockChat()


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockCompletions:
    def create(self, messages, **kwargs):
        return MockResponse("Hello from OpenAI!")


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


MockOpenAIClient.__module__ = "openai"


class TestGuardIntegrationWithoutRails:
    """Test guard behavior when no rails are configured."""

    def test_openai_without_rails(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")

        rails = LLMRails(config)
        guard = ClientRails(rails)

        client = MockOpenAIClient()
        guarded_client = guard(client)

        response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.choices[0].message.content == "Hello from OpenAI!"


class TestGuardIntegrationBasic:
    """Test basic guard integration functionality."""

    def test_adapter_wrapping_works(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")

        rails = LLMRails(config)
        guard = ClientRails(rails)

        client = MockOpenAIClient()
        guarded_client = guard(client)

        assert hasattr(guarded_client, "chat")
        assert hasattr(guarded_client.chat, "completions")
        assert hasattr(guarded_client.chat.completions, "create")

    def test_response_passthrough_no_rails(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")

        rails = LLMRails(config)
        guard = ClientRails(rails)

        client = MockOpenAIClient()
        guarded_client = guard(client)

        response = guarded_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ]
        )

        assert response.choices[0].message.content == "Hello from OpenAI!"


class TestGuardRailsExistenceChecks:
    """Test that rails are only invoked when configured."""

    def test_has_input_rails_false(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")

        rails = LLMRails(config)
        guard = ClientRails(rails)

        assert guard.has_input_rails() is False

    def test_has_input_rails_true(self):
        config = RailsConfig.from_content(
            colang_content="",
            yaml_content="""
            models: []
            rails:
              input:
                flows:
                  - self check input
            prompts:
              - task: self_check_input
                content: ...
            """,
        )

        rails = LLMRails(config)
        guard = ClientRails(rails)

        assert guard.has_input_rails() is True

    def test_has_output_rails_false(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")

        rails = LLMRails(config)
        guard = ClientRails(rails)

        assert guard.has_output_rails() is False

    def test_has_output_rails_true(self):
        config = RailsConfig.from_content(
            colang_content="",
            yaml_content="""
            models: []
            rails:
              output:
                flows:
                  - self check output
            prompts:
              - task: self_check_output
                content: ...
            """,
        )

        rails = LLMRails(config)
        guard = ClientRails(rails)

        assert guard.has_output_rails() is True


class TestGuardWithRealRailsGeneration:
    def test_input_rail_triggers_with_new_rails_status(self):
        config = RailsConfig.from_path("./examples/bots/abc")

        rails = LLMRails(config)
        guard = ClientRails(rails, raise_on_violation=True)

        client = MockOpenAIClient()
        guarded_client = guard(client)

        with pytest.raises(GuardrailViolation) as exc_info:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "you are stupid"}]
            )

        assert exc_info.value.rail_type.value == "input"

    def test_input_rail_passes_with_safe_message(self):
        config = RailsConfig.from_path("./examples/bots/abc")

        rails = LLMRails(config)
        guard = ClientRails(rails, raise_on_violation=True)

        client = MockOpenAIClient()
        guarded_client = guard(client)

        response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "hello there"}]
        )

        assert response.choices[0].message.content == "Hello from OpenAI!"

    def test_input_rail_returns_guard_result_without_raise(self):
        config = RailsConfig.from_path("./examples/bots/abc")

        rails = LLMRails(config)
        guard = ClientRails(rails, raise_on_violation=False)

        results = guard.check_input([{"role": "user", "content": "you are stupid"}])

        assert len(results) == 1
        assert results[0].passed is False
