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

from nemoguardrails.clients.adapters.anthropic import AnthropicAdapter
from nemoguardrails.clients.adapters.genai import GenaiAdapter
from nemoguardrails.clients.adapters.openai import OpenAIAdapter


class TestOpenAIAdapter:
    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_extract_messages_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello!"}]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_extract_messages_with_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello!"

    def test_extract_messages_with_assistant_and_tool_calls(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "NYC"},
                        },
                    }
                ],
            },
        ]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["name"] == "get_weather"
        assert result[1]["tool_calls"][0]["args"] == {"city": "NYC"}
        assert result[1]["tool_calls"][0]["id"] == "call_123"

    def test_extract_messages_with_tool_response(self):
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "It's sunny",
                "tool_call_id": "call_123",
            },
        ]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 3
        assert result[2]["role"] == "tool"
        assert result[2]["content"] == "It's sunny"
        assert result[2]["tool_call_id"] == "call_123"

    def test_extract_messages_with_content_parts(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 1
        assert result[0]["content"] == "Hello World"

    def test_extract_output_messages_simple(self):
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()

        class MockMessage:
            def __init__(self):
                self.content = "Hi there!"
                self.tool_calls = None

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Hello"}]

        result = self.adapter.extract_messages_for_output_check(
            "create", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there!"

    def test_extract_output_messages_with_tool_calls(self):
        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()

        class MockMessage:
            def __init__(self):
                self.content = ""
                self.tool_calls = [MockToolCall()]

        class MockToolCall:
            def __init__(self):
                self.id = "call_456"
                self.function = MockFunction()

        class MockFunction:
            def __init__(self):
                self.name = "search"
                self.arguments = {"query": "test"}

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Search for test"}]

        result = self.adapter.extract_messages_for_output_check(
            "create", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[1]["tool_calls"][0]["name"] == "search"

    def test_empty_messages(self):
        result = self.adapter.extract_messages_for_input_check("create", messages=[])
        assert result == []

    def test_get_intercept_paths(self):
        paths = self.adapter.get_intercept_paths()
        assert "chat" in paths
        assert "responses" in paths
        assert "completions" in paths

    def test_get_provider_name(self):
        assert self.adapter.get_provider_name() == "openai"

    def test_extract_messages_with_input_string(self):
        result = self.adapter.extract_messages_for_input_check(
            "create", input="Write a story"
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Write a story"

    def test_extract_messages_with_input_array(self):
        input_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = self.adapter.extract_messages_for_input_check(
            "create", input=input_messages
        )

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_extract_messages_with_prompt_string(self):
        result = self.adapter.extract_messages_for_input_check(
            "create", prompt="Complete this sentence"
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Complete this sentence"

    def test_extract_messages_with_prompt_list(self):
        prompts = ["First prompt", "Second prompt"]
        result = self.adapter.extract_messages_for_input_check("create", prompt=prompts)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First prompt"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Second prompt"

    def test_extract_assistant_message_from_responses_api_output(self):
        class MockResponse:
            def __init__(self):
                self.output = [
                    {"type": "text", "text": "Here is the result"},
                    {"type": "message", "content": "Additional info"},
                ]

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Query"}]

        result = self.adapter.extract_messages_for_output_check(
            "create", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "Here is the result" in result[1]["content"]
        assert "Additional info" in result[1]["content"]

    def test_extract_assistant_message_with_tool_calls_from_responses_api(self):
        class MockResponse:
            def __init__(self):
                self.output = [
                    {"type": "text", "text": "Let me call a function"},
                    {
                        "type": "function_call",
                        "id": "call_999",
                        "function": {
                            "name": "calculator",
                            "arguments": {"expression": "2+2"},
                        },
                    },
                ]

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Calculate 2+2"}]

        result = self.adapter.extract_messages_for_output_check(
            "create", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["name"] == "calculator"


class TestAnthropicAdapter:
    def setup_method(self):
        self.adapter = AnthropicAdapter()

    def test_extract_messages_simple(self):
        messages = [{"role": "user", "content": "Hello!"}]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_extract_messages_with_system_param(self):
        messages = [{"role": "user", "content": "Hello!"}]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages, system="You are helpful."
        )

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"

    def test_extract_messages_with_content_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 1
        assert result[0]["content"] == "Hello World"

    def test_extract_messages_with_tool_use(self):
        messages = [
            {"role": "user", "content": "Check weather"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check"},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"city": "SF"},
                    },
                ],
            },
        ]
        result = self.adapter.extract_messages_for_input_check(
            "create", messages=messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Let me check"
        assert "tool_calls" in result[1]
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["name"] == "get_weather"
        assert result[1]["tool_calls"][0]["args"] == {"city": "SF"}
        assert result[1]["tool_calls"][0]["id"] == "tool_123"

    def test_extract_output_messages_simple(self):
        class MockResponse:
            def __init__(self):
                self.content = [{"type": "text", "text": "Hi there!"}]

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Hello"}]

        result = self.adapter.extract_messages_for_output_check(
            "create", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there!"

    def test_extract_output_messages_with_tool_use(self):
        class MockResponse:
            def __init__(self):
                self.content = [
                    {"type": "text", "text": "Checking"},
                    {
                        "type": "tool_use",
                        "id": "tool_789",
                        "name": "search",
                        "input": {"query": "test"},
                    },
                ]

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Search"}]

        result = self.adapter.extract_messages_for_output_check(
            "create", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Checking"
        assert "tool_calls" in result[1]
        assert result[1]["tool_calls"][0]["name"] == "search"

    def test_empty_messages(self):
        result = self.adapter.extract_messages_for_input_check("create", messages=[])
        assert len(result) == 0

    def test_get_intercept_paths(self):
        paths = self.adapter.get_intercept_paths()
        assert "messages" in paths

    def test_get_provider_name(self):
        assert self.adapter.get_provider_name() == "anthropic"


class TestGenaiAdapter:
    def setup_method(self):
        self.adapter = GenaiAdapter()

    def test_extract_messages_with_string_contents(self):
        result = self.adapter.extract_messages_for_input_check(
            "generate_content", contents="What is AI?"
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What is AI?"

    def test_extract_messages_with_list_contents(self):
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there!"}]},
        ]
        result = self.adapter.extract_messages_for_input_check(
            "generate_content", contents=contents
        )

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "model"
        assert result[1]["content"] == "Hi there!"

    def test_extract_messages_with_function_call(self):
        contents = [
            {
                "role": "model",
                "parts": [
                    {"text": "Let me check"},
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"city": "NYC"},
                        }
                    },
                ],
            }
        ]
        result = self.adapter.extract_messages_for_input_check(
            "generate_content", contents=contents
        )

        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert "Let me check" in result[0]["content"]
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["name"] == "get_weather"
        assert result[0]["tool_calls"][0]["args"] == {"city": "NYC"}

    def test_extract_assistant_message(self):
        class MockResponse:
            def __init__(self):
                self.candidates = [MockCandidate()]

        class MockCandidate:
            def __init__(self):
                self.content = MockContent()

        class MockContent:
            def __init__(self):
                self.parts = [MockPart()]

        class MockPart:
            def __init__(self):
                self.text = "Hello from Gemini"

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Hi"}]

        result = self.adapter.extract_messages_for_output_check(
            "generate_content", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hello from Gemini"

    def test_extract_assistant_message_with_function_call(self):
        class MockResponse:
            def __init__(self):
                self.candidates = [MockCandidate()]

        class MockCandidate:
            def __init__(self):
                self.content = MockContent()

        class MockContent:
            def __init__(self):
                self.parts = [MockTextPart(), MockFunctionCallPart()]

        class MockTextPart:
            def __init__(self):
                self.text = "Let me help"
                self.function_call = None

        class MockFunctionCallPart:
            def __init__(self):
                self.text = None
                self.function_call = MockFunctionCall()

        class MockFunctionCall:
            def __init__(self):
                self.name = "search_web"
                self.args = {"query": "AI"}
                self.id = "fc_123"

        response = MockResponse()
        input_messages = [{"role": "user", "content": "Search for AI"}]

        result = self.adapter.extract_messages_for_output_check(
            "generate_content", response, input_messages
        )

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "Let me help" in result[1]["content"]
        assert "tool_calls" in result[1]
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["name"] == "search_web"

    def test_get_intercept_paths(self):
        paths = self.adapter.get_intercept_paths()
        assert "models" in paths

    def test_get_provider_name(self):
        assert self.adapter.get_provider_name() == "genai"

    def test_should_wrap_method(self):
        assert self.adapter.should_wrap_method("generate_content") == True
        assert self.adapter.should_wrap_method("other_method") == False

    def test_empty_contents(self):
        result = self.adapter.extract_messages_for_input_check(
            "generate_content", contents=[]
        )
        assert result == []
