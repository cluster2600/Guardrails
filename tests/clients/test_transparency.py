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


class CustomProviderError(Exception):
    pass


class NetworkError(Exception):
    pass


class RateLimitError(Exception):
    pass


class MockOpenAIClient:
    def __init__(self, behavior="normal"):
        self.behavior = behavior
        self.chat = MockChat(behavior)
        self.custom_attr = "preserved"

    def direct_method(self):
        return "direct_method_result"


class MockChat:
    def __init__(self, behavior):
        self.behavior = behavior
        self.completions = MockCompletions(behavior)


class MockCompletions:
    def __init__(self, behavior):
        self.behavior = behavior

    def create(self, messages, **kwargs):
        if self.behavior == "auth_error":
            raise CustomProviderError("Authentication failed: Invalid API key")
        elif self.behavior == "network_error":
            raise NetworkError("Connection timeout")
        elif self.behavior == "rate_limit":
            raise RateLimitError("Rate limit exceeded")
        elif self.behavior == "normal":
            last_msg = messages[-1] if messages else {}
            content = last_msg.get("content", "")
            return MockResponse(f"Echo: {content}")
        elif self.behavior == "streaming":
            return iter([MockResponse("chunk1"), MockResponse("chunk2")])
        elif self.behavior == "with_metadata":
            resp = MockResponse("Response with metadata")
            resp.metadata = {"tokens": 42, "model": "gpt-4"}
            return resp
        return MockResponse("Default response")

    async def acreate(self, messages, **kwargs):
        return self.create(messages, **kwargs)


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]
        self.id = "resp-123"
        self.model = "gpt-4"

    def __eq__(self, other):
        if not isinstance(other, MockResponse):
            return False
        return (
            self.choices[0].message.content == other.choices[0].message.content
            and self.id == other.id
        )


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


MockOpenAIClient.__module__ = "openai"


class TestTransparencyDropInReplacement:
    def setup_method(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")
        rails = LLMRails(config)
        self.guard = ClientRails(rails)

    def test_identical_successful_response(self):
        original_client = MockOpenAIClient("normal")
        guarded_client = self.guard(original_client)

        original_response = original_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}]
        )
        guarded_response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert (
            original_response.choices[0].message.content
            == guarded_response.choices[0].message.content
        )
        assert original_response.id == guarded_response.id
        assert original_response.model == guarded_response.model

    def test_identical_exception_auth_error(self):
        original_client = MockOpenAIClient("auth_error")
        guarded_client = self.guard(original_client)

        original_error = None
        guarded_error = None

        try:
            original_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
        except Exception as e:
            original_error = e

        try:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
        except Exception as e:
            guarded_error = e

        assert type(original_error) == type(guarded_error)
        assert str(original_error) == str(guarded_error)
        assert original_error.__class__.__name__ == "CustomProviderError"
        assert guarded_error.__class__.__name__ == "CustomProviderError"

    def test_identical_exception_network_error(self):
        original_client = MockOpenAIClient("network_error")
        guarded_client = self.guard(original_client)

        with pytest.raises(NetworkError) as original_exc:
            original_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )

        with pytest.raises(NetworkError) as guarded_exc:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )

        assert str(original_exc.value) == str(guarded_exc.value)

    def test_identical_exception_rate_limit(self):
        original_client = MockOpenAIClient("rate_limit")
        guarded_client = self.guard(original_client)

        with pytest.raises(RateLimitError) as original_exc:
            original_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )

        with pytest.raises(RateLimitError) as guarded_exc:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )

        assert type(original_exc.value) == type(guarded_exc.value)

    def test_preserves_response_attributes(self):
        original_client = MockOpenAIClient("with_metadata")
        guarded_client = self.guard(original_client)

        original_response = original_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}]
        )
        guarded_response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert hasattr(guarded_response, "metadata")
        assert guarded_response.metadata == original_response.metadata
        assert guarded_response.metadata["tokens"] == 42

    def test_preserves_client_attributes(self):
        original_client = MockOpenAIClient("normal")
        guarded_client = self.guard(original_client)

        assert guarded_client.custom_attr == original_client.custom_attr
        assert guarded_client.direct_method() == "direct_method_result"

    def test_identical_with_kwargs(self):
        original_client = MockOpenAIClient("normal")
        guarded_client = self.guard(original_client)

        original_response = original_client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )

        guarded_response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )

        assert (
            original_response.choices[0].message.content
            == guarded_response.choices[0].message.content
        )

    @pytest.mark.asyncio
    async def test_identical_async_behavior(self):
        original_client = MockOpenAIClient("normal")
        guarded_client = self.guard(original_client)

        original_response = await original_client.chat.completions.acreate(
            messages=[{"role": "user", "content": "Async test"}]
        )

        guarded_response = await guarded_client.chat.completions.acreate(
            messages=[{"role": "user", "content": "Async test"}]
        )

        assert (
            original_response.choices[0].message.content
            == guarded_response.choices[0].message.content
        )


class TestTransparencyWithGuardrails:
    def setup_method(self):
        config = RailsConfig.from_path("./examples/bots/abc")
        rails = LLMRails(config)
        self.guard = ClientRails(rails, raise_on_violation=True)

    def test_guardrail_exception_is_only_new_exception(self):
        original_client = MockOpenAIClient("normal")
        guarded_client = self.guard(original_client)

        original_exception_raised = False
        try:
            original_client.chat.completions.create(
                messages=[{"role": "user", "content": "you are stupid"}]
            )
        except Exception:
            original_exception_raised = True

        assert not original_exception_raised

        with pytest.raises(GuardrailViolation) as exc_info:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "you are stupid"}]
            )

        assert exc_info.value.rail_type.value == "input"
        assert (
            "refuse to respond" in exc_info.value.message.lower()
            or "self_check_input" in exc_info.value.message.lower()
        )

    def test_provider_error_still_passes_through_with_guardrails(self):
        original_client = MockOpenAIClient("auth_error")
        guarded_client = self.guard(original_client)

        with pytest.raises(CustomProviderError) as original_exc:
            original_client.chat.completions.create(
                messages=[{"role": "user", "content": "Safe message"}]
            )

        with pytest.raises(CustomProviderError) as guarded_exc:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "Safe message"}]
            )

        assert str(original_exc.value) == str(guarded_exc.value)

    def test_safe_message_returns_identical_response(self):
        original_client = MockOpenAIClient("normal")
        guarded_client = self.guard(original_client)

        original_response = original_client.chat.completions.create(
            messages=[{"role": "user", "content": "hello there"}]
        )

        guarded_response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "hello there"}]
        )

        assert (
            original_response.choices[0].message.content
            == guarded_response.choices[0].message.content
        )


class TestTransparencyExceptionChaining:
    def setup_method(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")
        rails = LLMRails(config)
        self.guard = ClientRails(rails)

    def test_exception_stack_trace_preserves_provider_info(self):
        client = MockOpenAIClient("auth_error")
        guarded_client = self.guard(client)

        try:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
            pytest.fail("Should have raised CustomProviderError")
        except CustomProviderError as e:
            import traceback

            tb = traceback.format_exc()
            assert "CustomProviderError" in tb
            assert "Authentication failed" in str(e)

    def test_no_wrapper_exceptions_in_stack(self):
        client = MockOpenAIClient("network_error")
        guarded_client = self.guard(client)

        try:
            guarded_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}]
            )
            pytest.fail("Should have raised NetworkError")
        except NetworkError as e:
            assert e.__class__.__name__ == "NetworkError"
            assert not isinstance(e, GuardrailViolation)


class TestTransparencyResponseIdentity:
    def setup_method(self):
        config = RailsConfig.from_content(colang_content="", yaml_content="models: []")
        rails = LLMRails(config)
        self.guard = ClientRails(rails)

    def test_response_is_identical_object(self):
        client = MockOpenAIClient("normal")
        guarded_client = self.guard(client)

        response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert isinstance(response, MockResponse)
        assert hasattr(response, "choices")
        assert hasattr(response, "id")
        assert response.id == "resp-123"

    def test_response_not_wrapped_or_modified(self):
        client = MockOpenAIClient("normal")
        guarded_client = self.guard(client)

        response = guarded_client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert type(response).__name__ == "MockResponse"
        assert not hasattr(response, "_wrapped")
        assert not hasattr(response, "_original")
