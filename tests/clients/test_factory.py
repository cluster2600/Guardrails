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
from nemoguardrails.clients.adapters.base import ProviderAdapter
from nemoguardrails.clients.adapters.factory import ProviderAdapterFactory
from nemoguardrails.clients.adapters.openai import OpenAIAdapter


class MockOpenAIClient:
    pass


class MockAnthropicClient:
    pass


class MockUnknownClient:
    pass


MockOpenAIClient.__module__ = "openai.resources.chat.completions"
MockAnthropicClient.__module__ = "anthropic.resources.messages"
MockUnknownClient.__module__ = "unknown_provider.client"


class TestProviderDetection:
    def test_detect_openai_provider(self):
        client = MockOpenAIClient()
        provider = ProviderAdapterFactory.detect_provider(client)
        assert provider == "openai"

    def test_detect_anthropic_provider(self):
        client = MockAnthropicClient()
        provider = ProviderAdapterFactory.detect_provider(client)
        assert provider == "anthropic"

    def test_detect_unknown_provider_raises(self):
        client = MockUnknownClient()
        with pytest.raises(ValueError) as exc_info:
            ProviderAdapterFactory.detect_provider(client)

        assert "Unknown provider" in str(exc_info.value)


class TestAdapterCreation:
    def test_create_openai_adapter(self):
        adapter = ProviderAdapterFactory.create_adapter("openai")
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.get_provider_name() == "openai"

    def test_create_anthropic_adapter(self):
        adapter = ProviderAdapterFactory.create_adapter("anthropic")
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter.get_provider_name() == "anthropic"

    def test_create_unknown_adapter_raises(self):
        with pytest.raises(ValueError) as exc_info:
            ProviderAdapterFactory.create_adapter("unknown")

        assert "No adapter for provider" in str(exc_info.value)


class TestAdapterForClient:
    def test_create_adapter_for_openai_client(self):
        client = MockOpenAIClient()
        adapter = ProviderAdapterFactory.create_adapter_for_client(client)

        assert isinstance(adapter, OpenAIAdapter)

    def test_create_adapter_for_anthropic_client(self):
        client = MockAnthropicClient()
        adapter = ProviderAdapterFactory.create_adapter_for_client(client)

        assert isinstance(adapter, AnthropicAdapter)

    def test_create_adapter_for_unknown_client_raises(self):
        client = MockUnknownClient()

        with pytest.raises(ValueError):
            ProviderAdapterFactory.create_adapter_for_client(client)


class TestCustomAdapterRegistration:
    def test_register_custom_adapter(self):
        class CustomAdapter(ProviderAdapter):
            def extract_messages_for_input_check(self, method_name, *args, **kwargs):
                return []

            def extract_messages_for_output_check(
                self, method_name, response, input_messages
            ):
                return []

            def extract_output(self, response):
                return ""

            def get_intercept_paths(self):
                return ["custom"]

            def get_provider_name(self):
                return "custom"

        ProviderAdapterFactory.register_adapter("custom", CustomAdapter, ["custom_llm"])

        assert "custom" in ProviderAdapterFactory.get_supported_providers()

        adapter = ProviderAdapterFactory.create_adapter("custom")
        assert isinstance(adapter, CustomAdapter)


class TestGetSupportedProviders:
    def test_get_supported_providers(self):
        providers = ProviderAdapterFactory.get_supported_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert len(providers) >= 2
