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

"""Tests for LLM parameter mapping functionality."""

from unittest.mock import Mock

import pytest

from nemoguardrails.llm.parameter_mapping import (
    PROVIDER_PARAMETER_MAPPINGS,
    get_llm_provider,
    register_llm_parameter_mapping,
    transform_llm_params,
)


class MockAnthropicLLM:
    """Mock Anthropic LLM for testing."""

    __module__ = "langchain_anthropic.chat_models"
    model_name = "claude-3"


class MockHuggingFacePipeline:
    """Mock HuggingFace LLM for testing."""

    __module__ = "langchain_huggingface.llms"
    model_name = "gpt2"


class MockGoogleVertexAI:
    """Mock Google LLM for testing."""

    __module__ = "langchain_google_vertexai.chat_models"
    model_name = "gemini-pro"


class MockUnknownLLM:
    """Mock unknown LLM for testing."""

    __module__ = "some_unknown_module"
    model_name = "unknown-model"


def test_infer_provider_from_llm_anthropic():
    """Test provider inference for Anthropic models."""
    llm = MockAnthropicLLM()
    provider = get_llm_provider(llm)
    assert provider == "anthropic"


def test_infer_provider_from_llm_huggingface():
    """Test provider inference for HuggingFace models."""
    llm = MockHuggingFacePipeline()
    provider = get_llm_provider(llm)
    assert provider == "huggingface"


def test_infer_provider_from_llm_google():
    """Test provider inference for Google models."""
    llm = MockGoogleVertexAI()
    provider = get_llm_provider(llm)
    assert provider == "google_vertexai"


def test_infer_provider_from_llm_unknown():
    """Test provider inference for unknown models."""
    llm = MockUnknownLLM()
    provider = get_llm_provider(llm)
    assert provider is None


def test_transform_llm_params_empty():
    """Test transformation with empty parameters."""
    result = transform_llm_params({})
    assert result == {}


def test_transform_llm_params_none():
    """Test transformation with None parameters."""
    result = transform_llm_params(None)
    assert result is None


def test_transform_llm_params_custom_mapping():
    """Test transformation with custom parameter mapping."""
    params = {"max_tokens": 100, "temperature": 0.7, "top_p": 0.9}
    mapping = {"max_tokens": "max_new_tokens", "temperature": "temp", "top_p": None}

    result = transform_llm_params(params, parameter_mapping=mapping)

    expected = {"max_new_tokens": 100, "temp": 0.7}
    assert result == expected


def test_transform_llm_params_anthropic_builtin():
    """Test transformation with built-in Anthropic mapping."""
    llm = MockAnthropicLLM()
    params = {"max_tokens": 100, "temperature": 0.7}
    provider = get_llm_provider(llm)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)

    expected = {"max_tokens": 100, "temperature": 0.7}
    assert result == expected


def test_transform_llm_params_huggingface_builtin():
    """Test transformation with built-in HuggingFace mapping."""
    llm = MockHuggingFacePipeline()
    params = {"max_tokens": 50, "temperature": 0.5, "top_p": 0.8}
    provider = get_llm_provider(llm)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)

    expected = {"max_new_tokens": 50, "temperature": 0.5, "top_p": 0.8}
    assert result == expected


def test_transform_llm_params_google_builtin():
    """Test transformation with built-in Google mapping."""
    llm = MockGoogleVertexAI()
    params = {"max_tokens": 200, "temperature": 1.0}
    provider = get_llm_provider(llm)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)

    expected = {"max_output_tokens": 200, "temperature": 1.0}
    assert result == expected


def test_transform_llm_params_unknown_provider():
    """Test transformation with unknown provider returns unchanged params."""
    llm = MockUnknownLLM()
    params = {"max_tokens": 100, "temperature": 0.7}
    provider = get_llm_provider(llm)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)

    assert result == params


def test_transform_llm_params_partial_mapping():
    """Test transformation with partial parameter mapping."""
    params = {"max_tokens": 100, "temperature": 0.7, "top_p": 0.9, "stop": ["END"]}
    mapping = {"max_tokens": "max_length"}

    result = transform_llm_params(params, parameter_mapping=mapping)

    expected = {"max_length": 100, "temperature": 0.7, "top_p": 0.9, "stop": ["END"]}
    assert result == expected


def test_transform_llm_params_drop_parameter():
    """Test dropping parameters by mapping to None."""
    params = {"max_tokens": 100, "temperature": 0.7, "unsupported_param": "value"}
    mapping = {"unsupported_param": None}

    result = transform_llm_params(params, parameter_mapping=mapping)

    expected = {"max_tokens": 100, "temperature": 0.7}
    assert result == expected


def test_provider_parameter_mappings_structure():
    """Test that provider mappings have expected structure."""
    assert "huggingface" in PROVIDER_PARAMETER_MAPPINGS
    assert "google_vertexai" in PROVIDER_PARAMETER_MAPPINGS

    assert PROVIDER_PARAMETER_MAPPINGS["huggingface"]["max_tokens"] == "max_new_tokens"
    assert (
        PROVIDER_PARAMETER_MAPPINGS["google_vertexai"]["max_tokens"]
        == "max_output_tokens"
    )


def test_custom_mapping_overrides_builtin():
    """Test that custom mapping overrides built-in provider mapping."""
    params = {"max_tokens": 100}
    custom_mapping = {"max_tokens": "custom_max_tokens"}

    result = transform_llm_params(params, parameter_mapping=custom_mapping)

    expected = {"custom_max_tokens": 100}
    assert result == expected


def test_registered_mapping_used_in_transform():
    """Test that registered mapping is used automatically in transform_llm_params."""
    llm = MockUnknownLLM()
    params = {"max_tokens": 100, "temperature": 0.7}
    mapping = {"max_tokens": "max_length", "temperature": "temp"}
    provider = get_llm_provider(llm)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)
    assert result == params

    register_llm_parameter_mapping(provider, llm.model_name, mapping)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)
    expected = {"max_length": 100, "temp": 0.7}
    assert result == expected


def test_registered_mapping_overrides_builtin():
    """Test that registered mapping overrides built-in provider mapping."""
    llm = MockHuggingFacePipeline()
    params = {"max_tokens": 100}
    provider = get_llm_provider(llm)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)
    assert result == {"max_new_tokens": 100}

    custom_mapping = {"max_tokens": "custom_max_tokens"}
    register_llm_parameter_mapping(provider, llm.model_name, custom_mapping)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)
    assert result == {"custom_max_tokens": 100}


def test_infer_provider_community_models():
    """Test provider inference for community models."""

    class MockCommunityOllama:
        __module__ = "langchain_community.chat_models.ollama"

    class MockCommunityGooglePalm:
        __module__ = "langchain_community.chat_models.google_palm"

    ollama_llm = MockCommunityOllama()
    provider = get_llm_provider(ollama_llm)
    assert provider == "ollama"

    palm_llm = MockCommunityGooglePalm()
    provider = get_llm_provider(palm_llm)
    assert provider == "google_palm"


def test_infer_provider_google_variants():
    """Test provider inference for different Google provider variants."""

    class MockGoogleGenAI:
        __module__ = "langchain_google_genai.chat_models"

    class MockGoogleVertexAI:
        __module__ = "langchain_google_vertexai.chat_models"

    genai_llm = MockGoogleGenAI()
    provider = get_llm_provider(genai_llm)
    assert provider == "google_genai"

    vertexai_llm = MockGoogleVertexAI()
    provider = get_llm_provider(vertexai_llm)
    assert provider == "google_vertexai"


def test_transform_params_google_genai():
    """Test parameter transformation for google_genai provider."""

    class MockGoogleGenAI:
        __module__ = "langchain_google_genai.chat_models"

    llm = MockGoogleGenAI()
    params = {"max_tokens": 150, "temperature": 0.8}

    result = transform_llm_params(params, llm)

    expected = {"max_tokens": 150, "temperature": 0.8}
    assert result == expected


def test_transform_params_google_vertexai():
    """Test parameter transformation for google_vertexai provider."""

    class MockGoogleVertexAI:
        __module__ = "langchain_google_vertexai.chat_models"
        model_name = "gemini-pro"

    llm = MockGoogleVertexAI()
    params = {"max_tokens": 200, "temperature": 0.9}
    provider = get_llm_provider(llm)

    result = transform_llm_params(params, provider=provider, model_name=llm.model_name)

    expected = {"max_output_tokens": 200, "temperature": 0.9}
    assert result == expected
