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

"""Tests for LLM parameter mapping integration in llm_call function."""

from unittest.mock import AsyncMock, Mock

import pytest

from nemoguardrails.actions.llm.utils import llm_call


class MockResponse:
    """Mock response object."""

    def __init__(self, content="Test response"):
        self.content = content


class MockHuggingFaceLLM:
    """Mock HuggingFace LLM for testing parameter mapping."""

    __module__ = "langchain_huggingface.llms"

    def __init__(self):
        self.model_name = "test-model"
        self.bind = Mock(return_value=self)
        self.ainvoke = AsyncMock(return_value=MockResponse())


@pytest.mark.asyncio
async def test_llm_call_with_registered_parameter_mapping():
    """Test llm_call applies registered parameter mapping correctly."""
    from nemoguardrails.llm.parameter_mapping import register_llm_parameter_mapping

    mock_llm = MockHuggingFaceLLM()
    register_llm_parameter_mapping(
        "huggingface", "test-model", {"max_tokens": "max_new_tokens"}
    )

    result = await llm_call(
        llm=mock_llm,
        prompt="Test prompt",
        llm_params={"max_tokens": 100, "temperature": 0.5},
    )

    mock_llm.bind.assert_called_once_with(max_new_tokens=100, temperature=0.5)
    assert result == "Test response"


@pytest.mark.asyncio
async def test_llm_call_with_builtin_mapping():
    """Test llm_call uses built-in provider mapping when no custom mapping provided."""
    mock_llm = MockHuggingFaceLLM()

    result = await llm_call(
        llm=mock_llm,
        prompt="Test prompt",
        llm_params={"max_tokens": 50, "temperature": 0.7},
    )

    mock_llm.bind.assert_called_once_with(max_new_tokens=50, temperature=0.7)
    assert result == "Test response"


@pytest.mark.asyncio
async def test_llm_call_with_dropped_parameter():
    """Test llm_call drops parameters mapped to None."""
    from nemoguardrails.llm.parameter_mapping import register_llm_parameter_mapping

    mock_llm = MockHuggingFaceLLM()
    register_llm_parameter_mapping(
        "huggingface",
        "test-model",
        {"max_tokens": "max_new_tokens", "unsupported_param": None},
    )

    result = await llm_call(
        llm=mock_llm,
        prompt="Test prompt",
        llm_params={"max_tokens": 100, "unsupported_param": "value"},
    )

    mock_llm.bind.assert_called_once_with(max_new_tokens=100)
    assert result == "Test response"


@pytest.mark.asyncio
async def test_llm_call_without_params():
    """Test llm_call works without llm_params."""
    mock_llm = MockHuggingFaceLLM()

    result = await llm_call(llm=mock_llm, prompt="Test prompt")

    mock_llm.bind.assert_not_called()
    mock_llm.ainvoke.assert_called_once()
    assert result == "Test response"


@pytest.mark.asyncio
async def test_llm_call_with_stop_tokens():
    """Test llm_call handles stop tokens correctly with parameter mapping."""
    from nemoguardrails.llm.parameter_mapping import register_llm_parameter_mapping

    mock_llm = MockHuggingFaceLLM()
    register_llm_parameter_mapping(
        "huggingface", "test-model", {"max_tokens": "max_new_tokens"}
    )

    result = await llm_call(
        llm=mock_llm,
        prompt="Test prompt",
        stop=["END", "STOP"],
        llm_params={"max_tokens": 100},
    )

    mock_llm.bind.assert_called_once_with(stop=["END", "STOP"], max_new_tokens=100)
    assert result == "Test response"
