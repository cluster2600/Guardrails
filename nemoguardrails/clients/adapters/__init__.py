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

"""Provider adapters for multi-provider LLM support.

This package contains adapter implementations for different LLM providers,
enabling the LLMRails wrapper to work with OpenAI, Anthropic, and others.
"""

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .factory import ProviderAdapterFactory
from .genai import GenaiAdapter
from .openai import OpenAIAdapter

__all__ = [
    "ProviderAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GenaiAdapter",
    "ProviderAdapterFactory",
]
