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

"""Type definitions for multi-provider guarded client.

This module defines the message types used for communication between
adapters and the guardrails system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, TypedDict


class RailType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class ToolCall(TypedDict):
    """Structure for tool call information in assistant messages.

    Attributes:
        name: Name of the tool/function being called
        args: Dictionary of arguments passed to the tool
        id: Unique identifier for this tool call
        type: Type discriminator, always "tool_call"
    """

    name: str
    args: Dict[str, Any]
    id: str
    type: Literal["tool_call"]


class RailsMessage(TypedDict, total=False):
    """Message format for NemoGuardrails input/output checking.

    This format is compatible with OpenAI's message format and supports
    the full conversation history including tool calls.

    Attributes:
        role: Message role (user, assistant, system, tool)
        content: Message content as string
        tool_calls: List of tool calls (only for assistant messages)
        tool_call_id: Tool call ID (only for tool messages)

    Examples:
        User message:
            {"role": "user", "content": "Hello!"}

        Assistant message with tool call:
            {
                "role": "assistant",
                "content": "I'll check the weather.",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "NYC"},
                        "id": "call_123",
                        "type": "tool_call"
                    }
                ]
            }

        Tool message:
            {
                "role": "tool",
                "content": "It's sunny in NYC",
                "tool_call_id": "call_123"
            }

        System message:
            {"role": "system", "content": "You are a helpful assistant."}
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: List[ToolCall]
    tool_call_id: str


@dataclass
class GuardResult:
    """Result from running a guardrail check."""

    passed: bool
    reason: str = ""
    guard_name: str = ""
