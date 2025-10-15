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

"""Anthropic provider adapter.

This module implements the ProviderAdapter for Anthropic's SDK, handling
extraction of input/output from Anthropic-specific formats using composition.
"""

from typing import Any, List, Optional

from ..types import RailsMessage, ToolCall
from .base import ProviderAdapter
from .extractors.format import extract
from .extractors.input import ParameterExtractor


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic SDK (anthropic>=0.18.0).

    Supports:
    - messages.create()

    Uses composition pattern with extractors for clean separation of concerns.
    """

    def __init__(self, validate_messages: bool = False):
        super().__init__(validate_messages)
        self.input_extractors = [
            ParameterExtractor("messages", self._messages_converter),
        ]

    def _messages_converter(self, value: Any) -> Optional[List[RailsMessage]]:
        if isinstance(value, list):
            return self._convert_messages_to_rails_format(value)
        return None

    def extract_messages_for_input_check(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> List[RailsMessage]:
        """Extract full conversation messages for input rails checking.

        Anthropic has a special format where system messages are passed
        as a separate parameter, not in the messages list.

        Args:
            method_name: Method being called (e.g., "create")
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of messages in NemoGuardrails format
        """
        if method_name != "create":
            return []

        rails_messages: List[RailsMessage] = []

        system_content = kwargs.get("system")
        if system_content:
            rails_messages.append(
                {
                    "role": "system",
                    "content": str(system_content),
                }
            )

        for extractor in self.input_extractors:
            result = extractor(method_name, *args, **kwargs)
            if result is not None:
                rails_messages.extend(result)
                self._validate_if_enabled(rails_messages)
                return rails_messages

        if args and isinstance(args[0], list):
            rails_messages.extend(self._convert_messages_to_rails_format(args[0]))

        self._validate_if_enabled(rails_messages)
        return rails_messages

    def _convert_messages_to_rails_format(
        self, messages: list[Any]
    ) -> List[RailsMessage]:
        """Convert Anthropic messages to NemoGuardrails format using universal extractor.

        Args:
            messages: List of Anthropic message dicts or objects

        Returns:
            List of RailsMessage objects
        """
        rails_messages: List[RailsMessage] = []

        for msg in messages:
            role = extract(msg, "role")
            content = extract(msg, "content")

            if role in ("user", "assistant"):
                content_str, tool_calls = self._extract_content_and_tool_calls(content)

                rails_msg: RailsMessage = {
                    "role": role,
                    "content": content_str,
                }

                if tool_calls:
                    rails_msg["tool_calls"] = tool_calls

                rails_messages.append(rails_msg)

        return rails_messages

    def _extract_content_and_tool_calls(
        self, content: Any
    ) -> tuple[str, List[ToolCall]]:
        """Extract text content and tool calls from Anthropic content using universal extractor.

        Anthropic content can be string or list of content blocks.
        Tool use blocks are converted to tool_calls format.

        Args:
            content: Content from Anthropic message

        Returns:
            Tuple of (text_content, tool_calls)
        """
        if isinstance(content, str):
            return (content, [])
        elif isinstance(content, list):
            return self._extract_from_content_blocks(content)
        else:
            return ("", [])

    def _extract_from_content_blocks(
        self, blocks: list[Any]
    ) -> tuple[str, List[ToolCall]]:
        """Extract text and tool calls from Anthropic content blocks using universal extractor.

        Args:
            blocks: List of content block dicts or objects

        Returns:
            Tuple of (concatenated_text, tool_calls)
        """
        texts = []
        tool_calls: List[ToolCall] = []

        for block in blocks:
            block_type = extract(block, "type")

            if block_type == "text":
                text = extract(block, "text")
                if text:
                    texts.append(str(text))
            elif block_type == "tool_use":
                tool_use_id = extract(block, "id")
                tool_name = extract(block, "name")
                tool_input = extract(block, "input")

                if tool_use_id and tool_name:
                    tool_calls.append(
                        {
                            "name": str(tool_name),
                            "args": tool_input or {},
                            "id": str(tool_use_id),
                            "type": "tool_call",
                        }
                    )

        return (" ".join(texts), tool_calls)

    def extract_messages_for_output_check(
        self, method_name: str, response: Any, input_messages: List[RailsMessage]
    ) -> List[RailsMessage]:
        """Extract conversation + response for output rails checking.

        Args:
            method_name: Method being called
            response: Anthropic response object
            input_messages: Original input messages

        Returns:
            List of messages including input and bot response
        """
        assistant_message = self._extract_assistant_message(response)
        if assistant_message:
            return input_messages + [assistant_message]
        return input_messages

    def _extract_assistant_message(self, response: Any) -> RailsMessage:
        """Extract assistant message from Anthropic response using universal extractor.

        Args:
            response: Anthropic response object

        Returns:
            Assistant message in RailsMessage format
        """
        content = ""
        tool_calls: List[ToolCall] = []

        response_content = extract(response, "content")
        if response_content:
            if isinstance(response_content, list):
                content, tool_calls = self._extract_from_content_blocks(
                    response_content
                )
            elif isinstance(response_content, str):
                content = response_content

        assistant_msg: RailsMessage = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        return assistant_msg

    def get_intercept_paths(self) -> list[str]:
        """Get Anthropic resource paths to intercept.

        Returns:
            List of attribute names to wrap: ["messages"]
        """
        return ["messages"]

    def get_provider_name(self) -> str:
        """Get provider name.

        Returns:
            "anthropic"
        """
        return "anthropic"
