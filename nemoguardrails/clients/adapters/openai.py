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

"""OpenAI provider adapter.

This module implements the ProviderAdapter for OpenAI's SDK, handling
extraction of input/output from OpenAI-specific formats using composition.
"""

from typing import Any, List, Optional

from ..types import RailsMessage, ToolCall
from .base import ProviderAdapter
from .extractors.content import extract_text_content
from .extractors.format import extract
from .extractors.input import ParameterExtractor


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI SDK (openai>=1.0.0).

    Supports:
    - chat.completions.create() with messages parameter
    - responses.create() with input parameter (string or message array)
    - completions.create() with prompt parameter (string or list)

    Handles both Chat Completions response format (choices[].message)
    and Responses API format (output with polymorphic items).

    Uses composition pattern with extractors for clean separation of concerns.
    """

    def __init__(self, validate_messages: bool = False):
        super().__init__(validate_messages)
        self.input_extractors = [
            ParameterExtractor("messages", self._messages_converter),
            ParameterExtractor("input", self._input_converter),
            ParameterExtractor("prompt", self._prompt_converter),
        ]

    def _messages_converter(self, value: Any) -> Optional[List[RailsMessage]]:
        if isinstance(value, list):
            return self._convert_messages_to_rails_format(value)
        return None

    def _input_converter(self, value: Any) -> Optional[List[RailsMessage]]:
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        elif isinstance(value, list):
            return self._convert_messages_to_rails_format(value)
        return None

    def _prompt_converter(self, value: Any) -> Optional[List[RailsMessage]]:
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        elif isinstance(value, list):
            messages_from_prompts = []
            for p in value:
                if isinstance(p, str):
                    messages_from_prompts.append({"role": "user", "content": p})
            return messages_from_prompts if messages_from_prompts else None
        return None

    def extract_messages_for_input_check(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> List[RailsMessage]:
        """Extract full conversation messages for input rails checking.

        Converts OpenAI message format to NemoGuardrails format, preserving
        system messages, user messages, assistant messages with tool_calls,
        and tool messages.

        Supports multiple input formats:
        - messages: List of message dicts (Chat Completions API)
        - input: String or list of messages (Responses API)
        - prompt: String or list of strings (Completions API)

        Args:
            method_name: Method being called (e.g., "create")
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of messages in NemoGuardrails format
        """
        if method_name == "create":
            for extractor in self.input_extractors:
                result = extractor(method_name, *args, **kwargs)
                if result is not None:
                    self._validate_if_enabled(result)
                    return result

            if args and isinstance(args[0], list):
                result = self._convert_messages_to_rails_format(args[0])
                self._validate_if_enabled(result)
                return result

        return []

    def _convert_messages_to_rails_format(
        self, messages: list[Any]
    ) -> List[RailsMessage]:
        """Convert OpenAI messages to NemoGuardrails format using universal extractor.

        Args:
            messages: List of OpenAI message dicts or objects

        Returns:
            List of RailsMessage objects
        """
        rails_messages: List[RailsMessage] = []

        for msg in messages:
            role = extract(msg, "role")
            content = extract(msg, "content")
            tool_calls = extract(msg, "tool_calls")
            tool_call_id = extract(msg, "tool_call_id")

            if role in ("user", "assistant", "system", "tool"):
                content_str = extract_text_content(content)
                rails_msg: RailsMessage = {
                    "role": role,
                    "content": content_str,
                }

                if role == "assistant" and tool_calls:
                    rails_msg["tool_calls"] = self._convert_tool_calls(tool_calls)

                if role == "tool" and tool_call_id:
                    rails_msg["tool_call_id"] = str(tool_call_id)

                rails_messages.append(rails_msg)

        return rails_messages

    def _convert_tool_calls(self, tool_calls: Any) -> List[ToolCall]:
        """Convert OpenAI tool calls to NemoGuardrails format using universal extractor.

        Args:
            tool_calls: Tool calls from OpenAI message

        Returns:
            List of ToolCall objects
        """
        if not tool_calls:
            return []

        converted: List[ToolCall] = []
        for tc in tool_calls:
            tc_id = extract(tc, "id", "")
            function = extract(tc, "function")

            if function:
                name = extract(function, "name", "")
                arguments = extract(function, "arguments", {})
            else:
                name = extract(tc, "name", "")
                arguments = extract(tc, "args", {})

            converted.append(
                {
                    "name": str(name),
                    "args": arguments,
                    "id": str(tc_id),
                    "type": "tool_call",
                }
            )

        return converted

    def extract_messages_for_output_check(
        self, method_name: str, response: Any, input_messages: List[RailsMessage]
    ) -> List[RailsMessage]:
        """Extract conversation + response for output rails checking.

        Args:
            method_name: Method being called
            response: OpenAI response object
            input_messages: Original input messages

        Returns:
            List of messages including input and bot response
        """
        assistant_message = self._extract_assistant_message(response)
        if assistant_message:
            return input_messages + [assistant_message]
        return input_messages

    def _extract_assistant_message(self, response: Any) -> RailsMessage:
        """Extract assistant message from OpenAI response using universal extractor.

        Handles both Chat Completions and Responses API formats.

        Args:
            response: OpenAI response object

        Returns:
            Assistant message in RailsMessage format
        """
        content = ""
        tool_calls = []

        choices = extract(response, "choices")
        if choices:
            first_choice = choices[0]
            message = extract(first_choice, "message")
            if message:
                content = str(extract(message, "content", "") or "")
                message_tool_calls = extract(message, "tool_calls")
                if message_tool_calls:
                    tool_calls = self._convert_tool_calls(message_tool_calls)
        else:
            output = extract(response, "output")
            if output and isinstance(output, list):
                texts = []
                output_tool_calls = []
                for item in output:
                    item_type = extract(item, "type")
                    if item_type in ("message", "text"):
                        text_content = extract(item, "text") or extract(
                            item, "content", ""
                        )
                        if text_content:
                            texts.append(str(text_content))
                    elif item_type in ("function_call", "tool_use", "tool_call"):
                        output_tool_calls.append(item)

                content = " ".join(texts)
                if output_tool_calls:
                    tool_calls = self._convert_tool_calls(output_tool_calls)

        assistant_msg: RailsMessage = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        return assistant_msg

    def get_intercept_paths(self) -> list[str]:
        """Get OpenAI resource paths to intercept.

        Returns:
            List of attribute names to wrap: ["chat", "responses", "completions"]
        """
        return ["chat", "responses", "completions"]

    def get_provider_name(self) -> str:
        """Get provider name.

        Returns:
            "openai"
        """
        return "openai"
