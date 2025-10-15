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

"""Google Genai (Gemini) provider adapter.

This module implements the ProviderAdapter for Google's Genai SDK (Gemini),
handling extraction of input/output from Google-specific formats using composition.
"""

from typing import Any, List, Optional

from ..types import RailsMessage, ToolCall
from .base import ProviderAdapter
from .extractors.content import extract_text_content
from .extractors.format import extract
from .extractors.input import ParameterExtractor


class GenaiAdapter(ProviderAdapter):
    """Adapter for Google Genai SDK (google-genai>=1.0.0).

    Supports:
    - models.generate_content() with contents parameter

    The Gemini API uses 'contents' parameter which can be:
    - A string (simple prompt)
    - A list (multimodal: text + images/files)
    - Structured with roles

    Uses composition pattern with extractors for clean separation of concerns.
    """

    def __init__(self, validate_messages: bool = False):
        super().__init__(validate_messages)
        self.input_extractors = [
            ParameterExtractor("contents", self._contents_converter),
        ]

    def _contents_converter(self, value: Any) -> Optional[List[RailsMessage]]:
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        elif isinstance(value, list):
            return self._convert_contents_to_rails_format(value)
        return None

    def extract_messages_for_input_check(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> List[RailsMessage]:
        """Extract full conversation messages for input rails checking.

        Google Genai uses 'contents' parameter for input.

        Args:
            method_name: Method being called (e.g., "generate_content")
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of messages in NemoGuardrails format
        """
        if method_name == "generate_content":
            for extractor in self.input_extractors:
                extracted = extractor(method_name, *args, **kwargs)
                if extracted is not None:
                    self._validate_if_enabled(extracted)
                    return extracted

            if args and args[0]:
                contents = args[0]
                if isinstance(contents, str):
                    result: List[RailsMessage] = [{"role": "user", "content": contents}]
                    self._validate_if_enabled(result)
                    return result
                elif isinstance(contents, list):
                    result = self._convert_contents_to_rails_format(contents)
                    self._validate_if_enabled(result)
                    return result

        return []

    def _convert_contents_to_rails_format(
        self, contents: list[Any]
    ) -> List[RailsMessage]:
        """Convert Google Genai contents to NemoGuardrails format using universal extractor.

        Args:
            contents: List of content items (strings or structured objects)

        Returns:
            List of RailsMessage objects
        """
        rails_messages: List[RailsMessage] = []

        for item in contents:
            if isinstance(item, str):
                rails_messages.append({"role": "user", "content": item})
            else:
                role = extract(item, "role", "user")
                parts = extract(item, "parts")

                if parts:
                    content_str, tool_calls = self._extract_from_parts(parts)
                    rails_msg: RailsMessage = {
                        "role": role,
                        "content": content_str,
                    }
                    if tool_calls:
                        rails_msg["tool_calls"] = tool_calls
                    rails_messages.append(rails_msg)
                else:
                    content = extract(item, "content", "")
                    rails_messages.append(
                        {
                            "role": role,
                            "content": str(content),
                        }
                    )

        return rails_messages

    def _extract_from_parts(self, parts: list[Any]) -> tuple[str, List[ToolCall]]:
        """Extract text and function calls from Gemini parts using universal extractor.

        Args:
            parts: List of part objects from Gemini response

        Returns:
            Tuple of (concatenated_text, tool_calls)
        """
        texts = []
        tool_calls: List[ToolCall] = []

        for part in parts:
            text = extract(part, "text")
            if text:
                texts.append(str(text))

            function_call = extract(part, "function_call")
            if function_call:
                name = extract(function_call, "name", "")
                args = extract(function_call, "args", {})
                call_id = extract(function_call, "id", name)

                if name:
                    tool_calls.append(
                        {
                            "name": str(name),
                            "args": args or {},
                            "id": str(call_id),
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
            response: Google Genai response object
            input_messages: Original input messages

        Returns:
            List of messages including input and bot response
        """
        assistant_message = self._extract_assistant_message(response)
        if assistant_message:
            return input_messages + [assistant_message]
        return input_messages

    def _extract_assistant_message(self, response: Any) -> RailsMessage:
        """Extract assistant message from Google Genai response using universal extractor.

        Google Genai response structure:
            response.candidates[0].content.parts[0].text

        Args:
            response: Google Genai response object

        Returns:
            Assistant message in RailsMessage format
        """
        content = ""
        tool_calls: List[ToolCall] = []

        candidates = extract(response, "candidates")
        if candidates and len(candidates) > 0:
            first_candidate = candidates[0]
            candidate_content = extract(first_candidate, "content")

            if candidate_content:
                parts = extract(candidate_content, "parts")
                if parts:
                    content, tool_calls = self._extract_from_parts(parts)

        assistant_msg: RailsMessage = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        return assistant_msg

    def get_intercept_paths(self) -> list[str]:
        """Get Google Genai resource paths to intercept.

        Returns:
            List of attribute names to wrap: ["models"]
        """
        return ["models"]

    def get_provider_name(self) -> str:
        """Get provider name.

        Returns:
            "genai"
        """
        return "genai"

    def should_wrap_method(self, method_name: str) -> bool:
        """Check if a method should be wrapped with guardrails.

        Args:
            method_name: Name of the method

        Returns:
            True if method should be wrapped, False otherwise
        """
        return method_name == "generate_content"
