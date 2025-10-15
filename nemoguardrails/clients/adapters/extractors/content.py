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

"""Content parsing utilities for extracting text from various formats.

This module provides reusable parsers for extracting text content from
different message content formats (string, list of parts, content blocks, etc.).
"""

from typing import Any

from .format import extract


def extract_text_from_string(content: Any) -> str:
    """Extract text from string content.

    Args:
        content: Content value

    Returns:
        Content as string, or empty string if not a string
    """
    if isinstance(content, str):
        return content
    return ""


def extract_text_from_parts(content: Any, text_types: tuple = ("text",)) -> str:
    """Extract text from list of content parts.

    Handles multimodal content where text is in parts with type indicators.

    Args:
        content: List of content parts
        text_types: Tuple of type values that indicate text content

    Returns:
        Concatenated text from all text parts
    """
    if not isinstance(content, list):
        return ""

    texts = []
    for part in content:
        part_type = extract(part, "type")
        if part_type in text_types:
            text = extract(part, "text") or extract(part, "content", "")
            if text:
                texts.append(str(text))

    return " ".join(texts)


def extract_text_content(content: Any, text_types: tuple = ("text",)) -> str:
    """Universal text extraction from string or parts.

    Args:
        content: Content in any format (string, list, etc.)
        text_types: Tuple of type values that indicate text content

    Returns:
        Extracted text as string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return extract_text_from_parts(content, text_types)
    elif content is None:
        return ""
    else:
        return str(content)
