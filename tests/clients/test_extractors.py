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

from nemoguardrails.clients.adapters.extractors.content import (
    extract_text_content,
    extract_text_from_parts,
    extract_text_from_string,
)
from nemoguardrails.clients.adapters.extractors.format import extract
from nemoguardrails.clients.adapters.extractors.input import ParameterExtractor


class TestExtract:
    def test_extract_from_dict(self):
        data = {"name": "test", "value": 42}
        assert extract(data, "name") == "test"
        assert extract(data, "value") == 42

    def test_extract_from_object(self):
        class TestObj:
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = TestObj()
        assert extract(obj, "name") == "test"
        assert extract(obj, "value") == 42

    def test_extract_missing_key_returns_default(self):
        data = {"name": "test"}
        assert extract(data, "missing") is None
        assert extract(data, "missing", "default") == "default"

    def test_extract_none_value(self):
        data = {"value": None}
        assert extract(data, "value") is None

    def test_extract_from_none(self):
        assert extract(None, "key") is None
        assert extract(None, "key", "default") == "default"

    def test_extract_nested_dict(self):
        data = {"user": {"name": "Alice", "age": 30}}
        user = extract(data, "user")
        assert extract(user, "name") == "Alice"

    def test_extract_nested_object(self):
        class Address:
            def __init__(self):
                self.city = "NYC"

        class User:
            def __init__(self):
                self.address = Address()

        user = User()
        address = extract(user, "address")
        assert extract(address, "city") == "NYC"


class TestParameterExtractor:
    def test_extract_existing_parameter(self):
        def converter(value):
            return [{"role": "user", "content": value}]

        extractor = ParameterExtractor("prompt", converter)
        result = extractor("create", prompt="Hello")

        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_extract_missing_parameter_returns_none(self):
        def converter(value):
            return [{"role": "user", "content": value}]

        extractor = ParameterExtractor("prompt", converter)
        result = extractor("create", other_param="value")

        assert result is None

    def test_converter_returns_none(self):
        def converter(value):
            return None

        extractor = ParameterExtractor("prompt", converter)
        result = extractor("create", prompt="Hello")

        assert result is None

    def test_extract_with_positional_args(self):
        def converter(value):
            return [{"role": "user", "content": value}]

        extractor = ParameterExtractor("messages", converter)
        result = extractor("create", "arg1", "arg2", messages="Hello")

        assert len(result) == 1
        assert result[0]["content"] == "Hello"


class TestContentExtraction:
    def test_extract_text_from_string(self):
        assert extract_text_from_string("Hello") == "Hello"
        assert extract_text_from_string(123) == ""
        assert extract_text_from_string(None) == ""

    def test_extract_text_from_parts_with_text_type(self):
        parts = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert extract_text_from_parts(parts) == "Hello World"

    def test_extract_text_from_parts_ignores_non_text_types(self):
        parts = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "http://example.com/img.jpg"},
            {"type": "text", "text": "World"},
        ]
        assert extract_text_from_parts(parts) == "Hello World"

    def test_extract_text_from_parts_empty_list(self):
        assert extract_text_from_parts([]) == ""

    def test_extract_text_from_parts_non_list(self):
        assert extract_text_from_parts("not a list") == ""

    def test_extract_text_content_from_string(self):
        assert extract_text_content("Hello") == "Hello"

    def test_extract_text_content_from_parts(self):
        parts = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert extract_text_content(parts) == "Hello World"

    def test_extract_text_content_from_none(self):
        assert extract_text_content(None) == ""

    def test_extract_text_content_from_number(self):
        assert extract_text_content(42) == "42"

    def test_extract_text_from_parts_with_content_field(self):
        parts = [
            {"type": "text", "content": "Hello"},
            {"type": "text", "content": "World"},
        ]
        assert extract_text_from_parts(parts) == "Hello World"

    def test_extract_text_content_custom_text_types(self):
        parts = [
            {"type": "message", "text": "Hello"},
            {"type": "text", "text": "Ignored"},
        ]
        result = extract_text_content(parts, text_types=("message",))
        assert result == "Hello"
