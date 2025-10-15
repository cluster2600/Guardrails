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

from nemoguardrails.clients.validators import MessageValidator, ValidationError


class TestMessageValidator:
    def setup_method(self):
        self.validator = MessageValidator()

    def test_validate_simple_user_message(self):
        message = {"role": "user", "content": "Hello"}
        self.validator.validate(message)

    def test_validate_assistant_message(self):
        message = {"role": "assistant", "content": "Hi there"}
        self.validator.validate(message)

    def test_validate_system_message(self):
        message = {"role": "system", "content": "You are helpful"}
        self.validator.validate(message)

    def test_validate_tool_message(self):
        message = {"role": "tool", "content": "Result", "tool_call_id": "call_123"}
        self.validator.validate(message)

    def test_validate_assistant_with_tool_calls(self):
        message = {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {
                    "name": "get_weather",
                    "args": {"city": "NYC"},
                    "id": "call_456",
                    "type": "tool_call",
                }
            ],
        }
        self.validator.validate(message)

    def test_missing_role_raises(self):
        message = {"content": "Hello"}
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "missing 'role' field" in str(exc_info.value)

    def test_invalid_role_raises(self):
        message = {"role": "invalid", "content": "Hello"}
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "Invalid role" in str(exc_info.value)

    def test_missing_content_raises(self):
        message = {"role": "user"}
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "missing 'content' field" in str(exc_info.value)

    def test_tool_message_missing_tool_call_id_raises(self):
        message = {"role": "tool", "content": "Result"}
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "missing 'tool_call_id'" in str(exc_info.value)

    def test_tool_calls_not_list_raises(self):
        message = {"role": "assistant", "content": "Test", "tool_calls": "not_a_list"}
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "tool_calls must be a list" in str(exc_info.value)

    def test_tool_call_not_dict_raises(self):
        message = {"role": "assistant", "content": "Test", "tool_calls": ["not_a_dict"]}
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "must be a dict" in str(exc_info.value)

    def test_tool_call_missing_name_raises(self):
        message = {
            "role": "assistant",
            "content": "Test",
            "tool_calls": [{"args": {}, "id": "call_123", "type": "tool_call"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "missing 'name' field" in str(exc_info.value)

    def test_tool_call_missing_args_raises(self):
        message = {
            "role": "assistant",
            "content": "Test",
            "tool_calls": [{"name": "test", "id": "call_123", "type": "tool_call"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "missing 'args' field" in str(exc_info.value)

    def test_tool_call_missing_id_raises(self):
        message = {
            "role": "assistant",
            "content": "Test",
            "tool_calls": [{"name": "test", "args": {}, "type": "tool_call"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "missing 'id' field" in str(exc_info.value)

    def test_tool_call_invalid_type_raises(self):
        message = {
            "role": "assistant",
            "content": "Test",
            "tool_calls": [
                {"name": "test", "args": {}, "id": "call_123", "type": "wrong_type"}
            ],
        }
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate(message)
        assert "invalid type" in str(exc_info.value)

    def test_validate_messages_list(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        self.validator.validate_messages(messages)

    def test_validate_messages_list_with_error(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "invalid", "content": "Bad"},
        ]
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_messages(messages)
        assert "Message at index 1" in str(exc_info.value)

    def test_validate_empty_messages_list(self):
        messages = []
        self.validator.validate_messages(messages)

    def test_validate_complex_conversation(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "args": {"city": "SF"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            },
            {"role": "tool", "content": "Sunny, 72F", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "It's sunny and 72F"},
        ]
        self.validator.validate_messages(messages)
