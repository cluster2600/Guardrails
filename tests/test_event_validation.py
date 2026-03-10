# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import sys

import pytest


def _import_utils():
    """Import nemoguardrails.utils directly, bypassing the top-level package __init__."""
    spec = importlib.util.spec_from_file_location(
        "nemoguardrails.utils",
        "nemoguardrails/utils.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


utils = _import_utils()
ensure_valid_event = utils.ensure_valid_event
is_valid_event = utils.is_valid_event


def _make_base_event(event_type, **extra):
    """Build a minimal event dict that satisfies common validators."""
    event = {
        "type": event_type,
        "uid": "test-uid",
        "event_created_at": "2026-01-01T00:00:00+00:00",
        "source_uid": "test",
        "action_uid": "test-action-uid",
    }
    event.update(extra)
    return event


class TestUtteranceBotActionScriptUpdatedValidation:
    """Tests for UtteranceBotActionScriptUpdated event validation."""

    def test_valid_event_with_interim_script(self):
        """An UtteranceBotActionScriptUpdated event with interim_script should pass."""
        event = _make_base_event(
            "UtteranceBotActionScriptUpdated",
            interim_script="Hello there",
        )
        assert is_valid_event(event) is True

    def test_missing_interim_script_is_invalid(self):
        """An UtteranceBotActionScriptUpdated event without interim_script should fail."""
        event = _make_base_event("UtteranceBotActionScriptUpdated")
        assert is_valid_event(event) is False

    def test_missing_interim_script_raises(self):
        """ensure_valid_event should raise AssertionError when interim_script is missing."""
        event = _make_base_event("UtteranceBotActionScriptUpdated")
        with pytest.raises(AssertionError, match="interim_script"):
            ensure_valid_event(event)

    def test_interim_script_wrong_type_is_invalid(self):
        """An UtteranceBotActionScriptUpdated event with non-str interim_script should fail."""
        event = _make_base_event(
            "UtteranceBotActionScriptUpdated",
            interim_script=42,
        )
        assert is_valid_event(event) is False
