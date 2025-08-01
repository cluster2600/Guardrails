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

import time
from typing import List

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nemoguardrails.eval.models import Span
from nemoguardrails.tracing import InteractionLog
from nemoguardrails.tracing.adapters.opentelemetry import OpenTelemetryAdapter


class InMemorySpanExporter:
    """Simple in-memory span exporter to capture spans for testing."""

    def __init__(self):
        self.spans: List = []

    def export(self, spans):
        self.spans.extend(spans)
        return 0  # Success

    def shutdown(self):
        pass


class TestOpenTelemetryTimingBehavior:
    """
    Test that verifies OpenTelemetry spans are created with correct timestamps.

    This test focuses on the ACTUAL timing behavior, not implementation details.
    It will fail with the old broken code (retrospective timing) and pass with
    the fixed code (historical timing).
    """

    def setup_method(self):
        self.exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))

        trace._TRACER_PROVIDER = None
        trace.set_tracer_provider(self.tracer_provider)

        self.adapter = OpenTelemetryAdapter()

    def teardown_method(self):
        # Clean up - reset to no-op
        trace._TRACER_PROVIDER = None
        trace.set_tracer_provider(trace.NoOpTracerProvider())

    def test_spans_use_historical_timestamps_not_current_time(self):
        """
        Test that spans are created with historical timestamps from span_data,
        not with the current time when transform() is called.

        This test will:
        - FAIL with old broken code (uses current time)
        - PASS with fixed code (uses historical time)
        """
        historical_start = 1234567890.5  # January 1, 2009
        historical_end = 1234567892.0  # 1.5 seconds later

        interaction_log = InteractionLog(
            id="timing_test",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="historical_operation",
                    span_id="span_1",
                    parent_id=None,
                    start_time=historical_start,
                    end_time=historical_end,
                    duration=1.5,
                    metrics={"test_metric": 42},
                )
            ],
        )

        current_time_before = time.time()

        self.adapter.transform(interaction_log)

        current_time_after = time.time()

        assert len(self.exporter.spans) == 1
        captured_span = self.exporter.spans[0]

        actual_start_time = captured_span.start_time / 1_000_000_000
        actual_end_time = captured_span.end_time / 1_000_000_000

        assert (
            abs(actual_start_time - historical_start) < 0.001
        ), f"Span start time ({actual_start_time}) should match historical time ({historical_start})"

        assert (
            abs(actual_end_time - historical_end) < 0.001
        ), f"Span end time ({actual_end_time}) should match historical time ({historical_end})"

        time_diff_start = abs(actual_start_time - current_time_before)
        time_diff_end = abs(actual_end_time - current_time_after)

        assert time_diff_start > 1000000, (
            f"Span start time should be very different from current time. "
            f"Difference: {time_diff_start} seconds. This suggests the old bug is present."
        )

        assert time_diff_end > 1000000, (
            f"Span end time should be very different from current time. "
            f"Difference: {time_diff_end} seconds. This suggests the old bug is present."
        )
        actual_duration = actual_end_time - actual_start_time
        expected_duration = historical_end - historical_start
        assert (
            abs(actual_duration - expected_duration) < 0.001
        ), f"Span duration should be {expected_duration}s, got {actual_duration}s"

        assert captured_span.name == "historical_operation"
        assert captured_span.attributes.get("test_metric") == 42
        assert captured_span.attributes.get("span_id") == "span_1"
        assert captured_span.attributes.get("trace_id") == "timing_test"
