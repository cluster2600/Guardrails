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

import asyncio
import unittest
import warnings
from importlib.metadata import version
from unittest.mock import MagicMock, patch

# TODO: check to see if we can add it as a dependency
# but now we try to import opentelemetry and set a flag if it's not available
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NoOpTracerProvider

from nemoguardrails.eval.models import Span
from nemoguardrails.tracing import InteractionLog
from nemoguardrails.tracing.adapters.opentelemetry import OpenTelemetryAdapter


class TestOpenTelemetryAdapter(unittest.TestCase):
    def setUp(self):
        # Set up a mock tracer provider for testing
        self.mock_tracer_provider = MagicMock(spec=TracerProvider)
        self.mock_tracer = MagicMock()
        self.mock_tracer_provider.get_tracer.return_value = self.mock_tracer

        # Patch the global tracer provider
        patcher_get_tracer_provider = patch("opentelemetry.trace.get_tracer_provider")
        self.mock_get_tracer_provider = patcher_get_tracer_provider.start()
        self.mock_get_tracer_provider.return_value = self.mock_tracer_provider
        self.addCleanup(patcher_get_tracer_provider.stop)

        # Patch get_tracer to return our mock
        patcher_get_tracer = patch("opentelemetry.trace.get_tracer")
        self.mock_get_tracer = patcher_get_tracer.start()
        self.mock_get_tracer.return_value = self.mock_tracer
        self.addCleanup(patcher_get_tracer.stop)

        # Get the actual version for testing
        self.actual_version = version("nemoguardrails")

        # Create the adapter - it should now use the global tracer
        self.adapter = OpenTelemetryAdapter()

    def test_initialization(self):
        """Test that the adapter initializes correctly using the global tracer."""

        self.mock_get_tracer.assert_called_once_with(
            "nemo_guardrails",
            instrumenting_library_version=self.actual_version,
            schema_url="https://opentelemetry.io/schemas/1.26.0",
        )
        # Verify that the adapter has the mock tracer
        self.assertEqual(self.adapter.tracer, self.mock_tracer)

    def test_transform(self):
        """Test that transform creates spans correctly with proper timing."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.5,  # historical timestamp
                    end_time=1234567891.5,  # historical timestamp
                    duration=1.0,
                    metrics={"key": 123},
                )
            ],
        )

        self.adapter.transform(interaction_log)

        # Verify that start_span was called with proper timing (not start_as_current_span)
        self.mock_tracer.start_span.assert_called_once_with(
            "test_span",
            context=None,
            start_time=1234567890500000000,  # Converted to nanoseconds
        )

        mock_span.set_attribute.assert_any_call("key", 123)
        mock_span.set_attribute.assert_any_call("span_id", "span_1")
        mock_span.set_attribute.assert_any_call("trace_id", "test_id")
        mock_span.set_attribute.assert_any_call("duration", 1.0)

        # Verify span was ended with correct end time
        mock_span.end.assert_called_once_with(
            end_time=1234567891500000000
        )  # Converted to nanoseconds

    def test_transform_span_attributes_various_types(self):
        """Test that different attribute types are handled correctly."""
        mock_span = MagicMock()
        self.mock_tracer.start_span.return_value = mock_span

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.0,
                    end_time=1234567891.0,
                    duration=1.0,
                    metrics={
                        "int_key": 42,
                        "float_key": 3.14,
                        "str_key": 123,  # Changed to a numeric value
                        "bool_key": 1,  # Changed to a numeric value
                    },
                )
            ],
        )

        self.adapter.transform(interaction_log)

        mock_span.set_attribute.assert_any_call("int_key", 42)
        mock_span.set_attribute.assert_any_call("float_key", 3.14)
        mock_span.set_attribute.assert_any_call("str_key", 123)
        mock_span.set_attribute.assert_any_call("bool_key", 1)
        mock_span.set_attribute.assert_any_call("span_id", "span_1")
        mock_span.set_attribute.assert_any_call("trace_id", "test_id")
        mock_span.set_attribute.assert_any_call("duration", 1.0)
        mock_span.end.assert_called_once_with(end_time=1234567891000000000)

    def test_transform_with_empty_trace(self):
        """Test transform with empty trace."""
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[],
        )

        self.adapter.transform(interaction_log)

        self.mock_tracer.start_span.assert_not_called()

    def test_transform_with_tracer_failure(self):
        """Test transform when tracer fails."""
        self.mock_tracer.start_span.side_effect = Exception("Tracer failure")

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.0,
                    end_time=1234567891.0,
                    duration=1.0,
                    metrics={"key": 123},
                )
            ],
        )

        with self.assertRaises(Exception) as context:
            self.adapter.transform(interaction_log)

        self.assertIn("Tracer failure", str(context.exception))

    def test_transform_with_parent_child_relationships(self):
        """Test that parent-child relationships are preserved with correct timing."""
        parent_mock_span = MagicMock()
        child_mock_span = MagicMock()
        self.mock_tracer.start_span.side_effect = [parent_mock_span, child_mock_span]

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="parent_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=1234567890.0,
                    end_time=1234567892.0,
                    duration=2.0,
                    metrics={"parent_key": 1},
                ),
                Span(
                    name="child_span",
                    span_id="span_2",
                    parent_id="span_1",
                    start_time=1234567890.5,  # child starts after parent
                    end_time=1234567891.5,  # child ends before parent
                    duration=1.0,
                    metrics={"child_key": 2},
                ),
            ],
        )

        with patch(
            "opentelemetry.trace.set_span_in_context"
        ) as mock_set_span_in_context:
            mock_set_span_in_context.return_value = "parent_context"

            self.adapter.transform(interaction_log)

            # verify parent span created first with no context
            self.assertEqual(self.mock_tracer.start_span.call_count, 2)
            first_call = self.mock_tracer.start_span.call_args_list[0]
            self.assertEqual(first_call[0][0], "parent_span")  # name
            self.assertEqual(first_call[1]["context"], None)  # no parent context
            self.assertEqual(
                first_call[1]["start_time"], 1234567890000000000
            )  # nanoseconds

            # verify child span created with parent context
            second_call = self.mock_tracer.start_span.call_args_list[1]
            self.assertEqual(second_call[0][0], "child_span")  # name
            self.assertEqual(
                second_call[1]["context"], "parent_context"
            )  # parent context
            self.assertEqual(
                second_call[1]["start_time"], 1234567890500000000
            )  # nanoseconds

            # verify parent context was set correctly
            mock_set_span_in_context.assert_called_once_with(parent_mock_span)

            # verify both spans ended with correct times
            parent_mock_span.end.assert_called_once_with(end_time=1234567892000000000)
            child_mock_span.end.assert_called_once_with(end_time=1234567891500000000)

    def test_transform_async(self):
        """Test async transform functionality."""

        async def run_test():
            mock_span = MagicMock()
            self.mock_tracer.start_span.return_value = mock_span

            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    Span(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=1234567890.5,
                        end_time=1234567891.5,
                        duration=1.0,
                        metrics={"key": 123},
                    )
                ],
            )

            await self.adapter.transform_async(interaction_log)

            self.mock_tracer.start_span.assert_called_once_with(
                "test_span",
                context=None,
                start_time=1234567890500000000,
            )

            mock_span.set_attribute.assert_any_call("key", 123)
            mock_span.set_attribute.assert_any_call("span_id", "span_1")
            mock_span.set_attribute.assert_any_call("trace_id", "test_id")
            mock_span.set_attribute.assert_any_call("duration", 1.0)
            mock_span.end.assert_called_once_with(end_time=1234567891500000000)

        asyncio.run(run_test())

    def test_transform_async_with_empty_trace(self):
        """Test async transform with empty trace."""

        async def run_test():
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[],
            )

            await self.adapter.transform_async(interaction_log)

            self.mock_tracer.start_span.assert_not_called()

        asyncio.run(run_test())

    def test_transform_async_with_tracer_failure(self):
        """Test async transform when tracer fails."""
        self.mock_tracer.start_span.side_effect = Exception("Tracer failure")

        async def run_test():
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    Span(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=1234567890.0,
                        end_time=1234567891.0,
                        duration=1.0,
                        metrics={"key": 123},
                    )
                ],
            )

            with self.assertRaises(Exception) as context:
                await self.adapter.transform_async(interaction_log)

            self.assertIn("Tracer failure", str(context.exception))

        asyncio.run(run_test())

    def test_backward_compatibility_with_old_config(self):
        """Test that old configuration parameters are still accepted."""
        # This should not fail even if old parameters are passed
        adapter = OpenTelemetryAdapter(
            service_name="test_service",
            exporter="console",  # this should be ignored gracefully
            resource_attributes={"test": "value"},  # this should be ignored gracefully
        )

        # Should still create the adapter successfully
        self.assertIsInstance(adapter, OpenTelemetryAdapter)
        self.assertEqual(adapter.tracer, self.mock_tracer)

    def test_deprecation_warning_for_old_parameters(self):
        """Test that deprecation warnings are raised for old configuration parameters."""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # adapter with deprecated parameters
            _adapter = OpenTelemetryAdapter(
                service_name="test_service",
                exporter="console",
                resource_attributes={"test": "value"},
                span_processor=MagicMock(),
            )

            # deprecation warning is issued
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[0].message))
            self.assertIn("exporter", str(w[0].message))
            self.assertIn("resource_attributes", str(w[0].message))
            self.assertIn("span_processor", str(w[0].message))

    def test_no_op_tracer_provider_warning(self):
        """Test that a warning is issued when NoOpTracerProvider is detected."""

        with patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider:
            mock_get_provider.return_value = NoOpTracerProvider()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                _adapter = OpenTelemetryAdapter()

                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                self.assertIn(
                    "No OpenTelemetry TracerProvider configured", str(w[0].message)
                )
                self.assertIn("Traces will not be exported", str(w[0].message))

    def test_no_warnings_with_proper_configuration(self):
        """Test that no warnings are issued when properly configured."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # adapter without deprecated parameters
            _adapter = OpenTelemetryAdapter(service_name="test_service")

            # no warnings is issued
            self.assertEqual(len(w), 0)

    def test_register_otel_exporter_deprecation(self):
        """Test that register_otel_exporter shows deprecation warning."""
        from nemoguardrails.tracing.adapters.opentelemetry import register_otel_exporter

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            mock_exporter_cls = MagicMock()

            register_otel_exporter("test-exporter", mock_exporter_cls)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("register_otel_exporter is deprecated", str(w[0].message))
            self.assertIn("0.16.0", str(w[0].message))

            from nemoguardrails.tracing.adapters.opentelemetry import (
                _exporter_name_cls_map,
            )

            self.assertEqual(_exporter_name_cls_map["test-exporter"], mock_exporter_cls)
