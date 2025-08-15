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
import importlib
import json
import os
import tempfile
import unittest

from nemoguardrails.eval.models import Span
from nemoguardrails.tracing import InteractionLog
from nemoguardrails.tracing.adapters.filesystem import FileSystemAdapter


class TestFileSystemAdapter(unittest.TestCase):
    def setUp(self):
        # creating a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filepath = os.path.join(self.temp_dir.name, "trace.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_default_path(self):
        adapter = FileSystemAdapter()
        self.assertEqual(adapter.filepath, "./.traces/trace.jsonl")

    def test_initialization_custom_path(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        self.assertEqual(adapter.filepath, self.filepath)
        self.assertTrue(os.path.exists(os.path.dirname(self.filepath)))

    def test_transform(self):
        adapter = FileSystemAdapter(filepath=self.filepath)

        #  Mock the InteractionLog
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    metrics={},
                )
            ],
        )

        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            self.assertEqual(log_dict["trace_id"], "test_id")
            self.assertEqual(len(log_dict["spans"]), 1)
            self.assertEqual(log_dict["spans"][0]["name"], "test_span")

    @unittest.skipIf(
        importlib.util.find_spec("aiofiles") is None, "aiofiles is not installed"
    )
    def test_transform_async(self):
        async def run_test():
            adapter = FileSystemAdapter(filepath=self.filepath)

            # Mock the InteractionLog
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    Span(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=0.0,
                        end_time=1.0,
                        duration=1.0,
                        metrics={},
                    )
                ],
            )

            await adapter.transform_async(interaction_log)

            with open(self.filepath, "r") as f:
                content = f.read()
                log_dict = json.loads(content.strip())
                self.assertEqual(log_dict["trace_id"], "test_id")
                self.assertEqual(len(log_dict["spans"]), 1)
                self.assertEqual(log_dict["spans"][0]["name"], "test_span")

        asyncio.run(run_test())

    def test_jsonl_format_single_record(self):
        """Test that output is valid JSONL format (single line per JSON object)."""
        adapter = FileSystemAdapter(filepath=self.filepath)

        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    metrics={},
                )
            ],
        )

        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 1, "Should have exactly one line")

        line = lines[0].strip()
        self.assertNotEqual(line, "", "Line should not be empty")

        log_dict = json.loads(line)
        self.assertEqual(log_dict["trace_id"], "test_id")
        self.assertEqual(len(log_dict["spans"]), 1)

        self.assertNotIn("\n", line, "JSON object should not contain embedded newlines")

    def test_jsonl_format_multiple_records(self):
        """Test that multiple records create valid JSONL with one JSON per line."""
        adapter = FileSystemAdapter(filepath=self.filepath)

        log1 = InteractionLog(
            id="trace_1",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="span_1",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    metrics={"count": 1},
                )
            ],
        )

        log2 = InteractionLog(
            id="trace_2",
            activated_rails=[],
            events=[],
            trace=[
                Span(
                    name="span_2a",
                    span_id="span_2a",
                    parent_id=None,
                    start_time=0.0,
                    end_time=0.5,
                    duration=0.5,
                    metrics={},
                ),
                Span(
                    name="span_2b",
                    span_id="span_2b",
                    parent_id="span_2a",
                    start_time=0.5,
                    end_time=1.0,
                    duration=0.5,
                    metrics={"score": 0.95},
                ),
            ],
        )

        adapter.transform(log1)
        adapter.transform(log2)

        with open(self.filepath, "r") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2, "Should have exactly two lines")

        parsed_logs = []
        for i, line in enumerate(lines):
            line = line.strip()
            self.assertNotEqual(line, "", f"Line {i + 1} should not be empty")

            log_dict = json.loads(line)
            parsed_logs.append(log_dict)

            self.assertNotIn(
                "\n", line, f"Line {i + 1} should not contain embedded newlines"
            )

        self.assertEqual(parsed_logs[0]["trace_id"], "trace_1")
        self.assertEqual(len(parsed_logs[0]["spans"]), 1)

        self.assertEqual(parsed_logs[1]["trace_id"], "trace_2")
        self.assertEqual(len(parsed_logs[1]["spans"]), 2)

    def test_jsonl_streaming_compatible(self):
        """Test that file can be processed as streaming JSONL."""
        adapter = FileSystemAdapter(filepath=self.filepath)

        for i in range(3):
            log = InteractionLog(
                id=f"trace_{i}",
                activated_rails=[],
                events=[],
                trace=[
                    Span(
                        name=f"span_{i}",
                        span_id=f"span_{i}",
                        parent_id=None,
                        start_time=float(i),
                        end_time=float(i + 1),
                        duration=1.0,
                        metrics={"index": i},
                    )
                ],
            )
            adapter.transform(log)

        trace_ids = []
        with open(self.filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        trace_ids.append(record["trace_id"])
                    except json.JSONDecodeError as e:
                        self.fail(f"Line {line_num} is not valid JSON: {e}")

        self.assertEqual(trace_ids, ["trace_0", "trace_1", "trace_2"])
        self.assertEqual(len(trace_ids), 3, "Should have processed 3 records")
