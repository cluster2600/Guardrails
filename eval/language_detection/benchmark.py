# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SUPPORTED_LANGUAGES = frozenset({"en", "es", "zh", "de", "fr", "hi", "ja", "ar", "th"})


def detect_with_fast_langdetect(text: str) -> Optional[str]:
    try:
        from fast_langdetect import detect

        result = detect(text, k=1)
        if result and len(result) > 0:
            return result[0].get("lang")
        return None
    except Exception as e:
        print(f"fast-langdetect error: {e}")
        return None


def detect_with_lingua(text: str) -> Optional[str]:
    try:
        from lingua import Language, LanguageDetectorBuilder

        languages = [
            Language.ENGLISH,
            Language.SPANISH,
            Language.CHINESE,
            Language.GERMAN,
            Language.FRENCH,
            Language.HINDI,
            Language.JAPANESE,
            Language.ARABIC,
            Language.THAI,
        ]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        detected = detector.detect_language_of(text)

        if detected:
            lang_map = {
                "ENGLISH": "en",
                "SPANISH": "es",
                "CHINESE": "zh",
                "GERMAN": "de",
                "FRENCH": "fr",
                "HINDI": "hi",
                "JAPANESE": "ja",
                "ARABIC": "ar",
                "THAI": "th",
            }
            return lang_map.get(detected.name)
        return None
    except Exception as e:
        print(f"lingua error: {e}")
        return None


def normalize_lang_code(lang: Optional[str]) -> Optional[str]:
    if lang is None:
        return None
    lang = lang.lower().split("-")[0]
    if lang == "cmn":
        return "zh"
    return lang


def warmup(mode: str = "direct"):
    print("Warming up models...")
    warmup_texts = [
        "Hello, how are you?",
        "Bonjour, comment allez-vous?",
        "你好，你今天怎么样？",
    ]
    for text in warmup_texts:
        detect_with_fast_langdetect(text)
        if mode == "direct":
            detect_with_lingua(text)
    print("Warm up complete.\n")


async def detect_with_action(text: str, dispatcher, custom_messages: Optional[dict] = None):
    from nemoguardrails.rails.llm.config import ContentSafetyConfig, MultilingualConfig

    multilingual_config = MultilingualConfig(enabled=True, refusal_messages=custom_messages)
    content_safety_config = ContentSafetyConfig(multilingual=multilingual_config)

    class MockRailsConfig:
        def __init__(self):
            self.config = type("obj", (object,), {"content_safety": content_safety_config})()

    class MockConfig:
        def __init__(self):
            self.rails = MockRailsConfig()

    params = {
        "context": {"user_message": text},
        "config": MockConfig(),
    }

    result, status = await dispatcher.execute_action("detect_language", params)
    if status == "success" and result:
        return result.get("language")
    return None


def run_benchmark(
    dataset_path: Path,
    max_samples: Optional[int] = None,
    report_dir: Optional[Path] = None,
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if max_samples and len(dataset) > max_samples:
        import random

        random.seed(42)
        dataset = random.sample(dataset, max_samples)

    warmup(mode="direct")

    dataset_name = dataset_path.stem.replace("_filtered", "")
    print(f"Dataset: {dataset_path.name}")
    print(f"Loaded {len(dataset)} samples")
    print("=" * 80)

    results = {
        "fast_langdetect": {"correct": 0, "total": 0, "latencies": [], "errors": [], "by_lang": {}},
        "lingua": {"correct": 0, "total": 0, "latencies": [], "errors": [], "by_lang": {}},
    }

    for lang in SUPPORTED_LANGUAGES:
        for name in results:
            results[name]["by_lang"][lang] = {"correct": 0, "total": 0}

    for item in dataset:
        text = item["text"]
        expected = item["language"]
        length = item.get("length", "unknown")

        start = time.perf_counter()
        fast_result = detect_with_fast_langdetect(text)
        fast_latency = (time.perf_counter() - start) * 1_000_000

        start = time.perf_counter()
        lingua_result = detect_with_lingua(text)
        lingua_latency = (time.perf_counter() - start) * 1_000_000

        fast_normalized = normalize_lang_code(fast_result)
        lingua_normalized = normalize_lang_code(lingua_result)

        results["fast_langdetect"]["total"] += 1
        results["fast_langdetect"]["latencies"].append(fast_latency)
        results["fast_langdetect"]["by_lang"][expected]["total"] += 1
        if fast_normalized == expected:
            results["fast_langdetect"]["correct"] += 1
            results["fast_langdetect"]["by_lang"][expected]["correct"] += 1
        else:
            results["fast_langdetect"]["errors"].append(
                {
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "expected": expected,
                    "got": fast_normalized,
                    "length": length,
                }
            )

        results["lingua"]["total"] += 1
        results["lingua"]["latencies"].append(lingua_latency)
        results["lingua"]["by_lang"][expected]["total"] += 1
        if lingua_normalized == expected:
            results["lingua"]["correct"] += 1
            results["lingua"]["by_lang"][expected]["correct"] += 1
        else:
            results["lingua"]["errors"].append(
                {
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "expected": expected,
                    "got": lingua_normalized,
                    "length": length,
                }
            )

    print_results(results)

    if report_dir:
        write_csv_reports(results, report_dir, dataset_name, "direct", len(dataset))


async def run_benchmark_action(
    dataset_path: Path,
    max_samples: Optional[int] = None,
    report_dir: Optional[Path] = None,
):
    from nemoguardrails.actions.action_dispatcher import ActionDispatcher

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if max_samples and len(dataset) > max_samples:
        import random

        random.seed(42)
        dataset = random.sample(dataset, max_samples)

    print("Initializing ActionDispatcher...")
    dispatcher = ActionDispatcher(load_all_actions=True)

    warmup(mode="action")

    dataset_name = dataset_path.stem.replace("_filtered", "")
    print(f"Dataset: {dataset_path.name}")
    print(f"Loaded {len(dataset)} samples")
    print("=" * 80)

    results = {
        "detect_language_action": {"correct": 0, "total": 0, "latencies": [], "errors": [], "by_lang": {}},
    }

    for lang in SUPPORTED_LANGUAGES:
        for name in results:
            results[name]["by_lang"][lang] = {"correct": 0, "total": 0}

    for item in dataset:
        text = item["text"]
        expected = item["language"]
        length = item.get("length", "unknown")

        start = time.perf_counter()
        action_result = await detect_with_action(text, dispatcher)
        action_latency = (time.perf_counter() - start) * 1_000_000

        action_normalized = normalize_lang_code(action_result)

        results["detect_language_action"]["total"] += 1
        results["detect_language_action"]["latencies"].append(action_latency)
        results["detect_language_action"]["by_lang"][expected]["total"] += 1
        if action_normalized == expected:
            results["detect_language_action"]["correct"] += 1
            results["detect_language_action"]["by_lang"][expected]["correct"] += 1
        else:
            results["detect_language_action"]["errors"].append(
                {
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "expected": expected,
                    "got": action_normalized,
                    "length": length,
                }
            )

    print_results(results)

    if report_dir:
        write_csv_reports(results, report_dir, dataset_name, "action", len(dataset))


def compute_latency_stats(latencies: list) -> dict:
    if not latencies:
        return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
    s = pd.Series(latencies)
    return {
        "avg": s.mean(),
        "min": s.min(),
        "max": s.max(),
        "p50": s.quantile(0.50),
        "p95": s.quantile(0.95),
    }


def write_csv_reports(
    results: dict,
    report_dir: Path,
    dataset_name: str,
    mode: str,
    total_samples: int,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    summary_path = report_dir / "summary.csv"
    per_language_path = report_dir / "per_language.csv"
    errors_path = report_dir / "errors.csv"

    summary_rows = []
    for library, data in results.items():
        stats = compute_latency_stats(data["latencies"])
        accuracy = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        summary_rows.append(
            {
                "timestamp": timestamp,
                "dataset": dataset_name,
                "samples": total_samples,
                "mode": mode,
                "library": library,
                "correct": data["correct"],
                "total": data["total"],
                "accuracy_pct": round(accuracy, 2),
                "avg_latency_us": round(stats["avg"], 2),
                "min_latency_us": round(stats["min"], 2),
                "max_latency_us": round(stats["max"], 2),
                "p50_latency_us": round(stats["p50"], 2),
                "p95_latency_us": round(stats["p95"], 2),
                "error_count": len(data["errors"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    per_language_rows = []
    for library, data in results.items():
        for lang in sorted(data["by_lang"].keys()):
            lang_data = data["by_lang"][lang]
            accuracy = (lang_data["correct"] / lang_data["total"] * 100) if lang_data["total"] > 0 else 0
            per_language_rows.append(
                {
                    "timestamp": timestamp,
                    "dataset": dataset_name,
                    "mode": mode,
                    "library": library,
                    "language": lang,
                    "correct": lang_data["correct"],
                    "total": lang_data["total"],
                    "accuracy_pct": round(accuracy, 2),
                }
            )

    per_language_df = pd.DataFrame(per_language_rows)
    if per_language_path.exists():
        existing_df = pd.read_csv(per_language_path)
        per_language_df = pd.concat([existing_df, per_language_df], ignore_index=True)
    per_language_df.to_csv(per_language_path, index=False)

    error_rows = []
    for library, data in results.items():
        for err in data["errors"]:
            error_rows.append(
                {
                    "timestamp": timestamp,
                    "dataset": dataset_name,
                    "mode": mode,
                    "library": library,
                    "expected": err["expected"],
                    "got": err["got"],
                    "text_length": err["length"],
                    "text_sample": err["text"],
                }
            )

    if error_rows:
        errors_df = pd.DataFrame(error_rows)
        if errors_path.exists():
            existing_df = pd.read_csv(errors_path)
            errors_df = pd.concat([existing_df, errors_df], ignore_index=True)
        errors_df.to_csv(errors_path, index=False)

    print(f"\nReports written to {report_dir}/")
    print("  - summary.csv")
    print("  - per_language.csv")
    print("  - errors.csv")


def print_results(results: dict):
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for name, data in results.items():
        accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        avg_latency = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0
        min_latency = min(data["latencies"]) if data["latencies"] else 0
        max_latency = max(data["latencies"]) if data["latencies"] else 0
        p50 = sorted(data["latencies"])[len(data["latencies"]) // 2] if data["latencies"] else 0
        p95_idx = int(len(data["latencies"]) * 0.95)
        p95 = sorted(data["latencies"])[p95_idx] if data["latencies"] else 0

        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"  Accuracy: {accuracy:.2f}% ({data['correct']}/{data['total']})")
        print("  Latency (μs):")
        print(f"    Average: {avg_latency:.2f}")
        print(f"    Min:     {min_latency:.2f}")
        print(f"    Max:     {max_latency:.2f}")
        print(f"    P50:     {p50:.2f}")
        print(f"    P95:     {p95:.2f}")

        print("\n  Per-language accuracy:")
        for lang in sorted(data["by_lang"].keys()):
            lang_data = data["by_lang"][lang]
            if lang_data["total"] > 0:
                lang_acc = (lang_data["correct"] / lang_data["total"]) * 100
                print(f"    {lang}: {lang_acc:>6.2f}% ({lang_data['correct']}/{lang_data['total']})")

        if data["errors"]:
            print(f"\n  Sample errors ({len(data['errors'])} total):")
            for err in data["errors"][:5]:
                print(f"    [{err['length']}] Expected '{err['expected']}', got '{err['got']}': {err['text']}")
            if len(data["errors"]) > 5:
                print(f"    ... and {len(data['errors']) - 5} more")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Library':<25} {'Accuracy':<15} {'Avg Latency':<15} {'P95 Latency':<15}")
    print("-" * 70)
    for name, data in results.items():
        accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        avg_latency = sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0
        p95_idx = int(len(data["latencies"]) * 0.95)
        p95 = sorted(data["latencies"])[p95_idx] if data["latencies"] else 0
        print(f"{name:<25} {accuracy:>6.2f}%        {avg_latency:>6.2f} μs       {p95:>6.2f} μs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark language detection libraries")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["custom", "wili", "papluca", "nemotron"],
        default="custom",
        help="Dataset to use: 'custom' (dataset.json), 'wili' (wili_filtered.json), 'papluca' (papluca_filtered.json), or 'nemotron' (nemotron_filtered.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (random subset if dataset is larger)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "action"],
        default="direct",
        help="Benchmark mode: 'direct' (call detection functions directly) or 'action' (use ActionDispatcher with detect_language action)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Directory to save CSV reports (summary.csv, per_language.csv, errors.csv). If not specified, no reports are generated.",
    )
    args = parser.parse_args()

    if args.dataset == "wili":
        dataset_path = Path(__file__).parent / "wili_filtered.json"
    elif args.dataset == "papluca":
        dataset_path = Path(__file__).parent / "papluca_filtered.json"
    elif args.dataset == "nemotron":
        dataset_path = Path(__file__).parent / "nemotron_filtered.json"
    else:
        dataset_path = Path(__file__).parent / "dataset.json"

    report_dir = Path(args.report) if args.report else None

    if args.mode == "action":
        asyncio.run(run_benchmark_action(dataset_path, args.max_samples, report_dir))
    else:
        run_benchmark(dataset_path, args.max_samples, report_dir)
