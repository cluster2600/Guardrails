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

import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATASETS = ["papluca", "nemotron"]
MODES = ["direct", "action"]
REPORT_DIR = SCRIPT_DIR / "reports"


def run_command(cmd: list[str]) -> int:
    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR.parent.parent)
    return result.returncode


def download_datasets():
    print("\n" + "=" * 80)
    print("DOWNLOADING DATASETS")
    print("=" * 80)

    downloaders = [
        ("papluca_filtered.json", "download_papluca.py"),
        ("nemotron_filtered.json", "download_nemotron.py"),
    ]

    for filename, script in downloaders:
        if (SCRIPT_DIR / filename).exists():
            print(f"\n{filename} already exists, skipping download.")
        else:
            print(f"\nDownloading {filename}...")
            run_command(["poetry", "run", "python", str(SCRIPT_DIR / script)])


def run_benchmarks():
    print("\n" + "=" * 80)
    print("RUNNING BENCHMARKS")
    print("=" * 80)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for mode in MODES:
            run_command(
                [
                    "poetry",
                    "run",
                    "python",
                    str(SCRIPT_DIR / "benchmark.py"),
                    "--dataset",
                    dataset,
                    "--mode",
                    mode,
                    "--report",
                    str(REPORT_DIR),
                ]
            )


def main():
    download_datasets()
    run_benchmarks()

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nReports saved to: {REPORT_DIR}/")
    print("  - summary.csv")
    print("  - per_language.csv")
    print("  - errors.csv")


if __name__ == "__main__":
    main()
