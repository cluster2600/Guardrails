# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from typing import Optional

import heuristics.checks as hc  # type: ignore[import-not-found]
import model_based.checks as mc  # type: ignore[import-not-found]
import typer  # type: ignore[import-not-found]
import uvicorn  # type: ignore[import-not-found]
from fastapi import FastAPI  # type: ignore[import-not-found]
from pydantic import BaseModel

from nemoguardrails.rails.llm.config import JailbreakDetectionConfig

# Use the JailbreakDetectionConfig defaults for consistency
LP_THRESHOLD_DEFAULT = JailbreakDetectionConfig.model_fields["length_per_perplexity_threshold"].default
PS_PPL_THRESHOLD_DEFAULT = JailbreakDetectionConfig.model_fields["prefix_suffix_perplexity_threshold"].default


app = FastAPI()
cli_app = typer.Typer()

device = os.environ.get("JAILBREAK_CHECK_DEVICE", "cpu")


class JailbreakHeuristicRequest(BaseModel):
    """
    prompt (str): User utterance to the model
    lp_threshold (float): Threshold value for length-perplexity heuristic.
    ps_ppl_threshold (float): Threshold value for prefix/suffix perplexity heuristic.
    """

    prompt: str
    lp_threshold: Optional[float] = LP_THRESHOLD_DEFAULT
    ps_ppl_threshold: Optional[float] = PS_PPL_THRESHOLD_DEFAULT


class JailbreakModelRequest(BaseModel):
    """
    Since the embedding model corresponds exactly to the classifier, we only need to provide the embedding model in the request.

    prompt (str): User utterance to the model
    """

    prompt: str


@app.get("/")
def hello_world():
    welcome_str = (
        "This is a development server for jailbreak detection.\n"
        "Hit the /heuristics endpoint to run all heuristics by sending a POST request with the user prompt.\n"
        "Hit the /model endpoint to run against the loaded classifier by sending a POST request with the user prompt."
        "Detailed documentation and all endpoints are included in the README."
    )
    return welcome_str


@app.post("/jailbreak_lp_heuristic")
def lp_heuristic_check(request: JailbreakHeuristicRequest):
    return hc.check_jailbreak_length_per_perplexity(request.prompt, request.lp_threshold or LP_THRESHOLD_DEFAULT)


@app.post("/jailbreak_ps_heuristic")
def ps_ppl_heuristic_check(request: JailbreakHeuristicRequest):
    return hc.check_jailbreak_prefix_suffix_perplexity(
        request.prompt, request.ps_ppl_threshold or PS_PPL_THRESHOLD_DEFAULT
    )


@app.post("/heuristics")
def run_all_heuristics(request: JailbreakHeuristicRequest):
    # Will add other heuristics as they become available
    lp_check = hc.check_jailbreak_length_per_perplexity(request.prompt, request.lp_threshold or LP_THRESHOLD_DEFAULT)
    ps_ppl_check = hc.check_jailbreak_prefix_suffix_perplexity(
        request.prompt, request.ps_ppl_threshold or PS_PPL_THRESHOLD_DEFAULT
    )
    jailbreak = any([lp_check["jailbreak"], ps_ppl_check["jailbreak"]])
    heuristic_checks = {
        "jailbreak": jailbreak,
        "length_per_perplexity": lp_check["jailbreak"],
        "prefix_suffix_perplexity": ps_ppl_check["jailbreak"],
    }
    return heuristic_checks


@app.post("/model")
def run_model_check(request: JailbreakModelRequest):
    result = mc.check_jailbreak(request.prompt)
    jailbreak = result["jailbreak"]
    score = result["score"]
    model_checks = {"jailbreak": jailbreak, "score": score}
    return model_checks


@cli_app.command()
def start(
    port: int = typer.Option(default=1337, help="The port that the server should listen on."),
    host: str = typer.Option(default="0.0.0.0", help="IP address of the host"),
):
    _ = mc.initialize_model()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli_app()
