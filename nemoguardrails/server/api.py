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

import asyncio
import contextvars
import importlib.util
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel, ValidationError
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from nemoguardrails import LLMRails, RailsConfig, utils
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server.datastore.datastore import DataStore
from nemoguardrails.server.schemas.openai import (
    GuardrailsChatCompletion,
    GuardrailsChatCompletionRequest,
    OpenAIModelsList,
)
from nemoguardrails.server.schemas.utils import (
    create_error_chat_completion,
    extract_bot_message_from_response,
    fetch_models,
    format_streaming_chunk_as_sse,  # Formats a single token/chunk into SSE wire format
    generation_response_to_chat_completion,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GuardrailsApp(FastAPI):
    """Custom FastAPI subclass with additional attributes for Guardrails server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize custom attributes
        self.default_config_id: Optional[str] = None
        self.rails_config_path: str = ""
        self.disable_chat_ui: bool = False
        self.auto_reload: bool = False
        self.stop_signal: bool = False
        self.single_config_mode: bool = False
        self.single_config_id: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.task: Optional[asyncio.Future] = None


# The list of registered loggers. Can be used to send logs to various
# backends and storage engines.
registered_loggers: List[Callable] = []

api_description = """Guardrails Sever API."""

# Per-request header storage via ContextVar — ensures each async coroutine sees
# its own request headers without cross-contamination between concurrent requests.
api_request_headers: contextvars.ContextVar = contextvars.ContextVar("headers")

# The datastore that the Server should use.
# This is currently used only for storing threads.
# TODO: refactor to wrap the FastAPI instance inside a RailsServer class
#  and get rid of all the global attributes.
datastore: Optional[DataStore] = None


@asynccontextmanager
async def lifespan(app: GuardrailsApp):
    # Startup logic here
    """Register any additional challenges, if available at startup."""
    challenges_files = os.path.join(app.rails_config_path, "challenges.json")

    if os.path.exists(challenges_files):
        with open(challenges_files) as f:
            register_challenges(json.load(f))

    # Detect single-config mode: when the rails_config_path itself contains a
    # config.yml/yaml, the server serves exactly one guardrails configuration
    # rather than a directory of multiple named configs.
    if os.path.exists(os.path.join(app.rails_config_path, "config.yml")) or os.path.exists(
        os.path.join(app.rails_config_path, "config.yaml")
    ):
        app.single_config_mode = True
        app.single_config_id = os.path.basename(app.rails_config_path)
    else:
        # Multi-config mode: optionally load a server-level config.py that can
        # register custom loggers, set the default config, or otherwise
        # customise the app instance before it starts serving.
        filepath = os.path.join(app.rails_config_path, "config.py")
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            # Dynamically import the config module using importlib so it can
            # execute arbitrary Python at server startup.
            spec = importlib.util.spec_from_file_location(filename, filepath)
            if spec is not None and spec.loader is not None:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
            else:
                config_module = None

            # Convention: if the module exposes an `init(app)` function, call
            # it so operators can programmatically configure the server.
            if config_module is not None and hasattr(config_module, "init"):
                config_module.init(app)

    # Finally, we register the static frontend UI serving

    if not app.disable_chat_ui:
        FRONTEND_DIR = utils.get_chat_ui_data_path("frontend")

        app.mount(
            "/",
            StaticFiles(
                directory=FRONTEND_DIR,
                html=True,
            ),
            name="chat",
        )
    else:

        @app.get("/")
        async def root_handler():
            return {"status": "ok"}

    if app.auto_reload:
        app.loop = asyncio.get_running_loop()
        # Run the watchdog file observer in a thread-pool executor so it
        # does not block the async event loop. The returned Future is kept
        # so we can cancel it on shutdown.
        app.task = app.loop.run_in_executor(None, start_auto_reload_monitoring)

    yield

    # Shutdown logic here
    if app.auto_reload:
        app.stop_signal = True
        if hasattr(app, "task") and app.task is not None:
            app.task.cancel()
        log.info("Shutting down file observer")
    else:
        pass


app = GuardrailsApp(
    title="Guardrails Server API",
    description=api_description,
    version="0.1.0",
    license_info={"name": "Apache License, Version 2.0"},
    lifespan=lifespan,
)

ENABLE_CORS = os.getenv("NEMO_GUARDRAILS_SERVER_ENABLE_CORS", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS", "*")

if ENABLE_CORS:
    # Split origins by comma
    origins = ALLOWED_ORIGINS.split(",")

    log.info(f"CORS enabled with the following origins: {origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.default_config_id = None

# By default, we use the rails in the examples folder
app.rails_config_path = utils.get_examples_data_path("bots")

# Weather the chat UI is enabled or not.
app.disable_chat_ui = False

# auto reload flag
app.auto_reload = False

# stop signal for observer
app.stop_signal = False

# Whether the server is pointed to a directory containing a single config.
app.single_config_mode = False
app.single_config_id = None


@app.get(
    "/v1/rails/configs",
    summary="Get List of available rails configurations.",
)
async def get_rails_configs():
    """Returns the list of available rails configurations."""

    # In single-config mode, we return a single config.
    if app.single_config_mode:
        # And we use the name of the root folder as the id of the config.
        return [{"id": app.single_config_id}]

    # We extract all folder names as config names
    config_ids = [
        f
        for f in os.listdir(app.rails_config_path)
        if os.path.isdir(os.path.join(app.rails_config_path, f))
        and f[0] != "."
        and f[0] != "_"
        # We filter out all the configs for which there is no `config.yml` file.
        and (
            os.path.exists(os.path.join(app.rails_config_path, f, "config.yml"))
            or os.path.exists(os.path.join(app.rails_config_path, f, "config.yaml"))
        )
    ]

    return [{"id": config_id} for config_id in config_ids]


@app.get(
    "/v1/models",
    response_model=OpenAIModelsList,
    summary="Get list of available models.",
)
async def list_models(request: Request):
    """Return the list of models available from the configured provider."""

    engine = os.environ.get("MAIN_MODEL_ENGINE", "openai")

    # Forward auth headers from the incoming request.
    request_headers: dict[str, str] = {}
    auth_header = request.headers.get("authorization")
    if auth_header:
        request_headers["Authorization"] = auth_header

    try:
        # Fetch the list of models from the configured provider
        models = await fetch_models(engine, request_headers)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Error fetching models from upstream: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error connecting to upstream model server: {str(exc)}",
        )

    return OpenAIModelsList(data=models)


# Module-level cache of LLMRails instances, keyed by a composite string of
# config IDs (and optionally model name). This avoids re-initialising the
# (potentially expensive) LLMRails pipeline on every request.
# NOTE: This dict is *not* thread-safe — it relies on the GIL for basic
# atomicity and on the assumption that FastAPI's async handlers serialise
# dict mutations within a single event loop. If the server were to use
# multiple worker processes (e.g. gunicorn with >1 worker), each process
# would hold its own independent copy of this cache.
llm_rails_instances: dict[str, LLMRails] = {}

# Separate cache that preserves conversation event histories across config
# reloads (triggered by the auto-reload watcher). When a config changes on
# disc, the corresponding LLMRails instance is evicted from
# llm_rails_instances, but its events_history_cache is stashed here so the
# newly created instance can resume with the same conversational state.
llm_rails_events_history_cache: dict[str, dict] = {}


def _generate_cache_key(config_ids: List[str], model_name: Optional[str] = None) -> str:
    """Generates a cache key for the given config ids and model name."""
    # Concatenate all config IDs with a hyphen separator, then append the
    # model name (if any) after a colon. This ensures that the same set of
    # configs with different model overrides produces distinct cache entries.
    # e.g. "configA-configB:gpt-4" vs "configA-configB:gpt-3.5-turbo"
    # CAVEAT: config IDs containing hyphens could theoretically collide
    # (e.g. ["a-b", "c"] vs ["a", "b-c"]), though this is unlikely in
    # practice because config IDs are typically simple directory names.
    key = "-".join(config_ids)
    if model_name:
        key = f"{key}:{model_name}"
    return key


def _update_models_in_config(config: RailsConfig, main_model: Model) -> RailsConfig:
    """Update the main model in the RailsConfig.

    If a model with type="main" exists, it replaces it. Otherwise, adds it.
    """
    models = config.models.copy()  # Shallow copy to avoid mutating the original config
    main_model_index = None

    # Locate the existing model entry that shares the same type (e.g. "main")
    for index, model in enumerate(models):
        if model.type == main_model.type:
            main_model_index = index
            break

    if main_model_index is not None:
        # Merge parameters: the existing config's parameters serve as defaults,
        # and the new model's parameters override them. This allows callers to
        # supply only the parameters they wish to change (e.g. base_url) whilst
        # retaining any provider-specific defaults already set in the YAML config.
        parameters = {**models[main_model_index].parameters, **main_model.parameters}
        models[main_model_index] = main_model
        models[main_model_index].parameters = parameters
    else:
        # No existing model of this type — simply append the new one
        models.append(main_model)

    # Return a new RailsConfig instance (immutable update via Pydantic)
    return config.model_copy(update={"models": models})


def _get_rails(config_ids: List[str], model_name: Optional[str] = None) -> LLMRails:
    """Returns the rails instance for the given config id and model.

    Args:
        config_ids: List of configuration IDs to load
        model_name: The model name from the request (overrides config's main model)
    """
    configs_cache_key = _generate_cache_key(config_ids, model_name)

    # Fast path: return the cached LLMRails instance if one already exists
    # for this exact combination of config IDs + model name.
    if configs_cache_key in llm_rails_instances:
        return llm_rails_instances[configs_cache_key]

    # --- Cache miss: build a new LLMRails instance from scratch ---

    # In single-config mode, we only load the main config directory
    if app.single_config_mode:
        if config_ids != [app.single_config_id]:
            raise ValueError(f"Invalid configuration ids: {config_ids}")

        # Replace with empty string so os.path.join(base_path, "") == base_path
        config_ids = [""]

    full_llm_rails_config: Optional[RailsConfig] = None

    # Iterate through each requested config and merge them together.
    # The first config becomes the base; subsequent configs are merged via
    # the RailsConfig.__add__ operator, which layers additional rails,
    # prompts, and flows on top of the base configuration.
    for config_id in config_ids:
        base_path = os.path.abspath(app.rails_config_path)
        full_path = os.path.normpath(os.path.join(base_path, config_id))

        # Path-traversal guard: reject config IDs containing slashes or ".."
        # to prevent directory-traversal attacks via crafted config_id values.
        if re.search(r"[\\/]|(\.\.)", config_id):
            raise ValueError("Invalid config_id.")

        # Secondary traversal guard: ensure the resolved path remains within
        # the configured base directory (belt-and-braces defence).
        if os.path.commonprefix([full_path, base_path]) != base_path:
            raise ValueError("Access to the specified path is not allowed.")

        rails_config = RailsConfig.from_path(full_path)

        if not full_llm_rails_config:
            full_llm_rails_config = rails_config
        else:
            # Merge: layers the new config's rails/flows onto the accumulated config
            full_llm_rails_config += rails_config

    if full_llm_rails_config is None:
        raise ValueError("No valid rails configuration found.")

    # If the caller specified a model name (via the OpenAI-compatible `model`
    # field), override the "main" model in the merged config. The engine and
    # base_url are sourced from environment variables, allowing operators to
    # point the server at different LLM providers without changing YAML configs.
    if model_name:
        engine = os.environ.get("MAIN_MODEL_ENGINE")
        if not engine:
            engine = "openai"
            log.warning("MAIN_MODEL_ENGINE not set, defaulting to 'openai'. ")

        parameters = {}
        base_url = os.environ.get("MAIN_MODEL_BASE_URL")
        if base_url:
            parameters["base_url"] = base_url

        main_model = Model(model=model_name, type="main", engine=engine, parameters=parameters)
        full_llm_rails_config = _update_models_in_config(full_llm_rails_config, main_model)

    # Initialise the LLMRails pipeline (loads models, compiles Colang, etc.)
    llm_rails = LLMRails(config=full_llm_rails_config, verbose=True)
    # Store in the module-level cache for subsequent requests
    llm_rails_instances[configs_cache_key] = llm_rails

    # Restore any previously saved events history (preserved across auto-reloads)
    # so that ongoing conversations are not lost when a config file changes on disc.
    llm_rails.events_history_cache = llm_rails_events_history_cache.get(configs_cache_key, {})

    return llm_rails


class ChunkErrorMetadata(BaseModel):
    message: str
    type: str
    param: str
    code: str


class ChunkError(BaseModel):
    error: ChunkErrorMetadata


async def _format_streaming_response(
    stream_iterator: AsyncIterator[Union[str, dict]], model_name: str
) -> AsyncIterator[str]:
    """
    Format streaming chunks from LLMRails.stream_async() as SSE events.

    Args:
        stream_iterator: AsyncIterator from stream_async() that yields str or dict chunks
        model_name: The model name to include in the chunks

    Yields:
        SSE-formatted strings (data: {...}\n\n)
    """
    model = model_name or "unknown"
    # Generate a single completion ID shared across all chunks in this stream,
    # matching OpenAI's behaviour where every chunk in a streamed response
    # carries the same `id` field.
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    try:
        async for chunk in stream_iterator:
            # Each chunk may be a plain text token or a JSON error object.
            # process_chunk() attempts to parse it as a ChunkError; if it
            # succeeds, the stream is terminated early with the error payload.
            processed_chunk = process_chunk(chunk)
            if isinstance(processed_chunk, ChunkError):
                # Yield the error as a single SSE event, then halt the stream
                yield f"data: {json.dumps(processed_chunk.model_dump())}\n\n"
                return
            else:
                # Normal token — wrap in the OpenAI streaming delta format
                yield format_streaming_chunk_as_sse(processed_chunk, model, chunk_id)

    finally:
        # The SSE protocol requires a sentinel [DONE] message so clients know
        # the stream has finished (mirrors the OpenAI streaming API contract).
        yield "data: [DONE]\n\n"


def process_chunk(chunk: Any) -> Union[Any, ChunkError]:
    """
    Processes a single chunk from the stream.

    Args:
        chunk: A single chunk from the stream (can be str, dict, or other type).
        model: The model name (not used in processing but kept for signature consistency).

    Returns:
        Union[Any, StreamingError]: StreamingError instance for errors or the original chunk.
    """
    # Normalise the chunk to a JSON string so we can attempt Pydantic validation.
    # Most chunks are plain str tokens; dicts are serialised, other types coerced.
    chunk_str = chunk if isinstance(chunk, str) else json.dumps(chunk) if isinstance(chunk, dict) else str(chunk)

    try:
        # Attempt to parse the chunk as a structured error (e.g. from upstream LLM).
        # If the chunk does not match the ChunkError schema, Pydantic raises
        # ValidationError and we fall through to treat it as a normal token.
        validated_data = ChunkError.model_validate_json(chunk_str)
        return validated_data
    except ValidationError:
        pass  # Not an error payload — treat as a normal content token
    except json.JSONDecodeError:
        pass  # Malformed JSON — safe to treat as a regular token
    except Exception as e:
        log.warning(
            f"Unexpected error processing stream chunk: {type(e).__name__}: {str(e)}",
            extra={"chunk": chunk_str},
        )

    # Pass the original chunk through unchanged for downstream SSE formatting
    return chunk


@app.post(
    "/v1/chat/completions",
    response_model=GuardrailsChatCompletion,
    response_model_exclude_none=True,
)
async def chat_completion(body: GuardrailsChatCompletionRequest, request: Request):
    """Chat completion for the provided conversation.

    TODO: add support for explicit state object.
    """
    log.info("Got request for config %s", body.guardrails.config_id)
    # Fire-and-forget: dispatch logging tasks without blocking the response.
    # Each registered logger runs as an independent async task on the event loop.
    for logger in registered_loggers:
        asyncio.get_running_loop().create_task(logger({"endpoint": "/v1/chat/completions", "body": body.json()}))

    # Stash the incoming HTTP headers in the ContextVar so that downstream
    # code (e.g. LLM provider calls) can forward authorisation tokens.
    api_request_headers.set(request.headers)

    # Resolve config IDs: prefer the request-level override, fall back to
    # the server-wide default. Absence of both is a client error (422).
    config_ids = body.guardrails.config_ids

    if not config_ids:
        if app.default_config_id:
            config_ids = [app.default_config_id]
        else:
            raise HTTPException(
                status_code=422,
                detail="No guardrails config_id provided and server has no default configuration",
            )

    try:
        llm_rails = _get_rails(config_ids, model_name=body.model)

    except ValueError as ex:
        log.exception(ex)
        return create_error_chat_completion(
            model=body.model,
            error_message=f"Could not load the {config_ids} guardrails configuration. An internal error has occurred.",
            config_id=config_ids[0] if config_ids else None,
        )

    try:
        messages = body.messages or []
        # Inject any caller-supplied context as a synthetic "context" role message
        # at the beginning of the conversation, ahead of user/assistant turns.
        if body.guardrails.context:
            messages.insert(0, {"role": "context", "content": body.guardrails.context})

        # If we have a `thread_id` specified, we need to look up the thread
        datastore_key = None

        if body.guardrails.thread_id:
            if datastore is None:
                raise RuntimeError("No DataStore has been configured.")
            # Enforce a minimum length to discourage trivially guessable thread IDs
            # (mitigates enumeration attacks against the datastore).
            if len(body.guardrails.thread_id) < 16:
                return create_error_chat_completion(
                    model=body.model,
                    error_message="The `thread_id` must have a minimum length of 16 characters.",
                    config_id=config_ids[0] if config_ids else None,
                )

            # Namespace datastore keys with a "thread-" prefix to avoid
            # collisions with other data stored in the same backend.
            datastore_key = "thread-" + body.guardrails.thread_id
            thread_messages = json.loads(await datastore.get(datastore_key) or "[]")

            # Prepend historical messages so the LLM sees the full conversation
            messages = thread_messages + messages

        generation_options = body.guardrails.options

        # Validate state format if provided
        if body.guardrails.state is not None and body.guardrails.state != {}:
            if "events" not in body.guardrails.state and "state" not in body.guardrails.state:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid state format: state must contain 'events' or 'state' key. Use an empty dict {} to start a new conversation.",
                )

        # Ensure llm_params dict exists before populating it
        if generation_options.llm_params is None:
            generation_options.llm_params = {}

        # Forward standard OpenAI-compatible sampling/generation parameters
        # into the guardrails options so they reach the underlying LLM call.
        if body.max_tokens:
            generation_options.llm_params["max_tokens"] = body.max_tokens
        if body.temperature is not None:
            generation_options.llm_params["temperature"] = body.temperature
        if body.top_p is not None:
            generation_options.llm_params["top_p"] = body.top_p
        if body.stop:
            generation_options.llm_params["stop"] = body.stop
        if body.presence_penalty is not None:
            generation_options.llm_params["presence_penalty"] = body.presence_penalty
        if body.frequency_penalty is not None:
            generation_options.llm_params["frequency_penalty"] = body.frequency_penalty

        if body.stream:
            # Streaming branch: yields Server-Sent Events (SSE) as each token
            # arrives from the LLM. Output rails are still applied via
            # stream_async, which may buffer or filter tokens before yielding.
            stream_iterator = llm_rails.stream_async(
                messages=messages,
                options=generation_options,
                state=body.guardrails.state,
            )

            # StreamingResponse consumes the async generator lazily, keeping
            # the connection open until the generator is exhausted or errors.
            return StreamingResponse(
                _format_streaming_response(stream_iterator, model_name=body.model),
                media_type="text/event-stream",
            )
        else:
            res = await llm_rails.generate_async(
                messages=messages,
                options=generation_options,
                state=body.guardrails.state,
            )

            # Extract the assistant's reply so we can persist it in the thread
            bot_message = extract_bot_message_from_response(res)

            # Persist the updated conversation (original + new turn) back to the
            # datastore so subsequent requests with the same thread_id see it.
            if body.guardrails.thread_id and datastore is not None and datastore_key is not None:
                await datastore.set(datastore_key, json.dumps(messages + [bot_message]))

            # Return the result in an OpenAI-compatible chat completion envelope.
            # GenerationResponse carries richer metadata (guardrails log, etc.);
            # plain dicts are wrapped in a minimal completion structure.
            if isinstance(res, GenerationResponse):
                return generation_response_to_chat_completion(
                    response=res,
                    model=body.model,
                    config_id=config_ids[0] if config_ids else None,
                )
            else:
                # For dict responses, convert to basic chat completion
                return GuardrailsChatCompletion(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=body.model,
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=bot_message.get("content", ""),
                            ),
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                )

    except HTTPException:
        raise  # Re-raise HTTP exceptions so FastAPI returns the correct status code
    except Exception as ex:
        # Catch-all: log the full traceback but return a sanitised error to the
        # client to avoid leaking internal details.
        log.exception(ex)
        return create_error_chat_completion(
            model=body.model,
            error_message="Internal server error",
            config_id=config_ids[0] if config_ids else None,
        )


# By default, there are no challenges
challenges = []


def register_challenges(additional_challenges: List[dict]):
    """Register additional challenges

    Args:
        additional_challenges: The new challenges to be registered.
    """
    challenges.extend(additional_challenges)


@app.get(
    "/v1/challenges",
    summary="Get list of available challenges.",
)
async def get_challenges():
    """Returns the list of available challenges for red teaming."""

    return challenges


def register_datastore(datastore_instance: DataStore):
    """Registers a DataStore to be used by the server."""
    global datastore

    datastore = datastore_instance


def register_logger(logger: Callable):
    """Register an additional logger"""
    registered_loggers.append(logger)


def start_auto_reload_monitoring():
    """Start a thread that monitors the config folder for changes."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class Handler(FileSystemEventHandler):
            # NOTE on thread safety: watchdog fires events from a background
            # OS thread, whilst the LLMRails cache is read/written from the
            # async event loop thread. Python's GIL protects the dict from
            # corruption, but there is a small race window where a request
            # could read a stale instance just before it is deleted. In
            # practice this is benign — the stale instance still works, and
            # the next request will pick up the fresh config.
            def on_any_event(self, event):
                if event.is_directory:
                    return None

                elif event.event_type == "created" or event.event_type == "modified":
                    log.info(f"Watchdog received {event.event_type} event for file {event.src_path}")

                    # Derive the config_id from the first path component relative
                    # to the configs root directory.
                    src_path_str = str(event.src_path)
                    rel_path = os.path.relpath(src_path_str, app.rails_config_path)

                    parts = rel_path.split(os.path.sep)
                    config_id = parts[0]

                    # Skip hidden files and Jupyter checkpoint artefacts
                    if (
                        not parts[-1].startswith(".")
                        and ".ipynb_checkpoints" not in parts
                        and os.path.isfile(src_path_str)
                    ):
                        # Evict the cached LLMRails instance so the next request
                        # triggers a full rebuild with the updated config files.
                        if config_id in llm_rails_instances:
                            instance = llm_rails_instances[config_id]
                            del llm_rails_instances[config_id]
                            if instance:
                                # Preserve the events history so ongoing
                                # conversations survive a config reload.
                                val = instance.events_history_cache
                                llm_rails_events_history_cache[config_id] = val

                            log.info(f"Configuration {config_id} has changed. Clearing cache.")

        observer = Observer()
        event_handler = Handler()
        observer.schedule(event_handler, app.rails_config_path, recursive=True)
        observer.start()
        try:
            while not app.stop_signal:
                time.sleep(5)
        finally:
            observer.stop()
            observer.join()

    except ImportError:
        # Since this is running in a separate thread, we just print the error.
        print("The auto-reload feature requires `watchdog`. Please install using `pip install watchdog`.")
        # Force close everything.
        os._exit(-1)


def set_default_config_id(config_id: str):
    app.default_config_id = config_id


class GuardrailsConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


# # Register a nicer error message for 422 error
# def register_exception(app: FastAPI):
#     @app.exception_handler(RequestValidationError)
#     async def validation_exception_handler(
#         request: Request, exc: RequestValidationError
#     ):
#         exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
#         # or logger.error(f'{exc}')
#         log.error(request, exc_str)
#         content = {"status_code": 10422, "message": exc_str, "data": None}
#         return JSONResponse(
#             content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
#         )
#
#
# register_exception(app)
