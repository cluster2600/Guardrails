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

import logging
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.thread_pool import RailThreadPool

log = logging.getLogger(__name__)


class Runtime:
    """Base Colang Runtime implementation.

    The Runtime is the central orchestrator for guardrail evaluation.
    It owns the action dispatcher, the LLM task manager, and (optionally)
    a thread pool for offloading CPU-bound actions.  Concrete subclasses
    (e.g. ``RuntimeV1_0``) provide version-specific flow execution logic.
    """

    def __init__(self, config: RailsConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose

        # Build the thread pool for CPU-bound actions (if enabled in config).
        # On free-threaded Python 3.14t this allows true parallel execution
        # of @cpu_bound-decorated actions without blocking the event loop.
        tp_config = config.thread_pool
        if tp_config.enabled:
            self._thread_pool: Optional[RailThreadPool] = RailThreadPool(
                max_workers=tp_config.max_workers,
                thread_name_prefix=tp_config.thread_name_prefix,
                enabled=True,
            )
        else:
            self._thread_pool = None

        # Initialise the action dispatcher, passing in the thread pool so
        # that @cpu_bound actions can be dispatched to worker threads.
        self.action_dispatcher = ActionDispatcher(
            config_path=config.config_path,
            import_paths=list(config.imported_paths.values()),
            thread_pool=self._thread_pool,
        )

        # Register parallel-execution actions if the subclass provides them.
        # These are invoked by Colang flows to run multiple rails concurrently.
        if hasattr(self, "_run_output_rails_in_parallel_streaming"):
            self.action_dispatcher.register_action(
                self._run_output_rails_in_parallel_streaming,
                name="run_output_rails_in_parallel_streaming",
            )

        if hasattr(self, "_run_flows_in_parallel"):
            self.action_dispatcher.register_action(self._run_flows_in_parallel, name="run_flows_in_parallel")

        if hasattr(self, "_run_input_rails_in_parallel"):
            self.action_dispatcher.register_action(
                self._run_input_rails_in_parallel, name="run_input_rails_in_parallel"
            )

        if hasattr(self, "_run_output_rails_in_parallel"):
            self.action_dispatcher.register_action(
                self._run_output_rails_in_parallel, name="run_output_rails_in_parallel"
            )

        # Additional parameters that can be injected into action callables
        # at dispatch time (e.g. ``llm``, ``config``, ``state``).
        self.registered_action_params: dict = {}

        self._init_flow_configs()

        # The LLM task manager handles prompt rendering and output parsing.
        self.llm_task_manager = LLMTaskManager(config)

        # Watchers are notified on each event cycle — used by the CLI to
        # report progress to the user.
        self.watchers = []

        # Safety limit to prevent infinite processing loops.
        self.max_events = 500

    def shutdown(self, wait: bool = True) -> None:
        """Release the thread pool and any other managed resources.

        Callers (e.g. ``LLMRails``) should invoke this when the runtime
        is no longer needed, to ensure deterministic cleanup of worker
        threads.  If not called, the ``__del__`` method on the thread
        pool will attempt best-effort cleanup.
        """
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=wait)
            self._thread_pool = None

    @abstractmethod
    def _init_flow_configs(self) -> None:
        pass

    def register_action(self, action: Callable, name: Optional[str] = None, override: bool = True) -> None:
        """Registers an action with the given name.

        :param name: The name of the action.
        :param action: The action function.
        :param override: If an action already exists, whether it should be overriden or not.
        """
        self.action_dispatcher.register_action(action, name, override=override)

    def register_actions(self, actions_obj: Any, override: bool = True) -> None:
        """Registers all the actions from the given object."""
        self.action_dispatcher.register_actions(actions_obj, override=override)

    @property
    def registered_actions(self) -> dict:
        """Return registered actions."""
        return self.action_dispatcher.registered_actions

    def register_action_param(self, name: str, value: Any) -> None:
        """Registers an additional parameter that can be passed to the actions.

        :param name: The name of the parameter.
        :param value: The value of the parameter.
        """
        self.registered_action_params[name] = value

    async def generate_events(self, events: List[dict], processing_log: Optional[List[dict]] = None) -> List[dict]:
        """Generates the next events based on the provided history.

        This is a wrapper around the `process_events` method, that will keep
        processing the events until the `listen` event is produced.

        Args:
            events (List[dict]): The list of events.
            processing_log (Optional[List[dict]]): The processing log so far. This will be mutated.

        :return: The list of events.
        """
        raise NotImplementedError()

    async def process_events(
        self, events: List[dict], state: Optional[Any] = None, blocking: bool = False
    ) -> Tuple[List[dict], Any]:
        """Process a sequence of events in a given state.

        The events will be processed one by one, in the input order.

        Args:
            events: A sequence of events that needs to be processed.
            state: The state that should be used as the starting point. If not provided,
              a clean state will be used.
            blocking: In blocking mode, the event processing will also wait for all
              local async actions.

        Returns:
            (output_events, output_state) Returns a sequence of output events and an output
              state.
        """
        raise NotImplementedError()
