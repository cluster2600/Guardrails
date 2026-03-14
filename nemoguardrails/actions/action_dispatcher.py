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

"""Module for dispatching action calls to the appropriate registered handler.

This module is responsible for:
  1. Discovering and registering action functions/classes from the filesystem.
  2. Normalising action names (CamelCase -> snake_case, stripping "Action" suffix).
  3. Dispatching execution to the correct handler, supporting:
       - Plain synchronous functions
       - Async coroutine functions
       - Class-based actions (with a ``run`` method)
       - LangChain ``Runnable`` instances
  4. Optionally offloading ``@cpu_bound``-decorated synchronous actions to a
     :class:`~nemoguardrails.rails.llm.thread_pool.RailThreadPool` so they
     do not block the asyncio event loop.
  5. Ensuring thread safety on free-threaded (no-GIL) Python builds by
     using :class:`ThreadSafeDict` for the action registry and per-action
     locks for lazy class instantiation.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import threading
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from langchain_core.runnables import Runnable

from nemoguardrails import utils

# ThreadSafeDict wraps a plain dict with a reentrant lock, providing
# atomic read/write operations.  ``is_free_threaded()`` returns True
# when the interpreter was built with ``--disable-gil`` (PEP 703).
from nemoguardrails._thread_safety import ThreadSafeDict, is_free_threaded
from nemoguardrails.exceptions import LLMCallException

if TYPE_CHECKING:
    # Imported only for type-checking to avoid circular imports at runtime.
    from nemoguardrails.rails.llm.thread_pool import RailThreadPool

log = logging.getLogger(__name__)


class ActionDispatcher:
    def __init__(
        self,
        load_all_actions: bool = True,
        config_path: Optional[str] = None,
        import_paths: Optional[List[str]] = None,
        thread_pool: Optional["RailThreadPool"] = None,
    ):
        """
        Initializes an actions dispatcher.
        Args:
            load_all_actions (bool, optional): When set to True, it loads all actions in the
                'actions' folder both in the current working directory and in the package.
            config_path (str, optional): The path from which the configuration was loaded.
                If there are actions at the specified path, it loads them as well.
            import_paths (List[str], optional): Additional imported paths from which actions
                should be loaded.
            thread_pool (Optional[RailThreadPool]): An optional thread-pool executor for
                dispatching ``@cpu_bound``-decorated synchronous action functions.
                When *None*, cpu_bound actions are executed inline (backward-compatible).
        """
        log.info("Initialising action dispatcher")

        # ------------------------------------------------------------------
        # Thread pool integration
        # ------------------------------------------------------------------
        # The thread pool is used to offload ``@cpu_bound``-decorated
        # synchronous action functions so that they do not block the
        # asyncio event loop.  It may be ``None`` at construction time and
        # set later via the ``thread_pool`` property (e.g. when
        # ``LLMRails`` builds the pool from its configuration).
        self._thread_pool = thread_pool

        # ------------------------------------------------------------------
        # ThreadSafeDict usage on free-threaded Python
        # ------------------------------------------------------------------
        # On free-threaded (no-GIL) builds, concurrent dict mutations are
        # *not* protected by the GIL, so we use ``ThreadSafeDict`` – a
        # dict subclass guarded by a reentrant lock – to prevent data
        # races when actions are registered or looked up from multiple
        # threads.  On standard GIL-enabled builds, a plain ``dict`` is
        # used for zero overhead (dict operations are already atomic under
        # the GIL).
        self._registered_actions: Dict[str, Union[Type, Callable[..., Any]]] = (
            ThreadSafeDict() if is_free_threaded() else {}
        )

        # ------------------------------------------------------------------
        # Per-action locking for lazy class instantiation
        # ------------------------------------------------------------------
        # Class-based actions are registered as *classes* and only
        # instantiated on first invocation (see ``_atomic_instantiate_action``).
        # On GIL-enabled builds the dict write is atomic, but on
        # free-threaded builds two threads could race and construct the
        # same action class twice.  We therefore maintain a *per-action-name*
        # lock so that unrelated actions are never serialised against one
        # another.
        #
        # ``_init_locks_guard`` protects *creation* of new entries in
        # ``_init_locks``; each value in ``_init_locks`` protects a single
        # action's class-to-instance promotion.
        self._init_locks: Dict[str, threading.Lock] = {}
        self._init_locks_guard = threading.Lock()

        # ------------------------------------------------------------------
        # Action discovery / loading
        # ------------------------------------------------------------------
        # Actions are discovered in a well-defined priority order so that
        # later registrations override earlier ones of the same name:
        #   1. Built-in package ``actions/`` directory
        #   2. Per-feature ``library/*/actions`` directories
        #   3. Current working directory (user project root)
        #   4. Explicit ``config_path`` (may be comma-separated)
        #   5. Explicit ``import_paths``
        if load_all_actions:
            # TODO: check for better way to find actions dir path or use constants.py
            current_file_path = Path(__file__).resolve()
            parent_directory_path = current_file_path.parents[1]

            # 1. Load built-in actions shipped with the package.
            self.load_actions_from_path(parent_directory_path)

            # 2. Walk the ``library/`` tree and load any sub-package that
            #    exposes an ``actions/`` folder or ``actions.py`` file.
            library_path = parent_directory_path / "library"

            for root, dirs, files in os.walk(library_path):
                if "actions" in dirs or "actions.py" in files:
                    self.load_actions_from_path(Path(root))

            # 3. Load user-defined actions from the current working directory.
            # TODO: add support for an explicit ACTIONS_PATH
            self.load_actions_from_path(Path.cwd())

            # 4. Load actions from the configuration path(s), if provided.
            #    ``config_path`` may be a comma-separated list of paths.
            if config_path:
                split_config_path: List[str] = config_path.split(",")

                if split_config_path:
                    for path in split_config_path:
                        self.load_actions_from_path(Path(path.strip()))

            # 5. Load actions from any additional import paths.
            if import_paths:
                for import_path in import_paths:
                    self.load_actions_from_path(Path(import_path.strip()))

        log.info(f"Registered Actions :: {sorted(self._registered_actions.keys())}")
        log.info("Action dispatcher initialised")

    @property
    def registered_actions(self):
        """
        Gets the dictionary of registered actions.
        Returns:
            dict: A dictionary where keys are action names and values are callable action functions.
        """
        return self._registered_actions

    # ------------------------------------------------------------------
    # Thread pool property
    # ------------------------------------------------------------------

    @property
    def thread_pool(self) -> Optional["RailThreadPool"]:
        """Return the thread-pool executor used for ``@cpu_bound`` actions, if any.

        The pool wraps a :class:`concurrent.futures.ThreadPoolExecutor` and
        exposes an ``async dispatch(fn, **kwargs)`` helper that runs *fn* in
        a worker thread and awaits the result, keeping the asyncio event loop
        free.
        """
        return self._thread_pool

    @thread_pool.setter
    def thread_pool(self, pool: Optional["RailThreadPool"]) -> None:
        """Set (or replace) the thread-pool executor.

        This allows the pool to be attached *after* the dispatcher is
        constructed -- for instance when the :class:`LLMRails` instance
        builds the pool from its configuration.  It is safe to call this
        setter at any time; subsequent ``execute_action`` calls will pick
        up the new pool reference.
        """
        self._thread_pool = pool

    def load_actions_from_path(self, path: Path):
        """Loads all actions from the specified path.

        This method loads all actions from the `actions.py` file if it exists and
        all actions inside the `actions` folder if it exists.

        Args:
            path (str): A string representing the path from which to load actions.

        """
        actions_path = path / "actions"
        if os.path.exists(actions_path):
            self._registered_actions.update(self._find_actions(actions_path))

        actions_py_path = os.path.join(path, "actions.py")
        if os.path.exists(actions_py_path):
            self._registered_actions.update(self._load_actions_from_module(actions_py_path))

    def register_action(self, action: Callable, name: Optional[str] = None, override: bool = True):
        """Registers an action with the given name.

        Args:
            action (Callable): The action function.
            name (Optional[str]): The name of the action. Defaults to None.
            override (bool): If an action already exists, whether it should be overridden or not.
        """
        if name is None:
            action_meta = getattr(action, "action_meta", None)
            action_name = action_meta["name"] if action_meta else action.__name__
        else:
            action_name = name

        # If we're not allowed to override, we stop.
        if action_name in self._registered_actions and not override:
            return

        self._registered_actions[action_name] = action

    def register_actions(self, actions_obj: Any, override: bool = True):
        """Registers all the actions from the given object.

        Args:
            actions_obj (any): The object containing actions.
            override (bool): If an action already exists, whether it should be overridden or not.
        """

        # Register the actions
        for attr in dir(actions_obj):
            val = getattr(actions_obj, attr)

            if hasattr(val, "action_meta"):
                self.register_action(val, override=override)

    def _normalize_action_name(self, name: str) -> str:
        """Normalise an action name to its canonical snake_case form.

        The normalisation proceeds as follows:
          1. If the name already exists verbatim in the registry, return it
             immediately (fast path, avoids unnecessary string work).
          2. Strip a trailing ``"Action"`` suffix if present
             (e.g. ``"GenerateUserIntentAction"`` -> ``"GenerateUserIntent"``).
          3. Convert from CamelCase to snake_case using
             :func:`nemoguardrails.utils.camelcase_to_snakecase`.

        Note: on this branch there is no normalisation cache.  If profiling
        shows ``_normalize_action_name`` as a hot-spot, a bounded LRU cache
        keyed on *name* would be a straightforward optimisation.
        """
        if name not in self.registered_actions:
            if name.endswith("Action"):
                name = name.replace("Action", "")
            name = utils.camelcase_to_snakecase(name)
        return name

    def has_registered(self, name: str) -> bool:
        """Check if an action is registered."""
        name = self._normalize_action_name(name)
        return name in self.registered_actions

    def get_action(self, name: str) -> Optional[Callable]:
        """Get the registered action by name.

        Args:
            name (str): The name of the action.

        Returns:
            callable: The registered action.
        """
        name = self._normalize_action_name(name)
        return self._registered_actions.get(name, None)

    # ------------------------------------------------------------------
    # Atomic class-to-instance promotion (per-action locking)
    # ------------------------------------------------------------------

    def _atomic_instantiate_action(self, action_name: str, cls: Type) -> Callable[..., Any]:
        """Instantiate a class-based action exactly once (thread-safe).

        Class-based actions are stored in the registry as their *class*
        object and promoted to a singleton *instance* on first use.  This
        lazy initialisation avoids paying the cost of constructing actions
        that are never invoked.

        **Free-threaded Python path:**

        Without the GIL, two (or more) threads calling ``execute_action``
        for the same class-based action could both see ``inspect.isclass``
        return ``True`` and race to construct the instance.  To prevent
        duplicate construction we employ a *double-checked locking* pattern
        with per-action-name granularity:

          1. Acquire ``_init_locks_guard`` (a coarse lock) just long enough
             to obtain or create the per-action ``threading.Lock``.
          2. Acquire the per-action lock.
          3. Re-read the registry entry (the *double check*).  If another
             thread has already replaced the class with an instance, return
             that instance immediately.
          4. Otherwise, construct the instance, store it in the registry,
             and return it.

        Using per-action locks ensures that instantiation of *unrelated*
        actions is never serialised against each other, keeping contention
        to a minimum.

        **GIL-enabled Python path:**

        On standard CPython with the GIL, dict operations are already
        atomic, so no locking is required.  We take the simple path and
        instantiate directly.

        Args:
            action_name: The canonical (normalised) name of the action.
            cls: The action class to instantiate.

        Returns:
            The callable instance that has replaced *cls* in the registry.
        """
        if is_free_threaded():
            # Step 1 -- obtain (or create) a dedicated lock for this action.
            with self._init_locks_guard:
                if action_name not in self._init_locks:
                    self._init_locks[action_name] = threading.Lock()
                lock = self._init_locks[action_name]

            # Step 2 -- acquire the per-action lock and double-check.
            with lock:
                current = self._registered_actions.get(action_name)
                if current is not None and not inspect.isclass(current):
                    # Another thread already completed the promotion.
                    return cast(Callable[..., Any], current)
                # First thread to arrive -- construct and store the instance.
                instance = cls()
                self._registered_actions[action_name] = instance
                return instance
        else:
            # GIL build -- no race possible; instantiate directly.
            instance = cls()
            self._registered_actions[action_name] = instance
            return instance

    async def execute_action(
        self, action_name: str, params: Dict[str, Any]
    ) -> Tuple[Union[Optional[str], Dict[str, Any]], str]:
        """Execute a registered action.

        Args:
            action_name (str): The name of the action to execute.
            params (Dict[str, Any]): Parameters for the action.

        Returns:
            Tuple[Union[str, Dict[str, Any]], str]: A tuple containing the result and status.
        """

        action_name = self._normalize_action_name(action_name)

        if action_name in self._registered_actions:
            log.info("Executing registered action: %s", action_name)
            maybe_fn: Optional[Callable] = self._registered_actions.get(action_name, None)
            if not maybe_fn:
                raise Exception(f"Action '{action_name}' is not registered.")

            fn = cast(Callable, maybe_fn)
            # Actions that are registered as classes are initialized lazily,
            # when they are first used.  On free-threaded Python, two threads
            # could race here, so we use a per-action lock to ensure each
            # class is instantiated exactly once.
            if inspect.isclass(fn):
                fn = self._atomic_instantiate_action(action_name, fn)

            if fn:
                try:
                    # We support both functions and classes as actions
                    if inspect.isfunction(fn) or inspect.ismethod(fn):
                        # We support both sync and async actions.
                        # Check if the function is marked @cpu_bound and we
                        # have a thread pool configured.
                        is_cpu_bound = getattr(fn, "_cpu_bound", False)

                        if is_cpu_bound and self._thread_pool is not None:
                            log.info(
                                "Dispatching cpu_bound action `%s` to thread pool.",
                                action_name,
                            )
                            result = await self._thread_pool.dispatch(fn, **params)
                        else:
                            if is_cpu_bound:
                                log.warning(
                                    "Action `%s` is @cpu_bound but no thread pool is configured; "
                                    "running inline and blocking the event loop.",
                                    action_name,
                                )
                            result = fn(**params)

                        if inspect.iscoroutine(result):
                            result = await result
                        elif not is_cpu_bound:
                            log.warning(f"Synchronous action `{action_name}` has been called.")

                    elif isinstance(fn, Runnable):
                        # If it's a Runnable, we invoke it as well
                        runnable = fn

                        result = await runnable.ainvoke(input=params)
                    else:
                        # TODO: there should be a common base class here
                        fn_run_func = getattr(fn, "run", None)
                        if not callable(fn_run_func):
                            raise Exception(f"No 'run' method defined for action '{action_name}'.")

                        # Check if the class's run method is @cpu_bound.
                        is_cpu_bound = getattr(fn_run_func, "_cpu_bound", False)

                        fn_run_func_with_signature = cast(
                            Callable[[], Union[Optional[str], Dict[str, Any]]],
                            fn_run_func,
                        )

                        if is_cpu_bound and self._thread_pool is not None:
                            log.info(
                                "Dispatching cpu_bound action `%s.run` to thread pool.",
                                action_name,
                            )
                            result = await self._thread_pool.dispatch(fn_run_func_with_signature, **params)
                        else:
                            if is_cpu_bound:
                                log.warning(
                                    "Action `%s.run` is @cpu_bound but no thread pool is configured; "
                                    "running inline and blocking the event loop.",
                                    action_name,
                                )
                            result = fn_run_func_with_signature(**params)
                    return result, "success"

                # We forward LLM Call exceptions
                except LLMCallException as e:
                    raise e

                except Exception as e:
                    filtered_params = {k: v for k, v in params.items() if k not in ["state", "events", "llm"]}
                    log.warning(
                        "Error while execution '%s' with parameters '%s': %s",
                        action_name,
                        filtered_params,
                        e,
                    )
                    log.exception(e)

        return None, "failed"

    def get_registered_actions(self) -> List[str]:
        """Get the list of available actions.

        Returns:
            List[str]: List of available actions.
        """
        return list(self._registered_actions.keys())

    @staticmethod
    def _load_actions_from_module(filepath: str):
        """Loads the actions from the specified python module.

        Args:
            filepath (str): The path of the Python module.

        Returns:
            Dict: Dictionary of loaded actions.
        """
        action_objects = {}
        filename = os.path.basename(filepath)
        module = None

        if not os.path.isfile(filepath):
            log.error(f"{filepath} does not exist or is not a file.")
            log.error(f"Failed to load actions from {filename}.")
            return action_objects

        try:
            log.debug(f"Analyzing file {filename}")
            # Import the module from the file

            spec: Optional[ModuleSpec] = importlib.util.spec_from_file_location(filename, filepath)
            if not spec:
                log.error(f"Failed to create a module spec from {filepath}.")
                return action_objects

            module = importlib.util.module_from_spec(spec)
            if spec.loader:
                spec.loader.exec_module(module)

            # Loop through all members in the module and check for the `@action` decorator
            # If class has action decorator is_action class member is true
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) or inspect.isclass(obj)) and hasattr(obj, "action_meta"):
                    try:
                        actionable_name: str = getattr(obj, "action_meta").get("name")
                        action_objects[actionable_name] = obj
                        log.info(f"Added {actionable_name} to actions")
                    except Exception as e:
                        log.error(f"Failed to register {name} in action dispatcher due to exception {e}")
        except Exception as e:
            if module is None:
                raise RuntimeError(f"Failed to load actions from module at {filepath}.")
            if not module.__file__:
                raise RuntimeError(f"No file found for module {module} at {filepath}.")

            try:
                relative_filepath = Path(module.__file__).relative_to(Path.cwd())
            except ValueError:
                relative_filepath = Path(module.__file__).resolve()
            log.error(f"Failed to register {filename} in action dispatcher due to exception: {e}")

        return action_objects

    def _find_actions(self, directory) -> Dict:
        """Loop through all the subdirectories and check for the class with @action
        decorator and add in action_classes dict.

        Args:
            directory: The directory to search for actions.

        Returns:
            Dict: Dictionary of found actions.
        """
        action_objects = {}

        if not os.path.exists(directory):
            log.debug(f"_find_actions: {directory} does not exist.")
            return action_objects

        # Loop through all files in the directory and its subdirectories
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = os.path.join(root, filename)
                    if is_action_file(filepath):
                        action_objects.update(ActionDispatcher._load_actions_from_module(filepath))
        if not action_objects:
            log.debug(f"No actions found in {directory}")
            log.exception(f"No actions found in the directory {directory}.")

        return action_objects


def is_action_file(filepath):
    """Heuristics for determining if a Python file can have actions or not.

    Currently, it only excludes the `__init__.py files.
    """
    if "__init__.py" in filepath:
        return False

    return True
