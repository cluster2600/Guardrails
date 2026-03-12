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

"""Thread-pool executor for CPU-bound rail actions.

Python 3.14's free-threaded build (``Py_GIL_DISABLED``) enables true
parallel CPU work via threads.  This module provides:

* :class:`RailThreadPool` -- a thin wrapper around
  :class:`concurrent.futures.ThreadPoolExecutor` that dispatches
  synchronous functions onto a thread pool from the asyncio event loop.
* :func:`cpu_bound` -- a decorator that marks synchronous action
  functions for automatic thread-pool dispatch.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Free-threaded build detection
# ---------------------------------------------------------------------------

_GIL_DISABLED: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
"""True when running on a free-threaded (no-GIL) CPython build."""


def is_free_threaded() -> bool:
    """Return *True* when the interpreter was built with ``Py_GIL_DISABLED``.

    On a free-threaded build, CPU-bound work dispatched to threads can
    execute in true parallel.  On a regular build the GIL serialises
    threads, so the thread pool still prevents event-loop starvation but
    does not yield a speed-up for pure-Python CPU work.
    """
    return _GIL_DISABLED


# ---------------------------------------------------------------------------
# @cpu_bound decorator
# ---------------------------------------------------------------------------


def cpu_bound(fn: F) -> F:
    """Mark a **synchronous** action function for thread-pool dispatch.

    The decorated function gains a ``_cpu_bound = True`` attribute that
    the :class:`~nemoguardrails.actions.action_dispatcher.ActionDispatcher`
    inspects before execution.  If a :class:`RailThreadPool` is available
    the function will be run via ``loop.run_in_executor()`` instead of
    blocking the event loop.

    Raises:
        TypeError: If *fn* is a coroutine function (async def).

    Example::

        from nemoguardrails.rails.llm.thread_pool import cpu_bound

        @cpu_bound
        def heavy_pattern_match(text: str) -> bool:
            ...
    """
    if inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"@cpu_bound cannot decorate async function {fn.__qualname__!r}. "
            "Only synchronous functions benefit from thread-pool dispatch."
        )

    fn._cpu_bound = True  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# RailThreadPool
# ---------------------------------------------------------------------------


class RailThreadPool:
    """A configurable thread-pool executor for CPU-bound rail actions.

    Wraps :class:`concurrent.futures.ThreadPoolExecutor` and exposes an
    ``async def dispatch()`` helper that runs a synchronous callable in
    the pool via ``loop.run_in_executor()``.

    Parameters:
        max_workers:
            Maximum number of worker threads.  Defaults to
            ``min(4, os.cpu_count() or 1)``.
        thread_name_prefix:
            Prefix applied to worker thread names for easier debugging.
        enabled:
            When *False*, :meth:`dispatch` will run the callable directly
            in the event loop (i.e. the pool is a no-op).  Useful for
            disabling thread dispatch without changing action code.

    The pool is designed to be a **singleton per LLMRails instance**.  It
    is created once and shared across all action dispatches for that
    instance.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "nemo-rail-cpu",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._max_workers = max_workers or min(4, os.cpu_count() or 1)
        self._thread_name_prefix = thread_name_prefix
        self._executor: Optional[ThreadPoolExecutor] = None

        if self._enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix=self._thread_name_prefix,
            )
            log.info(
                "RailThreadPool started: max_workers=%d, free_threaded=%s, prefix=%r",
                self._max_workers,
                is_free_threaded(),
                self._thread_name_prefix,
            )
        else:
            log.info("RailThreadPool created but disabled; CPU-bound actions run inline.")

    # -- properties ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the thread pool is active."""
        return self._enabled

    @property
    def max_workers(self) -> int:
        """Configured maximum worker thread count."""
        return self._max_workers

    # -- dispatch ------------------------------------------------------------

    async def dispatch(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run *fn* in the thread pool and return the result.

        If the pool is disabled (or was shut down), the function is called
        directly -- ensuring backward compatibility.

        Parameters:
            fn: A **synchronous** callable.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            Whatever *fn* returns.
        """
        if not self._enabled or self._executor is None:
            # Fall back to synchronous inline execution.
            return fn(*args, **kwargs)

        loop = asyncio.get_running_loop()
        # run_in_executor only accepts callables with no kwargs,
        # so we use functools.partial to bind them.
        call = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(self._executor, call)

    # -- lifecycle -----------------------------------------------------------

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shut down the underlying thread-pool executor.

        After shutdown, :meth:`dispatch` falls back to inline execution.

        Parameters:
            wait: Block until all pending futures finish.
            cancel_futures: Cancel pending futures that have not started.
        """
        if self._executor is not None:
            log.info("Shutting down RailThreadPool (wait=%s).", wait)
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None
            self._enabled = False

    def __del__(self) -> None:
        # Best-effort cleanup; callers should use shutdown() explicitly.
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        state = "enabled" if self._enabled else "disabled"
        return f"<RailThreadPool {state} workers={self._max_workers} free_threaded={is_free_threaded()}>"
