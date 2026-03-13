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

# Generic callable type variable, used to preserve the decorated
# function's signature when applying the @cpu_bound decorator.
F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Free-threaded build detection
# ---------------------------------------------------------------------------

# At module-load time, query the CPython build configuration to
# determine whether the interpreter was compiled with the GIL disabled.
# ``sysconfig.get_config_var("Py_GIL_DISABLED")`` returns 1 on a
# free-threaded build and 0 (or None) otherwise.  The result is cached
# in a module-level constant so that the check is performed only once.
_GIL_DISABLED: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
"""True when running on a free-threaded (no-GIL) CPython build."""


def is_free_threaded() -> bool:
    """Return *True* when the interpreter was built with ``Py_GIL_DISABLED``.

    On a free-threaded build, CPU-bound work dispatched to threads can
    execute in true parallel.  On a regular build the GIL serialises
    threads, so the thread pool still prevents event-loop starvation but
    does not yield a speed-up for pure-Python CPU work.

    This helper is used both internally (e.g. in ``__repr__``) and by
    external callers who wish to adapt their behaviour depending on
    whether genuine parallelism is available.
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
    # Coroutines are already non-blocking with respect to the event loop,
    # so decorating them with @cpu_bound is a programming error.  We
    # raise early to surface the mistake at import/decoration time rather
    # than at runtime dispatch.
    if inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"@cpu_bound cannot decorate async function {fn.__qualname__!r}. "
            "Only synchronous functions benefit from thread-pool dispatch."
        )

    # Stamp a sentinel attribute onto the function object.  The
    # ActionDispatcher checks for this attribute (via ``getattr(fn,
    # '_cpu_bound', False)``) to decide whether to route the call
    # through ``RailThreadPool.dispatch()`` rather than invoking it
    # directly on the event loop thread.
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

    Configuration is driven by the ``ThreadPoolConfig`` Pydantic model
    defined in ``nemoguardrails.rails.llm.config``.  That model exposes
    ``enabled``, ``max_workers``, and ``thread_name_prefix`` fields which
    are forwarded verbatim to this constructor by ``LLMRails`` at
    initialisation time.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "nemo-rail-cpu",
        enabled: bool = True,
    ) -> None:
        # Whether the pool is logically active.  When False, dispatch()
        # falls back to inline (same-thread) execution.
        self._enabled = enabled

        # Determine the worker count.  We cap at 4 by default to avoid
        # excessive context-switching on machines with many cores; the
        # caller (or ThreadPoolConfig) may override this.
        self._max_workers = max_workers or min(4, os.cpu_count() or 1)

        # Thread name prefix makes it straightforward to identify pool
        # threads in debuggers, log output, and ``threading.enumerate()``.
        self._thread_name_prefix = thread_name_prefix

        # The underlying executor instance.  Set to None when disabled or
        # after shutdown(), which also serves as a guard in dispatch().
        self._executor: Optional[ThreadPoolExecutor] = None

        if self._enabled:
            # Eagerly create the executor so worker threads are ready
            # before the first dispatch call.
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
            # Fallback path: execute synchronously on the current thread.
            # This keeps behaviour identical to a system without the pool,
            # which is important both when the pool is explicitly disabled
            # and after shutdown() has been called.  Note that this *will*
            # block the event loop for the duration of the call.
            return fn(*args, **kwargs)

        # Obtain the currently running asyncio event loop so we can
        # schedule the synchronous callable onto the thread pool.
        loop = asyncio.get_running_loop()

        # ``loop.run_in_executor()`` expects a zero-argument callable.
        # ``functools.partial`` bundles *args and **kwargs into a single
        # callable object that satisfies this requirement.
        call = functools.partial(fn, *args, **kwargs)

        # Submit the callable to the thread-pool executor and await its
        # completion.  The awaiting coroutine is suspended (yielding
        # control back to the event loop) until the worker thread
        # finishes, so other async tasks are not starved.
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
            # Delegate to the stdlib executor's own shutdown.  The *wait*
            # parameter controls whether this call blocks until every
            # submitted task has completed; *cancel_futures* allows
            # discarding tasks that are still queued but not yet running.
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

            # Clear the reference and mark the pool as disabled so that
            # subsequent calls to dispatch() fall through to inline
            # execution rather than raising.
            self._executor = None
            self._enabled = False

    def __del__(self) -> None:
        # Best-effort cleanup invoked by the garbage collector.  Callers
        # should prefer calling ``shutdown()`` explicitly (e.g. during
        # application teardown) because __del__ timing is
        # non-deterministic and may never run if reference cycles exist.
        # We pass ``wait=False`` to avoid blocking the GC thread.
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        state = "enabled" if self._enabled else "disabled"
        return f"<RailThreadPool {state} workers={self._max_workers} free_threaded={is_free_threaded()}>"
