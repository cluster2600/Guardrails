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

"""Thread-safety primitives for Python 3.14t (free-threaded / no-GIL) builds.

This module provides concurrency utilities that automatically adapt to the
runtime environment:

* On **free-threaded** Python (``Py_GIL_DISABLED=1``), every primitive uses
  real ``threading.RLock`` / ``threading.Lock`` guards so that concurrent
  threads cannot corrupt shared state.
* On **regular** (GIL-enabled) Python, the same primitives still work
  correctly but are effectively zero-overhead because the underlying lock
  objects are never contended under the GIL.

Public API
----------
``is_free_threaded``
    Detect whether the interpreter was built with ``--disable-gil``.

``ThreadSafeDict``
    A ``dict``-like wrapper that guards every mutation with an ``RLock``.

``ThreadSafeCache``
    A bounded LRU cache with lock-protected reads and writes.

``atomic_init``
    A decorator that guarantees a callable is executed exactly once, even
    when multiple threads race to trigger lazy initialization.
"""

from __future__ import annotations

import functools
import sysconfig
import threading
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
)

__all__ = [
    "is_free_threaded",
    "ThreadSafeDict",
    "ThreadSafeCache",
    "atomic_init",
]

# ---------------------------------------------------------------------------
# Runtime detection
# ---------------------------------------------------------------------------

_FREE_THREADED: Optional[bool] = None


def is_free_threaded() -> bool:
    """Return ``True`` if the running Python interpreter is a free-threaded build.

    Free-threaded Python (PEP 703) sets the ``Py_GIL_DISABLED`` sysconfig
    variable to ``1``.  This function caches the result after the first call.

    Returns:
        bool: ``True`` when running on a no-GIL / free-threaded interpreter,
              ``False`` otherwise.
    """
    global _FREE_THREADED
    if _FREE_THREADED is None:
        _FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    return _FREE_THREADED


# ---------------------------------------------------------------------------
# ThreadSafeDict
# ---------------------------------------------------------------------------

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_DT = TypeVar("_DT")


class ThreadSafeDict(dict):
    """A ``dict`` subclass that protects mutations with a ``threading.RLock``.

    All mutating operations (``__setitem__``, ``__delitem__``, ``pop``,
    ``update``, ``setdefault``, ``clear``) acquire the lock before
    modifying internal state.  Read operations (``__getitem__``,
    ``__contains__``, ``get``, ``keys``, ``values``, ``items``, ``__len__``,
    ``__iter__``) also acquire the lock to guarantee a consistent snapshot
    on free-threaded builds.

    On GIL-enabled Python the lock is still acquired but effectively
    uncontended, so the overhead is negligible.

    The class inherits from ``dict`` so it is a drop-in replacement
    everywhere a plain ``dict`` is expected (``isinstance`` checks,
    ``json.dumps``, etc.).

    Example::

        actions: ThreadSafeDict[str, Callable] = ThreadSafeDict()
        actions["greet"] = greet_action
        assert "greet" in actions
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)

    # -- mutating operations ------------------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            super().__delitem__(key)

    def pop(self, key: Any, *args: Any) -> Any:  # type: ignore[override]
        with self._lock:
            return super().pop(key, *args)

    def update(self, __m: Any = (), **kwargs: Any) -> None:  # type: ignore[override]
        with self._lock:
            super().update(__m, **kwargs)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return super().setdefault(key, default)

    def clear(self) -> None:
        with self._lock:
            super().clear()

    def popitem(self) -> Tuple[Any, Any]:
        with self._lock:
            return super().popitem()

    # -- read operations (consistent snapshot) ------------------------------

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return super().__contains__(key)

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        with self._lock:
            return super().get(key, default)

    def __len__(self) -> int:
        with self._lock:
            return super().__len__()

    def __iter__(self) -> Iterator[Any]:
        with self._lock:
            # Return an iterator over a snapshot of the keys so that the
            # caller is not affected by concurrent mutations.
            return iter(list(super().keys()))

    def keys(self) -> Any:  # type: ignore[override]
        with self._lock:
            return list(super().keys())

    def values(self) -> Any:  # type: ignore[override]
        with self._lock:
            return list(super().values())

    def items(self) -> Any:  # type: ignore[override]
        with self._lock:
            return list(super().items())

    def copy(self) -> "ThreadSafeDict":  # type: ignore[override]
        with self._lock:
            return ThreadSafeDict(super().copy())

    def __repr__(self) -> str:
        with self._lock:
            return f"{self.__class__.__name__}({super().__repr__()})"

    def __eq__(self, other: object) -> bool:
        with self._lock:
            return super().__eq__(other)

    def __bool__(self) -> bool:
        with self._lock:
            return super().__len__() > 0


# ---------------------------------------------------------------------------
# ThreadSafeCache (bounded LRU)
# ---------------------------------------------------------------------------


class ThreadSafeCache:
    """A bounded LRU cache protected by a ``threading.RLock``.

    The cache evicts the least-recently-used entry when ``maxsize`` is
    exceeded.  Both ``get`` and ``put`` are O(1) amortized (backed by an
    ``OrderedDict``).

    Args:
        maxsize: Maximum number of entries.  ``0`` means unlimited.

    Example::

        cache: ThreadSafeCache = ThreadSafeCache(maxsize=1024)
        cache.put("key", expensive_result)
        hit = cache.get("key")  # returns expensive_result
    """

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0

    # -- public API ---------------------------------------------------------

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Retrieve *key* from the cache, moving it to the MRU position.

        Args:
            key: Cache key.
            default: Value returned when *key* is absent.

        Returns:
            The cached value, or *default*.
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return default

    def put(self, key: Hashable, value: Any) -> None:
        """Insert or update *key* in the cache.

        If the cache exceeds ``maxsize``, the least-recently-used entry
        is evicted.

        Args:
            key: Cache key.
            value: Value to store.
        """
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                self._data[key] = value
                if self._maxsize > 0 and len(self._data) > self._maxsize:
                    self._data.popitem(last=False)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def invalidate(self, key: Hashable) -> None:
        """Remove a single entry.  No-op if *key* is absent."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        """Return a snapshot of cache statistics.

        Returns:
            dict: ``{"size": ..., "maxsize": ..., "hits": ..., "misses": ...}``
        """
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
            }


# ---------------------------------------------------------------------------
# atomic_init decorator
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


class _AtomicInitWrapper:
    """Internal descriptor that ensures a callable executes exactly once.

    After the first successful call, subsequent calls return the cached
    result immediately (fast-path only reads a ``bool`` flag which is safe
    even under free-threading thanks to the atomic nature of Python object
    attribute reads after publication).
    """

    def __init__(self, fn: Callable[..., _T]) -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self._initialized = False
        self._result: Any = None
        self._exc: Optional[BaseException] = None
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Fast path: already initialized.
        if self._initialized:
            if self._exc is not None:
                raise self._exc
            return self._result

        with self._lock:
            # Double-checked locking: another thread may have won the race.
            if self._initialized:
                if self._exc is not None:
                    raise self._exc
                return self._result

            try:
                self._result = self._fn(*args, **kwargs)
            except BaseException as exc:
                self._exc = exc
                self._initialized = True
                raise
            else:
                self._initialized = True
                return self._result

    def reset(self) -> None:
        """Reset the wrapper so the function can be called again.

        Intended for testing purposes only.
        """
        with self._lock:
            self._initialized = False
            self._result = None
            self._exc = None


def atomic_init(fn: Callable[..., _T]) -> _AtomicInitWrapper:
    """Decorator ensuring *fn* is executed exactly once (thread-safe).

    On the first invocation the wrapped function runs under a
    ``threading.Lock``.  All subsequent invocations return the cached
    result without acquiring the lock (double-checked locking pattern).

    This is useful for lazy, one-time initialization of expensive
    resources (models, indices, etc.) that may be triggered from
    multiple threads simultaneously.

    Args:
        fn: The callable to wrap.

    Returns:
        An ``_AtomicInitWrapper`` instance that behaves like the original
        callable but guarantees single execution.

    Example::

        @atomic_init
        def load_model():
            return HeavyModel()

        # Both threads get the same instance, load_model() runs once.
        model = load_model()
    """
    return _AtomicInitWrapper(fn)
