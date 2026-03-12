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

"""Compatibility shim for OpenTelemetry on Python 3.14+ (PEP 649).

Python 3.14 introduces PEP 649 (deferred evaluation of annotations).
Under PEP 649, type annotations are resolved lazily, which can cause
``TypeError`` when importing modules that use complex type annotations
if those annotations reference names that are shadowed or unavailable
at resolution time.

The OpenTelemetry SDK and API packages may trigger these issues on
Python 3.14 depending on the installed version.  This module provides
safe import helpers that catch both ``ImportError`` (package not
installed) and ``TypeError`` (PEP 649 annotation resolution failure)
so that tracing degrades gracefully instead of crashing.
"""

import logging
import sys

log = logging.getLogger(__name__)

_NEEDS_PEP649_GUARD = sys.version_info >= (3, 14)

# Sentinel indicating OTel is not available
OTEL_AVAILABLE = False

try:
    from opentelemetry import trace  # noqa: F401
    from opentelemetry.trace import NoOpTracerProvider  # noqa: F401

    OTEL_AVAILABLE = True
except (ImportError, TypeError) as exc:
    if _NEEDS_PEP649_GUARD and isinstance(exc, TypeError):
        log.warning(
            "OpenTelemetry API import failed on Python %s due to a TypeError "
            "(likely PEP 649 annotation resolution). Tracing will be disabled. "
            "Error: %s",
            sys.version,
            exc,
        )
    # If ImportError, OTel simply isn't installed — no warning needed.


def safe_import_otel_trace():
    """Import ``opentelemetry.trace`` with PEP 649 safety.

    Returns the ``trace`` module or ``None`` if OTel is not available.
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        from opentelemetry import trace

        return trace
    except (ImportError, TypeError):
        return None


def safe_import_otel_sdk_tracer_provider():
    """Import ``TracerProvider`` from the OTel SDK with PEP 649 safety.

    Returns the ``TracerProvider`` class or ``None`` if the SDK is not
    available.  This is only needed by tests or applications that configure
    the SDK; the adapter itself only uses the API.
    """
    try:
        from opentelemetry.sdk.trace import TracerProvider

        return TracerProvider
    except (ImportError, TypeError):
        return None
