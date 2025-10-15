# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any


def extract(obj: Any, key: str, default: Any = None) -> Any:
    """Extract key from dict or attribute from object.

    This utility handles both dictionary and object attribute access,
    eliminating the need for isinstance() checks in adapter code.

    Args:
        obj: Dict or object to extract from
        key: Key/attribute name
        default: Default value if not found

    Returns:
        Extracted value or default

    Example:
        >>> extract({"role": "user"}, "role")
        'user'
        >>> extract(message_obj, "role")
        'user'
        >>> extract({}, "missing", "default")
        'default'
        >>> extract({"value": None}, "value")
        None
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
