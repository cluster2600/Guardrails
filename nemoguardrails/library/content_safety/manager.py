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

import logging
from typing import Dict, Optional

from nemoguardrails.cache.interface import CacheInterface
from nemoguardrails.cache.lfu import LFUCache
from nemoguardrails.rails.llm.config import ModelConfig

log = logging.getLogger(__name__)


class ContentSafetyManager:
    """Manages all content safety related functionality."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._caches: Dict[str, CacheInterface] = {}
        self._initialize_caches()

    def _initialize_caches(self):
        """Initialize per-model caches based on configuration."""
        if not self.config.cache.enabled:
            return

        # We'll create caches on-demand for each model
        self._cache_config = self.config.cache

    def get_cache_for_model(self, model_name: str) -> Optional[CacheInterface]:
        """Get or create cache for a specific model."""
        if not self.config.cache.enabled:
            return None

        # Resolve model alias if configured
        actual_model = self.config.model_mapping.get(model_name, model_name)

        if actual_model not in self._caches:
            # Create cache based on store type
            if self._cache_config.store == "memory":
                # Determine persistence settings for this model
                persistence_path = None
                persistence_interval = None

                # Check if persistence is enabled and has a valid interval
                if (
                    self._cache_config.persistence.enabled
                    and self._cache_config.persistence.interval is not None
                ):
                    persistence_interval = self._cache_config.persistence.interval

                    if self._cache_config.persistence.path:
                        # Use configured path, replacing {model_name} if present
                        persistence_path = self._cache_config.persistence.path.replace(
                            "{model_name}", actual_model
                        )
                    else:
                        # Default path if persistence is enabled but no path specified
                        persistence_path = f"cache_{actual_model}.json"

                # Determine stats logging settings
                stats_logging_interval = None
                if (
                    self._cache_config.stats.enabled
                    and self._cache_config.stats.log_interval is not None
                ):
                    stats_logging_interval = self._cache_config.stats.log_interval

                self._caches[actual_model] = LFUCache(
                    capacity=self._cache_config.capacity_per_model,
                    track_stats=self._cache_config.stats.enabled,
                    persistence_interval=persistence_interval,
                    persistence_path=persistence_path,
                    stats_logging_interval=stats_logging_interval,
                )
            # elif self._cache_config.store == "filesystem":
            #     self._caches[actual_model] = FilesystemCache(...)
            # elif self._cache_config.store == "redis":
            #     self._caches[actual_model] = RedisCache(...)

        return self._caches[actual_model]

    def persist_all_caches(self):
        """Force immediate persistence of all caches that support it."""
        for model_name, cache in self._caches.items():
            if cache.supports_persistence():
                cache.persist_now()
                log.info(f"Persisted cache for model: {model_name}")
