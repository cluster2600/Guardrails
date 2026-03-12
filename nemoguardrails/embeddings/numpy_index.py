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

"""Numpy-based drop-in replacement for annoy.AnnoyIndex.

This module provides a pure-numpy alternative to the Annoy library for
nearest-neighbour search over embedding vectors. It is used as a fallback
when annoy is not installed (e.g. on Python 3.13+ where the annoy C++
extension triggers a SIGILL).

For the typical guardrails index sizes (tens to hundreds of items) the
brute-force cosine search is more than fast enough.
"""

from typing import List, Optional, Tuple, Union

import numpy as np


class NumpyAnnoyIndex:
    """A numpy-backed nearest-neighbour index that exposes the same API surface
    as ``annoy.AnnoyIndex`` for the subset used by NeMo Guardrails.

    Supported operations:
        * ``add_item(i, vector)``
        * ``build(n_trees)``  (no-op -- kept for interface compatibility)
        * ``get_nns_by_vector(vector, n, include_distances=False)``
        * ``save(path)`` / ``load(path)``

    The metric is *angular* distance, matching Annoy's default for text
    embeddings.  Angular distance is defined as
    ``sqrt(2 * (1 - cos_sim))`` so that it is ``0`` for identical vectors
    and ``2`` for diametrically opposed ones.
    """

    def __init__(self, embedding_size: int, metric: str = "angular"):
        if metric != "angular":
            raise ValueError(
                f"NumpyAnnoyIndex only supports metric='angular', got {metric!r}"
            )
        self._embedding_size = embedding_size
        self._metric = metric
        # Sparse storage during build phase (id -> vector)
        self._vectors_dict: dict = {}
        # Dense numpy matrix after build()
        self._vectors: Optional[np.ndarray] = None
        self._built = False

    # ------------------------------------------------------------------
    # Build interface
    # ------------------------------------------------------------------

    def add_item(self, i: int, vector) -> None:
        """Add a single vector with integer id *i*."""
        self._vectors_dict[i] = np.asarray(vector, dtype=np.float32)

    def build(self, n_trees: int = 10) -> None:
        """Finalise the index.  The *n_trees* parameter is ignored (kept
        for API compatibility with Annoy)."""
        if not self._vectors_dict:
            self._vectors = np.empty((0, self._embedding_size), dtype=np.float32)
        else:
            max_id = max(self._vectors_dict.keys())
            self._vectors = np.zeros(
                (max_id + 1, self._embedding_size), dtype=np.float32
            )
            for idx, vec in self._vectors_dict.items():
                self._vectors[idx] = vec
        self._vectors_dict = {}  # release per-item dict memory now stored in _vectors
        self._built = True

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_nns_by_vector(
        self, vector, n: int, include_distances: bool = False
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """Return the *n* nearest neighbours of *vector*.

        When *include_distances* is ``True`` the return value is a tuple
        ``(ids, distances)``; otherwise just ``ids``.
        """
        if self._vectors is None or len(self._vectors) == 0:
            return ([], []) if include_distances else []

        query = np.asarray(vector, dtype=np.float32)

        # Cosine similarity via normalised dot product
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        # Avoid division by zero for zero-vectors
        safe_norms = np.where(norms == 0, 1.0, norms)
        normed = self._vectors / safe_norms

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_normed = query
        else:
            query_normed = query / query_norm

        cos_sim = normed @ query_normed  # shape: (num_items,)

        # Angular distance (matches Annoy's definition)
        cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
        distances = np.sqrt(2.0 * (1.0 - cos_sim_clipped))

        # Get top-n indices (lowest distance first)
        n = min(n, len(distances))
        if n == len(distances):
            # All items requested -- just argsort the whole array
            top_indices = np.argsort(distances)[:n]
        else:
            top_indices = np.argpartition(distances, n)[:n]
            top_indices = top_indices[np.argsort(distances[top_indices])]

        ids = top_indices.tolist()
        if include_distances:
            return ids, distances[top_indices].tolist()
        return ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the index to disk as a ``.npy`` file.

        If the caller supplies a path ending in ``.ann`` (the annoy
        convention), we silently swap the extension to ``.npy`` so that
        both backends can coexist in the same cache directory.

        Note: ``numpy.save`` automatically appends ``.npy`` when the
        path does not already end with that suffix, so callers should
        always pass either an ``.ann`` path (which is converted here)
        or an explicit ``.npy`` path.
        """
        if not self._built:
            raise RuntimeError(
                "NumpyAnnoyIndex.save() called before build(); call build() first."
            )
        if path.endswith(".ann"):
            path = path[:-4] + ".npy"
        if self._vectors is not None:
            np.save(path, self._vectors)

    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""
        if path.endswith(".ann"):
            path = path[:-4] + ".npy"
        self._vectors_dict = {}  # discard any pre-build state
        self._vectors = np.load(path).astype(np.float32)
        self._built = True
