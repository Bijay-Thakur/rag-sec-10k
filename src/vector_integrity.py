"""Fail-closed checks for a Chroma collection used in semantic retrieval.

Collection metadata identifies how vectors were generated; checking their values
also catches empty, non-finite, or incompatible embeddings. This is intentionally
read-only and can be run against a staging copy before switching traffic.
"""

from __future__ import annotations

import math
from typing import Any

EMBEDDING_DIMENSIONS = {"text-embedding-3-small": 1536}
INDEX_VERSION = "openai-embeddings-v1"


class InvalidVectorIndex(ValueError):
    """The index cannot be trusted for semantic retrieval."""


def verify_collection(collection: Any, *, model: str, batch_size: int = 64) -> int:
    """Check provenance and every stored vector; return the number inspected."""
    dimensions = EMBEDDING_DIMENSIONS.get(model)
    if dimensions is None:
        raise InvalidVectorIndex(f"Unsupported embedding model: {model}")
    meta = collection.metadata or {}
    if (meta.get("index_version") != INDEX_VERSION
            or meta.get("embedding_source") != "openai"
            or meta.get("embedding_model") != model
            or meta.get("embedding_dimensions") != dimensions):
        raise InvalidVectorIndex("Missing or incompatible embedding provenance")
    count = collection.count()
    if count == 0:
        raise InvalidVectorIndex("Empty vector index")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for offset in range(0, count, batch_size):
        page = collection.get(limit=min(batch_size, count - offset),
                              offset=offset, include=["embeddings"])
        vectors = page.get("embeddings")
        if vectors is None or len(vectors) != min(batch_size, count - offset):
            raise InvalidVectorIndex("Missing vectors in index page")
        for vector in vectors:
            if len(vector) != dimensions:
                raise InvalidVectorIndex("Embedding dimension mismatch")
            if not all(math.isfinite(float(value)) for value in vector):
                raise InvalidVectorIndex("Non-finite embedding")
            if not any(float(value) != 0.0 for value in vector):
                raise InvalidVectorIndex("Zero embedding")
    return count
