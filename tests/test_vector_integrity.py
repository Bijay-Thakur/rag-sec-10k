"""Semantic search must fail closed on fixture or corrupt vectors."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vector_integrity import (
    EMBEDDING_DIMENSIONS,
    INDEX_VERSION,
    InvalidVectorIndex,
    verify_collection,
)

MODEL = "text-embedding-3-small"


class FakeCollection:
    def __init__(self, vectors, metadata=None):
        self.vectors = vectors
        self.metadata = metadata or {
            "index_version": INDEX_VERSION,
            "embedding_source": "openai",
            "embedding_model": MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS[MODEL],
        }

    def count(self):
        return len(self.vectors)

    def get(self, *, limit, offset, include):
        assert include == ["embeddings"]
        return {"embeddings": self.vectors[offset:offset + limit]}


def real_vector():
    return [1.0] + [0.0] * 1535


def test_checks_all_pages_and_accepts_declared_real_vectors():
    assert verify_collection(FakeCollection([real_vector()] * 3), model=MODEL, batch_size=2) == 3


@pytest.mark.parametrize("vector,reason", [
    ([0.0] * 1536, "Zero"),
    ([1.0], "dimension"),
    ([math.nan] + [0.0] * 1535, "Non-finite"),
    ([math.inf] + [0.0] * 1535, "Non-finite"),
])
def test_rejects_corrupt_vector_even_on_later_page(vector, reason):
    with pytest.raises(InvalidVectorIndex, match=reason):
        verify_collection(FakeCollection([real_vector(), vector]), model=MODEL, batch_size=1)


def test_rejects_unproven_and_wrong_model():
    with pytest.raises(InvalidVectorIndex, match="provenance"):
        verify_collection(FakeCollection([real_vector()], {"index_kind": "offline_test_fixture"}), model=MODEL)
    with pytest.raises(InvalidVectorIndex, match="Unsupported"):
        verify_collection(FakeCollection([real_vector()]), model="unknown")


def test_demo_does_not_require_index(monkeypatch, tmp_path: Path):
    from backend.app.adapters import rag_adapter
    from backend.app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "chroma_db_dir", tmp_path / "absent")
    rag_adapter._resources_by_filing.clear()
    resources = rag_adapter._get_resources("apple_2025", settings)
    assert resources.collection is None
    assert resources.bm25_retriever.search("Apple sales", top_k=1)
    assert not rag_adapter.index_is_ready("apple_2025")
    rag_adapter._query_cache.clear()
    response = rag_adapter.ask_question(
        "What did Apple report about its total net sales in 2025?",
        filing_id="apple_2025", demo_mode=True,
    )
    assert response.retrieved_chunks
    assert response.estimated_cost_usd == 0
    with pytest.raises(rag_adapter.RAGPipelineError) as exc:
        rag_adapter._get_resources("apple_2025", settings, require_semantic=True)
    assert exc.value.error_code == "index_unavailable"
    rag_adapter._resources_by_filing.clear()


def test_offline_chroma_fixture_cannot_enable_semantic(monkeypatch, tmp_path: Path):
    import chromadb

    from backend.app.adapters import rag_adapter
    from backend.app.config import get_settings

    db = tmp_path / "dummy-index"
    collection = chromadb.PersistentClient(path=str(db)).create_collection(
        "semantic_index", metadata={"index_kind": "offline_test_fixture"},
    )
    collection.add(ids=["dummy"], documents=["test"],
                   embeddings=[[0.0] * 1536], metadatas=[{"source_file": "Apple.html"}])
    settings = get_settings()
    monkeypatch.setattr(settings, "chroma_db_dir", db)
    rag_adapter._resources_by_filing.clear()
    with pytest.raises(rag_adapter.RAGPipelineError) as exc:
        rag_adapter._get_resources("apple_2025", settings, require_semantic=True)
    assert exc.value.error_code == "index_invalid"
    rag_adapter._resources_by_filing.clear()
