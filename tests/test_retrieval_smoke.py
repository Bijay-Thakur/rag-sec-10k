"""
Optional integration checks for embed + retrieval.

Requires:
  - OPENAI_API_KEY
  - data/chunks/*.jsonl and a populated ./db from Embed/embed.py

Run:
  set PYTHONPATH=src && pytest tests/test_retrieval_smoke.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def _have_index() -> bool:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        return False
    db = REPO / "db"
    if not db.is_dir():
        return False
    try:
        cl = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(db))
        )
        return (
            cl.get_collection("semantic_index").count() > 0
            and cl.get_collection("recursive_index").count() > 0
        )
    except Exception:
        return False


needs_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
needs_index = pytest.mark.skipif(
    not _have_index(),
    reason="Chroma db missing or empty — run: PYTHONPATH=src python src/Embed/embed.py",
)


@pytest.fixture(scope="module", autouse=True)
def path_setup():
    sys.path.insert(0, str(SRC))
    yield


@needs_key
@needs_index
def test_retrieve_semantic_returns_ranked_hits():
    from retrieval.retrieve import retrieve, SEMANTIC_INDEX

    hits = retrieve("What products or services does the company describe?", n_results=3)
    assert len(hits) == 3
    assert all("chunk_id" in h and "document" in h for h in hits)
    assert all("distance" in h for h in hits)
    assert all(isinstance(h["distance"], (int, float)) for h in hits)


@needs_key
@needs_index
def test_retrieve_fused_returns_hits():
    from retrieval.retrieve import retrieve_fused

    hits = retrieve_fused("revenue and financial performance", max_results=4)
    assert 1 <= len(hits) <= 4
    assert all("fusion_score" in h for h in hits)
