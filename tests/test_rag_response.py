"""Tests for standardized RAGResponse schema and builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from backend.app.rag.builder import build_rag_response
from backend.app.rag.schemas import Answerability, RAGResponse


SAMPLE_HITS = [
    {
        "id": "apple_item7_chunk_007",
        "text": "Total net sales were $416.161 billion in 2025.",
        "metadata": {
            "source_file": "Apple.html",
            "part": "PART II",
            "item": "Item 7",
            "section_title": "Management's Discussion and Analysis",
            "chunk_strategy": "semantic",
            "chunk_index": 7,
            "char_count": 48,
        },
        "rrf_score": 0.032,
    },
    {
        "id": "apple_item8_chunk_002",
        "text": "Consolidated net sales for 2025, 2024 and 2023.",
        "metadata": {
            "source_file": "Apple.html",
            "part": "PART II",
            "item": "Item 8",
            "section_title": "Financial Statements",
            "chunk_strategy": "semantic",
        },
        "rrf_score": 0.028,
    },
]


def test_rag_response_schema_has_required_fields():
    resp = build_rag_response(
        question="What were net sales?",
        answer="Net sales were $416.161 billion [1].",
        hits=SAMPLE_HITS,
        cited_indices=[1],
        filing_id="apple_2025",
        company="Apple Inc.",
        strategy="hybrid",
        top_k=5,
        demo_mode=False,
        model="gpt-4o-mini",
        retrieval_latency_ms=12.5,
        generation_latency_ms=850.0,
        collection_name="semantic_index",
        chunks_loaded=2812,
        sem_ranks={"apple_item7_chunk_007": 1, "apple_item8_chunk_002": 3},
        bm25_ranks={"apple_item7_chunk_007": 2, "apple_item8_chunk_002": 1},
        prompt_tokens=1200,
        completion_tokens=45,
        embedding_tokens=12,
        estimated_cost_usd=0.0002,
        answerability=Answerability(
            status="answerable",
            reason="Test fixture.",
            confidence=0.9,
            chunks_retrieved=2,
            answerable=True,
            relevant_chunk_count=2,
        ),
    )

    assert isinstance(resp, RAGResponse)
    data = json.loads(resp.to_json())

    for key in (
        "answer",
        "citations",
        "retrieved_chunks",
        "retrieval_trace",
        "answerability",
        "model",
        "latency_ms",
        "token_usage",
        "estimated_cost_usd",
        "cache_hit",
    ):
        assert key in data

    assert data["token_usage"]["prompt_tokens"] == 1200
    assert data["token_usage"]["total_tokens"] == 1257
    assert data["answerability"]["status"] == "answerable"
    assert data["answerability"]["confidence"] == 0.9
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "apple_item7_chunk_007"
    assert data["citations"][0]["filing_id"] == "apple_2025"
    assert data["citations"][0]["company"] == "Apple Inc."
    assert data["citations"][0]["section_title"] == "Management's Discussion and Analysis"
    assert data["citations"][0]["score"] == pytest.approx(0.032)

    chunk0 = data["retrieved_chunks"][0]
    assert chunk0["chunk_id"] == "apple_item7_chunk_007"
    assert chunk0["bm25_rank"] == 2
    assert chunk0["vector_rank"] == 1
    assert chunk0["rrf_score"] == pytest.approx(0.032)
    assert chunk0["source_metadata"]["item"] == "Item 7"
    assert "text_excerpt" in chunk0
    assert "text" in chunk0


def test_rag_response_json_is_frontend_ready():
    """Response JSON should be self-contained for UI rendering."""
    resp = build_rag_response(
        question="Test?",
        answer="Answer [1].",
        hits=SAMPLE_HITS[:1],
        cited_indices=[1],
        filing_id="apple_2025",
        company="Apple Inc.",
        strategy="bm25",
        top_k=5,
        demo_mode=True,
        model="demo-mode/no-llm",
        retrieval_latency_ms=5.0,
        generation_latency_ms=0.0,
        collection_name="semantic_index",
        chunks_loaded=100,
        sem_ranks={},
        bm25_ranks={"apple_item7_chunk_007": 1},
        answerability=Answerability(
            status="answerable",
            reason="Demo test.",
            confidence=0.7,
            chunks_retrieved=1,
            answerable=True,
        ),
    )

    parsed = json.loads(resp.to_json())
    assert parsed["retrieval_trace"]["question"] == "Test?"
    assert parsed["answerability"]["answerable"] is True
    assert parsed["cache_hit"] is False


@pytest.mark.skipif(
    not (ROOT / "db").exists(),
    reason="ChromaDB not built",
)
def test_ask_question_smoke_demo_mode():
    """Integration smoke: pipeline returns valid RAGResponse in demo mode."""
    from backend.app.adapters.rag_adapter import ask_question

    resp = ask_question(
        "What were Apple's total net sales in fiscal year 2025?",
        demo_mode=True,
    )
    assert isinstance(resp, RAGResponse)
    assert resp.answer
    assert len(resp.retrieved_chunks) > 0
    assert resp.retrieved_chunks[0].chunk_id
    assert resp.retrieval_trace.filing_id == "apple_2025"
    assert resp.estimated_cost_usd == 0.0

    # Must serialize cleanly
    json.loads(resp.to_json())
