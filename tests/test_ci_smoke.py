"""CI smoke tests — no OpenAI calls; uses demo_mode and offline bootstrap index."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _offline_index_ready() -> bool:
    if not (ROOT / "db").exists():
        return False
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


def test_backend_imports():
    from backend.app.adapters.rag_adapter import ask_question, index_is_ready
    from backend.app.filings.registry import list_filing_ids
    from backend.app.main import app

    assert app is not None
    assert len(list_filing_ids()) == 5
    assert callable(ask_question)
    assert callable(index_is_ready)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert body["version"]
    assert body["filing_count"] == 5
    assert len(body["available_filings"]) == 5


def test_sample_questions_endpoint(client):
    response = client.get("/api/sample-questions")
    assert response.status_code == 200
    body = response.json()
    assert body["filing_id"] == "apple_2025"
    assert len(body["questions"]) >= 1
    assert body["questions"][0]["question"]


def test_filings_catalog_endpoint(client):
    response = client.get("/api/filings")
    assert response.status_code == 200
    body = response.json()
    assert len(body["filings"]) == 5
    assert body["filings"][0]["label"]


@pytest.mark.skipif(
    not _offline_index_ready(),
    reason="Run scripts/ci_bootstrap_db.py with backend deps installed",
)
def test_ask_demo_mode_no_openai(client):
    """POST /api/ask in demo_mode — BM25 only, zero LLM cost."""
    response = client.post(
        "/api/ask",
        json={
            "question": "What were Apple's total net sales in fiscal year 2025?",
            "filing_id": "apple_2025",
            "demo_mode": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["estimated_cost_usd"] == 0.0
    assert len(body["retrieved_chunks"]) > 0
    assert body["retrieval_trace"]["demo_mode"] is True
    assert body["retrieved_chunks"][0]["source_metadata"]["source_file"] == "Apple.html"
