"""Tests for pre-generation answerability classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.rag.answerability import (
    NOT_ANSWERABLE_REFUSAL,
    AnswerabilityStatus,
    classify_answerability,
    result_to_schema,
)


def _apple_hit(
    chunk_id: str,
    text: str,
    *,
    item: str = "Item 7",
    score: float = 8.0,
    rrf_score: float | None = None,
) -> dict:
    hit = {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source_file": "Apple.html",
            "part": "PART II",
            "item": item,
            "section_title": "Management's Discussion and Analysis",
            "chunk_strategy": "semantic",
        },
        "score": score,
    }
    if rrf_score is not None:
        hit["rrf_score"] = rrf_score
        del hit["score"]
    return hit


APPLE_FINANCIAL_HITS = [
    _apple_hit(
        "apple_item7_chunk_007",
        "Total net sales were $416.161 billion in 2025, $391.035 billion in 2024, and $383.285 billion in 2023.",
        rrf_score=0.031,
    ),
    _apple_hit(
        "apple_item8_chunk_002",
        "Consolidated Statements of Operations show net sales of $416,161 million for fiscal 2025.",
        item="Item 8",
        rrf_score=0.028,
    ),
]

WEAK_UNRELATED_HITS = [
    _apple_hit(
        "apple_item1_chunk_001",
        "The company operates retail stores in multiple countries.",
        score=1.2,
    ),
]

PARTIAL_HITS = [
    _apple_hit(
        "apple_item1a_chunk_006",
        "Tariffs and other trade restrictions can increase costs and reduce consumer demand.",
        item="Item 1A",
        rrf_score=0.029,
    ),
]


FILING_META = {
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "source_file": "Apple.html",
}


def test_answerable_financial_question():
    result = classify_answerability(
        "What were Apple's total net sales in fiscal year 2025?",
        APPLE_FINANCIAL_HITS,
        **FILING_META,
        expected_source_file="Apple.html",
    )
    assert result.status == AnswerabilityStatus.ANSWERABLE
    assert result.confidence >= 0.5
    assert result.relevant_chunk_count >= 1
    assert result.company_terms_found is True

    schema = result_to_schema(result)
    assert schema.status == "answerable"
    assert schema.answerable is True
    assert 0.0 <= schema.confidence <= 1.0


def test_not_answerable_unrelated_question():
    result = classify_answerability(
        "What is the weather forecast in Tokyo next week?",
        WEAK_UNRELATED_HITS,
        **FILING_META,
        expected_source_file="Apple.html",
    )
    assert result.status == AnswerabilityStatus.NOT_ANSWERABLE
    assert result.confidence < 0.5
    assert result.answerable is False

    schema = result_to_schema(result)
    assert schema.status == "not_answerable"
    assert schema.reason


def test_not_answerable_future_prediction():
    result = classify_answerability(
        "What will Apple's total revenue be in 2030?",
        APPLE_FINANCIAL_HITS,
        **FILING_META,
        expected_source_file="Apple.html",
    )
    assert result.status == AnswerabilityStatus.NOT_ANSWERABLE
    assert "future" in result.reason.lower() or "prediction" in result.reason.lower()
    assert result.signals.get("future_prediction_question") is True


def test_partially_answerable_question():
    result = classify_answerability(
        "What tariff risks does Apple disclose, and also what were its total net sales in 2025?",
        PARTIAL_HITS,
        **FILING_META,
        expected_source_file="Apple.html",
    )
    assert result.status == AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert result.missing_aspects
    assert 0.0 < result.confidence < 0.9

    schema = result_to_schema(result)
    assert schema.status == "partially_answerable"
    assert schema.answerable is False


def test_calculation_required():
    result = classify_answerability(
        "What was Apple's year-over-year revenue growth rate in 2025?",
        APPLE_FINANCIAL_HITS,
        **FILING_META,
        expected_source_file="Apple.html",
    )
    assert result.status == AnswerabilityStatus.CALCULATION_REQUIRED
    assert result.requires_calculation is True
    assert result.should_generate is False

    schema = result_to_schema(result)
    assert schema.status == "calculation_required"
    assert schema.requires_calculation is True


def test_not_answerable_empty_retrieval():
    result = classify_answerability(
        "What were net sales?",
        [],
        **FILING_META,
    )
    assert result.status == AnswerabilityStatus.NOT_ANSWERABLE
    assert result.chunks_retrieved == 0


def test_refusal_message_constant():
    assert "could not find enough evidence" in NOT_ANSWERABLE_REFUSAL.lower()


def test_wrong_filing_chunks_rejected():
    wrong_filing_hit = _apple_hit(
        "walmart_item7_chunk_001",
        "Walmart net sales increased.",
    )
    wrong_filing_hit["metadata"]["source_file"] = "Walmart.html"

    result = classify_answerability(
        "What were Apple's net sales?",
        [wrong_filing_hit],
        **FILING_META,
        expected_source_file="Apple.html",
    )
    assert result.status == AnswerabilityStatus.NOT_ANSWERABLE
    assert result.signals.get("wrong_filing") is True
