"""
Pre-generation answerability classification using retrieval signals only.

No LLM calls by default — rules + scores + term overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOT_ANSWERABLE_REFUSAL = (
    "I could not find enough evidence in the retrieved 10-K sections to answer this question."
)

PARTIAL_GENERATION_HINT = (
    "Important: Only part of this question can be answered from the retrieved 10-K passages. "
    "Answer ONLY what the numbered context supports with citations. "
    "End with a brief note starting with 'Missing from the filing:' describing what could not be answered."
)

CALCULATION_RESPONSE_TEMPLATE = (
    "This question requires a deterministic calculation from disclosed figures rather than "
    "LLM estimation. The retrieved 10-K sections contain source data in {sections}. "
    "Use the calculation engine to compute the result from extracted values."
)


class AnswerabilityStatus(str, Enum):
    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    NOT_ANSWERABLE = "not_answerable"
    CALCULATION_REQUIRED = "calculation_required"


# Question-pattern heuristics (no LLM)
_FUTURE_PREDICTION_RE = re.compile(
    r"\b("
    r"will\s+.+\s+(grow|increase|decrease|reach|exceed|become)|"
    r"expected\s+to|forecast|predict|projection|"
    r"going\s+forward|outlook\s+for\s+\d{4}|"
    r"in\s+(20[3-9]\d|2[1-9]\d{2})|"  # years beyond typical near-term
    r"next\s+(decade|five\s+years)|"
    r"what\s+will\s+"
    r")\b",
    re.IGNORECASE,
)

_CALCULATION_RE = re.compile(
    r"\b("
    r"growth\s+rate|cagr|compound\s+annual|"
    r"percentage\s+change|percent\s+change|%+\s+change|"
    r"year[- ]over[- ]year\s+(change|growth|decline)|"
    r"y/y|yoy|"
    r"ratio\s+of|"
    r"difference\s+between|"
    r"how\s+much\s+(did|has)\s+.+\s+(increase|decrease|change)\s+by|"
    r"what\s+is\s+the\s+(percent|ratio|rate)\s+"
    r")\b",
    re.IGNORECASE,
)

_MULTI_PART_RE = re.compile(
    r"\b(and also|as well as|;\s|\.+\s+also|both\s+.+\s+and)\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "what", "how", "does", "do",
    "did", "in", "on", "for", "of", "to", "from", "with", "by", "at", "its",
    "their", "this", "that", "which", "who", "when", "where", "why", "any",
    "apple", "inc", "company", "filing", "10-k", "10k", "sec", "report",
})


@dataclass
class AnswerabilityResult:
    status: AnswerabilityStatus
    reason: str
    confidence: float
    chunks_retrieved: int
    relevant_chunk_count: int = 0
    top_score: Optional[float] = None
    top_score_normalized: float = 0.0
    term_overlap_ratio: float = 0.0
    company_terms_found: bool = False
    missing_aspects: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return self.status == AnswerabilityStatus.ANSWERABLE

    @property
    def requires_calculation(self) -> bool:
        return self.status == AnswerabilityStatus.CALCULATION_REQUIRED

    @property
    def should_generate(self) -> bool:
        return self.status in (
            AnswerabilityStatus.ANSWERABLE,
            AnswerabilityStatus.PARTIALLY_ANSWERABLE,
        )


def _question_terms(question: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _chunk_text_blob(hit: Dict[str, Any]) -> str:
    meta = hit.get("metadata") or {}
    parts = [
        hit.get("text") or "",
        str(meta.get("section_title") or ""),
        str(meta.get("item") or ""),
        str(meta.get("part") or ""),
        str(meta.get("source_file") or ""),
    ]
    return " ".join(parts).lower()


def _normalized_score(hit: Dict[str, Any]) -> float:
    """Map heterogeneous retriever scores to [0, 1] strength."""
    if hit.get("rerank_score") is not None:
        # MS-MARCO cross-encoder: roughly [-10, 10]; squash to 0-1
        raw = float(hit["rerank_score"])
        return max(0.0, min(1.0, 1.0 / (1.0 + pow(2.718, -raw))))

    if hit.get("rrf_score") is not None:
        # Typical RRF contributions ~0.01–0.035
        return max(0.0, min(1.0, float(hit["rrf_score"]) / 0.035))

    if hit.get("score") is not None:
        # BM25Okapi scores — scale loosely
        return max(0.0, min(1.0, float(hit["score"]) / 12.0))

    if hit.get("distance") is not None:
        return max(0.0, min(1.0, 1.0 - float(hit["distance"])))

    return 0.3  # unknown score type — neutral-low


def _company_terms(company: str, ticker: str, source_file: str) -> Set[str]:
    terms: Set[str] = set()
    for raw in (company, ticker, source_file):
        for w in re.findall(r"[a-z0-9]+", raw.lower()):
            if len(w) > 2:
                terms.add(w)
    # e.g. "Apple Inc." → apple
    short = company.split()[0].lower() if company else ""
    if len(short) > 2:
        terms.add(short)
    return terms


def _term_overlap_ratio(question_terms: Set[str], hits: List[Dict[str, Any]]) -> float:
    if not question_terms or not hits:
        return 0.0
    matched: Set[str] = set()
    for hit in hits[:5]:
        blob = _chunk_text_blob(hit)
        for term in question_terms:
            if term in blob:
                matched.add(term)
    return len(matched) / len(question_terms)


def _company_terms_in_hits(company_terms: Set[str], hits: List[Dict[str, Any]]) -> bool:
    if not company_terms or not hits:
        return False
    for hit in hits[:3]:
        blob = _chunk_text_blob(hit)
        if any(t in blob for t in company_terms):
            return True
    return False


def _count_relevant_chunks(hits: List[Dict[str, Any]], threshold: float = 0.25) -> int:
    return sum(1 for h in hits if _normalized_score(h) >= threshold)


def _infer_missing_aspects(question: str, question_terms: Set[str], hits: List[Dict[str, Any]]) -> List[str]:
    """Heuristic: question terms not present in any retrieved chunk."""
    if not hits:
        return ["No relevant filing sections were retrieved."]

    covered: Set[str] = set()
    for hit in hits:
        blob = _chunk_text_blob(hit)
        for term in question_terms:
            if term in blob:
                covered.add(term)

    missing_terms = sorted(question_terms - covered)
    if missing_terms:
        preview = ", ".join(missing_terms[:8])
        return [f"Topics not found in retrieved passages: {preview}"]

    if _MULTI_PART_RE.search(question):
        return ["One or more parts of this multi-part question lack direct support in the retrieved sections."]

    return ["Some aspects of the question could not be fully verified in the retrieved sections."]


def _sections_label(hits: List[Dict[str, Any]]) -> str:
    items: List[str] = []
    for hit in hits[:3]:
        meta = hit.get("metadata") or {}
        item = meta.get("item") or meta.get("section_title") or "unknown section"
        if item not in items:
            items.append(str(item))
    return ", ".join(items) if items else "the retrieved sections"


def classify_answerability(
    question: str,
    hits: List[Dict[str, Any]],
    *,
    company: str = "",
    ticker: str = "",
    source_file: str = "",
    expected_source_file: Optional[str] = None,
) -> AnswerabilityResult:
    """
    Classify whether a question should be answered, refused, partially answered,
    or routed to deterministic calculation — using retrieval signals only.
    """
    q = question.strip()
    n = len(hits)

    # Restrict to chunks from the requested filing when metadata is present
    if expected_source_file and hits:
        filing_hits = [
            h for h in hits
            if (h.get("metadata") or {}).get("source_file") == expected_source_file
        ]
        if filing_hits:
            hits = filing_hits
        elif any((h.get("metadata") or {}).get("source_file") for h in hits):
            return AnswerabilityResult(
                status=AnswerabilityStatus.NOT_ANSWERABLE,
                reason=(
                    "Retrieved passages belong to a different filing than requested; "
                    "no evidence from the target 10-K was found."
                ),
                confidence=0.05,
                chunks_retrieved=n,
                relevant_chunk_count=0,
                signals={"wrong_filing": True},
            )

    if n == 0:
        return AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason="No chunks were retrieved for this question.",
            confidence=0.0,
            chunks_retrieved=0,
            signals={"empty_retrieval": True},
        )

    top_strength = _normalized_score(hits[0])
    raw_top = hits[0].get("rerank_score") or hits[0].get("rrf_score") or hits[0].get("score") or hits[0].get("distance")
    relevant_count = _count_relevant_chunks(hits)
    q_terms = _question_terms(q)
    overlap = _term_overlap_ratio(q_terms, hits)
    comp_terms = _company_terms(company, ticker, source_file)
    company_found = _company_terms_in_hits(comp_terms, hits)

    signals: Dict[str, Any] = {
        "top_score_normalized": round(top_strength, 3),
        "relevant_chunk_count": relevant_count,
        "term_overlap_ratio": round(overlap, 3),
        "company_terms_found": company_found,
        "future_prediction_question": bool(_FUTURE_PREDICTION_RE.search(q)),
        "calculation_question": bool(_CALCULATION_RE.search(q)),
        "multi_part_question": bool(_MULTI_PART_RE.search(q)),
    }

    # --- Future prediction → not answerable (10-K is historical/disclosure, not prophecy) ---
    if signals["future_prediction_question"]:
        return AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason=(
                "The question asks for a future prediction; 10-K filings disclose historical "
                "and risk information, not forecasts of specific future outcomes."
            ),
            confidence=0.15 if top_strength > 0.3 else 0.05,
            chunks_retrieved=n,
            relevant_chunk_count=relevant_count,
            top_score=float(raw_top) if raw_top is not None else None,
            top_score_normalized=top_strength,
            term_overlap_ratio=overlap,
            company_terms_found=company_found,
            signals=signals,
        )

    # --- Weak retrieval → not answerable ---
    if top_strength < 0.18 and relevant_count == 0:
        return AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason=(
                "Retrieval scores are too low and no sufficiently relevant chunks were found "
                "in the 10-K sections."
            ),
            confidence=0.08,
            chunks_retrieved=n,
            relevant_chunk_count=relevant_count,
            top_score=float(raw_top) if raw_top is not None else None,
            top_score_normalized=top_strength,
            term_overlap_ratio=overlap,
            company_terms_found=company_found,
            signals=signals,
        )

    if overlap < 0.12 and top_strength < 0.35:
        return AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason=(
                "The question terms do not align with retrieved 10-K sections — "
                "the query appears unrelated to this filing."
            ),
            confidence=0.1,
            chunks_retrieved=n,
            relevant_chunk_count=relevant_count,
            top_score=float(raw_top) if raw_top is not None else None,
            top_score_normalized=top_strength,
            term_overlap_ratio=overlap,
            company_terms_found=company_found,
            signals=signals,
        )

    # --- Calculation required → route to deterministic engine, skip LLM math ---
    if signals["calculation_question"]:
        confidence = min(0.95, 0.5 + top_strength * 0.3 + overlap * 0.2)
        return AnswerabilityResult(
            status=AnswerabilityStatus.CALCULATION_REQUIRED,
            reason=(
                "The question requires computed metrics (growth rate, ratio, or change) "
                "that should be derived deterministically from disclosed figures, not estimated by the LLM."
            ),
            confidence=round(confidence, 3),
            chunks_retrieved=n,
            relevant_chunk_count=relevant_count,
            top_score=float(raw_top) if raw_top is not None else None,
            top_score_normalized=top_strength,
            term_overlap_ratio=overlap,
            company_terms_found=company_found,
            signals=signals,
        )

    # --- Partially answerable ---
    is_partial = (
        (signals["multi_part_question"] and overlap < 0.55)
        or (relevant_count == 1 and top_strength < 0.55)
        or (overlap < 0.35 and top_strength >= 0.25)
    )

    if is_partial:
        missing = _infer_missing_aspects(q, q_terms, hits)
        confidence = min(0.75, 0.35 + top_strength * 0.25 + overlap * 0.25)
        return AnswerabilityResult(
            status=AnswerabilityStatus.PARTIALLY_ANSWERABLE,
            reason=(
                "Some relevant 10-K sections were retrieved, but not enough to fully "
                "answer every part of the question."
            ),
            confidence=round(confidence, 3),
            chunks_retrieved=n,
            relevant_chunk_count=relevant_count,
            top_score=float(raw_top) if raw_top is not None else None,
            top_score_normalized=top_strength,
            term_overlap_ratio=overlap,
            company_terms_found=company_found,
            missing_aspects=missing,
            signals=signals,
        )

    # --- Answerable ---
    confidence = min(0.98, 0.4 + top_strength * 0.35 + overlap * 0.25 + min(relevant_count, 3) * 0.05)
    if not company_found and company:
        confidence *= 0.85

    return AnswerabilityResult(
        status=AnswerabilityStatus.ANSWERABLE,
        reason=f"Retrieved {relevant_count} relevant chunk(s) with sufficient evidence from the filing.",
        confidence=round(confidence, 3),
        chunks_retrieved=n,
        relevant_chunk_count=relevant_count,
        top_score=float(raw_top) if raw_top is not None else None,
        top_score_normalized=top_strength,
        term_overlap_ratio=overlap,
        company_terms_found=company_found,
        signals=signals,
    )


def result_to_schema(result: AnswerabilityResult):
    """Convert dataclass result to Pydantic Answerability model."""
    from backend.app.rag.schemas import Answerability

    return Answerability(
        status=result.status.value,
        reason=result.reason,
        confidence=result.confidence,
        chunks_retrieved=result.chunks_retrieved,
        answerable=result.answerable,
        relevant_chunk_count=result.relevant_chunk_count,
        missing_aspects=result.missing_aspects,
        requires_calculation=result.requires_calculation,
        signals=result.signals,
    )
