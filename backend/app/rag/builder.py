"""
Build a standardized RAGResponse from raw pipeline outputs.

Retrieval quality is unchanged — rank maps are computed post-hoc for metadata only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.app.rag.schemas import (
    Answerability,
    CalculationDetail,
    Citation,
    RAGResponse,
    RetrievedChunk,
    RetrievalTrace,
    SourceMetadata,
    TokenUsage,
)
from backend.app.rag.answerability import result_to_schema

REFUSAL_PHRASE = (
    "The provided context does not contain sufficient information to answer this question."
)

TEXT_EXCERPT_MAX = 500
CITATION_EXCERPT_MAX = 300


def _excerpt(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def _best_score(hit: Dict[str, Any]) -> Optional[float]:
    for key in ("rerank_score", "rrf_score", "score", "distance"):
        val = hit.get(key)
        if val is not None:
            return float(val)
    return None


def _source_metadata(meta: Dict[str, Any]) -> SourceMetadata:
    known = {"source_file", "part", "item", "section_title", "chunk_strategy", "chunk_index", "char_count", "token_count"}
    extra = {k: v for k, v in meta.items() if k not in known and v is not None}
    return SourceMetadata(
        source_file=str(meta.get("source_file") or ""),
        part=str(meta.get("part") or ""),
        item=str(meta.get("item") or ""),
        section_title=str(meta.get("section_title") or ""),
        chunk_strategy=str(meta.get("chunk_strategy") or ""),
        chunk_index=meta.get("chunk_index"),
        char_count=meta.get("char_count"),
        token_count=meta.get("token_count"),
        extra=extra,
    )


def compute_rank_maps(
    question: str,
    collection: Any,
    bm25_retriever: Any,
    *,
    pool: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Run semantic + BM25 searches independently to annotate final hits with ranks.
    Does not alter the primary retriever output.
    """
    from retrieval.retriever import semantic_search

    sem_hits = semantic_search(collection, question, top_k=pool)
    bm25_hits = bm25_retriever.search(question, top_k=pool)

    sem_ranks = {h["id"]: rank for rank, h in enumerate(sem_hits, start=1)}
    bm25_ranks = {h["id"]: rank for rank, h in enumerate(bm25_hits, start=1)}
    return sem_ranks, bm25_ranks


def assess_answerability(hits: List[Dict[str, Any]], answer: str) -> Answerability:
    """Legacy post-hoc helper — prefer pre-generation classify_answerability()."""
    from backend.app.rag.answerability import (
        NOT_ANSWERABLE_REFUSAL,
        AnswerabilityStatus,
        classify_answerability,
        result_to_schema,
    )

    if not hits:
        from backend.app.rag.answerability import AnswerabilityResult
        return result_to_schema(AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason="No chunks retrieved for this question.",
            confidence=0.0,
            chunks_retrieved=0,
        ))

    result = classify_answerability("", hits)
    if NOT_ANSWERABLE_REFUSAL in answer or (
        "The provided context does not contain sufficient information" in answer
    ):
        from backend.app.rag.answerability import AnswerabilityResult
        return result_to_schema(AnswerabilityResult(
            status=AnswerabilityStatus.NOT_ANSWERABLE,
            reason="Retrieved context appears insufficient for a grounded answer.",
            confidence=0.2,
            chunks_retrieved=len(hits),
        ))

    return result_to_schema(result)


def build_citations(
    hits: List[Dict[str, Any]],
    cited_indices: List[int],
    *,
    filing_id: str,
    company: str,
) -> List[Citation]:
    citations: List[Citation] = []
    for idx in cited_indices:
        if not (1 <= idx <= len(hits)):
            continue
        hit = hits[idx - 1]
        meta = hit.get("metadata") or {}
        citations.append(Citation(
            index=idx,
            chunk_id=hit.get("id", ""),
            filing_id=filing_id,
            company=company,
            section_title=str(meta.get("section_title") or "") or None,
            item=str(meta.get("item") or "") or None,
            source_text_excerpt=_excerpt(hit.get("text") or "", CITATION_EXCERPT_MAX),
            score=_best_score(hit),
        ))
    return citations


def build_retrieved_chunks(
    hits: List[Dict[str, Any]],
    *,
    sem_ranks: Dict[str, int],
    bm25_ranks: Dict[str, int],
) -> List[RetrievedChunk]:
    chunks: List[RetrievedChunk] = []
    for rank, hit in enumerate(hits, start=1):
        chunk_id = hit.get("id", "")
        text = hit.get("text") or ""
        meta = hit.get("metadata") or {}

        chunks.append(RetrievedChunk(
            chunk_id=chunk_id,
            rank=rank,
            text_excerpt=_excerpt(text, TEXT_EXCERPT_MAX),
            text=text,
            source_metadata=_source_metadata(meta),
            bm25_rank=bm25_ranks.get(chunk_id),
            vector_rank=sem_ranks.get(chunk_id),
            rrf_score=float(hit["rrf_score"]) if hit.get("rrf_score") is not None else None,
            rerank_score=float(hit["rerank_score"]) if hit.get("rerank_score") is not None else None,
            vector_distance=float(hit["distance"]) if hit.get("distance") is not None else None,
        ))
    return chunks


def build_rag_response(
    *,
    question: str,
    answer: str,
    hits: List[Dict[str, Any]],
    cited_indices: List[int],
    filing_id: str,
    company: str,
    strategy: str,
    top_k: int,
    demo_mode: bool,
    model: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    collection_name: str,
    chunks_loaded: int,
    sem_ranks: Dict[str, int],
    bm25_ranks: Dict[str, int],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    embedding_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    cache_hit: bool = False,
    answerability: Optional[Answerability] = None,
    calculation: Optional[CalculationDetail] = None,
) -> RAGResponse:
    token_usage: Optional[TokenUsage] = None
    if prompt_tokens or completion_tokens or embedding_tokens:
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens or None,
            completion_tokens=completion_tokens or None,
            embedding_tokens=embedding_tokens or None,
            total_tokens=(prompt_tokens + completion_tokens + embedding_tokens) or None,
        )

    trace = RetrievalTrace(
        strategy=strategy,
        top_k=top_k,
        filing_id=filing_id,
        question=question,
        retrieval_latency_ms=round(retrieval_latency_ms, 1),
        generation_latency_ms=round(generation_latency_ms, 1),
        demo_mode=demo_mode,
        stages={
            "collection": collection_name,
            "chunks_loaded": chunks_loaded,
            "hits_returned": len(hits),
        },
    )

    resolved_answerability = answerability or assess_answerability(hits, answer)

    return RAGResponse(
        answer=answer,
        citations=build_citations(hits, cited_indices, filing_id=filing_id, company=company),
        retrieved_chunks=build_retrieved_chunks(hits, sem_ranks=sem_ranks, bm25_ranks=bm25_ranks),
        retrieval_trace=trace,
        answerability=resolved_answerability,
        model=model,
        latency_ms=round(retrieval_latency_ms + generation_latency_ms, 1),
        token_usage=token_usage,
        estimated_cost_usd=estimated_cost_usd,
        cache_hit=cache_hit,
        calculation=calculation,
    )
