"""
Adapter layer over the existing v1 RAG pipeline (src/retrieval + src/generation).

Does not rewrite retrieval logic — delegates to get_retriever() and generate_answer(),
then normalizes output into RAGResponse.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import Settings, get_settings
from backend.app.filings.registry import FILING_REGISTRY
from backend.app.rag.answerability import (
    NOT_ANSWERABLE_REFUSAL,
    PARTIAL_GENERATION_HINT,
    AnswerabilityResult,
    AnswerabilityStatus,
    classify_answerability,
    result_to_schema,
)
from backend.app.rag.calculation.engine import CALC_EXPLAIN_HINT, format_calc_block, run_calculation
from backend.app.rag.builder import build_rag_response
from backend.app.rag.schemas import CalculationDetail, RAGResponse

SAMPLE_QUESTIONS: Dict[str, List[Dict[str, str]]] = {
    "apple_2025": [
        {"question": "What tariff and trade risks does Apple describe in its 2025 10-K?", "category": "risk_factors"},
        {"question": "What were Apple's total net sales in fiscal year 2025?", "category": "financials"},
        {"question": "Which geographic segments does Apple use to report its revenue?", "category": "segments"},
        {"question": "What cybersecurity risks does Apple disclose?", "category": "risk_factors"},
        {"question": "How did Apple's R&D spending change year-over-year?", "category": "financials"},
        {"question": "What does Apple say about competition in the smartphone market?", "category": "business"},
    ],
    "walmart_2026": [
        {"question": "What is Walmart's omnichannel strategy?", "category": "business"},
        {"question": "What were Walmart's total revenues in fiscal 2026?", "category": "financials"},
        {"question": "What cybersecurity risks does Walmart disclose?", "category": "risk_factors"},
        {"question": "How did Walmart U.S. comparable sales change in fiscal 2026?", "category": "financials"},
        {"question": "What eCommerce contribution did Walmart U.S. report?", "category": "segments"},
    ],
    "exxon_2025": [
        {"question": "What are ExxonMobil's reportable business segments?", "category": "segments"},
        {"question": "What oil-equivalent proved reserves does Exxon disclose?", "category": "financials"},
        {"question": "What climate and energy transition risks does Exxon highlight?", "category": "risk_factors"},
        {"question": "What were upstream earnings in 2025?", "category": "financials"},
    ],
    "elilily_2025": [
        {"question": "What are Eli Lilly's main pharmaceutical product areas?", "category": "business"},
        {"question": "What R&D risks does Eli Lilly describe?", "category": "risk_factors"},
        {"question": "What was Eli Lilly's total revenue in 2025?", "category": "financials"},
        {"question": "How does Lilly describe its cybersecurity program?", "category": "risk_factors"},
    ],
    "chase_2025": [
        {"question": "What are JPMorgan Chase's three reportable business segments?", "category": "segments"},
        {"question": "What capital and liquidity requirements apply to JPMorgan Chase?", "category": "regulation"},
        {"question": "What were JPMorgan Chase's total assets as of December 31, 2025?", "category": "financials"},
        {"question": "What consumer protection regulations apply to Chase?", "category": "regulation"},
    ],
}


class RAGPipelineError(Exception):
    """Raised when the underlying RAG pipeline fails."""

    def __init__(self, message: str, *, error_code: str = "pipeline_error") -> None:
        super().__init__(message)
        self.error_code = error_code


# ---------------------------------------------------------------------------
# Lazy-loaded pipeline resources (mirrors Streamlit cache pattern)
# ---------------------------------------------------------------------------

@dataclass
class _PipelineResources:
    collection: Any
    bm25_retriever: Any
    chunks: List[Dict[str, Any]]


_resources_by_filing: Dict[str, _PipelineResources] = {}
_resources_lock = Lock()

_query_cache: OrderedDict[str, RAGResponse] = OrderedDict()
_cache_lock = Lock()


def _resolve_filing_id(filing_id: Optional[str], settings: Settings) -> str:
    fid = (filing_id or settings.default_filing_id).strip().lower()
    if fid not in FILING_REGISTRY:
        raise RAGPipelineError(
            f"Unknown filing_id '{filing_id}'. Available: {list(FILING_REGISTRY)}",
            error_code="unknown_filing",
        )
    return fid


def _load_chunks(chunks_path, settings: Settings, source_file: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            c = json.loads(s)
            meta = c.get("metadata", {})
            if meta.get("source_file") != source_file:
                continue
            rows.append({
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "metadata": meta,
            })
    return rows


def _get_resources(filing_id: str, settings: Settings) -> _PipelineResources:
    meta = FILING_REGISTRY[filing_id]
    with _resources_lock:
        cached = _resources_by_filing.get(filing_id)
        if cached is not None:
            return cached

        try:
            import chromadb
            from retrieval.retriever import BM25Retriever
        except ImportError as exc:
            raise RAGPipelineError(
                f"Could not import v1 retrieval modules: {exc}",
                error_code="import_error",
            ) from exc

        chunks_path = settings.chunks_dir / meta["chunks_file"]
        if not chunks_path.is_file():
            raise RAGPipelineError(
                f"Chunks file not found: {chunks_path}",
                error_code="chunks_missing",
            )

        try:
            client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
            collection = client.get_collection(name=meta["collection"])
            if collection.count() == 0:
                raise RAGPipelineError(
                    f"Chroma collection '{meta['collection']}' is empty. "
                    "Run: python src/Embed/embed.py",
                    error_code="index_empty",
                )
        except RAGPipelineError:
            raise
        except Exception as exc:
            raise RAGPipelineError(
                f"ChromaDB unavailable: {exc}",
                error_code="index_unavailable",
            ) from exc

        chunks = _load_chunks(chunks_path, settings, meta["source_file"])
        if not chunks:
            raise RAGPipelineError(
                f"No chunks found for {filing_id} (source_file={meta['source_file']}). "
                "Re-run ingestion and embedding.",
                error_code="chunks_missing",
            )

        bm25 = BM25Retriever(chunks)
        resources = _PipelineResources(collection=collection, bm25_retriever=bm25, chunks=chunks)
        _resources_by_filing[filing_id] = resources
        return resources


def index_is_ready(filing_id: Optional[str] = None) -> bool:
    settings = get_settings()
    try:
        fid = _resolve_filing_id(filing_id, settings)
        _get_resources(fid, settings)
        return True
    except RAGPipelineError:
        return False


def get_sample_questions(filing_id: Optional[str] = None) -> Tuple[str, List[Dict[str, str]]]:
    settings = get_settings()
    fid = _resolve_filing_id(filing_id, settings)
    return fid, SAMPLE_QUESTIONS.get(fid, [])


def _cache_key(question: str, filing_id: str, demo_mode: bool, strategy: str, top_k: int) -> str:
    return f"{filing_id}|{demo_mode}|{strategy}|{top_k}|{question.strip().lower()}"


def _get_cached(key: str) -> Optional[RAGResponse]:
    with _cache_lock:
        if key in _query_cache:
            _query_cache.move_to_end(key)
            return _query_cache[key].model_copy(update={"cache_hit": True})
    return None


def _set_cache(key: str, response: RAGResponse, settings: Settings) -> None:
    with _cache_lock:
        _query_cache[key] = response
        _query_cache.move_to_end(key)
        while len(_query_cache) > settings.query_cache_max_entries:
            _query_cache.popitem(last=False)


def _normalize_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize retriever output to a consistent shape for generation."""
    return {
        "id": hit.get("id") or hit.get("chunk_id", ""),
        "text": hit.get("text") or hit.get("document") or "",
        "metadata": hit.get("metadata") or {},
        **{k: hit[k] for k in ("distance", "rrf_score", "score", "rerank_score") if k in hit},
    }


def _estimate_cost_usd(
    *,
    settings: Settings,
    embed_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    demo_mode: bool,
) -> float:
    if demo_mode:
        return 0.0
    embed_cost = (embed_tokens / 1_000_000) * settings.embed_input_price_per_1m
    gen_in = (prompt_tokens / 1_000_000) * settings.gpt4o_mini_input_price_per_1m
    gen_out = (completion_tokens / 1_000_000) * settings.gpt4o_mini_output_price_per_1m
    return round(embed_cost + gen_in + gen_out, 6)


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split()))


def _truncate_hits_for_input_budget(
    hits: List[Dict[str, Any]],
    max_input_tokens: int,
) -> List[Dict[str, Any]]:
    """Trim retrieved passages to stay within a rough input token budget."""
    if max_input_tokens <= 0 or not hits:
        return hits
    budget = max_input_tokens
    trimmed: List[Dict[str, Any]] = []
    for hit in hits:
        text = hit.get("text") or hit.get("document") or ""
        words = text.split()
        if not words:
            trimmed.append(hit)
            continue
        if len(words) > budget:
            short = " ".join(words[: max(budget, 1)])
            trimmed.append({**hit, "text": f"{short}…"})
            budget = 0
        else:
            trimmed.append(hit)
            budget -= len(words)
        if budget <= 0:
            break
    return trimmed or hits[:1]


def _generation_kwargs(settings: Settings) -> Dict[str, Any]:
    return {
        "model": settings.generation_model,
        "max_tokens": settings.max_output_tokens,
    }


def _demo_answer(hits: List[Dict[str, Any]], question: str) -> str:
    if not hits:
        return (
            "Demo mode: no relevant passages were retrieved. "
            "Try a different question or disable demo_mode for full generation."
        )
    top = hits[0]
    meta = top.get("metadata") or {}
    item = meta.get("item", "unknown section")
    preview = (top.get("text") or "")[:500].strip()
    return (
        f"**Demo mode** — retrieval preview only (no LLM call).\n\n"
        f"Top match from **{item}**:\n\n"
        f"> {preview}{'…' if len(top.get('text') or '') > 500 else ''}\n\n"
        f"_Question: {question}_"
    )


def _rank_maps_for_hits(
    question: str,
    resources: _PipelineResources,
    hits: List[Dict[str, Any]],
    top_k: int,
    *,
    include_semantic: bool,
    source_file: Optional[str] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Annotate hits with BM25/vector ranks without changing retrieval output."""
    pool = max(top_k * 2, len(hits), 10)
    bm25_hits = resources.bm25_retriever.search(question, top_k=pool)
    bm25_ranks = {h["id"]: rank for rank, h in enumerate(bm25_hits, start=1)}

    sem_ranks: Dict[str, int] = {}
    if include_semantic:
        try:
            from retrieval.retriever import semantic_search
            sem_hits = semantic_search(
                resources.collection,
                question,
                top_k=pool,
                source_file=source_file,
            )
            sem_ranks = {h["id"]: rank for rank, h in enumerate(sem_hits, start=1)}
        except Exception:
            sem_ranks = {}

    return sem_ranks, bm25_ranks


def ask_question(
    question: str,
    *,
    filing_id: Optional[str] = None,
    demo_mode: bool = False,
) -> RAGResponse:
    """
    Run the full RAG pipeline (or demo retrieval) and return a standardized RAGResponse.
    """
    settings = get_settings()
    fid = _resolve_filing_id(filing_id, settings)
    strategy = "bm25" if demo_mode else settings.retrieval_strategy
    top_k = settings.retrieval_top_k

    cache_key = _cache_key(question, fid, demo_mode, strategy, top_k)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    if not demo_mode and not settings.openai_api_key:
        raise RAGPipelineError(
            "OPENAI_API_KEY is not configured. Set it in .env or use demo_mode=true.",
            error_code="missing_api_key",
        )

    resources = _get_resources(fid, settings)
    meta = FILING_REGISTRY[fid]

    try:
        from retrieval.retriever import get_retriever
        from generation.generator import generate_answer
    except ImportError as exc:
        raise RAGPipelineError(
            f"Could not import pipeline modules: {exc}",
            error_code="import_error",
        ) from exc

    retriever = get_retriever(
        collection_name=meta["collection"],
        chunks=resources.chunks,
        strategy=strategy,  # type: ignore[arg-type]
        source_file=meta["source_file"],
    )

    t0 = time.perf_counter()
    try:
        raw_hits = retriever(question, top_k=top_k)
    except Exception as exc:
        raise RAGPipelineError(
            f"Retrieval failed: {exc}",
            error_code="retrieval_failed",
        ) from exc
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    hits = [_normalize_hit(h) for h in raw_hits]
    llm_hits = _truncate_hits_for_input_budget(hits, settings.max_input_tokens)
    gen_kwargs = _generation_kwargs(settings)

    sem_ranks, bm25_ranks = _rank_maps_for_hits(
        question,
        resources,
        hits,
        top_k,
        include_semantic=bool(settings.openai_api_key),
        source_file=meta["source_file"],
    )

    embed_tokens = 0 if demo_mode or strategy == "bm25" else _rough_token_count(question)
    prompt_tokens = 0
    completion_tokens = 0
    generation_ms = 0.0
    cited_indices: List[int] = []
    model = settings.generation_model
    calculation_detail: Optional[CalculationDetail] = None

    # --- Pre-generation answerability gate (retrieval signals only) ---
    answerability_result = classify_answerability(
        question,
        hits,
        company=meta["company"],
        ticker=meta.get("ticker", ""),
        source_file=meta.get("source_file", ""),
        expected_source_file=meta.get("source_file"),
    )
    answerability_schema = result_to_schema(answerability_result)

    if demo_mode:
        answer = _demo_answer(hits, question)
        model = "demo-mode/no-llm"
        if answerability_result.status == AnswerabilityStatus.CALCULATION_REQUIRED:
            calc_result = run_calculation(question, hits)
            if calc_result.success and calc_result.detail:
                calculation_detail = calc_result.detail
                answer = calc_result.answer_text
                cited_indices = _chunk_ids_to_citation_indices(
                    calc_result.detail.source_chunk_ids, hits,
                )
    elif answerability_result.status == AnswerabilityStatus.NOT_ANSWERABLE:
        answer = NOT_ANSWERABLE_REFUSAL
        model = "none/refusal"
    elif answerability_result.status == AnswerabilityStatus.CALCULATION_REQUIRED:
        calc_result = run_calculation(question, hits)
        if calc_result.success and calc_result.detail:
            calculation_detail = calc_result.detail
            cited_indices = _chunk_ids_to_citation_indices(
                calc_result.detail.source_chunk_ids, hits,
            )
            answerability_schema = result_to_schema(AnswerabilityResult(
                status=AnswerabilityStatus.ANSWERABLE,
                reason="Answer computed deterministically from disclosed filing figures.",
                confidence=calc_result.detail.confidence,
                chunks_retrieved=len(hits),
                relevant_chunk_count=answerability_result.relevant_chunk_count,
                signals={**answerability_result.signals, "deterministic_calculation": True},
            ))
            answerability_schema.requires_calculation = True

            calc_block = format_calc_block(calc_result.detail)
            gen_question = (
                f"{question}\n\n"
                f"{CALC_EXPLAIN_HINT.format(calc_block=calc_block)}"
            )
            t1 = time.perf_counter()
            try:
                gen_result = generate_answer(
                    gen_question,
                    llm_hits,
                    **gen_kwargs,
                )
            except Exception as exc:
                raise RAGPipelineError(
                    f"Generation failed: {exc}",
                    error_code="generation_failed",
                ) from exc
            generation_ms = (time.perf_counter() - t1) * 1000.0
            answer = gen_result.answer
            llm_cited = gen_result.cited_indices
            cited_indices = llm_cited or cited_indices
            prompt_tokens = gen_result.prompt_tokens
            completion_tokens = gen_result.completion_tokens
            model = f"{settings.generation_model}+python/calculation"
        else:
            # Extraction failed — do not let LLM invent numbers
            if demo_mode:
                answer = (
                    f"Could not confidently extract figures for a deterministic calculation. "
                    f"{calc_result.partial_reason}"
                )
                model = "python/calculation-failed"
            else:
                answerability_schema = result_to_schema(AnswerabilityResult(
                    status=AnswerabilityStatus.PARTIALLY_ANSWERABLE,
                    reason=calc_result.partial_reason or (
                        "Calculation required but disclosed figures could not be confidently extracted."
                    ),
                    confidence=0.35,
                    chunks_retrieved=len(hits),
                    relevant_chunk_count=answerability_result.relevant_chunk_count,
                    missing_aspects=[
                        calc_result.partial_reason or "Required numeric inputs not found in retrieved sections.",
                    ],
                    signals={**answerability_result.signals, "calculation_extraction_failed": True},
                ))
                gen_question = f"{question}\n\n{PARTIAL_GENERATION_HINT}"
                t1 = time.perf_counter()
                try:
                    gen_result = generate_answer(gen_question, llm_hits, **gen_kwargs)
                except Exception as exc:
                    raise RAGPipelineError(
                        f"Generation failed: {exc}",
                        error_code="generation_failed",
                    ) from exc
                generation_ms = (time.perf_counter() - t1) * 1000.0
                answer = gen_result.answer
                cited_indices = gen_result.cited_indices
                prompt_tokens = gen_result.prompt_tokens
                completion_tokens = gen_result.completion_tokens
                model = settings.generation_model
    elif not hits:
        answer = NOT_ANSWERABLE_REFUSAL
        model = "none/refusal"
    else:
        gen_question = question
        if answerability_result.status == AnswerabilityStatus.PARTIALLY_ANSWERABLE:
            gen_question = f"{question}\n\n{PARTIAL_GENERATION_HINT}"

        t1 = time.perf_counter()
        try:
            gen_result = generate_answer(
                gen_question,
                llm_hits,
                **gen_kwargs,
            )
        except Exception as exc:
            raise RAGPipelineError(
                f"Generation failed: {exc}",
                error_code="generation_failed",
            ) from exc
        generation_ms = (time.perf_counter() - t1) * 1000.0
        answer = gen_result.answer
        cited_indices = gen_result.cited_indices
        prompt_tokens = gen_result.prompt_tokens
        completion_tokens = gen_result.completion_tokens

    cost = _estimate_cost_usd(
        settings=settings,
        embed_tokens=embed_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        demo_mode=demo_mode,
    )

    response = build_rag_response(
        question=question,
        answer=answer,
        hits=hits,
        cited_indices=cited_indices,
        filing_id=fid,
        company=meta["company"],
        strategy=strategy,
        top_k=top_k,
        demo_mode=demo_mode,
        model=model,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=generation_ms,
        collection_name=meta["collection"],
        chunks_loaded=len(resources.chunks),
        sem_ranks=sem_ranks,
        bm25_ranks=bm25_ranks,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embedding_tokens=embed_tokens,
        estimated_cost_usd=cost,
        cache_hit=False,
        answerability=answerability_schema,
        calculation=calculation_detail,
    )

    _set_cache(cache_key, response, settings)
    return response


def _chunk_ids_to_citation_indices(
    chunk_ids: List[str],
    hits: List[Dict[str, Any]],
) -> List[int]:
    """Map source chunk IDs to 1-based citation indices in the hit list."""
    id_to_idx = {h.get("id"): i + 1 for i, h in enumerate(hits)}
    return [id_to_idx[cid] for cid in chunk_ids if cid in id_to_idx]


def _sections_label(hits: List[Dict[str, Any]]) -> str:
    items: List[str] = []
    for hit in hits[:3]:
        meta = hit.get("metadata") or {}
        item = meta.get("item") or meta.get("section_title") or "unknown section"
        label = str(item)
        if label not in items:
            items.append(label)
    return ", ".join(items) if items else "the retrieved sections"
