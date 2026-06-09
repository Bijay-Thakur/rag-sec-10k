"""
Canonical RAG response schema — frontend-ready, no extra parsing required.

Every pipeline answer (API, CLI, UI) should serialize to RAGResponse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token counts when available from the LLM / embedding calls."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    embedding_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class SourceMetadata(BaseModel):
    """Structured metadata attached to a retrieved chunk."""

    source_file: str = ""
    part: str = ""
    item: str = ""
    section_title: str = ""
    chunk_strategy: str = ""
    chunk_index: Optional[int] = None
    char_count: Optional[int] = None
    token_count: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Evidence reference cited in the generated answer."""

    index: int = Field(description="1-based citation number as it appears in the answer, e.g. [1]")
    chunk_id: str
    filing_id: str
    company: str
    section_title: Optional[str] = None
    item: Optional[str] = None
    source_text_excerpt: str = Field(description="Passage excerpt supporting the citation")
    score: Optional[float] = Field(default=None, description="Best available retrieval/rerank score")


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with rank/score breakdown for UI display."""

    chunk_id: str
    rank: int = Field(description="Final rank in the returned result list (1-based)")
    text_excerpt: str = Field(description="Truncated text for display")
    text: str = Field(description="Full chunk text")
    source_metadata: SourceMetadata
    bm25_rank: Optional[int] = Field(default=None, description="Rank in BM25-only results")
    vector_rank: Optional[int] = Field(default=None, description="Rank in dense vector results")
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    vector_distance: Optional[float] = None


class RetrievalTrace(BaseModel):
    """Audit trail for how retrieval was performed."""

    strategy: str
    top_k: int
    filing_id: str
    question: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    demo_mode: bool
    stages: Dict[str, Any] = Field(default_factory=dict)


class CalculationInput(BaseModel):
    label: str
    value: float
    unit: str = "USD millions"
    year: Optional[int] = None
    source_chunk_id: str = ""
    source_excerpt: str = ""


class CalculationDetail(BaseModel):
    """Deterministic calculation performed in Python (never LLM-estimated)."""

    calculation_type: str
    inputs: List[CalculationInput]
    formula: str
    result: str
    result_value: Optional[float] = None
    source_chunk_ids: List[str] = Field(default_factory=list)
    success: bool = True
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metric: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Answerability(BaseModel):
    """Pre-generation assessment of whether the question can be answered from context."""

    status: str = Field(
        description="One of: answerable, partially_answerable, not_answerable, calculation_required",
    )
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    chunks_retrieved: int = 0
    answerable: bool = Field(
        description="True only when status is 'answerable' (backward-compatible flag)",
    )
    relevant_chunk_count: int = 0
    missing_aspects: List[str] = Field(default_factory=list)
    requires_calculation: bool = False
    signals: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """
    Standardized output for every RAG pipeline invocation.

    Designed for direct frontend rendering — citations, evidence panels,
    and system metadata without additional parsing.
    """

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace
    answerability: Answerability
    model: str
    latency_ms: float
    token_usage: Optional[TokenUsage] = None
    estimated_cost_usd: float = 0.0
    cache_hit: bool = False
    calculation: Optional[CalculationDetail] = None

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)
