"""Pydantic request/response schemas for the SEC Insight AI API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from backend.app.rag.schemas import RAGResponse

# API alias — /api/ask returns the canonical RAGResponse shape
AskResponse = RAGResponse


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Natural language question")
    filing_id: Optional[str] = Field(
        default=None,
        description="Indexed filing identifier (defaults to apple_2025)",
    )
    demo_mode: bool = Field(
        default=False,
        description="If true, retrieval-only demo — no LLM generation (zero generation cost)",
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        q = v.strip()
        if not q:
            raise ValueError("question must not be empty")
        return q


class HealthResponse(BaseModel):
    status: str
    version: str
    index_ready: bool
    default_filing_id: str
    available_filings: List[str]
    filing_count: int = 0


class FilingInfo(BaseModel):
    filing_id: str
    company: str
    ticker: str = ""
    fiscal_year: str = ""
    source_file: str = ""
    label: str


class FilingsResponse(BaseModel):
    default_filing_id: str
    filings: List[FilingInfo]


class SampleQuestion(BaseModel):
    question: str
    category: Optional[str] = None


class SampleQuestionsResponse(BaseModel):
    filing_id: str
    questions: List[SampleQuestion]


class EvalSummaryResponse(BaseModel):
    retrieval_v1: List[Dict[str, Any]] = Field(default_factory=list)
    generation_v1_ragas: Dict[str, Any] = Field(default_factory=dict)
    retrieval_v2: List[Dict[str, Any]] = Field(default_factory=list)
    generation_v2: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


class EntitlementsResponse(BaseModel):
    authenticated: bool
    email: Optional[str] = None
    plan: str = "anonymous"
    llm_calls_used: int = 0
    llm_calls_limit: int = 0
    llm_calls_remaining: int = 0
    can_use_llm: bool = False
    demo_mode_only: bool = True
    daily_token_budget: int = 0
    daily_tokens_used: int = 0
