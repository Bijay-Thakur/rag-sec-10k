"""Operator-gated research preview, off by default; legacy Q&A unaffected."""

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from backend.app.middleware.rate_limit import limiter
from .generation import GroqGenerator, verify_answer
from .ingestion_api import get_research_repository
from .local_embedding import LocalSentenceEmbedding
from .models import DiscoveryError, normalize_cik
from .retrieval import retrieve
from .storage import LocalResearchRepository

router = APIRouter(prefix="/api/research", tags=["research-preview"])
_budget_lock = threading.Lock()
_budget_day = 0
_budget_calls = 0


class ResearchQuestion(BaseModel):
    cik: str
    accession: str
    revision_id: uuid.UUID
    question: str = Field(min_length=2, max_length=500)


@lru_cache
def get_local_model() -> LocalSentenceEmbedding:
    path = os.getenv("RESEARCH_LOCAL_MODEL_DIR", "")
    if not path or not Path(path).is_absolute():
        raise DiscoveryError("embedding_unconfigured", "Local model unavailable", 503)
    return LocalSentenceEmbedding(Path(path))


def require_research_preview(request: Request) -> None:
    secret = os.getenv("RESEARCH_QUERY_TOKEN", "")
    if (os.getenv("RESEARCH_QUERY_ENABLED", "").lower() != "true" or len(secret) < 24
            or os.getenv("RESEARCH_SINGLE_INSTANCE_MODE") != "true"):
        raise DiscoveryError("research_disabled", "Research preview disabled", 403)
    if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {secret}"):
        raise DiscoveryError("research_forbidden", "Research preview authorization required", 403)


def reserve_generation_budget() -> None:
    global _budget_day, _budget_calls
    raw = os.getenv("RESEARCH_DAILY_CALL_CAP", "0")
    cap = int(raw) if raw.isascii() and raw.isdigit() else 0
    if not 1 <= cap <= 100:
        raise DiscoveryError("research_budget_unconfigured", "Research call budget unavailable", 503)
    with _budget_lock:
        day = int(time.time() // 86400)
        if day != _budget_day:
            _budget_day, _budget_calls = day, 0
        if _budget_calls >= cap:
            raise DiscoveryError("research_budget_exhausted", "Research call budget exhausted", 429)
        _budget_calls += 1


@router.post("/query")
@limiter.limit("3/minute")
def query_research(request: Request, body: ResearchQuestion,
                   admission: None = Depends(require_research_preview),
                   repo: LocalResearchRepository = Depends(get_research_repository),
                   model: LocalSentenceEmbedding = Depends(get_local_model)) -> dict:
    # Numerical reasoning waits for typed XBRL operands and Phase 4 validation.
    if any(term in body.question.casefold() for term in ("how much", "percent", "ratio", "growth", "calculate", "compare")):
        return {"status": "refused", "reason": "numeric_or_comparison_not_enabled", "answer": None,
                "evidence": [], "trace_id": str(uuid.uuid4())}
    revision = str(body.revision_id)
    hits = retrieve(repo, body.cik, body.accession, revision, body.question, model)
    if not hits:
        return {"status": "refused", "reason": "insufficient_evidence", "answer": None,
                "evidence": [], "trace_id": str(uuid.uuid4())}
    reserve_generation_budget()
    generator = GroqGenerator(os.getenv("GROQ_API_KEY", ""), os.getenv("GROQ_MODEL", ""))
    answer, usage = generator.answer(body.question, hits)
    tags = verify_answer(answer, hits)
    cited = [{"id": f"E{i}", "cik": normalize_cik(body.cik), "accession": body.accession,
              "revision_id": revision, "chunk_id": hit["chunk_id"], "source_url": hit["source_url"],
              "part": hit["part"], "item": hit["item"], "excerpt": hit["text"][:300]}
             for i, hit in enumerate(hits, 1) if f"[E{i}]" in tags]
    return {"status": "evidence_cited_preview", "answer": answer, "evidence": cited,
            "revision_id": revision, "model": generator.model, "usage": usage, "trace_id": str(uuid.uuid4())}
