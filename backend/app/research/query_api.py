"""Operator-gated research preview, off by default; legacy Q&A unaffected."""

from __future__ import annotations

import hmac
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from backend.app.middleware.rate_limit import limiter
from .generation import GroqGenerator, verify_answer
from .ingestion_api import get_research_repository
from .local_embedding import LocalSentenceEmbedding
from .models import DiscoveryError, normalize_cik, validate_accession
from .planning import Intent, plan_query
from .retrieval import retrieve
from .storage import LocalResearchRepository

router = APIRouter(prefix="/api/research", tags=["research-preview"])


class ResearchQuestion(BaseModel):
    cik: str
    accession: str
    revision_id: uuid.UUID
    question: str = Field(min_length=2, max_length=500)

    @field_validator("cik")
    @classmethod
    def canonical_cik(cls, value: str) -> str:
        return normalize_cik(value)

    @field_validator("accession")
    @classmethod
    def canonical_accession(cls, value: str) -> str:
        return validate_accession(value)


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


def reserve_generation_budget(repo: LocalResearchRepository) -> None:
    raw = os.getenv("RESEARCH_DAILY_CALL_CAP", "0")
    cap = int(raw) if raw.isascii() and raw.isdigit() else 0
    repo.reserve_research_call(cap)


@router.post("/query")
@limiter.limit("3/minute")
def query_research(request: Request, body: ResearchQuestion,
                   admission: None = Depends(require_research_preview),
                   repo: LocalResearchRepository = Depends(get_research_repository),
                   model: LocalSentenceEmbedding = Depends(get_local_model)) -> dict:
    trace_id, started = str(uuid.uuid4()), time.monotonic()
    revision = str(body.revision_id)
    metrics: dict = {}
    try:
        intent = plan_query(body.question)
        if intent != Intent.NARRATIVE:
            repo.record_research_event(trace_id, "query_refused", "intent_unavailable", body.cik,
                                       body.accession, revision, {"elapsed_ms": (time.monotonic() - started) * 1000})
            return {"status": "refused", "reason": "intent_unavailable", "intent": intent,
                    "answer": None, "evidence": [], "trace_id": trace_id}
        hits = retrieve(repo, body.cik, body.accession, revision, body.question, model)
        metrics["retrieval_ms"] = (time.monotonic() - started) * 1000
        metrics["evidence_count"] = len(hits)
        if not hits:
            repo.record_research_event(trace_id, "query_refused", "insufficient_evidence", body.cik,
                                       body.accession, revision, metrics)
            return {"status": "refused", "reason": "insufficient_evidence", "answer": None,
                    "evidence": [], "trace_id": trace_id}
        generator = GroqGenerator(os.getenv("GROQ_API_KEY", ""), os.getenv("GROQ_MODEL", ""))
        reserve_generation_budget(repo)
        generation_started = time.monotonic()
        answer, usage = generator.answer(body.question, hits)
        metrics["generation_ms"] = (time.monotonic() - generation_started) * 1000
        tags = verify_answer(answer, hits)
        metrics.update({key: value for key, value in usage.items()
                        if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        and isinstance(value, int) and value >= 0})
        metrics["citation_count"] = len(tags)
        metrics["elapsed_ms"] = (time.monotonic() - started) * 1000
        repo.record_research_event(trace_id, "query_complete", "evidence_cited_preview", body.cik,
                                   body.accession, revision, metrics)
    except DiscoveryError as exc:
        metrics["elapsed_ms"] = (time.monotonic() - started) * 1000
        repo.record_research_event(trace_id, "query_failed", exc.code, body.cik, body.accession, revision, metrics)
        exc.trace_id = trace_id
        raise
    cited = [{"id": f"E{i}", "cik": normalize_cik(body.cik), "accession": body.accession,
              "revision_id": revision, "chunk_id": hit["chunk_id"], "source_url": hit["source_url"],
              "part": hit["part"], "item": hit["item"], "excerpt": hit["text"][:300]}
             for i, hit in enumerate(hits, 1) if f"[E{i}]" in tags]
    return {"status": "evidence_cited_preview", "intent": Intent.NARRATIVE, "answer": answer, "evidence": cited,
            "revision_id": revision, "model": generator.model, "usage": {k: v for k, v in metrics.items() if k.endswith("tokens")},
            "trace_id": trace_id}
