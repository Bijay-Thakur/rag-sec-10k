"""Protected local ingestion admission and read-only status/diagnostic routes."""

from __future__ import annotations

import hmac
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from backend.app.middleware.rate_limit import limiter

from .api import get_sec_client
from .models import DiscoveryError, normalize_cik, validate_accession
from .sec_client import SecClient
from .storage import LocalResearchRepository
from .worker import PIPELINE_VERSION
from .worker import LocalHashEmbedding

router = APIRouter(prefix="/api/research", tags=["research-ingestion"])


@lru_cache
def get_research_repository() -> LocalResearchRepository:
    db = os.getenv("RESEARCH_DB_PATH", "")
    if not db or not Path(db).is_absolute():
        raise DiscoveryError("research_storage_unconfigured", "Durable research storage unavailable", 503)
    return LocalResearchRepository(Path(db))


class IngestionRequest(BaseModel):
    cik: str
    accession: str


def require_admission(request: Request) -> None:
    token = os.getenv("RESEARCH_INGESTION_TOKEN", "")
    if os.getenv("RESEARCH_INGESTION_ENABLED", "").lower() != "true" or len(token) < 24:
        raise DiscoveryError("ingestion_disabled", "Ingestion is disabled", 403)
    header = request.headers.get("authorization", "")
    if not hmac.compare_digest(header, f"Bearer {token}"):
        raise DiscoveryError("ingestion_forbidden", "Ingestion authorization required", 403)


@router.post("/ingestions", status_code=202)
@limiter.limit("5/minute")
def enqueue_ingestion(request: Request, body: IngestionRequest,
                      admission: None = Depends(require_admission),
                      repository: LocalResearchRepository = Depends(get_research_repository),
                      sec: SecClient = Depends(get_sec_client)) -> dict:
    cik, accession = normalize_cik(body.cik), validate_accession(body.accession)
    # Resolve on SEC, never accept a client-supplied URL or document name.
    listing = sec.list_filings(cik, include_amendments=True, limit=100)
    match = next((filing for filing in listing.filings if filing.accession == accession), None)
    if not match or not match.primary_document_url:
        raise DiscoveryError("filing_not_discovered", "Accession unavailable in recent SEC metadata", 404)
    job = repository.enqueue(match, PIPELINE_VERSION)
    ready = repository.filing(cik, accession)
    return {"job_id": job["job_id"], "cik": cik, "accession": accession,
            "state": job["state"], "existing_ready_revision_id": ready["ready_revision"] if ready else None}


@router.get("/ingestions/{job_id}")
@limiter.limit("30/minute")
def ingestion_status(request: Request, job_id: uuid.UUID,
                     repository: LocalResearchRepository = Depends(get_research_repository)) -> dict:
    job = repository.job(str(job_id))
    if not job:
        raise DiscoveryError("job_not_found", "Unknown ingestion job", 404)
    return {key: value for key, value in job.items() if key != "progress_json"} | {
        "progress": json.loads(job["progress_json"])}


@router.get("/filings/{cik}/{accession}/status")
@limiter.limit("30/minute")
def filing_status(request: Request, cik: str, accession: str,
                  repository: LocalResearchRepository = Depends(get_research_repository)) -> dict:
    row = repository.filing(cik, accession)
    if not row:
        raise DiscoveryError("filing_not_indexed", "No local ingestion record", 404)
    research = repository.research_revision(cik, accession)
    return {"cik": row["cik"], "accession": row["accession"], "form": row["form"],
            "filed_at": row["filed_at"], "report_period": row["report_period"],
            "source_url": row["source_url"], "ready_revision_id": row["ready_revision"],
            "research_revision_id": research["revision_id"] if research else None,
            "index_scope": "operator research preview only" if research else "local diagnostic only; dynamic Q&A unavailable"}


@router.get("/filings/{cik}/{accession}/sections")
@limiter.limit("30/minute")
def filing_sections(request: Request, cik: str, accession: str,
                    repository: LocalResearchRepository = Depends(get_research_repository)) -> dict:
    return {"sections": repository.sections(cik, accession), "scope": "local diagnostic"}


@router.get("/filings/{cik}/{accession}/diagnostic")
@limiter.limit("20/minute")
def filing_diagnostic(request: Request, cik: str, accession: str, query: str,
                      mode: str = "lexical",
                      repository: LocalResearchRepository = Depends(get_research_repository)) -> dict:
    if not 2 <= len(query) <= 200:
        raise DiscoveryError("invalid_query", "Query must be 2–200 characters", 422)
    if mode == "lexical":
        hits = repository.search(cik, accession, query)
    elif mode == "local-vector":
        row = repository.filing(cik, accession)
        if not row or not row["ready_revision"]:
            hits = []
        else:
            with repository.connect() as db:
                revision = db.execute("SELECT index_kind FROM revisions WHERE revision_id=?",
                                      (row["ready_revision"],)).fetchone()
            if not revision or revision["index_kind"] != "lexical_diagnostic":
                raise DiscoveryError("diagnostic_unavailable", "Local hash index unavailable", 422)
            hits = repository.vector_search(cik, accession, LocalHashEmbedding().embed([query])[0])
    else:
        raise DiscoveryError("invalid_filter", "Unknown diagnostic mode", 422)
    return {"hits": hits, "scope": "accession-filtered diagnostic; not research Q&A", "mode": mode}
