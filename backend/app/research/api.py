"""Additive, read-only SEC discovery routes."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from backend.app.middleware.rate_limit import limiter

from .models import FilingListResponse, IssuerSearchResponse
from .sec_client import SecClient, SecHttpAdapter, SharedSQLiteRateLimiter
from .models import DiscoveryError

router = APIRouter(prefix="/api/research", tags=["research-discovery"])


@lru_cache
def get_sec_client() -> SecClient:
    # A process-local limiter is not global across Cloud Run instances. Live
    # traffic requires an explicitly configured single-instance deployment.
    shared_path = os.getenv("SEC_RATE_DB_PATH", "")
    if os.getenv("SEC_SINGLE_INSTANCE_MODE", "").lower() != "true":
        raise DiscoveryError("sec_policy_unconfigured",
                             "SEC discovery requires an operator-verified single-instance limit", 503)
    if shared_path and not Path(shared_path).is_absolute():
        raise DiscoveryError("sec_policy_unconfigured", "SEC rate store path must be absolute", 503)
    throttle = SharedSQLiteRateLimiter(Path(shared_path)) if shared_path else None
    return SecHttpAdapter(user_agent=os.getenv("SEC_USER_AGENT", ""), throttle=throttle)


@router.get("/issuers", response_model=IssuerSearchResponse)
@limiter.limit("20/minute")
def search_issuers(request: Request, query: str = Query(min_length=1, max_length=100),
                   sec: SecClient = Depends(get_sec_client)) -> IssuerSearchResponse:
    return sec.resolve_issuer(query)


@router.get("/issuers/{cik}/filings", response_model=FilingListResponse)
@limiter.limit("20/minute")
def list_research_filings(request: Request, cik: str, form: str | None = None,
                          include_amendments: bool = False, include_history: bool = False,
                          limit: int = Query(25, ge=1, le=100), cursor: str | None = None,
                          filed_from: date | None = None, filed_to: date | None = None,
                          sec: SecClient = Depends(get_sec_client)) -> FilingListResponse:
    return sec.list_filings(cik, form=form, include_amendments=include_amendments,
                            include_history=include_history, limit=limit, cursor=cursor,
                            filed_from=filed_from, filed_to=filed_to)
