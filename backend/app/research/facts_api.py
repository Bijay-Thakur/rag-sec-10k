"""Explicit-operand financial preview. No LLM chooses or calculates numbers."""

from __future__ import annotations

import os
import time
import uuid
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from backend.app.middleware.rate_limit import limiter
from .fact_storage import FactSnapshots
from .facts import calculate, select_as_filed
from .ingestion_api import get_research_repository
from .models import DiscoveryError
from .models import normalize_cik, validate_accession
from .query_api import require_research_preview
from .storage import RawObjectStore

router = APIRouter(prefix="/api/research", tags=["research-financial-preview"])
REVENUE = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"}
SUPPORTED = REVENUE | {"NetIncomeLoss", "OperatingIncomeLoss", "Assets", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent"}


class FactOperand(BaseModel):
    concept: str
    start: date | None = None
    end: date


class FinancialRequest(BaseModel):
    cik: str
    accession: str
    revision_id: uuid.UUID
    unit: str = "USD"
    formula: Literal["reported", "yoy", "margin", "ratio", "cagr"]
    current: FactOperand
    prior: FactOperand | None = None
    years: int | None = None
    as_of: date | None = None


def require_financial_preview(request: Request) -> None:
    require_research_preview(request)
    if os.getenv("RESEARCH_FACTS_ENABLED", "").lower() != "true":
        raise DiscoveryError("facts_disabled", "Financial preview disabled", 403)


@lru_cache
def get_fact_snapshots() -> FactSnapshots:
    root = os.getenv("RESEARCH_RAW_DIR", "")
    if not root or not Path(root).is_absolute():
        raise DiscoveryError("research_storage_unconfigured", "Raw source storage unavailable", 503)
    return FactSnapshots(get_research_repository(), RawObjectStore(Path(root)))


def financial_result(body: FinancialRequest, snapshots: FactSnapshots) -> dict:
    if body.unit != "USD" or body.current.concept not in SUPPORTED or (body.prior and body.prior.concept not in SUPPORTED):
        raise DiscoveryError("unsupported_concept", "Concept or unit needs review", 422)
    facts, source = snapshots.read(body.cik, body.accession, str(body.revision_id))

    def select(operand: FactOperand):
        return select_as_filed(facts, cik=body.cik, accession=body.accession, taxonomy="us-gaap",
                               concept=operand.concept, unit=body.unit, start=operand.start,
                               end=operand.end, as_of=body.as_of)

    current = select(body.current)
    operands = [current]
    if body.formula == "reported":
        if body.prior or body.years:
            raise DiscoveryError("invalid_operands", "Reported lookup takes one operand", 422)
        value, unit, kind = str(current.value), current.unit, "reported"
    else:
        if body.prior is None:
            raise DiscoveryError("missing_operand", "Calculation needs an explicit prior operand", 422)
        prior = select(body.prior)
        if body.formula == "margin" and not (current.concept in {"NetIncomeLoss", "OperatingIncomeLoss"} and prior.concept in REVENUE):
            raise DiscoveryError("unsupported_concept", "Margin concept mapping needs review", 422)
        if body.formula == "ratio" and (current.concept, prior.concept) not in {("AssetsCurrent", "LiabilitiesCurrent"), ("Liabilities", "Assets")}:
            raise DiscoveryError("unsupported_concept", "Ratio concept mapping needs review", 422)
        result = calculate(body.formula, current, prior, years=body.years)
        operands.append(prior)
        value, unit, kind = str(result.value), result.unit, "calculated"
    return {"status": "financial_preview", "kind": kind, "value": value, "unit": unit,
            "formula": body.formula, "rounding": "half_even_2_decimal_places" if kind == "calculated" else "as_reported",
            "revision_id": str(body.revision_id), "source": source,
            "operands": [{"id": fact.identity, "value": str(fact.value), "unit": fact.unit,
                          "cik": fact.cik, "accession": fact.accession, "concept": fact.concept,
                          "start": fact.start, "end": fact.end, "filed": fact.filed,
                          "form": fact.form, "fiscal_year": fact.fiscal_year, "fiscal_period": fact.fiscal_period,
                          "frame": fact.frame, "source_pointer": fact.source_pointer} for fact in operands]}


@router.post("/financial")
@limiter.limit("10/minute")
def financial_preview(request: Request, body: FinancialRequest,
                      admission: None = Depends(require_financial_preview),
                      snapshots: FactSnapshots = Depends(get_fact_snapshots)) -> dict:
    cik, accession = normalize_cik(body.cik), validate_accession(body.accession)
    trace_id, started = str(uuid.uuid4()), time.monotonic()
    try:
        result = financial_result(body, snapshots)
    except DiscoveryError as exc:
        snapshots.repo.record_research_event(trace_id, "financial_failed", exc.code, cik, accession,
                                              str(body.revision_id), {"elapsed_ms": (time.monotonic() - started) * 1000})
        exc.trace_id = trace_id
        raise
    snapshots.repo.record_research_event(trace_id, "financial_complete", "financial_preview", cik, accession,
                                          str(body.revision_id), {"elapsed_ms": (time.monotonic() - started) * 1000,
                                                                 "operand_count": len(result["operands"])})
    return {**result, "trace_id": trace_id}
