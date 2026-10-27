"""Typed public SEC discovery contracts and strict identifier validation."""

from __future__ import annotations

import re
from datetime import date, datetime
from urllib.parse import quote

from pydantic import BaseModel, Field


class DiscoveryError(Exception):
    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code, self.status = code, status


def normalize_cik(value: str) -> str:
    if not re.fullmatch(r"[0-9]{1,10}", value):
        raise DiscoveryError("invalid_cik", "CIK must contain 1–10 digits", 422)
    return value.zfill(10)


def validate_accession(value: str) -> str:
    if not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", value):
        raise DiscoveryError("invalid_accession", "Invalid SEC accession", 422)
    return value


def primary_document_url(cik: str, accession: str, document: str | None) -> str | None:
    cik = normalize_cik(cik)
    accession = validate_accession(accession)
    if not document:
        return None
    # One SEC-supplied basename only. No slashes, dot segments, URL escaping or query.
    if (len(document) > 180 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", document)
            or ".." in document or document.endswith(".")):
        raise DiscoveryError("invalid_document", "Unsafe SEC primary document name", 502)
    return ("https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession.replace('-', '')}/{quote(document, safe='._-')}")


class Issuer(BaseModel):
    cik: str
    canonical_name: str
    tickers: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)
    lookup_source: str
    observed_at: datetime


class FilingRef(BaseModel):
    cik: str
    accession: str
    form: str
    filed_at: date
    report_period: date | None
    primary_document: str | None
    primary_document_url: str | None
    source_submissions_url: str
    metadata_observed_at: datetime


class IssuerSearchResponse(BaseModel):
    candidates: list[Issuer]
    ambiguous: bool
    fetched_at: datetime
    source_url: str


class FilingListResponse(BaseModel):
    cik: str
    filings: list[FilingRef]
    next_cursor: str | None
    fetched_at: datetime
    source_url: str
