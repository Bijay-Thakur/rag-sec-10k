"""Server-only SEC discovery adapter with injected transport and access policy."""

from __future__ import annotations

import json
import hashlib
import random
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlsplit

import httpx

from .models import (
    DiscoveryError, FilingListResponse, FilingRef, Issuer, IssuerSearchResponse,
    normalize_cik, primary_document_url,
)

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_FILING_BYTES = 20 * 1024 * 1024


class SecClient(Protocol):
    def resolve_issuer(self, query: str) -> IssuerSearchResponse: ...
    def list_filings(self, cik: str, *, form: str | None = None,
                     include_amendments: bool = False, include_history: bool = False,
                     limit: int = 25, cursor: str | None = None,
                     filed_from: date | None = None, filed_to: date | None = None) -> FilingListResponse: ...
    def fetch_primary(self, filing: FilingRef) -> tuple[bytes, dict[str, str]]: ...


class LocalRateLimiter:
    """At most 2 requests/s in one process. Live multi-instance use is disabled by default."""

    def __init__(self, interval: float = 0.5, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            wait = max(0.0, self._next - self.clock())
            if wait:
                self.sleep(wait)
            self._next = self.clock() + self.interval


class SharedSQLiteRateLimiter:
    """Local-volume aggregate limit across the API and worker processes."""

    def __init__(self, path: Path, *, interval: float = 0.5,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.path, self.interval, self.clock, self.sleep = path, interval, clock, sleep
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS sec_budget (id INTEGER PRIMARY KEY CHECK(id=1), next_at REAL NOT NULL)")

    def acquire(self) -> None:
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT next_at FROM sec_budget WHERE id=1").fetchone()
            scheduled = max(self.clock(), row[0] if row else 0.0)
            db.execute("INSERT INTO sec_budget(id,next_at) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET next_at=excluded.next_at",
                       (scheduled + self.interval,))
        self.sleep(max(0.0, scheduled - self.clock()))


class SecHttpAdapter:
    def __init__(self, *, user_agent: str, transport: httpx.BaseTransport | None = None,
                 throttle: LocalRateLimiter | SharedSQLiteRateLimiter | None = None,
                 clock: Callable[[], datetime] | None = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 ttl_seconds: int = 3600, max_attempts: int = 3):
        if (not re.search(r"\S+@\S+\.\S+", user_agent)
                or re.search(r"example\.(com|org)|your[-_ ]?(email|company)|placeholder|@[^\s]+\.(test|invalid|local)\b", user_agent, re.I)) and transport is None:
            raise DiscoveryError("sec_contact_required", "Configure a real contact-bearing SEC_USER_AGENT", 503)
        self.user_agent = user_agent
        self.http = httpx.Client(transport=transport, timeout=httpx.Timeout(10.0), follow_redirects=False,
                                 headers={"User-Agent": user_agent, "Accept": "application/json"})
        self.throttle = throttle or LocalRateLimiter()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic, self.sleep = monotonic, sleep
        self.ttl_seconds, self.max_attempts = ttl_seconds, max_attempts
        self._cache: dict[str, tuple[float, object, str | None, datetime]] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def _allowlisted(url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.username or parts.password or parts.port or parts.query or parts.fragment:
            return False
        if parts.hostname == "www.sec.gov":
            return parts.path == "/files/company_tickers_exchange.json"
        if parts.hostname == "data.sec.gov":
            return bool(re.fullmatch(r"/submissions/CIK[0-9]{10}(?:-submissions-[0-9]{3})?\.json", parts.path))
        return False

    def _json(self, url: str) -> tuple[object, datetime]:
        if not self._allowlisted(url):
            raise DiscoveryError("unsafe_sec_url", "SEC request path is not allowlisted", 422)
        with self._cache_lock:
            cached = self._cache.get(url)
            if cached and cached[0] > self.monotonic():
                return cached[1], cached[3]
        headers = {"If-None-Match": cached[2]} if cached and cached[2] else {}
        for attempt in range(self.max_attempts):
            self.throttle.acquire()
            try:
                with self.http.stream("GET", url, headers=headers) as response:
                    if response.status_code == 304 and cached:
                        with self._cache_lock:
                            self._cache[url] = (self.monotonic() + self.ttl_seconds, cached[1], cached[2], cached[3])
                        return cached[1], cached[3]
                    if 300 <= response.status_code < 400:
                        raise DiscoveryError("sec_redirect", "SEC redirected a restricted request", 502)
                    if response.status_code in (403, 404):
                        raise DiscoveryError(f"sec_{response.status_code}", "SEC source unavailable", response.status_code)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise httpx.HTTPStatusError("SEC temporarily unavailable", request=response.request, response=response)
                    if 400 <= response.status_code < 500:
                        raise DiscoveryError(f"sec_{response.status_code}", "SEC rejected the request", 502)
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_JSON_BYTES:
                            raise DiscoveryError("sec_response_too_large", "SEC response exceeded size limit", 502)
                    try:
                        payload = json.loads(data)
                    except (ValueError, UnicodeError) as exc:
                        raise DiscoveryError("sec_invalid_json", "SEC returned invalid JSON", 502) from exc
                    observed = self.clock()
                    with self._cache_lock:
                        self._cache[url] = (self.monotonic() + self.ttl_seconds, payload,
                                            response.headers.get("etag"), observed)
                    return payload, observed
            except DiscoveryError:
                raise
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt + 1 == self.max_attempts:
                    raise DiscoveryError("sec_unavailable", "SEC request failed after bounded retries", 503) from exc
                delay = min(4.0, 0.5 * 2 ** attempt + random.uniform(0, 0.1))
                if isinstance(exc, httpx.HTTPStatusError):
                    raw = exc.response.headers.get("retry-after", "")
                    if raw.isdigit():
                        delay = min(10.0, max(delay, float(raw)))
                self.sleep(delay)
        raise AssertionError("unreachable")

    def resolve_issuer(self, query: str) -> IssuerSearchResponse:
        query = query.strip()
        if len(query) < 1 or len(query) > 100 or any(ord(c) < 32 for c in query):
            raise DiscoveryError("invalid_query", "Search needs 1–100 printable characters", 422)
        payload, observed = self._json(TICKERS_URL)
        if not isinstance(payload, dict) or not isinstance(payload.get("fields"), list) or not isinstance(payload.get("data"), list):
            raise DiscoveryError("sec_schema", "Unexpected SEC ticker mapping", 502)
        fields = payload["fields"]
        if not {"cik", "name", "ticker"}.issubset(fields):
            raise DiscoveryError("sec_schema", "Incomplete SEC ticker mapping", 502)
        ix = {name: fields.index(name) for name in fields}
        matched: dict[str, Issuer] = {}
        exact_ids: set[str] = set()
        is_cik = query.isascii() and query.isdigit()
        if is_cik:
            cik_query = normalize_cik(query)
        needle = query.casefold()
        for row in payload["data"]:
            if not isinstance(row, list) or len(row) != len(fields):
                raise DiscoveryError("sec_schema", "Malformed SEC ticker row", 502)
            cik = normalize_cik(str(row[ix["cik"]]))
            name, ticker = str(row[ix["name"]]), str(row[ix["ticker"]])
            exact = (is_cik and cik == cik_query) or needle == ticker.casefold() or needle == name.casefold()
            if not (exact or (len(needle) >= 3 and name.casefold().startswith(needle))):
                continue
            if exact:
                exact_ids.add(cik)
            if cik not in matched:
                matched[cik] = Issuer(cik=cik, canonical_name=name, lookup_source=TICKERS_URL,
                                      observed_at=observed)
            if ticker and ticker not in matched[cik].tickers:
                matched[cik].tickers.append(ticker)
            if "exchange" in ix and row[ix["exchange"]] and row[ix["exchange"]] not in matched[cik].exchanges:
                matched[cik].exchanges.append(str(row[ix["exchange"]]))
        if exact_ids:
            matched = {key: value for key, value in matched.items() if key in exact_ids}
        candidates = sorted(matched.values(), key=lambda x: (x.canonical_name.casefold(), x.cik))[:20]
        if is_cik and not candidates:
            source = f"https://data.sec.gov/submissions/CIK{cik_query}.json"
            try:
                submission, observed = self._json(source)
            except DiscoveryError as exc:
                if exc.code == "sec_404":
                    return IssuerSearchResponse(candidates=[], ambiguous=False,
                                                fetched_at=observed, source_url=TICKERS_URL)
                raise
            if not isinstance(submission, dict) or not isinstance(submission.get("name"), str):
                raise DiscoveryError("sec_schema", "Missing issuer identity in submissions", 502)
            candidates = [Issuer(cik=cik_query, canonical_name=submission["name"],
                                 tickers=submission.get("tickers") or [],
                                 exchanges=submission.get("exchanges") or [],
                                 lookup_source=source, observed_at=observed)]
        return IssuerSearchResponse(candidates=candidates, ambiguous=len(candidates) > 1,
                                    fetched_at=observed, source_url=(candidates[0].lookup_source
                                                if is_cik and candidates else TICKERS_URL))

    @staticmethod
    def _rows(cik: str, block: object, source: str, observed: datetime) -> list[FilingRef]:
        if not isinstance(block, dict) or not isinstance(block.get("accessionNumber"), list):
            raise DiscoveryError("sec_schema", "Malformed submissions rows", 502)
        keys = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument")
        n = len(block["accessionNumber"])
        for key in keys:
            if key not in block or not isinstance(block[key], list) or len(block[key]) != n:
                raise DiscoveryError("sec_schema", "Mismatched submissions columns", 502)
        rows = []
        for i in range(n):
            accession, form = block["accessionNumber"][i], block["form"][i]
            if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                continue
            url = primary_document_url(cik, accession, block["primaryDocument"][i])
            try:
                rows.append(FilingRef(cik=cik, accession=accession, form=form,
                                      filed_at=date.fromisoformat(block["filingDate"][i]),
                                      report_period=(date.fromisoformat(block["reportDate"][i])
                                                     if block["reportDate"][i] else None),
                                      primary_document=block["primaryDocument"][i] or None,
                                      primary_document_url=url, source_submissions_url=source,
                                      metadata_observed_at=observed))
            except (ValueError, TypeError) as exc:
                raise DiscoveryError("sec_schema", "Invalid submissions date or field", 502) from exc
        return rows

    def list_filings(self, cik: str, *, form: str | None = None,
                     include_amendments: bool = False, include_history: bool = False,
                     limit: int = 25, cursor: str | None = None,
                     filed_from: date | None = None, filed_to: date | None = None) -> FilingListResponse:
        cik = normalize_cik(cik)
        if form not in (None, "10-K", "10-Q") or not 1 <= limit <= 100:
            raise DiscoveryError("invalid_filter", "Invalid form or page size", 422)
        if cursor is not None and (not cursor.isdigit() or len(cursor) > 8):
            raise DiscoveryError("invalid_cursor", "Invalid pagination cursor", 422)
        offset = int(cursor or 0)
        if offset > 100000:
            raise DiscoveryError("invalid_cursor", "Pagination cursor too large", 422)
        if filed_from and filed_to and filed_from > filed_to:
            raise DiscoveryError("invalid_filter", "Invalid date range", 422)
        source = f"https://data.sec.gov/submissions/CIK{cik}.json"
        payload, observed = self._json(source)
        if not isinstance(payload, dict) or not isinstance(payload.get("filings"), dict):
            raise DiscoveryError("sec_schema", "Malformed SEC submissions", 502)
        filings = payload["filings"]
        rows = self._rows(cik, filings.get("recent"), source, observed)
        if include_history:
            for entry in filings.get("files", []):
                name = entry.get("name") if isinstance(entry, dict) else None
                if not isinstance(name, str) or not re.fullmatch(r"CIK[0-9]{10}-submissions-[0-9]{3}\.json", name) or not name.startswith(f"CIK{cik}-"):
                    raise DiscoveryError("sec_schema", "Unsafe SEC history filename", 502)
                history_url = f"https://data.sec.gov/submissions/{name}"
                history, seen = self._json(history_url)
                rows.extend(self._rows(cik, history, history_url, seen))
        distinct = {(r.cik, r.accession): r for r in reversed(rows)}
        eligible = [r for r in distinct.values() if (not form or r.form == form or
                    (include_amendments and r.form == f"{form}/A"))
                    and (include_amendments or not r.form.endswith("/A"))
                    and (filed_from is None or r.filed_at >= filed_from)
                    and (filed_to is None or r.filed_at <= filed_to)]
        eligible.sort(key=lambda r: (r.filed_at, r.accession), reverse=True)
        page = eligible[offset:offset + limit]
        next_cursor = str(offset + limit) if offset + limit < len(eligible) else None
        return FilingListResponse(cik=cik, filings=page, next_cursor=next_cursor,
                                  fetched_at=observed, source_url=source)

    def fetch_primary(self, filing: FilingRef) -> tuple[bytes, dict[str, str]]:
        """Fetch only the canonical URL reconstructed from a trusted filing ref."""
        expected = primary_document_url(filing.cik, filing.accession, filing.primary_document)
        if not expected or filing.primary_document_url != expected:
            raise DiscoveryError("unsafe_sec_url", "Filing URL disagrees with its identity", 422)
        for attempt in range(self.max_attempts):
            self.throttle.acquire()
            try:
                with self.http.stream("GET", expected, headers={"Accept": "text/html, text/plain"}) as response:
                    if 300 <= response.status_code < 400:
                        raise DiscoveryError("sec_redirect", "SEC redirected the filing request", 502)
                    if response.status_code in (403, 404):
                        raise DiscoveryError(f"sec_{response.status_code}", "SEC filing unavailable", response.status_code)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise httpx.HTTPStatusError("SEC temporarily unavailable", request=response.request, response=response)
                    if response.status_code >= 400:
                        raise DiscoveryError(f"sec_{response.status_code}", "SEC rejected filing request", 502)
                    mime = response.headers.get("content-type", "").split(";")[0].lower()
                    if mime not in ("text/html", "application/xhtml+xml", "text/plain"):
                        raise DiscoveryError("unsupported_document", "SEC primary document is not HTML/text", 422)
                    data = bytearray()
                    for chunk in response.iter_bytes():  # httpx decodes content encoding; cap decoded bytes
                        data.extend(chunk)
                        if len(data) > MAX_FILING_BYTES:
                            raise DiscoveryError("filing_too_large", "SEC filing exceeds byte limit", 413)
                    if not data:
                        raise DiscoveryError("empty_filing", "SEC filing was empty", 502)
                    return bytes(data), {"content_type": mime, "etag": response.headers.get("etag", ""),
                                         "last_modified": response.headers.get("last-modified", ""),
                                         "sha256": hashlib.sha256(data).hexdigest(),
                                         "fetched_at": self.clock().isoformat(), "source_url": expected}
            except DiscoveryError:
                raise
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt + 1 == self.max_attempts:
                    raise DiscoveryError("sec_unavailable", "SEC filing fetch exhausted retries", 503) from exc
                self.sleep(min(10.0, 0.5 * 2 ** attempt + random.uniform(0, 0.1)))
        raise AssertionError("unreachable")
