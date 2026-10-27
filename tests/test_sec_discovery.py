"""No external HTTP, credentials, embedding calls, or filing downloads."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pytest

from backend.app.research.models import DiscoveryError, primary_document_url
from backend.app.research.sec_client import LocalRateLimiter, SecHttpAdapter, SharedSQLiteRateLimiter

FIXTURES = Path(__file__).parent / "fixtures"
OBSERVED = datetime(2026, 10, 27, tzinfo=timezone.utc)


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def adapter(handler=None, **kwargs):
    mapping = fixture("sec_issuer_mapping.json")
    recent = fixture("sec_submissions_recent.json")
    history = fixture("sec_submissions_history.json")

    def default(request):
        path = request.url.path
        if path.endswith("company_tickers_exchange.json"):
            return httpx.Response(200, json=mapping, headers={"etag": '"mapping-1"'})
        if path.endswith("submissions-001.json"):
            return httpx.Response(200, json=history)
        if path.endswith("CIK0001234567.json"):
            return httpx.Response(200, json=recent)
        return httpx.Response(404)

    return SecHttpAdapter(user_agent="Fixture Test testing@invalid.test",
                          transport=httpx.MockTransport(handler or default),
                          clock=lambda: OBSERVED, sleep=lambda _: None, **kwargs)


def test_ticker_ambiguity_and_name_candidates():
    sec = adapter()
    result = sec.resolve_issuer("ORI")
    assert result.ambiguous and {x.cik for x in result.candidates} == {"0001234567", "0002222222"}
    assert sec.resolve_issuer("Orion Research Inc.").candidates[0].cik == "0001234567"
    assert len(sec.resolve_issuer("Orion").candidates) == 2
    assert sec.resolve_issuer("O").candidates == []  # valid one-letter lookup


def test_cik_lookup_uses_submissions_if_missing_from_ticker_mapping():
    def handler(request):
        if request.url.path.endswith("company_tickers_exchange.json"):
            return httpx.Response(200, json=fixture("sec_issuer_mapping.json"))
        return httpx.Response(200, json={"name": "Unlisted Issuer", "tickers": []})
    result = adapter(handler).resolve_issuer("9999999")
    assert result.candidates[0].cik == "0009999999"
    assert result.candidates[0].lookup_source.endswith("CIK0009999999.json")
    with pytest.raises(DiscoveryError, match="CIK"):
        adapter().list_filings("../1234567")


def test_forms_amendments_history_and_pagination():
    sec = adapter()
    standard = sec.list_filings("1234567", include_history=True, limit=2)
    assert [x.form for x in standard.filings] == ["10-Q", "10-K"]
    assert standard.next_cursor == "2"
    assert standard.filings[0].accession == "0001111111-26-000012"
    assert standard.filings[0].primary_document_url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000111111126000012/ori-10q.htm")
    older = sec.list_filings("1234567", include_history=True, cursor="2", limit=2)
    assert [x.form for x in older.filings] == ["10-Q"]
    all_forms = sec.list_filings("1234567", include_history=True, include_amendments=True,
                                 form="10-Q", limit=10)
    assert [x.form for x in all_forms.filings] == ["10-Q", "10-Q/A", "10-Q"]
    assert all_forms.filings[1].report_period == date(2024, 6, 30)
    assert all_forms.filings[1].filed_at == date(2024, 9, 2)
    assert len(sec.list_filings("1234567", filed_from=date(2026, 1, 1)).filings) == 1


@pytest.mark.parametrize("document", ["../evil.htm", "a/b.htm", "%2e%2e", "a?x", "a..htm", "a\\b"])
def test_rejects_unsafe_primary_document(document):
    with pytest.raises(DiscoveryError) as exc:
        primary_document_url("1234567", "0001111111-25-000010", document)
    assert exc.value.code == "invalid_document"


def test_rejects_bad_submission_columns_and_unsafe_history_name():
    def bad(request):
        payload = fixture("sec_submissions_recent.json")
        payload["filings"]["recent"]["form"].pop()
        return httpx.Response(200, json=payload)

    with pytest.raises(DiscoveryError, match="Mismatched"):
        adapter(bad).list_filings("1234567")
    def unsafe(request):
        payload = fixture("sec_submissions_recent.json")
        payload["filings"]["files"][0]["name"] = "../../elsewhere.json"
        return httpx.Response(200, json=payload)
    with pytest.raises(DiscoveryError, match="Unsafe"):
        adapter(unsafe).list_filings("1234567", include_history=True)


@pytest.mark.parametrize("status,code", [(403, "sec_403"), (404, "sec_404"), (429, "sec_unavailable"), (500, "sec_unavailable"), (302, "sec_redirect")])
def test_transport_errors(status, code):
    sec = adapter(lambda request: httpx.Response(status, headers={"location": "https://evil.example"}),
                  max_attempts=2)
    with pytest.raises(DiscoveryError) as exc:
        sec.list_filings("1234567")
    assert exc.value.code == code


def test_timeout_and_decompression_size_limit():
    sec = adapter(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("offline timeout")), max_attempts=2)
    with pytest.raises(DiscoveryError, match="bounded retries"):
        sec.list_filings("1234567")
    from backend.app.research import sec_client
    original = sec_client.MAX_JSON_BYTES
    sec_client.MAX_JSON_BYTES = 20
    try:
        with pytest.raises(DiscoveryError) as exc:
            adapter().list_filings("1234567")
        assert exc.value.code == "sec_response_too_large"
    finally:
        sec_client.MAX_JSON_BYTES = original


def test_cache_ttl_conditional_get_and_fake_limiter():
    now = [0.0]
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds
    limiter = LocalRateLimiter(interval=0.5, clock=lambda: now[0], sleep=sleep)
    limiter.acquire(); limiter.acquire()
    assert waits == [0.5]

    calls = []
    def handler(request):
        calls.append(request.headers.get("if-none-match"))
        return httpx.Response(304) if len(calls) == 2 else httpx.Response(
            200, json=fixture("sec_issuer_mapping.json"), headers={"etag": '"mapping-1"'})
    sec = adapter(handler, monotonic=lambda: now[0], ttl_seconds=1)
    sec.resolve_issuer("ORI"); sec.resolve_issuer("ORI")
    assert len(calls) == 1
    now[0] += 2
    sec.resolve_issuer("ORI")
    assert calls == [None, '"mapping-1"']


def test_local_volume_limiter_coordinates_two_clients(tmp_path):
    now, waits = [100.0], []
    def sleeper(seconds):
        waits.append(seconds)
        now[0] += seconds
    options = {"interval": 0.5, "clock": lambda: now[0], "sleep": sleeper}
    first = SharedSQLiteRateLimiter(tmp_path / "rate.sqlite3", **options)
    second = SharedSQLiteRateLimiter(tmp_path / "rate.sqlite3", **options)
    first.acquire(); second.acquire(); first.acquire()
    assert waits == [0.0, 0.5, 0.5]


def test_api_contract_and_legacy_catalog(client):
    from backend.app.research.api import get_sec_client
    from backend.app.main import app
    app.dependency_overrides[get_sec_client] = lambda: adapter()
    try:
        response = client.get("/api/research/issuers?query=ORI")
        assert response.status_code == 200 and response.json()["ambiguous"]
        response = client.get("/api/research/issuers/1234567/filings?form=10-Q&include_history=true")
        assert response.status_code == 200 and len(response.json()["filings"]) == 2
        assert client.get("/api/research/issuers/not-a-cik/filings").status_code == 422
        assert len(client.get("/api/filings").json()["filings"]) == 5
    finally:
        app.dependency_overrides.clear()
