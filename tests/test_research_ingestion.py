"""Durability and publication safety on an isolated local research volume."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import httpx
import pytest

from backend.app.research.models import DiscoveryError, FilingRef, primary_document_url
from backend.app.research.processing import chunk_sections, parse_filing
from backend.app.research.sec_client import SecHttpAdapter
from backend.app.research.storage import LocalResearchRepository, RawObjectStore
from backend.app.research.worker import IngestionWorker, LocalHashEmbedding, validate_vectors, PIPELINE_VERSION


def filing(cik="0001234567", accession="0001111111-26-000012", form="10-Q"):
    return FilingRef(cik=cik, accession=accession, form=form, filed_at=date(2026, 5, 1),
                     report_period=date(2026, 3, 31), primary_document="quarter.htm",
                     primary_document_url=primary_document_url(cik, accession, "quarter.htm"),
                     source_submissions_url=f"https://data.sec.gov/submissions/CIK{cik}.json",
                     metadata_observed_at=datetime.now(timezone.utc))


HTML = b"""<html><body><h1>PART I</h1><h2>Item 1. Business</h2>
<p>Orion develops optical instruments used by research laboratories and supplies equipment to clients worldwide.</p>
<h2>Item 1. Updated business note</h2><p>Additional material describes manufacturing sites, suppliers, and research investments during the quarter.</p>
<h2>Item 2. Management discussion</h2><p>Management discusses production capacity and inventory risk over the reporting period.</p>
<table><caption>Operations</caption><tr><th>Year</th><th>Revenue</th></tr><tr><td>2026</td><td>100</td></tr></table>
</body></html>"""
ANNUAL_HTML = b"""<html><body><div>PART II</div><div>ITEM 7 - Management's Discussion</div>
<p>This annual report discusses material changes to operations, cash resources, and market risks for the full fiscal year.</p>
<div>ITEM 8: Financial Statements</div><p>Statements contain consolidated information with notes explaining accounting policy changes.</p>
</body></html>"""


class FakeSec:
    def __init__(self, raw=HTML, fail=False):
        self.raw, self.fail, self.calls = raw, fail, 0

    def fetch_primary(self, ref):
        self.calls += 1
        if self.fail:
            raise DiscoveryError("sec_503", "SEC unavailable", 503)
        return self.raw, {"fetched_at": "2026-10-27T00:00:00+00:00", "source_url": ref.primary_document_url,
                          "content_type": "text/html"}


@pytest.fixture
def store(tmp_path):
    return LocalResearchRepository(tmp_path / "research.sqlite3"), RawObjectStore(tmp_path / "objects")


def test_parser_keeps_repeated_item_and_table_warning():
    parsed = parse_filing(HTML)
    assert parsed.qa["duplicate_items_retained"] == 1
    assert parsed.qa["table_count"] == 1
    assert [section.item for section in parsed.sections] == ["Item 1", "Item 1", "Item 2"]
    first = filing()
    ids = [chunk["chunk_id"] for chunk in chunk_sections(first, parsed, "a" * 64)]
    assert len(ids) == len(set(ids))
    other = filing(accession="0001111111-26-000013")
    assert ids[0] != chunk_sections(other, parsed, "a" * 64)[0]["chunk_id"]
    annual = parse_filing(ANNUAL_HTML)
    assert [s.item for s in annual.sections] == ["Item 7", "Item 8"]
    assert annual.qa["table_count"] == 0


def test_idempotency_restart_and_accession_scope(store):
    repo, objects = store
    ref = filing()
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = list(pool.map(lambda _: repo.enqueue(ref, PIPELINE_VERSION)["job_id"], range(12)))
    assert len(set(jobs)) == 1
    worker = IngestionWorker(repo, objects, FakeSec(), LocalHashEmbedding())
    result = worker.run_once()
    assert result and result["sections"] == 3
    ready = repo.filing(ref.cik, ref.accession)["ready_revision"]
    assert ready == result["revision_id"]
    assert repo.enqueue(ref, PIPELINE_VERSION)["job_id"] == jobs[0]
    assert worker.run_once() is None
    reopened = LocalResearchRepository(repo.db_path)
    assert reopened.filing(ref.cik, ref.accession)["ready_revision"] == ready
    assert len(reopened.sections(ref.cik, ref.accession)) == 3
    assert reopened.search(ref.cik, ref.accession, "manufacturing")
    query_vector = LocalHashEmbedding().embed(["manufacturing sites suppliers"])[0]
    assert reopened.vector_search(ref.cik, ref.accession, query_vector)[0]["source_url"] == ref.primary_document_url
    assert reopened.vector_search("0002222222", ref.accession, query_vector) == []
    assert reopened.search("0002222222", ref.accession, "manufacturing") == []
    assert reopened.search(ref.cik, "0001111111-26-000099", "manufacturing") == []


def test_failed_reprocess_and_before_commit_rollback_keep_ready(store):
    repo, objects = store
    ref = filing()
    repo.enqueue(ref, PIPELINE_VERSION)
    IngestionWorker(repo, objects, FakeSec(), LocalHashEmbedding()).run_once()
    old = repo.filing(ref.cik, ref.accession)["ready_revision"]
    repo.enqueue(ref, "research-local-v2")
    with pytest.raises(DiscoveryError):
        IngestionWorker(repo, objects, FakeSec(fail=True), LocalHashEmbedding()).run_once()
    assert repo.filing(ref.cik, ref.accession)["ready_revision"] == old
    assert repo.sections(ref.cik, ref.accession)
    repo.enqueue(ref, "research-local-v2")
    with pytest.raises(RuntimeError, match="simulated"):
        IngestionWorker(repo, objects, FakeSec(), LocalHashEmbedding()).run_once(
            before_commit=lambda: (_ for _ in ()).throw(RuntimeError("simulated crash")))
    assert repo.filing(ref.cik, ref.accession)["ready_revision"] == old
    with repo.connect() as db:
        assert db.execute("SELECT count(*) FROM revisions").fetchone()[0] == 1
    repo.enqueue(ref, "research-local-v2")
    result = IngestionWorker(repo, objects, FakeSec(), LocalHashEmbedding()).run_once()
    assert result["revision_id"] != old


def test_parse_and_embedding_failures_are_recorded_without_ready(store):
    repo, objects = store
    ref = filing()
    repo.enqueue(ref, PIPELINE_VERSION)
    with pytest.raises(DiscoveryError):
        IngestionWorker(repo, objects, FakeSec(raw=b"<html>Empty</html>"), LocalHashEmbedding()).run_once()
    assert repo.job_by_key(ref.cik, ref.accession, PIPELINE_VERSION)["error_code"] == "parser_quality"
    assert repo.sections(ref.cik, ref.accession) == []
    repo.enqueue(ref, PIPELINE_VERSION)
    class BadEmbedding(LocalHashEmbedding):
        def embed(self, texts):
            return [[0.0] * self.dimensions for _ in texts]
    with pytest.raises(DiscoveryError):
        IngestionWorker(repo, objects, FakeSec(), BadEmbedding()).run_once()
    assert repo.job_by_key(ref.cik, ref.accession, PIPELINE_VERSION)["error_code"] == "vector_invalid"
    assert repo.filing(ref.cik, ref.accession)["ready_revision"] is None


def test_expired_lease_is_reclaimed_without_duplicate_job(store):
    repo, objects = store
    ref = filing()
    initial = repo.enqueue(ref, PIPELINE_VERSION)
    claimed = repo.claim(lease_seconds=1)
    assert claimed["job_id"] == initial["job_id"]
    with repo.connect() as db:
        db.execute("UPDATE jobs SET lease_until=0 WHERE job_id=?", (claimed["job_id"],))
    restarted = LocalResearchRepository(repo.db_path)
    result = IngestionWorker(restarted, objects, FakeSec(), LocalHashEmbedding()).run_once()
    assert result["job_id"] == initial["job_id"]
    assert restarted.job(initial["job_id"])["attempt"] == 2


def test_stale_attempt_cannot_change_reclaimed_job(store):
    repo, _ = store
    ref = filing()
    repo.enqueue(ref, PIPELINE_VERSION)
    stale = repo.claim(lease_seconds=1)
    with repo.connect() as db:
        db.execute("UPDATE jobs SET lease_until=0 WHERE job_id=?", (stale["job_id"],))
    fresh = repo.claim()
    assert fresh["attempt"] == stale["attempt"] + 1
    with pytest.raises(DiscoveryError) as exc:
        repo.advance(stale["job_id"], "PARSING", attempt=stale["attempt"])
    assert exc.value.code == "invalid_job_state"
    repo.fail(stale["job_id"], "old_failure", attempt=stale["attempt"])
    assert repo.job(fresh["job_id"])["state"] == "FETCHING"


def test_real_client_factory_requires_configured_policy(monkeypatch, tmp_path):
    from backend.app.research.api import get_sec_client
    for key in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SEC_SINGLE_INSTANCE_MODE", "true")
    monkeypatch.setenv("SEC_RATE_DB_PATH", str(tmp_path / "sec-budget.sqlite3"))
    monkeypatch.setenv("SEC_USER_AGENT", "Example Research contact@real-domain.org")
    get_sec_client.cache_clear()
    try:
        assert get_sec_client().throttle.path.exists()
    finally:
        get_sec_client.cache_clear()


@pytest.mark.parametrize("vectors", [[[0.0] * 128], [[math.nan] + [0.0] * 127], [[1.0]]])
def test_bad_vectors_fail_publication(vectors):
    with pytest.raises(DiscoveryError):
        validate_vectors(vectors, 1, dimensions=128)


def test_immutable_artifact_and_legacy_db_is_rejected(store, tmp_path):
    repo, objects = store
    digest, path = objects.put(HTML)
    assert objects.put(HTML)[0] == digest
    assert objects.read(digest) == HTML
    assert path.name == digest
    LocalResearchRepository(repo.db_path)  # idempotent migration
    with pytest.raises(ValueError):
        LocalResearchRepository(tmp_path / "db" / "chroma.sqlite3")


def test_fetch_restricts_url_and_bytes():
    ref = filing()
    sec = SecHttpAdapter(user_agent="Fixture testing@invalid.test",
                         transport=httpx.MockTransport(lambda request: httpx.Response(
                             200, content=HTML, headers={"content-type": "text/html"})))
    raw, metadata = sec.fetch_primary(ref)
    assert raw == HTML and metadata["source_url"] == ref.primary_document_url
    changed = ref.model_copy(update={"primary_document_url": "https://evil.example/steal"})
    with pytest.raises(DiscoveryError, match="disagrees"):
        sec.fetch_primary(changed)


@pytest.mark.parametrize("status,code", [(403, "sec_403"), (404, "sec_404"), (429, "sec_unavailable"),
                                         (500, "sec_unavailable"), (302, "sec_redirect")])
def test_fetch_failure_policy(status, code):
    sec = SecHttpAdapter(user_agent="Fixture testing@invalid.test", max_attempts=2,
                         sleep=lambda _: None, transport=httpx.MockTransport(
                             lambda request: httpx.Response(status, headers={"location": "https://evil.example"})))
    with pytest.raises(DiscoveryError) as exc:
        sec.fetch_primary(filing())
    assert exc.value.code == code


def test_fetch_rejects_oversize_and_bad_mime(monkeypatch):
    from backend.app.research import sec_client
    monkeypatch.setattr(sec_client, "MAX_FILING_BYTES", 50)
    sec = SecHttpAdapter(user_agent="Fixture testing@invalid.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=HTML, headers={"content-type": "text/html"})))
    with pytest.raises(DiscoveryError) as exc:
        sec.fetch_primary(filing())
    assert exc.value.code == "filing_too_large"
    sec = SecHttpAdapter(user_agent="Fixture testing@invalid.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"PDF", headers={"content-type": "application/pdf"})))
    with pytest.raises(DiscoveryError) as exc:
        sec.fetch_primary(filing())
    assert exc.value.code == "unsupported_document"


def test_ingestion_api_default_off_and_read_only_status(client, store, monkeypatch):
    from backend.app.main import app
    from backend.app.research.api import get_sec_client
    from backend.app.research.ingestion_api import get_research_repository

    repo, objects = store
    fake = FakeSec()
    fake.list_filings = lambda cik, **kwargs: type("Listing", (), {"filings": [filing()]})()
    app.dependency_overrides[get_sec_client] = lambda: fake
    app.dependency_overrides[get_research_repository] = lambda: repo
    monkeypatch.delenv("RESEARCH_INGESTION_ENABLED", raising=False)
    try:
        response = client.post("/api/research/ingestions", json={"cik": "1234567", "accession": "0001111111-26-000012"})
        assert response.status_code == 403 and fake.calls == 0
        monkeypatch.setenv("RESEARCH_INGESTION_ENABLED", "true")
        monkeypatch.setenv("RESEARCH_INGESTION_TOKEN", "local-operator-token-over-24-characters")
        response = client.post("/api/research/ingestions", headers={"Authorization": "Bearer local-operator-token-over-24-characters"},
                               json={"cik": "1234567", "accession": "0001111111-26-000012"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert client.get(f"/api/research/ingestions/{job_id}").json()["state"] == "QUEUED"
        IngestionWorker(repo, objects, fake, LocalHashEmbedding()).run_once()
        assert client.get("/api/research/filings/1234567/0001111111-26-000012/sections").json()["sections"]
        diagnostic = client.get("/api/research/filings/1234567/0001111111-26-000012/diagnostic?query=suppliers&mode=local-vector")
        assert diagnostic.status_code == 200 and diagnostic.json()["hits"]
    finally:
        app.dependency_overrides.clear()
