"""Synthetic XBRL snapshot, exact source pointer and financial preview tests."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from backend.app.research.fact_storage import FactSnapshots
from backend.app.research.models import DiscoveryError
from backend.app.research.sec_client import SecHttpAdapter
from backend.app.research.storage import LocalResearchRepository, RawObjectStore
from backend.app.research.worker import IngestionWorker, LocalHashEmbedding, PIPELINE_VERSION
from tests.test_research_ingestion import FakeSec, filing


def raw_facts(ref, current=120):
    return json.dumps({"cik": int(ref.cik), "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"accn": ref.accession, "val": current, "start": "2026-01-01", "end": "2026-03-31", "filed": "2026-05-01", "form": ref.form},
        {"accn": ref.accession, "val": 100, "start": "2025-01-01", "end": "2025-03-31", "filed": "2026-05-01", "form": ref.form}
    ]}}}}}, indent=2).encode()


@pytest.fixture
def snapshots(tmp_path):
    repo = LocalResearchRepository(tmp_path / "research.sqlite3")
    objects = RawObjectStore(tmp_path / "objects")
    ref = filing()
    repo.enqueue(ref, PIPELINE_VERSION)
    revision = IngestionWorker(repo, objects, FakeSec(), LocalHashEmbedding()).run_once()["revision_id"]
    store = FactSnapshots(repo, objects)
    store.publish(ref.cik, ref.accession, revision, raw_facts(ref), datetime.now(timezone.utc))
    return store, ref, revision


def test_exact_source_bytes_and_pointers_survive_restart(snapshots):
    store, ref, revision = snapshots
    reopened = FactSnapshots(LocalResearchRepository(store.repo.db_path), store.objects)
    facts, source = reopened.read(ref.cik, ref.accession, revision)
    assert len(facts) == 2 and facts[0].value == 120
    assert store.objects.read(source["raw_sha256"]) == raw_facts(ref)
    assert facts[0].source_pointer == "/facts/us-gaap/Revenues/units/USD/0"
    assert reopened.publish(ref.cik, ref.accession, revision, raw_facts(ref), datetime.now(timezone.utc)) == source["snapshot_id"]
    with pytest.raises(DiscoveryError, match="selected revision"):
        reopened.read("0002222222", ref.accession, revision)
    with pytest.raises(DiscoveryError, match="exact filing revision"):
        reopened.publish(ref.cik, ref.accession, "wrong-revision", raw_facts(ref), datetime.now(timezone.utc))


def test_changed_snapshot_preserves_old_source_and_rejects_corruption(snapshots):
    store, ref, revision = snapshots
    _, old = store.read(ref.cik, ref.accession, revision)
    store.publish(ref.cik, ref.accession, revision, raw_facts(ref, 140), datetime.now(timezone.utc))
    facts, new = store.read(ref.cik, ref.accession, revision)
    assert old["snapshot_id"] != new["snapshot_id"] and facts[0].value == 140
    assert store.objects.read(old["raw_sha256"]) == raw_facts(ref)
    store.rollback(ref.cik, ref.accession, revision, old["snapshot_id"])
    assert store.read(ref.cik, ref.accession, revision)[0][0].value == 120
    store.rollback(ref.cik, ref.accession, revision, new["snapshot_id"])
    with store.repo.connect() as db:
        db.execute("UPDATE fact_snapshots SET facts_json='[]' WHERE snapshot_id=?", (new["snapshot_id"],))
    with pytest.raises(DiscoveryError, match="immutable source"):
        store.read(ref.cik, ref.accession, revision)


def test_financial_route_requires_flag_and_returns_validated_operands(client, snapshots, monkeypatch):
    from backend.app.main import app
    from backend.app.research.facts_api import get_fact_snapshots
    store, ref, revision = snapshots
    body = {"cik": ref.cik, "accession": ref.accession, "revision_id": revision, "formula": "yoy",
            "current": {"concept": "Revenues", "start": "2026-01-01", "end": "2026-03-31"},
            "prior": {"concept": "Revenues", "start": "2025-01-01", "end": "2025-03-31"}}
    app.dependency_overrides[get_fact_snapshots] = lambda: store
    headers = {"Authorization": "Bearer test-operator-secret-over-24-chars"}
    try:
        monkeypatch.setenv("RESEARCH_QUERY_ENABLED", "true")
        monkeypatch.setenv("RESEARCH_SINGLE_INSTANCE_MODE", "true")
        monkeypatch.setenv("RESEARCH_QUERY_TOKEN", "test-operator-secret-over-24-chars")
        monkeypatch.delenv("RESEARCH_FACTS_ENABLED", raising=False)
        assert client.post("/api/research/financial", json=body, headers=headers).status_code == 403
        monkeypatch.setenv("RESEARCH_FACTS_ENABLED", "true")
        response = client.post("/api/research/financial", json=body, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert result["value"] == "20.00" and result["kind"] == "calculated"
        assert result["operands"][0]["source_pointer"].endswith("/0")
        assert result["source"]["raw_sha256"] and result["revision_id"] == revision
        invalid = {**body, "unit": "shares"}
        assert client.post("/api/research/financial", json=invalid, headers=headers).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_sec_fact_source_retains_exact_bytes():
    ref = filing()
    original = raw_facts(ref)
    sec = SecHttpAdapter(user_agent="Fixture test@invalid.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=original, headers={"content-type": "application/json"})))
    data, observed, source = sec.fetch_companyfacts_bytes(ref.cik)
    assert data == original and observed.tzinfo and source.endswith(f"CIK{ref.cik}.json")


def test_amendment_never_replaces_original_and_fact_backup_restores(snapshots, tmp_path):
    from backend.app.research.backup import snapshot, restore
    store, original, original_revision = snapshots
    amended = filing(accession="0001111111-26-000013", form="10-Q/A")
    store.repo.enqueue(amended, PIPELINE_VERSION)
    revision = IngestionWorker(store.repo, store.objects, FakeSec(), LocalHashEmbedding()).run_once()["revision_id"]
    store.publish(amended.cik, amended.accession, revision, raw_facts(amended, 150), datetime.now(timezone.utc))
    assert store.read(original.cik, original.accession, original_revision)[0][0].value == 120
    assert store.read(amended.cik, amended.accession, revision)[0][0].value == 150
    snapshot(store.repo, store.objects, tmp_path / "backup")
    restored_repo = restore(tmp_path / "backup", tmp_path / "restored")
    restored = FactSnapshots(restored_repo, RawObjectStore(tmp_path / "restored" / "objects"))
    assert restored.read(original.cik, original.accession, original_revision)[0][0].value == 120
    assert restored.read(amended.cik, amended.accession, revision)[0][0].value == 150
