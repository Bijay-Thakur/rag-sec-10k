"""Append-only, raw-source-linked XBRL snapshots for local staging."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal

from .facts import Fact, normalize_companyfacts
from .models import DiscoveryError, normalize_cik, validate_accession

MAP_VERSION = "as-filed-v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_snapshots (
 snapshot_id TEXT PRIMARY KEY, cik TEXT NOT NULL, accession TEXT NOT NULL,
 revision_id TEXT NOT NULL, raw_sha256 TEXT NOT NULL, map_version TEXT NOT NULL,
 facts_json TEXT NOT NULL, created_at REAL NOT NULL,
 FOREIGN KEY(revision_id) REFERENCES revisions(revision_id),
 FOREIGN KEY(raw_sha256) REFERENCES raw_artifacts(sha256));
CREATE TABLE IF NOT EXISTS fact_ready (
 cik TEXT NOT NULL, accession TEXT NOT NULL, revision_id TEXT NOT NULL,
 snapshot_id TEXT NOT NULL, PRIMARY KEY(cik,accession,revision_id),
 FOREIGN KEY(snapshot_id) REFERENCES fact_snapshots(snapshot_id));
"""


class FactSnapshots:
    def __init__(self, repository, objects):
        self.repo, self.objects = repository, objects
        with self.repo.connect() as db:
            db.executescript(SCHEMA)

    def publish(self, cik: str, accession: str, revision_id: str, raw: bytes,
                observed_at: datetime) -> str:
        cik, accession = normalize_cik(cik), validate_accession(accession)
        if len(raw) > 8 * 1024 * 1024 or observed_at.tzinfo is None:
            raise DiscoveryError("xbrl_invalid", "XBRL snapshot size or timestamp invalid", 422)
        try:
            facts = normalize_companyfacts(json.loads(raw, parse_float=Decimal), cik, accession)
        except (ValueError, UnicodeError) as exc:
            raise DiscoveryError("xbrl_schema", "Invalid XBRL source JSON", 422) from exc
        if not facts:
            raise DiscoveryError("fact_unavailable", "No facts for selected accession", 404)
        sha, path = self.objects.put(raw)
        identity = hashlib.sha256(f"{cik}|{accession}|{revision_id}|{sha}|{MAP_VERSION}".encode()).hexdigest()
        source = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        with self.repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            revision = db.execute("SELECT f.form FROM revisions r JOIN filings f ON f.cik=r.cik AND f.accession=r.accession "
                                  "WHERE r.cik=? AND r.accession=? AND r.revision_id=?",
                                  (cik, accession, revision_id)).fetchone()
            if not revision or any(fact.form != revision["form"] for fact in facts):
                raise DiscoveryError("scope_mismatch", "Facts require an exact filing revision", 409)
            db.execute("INSERT OR IGNORE INTO raw_artifacts VALUES(?,?,?,?,?,?)",
                       (sha, str(path), len(raw), source, observed_at.isoformat(), "{}"))
            db.execute("INSERT OR IGNORE INTO fact_snapshots VALUES(?,?,?,?,?,?,?,?)",
                       (identity, cik, accession, revision_id, sha, MAP_VERSION,
                        json.dumps([asdict(fact) for fact in facts], default=str), time.time()))
            db.execute("INSERT INTO fact_ready VALUES(?,?,?,?) ON CONFLICT(cik,accession,revision_id) "
                       "DO UPDATE SET snapshot_id=excluded.snapshot_id", (cik, accession, revision_id, identity))
            db.commit()
        return identity

    def read(self, cik: str, accession: str, revision_id: str) -> tuple[list[Fact], dict]:
        cik, accession = normalize_cik(cik), validate_accession(accession)
        with self.repo.connect() as db:
            row = db.execute("SELECT s.*,a.source_url,a.fetched_at FROM fact_ready p "
                             "JOIN fact_snapshots s ON s.snapshot_id=p.snapshot_id JOIN raw_artifacts a ON a.sha256=s.raw_sha256 "
                             "WHERE p.cik=? AND p.accession=? AND p.revision_id=? "
                             "AND s.cik=p.cik AND s.accession=p.accession AND s.revision_id=p.revision_id",
                             (cik, accession, revision_id)).fetchone()
        if not row:
            raise DiscoveryError("fact_unavailable", "No fact snapshot for selected revision", 404)
        raw = self.objects.read(row["raw_sha256"])
        facts = normalize_companyfacts(json.loads(raw, parse_float=Decimal), cik, accession)
        if json.dumps([asdict(fact) for fact in facts], default=str) != row["facts_json"]:
            raise DiscoveryError("fact_corrupt", "Fact records differ from immutable source", 503)
        return facts, {"snapshot_id": row["snapshot_id"], "raw_sha256": row["raw_sha256"],
                       "source_url": row["source_url"], "fetched_at": row["fetched_at"], "map_version": row["map_version"],
                       "filing_source_url": self.repo.filing(cik, accession)["source_url"]}

    def rollback(self, cik: str, accession: str, revision_id: str, snapshot_id: str) -> None:
        cik, accession = normalize_cik(cik), validate_accession(accession)
        with self.repo.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT raw_sha256,facts_json FROM fact_snapshots WHERE snapshot_id=? AND cik=? AND accession=? AND revision_id=?",
                             (snapshot_id, cik, accession, revision_id)).fetchone()
            if not row:
                raise DiscoveryError("scope_mismatch", "Rollback snapshot must match exact filing revision", 409)
            facts = normalize_companyfacts(json.loads(self.objects.read(row["raw_sha256"]), parse_float=Decimal), cik, accession)
            if json.dumps([asdict(fact) for fact in facts], default=str) != row["facts_json"]:
                raise DiscoveryError("fact_corrupt", "Rollback fact records differ from source", 503)
            db.execute("INSERT INTO fact_ready VALUES(?,?,?,?) ON CONFLICT(cik,accession,revision_id) "
                       "DO UPDATE SET snapshot_id=excluded.snapshot_id", (cik, accession, revision_id, snapshot_id))
            db.commit()
