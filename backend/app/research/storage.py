"""Separate durable local research store; never opens the legacy Chroma DB."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .models import DiscoveryError, FilingRef, normalize_cik, validate_accession


class RawObjectStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> tuple[str, Path]:
        digest = hashlib.sha256(data).hexdigest()
        target = self.root / digest
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise DiscoveryError("raw_corrupt", "Existing source object failed checksum", 503)
            return digest, target
        temp = self.root / f".{uuid.uuid4().hex}.tmp"
        try:
            with temp.open("xb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            try:
                os.link(temp, target)  # exclusive publication; never overwrite same hash
            except FileExistsError:
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise DiscoveryError("raw_corrupt", "Concurrent source object failed checksum", 503)
            os.chmod(target, 0o444)
        finally:
            temp.unlink(missing_ok=True)
        return digest, target

    def read(self, digest: str) -> bytes:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise DiscoveryError("raw_invalid", "Invalid object checksum", 422)
        data = (self.root / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise DiscoveryError("raw_corrupt", "Source object failed checksum", 503)
        return data


SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
 cik TEXT NOT NULL, accession TEXT NOT NULL, form TEXT NOT NULL,
 filed_at TEXT NOT NULL, report_period TEXT, source_url TEXT NOT NULL,
 ref_json TEXT NOT NULL, ready_revision TEXT,
 PRIMARY KEY(cik, accession));
CREATE TABLE IF NOT EXISTS jobs (
 job_id TEXT PRIMARY KEY, cik TEXT NOT NULL, accession TEXT NOT NULL,
 pipeline_version TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
 queued_at REAL NOT NULL, updated_at REAL NOT NULL, lease_until REAL,
 error_code TEXT, progress_json TEXT NOT NULL DEFAULT '{}',
 UNIQUE(cik, accession, pipeline_version),
 FOREIGN KEY(cik, accession) REFERENCES filings(cik, accession));
CREATE TABLE IF NOT EXISTS raw_artifacts (
 sha256 TEXT PRIMARY KEY, object_path TEXT NOT NULL, bytes INTEGER NOT NULL,
 source_url TEXT NOT NULL, fetched_at TEXT NOT NULL, headers_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source_fetches (
 cik TEXT NOT NULL, accession TEXT NOT NULL, sha256 TEXT NOT NULL,
 source_url TEXT NOT NULL, fetched_at TEXT NOT NULL,
 PRIMARY KEY(cik, accession, sha256),
 FOREIGN KEY(cik, accession) REFERENCES filings(cik, accession),
 FOREIGN KEY(sha256) REFERENCES raw_artifacts(sha256));
CREATE TABLE IF NOT EXISTS revisions (
 revision_id TEXT PRIMARY KEY, cik TEXT NOT NULL, accession TEXT NOT NULL,
 raw_sha256 TEXT NOT NULL, pipeline_version TEXT NOT NULL,
 parser_version TEXT NOT NULL, chunker_version TEXT NOT NULL,
 embedding_provider TEXT NOT NULL, embedding_model TEXT NOT NULL,
 dimensions INTEGER NOT NULL, index_kind TEXT NOT NULL, qa_json TEXT NOT NULL,
 state TEXT NOT NULL, created_at REAL NOT NULL,
 FOREIGN KEY(cik, accession) REFERENCES filings(cik, accession),
 FOREIGN KEY(raw_sha256) REFERENCES raw_artifacts(sha256));
CREATE TABLE IF NOT EXISTS sections (
 revision_id TEXT NOT NULL, ordinal INTEGER NOT NULL, part TEXT, item TEXT NOT NULL,
 heading TEXT NOT NULL, text TEXT NOT NULL,
 PRIMARY KEY(revision_id, ordinal), FOREIGN KEY(revision_id) REFERENCES revisions(revision_id));
CREATE TABLE IF NOT EXISTS chunks (
 revision_id TEXT NOT NULL, chunk_id TEXT NOT NULL, section_ordinal INTEGER NOT NULL,
 position INTEGER NOT NULL, text TEXT NOT NULL, sha256 TEXT NOT NULL,
 token_count INTEGER NOT NULL, source_url TEXT NOT NULL, embedding_json TEXT NOT NULL,
 PRIMARY KEY(revision_id, chunk_id), FOREIGN KEY(revision_id, section_ordinal)
 REFERENCES sections(revision_id, ordinal));
CREATE INDEX IF NOT EXISTS jobs_state_lease ON jobs(state, lease_until);
CREATE INDEX IF NOT EXISTS chunks_revision_section ON chunks(revision_id, section_ordinal);
CREATE TABLE IF NOT EXISTS research_ready (
 cik TEXT NOT NULL, accession TEXT NOT NULL, revision_id TEXT NOT NULL,
 promoted_at REAL NOT NULL, PRIMARY KEY(cik, accession),
 FOREIGN KEY(cik, accession) REFERENCES filings(cik, accession),
 FOREIGN KEY(revision_id) REFERENCES revisions(revision_id));
CREATE TABLE IF NOT EXISTS research_budget (
 day INTEGER PRIMARY KEY, calls INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS research_events (
 trace_id TEXT NOT NULL, event TEXT NOT NULL, code TEXT NOT NULL,
 cik TEXT NOT NULL, accession TEXT NOT NULL, revision_id TEXT NOT NULL,
 metrics_json TEXT NOT NULL, occurred_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS research_events_time ON research_events(occurred_at);
"""


class LocalResearchRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path.resolve()
        if self.db_path.name == "chroma.sqlite3" or "/db/" in str(self.db_path):
            raise ValueError("Research storage cannot open the legacy Chroma index")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
        finally:
            db.close()

    def enqueue(self, filing: FilingRef, pipeline_version: str) -> dict:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO filings(cik,accession,form,filed_at,report_period,source_url,ref_json)"
                       " VALUES(?,?,?,?,?,?,?) ON CONFLICT(cik,accession) DO UPDATE SET ref_json=excluded.ref_json",
                       (filing.cik, filing.accession, filing.form, filing.filed_at.isoformat(),
                        filing.report_period.isoformat() if filing.report_period else None,
                        filing.primary_document_url, filing.model_dump_json()))
            existing = db.execute("SELECT * FROM jobs WHERE cik=? AND accession=? AND pipeline_version=?",
                                  (filing.cik, filing.accession, pipeline_version)).fetchone()
            if existing and existing["state"] == "FAILED":
                db.execute("UPDATE jobs SET state='QUEUED', error_code=NULL, updated_at=?, lease_until=NULL WHERE job_id=?",
                           (now, existing["job_id"]))
            elif not existing:
                db.execute("INSERT INTO jobs(job_id,cik,accession,pipeline_version,state,queued_at,updated_at)"
                           " VALUES(?,?,?,?,'QUEUED',?,?)",
                           (str(uuid.uuid4()), filing.cik, filing.accession, pipeline_version, now, now))
            db.commit()
        return self.job_by_key(filing.cik, filing.accession, pipeline_version)

    def job_by_key(self, cik: str, accession: str, version: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE cik=? AND accession=? AND pipeline_version=?",
                             (cik, accession, version)).fetchone()
            return dict(row) if row else {}

    def job(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def claim(self, *, lease_seconds: int = 300) -> dict | None:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state='QUEUED' OR "
                             "(state IN ('FETCHING','PARSING','EMBEDDING','VALIDATING') AND lease_until<?)"
                             " ORDER BY queued_at LIMIT 1", (now,)).fetchone()
            if not row:
                db.commit(); return None
            db.execute("UPDATE jobs SET state='FETCHING', attempt=attempt+1, updated_at=?, lease_until=?,"
                       " progress_json=? WHERE job_id=?", (now, now + lease_seconds,
                       json.dumps({"step": "FETCHING"}), row["job_id"]))
            db.commit()
            return self.job(row["job_id"])

    def advance(self, job_id: str, state: str, *, attempt: int, extra: dict | None = None) -> None:
        previous = {"PARSING": "FETCHING", "EMBEDDING": "PARSING", "VALIDATING": "EMBEDDING"}
        if state not in previous:
            raise ValueError("Invalid processing state")
        now = time.time()
        with self.connect() as db:
            result = db.execute("UPDATE jobs SET state=?, updated_at=?, lease_until=?, progress_json=? WHERE job_id=? AND state=? AND attempt=? AND lease_until>?",
                       (state, now, now + 300, json.dumps({"step": state, **(extra or {})}), job_id,
                        previous[state], attempt, now))
            if result.rowcount != 1:
                raise DiscoveryError("invalid_job_state", "Job transition rejected", 409)

    def fail(self, job_id: str, code: str, *, attempt: int) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='FAILED', error_code=?, updated_at=?, lease_until=NULL WHERE job_id=? "
                       "AND attempt=? AND state IN ('FETCHING','PARSING','EMBEDDING','VALIDATING')",
                       (code, time.time(), job_id, attempt))

    def filing(self, cik: str, accession: str) -> dict | None:
        cik, accession = normalize_cik(cik), validate_accession(accession)
        with self.connect() as db:
            row = db.execute("SELECT * FROM filings WHERE cik=? AND accession=?", (cik, accession)).fetchone()
            return dict(row) if row else None

    def sections(self, cik: str, accession: str) -> list[dict]:
        filing = self.filing(cik, accession)
        if not filing or not filing["ready_revision"]:
            return []
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT ordinal,part,item,heading,substr(text,1,500) AS excerpt FROM sections "
                "WHERE revision_id=? ORDER BY ordinal", (filing["ready_revision"],))]

    def search(self, cik: str, accession: str, query: str, limit: int = 5) -> list[dict]:
        filing = self.filing(cik, accession)
        if not filing or not filing["ready_revision"]:
            return []
        terms = set(query.lower().split())
        with self.connect() as db:
            rows = [dict(row) for row in db.execute(
                "SELECT chunk_id,text,source_url,section_ordinal FROM chunks WHERE revision_id=?",
                (filing["ready_revision"],))]
        rows.sort(key=lambda row: len(terms & set(row["text"].lower().split())), reverse=True)
        return rows[:limit]

    def vector_search(self, cik: str, accession: str, vector: list[float], limit: int = 5) -> list[dict]:
        filing = self.filing(cik, accession)
        if not filing or not filing["ready_revision"]:
            return []
        with self.connect() as db:
            revision = db.execute("SELECT dimensions FROM revisions WHERE revision_id=?",
                                  (filing["ready_revision"],)).fetchone()
            if not revision or len(vector) != revision["dimensions"] or not all(math.isfinite(v) for v in vector):
                raise DiscoveryError("vector_invalid", "Query vector differs from published model", 422)
            rows = [dict(row) for row in db.execute(
                "SELECT chunk_id,text,source_url,section_ordinal,embedding_json FROM chunks WHERE revision_id=?",
                (filing["ready_revision"],))]
        for row in rows:
            row["score"] = sum(a * b for a, b in zip(vector, json.loads(row.pop("embedding_json"))))
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows[:limit]

    def research_revision(self, cik: str, accession: str) -> dict | None:
        cik, accession = normalize_cik(cik), validate_accession(accession)
        with self.connect() as db:
            row = db.execute("SELECT r.* FROM research_ready p JOIN revisions r ON r.revision_id=p.revision_id "
                             "WHERE p.cik=? AND p.accession=? AND r.cik=p.cik AND r.accession=p.accession "
                             "AND r.state='READY_RESEARCH'", (cik, accession)).fetchone()
            return dict(row) if row else None

    def reserve_research_call(self, cap: int, *, now: float | None = None) -> None:
        if not 1 <= cap <= 100:
            raise DiscoveryError("research_budget_unconfigured", "Research call budget unavailable", 503)
        day = int((time.time() if now is None else now) // 86400)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT calls FROM research_budget WHERE day=?", (day,)).fetchone()
            if row and row["calls"] >= cap:
                raise DiscoveryError("research_budget_exhausted", "Research call budget exhausted", 429)
            db.execute("INSERT INTO research_budget VALUES(?,1) ON CONFLICT(day) DO UPDATE SET calls=calls+1", (day,))
            db.commit()

    def record_research_event(self, trace_id: str, event: str, code: str, cik: str,
                              accession: str, revision_id: str, metrics: dict) -> None:
        # Strict keys prevent prompts, headers, text, credentials or arbitrary payloads entering telemetry.
        allowed = {"elapsed_ms", "retrieval_ms", "generation_ms", "evidence_count", "citation_count",
                   "prompt_tokens", "completion_tokens", "total_tokens", "operand_count"}
        if event not in {"query_complete", "query_refused", "query_failed", "financial_complete", "financial_failed"} or not re.fullmatch(r"[a-z_]{1,80}", code):
            raise ValueError("Invalid research event")
        if set(metrics) - allowed or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                                         for v in metrics.values()):
            raise ValueError("Invalid research metrics")
        cik, accession = normalize_cik(cik), validate_accession(accession)
        uuid.UUID(trace_id); uuid.UUID(revision_id)
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM research_events WHERE occurred_at<?", (now - 7 * 86400,))
            db.execute("INSERT INTO research_events VALUES(?,?,?,?,?,?,?,?)",
                       (trace_id, event, code, cik, accession, revision_id, json.dumps(metrics), now))
            db.commit()

    def promote_research(self, cik: str, accession: str, revision_id: str) -> None:
        """Explicit offline operator gate; old pointer remains until one atomic commit."""
        cik, accession = normalize_cik(cik), validate_accession(accession)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM revisions WHERE revision_id=? AND cik=? AND accession=?",
                             (revision_id, cik, accession)).fetchone()
            filing = db.execute("SELECT ready_revision,source_url FROM filings WHERE cik=? AND accession=?",
                                (cik, accession)).fetchone()
            if (not row or not filing or row["state"] not in ("READY_LOCAL", "READY_RESEARCH")
                    or row["index_kind"] != "real_local_embeddings"
                    or row["embedding_provider"] != "local_sentence_transformer"
                    or not re.search(r"@sha256:[0-9a-f]{64}$", row["embedding_model"])
                    or filing["ready_revision"] != revision_id
                    or json.loads(row["qa_json"]).get("table_count", 0) > 0):
                raise DiscoveryError("research_not_ready", "Revision requires local-model and parser review", 409)
            chunks = db.execute("SELECT text,sha256,source_url,embedding_json FROM chunks WHERE revision_id=?",
                                (revision_id,)).fetchall()
            if not chunks:
                raise DiscoveryError("research_not_ready", "No chunks to publish", 409)
            for chunk in chunks:
                vector = json.loads(chunk["embedding_json"])
                if (chunk["source_url"] != filing["source_url"]
                        or hashlib.sha256(chunk["text"].encode()).hexdigest() != chunk["sha256"]
                        or len(vector) != row["dimensions"] or not all(math.isfinite(v) for v in vector)
                        or abs(sum(v * v for v in vector) - 1) > .01):
                    raise DiscoveryError("research_not_ready", "Chunk integrity failure", 409)
            db.execute("UPDATE revisions SET state='READY_RESEARCH' WHERE revision_id=?", (revision_id,))
            db.execute("INSERT INTO research_ready VALUES(?,?,?,?) ON CONFLICT(cik,accession) DO UPDATE "
                       "SET revision_id=excluded.revision_id,promoted_at=excluded.promoted_at",
                       (cik, accession, revision_id, time.time()))
            db.commit()

    def rollback_research(self, cik: str, accession: str, revision_id: str) -> None:
        cik, accession = normalize_cik(cik), validate_accession(accession)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            revision = db.execute("SELECT r.*,a.object_path,f.source_url FROM revisions r JOIN raw_artifacts a "
                                  "ON a.sha256=r.raw_sha256 JOIN filings f ON f.cik=r.cik AND f.accession=r.accession "
                                  "WHERE r.revision_id=? AND r.cik=? AND r.accession=? AND r.state='READY_RESEARCH'",
                                  (revision_id, cik, accession)).fetchone()
            if not revision:
                raise DiscoveryError("research_not_ready", "Rollback requires a prior validated revision in this scope", 409)
            if hashlib.sha256(Path(revision["object_path"]).read_bytes()).hexdigest() != revision["raw_sha256"]:
                raise DiscoveryError("raw_corrupt", "Rollback source checksum mismatch", 503)
            rows = db.execute("SELECT text,sha256,source_url,embedding_json FROM chunks WHERE revision_id=?",
                              (revision_id,)).fetchall()
            if not rows:
                raise DiscoveryError("research_not_ready", "Rollback index is empty", 409)
            for row in rows:
                vector = json.loads(row["embedding_json"])
                if (hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]
                        or row["source_url"] != revision["source_url"] or len(vector) != revision["dimensions"]
                        or not all(math.isfinite(value) for value in vector)
                        or abs(sum(value * value for value in vector) - 1) > .01):
                    raise DiscoveryError("chunk_corrupt", "Rollback index integrity failed", 503)
            db.execute("INSERT INTO research_ready VALUES(?,?,?,?) ON CONFLICT(cik,accession) DO UPDATE "
                       "SET revision_id=excluded.revision_id,promoted_at=excluded.promoted_at",
                       (cik, accession, revision_id, time.time()))
            db.commit()

    def publish(self, job_id: str, filing: FilingRef, raw_sha: str, object_path: Path,
                headers: dict, parsed, chunks: list[dict], vectors: list[list[float]],
                provider: str, model: str, index_kind: str, *, attempt: int, before_commit=None) -> str:
        from .processing import CHUNKER_VERSION, PARSER_VERSION
        from .worker import validate_vectors

        validate_vectors(vectors, len(chunks))
        if len(parsed.sections) != parsed.qa["section_count"] or len(chunks) != len(vectors):
            raise DiscoveryError("publication_mismatch", "Staged record counts disagree", 422)
        revision = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT state,cik,accession,pipeline_version,attempt,lease_until FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if (not job or job["state"] != "VALIDATING" or job["attempt"] != attempt
                    or job["lease_until"] <= time.time()
                    or (job["cik"], job["accession"]) != (filing.cik, filing.accession)):
                raise DiscoveryError("invalid_job_state", "Job cannot publish this filing", 409)
            if hashlib.sha256(object_path.read_bytes()).hexdigest() != raw_sha:
                raise DiscoveryError("raw_corrupt", "Raw object checksum mismatch", 503)
            db.execute("INSERT OR IGNORE INTO raw_artifacts VALUES(?,?,?,?,?,?)",
                       (raw_sha, str(object_path), object_path.stat().st_size,
                        filing.primary_document_url, headers["fetched_at"], json.dumps(headers)))
            db.execute("INSERT OR IGNORE INTO source_fetches VALUES(?,?,?,?,?)",
                       (filing.cik, filing.accession, raw_sha, filing.primary_document_url,
                        headers["fetched_at"]))
            db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (revision, filing.cik, filing.accession, raw_sha, job["pipeline_version"],
                        PARSER_VERSION, CHUNKER_VERSION, provider, model, len(vectors[0]),
                        index_kind, json.dumps(parsed.qa), "STAGED", time.time()))
            for section in parsed.sections:
                db.execute("INSERT INTO sections VALUES(?,?,?,?,?,?)",
                           (revision, section.ordinal, section.part, section.item, section.heading, section.text))
            for chunk, vector in zip(chunks, vectors):
                db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                           (revision, chunk["chunk_id"], chunk["section_ordinal"], chunk["position"],
                            chunk["text"], chunk["sha256"], chunk["token_count"],
                            chunk["source_url"], json.dumps(vector)))
            if before_commit:
                before_commit()
            db.execute("UPDATE revisions SET state='READY_LOCAL' WHERE revision_id=?", (revision,))
            db.execute("UPDATE filings SET ready_revision=? WHERE cik=? AND accession=?",
                       (revision, filing.cik, filing.accession))
            db.execute("UPDATE jobs SET state='READY_LOCAL', updated_at=?, lease_until=NULL,"
                       " progress_json=? WHERE job_id=?", (time.time(),
                       json.dumps({"step": "READY_LOCAL", "sections": len(parsed.sections),
                                   "chunks": len(chunks), "raw_bytes": object_path.stat().st_size}), job_id))
            db.commit()
        return revision
