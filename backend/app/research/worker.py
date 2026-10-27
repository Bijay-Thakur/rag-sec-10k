"""One leased ingestion attempt; local hash diagnostics are not semantic readiness."""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Protocol

from .models import DiscoveryError, FilingRef
from .processing import chunk_sections, parse_filing
from .storage import LocalResearchRepository, RawObjectStore

PIPELINE_VERSION = "research-local-v1"


class EmbeddingBackend(Protocol):
    provider: str
    model: str
    dimensions: int
    index_kind: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashEmbedding:
    """Deterministic lexical diagnostic. Never claim this is a semantic model."""

    provider = "local"
    model = "token-hash-v1"
    dimensions = 128
    index_kind = "lexical_diagnostic"

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for term in re.findall(r"[a-z0-9]+", text.lower()):
                index = int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "big") % self.dimensions
                vector[index] += 1.0
            norm = math.sqrt(sum(v * v for v in vector))
            result.append([v / norm for v in vector] if norm else vector)
        return result


class OpenAIEmbedding:
    provider = "openai"
    model = "text-embedding-3-small"
    dimensions = 1536
    index_kind = "real_embeddings_local_store"

    def __init__(self, api_key: str):
        if not api_key:
            raise DiscoveryError("embedding_unconfigured", "Embedding API key unavailable", 503)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
        self.tokens_used: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for start in range(0, len(texts), 64):
            response = self.client.embeddings.create(model=self.model, input=texts[start:start + 64])
            result.extend([list(row.embedding) for row in sorted(response.data, key=lambda row: row.index)])
            if response.usage:
                self.tokens_used = (self.tokens_used or 0) + response.usage.total_tokens
        return result


def validate_vectors(vectors: list[list[float]], expected: int, dimensions: int | None = None) -> None:
    if len(vectors) != expected or expected <= 0:
        raise DiscoveryError("vector_mismatch", "Embedding count does not match chunks", 422)
    width = dimensions or len(vectors[0])
    if width <= 0 or any(len(row) != width or not all(math.isfinite(v) for v in row)
                         or not any(v != 0 for v in row) for row in vectors):
        raise DiscoveryError("vector_invalid", "Invalid embedding dimensions or values", 422)


class IngestionWorker:
    def __init__(self, repository: LocalResearchRepository, objects: RawObjectStore,
                 sec, embeddings: EmbeddingBackend):
        self.repository, self.objects, self.sec, self.embeddings = repository, objects, sec, embeddings

    def run_once(self, *, before_commit=None) -> dict | None:
        job = self.repository.claim()
        if not job:
            return None
        started = time.monotonic()
        try:
            filing_row = self.repository.filing(job["cik"], job["accession"])
            if not filing_row:
                raise DiscoveryError("filing_missing", "Queued filing metadata unavailable", 503)
            filing = FilingRef.model_validate_json(filing_row["ref_json"])
            raw, headers = self.sec.fetch_primary(filing)
            raw_sha, path = self.objects.put(raw)
            self.repository.advance(job["job_id"], "PARSING", attempt=job["attempt"], extra={"raw_bytes": len(raw)})
            parsed = parse_filing(raw)
            chunks = chunk_sections(filing, parsed, raw_sha)
            self.repository.advance(job["job_id"], "EMBEDDING", attempt=job["attempt"], extra={"sections": len(parsed.sections),
                                                                         "chunks": len(chunks)})
            vectors = self.embeddings.embed([chunk["text"] for chunk in chunks])
            validate_vectors(vectors, len(chunks), self.embeddings.dimensions)
            self.repository.advance(job["job_id"], "VALIDATING", attempt=job["attempt"])
            revision = self.repository.publish(job["job_id"], filing, raw_sha, path, headers,
                                                parsed, chunks, vectors, self.embeddings.provider,
                                                self.embeddings.model, self.embeddings.index_kind, attempt=job["attempt"],
                                                before_commit=before_commit)
            return {"job_id": job["job_id"], "revision_id": revision,
                    "raw_bytes": len(raw), "sections": len(parsed.sections), "chunks": len(chunks),
                    "embedding_tokens": getattr(self.embeddings, "tokens_used", None),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "index_kind": self.embeddings.index_kind}
        except Exception as exc:
            code = exc.code if isinstance(exc, DiscoveryError) else "worker_error"
            self.repository.fail(job["job_id"], code, attempt=job["attempt"])
            raise
