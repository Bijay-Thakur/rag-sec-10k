"""Pinned accession/revision retrieval; no query ever scans another filing."""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from .models import DiscoveryError, normalize_cik, validate_accession

WORDS = re.compile(r"[a-z0-9]+")


def terms(value: str) -> list[str]:
    return WORDS.findall(value.casefold())


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, identity in enumerate(dict.fromkeys(ranking), 1):
            scores[identity] = scores.get(identity, 0.0) + 1 / (k + position)
    return sorted(scores, key=lambda identity: (-scores[identity], identity))


def retrieve(repo, cik: str, accession: str, revision_id: str, query: str,
             embedder, *, limit: int = 5) -> list[dict]:
    cik, accession = normalize_cik(cik), validate_accession(accession)
    ready = repo.research_revision(cik, accession)
    if not ready or ready["revision_id"] != revision_id:
        raise DiscoveryError("research_not_ready", "Selected filing revision unavailable", 409)
    if ready["embedding_model"] != embedder.model or ready["dimensions"] != embedder.dimensions:
        raise DiscoveryError("model_mismatch", "Query model differs from index model", 409)
    if not 1 <= limit <= 10 or not 2 <= len(query) <= 500:
        raise DiscoveryError("invalid_query", "Invalid research query or result limit", 422)
    with repo.connect() as db:
        rows = [dict(row) for row in db.execute(
            "SELECT c.chunk_id,c.text,c.sha256,c.source_url,c.embedding_json,c.section_ordinal,"
            "s.part,s.item FROM chunks c JOIN sections s ON s.revision_id=c.revision_id "
            "AND s.ordinal=c.section_ordinal JOIN revisions r ON r.revision_id=c.revision_id "
            "WHERE c.revision_id=? AND r.cik=? AND r.accession=?",
            (revision_id, cik, accession))]
    if not rows:
        raise DiscoveryError("research_not_ready", "Index contains no chunks", 409)
    q = Counter(terms(query))
    df = Counter(word for row in rows for word in set(terms(row["text"])))
    average = sum(len(terms(row["text"])) for row in rows) / len(rows)
    for row in rows:
        words = terms(row["text"]); tf = Counter(words)
        row["lexical_score"] = sum(math.log(1 + (len(rows) - df[word] + .5) / (df[word] + .5))
                                   * (count * 2.2 / (count + 1.2 * (.25 + .75 * len(words) / average)))
                                   for word in q for count in [tf[word]] if count)
    query_vector = embedder.embed([query])[0]
    if len(query_vector) != ready["dimensions"] or not all(math.isfinite(v) for v in query_vector):
        raise DiscoveryError("vector_invalid", "Invalid query vector", 422)
    for row in rows:
        vector = json.loads(row.pop("embedding_json"))
        row["dense_score"] = sum(a * b for a, b in zip(query_vector, vector))
    lexical = sorted(rows, key=lambda r: (-r["lexical_score"], r["chunk_id"]))
    dense = sorted(rows, key=lambda r: (-r["dense_score"], r["chunk_id"]))
    # A zero lexical match is not evidence. Dense hits remain bounded to this revision.
    fused = rrf([r["chunk_id"] for r in lexical if r["lexical_score"] > 0][:30],
                [r["chunk_id"] for r in dense[:30]])
    mapping = {r["chunk_id"]: r for r in rows}
    result = [mapping[identity] for identity in fused[:limit]]
    if any(r["source_url"] != repo.filing(cik, accession)["source_url"] for r in result):
        raise DiscoveryError("scope_violation", "Source scope mismatch", 503)
    return result
