"""Vector retrieval against Chroma collections built by Embed/embed.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_DIR = str(PROJECT_ROOT / "db")
EMBEDDING_MODEL = "text-embedding-3-small"

SEMANTIC_INDEX = "semantic_index"
RECURSIVE_INDEX = "recursive_index"

load_dotenv(PROJECT_ROOT / ".env")
_client = OpenAI()
_chroma: Optional[chromadb.PersistentClient] = None


def _get_chroma() -> chromadb.PersistentClient:
    """Same persistence backend as Embed/embed.py (PersistentClient on ./db)."""
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _chroma


def embed_query(text: str) -> List[float]:
    text = text.strip()
    if not text:
        raise ValueError("Query text must be non-empty")
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return list(response.data[0].embedding)


def _hit_fusion_key(h: Mapping[str, Any]) -> str:
    """Distinct across indexes: chunk_ids repeat between semantic/recursive splits."""
    strat = (h.get("metadata") or {}).get("chunk_strategy") or "unknown"
    return f'{strat}:{h["chunk_id"]}'


RetrievalHit = Dict[str, Any]


def retrieve(
    query: str,
    *,
    collection_name: str = SEMANTIC_INDEX,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> List[RetrievalHit]:
    """
    Run embedding-model similarity search against a collection.

    Uses the same model as indexing (text-embedding-3-small); distances are
    Chroma-space (lower is more similar for cosine-configured embeddings).
    """
    q_emb = embed_query(query)
    col = _get_chroma().get_collection(name=collection_name)
    raw = col.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        where=where,
        where_document=where_document,
    )

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    hits: List[RetrievalHit] = []
    for i, cid in enumerate(ids):
        hit: RetrievalHit = {
            "chunk_id": cid,
            "document": docs[i] if docs else "",
            "metadata": dict(metas[i]) if metas and metas[i] else {},
        }
        if dists:
            hit["distance"] = float(dists[i])
        hits.append(hit)
    return hits


def retrieve_fused(
    query: str,
    *,
    n_results_semantic: int = 8,
    n_results_recursive: int = 8,
    max_results: int = 10,
) -> List[RetrievalHit]:
    """
    Query semantic and recursive indexes; fuse with RRF keyed by chunk_strategy +
    chunk_id (IDs overlap across strategies but text differs).

    Fusion is reciprocal-rank fusion over each ranked list (k=60).
    """
    sem = retrieve(query, collection_name=SEMANTIC_INDEX, n_results=n_results_semantic)
    rec = retrieve(query, collection_name=RECURSIVE_INDEX, n_results=n_results_recursive)

    scores: Dict[str, float] = {}
    by_key: Dict[str, RetrievalHit] = {}

    k = 60.0
    for rank, h in enumerate(sem, start=1):
        key = _hit_fusion_key(h)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in by_key:
            by_key[key] = h

    for rank, h in enumerate(rec, start=1):
        key = _hit_fusion_key(h)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in by_key:
            by_key[key] = h

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    out = [by_key[key] for key in ranked_keys[:max_results]]
    for h in out:
        h["fusion_score"] = scores[_hit_fusion_key(h)]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Query SEC 10-K chunk indexes.")
    parser.add_argument("query", help="Natural language question")
    parser.add_argument(
        "--collection",
        choices=("semantic", "recursive", "fused"),
        default="semantic",
        help="Index to query (default: semantic)",
    )
    parser.add_argument("-n", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    if args.collection == "fused":
        hits = retrieve_fused(args.query, max_results=max(args.n * 2, args.n))
        hits = hits[: args.n]
    elif args.collection == "semantic":
        hits = retrieve(args.query, collection_name=SEMANTIC_INDEX, n_results=args.n)
    else:
        hits = retrieve(args.query, collection_name=RECURSIVE_INDEX, n_results=args.n)

    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        src = meta.get("source_file", "?")
        dist = h.get("distance")
        fusion = h.get("fusion_score")
        prefix = f"{i}. {h['chunk_id']} | {src}"
        if dist is not None:
            prefix += f" | distance={dist:.4f}"
        elif fusion is not None:
            prefix += f" | fusion={fusion:.4f}"
        print(prefix)
        print((h.get("document") or "")[:400])
        print()


if __name__ == "__main__":
    main()
