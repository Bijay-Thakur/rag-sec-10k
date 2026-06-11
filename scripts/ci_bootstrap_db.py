#!/usr/bin/env python3
"""
Bootstrap a minimal ChromaDB index for CI — no OpenAI calls.

Uses dummy embeddings so demo_mode (BM25) and index_is_ready() work offline.
Loads all chunks from data/chunks/semantic_chunks.jsonl (committed to git).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"
DB_PATH = ROOT / "db"
COLLECTION = "semantic_index"
EMBED_DIM = 1536
BATCH = 128


def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            out[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def load_chunks() -> List[dict]:
    rows: List[dict] = []
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    if not CHUNKS_PATH.is_file():
        print(f"Missing chunks file: {CHUNKS_PATH}", file=sys.stderr)
        return 1

    import chromadb

    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = load_chunks()
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        docs.append(chunk["text"])
        metas.append(normalize_metadata(chunk.get("metadata", {})))

    dummy = [0.0] * EMBED_DIM
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=[dummy] * (end - start),
        )

    print(f"CI bootstrap: indexed {len(ids)} chunks into {COLLECTION} at {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
