#!/usr/bin/env python3
"""
Smoke-test OpenAI embeddings + Chroma retrieval before wiring a UI.

From the repository root (PowerShell):

  $env:PYTHONPATH = "src"
  $env:OPENAI_API_KEY = "sk-..."   # or rely on .env via python-dotenv
  python scripts/verify_retrieval_stack.py

  python scripts/verify_retrieval_stack.py --query "What are the company's main products?"

Prerequisites:
  - data/chunks/semantic_chunks.jsonl and recursive_chunks.jsonl (run chunkers first)
  - Chroma DB populated: PYTHONPATH=src python src/Embed/embed.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CHUNK_SEM = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"
CHUNK_REC = ROOT / "data" / "chunks" / "recursive_chunks.jsonl"
CHROMA_DIR = ROOT / "db"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify OpenAI embed + Chroma retrieval.")
    parser.add_argument(
        "--query",
        default="What are the main risk factors discussed?",
        help="Natural-language query to run against the indexes.",
    )
    args = parser.parse_args()

    _load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in the environment or in a .env file at the repo root.", file=sys.stderr)
        return 1

    if not CHUNK_SEM.is_file() or not CHUNK_REC.is_file():
        print(f"Missing chunk files. Expected:\n  {CHUNK_SEM}\n  {CHUNK_REC}", file=sys.stderr)
        print("Generate them first (e.g. run chunkers on data/raw/*.html).", file=sys.stderr)
        return 1

    sys.path.insert(0, str(SRC))

    import chromadb
    from openai import OpenAI

    from retrieval.retrieve import (
        EMBEDDING_MODEL,
        RECURSIVE_INDEX,
        SEMANTIC_INDEX,
        retrieve,
        retrieve_fused,
    )

    # 1) OpenAI embedding API (same model as indexing)
    print(f"1) OpenAI embedding model: {EMBEDDING_MODEL}")
    probe = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input="health check")
    vec = probe.data[0].embedding
    print(f"   OK — single-vector dimension: {len(vec)}")

    # 2) Chroma collections exist and are non-empty (PersistentClient matches embed.py)
    print("2) Chroma collections")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        n_sem = client.get_collection(SEMANTIC_INDEX).count()
        n_rec = client.get_collection(RECURSIVE_INDEX).count()
    except Exception as e:
        print(
            f"   Failed to open collections under {CHROMA_DIR}: {e}",
            file=sys.stderr,
        )
        print(
            "   Build indexes first:\n"
            f"     $env:PYTHONPATH = \"src\"\n"
            f"     python src/Embed/embed.py",
            file=sys.stderr,
        )
        return 1
    print(f"   semantic_index: {n_sem} rows")
    print(f"   recursive_index: {n_rec} rows")
    if n_sem == 0 or n_rec == 0:
        print("   One or both indexes are empty — run embed.py after chunking.", file=sys.stderr)
        return 1

    # 3) Retrieval
    print(f"3) Retrieval (semantic, top 3) for: {args.query!r}")
    hits = retrieve(args.query, n_results=3)
    for i, h in enumerate(hits, 1):
        dist = h.get("distance")
        meta = h.get("metadata") or {}
        print(f"   {i}. {h.get('chunk_id')} | {meta.get('source_file')} | distance={dist}")
        snippet = (h.get("document") or "")[:220].replace("\n", " ")
        print(f"      {snippet}…")

    print("4) Fused retrieval (top 3)")
    fused = retrieve_fused(args.query, max_results=3)
    for i, h in enumerate(fused, 1):
        print(
            f"   {i}. {h.get('chunk_id')} | fusion={h.get('fusion_score'):.4f} | "
            f"{(h.get('metadata') or {}).get('chunk_strategy')}"
        )

    print("\nAll checks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
