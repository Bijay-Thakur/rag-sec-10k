#!/usr/bin/env python3
"""
Compare semantic (Chroma + OpenAI), BM25, and hybrid (RRF) retrievers from retrieval/retriever.py.

PowerShell (repo root):

  $env:PYTHONPATH = "src"
  $env:OPENAI_API_KEY = "sk-..."
  python scripts/benchmark_retrieval_strategies.py

  python scripts/benchmark_retrieval_strategies.py --collection semantic_index --top-k 10

Prerequisites:
  - data/chunks/semantic_chunks.jsonl (chunk_id, text, metadata)
  - Chroma populated for that collection (PYTHONPATH=src python src/Embed/embed.py)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CHUNK_PATH = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"


def load_jsonl_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def to_bm25_rows(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"chunk_id": c["chunk_id"], "text": c["text"], "metadata": dict(c.get("metadata") or {})}
        for c in chunks
    ]


def synthetic_queries(
    chunks: Sequence[Dict[str, Any]],
    *,
    n: int,
    seed: int,
    min_words: int = 10,
    max_words: int = 18,
    skip_lead: int = 40,
) -> List[Tuple[str, str]]:
    """
    (query, gold_chunk_id): query is a consecutive word span from the chunk body
    so lexical match is possible; semantic should also align if embeddings track the corpus.
    """
    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    usable = [c for c in chunks if len(c.get("text") or "") > skip_lead + 80]
    rng.shuffle(usable)
    for c in usable[:n]:
        text = (c["text"] or "")[skip_lead:]
        words = text.split()
        if len(words) < max_words + 5:
            continue
        start = rng.randint(0, max(0, len(words) - max_words - 1))
        span = rng.randint(min_words, max_words)
        qwords = words[start : start + span]
        q = " ".join(qwords).strip()
        if len(q) < 20:
            continue
        out.append((q, c["chunk_id"]))
    return out


def fixed_queries() -> List[str]:
    return [
        "What products or services does the company describe?",
        "revenue and financial performance",
        "What are the main risk factors discussed?",
        "AppleCare service and support offerings",
        "reportable geographic segments Americas Europe China",
        "stock-based compensation expense",
        "debt covenants and credit facilities",
        "cybersecurity and data privacy risks",
    ]


def top_ids(hits: Sequence[Dict[str, Any]], k: int) -> List[str]:
    return [h["id"] for h in hits[:k]]


def recall_at_k(gold: str, ranked_ids: Sequence[str], k: int) -> float:
    return 1.0 if gold in set(ranked_ids[:k]) else 0.0


def reciprocal_rank(gold: str, ranked_ids: Sequence[str]) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == gold:
            return 1.0 / i
    return 0.0


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def time_call(fn, *, repeats: int, warmup: int) -> Tuple[Any, float]:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn()
        samples.append(time.perf_counter() - t0)
    return r, statistics.mean(samples) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="semantic_index", help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--synthetic-n", type=int, default=45, help="Number of synthetic span queries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-repeats", type=int, default=2, help="Per-query repeats for latency (semantic is costly)")
    parser.add_argument("--timing-warmup", type=int, default=0)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY required (env or .env).", file=sys.stderr)
        return 1

    if not CHUNK_PATH.is_file():
        print(f"Missing {CHUNK_PATH}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(SRC))
    from retrieval.retriever import get_retriever

    chunks = load_jsonl_chunks(CHUNK_PATH)
    rows = to_bm25_rows(chunks)

    sem = get_retriever(args.collection, rows, "semantic")
    bm = get_retriever(args.collection, rows, "bm25")
    hy = get_retriever(args.collection, rows, "hybrid")

    k = args.top_k
    syn = synthetic_queries(chunks, n=args.synthetic_n, seed=args.seed)

    print("=== Synthetic in-corpus span queries ===")
    print(f"Queries: {len(syn)} | top_k={k} | collection={args.collection!r}\n")

    sem_r5: List[float] = []
    sem_r10: List[float] = []
    sem_mrr: List[float] = []
    bm_r5: List[float] = []
    bm_r10: List[float] = []
    bm_mrr: List[float] = []
    hy_r5: List[float] = []
    hy_r10: List[float] = []
    hy_mrr: List[float] = []

    lat_sem: List[float] = []
    lat_bm: List[float] = []
    lat_hy: List[float] = []

    for q, gold in syn:

        def _s():
            return sem(q, top_k=k)

        def _b():
            return bm(q, top_k=k)

        def _h():
            return hy(q, top_k=k)

        hs, ms = time_call(_s, repeats=args.timing_repeats, warmup=args.timing_warmup)
        hb, mb = time_call(_b, repeats=max(1, args.timing_repeats), warmup=0)
        hh, mh = time_call(_h, repeats=args.timing_repeats, warmup=args.timing_warmup)
        lat_sem.append(ms)
        lat_bm.append(mb)
        lat_hy.append(mh)

        ids_s = top_ids(hs, k)
        ids_b = top_ids(hb, k)
        ids_h = top_ids(hh, k)

        sem_r5.append(recall_at_k(gold, ids_s, 5))
        sem_r10.append(recall_at_k(gold, ids_s, 10))
        sem_mrr.append(reciprocal_rank(gold, ids_s))

        bm_r5.append(recall_at_k(gold, ids_b, 5))
        bm_r10.append(recall_at_k(gold, ids_b, 10))
        bm_mrr.append(reciprocal_rank(gold, ids_b))

        hy_r5.append(recall_at_k(gold, ids_h, 5))
        hy_r10.append(recall_at_k(gold, ids_h, 10))
        hy_mrr.append(reciprocal_rank(gold, ids_h))

    def summarize(name: str, r5, r10, mrr, lat) -> None:
        print(
            f"{name:10}  R@5={statistics.mean(r5):.3f}  R@10={statistics.mean(r10):.3f}  "
            f"MRR={statistics.mean(mrr):.3f}  latency≈{statistics.mean(lat):.1f} ms/query"
        )

    summarize("semantic", sem_r5, sem_r10, sem_mrr, lat_sem)
    summarize("bm25", bm_r5, bm_r10, bm_mrr, lat_bm)
    summarize("hybrid", hy_r5, hy_r10, hy_mrr, lat_hy)

    print("\n=== Fixed natural-language queries (rank overlap, top-5) ===\n")
    jq_sem_bm: List[float] = []
    jq_sem_hy: List[float] = []
    jq_bm_hy: List[float] = []

    for q in fixed_queries():
        s_ids = set(top_ids(sem(q, top_k=5), 5))
        b_ids = set(top_ids(bm(q, top_k=5), 5))
        h_ids = set(top_ids(hy(q, top_k=5), 5))
        jq_sem_bm.append(jaccard(s_ids, b_ids))
        jq_sem_hy.append(jaccard(s_ids, h_ids))
        jq_bm_hy.append(jaccard(b_ids, h_ids))

    print(f"Mean Jaccard(semantic, BM25):   {statistics.mean(jq_sem_bm):.3f}")
    print(f"Mean Jaccard(semantic, hybrid): {statistics.mean(jq_sem_hy):.3f}")
    print(f"Mean Jaccard(BM25, hybrid):     {statistics.mean(jq_bm_hy):.3f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
