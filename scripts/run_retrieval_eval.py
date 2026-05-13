#!/usr/bin/env python3
"""
Evaluate all retrieval strategies against the gold Q&A set.

Optimised approach
------------------
All 50 query embeddings are computed in ONE batch API call upfront.
Subsequent lookups (semantic / hybrid / rerank) reuse those vectors,
so only ~1 OpenAI call is made regardless of how many strategies run.

Usage (repo root, venv activated):
    $env:PYTHONPATH = "src"
    python scripts/run_retrieval_eval.py

Output
------
    data/eval/retrieval_results.json   per-question hit lists for every strategy
    data/eval/retrieval_summary.json   aggregated metric table
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"

GOLD_PATH      = ROOT / "data" / "eval" / "gold_questions" / "apple_2025_10k_gold_eval_50_chunked_minimal.jsonl"
CHUNK_PATH     = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"
COLLECTION     = "semantic_index"
RESULTS_PATH   = ROOT / "data" / "eval" / "retrieval_results.json"
SUMMARY_PATH   = ROOT / "data" / "eval" / "retrieval_summary.json"

TOP_K          = 10   # evaluate R@1 / R@5 / R@10
RERANK_POOL    = 20   # hybrid candidates fed to cross-encoder


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_gold(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def all_gold_ids(row: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for key in ("gold_chunk_ids", "primary_gold_chunk_ids", "supporting_gold_chunk_ids"):
        ids.update(row.get(key) or [])
    return ids


def primary_gold_ids(row: Dict[str, Any]) -> Set[str]:
    return set(row.get("primary_gold_chunk_ids") or [])


def recall_at(hit_ids: List[str], gold: Set[str], k: int) -> float:
    return 1.0 if any(h in gold for h in hit_ids[:k]) else 0.0


def mrr_score(hit_ids: List[str], gold: Set[str]) -> float:
    for rank, h in enumerate(hit_ids, start=1):
        if h in gold:
            return 1.0 / rank
    return 0.0


def summarise(name: str, rows, hit_lists, latencies) -> Dict[str, Any]:
    r1, r5, r10, mrrs = [], [], [], []
    pr1, pr5, pr10, pmrrs = [], [], [], []
    for row, hits in zip(rows, hit_lists):
        g  = all_gold_ids(row)
        pg = primary_gold_ids(row)
        r1.append(recall_at(hits, g, 1));  r5.append(recall_at(hits, g, 5));  r10.append(recall_at(hits, g, 10))
        mrrs.append(mrr_score(hits, g))
        pr1.append(recall_at(hits, pg, 1)); pr5.append(recall_at(hits, pg, 5)); pr10.append(recall_at(hits, pg, 10))
        pmrrs.append(mrr_score(hits, pg))
    avg = lambda xs: round(statistics.mean(xs), 4) if xs else 0.0
    return {
        "strategy":          name,
        "n_questions":       len(rows),
        "recall@1":          avg(r1),   "recall@5":          avg(r5),   "recall@10":          avg(r10),   "mrr":          avg(mrrs),
        "primary_recall@1":  avg(pr1),  "primary_recall@5":  avg(pr5),  "primary_recall@10":  avg(pr10),  "primary_mrr":  avg(pmrrs),
        "mean_latency_ms":   avg(latencies),
    }


def print_table(summaries: List[Dict[str, Any]]) -> None:
    cols = [
        ("Strategy",      "strategy",           16),
        ("R@1",           "recall@1",             6),
        ("R@5",           "recall@5",             6),
        ("R@10",          "recall@10",            7),
        ("MRR",           "mrr",                  6),
        ("pR@1",          "primary_recall@1",     6),
        ("pR@5",          "primary_recall@5",     6),
        ("pR@10",         "primary_recall@10",    7),
        ("pMRR",          "primary_mrr",          6),
        ("ms/q",          "mean_latency_ms",      8),
    ]
    hdr = "  ".join(f"{lbl:<{w}}" for lbl, _, w in cols)
    sep = "  ".join("-" * w for _, _, w in cols)
    print("\n" + hdr)
    print(sep)
    for s in summaries:
        print("  ".join(f"{str(s[k]):<{w}}" for _, k, w in cols))
    print()
    print("R@k  = any gold chunk (primary + supporting) in top-k")
    print("pR@k = primary gold chunk only in top-k")


# ---------------------------------------------------------------------------
# Semantic search using a pre-computed query vector
# ---------------------------------------------------------------------------

def semantic_search_vec(collection, q_vec: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Query Chroma with an already-computed embedding vector."""
    res  = collection.query(query_embeddings=[q_vec], n_results=top_k)
    ids  = res["ids"][0]
    docs = res["documents"][0]
    metas= res["metadatas"][0]
    dists= res["distances"][0]
    return [{"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]}
            for i in range(len(ids))]


def hybrid_search_vec(collection, bm25_retriever, q_vec, query_text, top_k, k_rrf=60):
    """RRF fusion with a pre-computed semantic vector (no extra API call)."""
    sem_hits  = semantic_search_vec(collection, q_vec, top_k)
    bm25_hits = bm25_retriever.search(query_text, top_k=top_k)

    sem_ranks  = {h["id"]: rank for rank, h in enumerate(sem_hits,  start=1)}
    bm25_ranks = {h["id"]: rank for rank, h in enumerate(bm25_hits, start=1)}
    all_ids    = set(sem_ranks) | set(bm25_ranks)

    fused = {
        doc_id: (1.0 / (k_rrf + sem_ranks.get(doc_id,  9999)) +
                 1.0 / (k_rrf + bm25_ranks.get(doc_id, 9999)))
        for doc_id in all_ids
    }
    id_to_chunk = {c["chunk_id"]: c for c in bm25_retriever.chunks}
    sorted_ids  = sorted(fused, key=fused.__getitem__, reverse=True)

    return [
        {"id": doc_id, "text": id_to_chunk[doc_id]["text"],
         "metadata": id_to_chunk[doc_id]["metadata"], "rrf_score": fused[doc_id]}
        for doc_id in sorted_ids[:top_k]
        if doc_id in id_to_chunk
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing.", file=sys.stderr)
        return 1
    for p in (GOLD_PATH, CHUNK_PATH):
        if not p.is_file():
            print(f"Not found: {p}", file=sys.stderr)
            return 1

    sys.path.insert(0, str(SRC))
    from openai import OpenAI
    from retrieval.retriever import BM25Retriever, get_collection, rerank

    # -- load data ----------------------------------------------------------
    print(f"Loading gold eval  ({GOLD_PATH.name}) ...")
    gold_rows = load_gold(GOLD_PATH)
    print(f"  {len(gold_rows)} questions")

    print(f"Loading chunks     ({CHUNK_PATH.name}) ...")
    all_chunks = load_chunks(CHUNK_PATH)
    bm25_rows  = [{"chunk_id": c["chunk_id"], "text": c["text"], "metadata": c["metadata"]}
                  for c in all_chunks]
    print(f"  {len(all_chunks)} chunks total")

    print(f"Opening Chroma     ({COLLECTION!r}) ...")
    try:
        collection = get_collection(COLLECTION)
        n = collection.count()
        print(f"  {n} vectors indexed")
        if n == 0:
            print("Collection empty — run: python src/Embed/embed.py --force", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Chroma error: {exc}", file=sys.stderr)
        return 1

    bm25_retriever = BM25Retriever(bm25_rows)

    # -- pre-compute ALL query embeddings in one batch API call -------------
    print("\nEmbedding all queries in one batch ...")
    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    questions   = [r["question"] for r in gold_rows]
    t0_batch    = time.perf_counter()
    resp        = oai.embeddings.create(model="text-embedding-3-small", input=questions)
    batch_ms    = (time.perf_counter() - t0_batch) * 1000
    q_vecs      = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    print(f"  Done — {len(q_vecs)} vectors in {batch_ms:.0f} ms total "
          f"({batch_ms/len(q_vecs):.1f} ms/query amortised)")

    # -- warm up cross-encoder (downloads model once, then cached) ----------
    print("\nLoading cross-encoder model (downloads once, ~90 MB) ...")
    t0_ce = time.perf_counter()
    _ = rerank("test", [{"id": "x", "text": "test"}], top_k=1)
    print(f"  Cross-encoder ready in {(time.perf_counter()-t0_ce)*1000:.0f} ms")

    # -- run each strategy --------------------------------------------------
    all_results: Dict[str, Any] = {}
    summaries:   List[Dict[str, Any]] = []

    STRATEGIES = ["semantic", "bm25", "hybrid", "hybrid_rerank"]

    for strategy in STRATEGIES:
        print(f"\n=== {strategy} ===")
        hit_lists:  List[List[str]] = []
        latencies:  List[float]     = []
        per_q:      List[Dict]      = []

        for i, (row, q_vec) in enumerate(zip(gold_rows, q_vecs)):
            q = row["question"]
            t0 = time.perf_counter()

            if strategy == "semantic":
                hits = semantic_search_vec(collection, q_vec, TOP_K)

            elif strategy == "bm25":
                hits = bm25_retriever.search(q, top_k=TOP_K)

            elif strategy == "hybrid":
                hits = hybrid_search_vec(collection, bm25_retriever,
                                         q_vec, q, TOP_K)

            elif strategy == "hybrid_rerank":
                candidates = hybrid_search_vec(collection, bm25_retriever,
                                               q_vec, q, RERANK_POOL)
                hits = rerank(q, candidates, top_k=TOP_K)

            elapsed = (time.perf_counter() - t0) * 1000
            hit_ids = [h["id"] for h in hits]
            hit_lists.append(hit_ids)
            latencies.append(elapsed)
            per_q.append({
                "question_id": row["question_id"],
                "question":    q,
                "hit_ids":     hit_ids,
                "gold_ids":    sorted(all_gold_ids(row)),
                "primary_ids": sorted(primary_gold_ids(row)),
                "latency_ms":  round(elapsed, 2),
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(gold_rows):
                print(f"  {i+1}/{len(gold_rows)} done")

        summary = summarise(strategy, gold_rows, hit_lists, latencies)
        summaries.append(summary)
        all_results[strategy] = {"summary": summary, "per_question": per_q}
        print(f"  R@5={summary['recall@5']}  R@10={summary['recall@10']}  "
              f"MRR={summary['mrr']}  pR@5={summary['primary_recall@5']}  "
              f"lat={summary['mean_latency_ms']:.1f} ms/q")

        # incremental save after each strategy
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # -- final summary ------------------------------------------------------
    print_table(summaries)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nResults saved to:\n  {RESULTS_PATH}\n  {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
