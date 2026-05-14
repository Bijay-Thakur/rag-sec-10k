"""
v2 Retrieval Evaluation — all strategies via LlamaIndex retrievers.

Optimization: all 50 query embeddings are computed in ONE batch API call
(same approach as v1's run_retrieval_eval.py), so the script runs in ~60s
regardless of how many strategies are evaluated.

BM25 needs no API calls at all.

Writes:
  data/eval/v2_retrieval_summary.json   (same schema as v1, for side-by-side comparison)
  data/eval/v2_retrieval_results.json   (per-question detail)

Usage
-----
    $env:PYTHONPATH = ".;src"
    python scripts/run_v2_retrieval_eval.py
    python scripts/run_v2_retrieval_eval.py --strategies bm25   # free, no API
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from llama_index.core.schema import TextNode
from v2.indexing  import load_v2_index, v2_index_ready
from v2.retrieval import get_v2_retriever

GOLD_PATH   = ROOT / "data/eval/gold_questions/apple_2025_10k_gold_eval_50_chunked_minimal.jsonl"
CHUNKS_PATH = ROOT / "data/chunks/semantic_chunks.jsonl"
EVAL_DIR    = ROOT / "data/eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_LIST     = [1, 5, 10]
DEFAULT_TOP_K  = 10
STRATEGIES_ALL = ["semantic", "bm25", "hybrid"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_gold() -> List[Dict[str, Any]]:
    rows = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _load_nodes() -> List[TextNode]:
    nodes = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            c = json.loads(s)
            nodes.append(TextNode(
                text=c["text"],
                id_=c["chunk_id"],
                metadata=c.get("metadata", {}),
            ))
    return nodes


def _gold_ids(row: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for key in ("gold_chunk_ids", "primary_gold_chunk_ids", "supporting_gold_chunk_ids"):
        ids.update(row.get(key) or [])
    return ids


def _primary_ids(row: Dict[str, Any]) -> Set[str]:
    return set(row.get("primary_gold_chunk_ids") or [])


# ---------------------------------------------------------------------------
# Batch embed all queries in ONE API call (same trick as v1)
# ---------------------------------------------------------------------------

def _batch_embed(questions: List[str]) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=questions,
    )
    data = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in data]


# ---------------------------------------------------------------------------
# Per-strategy evaluation
# ---------------------------------------------------------------------------

def _eval_bm25(
    gold: List[Dict],
    nodes: List[TextNode],
    top_k: int,
) -> Dict[str, Any]:
    """BM25 evaluation — no API calls needed."""
    from v2.retrieval.retrievers import BM25LlamaRetriever
    bm25 = BM25LlamaRetriever(nodes=nodes, top_k=top_k)

    per_q, latencies = [], []
    r_at  = {k: [] for k in TOP_K_LIST}
    pr_at = {k: [] for k in TOP_K_LIST}
    rr_all, rr_prim = [], []

    for row in gold:
        gids = _gold_ids(row)
        pids = _primary_ids(row)

        t0 = time.perf_counter()
        hits = bm25.retrieve(row["question"])
        lat  = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        hit_ids = [ns.node_id for ns in hits]

        for k in TOP_K_LIST:
            r_at[k].append(1.0 if any(h in gids for h in hit_ids[:k]) else 0.0)
            pr_at[k].append(1.0 if any(h in pids for h in hit_ids[:k]) else 0.0)

        rr  = next((1.0/(i+1) for i, h in enumerate(hit_ids) if h in gids), 0.0)
        prr = next((1.0/(i+1) for i, h in enumerate(hit_ids) if h in pids), 0.0)
        rr_all.append(rr); rr_prim.append(prr)

        per_q.append({
            "question_id": row.get("question_id", ""),
            "question": row["question"],
            "hit_ids": hit_ids,
            "gold_ids": list(gids),
            "rr": rr,
            "latency_ms": round(lat, 1),
        })

    n = len(gold)
    avg = lambda xs: round(sum(xs)/len(xs), 4) if xs else 0.0
    summary = {
        "strategy": "bm25", "version": "v2", "n_questions": n,
        "mean_latency_ms": avg(latencies), "mrr": avg(rr_all), "primary_mrr": avg(rr_prim),
    }
    for k in TOP_K_LIST:
        summary[f"recall@{k}"] = avg(r_at[k])
        summary[f"primary_recall@{k}"] = avg(pr_at[k])
    return {"summary": summary, "per_question": per_q}


def _eval_vector(
    strategy: str,
    gold: List[Dict],
    query_vecs: List[List[float]],
    nodes: List[TextNode],
    top_k: int,
    index,
) -> Dict[str, Any]:
    """Evaluate semantic or hybrid using pre-computed query embeddings."""
    import chromadb
    from rank_bm25 import BM25Okapi

    # Access Chroma directly for vector queries (avoids per-query embedding)
    chroma_client = chromadb.PersistentClient(path=str(ROOT / "db"))
    col = chroma_client.get_collection("semantic_index")

    # BM25 index for hybrid
    corpus = [n.text.split() for n in nodes]
    id_list = [n.node_id for n in nodes]
    bm25_engine = BM25Okapi(corpus)
    id_to_node = {n.node_id: n for n in nodes}

    per_q, latencies = [], []
    r_at  = {k: [] for k in TOP_K_LIST}
    pr_at = {k: [] for k in TOP_K_LIST}
    rr_all, rr_prim = [], []

    for row, q_vec in zip(gold, query_vecs):
        gids = _gold_ids(row)
        pids = _primary_ids(row)

        t0 = time.perf_counter()

        if strategy == "semantic":
            res = col.query(query_embeddings=[q_vec], n_results=top_k)
            hit_ids = res["ids"][0]

        elif strategy == "hybrid":
            n_pool = top_k
            # Semantic pool
            res = col.query(query_embeddings=[q_vec], n_results=n_pool)
            sem_ids = res["ids"][0]
            # BM25 pool
            tokens = row["question"].split()
            bm25_scores = bm25_engine.get_scores(tokens)
            bm25_ranked = sorted(range(len(bm25_scores)),
                                 key=lambda i: bm25_scores[i], reverse=True)[:n_pool]
            bm25_ids = [id_list[i] for i in bm25_ranked]
            # RRF fusion
            k_rrf = 60
            sem_ranks  = {d: r for r, d in enumerate(sem_ids,  1)}
            bm25_ranks = {d: r for r, d in enumerate(bm25_ids, 1)}
            all_ids = set(sem_ranks) | set(bm25_ranks)
            fused = {
                d: 1/(k_rrf + sem_ranks.get(d, 9999)) + 1/(k_rrf + bm25_ranks.get(d, 9999))
                for d in all_ids
            }
            hit_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]

        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        for k in TOP_K_LIST:
            r_at[k].append(1.0 if any(h in gids for h in hit_ids[:k]) else 0.0)
            pr_at[k].append(1.0 if any(h in pids for h in hit_ids[:k]) else 0.0)

        rr  = next((1.0/(i+1) for i, h in enumerate(hit_ids) if h in gids), 0.0)
        prr = next((1.0/(i+1) for i, h in enumerate(hit_ids) if h in pids), 0.0)
        rr_all.append(rr); rr_prim.append(prr)

        per_q.append({
            "question_id": row.get("question_id", ""),
            "question": row["question"],
            "hit_ids": list(hit_ids),
            "gold_ids": list(gids),
            "rr": rr,
            "latency_ms": round(lat, 1),
        })

    n = len(gold)
    avg = lambda xs: round(sum(xs)/len(xs), 4) if xs else 0.0
    summary = {
        "strategy": strategy, "version": "v2", "n_questions": n,
        "mean_latency_ms": avg(latencies), "mrr": avg(rr_all), "primary_mrr": avg(rr_prim),
    }
    for k in TOP_K_LIST:
        summary[f"recall@{k}"] = avg(r_at[k])
        summary[f"primary_recall@{k}"] = avg(pr_at[k])
    return {"summary": summary, "per_question": per_q}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="v2 retrieval evaluation.")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES_ALL,
                        choices=["semantic", "bm25", "hybrid"])
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not v2_index_ready():
        print("ERROR: v2 index not ready. Run `python scripts/run_v2_index.py` first.")
        sys.exit(1)

    gold  = _load_gold()
    if args.limit:
        gold = gold[:args.limit]
    print(f"[v2] {len(gold)} gold questions loaded.")

    print("[v2] Loading nodes …")
    nodes = _load_nodes()
    print(f"[v2] {len(nodes)} nodes loaded.")

    # Load index (needed for type-checking; actual queries go direct to Chroma)
    index = load_v2_index()

    # Pre-batch embeddings for vector strategies (1 API call total)
    needs_embeddings = any(s in ("semantic", "hybrid") for s in args.strategies)
    query_vecs: List[List[float]] = []
    if needs_embeddings:
        print(f"[v2] Embedding {len(gold)} queries in one batch …")
        t0 = time.perf_counter()
        query_vecs = _batch_embed([r["question"] for r in gold])
        print(f"[v2] Batch embed done in {(time.perf_counter()-t0)*1000:.0f} ms")

    all_summaries, all_per_q = [], {}

    for strategy in args.strategies:
        print(f"\n[v2] Evaluating: {strategy} …")
        if strategy == "bm25":
            result = _eval_bm25(gold, nodes, args.top_k)
        else:
            result = _eval_vector(strategy, gold, query_vecs, nodes, args.top_k, index)

        s = result["summary"]
        all_summaries.append(s)
        all_per_q[strategy] = result

        print(
            f"  R@1={s.get('recall@1',0):.3f}  "
            f"R@5={s.get('recall@5',0):.3f}  "
            f"R@10={s.get('recall@10',0):.3f}  "
            f"MRR={s['mrr']:.4f}  "
            f"lat={s['mean_latency_ms']:.1f} ms"
        )

    (EVAL_DIR / "v2_retrieval_summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    (EVAL_DIR / "v2_retrieval_results.json").write_text(
        json.dumps(all_per_q, indent=2), encoding="utf-8"
    )
    print(f"\n[v2] Results written to data/eval/v2_retrieval_*.json")


if __name__ == "__main__":
    main()
