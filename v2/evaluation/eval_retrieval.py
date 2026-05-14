"""
v2 Retrieval Evaluation using LlamaIndex's RetrieverEvaluator.

Metrics (built into LlamaIndex):
  - HitRate@k  : fraction of queries where any gold chunk appears in top-k
                 (equivalent to v1's Recall@k)
  - MRR@k      : Mean Reciprocal Rank of first gold chunk hit

These are computed natively by RetrieverEvaluator — no custom metric code needed.
Latency is measured with time.perf_counter just like v1.

The same gold eval set is used:
  data/eval/gold_questions/apple_2025_10k_gold_eval_50_chunked_minimal.jsonl

Gold chunk IDs must exist in the v2 index for a fair comparison.
Since v2 node IDs are assigned as "<source>::<part>::<item>::<chunk_index>",
they differ from v1 IDs.  The eval therefore matches on text overlap:
a retrieved node is a "hit" if it covers the same text span as any gold chunk.

Algorithm (text-overlap matching)
----------------------------------
For each question:
  1. Retrieve top-k nodes from the v2 retriever.
  2. Load the gold answer chunk text from v1's semantic_chunks.jsonl.
  3. A hit is counted when the gold text appears verbatim (or as a substring)
     in any retrieved node's text.
This is fairer than ID matching across pipeline versions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

GOLD_PATH    = _PROJECT_ROOT / "data" / "eval" / "gold_questions" / \
               "apple_2025_10k_gold_eval_50_chunked_minimal.jsonl"
CHUNKS_V1    = _PROJECT_ROOT / "data" / "chunks" / "semantic_chunks.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gold(path: Path = GOLD_PATH) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _load_chunk_texts(path: Path = CHUNKS_V1) -> Dict[str, str]:
    """Map v1 chunk_id -> text."""
    mapping: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                c = json.loads(s)
                mapping[c["chunk_id"]] = c["text"]
    return mapping


def _hit_by_text(
    retrieved_texts: List[str],
    gold_texts: List[str],
    top_k: int,
) -> bool:
    """
    Check whether any of the gold texts is a substring of any retrieved text
    in the top-k window (using first 200 chars as a fingerprint for speed).
    """
    fingerprints = [t[:200].strip() for t in gold_texts if t]
    for ret_text in retrieved_texts[:top_k]:
        snippet = ret_text[:400]
        for fp in fingerprints:
            if fp[:100] in snippet:
                return True
    return False


def _reciprocal_rank(
    retrieved_texts: List[str],
    gold_texts: List[str],
) -> float:
    fingerprints = [t[:200].strip() for t in gold_texts if t]
    for rank, ret_text in enumerate(retrieved_texts, start=1):
        snippet = ret_text[:400]
        for fp in fingerprints:
            if fp[:100] in snippet:
                return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_v2_retrieval_eval(
    retriever: BaseRetriever,
    strategy_name: str,
    top_k_list: List[int] = (1, 5, 10),
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a v2 retriever against the gold question set.

    Returns a summary dict compatible with v1's retrieval_summary.json schema
    so both can be loaded side-by-side in the comparison notebook.
    """
    gold   = _load_gold()
    id_to_text = _load_chunk_texts()

    if limit:
        gold = gold[:limit]

    n = len(gold)
    hits_at: Dict[int, int] = {k: 0 for k in top_k_list}
    rr_list: List[float] = []
    latencies: List[float] = []
    per_question = []

    max_k = max(top_k_list)

    for row in gold:
        question = row["question"]
        gold_ids_all: List[str] = []
        for key in ("gold_chunk_ids", "primary_gold_chunk_ids", "supporting_gold_chunk_ids"):
            gold_ids_all += [c for c in (row.get(key) or []) if c]
        gold_ids_all = list(dict.fromkeys(gold_ids_all))  # deduplicate, preserve order

        gold_texts = [id_to_text.get(cid, "") for cid in gold_ids_all if cid]

        # Retrieve
        t0 = time.perf_counter()
        nodes = retriever.retrieve(question)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        retrieved_texts = [ns.node.text for ns in nodes]

        for k in top_k_list:
            if _hit_by_text(retrieved_texts, gold_texts, k):
                hits_at[k] += 1

        rr = _reciprocal_rank(retrieved_texts, gold_texts)
        rr_list.append(rr)

        per_question.append({
            "question_id": row.get("question_id", ""),
            "question":    question,
            "hit_ids":     [ns.node_id for ns in nodes],
            "gold_ids":    gold_ids_all,
            "rr":          rr,
            "latency_ms":  round(latency_ms, 1),
        })

    summary = {
        "strategy":       strategy_name,
        "version":        "v2",
        "n_questions":    n,
        "mean_latency_ms": round(sum(latencies) / n, 1),
        "mrr":             round(sum(rr_list) / n, 4),
    }
    for k in top_k_list:
        summary[f"recall@{k}"] = round(hits_at[k] / n, 4)

    return {"summary": summary, "per_question": per_question}
