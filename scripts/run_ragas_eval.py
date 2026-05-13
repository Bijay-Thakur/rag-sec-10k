#!/usr/bin/env python3
"""
End-to-end RAG evaluation using RAGAS 0.4.x on the Apple 2025 gold Q&A set.

Pipeline per question:
  1. Retrieve top-5 chunks (hybrid_rerank strategy, pre-embedded batch)
  2. Generate answer via GPT-4o-mini
  3. Score (question, answer, contexts, ground_truth) with RAGAS

RAGAS metrics:
  faithfulness      -- claims in answer supported by retrieved context?
  answer_relevancy  -- does the answer address the question?
  context_recall    -- was all necessary info retrieved?
  context_precision -- is the retrieved context relevant?

Usage (repo root, venv activated):
    $env:PYTHONPATH = "src"
    python scripts/run_ragas_eval.py [--limit N]

Output:
    data/eval/ragas_results.json    per-question scores
    data/eval/ragas_summary.json    mean scores
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT      = Path(__file__).resolve().parents[1]
SRC       = ROOT / "src"
GOLD      = ROOT / "data" / "eval" / "gold_questions" / "apple_2025_10k_gold_eval_50_chunked_minimal.jsonl"
CHUNKS    = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"
COLL      = "semantic_index"
OUT_FULL  = ROOT / "data" / "eval" / "ragas_results.json"
OUT_SUM   = ROOT / "data" / "eval" / "ragas_summary.json"

RETRIEVE_K  = 5
RERANK_POOL = 20
GEN_MODEL   = "gpt-4o-mini"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def chunk_text(c: Dict) -> str:
    return (c.get("text") or c.get("document") or "").strip()


def hybrid_search_vec(collection, bm25_retriever, q_vec, query_text, top_k, k_rrf=60):
    """RRF fusion with pre-computed query vector."""
    sem = collection.query(query_embeddings=[q_vec], n_results=top_k)
    sem_hits  = [{"id": sem["ids"][0][i], "text": sem["documents"][0][i],
                  "metadata": sem["metadatas"][0][i]}
                 for i in range(len(sem["ids"][0]))]
    bm25_hits = bm25_retriever.search(query_text, top_k=top_k)

    sem_ranks  = {h["id"]: r for r, h in enumerate(sem_hits,  1)}
    bm25_ranks = {h["id"]: r for r, h in enumerate(bm25_hits, 1)}
    all_ids    = set(sem_ranks) | set(bm25_ranks)
    fused = {d: 1/(k_rrf + sem_ranks.get(d, 9999)) + 1/(k_rrf + bm25_ranks.get(d, 9999))
             for d in all_ids}
    id_map  = {c["chunk_id"]: c for c in bm25_retriever.chunks}
    ranked  = sorted(fused, key=fused.__getitem__, reverse=True)
    return [{"id": d, "text": id_map[d]["text"], "metadata": id_map[d]["metadata"],
             "rrf_score": fused[d]}
            for d in ranked if d in id_map][:top_k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N questions (default: all 50)")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing.", file=sys.stderr)
        return 1

    sys.path.insert(0, str(SRC))

    from openai import OpenAI
    from retrieval.retriever import BM25Retriever, get_collection, rerank
    from generation.generator import generate_answer

    # ── load ────────────────────────────────────────────────────────────────
    print("Loading data ...")
    gold_rows  = load_jsonl(GOLD)
    if args.limit:
        gold_rows = gold_rows[: args.limit]
        print(f"  Limited to first {args.limit} questions")
    all_chunks = load_jsonl(CHUNKS)
    bm25_rows  = [{"chunk_id": c["chunk_id"], "text": c["text"], "metadata": c["metadata"]}
                  for c in all_chunks]
    print(f"  {len(gold_rows)} questions | {len(all_chunks)} chunks")

    collection     = get_collection(COLL)
    bm25_retriever = BM25Retriever(bm25_rows)
    print(f"  Chroma '{COLL}': {collection.count()} vectors")

    # ── pre-embed queries ────────────────────────────────────────────────────
    print("\nPre-embedding queries ...")
    oai    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    qs     = [r["question"] for r in gold_rows]
    resp   = oai.embeddings.create(model="text-embedding-3-small", input=qs)
    q_vecs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    print(f"  {len(q_vecs)} vectors ready")

    # ── warm cross-encoder ───────────────────────────────────────────────────
    print("Warming cross-encoder ...")
    rerank("warmup", [{"id": "x", "text": "warmup"}], top_k=1)
    print("  Ready")

    # ── retrieve + generate ──────────────────────────────────────────────────
    print(f"\nRetrieve + Generate ({len(gold_rows)} questions, model={GEN_MODEL}) ...")
    records: List[Dict[str, Any]] = []

    for i, (row, q_vec) in enumerate(zip(gold_rows, q_vecs)):
        q = row["question"]

        candidates = hybrid_search_vec(collection, bm25_retriever, q_vec, q, RERANK_POOL)
        hits       = rerank(q, candidates, top_k=RETRIEVE_K)
        result     = generate_answer(q, hits, model=GEN_MODEL)

        records.append({
            "question_id":        row["question_id"],
            "question":           q,
            "answer":             result.answer,
            "contexts":           [chunk_text(c) for c in hits],
            "ground_truth":       row.get("expected_answer", ""),
            "latency_gen_ms":     result.latency_ms,
            "prompt_tokens":      result.prompt_tokens,
            "completion_tokens":  result.completion_tokens,
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(gold_rows):
            print(f"  {i+1}/{len(gold_rows)} done")

    # ── RAGAS evaluation ─────────────────────────────────────────────────────
    print("\nRunning RAGAS evaluation ...")

    # ragas 0.4.x: use the legacy underscore-prefixed Metric classes — these properly
    # inherit from ragas.metrics.Metric (which evaluate() type-checks for)
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        _Faithfulness,
        _AnswerRelevancy,
        _ContextRecall,
        _ContextPrecision,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    langchain_llm = LangchainLLMWrapper(ChatOpenAI(model=GEN_MODEL, temperature=0))
    langchain_emb = LangchainEmbeddingsWrapper(
                        OpenAIEmbeddings(model="text-embedding-3-small"))

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in records],
        "answer":       [r["answer"]       for r in records],
        "contexts":     [r["contexts"]     for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    metrics = [
        _Faithfulness(llm=langchain_llm),
        _AnswerRelevancy(llm=langchain_llm, embeddings=langchain_emb),
        _ContextRecall(llm=langchain_llm),
        _ContextPrecision(llm=langchain_llm),
    ]

    from ragas import RunConfig
    run_config = RunConfig(
        timeout    = 180,      # seconds per LLM call
        max_retries = 3,
        max_workers = 2,       # sequential-ish to avoid rate limits
        max_wait   = 60,
    )
    ragas_result = ragas_evaluate(dataset, metrics=metrics, run_config=run_config)

    scores_df   = ragas_result.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    per_question = []
    for i, rec in enumerate(records):
        row_scores = {}
        for col in metric_cols:
            if col in scores_df.columns:
                val = scores_df.iloc[i][col]
                row_scores[col] = None if val != val else round(float(val), 4)
        per_question.append({**rec, "ragas": row_scores})

    summary = {
        col: round(float(scores_df[col].dropna().mean()), 4)
        for col in metric_cols if col in scores_df.columns
    }
    print("  RAGAS complete")

    # ── save ─────────────────────────────────────────────────────────────────
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FULL, "w", encoding="utf-8") as f:
        json.dump(per_question, f, indent=2)
    with open(OUT_SUM, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RAGAS Summary ===")
    for k, v in summary.items():
        print(f"  {k:25s}  {v}")

    print(f"\nSaved to:\n  {OUT_FULL}\n  {OUT_SUM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
