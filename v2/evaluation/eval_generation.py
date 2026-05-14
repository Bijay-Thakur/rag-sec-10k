"""
v2 Generation Evaluation using LlamaIndex's built-in evaluators.

LlamaIndex evaluators vs RAGAS (used in v1):
---------------------------------------------------------------------------
| Metric                | v1 (RAGAS)               | v2 (LlamaIndex)           |
|-----------------------|--------------------------|---------------------------|
| Faithfulness          | ragas.Faithfulness       | FaithfulnessEvaluator     |
| Answer Relevancy      | ragas.AnswerRelevancy    | RelevancyEvaluator        |
| Context Recall        | ragas.ContextRecall      | (no direct equiv.; skip)  |
| Context Precision     | ragas.ContextPrecision   | (no direct equiv.; skip)  |
---------------------------------------------------------------------------

Both use LLM-as-judge with GPT-4o-mini as the judge.  The underlying
approach is comparable: the LLM is asked whether each claim in the answer
is supported by the retrieved context (Faithfulness) and whether the answer
addresses the question (Relevancy).

Usage
-----
    from v2.evaluation.eval_generation import run_v2_generation_eval
    results = run_v2_generation_eval(engine, limit=20)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

GOLD_PATH = _PROJECT_ROOT / "data" / "eval" / "gold_questions" / \
            "apple_2025_10k_gold_eval_50_chunked_minimal.jsonl"


def _load_gold(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows[:limit] if limit else rows


def run_v2_generation_eval(
    engine: RetrieverQueryEngine,
    *,
    limit: Optional[int] = 20,
    judge_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Run generation quality evaluation using LlamaIndex evaluators.

    Parameters
    ----------
    engine       : RetrieverQueryEngine from build_v2_query_engine().
    limit        : Number of gold questions to evaluate (default 20).
    judge_model  : OpenAI model to use as the LLM judge.

    Returns
    -------
    Dict with:
      - summary: {faithfulness, relevancy, n_questions, mean_latency_ms}
      - per_question: list of per-question scores
    """
    judge_llm = LlamaOpenAI(
        model=judge_model,
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    faithfulness_eval = FaithfulnessEvaluator(llm=judge_llm)
    relevancy_eval    = RelevancyEvaluator(llm=judge_llm)

    gold = _load_gold(limit=limit)
    per_question = []
    faith_scores: List[float] = []
    relev_scores: List[float] = []
    latencies: List[float] = []

    for i, row in enumerate(gold):
        question = row["question"]
        ref_answer = row.get("answer", "")
        print(f"  [{i+1}/{len(gold)}] {question[:70]}")

        t0 = time.perf_counter()
        response = engine.query(question)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        answer   = str(response.response) if response.response else ""
        contexts = [ns.node.text for ns in (response.source_nodes or [])]

        # --- Faithfulness ---
        # LlamaIndex checks if every claim in the answer is supported by context
        try:
            faith_result = faithfulness_eval.evaluate(
                query=question,
                response=answer,
                contexts=contexts,
            )
            faith_score = faith_result.score if faith_result.score is not None else 0.0
        except Exception as exc:
            print(f"    Faithfulness eval error: {exc}")
            faith_score = None

        # --- Relevancy ---
        # LlamaIndex checks if the answer is relevant to the query given the contexts
        try:
            relev_result = relevancy_eval.evaluate(
                query=question,
                response=answer,
                contexts=contexts,
            )
            relev_score = relev_result.score if relev_result.score is not None else 0.0
        except Exception as exc:
            print(f"    Relevancy eval error: {exc}")
            relev_score = None

        if faith_score is not None:
            faith_scores.append(faith_score)
        if relev_score is not None:
            relev_scores.append(relev_score)

        per_question.append({
            "question_id":  row.get("question_id", f"q{i+1:02d}"),
            "question":     question,
            "answer":       answer,
            "latency_ms":   round(latency_ms, 1),
            "scores": {
                "faithfulness": faith_score,
                "relevancy":    relev_score,
            },
        })

    n = len(gold)
    summary = {
        "version":          "v2",
        "n_questions":      n,
        "mean_latency_ms":  round(sum(latencies) / n, 1) if latencies else 0,
        "faithfulness":     round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
        "relevancy":        round(sum(relev_scores) / len(relev_scores), 4)  if relev_scores else None,
    }

    return {"summary": summary, "per_question": per_question}
