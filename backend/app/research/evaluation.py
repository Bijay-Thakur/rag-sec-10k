"""Deterministic metrics on accession-labeled held-out predictions."""

from __future__ import annotations

from statistics import mean
import math


def score_cases(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("Evaluation corpus is empty")
    recall, recall1, reciprocal, ndcg, valid_scope, valid_citations, refusals = [], [], [], [], [], [], []
    true_refusal = predicted_refusal = expected_refusal = 0
    for case in cases:
        gold = set(case["relevant_chunk_ids"])
        ranked = list(dict.fromkeys(case["ranked_chunk_ids"]))[:5]
        if gold:
            recall.append(len(gold.intersection(ranked)) / len(gold))
            recall1.append(len(gold.intersection(ranked[:1])) / len(gold))
            reciprocal.append(next((1 / (i + 1) for i, chunk in enumerate(ranked) if chunk in gold), 0.0))
            dcg = sum(1 / math.log2(i + 2) for i, chunk in enumerate(ranked) if chunk in gold)
            ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(gold))))
            ndcg.append(dcg / ideal)
        valid_scope.append(all(hit["cik"] == case["cik"] and hit["accession"] == case["accession"]
                               and hit["revision_id"] == case["revision_id"] for hit in case["hits"]))
        if case["cited_chunk_ids"]:
            valid_citations.append(all(citation in {hit["chunk_id"] for hit in case["hits"]}
                                       for citation in case["cited_chunk_ids"]))
        refusals.append(case["refused"] == case["unanswerable"])
        predicted_refusal += bool(case["refused"])
        expected_refusal += bool(case["unanswerable"])
        true_refusal += bool(case["refused"] and case["unanswerable"])
    return {"cases": len(cases), "retrieval_cases": len(recall),
            "recall_at_5": mean(recall) if recall else None,
            "recall_at_1": mean(recall1) if recall1 else None,
            "mrr_at_5": mean(reciprocal) if reciprocal else None,
            "ndcg_at_5": mean(ndcg) if ndcg else None,
            "scope_valid_rate": mean(valid_scope), "citation_target_cases": len(valid_citations),
            "citation_target_rate": mean(valid_citations) if valid_citations else None,
            "refusal_agreement": mean(refusals),
            "refusal_precision": true_refusal / predicted_refusal if predicted_refusal else None,
            "refusal_recall": true_refusal / expected_refusal if expected_refusal else None}
