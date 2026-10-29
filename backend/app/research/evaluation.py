"""Deterministic metrics on accession-labeled held-out predictions."""

from __future__ import annotations

from statistics import mean


def score_cases(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("Evaluation corpus is empty")
    recall, reciprocal, valid_scope, valid_citations, refusals = [], [], [], [], []
    for case in cases:
        gold = set(case["relevant_chunk_ids"])
        ranked = case["ranked_chunk_ids"][:5]
        recall.append(len(gold.intersection(ranked)) / len(gold) if gold else 1.0)
        reciprocal.append(next((1 / (i + 1) for i, chunk in enumerate(ranked) if chunk in gold), 0.0))
        valid_scope.append(all(hit["cik"] == case["cik"] and hit["accession"] == case["accession"]
                               and hit["revision_id"] == case["revision_id"] for hit in case["hits"]))
        valid_citations.append(all(citation in {hit["chunk_id"] for hit in case["hits"]}
                                   for citation in case["cited_chunk_ids"]))
        refusals.append(case["refused"] == case["unanswerable"])
    return {"cases": len(cases), "recall_at_5": mean(recall), "mrr_at_5": mean(reciprocal),
            "scope_valid_rate": mean(valid_scope), "citation_target_rate": mean(valid_citations),
            "refusal_agreement": mean(refusals)}
