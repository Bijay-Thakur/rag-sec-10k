"""Canonical filing registry for SEC Insight AI."""

from __future__ import annotations

from typing import Any, Dict, List

FILING_REGISTRY: Dict[str, Dict[str, str]] = {
    "apple_2025": {
        "collection": "semantic_index",
        "chunks_file": "semantic_chunks.jsonl",
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "source_file": "Apple.html",
        "fiscal_year": "2025",
    },
    "walmart_2026": {
        "collection": "semantic_index",
        "chunks_file": "semantic_chunks.jsonl",
        "company": "Walmart Inc.",
        "ticker": "WMT",
        "source_file": "Walmart.html",
        "fiscal_year": "2026",
    },
    "exxon_2025": {
        "collection": "semantic_index",
        "chunks_file": "semantic_chunks.jsonl",
        "company": "Exxon Mobil Corporation",
        "ticker": "XOM",
        "source_file": "Exxon.html",
        "fiscal_year": "2025",
    },
    "elilily_2025": {
        "collection": "semantic_index",
        "chunks_file": "semantic_chunks.jsonl",
        "company": "Eli Lilly and Company",
        "ticker": "LLY",
        "source_file": "Elilily.html",
        "fiscal_year": "2025",
    },
    "chase_2025": {
        "collection": "semantic_index",
        "chunks_file": "semantic_chunks.jsonl",
        "company": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "source_file": "Chase.html",
        "fiscal_year": "2025",
    },
}


def list_filing_ids() -> List[str]:
    """Return all registered filing identifiers (stable order)."""
    return list(FILING_REGISTRY.keys())


def get_filing_meta(filing_id: str) -> Dict[str, str]:
    return FILING_REGISTRY[filing_id]


def filing_catalog() -> List[Dict[str, Any]]:
    """Frontend-ready filing list with display labels."""
    rows: List[Dict[str, Any]] = []
    for filing_id, meta in FILING_REGISTRY.items():
        rows.append({
            "filing_id": filing_id,
            "company": meta["company"],
            "ticker": meta.get("ticker", ""),
            "fiscal_year": meta.get("fiscal_year", ""),
            "source_file": meta.get("source_file", ""),
            "label": f"{meta['company']} — FY{meta.get('fiscal_year', '?')} 10-K",
        })
    return rows
