#!/usr/bin/env python3
"""
Smoke test: run the RAG pipeline and print standardized RAGResponse JSON.

Usage (repo root, venv activated):
    python scripts/smoke_rag_response.py
    python scripts/smoke_rag_response.py --question "What were Apple's total net sales in 2025?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.adapters.rag_adapter import RAGPipelineError, ask_question


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test RAGResponse output")
    parser.add_argument(
        "--question",
        default="What were Apple's total net sales in fiscal year 2025?",
        help="Question to ask",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Use demo_mode (BM25 only, no LLM) — default true",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline with LLM generation (requires OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    demo_mode = not args.full

    try:
        response = ask_question(args.question, demo_mode=demo_mode)
    except RAGPipelineError as exc:
        print(json.dumps({"error": str(exc), "error_code": exc.error_code}, indent=2))
        return 1

    print(response.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
