"""
Generation module: takes a question + retrieved chunks, builds a grounded prompt,
calls the OpenAI chat API, and returns a structured result with inline citations.

Usage
-----
    from generation.generator import generate_answer

    result = generate_answer(question="What are Apple's main risk factors?", chunks=hits)
    print(result.answer)
    for i, c in enumerate(result.cited_chunks, 1):
        print(f"[{i}] {c['source_file']} — {c['item']}")
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DEFAULT_MODEL  = "gpt-4o-mini"
DEFAULT_TOP_K  = 5
MAX_TOKENS     = 1024

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a financial analyst assistant specialising in SEC 10-K annual report analysis.

Rules:
1. Answer using ONLY the numbered context passages provided. Do not use outside knowledge.
2. Cite every factual claim with an inline reference like [1] or [2]. Multiple citations
   per sentence are fine, e.g. [1][3].
3. If the context passages do not contain enough information to answer, say:
   "The provided context does not contain sufficient information to answer this question."
4. Be concise and precise. Avoid repeating the question. Do not speculate.
5. Financial figures should be quoted exactly as they appear in the source passages.\
"""


def build_messages(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build the chat messages list for the OpenAI API.

    Each chunk becomes a numbered context block so the model can cite [1], [2], etc.
    Accepts chunks from any retriever (retriever.py or retrieve.py shape).
    """
    context_parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text") or chunk.get("document") or ""
        meta = chunk.get("metadata") or {}
        company  = str(meta.get("source_file", "?")).replace(".html", "")
        item     = meta.get("item", "?")
        title    = meta.get("section_title", "")
        label    = f"{company} | {item}" + (f" — {title}" if title else "")
        context_parts.append(f"[{i}] {label}\n{text.strip()}")

    context_block = "\n\n---\n\n".join(context_parts)
    user_content  = (
        f"Context passages from SEC 10-K filings:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite each claim with [n]):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    question:      str
    answer:        str
    chunks:        List[Dict[str, Any]]   # all chunks passed to the model
    cited_chunks:  List[Dict[str, Any]]   # chunks actually cited in the answer
    cited_indices: List[int]              # 1-based citation numbers found in answer
    model:         str
    latency_ms:    float
    prompt_tokens: int  = 0
    completion_tokens: int = 0

    @property
    def context_texts(self) -> List[str]:
        """Plain text of each chunk — used by RAGAS."""
        return [c.get("text") or c.get("document") or "" for c in self.chunks]

    @property
    def cited_context_texts(self) -> List[str]:
        """Plain text of cited chunks only."""
        return [c.get("text") or c.get("document") or "" for c in self.cited_chunks]


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def _extract_citations(answer: str, chunks: List[Dict[str, Any]]) -> tuple[List[int], List[Dict]]:
    """Return (sorted unique citation indices, corresponding chunk dicts)."""
    nums   = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer)))
    cited  = [chunks[i - 1] for i in nums if 1 <= i <= len(chunks)]
    return nums, cited


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

_oai_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _oai_client


def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
) -> GenerationResult:
    """
    Generate a grounded answer with inline citations.

    Parameters
    ----------
    question  : Natural language question.
    chunks    : Retrieved chunks (list of dicts with 'text'/'document' + 'metadata').
    model     : OpenAI chat model name.
    max_tokens: Maximum completion tokens.
    temperature: Sampling temperature (0 = deterministic).

    Returns
    -------
    GenerationResult with answer, cited chunks, latencies, and token counts.
    """
    if not chunks:
        return GenerationResult(
            question=question,
            answer="No context passages were retrieved for this question.",
            chunks=[],
            cited_chunks=[],
            cited_indices=[],
            model=model,
            latency_ms=0.0,
        )

    messages = build_messages(question, chunks)

    t0       = time.perf_counter()
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    answer              = response.choices[0].message.content or ""
    cited_indices, cited = _extract_citations(answer, chunks)

    return GenerationResult(
        question          = question,
        answer            = answer,
        chunks            = chunks,
        cited_chunks      = cited,
        cited_indices     = cited_indices,
        model             = model,
        latency_ms        = round(latency_ms, 1),
        prompt_tokens     = response.usage.prompt_tokens     if response.usage else 0,
        completion_tokens = response.usage.completion_tokens if response.usage else 0,
    )
