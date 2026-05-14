"""
v2 Generation: RetrieverQueryEngine with a citation-grounded PromptTemplate.

v1 used raw OpenAI API calls with manually formatted messages.
v2 uses LlamaIndex's RetrieverQueryEngine which:
  1. Calls the retriever to get context nodes.
  2. Formats them into a prompt via a PromptTemplate.
  3. Calls the LLM (gpt-4o-mini) and returns a Response object.

The citation rules and financial-analyst persona are identical to v1 to make
the generation quality comparison fair.

Usage
-----
    from v2.generation import build_v2_query_engine

    engine = build_v2_query_engine(retriever)
    response = engine.query("What are Apple's main risk factors?")
    print(response.response)
    for node in response.source_nodes:
        print(node.metadata["item"], node.score)
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI as LlamaOpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DEFAULT_MODEL  = "gpt-4o-mini"
MAX_TOKENS     = 1024

# ---------------------------------------------------------------------------
# Citation-grounded prompt (identical rules to v1 SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

# LlamaIndex uses {context_str} and {query_str} as standard template variables
CITATION_PROMPT_TMPL = """\
You are a financial analyst assistant specialising in SEC 10-K annual report analysis.

Rules:
1. Answer using ONLY the numbered context passages provided. Do not use outside knowledge.
2. Cite every factual claim with an inline reference like [1] or [2]. Multiple citations
   per sentence are fine, e.g. [1][3].
3. If the context passages do not contain enough information to answer, say:
   "The provided context does not contain sufficient information to answer this question."
4. Be concise and precise. Avoid repeating the question. Do not speculate.
5. Financial figures should be quoted exactly as they appear in the source passages.

Context passages from SEC 10-K filings:

{context_str}

---

Question: {query_str}

Answer (cite each claim with [n]):"""

CITATION_PROMPT = PromptTemplate(CITATION_PROMPT_TMPL)


# ---------------------------------------------------------------------------
# Result dataclass — mirrors v1 GenerationResult for easy comparison
# ---------------------------------------------------------------------------

@dataclass
class V2QueryResult:
    question:      str
    answer:        str
    source_nodes:  List[Dict[str, Any]]  # node text + metadata + score
    cited_indices: List[int]
    model:         str
    latency_ms:    float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def context_texts(self) -> List[str]:
        return [n["text"] for n in self.source_nodes]


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------

def build_v2_query_engine(
    retriever: BaseRetriever,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
) -> RetrieverQueryEngine:
    """
    Build a RetrieverQueryEngine with the citation-grounded prompt.

    The engine combines:
    - Any LlamaIndex BaseRetriever (semantic / BM25 / hybrid / hybrid_rerank)
    - A custom PromptTemplate with citation rules
    - gpt-4o-mini as the generation LLM

    Returns
    -------
    RetrieverQueryEngine — call engine.query("your question") to get a Response.
    """
    llm = LlamaOpenAI(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=CITATION_PROMPT,
        response_mode="compact",
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    return engine


def query_with_timing(
    engine: RetrieverQueryEngine,
    question: str,
) -> V2QueryResult:
    """
    Run a query through the engine and return a structured V2QueryResult.

    Adds wall-clock latency and extracts inline citation numbers from the answer.
    """
    t0       = time.perf_counter()
    response = engine.query(question)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    answer = str(response.response) if response.response else ""
    cited_indices = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer)))

    source_nodes = []
    for ns in (response.source_nodes or []):
        source_nodes.append({
            "node_id":  ns.node_id,
            "text":     ns.node.text if ns.node else "",
            "metadata": dict(ns.node.metadata) if ns.node else {},
            "score":    float(ns.score) if ns.score is not None else 0.0,
        })

    # Token usage (LlamaIndex OpenAI LLM stores it in the callback manager)
    prompt_tok = completion_tok = 0
    try:
        usage = response.metadata.get("usage") if response.metadata else None
        if usage:
            prompt_tok     = getattr(usage, "prompt_tokens", 0) or 0
            completion_tok = getattr(usage, "completion_tokens", 0) or 0
    except Exception:
        pass

    return V2QueryResult(
        question      = question,
        answer        = answer,
        source_nodes  = source_nodes,
        cited_indices = cited_indices,
        model         = DEFAULT_MODEL,
        latency_ms    = round(latency_ms, 1),
        prompt_tokens = prompt_tok,
        completion_tokens = completion_tok,
    )
