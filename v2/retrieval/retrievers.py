"""
v2 Retriever factory using LlamaIndex abstractions.

Strategies
----------
semantic        – VectorIndexRetriever (dense OpenAI embeddings via Chroma)
bm25            – Custom BM25BaseRetriever wrapping rank_bm25
                  (llama-index-retrievers-bm25 requires pystemmer which needs
                   MSVC build tools on Windows; we wrap rank_bm25 directly)
hybrid          – QueryFusionRetriever (LlamaIndex RRF, mode="reciprocal_rerank")
hybrid_rerank   – hybrid + SentenceTransformerRerank post-processor
                  (disabled on Windows/Python 3.14 due to PyTorch crash in Streamlit;
                   fully functional as a standalone script)

All strategies return a standard LlamaIndex BaseRetriever so they can be plugged
into RetrieverQueryEngine or RetrieverEvaluator without changes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

V2RetrieverStrategy = Literal["semantic", "bm25", "hybrid", "hybrid_rerank"]


# ---------------------------------------------------------------------------
# BM25 retriever — wraps rank_bm25 as a LlamaIndex BaseRetriever
# ---------------------------------------------------------------------------

class BM25LlamaRetriever(BaseRetriever):
    """
    LlamaIndex-compatible BM25 retriever backed by rank_bm25 (BM25Okapi).

    This wraps the same rank_bm25 library used in v1 but exposes it through
    the LlamaIndex BaseRetriever interface so it can be used in
    QueryFusionRetriever, RetrieverQueryEngine, and RetrieverEvaluator
    without any special-casing.
    """

    def __init__(self, nodes: List[TextNode], top_k: int = 5) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        self._nodes = nodes
        self._top_k = top_k
        corpus = [n.text.split() for n in nodes]
        self._bm25 = BM25Okapi(corpus)
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        tokens = query_bundle.query_str.split()
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[NodeWithScore] = []
        for idx in ranked[: self._top_k]:
            node = self._nodes[idx]
            results.append(NodeWithScore(node=node, score=float(scores[idx])))
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_v2_retriever(
    index: VectorStoreIndex,
    nodes: List[TextNode],
    strategy: V2RetrieverStrategy = "semantic",
    top_k: int = 5,
    candidate_pool: int = 20,
) -> BaseRetriever:
    """
    Return a LlamaIndex BaseRetriever for the requested strategy.

    Parameters
    ----------
    index          : VectorStoreIndex (v2 Chroma-backed index).
    nodes          : All TextNode objects (needed for BM25 corpus).
    strategy       : One of "semantic", "bm25", "hybrid", "hybrid_rerank".
    top_k          : Number of results to return.
    candidate_pool : For hybrid_rerank — how many RRF candidates to rerank.
    """

    # --- Semantic -----------------------------------------------------------
    if strategy == "semantic":
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

    # --- BM25 ---------------------------------------------------------------
    if strategy == "bm25":
        return BM25LlamaRetriever(nodes=nodes, top_k=top_k)

    # --- Hybrid RRF ---------------------------------------------------------
    if strategy in ("hybrid", "hybrid_rerank"):
        semantic_ret = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k if strategy == "hybrid" else candidate_pool,
        )
        bm25_ret = BM25LlamaRetriever(
            nodes=nodes,
            top_k=top_k if strategy == "hybrid" else candidate_pool,
        )

        # QueryFusionRetriever with mode="reciprocal_rerank" implements RRF
        fusion_ret = QueryFusionRetriever(
            retrievers=[semantic_ret, bm25_ret],
            similarity_top_k=top_k if strategy == "hybrid" else candidate_pool,
            num_queries=1,          # do NOT generate extra LLM queries — pure fusion
            mode="reciprocal_rerank",
            use_async=False,
        )

        if strategy == "hybrid":
            return fusion_ret

        # --- Hybrid + Rerank ------------------------------------------------
        # SentenceTransformerRerank is a post-processor, not a retriever.
        # We wrap it inside a thin BaseRetriever adapter.
        return _HybridRerankerRetriever(
            fusion_retriever=fusion_ret,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )

    raise ValueError(
        f"Unknown strategy {strategy!r}. "
        "Choose from: semantic, bm25, hybrid, hybrid_rerank"
    )


class _HybridRerankerRetriever(BaseRetriever):
    """
    Combines QueryFusionRetriever (RRF) with SentenceTransformerRerank.

    NOTE: Disabled in Streamlit on Windows/Python 3.14 (PyTorch crash).
    Works fine as a standalone script on any platform with PyTorch installed.
    """

    def __init__(
        self,
        fusion_retriever: QueryFusionRetriever,
        top_k: int,
        candidate_pool: int,
    ) -> None:
        self._fusion = fusion_retriever
        self._top_k = top_k
        self._candidate_pool = candidate_pool
        self._reranker: Optional[Any] = None
        super().__init__()

    def _get_reranker(self):
        if self._reranker is None:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            self._reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=self._top_k,
            )
        return self._reranker

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        candidates = self._fusion.retrieve(query_bundle.query_str)
        reranker = self._get_reranker()
        reranked = reranker.postprocess_nodes(
            candidates, query_bundle=query_bundle
        )
        return reranked[: self._top_k]
