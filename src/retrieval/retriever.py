from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Literal, Optional

import os
import chromadb
from chromadb import Collection
from rank_bm25 import BM25Okapi

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "db"

EMBEDDING_MODEL = "text-embedding-3-small"

# Cross-encoder model — compact (22 M params), strong MS-MARCO ranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Chroma client & collection
# ---------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

# Lazy-loaded cross-encoder (only imported when reranking is requested)
_cross_encoder: Optional[Any] = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder  # type: ignore
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def get_collection(name: str) -> Collection:
    return chroma_client.get_collection(name=name)


# ---------------------------------------------------------
# Semantic embedding helper
# ---------------------------------------------------------
def embed_text(text: str) -> List[float]:
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty text")

    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


# ---------------------------------------------------------
# Semantic retriever (Chroma)
# ---------------------------------------------------------
def semantic_search(
    collection: Collection,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Semantic retrieval using OpenAI embeddings + Chroma."""
    q_embed = embed_text(query)

    res = collection.query(
        query_embeddings=[q_embed],
        n_results=top_k,
    )

    out: List[Dict[str, Any]] = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    for i in range(len(ids)):
        out.append(
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            }
        )
    return out


# ---------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------
class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        """
        chunks: list of {"chunk_id", "text", "metadata", ...}
        """
        self.chunks = chunks
        self.corpus_texts = [c["text"] for c in chunks]
        self.tokenized_corpus = [doc.split() for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices[:top_k]:
            c = self.chunks[idx]
            results.append(
                {
                    "id": c["chunk_id"],
                    "text": c["text"],
                    "metadata": c["metadata"],
                    "score": float(scores[idx]),
                }
            )
        return results


# ---------------------------------------------------------
# Hybrid retrieval via Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------
def hybrid_search(
    collection: Collection,
    bm25_retriever: BM25Retriever,
    query: str,
    top_k: int = 5,
    k_rrf: int = 60,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
    - semantic (Chroma + OpenAI)
    - lexical (BM25)
    merged via Reciprocal Rank Fusion.
    """
    semantic_results = semantic_search(collection, query, top_k=top_k)
    bm25_results = bm25_retriever.search(query, top_k=top_k)

    # Build rank maps: doc_id -> rank (1-based)
    semantic_ranks: Dict[str, int] = {}
    for rank, item in enumerate(semantic_results, start=1):
        semantic_ranks[item["id"]] = rank

    bm25_ranks: Dict[str, int] = {}
    for rank, item in enumerate(bm25_results, start=1):
        bm25_ranks[item["id"]] = rank

    # Collect all unique doc IDs
    all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

    # Compute RRF scores
    fused_scores: Dict[str, float] = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in semantic_ranks:
            score += 1.0 / (k_rrf + semantic_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 1.0 / (k_rrf + bm25_ranks[doc_id])
        fused_scores[doc_id] = score

    # Build id -> chunk lookup from BM25 corpus (has full texts/metadata)
    id_to_chunk = {c["chunk_id"]: c for c in bm25_retriever.chunks}

    # Sort by fused score
    sorted_ids = sorted(
        fused_scores.keys(),
        key=lambda i: fused_scores[i],
        reverse=True,
    )

    results: List[Dict[str, Any]] = []
    for doc_id in sorted_ids[:top_k]:
        c = id_to_chunk[doc_id]
        results.append(
            {
                "id": doc_id,
                "text": c["text"],
                "metadata": c["metadata"],
                "rrf_score": fused_scores[doc_id],
            }
        )

    return results


# ---------------------------------------------------------
# Reranker (cross-encoder over a candidate set)
# ---------------------------------------------------------
def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Score every candidate with a cross-encoder and return the top_k highest-scored chunks.

    Each item in `candidates` must have an "id" and "text" key (same shape as results
    from semantic_search / hybrid_search).  A "rerank_score" key is added to each
    returned item.
    """
    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: float(x[0]),
        reverse=True,
    )

    results = []
    for score, chunk in scored[:top_k]:
        out = dict(chunk)
        out["rerank_score"] = float(score)
        results.append(out)
    return results


def hybrid_rerank_search(
    collection: Collection,
    bm25_retriever: BM25Retriever,
    query: str,
    top_k: int = 5,
    candidate_pool: int = 20,
    k_rrf: int = 60,
) -> List[Dict[str, Any]]:
    """
    Hybrid RRF over a wider candidate pool, then reranked with a cross-encoder.

    candidate_pool: how many hybrid candidates to pass to the reranker (default 20).
    top_k: final results to return after reranking.
    """
    candidates = hybrid_search(
        collection=collection,
        bm25_retriever=bm25_retriever,
        query=query,
        top_k=candidate_pool,
        k_rrf=k_rrf,
    )
    return rerank(query, candidates, top_k=top_k)


# ---------------------------------------------------------
# High-level factory
# ---------------------------------------------------------
RetrieverStrategy = Literal["semantic", "bm25", "hybrid", "hybrid_rerank"]


def get_retriever(
    collection_name: str,
    chunks: List[Dict[str, Any]],
    strategy: RetrieverStrategy = "semantic",
    candidate_pool: int = 20,
):
    """
    Factory that returns a callable retriever:
        retriever(query: str, top_k: int) -> List[Dict]

    Strategies
    ----------
    semantic       – dense OpenAI embeddings via Chroma
    bm25           – lexical BM25Okapi
    hybrid         – RRF fusion of semantic + BM25
    hybrid_rerank  – hybrid RRF (candidate_pool docs) then cross-encoder reranker
    """
    collection = get_collection(collection_name)
    bm25_retriever = BM25Retriever(chunks)

    if strategy == "semantic":
        def _retriever(query: str, top_k: int = 5):
            return semantic_search(collection, query, top_k=top_k)
        return _retriever

    if strategy == "bm25":
        def _retriever(query: str, top_k: int = 5):
            return bm25_retriever.search(query, top_k=top_k)
        return _retriever

    if strategy == "hybrid":
        def _retriever(query: str, top_k: int = 5):
            return hybrid_search(
                collection=collection,
                bm25_retriever=bm25_retriever,
                query=query,
                top_k=top_k,
            )
        return _retriever

    if strategy == "hybrid_rerank":
        def _retriever(query: str, top_k: int = 5):
            return hybrid_rerank_search(
                collection=collection,
                bm25_retriever=bm25_retriever,
                query=query,
                top_k=top_k,
                candidate_pool=candidate_pool,
            )
        return _retriever

    raise ValueError(f"Unknown strategy: {strategy!r}. "
                     f"Choose from: semantic, bm25, hybrid, hybrid_rerank")
