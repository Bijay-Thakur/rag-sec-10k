"""
Compare semantic / BM25 / hybrid from retrieval/retriever.py without a real Chroma install.

Uses a fake Chroma collection with deterministic hash embeddings (cosine similarity)
plus real BM25 and RRF fusion — same code paths as production except embed_text + query.

Run:
  pytest tests/test_retriever_strategies_compare.py -v --tb=short
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
import unittest.mock as um
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
CHUNK_PATH = REPO / "data" / "chunks" / "semantic_chunks.jsonl"

EMBED_DIM = 96


def _fake_vec(s: str) -> List[float]:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    raw: List[float] = []
    i = 0
    while len(raw) < EMBED_DIM:
        raw.extend(float(b) / 255.0 for b in h)
        h = hashlib.sha256(h + str(i).encode()).digest()
        i += 1
    raw = raw[:EMBED_DIM]
    n = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / n for x in raw]


def _cos_dist(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return 1.0 - max(-1.0, min(1.0, dot))


class FakeCollection:
    def __init__(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.ids = [c["chunk_id"] for c in chunks]
        self.docs = [c["text"] for c in chunks]
        self.metas = [dict(c.get("metadata") or {}) for c in chunks]
        self._emb = [_fake_vec(t) for t in self.docs]

    def query(
        self,
        *,
        query_embeddings: List[List[float]],
        n_results: int,
    ) -> Dict[str, Any]:
        q = query_embeddings[0]
        scored = sorted(
            range(len(self.ids)),
            key=lambda i: _cos_dist(q, self._emb[i]),
        )[:n_results]
        return {
            "ids": [[self.ids[i] for i in scored]],
            "documents": [[self.docs[i] for i in scored]],
            "metadatas": [[self.metas[i] for i in scored]],
            "distances": [[_cos_dist(q, self._emb[i]) for i in scored]],
        }


@pytest.fixture(scope="module")
def retriever_mod():
    """Load retriever.py directly (avoids retrieval/__init__.py → chromadb/openai chain)."""
    if not CHUNK_PATH.is_file():
        pytest.skip(f"Missing chunk file: {CHUNK_PATH}")

    chunks = []
    with open(CHUNK_PATH, encoding="utf-8") as f:
        for _ in range(400):
            line = f.readline()
            if not line:
                break
            chunks.append(json.loads(line))

    fake_col = FakeCollection(chunks)
    fake_client = type("C", (), {})()
    fake_client.get_collection = lambda name: fake_col

    mock_chroma = um.MagicMock()
    mock_chroma.PersistentClient.return_value = fake_client

    retriever_path = SRC / "retrieval" / "retriever.py"
    spec = importlib.util.spec_from_file_location("_retriever_under_test", retriever_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)

    mock_openai = um.MagicMock()
    with um.patch.dict(
        sys.modules,
        {
            "chromadb": mock_chroma,
            "openai": mock_openai,
        },
    ):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    with um.patch.object(mod, "embed_text", side_effect=_fake_vec):
        yield mod, chunks


def _synthetic(
    chunks: Sequence[Dict[str, Any]], n: int, seed: int
) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    usable = [c for c in chunks if len(c.get("text") or "") > 120]
    rng.shuffle(usable)
    for c in usable[:n]:
        words = c["text"][40:].split()
        if len(words) < 25:
            continue
        start = rng.randint(0, len(words) - 20)
        q = " ".join(words[start : start + 14])
        out.append((q, c["chunk_id"]))
    return out


def _rows(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"chunk_id": c["chunk_id"], "text": c["text"], "metadata": dict(c.get("metadata") or {})}
        for c in chunks
    ]


def test_strategy_recall_and_overlap(retriever_mod):
    mod, chunks = retriever_mod
    rows = _rows(chunks)
    syn = _synthetic(chunks, n=50, seed=7)
    k = 10

    sem = mod.get_retriever("semantic_index", rows, "semantic")
    bm = mod.get_retriever("semantic_index", rows, "bm25")
    hy = mod.get_retriever("semantic_index", rows, "hybrid")

    def recall_at(gold: str, hits: List[Dict[str, Any]], kk: int) -> float:
        ids = [h["id"] for h in hits[:kk]]
        return 1.0 if gold in ids else 0.0

    def mmean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    sr5, sr10, br5, br10, hr5, hr10 = [], [], [], [], [], []
    for q, gold in syn:
        sr5.append(recall_at(gold, sem(q, top_k=k), 5))
        sr10.append(recall_at(gold, sem(q, top_k=k), 10))
        br5.append(recall_at(gold, bm(q, top_k=k), 5))
        br10.append(recall_at(gold, bm(q, top_k=k), 10))
        hr5.append(recall_at(gold, hy(q, top_k=k), 5))
        hr10.append(recall_at(gold, hy(q, top_k=k), 10))

    # Smoke: all methods retrieve something
    h0 = sem(syn[0][0], top_k=3)
    assert len(h0) == 3 and all("id" in x for x in h0)

    # Print a compact report for pytest -s users
    print(
        f"\n[mock-semantic + real BM25 + hybrid RRF] n={len(syn)} synthetic span queries, top_k={k}\n"
        f"  semantic  R@5={mmean(sr5):.3f} R@10={mmean(sr10):.3f}\n"
        f"  bm25      R@5={mmean(br5):.3f} R@10={mmean(br10):.3f}\n"
        f"  hybrid    R@5={mmean(hr5):.3f} R@10={mmean(hr10):.3f}\n"
    )

    fixed = [
        "What products or services does the company describe?",
        "AppleCare service and support offerings",
        "reportable geographic segments Americas Europe China",
    ]
    j_sem_bm, j_sem_hy, j_bm_hy = [], [], []
    for q in fixed:
        s = {h["id"] for h in sem(q, top_k=5)}
        b = {h["id"] for h in bm(q, top_k=5)}
        hset = {h["id"] for h in hy(q, top_k=5)}
        def jac(a, c):
            return len(a & c) / len(a | c) if (a | c) else 1.0
        j_sem_bm.append(jac(s, b))
        j_sem_hy.append(jac(s, hset))
        j_bm_hy.append(jac(b, hset))
    print(
        f"  Fixed-query mean Jaccard@5: sem_vs_bm25={mmean(j_sem_bm):.3f} "
        f"sem_vs_hybrid={mmean(j_sem_hy):.3f} bm25_vs_hybrid={mmean(j_bm_hy):.3f}\n"
    )

    # Verbatim span queries: BM25 should beat non-semantic fake embeddings; hybrid tracks BM25 here.
    assert mmean(br10) > mmean(sr10) + 0.5
    assert mmean(hr10) >= mmean(br10) - 0.01
