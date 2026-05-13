"""
SEC 10-K RAG Q&A — Streamlit UI  (Apple 2025 10-K)

Run:
    $env:PYTHONPATH = "src"
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── path setup (must happen before local imports) ───────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ── top-level imports from the project ──────────────────────────────────────
try:
    from generation.generator import generate_answer, GenerationResult
    _GEN_OK = True
except Exception as _e:
    _GEN_OK = False
    _GEN_ERR = str(_e)

try:
    from retrieval.retriever import BM25Retriever
    _RET_OK = True
except Exception as _e:
    _RET_OK = False
    _RET_ERR = str(_e)

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple 10-K RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── fatal import guard ───────────────────────────────────────────────────────
if not _GEN_OK:
    st.error(f"Failed to import generation module: {_GEN_ERR}")
    st.stop()
if not _RET_OK:
    st.error(f"Failed to import retrieval module: {_RET_ERR}")
    st.stop()


# ── cached resources (each loads once per server session) ───────────────────

@st.cache_resource(show_spinner="Connecting to vector store …")
def _get_collection():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(ROOT / "db"))
        col = client.get_collection(name="semantic_index")
        return col, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Building BM25 index …")
def _get_bm25():
    try:
        path = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"
        rows: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    c = json.loads(s)
                    rows.append({
                        "chunk_id": c["chunk_id"],
                        "text":     c["text"],
                        "metadata": c["metadata"],
                    })
        bm25 = BM25Retriever(rows)
        return bm25, rows, None
    except Exception as exc:
        return None, [], str(exc)




# ── retrieve ─────────────────────────────────────────────────────────────────

def _embed(question: str) -> List[float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    resp = client.embeddings.create(model="text-embedding-3-small", input=[question])
    return resp.data[0].embedding


def _retrieve(
    question: str,
    strategy: str,
    top_k: int,
    collection,
    bm25: Any,
    bm25_rows: List[Dict],
    k_rrf: int = 60,
    pool: int = 20,
) -> List[Dict[str, Any]]:

    # BM25 — no embedding needed
    if strategy == "bm25":
        return bm25.search(question, top_k=top_k)

    # All other strategies need a query vector
    q_vec = _embed(question)

    if strategy == "semantic":
        res   = collection.query(query_embeddings=[q_vec], n_results=top_k)
        ids   = res["ids"][0]
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            {"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]}
            for i in range(len(ids))
        ]

    # hybrid / hybrid_rerank — RRF fusion
    n          = pool if strategy == "hybrid_rerank" else top_k
    sem_res    = collection.query(query_embeddings=[q_vec], n_results=n)
    sem_hits   = [
        {"id":       sem_res["ids"][0][i],
         "text":     sem_res["documents"][0][i],
         "metadata": sem_res["metadatas"][0][i]}
        for i in range(len(sem_res["ids"][0]))
    ]
    bm25_hits  = bm25.search(question, top_k=n)
    sem_ranks  = {h["id"]: r for r, h in enumerate(sem_hits,  1)}
    bm25_ranks = {h["id"]: r for r, h in enumerate(bm25_hits, 1)}
    all_ids    = set(sem_ranks) | set(bm25_ranks)
    fused      = {
        d: 1 / (k_rrf + sem_ranks.get(d, 9999)) + 1 / (k_rrf + bm25_ranks.get(d, 9999))
        for d in all_ids
    }
    id_map     = {c["chunk_id"]: c for c in bm25_rows}
    ranked     = sorted(fused, key=fused.__getitem__, reverse=True)
    candidates = [
        {"id": d, "text": id_map[d]["text"],
         "metadata": id_map[d]["metadata"], "rrf_score": fused[d]}
        for d in ranked if d in id_map
    ]

    return candidates[:top_k]


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    strategy = st.selectbox(
        "Retrieval strategy",
        options=["hybrid", "semantic", "bm25"],
        index=0,
        help=(
            "**hybrid** — RRF fusion of dense + BM25 (~20 ms). Best live option.\n\n"
            "**semantic** — Dense OpenAI embeddings only (~5 ms).\n\n"
            "**bm25** — Keyword matching, no API call (~10 ms).\n\n"
            "*hybrid_rerank uses PyTorch which crashes on Windows/Py3.14 inside Streamlit. "
            "Its benchmark results are shown in the eval panel below.*"
        ),
    )

    top_k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=10, value=5)

    st.markdown("---")
    st.markdown("**Indexed filing**")
    st.markdown("🍎 **Apple Inc. (AAPL)**  \n*Technology · FY 2025*")

    st.markdown("---")
    st.markdown("**Retrieval eval (50 gold Qs)**")
    _rsum = ROOT / "data" / "eval" / "retrieval_summary.json"
    if _rsum.is_file():
        try:
            with open(_rsum) as f:
                _evals = json.load(f)
            for row in _evals:
                badge = " ✅" if row.get("strategy") == strategy else ""
                st.markdown(
                    f"`{row.get('strategy','?')}`{badge}  "
                    f"R@5={row.get('recall@5','?')}  MRR={row.get('mrr','?')}"
                )
        except Exception:
            st.caption("Could not load retrieval_summary.json")
    else:
        st.caption("Run scripts/run_retrieval_eval.py to populate.")

    _rgas = ROOT / "data" / "eval" / "ragas_summary.json"
    if _rgas.is_file():
        try:
            with open(_rgas) as f:
                _rg = json.load(f)
            if "error" not in _rg:
                st.markdown("---")
                st.markdown("**RAGAS scores (20 Qs)**")
                _labels = {
                    "faithfulness":      "Faithfulness",
                    "answer_relevancy":  "Answer Relevancy",
                    "context_recall":    "Context Recall",
                    "context_precision": "Context Precision",
                }
                for k, v in _rg.items():
                    st.markdown(f"- {_labels.get(k, k)}: **{v}**")
        except Exception:
            pass

    st.markdown("---")
    st.caption("No LangChain · No LlamaIndex · GPT-4o-mini")


# ── main ─────────────────────────────────────────────────────────────────────

st.title("📄 Apple 10-K RAG Q&A")
st.markdown(
    "Ask any question about **Apple's 2025 annual report (10-K)**. "
    "Answers are grounded in retrieved passages with inline citations."
)

# ── input row ────────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "Question",
        placeholder='e.g. "What are Apple\'s main risk factors related to tariffs?"',
        label_visibility="collapsed",
    )
with col_btn:
    ask = st.button("Ask", type="primary", use_container_width=True)

with st.expander("Example questions"):
    examples = [
        "What tariff and trade risks does Apple describe in its 2025 10-K?",
        "How does Apple manage its supply chain concentration risk in China?",
        "What were Apple's total net sales in fiscal year 2025?",
        "Which geographic segments does Apple use to report its revenue?",
        "What cybersecurity risks does Apple disclose?",
        "How did Apple's R&D spending change year-over-year?",
        "What does Apple say about competition in the smartphone market?",
        "What are the key risks related to Apple's services business?",
        "How does Apple describe its capital return program?",
        "What regulatory risks does Apple highlight in Item 1A?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            question = ex
            ask = True

if not question or not ask:
    st.stop()

# ── API key guard ─────────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY") and strategy in ("semantic", "hybrid", "hybrid_rerank"):
    st.error(
        "OPENAI_API_KEY is not set. "
        "Add `OPENAI_API_KEY=sk-...` to your `.env` file and restart Streamlit."
    )
    st.stop()

# ── load resources ────────────────────────────────────────────────────────────
collection, _col_err = _get_collection()
if _col_err:
    st.error(f"Could not connect to ChromaDB: {_col_err}")
    st.stop()

bm25, bm25_rows, _bm_err = _get_bm25()
if _bm_err:
    st.error(f"Could not build BM25 index: {_bm_err}")
    st.stop()

# ── retrieve ──────────────────────────────────────────────────────────────────
hits: List[Dict[str, Any]] = []
retrieval_ms = 0.0

with st.spinner(f"Retrieving with **{strategy}** …"):
    try:
        t0           = time.perf_counter()
        hits         = _retrieve(question, strategy, top_k, collection, bm25, bm25_rows)
        retrieval_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

if not hits:
    st.warning("No chunks were retrieved. Try a different question or strategy.")
    st.stop()

# ── generate ──────────────────────────────────────────────────────────────────
with st.spinner("Generating answer with GPT-4o-mini …"):
    try:
        result: GenerationResult = generate_answer(question, hits)
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

# ── answer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Answer")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy",    strategy)
c2.metric("Chunks used", len(hits))
c3.metric("Retrieval",   f"{retrieval_ms:.0f} ms")
c4.metric("Generation",  f"{result.latency_ms:.0f} ms")

st.markdown(result.answer)

# ── sources ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Sources  ({len(hits)} retrieved chunks)")

for i, chunk in enumerate(hits, start=1):
    meta      = chunk.get("metadata") or {}
    item      = meta.get("item", "")
    title     = meta.get("section_title", "")
    strategy_ = meta.get("chunk_strategy", "")
    chars     = meta.get("char_count", len(chunk.get("text") or ""))
    text      = (chunk.get("text") or chunk.get("document") or "").strip()
    is_cited  = i in result.cited_indices

    marker = " 📌 cited" if is_cited else ""
    header = f"[{i}] Apple · {item}" + (f" — {title}" if title else "") + marker

    with st.expander(header, expanded=is_cited):
        st.markdown(
            f"<div style='background:#f0f4ff;padding:12px;border-radius:6px;"
            f"font-size:0.88rem;line-height:1.5'>{text}</div>",
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        d1.caption(f"Strategy: {strategy_}")
        d2.caption(f"Chars: {chars}")
        _sk = (
            "rerank_score" if "rerank_score" in chunk else
            "rrf_score"    if "rrf_score"    in chunk else
            "score"        if "score"        in chunk else
            "distance"
        )
        if _sk in chunk:
            d3.caption(f"{_sk}: {chunk[_sk]:.4f}")
