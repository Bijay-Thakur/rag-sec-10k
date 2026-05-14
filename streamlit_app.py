"""
SEC 10-K RAG Q&A — Streamlit UI  (Apple 2025 10-K)
Supports two pipeline versions selectable via a pop-up dialog:
  v1 — Manual pipeline (custom code, no framework)
  v2 — LlamaIndex pipeline (framework-based)

Run:
    $env:PYTHONPATH = ".;src"
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
sys.path.insert(0, str(ROOT))        # needed for v2.* imports

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ── top-level imports ────────────────────────────────────────────────────────

# v1 generation
try:
    from generation.generator import generate_answer, GenerationResult
    _V1_GEN_OK = True
except Exception as _e:
    _V1_GEN_OK = False
    _V1_GEN_ERR = str(_e)

# v1 retrieval
try:
    from retrieval.retriever import BM25Retriever as V1BM25Retriever
    _V1_RET_OK = True
except Exception as _e:
    _V1_RET_OK = False
    _V1_RET_ERR = str(_e)

# v2 imports (non-fatal — deferred to when v2 is actually selected)
_V2_OK  = False
_V2_ERR = ""
try:
    from v2.indexing   import load_v2_index, v2_index_ready
    from v2.retrieval  import get_v2_retriever
    from v2.generation import build_v2_query_engine
    from v2.generation.query_engine import query_with_timing
    _V2_OK = True
except Exception as _e:
    _V2_ERR = str(_e)

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple 10-K RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── fatal guard: v1 must always load ────────────────────────────────────────
if not _V1_GEN_OK:
    st.error(f"Failed to import v1 generation module: {_V1_GEN_ERR}")
    st.stop()
if not _V1_RET_OK:
    st.error(f"Failed to import v1 retrieval module: {_V1_RET_ERR}")
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cached resources
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── v1 resources ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to v1 vector store …")
def _get_v1_collection():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(ROOT / "db"))
        col = client.get_collection(name="semantic_index")
        return col, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Building v1 BM25 index …")
def _get_v1_bm25():
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
        bm25 = V1BM25Retriever(rows)
        return bm25, rows, None
    except Exception as exc:
        return None, [], str(exc)


# ── v2 resources ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading v2 TextNodes for BM25 …")
def _get_v2_nodes():
    """Load TextNodes from v1's semantic_chunks.jsonl — shared data, no re-embedding."""
    try:
        from llama_index.core.schema import TextNode
        path = ROOT / "data" / "chunks" / "semantic_chunks.jsonl"
        nodes = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                c = json.loads(s)
                nodes.append(TextNode(
                    text=c["text"],
                    id_=c["chunk_id"],
                    metadata=c.get("metadata", {}),
                ))
        return nodes, None
    except Exception as exc:
        return [], str(exc)


@st.cache_resource(show_spinner="Connecting to v2 vector store …")
def _get_v2_index():
    """
    Load the pre-built VectorStoreIndex from ChromaDB collection 'v2_semantic_index'.
    Returns (index, error_str).  Never re-embeds — raises a friendly error if not built.
    """
    try:
        if not _V2_OK:
            return None, f"v2 modules not available: {_V2_ERR}"
        if not v2_index_ready():
            return None, (
                "v2 index not ready.\n\n"
                "Run this command once to build it:\n"
                "```\n$env:PYTHONPATH = '.;src'\n"
                "python scripts/run_v2_index.py\n```"
            )
        idx = load_v2_index()
        return idx, None
    except Exception as exc:
        return None, str(exc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Version selection dialog
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.dialog("🚀 Choose Pipeline Version", width="small")
def _show_version_picker() -> None:
    """
    Modal dialog shown on first load and when the user clicks
    'Change version' in the sidebar.
    """
    st.markdown(
        "Both pipelines answer the same questions from **Apple's 2025 10-K**."
        "  \nSelect the version to use:"
    )

    v1_desc = (
        "**v1 — Manual pipeline**  \n"
        "Custom Python code: chromadb, rank_bm25, OpenAI API.  \n"
        "No framework. Every step is explicit and transparent."
    )
    v2_desc = (
        "**v2 — LlamaIndex pipeline**  \n"
        "Framework-based: VectorStoreIndex, QueryFusionRetriever,  \n"
        "RetrieverQueryEngine with citation PromptTemplate.  \n"
        "*Requires running `python scripts/run_v2_index.py` once.*"
    )

    choice = st.radio(
        "version",
        options=["v1 — Manual pipeline", "v2 — LlamaIndex pipeline"],
        index=0 if st.session_state.get("pipeline_version", "v1") == "v1" else 1,
        label_visibility="collapsed",
    )

    with st.expander("What's the difference?", expanded=False):
        st.markdown(
            "| | v1 | v2 |\n"
            "|---|---|---|\n"
            "| Retrieval | Custom RRF | QueryFusionRetriever |\n"
            "| Generation | Raw OpenAI call | RetrieverQueryEngine |\n"
            "| BM25 | rank_bm25 class | BM25LlamaRetriever |\n"
            "| Eval | RAGAS | LlamaIndex evaluators |\n"
        )

    st.markdown("")
    if st.button("Confirm", type="primary", use_container_width=True):
        st.session_state["pipeline_version"] = (
            "v1" if choice.startswith("v1") else "v2"
        )
        st.rerun()


# Show dialog on first load
if "pipeline_version" not in st.session_state:
    _show_version_picker()
    st.stop()

pipeline_version: str = st.session_state["pipeline_version"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retrieve helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _embed_v1(question: str) -> List[float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    resp = client.embeddings.create(model="text-embedding-3-small", input=[question])
    return resp.data[0].embedding


def _retrieve_v1(
    question: str,
    strategy: str,
    top_k: int,
    collection,
    bm25: Any,
    bm25_rows: List[Dict],
    k_rrf: int = 60,
    pool: int = 20,
) -> List[Dict[str, Any]]:
    if strategy == "bm25":
        return bm25.search(question, top_k=top_k)

    q_vec = _embed_v1(question)

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

    # hybrid — RRF fusion
    n         = pool if strategy == "hybrid_rerank" else top_k
    sem_res   = collection.query(query_embeddings=[q_vec], n_results=n)
    sem_hits  = [
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
    id_map   = {c["chunk_id"]: c for c in bm25_rows}
    ranked   = sorted(fused, key=fused.__getitem__, reverse=True)
    return [
        {"id": d, "text": id_map[d]["text"],
         "metadata": id_map[d]["metadata"], "rrf_score": fused[d]}
        for d in ranked if d in id_map
    ][:top_k]


def _retrieve_v2_and_generate(
    question: str,
    strategy: str,
    top_k: int,
    v2_index,
    v2_nodes,
):
    """Run v2 retrieve + generate, return (hits_list, answer, cited_indices, latency_ms)."""
    retriever = get_v2_retriever(
        index    = v2_index,
        nodes    = v2_nodes,
        strategy = strategy,
        top_k    = top_k,
    )
    engine = build_v2_query_engine(retriever)

    t0 = time.perf_counter()
    result = query_with_timing(engine, question)
    # latency is already inside result.latency_ms

    hits = [
        {
            "id":       n["node_id"],
            "text":     n["text"],
            "metadata": n["metadata"],
            "score":    n["score"],
        }
        for n in result.source_nodes
    ]
    return hits, result.answer, result.cited_indices, result.latency_ms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    # ── version badge + change button ───────────────────────────────────────
    version_label = (
        "🔵 v1 — Manual pipeline" if pipeline_version == "v1"
        else "🟠 v2 — LlamaIndex pipeline"
    )
    st.markdown(f"### {version_label}")
    if st.button("🔄 Change version", use_container_width=True):
        _show_version_picker()

    st.markdown("---")
    st.title("⚙️ Settings")

    strategy = st.selectbox(
        "Retrieval strategy",
        options=["hybrid", "semantic", "bm25"],
        index=0,
        help=(
            "**hybrid** — RRF fusion of dense + BM25. Best live option.\n\n"
            "**semantic** — Dense OpenAI embeddings only.\n\n"
            "**bm25** — Keyword matching, no API call.\n\n"
            "*hybrid_rerank: benchmarked but excluded from live UI "
            "(PyTorch crash on Windows/Py3.14 inside Streamlit).*"
        ),
    )

    top_k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=10, value=5)

    st.markdown("---")
    st.markdown("**Indexed filing**")
    st.markdown("🍎 **Apple Inc. (AAPL)**  \n*Technology · FY 2025*")

    # ── eval metrics ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Retrieval eval (50 gold Qs)**")

    if pipeline_version == "v1":
        _rsum = ROOT / "data" / "eval" / "retrieval_summary.json"
        if _rsum.is_file():
            try:
                _evals = json.loads(_rsum.read_text())
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
                _rg = json.loads(_rgas.read_text())
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

    else:  # v2 sidebar metrics
        _v2_rsum = ROOT / "data" / "eval" / "v2_retrieval_summary.json"
        if _v2_rsum.is_file():
            try:
                _evals = json.loads(_v2_rsum.read_text())
                for row in _evals:
                    badge = " ✅" if row.get("strategy") == strategy else ""
                    st.markdown(
                        f"`{row.get('strategy','?')}`{badge}  "
                        f"R@5={row.get('recall@5','?')}  MRR={row.get('mrr','?')}"
                    )
            except Exception:
                st.caption("Could not load v2_retrieval_summary.json")
        else:
            st.caption("Run scripts/run_v2_retrieval_eval.py to populate.")

        _v2_gen = ROOT / "data" / "eval" / "v2_generation_summary.json"
        if _v2_gen.is_file():
            try:
                _rg = json.loads(_v2_gen.read_text())
                if "error" not in _rg:
                    st.markdown("---")
                    st.markdown("**LlamaIndex eval scores (20 Qs)**")
                    for k in ("faithfulness", "relevancy"):
                        if _rg.get(k) is not None:
                            label = k.replace("_", " ").title()
                            st.markdown(f"- {label}: **{_rg[k]}**")
            except Exception:
                pass

    st.markdown("---")
    if pipeline_version == "v1":
        st.caption("v1 · No framework · Custom code · GPT-4o-mini")
    else:
        st.caption("v2 · LlamaIndex 0.14 · QueryFusionRetriever · GPT-4o-mini")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main page header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

version_badge = (
    ":blue[v1 Manual]" if pipeline_version == "v1" else ":orange[v2 LlamaIndex]"
)
st.title(f"📄 Apple 10-K RAG Q&A  {version_badge}")
st.markdown(
    "Ask any question about **Apple's 2025 annual report (10-K)**. "
    "Answers are grounded in retrieved passages with inline citations."
)

# ── input row ─────────────────────────────────────────────────────────────
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

# ── API key guard ─────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY") and strategy in ("semantic", "hybrid"):
    st.error(
        "OPENAI_API_KEY is not set. "
        "Add `OPENAI_API_KEY=sk-...` to your `.env` file and restart Streamlit."
    )
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline dispatch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

hits: List[Dict[str, Any]] = []
answer          = ""
cited_indices:  List[int]   = []
retrieval_ms    = 0.0
generation_ms   = 0.0

# ─── v1 branch ──────────────────────────────────────────────────────────────
if pipeline_version == "v1":
    collection, _col_err = _get_v1_collection()
    if _col_err:
        st.error(f"Could not connect to v1 ChromaDB: {_col_err}")
        st.stop()

    bm25_v1, bm25_rows, _bm_err = _get_v1_bm25()
    if _bm_err:
        st.error(f"Could not build v1 BM25 index: {_bm_err}")
        st.stop()

    with st.spinner(f"**v1** — Retrieving with **{strategy}** …"):
        try:
            t0 = time.perf_counter()
            hits = _retrieve_v1(question, strategy, top_k, collection, bm25_v1, bm25_rows)
            retrieval_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            st.error(f"v1 Retrieval failed: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            st.stop()

    if not hits:
        st.warning("No chunks retrieved. Try a different question or strategy.")
        st.stop()

    with st.spinner("**v1** — Generating answer with GPT-4o-mini …"):
        try:
            gen_result: GenerationResult = generate_answer(question, hits)
            answer        = gen_result.answer
            cited_indices = gen_result.cited_indices
            generation_ms = gen_result.latency_ms
        except Exception as exc:
            st.error(f"v1 Generation failed: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            st.stop()

# ─── v2 branch ──────────────────────────────────────────────────────────────
else:
    if not _V2_OK:
        st.error(f"v2 pipeline could not be loaded: {_V2_ERR}")
        st.stop()

    v2_index, _v2_err = _get_v2_index()
    if _v2_err:
        st.error(_v2_err)
        st.info(
            "To build the v2 index, open a terminal and run:\n\n"
            "```bash\n$env:PYTHONPATH = '.;src'\n"
            "python scripts/run_v2_index.py\n```\n\n"
            "This only needs to run once. The index is then reused on every subsequent launch."
        )
        st.stop()

    v2_nodes, _nd_err = _get_v2_nodes()
    if _nd_err:
        st.error(f"Could not load v2 nodes: {_nd_err}")
        st.stop()

    with st.spinner(f"**v2** — Retrieving + generating with **{strategy}** …"):
        try:
            hits, answer, cited_indices, generation_ms = _retrieve_v2_and_generate(
                question, strategy, top_k, v2_index, v2_nodes
            )
            # v2 combines retrieve + generate in one call; retrieval ms is embedded inside
            retrieval_ms = 0.0
        except Exception as exc:
            st.error(f"v2 query failed: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            st.stop()

    if not hits:
        st.warning("No chunks retrieved by v2. Try a different question or strategy.")
        st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Answer display  (identical layout for both pipelines)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("---")
st.subheader("Answer")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pipeline",   pipeline_version.upper())
c2.metric("Strategy",   strategy)
c3.metric("Chunks",     len(hits))
c4.metric("Retrieval",  f"{retrieval_ms:.0f} ms" if retrieval_ms > 0 else "—")
c5.metric("Generation", f"{generation_ms:.0f} ms")

st.markdown(answer)

# ── sources ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Sources  ({len(hits)} retrieved chunks)")

for i, chunk in enumerate(hits, start=1):
    meta      = chunk.get("metadata") or {}
    item      = meta.get("item", "")
    title     = meta.get("section_title", "")
    strategy_ = meta.get("chunk_strategy", "")
    chars     = meta.get("char_count", len(chunk.get("text") or ""))
    text      = (chunk.get("text") or chunk.get("document") or "").strip()
    is_cited  = i in cited_indices

    marker = " 📌 cited" if is_cited else ""
    header = f"[{i}] Apple · {item}" + (f" — {title}" if title else "") + marker

    with st.expander(header, expanded=is_cited):
        st.markdown(
            f"<div style='background:#f0f4ff;padding:12px;border-radius:6px;"
            f"font-size:0.88rem;line-height:1.5'>{text}</div>",
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        d1.caption(f"Strategy: {strategy_ or pipeline_version}")
        d2.caption(f"Chars: {chars}")
        _sk = (
            "rerank_score" if "rerank_score" in chunk else
            "rrf_score"    if "rrf_score"    in chunk else
            "score"        if "score"        in chunk else
            "distance"
        )
        if _sk in chunk:
            d3.caption(f"{_sk}: {chunk[_sk]:.4f}")
