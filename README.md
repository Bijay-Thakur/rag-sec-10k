# SEC 10-K RAG: Production Q&A over Annual Reports

> A measured, production-grade Retrieval-Augmented Generation system that answers questions about Apple's 2025 10-K filing with source-grounded, cited answers — implemented **twice**: once from scratch in pure Python, once with LlamaIndex, with a full empirical comparison between the two.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![LlamaIndex 0.14](https://img.shields.io/badge/LlamaIndex-0.14-orange.svg)](https://docs.llamaindex.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-green.svg)](https://www.trychroma.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-black.svg)](https://platform.openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

---

## TL;DR

| | **Headline number** |
|---|---|
| Hybrid retrieval **Recall@5** | **0.86** (50 hand-curated gold questions) |
| Reranker **Recall@1** | **0.74** (+8 pp over semantic alone) |
| RAGAS **Faithfulness** | **0.99** — answers don't hallucinate |
| LlamaIndex **Relevancy** | **0.95** |
| End-to-end query latency | ~3 s (retrieve + GPT-4o-mini) |
| Pipeline implementations | **2** (manual + LlamaIndex), benchmarked side-by-side |

---

## Why this project

Reading a single 10-K to answer a specific question — *"What tariff risks does Apple disclose?"* or *"How did R&D spending change year-over-year?"* — takes equity analysts **3–6 hours per filing**. The filing runs 100–300 pages; the relevant fact lives in a single paragraph buried in a section called `PART I, Item 1A` or `PART II, Item 7`.

This project builds the system you actually want for that workflow:

- Type a question. Get an answer in **3 seconds**.
- Every claim has an inline `[1]` citation pointing back to the exact passage.
- The retrieved chunks are shown so you can verify the model's work.
- The whole pipeline is **measured against a 50-question gold set** — so you know exactly how reliable it is.

And because architectural decisions matter, the entire pipeline is implemented twice:
- **v1** — from-scratch Python (no framework). Every line is debuggable.
- **v2** — LlamaIndex. Less code, same quality, framework-grade extensibility.

You can switch between them at runtime with a single click in the Streamlit UI.

---

## Live demo

```bash
$env:PYTHONPATH = ".;src"
streamlit run streamlit_app.py
```

The app opens with a version picker dialog: **v1 (manual)** or **v2 (LlamaIndex)**. Both produce identical retrieval quality on the same data — the comparison is about engineering trade-offs (lines of code, extensibility, debuggability), not numbers.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Apple 2025 10-K (HTML)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
                  ┌──────────────────────────────────────┐
                  │  Ingestion  (Part/Item segmentation) │
                  │  beautifulsoup4 + lxml + custom regex │
                  └──────────────────┬───────────────────┘
                                     ▼
                  ┌──────────────────────────────────────┐
                  │  Chunking — 3 strategies              │
                  │  · semantic (Jaccard boundary, 700-1300 chars) │
                  │  · recursive_hierarchical             │
                  │  · fixed_size                         │
                  └──────────────────┬───────────────────┘
                                     ▼
                  ┌──────────────────────────────────────┐
                  │  Embedding  text-embedding-3-small   │
                  │  ChromaDB (cosine HNSW, 1536-dim)    │
                  │  2,812 vectors · embed-once          │
                  └──────────────────┬───────────────────┘
                                     ▼
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                          ▼
┌────────────────────┐                              ┌──────────────────────┐
│ v1 Retrieval        │                              │ v2 Retrieval (LI)    │
│ semantic / bm25 /   │                              │ VectorIndexRetriever │
│ hybrid (RRF) /      │                              │ QueryFusionRetriever │
│ + cross-encoder     │                              │ BM25LlamaRetriever   │
└──────────┬─────────┘                              └──────────┬───────────┘
           ▼                                                    ▼
┌──────────────────────┐                          ┌──────────────────────────┐
│ v1 Generation         │                          │ v2 Generation             │
│ Raw OpenAI call +     │                          │ RetrieverQueryEngine +    │
│ citation-grounded     │                          │ PromptTemplate            │
│ prompt + regex parser │                          │ (citation-grounded)       │
└──────────┬───────────┘                          └──────────────┬───────────┘
           └───────────────────────┬──────────────────────────────┘
                                   ▼
                  ┌──────────────────────────────────────┐
                  │  Streamlit UI                         │
                  │  · pop-up version picker (v1/v2)      │
                  │  · strategy selector                  │
                  │  · cited answer + source panels       │
                  └──────────────────────────────────────┘
```

---

## Results — real numbers, not marketing

All metrics below are produced by running the eval scripts against the live ChromaDB and the gold question set. No numbers are fabricated.

### Retrieval (50 hand-curated gold questions, Apple 2025 10-K)

| Strategy            | R@1  | R@5  | R@10 | MRR    | Latency (ms/q) |
|---------------------|------|------|------|--------|----------------|
| semantic            | 0.66 | 0.86 | 0.92 | 0.7412 |   4 ms         |
| bm25                | 0.50 | 0.78 | 0.82 | 0.6168 |  12 ms         |
| hybrid (RRF)        | 0.64 | 0.86 | 0.90 | 0.7324 |  17 ms         |
| **hybrid + rerank** | **0.74** | **0.88** | 0.92 | **0.8032** | 1,056 ms |

> **Headline:** the cross-encoder reranker delivers **+8 pp R@1** and **MRR 0.80** at the cost of ~1 s/query (CPU). On GPU, that latency drops to ~50 ms.

### Generation quality (20 questions, GPT-4o-mini, hybrid retrieval)

| | **v1 — RAGAS** | **v2 — LlamaIndex evaluators** |
|---|---|---|
| Faithfulness     | **0.99** | **0.90** |
| Answer Relevancy | **0.83** | **0.95** |
| Context Recall   | 0.84 | _(not implemented in LI)_ |
| Context Precision| 0.86 | _(not implemented in LI)_ |

> **Faithfulness 0.99** means the model essentially never hallucinates claims outside the retrieved context — exactly what citation-grounded prompts are designed to enforce. RAGAS and LlamaIndex use different LLM-as-judge methodologies (NLI decomposition vs. direct scoring), so the absolute numbers differ by ~10 pp on the same answers; both confirm the system is highly grounded.

### v1 vs v2 — engineering comparison

| Dimension | v1 (manual)                                  | v2 (LlamaIndex)                          |
|-----------|----------------------------------------------|------------------------------------------|
| Pipeline core (retrieval + generation + indexing) | 546 lines | 426 lines (**-22%**)         |
| Hybrid RRF | ~40-line manual rank-fusion loop           | One `QueryFusionRetriever(mode="reciprocal_rerank")` |
| Re-ranking | Direct `cross_encoder.predict()`           | `SentenceTransformerRerank` post-processor |
| Generation | Raw OpenAI call + manual prompt format     | `RetrieverQueryEngine` + `PromptTemplate` |
| Eval — retrieval | Custom R@k / MRR loops               | Same metrics, integrated harness         |
| Eval — generation | RAGAS (Faithfulness / Relevancy / Recall / Precision) | LlamaIndex `FaithfulnessEvaluator` + `RelevancyEvaluator` |
| Same retrieval quality? | **Yes — both share the same Chroma vectors** | (numerically identical) |
| When to choose | Learning, debugging, full transparency | Production, extensibility, agent workflows |

Both versions use the same underlying ChromaDB collection, so retrieval **R@k** and **MRR** are numerically identical. The interesting comparison is **developer experience**: v2 is ~22% less code with the same quality, while v1 stays maximally debuggable because every step is explicit.

---

## Project status

| Milestone | Status |
|---|---|
| HTML ingestion + Part/Item section extraction | Done |
| Three chunking strategies with structured metadata | Done |
| Dense Chroma (semantic) retrieval | Done |
| BM25 lexical retrieval | Done |
| Hybrid RRF retrieval | Done |
| Cross-encoder reranker (CPU) | Done |
| 50-question gold eval set (Apple 2025) | Done |
| Retrieval metrics (R@k, MRR) on gold set | Done |
| Citation-grounded generation with GPT-4o-mini | Done |
| RAGAS generation eval (4 metrics) | Done |
| **LlamaIndex v2 pipeline (full parity)** | **Done** |
| **v2 retrieval + generation eval** | **Done** |
| **Streamlit UI with v1/v2 version picker** | **Done** |
| **README with full real results** | **Done** |

---

## Tech stack

| Layer | Technology | Why |
|-------|------------|-----|
| HTML parsing | `beautifulsoup4` + `lxml` | XBRL-aware, structure-preserving |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) | Strong recall/$ ratio |
| Vector store | `chromadb` (cosine HNSW) | Persistent, batteries-included |
| Lexical search | `rank_bm25` (BM25Okapi) | Catches exact terms semantic misses |
| Reranker | `sentence-transformers` — `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M params, MS-MARCO trained |
| Generation | OpenAI `gpt-4o-mini` | Cheap, fast, grounded with citation prompt |
| Generation eval | `ragas` 0.4 + LlamaIndex evaluators | Two independent LLM-as-judge stacks |
| Framework (v2) | `llama-index-core` 0.14 + Chroma adapter | Production composition primitives |
| UI | `streamlit` 1.57 | Single-file Python app, no JS |
| Testing | `pytest` | Ingestion regression tests |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Bijay-Thakur/rag-sec-10k.git
cd rag-sec-10k

# 2. Environment
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # macOS / Linux

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env                   # add OPENAI_API_KEY

# 5. Build the index (one-time, ~30s; embed-once guard skips on reruns)
$env:PYTHONPATH = ".;src"
python src/Embed/embed.py

# 6. Launch the app
streamlit run streamlit_app.py
```

---

## Full usage

### Build / rebuild the index

```bash
python src/Embed/embed.py            # skip if already populated
python src/Embed/embed.py --force    # rebuild from data/chunks/*.jsonl
```

### CLI Q&A (no UI)

```bash
python -m cli.rag ingest --html Apple --strategy semantic
python -m cli.rag query "What are Apple's main risk factors?" -k 5
```

### Retrieval evaluation — v1

```bash
python scripts/run_retrieval_eval.py
# Outputs:
#   data/eval/retrieval_summary.json
#   data/eval/retrieval_results.json
```

### Retrieval evaluation — v2

```bash
python scripts/run_v2_retrieval_eval.py
# Outputs:
#   data/eval/v2_retrieval_summary.json
#   data/eval/v2_retrieval_results.json
```

### Generation evaluation — v1 (RAGAS)

```bash
python scripts/run_ragas_eval.py --limit 20
# Outputs:
#   data/eval/ragas_summary.json
#   data/eval/ragas_results.json
```

### Generation evaluation — v2 (LlamaIndex)

```bash
python scripts/run_v2_generation_eval.py --limit 20
# Outputs:
#   data/eval/v2_generation_summary.json
#   data/eval/v2_generation_results.json
```

### Print all metrics at once

```bash
python scripts/print_all_metrics.py
```

---

## Project structure

```text
rag-sec-10k/
├── data/
│   ├── raw/                          # Source 10-K HTML (gitignored)
│   ├── chunks/                       # Chunk JSONL by strategy
│   └── eval/
│       ├── gold_questions/           # 50-question hand-curated gold set
│       ├── retrieval_summary.json    # v1 retrieval metrics
│       ├── retrieval_results.json    #   per-question detail
│       ├── ragas_summary.json        # v1 generation metrics (RAGAS)
│       ├── ragas_results.json        #   per-question detail
│       ├── v2_retrieval_summary.json # v2 retrieval metrics
│       ├── v2_retrieval_results.json #   per-question detail
│       ├── v2_generation_summary.json# v2 generation metrics (LI evaluators)
│       └── v2_generation_results.json#   per-question detail
├── db/                               # ChromaDB persistent store (gitignored)
├── notebooks/
│   ├── retrieval_eval.ipynb          # v1 charts + per-question heatmaps
│   └── v2_comparison.ipynb           # v1 vs v2 side-by-side analysis
├── scripts/
│   ├── run_retrieval_eval.py         # v1 retrieval eval (50 gold Qs)
│   ├── run_ragas_eval.py             # v1 generation eval (RAGAS)
│   ├── run_v2_index.py               # build v2 LlamaIndex index
│   ├── run_v2_retrieval_eval.py      # v2 retrieval eval
│   ├── run_v2_generation_eval.py     # v2 generation eval
│   └── print_all_metrics.py          # consolidated metrics dump
├── src/                              # v1 manual pipeline
│   ├── ingestion/
│   │   ├── html_loader.py            # Part+Item section extractor
│   │   ├── chunkers.py               # 3 chunking strategies
│   │   └── validate_sections.py      # ingestion QA report
│   ├── Embed/embed.py                # batch embed + Chroma (embed-once)
│   ├── retrieval/
│   │   ├── retriever.py              # semantic / bm25 / hybrid / hybrid_rerank
│   │   └── retrieve.py               # CLI-oriented retrieval API
│   ├── generation/generator.py       # citation-grounded prompt + GPT-4o-mini
│   └── cli/rag.py                    # `python -m cli.rag` CLI
├── v2/                               # v2 LlamaIndex pipeline (mirrors src/)
│   ├── ingestion/
│   │   ├── html_reader.py            # SEC10KReader (BaseReader wrapper)
│   │   └── node_parser.py            # SentenceSplitter → TextNodes
│   ├── indexing/build_index.py       # VectorStoreIndex + ChromaVectorStore
│   ├── retrieval/retrievers.py       # all 4 strategies via LlamaIndex primitives
│   ├── generation/query_engine.py    # RetrieverQueryEngine + PromptTemplate
│   └── evaluation/
│       ├── eval_retrieval.py         # R@k / MRR with text-overlap matching
│       └── eval_generation.py        # FaithfulnessEvaluator + RelevancyEvaluator
├── streamlit_app.py                  # UI: version picker dialog + Q&A
├── tests/
│   ├── test_extractor_sizes.py       # html_loader regression tests
│   └── test_retriever_strategies_compare.py  # strategy comparison test
├── requirements.txt
├── .env.example
└── README.md
```

---

## Design decisions (and the reasoning behind them)


**Two implementations.** The from-scratch v1 forces you to understand every step (rank fusion, citation parsing, batch embedding). The LlamaIndex v2 then proves the abstraction is worth it: ~36% less code, same retrieval quality, more composable. Recruiters can see I know **why** the framework helps, not just that it does.

**HTML over PDF.** SEC EDGAR's native format is HTML (inline XBRL). Parsing HTML preserves the Part/Item structure that PDF extraction loses.

**Hierarchical section detection before chunking.** Every 10-K has a fixed SEC Regulation S-K structure. Detecting Part/Item boundaries from the document itself — including synthetic headings like *"Management's Discussion and Analysis"* that some filers use instead of *"Item 7"* — gives every chunk topical context that retrieval can filter on.

**Inline XBRL `unwrap`, not `decompose`.** Modern filings wrap financial facts in `<ix:*>` tags. Stripping with `decompose()` deletes the underlying text; `unwrap()` keeps the human-readable content while discarding the XBRL wrappers.

**Embed-once.** Both pipelines check `collection.count() > 0` before calling the OpenAI API. Pass `--force` to rebuild. This prevents accidental re-spend on embeddings (each full rebuild = ~$0.10 against `text-embedding-3-small`).

**Citation-grounded prompt.** The system prompt enforces three rules: use only the provided context, cite every claim with `[n]`, and refuse with a fixed sentence if context is insufficient. The 0.99 RAGAS Faithfulness score confirms this works.

**Reranker is benchmarked, not served in the UI.** On Windows + Python 3.14, the cross-encoder's PyTorch native libraries trigger a hard access violation inside Streamlit's threading model. The reranker still runs in the eval scripts; the UI uses hybrid RRF (which is within 1–2 pp of the reranker on R@5).

**Text-overlap matching for v2 retrieval eval.** v2 node IDs (LlamaIndex auto-generated) differ from v1 chunk IDs, so the v2 eval matches retrieved text against gold text using a 100-character fingerprint. This is fairer than ID matching across pipeline versions — and the numbers come out **identical to v1** because the underlying vectors are the same.

---


## Contact

Built by **Bijay Thakur**. Open to AI / ML engineering roles.

- GitHub: [@Bijay-Thakur](https://github.com/Bijay-Thakur)
- Repo:   [rag-sec-10k](https://github.com/Bijay-Thakur/rag-sec-10k)
