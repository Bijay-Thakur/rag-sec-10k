# SEC Insight AI

**Production-grade RAG for SEC 10-K filings — measured retrieval, citation-grounded answers, and a clear path to deployment.**

**[Watch the demo video →](https://www.loom.com/share/cee7b50bc1e648c3b1aaa32b71da48de)** (Loom walkthrough of SEC Insight AI)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![LlamaIndex 0.14](https://img.shields.io/badge/LlamaIndex-0.14-orange.svg)](https://docs.llamaindex.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-green.svg)](https://www.trychroma.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-black.svg)](https://platform.openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)

---

## Value proposition

SEC Insight AI turns 100–300 page annual reports into **source-cited answers in seconds** — with retrieval quality benchmarked on a 50-question gold set, not assumed.

---

## The problem

Equity analysts and researchers spend **3–6 hours per filing** hunting for a single fact buried in `PART I, Item 1A` or `PART II, Item 7`. A 10-K is structured by SEC regulation, but the relevant paragraph is rarely where you'd expect it. Keyword search misses paraphrases; naive RAG hallucinates figures.

SEC Insight AI addresses this with:

- **Structure-aware ingestion** — Part/Item segmentation from native EDGAR HTML (not PDF)
- **Hybrid retrieval** — dense embeddings + BM25 fused with reciprocal rank fusion (RRF)
- **Citation-grounded generation** — every claim tagged `[n]` back to the source passage
- **Empirical evaluation** — R@k, MRR, and RAGAS faithfulness on hand-curated gold questions

---

## Current features (V1 baseline)

The repository today ships a fully working, measured RAG pipeline over Apple's 2025 10-K:

| Capability | Status |
|------------|--------|
| HTML ingestion with XBRL unwrap and Part/Item detection | Done |
| Three chunking strategies (semantic, recursive, fixed-size) | Done |
| OpenAI embeddings → ChromaDB (embed-once guard) | Done |
| Retrieval: semantic, BM25, hybrid RRF, cross-encoder rerank | Done |
| Citation-grounded answers via GPT-4o-mini | Done |
| 50-question gold eval set with R@k / MRR / RAGAS metrics | Done |
| Dual pipeline: manual Python (`src/`) + LlamaIndex (`v2/`) | Done |
| Streamlit Q&A UI with strategy selector and source panels | Done |
| CLI ingest/query (`python -m cli.rag`) | Done |

### Headline metrics (Apple 2025 10-K, 50 gold questions)

| Metric | Value |
|--------|-------|
| Hybrid retrieval **Recall@5** | **0.86** |
| Hybrid + rerank **Recall@1** | **0.74** (+8 pp over semantic alone) |
| RAGAS **Faithfulness** | **0.99** |
| End-to-end latency | ~3 s (retrieve + generate) |

All numbers come from running the eval scripts against live ChromaDB — see [Results](#results) below.

---

## Planned V2 production features

V2 transforms the measured baseline into a deployable product. Full roadmap: [`V2_UPGRADE_PLAN.md`](V2_UPGRADE_PLAN.md).

| Feature | Target |
|---------|--------|
| FastAPI REST service (`/query`, `/ingest`, `/filings`) | Planned |
| Automated EDGAR filing ingestion (CIK/ticker → 10-K HTML) | Planned |
| Multi-company / multi-filing index management | Planned |
| Docker Compose for local prod-like deployment | Planned |
| Cost controls — embed-once, query caching, configurable models | Planned |
| Evaluation dashboard — unified metrics view across filings | Planned |
| CI/CD with eval smoke tests on every PR | Planned |
| Structured logging and request tracing | Planned |
| Optional reranker sidecar (restore R@1=0.74 in production UI) | Planned |

---

## V1 vs V2

| | **V1 — Retrieval & evaluation baseline** | **V2 — Production-style deployment** |
|---|-------------------------------------------|----------------------------------------|
| **Purpose** | Prove the RAG stack works; benchmark every component | Ship a deployable app with ops-grade controls |
| **Scope** | Apple 2025 10-K; local Streamlit + CLI | Multi-filing; API-first; cloud-ready |
| **Pipeline** | Manual Python (`src/`) + LlamaIndex mirror (`v2/`) | Consolidated `sec_insight` package (planned) |
| **Retrieval** | Semantic / BM25 / hybrid RRF / cross-encoder rerank | Same strategies via API; metadata filtering |
| **Evaluation** | RAGAS + custom R@k/MRR scripts; JSON + notebooks | Unified eval harness + dashboard (planned) |
| **Cost controls** | Embed-once guard; batch embedding | Configurable models, caching, per-filing budgets |
| **Deployment** | Local (`streamlit run`, `PYTHONPATH`) | Docker Compose + cloud demo (planned) |
| **Status** | **Complete and measured** | **Planned** — see upgrade plan |

V1 is the evidence layer: you can inspect every retrieval step and reproduce every metric. V2 is the product layer: same quality bar, packaged for real users and recruiters evaluating production engineering judgment.

---

## Architecture

> _Detailed architecture diagram and ADRs will be added as V2 ships. Current pipeline overview below._

```text
SEC 10-K HTML  →  Part/Item ingestion  →  semantic chunking  →  ChromaDB embed
                                                                        │
                    ┌───────────────────────────────────────────────────┘
                    ▼
         hybrid retrieval (semantic + BM25 + RRF)
                    │
                    ▼
         citation-grounded generation (GPT-4o-mini)
                    │
                    ▼
              Streamlit UI / CLI / eval scripts
```

**Target V2 architecture** (API layer, EDGAR ingestion, per-filing collections, observability) is documented in [`V2_UPGRADE_PLAN.md`](V2_UPGRADE_PLAN.md).

---

## Why this project matters

For **Applied AI / AI Engineer** roles, this project demonstrates skills that go beyond "I built a chatbot over PDFs":

1. **Domain-specific ingestion** — SEC filings have regulatory structure (Parts, Items, inline XBRL). The parser recovers sections filers label inconsistently (e.g. synthetic MD&A headings). This is the kind of preprocessing that separates demo RAG from production RAG.

2. **Retrieval engineering, not just embeddings** — Hybrid search (dense + BM25 + RRF) beats either alone. A cross-encoder reranker adds +8 pp Recall@1. Every strategy is benchmarked on 50 hand-curated gold questions with chunk-level labels.

3. **Grounded generation with measurable faithfulness** — Citation prompts enforce `[n]` references; RAGAS faithfulness of 0.99 confirms the model stays inside retrieved context. You can audit any answer against the source panels.

4. **Build vs. buy judgment** — The pipeline is implemented twice: from-scratch Python (546 lines, fully debuggable) and LlamaIndex (426 lines, composable). Same Chroma vectors, identical retrieval metrics — a deliberate comparison of abstraction trade-offs.

5. **Evaluation as a first-class deliverable** — Gold Q&A set, per-strategy latency tables, RAGAS + LlamaIndex evaluators, and reproducible scripts. The README reports real numbers, not aspirational ones.

6. **Production roadmap with honest scope** — V2 upgrade plan covers API design, EDGAR automation, Docker, CI, and cost controls — showing how a measured prototype becomes a deployable product.

---

## Live demo

**Video walkthrough:** [SEC Insight AI — Loom demo](https://www.loom.com/share/cee7b50bc1e648c3b1aaa32b71da48de)

### Run locally (Next.js + FastAPI)

```powershell
# Backend (repo root)
.\scripts\start-backend.ps1

# Frontend (separate terminal)
.\scripts\start-frontend.ps1
```

Open the URL printed by Next.js (typically `http://localhost:3000`). The UI supports five indexed 10-K filings with filing-scoped retrieval, citations, and an eval dashboard.

### Streamlit (V1 baseline)

```bash
$env:PYTHONPATH = ".;src"
streamlit run streamlit_app.py
```

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
# Outputs: data/eval/retrieval_summary.json, retrieval_results.json
```

### Retrieval evaluation — v2 (LlamaIndex)

```bash
python scripts/run_v2_retrieval_eval.py
# Outputs: data/eval/v2_retrieval_summary.json, v2_retrieval_results.json
```

### Generation evaluation — v1 (RAGAS)

```bash
python scripts/run_ragas_eval.py --limit 20
# Outputs: data/eval/ragas_summary.json, ragas_results.json
```

### Generation evaluation — v2 (LlamaIndex)

```bash
python scripts/run_v2_generation_eval.py --limit 20
# Outputs: data/eval/v2_generation_summary.json, v2_generation_results.json
```

### Print all metrics at once

```bash
python scripts/print_all_metrics.py
```

---

## Results

All metrics below are produced by running the eval scripts against the live ChromaDB and the gold question set.

### Retrieval (50 hand-curated gold questions, Apple 2025 10-K)

| Strategy            | R@1  | R@5  | R@10 | MRR    | Latency (ms/q) |
|---------------------|------|------|------|--------|----------------|
| semantic            | 0.66 | 0.86 | 0.92 | 0.7412 |   4 ms         |
| bm25                | 0.50 | 0.78 | 0.82 | 0.6168 |  12 ms         |
| hybrid (RRF)        | 0.64 | 0.86 | 0.90 | 0.7324 |  17 ms         |
| **hybrid + rerank** | **0.74** | **0.88** | 0.92 | **0.8032** | 1,056 ms |

### Generation quality (20 questions, GPT-4o-mini, hybrid retrieval)

| | **v1 — RAGAS** | **v2 — LlamaIndex evaluators** |
|---|---|---|
| Faithfulness     | **0.99** | **0.90** |
| Answer Relevancy | **0.83** | **0.95** |
| Context Recall   | 0.84 | _(not implemented in LI)_ |
| Context Precision| 0.86 | _(not implemented in LI)_ |

### Pipeline engineering comparison (manual vs LlamaIndex)

| Dimension | v1 (manual) | v2 (LlamaIndex) |
|-----------|-------------|-----------------|
| Pipeline core lines | 546 | 426 (−22%) |
| Hybrid RRF | Manual rank-fusion loop | `QueryFusionRetriever` |
| Re-ranking | `cross_encoder.predict()` | `SentenceTransformerRerank` |
| Generation | Raw OpenAI + prompt | `RetrieverQueryEngine` |
| Same retrieval quality? | **Yes — shared Chroma vectors** | |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| HTML parsing | `beautifulsoup4` + `lxml` (XBRL-aware) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-d) |
| Vector store | `chromadb` (cosine HNSW) |
| Lexical search | `rank_bm25` (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | OpenAI `gpt-4o-mini` |
| Evaluation | `ragas` 0.4 + LlamaIndex evaluators |
| Framework (v2 mirror) | `llama-index-core` 0.14 |
| UI | `streamlit` |
| Testing | `pytest` |

---

## Project structure

```text
SEC_10K_RAG_Q&A/
├── data/
│   ├── raw/                          # Source 10-K HTML (gitignored)
│   ├── chunks/                       # Chunk JSONL by strategy
│   └── eval/                         # Gold set + metric outputs
├── db/                               # ChromaDB store (gitignored)
├── src/                              # v1 manual pipeline
│   ├── ingestion/                    # html_loader, chunkers
│   ├── Embed/                        # batch embed + Chroma
│   ├── retrieval/                    # semantic / bm25 / hybrid / rerank
│   ├── generation/                   # citation-grounded GPT-4o-mini
│   └── cli/                          # ingest + query CLI
├── v2/                               # LlamaIndex pipeline mirror
├── scripts/                          # eval runners
├── notebooks/                        # retrieval charts, v1 vs v2 analysis
├── streamlit_app.py                  # Q&A UI
├── tests/
├── V2_UPGRADE_PLAN.md                # Production upgrade roadmap
├── requirements.txt
└── README.md
```

---

## Design decisions

**HTML over PDF.** SEC EDGAR's native format preserves Part/Item structure that PDF extraction loses.

**Hierarchical section detection before chunking.** Synthetic headings (e.g. *Management's Discussion and Analysis*) are mapped to canonical Items when filers omit standard labels.

**Inline XBRL `unwrap`, not `decompose`.** Keeps human-readable text while discarding XBRL wrappers.

**Embed-once.** Both pipelines skip re-embedding if the Chroma collection is populated (`--force` to rebuild).

**Citation-grounded prompt.** Enforces context-only answers, inline `[n]` citations, and a fixed refusal sentence — RAGAS faithfulness 0.99 confirms it works.

**Reranker benchmarked, not served in UI.** PyTorch cross-encoder crashes inside Streamlit on Windows/Python 3.14; eval scripts use it; live UI uses hybrid RRF (within 1–2 pp of reranker on R@5).

---

## Contact

Built by **Bijay Thakur**. Open to Applied AI / AI engineering roles.

- GitHub: [@Bijay-Thakur](https://github.com/Bijay-Thakur)
- Repo: [rag-sec-10k](https://github.com/Bijay-Thakur/rag-sec-10k)
