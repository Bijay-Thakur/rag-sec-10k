# RAG-SEC-10K: Production Retrieval-Augmented Generation over SEC 10-K Filings

A production-grade RAG system that answers questions about Apple's 2025 annual report (10-K filing) with cited, source-grounded answers. Built without LangChain or LlamaIndex to demonstrate fundamentals.

---

## Problem

Investors, analysts, and finance students spend 3 to 6 hours reading a single 10-K to answer questions like *"What are this company's top risks?"* or *"How did R&D spending change year-over-year?"* Each filing runs 100 to 300 pages and the relevant information is scattered across narrative sections, tables, and footnotes.

This project builds a system that:

- Ingests raw 10-K filings from SEC EDGAR
- Retrieves the exact passages relevant to a user's question
- Generates an answer grounded in those passages, with inline citations back to the source

**Target users:** equity analysts, finance students, and any analyst doing due diligence on a public company.

---

## Architecture

At a high level:

1. **Ingestion.** Parse 10-K HTML filings from SEC EDGAR, extract `ITEM`-based sections with metadata (company, part, item, title), chunk with three strategies (semantic / recursive / fixed-size), embed with OpenAI `text-embedding-3-small`, and store in ChromaDB.
2. **Retrieval.** Four strategies compared: dense semantic search, BM25 lexical search, hybrid Reciprocal Rank Fusion (RRF), and hybrid RRF followed by a cross-encoder reranker.
3. **Generation.** A citation-grounded prompt forces the LLM (GPT-4o-mini) to cite every claim with `[1]`, `[2]` etc. A regex parser maps citations back to source chunks for display.
4. **Evaluation.** 50 hand-curated gold Q&A pairs (Apple 2025 10-K) measured on Recall@k and MRR across all retrieval strategies; RAGAS scores for generation quality.
5. **UI.** Streamlit frontend with strategy selector, inline citations, and expandable source chunks.

---

## Dataset

**Apple Inc. (AAPL) — FY 2025 10-K**, downloaded from [SEC EDGAR](https://www.sec.gov/edgar).

| Company | Ticker | Sector     | Fiscal Year | Chunks (semantic) |
|---------|--------|------------|-------------|-------------------|
| Apple   | AAPL   | Technology | 2025        | ~563              |

The filing is parsed into Part/Item sections, chunked with three strategies, and embedded with OpenAI `text-embedding-3-small`. The gold evaluation set (50 hand-curated Q&A pairs) is also Apple-only.

> The pipeline is designed to scale to any number of filings — adding a new company is a single `python src/Embed/embed.py` run against its HTML file.

---

## Project Status

| Milestone                              | Status          |
| -------------------------------------- | --------------- |
| Ingestion + Part/Item section extraction | **Done**      |
| Three chunking strategies with metadata | **Done**       |
| Dense Chroma (semantic) retrieval       | **Done**       |
| BM25 lexical retrieval                  | **Done**       |
| Hybrid RRF retrieval                    | **Done**       |
| Cross-encoder reranker                  | **Done**       |
| 50-question gold eval set (Apple 2025)  | **Done**       |
| Retrieval metrics on gold set           | **Done**       |
| Citation-grounded generation            | **Done**       |
| RAGAS generation eval                   | **Done**       |
| README with full results                | **Done**       |
| Streamlit demo                          | **Done**       |

---

## Retrieval Evaluation Results

Evaluated on **50 hand-curated questions** from the Apple 2025 10-K.  
Gold chunk IDs come from `data/eval/gold_questions/apple_2025_10k_gold_eval_50_chunked_minimal.jsonl`.  
All numbers produced by `scripts/run_retrieval_eval.py` against the live ChromaDB index — no numbers are fabricated.

### Metrics explained

| Metric | Definition |
|--------|------------|
| **R@k** | Fraction of questions where **any** gold chunk (primary + supporting) appears in the top-k results |
| **pR@k** | Same but only counting **primary** gold chunks (stricter) |
| **MRR** | Mean Reciprocal Rank of the first gold chunk hit |
| **pMRR** | MRR using only primary gold chunks |
| **ms/q** | Mean wall-clock time per query (excludes one-time model load) |

### Results table (top-10 candidate pool)

| Strategy        | R@1  | R@5  | R@10 | MRR    | pR@1 | pR@5 | pR@10 | pMRR   | ms/q   |
|-----------------|------|------|------|--------|------|------|-------|--------|--------|
| **semantic**    | 0.66 | 0.86 | 0.92 | 0.7412 | 0.62 | 0.82 | 0.90  | 0.7041 | 3.8    |
| **bm25**        | 0.50 | 0.78 | 0.82 | 0.6168 | 0.50 | 0.74 | 0.80  | 0.6102 | 11.5   |
| **hybrid**      | 0.64 | 0.86 | 0.90 | 0.7324 | 0.62 | 0.82 | 0.86  | 0.7034 | 16.7   |
| **hybrid+rerank** | **0.74** | **0.88** | **0.92** | **0.8032** | **0.72** | **0.84** | **0.88** | **0.7732** | 1056   |

**Key takeaways:**

- **Semantic search alone is the strongest single strategy** — dense OpenAI embeddings capture paraphrase and concept-level similarity that BM25 misses entirely.
- **BM25 underperforms** on this eval set because the gold questions are written in natural language, not as keyword phrases. It still has a role as a fallback for exact-term queries.
- **Hybrid RRF** does not beat semantic alone here — the BM25 component introduces noise that dilutes the semantic signal for these question types.
- **Hybrid + cross-encoder reranker** achieves the best R@1 (+8 pp over semantic), best MRR (0.80 vs 0.74), and best pR@5/pMRR — at the cost of ~1 s latency due to cross-encoder inference. This is the recommended strategy for production where latency budget allows.
- The **1 s reranker latency** is dominated by CPU-based cross-encoder inference over 20 candidates; a GPU deployment would reduce this to ~50 ms.

Full per-question results in `data/eval/retrieval_results.json`.

---

## Generation & RAGAS Evaluation

The generation layer (`src/generation/generator.py`) takes a question and a list of retrieved chunks and calls **GPT-4o-mini** with a citation-grounded system prompt that:

1. Restricts the model to only the provided context passages.
2. Requires every factual claim to be cited inline as `[1]`, `[2]`, etc.
3. Returns a clean refusal if context is insufficient.

RAGAS scores below were measured on **20 questions** from the Apple 2025 gold eval set (same question pool, same `hybrid_rerank` retrieval, same GPT-4o-mini generation).  
Generated by `scripts/run_ragas_eval.py` against the live system — no numbers fabricated.

### RAGAS Results (hybrid\_rerank + GPT-4o-mini)

| Metric                  | Score  | What it measures |
|-------------------------|--------|------------------|
| **Faithfulness**        | **0.99** | Are all claims in the answer supported by the retrieved context? |
| **Answer Relevancy**    | **0.83** | Does the answer actually address the question asked? |
| **Context Recall**      | **0.84** | Did the retriever surface all information needed to answer? |
| **Context Precision**   | **0.86** | Is the retrieved context relevant (signal-to-noise ratio)? |

**Key takeaways:**

- **Faithfulness of 0.99** confirms the citation-grounded prompt is working — the model is almost never fabricating claims outside the retrieved passages.
- **Answer Relevancy of 0.83** shows the answers are directly on-topic; some questions have multi-part answers where the model addresses all parts but RAGAS counts the secondary parts as slight drift.
- **Context Recall of 0.84** aligns with the retrieval R@5 of 0.88 — occasional misses occur when the relevant chunk is outside the top-5 window.
- **Context Precision of 0.86** reflects that hybrid + rerank produces a tight, relevant candidate set with limited noise.

Full per-question RAGAS scores in `data/eval/ragas_results.json`.

---

## Chunking Strategies

Three strategies are implemented in `src/ingestion/chunkers.py`:

| Strategy | Chunk target | Overlap | How it splits |
|----------|-------------|---------|---------------|
| **semantic** | 700–1300 chars | none | Sentence stream; breaks when Jaccard similarity of word-token sets drops below 0.18 |
| **recursive_hierarchical** | ~1200 chars | none | Paragraph -> sentence -> hard-cut, recursively |
| **fixed_size** | ~1200 chars | ~300 chars sentence-aligned | Fixed window with sentence-level overlap carry-over |

Semantic and recursive chunks are embedded and stored as separate Chroma collections (`semantic_index`, `recursive_index`). The gold eval runs against `semantic_index` (Apple 2025 10-K).

---

## Tech Stack

| Layer | Library |
|-------|---------|
| HTML parsing | `beautifulsoup4` + `lxml` |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Vector store | `chromadb` (cosine HNSW) |
| Lexical search | `rank_bm25` (BM25Okapi) |
| Reranker | `sentence-transformers` — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | OpenAI `gpt-4o-mini` with citation-grounded prompt |
| Generation eval | `ragas` 0.4 (Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision) |
| UI | `streamlit` |
| Testing | `pytest` |

No LangChain or LlamaIndex. Every component is built directly on primitive libraries so the behavior is transparent and debuggable.

---

## Installation

```bash
git clone https://github.com/Bijay-Thakur/rag-sec-10k
cd rag-sec-10k
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY
```

---

## Usage

### Ingestion + indexing (run once)

```bash
# Chunk the Apple HTML filing (writes data/chunks/*.jsonl)
$env:PYTHONPATH = "src"
python src/ingestion/chunkers.py

# Embed and store in ChromaDB (run once; skips if already indexed)
python src/Embed/embed.py

# Force a full rebuild
python src/Embed/embed.py --force
```

### Query from CLI

```bash
$env:PYTHONPATH = "src"

# Ingest a single filing into a named collection
python -m cli.rag ingest --html Apple --strategy semantic

# Query that collection
python -m cli.rag query "What are the main risk factors?" -k 5

# Force re-ingest (overwrite existing collection)
python -m cli.rag ingest --html Apple --strategy semantic --force
```

### Run retrieval evaluation

```bash
$env:PYTHONPATH = "src"
python scripts/run_retrieval_eval.py
# Results written to data/eval/retrieval_summary.json and retrieval_results.json
```

### Run RAGAS generation evaluation

```bash
$env:PYTHONPATH = "src"
python scripts/run_ragas_eval.py              # all 50 questions
python scripts/run_ragas_eval.py --limit 20   # faster: first 20 questions
# Results written to data/eval/ragas_summary.json and ragas_results.json
```

### Launch the Streamlit UI

```bash
$env:PYTHONPATH = "src"
streamlit run streamlit_app.py
# Opens http://localhost:8501
```

> **Available strategies in the live UI:** `hybrid`, `semantic`, `bm25`.  
> `hybrid_rerank` is benchmarked (results shown in the sidebar) but excluded from  
> live queries because PyTorch causes a native crash inside Streamlit on  
> Python 3.14 / Windows. On Linux/macOS this limitation does not apply.

> **Available strategies in the UI:** `hybrid`, `semantic`, `bm25`.  
> `hybrid_rerank` is benchmarked (results in sidebar) but excluded from live queries --  
> PyTorch triggers a native crash inside Streamlit on Python 3.14 / Windows.  
> On Linux/macOS or with PyTorch >= 2.5 this limitation does not apply.

### Inspect section extraction

```bash
python src/ingestion/html_loader.py data/raw/Apple.html
python src/ingestion/validate_sections.py
```

---

## Project Structure

```text
rag-sec-10k/
├── data/
│   ├── raw/                          # Original 10-K HTML files (gitignored)
│   ├── chunks/                       # semantic_chunks.jsonl, recursive_chunks.jsonl
│   ├── eval/
│   │   ├── gold_questions/           # 50-question Apple gold eval set
│   │   ├── retrieval_results.json    # per-question retrieval hit lists (generated)
│   │   ├── retrieval_summary.json    # aggregated retrieval metrics (generated)
│   │   ├── ragas_results.json        # per-question RAGAS scores (generated)
│   │   └── ragas_summary.json        # mean RAGAS scores (generated)
│   └── processed/                    # Structured section text per company
├── db/                               # ChromaDB persistent store
├── notebooks/
│   └── retrieval_eval.ipynb          # Visual eval report with charts
├── scripts/
│   ├── run_retrieval_eval.py         # Gold-set retrieval benchmark (all 4 strategies)
│   ├── run_ragas_eval.py             # End-to-end RAG quality eval (RAGAS)
│   ├── benchmark_retrieval_strategies.py  # Synthetic span-query benchmark
│   └── verify_retrieval_stack.py     # Smoke test
├── src/
│   ├── ingestion/
│   │   ├── html_loader.py            # Part+Item section extractor
│   │   ├── chunkers.py               # Three chunking strategies
│   │   └── validate_sections.py      # Per-filing QA report
│   ├── Embed/
│   │   └── embed.py                  # Batch embedding + Chroma indexing (embed-once)
│   ├── retrieval/
│   │   ├── retriever.py              # semantic / BM25 / hybrid / hybrid_rerank
│   │   └── retrieve.py               # CLI-oriented single-collection retrieval
│   ├── generation/
│   │   └── generator.py              # Citation-grounded prompt + GPT-4o-mini call
│   └── cli/
│       └── rag.py                    # ingest + query CLI
├── streamlit_app.py                  # Streamlit Q&A demo (Apple 2025 10-K, hybrid/semantic/bm25)
├── tests/
│   ├── test_retriever_strategies_compare.py
│   ├── test_retrieval_smoke.py
│   └── test_extractor_sizes.py
├── requirements.txt
└── README.md
```


---

## Design Decisions

**No LangChain or LlamaIndex (v1).** Frameworks hide behavior. Writing retrieval, chunking, and prompt orchestration by hand makes every line debuggable and forces real understanding.

**Filings as HTML, not PDF.** SEC EDGAR's native format is HTML (inline XBRL). Parsing HTML preserves structure that PDF extraction loses.

**Hierarchical Part->Item section extraction before chunking.** Every 10-K has a fixed SEC Regulation S-K structure. The ingestion layer detects Parts and Items from the document itself, deduplicates ToC artifacts, and assigns each Item to its enclosing Part. This preserves topical context at retrieval time.

**Inline XBRL tags unwrapped, not stripped.** Modern filings wrap financial facts in `<ix:*>` tags. Stripping with `decompose()` deletes the underlying text; `unwrap()` keeps human-readable content while discarding the XBRL wrappers.

**Embed once, skip on re-run.** `embed.py` checks `collection.count() > 0` before calling the OpenAI API. Pass `--force` to rebuild. This prevents accidental re-spend on embeddings.
