# RAG-SEC-10K: Production Retrieval-Augmented Generation over SEC 10-K Filings

A production-grade RAG system that answers questions about public company annual reports (10-K filings) with cited, source-grounded answers. Built without LangChain or LlamaIndex to demonstrate fundamentals.

---

## Problem

Investors, analysts, and finance students spend 3 to 6 hours reading a single 10-K to answer questions like *"What are this company's top risks?"* or *"How did R&D spending change year-over-year?"* Each filing runs 100 to 300 pages and the relevant information is scattered across narrative sections, tables, and footnotes.

This project builds a system that:

- Ingests raw 10-K filings from SEC EDGAR
- Retrieves the exact passages relevant to a user's question
- Generates an answer grounded in those passages, with inline citations back to the source

**Target users:** equity analysts, finance students, and any analyst doing due diligence on a public company.

---

## Demo

> Coming in Week 4. Live URL plus a 2-minute Loom walkthrough.

---

## Architecture

> Coming in Week 2. Will include an Excalidraw diagram of the ingestion, retrieval, generation, and evaluation pipeline.

At a high level:

1. **Ingestion.** Parse 10-K HTML filings from SEC EDGAR, extract `ITEM`-based sections with metadata (company, part, item, title), chunk semantically, embed, and store in ChromaDB.
2. **Retrieval.** Hybrid search combining dense embeddings and BM25, fused via reciprocal rank fusion, then reranked with a cross-encoder.
3. **Generation.** Context-aware prompt forces the LLM to cite retrieved chunks. A parser extracts citations and surfaces them in the response.
4. **Evaluation.** 50+ hand-curated gold Q&A pairs measured on retrieval precision@k, context recall, answer faithfulness, and answer relevance using RAGAS.
5. **API + UI.** FastAPI backend, Streamlit frontend with clickable citations.

---

## Dataset

Five 10-K filings across diverse sectors, downloaded from [SEC EDGAR](https://www.sec.gov/edgar):

| Company        | Ticker | Sector             | Fiscal Year |
| -------------- | ------ | ------------------ | ----------- |
| Apple          | AAPL   | Technology         | 2025        |
| Walmart        | WMT    | Retail             | 2025        |
| JPMorgan Chase | JPM    | Financial Services | 2025        |
| Eli Lilly      | LLY    | Pharmaceuticals    | 2025        |
| Exxon Mobil    | XOM    | Energy             | 2025        |

Sector diversity was intentional. It stress-tests the pipeline (banks structure filings very differently from tech companies) and produces a more realistic eval set.

---

## Status

| Week | Focus                                                                       | State           |
| ---- | --------------------------------------------------------------------------- | --------------- |
| 1    | Ingestion, Part to Item section extraction, chunking, embeddings, ChromaDB  | **in progress** |
| 2    | Hybrid retrieval, reranker, FastAPI endpoints, citation-grounded generation | planned         |
| 3    | Gold evaluation set (50+ Q&A pairs), RAGAS metrics, ablation experiments    | planned         |
| 4    | Frontend, Docker, deployment, demo                                          | planned         |

---

## Tech stack

- **Language:** Python 3.11+
- **Ingestion:** BeautifulSoup (`lxml` parser), tiktoken
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** ChromaDB
- **Retrieval:** `rank_bm25`, `sentence-transformers` (cross-encoder reranker)
- **LLM:** Anthropic Claude or OpenAI GPT-4 class
- **API:** FastAPI + Pydantic
- **Evaluation:** RAGAS
- **Frontend:** Streamlit (may switch to React)
- **Deployment:** Docker + Railway / Fly.io

No LangChain or LlamaIndex. Every component is built directly on primitive libraries so the behavior is transparent and debuggable.

---

## Installation

> Full install instructions coming in Week 4. For now, development setup:

```bash
git clone https://github.com/Bijay-Thakur/rag-sec-10k
cd rag-sec-10k
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # add your API keys
```

---

## Usage

> CLI and API examples coming at the end of Week 2. Current read-only entry points:

```bash
# Inspect section extraction on a single filing
python src/ingestion/html_loader.py data/raw/Apple.html

# Run the validation report across all filings in data/raw/
python src/ingestion/validate_sections.py
```

The validation report prints the detected Part and Item hierarchy per filing, flags any numbering gaps, and reports duplicates suppressed by the dedup layer.

---

## Evaluation

> Results table coming in Week 3. Will report retrieval precision@k, context recall, answer faithfulness, and answer relevance across chunk-size and reranker ablations.

---

## Design decisions and tradeoffs

This section will grow throughout the project. Current decisions:

**No LangChain or LlamaIndex (v1).** Frameworks hide behavior. Writing retrieval, chunking, and prompt orchestration by hand makes every line debuggable and forces real understanding. A v2 refactor using LlamaIndex is planned to compare lines of code and performance.

**Filings as HTML, not PDF.** SEC EDGAR's native format is HTML (inline XBRL). Parsing HTML preserves structure (headings, tables, lists, and XBRL-tagged facts) that PDF extraction loses.

**Hierarchical `Part` to `Item` section extraction.** 10-Ks have a fixed structure defined by SEC Regulation S-K. Every filing has *Item 1A. Risk Factors*, *Item 7. MD&A*, and so on, grouped under *Part I* through *Part IV*. The ingestion layer detects Parts and Items **from the document itself** (not a hardcoded mapping), assigns each Item to its enclosing Part, and deduplicates against Table-of-Contents and running-header artifacts. This produces semantically meaningful sections *before* any size-based chunking, which preserves topical context at retrieval time. Filings that omit explicit `PART` markers (for example JPMorgan Chase) fall through with `part=None` and are handled at the retrieval layer.

**Inline XBRL tags unwrapped, not stripped.** Modern 10-K filings wrap financial facts in `<ix:*>` tags. Stripping these with `decompose()` deletes the underlying text; we `unwrap()` instead, keeping the human-readable content and discarding the XBRL wrappers.

**Financial statement sections (Items 8, 15, 16) excluded from v1 index.** These contain dense financial tables that need structured extraction rather than text chunking. Handling them would require a separate tabular-data pipeline; including them raw would pollute retrieval with numerical noise. Narrative sections (*Business*, *Risk Factors*, *MD&A*) contain the substance of most analyst questions and are the focus of v1.

---

## Project structure

```text
rag-sec-10k/
├── data/
│   ├── raw/                     # Original 10-K HTML files (gitignored)
│   └── processed/               # (Week 1) Cleaned, sectioned output
├── src/
│   ├── ingestion/
│   │   ├── html_loader.py       # Part+Item hierarchical section extractor
│   │   └── validate_sections.py # Per-filing structure report / QA harness
│   ├── retrieval/               # (Week 2) Hybrid search + reranker
│   ├── generation/              # (Week 2) Prompt templates, citation parser
│   └── evaluation/              # (Week 3) Metrics, gold dataset, eval runner
├── tests/
├── notebooks/                   # Exploratory analysis
├── requirements.txt
├── docker-compose.yml           # (Week 4)
├── .env.example
└── README.md
```

---

