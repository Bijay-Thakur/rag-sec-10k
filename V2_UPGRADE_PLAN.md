# SEC Insight AI — V2 Upgrade Plan

> Technical audit and production upgrade roadmap for transforming the existing **SEC 10-K RAG** project into **SEC Insight AI**, a production-style flagship RAG application.

**Audit date:** 2026-06-08  
**Scope:** Full repository inspection — no application code modified during this audit.

---

## Executive Summary

The current repository is a **well-measured research/prototype RAG system** with two parallel implementations (v1 manual Python + v2 LlamaIndex), strong retrieval benchmarks (Recall@5 = 0.86, hybrid+rerank R@1 = 0.74), and high generation faithfulness (RAGAS 0.99). It is **not yet production-ready** as a multi-tenant, deployable product.

**Strengths to preserve:**
- SEC-aware HTML ingestion with Part/Item segmentation and XBRL handling
- Hybrid retrieval (semantic + BM25 + RRF) with measured cross-encoder reranking
- Citation-grounded generation with 50-question gold eval set
- Dual-pipeline comparison (debuggability vs. framework extensibility)

**Primary gaps for "SEC Insight AI":**
- Single-company (Apple), single-filing scope in UI and eval
- No automated EDGAR ingestion or orchestration
- No API layer, auth, observability, or deployment infrastructure
- Dual v1/v2 codepaths create maintenance burden
- Hardcoded config, local Chroma only, no CI/CD

---

## 1. Current Architecture Summary

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  data/raw/*.html  (manual placement — no EDGAR downloader)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INGESTION  src/ingestion/html_loader.py                                     │
│  BeautifulSoup + lxml · XBRL ix:* unwrap · Part/Item/synthetic headings     │
│  QA: validate_sections.py · tests: 5 filers (Apple, Chase, Exxon, …)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHUNKING  src/ingestion/chunkers.py                                         │
│  semantic (Jaccard 700–1300 chars) · recursive_hierarchical · fixed_size    │
│  → data/chunks/semantic_chunks.jsonl (~2,812 Apple chunks)                  │
│  → data/chunks/recursive_chunks.jsonl (embedded but not default path)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INDEXING  src/Embed/embed.py                                                │
│  OpenAI text-embedding-3-small (1536-d) → ChromaDB db/ (cosine HNSW)        │
│  Collections: semantic_index, recursive_index · embed-once guard            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                          ▼
┌───────────────────────┐                          ┌───────────────────────────┐
│ v1  src/retrieval/    │                          │ v2  v2/retrieval/         │
│ retriever.py          │                          │ retrievers.py             │
│ semantic · bm25 ·     │                          │ VectorIndexRetriever ·    │
│ hybrid RRF ·          │                          │ BM25LlamaRetriever ·      │
│ cross-encoder rerank  │                          │ QueryFusionRetriever ·    │
└───────────┬───────────┘                          │ SentenceTransformerRerank │
            ▼                                      └─────────────┬─────────────┘
┌───────────────────────┐                          ┌─────────────▼─────────────┐
│ v1  src/generation/   │                          │ v2  v2/generation/        │
│ generator.py          │                          │ query_engine.py           │
│ GPT-4o-mini + [n]     │                          │ RetrieverQueryEngine      │
└───────────┬───────────┘                          └─────────────┬─────────────┘
            └───────────────────────┬──────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  UI  streamlit_app.py  (Apple-only · v1/v2 picker · hybrid/semantic/bm25)   │
│  EVAL  scripts/run_*_eval.py · notebooks · data/eval/*.json                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Baseline Metrics (Apple 2025 10-K, 50 gold questions)

| Layer | Metric | Value |
|-------|--------|-------|
| Retrieval (hybrid) | Recall@5 / MRR | 0.86 / 0.73 |
| Retrieval (hybrid+rerank) | Recall@1 / MRR | 0.74 / 0.80 |
| Generation (v1 RAGAS, n=20) | Faithfulness / Relevancy | 0.99 / 0.83 |
| Generation (v2 LI, n=20) | Faithfulness / Relevancy | 0.90 / 0.95 |
| E2E latency | ~3 s (hybrid, no rerank in UI) | |

---

## 2. Technical Audit

### 2.1 Folder Structure

```text
SEC_10K_RAG_Q&A/
├── data/
│   ├── raw/                    # gitignored — manual HTML placement
│   ├── chunks/                 # semantic + recursive JSONL (committed)
│   ├── processed/structured/   # gitignored — 5-filer artifacts, no generator
│   └── eval/
│       ├── gold_questions/     # 50 Apple gold Qs (rich schema)
│       └── *.json              # committed eval results
├── db/                         # gitignored — Chroma persistent store
├── src/                        # v1 manual pipeline
│   ├── ingestion/
│   ├── Embed/
│   ├── retrieval/
│   ├── generation/
│   └── cli/
├── v2/                         # v2 LlamaIndex pipeline (mirrors src/)
│   ├── ingestion/
│   ├── indexing/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
├── scripts/                    # eval + index runners (not package entrypoints)
├── notebooks/                  # retrieval_eval, v2_comparison
├── tests/                      # 2 test files (ingestion + retriever smoke)
├── streamlit_app.py            # single-file UI at repo root
├── requirements.txt
├── .env.example                # OPENAI_API_KEY only
└── README.md
```

**Observations:**
- Flat, script-oriented layout — no `pyproject.toml`, no package namespace
- `src/` and `v2/` duplicate concerns; `PYTHONPATH=.;src` required everywhere
- `data/processed/structured/` exists for 5 filers but no code generates it
- No `Dockerfile`, `.github/workflows`, `docker-compose`, or `Makefile`
- `db/` contains multiple Chroma segment UUIDs (artifact of rebuilds)

---

### 2.2 Ingestion Pipeline

| Aspect | Status | Details |
|--------|--------|---------|
| SEC download | **Missing** | HTML must be manually placed in `data/raw/` |
| HTML parsing | **Complete** | `src/ingestion/html_loader.py` — 220 lines |
| XBRL handling | **Complete** | `ix:*` tags unwrapped (not decomposed) |
| Section detection | **Complete** | PART/Item regex + synthetic headings (MD&A, etc.) |
| Multi-filer support | **Partial** | Parser tested on 5 filers; pipeline defaults to Apple |
| Structured export | **Missing** | `data/processed/structured/*.json` has no generator |
| v2 wrapper | **Complete** | `v2/ingestion/html_reader.py` → LlamaIndex `Document` |

**Data flow:** `load_html()` → `clean_soup()` → `extract_sections()` → list of `{part, item, section_title, text, metadata}` dicts.

**QA tooling:** `validate_sections.py` (CLI report), `tests/test_extractor_sizes.py` (regression on section char counts).

---

### 2.3 Chunking / Parsing Logic

| Strategy | File | Parameters | Persisted? | Used in prod path? |
|----------|------|------------|------------|-------------------|
| `semantic` | `chunkers.py` | 700–1300 chars, Jaccard threshold 0.18 | Yes → `semantic_chunks.jsonl` | **Yes** (default) |
| `recursive_hierarchical` | `chunkers.py` | target 1200 chars, part→item→para→sentence | Yes → `recursive_chunks.jsonl` | No (embedded only) |
| `fixed_size` | `chunkers.py` | 1200 chars, 300 overlap | No | No |
| v2 `SentenceSplitter` | `v2/ingestion/node_parser.py` | 512 tokens, 50 overlap | Via `run_v2_index.py` only | No (live path reuses v1 chunks) |

**Chunk ID scheme (v1):** `{source_slug}_{item_slug}_chunk_{NNN}` (e.g. `apple_item1a_chunk_006`).

**Metadata per chunk:** `source_file`, `part`, `item`, `section_title`, `chunk_strategy`, `char_count`, `token_count`.

**Gap:** Gold eval set references v1 chunk IDs exclusively. v2 native node IDs (`source::part::item::idx`) are incompatible without text-overlap matching (implemented in `v2/evaluation/eval_retrieval.py` but **not wired** into `scripts/run_v2_retrieval_eval.py`).

---

### 2.4 Embedding / Indexing Logic

| Component | v1 | v2 |
|-----------|----|----|
| Embed model | `text-embedding-3-small` | Same via `OpenAIEmbedding` |
| Vector store | Chroma `PersistentClient` at `db/` | Same, collection `v2_semantic_index` |
| Collections | `semantic_index`, `recursive_index` | `v2_semantic_index` (+ fallback to v1) |
| Batch size | 64 (manual) | LlamaIndex auto-batch |
| Embed-once guard | `collection.count() > 0` | Same pattern |
| Rebuild flag | `--force` | `--force` |

**Alternate path:** `src/retrieval/retrieve.py` implements `retrieve_fused()` — RRF across `semantic_index` + `recursive_index`. **Not used** in Streamlit or gold eval.

**Gap:** No metadata filtering at query time (e.g. filter by `item`, `part`, `source_file`) despite Chroma supporting `where` clauses in `retrieve.py`.

---

### 2.5 BM25 / Vector Retrieval / RRF / Reranking

#### v1 (`src/retrieval/retriever.py`)

| Strategy | Implementation | Latency (50 Q avg) |
|----------|----------------|-------------------|
| `semantic` | OpenAI embed → Chroma query | 4 ms |
| `bm25` | `rank_bm25.BM25Okapi`, whitespace tokenization | 12 ms |
| `hybrid` | Manual RRF, k=60 | 17 ms |
| `hybrid_rerank` | RRF pool=20 → `cross-encoder/ms-marco-MiniLM-L-6-v2` | 1,056 ms |

#### v2 (`v2/retrieval/retrievers.py`)

| Strategy | Implementation |
|----------|----------------|
| `semantic` | `VectorIndexRetriever` |
| `bm25` | Custom `BM25LlamaRetriever` (rank_bm25 wrapper) |
| `hybrid` | `QueryFusionRetriever(mode="reciprocal_rerank")` |
| `hybrid_rerank` | Fusion + `SentenceTransformerRerank` post-processor |

**Where strategies run:**

| Context | Strategies available |
|---------|---------------------|
| Streamlit UI | hybrid, semantic, bm25 only |
| v1 eval script | all four |
| v2 eval script | semantic, bm25, hybrid (no rerank) |
| RAGAS eval | hybrid_rerank |

**Known issue:** Cross-encoder reranker causes PyTorch access violation inside Streamlit on Windows/Python 3.14 — intentionally disabled in live UI.

---

### 2.6 LLM Answer Generation

| Aspect | v1 (`generator.py`) | v2 (`query_engine.py`) |
|--------|---------------------|------------------------|
| Model | `gpt-4o-mini`, temp=0 | Same |
| Prompt | System + numbered context blocks | `PromptTemplate` with identical rules |
| Citations | Regex extract `[n]` from answer | Same |
| Refusal | Fixed sentence if insufficient context | Same |
| Output | `GenerationResult` dataclass | `V2QueryResult` dataclass |
| Framework | Raw OpenAI API | `RetrieverQueryEngine` + compact synthesizer |

**No streaming, no conversation history, no query rewriting, no structured output (JSON).**

---

### 2.7 Evaluation Scripts and Metrics

| Script | Pipeline | Metrics | Output |
|--------|----------|---------|--------|
| `scripts/run_retrieval_eval.py` | v1 | R@1/5/10, primary R@k, MRR, latency | `retrieval_*.json` |
| `scripts/run_ragas_eval.py` | v1 E2E | faithfulness, answer_relevancy, context_recall, context_precision | `ragas_*.json` |
| `scripts/run_v2_retrieval_eval.py` | v2 | Same as v1 (ID matching on v1 chunks) | `v2_retrieval_*.json` |
| `scripts/run_v2_generation_eval.py` | v2 E2E | faithfulness, relevancy (LI evaluators) | `v2_generation_*.json` |
| `scripts/print_all_metrics.py` | — | Consolidated stdout dump | — |
| `notebooks/retrieval_eval.ipynb` | v1 | Charts, heatmaps | — |
| `notebooks/v2_comparison.ipynb` | v1 vs v2 | Side-by-side | — |

**Gold set:** `data/eval/gold_questions/apple_2025_10k_gold_eval_50_chunked_minimal.jsonl`
- Rich schema: `question_id`, `category`, `difficulty`, `gold_chunk_ids`, `primary_gold_chunk_ids`, `expected_answer`, `expected_keywords`, `gold_items`
- **Apple-only**, hand-curated, chunk-ID-linked

**Test coverage:** 2 test files — ingestion regression (5 filers) + retriever strategy smoke test. **No tests** for generation, eval scripts, or Streamlit.

---

### 2.8 Frontend

**Single file:** `streamlit_app.py` (~620 lines)

| Feature | Status |
|---------|--------|
| v1/v2 pipeline picker (modal) | Done |
| Strategy selector (hybrid/semantic/bm25) | Done |
| Top-k slider | Done |
| Example questions | Done (Apple-specific) |
| Cited answer + source panels | Done |
| Eval metric badges (from JSON) | Done |
| Multi-company selector | **Missing** |
| Filing/year selector | **Missing** |
| Chat history | **Missing** |
| Streaming responses | **Missing** |
| Export/share answers | **Missing** |
| Auth | **Missing** |

No React/Next.js, no API-backed frontend, no mobile layout considerations.

---

### 2.9 Missing Production Features

| Category | Gap | Severity |
|----------|-----|----------|
| **Data acquisition** | No EDGAR API/downloader (CIK lookup, 10-K fetch, rate limiting) | High |
| **Orchestration** | No pipeline DAG (ingest → chunk → embed → index as jobs) | High |
| **API layer** | No REST/GraphQL; Streamlit-only | High |
| **Multi-tenancy** | Single filing, single company | High |
| **Config management** | Hardcoded models, paths, k_rrf=60; only `OPENAI_API_KEY` in env | Medium |
| **Observability** | No structured logging, tracing, or cost tracking | High |
| **Deployment** | No Docker, K8s, CI/CD, health checks | High |
| **Auth & security** | No API keys, RBAC, input sanitization, rate limiting | High |
| **Vector store scaling** | Local Chroma only; no managed option (Pinecone, pgvector) | Medium |
| **BM25 persistence** | In-memory rebuild on every Streamlit session | Medium |
| **Reranker in prod** | Disabled in UI due to PyTorch/Streamlit crash | Medium |
| **Metadata filtering** | Chroma `where` supported but unused | Medium |
| **Caching** | Streamlit `@cache_resource` only | Medium |
| **Error handling** | Minimal; no retry/backoff on OpenAI calls (except LI v2 embed) | Medium |
| **Documentation** | Strong README; no API docs, architecture ADRs, runbooks | Low |
| **CI/CD** | No GitHub Actions, no lint/format hooks | Medium |
| **Package structure** | No installable package; path hacks everywhere | Medium |
| **Dual pipeline debt** | v1 + v2 maintained in parallel | Medium |

---

### 2.10 Recommended V2 Architecture (SEC Insight AI)

Consolidate on **one production pipeline** (LlamaIndex-based, retaining v1 as a `legacy/` reference or removing after parity tests pass). Wrap it in a proper service layer.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SEC Insight AI — Target Architecture                 │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐
  │  Web UI      │     │  REST API    │     │  CLI                         │
  │  (Next.js or │     │  FastAPI     │     │  sec-insight ingest/query    │
  │   Streamlit+)│     │  /v1/query   │     │                              │
  └──────┬───────┘     └──────┬───────┘     └──────────────┬───────────────┘
         │                    │                            │
         └────────────────────┼────────────────────────────┘
                              ▼
         ┌────────────────────────────────────────────────────────┐
         │  sec_insight/  (installable Python package)             │
         │  ├── api/          routes, schemas, middleware          │
         │  ├── core/         config, logging, exceptions          │
         │  ├── ingestion/    EDGAR fetch + html_loader (merged)   │
         │  ├── processing/   chunkers + node_parser (unified)     │
         │  ├── indexing/     embed + Chroma/pgvector adapter      │
         │  ├── retrieval/    retriever factory (single impl)      │
         │  ├── generation/   query engine + citation parser       │
         │  ├── evaluation/   unified eval harness                   │
         │  └── jobs/         Celery/Arq tasks for async pipelines   │
         └────────────────────────┬───────────────────────────────┘
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  Data Layer                                             │
         │  ├── PostgreSQL   filings registry, job status, users   │
         │  ├── Chroma/pgvector   vectors (per-filing collections) │
         │  ├── Redis          BM25 cache, query cache, rate limits│
         │  └── Object storage  raw HTML, chunk JSONL (S3/local)   │
         └────────────────────────────────────────────────────────┘
                                  ▲
         ┌────────────────────────┴───────────────────────────────┐
         │  Ingestion Worker                                         │
         │  EDGAR API → HTML → sections → chunks → embed → index    │
         └──────────────────────────────────────────────────────────┘
```

**Key architectural decisions:**

1. **Single pipeline** — LlamaIndex core with v1 algorithms preserved where LI lacks equivalents (custom BM25, citation prompt).
2. **Per-filing collections** — `{ticker}_{fiscal_year}_semantic` instead of one global `semantic_index`.
3. **FastAPI service** — `/query`, `/ingest`, `/filings`, `/health`, `/metrics`.
4. **EDGAR integration** — SEC EDGAR full-text search API or `sec-edgar-downloader` with User-Agent compliance.
5. **Reranker as optional microservice** — isolate PyTorch cross-encoder from Streamlit threading (HTTP sidecar or ONNX runtime).
6. **Unified config** — `pydantic-settings` with `.env` + YAML profiles (dev/staging/prod).
7. **Observability** — OpenTelemetry traces, structured JSON logs, Langfuse or Phoenix for RAG-specific tracing.

---

## 3. Production Gaps (Prioritized)

### P0 — Must-have for "production-style flagship"

1. Installable package with clean imports (eliminate `PYTHONPATH` hacks)
2. FastAPI REST API with OpenAPI docs
3. EDGAR filing fetcher (at minimum: CIK → latest 10-K HTML)
4. Multi-filing index management (company + fiscal year registry)
5. Docker Compose for local prod-like stack (API + Chroma + optional Redis)
6. CI pipeline (pytest, lint, eval smoke on PR)
7. Consolidated single pipeline (deprecate dual v1/v2 maintenance)
8. Structured logging + request tracing

### P1 — Strong production signals

9. Rebrand UI to "SEC Insight AI" with company/filing selector
10. Persistent BM25 index (pickle or dedicated search engine)
11. Metadata-filtered retrieval (pre-filter by Item for section-aware Q&A)
12. Reranker sidecar or ONNX deployment (restore R@1=0.74 in prod)
13. Query/embedding cache (Redis)
14. Cost and latency dashboards
15. Expand gold eval to 2+ companies

### P2 — Portfolio polish

16. Next.js frontend (or significantly upgraded Streamlit multi-page app)
17. Streaming generation (SSE)
18. Conversation history with citation continuity
19. Managed vector store option (pgvector)
20. Auth (API keys or OAuth)
21. ADRs and runbooks

---

## 4. Exact File Changes Needed

### Phase 0 — Restructure & Rebrand (no behavior change)

| Action | Files |
|--------|-------|
| **Create** | `pyproject.toml` — package `sec_insight`, entrypoints |
| **Create** | `sec_insight/__init__.py`, `sec_insight/core/config.py` |
| **Move** | `src/ingestion/*` → `sec_insight/ingestion/` |
| **Move** | `src/Embed/embed.py` → `sec_insight/indexing/chroma_index.py` |
| **Move** | `src/retrieval/retriever.py` → `sec_insight/retrieval/retriever.py` |
| **Move** | `src/generation/generator.py` → `sec_insight/generation/generator.py` |
| **Move** | `v2/retrieval/retrievers.py` → merge into `sec_insight/retrieval/` |
| **Move** | `v2/generation/query_engine.py` → `sec_insight/generation/query_engine.py` |
| **Move** | `v2/indexing/build_index.py` → `sec_insight/indexing/llama_index.py` |
| **Move** | `scripts/*.py` → `sec_insight/evaluation/scripts/` or `scripts/` with proper imports |
| **Move** | `tests/*` → `tests/` (update imports to `sec_insight.*`) |
| **Archive** | `src/`, `v2/` → `legacy/` (keep for 1 release, then delete) |
| **Update** | `README.md` — rebrand to SEC Insight AI |
| **Update** | `.env.example` — add all config vars (see below) |
| **Create** | `config/default.yaml` — models, chunk params, retrieval defaults |

### Phase 1 — API Layer

| Action | Files |
|--------|-------|
| **Create** | `sec_insight/api/main.py` — FastAPI app factory |
| **Create** | `sec_insight/api/routes/query.py` — POST `/v1/query` |
| **Create** | `sec_insight/api/routes/filings.py` — GET `/v1/filings`, POST `/v1/filings/ingest` |
| **Create** | `sec_insight/api/routes/health.py` — GET `/health`, `/ready` |
| **Create** | `sec_insight/api/schemas.py` — Pydantic request/response models |
| **Create** | `sec_insight/api/deps.py` — DI for retriever, index, config |
| **Create** | `Dockerfile`, `docker-compose.yml` |
| **Update** | `streamlit_app.py` → `apps/streamlit_app.py` — call API instead of direct imports |
| **Update** | `requirements.txt` — add `fastapi`, `uvicorn`, `pydantic-settings` |

### Phase 2 — EDGAR Ingestion & Multi-Filing

| Action | Files |
|--------|-------|
| **Create** | `sec_insight/ingestion/edgar_client.py` — fetch 10-K by CIK/ticker |
| **Create** | `sec_insight/ingestion/filing_registry.py` — track ingested filings in SQLite/Postgres |
| **Create** | `sec_insight/jobs/ingest_pipeline.py` — orchestrated ingest → chunk → embed |
| **Update** | `sec_insight/indexing/chroma_index.py` — per-filing collection naming |
| **Update** | `sec_insight/retrieval/retriever.py` — accept `filing_id` param, metadata filters |
| **Update** | `sec_insight/ingestion/chunkers.py` — batch process all filings in registry |
| **Create** | `data/filings_registry.json` or DB migration for filing metadata |

### Phase 3 — Production Hardening

| Action | Files |
|--------|-------|
| **Create** | `sec_insight/core/logging.py` — structlog/json logging |
| **Create** | `sec_insight/core/telemetry.py` — OpenTelemetry setup |
| **Create** | `sec_insight/retrieval/bm25_store.py` — persistent BM25 (pickle per collection) |
| **Create** | `services/reranker/` — optional FastAPI reranker microservice |
| **Create** | `.github/workflows/ci.yml` — lint, test, eval smoke |
| **Create** | `tests/test_api.py`, `tests/test_edgar_client.py` |
| **Update** | `sec_insight/evaluation/` — unified eval harness (single entrypoint) |
| **Create** | `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md` |

### Phase 4 — UI Upgrade

| Action | Files |
|--------|-------|
| **Update** | `apps/streamlit_app.py` — SEC Insight AI branding, company/filing picker |
| **Create** | `apps/web/` (optional) — Next.js frontend consuming `/v1/query` |
| **Update** | Page title, example questions per company |

### Config additions for `.env.example`

```env
OPENAI_API_KEY=
SEC_EDGAR_USER_AGENT=YourName your@email.com   # SEC requirement
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o-mini
CHROMA_PERSIST_DIR=./db
DEFAULT_RETRIEVAL_STRATEGY=hybrid
RRF_K=60
RERANKER_ENABLED=false
RERANKER_SERVICE_URL=http://reranker:8001
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## 5. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Dual-pipeline refactor breaks eval parity** | Loss of benchmark credibility | Run v1 vs consolidated pipeline A/B on gold set before deleting `legacy/` |
| **EDGAR rate limiting / ToS violations** | Blocked IP, legal issues | Respect 10 req/s limit; set User-Agent; cache filings locally |
| **Chroma per-filing collections don't scale** | Slow startup, disk bloat | Migrate to pgvector or single collection with metadata filters at 50+ filings |
| **Reranker PyTorch crash persists** | Can't ship R@1=0.74 in UI | Deploy reranker as separate process; evaluate ONNX Runtime cross-encoder |
| **Gold eval Apple-only** | Overfit to one filing's structure | Expand gold set to Chase/Walmart; re-benchmark after multi-filing |
| **OpenAI cost at scale** | Embedding 100+ filings adds up | Embed-once per filing; cache query embeddings; consider local embed model |
| **Chunk ID schema change** | Breaks gold eval linkage | Version chunk schema; maintain ID mapping table or text-overlap eval fallback |
| **Scope creep on frontend** | Delays core pipeline | Ship FastAPI + upgraded Streamlit first; Next.js as P2 |
| **LlamaIndex version drift** | Breaking API changes | Pin versions in `pyproject.toml`; abstract LI behind internal interfaces |
| **Windows dev vs Linux deploy** | Path/PyTorch issues | Docker-first dev; CI on Linux |

---

## 6. Ordered Implementation Plan

### Milestone 1 — Foundation (Week 1–2)
**Goal:** Installable package, single import path, no behavior change.

- [ ] Add `pyproject.toml` with `sec_insight` package
- [ ] Move v1 + v2 modules into unified `sec_insight/` namespace
- [ ] Fix all imports; remove `PYTHONPATH` requirement
- [ ] Archive `src/` and `v2/` to `legacy/`
- [ ] Update README branding to SEC Insight AI
- [ ] Verify all existing tests pass
- [ ] Verify eval scripts produce identical metrics

**Exit criteria:** `pip install -e .` + `sec-insight eval retrieval` reproduces current numbers.

---

### Milestone 2 — API Shell (Week 2–3)
**Goal:** FastAPI service wrapping existing RAG pipeline.

- [ ] Implement `POST /v1/query` (question, filing_id, strategy, top_k)
- [ ] Implement `GET /health`, `GET /v1/filings` (static Apple entry initially)
- [ ] Add Pydantic schemas for request/response with citations
- [ ] Docker Compose: API + Chroma volume mount
- [ ] Refactor Streamlit to call local API
- [ ] Add basic structured logging

**Exit criteria:** Streamlit works exclusively through API; same answers as direct imports.

---

### Milestone 3 — EDGAR + Multi-Filing (Week 3–5)
**Goal:** Automated ingestion for any public company 10-K.

- [ ] Build `edgar_client.py` (ticker → CIK → latest 10-K HTML)
- [ ] Filing registry (SQLite initially)
- [ ] Per-filing Chroma collections
- [ ] Ingest CLI/API: `POST /v1/filings/ingest {ticker, form_type, year}`
- [ ] UI company/filing selector
- [ ] Ingest remaining 4 test filers (Chase, Exxon, Elilily, Walmart)

**Exit criteria:** User selects Walmart 10-K in UI and gets grounded answers with citations.

---

### Milestone 4 — Production Hardening (Week 5–7)
**Goal:** CI/CD, observability, reranker, persistent BM25.

- [ ] GitHub Actions CI (pytest + lint + retrieval eval smoke)
- [ ] Persistent BM25 store (avoid rebuild per request)
- [ ] Reranker sidecar or ONNX integration
- [ ] Metadata filtering (optional `items` filter on query)
- [ ] OpenTelemetry tracing on query path
- [ ] Unified eval CLI with HTML report output

**Exit criteria:** hybrid_rerank available in prod API; CI green on every PR.

---

### Milestone 5 — Flagship Polish (Week 7–10)
**Goal:** Portfolio-ready SEC Insight AI.

- [ ] Expand gold eval to 2+ companies (25 Qs each minimum)
- [ ] Rebrand Streamlit UI (SEC Insight AI theme, landing page)
- [ ] Architecture docs + demo video script
- [ ] Optional Next.js frontend (if time permits)
- [ ] Remove `legacy/` after 30-day parity window
- [ ] Deploy demo to cloud (Railway/Fly.io/AWS)

**Exit criteria:** Public demo URL, README with architecture diagram and live metrics.

---

## 7. What to Keep vs. Deprecate

| Keep | Deprecate / Merge |
|------|-------------------|
| `html_loader.py` section extraction logic | Dual v1/v2 pipeline directories |
| Semantic chunking as default strategy | `fixed_size` unless benchmarked |
| Hybrid RRF retrieval | `retrieve_fused()` unless recursive index proves value |
| Citation-grounded prompt (v1 wording) | Duplicate retrieval logic in Streamlit |
| 50-question gold eval set | v2 text-overlap eval (use unified ID scheme instead) |
| Cross-encoder reranker (eval + prod sidecar) | In-process reranker in Streamlit |
| RAGAS eval metrics | Separate v1/v2 eval scripts (unify) |
| Ingestion regression tests | Manual HTML placement workflow |

---

## 8. Success Metrics for SEC Insight AI V2

| Metric | Current | Target |
|--------|---------|--------|
| Companies queryable | 1 (Apple) | 10+ |
| Ingestion | Manual | Automated EDGAR |
| Deployment | Local Streamlit | Docker + cloud demo |
| API | None | FastAPI with OpenAPI |
| Retrieval R@5 (Apple) | 0.86 | ≥ 0.86 (no regression) |
| R@1 with reranker in prod | N/A (disabled) | ≥ 0.74 |
| Faithfulness | 0.99 | ≥ 0.95 |
| CI | None | Green on every PR |
| Test files | 2 | 15+ |
| Time-to-first-query (new filing) | Manual hours | < 5 min automated |

---

## Appendix A — Key File Reference (Current State)

| File | Lines (approx) | Role |
|------|----------------|------|
| `src/ingestion/html_loader.py` | 220 | SEC HTML → Part/Item sections |
| `src/ingestion/chunkers.py` | 430 | 3 chunking strategies |
| `src/Embed/embed.py` | 155 | Batch embed → Chroma |
| `src/retrieval/retriever.py` | 324 | Semantic/BM25/RRF/rerank |
| `src/generation/generator.py` | 193 | GPT-4o-mini + citations |
| `v2/retrieval/retrievers.py` | 179 | LlamaIndex retriever factory |
| `v2/generation/query_engine.py` | 183 | RetrieverQueryEngine |
| `v2/indexing/build_index.py` | 160 | VectorStoreIndex builder |
| `streamlit_app.py` | 620 | Full Q&A UI |
| `scripts/run_retrieval_eval.py` | 313 | v1 retrieval benchmark |
| `scripts/run_ragas_eval.py` | 236 | v1 RAGAS benchmark |

**Total core pipeline:** ~546 lines (v1) + ~426 lines (v2) = significant duplication to eliminate.

---

## Appendix B — Environment & Dependencies (Current)

**Required env:** `OPENAI_API_KEY` only.

**Key dependencies:** chromadb, openai, beautifulsoup4, lxml, rank-bm25, sentence-transformers, ragas, streamlit, llama-index-core + adapters.

**Not present:** fastapi, uvicorn, pydantic-settings, redis, sqlalchemy, docker, pytest-cov, ruff, pre-commit, httpx (for EDGAR).

---

*This document is the authoritative upgrade plan. Implementation should proceed milestone-by-milestone with eval parity gates between each phase.*
