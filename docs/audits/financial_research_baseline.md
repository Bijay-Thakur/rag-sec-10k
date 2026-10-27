# Financial research transformation — Phase 0 baseline

**Observed:** 2026-10-27. **Base:** `main` at `27f47a3a122c7317511ad62cdb4eba540a24a9c1`; clean before edits. **Implementation branch:** `codex/phase0-vector-integrity`. All observations below concern this checkout, not a live deployment.

## Evidence and deviations from the proposed plan

| Status | Evidence | Consequence |
| --- | --- | --- |
| VERIFIED | `cloudbuild.yaml` originally ran `scripts/ci_bootstrap_db.py` before building `backend/Dockerfile`; bootstrap inserted 2,812 chunks with `[0.0] * 1536`; Docker copied `db/`. | A build using that configuration could serve a dummy semantic index. This is a build-path finding, not proof of what a currently running revision serves. |
| VERIFIED | `backend/app/adapters/rag_adapter.py` originally accepted any nonempty Chroma collection. `src/retrieval/retriever.py` used OpenAI query embeddings against it. | Count alone did not establish vector correctness or provenance. Hybrid results could appear valid while the dense branch was meaningless. |
| VERIFIED | `db/chroma.sqlite3` is **tracked** in Git (`git ls-files db`), about 26 MB here. SQLite reports `semantic_index` dimension 1536, only `hnsw:space=cosine` collection metadata. `python scripts/check_vector_index.py --db db` returns invalid with a local Chroma `'_type'` error. | The RFC's claim that the DB was only gitignored/unuploaded is stale for this checkout. Its generation model and contents are unverified. We preserved it and excluded `db/` in both Docker ignore files. |
| VERIFIED | `backend/app/filings/registry.py` has five legacy filing IDs. `/health`, `/api/filings`, `/api/ask` are in `backend/app/main.py`; demo requests choose BM25 in the adapter. | Phase 0 preserves the old catalog and routes. |
| VERIFIED | New builds omit `db/`. `src/vector_integrity.py` requires explicit `openai`/model/version/dimensions collection metadata and checks **every** stored vector for finite, nonzero values and the expected dimension. An unverified index is rejected with 503 for semantic requests and makes `/health` degraded. | Honest fail-closed behavior; a real compatible index is not provisioned by this phase. |
| VERIFIED | `src/Embed/embed.py` records metadata for newly created real indexes, while the isolated fixture labels itself `offline_test_fixture`. `scripts/ci_bootstrap_db.py --output NEW_PATH` refuses existing directories. | CI fixtures cannot silently become valid semantic indexes, and bootstrap no longer erases an existing `db/`. |
| VERIFIED | `src/retrieval/retriever.py` and `src/retrieval/retrieve.py` now create OpenAI clients only on the embedding path; BM25 does not open Chroma. | No API key or vector index is needed for the lexical demo path. |
| UNKNOWN | Current Cloud Run image digest, mounted volumes, secrets, index content, deployment revision, and live response trace. | Do not claim production retrieval is repaired or deploy this branch without separate verification. |

## Current architecture and verification boundary

Next.js proxies to FastAPI. The legacy adapter scopes one of five registered source files, loads committed semantic chunks for BM25, and conditionally uses Chroma for semantic/hybrid retrieval. Generation uses an external model for authorized live requests. Chroma on the local container filesystem is neither an externally durable system of record nor verified as a production semantic index. The five-filing catalog, citation format, and numeric engine remain legacy work for later phases.

The new build image includes chunks/evaluation summaries and **no** index. Demo mode still uses BM25. Live semantic/hybrid use requires a separately provisioned index that passes the integrity check. `index_ready` means verified semantic readiness; it no longer means a collection merely has rows. The check validates declared provenance plus numerical vector integrity, **not** that the metadata truthfully reflects an upstream API response, the semantic quality of the embeddings, or that filing citations link to the official accession. Treat that distinction as a release gate.

## Commands and results

| Command | Result on this checkout |
| --- | --- |
| `git status --short --branch; git rev-parse HEAD` | Clean `main`, SHA above, before feature branch. |
| `python -m pytest tests/ -q` before dependency installation | BLOCKED: `No module named pytest`. |
| `python -m pip install -r backend/requirements.txt pytest` | PASS; local dependencies installed. Exact versions were not pinned in the repository. |
| `python -m pytest tests/ -q` after changes | PASS: 58 tests; an existing BeautifulSoup XML parser warning and Chroma/Pydantic deprecation warnings. |
| `cd frontend && npm ci && npm run build` | PASS: Next.js 15.5.19; warnings for missing `NEXT_PUBLIC_API_BASE_URL` in local build and Supabase code in Edge runtime. |
| `cd frontend && npm run lint` | PASS: no ESLint warnings/errors; Next lint deprecation notice. |
| `python scripts/ci_bootstrap_db.py --output /workspace/scratch/1a3f7ab284e5/offline-fixture-db` | PASS: 2,812 fixture chunks in disposable scratch directory. |
| `python scripts/check_vector_index.py --db /workspace/scratch/1a3f7ab284e5/offline-fixture-db` | Expected rejection, exit 2: `Missing or incompatible embedding provenance`. |
| `python scripts/check_vector_index.py --db /workspace/scratch/1a3f7ab284e5/verified-fixture-db` | PASS on **one synthetic vector** with matching metadata, to exercise the valid code path. This does not establish a real OpenAI-generated vector or retrieval quality. |
| `python scripts/check_vector_index.py --db db` | BLOCKED for assessing the tracked index: installed Chroma reports `'_type'`; metadata lacks embedding provenance. No trusted production vector sample established. |
| `python -m compileall -q backend/app src scripts/check_vector_index.py; git diff --check` | PASS. |

Historical `data/eval/retrieval_summary.json` and `data/eval/ragas_summary.json` refer to offline Apple-centric runs. They have **not** been rerun with a proven real index. New evaluation denominator: 58 unit/smoke tests, including vector integrity cases; zero live SEC retrieval, citation, real embedding or production latency measurements. No live Apple question/citation was independently verified.

## Risks, stop conditions, and rollback

- Do not deploy this branch to production as a semantic service until a real index has model provenance, passes the CLI check on a safe staging copy, and can be mounted durably. Restoring the old image is the deploy rollback; there is no database migration or destructive index operation here.
- The tracked local SQLite file is neither deleted nor rebuilt. A current Chroma client cannot inspect it successfully, so compatibility/version diagnosis remains open. The CLI opens a Chroma client and may maintain local bookkeeping; run it on a staging copy if pristine bytes matter.
- Metadata can be mislabeled; vector norm/dimension tests alone cannot establish semantic quality. Compare actual retrieval with a frozen accession-labeled benchmark and verify a source-linked Apple citation before claiming full Phase 0 readiness.
- Existing in-memory caches, local index persistence and five hardcoded filings are outside this slice. No production cloud service, secret, Supabase data, or live endpoint was accessed.

**Decision:** Phase 0 code safety repair is verified offline; **NO-GO for production semantic cutover**. The master prompt's known-good real embedding/citation gate remains blocked pending a verified staging index and deployment access.

## Phase 1 SEC adapter sketch and offline test plan (not implemented)

Proposed narrow interface: `resolve_issuer(query) -> list[Issuer(cik, name, tickers)]`, `list_filings(cik, forms, cursor) -> list[FilingRef(cik, accession, form, filed, report_period, primary_document, url)]`, and `fetch_companyfacts(cik) -> CompanyFacts` later when the numeric phase requires it. Keep CIK and accession as durable identities. A server-only client will use configured contact-bearing User-Agent, a **shared aggregate** limiter, bounded responses/timeouts/backoff, allowlisted SEC hosts, and validated accession/document paths. Add `GET /api/research/issuers` and `/api/research/filings` without changing legacy `/api/filings`. Fixtures will cover CIK normalization, ambiguous ticker/name, pagination into older submissions files, amendments, malformed document names, 403/404/429, oversized response, and timeout. Live EDGAR smoke is opt-in and requires the maintainer contact configuration. No credentials or new resources are assumed.
