# Phase 2 — local ingestion slice

Date: 2026-10-27. Branch: `codex/phase2-local-ingestion`, dependent on Phase 1. **Local development only; no production cutover.**

## Design and delivered paths

- [Storage ADR](../adr/001-research-storage.md): a separate SQLite metadata/vector diagnostic store on a durable local volume. No writes to the legacy Chroma DB or tracked BM25 chunks. PostgreSQL/pgvector production provisioning remains undecided.
- [Raw original ADR](../adr/002-raw-object-storage.md): exact response bytes, SHA-256 content address, immutable object, source URL, fetch time and response metadata. Production object storage remains undecided.
- [Execution ADR](../adr/003-job-execution.md): token-protected admission returns 202; a separate worker leases jobs, retries expired attempts, fences stale attempts, and atomically publishes a revision. A failed attempt preserves the last ready pointer. No in-request ingestion.
- The worker parses repeated Item occurrences, flags flattened tables and insufficient text, emits stable section/position chunks and validates vector counts/dimensions before publication. Citation provenance retains accession, official source URL and original checksum. `/api/research/filings/{cik}/{accession}/sections` and `/diagnostic` are explicitly local-only. `mode=local-vector` uses a token-hash lexical diagnostic, **not semantic embeddings**.
- `OpenAIEmbedding` is an opt-in real embedding adapter for controlled staging, but it has not been exercised against real filings. There is no research Q&A route or switch into the old five-filing answer path. The Explorer's local-only Index action sends CIK/accession to a Next.js server route. That route is disabled on Vercel, requires a loopback Host and same-origin browser request, and passes the operator token server-side. The UI also refreshes job progress and shows locally published sections.

## Local operation

The overlay `compose.research.yml` mounts `research_data` into the API and the one worker. Set an authentic organization/contact `SEC_USER_AGENT`, a private `RESEARCH_INGESTION_TOKEN` of at least 24 characters, and verify a single backend replica and worker before starting. `SEC_RATE_DB_PATH` shares a conservative 2 requests/s budget across those processes on the local volume. Admission stays disabled in the base configuration.

```bash
docker compose -f docker-compose.yml -f compose.research.yml up --build
curl -X POST http://127.0.0.1:8080/api/research/ingestions \
  -H "Authorization: Bearer $RESEARCH_INGESTION_TOKEN" -H 'Content-Type: application/json' \
  -d '{"cik":"0000320193","accession":"0000320193-25-000079"}'
```

The example accession must first appear in current discovery metadata; replace it with one returned by the issuer filings route. Check `GET /api/research/ingestions/{job_id}` and the accession status/sections routes. To use real embeddings in an approved staging environment, set `RESEARCH_EMBEDDING_MODE=openai` and supply `OPENAI_API_KEY`; the default overlay uses `local-hash` diagnostic. Never pass the admission token to browser code. To roll back locally, stop the worker and disable the overlay; retain the research volume for inspection. No destructive migration is needed.

## Verification and remaining gates

Synthetic, SEC-shaped fixtures exercise 10-K and 10-Q layouts, repeated Items, source fetch restrictions, concurrency, idempotency, durable reopen, expired lease reclaim, stale-attempt fencing, failure rollback, revision and accession isolation, and disabled admission. `python -m pytest tests/ -q` and `npm run lint && npm run build` are the offline gates. Container restart/replacement could not be run in this workspace because Docker is unavailable. A local SQLite reopen proves persistence across a new process object, not container or host replacement.

**BLOCKED for live Phase 2:** Phase 1's authentic SEC contact and live smoke; an approved persistent production database/object store and queue; staging ingestion of at least two unseen issuers and a 10-Q from live SEC; real embeddings with measured cost; restart/replacement, retrieval scope and attribution audits. No live SEC fetch, paid embedding call, cloud provisioning or production deployment was made. The local diagnostic state `READY_LOCAL` must not be interpreted as production semantic readiness.
