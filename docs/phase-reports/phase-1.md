# Phase 1 — SEC discovery delivery report

Date: 2026-10-27. Branch: `codex/phase1-sec-discovery`, based on Phase 0 tree from PR #1. This is a dependent code change and **not a production cutover**.

## Evidence and design

- GitHub Actions for Phase 0 head `8009e82d...` succeeded; the Vercel preview failed before build with `BUILD_UTILS_NODE_VERSION_DISCONTINUED` (project previously targeted Node 20). At inspection, Vercel project `prj_9TvilHCfcWpAdaLooLAfLvucYdgP` already reported Node 24.x. A fresh Vercel preview remains to be checked.
- Legacy `backend/app/main.py` retains `/health`, `/api/filings`, `/api/ask` and existing access policy. New read-only `/api/research` routes use `SecClient` dependency injection, typed Pydantic contracts, SEC-only URL construction and bounded HTTP. The Next.js Explorer uses the existing `/api-proxy` rewrite.
- SEC official docs consulted 2026-10-27: [submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [archive paths, ticker associations and fair access](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data). The API supports 10-digit CIK submissions and referenced older files. Ticker associations are not guaranteed complete. SEC requests should identify an authentic contact and remain below 10/s aggregate.
- A process-local 2/s limiter cannot coordinate Cloud Run replicas. Live mode therefore requires `SEC_SINGLE_INSTANCE_MODE=true` plus an authentic `SEC_USER_AGENT`, and the operator must separately verify a **one-instance** deployment. Without these settings the routes return a typed 503. No SEC calls run in CI.
- `GET /api/research/issuers?query=MSFT` returns explicit candidates. `GET /api/research/issuers/{cik}/filings?form=10-K&include_amendments=false&include_history=false&limit=25` lists metadata and official links. `include_history=true` fetches referenced history pages. Numeric cursor uses a stable offset over a snapshot; changing live SEC data between pages can shift the snapshot. This is a known limitation.
- The synthetic SEC-shaped fixtures are labeled as such. They cover ambiguous tickers, recent/history parallel arrays, amendments, official archive URLs, unsafe document paths, 403/404/429/5xx, redirects, timeouts, response limits and conditional cache refresh.

## Verification

`python -m pytest tests/ -q`: 76 passed (legacy and new tests; parser/dependency warnings). `cd frontend && npm run lint && npm run build`: passed on local Node 24.22.1. The Explorer labels rows “Discovered · not indexed”; it does not invoke document fetch, embeddings, indexing or LLM. Old five-filing API smoke tests pass.

Live SEC smoke: **BLOCKED** without the maintainer's authentic `SEC_USER_AGENT`, a confirmed single-instance limit and an allowed SEC network route. Opt-in command: `SEC_SINGLE_INSTANCE_MODE=true SEC_USER_AGENT='<real organization and contact email>' python scripts/smoke_edgar_discovery.py`. Run only after confirming deployment policy; record the JSON output and accession checks. Do not infer live success from fixture tests.

Release verdict: **offline code review ready; live Phase 1 gate BLOCKED**. No billed resource, production setting, secret, or live SEC request was changed. Phase 2 may be developed locally under the user's express combined-phase request, but cannot be labeled live-ready or released on this evidence.
