# Phase 3 baseline and G0 evidence

Date 2026-10-27 UTC. Inspected `main` at `d71a25d6e52146e598ce8e27df122ecf715540c6`, clean worktree before creating `codex/phases3-5-local`. No `AGENTS.md` was found in the repository. The Phase 1 and 2 reports, ADRs, CI, research routes/worker/store, legacy adapter, frontend and deployment configuration were inspected. `V2_UPGRADE_PLAN.md` predates the merged research code and is historical.

| Claim | Evidence and command | Environment / result | Limit |
| --- | --- | --- | --- |
| Main contains Phase 0–2 | `git log --oneline -7 origin/main`, 2026-10-27 | Source observed: merge `d71a25d` includes PR #2 and dependent PR #3 | Merge is not release proof. |
| Legacy path still exists | `backend/app/main.py`, `backend/app/adapters/rag_adapter.py`; `python -m pytest tests/ -q` | Offline: 96 passed before new edits | Does not identify deployed backend revision/index. |
| Frontend builds | `cd frontend && npm run lint && npm run build` | Offline: both passed, Node 24; local rewrite warning with missing API base | Does not verify hydrated production flow. |
| CI omitted research tests | `.github/workflows/ci.yml` before edit selected only smoke and six test files | Source observed; changed to full offline `tests/` suite and Node 24 | Remote CI has not run on this unpushed branch. |
| Live discovery/ingestion | Phase 1 and 2 reports | Reported BLOCKED: authentic SEC contact, aggregate limiter/deployment topology, live samples, real local vectors, container replacement | No live request, paid call or Docker test was run in this work. |
| Production backend and vector provenance | No Cloud Run revision/image digest or mounted index verification in this workspace | UNKNOWN | `/health` and Vercel status cannot prove dynamic research readiness. |

The Phase 3 frozen multi-issuer benchmark is **not available**. Its version, labeled accession corpus, predeclared quality/latency thresholds and staging sample sizes remain BLOCKED. `tests/test_research_phases_3_5.py` is a synthetic safety fixture, not a benchmark. The old Apple metrics must not be reused as Phase 3 results.

**G0 live verdict: NO-GO.** Continue offline code only. Before enabling research: verify backend image digest, index provenance and rollback target; configure authentic SEC contact and one aggregate limiter; run three-issuer historical discovery smoke; stage two unseen issuers, a 10-Q and amendment with a pinned local model; inspect table/footnote attribution and survive container replacement. No production flag or service was changed.
