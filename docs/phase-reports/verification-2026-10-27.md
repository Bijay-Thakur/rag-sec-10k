# Local verification — 2026-10-27 America/New_York

Implementation commit: `135367c802f162cb5a89e23c9d7ec04f68594544`, branch `codex/phases3-5-local`; baseline `d71a25d6e52146e598ce8e27df122ecf715540c6`. The supplied acceptance plan and prompt are in `docs/plans/` (Markdown trailing spaces normalized). Dependencies are recorded in `verification-environment.json`; the local embedding pin/manifest is described in ADR 004.

| Check | Exact command or operation | Result | Boundary |
| --- | --- | --- | --- |
| Research safety and source contracts | `.venv/bin/python -m pytest tests/test_research_phases_3_5.py tests/test_research_financial.py tests/test_research_ingestion.py tests/test_sec_discovery.py -q` | 56 passed in 5.72s | Synthetic SEC HTTP and fake Groq in unit tests; no paid calls |
| Syntax | `.venv/bin/python -m compileall -q backend/app/research scripts/evaluate_research.py scripts/ingest_companyfacts.py scripts/research_backup.py scripts/research_status.py scripts/prepare_research_model.py scripts/smoke_local_research.py` | Passed | Static/import-independent syntax check |
| Frontend | `NEXT_TELEMETRY_DISABLED=1 npm run lint && NEXT_TELEMETRY_DISABLED=1 npm run build` in `frontend/` | Passed, Next.js 15.5.19 | Local API-base warning; no hydrated browser/mobile test |
| Local vector integration | `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 .venv/bin/python scripts/smoke_local_research.py --model-dir .research/models/all-MiniLM-L6-v2 --output docs/phase-reports/local-model-smoke.json` | 9 queries, 384-dimensional real local vectors, all expected synthetic Items ranked first | Three short synthetic scopes reuse the same three sections; not a real relevance or latency benchmark |
| Server boundary | Started the production frontend on loopback with both local operator flags enabled and fake test tokens; POSTed both `/api/research/local-query` and `/api/research/local-ingestions` with a matching Origin | Both returned 403 | Confirms production-mode refusal; no real key or backend request used |
| Backup/restore and rollback | Included in the 56-case suite | Preserved source bytes, sections, budget, original/amended fact maps; rejected corrupt sources and cross-scope rollback | Local SQLite/filesystem only; no container/host/cloud replacement |
| Secret/context checks | `git check-ignore .env .research/models/all-MiniLM-L6-v2/model.safetensors`, `git ls-files .env`, changed-file secret-pattern scan, public bundle secret-name scan | Ignored/untracked secrets/model; zero secret-pattern or public secret-name hits; no new files >1 MiB | Pattern scan is not a guarantee against every secret format |
| Diff | `git diff --check`, then `git diff --cached --check` | Passed | Source plan hard-break whitespace was normalized |

## Blocked full regression

A current-runtime full `pytest tests/` attempt was stopped by automatic approval review for unexpected Microsoft telemetry egress with an unknown payload. A subsequent attempt after the documented telemetry opt-out was also rejected. The native Chroma/ONNX path is suspected from dependency inspection, but the exact source and payload were not independently established. No further full-suite attempt was made. The completed research test commands avoid initializing that path and use an outbound Python socket guard. This is a remaining full-regression gate; neither the earlier runtime's full pass nor the current 56-case result substitutes for it.

## Live observations and unresolved activation gates

Vercel read-only inspection returned a READY production deployment of `d71a25d`, Node 24. The proxied legacy catalog returned HTTP 200 with five filings; the health request timed out. These observations do not establish the backend image, real index provenance or new worker state. Those remain UNKNOWN.

The runtime has no authentic SEC contact/rate configuration, no configured Groq key, and no Docker executable. No live SEC ingestion, Groq generation, paid model call, container replacement, production setting change, remote push or deployment was performed. Real-file table attribution, adjudicated held-out relevance/numeric evaluation, Groq human review and staging rollback still block G0/G3/G4/G5 activation. Both research flags remain off by default.
