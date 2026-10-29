# Research operator runbook (pre-release)

1. Record frontend deployment SHA and backend Cloud Run revision/image digest separately; inspect instance count, worker count, mount and index provenance. Do not interpret legacy `/health` as new research readiness.
2. Set a real organization/contact `SEC_USER_AGENT`; verify one aggregate shared limiter for API and worker at <=2 requests/s. Run `python scripts/smoke_edgar_discovery.py` only as an opt-in bounded smoke. Record exact accession URLs and timestamps. Stop on 403/429 and review policy.
3. On an isolated durable staging volume, provision an audited local sentence-transformer artifact. Set `RESEARCH_EMBEDDING_MODE=local-model` and absolute `RESEARCH_LOCAL_MODEL_DIR`. Record model directory checksum, dimensions and software revision. Run the worker on two unseen issuers including a 10-Q and amendment; verify source SHA, parser QA, failure recovery, restart and cross-scope tests. Keep query flag off.
4. Review table and footnote attribution. Only an operator may call `LocalResearchRepository.promote_research(cik, accession, revision_id)` after QA; promotion refuses diagnostics and flattened tables. Pin a frozen labeled benchmark and verify retrieval/citation and generation with a bounded staging budget.
5. To enable a **single-instance staging preview only**, set `RESEARCH_QUERY_ENABLED=true`, `RESEARCH_SINGLE_INSTANCE_MODE=true`, a small `RESEARCH_DAILY_CALL_CAP` (1–100), `RESEARCH_QUERY_TOKEN` (>=24 random characters), `GROQ_MODEL` and a server-side `GROQ_API_KEY` supplied by a secret manager. The daily call cap is stored transactionally in the research database and survives process restart. SQLite still requires one shared local volume and a verified single host topology. Keep the token and key out of browser code and logs. Rotate the key shared in chat before any production use. The query endpoint is operator-only and rate-limited.
6. Incident response: turn off `RESEARCH_QUERY_ENABLED`; preserve raw objects and the failed revision, restore the previous `research_ready` pointer in a transaction if needed, inspect trace/error codes without logging questions or secrets. Legacy `/api/ask` remains the fallback. Verify backup/restore and rollback in staging before release.

Release remains NO-GO until the phase reports' live/staging gates and operator sign-off are met. No production resource should be created by following this document alone.

## Reproducible local commands

Use a virtual environment, then install the tested optional CPU runtime after the backend dependencies:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements-research-local.txt
python scripts/prepare_research_model.py .research/models/all-MiniLM-L6-v2 --download
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 python scripts/smoke_local_research.py \
  --model-dir .research/models/all-MiniLM-L6-v2 --output /tmp/local-model-smoke.json
```

Preparation contacts the official public model host; the smoke uses real local vectors on synthetic source text and makes no SEC or generation calls. Model checksums are compared with the reviewed `backend/app/research/embedding_manifest.json` and extra model files are rejected.

After authentic SEC access and an exact local filing revision exist, `scripts/ingest_companyfacts.py --cik CIK --accession ACCESSION --revision REVISION --allow-sec-fetch` fetches one bounded companyfacts snapshot through the SEC adapter. It requires absolute `RESEARCH_DB_PATH` and `RESEARCH_RAW_DIR`, the real SEC contact and verified request budget. The financial endpoint additionally requires `RESEARCH_FACTS_ENABLED=true`; its request specifies exact concepts, units and periods. It never infers a missing concept.

For a local browser preview, set server-only `RESEARCH_LOCAL_QUERY_ENABLED=true`, `RESEARCH_QUERY_TOKEN`, `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8080` (or the local backend service) and public visibility flag `NEXT_PUBLIC_RESEARCH_LOCAL_QUERY_ENABLED=true` in the frontend environment. Set the backend preview flags, model path and bounded call cap as described above. A research-ready status then reveals the question form. The server route refuses Vercel and production-mode execution; use the development server bound to `127.0.0.1`. Local Compose now publishes the frontend only on loopback. Keep secrets out of public environment variables.

The following commands operate on existing local development data and require new destinations:

```bash
python scripts/research_status.py /absolute/research.sqlite3
python scripts/research_backup.py backup /absolute/research.sqlite3 /new/backup --objects /absolute/objects
python scripts/research_backup.py restore /new/backup /new/restore-drill
python scripts/evaluate_research.py /absolute/predictions.json --output /new/evaluation.json
```

Backup copies the SQLite snapshot and every referenced raw object, including XBRL sources, and validates checksums during restore. Local tests restore original/amended snapshots and preserve budget reservations. To roll back a research pointer use `repository.rollback_research(cik, accession, prior_revision_id)`; to roll back a fact map use `snapshots.rollback(cik, accession, revision_id, prior_snapshot_id)`. Both require an exact prior scope and retain all versions. These local tests do not establish cloud backup or container replacement readiness.

Query/financial trace records expire after seven days when new events are written. Reserved generation-call counts persist by UTC day. The status command prints actionable local warnings; configure scheduling, external alerts, retention and billing reconciliation only in an approved deployment plan.
