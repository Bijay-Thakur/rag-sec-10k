# Phase 3–5 release readiness — 2026-10-27

| Gate | Implemented and checked locally | Remaining release evidence | Decision |
| --- | --- | --- | --- |
| G0 provenance and SEC access | Merged source baseline and current Vercel main SHA reconciled; full-test workflow configured | Backend image/mount/instances unknown; authentic SEC contact/rate policy and live samples absent; native full-suite auto-review blocker | NO-GO |
| G3 dense retrieval and citations | Pinned model ran offline, scoped BM25/dense/RRF, revision publication, generated citation rejection, protected local UI | Real multi-issuer held-out corpus, table attribution, Groq human review and container replacement | NO-GO |
| G4 financial reasoning | Immutable XBRL snapshots, exact source pointers, explicit as-filed selection, Decimal preview and amendment isolation | Live numeric accuracy, reviewed wider concept map, table/custom-taxonomy fallback | NO-GO |
| G5 evaluation and operation | Metric CLI, private trace records, durable call cap, status signals, local backup/restore tests | Native full regression, real staging traces/cost, adjudicated gold set, staged rollback and production sign-off | NO-GO |

Both research features remain disabled by default. Local synthetic integration is verified within the limits recorded in the phase reports. Nothing in this branch authorizes production deployment or a GitHub push. See [runbook](../research-operator-runbook.md).
