# Phase 3–5 release readiness (2026-10-27)

| Gate | Offline code | Staging/live evidence | Decision |
| --- | --- | --- | --- |
| G0 deployed provenance, authentic SEC smoke and ingestion | Baseline suite passes; CI expanded | Backend digest/mount unknown; SEC and container replacement not run | NO-GO |
| G3 real local embedding, scoped retrieval, citation quality | Model adapter, pointer, retrieval and adversarial fixtures | Model artifact absent; no frozen multi-issuer benchmark, Groq staging review or measured latency | NO-GO |
| G4 XBRL financial reasoning | SEC allowlist, exact-accession normalization and Decimal guardrails | No persisted raw facts, amendment precedence, verified table cells or held-out numeric suite | NO-GO |
| G5 evaluation, observability, product readiness | Deterministic metric function and full offline CI configured | No adjudicated gold set, staging traces, alerts, backup/restore or rollback drill | NO-GO |

The public research query feature remains disabled. The local code is not a production Financial Research System. See the individual phase reports and [runbook](../research-operator-runbook.md) for exact missing checks and rollback. No remote push or deployment is included in this branch.
