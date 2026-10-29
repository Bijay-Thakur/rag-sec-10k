# Phase 5 — offline evaluation and operational preparation

CI now runs the complete offline backend suite, including discovery, ingestion and Phase 3–5 safety tests, plus frontend lint/build on Node 24. `evaluation.score_cases` produces deterministic recall@5, MRR@5, scope validity, citation target validity and refusal agreement from accession/revision-labeled cases. The current single synthetic case only validates scoring behavior; it is **not** an evaluation corpus or quality metric.

No Groq call, staging trace, held-out multi-issuer set, accessibility/mobile review, backup/restore or rollback drill was run. Retention, alerts and cost accounting remain design requirements. G5 is **NO-GO**. The [operator runbook](../research-operator-runbook.md) records verification and rollback procedures. Nothing was deployed or pushed.
