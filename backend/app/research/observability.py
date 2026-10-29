"""Local operational signals without request text, sources or credentials."""

import time


def operational_status(repo, *, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    with repo.connect() as db:
        counts = {row["state"]: row["n"] for row in db.execute("SELECT state,count(*) AS n FROM jobs GROUP BY state")}
        expired = db.execute("SELECT count(*) FROM jobs WHERE lease_until<? AND state IN "
                             "('FETCHING','PARSING','EMBEDDING','VALIDATING')", (now,)).fetchone()[0]
        queued = db.execute("SELECT min(queued_at) FROM jobs WHERE state='QUEUED'").fetchone()[0]
        citation_failures = db.execute("SELECT count(*) FROM research_events WHERE code='citation_invalid' AND occurred_at>=?",
                                      (now - 86400,)).fetchone()[0]
        scope_failures = db.execute("SELECT count(*) FROM research_events WHERE code IN ('scope_violation','scope_mismatch') AND occurred_at>=?",
                                   (now - 86400,)).fetchone()[0]
        ready = db.execute("SELECT count(*) FROM research_ready").fetchone()[0]
        usage = db.execute("SELECT calls FROM research_budget WHERE day=?", (int(now // 86400),)).fetchone()
    alerts = []
    if expired:
        alerts.append("expired_worker_lease: inspect worker health before reclaiming jobs")
    if counts.get("FAILED", 0):
        alerts.append("failed_ingestion: inspect error_code and parser QA before retry")
    if queued and now - queued > 900:
        alerts.append("stale_queue: inspect worker and SEC access policy")
    if citation_failures:
        alerts.append("citation_failure: keep preview disabled and review evidence validation")
    if scope_failures:
        alerts.append("scope_failure: disable preview and investigate revision isolation")
    return {"job_states": counts, "expired_leases": expired, "research_ready_filings": ready,
            "citation_failures_24h": citation_failures, "scope_failures_24h": scope_failures,
            "reserved_generation_calls_today": usage[0] if usage else 0, "alerts": alerts}
