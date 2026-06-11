"""Process-local daily LLM/cost caps for Cloud Run cost safety.

Counters reset on container cold start. Use billing alerts and OpenAI usage
limits in addition to these guards.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timezone

from backend.app.adapters.rag_adapter import RAGPipelineError
from backend.app.config import Settings

_lock = threading.Lock()
_state: dict[str, date | int | float | None] = {
    "day": None,
    "llm_calls": 0,
    "estimated_cost_usd": 0.0,
}


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _reset_if_new_day() -> None:
    today = _utc_today()
    if _state["day"] != today:
        _state["day"] = today
        _state["llm_calls"] = 0
        _state["estimated_cost_usd"] = 0.0


def assert_within_global_limits(settings: Settings) -> None:
    """Fail closed when deployment-wide daily caps are exceeded."""
    if settings.max_daily_llm_calls <= 0 and settings.max_daily_estimated_cost_usd <= 0:
        return

    with _lock:
        _reset_if_new_day()
        calls = int(_state["llm_calls"] or 0)
        cost = float(_state["estimated_cost_usd"] or 0.0)

        if settings.max_daily_llm_calls > 0 and calls >= settings.max_daily_llm_calls:
            raise RAGPipelineError(
                "Daily LLM call limit reached for this deployment. Try again tomorrow "
                "or use demo mode (retrieval only).",
                error_code="budget_exceeded",
            )
        if (
            settings.max_daily_estimated_cost_usd > 0
            and cost >= settings.max_daily_estimated_cost_usd
        ):
            raise RAGPipelineError(
                "Daily estimated LLM cost limit reached for this deployment. "
                "Try again tomorrow or use demo mode.",
                error_code="budget_exceeded",
            )


def record_global_llm_usage(settings: Settings, *, estimated_cost_usd: float = 0.0) -> None:
    """Increment deployment-wide counters after a successful live LLM call."""
    if settings.max_daily_llm_calls <= 0 and settings.max_daily_estimated_cost_usd <= 0:
        return

    with _lock:
        _reset_if_new_day()
        _state["llm_calls"] = int(_state["llm_calls"] or 0) + 1
        if estimated_cost_usd > 0:
            _state["estimated_cost_usd"] = float(_state["estimated_cost_usd"] or 0.0) + estimated_cost_usd
