"""Server-side access policy for /api/ask — demo mode, auth, and quotas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.adapters.rag_adapter import RAGPipelineError
from backend.app.auth.models import UserContext
from backend.app.config import Settings, get_settings
from backend.app.services.cost_guard import assert_within_global_limits
from backend.app.services.quota_service import reserve_llm_call


@dataclass(frozen=True)
class AskAccessDecision:
    effective_demo_mode: bool
    user: Optional[UserContext] = None


def resolve_ask_access(
    *,
    requested_demo_mode: bool,
    user: Optional[UserContext],
    settings: Optional[Settings] = None,
) -> AskAccessDecision:
    """
    Enforce server-side policy for /api/ask.

    - Anonymous: demo only (BM25, no LLM). Reject live LLM requests.
    - Authenticated + demo_mode: allowed without consuming LLM quota.
    - Authenticated + live LLM: check and consume quota.
    """
    settings = settings or get_settings()

    if not settings.enable_live_llm_calls and not requested_demo_mode:
        raise RAGPipelineError(
            "Live LLM calls are disabled on this deployment. Use demo mode "
            "(retrieval-only) or set ENABLE_LIVE_LLM_CALLS=true for operators.",
            error_code="missing_api_key",
        )

    if not settings.enforce_access_policy:
        return AskAccessDecision(
            effective_demo_mode=requested_demo_mode,
            user=user,
        )

    if user is None:
        if settings.anonymous_demo_only and not requested_demo_mode:
            raise RAGPipelineError(
                "Sign in with email to use live AI generation. "
                "Anonymous visitors may only use demo mode (retrieval-only).",
                error_code="auth_required",
            )
        return AskAccessDecision(effective_demo_mode=True, user=None)

    if requested_demo_mode:
        return AskAccessDecision(effective_demo_mode=True, user=user)

    assert_within_global_limits(settings)
    reserve_llm_call(user, settings)
    return AskAccessDecision(effective_demo_mode=False, user=user)
