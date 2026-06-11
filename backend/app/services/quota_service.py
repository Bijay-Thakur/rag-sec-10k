"""Per-user LLM quota tracking via Supabase (with in-memory fallback for dev)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from backend.app.adapters.rag_adapter import RAGPipelineError
from backend.app.auth.models import UserContext
from backend.app.config import Settings, get_settings
from backend.app.services.supabase_client import SupabaseClient

logger = logging.getLogger("sec_insight_api.quota")

_lock = threading.Lock()
_memory_profiles: Dict[str, Dict[str, int | str]] = {}


@dataclass(frozen=True)
class UserEntitlements:
    user_id: str
    email: str
    plan: str
    llm_calls_used: int
    llm_calls_limit: int
    llm_calls_remaining: int
    can_use_llm: bool
    daily_token_budget: int
    daily_tokens_used: int


def _default_limit(settings: Settings, plan: str) -> int:
    if plan == "pro":
        return settings.pro_tier_llm_calls_per_day
    return settings.free_tier_llm_calls


class SupabaseUnavailableError(Exception):
    """Raised when Supabase is configured but cannot be reached.

    When Supabase is configured (``client.enabled``), falling back to an
    in-memory profile with ``llm_calls_used=0`` would allow quota bypass.
    Callers must catch this and fail closed (block the LLM call / return 503).
    """


def _ensure_profile(user: UserContext, settings: Settings, client: SupabaseClient) -> Dict:
    """Return the user's profile, creating it only if it truly does not exist.

    Safety invariants
    -----------------
    * NEVER resets ``llm_calls_used`` on an existing row.
    * When Supabase is configured but unreachable, raises
      ``SupabaseUnavailableError`` so callers can fail closed instead of
      silently granting quota from a stale in-memory state.
    """
    new_profile: Dict = {
        "id": user.user_id,
        "email": user.email,
        "plan": "free",
        "llm_calls_used": 0,
        "llm_calls_limit": settings.free_tier_llm_calls,
        "daily_tokens_used": 0,
    }

    if client.enabled:
        # 1. Try to read the existing profile.
        existing = client.get_profile(user.user_id)
        if existing:
            return existing

        # 2. Profile not found (new user or transient read error).
        #    INSERT OR IGNORE — never overwrites an existing row's counters.
        inserted = client.insert_profile_if_new(new_profile)
        if inserted:
            return inserted  # Newly created row

        # 3. insert returned None → row existed but get_profile failed.
        #    Retry the read once before giving up.
        retry = client.get_profile(user.user_id)
        if retry:
            return retry

        # 4. Both reads and the insert all failed — Supabase is unavailable.
        #    Fail closed: do NOT fall back to in-memory when Supabase is
        #    configured, as that would grant fresh quota to exhausted users.
        raise SupabaseUnavailableError(
            f"Cannot read profile for user {user.user_id} — Supabase unavailable."
        )

    # Supabase not configured (pure local/dev mode) — use in-memory.
    with _lock:
        if user.user_id not in _memory_profiles:
            _memory_profiles[user.user_id] = dict(new_profile)
        return dict(_memory_profiles[user.user_id])  # type: ignore[arg-type]


def load_user_plan(user_id: str, settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    client = SupabaseClient(settings)
    if client.enabled:
        profile = client.get_profile(user_id)
        if profile:
            plan = str(profile.get("plan", "free"))
            return plan if plan in ("free", "pro") else "free"
    with _lock:
        profile = _memory_profiles.get(user_id)
        if profile:
            plan = str(profile.get("plan", "free"))
            return plan if plan in ("free", "pro") else "free"
    return "free"


def get_entitlements(user: UserContext, settings: Optional[Settings] = None) -> UserEntitlements:
    settings = settings or get_settings()
    client = SupabaseClient(settings)
    # SupabaseUnavailableError propagates to the caller (FastAPI endpoint
    # returns 503; reserve_llm_call blocks the LLM call).
    profile = _ensure_profile(user, settings, client)

    plan = str(profile.get("plan", user.plan))
    used = int(profile.get("llm_calls_used", 0))
    limit = int(profile.get("llm_calls_limit", _default_limit(settings, plan)))
    daily_used = int(profile.get("daily_tokens_used", 0))
    daily_budget = (
        settings.pro_tier_daily_token_budget
        if plan == "pro"
        else settings.free_tier_daily_token_budget
    )

    if plan == "pro":
        remaining = max(0, limit - used)
        can_use = remaining > 0 and daily_used < daily_budget
    else:
        remaining = max(0, limit - used)
        can_use = remaining > 0

    return UserEntitlements(
        user_id=user.user_id,
        email=user.email,
        plan=plan,
        llm_calls_used=used,
        llm_calls_limit=limit,
        llm_calls_remaining=remaining,
        can_use_llm=can_use,
        daily_token_budget=daily_budget,
        daily_tokens_used=daily_used,
    )


def reserve_llm_call(user: UserContext, settings: Optional[Settings] = None) -> UserEntitlements:
    """Verify the user may make an LLM call and increment usage.

    Raises RAGPipelineError with error_code=budget_exceeded when over quota.
    Raises RAGPipelineError with error_code=quota_unavailable when Supabase
    is configured but unreachable (fail-closed: blocks the LLM call).
    """
    settings = settings or get_settings()
    client = SupabaseClient(settings)
    try:
        entitlements = get_entitlements(user, settings)
    except SupabaseUnavailableError:
        logger.error(
            "Supabase unavailable for user %s — blocking LLM call (fail-closed).",
            user.user_id,
        )
        raise RAGPipelineError(
            "Quota service is temporarily unavailable. Please try again in a moment.",
            error_code="quota_unavailable",
        )

    if not entitlements.can_use_llm:
        if entitlements.plan == "free":
            detail = (
                "You have used your free AI answer. Subscribe to Premium "
                f"({settings.premium_price_label}) for more LLM-powered queries."
            )
        else:
            detail = (
                "Daily LLM quota exceeded. Try again tomorrow or contact support."
            )
        raise RAGPipelineError(detail, error_code="budget_exceeded")

    if client.enabled:
        updated = client.increment_llm_calls(user.user_id)
        if updated:
            used = int(updated.get("llm_calls_used", entitlements.llm_calls_used + 1))
            limit = int(updated.get("llm_calls_limit", entitlements.llm_calls_limit))
            return UserEntitlements(
                user_id=user.user_id,
                email=user.email,
                plan=str(updated.get("plan", entitlements.plan)),
                llm_calls_used=used,
                llm_calls_limit=limit,
                llm_calls_remaining=max(0, limit - used),
                can_use_llm=max(0, limit - used) > 0,
                daily_token_budget=entitlements.daily_token_budget,
                daily_tokens_used=entitlements.daily_tokens_used,
            )
        # Supabase write failed — fail closed to prevent unbounded LLM spend.
        logger.error(
            "Quota increment failed for user %s — blocking LLM call to protect cost.",
            user.user_id,
        )
        raise RAGPipelineError(
            "Could not record quota usage. Please try again in a moment.",
            error_code="quota_write_failed",
        )

    with _lock:
        profile = _memory_profiles.setdefault(
            user.user_id,
            {
                "id": user.user_id,
                "email": user.email,
                "plan": "free",
                "llm_calls_used": 0,
                "llm_calls_limit": settings.free_tier_llm_calls,
                "daily_tokens_used": 0,
            },
        )
        profile["llm_calls_used"] = int(profile.get("llm_calls_used", 0)) + 1

    return get_entitlements(user, settings)


def record_token_usage(user_id: str, total_tokens: int, settings: Optional[Settings] = None) -> None:
    """Track token usage for pro-tier daily budgets (best-effort)."""
    if total_tokens <= 0:
        return
    settings = settings or get_settings()
    client = SupabaseClient(settings)
    if client.enabled:
        profile = client.get_profile(user_id)
        if profile is None:
            return
        used = int(profile.get("daily_tokens_used", 0)) + total_tokens
        client.upsert_profile({"id": user_id, "daily_tokens_used": used})
        return

    with _lock:
        if user_id in _memory_profiles:
            profile = _memory_profiles[user_id]
            profile["daily_tokens_used"] = int(profile.get("daily_tokens_used", 0)) + total_tokens
