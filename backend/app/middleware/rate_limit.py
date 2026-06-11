"""Rate limiting helpers for /api/ask."""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.auth.deps import get_optional_user
from backend.app.config import get_settings


def rate_limit_key(request: Request) -> str:
    user = get_optional_user(request)
    if user is not None:
        return f"user:{user.user_id}"
    return get_remote_address(request)


def tiered_rate_limit(key: str) -> str:
    """Return the appropriate rate-limit string for the current caller tier.

    slowapi calls this as tiered_rate_limit(key_func(request)) when the
    parameter is named ``key`` — see slowapi wrappers.py __iter__ logic.
    The key is produced by rate_limit_key(), which returns "user:<id>" for
    authenticated callers and an IP address for anonymous ones.
    """
    settings = get_settings()
    if key.startswith("user:"):
        user_id = key[len("user:"):]
        from backend.app.services.quota_service import load_user_plan  # noqa: PLC0415
        plan = load_user_plan(user_id, settings)
        if plan == "pro":
            return settings.rate_limit_pro_per_hour
        return settings.rate_limit_authenticated_per_hour
    return settings.rate_limit_anonymous_per_hour


limiter = Limiter(key_func=rate_limit_key)
