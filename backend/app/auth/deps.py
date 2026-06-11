"""FastAPI dependencies for optional Supabase authentication."""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from backend.app.auth.jwt import verify_supabase_jwt
from backend.app.auth.models import UserContext
from backend.app.config import get_settings
from backend.app.services.quota_service import load_user_plan


def extract_bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def get_optional_user(request: Request) -> Optional[UserContext]:
    """Return authenticated user if a valid Supabase JWT is present.

    Supports both ES256 (JWKS, newer Supabase projects) and HS256 (legacy).
    For ES256, supabase_jwt_secret is not required — only supabase_url is needed.
    """
    settings = get_settings()

    # At minimum we need the Supabase URL to fetch JWKS for ES256 verification.
    if not settings.supabase_url and not settings.supabase_jwt_secret:
        return None

    token = extract_bearer_token(request)
    if not token:
        return None

    user = verify_supabase_jwt(token, settings.supabase_jwt_secret, settings.supabase_url)
    if user is not None:
        plan = load_user_plan(user.user_id, settings)
        user = UserContext(user_id=user.user_id, email=user.email, plan=plan)  # type: ignore[arg-type]
        request.state.user = user
    return user
