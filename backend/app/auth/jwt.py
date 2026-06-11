"""Verify Supabase-issued JWT access tokens.

Supabase projects created after mid-2024 use ES256 (asymmetric ECDSA) instead
of HS256. We fetch the public key once from the JWKS endpoint and cache it.
Older projects using HS256 are still supported as a fallback.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from typing import Any, Dict, Optional

import httpx
import jwt
from jwt import PyJWTError
from jwt.algorithms import ECAlgorithm

from backend.app.auth.models import UserContext

logger = logging.getLogger("sec_insight_api.auth")

_jwks_lock = threading.Lock()
_jwks_cache: Dict[str, Any] = {}   # kid -> public key object


def _fetch_public_key(supabase_url: str, kid: str) -> Optional[Any]:
    """Fetch JWKS from Supabase and return the matching public key (cached)."""
    with _jwks_lock:
        if kid in _jwks_cache:
            return _jwks_cache[kid]

    try:
        url = supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        keys = resp.json().get("keys", [])
        for jwk in keys:
            if jwk.get("kid") == kid:
                pub = ECAlgorithm.from_jwk(json.dumps(jwk))
                with _jwks_lock:
                    _jwks_cache[kid] = pub
                return pub
    except Exception as exc:
        logger.warning("JWKS fetch failed: %s", exc)
    return None


def verify_supabase_jwt(
    token: str,
    jwt_secret: str,
    supabase_url: str = "",
) -> Optional[UserContext]:
    """
    Validate a Supabase JWT (ES256 or HS256) and return user context.
    Returns None if the token is invalid or expired.
    """
    if not token:
        return None

    # Peek at the header to detect the algorithm.
    try:
        header = jwt.get_unverified_header(token)
    except PyJWTError as exc:
        logger.debug("JWT header parse failed: %s", exc)
        return None

    alg = header.get("alg", "HS256")
    kid = header.get("kid", "")

    try:
        if alg == "ES256" and supabase_url and kid:
            public_key = _fetch_public_key(supabase_url, kid)
            if public_key is None:
                logger.debug("No JWKS key found for kid=%s", kid)
                return None
            payload: Dict[str, Any] = jwt.decode(
                token,
                public_key,
                algorithms=["ES256"],
                options={"verify_aud": False},
            )
        else:
            # Legacy HS256 — try raw bytes first, then base64-decoded.
            if not jwt_secret:
                return None
            secret: Any = jwt_secret.encode("utf-8")
            try:
                payload = jwt.decode(
                    token, secret, algorithms=["HS256"], options={"verify_aud": False}
                )
            except PyJWTError:
                try:
                    secret = base64.b64decode(jwt_secret)
                    payload = jwt.decode(
                        token, secret, algorithms=["HS256"], options={"verify_aud": False}
                    )
                except Exception as exc:
                    logger.debug("HS256 JWT verification failed: %s", exc)
                    return None

    except PyJWTError as exc:
        logger.debug("JWT verification failed: %s", exc)
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    email = payload.get("email") or payload.get("user_metadata", {}).get("email") or ""
    app_meta = payload.get("app_metadata") or {}
    plan = app_meta.get("plan", "free")
    if plan not in ("free", "pro"):
        plan = "free"

    return UserContext(user_id=str(user_id), email=str(email), plan=plan)  # type: ignore[arg-type]
