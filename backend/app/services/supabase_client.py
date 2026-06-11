"""Minimal Supabase REST client for backend quota/profile operations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from backend.app.config import Settings

logger = logging.getLogger("sec_insight_api.supabase")


class SupabaseClient:
    def __init__(self, settings: Settings) -> None:
        self._url = settings.supabase_url.rstrip("/") if settings.supabase_url else ""
        self._service_key = settings.supabase_service_role_key
        self._enabled = bool(self._url and self._service_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _headers(self, prefer: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "apikey": self._service_key,
            "Authorization": f"Bearer {self._service_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        url = f"{self._url}/rest/v1/user_profiles"
        params = {"id": f"eq.{user_id}", "select": "*"}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=self._headers(), params=params)
                resp.raise_for_status()
                rows: List[Dict[str, Any]] = resp.json()
                return rows[0] if rows else None
        except Exception as exc:
            logger.warning("Supabase get_profile failed: %s", exc)
            return None

    def upsert_profile(self, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge non-counter fields into an existing profile row.

        Counter fields (``llm_calls_used``, ``daily_tokens_used``) are stripped
        before merging so that a merge-duplicates upsert never resets usage
        data that was written by ``increment_llm_calls``.
        """
        if not self._enabled:
            return None
        # Never allow a merge upsert to reset counter columns.
        safe_profile = {
            k: v for k, v in profile.items()
            if k not in ("llm_calls_used", "daily_tokens_used")
        }
        url = f"{self._url}/rest/v1/user_profiles"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    url,
                    headers=self._headers(prefer="resolution=merge-duplicates,return=representation"),
                    params={"on_conflict": "id"},
                    json=safe_profile,
                )
                resp.raise_for_status()
                rows: List[Dict[str, Any]] = resp.json()
                return rows[0] if rows else profile
        except Exception as exc:
            logger.warning("Supabase upsert_profile failed: %s", exc)
            return None

    def insert_profile_if_new(self, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """INSERT the profile row only if it does not already exist.

        Uses ON CONFLICT DO NOTHING (``ignore-duplicates``) so an existing
        row — and its usage counters — are never overwritten.  Returns the
        newly inserted row, or ``None`` when the row already existed.
        """
        if not self._enabled:
            return None
        url = f"{self._url}/rest/v1/user_profiles"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    url,
                    headers=self._headers(prefer="resolution=ignore-duplicates,return=representation"),
                    params={"on_conflict": "id"},
                    json=profile,
                )
                resp.raise_for_status()
                rows: List[Dict[str, Any]] = resp.json()
                return rows[0] if rows else None  # None → row already existed
        except Exception as exc:
            logger.warning("Supabase insert_profile_if_new failed: %s", exc)
            return None

    def increment_llm_calls(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Atomically increment llm_calls_used with optimistic locking + limit guard.

        The PATCH enforces two conditions at the DB level:
        1. ``llm_calls_used = current_used`` (optimistic lock) — a concurrent
           request that already incremented causes zero rows matched → fail closed.
        2. ``llm_calls_used < llm_calls_limit`` (limit guard) — defense-in-depth
           so the DB never stores a value above the configured limit, even if
           reserve_llm_call is called with a stale quota view.

        Returns the updated row, or None on any failure (caller blocks the call).
        """
        if not self._enabled:
            return None
        profile = self.get_profile(user_id)
        if profile is None:
            return None
        current_used = int(profile.get("llm_calls_used", 0))
        limit = int(profile.get("llm_calls_limit", 0))

        # Pre-check: do not even attempt the write if already at/over limit.
        if current_used >= limit:
            logger.warning(
                "increment_llm_calls called for user %s but used=%d >= limit=%d; blocking.",
                user_id, current_used, limit,
            )
            return None

        new_used = current_used + 1
        url = f"{self._url}/rest/v1/user_profiles"
        # Pass params as list of tuples to allow two conditions on the same
        # column (httpx serialises them as separate query-string pairs which
        # PostgREST ANDs together).
        params_list = [
            ("id", f"eq.{user_id}"),
            ("llm_calls_used", f"eq.{current_used}"),   # optimistic lock
            ("llm_calls_used", f"lt.{limit}"),           # limit guard
        ]
        payload = {"llm_calls_used": new_used}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.patch(
                    url,
                    headers=self._headers(prefer="return=representation"),
                    params=params_list,
                    json=payload,
                )
                resp.raise_for_status()
                rows: List[Dict[str, Any]] = resp.json()
                return rows[0] if rows else None
        except Exception as exc:
            logger.warning("Supabase increment_llm_calls failed: %s", exc)
            return None

    def update_plan(
        self,
        user_id: str,
        *,
        plan: str,
        llm_calls_limit: int,
        stripe_customer_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        payload: Dict[str, Any] = {
            "plan": plan,
            "llm_calls_limit": llm_calls_limit,
        }
        if stripe_customer_id:
            payload["stripe_customer_id"] = stripe_customer_id
        url = f"{self._url}/rest/v1/user_profiles"
        params = {"id": f"eq.{user_id}"}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.patch(
                    url,
                    headers=self._headers(prefer="return=representation"),
                    params=params,
                    json=payload,
                )
                resp.raise_for_status()
                rows: List[Dict[str, Any]] = resp.json()
                return rows[0] if rows else None
        except Exception as exc:
            logger.warning("Supabase update_plan failed: %s", exc)
            return None
