"""Tests for server-side access policy enforcement."""

from __future__ import annotations

import pytest

from backend.app.adapters.rag_adapter import RAGPipelineError
from backend.app.auth.models import UserContext
from backend.app.config import Settings
from backend.app.services.access_policy import resolve_ask_access


def _settings(**overrides) -> Settings:
    s = Settings()
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def test_anonymous_forced_to_demo():
    access = resolve_ask_access(
        requested_demo_mode=True,
        user=None,
        settings=_settings(enforce_access_policy=True, anonymous_demo_only=True),
    )
    assert access.effective_demo_mode is True
    assert access.user is None


def test_anonymous_live_llm_rejected():
    with pytest.raises(RAGPipelineError) as exc:
        resolve_ask_access(
            requested_demo_mode=False,
            user=None,
            settings=_settings(enforce_access_policy=True, anonymous_demo_only=True),
        )
    assert exc.value.error_code == "auth_required"


def test_authenticated_demo_skips_quota():
    user = UserContext(user_id="u1", email="a@b.com", plan="free")
    access = resolve_ask_access(
        requested_demo_mode=True,
        user=user,
        settings=_settings(enforce_access_policy=True, free_tier_llm_calls=1),
    )
    assert access.effective_demo_mode is True


def test_policy_disabled_allows_client_demo_flag():
    access = resolve_ask_access(
        requested_demo_mode=False,
        user=None,
        settings=_settings(enforce_access_policy=False),
    )
    assert access.effective_demo_mode is False
