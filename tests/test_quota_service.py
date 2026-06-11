"""Tests for LLM quota consumption."""

from __future__ import annotations

import pytest

from backend.app.adapters.rag_adapter import RAGPipelineError
from backend.app.auth.models import UserContext
from backend.app.config import Settings
from backend.app.services import quota_service
from backend.app.services.quota_service import get_entitlements, reserve_llm_call


def _settings(**overrides) -> Settings:
    s = Settings()
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


@pytest.fixture(autouse=True)
def reset_memory_profiles():
    quota_service._memory_profiles.clear()
    yield
    quota_service._memory_profiles.clear()


def test_free_tier_allows_one_llm_call():
    user = UserContext(user_id="u-free", email="free@test.com", plan="free")
    settings = _settings(free_tier_llm_calls=1, enforce_access_policy=True)
    ent = reserve_llm_call(user, settings)
    assert ent.llm_calls_used == 1
    assert ent.llm_calls_remaining == 0


def test_free_tier_blocks_second_llm_call():
    user = UserContext(user_id="u-free2", email="free2@test.com", plan="free")
    settings = _settings(free_tier_llm_calls=1)
    reserve_llm_call(user, settings)
    with pytest.raises(RAGPipelineError) as exc:
        reserve_llm_call(user, settings)
    assert exc.value.error_code == "budget_exceeded"


def test_pro_tier_has_higher_limit():
    user = UserContext(user_id="u-pro", email="pro@test.com", plan="pro")
    settings = _settings(pro_tier_llm_calls_per_day=100)
    quota_service._memory_profiles[user.user_id] = {
        "id": user.user_id,
        "email": user.email,
        "plan": "pro",
        "llm_calls_used": 0,
        "llm_calls_limit": 100,
        "daily_tokens_used": 0,
    }
    ent = get_entitlements(user, settings)
    assert ent.plan == "pro"
    assert ent.llm_calls_limit == 100
    assert ent.can_use_llm is True
