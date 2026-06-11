"""API integration tests for auth enforcement on /api/ask."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def enforce_policy(monkeypatch):
    monkeypatch.setenv("ENFORCE_ACCESS_POLICY", "true")
    monkeypatch.setenv("ANONYMOUS_DEMO_ONLY", "true")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    get_settings = __import__(
        "backend.app.config", fromlist=["get_settings"]
    ).get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_anonymous_live_llm_returns_403(client):
    response = client.post(
        "/api/ask",
        json={
            "question": "What were Apple's total net sales in fiscal year 2025?",
            "filing_id": "apple_2025",
            "demo_mode": False,
        },
    )
    assert response.status_code == 403
    body = response.json()
    assert body["error_code"] == "auth_required"


def test_anonymous_entitlements(client):
    response = client.get("/api/me/entitlements")
    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is False
    assert body["plan"] == "anonymous"
    assert body["can_use_llm"] is False
    assert body["demo_mode_only"] is True
