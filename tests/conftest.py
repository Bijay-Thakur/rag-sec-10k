"""Shared pytest fixtures for API smoke tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Test defaults — disable rate limits; keep access policy on for auth tests.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("ENFORCE_ACCESS_POLICY", "true")
os.environ.setdefault("ANONYMOUS_DEMO_ONLY", "true")


@pytest.fixture(scope="module")
def client() -> TestClient:
    from backend.app.config import get_settings
    from backend.app.main import app

    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
