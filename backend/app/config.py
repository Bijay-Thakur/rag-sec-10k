"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _env_bool_prefer(*names: str, default: str = "true") -> bool:
    """Return bool for the first env var that is explicitly set."""
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.lower() in ("1", "true", "yes")
    return default.lower() in ("1", "true", "yes")

def _find_repo_root() -> Path:
    """Repo root contains src/ and data/ (local dev or Docker /app layout)."""
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, here.parent.parent):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    return here.parent.parent


REPO_ROOT = _find_repo_root()
load_dotenv(REPO_ROOT / ".env")

# Make v1 pipeline importable (generation, retrieval, Embed)
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Settings:
    """Runtime settings — never embed secrets in code."""

    app_name: str = "SEC Insight AI"
    app_version: str = "1.0.0"

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    frontend_origin: str = os.getenv(
        "FRONTEND_ORIGIN",
        "http://localhost:3000,http://localhost:3001,http://localhost:3002,"
        "http://localhost:3003,http://127.0.0.1:3000,http://127.0.0.1:3001,"
        "http://localhost:8501",
    )

    chroma_db_dir: Path = REPO_ROOT / "db"
    chunks_dir: Path = REPO_ROOT / "data" / "chunks"
    eval_dir: Path = REPO_ROOT / "data" / "eval"

    default_filing_id: str = os.getenv("DEFAULT_FILING_ID", "apple_2025")
    retrieval_strategy: str = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    # LLM_MODEL is the Cloud Run alias; GENERATION_MODEL is the legacy name.
    generation_model: str = _first_env("LLM_MODEL", "GENERATION_MODEL", default="gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Cloud Run cost-safety knobs (0 = unlimited for daily caps)
    enable_live_llm_calls: bool = _env_bool("ENABLE_LIVE_LLM_CALLS", "true")
    max_daily_llm_calls: int = int(os.getenv("MAX_DAILY_LLM_CALLS", "0"))
    max_daily_estimated_cost_usd: float = float(
        os.getenv("MAX_DAILY_ESTIMATED_COST_USD", "0")
    )
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "4000"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "500"))

    # Rough OpenAI pricing (USD per 1M tokens) for cost estimates
    embed_input_price_per_1m: float = float(os.getenv("EMBED_PRICE_PER_1M", "0.02"))
    gpt4o_mini_input_price_per_1m: float = float(os.getenv("GPT_INPUT_PRICE_PER_1M", "0.15"))
    gpt4o_mini_output_price_per_1m: float = float(os.getenv("GPT_OUTPUT_PRICE_PER_1M", "0.60"))

    query_cache_max_entries: int = int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "128"))

    # ── Auth & access policy ─────────────────────────────────────────────
    enforce_access_policy: bool = os.getenv("ENFORCE_ACCESS_POLICY", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    anonymous_demo_only: bool = _env_bool_prefer(
        "DEMO_MODE_ONLY",
        "ANONYMOUS_DEMO_ONLY",
        default="true",
    )
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    # ── Quotas ───────────────────────────────────────────────────────────
    free_tier_llm_calls: int = int(os.getenv("FREE_TIER_LLM_CALLS", "3"))
    pro_tier_llm_calls_per_day: int = int(os.getenv("PRO_TIER_LLM_CALLS_PER_DAY", "100"))
    free_tier_daily_token_budget: int = int(os.getenv("FREE_TIER_DAILY_TOKEN_BUDGET", "0"))
    pro_tier_daily_token_budget: int = int(os.getenv("PRO_TIER_DAILY_TOKEN_BUDGET", "500000"))
    premium_price_label: str = os.getenv("PREMIUM_PRICE_LABEL", "$19.99/month")

    # ── Rate limiting ────────────────────────────────────────────────────
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    rate_limit_anonymous_per_hour: str = os.getenv("RATE_LIMIT_ANONYMOUS_PER_HOUR", "30/hour")
    rate_limit_authenticated_per_hour: str = os.getenv(
        "RATE_LIMIT_AUTHENTICATED_PER_HOUR", "120/hour"
    )
    rate_limit_pro_per_hour: str = os.getenv("RATE_LIMIT_PRO_PER_HOUR", "300/hour")


@lru_cache
def get_settings() -> Settings:
    return Settings()
