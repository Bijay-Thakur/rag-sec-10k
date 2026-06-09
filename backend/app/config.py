"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Repo root: backend/app/config.py → parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
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
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Rough OpenAI pricing (USD per 1M tokens) for cost estimates
    embed_input_price_per_1m: float = float(os.getenv("EMBED_PRICE_PER_1M", "0.02"))
    gpt4o_mini_input_price_per_1m: float = float(os.getenv("GPT_INPUT_PRICE_PER_1M", "0.15"))
    gpt4o_mini_output_price_per_1m: float = float(os.getenv("GPT_OUTPUT_PRICE_PER_1M", "0.60"))

    query_cache_max_entries: int = int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "128"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
