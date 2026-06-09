"""Load persisted evaluation summaries from data/eval/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from backend.app.config import get_settings
from backend.app.schemas import EvalSummaryResponse


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_eval_summary() -> EvalSummaryResponse:
    settings = get_settings()
    eval_dir = settings.eval_dir

    retrieval_v1 = _load_json(eval_dir / "retrieval_summary.json") or []
    ragas_v1 = _load_json(eval_dir / "ragas_summary.json") or {}
    retrieval_v2 = _load_json(eval_dir / "v2_retrieval_summary.json") or []
    gen_v2 = _load_json(eval_dir / "v2_generation_summary.json") or {}

    if not isinstance(retrieval_v1, list):
        retrieval_v1 = []
    if not isinstance(ragas_v1, dict):
        ragas_v1 = {}
    if not isinstance(retrieval_v2, list):
        retrieval_v2 = []
    if not isinstance(gen_v2, dict):
        gen_v2 = {}

    return EvalSummaryResponse(
        retrieval_v1=retrieval_v1,
        generation_v1_ragas=ragas_v1,
        retrieval_v2=retrieval_v2,
        generation_v2=gen_v2,
    )
