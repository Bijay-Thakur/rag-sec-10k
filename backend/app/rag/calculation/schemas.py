"""Re-export calculation schemas from canonical RAG models."""

from backend.app.rag.schemas import CalculationDetail, CalculationInput
from pydantic import BaseModel, Field


class CalculationEngineResult(BaseModel):
    """Outcome of attempting a deterministic calculation."""

    success: bool
    detail: CalculationDetail | None = None
    answer_text: str = ""
    downgrade_to_partial: bool = False
    partial_reason: str = ""


__all__ = [
    "CalculationDetail",
    "CalculationInput",
    "CalculationEngineResult",
]
