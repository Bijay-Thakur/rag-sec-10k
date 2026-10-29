"""Conservative deterministic routing; numeric operands never come from an LLM."""

import re
from enum import Enum


class Intent(str, Enum):
    NARRATIVE = "narrative"
    SINGLE_FACT = "single_fact"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    UNSUPPORTED = "unsupported"


def plan_query(question: str) -> Intent:
    value = question.casefold()
    if re.search(r"\b(predict|forecast|buy|sell|recommend|execute|password|api key)\b", value):
        return Intent.UNSUPPORTED
    if re.search(r"\b(compare|comparison|versus|vs)\b", value):
        return Intent.COMPARISON
    if re.search(r"\b(calculate|percent|percentage|ratio|margin|growth|yoy|cagr)\b", value):
        return Intent.CALCULATION
    if re.search(r"\b(how much|how many|revenue|net income|total assets|earnings per share)\b", value):
        return Intent.SINGLE_FACT
    return Intent.NARRATIVE
