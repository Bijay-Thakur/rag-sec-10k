"""Detect financial calculation intent from natural language questions."""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Tuple


class CalculationType(str, Enum):
    YEAR_OVER_YEAR_CHANGE = "year_over_year_change"
    PERCENTAGE_CHANGE = "percentage_change"
    ABSOLUTE_DIFFERENCE = "absolute_difference"
    REVENUE_COMPARISON = "revenue_comparison"
    OPERATING_INCOME_COMPARISON = "operating_income_comparison"
    SEGMENT_PRODUCT_NET_SALES = "segment_product_net_sales"
    MARGIN_CHANGE = "margin_change"


# Metric detection from question text — specific labels before generic "net sales"
_METRIC_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("iPhone", re.compile(r"\biphone\b", re.I)),
    ("Mac", re.compile(r"\bmac\b(?!\s*os)", re.I)),
    ("iPad", re.compile(r"\bipad\b", re.I)),
    ("Services", re.compile(r"\bservices\b(?!\s+net)", re.I)),
    ("Wearables, Home and Accessories", re.compile(r"\bwearables\b|\bhome\s+and\s+accessories\b", re.I)),
    ("operating income", re.compile(r"\boperating\s+income\b", re.I)),
    ("Research and development", re.compile(r"\b(r&d|research\s+and\s+development)\b", re.I)),
    ("Total gross margin", re.compile(r"\btotal\s+gross\s+margin\b|\bgross\s+margin\b", re.I)),
    ("Americas", re.compile(r"\bamericas\b", re.I)),
    ("Europe", re.compile(r"\beurope\b", re.I)),
    ("Greater China", re.compile(r"\bgreater\s+china\b|\bchina\b", re.I)),
    ("Japan", re.compile(r"\bjapan\b", re.I)),
    ("Rest of Asia Pacific", re.compile(r"\brest\s+of\s+asia\s+pacific\b", re.I)),
    ("total net sales", re.compile(r"\b(total\s+)?net\s+sales\b|\brevenue\b", re.I)),
)

_PERCENT_RE = re.compile(
    r"\b("
    r"growth\s+rate|percentage\s+change|percent\s+change|%+\s*change|"
    r"year[- ]over[- ]year|y/y|yoy|"
    r"what\s+percent|what\s+percentage"
    r")\b",
    re.I,
)

_ABSOLUTE_CHANGE_RE = re.compile(
    r"\b("
    r"difference\s+between|"
    r"how\s+much\s+(did|has)\s+.+\s+(increase|decrease|change)|"
    r"change\s+in|increase\s+in|decrease\s+in"
    r")\b",
    re.I,
)

_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|compared\s+to)\b", re.I)

_MARGIN_RE = re.compile(r"\b(margin|gross\s+margin\s+percentage|margin\s+percentage)\b", re.I)

_SEGMENT_PRODUCT_RE = re.compile(
    r"\b(segment|iphone|mac|ipad|services|americas|europe|china|japan|product\s+categor)",
    re.I,
)


def detect_metric(question: str) -> Optional[str]:
    for label, pattern in _METRIC_PATTERNS:
        if pattern.search(question):
            return label
    return None


def detect_calculation_type(question: str, metric: Optional[str]) -> CalculationType:
    q = question.lower()
    has_percent = bool(_PERCENT_RE.search(q))
    has_absolute = bool(_ABSOLUTE_CHANGE_RE.search(q))
    wants_absolute = has_absolute and not re.search(r"\b(rate|ratio|percent|%)\b", q, re.I)

    if metric == "operating income" and _COMPARE_RE.search(q):
        return CalculationType.OPERATING_INCOME_COMPARISON

    if metric and metric not in ("total net sales", "operating income", "Total gross margin") and _SEGMENT_PRODUCT_RE.search(q):
        if wants_absolute:
            return CalculationType.ABSOLUTE_DIFFERENCE
        if has_percent:
            return CalculationType.PERCENTAGE_CHANGE
        return CalculationType.SEGMENT_PRODUCT_NET_SALES

    if _MARGIN_RE.search(q):
        return CalculationType.MARGIN_CHANGE

    if wants_absolute:
        return CalculationType.ABSOLUTE_DIFFERENCE

    if has_percent:
        return CalculationType.PERCENTAGE_CHANGE

    if has_absolute:
        return CalculationType.ABSOLUTE_DIFFERENCE

    if _COMPARE_RE.search(q) and metric in ("total net sales", None):
        return CalculationType.REVENUE_COMPARISON

    if metric == "operating income":
        return CalculationType.OPERATING_INCOME_COMPARISON

    return CalculationType.YEAR_OVER_YEAR_CHANGE


def parse_target_years(question: str) -> Tuple[int, int]:
    """Return (newer_year, older_year) for comparisons. Defaults to 2025 vs 2024."""
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", question)]
    if len(years) >= 2:
        newer, older = max(years[0], years[1]), min(years[0], years[1])
        return newer, older
    if len(years) == 1:
        y = years[0]
        return y, y - 1
    return 2025, 2024
