"""Extract financial figures from SEC 10-K chunk text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Default fiscal years for Apple 2025 10-K tables (newest first)
DEFAULT_YEARS: Tuple[int, ...] = (2025, 2024, 2023)

_DOLLAR_AMOUNT_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:billion|million|bn|mm)?",
    re.IGNORECASE,
)

_PCT_VALUE_RE = re.compile(r"([\d.]+)\s*%")


@dataclass
class ExtractedSeries:
    """Time series extracted from one chunk."""

    metric: str
    values_by_year: Dict[int, float]
    unit: str
    chunk_id: str
    source_excerpt: str
    confidence: float
    is_percentage: bool = False


@dataclass
class ExtractionResult:
    series: Optional[ExtractedSeries] = None
    confident: bool = False
    reason: str = ""


def _parse_number(token: str) -> Optional[float]:
    token = token.strip().replace(",", "").replace("$", "")
    token = token.replace("(", "-").replace(")", "")
    if not token or token == "—" or token == "-":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _normalize_metric(label: str) -> str:
    return label.strip()


def _extract_line_series(
    text: str,
    metric: str,
    chunk_id: str,
    requested_metric: str,
) -> Optional[ExtractedSeries]:
    """
    Parse Item 8 style line tables:
        Operating income
        133,050
        123,216
        114,301
    """
    lines = [ln.strip() for ln in text.splitlines()]
    metric_lower = metric.lower()

    for i, line in enumerate(lines):
        if metric_lower not in line.lower():
            continue
        # Skip long narrative lines that merely mention the metric
        if len(line) > len(metric) + 40:
            continue

        values: List[float] = []
        for j in range(i + 1, min(i + 8, len(lines))):
            val = _parse_number(lines[j])
            if val is not None and val >= 0:
                values.append(val)
            elif values:
                break

        if len(values) >= 2:
            years = DEFAULT_YEARS[: len(values)]
            excerpt = "\n".join(lines[max(0, i - 1) : i + len(values) + 2])[:400]
            return ExtractedSeries(
                metric=requested_metric,
                values_by_year=dict(zip(years, values)),
                unit="USD millions",
                chunk_id=chunk_id,
                source_excerpt=excerpt,
                confidence=0.95 if len(values) >= 3 else 0.85,
            )
    return None


def _extract_dollar_table_row(text: str, metric: str, chunk_id: str, requested_metric: str) -> Optional[ExtractedSeries]:
    """
    Parse MD&A table rows — collect only $-prefixed amounts after the row label
    (skips percentage change columns like ``6 %``).
    """
    idx = text.lower().find(metric.lower())
    if idx == -1:
        return None

    segment = text[idx : idx + 600]
    dollar_matches = re.findall(r"\$\s*([\d,]+)", segment)
    if len(dollar_matches) < 2:
        return None

    values = [float(v.replace(",", "")) for v in dollar_matches[:3]]
    years = DEFAULT_YEARS[: len(values)]
    excerpt = segment[:400]

    return ExtractedSeries(
        metric=requested_metric,
        values_by_year=dict(zip(years, values)),
        unit="USD millions",
        chunk_id=chunk_id,
        source_excerpt=excerpt,
        confidence=0.9,
    )


def _extract_percentage_row(
    text: str,
    metric: str,
    chunk_id: str,
    requested_metric: str,
) -> Optional[ExtractedSeries]:
    """Parse gross margin percentage rows: Products 36.8 % 37.2 % 36.5 %"""
    pattern = (
        rf"{re.escape(metric)}\s*"
        r"([\d.]+)\s*%\s*"
        r"([\d.]+)\s*%\s*"
        r"([\d.]+)\s*%"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None

    values = [float(g) for g in m.groups()]
    years = DEFAULT_YEARS[: len(values)]
    excerpt = text[max(0, m.start() - 20) : m.end() + 20][:400]

    return ExtractedSeries(
        metric=requested_metric,
        values_by_year=dict(zip(years, values)),
        unit="percent",
        chunk_id=chunk_id,
        source_excerpt=excerpt,
        confidence=0.9,
        is_percentage=True,
    )


# Aliases for row labels in filing text
METRIC_ROW_LABELS: Dict[str, List[str]] = {
    "total net sales": ["Total net sales", "Net sales"],
    "iPhone": ["iPhone"],
    "Mac": ["Mac"],
    "iPad": ["iPad"],
    "Services": ["Services"],
    "Wearables, Home and Accessories": ["Wearables, Home and Accessories"],
    "operating income": ["Operating income", "Operating income/(loss)"],
    "Research and development": ["Research and development"],
    "Total gross margin": ["Total gross margin"],
    "Products": ["Products"],
    "Americas": ["Americas"],
    "Europe": ["Europe"],
    "Greater China": ["Greater China"],
    "Japan": ["Japan"],
    "Rest of Asia Pacific": ["Rest of Asia Pacific"],
}


def extract_metric_from_hits(
    hits: List[Dict[str, Any]],
    metric: str,
    *,
    prefer_percentage: bool = False,
) -> ExtractionResult:
    """Search retrieved chunks for a metric time series."""
    labels = METRIC_ROW_LABELS.get(metric, [metric])
    best: Optional[ExtractedSeries] = None

    for hit in hits:
        text = hit.get("text") or ""
        chunk_id = hit.get("id") or hit.get("chunk_id") or ""

        for label in labels:
            candidates: List[Optional[ExtractedSeries]] = []

            if prefer_percentage:
                candidates.append(_extract_percentage_row(text, label, chunk_id, metric))

            candidates.extend([
                _extract_line_series(text, label, chunk_id, metric),
                _extract_dollar_table_row(text, label, chunk_id, metric),
            ])

            for series in candidates:
                if series is None:
                    continue
                if best is None or series.confidence > best.confidence:
                    best = series

    if best is None:
        return ExtractionResult(
            confident=False,
            reason=f"Could not find '{metric}' figures in retrieved chunks.",
        )

    if len(best.values_by_year) < 2:
        return ExtractionResult(
            confident=False,
            reason=f"Only one year of '{metric}' data found — need at least two for comparison.",
        )

    return ExtractionResult(series=best, confident=best.confidence >= 0.8, reason="")


def get_value_for_year(series: ExtractedSeries, year: int) -> Optional[float]:
    return series.values_by_year.get(year)
