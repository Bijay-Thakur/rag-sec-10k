"""
Deterministic financial calculation engine.

All numeric results are computed in Python — never by the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.app.rag.calculation.extractors import (
    ExtractedSeries,
    extract_metric_from_hits,
    get_value_for_year,
)
from backend.app.rag.calculation.intents import (
    CalculationType,
    detect_calculation_type,
    detect_metric,
    parse_target_years,
)
from backend.app.rag.calculation.schemas import (
    CalculationDetail,
    CalculationEngineResult,
    CalculationInput,
)

CALC_EXPLAIN_HINT = """
IMPORTANT — Pre-computed figures (do NOT recalculate or change any number):
{calc_block}

Explain this result in plain language for a financial analyst.
Cite source passages with [n]. Use EXACTLY the numeric values above.
Do not invent or estimate any figures not listed above.
"""


def _fmt_millions(value: float) -> str:
    return f"${value:,.0f} million"


def _fmt_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def _series_inputs(series: ExtractedSeries, years: List[int]) -> List[CalculationInput]:
    inputs: List[CalculationInput] = []
    for year in years:
        val = get_value_for_year(series, year)
        if val is None:
            continue
        inputs.append(CalculationInput(
            label=f"{series.metric} ({year})",
            value=val,
            unit=series.unit,
            year=year,
            source_chunk_id=series.chunk_id,
            source_excerpt=series.source_excerpt[:200],
        ))
    return inputs


def _compute_yoy_change(
    series: ExtractedSeries,
    newer_year: int,
    older_year: int,
    calc_type: CalculationType,
) -> CalculationEngineResult:
    newer = get_value_for_year(series, newer_year)
    older = get_value_for_year(series, older_year)

    if newer is None or older is None:
        return CalculationEngineResult(
            success=False,
            downgrade_to_partial=True,
            partial_reason=(
                f"Found {series.metric} data but not for both {newer_year} and {older_year}."
            ),
        )

    if older == 0:
        return CalculationEngineResult(
            success=False,
            downgrade_to_partial=True,
            partial_reason="Cannot compute percentage change — prior year value is zero.",
        )

    abs_change = newer - older
    pct_change = (abs_change / older) * 100.0
    inputs = _series_inputs(series, [newer_year, older_year])

    if series.is_percentage or calc_type == CalculationType.MARGIN_CHANGE:
        formula = f"{series.metric} ({newer_year}) − {series.metric} ({older_year})"
        result_str = f"{_fmt_pct(newer - older)} percentage points ({newer:.1f}% → {older:.1f}%)"
        answer = (
            f"The {series.metric.lower()} was {newer:.1f}% in {newer_year} and "
            f"{older:.1f}% in {older_year}, a change of {_fmt_pct(newer - older)} percentage points."
        )
        detail_type = "margin_change"
        result_value = newer - older
    elif calc_type in (CalculationType.PERCENTAGE_CHANGE, CalculationType.YEAR_OVER_YEAR_CHANGE):
        formula = f"(({newer:,.0f} − {older:,.0f}) / {older:,.0f}) × 100"
        result_str = _fmt_pct(pct_change)
        answer = (
            f"{series.metric} {_fmt_millions(abs_change).replace('$', '$')} "
            f"({'increased' if abs_change >= 0 else 'decreased'}) year-over-year from "
            f"{older_year} to {newer_year}, a {_fmt_pct(pct_change)} change "
            f"({_fmt_millions(older)} → {_fmt_millions(newer)})."
        )
        detail_type = "percentage_change"
        result_value = pct_change
    else:
        formula = f"{newer:,.0f} − {older:,.0f}"
        result_str = _fmt_millions(abs_change)
        answer = (
            f"{series.metric} changed by {_fmt_millions(abs_change)} from {older_year} to {newer_year} "
            f"({_fmt_millions(older)} → {_fmt_millions(newer)})."
        )
        detail_type = "absolute_difference"
        result_value = abs_change

    detail = CalculationDetail(
        calculation_type=detail_type,
        inputs=inputs,
        formula=formula,
        result=result_str,
        result_value=round(result_value, 4),
        source_chunk_ids=[series.chunk_id],
        success=True,
        confidence=series.confidence,
        metric=series.metric,
        extra={"newer_year": newer_year, "older_year": older_year},
    )
    return CalculationEngineResult(success=True, detail=detail, answer_text=answer)


def _compute_comparison(
    series: ExtractedSeries,
    newer_year: int,
    older_year: int,
    comparison_type: CalculationType,
) -> CalculationEngineResult:
    newer = get_value_for_year(series, newer_year)
    older = get_value_for_year(series, older_year)

    if newer is None or older is None:
        return CalculationEngineResult(
            success=False,
            downgrade_to_partial=True,
            partial_reason=f"Insufficient {series.metric} data for {newer_year} vs {older_year}.",
        )

    abs_change = newer - older
    pct_change = (abs_change / older * 100.0) if older else 0.0
    inputs = _series_inputs(series, [newer_year, older_year])

    type_label = comparison_type.value
    answer = (
        f"{series.metric} was {_fmt_millions(newer)} in {newer_year} vs "
        f"{_fmt_millions(older)} in {older_year} "
        f"({'+' if abs_change >= 0 else ''}{abs_change:,.0f} million, {_fmt_pct(pct_change)})."
    )

    detail = CalculationDetail(
        calculation_type=type_label,
        inputs=inputs,
        formula=f"Compare {newer_year} vs {older_year}; Δ = {newer:,.0f} − {older:,.0f}",
        result=f"{_fmt_millions(newer)} ({newer_year}) vs {_fmt_millions(older)} ({older_year})",
        result_value=abs_change,
        source_chunk_ids=[series.chunk_id],
        success=True,
        confidence=series.confidence,
        metric=series.metric,
    )
    return CalculationEngineResult(success=True, detail=detail, answer_text=answer)


def _compute_segment_lookup(series: ExtractedSeries, year: int) -> CalculationEngineResult:
    val = get_value_for_year(series, year)
    if val is None:
        return CalculationEngineResult(
            success=False,
            downgrade_to_partial=True,
            partial_reason=f"No {series.metric} figure found for {year}.",
        )

    inputs = _series_inputs(series, [year])
    answer = f"{series.metric} net sales were {_fmt_millions(val)} in {year}."

    detail = CalculationDetail(
        calculation_type="segment_product_net_sales",
        inputs=inputs,
        formula=f"Direct disclosure for {year}",
        result=_fmt_millions(val),
        result_value=val,
        source_chunk_ids=[series.chunk_id],
        success=True,
        confidence=series.confidence,
        metric=series.metric,
    )
    return CalculationEngineResult(success=True, detail=detail, answer_text=answer)


def format_calc_block(detail: CalculationDetail) -> str:
    lines = [
        f"Metric: {detail.metric}",
        f"Calculation type: {detail.calculation_type}",
        f"Formula: {detail.formula}",
        f"Result: {detail.result}",
    ]
    for inp in detail.inputs:
        lines.append(f"  - {inp.label}: {inp.value:,.2f} {inp.unit}")
    return "\n".join(lines)


def run_calculation(question: str, hits: List[Dict[str, Any]]) -> CalculationEngineResult:
    """
    Attempt a deterministic calculation from retrieved chunks.

    Returns success=False with downgrade_to_partial=True when figures
    cannot be confidently extracted — caller must not let the LLM guess.
    """
    if not hits:
        return CalculationEngineResult(
            success=False,
            downgrade_to_partial=True,
            partial_reason="No retrieved chunks available for calculation.",
        )

    metric = detect_metric(question) or "total net sales"
    calc_type = detect_calculation_type(question, metric)
    newer_year, older_year = parse_target_years(question)

    prefer_pct = calc_type == CalculationType.MARGIN_CHANGE
    lookup_metric = metric
    if calc_type == CalculationType.MARGIN_CHANGE and "Products" in question:
        lookup_metric = "Products"

    extraction = extract_metric_from_hits(
        hits,
        lookup_metric,
        prefer_percentage=prefer_pct,
    )

    if not extraction.confident or extraction.series is None:
        return CalculationEngineResult(
            success=False,
            downgrade_to_partial=True,
            partial_reason=extraction.reason or "Could not confidently extract figures from chunks.",
        )

    series = extraction.series

    if calc_type == CalculationType.SEGMENT_PRODUCT_NET_SALES:
        year = newer_year if str(newer_year) in question else newer_year
        return _compute_segment_lookup(series, year)

    if calc_type in (CalculationType.REVENUE_COMPARISON, CalculationType.OPERATING_INCOME_COMPARISON):
        return _compute_comparison(series, newer_year, older_year, calc_type)

    return _compute_yoy_change(series, newer_year, older_year, calc_type)
