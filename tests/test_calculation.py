"""Tests for deterministic financial calculation engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.rag.calculation.engine import run_calculation
from backend.app.rag.calculation.extractors import extract_metric_from_hits

# Real excerpt patterns from Apple 2025 semantic_chunks.jsonl
NET_SALES_CHUNK = {
    "id": "apple_item7_chunk_007",
    "text": (
        "Apple Inc. | 2025 Form 10-K | 22\nProducts and Services Performance\n"
        "The following table shows net sales by category for 2025, 2024 and 2023 (dollars in millions):\n"
        "2025\nChange\n2024\nChange\n2023\niPhone\n$\n209,586\n4\n%\n$\n201,183\n—\n%\n$\n200,583\n"
        "Mac\n33,708\n12\n%\n29,984\n2\n%\n29,357\niPad\n28,023\n5\n%\n26,694\n(6)\n%\n28,300\n"
        "Wearables, Home and Accessories\n35,686\n(4)\n%\n37,005\n(7)\n%\n39,845\nServices\n(1)\n"
        "109,158\n14\n%\n96,169\n13\n%\n85,200\nTotal net sales\n$\n416,161\n6\n%\n$\n391,035\n2\n%\n$\n383,285\n"
        "(1)\nServices net sales include amortization of the deferred value of services bundled in the sales price of certain products."
    ),
    "metadata": {"source_file": "Apple.html", "item": "Item 7", "part": "PART II"},
}

OPERATING_INCOME_CHUNK = {
    "id": "apple_item8_chunk_002",
    "text": (
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions, except number of shares, which are reflected in thousands, and per-share amounts)\n"
        "Years ended\nSeptember 27,\n2025\nSeptember 28,\n2024\nSeptember 30,\n2023\n"
        "Net sales:\nProducts\n$\n307,003\n$\n294,866\n$\n298,085\nServices\n109,158\n96,169\n85,200\n"
        "Total net sales\n416,161\n391,035\n383,285\nCost of sales:\nProducts\n194,116\n185,233\n189,282\n"
        "Services\n26,844\n25,119\n24,855\nTotal cost of sales\n220,960\n210,352\n214,137\n"
        "Gross margin\n195,201\n180,683\n169,148\nOperating expenses:\nResearch and development\n34,550\n31,370\n29,915\n"
        "Selling, general and administrative\n27,601\n26,097\n24,932\nTotal operating expenses\n62,151\n57,467\n54,847\n"
        "Operating income\n133,050\n123,216\n114,301\n"
    ),
    "metadata": {"source_file": "Apple.html", "item": "Item 8", "part": "PART II"},
}

IPHONE_ROW_CHUNK = {
    "id": "apple_item7_chunk_007b",
    "text": NET_SALES_CHUNK["text"],
    "metadata": NET_SALES_CHUNK["metadata"],
}


def test_extract_total_net_sales_from_mda_table():
    result = extract_metric_from_hits([NET_SALES_CHUNK], "total net sales")
    assert result.confident
    assert result.series is not None
    assert result.series.values_by_year[2025] == pytest.approx(416_161)
    assert result.series.values_by_year[2024] == pytest.approx(391_035)


def test_yoy_revenue_growth_rate_calculation():
    question = "What was Apple's year-over-year revenue growth rate in 2025?"
    result = run_calculation(question, [NET_SALES_CHUNK])

    assert result.success
    assert result.detail is not None
    assert result.detail.calculation_type == "percentage_change"
    assert result.detail.metric == "total net sales"
    assert len(result.detail.inputs) == 2
    assert result.detail.source_chunk_ids == ["apple_item7_chunk_007"]

    # (416161 - 391035) / 391035 * 100 ≈ 6.42%
    assert result.detail.result_value == pytest.approx(6.4226, rel=0.01)
    assert "6.42" in result.detail.result or "6.4" in result.detail.result
    assert "416" in result.answer_text
    assert "391" in result.answer_text


def test_operating_income_comparison():
    question = "Compare Apple's operating income in 2025 versus 2024."
    result = run_calculation(question, [OPERATING_INCOME_CHUNK])

    assert result.success
    assert result.detail is not None
    assert result.detail.calculation_type == "operating_income_comparison"
    assert result.detail.inputs[0].value == pytest.approx(133_050)
    assert result.detail.inputs[1].value == pytest.approx(123_216)
    assert result.detail.result_value == pytest.approx(9_834)
    assert result.detail.formula


def test_iphone_net_sales_absolute_change():
    question = "How much did iPhone net sales change year-over-year in 2025?"
    result = run_calculation(question, [IPHONE_ROW_CHUNK])

    assert result.success
    assert result.detail is not None
    assert result.detail.metric == "iPhone"
    # 209586 - 201183 = 8403
    assert result.detail.result_value == pytest.approx(8_403)
    assert result.detail.calculation_type in ("absolute_difference", "percentage_change", "year_over_year_change")


def test_extraction_failure_returns_partial_not_guess():
    question = "What is the year-over-year growth rate of Martian colony revenue?"
    weak_hit = {
        "id": "apple_item1_chunk_001",
        "text": "The Company designs and markets smartphones and services worldwide.",
        "metadata": {"source_file": "Apple.html", "item": "Item 1"},
    }
    result = run_calculation(question, [weak_hit])

    assert not result.success
    assert result.downgrade_to_partial
    assert result.detail is None
    assert "Could not find" in result.partial_reason or "confidently" in result.partial_reason.lower()
