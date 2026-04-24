import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "ingestion"))

from html_loader import clean_soup, extract_sections, load_html  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def parsed():
    out = {}
    for name in ("Apple", "Chase", "Elilily", "Exxon", "Walmart"):
        path = RAW_DIR / f"{name}.html"
        soup = clean_soup(load_html(path))
        sections, stats = extract_sections(soup, path.name)
        index = {(s["part"], s["item"]): s for s in sections}
        out[name] = {"sections": sections, "stats": stats, "index": index}
    return out


def _chars(parsed, name, part, item):
    s = parsed[name]["index"].get((part, item))
    return len(s["text"]) if s else 0


def test_exxon_md_a_recovered(parsed):
    assert _chars(parsed, "Exxon", "PART II", "Item 7") > 50_000


def test_exxon_item_16_is_small(parsed):
    assert _chars(parsed, "Exxon", "PART IV", "Item 16") < 500


def test_exxon_item_8_recovered(parsed):
    assert _chars(parsed, "Exxon", "PART II", "Item 8") > 50_000


def test_exxon_all_canonical_parts_present(parsed):
    pd = parsed["Exxon"]["stats"]["parts_detected"]
    assert set(pd) == {"PART I", "PART II", "PART III", "PART IV"}


def test_chase_item_15_is_small(parsed):
    assert _chars(parsed, "Chase", "PART IV", "Item 15") < 20_000


def test_chase_md_a_recovered(parsed):
    assert _chars(parsed, "Chase", "PART II", "Item 7") > 50_000


def test_chase_item_8_recovered(parsed):
    assert _chars(parsed, "Chase", "PART II", "Item 8") > 50_000


def test_chase_items_have_canonical_parts(parsed):
    # Chase's HTML only contains literal "Part I" and "Part IV" text nodes,
    # but items 5-14 must still be mapped to Part II / III via the canonical
    # item-number -> part mapping.
    idx = parsed["Chase"]["index"]
    assert ("PART II", "Item 7") in idx
    assert ("PART II", "Item 8") in idx
    assert ("PART III", "Item 10") in idx


@pytest.mark.parametrize("name", ["Apple", "Walmart", "Elilily"])
def test_standard_filings_have_substantial_md_a(parsed, name):
    # Sanity check: well-formed filers retain real MD&A under Part II / Item 7.
    assert _chars(parsed, name, "PART II", "Item 7") > 10_000


@pytest.mark.parametrize("name", ["Apple", "Walmart", "Elilily"])
def test_standard_filings_have_substantial_item_8(parsed, name):
    assert _chars(parsed, name, "PART II", "Item 8") > 40_000


@pytest.mark.parametrize("name", ["Apple", "Walmart", "Elilily"])
def test_standard_filings_item_16_stays_small(parsed, name):
    # Form 10-K Summary is either "None." or a tiny block for these filers;
    # ensure no trailing content leaked in.
    assert _chars(parsed, name, "PART IV", "Item 16") < 500


def test_ibr_flag_set_on_stub(parsed):
    # Apple's Part III items are short pointers to the proxy statement;
    # they should be flagged incorporated_by_reference.
    idx = parsed["Apple"]["index"]
    s = idx.get(("PART III", "Item 11"))
    assert s is not None
    assert s.get("incorporated_by_reference") is True


def test_no_oversized_sections(parsed):
    # After the fix, no section should exceed ~700k chars (Chase's Item 8 is
    # the largest at ~580k). Guards against future regressions that would
    # re-absorb the signatures/exhibits tail.
    for name, data in parsed.items():
        for s in data["sections"]:
            assert len(s["text"]) < 700_000, (
                f"{name} {s['part']}/{s['item']} = {len(s['text'])} chars"
            )
