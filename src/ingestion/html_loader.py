from bs4 import BeautifulSoup
import re
import warnings
from pathlib import Path
import sys

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

PART_PATTERN = re.compile(r"^\s*PART\s+(I{1,3}V?|IV|V)\s*\.?\s*$", re.IGNORECASE)
SIGNATURES_PATTERN = re.compile(r"^\s*signatures?\s*\.?\s*$", re.IGNORECASE)

SYNTHETIC_ITEM_HEADINGS = {
    "management's discussion and analysis of financial condition and results of operations": ("PART II", "Item 7"),
    "management's discussion and analysis": ("PART II", "Item 7"),
    "quantitative and qualitative disclosures about market risk": ("PART II", "Item 7A"),
    "report of independent registered public accounting firm": ("PART II", "Item 8"),
}
SYNTHETIC_ABSORPTION_ITEMS = {"Item 15", "Item 16"}
IBR_PATTERN = re.compile(
    r"incorporated\s+(herein\s+)?by\s+reference|reference\s+is\s+made\s+to",
    re.IGNORECASE,
)
IBR_SHORT_THRESHOLD = 500
ITEM_PATTERN = re.compile(
    r"^\s*ITEM\s+(\d{1,2}[A-Z]?)\s*[\.\u2014\u2013\-:]",
    re.IGNORECASE,
)


def _normalize_heading(text):
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .lower()
        .strip()
        .rstrip(".:;—–-")
        .strip()
    )


def _inside_anchor(node):
    p = node.parent
    while p is not None and p.name is not None:
        if p.name == "a":
            return True
        p = p.parent
    return False


def _normalize_part(roman):
    return f"PART {roman.upper()}"


def _normalize_item(num):
    num = num.upper()
    if num[-1].isalpha():
        return f"Item {num[:-1]}{num[-1]}"
    return f"Item {num}"


def _canonical_part_for_item(item_label, fallback):
    m = re.match(r"Item\s+(\d{1,2})", item_label)
    if not m:
        return fallback
    n = int(m.group(1))
    if 1 <= n <= 4:
        return "PART I"
    if 5 <= n <= 9:
        return "PART II"
    if 10 <= n <= 14:
        return "PART III"
    if 15 <= n <= 16:
        return "PART IV"
    return fallback


def load_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    return BeautifulSoup(html, "lxml")


def clean_soup(soup):
    for tag in soup(["script", "style"]):
        tag.decompose()
    for ix_tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):
        ix_tag.unwrap()
    return soup


def extract_sections(soup, source_file):
    sections_by_key = {}
    parts_seen = []
    duplicates_suppressed = 0
    synthetic_items_fired = set()
    in_synthetic_section = False

    current_part = None
    current_item = None
    current_item_part = None
    current_title = None
    current_text = []
    expecting_title = False

    def flush():
        nonlocal duplicates_suppressed
        if current_item is None:
            return
        section_text = "\n".join(current_text).strip()
        if not section_text:
            return
        key = (current_item_part, current_item)
        new_section = {
            "part": current_item_part,
            "item": current_item,
            "title": current_title,
            "text": section_text,
            "source_file": source_file,
        }
        if key in sections_by_key:
            duplicates_suppressed += 1
            if len(section_text) > len(sections_by_key[key]["text"]):
                sections_by_key[key] = new_section
        else:
            sections_by_key[key] = new_section

    for node in soup.find_all(string=True):
        text = str(node).replace("\xa0", " ").strip()
        if not text or _inside_anchor(node):
            continue

        if SIGNATURES_PATTERN.match(text):
            flush()
            current_item = None
            break

        pm = PART_PATTERN.match(text)
        if pm:
            current_part = _normalize_part(pm.group(1))
            if current_part not in parts_seen:
                parts_seen.append(current_part)
            expecting_title = False
            continue

        im = ITEM_PATTERN.match(text)
        if im:
            flush()
            current_item = _normalize_item(im.group(1))
            current_item_part = _canonical_part_for_item(current_item, current_part)
            tail = ITEM_PATTERN.sub("", text, count=1).strip()
            current_title = tail
            current_text = []
            expecting_title = not tail
            in_synthetic_section = False
            continue

        if len(text) < 150:
            mapping = SYNTHETIC_ITEM_HEADINGS.get(_normalize_heading(text))
            if mapping and mapping not in synthetic_items_fired and (
                in_synthetic_section or current_item in SYNTHETIC_ABSORPTION_ITEMS
            ):
                synthetic_items_fired.add(mapping)
                in_synthetic_section = True
                flush()
                current_item_part, current_item = mapping
                current_title = text
                current_text = []
                expecting_title = False
                continue

        if expecting_title and current_item is not None:
            current_title = text
            expecting_title = False
            continue

        if current_item is not None:
            current_text.append(text)

    flush()

    sections = list(sections_by_key.values())
    _mark_incorporated_by_reference(sections)

    stats = {
        "parts_detected": parts_seen,
        "duplicates_suppressed": duplicates_suppressed,
        "synthetic_items": sorted(f"{p}/{i}" for (p, i) in synthetic_items_fired),
    }
    return sections, stats


def _mark_incorporated_by_reference(sections):
    for s in sections:
        t = s["text"]
        s["incorporated_by_reference"] = (
            len(t) < IBR_SHORT_THRESHOLD and bool(IBR_PATTERN.search(t))
        )
def flag_oversized_sections(sections, max_chars=200_000):
    for s in sections:
        s["oversized"] = len(s["text"]) > max_chars
    return sections

if __name__ == "__main__":
    path = sys.argv[1]
    soup = clean_soup(load_html(path))
    sections, stats = extract_sections(soup, Path(path).name)
    print(f"found {len(sections)} sections in {path}")
    print(f"parts detected: {stats['parts_detected']}")
    print(f"duplicates suppressed: {stats['duplicates_suppressed']}\n")
    current_part = object()
    for s in sections:
        if s["part"] != current_part:
            current_part = s["part"]
            print(f"\n[{current_part or 'NO PART'}]")
        print(f"  {s['item']:8s} {s['title'][:60]:60s} ({len(s['text']):>7d} chars)")
