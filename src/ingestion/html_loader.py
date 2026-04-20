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

PART_PATTERN = re.compile(r"^\s*PART\s+(I{1,3}V?|IV|V)\s*\.?\s*$")
ITEM_PATTERN = re.compile(
    r"^\s*ITEM\s+(\d{1,2}[A-Z]?)\s*[\.\u2014\u2013\-:]",
    re.IGNORECASE,
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
            current_item_part = current_part
            tail = ITEM_PATTERN.sub("", text, count=1).strip()
            current_title = tail
            current_text = []
            expecting_title = not tail
            continue

        if expecting_title and current_item is not None:
            current_title = text
            expecting_title = False
            continue

        if current_item is not None:
            current_text.append(text)

    flush()

    stats = {
        "parts_detected": parts_seen,
        "duplicates_suppressed": duplicates_suppressed,
    }
    return list(sections_by_key.values()), stats
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
