import re
import sys
from pathlib import Path

from html_loader import clean_soup, extract_sections, load_html

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
SHORT_THRESHOLD = 500
TITLE_TRUNC = 60


def _item_sort_key(item_label):
    m = re.match(r"Item\s+(\d{1,2})([A-Z]?)", item_label)
    if not m:
        return (999, "")
    return (int(m.group(1)), m.group(2))


def _detect_gaps(item_labels):
    nums = sorted({_item_sort_key(i)[0] for i in item_labels})
    gaps = []
    for prev, cur in zip(nums, nums[1:]):
        for missing in range(prev + 1, cur):
            gaps.append(f"Item {missing}")
    by_num = {}
    for i in item_labels:
        n, letter = _item_sort_key(i)
        if letter:
            by_num.setdefault(n, []).append(letter)
    for n, letters in by_num.items():
        letters_sorted = sorted(letters)
        for prev, cur in zip(letters_sorted, letters_sorted[1:]):
            for missing in range(ord(prev) + 1, ord(cur)):
                gaps.append(f"Item {n}{chr(missing)}")
    return gaps


def _group_by_part(sections):
    groups = {}
    order = []
    for s in sections:
        key = s["part"]
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(s)
    return [(p, groups[p]) for p in order]


def report_filing(path):
    soup = clean_soup(load_html(path))
    sections, stats = extract_sections(soup, path.name)

    print(f"\n=== {path.name} ===")
    print(f"Parts detected:        {stats['parts_detected']}")
    print(f"Sections:              {len(sections)}")
    print(f"Duplicates suppressed: {stats['duplicates_suppressed']}")

    short = []
    for part, group in _group_by_part(sections):
        label = part if part is not None else "[NO PART]"
        print(f"  {label}")
        for s in group:
            title = (s["title"] or "")[:TITLE_TRUNC]
            print(f"    {s['item']:8s} {title:<{TITLE_TRUNC}s} ({len(s['text']):>7d} chars)")
            if len(s["text"]) < SHORT_THRESHOLD:
                short.append(f"{s['item']} ({len(s['text'])} chars)")

    all_items = [s["item"] for s in sections]
    gaps = _detect_gaps(all_items)
    print(f"Gaps:                  {gaps if gaps else 'none'}")
    print(f"Suspiciously short:    {short if short else 'none'}")

    return {
        "file": path.name,
        "parts": len(stats["parts_detected"]),
        "items": len(sections),
        "gaps": len(gaps),
        "dup": stats["duplicates_suppressed"],
        "no_part": not stats["parts_detected"],
    }


def main():
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else sorted(RAW_DIR.glob("*.html"))
    if not paths:
        print(f"No HTML files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    summaries = [report_filing(p) for p in paths]

    print("\n=== Summary ===")
    for s in summaries:
        note = " [no PART markers]" if s["no_part"] else ""
        print(f"  {s['file']:20s} parts={s['parts']} items={s['items']:3d} "
              f"gaps={s['gaps']} dup={s['dup']}{note}")


if __name__ == "__main__":
    main()
