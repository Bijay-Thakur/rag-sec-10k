import argparse
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from html_loader import clean_soup, extract_sections, load_html
CHUNK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

DEFAULT_FIXED_CHARS = 1200
DEFAULT_FIXED_OVERLAP = 300
DEFAULT_RECURSIVE_TARGET = 1200
DEFAULT_SEMANTIC_MIN = 700
DEFAULT_SEMANTIC_MAX = 1300

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\b[\w'-]+\b")


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _item_slug(item: str) -> str:
    return _slug(item.replace(" ", ""))


def _token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _build_chunk(
    *,
    source_file: str,
    part: Optional[str],
    item: str,
    section_title: Optional[str],
    strategy: str,
    chunk_index: int,
    text: str,
) -> Dict:
    item_id = _item_slug(item)
    src_id = _slug(Path(source_file).stem)
    chunk_id = f"{src_id}_{item_id}_chunk_{chunk_index:03d}"
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "source_file": source_file,
            "part": part,
            "item": item,
            "section_title": section_title or "",
            "chunk_strategy": strategy,
            "chunk_index": chunk_index,
            "char_count": len(text),
            "token_count": _token_count(text),
        },
    }


def fixed_size_chunk_section(
    section: Dict,
    chunk_size: int = DEFAULT_FIXED_CHARS,
    overlap: int = DEFAULT_FIXED_OVERLAP,
) -> List[Dict]:
    text = section["text"]
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Dict] = []
    chunk_texts: List[str] = []

    cur_sentences: List[str] = []
    cur_len = 0

    for sentence in sentences:
        # +1 for the space when joining, if not the first sentence
        sl = len(sentence) + (1 if cur_sentences else 0)
        if cur_len + sl <= chunk_size:
            cur_sentences.append(sentence)
            cur_len += sl
        else:
            # finalize current chunk
            chunk_text = " ".join(cur_sentences).strip()
            if chunk_text:
                chunk_texts.append(chunk_text)

            # build new chunk with overlap from previous chunk
            if overlap > 0 and chunk_texts:
                prev = chunk_texts[-1]
                prev_sents = _split_sentences(prev)
                tail: List[str] = []
                tail_len = 0
                # take sentences from the end of previous chunk until we hit overlap
                for s in reversed(prev_sents):
                    s_len = len(s) + (1 if tail else 0)
                    if tail_len + s_len <= overlap:
                        tail.insert(0, s)
                        tail_len += s_len
                    else:
                        break
                cur_sentences = tail + [sentence]
                cur_len = sum(len(s) + (1 if i > 0 else 0) for i, s in enumerate(cur_sentences))
            else:
                cur_sentences = [sentence]
                cur_len = len(sentence)

    # flush last chunk
    if cur_sentences:
        chunk_text = " ".join(cur_sentences).strip()
        if chunk_text:
            chunk_texts.append(chunk_text)

    # build final chunk objects
    out: List[Dict] = []
    for idx, chunk_text in enumerate(chunk_texts, start=1):
        out.append(
            _build_chunk(
                source_file=section["source_file"],
                part=section["part"],
                item=section["item"],
                section_title=section.get("title"),
                strategy="fixed_size",
                chunk_index=idx,
                text=chunk_text,
            )
        )
    return out



def _recursive_split_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text.strip()]

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) > 1:
        out = []
        for p in paragraphs:
            if len(p) <= max_chars:
                out.append(p)
            else:
                out.extend(_recursive_split_text(p, max_chars))
        return [x.strip() for x in out if x.strip()]

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        out = []
        cur = []
        cur_len = 0
        for s in sentences:
            sl = len(s) + (1 if cur else 0)
            if cur_len + sl <= max_chars:
                cur.append(s)
                cur_len += sl
            else:
                out.append(" ".join(cur))
                cur = [s]
                cur_len = len(s)
        if cur:
            out.append(" ".join(cur))
        return [x.strip() for x in out if x.strip()]

    # final fallback for extremely long single-unit content
    hard = []
    i = 0
    while i < len(text):
        hard.append(text[i : i + max_chars].strip())
        i += max_chars
    return [x for x in hard if x]


def recursive_chunk_sections(
    sections: List[Dict], target_chars: int = DEFAULT_RECURSIVE_TARGET
) -> List[Dict]:
    grouped_by_part: Dict[Optional[str], List[Dict]] = {}
    for s in sections:
        grouped_by_part.setdefault(s["part"], []).append(s)

    out = []
    # Part -> Item -> Paragraph -> Sentence -> hard split
    for _, part_sections in grouped_by_part.items():
        for section in part_sections:
            split_units = _recursive_split_text(section["text"], target_chars)
            for idx, unit in enumerate(split_units, start=1):
                out.append(
                    _build_chunk(
                        source_file=section["source_file"],
                        part=section["part"],
                        item=section["item"],
                        section_title=section.get("title"),
                        strategy="recursive_hierarchical",
                        chunk_index=idx,
                        text=unit,
                    )
                )
    return out


def _sentence_tokens(sentence: str) -> set:
    return {w.lower() for w in WORD_RE.findall(sentence)}


def _jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def semantic_chunk_section(
    section: Dict,
    min_chars: int = DEFAULT_SEMANTIC_MIN,
    max_chars: int = DEFAULT_SEMANTIC_MAX,
    break_threshold: float = 0.18,
) -> List[Dict]:
    sentences = _split_sentences(section["text"])
    if not sentences:
        return []

    chunks = []
    cur_sentences = [sentences[0]]
    cur_tokens = _sentence_tokens(sentences[0])
    cur_len = len(sentences[0])

    for sentence in sentences[1:]:
        sent_tokens = _sentence_tokens(sentence)
        sim = _jaccard_similarity(cur_tokens, sent_tokens)
        proposed_len = cur_len + 1 + len(sentence)

        should_break = False
        if proposed_len > max_chars:
            should_break = True
        elif cur_len >= min_chars and sim < break_threshold:
            should_break = True

        if should_break:
            chunk_text = " ".join(cur_sentences).strip()
            chunks.append(chunk_text)
            cur_sentences = [sentence]
            cur_tokens = sent_tokens
            cur_len = len(sentence)
        else:
            cur_sentences.append(sentence)
            cur_tokens = cur_tokens | sent_tokens
            cur_len = proposed_len

    if cur_sentences:
        chunks.append(" ".join(cur_sentences).strip())

    return [
        _build_chunk(
            source_file=section["source_file"],
            part=section["part"],
            item=section["item"],
            section_title=section.get("title"),
            strategy="semantic",
            chunk_index=i,
            text=chunk_text,
        )
        for i, chunk_text in enumerate(chunks, start=1)
        if chunk_text
    ]


def run_fixed(sections: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for s in sections:
        out.extend(fixed_size_chunk_section(s, chunk_size=chunk_size, overlap=overlap))
    return out


def run_recursive(sections: List[Dict], target_chars: int) -> List[Dict]:
    return recursive_chunk_sections(sections, target_chars=target_chars)


def run_semantic(
    sections: List[Dict], min_chars: int, max_chars: int, break_threshold: float
) -> List[Dict]:
    out = []
    for s in sections:
        out.extend(
            semantic_chunk_section(
                s,
                min_chars=min_chars,
                max_chars=max_chars,
                break_threshold=break_threshold,
            )
        )
    return out


def load_sections_for_file(path: Path) -> List[Dict]:
    soup = clean_soup(load_html(path))
    sections, _ = extract_sections(soup, path.name)
    return sections


def iter_filing_paths(paths: Iterable[str]) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]
    return sorted(RAW_DIR.glob("*.html"))


def chunk_filing_for_strategy(
    html_path: Path,
    strategy: str,
    *,
    recursive_target: int = DEFAULT_RECURSIVE_TARGET,
    semantic_min: int = DEFAULT_SEMANTIC_MIN,
    semantic_max: int = DEFAULT_SEMANTIC_MAX,
    semantic_break_threshold: float = 0.18,
) -> List[Dict]:
    """
    Chunk a single 10-K HTML file with one strategy (for interactive / GUI flows).

    strategy: "semantic" or "recursive" (aliases: recursive_hierarchical).
    """
    path = Path(html_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    key = strategy.strip().lower().replace(" ", "_")
    sections = load_sections_for_file(path)
    if key in ("semantic",):
        return run_semantic(
            sections,
            semantic_min,
            semantic_max,
            semantic_break_threshold,
        )
    if key in ("recursive", "recursive_hierarchical"):
        return run_recursive(sections, recursive_target)
    raise ValueError(
        f"Unknown strategy {strategy!r}. Use 'semantic' or 'recursive'."
    )


def print_chunks(label: str, chunks: List[Dict], limit: Optional[int] = None) -> None:
    print(f"\n===== {label} =====")
    to_show = chunks if limit is None else chunks[:limit]
    for chunk in to_show:
        print(json.dumps(chunk, indent=4))
    print(f"total_chunks={len(chunks)} shown={len(to_show)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed, recursive, and semantic chunking for SEC 10-K sections."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional paths to HTML files. Defaults to data/raw/*.html",
    )
    parser.add_argument("--fixed-size", type=int, default=DEFAULT_FIXED_CHARS)
    parser.add_argument("--fixed-overlap", type=int, default=DEFAULT_FIXED_OVERLAP)
    parser.add_argument("--recursive-target", type=int, default=DEFAULT_RECURSIVE_TARGET)
    parser.add_argument("--semantic-min", type=int, default=DEFAULT_SEMANTIC_MIN)
    parser.add_argument("--semantic-max", type=int, default=DEFAULT_SEMANTIC_MAX)
    parser.add_argument("--semantic-break-threshold", type=float, default=0.18)
    parser.add_argument(
        "--max-output-per-strategy",
        type=int,
        default=None,
        help="If set, prints only first N chunks per strategy (counts remain full).",
    )
    args = parser.parse_args()

    all_sections = []
    file_paths = iter_filing_paths(args.files)
    if not file_paths:
        raise SystemExit(f"No files found under {RAW_DIR}")

    for path in file_paths:
        all_sections.extend(load_sections_for_file(path))

    fixed_chunks = run_fixed(all_sections, args.fixed_size, args.fixed_overlap)
    recursive_chunks = run_recursive(all_sections, args.recursive_target)
    semantic_chunks = run_semantic(
        all_sections,
        min_chars=args.semantic_min,
        max_chars=args.semantic_max,
        break_threshold=args.semantic_break_threshold,
    )

    print_chunks("fixed_size", fixed_chunks, limit=args.max_output_per_strategy)
    print_chunks(
        "recursive_hierarchical",
        recursive_chunks,
        limit=args.max_output_per_strategy,
    )
    print_chunks("semantic", semantic_chunks, limit=args.max_output_per_strategy)
    # --- Save chunks to disk for embedding ---
    semantic_path = CHUNK_DIR / "semantic_chunks.jsonl"
    recursive_path = CHUNK_DIR / "recursive_chunks.jsonl"
    with open(semantic_path, "w", encoding="utf-8") as f:
        for c in semantic_chunks:
            f.write(json.dumps(c) + "\n")

    with open(recursive_path, "w", encoding="utf-8") as f:
        for c in recursive_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Saved {semantic_path} and {recursive_path}")



if __name__ == "__main__":
    main()
