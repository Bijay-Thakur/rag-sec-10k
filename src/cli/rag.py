"""
Chunk + embed one HTML filing, then query top-k similar chunks.

Run with the project ``src`` directory on ``PYTHONPATH`` (repo root):

  PowerShell:
    $env:PYTHONPATH = "src"
    python -m cli.rag ingest --html Apple --strategy semantic
    python -m cli.rag query "What are the main risk factors?" -k 5

  Or pass a full path to --html:

    python -m cli.rag ingest --html "C:/path/to/Filing.html" --strategy recursive

Requires OPENAI_API_KEY (e.g. in .env at repo root).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_COLLECTION = "cli_active_index"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _ensure_paths() -> None:
    ing = PROJECT_ROOT / "src" / "ingestion"
    src = PROJECT_ROOT / "src"
    for p in (str(ing), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_html_path(html_arg: str) -> Path:
    p = Path(html_arg)
    if p.is_file():
        return p.resolve()
    cand = RAW_DIR / f"{html_arg}.html"
    if cand.is_file():
        return cand.resolve()
    raise SystemExit(f"HTML not found: {html_arg!r} (tried path and {cand})")


def cmd_ingest(args: argparse.Namespace) -> int:
    _ensure_paths()
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in the environment or .env", file=sys.stderr)
        return 1

    from chunkers import chunk_filing_for_strategy
    from Embed.embed import build_index, chroma

    html_path = _resolve_html_path(args.html)
    print(f"Chunking {html_path.name} ({args.strategy})…")
    chunks = chunk_filing_for_strategy(html_path, args.strategy)
    print(f"  {len(chunks)} chunks")

    name = args.collection
    try:
        chroma.delete_collection(name)
    except Exception:
        pass

    print(f"Embedding into Chroma collection {name!r}…")
    build_index(name, chunks)
    print("Done. Run: python -m cli.rag query \"…\" -k 5")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    _ensure_paths()
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in the environment or .env", file=sys.stderr)
        return 1

    from retrieval.retrieve import retrieve

    hits = retrieve(
        args.query_text,
        collection_name=args.collection,
        n_results=args.k,
    )
    if not hits:
        print("No hits (empty collection or wrong name?). Ingest first.", file=sys.stderr)
        return 1

    for i, h in enumerate(hits, 1):
        meta = h.get("metadata") or {}
        dist = h.get("distance")
        line = f"{i}. {h.get('chunk_id')} | {meta.get('source_file', '')} | {meta.get('item', '')}"
        if dist is not None:
            line += f" | distance={dist:.4f}"
        print(line)
        text = (h.get("document") or "").strip()
        preview = text[:500] + ("…" if len(text) > 500 else "")
        print(preview)
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cli.rag",
        description="Ingest one SEC HTML filing into Chroma, then run top-k retrieval.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_in = sub.add_parser("ingest", help="Chunk + embed a single HTML 10-K into Chroma")
    p_in.add_argument(
        "--html",
        required=True,
        help="Path to a .html file, or a stem under data/raw/ (e.g. Apple loads data/raw/Apple.html)",
    )
    p_in.add_argument(
        "--strategy",
        choices=("semantic", "recursive"),
        default="semantic",
        help="Chunking strategy (default: semantic)",
    )
    p_in.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION})",
    )
    p_in.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="Retrieve top-k chunks similar to a query")
    p_q.add_argument("query_text", help="Natural language query")
    p_q.add_argument(
        "-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of chunks to return (default: 5)",
    )
    p_q.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection to search (default: {DEFAULT_COLLECTION})",
    )
    p_q.set_defaults(func=cmd_query)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
