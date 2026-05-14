"""
Build the v2 LlamaIndex VectorStoreIndex.

Steps
-----
1. Load Apple.html with SEC10KReader (same html_loader.py logic as v1).
2. Parse sections into TextNodes with SentenceSplitter (512 tokens, 50 overlap).
3. Embed with OpenAI text-embedding-3-small and store in ChromaDB collection
   "v2_semantic_index".
4. Embed-once: skip if already populated (use --force to rebuild).

Usage
-----
    $env:PYTHONPATH = "."
    python scripts/run_v2_index.py
    python scripts/run_v2_index.py --force
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from v2.ingestion import SEC10KReader, build_nodes_from_sections
from v2.indexing  import build_v2_index, index_is_populated


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 LlamaIndex index.")
    parser.add_argument("--force", action="store_true",
                        help="Delete and rebuild even if already indexed.")
    parser.add_argument(
        "--html",
        default=str(ROOT / "data" / "raw" / "Apple.html"),
        help="Path to the 10-K HTML filing (default: data/raw/Apple.html).",
    )
    args = parser.parse_args()

    html_path = Path(args.html)
    if not html_path.exists():
        print(f"ERROR: HTML file not found at {html_path}")
        sys.exit(1)

    # --- Step 1: Load sections ---
    print(f"[v2] Loading {html_path.name} …")
    reader    = SEC10KReader()
    documents = reader.load_data(file=html_path)
    print(f"[v2] {len(documents)} sections extracted.")

    # --- Step 2: Parse into nodes ---
    print("[v2] Splitting into nodes (SentenceSplitter 512 tok / 50 overlap) …")
    nodes = build_nodes_from_sections(documents)
    print(f"[v2] {len(nodes)} nodes created.")

    # --- Step 3: Build / skip index ---
    build_v2_index(nodes, force=args.force)


if __name__ == "__main__":
    main()
