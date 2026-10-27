#!/usr/bin/env python3
"""Read-only staging/runtime vector integrity check; exits nonzero if unsafe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_integrity import verify_collection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Path to a staging copy or mounted index")
    parser.add_argument("--collection", default="semantic_index")
    parser.add_argument("--model", default="text-embedding-3-small")
    args = parser.parse_args()
    if not args.db.is_dir():
        print(json.dumps({"status": "invalid", "reason": "Index directory unavailable"}))
        return 2
    try:
        import chromadb

        collection = chromadb.PersistentClient(path=str(args.db)).get_collection(args.collection)
        count = verify_collection(collection, model=args.model)
    except Exception as exc:
        print(json.dumps({"status": "invalid", "reason": str(exc)}))
        return 2
    print(json.dumps({"status": "verified", "collection": args.collection,
                      "model": args.model, "vectors_checked": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
