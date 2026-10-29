#!/usr/bin/env python3
"""Local backup/restore drill. Destination must be a new directory."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.research.backup import restore, snapshot
from backend.app.research.storage import LocalResearchRepository, RawObjectStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["backup", "restore"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--objects", type=Path, help="Required for backup; immutable raw-object directory")
    args = parser.parse_args()
    if args.operation == "backup":
        if args.objects is None or not args.source.is_file():
            parser.error("An existing source SQLite DB and --objects directory are required")
        result = snapshot(LocalResearchRepository(args.source), RawObjectStore(args.objects), args.destination)
        print(json.dumps({"format": result["format"], "objects": len(result["objects"])}))
    else:
        restore(args.source, args.destination)
        print(json.dumps({"restored": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
