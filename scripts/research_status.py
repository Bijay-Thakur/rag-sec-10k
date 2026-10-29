#!/usr/bin/env python3
"""Inspect local jobs, query validation failures and reserved call counts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.research.observability import operational_status
from backend.app.research.storage import LocalResearchRepository


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("An existing research database is required")
    print(json.dumps(operational_status(LocalResearchRepository(args.database)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
