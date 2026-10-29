#!/usr/bin/env python3
"""Opt-in SEC XBRL snapshot for an existing local filing revision."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.research.api import get_sec_client
from backend.app.research.fact_storage import FactSnapshots
from backend.app.research.ingestion_api import get_research_repository
from backend.app.research.models import normalize_cik, validate_accession
from backend.app.research.storage import RawObjectStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cik", required=True)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--allow-sec-fetch", action="store_true")
    args = parser.parse_args()
    root = os.getenv("RESEARCH_RAW_DIR", "")
    if not args.allow_sec_fetch or not root or not Path(root).is_absolute():
        parser.error("Explicit --allow-sec-fetch and absolute RESEARCH_RAW_DIR are required")
    cik, accession = normalize_cik(args.cik), validate_accession(args.accession)
    repo = get_research_repository()
    with repo.connect() as db:
        if not db.execute("SELECT 1 FROM revisions WHERE cik=? AND accession=? AND revision_id=?",
                          (cik, accession, args.revision)).fetchone():
            parser.error("An existing exact filing revision is required before any network request")
    sec = get_sec_client()  # verifies authentic contact and configured rate policy
    raw, observed, _ = sec.fetch_companyfacts_bytes(cik)
    identity = FactSnapshots(repo, RawObjectStore(Path(root))).publish(cik, accession, args.revision, raw, observed)
    print(json.dumps({"snapshot_id": identity, "raw_bytes": len(raw), "observed_at": observed.isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
