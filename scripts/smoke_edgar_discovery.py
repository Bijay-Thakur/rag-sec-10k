#!/usr/bin/env python3
"""Opt-in, low-rate live SEC metadata smoke; no document download or embeddings."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.research.sec_client import SecHttpAdapter  # noqa: E402


def main() -> int:
    contact = os.getenv("SEC_USER_AGENT", "")
    if not contact or os.getenv("SEC_SINGLE_INSTANCE_MODE") != "true":
        print("BLOCKED: set an authentic SEC_USER_AGENT and verify single-instance policy before live smoke")
        return 2
    sec = SecHttpAdapter(user_agent=contact)
    report: dict = {"checked_at": datetime.now(timezone.utc).isoformat(), "filers": [],
                    "result": "BLOCKED"}
    # Distinct issuers outside the five-filing legacy registry.
    for ticker in ("MSFT", "NVDA", "COST"):
        search = sec.resolve_issuer(ticker)
        exact = [issuer for issuer in search.candidates if ticker in issuer.tickers]
        if len(exact) != 1:
            report["error"] = f"Ambiguous or absent SEC ticker mapping: {ticker}"
            print(json.dumps(report, indent=2))
            return 1
        issuer = exact[0]
        listing = sec.list_filings(issuer.cik, limit=5)
        if not listing.filings or any(f.cik != issuer.cik or not f.primary_document_url
                                      or not f.primary_document_url.startswith("https://www.sec.gov/Archives/edgar/data/")
                                      for f in listing.filings):
            report["error"] = f"Missing or mismatched official filing metadata: {ticker}"
            print(json.dumps(report, indent=2))
            return 1
        report["filers"].append({"ticker": ticker, "cik": issuer.cik,
                                  "source": listing.source_url, "count": len(listing.filings),
                                  "filings": [{"accession": f.accession, "form": f.form,
                                               "filed_at": f.filed_at.isoformat(),
                                               "url": f.primary_document_url} for f in listing.filings]})
    report["result"] = "PASS"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
