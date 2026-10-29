#!/usr/bin/env python3
"""Explicit local worker using dedicated persistent research volume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.research.api import get_sec_client  # noqa: E402
from backend.app.research.storage import LocalResearchRepository, RawObjectStore  # noqa: E402
from backend.app.research.worker import IngestionWorker, LocalHashEmbedding  # noqa: E402
from backend.app.research.local_embedding import LocalSentenceEmbedding  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one queued attempt and exit")
    args = parser.parse_args()
    db, root = os.getenv("RESEARCH_DB_PATH", ""), os.getenv("RESEARCH_RAW_DIR", "")
    if (not db or not root or not Path(db).is_absolute() or not Path(root).is_absolute()
            or os.getenv("RESEARCH_SINGLE_WORKER") != "true"
            or os.getenv("SEC_SINGLE_INSTANCE_MODE") != "true"):
        print("BLOCKED: configure absolute persistent paths and verified single-worker/SEC limits")
        return 2
    mode = os.getenv("RESEARCH_EMBEDDING_MODE", "")
    if mode == "local-model":
        model_dir = os.getenv("RESEARCH_LOCAL_MODEL_DIR", "")
        if not model_dir or not Path(model_dir).is_absolute():
            print("BLOCKED: set RESEARCH_LOCAL_MODEL_DIR to a preinstalled model directory")
            return 2
        embeddings = LocalSentenceEmbedding(Path(model_dir))
    elif mode == "local-hash" and os.getenv("RESEARCH_ALLOW_LOCAL_HASH") == "true":
        embeddings = LocalHashEmbedding()
    else:
        print("BLOCKED: select explicit embedding mode (local-model or local-hash diagnostic)")
        return 2
    worker = IngestionWorker(LocalResearchRepository(Path(db)), RawObjectStore(Path(root)),
                             get_sec_client(), embeddings)
    while True:
        result = None
        try:
            result = worker.run_once()
            if result:
                print(json.dumps(result), flush=True)
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        if not result:
            time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
