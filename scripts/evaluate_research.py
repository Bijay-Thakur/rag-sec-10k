#!/usr/bin/env python3
"""Score a local accession-labeled prediction file; never invokes SEC or an LLM."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.research.evaluation import score_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.predictions.stat().st_size > 10 * 1024 * 1024:
        parser.error("Prediction file exceeds 10 MiB")
    corpus = json.loads(args.predictions.read_text())
    if not corpus.get("dataset_version") or corpus.get("environment") not in {"synthetic", "staging", "live"}:
        parser.error("Explicit dataset_version and environment labels are required")
    report = {"dataset_version": corpus["dataset_version"], "environment": corpus["environment"],
              "metrics": score_cases(corpus["cases"]), "slices": {}}
    for name in sorted({case.get("slice", "unspecified") for case in corpus["cases"]}):
        report["slices"][name] = score_cases([case for case in corpus["cases"]
                                            if case.get("slice", "unspecified") == name])
    # Output contains metrics and version only, never questions or evidence text.
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
