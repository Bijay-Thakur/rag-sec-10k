#!/usr/bin/env python3
"""Real local vectors on explicitly synthetic filings. No SEC or generation calls."""

import argparse
import hashlib
import importlib.metadata
import json
import resource
import statistics
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.research.evaluation import score_cases
from backend.app.research.local_embedding import LocalSentenceEmbedding
from backend.app.research.models import FilingRef, primary_document_url
from backend.app.research.retrieval import retrieve
from backend.app.research.storage import LocalResearchRepository, RawObjectStore
from backend.app.research.worker import IngestionWorker, PIPELINE_VERSION

SECTIONS = [
    ("1", "Business", "Optical instruments are manufactured in research laboratories. Production workers assemble microscope lenses and precision imaging equipment for laboratory customers.", "What products does the company manufacture?"),
    ("1A", "Risk factors", "Dependence on a small number of suppliers creates component shortages. Factory disruptions and concentrated supplier relationships could delay delivery to customers.", "What risk comes from dependence on suppliers?"),
    ("2", "Liquidity", "Liquidity is maintained through operating cash flows and revolving credit facilities. Management uses cash reserves to fund operations and repay borrowings.", "How does management fund operations?")]


class SyntheticSec:
    def fetch_primary(self, filing):
        raw = ("<html><body>" + "".join(f"<h2>Item {item}. {heading}</h2><p>{text}</p>" for item, heading, text, _ in SECTIONS) + "</body></html>").encode()
        return raw, {"fetched_at": datetime.now(timezone.utc).isoformat(), "source_url": filing.primary_document_url}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    model = LocalSentenceEmbedding(args.model_dir)
    model_load = time.monotonic() - started
    cases, timings = [], []
    with tempfile.TemporaryDirectory(prefix="research-model-smoke-") as directory:
        repo = LocalResearchRepository(Path(directory) / "research.sqlite3")
        objects = RawObjectStore(Path(directory) / "objects")
        for number, form in enumerate(("10-K", "10-Q", "10-K/A"), 1):
            cik = str(1234500 + number).zfill(10)
            accession = f"0001111111-26-{number:06d}"
            ref = FilingRef(cik=cik, accession=accession, form=form, filed_at=date(2026, 5, 1),
                            report_period=date(2026, 3, 31), primary_document="synthetic.htm",
                            primary_document_url=primary_document_url(cik, accession, "synthetic.htm"),
                            source_submissions_url=f"https://data.sec.gov/submissions/CIK{cik}.json",
                            metadata_observed_at=datetime.now(timezone.utc))
            repo.enqueue(ref, PIPELINE_VERSION)
            result = IngestionWorker(repo, objects, SyntheticSec(), model).run_once()
            revision = result["revision_id"]
            repo.promote_research(cik, accession, revision)
            # Reopen SQLite to exercise persisted publication, not just in-memory vectors.
            repo = LocalResearchRepository(repo.db_path)
            for item, _, _, query in SECTIONS:
                before = time.monotonic()
                hits = retrieve(repo, cik, accession, revision, query, model, limit=10)
                timings.append((time.monotonic() - before) * 1000)
                with repo.connect() as db:
                    relevant = [row[0] for row in db.execute(
                        "SELECT c.chunk_id FROM chunks c JOIN sections s ON s.revision_id=c.revision_id "
                        "AND s.ordinal=c.section_ordinal WHERE c.revision_id=? AND s.item=?",
                        (revision, f"Item {item}"))]
                for mode in ("lexical", "dense", "rrf"):
                    ranked = hits if mode == "rrf" else sorted(hits, key=lambda hit: -hit[f"{mode}_score"])
                    cases.append({"mode": mode, "cik": cik, "accession": accession, "revision_id": revision,
                                  "slice": form, "relevant_chunk_ids": relevant,
                                  "ranked_chunk_ids": [hit["chunk_id"] for hit in ranked],
                                  "hits": [{"cik": cik, "accession": accession, "revision_id": revision,
                                            "chunk_id": hit["chunk_id"]} for hit in hits],
                                  "cited_chunk_ids": [], "refused": False, "unanswerable": False})
    ordered = sorted(timings)
    report = {"environment": "synthetic_filings_real_local_embeddings", "dataset_version": "research-synthetic-v1",
              "corpus_sha256": hashlib.sha256(json.dumps(SECTIONS).encode()).hexdigest(),
              "observed_at": datetime.now(timezone.utc).isoformat(),
              "forms": ["10-K", "10-Q", "10-K/A"], "filings": 3, "chunks_per_filing": len(SECTIONS),
              "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "sentence-transformers")},
              "model": model.model, "dimensions": model.dimensions, "queries": len(timings),
              "model_load_seconds": model_load, "retrieval_p50_ms": statistics.median(timings),
              "retrieval_p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
              "process_peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "generation_calls": 0, "embedding_api_calls": 0,
              "modes": {mode: score_cases([case for case in cases if case["mode"] == mode]) for mode in ("lexical", "dense", "rrf")},
              "limitations": ["Synthetic source text only; no SEC filings, LLM, real-world relevance or live citation quality verified.",
                              "No generated citation or unanswerable-query sample is scored in this smoke.",
                              "Modes share one retrieval pass; only the full retrieval path is timed. Peak RSS is measured on Linux."]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"queries": report["queries"], "dimensions": report["dimensions"], "environment": report["environment"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
