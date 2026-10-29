# ADR 004 — pinned local CPU embeddings

2026-10-27. Local development decision; production release remains gated.

Choose the public [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) artifact, Apache 2.0 as declared by its model card, pinned to repository revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Download only safetensors, tokenizer, standard model/pooling configuration and model card. No remote model code is permitted. The preparation command writes checksums for each file; the loader verifies them and the approved model/revision before loading. Runtime uses CPU, offline-only loading and normalized 384-dimensional vectors. Its model identity includes the revision and aggregate artifact checksum.

The tested optional environment is PyTorch `2.6.0+cpu`, sentence-transformers `3.4.1`, transformers `4.46.3`. `backend/requirements-research-local.txt` records the compatible versions. No embedding API or Groq credential is used. Model artifacts remain in ignored `.research/` and are excluded from Docker and Cloud Build contexts. Production must mount the audited artifact read-only; an absent or changed artifact fails closed.

`scripts/smoke_local_research.py` ran with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, persisted vectors and reopened SQLite across three synthetic filings (10-K, 10-Q, 10-K/A), then tested nine queries. See `docs/phase-reports/local-model-smoke.json`. These easy synthetic cases establish adapter integration, not real SEC relevance, semantic faithfulness or production latency. Reranking remains deferred until a held-out real benchmark demonstrates a measured gain.
