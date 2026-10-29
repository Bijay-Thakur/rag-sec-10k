"""Offline-only dense model adapter; never downloads a model at runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import DiscoveryError


class LocalSentenceEmbedding:
    provider = "local_sentence_transformer"
    index_kind = "real_local_embeddings"

    def __init__(self, model_dir: Path):
        try:
            path = model_dir.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DiscoveryError("embedding_unconfigured", "Local model directory unavailable", 503) from exc
        if not path.is_dir() or model_dir.is_symlink():
            raise DiscoveryError("embedding_unconfigured", "Local model directory required", 503)
        files = sorted(p for p in path.rglob("*") if p.is_file())
        if not files or any(p.is_symlink() for p in files):
            raise DiscoveryError("embedding_unconfigured", "Local model artifact invalid", 503)
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.relative_to(path).as_posix().encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        self.artifact_sha256 = digest.hexdigest()
        self.model = f"{path.name}@sha256:{self.artifact_sha256}"
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(str(path), trust_remote_code=False)
        self.dimensions = self.encoder.get_sentence_embedding_dimension()
        if not self.dimensions:
            raise DiscoveryError("embedding_unconfigured", "Model dimensions unavailable", 503)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or len(texts) > 5000:
            raise DiscoveryError("embedding_limit", "Embedding batch outside budget", 422)
        return self.encoder.encode(texts, normalize_embeddings=True, batch_size=32,
                                   show_progress_bar=False).tolist()
