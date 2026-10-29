"""Offline-only dense model adapter; never downloads a model at runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import DiscoveryError

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


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
        try:
            manifest = json.loads((path / "artifact_manifest.json").read_text())
            trusted = json.loads(Path(__file__).with_name("embedding_manifest.json").read_text())
            if manifest != trusted or manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
                raise ValueError("Model revision differs")
            names = manifest["files"]
            required = {"model.safetensors", "modules.json", "tokenizer.json", "1_Pooling/config.json", "config.json"}
            if not required.issubset(names) or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise ValueError("Model manifest incomplete")
            files = [path / name for name in sorted(names)]
            visible = {p.relative_to(path).as_posix() for p in path.rglob("*")
                       if p.is_file() and ".cache" not in p.relative_to(path).parts}
            if visible != set(names) | {"artifact_manifest.json"}:
                raise ValueError("Unexpected model artifacts")
            if any(file.is_symlink() or not file.resolve().is_relative_to(path) for file in files):
                raise ValueError("Model artifacts cannot use symlinks")
            if any(hashlib.sha256(file.read_bytes()).hexdigest() != names[str(file.relative_to(path))] for file in files):
                raise ValueError("Model checksum differs")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DiscoveryError("embedding_unconfigured", "Pinned model artifact failed verification", 503) from exc
        if not files or any(p.is_symlink() for p in files):
            raise DiscoveryError("embedding_unconfigured", "Local model artifact invalid", 503)
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.relative_to(path).as_posix().encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        self.artifact_sha256 = digest.hexdigest()
        self.model = f"{MODEL_ID}@{MODEL_REVISION}@sha256:{self.artifact_sha256}"
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(str(path), device="cpu", trust_remote_code=False, local_files_only=True)
        self.dimensions = self.encoder.get_sentence_embedding_dimension()
        if self.dimensions != 384:
            raise DiscoveryError("embedding_unconfigured", "Model dimensions unavailable", 503)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or len(texts) > 5000:
            raise DiscoveryError("embedding_limit", "Embedding batch outside budget", 422)
        return self.encoder.encode(texts, normalize_embeddings=True, batch_size=32,
                                   show_progress_bar=False).tolist()
