#!/usr/bin/env python3
"""Opt-in download of the fixed public CPU embedding artifact; runtime stays offline."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.research.local_embedding import MODEL_ID, MODEL_REVISION

FILES = ["modules.json", "config.json", "config_sentence_transformers.json", "sentence_bert_config.json",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
         "model.safetensors", "1_Pooling/config.json", "README.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not args.download:
        parser.error("Explicit --download is required; this command contacts Hugging Face")
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False, local_dir=args.destination,
                      allow_patterns=FILES)
    manifest = {name: hashlib.sha256((args.destination / name).read_bytes()).hexdigest() for name in FILES}
    result = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": manifest}
    trusted = json.loads((Path(__file__).resolve().parents[1] / "backend/app/research/embedding_manifest.json").read_text())
    if result != trusted:
        raise ValueError("Downloaded model differs from the reviewed artifact manifest")
    (args.destination / "artifact_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": len(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
