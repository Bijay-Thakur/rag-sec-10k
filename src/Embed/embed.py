import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths (define before load_dotenv so we can load repo-root .env)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"
CHROMA_DB_DIR = PROJECT_ROOT / "db"

load_dotenv(PROJECT_ROOT / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"
# OpenAI allows many strings per request; smaller batches = smoother progress + fewer timeouts
EMBED_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# OpenAI + Chroma
# ---------------------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def load_chunks(path: Path) -> List[dict]:
    """Load JSONL chunks from disk."""
    chunks: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma rejects None in metadata; coerce unsupported values."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def embed_text(text: str) -> List[float]:
    """Embed a single text chunk using OpenAI."""
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty text")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return list(response.data[0].embedding)


def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Embed many texts in one API call (order preserved)."""
    if not texts:
        return []
    cleaned = [t.strip() or " " for t in texts]  # API rejects empty; use space fallback
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=cleaned)
    data = list(response.data)
    if data and hasattr(data[0], "index") and getattr(data[0], "index", None) is not None:
        data.sort(key=lambda d: d.index)
    return [list(d.embedding) for d in data]


def collection_is_populated(name: str) -> bool:
    """Return True if a Chroma collection named `name` already exists and contains at least one document."""
    try:
        col = chroma.get_collection(name=name)
        return col.count() > 0
    except Exception:
        return False


def build_index(name: str, chunks: List[dict], *, force: bool = False) -> None:
    """
    Build a Chroma collection from `chunks`.

    By default (force=False) the function is a no-op when the collection already
    exists and is non-empty, so re-running the script never re-embeds unless you
    pass force=True (or use the --force CLI flag).
    """
    if not force and collection_is_populated(name):
        existing = chroma.get_collection(name=name).count()
        print(f"Collection '{name}' already has {existing} vectors — skipping embedding. "
              "Pass --force to rebuild.")
        return

    # Drop stale data so collection.add() never hits duplicate-ID errors.
    try:
        chroma.delete_collection(name)
    except Exception:
        pass

    collection = chroma.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    embeds: List[List[float]] = []

    for c in chunks:
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metas.append(normalize_metadata(c["metadata"]))

    for i in range(0, len(docs), EMBED_BATCH_SIZE):
        batch_docs = docs[i : i + EMBED_BATCH_SIZE]
        batch_emb = embed_texts_batch(batch_docs)
        embeds.extend(batch_emb)

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeds,
    )
    print(f"Indexed {len(ids)} chunks into collection '{name}'")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Chroma indexes from pre-computed chunk JSONL files."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild indexes even if they already exist. "
             "Without this flag the script is a no-op when data is already stored.",
    )
    args = parser.parse_args()

    semantic_chunks = load_chunks(CHUNK_DIR / "semantic_chunks.jsonl")
    recursive_chunks = load_chunks(CHUNK_DIR / "recursive_chunks.jsonl")

    build_index("semantic_index", semantic_chunks, force=args.force)
    build_index("recursive_index", recursive_chunks, force=args.force)


if __name__ == "__main__":
    main()
