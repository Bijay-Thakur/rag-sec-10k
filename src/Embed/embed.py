import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

GUI_QA_COLLECTION = "gui_qa_index"

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


def build_index(name: str, chunks: List[dict]) -> None:
    """Create or replace a Chroma collection with OpenAI embeddings (CLI / batch files)."""
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


def reset_gui_qa_collection(name: str = GUI_QA_COLLECTION) -> None:
    try:
        chroma.delete_collection(name)
    except Exception:
        pass


def embed_query_text(text: str) -> List[float]:
    """Same model as indexing — for retrieval in the GUI."""
    return embed_text(text)


def query_gui_index(
    question: str,
    *,
    n_results: int = 6,
    collection_name: str = GUI_QA_COLLECTION,
) -> List[Dict[str, Any]]:
    col = chroma.get_collection(collection_name)
    q_emb = embed_query_text(question)
    raw = col.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    hits: List[Dict[str, Any]] = []
    for i, cid in enumerate(ids):
        h: Dict[str, Any] = {
            "chunk_id": cid,
            "document": docs[i] if docs else "",
            "metadata": dict(metas[i]) if metas and metas[i] else {},
        }
        if dists:
            h["distance"] = float(dists[i])
        hits.append(h)
    return hits


def build_gui_qa_index(
    chunks: List[dict],
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Replace gui_qa_index with embedded chunks (Streamlit).

    progress_callback(done_chunks, total_chunks) after each embedding batch.
    """
    reset_gui_qa_collection()
    if not chunks:
        return 0

    collection = chroma.get_or_create_collection(
        name=GUI_QA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    for c in chunks:
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metas.append(normalize_metadata(c["metadata"]))

    n = len(docs)
    all_embeds: List[List[float]] = []
    for start in range(0, n, EMBED_BATCH_SIZE):
        batch_docs = docs[start : start + EMBED_BATCH_SIZE]
        batch_emb = embed_texts_batch(batch_docs)
        all_embeds.extend(batch_emb)
        if progress_callback:
            progress_callback(min(start + len(batch_docs), n), n)

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=all_embeds,
    )
    return n


def main() -> None:
    for name in ("semantic_index", "recursive_index"):
        try:
            chroma.delete_collection(name)
        except Exception:
            pass

    semantic_chunks = load_chunks(CHUNK_DIR / "semantic_chunks.jsonl")
    recursive_chunks = load_chunks(CHUNK_DIR / "recursive_chunks.jsonl")

    build_index("semantic_index", semantic_chunks)
    build_index("recursive_index", recursive_chunks)


if __name__ == "__main__":
    main()
