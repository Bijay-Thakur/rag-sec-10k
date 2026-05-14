"""
v2 Index builder: VectorStoreIndex over ChromaDB.

Key differences from v1 embed.py:
  - Uses LlamaIndex's VectorStoreIndex (not raw chromadb.Collection.add).
  - The OpenAI embedding call is handled by LlamaIndex's OpenAIEmbedding model,
    with automatic batching and retry logic built in.
  - embed-once: same `collection_is_populated` guard as v1 — skip if already indexed.
  - ChromaDB collection name is prefixed with "v2_" to avoid clobbering v1 data.

Usage (script)
--------------
    python scripts/run_v2_index.py            # build index (skip if populated)
    python scripts/run_v2_index.py --force    # force rebuild
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import chromadb
from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DB_DIR          = _PROJECT_ROOT / "db"
COLLECTION_NAME = "v2_semantic_index"
EMBED_MODEL     = "text-embedding-3-small"


def _chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(DB_DIR))


def index_is_populated(collection_name: str = COLLECTION_NAME) -> bool:
    """Return True if the v2 Chroma collection already has at least one document."""
    try:
        col = _chroma_client().get_collection(collection_name)
        return col.count() > 0
    except Exception:
        return False


def build_v2_index(
    nodes: List[TextNode],
    *,
    collection_name: str = COLLECTION_NAME,
    force: bool = False,
) -> VectorStoreIndex:
    """
    Build (or load) a VectorStoreIndex backed by ChromaDB.

    Parameters
    ----------
    nodes            : TextNode list from v2.ingestion.build_nodes_from_sections.
    collection_name  : ChromaDB collection name (default "v2_semantic_index").
    force            : If True, delete and rebuild even if already indexed.

    Returns
    -------
    VectorStoreIndex ready for querying.
    """
    client = _chroma_client()

    if not force and index_is_populated(collection_name):
        existing = client.get_collection(collection_name).count()
        print(
            f"[v2] Collection '{collection_name}' already has {existing} vectors "
            "— loading existing index (pass --force to rebuild)."
        )
        return _load_existing(client, collection_name)

    # Wipe old data to avoid duplicate-ID errors on partial rebuilds
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    chroma_col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store   = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx    = StorageContext.from_defaults(vector_store=vector_store)
    embed_model    = OpenAIEmbedding(
        model=EMBED_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    print(f"[v2] Indexing {len(nodes)} nodes into '{collection_name}' …")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_ctx,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"[v2] Done — {client.get_collection(collection_name).count()} vectors stored.")
    return index


def load_v2_index(
    collection_name: str = COLLECTION_NAME,
) -> VectorStoreIndex:
    """
    Load an existing v2 VectorStoreIndex from ChromaDB without re-embedding.
    Falls back to v1's 'semantic_index' collection if v2_semantic_index is not
    built yet — this is the zero-cost production path that reuses existing vectors.
    """
    client = _chroma_client()
    if index_is_populated(collection_name):
        return _load_existing(client, collection_name)

    # Fallback: wrap v1's semantic_index
    V1_COLLECTION = "semantic_index"
    if index_is_populated(V1_COLLECTION):
        print(
            f"[v2] '{collection_name}' not found — "
            f"using v1 '{V1_COLLECTION}' collection as backend "
            "(same embeddings, zero API cost)."
        )
        return _load_existing(client, V1_COLLECTION)

    raise RuntimeError(
        f"Neither '{collection_name}' nor 'semantic_index' are populated. "
        "Run `python scripts/run_v2_index.py` first."
    )


def v2_index_ready() -> bool:
    """
    Return True if the v2 index is usable — either v2_semantic_index or
    v1's semantic_index is populated.
    """
    return index_is_populated(COLLECTION_NAME) or index_is_populated("semantic_index")


def _load_existing(
    client: chromadb.ClientAPI,
    collection_name: str,
) -> VectorStoreIndex:
    chroma_col  = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_col)
    storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)
    embed_model  = OpenAIEmbedding(
        model=EMBED_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
