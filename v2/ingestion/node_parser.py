"""
Node parser for v2: converts LlamaIndex Documents into TextNodes.

Uses LlamaIndex's SentenceSplitter (sentence-boundary-aware recursive splitting)
instead of the manual chunkers.py strategies.  This maps most closely to v1's
"recursive_hierarchical" strategy, but is driven by the LlamaIndex framework.

Metadata from the source Document is automatically propagated to every child node.
A stable `node_id` (chunk_id) is assigned as:
    <source_file>::<part>::<item>::<chunk_index_within_section>

so ids are deterministic and compatible with the gold eval set if regenerated.

Usage
-----
    from v2.ingestion import build_nodes_from_sections

    nodes = build_nodes_from_sections(documents)
    print(f"{len(nodes)} nodes created")
"""
from __future__ import annotations

from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

# Matches v1 semantic/recursive chunk target (~700 tokens ≈ 2800 chars at ~4 chars/token)
CHUNK_SIZE    = 512   # tokens (LlamaIndex counts by tokens via tiktoken)
CHUNK_OVERLAP = 50    # token overlap between consecutive chunks


def build_nodes_from_sections(
    documents: List[Document],
    chunk_size: int   = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[TextNode]:
    """
    Split each Document (one section) into TextNodes.

    LlamaIndex's SentenceSplitter:
    - Tries to keep complete sentences together.
    - Falls back to word boundaries, then character boundaries.
    - Propagates all Document metadata to every child node.

    Returns a flat list of TextNode objects ready for indexing.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # LlamaIndex uses tiktoken by default for token counting.
        # tokenizer=None uses cl100k_base which is what text-embedding-3-small uses.
    )

    all_nodes: List[TextNode] = []
    for doc in documents:
        nodes = splitter.get_nodes_from_documents([doc])
        # Assign stable, descriptive IDs
        src   = doc.metadata.get("source_file", "unknown")
        part  = doc.metadata.get("part", "")
        item  = doc.metadata.get("item", "")
        for idx, node in enumerate(nodes):
            node.node_id = f"{src}::{part}::{item}::{idx}"
            # LlamaIndex propagates metadata automatically, but ensure it
            # also carries char_count of the chunk (not the parent section)
            node.metadata["chunk_char_count"] = len(node.text)
            node.metadata["chunk_index"] = idx
        all_nodes.extend(nodes)

    return all_nodes
