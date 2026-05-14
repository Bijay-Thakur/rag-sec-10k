"""
LlamaIndex custom reader for SEC 10-K HTML filings.

Wraps the v1 html_loader.py to produce LlamaIndex Document objects,
each representing one Part/Item section with structured metadata.

Usage
-----
    from v2.ingestion import SEC10KReader

    reader = SEC10KReader()
    documents = reader.load_data(file=Path("data/raw/Apple.html"))
    # Each document = one Item section with metadata
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

# Allow running from project root without installing the package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ingestion.html_loader import clean_soup, extract_sections, load_html


class SEC10KReader(BaseReader):
    """
    Reads a single SEC 10-K HTML filing and returns one LlamaIndex Document
    per Part/Item section detected by the v1 html_loader.

    Each Document carries rich metadata:
        - source_file  : filename of the original HTML (e.g. "Apple.html")
        - company      : stem of source_file (e.g. "Apple")
        - part         : SEC filing part ("PART I", "PART II", …)
        - item         : SEC item number ("Item 1", "Item 1A", …)
        - section_title: human-readable section title
        - incorporated_by_reference : bool (IBR sections are short stubs)
        - char_count   : length of section text in characters
    """

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        source_file = file.name
        soup = clean_soup(load_html(file))
        sections, stats = extract_sections(soup, source_file)

        documents: List[Document] = []
        for sec in sections:
            text = sec.get("text", "").strip()
            if not text:
                continue

            metadata: Dict[str, Any] = {
                "source_file": source_file,
                "company":     source_file.replace(".html", ""),
                "part":        sec.get("part") or "",
                "item":        sec.get("item") or "",
                "section_title": sec.get("title") or "",
                "incorporated_by_reference": bool(sec.get("incorporated_by_reference")),
                "char_count":  len(text),
            }
            if extra_info:
                metadata.update(extra_info)

            documents.append(
                Document(
                    text=text,
                    metadata=metadata,
                    # LlamaIndex uses doc_id for dedup; make it deterministic
                    doc_id=f"{source_file}::{sec.get('part','')}::{sec.get('item','')}",
                )
            )

        return documents
