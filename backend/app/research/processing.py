"""Versioned SEC HTML extraction retaining repeated Item occurrences."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .models import DiscoveryError, FilingRef

PARSER_VERSION = "sec-html-sections-v1"
CHUNKER_VERSION = "section-window-1200-v1"
ITEM = re.compile(r"^\s*item\s+(\d{1,2}[a-z]?)\b[\s.\-:—–]*(.*)$", re.I)
PART = re.compile(r"^\s*part\s+(I{1,3}V?|IV)\s*\.?\s*$", re.I)


@dataclass(frozen=True)
class ParsedSection:
    ordinal: int
    part: str | None
    item: str
    heading: str
    text: str


@dataclass(frozen=True)
class ParseResult:
    sections: list[ParsedSection]
    qa: dict


def parse_filing(raw: bytes) -> ParseResult:
    if len(raw) > 20 * 1024 * 1024:
        raise DiscoveryError("filing_too_large", "Filing exceeds parser byte limit", 413)
    soup = BeautifulSoup(raw, "lxml")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    for element in soup.find_all(lambda tag: tag.name and tag.name.startswith("ix:")):
        element.unwrap()

    part: str | None = None
    item: str | None = None
    heading = ""
    text: list[str] = []
    sections: list[ParsedSection] = []

    def flush():
        if item and text:
            sections.append(ParsedSection(len(sections), part, item, heading, "\n".join(text)))

    for node in soup.find_all(string=True):
        value = " ".join(str(node).replace("\xa0", " ").split())
        if not value or node.find_parent("a"):
            continue
        if len(value) < 90:
            part_match = PART.fullmatch(value)
            if part_match:
                flush()
                item, text = None, []
                part = f"PART {part_match.group(1).upper()}"
                continue
            match = ITEM.match(value)
            if match:
                flush()
                item = f"Item {match.group(1).upper()}"
                heading, text = match.group(2).strip(), []
                continue
        if item:
            text.append(value)
    flush()
    characters = sum(len(section.text) for section in sections)
    qa = {"parser_version": PARSER_VERSION, "raw_bytes": len(raw),
          "extracted_characters": characters, "section_count": len(sections),
          "paragraph_count": len(soup.find_all("p")), "table_count": len(soup.find_all("table")),
          "duplicate_items_retained": len(sections) - len({(s.part, s.item) for s in sections}),
          "warnings": ["Tables flattened into text; cell/footnote coordinates are not verified"]
                      if soup.find("table") else []}
    if not sections or characters < 50:
        raise DiscoveryError("parser_quality", "Filing has insufficient attributable Item text", 422)
    return ParseResult(sections, qa)


def chunk_sections(filing: FilingRef, parsed: ParseResult, raw_sha: str) -> list[dict]:
    chunks: list[dict] = []
    for section in parsed.sections:
        words = section.text.split()
        current: list[str] = []
        current_length = 0
        windows: list[str] = []
        for word in words:
            if current and current_length + len(current) + len(word) > 1200:
                windows.append(" ".join(current))
                current = []
                current_length = 0
            current.append(word)
            current_length += len(word)
        if current:
            windows.append(" ".join(current))
        for position, body in enumerate(windows):
            identity = (f"{filing.cik}|{filing.accession}|{filing.primary_document}|"
                        f"{PARSER_VERSION}|{CHUNKER_VERSION}|{section.ordinal}|{position}")
            chunks.append({"chunk_id": hashlib.sha256(identity.encode()).hexdigest(),
                           "section_ordinal": section.ordinal, "position": position,
                           "text": body, "sha256": hashlib.sha256(body.encode()).hexdigest(),
                           "token_count": len(body.split()), "raw_sha256": raw_sha,
                           "source_url": filing.primary_document_url})
    if not chunks or len(chunks) > 5000:
        raise DiscoveryError("chunk_limit", "Invalid chunk count", 422)
    return chunks
