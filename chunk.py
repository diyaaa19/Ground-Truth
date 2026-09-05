"""Sentence-aware chunking with overlap, keeping page provenance for citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .extract import Page

_SENT = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc} · p.{self.page}"


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_pages(
    pages: Iterable[Page],
    doc_name: str,
    target_chars: int = 1100,
    overlap_chars: int = 180,
    start_id: int = 0,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    next_id = start_id

    for page in pages:
        sentences = _split_sentences(page.text)
        buf = ""
        for sentence in sentences:
            if len(sentence) > target_chars * 2:
                # very long unbroken block (tables, code): hard-wrap it
                for i in range(0, len(sentence), target_chars):
                    piece = sentence[i : i + target_chars]
                    chunks.append(Chunk(next_id, doc_name, page.number, piece))
                    next_id += 1
                continue
            if len(buf) + len(sentence) + 1 <= target_chars:
                buf = f"{buf} {sentence}".strip()
            else:
                if buf:
                    chunks.append(Chunk(next_id, doc_name, page.number, buf))
                    next_id += 1
                    tail = buf[-overlap_chars:] if overlap_chars else ""
                    buf = f"{tail} {sentence}".strip()
                else:
                    buf = sentence
        if buf.strip():
            chunks.append(Chunk(next_id, doc_name, page.number, buf.strip()))
            next_id += 1

    return chunks
