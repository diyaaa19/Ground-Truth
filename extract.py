"""PDF text extraction that works for all kinds of PDFs.

Strategy (first one that yields text wins, per page):
1. pdfplumber  - best layout fidelity, also pulls tables
2. pypdf       - fast fallback for odd/compressed PDFs
3. OCR         - optional, for scanned/image-only pages
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Page:
    number: int
    text: str


_WS = re.compile(r"[ \t\x0b\f\r]+")
_NL = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    # de-hyphenate words broken across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


def _tables_to_text(tables) -> str:
    out = []
    for table in tables or []:
        rows = [
            " | ".join((cell or "").strip() for cell in row)
            for row in table
            if row and any(cell for cell in row)
        ]
        if rows:
            out.append("\n".join(rows))
    return "\n\n".join(out)


def _plumber_pages(path: str) -> List[Page]:
    import pdfplumber

    pages: List[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                body = page.extract_text() or ""
            except Exception:
                body = ""
            try:
                tables = _tables_to_text(page.extract_tables())
            except Exception:
                tables = ""
            merged = "\n\n".join(part for part in (body, tables) if part)
            pages.append(Page(i, clean(merged)))
    return pages


def _pypdf_page(path: str, index: int) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        return clean(reader.pages[index].extract_text() or "")
    except Exception:
        return ""


def _ocr_page(path: str, page_number: int, dpi: int = 200) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(
            path, dpi=dpi, first_page=page_number, last_page=page_number
        )
        if not images:
            return ""
        return clean(pytesseract.image_to_string(images[0]))
    except Exception:
        return ""


def extract_pages(path: str, use_ocr: bool = True, min_chars: int = 25) -> List[Page]:
    """Return per-page text for any PDF. Empty pages are kept out of the result."""
    try:
        pages = _plumber_pages(path)
    except Exception:
        pages = []

    if not pages:
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            pages = [
                Page(i + 1, clean(p.extract_text() or ""))
                for i, p in enumerate(reader.pages)
            ]
        except Exception:
            pages = []

    filled: List[Page] = []
    for page in pages:
        text = page.text
        if len(text) < min_chars:
            text = _pypdf_page(path, page.number - 1) or text
        if use_ocr and len(text) < min_chars:
            text = _ocr_page(path, page.number) or text
        if text.strip():
            filled.append(Page(page.number, text))
    return filled
