"""
PDF text extraction via PyMuPDF.

Returns per-page data with paragraph blocks for accurate page/paragraph indexing.
Arabic RTL: PyMuPDF >= 1.23 handles shaping correctly in "text" mode.
Fallback to pdfplumber not implemented here — if < 200 chars extracted, caller
should raise INVALID_PDF.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF


def file_hash(path: str | Path) -> str:
    """SHA256 of file bytes, first 16 hex chars used as document_id."""
    with open(path, "rb") as f:
        return file_hash_bytes_iter(iter(lambda: f.read(65536), b""))


def file_hash_bytes(contents: bytes) -> str:
    """SHA256 of in-memory PDF bytes, first 16 hex chars used as document_id."""
    return file_hash_bytes_iter([contents])


def file_hash_bytes_iter(blocks) -> str:
    """SHA256 helper shared by file- and bytes-based hashing."""
    h = hashlib.sha256()
    for block in blocks:
        h.update(block)
    return h.hexdigest()[:16]


def extract_pdf(path: str | Path) -> List[Dict[str, Any]]:
    """
    Extract all pages from a PDF.

    Returns:
        List of page dicts:
            page_number: int (1-indexed)
            text: str  (full page text, reading order)
            blocks: list of {text, bbox}  (paragraph-level blocks)
            char_count: int
    """
    path = str(path)
    doc = fitz.open(path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        raw_blocks = page.get_text("blocks")

        # block type 0 = text, type 1 = image
        text_blocks = [
            {"text": b[4], "bbox": list(b[:4])}
            for b in raw_blocks
            if b[6] == 0 and b[4].strip()
        ]

        pages.append({
            "page_number": page_num,
            "text": text,
            "blocks": text_blocks,
            "char_count": len(text.strip()),
        })

    doc.close()
    return pages


def validate_extraction(pages: List[Dict]) -> tuple[bool, str]:
    """
    Returns (ok, reason). Callers should raise INVALID_PDF if ok=False.
    """
    total_chars = sum(p["char_count"] for p in pages)
    if total_chars < 200:
        return False, "This PDF appears to be scanned. OCR is not supported in this version."

    # Check for replacement char contamination (encoding failure)
    replacement_chars = sum(p["text"].count("�") for p in pages)
    if replacement_chars > len(pages) * 5:
        return False, f"{replacement_chars} replacement chars — encoding failure"

    full_text = "".join(page["text"] for page in pages).strip()
    if is_garbled(full_text):
        return False, "This PDF appears to be scanned or garbled. OCR is not supported in this version."

    return True, "ok"


def is_garbled(text: str) -> bool:
    if len(text) < 50:
        return True
    printable_ratio = sum(1 for char in text if char.isprintable()) / max(len(text), 1)
    return printable_ratio < 0.85
