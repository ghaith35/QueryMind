"""
Multilingual text chunker.

Parameters (locked per Section 3):
    target:  900 chars
    max:    1200 chars
    overlap: 150 chars

Strategy:
    1. Split page text on blank-line paragraph boundaries.
    2. Accumulate paragraphs into a buffer up to 1200 chars.
    3. When a paragraph would overflow, flush buffer as a chunk.
    4. If a single paragraph > 1200 chars, sentence-split it.
    5. After all chunks per page, apply 150-char overlap prefix from previous chunk.
"""

import re
import unicodedata
from datetime import datetime, timezone
from typing import List, Optional

from backend.schemas.chunk import Chunk

TARGET = 900
MAX_CHARS = 1200
OVERLAP = 150

# Sentence boundary: after . ! ? followed by whitespace + uppercase (Latin or Arabic)
_SENT_END = re.compile(r'(?<=[.!?؟])\s+(?=[A-ZÀ-Öa-z؀-ۿ])')


def detect_language(text: str) -> str:
    """Heuristic language detection — avoids adding langdetect dependency."""
    arabic = sum(
        1 for c in text
        if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ"
    )
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "en"
    arabic_ratio = arabic / total_alpha
    if arabic_ratio > 0.4:
        return "ar"

    accented = sum(
        1 for c in text
        if c.isalpha() and ord(c) > 127 and "LATIN" in unicodedata.name(c, "")
    )
    # Ratio-based: French text reliably has >= 2% accented Latin chars
    if total_alpha > 0 and (accented / total_alpha) > 0.02:
        return "fr"
    # Absolute fallback for short strings: at least 2 accented chars
    if accented >= 2:
        return "fr"
    return "en"


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences; keeps each under MAX_CHARS."""
    parts = _SENT_END.split(text)
    chunks: List[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) + 1 <= MAX_CHARS:
            buf = (buf + " " + part).strip()
        else:
            if buf:
                chunks.append(buf)
            # If a single sentence > MAX_CHARS, hard-split it
            if len(part) > MAX_CHARS:
                for i in range(0, len(part), MAX_CHARS - OVERLAP):
                    chunks.append(part[i : i + MAX_CHARS])
            else:
                buf = part
    if buf:
        chunks.append(buf)
    return chunks


def _apply_overlap(chunks: List[str]) -> List[str]:
    """Prepend last OVERLAP chars of chunk[i-1] to chunk[i]."""
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prefix = chunks[i - 1][-OVERLAP:].strip()
        result.append(prefix + " " + chunks[i] if prefix else chunks[i])
    return result


def chunk_page(
    page_text: str,
    page_number: int,
    document_id: str,
    document_name: str,
    document_set_id: str,
    start_index: int,
    timestamp: Optional[datetime] = None,
) -> List[Chunk]:
    """
    Chunk a single page's text. Returns Chunk objects (no embeddings yet).
    start_index: global chunk counter offset for this document.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    raw_texts: List[str] = []

    buffer = ""
    para_idx = 0
    current_para_start = 0

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= MAX_CHARS:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                raw_texts.append(buffer)
            if len(para) > MAX_CHARS:
                raw_texts.extend(_sentence_split(para))
                buffer = ""
            else:
                buffer = para
        para_idx += 1

    if buffer:
        raw_texts.append(buffer)

    # Apply overlap between adjacent text segments
    overlapped = _apply_overlap(raw_texts)

    chunks: List[Chunk] = []
    char_cursor = 0

    for i, text in enumerate(overlapped):
        text = text[:MAX_CHARS]  # hard ceiling
        lang = detect_language(text)
        word_count = len(text.split())

        chunk_idx = start_index + i
        chunk_id = f"{document_id}_{chunk_idx:04d}"

        chunk = Chunk(
            chunk_id=chunk_id,
            document_name=document_name,
            document_id=document_id,
            document_set_id=document_set_id,
            page_number=page_number,
            paragraph_index=i,
            text=text,
            entities=[],  # filled by entity_extractor
            char_start=char_cursor,
            char_end=char_cursor + len(text),
            word_count=word_count,
            chunk_index_in_document=chunk_idx,
            language=lang,
            timestamp_indexed=timestamp,
        )
        chunks.append(chunk)
        char_cursor += len(text)

    return chunks


def chunk_document(
    pages: list,
    document_id: str,
    document_name: str,
    document_set_id: str,
    timestamp: Optional[datetime] = None,
) -> List[Chunk]:
    """Chunk all pages of a document. pages is the output of extractor.extract_pdf()."""
    all_chunks: List[Chunk] = []
    for page in pages:
        page_chunks = chunk_page(
            page_text=page["text"],
            page_number=page["page_number"],
            document_id=document_id,
            document_name=document_name,
            document_set_id=document_set_id,
            start_index=len(all_chunks),
            timestamp=timestamp,
        )
        all_chunks.extend(page_chunks)
    return all_chunks
