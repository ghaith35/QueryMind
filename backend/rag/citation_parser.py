"""
Citation extraction and validation.

Flow:
    1. Regex-extract all [CITE: id] markers from LLM answer.
    2. Intersect with retrieved chunk IDs.
    3. Build Citation objects for valid ones; collect hallucinated IDs.
    4. De-duplicate citations (same chunk cited multiple times → one Citation).
"""

import re
from typing import List, Tuple, Dict

from backend.rag.retriever import RetrievedChunk
from backend.schemas.chunk import Citation

# Matches [CITE: id], [CITE id], [Cite: id] with optional whitespace.
# Also accepts alphanumeric IDs for forward-compatibility with test fixtures.
_CITATION_RE = re.compile(
    r"\[\s*CITE\s*[:\s]\s*([A-Za-z0-9_]+)\s*\]",
    re.IGNORECASE,
)


def parse_citations(
    answer_text: str,
    retrieved_chunks: List[RetrievedChunk],
) -> Tuple[str, List[Citation], List[str]]:
    """
    Extract and validate citations from LLM answer.

    Returns:
        answer_text: unchanged (caller decides what to do with hallucinated refs)
        valid_citations: de-duplicated Citation objects in order of first appearance
        hallucinated_ids: IDs cited by LLM that are NOT in the retrieved set
    """
    retrieved_map: Dict[str, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}
    cited_ids = _CITATION_RE.findall(answer_text)

    valid: List[Citation] = []
    hallucinated: List[str] = []
    seen_valid: set = set()

    for cid in cited_ids:
        if cid in retrieved_map:
            if cid not in seen_valid:
                chunk = retrieved_map[cid]
                valid.append(Citation(
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    paragraph_index=chunk.paragraph_index,
                    chunk_id=chunk.chunk_id,
                    relevance_score=chunk.relevance_score,
                    excerpt=chunk.text[:150],
                ))
                seen_valid.add(cid)
        else:
            if cid not in hallucinated:
                hallucinated.append(cid)

    return answer_text, valid, hallucinated


def strip_hallucinated_citations(answer_text: str, hallucinated_ids: List[str]) -> str:
    """Replace hallucinated [CITE: id] markers with [citation omitted]."""
    for bad_id in hallucinated_ids:
        pattern = re.compile(
            r'\[CITE:\s*' + re.escape(bad_id) + r'\s*\]', re.IGNORECASE
        )
        answer_text = pattern.sub("[citation omitted]", answer_text)
    return answer_text


def is_refusal(answer_text: str) -> bool:
    """Detect when the LLM correctly refused to answer."""
    refusal_fragments = [
        "do not contain information",
        "cannot answer",
        "no information",
        "ne contient pas",       # French
        "لا تحتوي",              # Arabic
        "لا يمكنني",             # Arabic
    ]
    lower = answer_text.lower()
    return any(f.lower() in lower for f in refusal_fragments)
