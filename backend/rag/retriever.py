"""
Vector retrieval: embed query → ChromaDB cosine search → SQLite hydration.

Returns RetrievedChunk objects (Chunk + relevance_score) ordered by score desc.

Retrieval pipeline (in order):
  1. HyDE: generate a hypothetical expert answer, embed it — searches in "answer
     space" rather than "question space", dramatically improving precision.
  2. Vector search on original query + HyDE text — two independent semantic signals.
  3. Lexical search — catches exact terms / proper nouns the vector model may miss.
  4. Overview sampling — representative chunks from across the document set for
     summary/overview questions.
  5. RRF merge — Reciprocal Rank Fusion combines all rankers mathematically.
  6. Contextual expansion — add ±1 adjacent chunks so the LLM gets full passage
     context, not isolated 900-char windows.
"""

import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

from backend.indexing.embedder import embed_query
from backend.indexing.entity_extractor import stored_entity_labels
from backend.indexing.pipeline import _get_collection
from backend.db.connection import get_db
from backend.schemas.chunk import Chunk

log = logging.getLogger(__name__)

DEFAULT_K = 10
OVERVIEW_K = 14
VECTOR_FETCH_MULTIPLIER = 3
LEXICAL_LIMIT = 6
OVERVIEW_LIMIT = 8
RRF_K = 60          # standard constant from RRF literature
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.25"))
USE_HYDE = os.getenv("USE_HYDE", "true").lower() in ("1", "true", "yes")
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "1"))  # ±N neighbor chunks to expand

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[؀-ۿ]+")
_OVERVIEW_HINTS = (
    "summary",
    "summarize",
    "summarise",
    "overview",
    "what is this document",
    "what is the document",
    "what does the document contain",
    "what the document contain",
    "what is this pdf",
    "what is the pdf",
    "explain the document",
    "explain this document",
    "describe the document",
    "main idea",
    "talking about",
    "about this pdf",
    "about the pdf",
    "explain it",
    "اشرح",
    "لخص",
    "ملخص",
    "ما الذي يحتويه",
    "عن ماذا",
    "بالعربية",
)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for", "and",
    "or", "what", "which", "who", "whom", "whose", "this", "that", "these", "those",
    "it", "please", "can", "you", "make", "about", "explain", "summary", "document",
    "pdf", "tell", "me", "do", "does", "contain", "talking",
}


class RetrievedChunk(Chunk):
    """Chunk enriched with a retrieval relevance score (0-1)."""
    relevance_score: float = 0.0


def _row_to_chunk(row) -> RetrievedChunk:
    """Reconstruct a RetrievedChunk from a sqlite3.Row."""
    from datetime import datetime
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        document_name=row["document_name"],
        document_id=row["document_id"],
        document_set_id=row["document_set_id"],
        page_number=row["page_number"],
        paragraph_index=row["paragraph_index"],
        text=row["text"],
        entities=stored_entity_labels(row["entities_json"]),
        char_start=row["char_start"] or 0,
        char_end=row["char_end"] or 0,
        word_count=row["word_count"] or 0,
        chunk_index_in_document=row["chunk_index_in_document"] or 0,
        language=row["language"] or "en",
        timestamp_indexed=datetime.fromisoformat(row["timestamp_indexed"]),
        relevance_score=0.0,
    )


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]


def _is_overview_query(query: str) -> bool:
    lower = query.lower()
    return any(hint in lower for hint in _OVERVIEW_HINTS)


def _hydrate_chunks(chunk_ids: List[str]) -> Dict[str, RetrievedChunk]:
    if not chunk_ids:
        return {}
    with get_db() as conn:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    return {r["chunk_id"]: _row_to_chunk(r) for r in rows}


def _vector_candidates(
    query: str,
    document_set_id: str,
    limit: int,
) -> List[RetrievedChunk]:
    query_vec = embed_query(query)
    collection = _get_collection(document_set_id)
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(limit, count),
        include=["metadatas", "distances"],
    )
    if not results["ids"][0]:
        log.warning("No results from ChromaDB for document_set_id=%s", document_set_id)
        return []

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    chunk_ids = [m["chunk_id"] for m in metadatas]
    rows_by_id = _hydrate_chunks(chunk_ids)

    retrieved: List[RetrievedChunk] = []
    for cid, dist in zip(chunk_ids, distances):
        chunk = rows_by_id.get(cid)
        if chunk is None:
            log.warning("chunk_id %s in ChromaDB but missing from SQLite", cid)
            continue
        chunk.relevance_score = round(max(0.0, 1.0 - dist), 4)
        retrieved.append(chunk)
    return retrieved


def _hyde_candidates(
    question: str,
    document_set_id: str,
    limit: int,
) -> List[RetrievedChunk]:
    """
    Hypothetical Document Embeddings (HyDE):

    Instead of embedding the raw question (which lives in "question space"),
    generate a short hypothetical expert answer and embed THAT (which lives in
    "answer space" — the same space as actual document chunks). This dramatically
    improves retrieval precision for specific factual questions.

    Uses generate_fast() (thinking disabled, temp=0) to keep latency low.
    Gracefully returns [] on any failure so normal retrieval continues.
    """
    if not USE_HYDE:
        return []
    try:
        from backend.rag.llm_client import generate_fast
        hyde_system = (
            "You are an expert document analyst. Write a precise 2-3 sentence answer "
            "to the question below. Be specific: include technical terms, named entities, "
            "numerical values, and professional vocabulary that would appear in a formal "
            "document. Do not say you don't know — make your best expert estimation."
        )
        hypothetical = generate_fast(question, hyde_system)
        if not hypothetical or len(hypothetical.strip()) < 20:
            return []
        hyde_query = hypothetical.strip()[:600]
        log.debug("HyDE query generated (%d chars)", len(hyde_query))
        return _vector_candidates(hyde_query, document_set_id, limit)
    except Exception as e:
        log.warning("HyDE failed, falling back to raw query: %s", e)
        return []


def _lexical_candidates(
    query: str,
    document_set_id: str,
    limit: int,
) -> List[RetrievedChunk]:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM chunks
            WHERE document_set_id = ?
            ORDER BY chunk_index_in_document
            """,
            (document_set_id,),
        ).fetchall()

    scored: List[tuple[float, RetrievedChunk]] = []
    for row in rows:
        chunk = _row_to_chunk(row)
        chunk_tokens = set(_tokenize(chunk.text))
        if not chunk_tokens:
            continue
        overlap = len(query_tokens & chunk_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(query_tokens), 1)
        chunk.relevance_score = round(min(0.99, 0.45 + score), 4)
        scored.append((score, chunk))

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].relevance_score,
            -item[1].page_number,
            -item[1].chunk_index_in_document,
        ),
        reverse=True,
    )
    return [chunk for _, chunk in scored[:limit]]


def _overview_candidates(
    document_set_id: str,
    limit: int,
) -> List[RetrievedChunk]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM chunks
            WHERE document_set_id = ?
            ORDER BY document_id, chunk_index_in_document
            """,
            (document_set_id,),
        ).fetchall()

    by_doc: Dict[str, List] = defaultdict(list)
    for row in rows:
        by_doc[row["document_id"]].append(row)

    selected: List[RetrievedChunk] = []
    for doc_rows in by_doc.values():
        if not doc_rows:
            continue
        picks = {0, min(1, len(doc_rows) - 1), len(doc_rows) // 2, len(doc_rows) - 1}
        if len(doc_rows) > 4:
            picks.add(len(doc_rows) // 3)
            picks.add((2 * len(doc_rows)) // 3)
        for idx in sorted(picks):
            row = doc_rows[idx]
            chunk = _row_to_chunk(row)
            chunk.relevance_score = round(max(chunk.relevance_score, 0.52), 4)
            selected.append(chunk)

    selected.sort(key=lambda c: (c.document_name, c.chunk_index_in_document))
    return selected[:limit]


def _rrf_merge(
    *groups: List[RetrievedChunk],
    limit: int,
) -> List[RetrievedChunk]:
    """
    Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d))

    Far more principled than adding raw similarity scores.
    Respects relative ranking within each retriever, not absolute scores.
    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    """
    rrf_scores: Dict[str, float] = {}
    chunks_by_id: Dict[str, RetrievedChunk] = {}

    for group in groups:
        for rank, chunk in enumerate(group):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if cid not in chunks_by_id or chunk.relevance_score > chunks_by_id[cid].relevance_score:
                chunks_by_id[cid] = chunk

    if not rrf_scores:
        return []

    # Normalise to 0-1 so relevance_score % in the prompt is meaningful
    max_score = max(rrf_scores.values())
    for cid, rrf in rrf_scores.items():
        chunks_by_id[cid].relevance_score = round(rrf / max_score, 4)

    return sorted(
        chunks_by_id.values(),
        key=lambda c: rrf_scores[c.chunk_id],
        reverse=True,
    )[:limit]


def _expand_with_neighbors(
    chunks: List[RetrievedChunk],
    window: int = CONTEXT_WINDOW,
) -> List[RetrievedChunk]:
    """
    For each retrieved chunk, also fetch its ±window adjacent chunks from SQLite.

    This solves the "isolated window" problem: when chunk 42 is the most relevant,
    chunks 41 and 43 complete the passage. The LLM gets coherent reading context
    instead of sentences that start and end mid-thought.

    Neighbor chunks are assigned relevance_score=0.30 — above MIN_RELEVANCE
    so they survive filtering, but below semantic hits so they don't crowd out
    the primary evidence in the prompt.
    """
    seen_ids = {c.chunk_id for c in chunks}
    neighbors: List[RetrievedChunk] = []

    to_fetch: List[tuple] = []
    for chunk in chunks:
        idx = chunk.chunk_index_in_document
        doc_id = chunk.document_id
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor_idx = idx + offset
            if neighbor_idx >= 0:
                to_fetch.append((doc_id, neighbor_idx))

    if not to_fetch:
        return chunks

    with get_db() as conn:
        for doc_id, idx in to_fetch:
            row = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND chunk_index_in_document = ?",
                (doc_id, idx),
            ).fetchone()
            if row:
                c = _row_to_chunk(row)
                if c.chunk_id not in seen_ids:
                    c.relevance_score = 0.30
                    neighbors.append(c)
                    seen_ids.add(c.chunk_id)

    return chunks + neighbors


def get_document_names(document_set_id: str) -> List[str]:
    """Return unique document names in this set, ordered alphabetically."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT document_name FROM chunks WHERE document_set_id = ? ORDER BY document_name",
            (document_set_id,),
        ).fetchall()
    return [r["document_name"] for r in rows]


def retrieve(
    query: str,
    document_set_id: str,
    k: int = DEFAULT_K,
) -> List[RetrievedChunk]:
    """
    Multi-signal hybrid retrieval pipeline:
        1. HyDE vector pass  — hypothetical answer embedding (answer space search)
        2. Direct vector pass — raw query embedding (question space search)
        3. Lexical pass      — exact-term token overlap
        4. Overview sampling — for broad summary/overview questions
        5. RRF merge         — mathematically principled rank fusion
        6. Neighbor expansion — ±1 adjacent chunks for full passage context
    """
    overview_mode = _is_overview_query(query)
    final_k = max(k, OVERVIEW_K if overview_mode else DEFAULT_K)
    vector_limit = max(final_k * VECTOR_FETCH_MULTIPLIER, 15)

    # For overview queries, skip HyDE (sampling is more representative)
    hyde_hits = _hyde_candidates(query, document_set_id, vector_limit) if not overview_mode else []
    vector_hits = _vector_candidates(query, document_set_id, vector_limit)
    lexical_hits = _lexical_candidates(query, document_set_id, LEXICAL_LIMIT)
    overview_hits = _overview_candidates(document_set_id, OVERVIEW_LIMIT) if overview_mode else []

    active_groups = [g for g in [hyde_hits, vector_hits, lexical_hits, overview_hits] if g]
    if not active_groups:
        return []

    retrieved = _rrf_merge(*active_groups, limit=final_k)

    if not retrieved and overview_mode:
        return _overview_candidates(document_set_id, final_k)

    # Contextual expansion: add adjacent chunks for full passage context
    if retrieved and CONTEXT_WINDOW > 0:
        retrieved = _expand_with_neighbors(retrieved, window=CONTEXT_WINDOW)

    if overview_mode:
        filtered = [c for c in retrieved if c.relevance_score >= MIN_RELEVANCE * 0.6]
        return sorted(filtered, key=lambda c: c.relevance_score, reverse=True) or _overview_candidates(document_set_id, final_k)

    filtered = [c for c in retrieved if c.relevance_score >= MIN_RELEVANCE]
    return sorted(filtered, key=lambda c: c.relevance_score, reverse=True)
