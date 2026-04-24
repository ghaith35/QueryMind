"""
Indexing pipeline orchestrator.

Flow per document:
    extract → chunk → entity extraction → embed (batched) →
    store ChromaDB + SQLite → update graph → emit progress

Progress callbacks are plain callables (not async) so the pipeline can run
in a thread pool. FastAPI routes wrap these with asyncio.run_coroutine_threadsafe
when broadcasting over WebSocket.
"""

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import chromadb
from chromadb.config import Settings

from backend.db.connection import (
    get_db,
    insert_chunk,
    upsert_document_set,
    increment_chunk_count,
)
from backend.errors import AppError, ErrorCode
from backend.indexing.chunker import chunk_document
from backend.indexing.embedder import embed_chunks
from backend.indexing.entity_extractor import (
    entity_labels,
    extract_entities,
    serialise_entities,
)
from backend.indexing.extractor import extract_pdf, file_hash, validate_extraction
from backend.indexing.graph_builder import get_graph
from backend.schemas.graph import GraphDiff
from backend.schemas.websocket import (
    IndexProgressPayload,
    GraphUpdatePayload,
    index_progress_msg,
    graph_update_msg,
)

log = logging.getLogger(__name__)

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
GRAPH_BATCH_CHUNKS = int(os.getenv("GRAPH_BATCH_SIZE", "6"))
GRAPH_BATCH_MS = int(os.getenv("GRAPH_BATCH_MS", "250"))

_chroma_client: Optional[chromadb.PersistentClient] = None

def init_chroma(persist_dir: str | Path) -> None:
    global _chroma_client
    _chroma_client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    log.info("ChromaDB initialised at %s", persist_dir)


def _get_collection(document_set_id: str):
    if _chroma_client is None:
        raise RuntimeError("init_chroma() must be called before indexing")
    return _chroma_client.get_or_create_collection(
        name="set_" + document_set_id.replace("-", "")[:32],
        metadata={"hnsw:space": "cosine"},
    )


def delete_set_collection(document_set_id: str) -> None:
    """Delete the entire ChromaDB collection for a document set."""
    if _chroma_client is None:
        return
    collection_name = "set_" + document_set_id.replace("-", "")[:32]
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass


def document_exists(document_id: str, document_set_id: str | None = None) -> bool:
    """Check if a document_id is already in SQLite."""
    with get_db() as conn:
        if document_set_id:
            row = conn.execute(
                """
                SELECT 1 FROM chunks
                WHERE document_id = ? AND document_set_id = ?
                LIMIT 1
                """,
                (document_id, document_set_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1",
                (document_id,),
            ).fetchone()
    return row is not None


def index_document(
    pdf_path: str | Path,
    document_set_id: Optional[str] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    on_graph_update: Optional[Callable[[dict], None]] = None,
) -> dict:
    """
    Index a single PDF document end-to-end.

    Args:
        pdf_path: path to the PDF file
        document_set_id: UUID of the document set (created if None)
        on_progress: called with serialised IndexProgressPayload dicts
        on_graph_update: called with serialised GraphUpdatePayload dicts

    Returns:
        dict with document_id, chunk_count, elapsed_seconds
    """
    t_start = time.time()
    pdf_path = Path(pdf_path)
    document_name = pdf_path.name
    doc_id = file_hash(pdf_path)

    if document_set_id is None:
        document_set_id = str(uuid.uuid4())

    if document_exists(doc_id, document_set_id):
        log.info("Skipped duplicate: %s", document_name)
        return {
            "document_id": doc_id,
            "document_set_id": document_set_id,
            "document_name": document_name,
            "chunk_count": 0,
            "elapsed_seconds": 0.0,
            "skipped_duplicate": True,
        }

    job_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    def emit_progress(stage: str, processed: int, total: int) -> None:
        if on_progress is None:
            return
        payload = IndexProgressPayload(
            job_id=job_id,
            document_name=document_name,
            chunks_processed=processed,
            total_chunks_estimated=total,
            stage=stage,
            percent=round(100 * processed / max(total, 1), 1),
        )
        on_progress(index_progress_msg(payload).model_dump())

    def emit_graph(diff: GraphDiff) -> None:
        if on_graph_update is None or diff.is_empty():
            return
        payload = GraphUpdatePayload(
            document_set_id=document_set_id,
            added_nodes=diff.added_nodes,
            added_edges=diff.added_edges,
            updated_nodes=diff.updated_nodes,
        )
        on_graph_update(graph_update_msg(payload).model_dump())

    # ── 1. Extract ─────────────────────────────────────────────
    log.info("[%s] Extracting %s", job_id[:8], document_name)
    emit_progress("extracting", 0, 1)

    pages = extract_pdf(pdf_path)
    ok, reason = validate_extraction(pages)
    if not ok:
        raise AppError(
            ErrorCode.INVALID_PDF,
            reason,
            recoverable=False,
            status_code=400,
        )

    emit_progress("chunking", 0, 1)

    # ── 2. Chunk ───────────────────────────────────────────────
    log.info("[%s] Chunking %d pages", job_id[:8], len(pages))
    chunks = chunk_document(
        pages=pages,
        document_id=doc_id,
        document_name=document_name,
        document_set_id=document_set_id,
        timestamp=timestamp,
    )
    total = len(chunks)
    log.info("[%s] %d chunks produced", job_id[:8], total)

    # ── 3. Ensure document_set row exists ──────────────────────
    with get_db() as conn:
        upsert_document_set(conn, document_set_id, document_name)
        conn.commit()

    # ── 4. Entity extraction ───────────────────────────────────
    emit_progress("embedding", 0, total)
    log.info("[%s] Entity extraction + embedding", job_id[:8])

    graph = get_graph(document_set_id)
    collection = _get_collection(document_set_id)

    pending_diff = GraphDiff(document_set_id=document_set_id)
    last_graph_emit = time.time()

    # Process in embed batches
    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        batch_entities: dict[str, list[tuple[str, str]]] = {}

        # Entity extraction per chunk (before embedding — we need entities for graph)
        for chunk in batch:
            ents = extract_entities(chunk.text, chunk.language)
            batch_entities[chunk.chunk_id] = ents
            chunk.entities = entity_labels(ents)
            diff = graph.add_chunk_entities(
                chunk_id=chunk.chunk_id,
                document_name=document_name,
                entities=ents,
            )
            # Merge diffs
            pending_diff.added_nodes.extend(diff.added_nodes)
            pending_diff.added_edges.extend(diff.added_edges)
            pending_diff.updated_nodes.extend(diff.updated_nodes)

        # Embed batch
        embeddings = embed_chunks([c.text for c in batch])

        # ── 5. Store in ChromaDB ───────────────────────────────
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[c.to_chroma_metadata() for c in batch],
        )

        # ── 6. Store in SQLite ────────────────────────────────
        with get_db() as conn:
            for chunk in batch:
                row = chunk.to_sqlite_row()
                row["entities_json"] = serialise_entities(batch_entities[chunk.chunk_id])
                insert_chunk(conn, row)
            increment_chunk_count(conn, document_set_id, len(batch))
            conn.commit()

        processed = min(batch_start + EMBED_BATCH_SIZE, total)
        emit_progress("embedding", processed, total)

        # Emit batched graph update every 20 chunks or 500ms
        elapsed_ms = (time.time() - last_graph_emit) * 1000
        if (
            len(pending_diff.added_nodes) + len(pending_diff.added_edges) >= GRAPH_BATCH_CHUNKS
            or elapsed_ms >= GRAPH_BATCH_MS
        ):
            emit_graph(pending_diff)
            pending_diff = GraphDiff(document_set_id=document_set_id)
            last_graph_emit = time.time()

    # Flush remaining graph diff
    emit_graph(pending_diff)

    emit_progress("complete", total, total)

    elapsed = round(time.time() - t_start, 2)
    log.info("[%s] Indexed %s: %d chunks in %.1fs", job_id[:8], document_name, total, elapsed)

    return {
        "document_id": doc_id,
        "document_set_id": document_set_id,
        "document_name": document_name,
        "chunk_count": total,
        "elapsed_seconds": elapsed,
    }
