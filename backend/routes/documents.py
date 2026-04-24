"""
Document management routes.

POST  /documents/set      — create a new document set
GET   /documents/sets     — list all sets with stats
POST  /documents/upload   — upload PDF, fire-and-forget indexing
DELETE /documents/{doc_id} — remove document from ChromaDB + SQLite
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

import fitz
from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.errors import AppError, ErrorCode, parse_prefixed_error
from backend.db.connection import get_db, upsert_document_set
from backend.indexing.extractor import file_hash_bytes
from backend.indexing.pipeline import _get_collection, delete_set_collection, document_exists, index_document
from backend.ws.manager import ws_manager

router = APIRouter(prefix="/documents", tags=["documents"])

PDF_WATCH_DIR = os.getenv("PDF_WATCH_DIR", "./data/pdfs")
MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB
MAX_PAGES = 500


# ── Pydantic models ───────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: Literal["queued", "indexing", "complete", "failed"]
    message: str | None = None


class CreateSetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class RenameSetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class DocumentInSetOut(BaseModel):
    doc_id: str
    document_name: str
    chunk_count: int


class DocumentSetOut(BaseModel):
    id: str
    name: str
    created_at: datetime
    chunk_count: int = 0


class DocumentSetWithStats(BaseModel):
    id: str
    name: str
    created_at: datetime
    doc_count: int = 0
    chunk_count: int = 0


# ── Helpers ───────────────────────────────────────────────────────

def _validate_pdf_bytes(contents: bytes) -> None:
    if contents[:5] != b"%PDF-":
        raise HTTPException(400, "Not a valid PDF (missing magic bytes)")
    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except Exception as e:
        raise HTTPException(400, f"Corrupt PDF: {e}")
    if page_count == 0:
        raise HTTPException(400, "PDF has no pages")
    if page_count > MAX_PAGES:
        raise HTTPException(413, f"PDF exceeds {MAX_PAGES} pages ({page_count})")


async def _run_indexing(save_path: str, document_set_id: str, session_id: str) -> None:
    """Background task: run sync pipeline in thread pool, push progress via WS."""
    loop = asyncio.get_event_loop()

    def on_progress(msg: dict) -> None:
        ws_manager.send_sync(session_id, msg)

    def on_graph_update(msg: dict) -> None:
        ws_manager.broadcast_sync(msg)

    try:
        result = await loop.run_in_executor(
            None,
            lambda: index_document(
                pdf_path=save_path,
                document_set_id=document_set_id,
                on_progress=on_progress,
                on_graph_update=on_graph_update,
            ),
        )
        from backend.schemas.websocket import IndexCompletePayload, index_complete_msg
        ws_manager.send_sync(session_id, index_complete_msg(IndexCompletePayload(
            job_id=result.get("document_id", ""),
            document_name=result.get("document_name", ""),
            chunk_count=result.get("chunk_count", 0),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
        )).model_dump())
    except AppError as e:
        from backend.schemas.websocket import ErrorPayload, error_msg
        ws_manager.send_sync(session_id, error_msg(ErrorPayload(
            code=e.code,
            message=e.message,
            recoverable=e.recoverable,
        )).model_dump())
    except Exception as e:
        parsed = parse_prefixed_error(str(e), default_code=ErrorCode.INDEX_FAILED)
        from backend.schemas.websocket import ErrorPayload, error_msg
        ws_manager.send_sync(session_id, error_msg(ErrorPayload(
            code=parsed.code,
            message=parsed.message,
            recoverable=parsed.recoverable,
        )).model_dump())


# ── Routes ────────────────────────────────────────────────────────

@router.post("/set", response_model=DocumentSetOut)
def create_document_set(req: CreateSetRequest):
    set_id = str(uuid.uuid4())
    with get_db() as conn:
        existing = conn.execute(
            "SELECT 1 FROM document_sets WHERE LOWER(name) = LOWER(?)", (req.name.strip(),)
        ).fetchone()
        if existing:
            raise HTTPException(409, f'A document set named "{req.name.strip()}" already exists.')
        upsert_document_set(conn, set_id, req.name.strip())
        conn.commit()
    return DocumentSetOut(
        id=set_id,
        name=req.name.strip(),
        created_at=datetime.now(timezone.utc),
        chunk_count=0,
    )


@router.get("/sets", response_model=List[DocumentSetWithStats])
def list_sets():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT
                ds.id, ds.name, ds.created_at,
                COUNT(DISTINCT c.document_id)  AS doc_count,
                COUNT(c.chunk_id)              AS chunk_count
            FROM document_sets ds
            LEFT JOIN chunks c ON c.document_set_id = ds.id
            GROUP BY ds.id
            ORDER BY ds.created_at DESC
        """).fetchall()
    return [
        DocumentSetWithStats(
            id=r["id"],
            name=r["name"],
            created_at=datetime.fromisoformat(r["created_at"]),
            doc_count=r["doc_count"] or 0,
            chunk_count=r["chunk_count"] or 0,
        )
        for r in rows
    ]


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    document_set_id: str = Form(...),
    session_id: str = Form(...),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    contents = await file.read()
    doc_id = file_hash_bytes(contents)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE // 1024 // 1024}MB")
    if len(contents) < 100:
        raise HTTPException(400, "File appears empty")

    _validate_pdf_bytes(contents)

    # Ensure document_set exists
    with get_db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM document_sets WHERE id = ?", (document_set_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(404, f"Document set {document_set_id} not found")

    if document_exists(doc_id, document_set_id):
        return UploadResponse(
            job_id=doc_id,
            filename=file.filename,
            status="complete",
            message=f"Skipped duplicate: {file.filename}",
        )

    # Save to watched folder
    save_dir = os.path.join(PDF_WATCH_DIR, document_set_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)

    with open(save_path, "wb") as f:
        f.write(contents)

    job_id = str(uuid.uuid4())
    asyncio.create_task(_run_indexing(save_path, document_set_id, session_id))

    return UploadResponse(job_id=job_id, filename=file.filename, status="queued")


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT document_set_id FROM chunks WHERE document_id = ? LIMIT 1",
            (doc_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Document not found")

        set_id = row["document_set_id"]
        chunk_ids = [
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM chunks WHERE document_id = ?", (doc_id,)
            ).fetchall()
        ]

        # Remove from ChromaDB
        try:
            collection = _get_collection(set_id)
            if chunk_ids:
                collection.delete(ids=chunk_ids)
        except Exception as e:
            raise HTTPException(500, f"ChromaDB delete failed: {e}")

        # Remove from SQLite
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.commit()

    return {"deleted": True, "chunks_removed": len(chunk_ids)}


@router.get("/set/{set_id}/documents", response_model=List[DocumentInSetOut])
def list_documents_in_set(set_id: str):
    with get_db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM document_sets WHERE id = ?", (set_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(404, "Document set not found")

        rows = conn.execute(
            """
            SELECT document_id, document_name, COUNT(*) AS chunk_count
            FROM chunks
            WHERE document_set_id = ?
            GROUP BY document_id, document_name
            ORDER BY document_name
            """,
            (set_id,),
        ).fetchall()

    return [
        DocumentInSetOut(
            doc_id=r["document_id"],
            document_name=r["document_name"],
            chunk_count=r["chunk_count"],
        )
        for r in rows
    ]


@router.patch("/set/{set_id}")
def rename_document_set(set_id: str, req: RenameSetRequest):
    with get_db() as conn:
        conflict = conn.execute(
            "SELECT 1 FROM document_sets WHERE LOWER(name) = LOWER(?) AND id != ?",
            (req.name.strip(), set_id),
        ).fetchone()
        if conflict:
            raise HTTPException(409, f'A document set named "{req.name.strip()}" already exists.')
        result = conn.execute(
            "UPDATE document_sets SET name = ? WHERE id = ?",
            (req.name.strip(), set_id),
        )
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(404, "Document set not found")
    return {"id": set_id, "name": req.name.strip()}


@router.delete("/set/{set_id}")
def delete_document_set(set_id: str):
    with get_db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM document_sets WHERE id = ?", (set_id,)
        ).fetchone()
        if not exists:
            raise HTTPException(404, "Document set not found")

        chunk_ids = [
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM chunks WHERE document_set_id = ?", (set_id,)
            ).fetchall()
        ]

        # Delete entire ChromaDB collection for this set
        delete_set_collection(set_id)

        # Cascade delete in SQLite
        session_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM sessions WHERE document_set_id = ?", (set_id,)
            ).fetchall()
        ]
        for sid in session_ids:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM sessions WHERE document_set_id = ?", (set_id,))
        conn.execute("DELETE FROM chunks WHERE document_set_id = ?", (set_id,))
        conn.execute("DELETE FROM document_sets WHERE id = ?", (set_id,))
        conn.commit()

    # Clean up PDF files on disk
    save_dir = os.path.join(PDF_WATCH_DIR, set_id)
    if os.path.isdir(save_dir):
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)

    return {"deleted": True, "set_id": set_id, "chunks_removed": len(chunk_ids)}
