"""
Chat routes.

POST /chat             — fire-and-forget RAG, answer streamed via WebSocket
GET  /chat/history/{session_id}
"""

import asyncio
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.db.connection import get_db, get_session_turns
from backend.errors import AppError
from backend.rag.answer_generator import ask_streaming
from backend.schemas.conversation import Turn
from backend.services.session import delete_session, ensure_session
from backend.ws.manager import ws_manager

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    document_set_id: str
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    turn_id: str
    status: str = "streaming"


class SessionSwitchRequest(BaseModel):
    document_set_id: str
    previous_session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    document_set_id: str


async def _run_rag(
    session_id: str,
    document_set_id: str,
    question: str,
    turn_id: str,
) -> None:
    """Background task: stream answer tokens to the WebSocket session."""

    def on_token(msg: dict) -> None:
        asyncio.create_task(ws_manager.send(session_id, msg))

    def on_complete(msg: dict) -> None:
        asyncio.create_task(ws_manager.send(session_id, msg))

    def on_error(msg: dict) -> None:
        asyncio.create_task(ws_manager.send(session_id, msg))

    await ask_streaming(
        question=question,
        document_set_id=document_set_id,
        session_id=session_id,
        turn_id=turn_id,
        on_token=on_token,
        on_complete=on_complete,
        on_error=on_error,
    )


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    with get_db() as conn:
        set_exists = conn.execute(
            "SELECT 1 FROM document_sets WHERE id = ?", (req.document_set_id,)
        ).fetchone()
        if not set_exists:
            raise HTTPException(404, f"Document set {req.document_set_id} not found")

        try:
            ensure_session(conn, req.session_id, req.document_set_id)
        except AppError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc
        conn.commit()

    turn_id = str(uuid.uuid4())
    asyncio.create_task(_run_rag(req.session_id, req.document_set_id, req.message, turn_id))

    return ChatResponse(turn_id=turn_id, status="streaming")


@router.get("/history/{session_id}", response_model=List[Turn])
def get_history(session_id: str, limit: int = Query(50, ge=1, le=200)):
    with get_db() as conn:
        rows = get_session_turns(conn, session_id)
    return [Turn.from_row(dict(r)) for r in rows[:limit]]


@router.delete("/session/{session_id}")
def reset_session(session_id: str):
    with get_db() as conn:
        deleted = delete_session(conn, session_id)
        conn.commit()
    return {"deleted": True, **deleted}


@router.post("/session/switch", response_model=SessionResponse)
def switch_session(req: SessionSwitchRequest):
    session_id = str(uuid.uuid4())
    with get_db() as conn:
        set_exists = conn.execute(
            "SELECT 1 FROM document_sets WHERE id = ?",
            (req.document_set_id,),
        ).fetchone()
        if not set_exists:
            raise HTTPException(404, f"Document set {req.document_set_id} not found")
        try:
            session = ensure_session(conn, session_id, req.document_set_id)
        except AppError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc
        conn.commit()
    return SessionResponse(
        session_id=session.id,
        document_set_id=session.document_set_id,
    )
