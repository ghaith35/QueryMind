"""
Answer generation orchestrator.

Flow:
    retrieve → format prompt → LLM call → parse citations →
    validate (retry if hallucinated) → compute active graph → return

The hallucination recovery loop:
    - Attempt 1: standard prompt
    - Attempt 2 (on hallucination): stricter prompt with explicit warning
    - After 2 failures: strip hallucinated citations, flag answer, emit error event
"""

import logging
import re
import time
import uuid
from typing import List, Optional, Tuple, Callable, AsyncGenerator

from backend.db.connection import get_db, insert_turn
from backend.indexing.graph_builder import get_graph
from backend.rag.citation_parser import (
    parse_citations,
    strip_hallucinated_citations,
    is_refusal,
)
from backend.rag.llm_client import generate, stream_generate
from backend.rag.prompt import build_system_prompt
from backend.rag.retriever import RetrievedChunk, retrieve, get_document_names
from backend.services.conversation import build_history_window, serialise_history
from backend.services.session import touch_session
from backend.schemas.chunk import Citation
from backend.schemas.conversation import Turn
from backend.schemas.websocket import (
    AnswerCompletePayload,
    AnswerStreamPayload,
    ErrorPayload,
    answer_complete_msg,
    answer_stream_msg,
    error_msg,
)

log = logging.getLogger(__name__)

MAX_RETRIES = 1  # one retry with strict prompt before flagging


def _load_history(session_id: str) -> list:
    return serialise_history(build_history_window(session_id))


def _save_turn(
    session_id: str,
    role: str,
    content: str,
    citations: List[Citation],
    chunk_ids: List[str],
    turn_id: str | None = None,
) -> str:
    turn_id = turn_id or str(uuid.uuid4())
    turn = Turn(
        turn_id=turn_id,
        session_id=session_id,
        role=role,
        content=content,
        citations=citations,
        retrieved_chunk_ids=chunk_ids,
    )
    with get_db() as conn:
        insert_turn(conn, turn.to_sqlite_row())
        touch_session(conn, session_id)
        conn.commit()
    return turn_id


def _should_retry_citations(
    answer_text: str,
    valid_citations: List[Citation],
    hallucinated: List[str],
) -> bool:
    if hallucinated:
        return True
    return len(answer_text.strip()) > 100 and not valid_citations and not is_refusal(answer_text)


def _build_retrieval_query(question: str, history: list) -> str:
    """
    Expand short/follow-up prompts using recent context so queries like
    "in Arabic please" or "explain it" still retrieve the right chunks.
    """
    q = question.strip()
    lower = q.lower()
    user_turns = [t["content"] for t in history if t["role"] == "user"]
    if user_turns and user_turns[-1].strip() == q:
        user_turns = user_turns[:-1]
    recent_user = user_turns[-1] if user_turns else ""
    recent_assistant = next((t["content"] for t in reversed(history) if t["role"] == "assistant"), "")

    followup_markers = (
        len(q) < 40
        or any(token in lower for token in [" it", "that", "this", "translate", "summary", "summar", "explain", "arabic", "بالعربية", "اشرح", "لخص"])
        or any(token in q for token in ["هذا", "ذلك", "بالعربية", "اشرح", "لخص"])
    )
    if not followup_markers:
        return q

    context_parts = [part.strip() for part in [recent_user, recent_assistant] if part.strip()]
    if not context_parts:
        return q

    assistant_snippet = re.sub(r"\s+", " ", recent_assistant).strip()[:400]
    return "\n".join([
        f"Current request: {q}",
        *(["Previous user request: " + recent_user.strip()] if recent_user.strip() else []),
        *(["Previous assistant answer: " + assistant_snippet] if assistant_snippet else []),
    ])


def ask(
    question: str,
    document_set_id: str,
    session_id: str,
    k: int = 12,
) -> AnswerCompletePayload:
    """
    Blocking (non-streaming) answer generation. Used for testing.

    Returns AnswerCompletePayload — the same structure sent over WebSocket.
    """
    started_at = time.perf_counter()
    # Save user turn
    _save_turn(session_id, "user", question, [], [])

    history = _load_history(session_id)
    retrieval_query = _build_retrieval_query(question, history)
    chunks = retrieve(retrieval_query, document_set_id, k=k)

    if not chunks:
        answer_text = "The provided documents do not contain information to answer this question."
        turn_id = _save_turn(session_id, "assistant", answer_text, [], [])
        return AnswerCompletePayload(
            session_id=session_id,
            turn_id=turn_id,
            full_answer=answer_text,
            citations=[],
            active_node_ids=[],
            active_edge_ids=[],
        )

    chunk_ids = [c.chunk_id for c in chunks]
    doc_names = get_document_names(document_set_id)
    valid_citations: List[Citation] = []
    answer_text = ""
    all_hallucinated: List[str] = []

    for attempt in range(MAX_RETRIES + 1):
        strict = attempt > 0
        system_prompt = build_system_prompt(chunks, history, strict=strict, document_names=doc_names)
        answer_text = generate(question, system_prompt)
        _, valid_citations, hallucinated = parse_citations(answer_text, chunks)

        if not _should_retry_citations(answer_text, valid_citations, hallucinated):
            break

        log.warning(
            "Attempt %d: hallucinated citation IDs %s", attempt + 1, hallucinated
        )
        all_hallucinated = hallucinated

        if attempt == MAX_RETRIES:
            answer_text = strip_hallucinated_citations(answer_text, hallucinated)
            log.error("Hallucinated citations remain after retry — stripped")

    # Compute graph highlighting
    graph = get_graph(document_set_id)
    cited_chunk_ids = [c.chunk_id for c in valid_citations]
    active_node_ids, active_edge_ids = graph.get_active_ids_for_chunks(cited_chunk_ids)

    # Save assistant turn
    turn_id = _save_turn(
        session_id, "assistant", answer_text, valid_citations, chunk_ids
    )

    return AnswerCompletePayload(
        session_id=session_id,
        turn_id=turn_id,
        full_answer=answer_text,
        citations=valid_citations,
        active_node_ids=active_node_ids,
        active_edge_ids=active_edge_ids,
        elapsed_seconds=round(time.perf_counter() - started_at, 2),
    )


async def ask_streaming(
    question: str,
    document_set_id: str,
    session_id: str,
    turn_id: str,
    on_token: Callable[[dict], None],
    on_complete: Callable[[dict], None],
    on_error: Callable[[dict], None],
    k: int = 12,
) -> None:
    """
    Streaming answer generation for WebSocket delivery.

    Tokens are emitted via on_token as answer_stream messages.
    Final validated answer + citations emitted via on_complete.
    Errors emitted via on_error.

    Note: streaming uses the non-strict prompt on attempt 1. If hallucinations
    are found post-generation, we fall back to a blocking retry with the strict
    prompt (streaming is reset — the frontend receives [citation omitted] markers).
    """
    started_at = time.perf_counter()
    _save_turn(session_id, "user", question, [], [])

    history = _load_history(session_id)
    retrieval_query = _build_retrieval_query(question, history)
    chunks = retrieve(retrieval_query, document_set_id, k=k)

    if not chunks:
        answer_text = "The provided documents do not contain information to answer this question."
        _save_turn(session_id, "assistant", answer_text, [], [], turn_id=turn_id)
        on_complete(answer_complete_msg(AnswerCompletePayload(
            session_id=session_id,
            turn_id=turn_id,
            full_answer=answer_text,
            citations=[],
            active_node_ids=[],
            active_edge_ids=[],
        )).model_dump())
        return

    chunk_ids = [c.chunk_id for c in chunks]
    doc_names = get_document_names(document_set_id)
    system_prompt = build_system_prompt(chunks, history, strict=False, document_names=doc_names)

    # Stream tokens
    full_text = ""
    try:
        async for token in stream_generate(question, system_prompt):
            full_text += token
            on_token(answer_stream_msg(AnswerStreamPayload(
                session_id=session_id,
                turn_id=turn_id,
                token=token,
                is_final=False,
            )).model_dump())
    except Exception as e:
        log.exception("Streaming failed: %s", e)
        on_error(error_msg(ErrorPayload(
            code="LLM_FAILED",
            message=str(e),
            recoverable=True,
        )).model_dump())
        return

    # Validate citations on full text
    _, valid_citations, hallucinated = parse_citations(full_text, chunks)

    if _should_retry_citations(full_text, valid_citations, hallucinated):
        if hallucinated:
            log.warning("Stream had hallucinated IDs %s — retrying blocking", hallucinated)
        else:
            log.warning("Stream answer had no valid citations — retrying blocking")
        try:
            strict_prompt = build_system_prompt(chunks, history, strict=True, document_names=doc_names)
            full_text = generate(question, strict_prompt)
            _, valid_citations, hallucinated = parse_citations(full_text, chunks)
            if hallucinated:
                full_text = strip_hallucinated_citations(full_text, hallucinated)
                on_error(error_msg(ErrorPayload(
                    code="CITATION_HALLUCINATED",
                    message=f"Stripped {len(hallucinated)} hallucinated IDs",
                    recoverable=False,
                )).model_dump())
            elif not valid_citations and not is_refusal(full_text):
                on_error(error_msg(ErrorPayload(
                    code="CITATION_MISSING",
                    message="Answer generated without verifiable citations. Review the response carefully.",
                    recoverable=True,
                )).model_dump())
        except Exception as e:
            on_error(error_msg(ErrorPayload(
                code="LLM_FAILED", message=str(e), recoverable=True
            )).model_dump())
            return

    # Emit final token marker
    on_token(answer_stream_msg(AnswerStreamPayload(
        session_id=session_id,
        turn_id=turn_id,
        token="",
        is_final=True,
    )).model_dump())

    graph = get_graph(document_set_id)
    cited_ids = [c.chunk_id for c in valid_citations]
    active_node_ids, active_edge_ids = graph.get_active_ids_for_chunks(cited_ids)

    _save_turn(
        session_id,
        "assistant",
        full_text,
        valid_citations,
        chunk_ids,
        turn_id=turn_id,
    )

    on_complete(answer_complete_msg(AnswerCompletePayload(
        session_id=session_id,
        turn_id=turn_id,
        full_answer=full_text,
        citations=valid_citations,
        active_node_ids=active_node_ids,
        active_edge_ids=active_edge_ids,
        elapsed_seconds=round(time.perf_counter() - started_at, 2),
    )).model_dump())
