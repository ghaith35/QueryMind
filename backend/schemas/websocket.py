from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Generic, TypeVar, List, Literal, Optional
from .chunk import Citation
from .graph import GraphNode, GraphEdge

T = TypeVar("T")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WSMessage(BaseModel, Generic[T]):
    type: str
    timestamp: str = Field(default_factory=_now_iso)
    payload: T


# ── Payload types ─────────────────────────────────────────────────

class IndexProgressPayload(BaseModel):
    job_id: str
    document_name: str
    chunks_processed: int
    total_chunks_estimated: int
    stage: Literal["extracting", "chunking", "embedding", "storing", "complete"]
    percent: float  # 0-100


class GraphUpdatePayload(BaseModel):
    document_set_id: str
    added_nodes: List[GraphNode] = Field(default_factory=list)
    added_edges: List[GraphEdge] = Field(default_factory=list)
    updated_nodes: List[dict] = Field(default_factory=list)  # {id, frequency}


class AnswerStreamPayload(BaseModel):
    session_id: str
    turn_id: str
    token: str
    is_final: bool = False


class AnswerCompletePayload(BaseModel):
    session_id: str
    turn_id: str
    full_answer: str
    citations: List[Citation]
    active_node_ids: List[str]
    active_edge_ids: List[str]  # format "source_id->target_id"
    elapsed_seconds: float | None = None


class IndexCompletePayload(BaseModel):
    job_id: str
    document_name: str
    chunk_count: int
    elapsed_seconds: float


class ErrorPayload(BaseModel):
    code: Literal[
        "INDEX_FAILED",
        "LLM_FAILED",
        "CITATION_HALLUCINATED",
        "CITATION_MISSING",
        "INVALID_PDF",
        "DUPLICATE_DOCUMENT",
        "SESSION_SCOPE_MISMATCH",
        "GRAPH_LOAD_FAILED",
    ]
    message: str
    recoverable: bool


# ── Typed constructors (avoids stringly-typed type= at call sites) ─

def index_progress_msg(payload: IndexProgressPayload) -> WSMessage:
    return WSMessage(type="index_progress", payload=payload)

def index_complete_msg(payload: IndexCompletePayload) -> WSMessage:
    return WSMessage(type="index_complete", payload=payload)

def graph_update_msg(payload: GraphUpdatePayload) -> WSMessage:
    return WSMessage(type="graph_update", payload=payload)

def answer_stream_msg(payload: AnswerStreamPayload) -> WSMessage:
    return WSMessage(type="answer_stream", payload=payload)

def answer_complete_msg(payload: AnswerCompletePayload) -> WSMessage:
    return WSMessage(type="answer_complete", payload=payload)

def error_msg(payload: ErrorPayload) -> WSMessage:
    return WSMessage(type="error", payload=payload)
