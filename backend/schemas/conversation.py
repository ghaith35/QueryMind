import json
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Literal
from .chunk import Citation


class DocumentSet(BaseModel):
    id: str           # UUID
    name: str
    created_at: datetime
    chunk_count: int = 0


class Session(BaseModel):
    id: str           # UUID from browser localStorage
    document_set_id: str
    created_at: datetime
    last_activity: datetime


class Turn(BaseModel):
    turn_id: str      # UUID
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_row(cls, row) -> "Turn":
        citations_raw = json.loads(row["citations_json"] or "[]")
        chunk_ids = json.loads(row["retrieved_chunk_ids_json"] or "[]")
        return cls(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            citations=[Citation(**c) for c in citations_raw],
            retrieved_chunk_ids=chunk_ids,
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def to_sqlite_row(self) -> dict:
        import json
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "citations_json": json.dumps(
                [c.model_dump() for c in self.citations], ensure_ascii=False
            ),
            "retrieved_chunk_ids_json": json.dumps(self.retrieved_chunk_ids),
            "timestamp": self.timestamp.isoformat(),
        }
