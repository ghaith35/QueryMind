from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime
from typing import List, Literal


class Chunk(BaseModel):
    chunk_id: str  # "{doc_hash[:8]}_{chunk_index:04d}"  e.g. "a3f9c2b1_0042"
    document_name: str
    document_id: str   # sha256(filename + filesize)[:16]
    document_set_id: str  # UUID
    page_number: int   # 1-indexed
    paragraph_index: int  # 0-indexed within page
    text: str          # raw chunk text, max 1200 chars
    entities: List[str] = Field(default_factory=list)  # max 15 per chunk
    char_start: int
    char_end: int
    word_count: int
    chunk_index_in_document: int  # 0-indexed globally within doc
    language: Literal["en", "fr", "ar", "mixed"]
    timestamp_indexed: datetime

    @field_validator("text")
    @classmethod
    def text_max_length(cls, v: str) -> str:
        if len(v) > 1200:
            raise ValueError(f"chunk text exceeds 1200 chars ({len(v)})")
        return v

    @field_validator("entities")
    @classmethod
    def entities_max_count(cls, v: List[str]) -> List[str]:
        return v[:15]

    def to_chroma_metadata(self) -> dict:
        """Stripped metadata for ChromaDB — only filter fields, no blowup."""
        entities_csv = ",".join(self.entities)[:500]
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_set_id": self.document_set_id,
            "page_number": self.page_number,
            "language": self.language,
            "entities_csv": entities_csv,
        }

    def to_sqlite_row(self) -> dict:
        """Full data row for SQLite chunks table."""
        import json
        return {
            "chunk_id": self.chunk_id,
            "document_set_id": self.document_set_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "paragraph_index": self.paragraph_index,
            "text": self.text,
            "entities_json": json.dumps(self.entities, ensure_ascii=False),
            "char_start": self.char_start,
            "char_end": self.char_end,
            "word_count": self.word_count,
            "chunk_index_in_document": self.chunk_index_in_document,
            "language": self.language,
            "timestamp_indexed": self.timestamp_indexed.isoformat(),
        }

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "chunk_id": "a3f9c2b1_0042",
                "document_name": "cardiology_guidelines_2024.pdf",
                "document_id": "a3f9c2b1d4e5f6a7",
                "document_set_id": "550e8400-e29b-41d4-a716-446655440000",
                "page_number": 12,
                "paragraph_index": 3,
                "text": "Hypertension management in adults over 60...",
                "entities": ["hypertension", "cardiology", "ACE inhibitor"],
                "char_start": 450,
                "char_end": 1620,
                "word_count": 187,
                "chunk_index_in_document": 42,
                "language": "en",
                "timestamp_indexed": "2026-04-22T10:30:00Z",
            }
        })


class Citation(BaseModel):
    document_name: str
    page_number: int
    paragraph_index: int
    chunk_id: str
    relevance_score: float  # 0-1, from retrieval cosine similarity
    excerpt: str  # first 150 chars of chunk.text

    @field_validator("relevance_score")
    @classmethod
    def score_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"relevance_score must be 0-1, got {v}")
        return round(v, 4)

    @field_validator("excerpt")
    @classmethod
    def excerpt_max(cls, v: str) -> str:
        return v[:150]
