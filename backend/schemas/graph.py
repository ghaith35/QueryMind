from pydantic import BaseModel, Field, field_validator
from typing import List, Literal
import hashlib


def node_id_from_label(label: str) -> str:
    """Stable ID: sha256(lowercased label)[:12]."""
    return hashlib.sha256(label.lower().encode()).hexdigest()[:12]


class GraphNode(BaseModel):
    id: str             # sha256(label.lower())[:12]
    label: str          # canonical display form
    type: Literal["concept", "person", "organization", "location", "technique"]
    document_sources: List[str] = Field(default_factory=list)  # unique doc names
    chunk_ids: List[str] = Field(default_factory=list)
    frequency: int = 0  # len(chunk_ids)
    is_active: bool = False  # true while answer highlighting

    @classmethod
    def from_label(
        cls,
        label: str,
        entity_type: Literal["concept", "person", "organization", "location", "technique"] = "concept",
    ) -> "GraphNode":
        return cls(id=node_id_from_label(label), label=label, type=entity_type)

    def add_chunk(self, chunk_id: str, document_name: str) -> None:
        if chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)
            self.frequency = len(self.chunk_ids)
        if document_name not in self.document_sources:
            self.document_sources.append(document_name)


class GraphEdge(BaseModel):
    source: str       # node_id
    target: str       # node_id
    relation_type: Literal["co_occurrence"] = "co_occurrence"
    weight: float = 0.0  # normalized 0-1
    co_occurrence_count: int = 0
    is_active: bool = False

    @property
    def edge_id(self) -> str:
        """Canonical ID for WS active_edge_ids list."""
        return f"{self.source}->{self.target}"

    @field_validator("weight")
    @classmethod
    def weight_range(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    def increment(self) -> None:
        self.co_occurrence_count += 1


class GraphDiff(BaseModel):
    """Batched graph update payload sent over WebSocket every 20 chunks or 500ms."""
    document_set_id: str
    added_nodes: List[GraphNode] = Field(default_factory=list)
    added_edges: List[GraphEdge] = Field(default_factory=list)
    updated_nodes: List[dict] = Field(default_factory=list)  # {id, frequency}

    def is_empty(self) -> bool:
        return not (self.added_nodes or self.added_edges or self.updated_nodes)
