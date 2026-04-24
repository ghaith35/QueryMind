"""Graph routes for the knowledge graph view and navigation."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.graph.builder import (
    get_chunk_or_none,
    get_graph_node_chunk_ids,
    get_or_rebuild_graph,
)
from backend.graph.filters import apply_graph_filters
from backend.indexing.entity_extractor import stored_entity_labels
from backend.schemas.graph import GraphEdge, GraphNode

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    node_count: int
    edge_count: int
    total_node_count: int
    total_edge_count: int
    truncated: bool = False
    message: Optional[str] = None
    available_documents: List[str] = Field(default_factory=list)


class GraphChunkPreview(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    paragraph_index: int
    excerpt: str


class ChunkDetailResponse(BaseModel):
    chunk_id: str
    document_set_id: str
    document_name: str
    page_number: int
    paragraph_index: int
    text: str
    language: str
    entities: List[str]


@router.get("/{document_set_id}", response_model=GraphResponse)
def get_graph_endpoint(
    document_set_id: str,
    min_frequency: int = Query(2, ge=1),
    entity_types: Optional[List[str]] = Query(None),
    document_names: Optional[List[str]] = Query(None),
):
    graph = get_or_rebuild_graph(document_set_id)
    result = apply_graph_filters(
        graph,
        min_frequency=min_frequency,
        entity_types=entity_types,
        document_names=document_names or [],
    )
    return GraphResponse(
        nodes=result.nodes,
        edges=result.edges,
        node_count=len(result.nodes),
        edge_count=len(result.edges),
        total_node_count=result.total_node_count,
        total_edge_count=result.total_edge_count,
        truncated=result.truncated,
        message=result.message,
        available_documents=result.available_documents,
    )


@router.get("/{document_set_id}/node/{node_id}/chunks", response_model=List[GraphChunkPreview])
def get_node_chunks_endpoint(
    document_set_id: str,
    node_id: str,
    document_name: Optional[str] = Query(None),
    limit: int = Query(25, ge=1, le=100),
):
    chunk_ids = get_graph_node_chunk_ids(document_set_id, node_id, document_name=document_name)
    if not chunk_ids:
        return []

    previews: List[GraphChunkPreview] = []
    for chunk_id in chunk_ids[:limit]:
        row = get_chunk_or_none(chunk_id)
        if not row:
            continue
        text = row["text"] or ""
        previews.append(GraphChunkPreview(
            chunk_id=row["chunk_id"],
            document_name=row["document_name"],
            page_number=row["page_number"],
            paragraph_index=row["paragraph_index"],
            excerpt=(text[:220] + ("..." if len(text) > 220 else "")).strip(),
        ))
    return previews


@router.get("/chunk/{chunk_id}", response_model=ChunkDetailResponse)
def get_chunk_detail_endpoint(chunk_id: str):
    row = get_chunk_or_none(chunk_id)
    if not row:
        raise HTTPException(404, f"Chunk {chunk_id} not found")

    return ChunkDetailResponse(
        chunk_id=row["chunk_id"],
        document_set_id=row["document_set_id"],
        document_name=row["document_name"],
        page_number=row["page_number"],
        paragraph_index=row["paragraph_index"],
        text=row["text"],
        language=row["language"] or "en",
        entities=stored_entity_labels(row["entities_json"]),
    )
