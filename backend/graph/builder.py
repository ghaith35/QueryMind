"""
Graph access helpers used by routes.
"""

from typing import List, Optional

from backend.db.connection import get_chunk_by_id, get_chunks_by_doc_set, get_db
from backend.indexing.graph_builder import GraphState, get_graph, rebuild_graph_from_chunks


def get_or_rebuild_graph(document_set_id: str) -> GraphState:
    graph = get_graph(document_set_id)
    if graph.nodes:
        return graph

    with get_db() as conn:
        rows = get_chunks_by_doc_set(conn, document_set_id)
    if rows:
        graph = rebuild_graph_from_chunks(document_set_id, rows)
    return graph


def get_graph_node_chunk_ids(
    document_set_id: str,
    node_id: str,
    document_name: str | None = None,
) -> List[str]:
    graph = get_or_rebuild_graph(document_set_id)
    node = graph.nodes.get(node_id)
    if not node:
        return []

    chunk_ids = list(node.chunk_ids)
    if document_name:
        chunk_ids = [
            chunk_id
            for chunk_id in chunk_ids
            if graph.chunk_documents.get(chunk_id) == document_name
        ]
    return chunk_ids


def get_chunk_or_none(chunk_id: str):
    with get_db() as conn:
        return get_chunk_by_id(conn, chunk_id)
