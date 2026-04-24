"""
Graph filtering helpers for the knowledge graph API.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

from backend.indexing.graph_builder import GraphState
from backend.schemas.graph import GraphEdge, GraphNode

DEFAULT_TRUNCATE_THRESHOLD = 500
DEFAULT_HARD_LIMIT = 300


@dataclass
class GraphFilterResult:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_node_count: int
    total_edge_count: int
    truncated: bool
    message: str | None
    available_documents: List[str]


def _filtered_chunk_ids(
    graph: GraphState,
    node: GraphNode,
    document_names_set: set[str],
) -> List[str]:
    if not document_names_set:
        return list(node.chunk_ids)
    return [
        chunk_id
        for chunk_id in node.chunk_ids
        if graph.chunk_documents.get(chunk_id) in document_names_set
    ]


def apply_graph_filters(
    graph: GraphState,
    min_frequency: int = 1,
    entity_types: Optional[Iterable[str]] = None,
    document_names: Optional[List[str]] = None,
    truncate_threshold: int = DEFAULT_TRUNCATE_THRESHOLD,
    hard_limit: int = DEFAULT_HARD_LIMIT,
) -> GraphFilterResult:
    entity_types_set = set(entity_types or [])
    filter_by_type = len(entity_types_set) > 0
    doc_names_set: set[str] = set(document_names) if document_names else set()

    filtered_nodes: List[GraphNode] = []
    for node in graph.nodes.values():
        chunk_ids = _filtered_chunk_ids(graph, node, doc_names_set)
        frequency = len(chunk_ids)
        if frequency < min_frequency:
            continue
        if filter_by_type and node.type not in entity_types_set:
            continue
        clone = node.model_copy(deep=True)
        clone.chunk_ids = chunk_ids
        clone.frequency = frequency
        if doc_names_set:
            clone.document_sources = [s for s in clone.document_sources if s in doc_names_set]
        filtered_nodes.append(clone)

    filtered_nodes.sort(key=lambda n: (-n.frequency, n.label.lower()))
    total_node_count = len(filtered_nodes)

    filtered_node_ids = {n.id for n in filtered_nodes}
    relevant_edge_counts: dict[tuple[str, str], int] = {}
    for key, edge in graph.edges.items():
        if edge.source not in filtered_node_ids or edge.target not in filtered_node_ids:
            continue
        if doc_names_set:
            count = sum(
                1
                for chunk_id in graph.edge_chunk_ids.get(key, set())
                if graph.chunk_documents.get(chunk_id) in doc_names_set
            )
        else:
            count = edge.co_occurrence_count
        if count > 0:
            relevant_edge_counts[key] = count

    total_edge_count = len(relevant_edge_counts)
    truncated = total_node_count > truncate_threshold
    message = None

    displayed_nodes = filtered_nodes
    if truncated:
        displayed_nodes = filtered_nodes[:hard_limit]
        message = (
            f"Showing top {len(displayed_nodes)} of {total_node_count} nodes - "
            "adjust filters to see more."
        )

    displayed_node_ids = {n.id for n in displayed_nodes}
    max_count = max(relevant_edge_counts.values()) if relevant_edge_counts else 1
    displayed_edges: List[GraphEdge] = []
    for key, count in relevant_edge_counts.items():
        src, tgt = key
        if src not in displayed_node_ids or tgt not in displayed_node_ids:
            continue
        edge = graph.edges[key].model_copy(deep=True)
        edge.co_occurrence_count = count
        edge.weight = round(count / max_count, 4)
        displayed_edges.append(edge)

    displayed_edges.sort(key=lambda e: (-e.co_occurrence_count, e.source, e.target))

    return GraphFilterResult(
        nodes=displayed_nodes,
        edges=displayed_edges,
        total_node_count=total_node_count,
        total_edge_count=total_edge_count,
        truncated=truncated,
        message=message,
        available_documents=graph.available_document_names(),
    )

