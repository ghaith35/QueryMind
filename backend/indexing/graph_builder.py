"""
Co-occurrence graph builder.

Maintains an in-memory graph per document_set_id. When chunks are indexed,
entities are extracted and co-occurrence edges are formed between entities
that appear in the same chunk.

This is NOT stored persistently — the graph is rebuilt from the chunks table
on startup (fast: ~1s for 800 chunks). During indexing, it is updated
incrementally and diffs are sent over WebSocket.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from backend.indexing.entity_extractor import parse_stored_entities
from backend.schemas.graph import GraphNode, GraphEdge, GraphDiff, node_id_from_label


class GraphState:
    """In-memory graph for one document set."""

    def __init__(self, document_set_id: str):
        self.document_set_id = document_set_id
        self.nodes: Dict[str, GraphNode] = {}  # node_id → GraphNode
        self.edges: Dict[Tuple[str, str], GraphEdge] = {}  # (src, tgt) → GraphEdge
        self._max_co_occurrence = 1  # for weight normalisation
        self.chunk_documents: Dict[str, str] = {}  # chunk_id -> document_name
        self.edge_chunk_ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.edge_document_sources: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def add_chunk_entities(
        self,
        chunk_id: str,
        document_name: str,
        entities: List[Tuple[str, str]],  # [(label, type), ...]
    ) -> GraphDiff:
        """
        Process entities from one chunk. Returns a GraphDiff containing
        only what changed (new nodes, new edges, updated frequencies).
        """
        diff = GraphDiff(document_set_id=self.document_set_id)

        # Upsert nodes
        entity_ids: List[str] = []
        self.chunk_documents[chunk_id] = document_name
        for label, etype in entities:
            node_id = node_id_from_label(label)
            is_new = node_id not in self.nodes

            if is_new:
                node = GraphNode.from_label(label, etype)  # type: ignore[arg-type]
                self.nodes[node_id] = node
                diff.added_nodes.append(node)

            node = self.nodes[node_id]
            node.add_chunk(chunk_id, document_name)

            if not is_new:
                diff.updated_nodes.append({"id": node_id, "frequency": node.frequency})

            entity_ids.append(node_id)

        # Upsert co-occurrence edges for every pair in this chunk
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                src, tgt = sorted([entity_ids[i], entity_ids[j]])
                key = (src, tgt)
                is_new_edge = key not in self.edges

                if is_new_edge:
                    edge = GraphEdge(source=src, target=tgt)
                    self.edges[key] = edge

                edge = self.edges[key]
                if chunk_id in self.edge_chunk_ids[key]:
                    continue
                edge.increment()
                self.edge_chunk_ids[key].add(chunk_id)
                self.edge_document_sources[key].add(document_name)

                if edge.co_occurrence_count > self._max_co_occurrence:
                    self._max_co_occurrence = edge.co_occurrence_count

                # Renormalise this edge's weight
                edge.weight = round(
                    edge.co_occurrence_count / self._max_co_occurrence, 4
                )

                if is_new_edge:
                    diff.added_edges.append(edge)

        return diff

    def available_document_names(self) -> List[str]:
        return sorted({name for name in self.chunk_documents.values() if name})

    def get_active_ids_for_chunks(
        self, chunk_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Given retrieved chunk_ids, return node_ids and edge_ids that
        contributed to the answer (for graph highlighting).
        """
        chunk_set = set(chunk_ids)
        active_nodes = [
            node.id
            for node in self.nodes.values()
            if any(cid in chunk_set for cid in node.chunk_ids)
        ]
        active_edges = [
            edge.edge_id
            for key, edge in self.edges.items()
            if any(cid in chunk_set for cid in self.edge_chunk_ids.get(key, set()))
        ]
        return active_nodes, active_edges


# Global registry: document_set_id → GraphState
_graphs: Dict[str, GraphState] = {}


def get_graph(document_set_id: str) -> GraphState:
    if document_set_id not in _graphs:
        _graphs[document_set_id] = GraphState(document_set_id)
    return _graphs[document_set_id]


def rebuild_graph_from_chunks(document_set_id: str, chunk_rows: list) -> GraphState:
    """
    Rebuild GraphState from SQLite chunk rows on startup.
    chunk_rows: list of sqlite3.Row objects from get_chunks_by_doc_set().
    """
    graph = GraphState(document_set_id)
    _graphs[document_set_id] = graph

    for row in chunk_rows:
        entities_raw = parse_stored_entities(row["entities_json"])
        lang = row["language"]
        if lang in ("en", "fr") and entities_raw:
            graph.add_chunk_entities(
                chunk_id=row["chunk_id"],
                document_name=row["document_name"],
                entities=entities_raw,
            )

    return graph
