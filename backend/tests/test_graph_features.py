import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.graph.filters import apply_graph_filters
from backend.indexing.graph_builder import GraphState, rebuild_graph_from_chunks
from backend.schemas.graph import node_id_from_label


def test_graph_filters_recompute_document_frequency_and_edges():
    graph = GraphState("set-graph-001")
    graph.add_chunk_entities(
        "chunk-1",
        "alpha.pdf",
        [("Alice", "person"), ("Paris", "location"), ("IELTS", "concept")],
    )
    graph.add_chunk_entities(
        "chunk-2",
        "beta.pdf",
        [("Alice", "person"), ("London", "location")],
    )

    result = apply_graph_filters(
        graph,
        min_frequency=1,
        entity_types=["person", "location"],
        document_names=["alpha.pdf"],
    )

    assert {node.label: node.frequency for node in result.nodes} == {
        "Alice": 1,
        "Paris": 1,
    }
    assert {
        frozenset((edge.source, edge.target))
        for edge in result.edges
    } == {
        frozenset((node_id_from_label("Alice"), node_id_from_label("Paris")))
    }
    assert result.available_documents == ["alpha.pdf", "beta.pdf"]


def test_graph_filters_truncate_large_graph():
    graph = GraphState("set-graph-002")
    for index in range(501):
        graph.add_chunk_entities(
            f"chunk-{index}",
            "big.pdf",
            [(f"Entity {index}", "concept")],
        )

    result = apply_graph_filters(graph, min_frequency=1)

    assert result.truncated is True
    assert len(result.nodes) == 300
    assert result.total_node_count == 501
    assert result.message is not None


def test_rebuild_graph_from_chunks_preserves_entity_types():
    rows = [
        {
            "chunk_id": "chunk-1",
            "document_name": "typed.pdf",
            "entities_json": json.dumps(
                [
                    {"label": "Alice", "type": "person"},
                    {"label": "Paris", "type": "location"},
                ]
            ),
            "language": "en",
        }
    ]

    graph = rebuild_graph_from_chunks("set-graph-003", rows)

    assert graph.nodes[node_id_from_label("Alice")].type == "person"
    assert graph.nodes[node_id_from_label("Paris")].type == "location"


def test_active_edges_only_include_retrieved_chunks():
    graph = GraphState("set-graph-004")
    graph.add_chunk_entities("chunk-1", "doc.pdf", [("A", "concept"), ("B", "concept")])
    graph.add_chunk_entities("chunk-2", "doc.pdf", [("B", "concept"), ("C", "concept")])
    graph.add_chunk_entities("chunk-3", "doc.pdf", [("A", "concept"), ("C", "concept")])

    active_nodes, active_edges = graph.get_active_ids_for_chunks(["chunk-1", "chunk-2"])

    assert set(active_nodes) == {
        node_id_from_label("A"),
        node_id_from_label("B"),
        node_id_from_label("C"),
    }
    expected_edges = {
        "->".join(sorted((node_id_from_label("A"), node_id_from_label("B")))),
        "->".join(sorted((node_id_from_label("B"), node_id_from_label("C")))),
    }
    assert set(active_edges) == expected_edges


def test_duplicate_chunk_entities_do_not_double_count_edges():
    graph = GraphState("set-graph-005")
    graph.add_chunk_entities("chunk-1", "doc.pdf", [("A", "concept"), ("B", "concept")])
    graph.add_chunk_entities("chunk-1", "doc.pdf", [("A", "concept"), ("B", "concept")])

    edge = next(iter(graph.edges.values()))
    assert edge.co_occurrence_count == 1
