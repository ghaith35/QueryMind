"""
Unit tests for all Pydantic schemas — serialize/deserialize round-trips
and constraint validation.
"""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.schemas.chunk import Chunk, Citation
from backend.schemas.graph import GraphNode, GraphEdge, GraphDiff, node_id_from_label
from backend.schemas.conversation import Turn
from backend.schemas.websocket import (
    WSMessage,
    IndexProgressPayload,
    GraphUpdatePayload,
    AnswerStreamPayload,
    AnswerCompletePayload,
    ErrorPayload,
    index_progress_msg,
    graph_update_msg,
    answer_complete_msg,
    error_msg,
)


# ── Chunk ─────────────────────────────────────────────────────────

def make_chunk(**overrides) -> Chunk:
    base = dict(
        chunk_id="a3f9c2b1_0042",
        document_name="cardiology_guidelines_2024.pdf",
        document_id="a3f9c2b1d4e5f6a7",
        document_set_id="550e8400-e29b-41d4-a716-446655440000",
        page_number=12,
        paragraph_index=3,
        text="Hypertension management in adults over 60 requires careful monitoring.",
        entities=["hypertension", "cardiology", "ACE inhibitor"],
        char_start=450,
        char_end=520,
        word_count=11,
        chunk_index_in_document=42,
        language="en",
        timestamp_indexed=datetime(2026, 4, 22, 10, 30, 0, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return Chunk(**base)


def test_chunk_round_trip_json():
    chunk = make_chunk()
    serialised = chunk.model_dump_json()
    restored = Chunk.model_validate_json(serialised)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.entities == chunk.entities
    assert restored.language == chunk.language


def test_chunk_text_max_length():
    with pytest.raises(Exception):
        make_chunk(text="x" * 1201)


def test_chunk_entities_truncated_to_15():
    chunk = make_chunk(entities=[f"entity_{i}" for i in range(20)])
    assert len(chunk.entities) == 15


def test_chunk_to_chroma_metadata():
    chunk = make_chunk(entities=["hypertension", "cardiology"])
    meta = chunk.to_chroma_metadata()
    assert set(meta.keys()) == {
        "chunk_id", "document_id", "document_set_id", "page_number",
        "language", "entities_csv",
    }
    assert "text" not in meta
    assert "hypertension" in meta["entities_csv"]


def test_chunk_chroma_metadata_entities_csv_max_500():
    # 15 entities each ~40 chars → potential overflow
    long_entities = [f"very_long_entity_name_number_{i:03d}" for i in range(15)]
    chunk = make_chunk(entities=long_entities)
    meta = chunk.to_chroma_metadata()
    assert len(meta["entities_csv"]) <= 500


def test_chunk_to_sqlite_row():
    chunk = make_chunk()
    row = chunk.to_sqlite_row()
    entities_back = json.loads(row["entities_json"])
    assert entities_back == chunk.entities
    assert row["chunk_id"] == chunk.chunk_id


def test_chunk_arabic():
    chunk = make_chunk(
        text="يتطلب ارتفاع ضغط الدم لدى المرضى مراقبة دقيقة.",
        language="ar",
        entities=["ضغط الدم", "المرضى"],
    )
    serialised = chunk.model_dump_json()
    restored = Chunk.model_validate_json(serialised)
    assert "ضغط الدم" in restored.entities


# ── Citation ──────────────────────────────────────────────────────

def test_citation_round_trip():
    c = Citation(
        document_name="cardiology.pdf",
        page_number=12,
        paragraph_index=3,
        chunk_id="a3f9c2b1_0042",
        relevance_score=0.923456,
        excerpt="Hypertension management in adults over 60…",
    )
    restored = Citation.model_validate_json(c.model_dump_json())
    assert restored.chunk_id == c.chunk_id
    assert restored.relevance_score == round(0.923456, 4)


def test_citation_score_out_of_range():
    with pytest.raises(Exception):
        Citation(
            document_name="x.pdf", page_number=1, paragraph_index=0,
            chunk_id="abc_0001", relevance_score=1.5, excerpt="text",
        )


def test_citation_excerpt_truncated():
    long_text = "word " * 100
    c = Citation(
        document_name="x.pdf", page_number=1, paragraph_index=0,
        chunk_id="abc_0001", relevance_score=0.8, excerpt=long_text,
    )
    assert len(c.excerpt) <= 150


# ── GraphNode ────────────────────────────────────────────────────

def test_graph_node_id_stable():
    id1 = node_id_from_label("Hypertension")
    id2 = node_id_from_label("hypertension")
    id3 = node_id_from_label("HYPERTENSION")
    assert id1 == id2 == id3
    assert len(id1) == 12


def test_graph_node_from_label():
    node = GraphNode.from_label("ACE inhibitor", "technique")
    assert node.label == "ACE inhibitor"
    assert node.type == "technique"
    assert len(node.id) == 12


def test_graph_node_add_chunk():
    node = GraphNode.from_label("hypertension")
    node.add_chunk("a3f9c2b1_0001", "cardiology.pdf")
    node.add_chunk("a3f9c2b1_0002", "cardiology.pdf")
    node.add_chunk("a3f9c2b1_0001", "cardiology.pdf")  # duplicate — ignored
    assert node.frequency == 2
    assert len(node.document_sources) == 1


def test_graph_node_round_trip():
    node = GraphNode.from_label("machine learning", "technique")
    node.add_chunk("abc_0001", "ml_paper.pdf")
    restored = GraphNode.model_validate_json(node.model_dump_json())
    assert restored.id == node.id
    assert restored.frequency == node.frequency


# ── GraphEdge ────────────────────────────────────────────────────

def test_graph_edge_id():
    edge = GraphEdge(source="aaa111", target="bbb222")
    assert edge.edge_id == "aaa111->bbb222"


def test_graph_edge_weight_clamped():
    edge = GraphEdge(source="a", target="b", weight=1.5)
    assert edge.weight == 1.0
    edge2 = GraphEdge(source="a", target="b", weight=-0.3)
    assert edge2.weight == 0.0


def test_graph_edge_increment():
    edge = GraphEdge(source="a", target="b")
    edge.increment()
    edge.increment()
    assert edge.co_occurrence_count == 2


# ── GraphDiff ────────────────────────────────────────────────────

def test_graph_diff_empty():
    diff = GraphDiff(document_set_id="set-001")
    assert diff.is_empty()


def test_graph_diff_not_empty():
    diff = GraphDiff(
        document_set_id="set-001",
        added_nodes=[GraphNode.from_label("test")],
    )
    assert not diff.is_empty()


# ── Turn ─────────────────────────────────────────────────────────

def test_turn_sqlite_row_citations_json():
    citation = Citation(
        document_name="doc.pdf",
        page_number=1,
        paragraph_index=0,
        chunk_id="abc_0001",
        relevance_score=0.9,
        excerpt="some text",
    )
    turn = Turn(
        turn_id="t-001",
        session_id="s-001",
        role="assistant",
        content="The answer is...",
        citations=[citation],
        retrieved_chunk_ids=["abc_0001"],
    )
    row = turn.to_sqlite_row()
    citations_back = json.loads(row["citations_json"])
    assert citations_back[0]["chunk_id"] == "abc_0001"
    ids_back = json.loads(row["retrieved_chunk_ids_json"])
    assert ids_back == ["abc_0001"]


# ── WebSocket messages ────────────────────────────────────────────

def test_index_progress_msg():
    msg = index_progress_msg(IndexProgressPayload(
        job_id="job-001",
        document_name="paper.pdf",
        chunks_processed=20,
        total_chunks_estimated=100,
        stage="embedding",
        percent=20.0,
    ))
    assert msg.type == "index_progress"
    data = json.loads(msg.model_dump_json())
    assert data["payload"]["stage"] == "embedding"
    assert "timestamp" in data


def test_graph_update_msg_serialises_nodes():
    node = GraphNode.from_label("neural network", "technique")
    msg = graph_update_msg(GraphUpdatePayload(
        document_set_id="set-001",
        added_nodes=[node],
    ))
    data = json.loads(msg.model_dump_json())
    assert data["type"] == "graph_update"
    assert data["payload"]["added_nodes"][0]["label"] == "neural network"


def test_answer_complete_msg():
    citation = Citation(
        document_name="doc.pdf", page_number=5, paragraph_index=2,
        chunk_id="abc_0005", relevance_score=0.91, excerpt="excerpt text",
    )
    msg = answer_complete_msg(AnswerCompletePayload(
        session_id="s-001",
        turn_id="t-002",
        full_answer="The answer is X.",
        citations=[citation],
        active_node_ids=["node123"],
        active_edge_ids=["node123->node456"],
    ))
    data = json.loads(msg.model_dump_json())
    assert data["payload"]["citations"][0]["chunk_id"] == "abc_0005"
    assert data["payload"]["active_edge_ids"] == ["node123->node456"]


def test_error_msg():
    msg = error_msg(ErrorPayload(
        code="CITATION_HALLUCINATED",
        message="Cited chunk not in retrieved set",
        recoverable=True,
    ))
    data = json.loads(msg.model_dump_json())
    assert data["payload"]["code"] == "CITATION_HALLUCINATED"
    assert data["payload"]["recoverable"] is True


def test_ws_message_has_iso_timestamp():
    msg = error_msg(ErrorPayload(
        code="LLM_FAILED", message="timeout", recoverable=False
    ))
    # must parse as ISO 8601 without raising
    datetime.fromisoformat(msg.timestamp.replace("Z", "+00:00"))
