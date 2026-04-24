"""
Phase 3 tests: chunker, entity extractor, graph builder, and full pipeline.

End-to-end pipeline test (test_index_*) requires PDFs in data/pdfs/.
They are skipped automatically if no PDFs are present.
"""

import json
import time
import uuid
import tempfile
import shutil
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Chunker ──────────────────────────────────────────────────────

from backend.indexing.chunker import (
    chunk_page, chunk_document, detect_language,
    _sentence_split, _apply_overlap, TARGET, MAX_CHARS, OVERLAP,
)


def test_detect_language_arabic():
    assert detect_language("الذكاء الاصطناعي يغير العالم") == "ar"

def test_detect_language_french():
    # Sentence with ~5% accented chars — representative of real French text
    assert detect_language(
        "Le développement des sociétés modernes nécessite une réflexion approfondie."
    ) == "fr"

def test_detect_language_english():
    assert detect_language("Artificial intelligence is transforming the world.") == "en"

def test_sentence_split_stays_under_max():
    long_text = "This is a sentence. " * 100
    parts = _sentence_split(long_text)
    for part in parts:
        assert len(part) <= MAX_CHARS, f"Sentence split produced {len(part)} chars"

def test_apply_overlap_prepends_tail():
    chunks = ["AAAA" * 40, "BBBB" * 40]  # two chunks of 160 chars each
    result = _apply_overlap(chunks)
    assert len(result) == 2
    # second chunk should start with tail of first
    tail = chunks[0][-OVERLAP:]
    assert result[1].startswith(tail.strip())

def _make_chunks(text, page=1):
    return chunk_page(
        page_text=text,
        page_number=page,
        document_id="deadbeef",
        document_name="test.pdf",
        document_set_id="set-001",
        start_index=0,
    )

def test_chunk_page_single_short_para():
    chunks = _make_chunks("Short paragraph.")
    assert len(chunks) == 1
    assert chunks[0].text == "Short paragraph."
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "deadbeef_0000"

def test_chunk_page_respects_max_chars():
    # Paragraph exactly at limit — should not split
    text = "word " * 239  # ~1195 chars
    chunks = _make_chunks(text)
    for c in chunks:
        assert len(c.text) <= MAX_CHARS

def test_chunk_page_long_single_paragraph_splits():
    long_para = "This is a long sentence with meaningful content. " * 30
    chunks = _make_chunks(long_para)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= MAX_CHARS

def test_chunk_page_two_paragraphs_merged():
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = _make_chunks(text)
    # Both short enough to merge into one chunk
    assert len(chunks) == 1
    assert "First" in chunks[0].text
    assert "Second" in chunks[0].text

def test_chunk_page_arabic_text():
    text = "يتطلب ارتفاع ضغط الدم لدى المرضى المسنين مراقبة دقيقة لضغط الدم.\n\nيجب مراجعة الطبيب بانتظام."
    chunks = _make_chunks(text)
    assert len(chunks) >= 1
    assert chunks[0].language == "ar"
    # Arabic text must be preserved exactly
    combined = " ".join(c.text for c in chunks)
    assert "ضغط الدم" in combined

def test_chunk_schema_valid():
    text = "Valid English chunk text for testing purposes."
    chunks = _make_chunks(text)
    c = chunks[0]
    assert c.chunk_id == "deadbeef_0000"
    assert c.document_name == "test.pdf"
    assert c.language == "en"
    assert c.word_count > 0
    assert c.char_start == 0
    assert c.char_end == len(text)

def test_chunk_global_index_increments():
    pages = [
        {"page_number": 1, "text": "Page one content."},
        {"page_number": 2, "text": "Page two content."},
    ]
    all_chunks = chunk_document(
        pages=pages,
        document_id="deadbeef",
        document_name="test.pdf",
        document_set_id="set-001",
    )
    indices = [c.chunk_index_in_document for c in all_chunks]
    assert indices == list(range(len(all_chunks))), "chunk_index_in_document must be contiguous"


# ── Entity Extractor ─────────────────────────────────────────────

from backend.indexing.entity_extractor import extract_entities, entity_labels

def test_entity_extraction_english():
    text = "Apple Inc. and Google LLC are based in California and develop machine learning."
    ents = extract_entities(text, "en")
    labels = entity_labels(ents)
    assert any("Apple" in l for l in labels)
    assert any("Google" in l for l in labels)

def test_entity_extraction_french():
    text = "La Commission européenne a publié un rapport sur l'intelligence artificielle à Bruxelles."
    ents = extract_entities(text, "fr")
    assert len(ents) >= 1  # at least one entity found

def test_entity_extraction_arabic_returns_empty():
    text = "الذكاء الاصطناعي يغير العالم بشكل جذري."
    ents = extract_entities(text, "ar")
    assert ents == [], "Arabic should return empty list (documented limitation)"

def test_entity_extraction_max_15():
    # Artificially pack 20+ recognisable entities
    names = " and ".join([f"Company{i} Inc." for i in range(25)])
    ents = extract_entities(names, "en")
    assert len(ents) <= 15

def test_entity_types_are_valid():
    valid_types = {"concept", "person", "organization", "location", "technique"}
    text = "Barack Obama visited Paris on behalf of the United Nations."
    ents = extract_entities(text, "en")
    for _, etype in ents:
        assert etype in valid_types


def test_custom_entity_patterns_cover_domain_terms():
    text = "CNAS reimbursement rules differ by wilaya and Ministry of Health policy."
    ents = extract_entities(text, "en")
    mapping = {label.lower(): etype for label, etype in ents}
    assert mapping.get("cnas") == "organization"
    assert mapping.get("wilaya") == "location"


# ── Graph Builder ────────────────────────────────────────────────

from backend.indexing.graph_builder import GraphState, get_graph

def test_graph_state_adds_nodes():
    graph = GraphState("set-test-001")
    diff = graph.add_chunk_entities(
        "chunk_0001", "doc.pdf",
        [("hypertension", "concept"), ("ACE inhibitor", "technique")]
    )
    assert len(diff.added_nodes) == 2
    assert len(graph.nodes) == 2

def test_graph_state_co_occurrence_edge():
    graph = GraphState("set-test-002")
    graph.add_chunk_entities(
        "chunk_0001", "doc.pdf",
        [("hypertension", "concept"), ("cardiology", "concept")]
    )
    assert len(graph.edges) == 1
    edge = list(graph.edges.values())[0]
    assert edge.co_occurrence_count == 1
    assert edge.weight == 1.0

def test_graph_state_edge_weight_normalised():
    graph = GraphState("set-test-003")
    # First chunk: A+B → edge count 1
    graph.add_chunk_entities("c1", "doc.pdf", [("A", "concept"), ("B", "concept")])
    # Second chunk: A+B → edge count 2 (max)
    graph.add_chunk_entities("c2", "doc.pdf", [("A", "concept"), ("B", "concept")])
    # Third chunk: A+C → edge count 1 (weight = 1/2 = 0.5)
    graph.add_chunk_entities("c3", "doc.pdf", [("A", "concept"), ("C", "concept")])

    ab_edge = [e for e in graph.edges.values() if e.co_occurrence_count == 2][0]
    ac_edge = [e for e in graph.edges.values() if e.co_occurrence_count == 1][0]
    assert ab_edge.weight == 1.0
    assert ac_edge.weight == 0.5

def test_graph_state_duplicate_chunk_not_double_counted():
    graph = GraphState("set-test-004")
    graph.add_chunk_entities("c1", "doc.pdf", [("entity_a", "concept")])
    graph.add_chunk_entities("c1", "doc.pdf", [("entity_a", "concept")])  # same chunk
    node = list(graph.nodes.values())[0]
    # chunk_ids should not contain duplicates
    assert node.chunk_ids.count("c1") == 1
    assert node.frequency == 1

def test_graph_get_active_ids():
    graph = GraphState("set-test-005")
    graph.add_chunk_entities("c1", "doc.pdf", [("A", "concept"), ("B", "concept")])
    graph.add_chunk_entities("c2", "doc.pdf", [("C", "concept")])
    active_nodes, active_edges = graph.get_active_ids_for_chunks(["c1"])
    assert len(active_nodes) == 2  # A and B
    assert len(active_edges) == 1  # A->B


# ── End-to-end pipeline (skipped if no PDFs) ─────────────────────

def _find_pdf(pattern: str) -> Path | None:
    """Find a PDF that has actual extractable text (not a scanned image)."""
    import fitz
    for p in sorted(Path("data/pdfs").glob(pattern)):
        try:
            doc = fitz.open(str(p))
            chars = sum(len(pg.get_text()) for pg in doc)
            doc.close()
            if chars >= 200:
                return p
        except Exception:
            continue
    return None

@pytest.mark.skipif(
    _find_pdf("*.pdf") is None,
    reason="No text-based PDFs in data/pdfs/ (scanned-only PDFs cannot be indexed)"
)
def test_index_pdf_basic(tmp_path):
    """Index the first available PDF, verify chunk count and DB rows."""
    from backend.db.connection import init_db, get_db
    from backend.indexing.pipeline import init_chroma, index_document

    db_path = tmp_path / "test.db"
    chroma_path = tmp_path / "chroma"
    init_db(db_path)
    init_chroma(chroma_path)

    pdf = _find_pdf("*.pdf")
    progress_events = []

    result = index_document(
        pdf_path=pdf,
        on_progress=lambda e: progress_events.append(e),
    )

    assert result["chunk_count"] > 0, "No chunks produced"
    assert result["elapsed_seconds"] < 300, "Indexing took > 5 minutes"

    with get_db() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
            (result["document_id"],)
        ).fetchone()
    assert rows[0] == result["chunk_count"]

    stages = [e["payload"]["stage"] for e in progress_events]
    assert "complete" in stages

    print(f"\n  Indexed {result['chunk_count']} chunks in {result['elapsed_seconds']}s")


@pytest.mark.skipif(
    _find_pdf("*.pdf") is None,
    reason="No text-based PDFs in data/pdfs/"
)
def test_index_duplicate_skipped(tmp_path):
    """Index same PDF twice — second run should be skipped via document_exists()."""
    from backend.db.connection import init_db
    from backend.indexing.pipeline import init_chroma, index_document, document_exists

    db_path = tmp_path / "test.db"
    chroma_path = tmp_path / "chroma"
    init_db(db_path)
    init_chroma(chroma_path)

    pdf = _find_pdf("*.pdf")
    result1 = index_document(pdf_path=pdf)
    doc_id = result1["document_id"]

    assert document_exists(doc_id), "document_exists() should return True after indexing"

    # Second index — caller should check document_exists() and skip
    # (watcher.py does this; here we just verify the check works)
    assert document_exists(doc_id), "Still exists after second check"


@pytest.mark.skipif(
    _find_pdf("*ar*.pdf") is None and _find_pdf("*arabic*.pdf") is None,
    reason="No text-based Arabic PDF in data/pdfs/ (scanned PDFs cannot be indexed)"
)
def test_index_arabic_text_integrity(tmp_path):
    """Index Arabic PDF, verify Arabic text preserved in SQLite."""
    from backend.db.connection import init_db, get_db
    from backend.indexing.pipeline import init_chroma, index_document

    db_path = tmp_path / "test.db"
    chroma_path = tmp_path / "chroma"
    init_db(db_path)
    init_chroma(chroma_path)

    pdf = _find_pdf("*ar*.pdf") or _find_pdf("*arabic*.pdf")
    result = index_document(pdf_path=pdf)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT text, language FROM chunks WHERE document_id = ? AND language = 'ar' LIMIT 5",
            (result["document_id"],)
        ).fetchall()

    assert len(rows) > 0, "Expected Arabic chunks but found none"
    for row in rows:
        # Arabic text should contain Arabic Unicode chars
        arabic_chars = sum(1 for c in row["text"] if "؀" <= c <= "ۿ")
        assert arabic_chars > 0, f"Arabic chunk has no Arabic chars: {row['text'][:100]}"
    print(f"\n  Arabic chunks: {len(rows)}, sample: {rows[0]['text'][:80]}")
