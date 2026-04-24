"""
Unit tests for DB connection layer — SQLite init, CRUD helpers, migrations.
"""

import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.connection import (
    init_db, get_db, insert_chunk, get_chunk_by_id, upsert_document_set
)
from backend.schemas.chunk import Chunk


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    return db_path


def make_chunk_row(**overrides) -> dict:
    chunk = Chunk(
        chunk_id="a3f9c2b1_0001",
        document_name="test.pdf",
        document_id="a3f9c2b1d4e5f6a7",
        document_set_id="set-001",
        page_number=1,
        paragraph_index=0,
        text="Test chunk text for database validation.",
        entities=["testing", "database"],
        char_start=0,
        char_end=40,
        word_count=6,
        chunk_index_in_document=1,
        language="en",
        timestamp_indexed=datetime(2026, 4, 22, tzinfo=timezone.utc),
    )
    row = chunk.to_sqlite_row()
    row.update(overrides)
    return row


def test_db_init_creates_tables(tmp_db):
    with get_db() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"chunks", "document_sets", "sessions", "turns", "schema_migrations"} <= tables


def test_schema_migration_version(tmp_db):
    with get_db() as conn:
        version = conn.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0]
    assert version == 1


def _seed_doc_set(conn, set_id="set-001", name="Test Set"):
    """Insert a document_set row so FK constraints pass."""
    upsert_document_set(conn, set_id, name)
    conn.commit()


def test_insert_and_retrieve_chunk(tmp_db):
    row = make_chunk_row()
    with get_db() as conn:
        _seed_doc_set(conn)
        insert_chunk(conn, row)
        conn.commit()
        retrieved = get_chunk_by_id(conn, "a3f9c2b1_0001")

    assert retrieved is not None
    assert retrieved["chunk_id"] == "a3f9c2b1_0001"
    assert retrieved["document_name"] == "test.pdf"
    entities = json.loads(retrieved["entities_json"])
    assert "testing" in entities


def test_insert_chunk_arabic_unicode(tmp_db):
    with get_db() as conn:
        _seed_doc_set(conn)
    row = make_chunk_row(
        chunk_id="ar_test_0001",
        text="يتطلب ارتفاع ضغط الدم مراقبة دقيقة.",
        entities_json=json.dumps(["ضغط الدم", "المرضى"], ensure_ascii=False),
        language="ar",
    )
    with get_db() as conn:
        insert_chunk(conn, row)
        conn.commit()
        retrieved = get_chunk_by_id(conn, "ar_test_0001")

    assert retrieved["language"] == "ar"
    text = retrieved["text"]
    assert "ضغط" in text
    entities = json.loads(retrieved["entities_json"])
    assert "ضغط الدم" in entities


def test_insert_chunk_max_entities(tmp_db):
    """15 entities × ~40 chars each — entities_json must store all of them."""
    entities = [f"very_long_entity_name_{i:03d}" for i in range(15)]
    row = make_chunk_row(
        chunk_id="maxent_0001",
        entities_json=json.dumps(entities),
    )
    with get_db() as conn:
        _seed_doc_set(conn)
        insert_chunk(conn, row)
        conn.commit()
        retrieved = get_chunk_by_id(conn, "maxent_0001")

    entities_back = json.loads(retrieved["entities_json"])
    assert len(entities_back) == 15


def test_insert_duplicate_chunk_replaces(tmp_db):
    row = make_chunk_row(text="original text")
    row2 = make_chunk_row(text="updated text")
    with get_db() as conn:
        _seed_doc_set(conn)
        insert_chunk(conn, row)
        conn.commit()
        insert_chunk(conn, row2)
        conn.commit()
        retrieved = get_chunk_by_id(conn, "a3f9c2b1_0001")

    assert retrieved["text"] == "updated text"


def test_chroma_metadata_under_40kb():
    """Verify ChromaDB metadata dict stays well under the 40KB per-record limit."""
    chunk = Chunk(
        chunk_id="a3f9c2b1_0001",
        document_name="test.pdf",
        document_id="a3f9c2b1d4e5f6a7",
        document_set_id="set-001",
        page_number=1,
        paragraph_index=0,
        text="Some text",
        entities=[f"entity_{i}" for i in range(15)],
        char_start=0,
        char_end=9,
        word_count=2,
        chunk_index_in_document=0,
        language="en",
        timestamp_indexed=datetime(2026, 4, 22, tzinfo=timezone.utc),
    )
    meta = chunk.to_chroma_metadata()
    serialised = json.dumps(meta, ensure_ascii=False)
    assert len(serialised.encode("utf-8")) < 40_000, (
        f"ChromaDB metadata too large: {len(serialised)} bytes"
    )
