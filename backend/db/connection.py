"""
SQLite connection pool + migration runner.

Uses a thread-local connection per worker (FastAPI + uvicorn run in a
threadpool for sync DB calls). WAL mode is set once at DB creation so
concurrent reads don't block writes from the indexing pipeline.
"""

import sqlite3
import threading
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

log = logging.getLogger(__name__)

_DB_PATH: Path | None = None
_local = threading.local()

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
CURRENT_SCHEMA_VERSION = 1


def init_db(db_path: str | Path) -> None:
    """Call once at startup. Creates the DB file and runs migrations."""
    global _DB_PATH
    new_path = Path(db_path)
    new_path.parent.mkdir(parents=True, exist_ok=True)

    # If path changed (e.g. in tests), close the old thread-local connection
    if _DB_PATH != new_path:
        close_thread_connection()

    _DB_PATH = new_path

    with _connect(_DB_PATH) as conn:
        _run_migrations(conn)

    log.info("SQLite initialised at %s", _DB_PATH)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, faster than FULL
    return conn


def _run_migrations(conn: sqlite3.Connection) -> None:
    schema_sql = SCHEMA_PATH.read_text()
    conn.executescript(schema_sql)
    conn.commit()

    version = conn.execute(
        "SELECT MAX(version) FROM schema_migrations"
    ).fetchone()[0]
    log.info("DB schema version: %s", version)


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Yields a thread-local SQLite connection.
    Use as a FastAPI dependency or plain context manager.
    """
    if _DB_PATH is None:
        raise RuntimeError("init_db() must be called before get_db()")

    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = _connect(_DB_PATH)

    try:
        yield _local.conn
    except Exception:
        _local.conn.rollback()
        raise


def close_thread_connection() -> None:
    """Call in thread cleanup (e.g. FastAPI shutdown) to close thread-local conn."""
    if hasattr(_local, "conn") and _local.conn:
        _local.conn.close()
        _local.conn = None


# ── Convenience helpers ───────────────────────────────────────────

def insert_chunk(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO chunks
            (chunk_id, document_set_id, document_id, document_name,
             page_number, paragraph_index, text, entities_json,
             char_start, char_end, word_count, chunk_index_in_document,
             language, timestamp_indexed)
        VALUES
            (:chunk_id, :document_set_id, :document_id, :document_name,
             :page_number, :paragraph_index, :text, :entities_json,
             :char_start, :char_end, :word_count, :chunk_index_in_document,
             :language, :timestamp_indexed)
        """,
        row,
    )


def get_chunk_by_id(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()


def get_chunks_by_doc_set(
    conn: sqlite3.Connection, document_set_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE document_set_id = ? ORDER BY chunk_index_in_document",
        (document_set_id,),
    ).fetchall()


def upsert_document_set(
    conn: sqlite3.Connection, set_id: str, name: str
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO document_sets (id, name) VALUES (?, ?)",
        (set_id, name),
    )


def increment_chunk_count(
    conn: sqlite3.Connection, document_set_id: str, delta: int = 1
) -> None:
    conn.execute(
        "UPDATE document_sets SET chunk_count = chunk_count + ? WHERE id = ?",
        (delta, document_set_id),
    )


def insert_turn(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO turns
            (turn_id, session_id, role, content,
             citations_json, retrieved_chunk_ids_json, timestamp)
        VALUES
            (:turn_id, :session_id, :role, :content,
             :citations_json, :retrieved_chunk_ids_json, :timestamp)
        """,
        row,
    )


def get_session_turns(
    conn: sqlite3.Connection, session_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
    ).fetchall()
