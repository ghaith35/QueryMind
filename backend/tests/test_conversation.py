import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.connection import get_db, init_db, insert_turn, upsert_document_set
from backend.errors import AppError
from backend.schemas.conversation import Turn
from backend.services.conversation import build_history_window, estimate_tokens
from backend.services.session import cleanup_expired_sessions, delete_session, ensure_session


@pytest.fixture
def conversation_db(tmp_path):
    init_db(tmp_path / "conversation.db")
    with get_db() as conn:
        upsert_document_set(conn, "set-001", "Conversation Set")
        upsert_document_set(conn, "set-002", "Another Set")
        conn.commit()
    return tmp_path


def _insert_turn(session_id: str, role: str, content: str) -> None:
    turn = Turn(
        turn_id=f"{session_id}-{role}-{len(content)}",
        session_id=session_id,
        role=role,
        content=content,
        citations=[],
        retrieved_chunk_ids=[],
        timestamp=datetime.now(timezone.utc),
    )
    with get_db() as conn:
        insert_turn(conn, turn.to_sqlite_row())
        conn.commit()


def test_estimate_tokens_counts_arabic_more_aggressively():
    english = estimate_tokens("This is a short English sentence.")
    arabic = estimate_tokens("هذه جملة عربية قصيرة للاختبار")
    assert english > 0
    assert arabic > 0
    assert arabic >= english - 2


def test_build_history_window_respects_token_budget(conversation_db):
    with get_db() as conn:
        ensure_session(conn, "session-001", "set-001")
        conn.commit()

    _insert_turn("session-001", "user", "short question")
    _insert_turn("session-001", "assistant", "x" * 2000)
    _insert_turn("session-001", "user", "latest question")

    window = build_history_window("session-001", max_history_tokens=50)

    assert [turn.content for turn in window] == ["latest question"]


def test_ensure_session_rejects_cross_set_scope(conversation_db):
    with get_db() as conn:
        ensure_session(conn, "session-002", "set-001")
        conn.commit()

    with get_db() as conn:
        with pytest.raises(AppError):
            ensure_session(conn, "session-002", "set-002")


def test_delete_session_removes_turns(conversation_db):
    with get_db() as conn:
        ensure_session(conn, "session-003", "set-001")
        conn.commit()

    _insert_turn("session-003", "user", "hello")

    with get_db() as conn:
        deleted = delete_session(conn, "session-003")
        conn.commit()

    assert deleted["turns_deleted"] == 1
    assert deleted["sessions_deleted"] == 1


def test_cleanup_expired_sessions_removes_old_rows(conversation_db):
    with get_db() as conn:
        ensure_session(conn, "session-004", "set-001")
        conn.execute(
            """
            UPDATE sessions
            SET last_activity = datetime('now', '-31 days')
            WHERE id = ?
            """,
            ("session-004",),
        )
        conn.commit()

    _insert_turn("session-004", "user", "stale turn")

    with get_db() as conn:
        cleanup = cleanup_expired_sessions(conn, retention_days=30)
        conn.commit()
        row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", ("session-004",)).fetchone()

    assert cleanup["sessions_deleted"] == 1
    assert cleanup["turns_deleted"] == 1
    assert row is None
