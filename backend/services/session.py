"""
Session lifecycle helpers.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from backend.errors import AppError, ErrorCode
from backend.schemas.conversation import Session

DEFAULT_RETENTION_DAYS = 30


def get_session(conn: sqlite3.Connection, session_id: str) -> Session | None:
    row = conn.execute(
        """
        SELECT id, document_set_id, created_at, last_activity
        FROM sessions
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if not row:
        return None

    return Session(
        id=row["id"],
        document_set_id=row["document_set_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        last_activity=datetime.fromisoformat(row["last_activity"]),
    )


def ensure_session(
    conn: sqlite3.Connection,
    session_id: str,
    document_set_id: str,
) -> Session:
    existing = get_session(conn, session_id)

    if existing and existing.document_set_id != document_set_id:
        raise AppError(
            ErrorCode.SESSION_SCOPE_MISMATCH,
            "This session belongs to another document set. Start a fresh session for the new set.",
            recoverable=True,
            status_code=409,
        )

    conn.execute(
        """
        INSERT OR IGNORE INTO sessions (id, document_set_id)
        VALUES (?, ?)
        """,
        (session_id, document_set_id),
    )
    touch_session(conn, session_id)

    current = get_session(conn, session_id)
    if current is None:
        raise AppError(
            ErrorCode.SESSION_SCOPE_MISMATCH,
            "Failed to create chat session.",
            recoverable=False,
            status_code=500,
        )
    return current


def touch_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        """
        UPDATE sessions
        SET last_activity = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (session_id,),
    )


def delete_session(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    turns_deleted = conn.execute(
        "DELETE FROM turns WHERE session_id = ?",
        (session_id,),
    ).rowcount
    sessions_deleted = conn.execute(
        "DELETE FROM sessions WHERE id = ?",
        (session_id,),
    ).rowcount
    return {
        "turns_deleted": turns_deleted,
        "sessions_deleted": sessions_deleted,
    }


def cleanup_expired_sessions(
    conn: sqlite3.Connection,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> dict[str, int]:
    turns_deleted = conn.execute(
        """
        DELETE FROM turns
        WHERE session_id IN (
            SELECT id FROM sessions
            WHERE last_activity < datetime('now', ?)
        )
        """,
        (f"-{retention_days} days",),
    ).rowcount
    sessions_deleted = conn.execute(
        """
        DELETE FROM sessions
        WHERE last_activity < datetime('now', ?)
        """,
        (f"-{retention_days} days",),
    ).rowcount
    return {
        "turns_deleted": turns_deleted,
        "sessions_deleted": sessions_deleted,
    }
