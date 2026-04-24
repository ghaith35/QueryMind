"""
Conversation-memory helpers.
"""

from __future__ import annotations

import os

from backend.db.connection import get_db
from backend.schemas.conversation import Turn

MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))


def estimate_tokens(text: str) -> int:
    arabic_chars = sum(1 for char in text if "\u0600" <= char <= "\u06FF")
    latin_chars = max(0, len(text) - arabic_chars)
    return max(1, latin_chars // 4 + arabic_chars // 2)


def build_history_window(
    session_id: str,
    max_history_tokens: int = MAX_HISTORY_TOKENS,
) -> list[Turn]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM turns
            WHERE session_id = ?
            ORDER BY timestamp DESC
            """,
            (session_id,),
        ).fetchall()

    selected: list[Turn] = []
    budget = max_history_tokens

    for row in rows:
        turn = Turn.from_row(dict(row))
        estimated = estimate_tokens(turn.content)
        if estimated > budget:
            break
        selected.append(turn)
        budget -= estimated

    selected.reverse()
    return selected


def serialise_history(turns: list[Turn]) -> list[dict[str, str]]:
    return [{"role": turn.role, "content": turn.content} for turn in turns]
