"""
Typed application errors used across routes, indexing, and WebSocket recovery.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    INDEX_FAILED = "INDEX_FAILED"
    LLM_FAILED = "LLM_FAILED"
    CITATION_HALLUCINATED = "CITATION_HALLUCINATED"
    CITATION_MISSING = "CITATION_MISSING"
    INVALID_PDF = "INVALID_PDF"
    DUPLICATE_DOCUMENT = "DUPLICATE_DOCUMENT"
    SESSION_SCOPE_MISMATCH = "SESSION_SCOPE_MISMATCH"
    GRAPH_LOAD_FAILED = "GRAPH_LOAD_FAILED"


class AppError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        recoverable: bool = True,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable
        self.status_code = status_code


def parse_prefixed_error(message: str, *, default_code: ErrorCode) -> AppError:
    """
    Convert legacy `CODE: message` strings into structured errors.
    """
    prefix, sep, remainder = message.partition(":")
    if not sep:
        return AppError(default_code, message)

    normalized = prefix.strip().upper()
    for code in ErrorCode:
        if code.value == normalized:
            return AppError(code, remainder.strip() or message)

    return AppError(default_code, message)
