import os
from fastapi import APIRouter
from backend.db.connection import get_db
import backend.indexing.pipeline as _pipeline
from backend.monitoring.benchmark import get_benchmark_state

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    # SQLite check
    try:
        with get_db() as conn:
            sqlite_ok = conn.execute("SELECT 1").fetchone() is not None
    except Exception:
        sqlite_ok = False

    # ChromaDB check — read from module at call time (not import time)
    try:
        client = _pipeline._chroma_client
        chroma_ok = client is not None and client.heartbeat() is not None
    except Exception:
        chroma_ok = False

    return {
        "status": "ok" if (sqlite_ok and chroma_ok) else "degraded",
        "chromadb": chroma_ok,
        "sqlite": sqlite_ok,
        "llm_configured": bool(os.getenv("GEMINI_API_KEY")),
        "llm_model": os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"),
        "embedding_benchmark": get_benchmark_state(),
    }
