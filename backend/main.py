"""
QueryMind FastAPI application.

Startup order:
    1. Init SQLite (create tables if needed)
    2. Init ChromaDB (persistent client)
    3. Set event loop on WSManager (for thread-safe sends from pipeline)
    4. Scan for orphaned PDFs and re-queue
    5. Register all routes
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    dotenv_path=BASE_DIR / ".env",
    override=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
def _resolve_env_path(name: str, default: str) -> str:
    raw = os.getenv(name, default)
    path = Path(raw)
    if not path.is_absolute():
        path = (BASE_DIR / raw).resolve()
    return str(path)


DB_PATH = _resolve_env_path("SQLITE_PATH", "./data/sqlite/querymind.db")
CHROMA_PATH = _resolve_env_path("CHROMA_PERSIST_DIR", "./data/chroma")
PDF_DIR = _resolve_env_path("PDF_WATCH_DIR", "./data/pdfs")
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://ghaith.com",
]


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────
    from backend.db.connection import init_db
    from backend.indexing.pipeline import init_chroma
    from backend.routes.documents import _run_indexing
    from backend.services.session import cleanup_expired_sessions
    from backend.services.orphan_scanner import scan_and_requeue
    from backend.ws.manager import ws_manager
    from backend.indexing.embedder import load_model as load_embedder
    from backend.indexing.entity_extractor import _load_models as load_ner_models
    from backend.monitoring.benchmark import benchmark_embedding
    from backend.db.connection import get_db

    log.info("QueryMind starting up…")
    log.info("Configured LLM model: %s", os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"))

    init_db(DB_PATH)
    with get_db() as conn:
        cleanup = cleanup_expired_sessions(conn)
        conn.commit()
    if cleanup["sessions_deleted"] or cleanup["turns_deleted"]:
        log.info(
            "Session cleanup removed %d sessions and %d turns",
            cleanup["sessions_deleted"],
            cleanup["turns_deleted"],
        )
    init_chroma(CHROMA_PATH)

    loop = asyncio.get_running_loop()
    ws_manager.set_loop(loop)

    # Scan for PDFs that were mid-index when server last stopped
    await scan_and_requeue(PDF_DIR, _run_indexing)

    async def _warm_models() -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, load_embedder)
            await loop.run_in_executor(None, load_ner_models)
            await loop.run_in_executor(None, benchmark_embedding)
            log.info("Model warmup complete.")
        except Exception as e:
            log.warning("Model warmup skipped: %s", e)

    asyncio.create_task(_warm_models())

    log.info("QueryMind ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────
    log.info("QueryMind shutting down.")


# ── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="QueryMind API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────

from backend.routes.documents import router as documents_router
from backend.routes.chat import router as chat_router
from backend.routes.graph import router as graph_router
from backend.routes.health import router as health_router

app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(graph_router)
app.include_router(health_router)


# ── WebSocket ─────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    from backend.ws.manager import ws_manager

    await ws_manager.connect(session_id, ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
    except Exception as e:
        log.warning("WS error for %s: %s", session_id[:8], e)
        ws_manager.disconnect(session_id)
