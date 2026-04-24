"""
WebSocket connection manager.

Maintains a dict of session_id → WebSocket. Provides both async methods
(for use inside async routes) and thread-safe sync methods (for use from
the indexing pipeline running in a thread pool).
"""

import asyncio
import logging
from typing import Dict, Optional

from fastapi import WebSocket

log = logging.getLogger(__name__)


class WSManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once at startup with the running event loop."""
        self._loop = loop

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self.connections[session_id] = ws
        log.info("WS connected: session=%s (total=%d)", session_id[:8], len(self.connections))

    def disconnect(self, session_id: str) -> None:
        self.connections.pop(session_id, None)
        log.info("WS disconnected: session=%s", session_id[:8])

    async def send(self, session_id: str, message: dict) -> None:
        """Async send to a specific session."""
        ws = self.connections.get(session_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                log.warning("WS send failed for %s: %s", session_id[:8], e)
                self.disconnect(session_id)

    async def broadcast(self, message: dict) -> None:
        """Async broadcast to all connected sessions."""
        for session_id in list(self.connections.keys()):
            await self.send(session_id, message)

    def send_sync(self, session_id: str, message: dict) -> None:
        """
        Thread-safe sync send. Use from pipeline thread pool.
        Falls back to no-op if no event loop is set.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send(session_id, message), self._loop
            )

    def broadcast_sync(self, message: dict) -> None:
        """Thread-safe broadcast. Use from pipeline thread pool."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.broadcast(message), self._loop
            )


ws_manager = WSManager()
