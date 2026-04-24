"""
Watchdog-based filesystem observer for the PDF drop folder.

When a new .pdf appears in the watched directory:
1. Hash it to get document_id.
2. Check if already indexed (duplicate prevention).
3. Fire the pipeline.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

log = logging.getLogger(__name__)


class PDFEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        on_new_pdf: Callable[[str], None],
        document_exists: Callable[[str], bool],
    ):
        super().__init__()
        self._on_new_pdf = on_new_pdf
        self._document_exists = document_exists
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = str(event.src_path)
        if not path.lower().endswith(".pdf"):
            return

        # Brief wait — some copy operations fire created before the file is complete
        time.sleep(0.5)

        doc_id = self._hash_file(path)
        if not doc_id:
            return

        with self._lock:
            if doc_id in self._in_flight:
                log.debug("Already processing %s", path)
                return
            if self._document_exists(doc_id):
                log.info("Skip (already indexed): %s", Path(path).name)
                return
            self._in_flight.add(doc_id)

        try:
            log.info("New PDF detected: %s", Path(path).name)
            self._on_new_pdf(path)
        except Exception:
            log.exception("Pipeline failed for %s", path)
        finally:
            with self._lock:
                self._in_flight.discard(doc_id)

    @staticmethod
    def _hash_file(path: str) -> Optional[str]:
        import hashlib
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
            return h.hexdigest()[:16]
        except OSError as e:
            log.warning("Could not hash %s: %s", path, e)
            return None


class PDFWatcher:
    def __init__(
        self,
        watch_dir: str | Path,
        on_new_pdf: Callable[[str], None],
        document_exists: Callable[[str], bool],
    ):
        self._dir = str(watch_dir)
        self._handler = PDFEventHandler(on_new_pdf, document_exists)
        self._observer: Optional[Observer] = None

    def start(self) -> None:
        self._observer = Observer()
        self._observer.schedule(self._handler, self._dir, recursive=False)
        self._observer.start()
        log.info("Watching %s for new PDFs", self._dir)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            log.info("PDF watcher stopped")
