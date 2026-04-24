"""
Startup orphan scanner.

Scans data/pdfs/ for PDF files that have no corresponding SQLite entry.
This handles the case where the server restarted mid-indexing.
Re-queues orphaned files as background indexing tasks.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


async def scan_and_requeue(
    pdf_root: str,
    run_indexing: Callable[[str, str, str], None],
    dummy_session_id: str = "system",
) -> int:
    """
    Walk pdf_root (structure: pdf_root/{document_set_id}/{filename}.pdf).
    For each PDF, check if it has been indexed in SQLite.
    Re-queue any that haven't been.

    Returns the number of orphaned PDFs re-queued.
    """
    from backend.indexing.extractor import file_hash
    from backend.indexing.pipeline import document_exists

    requeued = 0
    root = Path(pdf_root)

    if not root.exists():
        return 0

    for set_dir in root.iterdir():
        if not set_dir.is_dir():
            continue
        document_set_id = set_dir.name

        for pdf_path in set_dir.glob("*.pdf"):
            try:
                doc_id = file_hash(pdf_path)
                if not document_exists(doc_id):
                    log.info(
                        "Orphan found: %s (set=%s) — re-queuing",
                        pdf_path.name,
                        document_set_id[:8],
                    )
                    asyncio.create_task(
                        run_indexing(str(pdf_path), document_set_id, dummy_session_id)
                    )
                    requeued += 1
            except Exception:
                log.exception("Error scanning %s", pdf_path)

    if requeued:
        log.info("Orphan scanner: re-queued %d PDF(s)", requeued)
    else:
        log.info("Orphan scanner: no orphaned PDFs found")

    return requeued
