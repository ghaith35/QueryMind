"""
Lightweight startup benchmarks for environment health.
"""

from __future__ import annotations

import logging
import time

from backend.indexing.embedder import BATCH_SIZE, get_device, load_model

log = logging.getLogger(__name__)

_benchmark_state: dict[str, float | str | bool | None] = {
    "device": None,
    "elapsed_seconds": None,
    "batch_size": BATCH_SIZE,
    "slow": None,
}


def benchmark_embedding(sample_size: int = 16) -> dict[str, float | str | bool | None]:
    model = load_model()
    device = get_device()
    samples = ["test sentence for warm benchmark"] * sample_size

    started_at = time.perf_counter()
    model.encode(samples, batch_size=min(BATCH_SIZE, sample_size), show_progress_bar=False)
    elapsed = round(time.perf_counter() - started_at, 3)

    slow = elapsed > 5.0
    _benchmark_state.update(
        {
            "device": device,
            "elapsed_seconds": elapsed,
            "batch_size": sample_size,
            "slow": slow,
        }
    )

    if slow:
        log.warning(
            "Embedding benchmark slow on %s (%.2fs for %d samples). MPS acceleration may be unavailable.",
            device,
            elapsed,
            sample_size,
        )
    else:
        log.info("Embedding benchmark: %s in %.2fs for %d samples", device, elapsed, sample_size)

    return dict(_benchmark_state)


def get_benchmark_state() -> dict[str, float | str | bool | None]:
    return dict(_benchmark_state)
