"""
Multilingual-E5-small embedder with MPS (Apple Silicon) acceleration.

e5 convention:
    - Documents (chunks stored in vector DB): "passage: {text}"
    - Queries (user questions): "query: {text}"
Mixing these incorrectly degrades retrieval quality significantly.
"""

import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 16  # M1 8GB sweet spot; drop to 8 if OOM

_model: SentenceTransformer | None = None
_device: str | None = None


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model() -> SentenceTransformer:
    global _model, _device
    if _model is None:
        _device = get_device()
        log.info("Loading %s on %s", MODEL_NAME, _device)
        _model = SentenceTransformer(MODEL_NAME, device=_device)
        log.info("Embedder ready")
    return _model


def embed_chunks(texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """
    Embed document chunks. Applies 'passage: ' prefix as required by e5.
    Returns list of L2-normalised float vectors (dim=384 for e5-small).
    """
    model = load_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a user query. Applies 'query: ' prefix as required by e5.
    Returns a single L2-normalised float vector.
    """
    model = load_model()
    embedding = model.encode(
        f"query: {query}",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.tolist()
