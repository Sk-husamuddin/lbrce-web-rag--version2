"""Local BGE embedding provider for the migration-copy retrieval path.

This module never calls Pinecone inference. The same model and encoding
settings used by the migration scripts are used for user-query vectors.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # Optional for Pinecone-only test/deployment environments.
    SentenceTransformer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required when EMBEDDING_PROVIDER=local. "
            "Install it with: pip install sentence-transformers"
        )
    logger.info("Loading local embedding model %s", model_name)
    return SentenceTransformer(model_name)


class LocalBGEEmbeddingGenerator:
    """Generate normalized local BGE query vectors."""

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", dimension: int = 1024):
        self.model_name = model_name
        self.dimension = int(dimension)
        self.model = _load_model(model_name)
        logger.info(
            "LocalBGEEmbeddingGenerator initialized with %s (%sdim)",
            model_name,
            self.dimension,
        )

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        prefixed_text = self.QUERY_INSTRUCTION + text
        vector = self.model.encode(
            [prefixed_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Local embedding dimension mismatch: expected {self.dimension}, got {vector.shape}"
            )
        return vector.tolist()

    def embed_chunks(self, chunks) -> List[List[float]]:
        texts = [str(chunk.text) for chunk in chunks]
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                f"Local embedding dimension mismatch: expected {(len(texts), self.dimension)}, got {vectors.shape}"
            )
        return vectors.tolist()
