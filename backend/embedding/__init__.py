"""
Embedding generation for LBRCE AI Assistant.

Provides the configured embedding implementation for both live retrieval and
all ingestion scripts. The provider is selected by EMBEDDING_PROVIDER in the
application settings so document and query vectors cannot silently diverge.
"""

from __future__ import annotations

from typing import Any

from backend.config.settings import settings as _settings
from .embedding_generator import EmbeddingGenerator
from .local_bge_embedding_generator import LocalBGEEmbeddingGenerator


def get_configured_embedding_generator() -> Any:
    """Return the embedding generator selected by application settings.

    ``EMBEDDING_PROVIDER=local`` selects the local BGE model. Any other value
    preserves the Pinecone inference embedder for production-compatible
    deployments.
    """
    provider = (
        getattr(_settings, "EMBEDDING_PROVIDER", "pinecone")
        if _settings is not None
        else "pinecone"
    )
    provider = str(provider).strip().lower()

    if provider == "local":
        return LocalBGEEmbeddingGenerator(
            model_name=getattr(
                _settings,
                "EMBEDDING_MODEL",
                "BAAI/bge-large-en-v1.5",
            ),
            dimension=int(getattr(_settings, "EMBEDDING_DIMENSION", 1024)),
        )

    return EmbeddingGenerator(
        api_key=(
            getattr(_settings, "PINECONE_API_KEY", None)
            if _settings is not None
            else None
        )
    )


__all__ = [
    "EmbeddingGenerator",
    "LocalBGEEmbeddingGenerator",
    "get_configured_embedding_generator",
]
