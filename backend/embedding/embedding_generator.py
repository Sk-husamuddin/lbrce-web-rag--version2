"""
Embedding generation for LBRCE AI Assistant using Pinecone inference services.

Converts Chunk objects into vector embeddings for Pinecone indexing.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import List, Optional
from pinecone import Pinecone

logger = logging.getLogger(__name__)

_MOCK_PINECONE_KEYS = {
    "mock_pinecone_key",
    "mock-pinecone",
    "mock_pinecone",
}


def _stable_mock_vector(text: str, dimension: int) -> List[float]:
    """Generate a reproducible mock vector without Python's randomized hash()."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    generator = random.Random(seed)
    return [generator.random() for _ in range(dimension)]


class EmbeddingGenerator:
    """Generates vector embeddings using Pinecone's integrated inference API.

    Ensures that document chunk embeddings and query embeddings are generated
    using the same model configuration.
    """

    def __init__(
        self,
        model_name: str = "llama-text-embed-v2",
        dimension: int = 1024,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.dimension = dimension

        if not api_key:
            try:
                from backend.config.settings import settings
                if settings:
                    api_key = settings.PINECONE_API_KEY
            except ImportError:
                pass
        
        self.api_key = api_key or "mock_pinecone_key"
        self.pc = Pinecone(api_key=self.api_key)
        logger.info(f"EmbeddingGenerator initialized with {model_name} ({dimension}dim)")

    def embed(self, text: str) -> List[float]:
        """Generate a dense embedding vector for a single query text.

        Args:
            text: Input query string.

        Returns:
            List of float values representing the embedding vector.
        """
        if not text:
            return [0.0] * self.dimension

        # Safe mock path for testing or missing API keys
        if self.api_key in _MOCK_PINECONE_KEYS:
            return _stable_mock_vector(text, self.dimension)

        try:
            response = self.pc.inference.embed(
                model=self.model_name,
                inputs=[text],
                parameters={"input_type": "query", "truncate": "END"}
            )
            return response.data[0].values
        except Exception as e:
            logger.error(f"Pinecone embedding generation failed: {e}")
            raise e

    def embed_chunks(self, chunks: List["Chunk"]) -> List[List[float]]:
        """Generate embeddings for a list of Chunk objects.

        Args:
            chunks: List of Chunk objects (from backend.ingestion.chunker).

        Returns:
            List of embedding vectors, one per chunk.
        """
        from backend.ingestion.chunker import Chunk as ChunkType

        if not chunks:
            return []

        texts = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, ChunkType):
                raise TypeError(f"Expected Chunk, got {type(chunk)}")
            texts.append(str(chunk.text))

        # Safe mock path for testing or missing API keys
        if self.api_key in _MOCK_PINECONE_KEYS:
            return [_stable_mock_vector(text, self.dimension) for text in texts]

        try:
            response = self.pc.inference.embed(
                model=self.model_name,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            return [emb.values for emb in response.data]
        except Exception as e:
            logger.error(f"Pinecone batch embedding generation failed: {e}")
            raise e