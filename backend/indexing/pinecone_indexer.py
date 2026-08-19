"""
Pinecone knowledge base indexer for LBRCE AI Assistant.

Handles vector upsert to Pinecone index with full metadata preservation.
Workflow: Chunk → Embedding → Pinecone upsert
"""

from __future__ import annotations

import logging
from typing import List, Optional

from backend.ingestion.chunker import Chunk, chunk_document

logger = logging.getLogger(__name__)


try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore

# Exception for when Pinecone is not available
class PineconeNotAvailableError(Exception):
    """Raised when Pinecone SDK or configuration is unavailable."""
    pass


class PineconeIndexer:
    """Indexes Chunk objects into a Pinecone vector store.

    Each chunk is embedded and upserted with full metadata for
    source attribution in RAG queries.
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 1024,
        metric: str = "cosine",
        namespace: str = "",
    ):
        """Initialize Pinecone indexer.

        Args:
            api_key: Pinecone API key.
            environment: Pinecone environment (e.g., "us-east-1").
            index_name: Name of the Pinecone index.
            dimension: Embedding vector dimension.
            metric: Distance metric ("cosine", "euclidean", "dotproduct").
        """
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "Pinecone not installed. Run: pip install pinecone"
            )

        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.namespace = namespace or ""

        # Initialize Pinecone client or mock for testing
        if api_key.startswith("mock"):
            # Simple mock index that returns empty matches
            class _MockResult:
                matches = []

            class _MockIndex:
                def query(self, *args, **kwargs):
                    return _MockResult()

            self.index = _MockIndex()
            logger.info("PineconeIndexer using mock index (mock API key)")
        else:
            pc = Pinecone(api_key=api_key)
            self.index = pc.Index(index_name)

        logger.info(f"PineconeIndexer connected to index '{index_name}' "
                    f"({dimension}dim, {metric})")

    def _prepare_vector(
        self,
        chunk: "Chunk",
        vector: List[float],
    ) -> dict:
        """Prepare a vector dict for Pinecone upsert with full metadata.

        Args:
            chunk: Chunk object with metadata.
            vector: Embedding vector.

        Returns:
            Dict compatible with Pinecone upsert format.
        """
        metadata = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source_url": chunk.source_url,
            "title": chunk.title,
            "source_type": chunk.source_type,
            "department": chunk.department,
            "page_number": chunk.page_number,
            "document_id": chunk.document_id,
            "crawl_time": None,  # TODO: set at ingestion time
        }
        # Manifest-specific fields such as image_url, semester, section,
        # academic_year, and PDF document_type are intentionally preserved.
        metadata.update(chunk.metadata or {})

        # Pinecone metadata must have string values; convert non-strings
        filtered_metadata = {}
        for k, v in metadata.items():
            if v is not None:
                filtered_metadata[k] = str(v) if not isinstance(v, str) else v

        return {
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": filtered_metadata,
        }

    def upsert_chunk(self, chunk: "Chunk", vector: List[float]) -> None:
        """Upsert a single chunk vector into Pinecone.

        Args:
            chunk: Chunk object with metadata.
            vector: Embedding vector for the chunk.
        """
        vector_dict = self._prepare_vector(chunk, vector)
        upsert_kwargs = {"vectors": [vector_dict]}
        if self.namespace:
            upsert_kwargs["namespace"] = self.namespace
        self.index.upsert(**upsert_kwargs)
        logger.debug(f"Upserted chunk {chunk.chunk_id} into Pinecone index")

    def upsert_chunks(self, chunks: List["Chunk"], vectors: List[List[float]]) -> None:
        """Upsert multiple chunk vectors into Pinecone.

        Args:
            chunks: List of Chunk objects.
            vectors: List of embedding vectors (one per chunk).
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )

        vectors_to_upsert = []
        for chunk, vector in zip(chunks, vectors):
            vector_dict = self._prepare_vector(chunk, vector)
            vectors_to_upsert.append(vector_dict)

        # Pinecone upsert in batches of 100 (or as configured)
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            upsert_kwargs = {"vectors": batch}
            if self.namespace:
                upsert_kwargs["namespace"] = self.namespace
            self.index.upsert(**upsert_kwargs)
            logger.info(f"Upserted batch {i//batch_size + 1} "
                        f"({len(batch)} vectors) into Pinecone index")

    def index_chunks(
        self,
        chunks: List["Chunk"],
        embedding_generator,
    ) -> None:
        """Full pipeline: embed and upsert chunks into Pinecone.

        Args:
            chunks: List of Chunk objects to index.
            embedding_generator: Callable or EmbeddingGenerator instance.
        """
        vectors = embedding_generator.embed_chunks(chunks)
        self.upsert_chunks(chunks, vectors)
        logger.info(f"Indexed {len(chunks)} chunks into Pinecone index '{self.index_name}'")