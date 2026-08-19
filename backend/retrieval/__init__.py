"""
RAG Retrieval layer for LBRCE AI Assistant.

Handles query embedding, Pinecone similarity search, and result formatting.
Reuses existing abstractions from Phases 5-6:
  - backend.ingestion.chunker.Chunk for metadata structure
  - backend.embedding.EmbeddingGenerator for query embedding
  - backend.indexing.pinecone_indexer.PineconeIndexer for Pinecone client
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from backend.embedding import EmbeddingGenerator
from backend.indexing.pinecone_indexer import PineconeIndexer, PineconeNotAvailableError
from backend.ingestion.chunker import Chunk
from backend.retrieval.rag import RAGPipeline

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Base exception for retrieval-layer errors."""
    pass


class EmptyQueryError(RetrievalError):
    """Raised when the query is empty or whitespace-only."""
    pass


class NoResultsError(RetrievalError):
    """Raised when no matching chunks are found."""
    pass


class PineconeUnavailableError(RetrievalError):
    """Raised when Pinecone service is unavailable."""
    pass


def _validate_query(query: str) -> None:
    """Validate that the query is non-empty after stripping."""
    if not query or not query.strip():
        raise EmptyQueryError("Query must be a non-empty string")


def _format_chunk(chunk: Chunk) -> dict:
    """Format a Chunk into a result dict with standard and manifest metadata."""
    result = {
        "chunk_id": chunk.chunk_id,
        "chunk_text": chunk.text,
        "similarity_score": getattr(chunk, "_similarity_score", 0.0),
        "source_url": chunk.source_url,
        "title": chunk.title,
        "source_type": chunk.source_type,
        "department": chunk.department,
        "page_number": chunk.page_number,
        "document_id": chunk.document_id,
    }
    result.update(chunk.metadata or {})
    return result


def retrieve(
    query: str,
    embedding_generator: EmbeddingGenerator,
    pinecone_indexer: PineconeIndexer,
    top_k: int = 4,
    metadata_filter: Optional[dict] = None,
) -> List[dict]:
    """Run a retrieval query: embed → Pinecone search → format results.

    Args:
        query: User's search query string.
        embedding_generator: Initialised EmbeddingGenerator instance.
        pinecone_indexer: Initialised PineconeIndexer instance.
        top_k: Number of top results to return (default: 4).

    Returns:
        List of result dicts with keys: chunk_id, chunk_text,
        similarity_score, source_url, title, source_type, department,
        page_number, and document_id.

    Raises:
        EmptyQueryError: If query is empty/whitespace.
        PineconeUnavailableError: If Pinecone service is unavailable.
        NoResultsError: If no chunks match the query.
    """
    _validate_query(query)

    # Step 1: Embed the query
    query_vector = embedding_generator.embed(query)

    # Step 2: Query Pinecone
    try:
        query_kwargs = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True,
        }
        namespace = getattr(pinecone_indexer, "namespace", "") or ""
        if namespace:
            query_kwargs["namespace"] = namespace
            logger.info("Querying Pinecone namespace %r", namespace)
        if metadata_filter:
            query_kwargs["filter"] = metadata_filter
            logger.info("Applying Pinecone metadata filter %s", metadata_filter)
        search_results = pinecone_indexer.index.query(**query_kwargs)
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        raise PineconeUnavailableError(
            f"Pinecone query failed: {e}"
        ) from e

    # Step 3: Format results
    results = []
    matches = getattr(search_results, "matches", []) or []
    for match in matches:
        meta = match.metadata or {}
        score = match.score if match.score is not None else 0.0
        raw_page = meta.get("page_number")
        try:
            page_num = int(raw_page) if raw_page not in (None, "", "None") else None
        except (ValueError, TypeError):
            page_num = None

        reserved = {
            "chunk_id", "text", "source_url", "title", "source_type",
            "department", "page_number", "document_id"
        }
        extra_metadata = {key: value for key, value in meta.items() if key not in reserved}

        # Build a minimal Chunk-like object from metadata.
        # We reconstruct enough to return the required fields.
        chunk = Chunk(
            chunk_id=meta.get("chunk_id") or getattr(match, "id", "") or "",
            text=meta.get("text", ""),
            source_url=meta.get("source_url", ""),
            title=meta.get("title", ""),
            source_type=meta.get("source_type", ""),
            department=meta.get("department", ""),
            page_number=page_num,
            document_id=meta.get("document_id", ""),
            metadata=extra_metadata,
        )
        # Store the similarity score on the chunk for formatting
        chunk._similarity_score = score
        results.append(_format_chunk(chunk))

    if not results:
        raise NoResultsError(
            f"No retrieval results for query: {query!r} "
            f"(top_k={top_k})"
        )

    return results


def retrieve_with_fallback(
    query: str,
    embedding_generator: EmbeddingGenerator,
    pinecone_indexer: Optional[PineconeIndexer],
    top_k: int = 4,
    metadata_filter: Optional[dict] = None,
) -> Tuple[List[dict], bool]:
    """Run retrieval with graceful fallback when Pinecone is unavailable.

    Returns:
        (results, pinecone_available) tuple.
        - results: List of result dicts (empty if Pinecone unavailable).
        - pinecone_available: Whether Pinecone service was reachable.
    """
    pinecone_available = True
    try:
        results = retrieve(
            query,
            embedding_generator,
            pinecone_indexer,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return results, True
    except PineconeUnavailableError:
        pinecone_available = False
        return [], False
    except NoResultsError:
        # The service was reachable, but no matching records were found.
        return [], True
    except EmptyQueryError:
        # Input errors should remain visible to the caller instead of being
        # mislabeled as a Pinecone outage.
        raise
    except Exception as e:
        logger.error(f"Unexpected retrieval error: {e}")
        raise
