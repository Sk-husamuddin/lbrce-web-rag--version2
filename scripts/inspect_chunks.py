"""
Inspect what's actually sitting in Pinecone for a given department contact
page or free-text query — run this BEFORE re-chunking anything.

Two modes:

1. By URL (exact evidence check):
   python scripts/inspect_chunks.py --url https://lbrce.ac.in/ece/ececontact.php

2. By query (what would retrieve_node actually see):
   python scripts/inspect_chunks.py --query "who is the HOD of AIDS?"

Prints the raw chunk text, similarity score, and full metadata for each
match so you can see exactly what got embedded — no guessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import settings
from backend.embedding.embedding_generator import EmbeddingGenerator
from backend.indexing.pinecone_indexer import PineconeIndexer


def get_indexer() -> PineconeIndexer:
    return PineconeIndexer(
        api_key=settings.PINECONE_API_KEY if settings else "",
        environment="",
        index_name=settings.PINECONE_INDEX_NAME if settings else "lbrce-index",
    )


def get_embedder() -> EmbeddingGenerator:
    return EmbeddingGenerator(api_key=settings.PINECONE_API_KEY if settings else None)


def print_match(i: int, score: float, metadata: dict) -> None:
    text = metadata.get("chunk_text") or metadata.get("text") or ""
    print(f"\n--- match {i} | score={score:.4f} ---")
    print(f"source_url : {metadata.get('source_url')}")
    print(f"title      : {metadata.get('title')}")
    print(f"department : {metadata.get('department')}")
    print(f"source_type: {metadata.get('source_type')}")
    print(f"page_number: {metadata.get('page_number')}")
    print(f"chunk_text :\n{text}")
    print("-" * 60)


def inspect_by_url(url: str, top_k: int = 20) -> None:
    """Fetch every chunk whose source_url metadata matches exactly."""
    indexer = get_indexer()
    embedder = get_embedder()

    # Pinecone requires a query vector even for a metadata-only filter, so
    # embed a neutral probe string — the filter does the real narrowing.
    probe_vector = embedder.embed("department contact information")

    result = indexer.index.query(
        vector=probe_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"source_url": {"$eq": url}},
    )
    matches = getattr(result, "matches", []) or []
    if not matches:
        print(f"No chunks found in Pinecone with source_url == {url!r}.")
        print("Either it wasn't ingested, or the stored URL differs slightly "
              "(check for a trailing slash, http vs https, or www prefix).")
        return

    print(f"Found {len(matches)} chunk(s) for {url}:")
    for i, m in enumerate(matches, 1):
        print_match(i, getattr(m, "score", 0.0), getattr(m, "metadata", {}) or {})


def inspect_by_query(query: str, top_k: int = 5) -> None:
    """Run the exact same retrieval path retrieve_node uses, and print results."""
    indexer = get_indexer()
    embedder = get_embedder()

    query_vector = embedder.embed(query)
    result = indexer.index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
    matches = getattr(result, "matches", []) or []
    if not matches:
        print("No matches returned at all — check index name / API key / dimension.")
        return

    print(f"Top {len(matches)} Pinecone matches for query: {query!r}")
    for i, m in enumerate(matches, 1):
        print_match(i, getattr(m, "score", 0.0), getattr(m, "metadata", {}) or {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Exact source_url to fetch all chunks for.")
    parser.add_argument("--query", help="Free-text query to run through real retrieval.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if not args.url and not args.query:
        parser.error("Provide --url or --query (or both).")

    if args.url:
        inspect_by_url(args.url, top_k=args.top_k)
    if args.query:
        inspect_by_query(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()