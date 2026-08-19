"""Verify the local BGE embedding model without Pinecone or project ingestion."""

from __future__ import annotations

import sys
from pathlib import Path


MODEL_NAME = "BAAI/bge-large-en-v1.5"
EXPECTED_DIMENSION = 1024


def main() -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print("ERROR: sentence-transformers is not installed.")
        print("Install it with: python -m pip install sentence-transformers")
        print(f"Details: {exc}")
        return 1

    print(f"Loading local embedding model: {MODEL_NAME}")
    print("The first run may download the model from Hugging Face.")
    model = SentenceTransformer(MODEL_NAME)

    query_text = "Represent this sentence for searching relevant passages: Where is LBRCE located?"
    passage_text = (
        "Lakireddy Bali Reddy College of Engineering is located in Mylavaram, "
        "Krishna District, Andhra Pradesh."
    )

    query_vector = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    passage_vector = model.encode(
        passage_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_shape = tuple(query_vector.shape)
    passage_shape = tuple(passage_vector.shape)

    print(f"Query vector shape:   {query_shape}")
    print(f"Passage vector shape: {passage_shape}")
    print(f"Query vector norm:    {float((query_vector ** 2).sum() ** 0.5):.6f}")
    print(f"Passage vector norm:  {float((passage_vector ** 2).sum() ** 0.5):.6f}")

    if query_shape != (EXPECTED_DIMENSION,) or passage_shape != (EXPECTED_DIMENSION,):
        print(
            f"ERROR: Expected {EXPECTED_DIMENSION}-dimensional vectors, "
            f"got query={query_shape}, passage={passage_shape}."
        )
        return 1

    print("PASS: local model generated two normalized 1024-dimensional vectors.")
    print("PASS: no Pinecone API or Pinecone embedding call was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
