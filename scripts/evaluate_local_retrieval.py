"""Evaluate metadata-filtered local retrieval without Pinecone.

The evaluator loads benchmark_chunks.jsonl and benchmark_vectors.npy, embeds
queries with the same local BGE model, applies metadata filters locally, and
reports whether the expected official page appears at rank 1 or in top-k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EXPECTED_DIMENSION = 1024


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matches_filter(record: dict, query: dict) -> bool:
    if query.get("page_category") and record.get("page_category") != query["page_category"]:
        return False
    if query.get("department") and record.get("department") != query["department"]:
        return False
    if query.get("topic") and record.get("topic") != query["topic"]:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="migration_artifacts/local_benchmark")
    parser.add_argument("--queries", default="scripts/local_retrieval_queries.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    chunks = load_jsonl(artifact_dir / "benchmark_chunks.jsonl")
    vectors = np.load(artifact_dir / "benchmark_vectors.npy")
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]

    if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIMENSION:
        raise RuntimeError(f"Expected vectors with dimension {EXPECTED_DIMENSION}, got {vectors.shape}")
    if len(chunks) != len(vectors):
        raise RuntimeError(f"Chunk/vector mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")

    model = SentenceTransformer(MODEL_NAME)
    results = []
    passed = 0

    for query in queries:
        eligible_indices = [index for index, record in enumerate(chunks) if matches_filter(record, query)]
        if not eligible_indices:
            results.append({
                "id": query["id"],
                "query": query["query"],
                "status": "FAIL",
                "reason": "metadata filter returned zero chunks",
            })
            continue

        query_text = "Represent this sentence for searching relevant passages: " + query["query"]
        query_vector = model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = vectors[eligible_indices] @ query_vector
        ranked = sorted(zip(eligible_indices, scores.tolist()), key=lambda pair: pair[1], reverse=True)
        top = []
        for index, score in ranked[: args.top_k]:
            record = chunks[index]
            top.append({
                "score": round(float(score), 6),
                "url": record.get("source_url"),
                "page_category": record.get("page_category"),
                "topic": record.get("topic"),
                "department": record.get("department"),
                "chunk_index": record.get("chunk_index"),
            })

        expected = query.get("expected_url_contains", "").lower()
        top_urls = [str(item["url"]).lower() for item in top]
        matched_rank = next(
            (rank + 1 for rank, url in enumerate(top_urls) if expected in url),
            None,
        )
        status = "PASS" if matched_rank is not None else "REVIEW"
        if status == "PASS":
            passed += 1
        results.append({
            "id": query["id"],
            "query": query["query"],
            "eligible_chunk_count": len(eligible_indices),
            "expected_url_contains": expected,
            "matched_rank": matched_rank,
            "status": status,
            "top": top,
        })

    summary = {
        "model": MODEL_NAME,
        "dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "query_count": len(queries),
        "passes": passed,
        "reviews": sum(1 for result in results if result["status"] == "REVIEW"),
        "failures": sum(1 for result in results if result["status"] == "FAIL"),
        "pinecone_written": False,
        "results": results,
    }
    output_path = artifact_dir / "local_retrieval_evaluation.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Evaluation written to: {output_path.resolve()}")
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
