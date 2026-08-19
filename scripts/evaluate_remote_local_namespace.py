"""Evaluate the separate local-BGE Pinecone namespace with local query vectors.

This script never calls Pinecone inference. Query embeddings are generated with
BAAI/bge-large-en-v1.5 locally, then sent to the non-production namespace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PRODUCTION_INDEX = "lbrce-index"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_v1"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
EXPECTED_DIMENSION = 1024


def pinecone_filter(query: dict) -> dict | None:
    conditions = []
    if query.get("page_category"):
        conditions.append({"page_category": {"$eq": query["page_category"]}})
    if query.get("department"):
        conditions.append({"department": {"$eq": query["department"]}})
    if query.get("topic"):
        conditions.append({"topic": {"$eq": query["topic"]}})
    if query.get("source_url"):
        conditions.append({"source_url": {"$eq": query["source_url"]}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--queries", default="scripts/local_retrieval_queries.json")
    parser.add_argument("--output", default="migration_artifacts/local_benchmark/remote_local_namespace_evaluation.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: remote evaluator cannot target production index lbrce-index")
    if args.namespace in {"", "default"} or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: use the explicit local migration namespace")

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is missing from the migration environment")

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))["queries"]
    model = SentenceTransformer(MODEL_NAME)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index_name)
    stats = index.describe_index_stats()
    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else str(stats)

    results = []
    for query in queries:
        text = "Represent this sentence for searching relevant passages: " + query["query"]
        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if len(vector) != EXPECTED_DIMENSION:
            raise SystemExit(f"REFUSED: local query vector has dimension {len(vector)}")

        response = index.query(
            namespace=args.namespace,
            vector=vector.tolist(),
            top_k=args.top_k,
            include_metadata=True,
            filter=pinecone_filter(query),
        )
        matches = getattr(response, "matches", [])
        top = []
        for match in matches:
            metadata = getattr(match, "metadata", {}) or {}
            top.append({
                "id": getattr(match, "id", None),
                "score": round(float(getattr(match, "score", 0.0)), 6),
                "url": metadata.get("source_url"),
                "page_category": metadata.get("page_category"),
                "topic": metadata.get("topic"),
                "department": metadata.get("department"),
                "chunk_index": metadata.get("chunk_index"),
            })
        expected = query.get("expected_url_contains", "").lower()
        matched_rank = next(
            (rank + 1 for rank, item in enumerate(top) if expected in str(item.get("url", "")).lower()),
            None,
        )
        results.append({
            "id": query["id"],
            "query": query["query"],
            "filter": pinecone_filter(query),
            "matched_rank": matched_rank,
            "status": "PASS" if matched_rank is not None else "FAIL",
            "top": top,
        })

    summary = {
        "index": args.index_name,
        "namespace": args.namespace,
        "model": MODEL_NAME,
        "dimension": EXPECTED_DIMENSION,
        "query_count": len(queries),
        "passes": sum(1 for item in results if item["status"] == "PASS"),
        "failures": sum(1 for item in results if item["status"] == "FAIL"),
        "pinecone_inference_used": False,
        "index_stats": stats_dict,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Evaluation written to: {output.resolve()}")
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
