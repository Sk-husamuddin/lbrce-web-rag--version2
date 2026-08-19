"""Create and upload the local BGE benchmark into a separate Pinecone namespace.

Safety properties:
- Refuses the production index name ``lbrce-index``.
- Refuses an empty or production-looking namespace.
- Defaults to dry-run; actual upsert requires ``--confirm-local-migration``.
- Uploads only local benchmark artifacts and never calls Pinecone inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise SystemExit("Install python-dotenv first: python -m pip install python-dotenv") from exc

from pinecone import Pinecone, ServerlessSpec

PRODUCTION_INDEX = "lbrce-index"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_v1"
DIMENSION = 1024
METRIC = "cosine"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_metadata(record: dict) -> dict:
    metadata = {}
    for key, value in record.items():
        if key in {"id", "text"} or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    metadata["text"] = record["text"]
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="migration_artifacts/local_benchmark")
    parser.add_argument("--index-name", default=os.getenv("MIGRATION_PINECONE_INDEX", DEFAULT_INDEX))
    parser.add_argument("--namespace", default=os.getenv("MIGRATION_PINECONE_NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--cloud", default=os.getenv("PINECONE_CLOUD", "aws"))
    parser.add_argument("--region", default=os.getenv("PINECONE_REGION", "us-east-1"))
    parser.add_argument("--confirm-local-migration", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: migration script cannot target production index lbrce-index")
    if not args.index_name or not args.namespace:
        raise SystemExit("REFUSED: index name and namespace are required")
    if "prod" in args.namespace.lower() or args.namespace.lower() in {"", "default"}:
        raise SystemExit("REFUSED: use an explicit local migration namespace")

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is missing from the migration environment")

    artifact_dir = Path(args.artifact_dir)
    chunks = load_jsonl(artifact_dir / "benchmark_chunks.jsonl")
    vectors = np.load(artifact_dir / "benchmark_vectors.npy")
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION:
        raise SystemExit(f"REFUSED: expected vectors with dimension {DIMENSION}, got {vectors.shape}")
    if len(chunks) != len(vectors):
        raise SystemExit(f"REFUSED: {len(chunks)} chunks but {len(vectors)} vectors")

    print(f"Target index: {args.index_name}")
    print(f"Target namespace: {args.namespace}")
    print(f"Dimension/metric: {DIMENSION}/{METRIC}")
    print(f"Vectors to upload: {len(chunks)}")
    print("Pinecone inference calls: 0")

    pc = Pinecone(api_key=api_key)
    index_names = {item["name"] if isinstance(item, dict) else item.name for item in pc.list_indexes()}
    if args.index_name not in index_names:
        if not args.confirm_local_migration:
            print("DRY RUN: index does not exist; rerun with --confirm-local-migration to create it")
            return 0
        print(f"Creating new index {args.index_name}...")
        pc.create_index(
            name=args.index_name,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=args.cloud, region=args.region),
        )
        while not pc.describe_index(args.index_name).status["ready"]:
            time.sleep(5)

    if not args.confirm_local_migration:
        print("DRY RUN: no index creation, deletion, or upsert performed")
        return 0

    index = pc.Index(args.index_name)
    vectors_to_upsert = []
    for record, vector in zip(chunks, vectors):
        vectors_to_upsert.append({
            "id": record["id"],
            "values": vector.astype(float).tolist(),
            "metadata": safe_metadata(record),
        })
    for start in range(0, len(vectors_to_upsert), 100):
        batch = vectors_to_upsert[start:start + 100]
        index.upsert(vectors=batch, namespace=args.namespace)
        print(f"Uploaded {min(start + len(batch), len(vectors_to_upsert))}/{len(vectors_to_upsert)}")

    stats = index.describe_index_stats()
    print(json.dumps({
        "index": args.index_name,
        "namespace": args.namespace,
        "dimension": DIMENSION,
        "uploaded": len(vectors_to_upsert),
        "pinecone_inference_used": False,
        "stats": stats.to_dict() if hasattr(stats, "to_dict") else str(stats),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
