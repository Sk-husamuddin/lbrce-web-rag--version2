"""Validate and upload the completed local BGE vectors to the full migration namespace.

This script does not generate embeddings and never calls Pinecone inference. It
reads local ``chunks.jsonl`` and ``vectors.npy``, verifies exact alignment, and
refuses the production index and benchmark namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_MODEL = "BAAI/bge-large-en-v1.5"
EXPECTED_DIMENSION = 1024
PRODUCTION_INDEX = "lbrce-index"
BENCHMARK_NAMESPACE = "lbrce_local_bge_v1"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_full_v1"


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metadata_for(chunk: dict) -> dict:
    result = {}
    for key, value in chunk.items():
        if key == "id" or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


def validate_local_artifacts(artifact_dir: Path) -> tuple[list[dict], np.ndarray]:
    chunks_path = artifact_dir / "chunks.jsonl"
    vectors_path = artifact_dir / "vectors.npy"
    manifest_path = artifact_dir / "vector_manifest.json"
    if not chunks_path.exists() or not vectors_path.exists():
        raise SystemExit("REFUSED: chunks.jsonl or vectors.npy is missing")
    chunks = load_chunks(chunks_path)
    vectors = np.load(vectors_path)
    if not chunks:
        raise SystemExit("REFUSED: chunks.jsonl is empty")
    if vectors.shape != (len(chunks), EXPECTED_DIMENSION):
        raise SystemExit(f"REFUSED: vectors/chunks mismatch: vectors={vectors.shape}, chunks={len(chunks)}")
    ids = [chunk.get("id") for chunk in chunks]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise SystemExit("REFUSED: chunk IDs are missing or duplicated")
    if not np.all(np.isfinite(vectors)):
        raise SystemExit("REFUSED: vector file contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit("REFUSED: local vectors are not normalized")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != EXPECTED_MODEL:
            raise SystemExit("REFUSED: vector manifest model mismatch")
        if int(manifest.get("dimension", -1)) != EXPECTED_DIMENSION:
            raise SystemExit("REFUSED: vector manifest dimension mismatch")
        if int(manifest.get("chunk_count", -1)) != len(chunks):
            raise SystemExit("REFUSED: vector manifest chunk count mismatch")
        if manifest.get("pinecone_written") is not False:
            raise SystemExit("REFUSED: vector manifest does not prove local-only generation")
    return chunks, vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="migration_artifacts/full_registry_local_bge")
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--confirm-full-migration", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: cannot target production lbrce-index")
    if args.namespace in {"", "default", BENCHMARK_NAMESPACE} or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: invalid full-migration namespace")

    chunks, vectors = validate_local_artifacts(Path(args.artifact_dir))
    print(json.dumps({
        "target_index": args.index_name,
        "target_namespace": args.namespace,
        "model": EXPECTED_MODEL,
        "dimension": EXPECTED_DIMENSION,
        "chunks": len(chunks),
        "vectors": list(vectors.shape),
        "pinecone_inference_used": False,
    }, indent=2))

    if not args.confirm_full_migration:
        print("DRY RUN: local vectors validated; no Pinecone connection or upsert performed")
        return 0

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is missing")
    pc = Pinecone(api_key=api_key)
    names = {item["name"] if isinstance(item, dict) else item.name for item in pc.list_indexes()}
    if args.index_name not in names:
        raise SystemExit(f"REFUSED: target index {args.index_name} does not exist")
    description = pc.describe_index(args.index_name)
    if int(description.dimension) != EXPECTED_DIMENSION or description.metric != "cosine":
        raise SystemExit("REFUSED: target index is not 1024-dimensional cosine")

    index = pc.Index(args.index_name)
    uploaded = 0
    for start in range(0, len(chunks), args.batch_size):
        batch_chunks = chunks[start:start + args.batch_size]
        batch_vectors = vectors[start:start + args.batch_size]
        payload = [
            {
                "id": chunk["id"],
                "values": vector.astype(float).tolist(),
                "metadata": metadata_for(chunk),
            }
            for chunk, vector in zip(batch_chunks, batch_vectors)
        ]
        index.upsert(vectors=payload, namespace=args.namespace)
        uploaded += len(payload)
        print(f"Uploaded {uploaded}/{len(chunks)}")

    stats = index.describe_index_stats()
    print(json.dumps({
        "index": args.index_name,
        "namespace": args.namespace,
        "uploaded": uploaded,
        "pinecone_inference_used": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats.to_dict() if hasattr(stats, "to_dict") else str(stats),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
