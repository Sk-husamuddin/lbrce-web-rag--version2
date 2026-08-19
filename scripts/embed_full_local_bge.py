"""Generate local BGE vectors for the validated full HTML corpus.

This script reads only local ``chunks.jsonl`` and writes local NumPy/checkpoint
artifacts. It never imports Pinecone and never makes network calls.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EXPECTED_DIMENSION = 1024


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="migration_artifacts/full_registry_local_bge")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="Discard existing local vector/checkpoint artifacts")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    chunks_path = artifact_dir / "chunks.jsonl"
    vectors_path = artifact_dir / "vectors.npy"
    checkpoint_path = artifact_dir / "embedding_checkpoint.json"
    manifest_path = artifact_dir / "vector_manifest.json"

    if not chunks_path.exists():
        raise SystemExit(f"Missing validated chunk file: {chunks_path}")
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise SystemExit("REFUSED: chunk file is empty")

    ids = [chunk.get("id") for chunk in chunks]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise SystemExit("REFUSED: missing or duplicate chunk IDs")

    if args.force:
        for path in (vectors_path, checkpoint_path, manifest_path):
            if path.exists():
                path.unlink()

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    completed = int(checkpoint.get("completed_chunks", 0))
    stored_vectors = np.load(vectors_path) if vectors_path.exists() else np.empty((0, EXPECTED_DIMENSION), dtype=np.float32)
    if stored_vectors.ndim != 2 or stored_vectors.shape[1] != EXPECTED_DIMENSION:
        raise SystemExit(f"REFUSED: invalid existing vector shape: {stored_vectors.shape}")
    if stored_vectors.shape[0] != completed:
        raise SystemExit(
            f"REFUSED: checkpoint/vector mismatch: checkpoint={completed}, vectors={stored_vectors.shape[0]}"
        )
    if completed > len(chunks):
        raise SystemExit("REFUSED: checkpoint exceeds chunk count")

    if completed == len(chunks):
        norms = np.linalg.norm(stored_vectors, axis=1)
        print(json.dumps({
            "model": MODEL_NAME,
            "dimension": EXPECTED_DIMENSION,
            "chunk_count": len(chunks),
            "completed_chunks": completed,
            "mean_norm": round(float(norms.mean()), 6),
            "pinecone_written": False,
            "status": "already_complete",
        }, indent=2))
        return 0

    model = SentenceTransformer(MODEL_NAME)
    vectors = stored_vectors
    for start in range(completed, len(chunks), args.batch_size):
        batch = chunks[start:start + args.batch_size]
        encoded = model.encode(
            [chunk["text"] for chunk in batch],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        encoded = np.asarray(encoded, dtype=np.float32)
        if encoded.shape != (len(batch), EXPECTED_DIMENSION):
            raise SystemExit(f"REFUSED: invalid batch vector shape at {start}: {encoded.shape}")
        vectors = np.concatenate([vectors, encoded], axis=0)
        temporary_vectors = vectors_path.with_suffix(".npy.tmp")
        with temporary_vectors.open("wb") as handle:
            np.save(handle, vectors)
        temporary_vectors.replace(vectors_path)
        completed = start + len(batch)
        atomic_json(checkpoint_path, {
            "model": MODEL_NAME,
            "dimension": EXPECTED_DIMENSION,
            "completed_chunks": completed,
            "total_chunks": len(chunks),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pinecone_written": False,
        })
        print(f"Embedded {completed}/{len(chunks)} chunks")

    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit("REFUSED: generated vectors are not finite normalized 1024-dimensional vectors")
    if vectors.shape != (len(chunks), EXPECTED_DIMENSION):
        raise SystemExit(f"REFUSED: final vector shape: {vectors.shape}")

    atomic_json(manifest_path, {
        "model": MODEL_NAME,
        "dimension": EXPECTED_DIMENSION,
        "chunk_count": len(chunks),
        "vector_file": vectors_path.name,
        "mean_norm": round(float(norms.mean()), 6),
        "min_norm": round(float(norms.min()), 6),
        "max_norm": round(float(norms.max()), 6),
        "pinecone_written": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({
        "model": MODEL_NAME,
        "dimension": EXPECTED_DIMENSION,
        "chunk_count": len(chunks),
        "vector_shape": list(vectors.shape),
        "mean_norm": round(float(norms.mean()), 6),
        "pinecone_written": False,
        "status": "complete",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
