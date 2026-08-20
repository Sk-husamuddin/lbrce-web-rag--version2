"""Backfill missing metadata on existing R23 regulation PDF vectors.

This script never re-embeds or re-upserts vectors. It finds existing vectors by
exact source URL and uses Pinecone update(set_metadata=...) only. It reads the
actual migration index and namespace from the project .env, refuses production,
and requires explicit confirmation before updates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRODUCTION_INDEX = "lbrce-index"
PRODUCTION_NAMESPACE = "lbrce_local_bge_v1"
REQUIRED_PAGE_CATEGORY = "regulation_directory"
REQUIRED_TOPIC = "regulations"
EXPECTED_TOTAL = 4


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("pdfs") if isinstance(data, dict) else data
    if not isinstance(records, list) or len(records) != 4:
        raise SystemExit("REFUSED: regulation manifest must contain exactly four pdfs")
    for record in records:
        required = {"url", "title", "regulation", "document_type", "resource_type", "page_category", "topic"}
        missing = required - set(record)
        if missing:
            raise SystemExit(f"REFUSED: manifest record missing fields {sorted(missing)}")
        if record["page_category"] != REQUIRED_PAGE_CATEGORY or record["topic"] != REQUIRED_TOPIC:
            raise SystemExit("REFUSED: manifest regulation metadata does not match the approved filter")
    return records


def query_ids(index, namespace: str, source_url: str, dimension: int, top_k: int = 100) -> list[dict]:
    result = index.query(
        vector=np.zeros(dimension, dtype=np.float32).tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter={"source_url": {"$eq": source_url}},
    )
    matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
    found = []
    for match in matches:
        if isinstance(match, dict):
            found.append(match)
        else:
            found.append({"id": match.id, "metadata": getattr(match, "metadata", {})})
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="scripts/r23_regulations_manifest.json")
    parser.add_argument("--confirm-backfill", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    namespace = os.getenv("PINECONE_NAMESPACE")
    if not api_key or not index_name or not namespace:
        raise SystemExit("REFUSED: PINECONE_API_KEY, PINECONE_INDEX_NAME, and PINECONE_NAMESPACE are required in .env")
    if index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: production index lbrce-index is never allowed")
    if namespace in {"", "default", PRODUCTION_NAMESPACE} or "prod" in namespace.lower():
        raise SystemExit(f"REFUSED: unsafe namespace {namespace!r}")

    records = load_manifest(PROJECT_ROOT / args.manifest)
    pc = Pinecone(api_key=api_key)
    description = pc.describe_index(index_name)
    dimension = int(description.dimension)
    if dimension != 1024 or str(description.metric).lower() != "cosine":
        raise SystemExit("REFUSED: expected a 1024-dimensional cosine migration index")
    index = pc.Index(index_name)

    summary = []
    all_ids: list[str] = []
    for record in records:
        matches = query_ids(index, namespace, record["url"], dimension, args.top_k)
        ids = [match["id"] for match in matches if match.get("id")]
        all_ids.extend(ids)
        summary.append({"url": record["url"], "title": record["title"], "matched_vectors": len(ids), "ids": ids})

    total = sum(item["matched_vectors"] for item in summary)
    print(json.dumps({
        "index": index_name,
        "namespace": namespace,
        "mode": "backfill" if args.confirm_backfill else "dry_run",
        "expected_total": EXPECTED_TOTAL,
        "matched_total": total,
        "by_url": summary,
        "metadata_patch": {"page_category": REQUIRED_PAGE_CATEGORY, "topic": REQUIRED_TOPIC},
        "embedding_changed": False,
        "upsert_performed": False,
    }, indent=2))

    if total < EXPECTED_TOTAL:
        raise SystemExit(f"REFUSED: expected at least {EXPECTED_TOTAL} URL-only regulation vectors, found {total}")
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("REFUSED: duplicate vector IDs appeared across regulation source URLs")
    for item in summary:
        if item["matched_vectors"] < 1:
            raise SystemExit(f"REFUSED: no existing URL-only vector found for {item['url']}")

    if not args.confirm_backfill:
        print("DRY RUN: no metadata updates performed")
        return 0

    patched = 0
    for vector_id in all_ids:
        index.update(
            id=vector_id,
            set_metadata={"page_category": REQUIRED_PAGE_CATEGORY, "topic": REQUIRED_TOPIC},
            namespace=namespace,
        )
        patched += 1
    print(json.dumps({
        "patched": patched,
        "index": index_name,
        "namespace": namespace,
        "metadata_only": True,
        "reembedded": False,
        "upsert_performed": False,
    }, indent=2))

    verify = index.query(
        vector=np.zeros(dimension, dtype=np.float32).tolist(),
        top_k=100,
        include_metadata=True,
        namespace=namespace,
        filter={
            "$and": [
                {"page_category": {"$eq": REQUIRED_PAGE_CATEGORY}},
                {"topic": {"$eq": REQUIRED_TOPIC}},
            ]
        },
    )
    verify_matches = verify.get("matches", []) if isinstance(verify, dict) else getattr(verify, "matches", [])
    print(json.dumps({"verified_filter_matches": len(verify_matches), "expected": EXPECTED_TOTAL}, indent=2))
    if len(verify_matches) < EXPECTED_TOTAL:
        raise SystemExit(f"REFUSED: post-backfill verification returned fewer than {EXPECTED_TOTAL} matching vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
