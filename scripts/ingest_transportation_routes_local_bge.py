"""Ingest official LBRCE 2026-27 transportation routes as route-specific local-BGE vectors.

This is a migration-only script. It fetches the official transportation HTML page,
parses one structured record per route, and never uses Pinecone hosted embeddings.
By default it prepares local JSONL artifacts without Pinecone writes. The confirmed
path deletes only existing vectors for the exact transportation source URL and then
upserts the route-specific records into the migration namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # Parsing-only preparation does not require the model package.
    SentenceTransformer = None

from scripts.benchmark_local_embeddings import EXPECTED_DIMENSION, MODEL_NAME

SOURCE_URL = "https://www.lbrce.ac.in/studentcorner_pages/transportation.php"
PRODUCTION_INDEX = "lbrce-index"
BENCHMARK_NAMESPACE = "lbrce_local_bge_v1"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_full_v1"
EXPECTED_ROUTE_COUNT = 41
USER_AGENT = "LBRCE-Transportation-Routes-Local-BGE/1.0"


def canonical_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, parsed.query, ""))


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def flat_metadata(record: dict) -> dict:
    """Keep Pinecone metadata scalar/list-compatible; never send nested dicts."""
    allowed = (str, int, float, bool)
    output = {}
    for key, value in record.items():
        if key == "id" or value is None:
            continue
        if isinstance(value, allowed):
            output[key] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            output[key] = value
    return output


def parse_route_tables(html: str, source_url: str = SOURCE_URL) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    canonical = canonical_url(source_url)
    document_prefix = f"transport-routes-bge-{stable_id(canonical)}"
    records: list[dict] = []

    for table in soup.find_all("table"):
        heading = table.find_previous("h3")
        route_code = " ".join(heading.get_text(" ", strip=True).split()).upper() if heading else ""
        if not route_code or not ((route_code[0] in {"J", "S"} and route_code[1:].isdigit()) or route_code == "SB"):
            continue

        rows = []
        for row in table.find_all("tr"):
            cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in row.find_all(["th", "td"])]
            if not cells or cells[0].lower().startswith("s.no"):
                continue
            if len(cells) >= 4 and cells[0].isdigit():
                rows.append({
                    "number": cells[0],
                    "point": cells[1],
                    "fee": cells[2],
                    "start_time": cells[3],
                })

        if not rows:
            continue

        lines = [
            f"### {route_code}",
            "| S.No. | Route Point | Bus Fee | Start Time |",
            "|---:|---|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['number']} | {row['point']} | {row['fee']} | {row['start_time']} |"
            )
        text = (
            f"Official LBRCE College Transportation bus route {route_code}, "
            "Academic Year 2026-27.\n\n" + "\n".join(lines)
        )
        records.append({
            "id": f"{document_prefix}-{route_code.lower()}",
            "document_id": f"{document_prefix}-{route_code.lower()}",
            "source_url": source_url,
            "canonical_url": canonical,
            "title": f"LBRCE Bus Route {route_code} and Bus Fare 2026-27",
            "page_category": "transportation",
            "topic": "transportation",
            "route_code": route_code,
            "route_points": [row["point"] for row in rows],
            "stop_count": len(rows),
            "academic_year": "2026-27",
            "source_type": "html",
            "resource_type": "transportation_route",
            "approved_source": True,
            "text": text,
            "migration_embedding_model": MODEL_NAME,
            "migration_version": "local_bge_transport_routes_v1",
        })

    records.sort(key=lambda record: (record["route_code"] != "SB", record["route_code"]))
    if len(records) != EXPECTED_ROUTE_COUNT:
        raise SystemExit(
            f"REFUSED: expected {EXPECTED_ROUTE_COUNT} route tables, found {len(records)}"
        )
    if {record["route_code"] for record in records} != {
        *(f"J{number:02d}" for number in range(1, 21)),
        *(f"S{number:02d}" for number in range(1, 21)),
        "SB",
    }:
        raise SystemExit("REFUSED: route-code set is incomplete or unexpected")
    return records


def fetch_and_prepare(output_dir: Path, timeout: float) -> list[dict]:
    response = requests.get(
        SOURCE_URL,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    response.raise_for_status()
    records = parse_route_tables(response.text, response.url)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transportation_route_chunks.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="migration_artifacts/transportation_routes_local_bge")
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--confirm-transportation-migration", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: cannot target production lbrce-index")
    if args.namespace in {"", "default", BENCHMARK_NAMESPACE} or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: invalid transportation namespace")

    output_dir = Path(args.output_dir)
    records = fetch_and_prepare(output_dir, args.timeout)
    print(json.dumps({
        "source_url": SOURCE_URL,
        "route_count": len(records),
        "route_codes": [record["route_code"] for record in records],
        "target_index": args.index_name,
        "target_namespace": args.namespace,
        "pinecone_inference_used": False,
    }, indent=2))

    if SentenceTransformer is None:
        if args.confirm_transportation_migration:
            raise SystemExit("sentence-transformers is required for confirmed local-BGE upsert")
        print("PREPARE-ONLY: sentence-transformers is not installed; route parsing completed")
        return 0

    model = SentenceTransformer(MODEL_NAME)
    vectors = np.asarray(
        model.encode(
            [record["text"] for record in records],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if vectors.shape != (len(records), EXPECTED_DIMENSION):
        raise SystemExit(f"REFUSED: unexpected local-BGE vector shape {vectors.shape}")
    np.save(output_dir / "transportation_route_vectors.npy", vectors)

    if not args.confirm_transportation_migration:
        print("DRY RUN: route vectors created locally; no Pinecone writes performed")
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
    # Exact canonical URL replacement removes the old summary vector and any
    # previous route-vector attempt without touching other transportation data.
    index.delete(
        filter={"canonical_url": {"$eq": canonical_url(SOURCE_URL)}},
        namespace=args.namespace,
    )
    print(f"Deleted existing transportation vectors for canonical_url={canonical_url(SOURCE_URL)}")

    uploaded = 0
    for start in range(0, len(records), args.batch_size):
        payload = [
            {
                "id": record["id"],
                "values": vector.astype(float).tolist(),
                "metadata": flat_metadata(record),
            }
            for record, vector in zip(records[start:start + args.batch_size], vectors[start:start + args.batch_size])
        ]
        index.upsert(vectors=payload, namespace=args.namespace)
        uploaded += len(payload)
        print(f"Uploaded {uploaded}/{len(records)} transportation route vectors")

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
