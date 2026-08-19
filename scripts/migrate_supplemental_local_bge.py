"""Fetch, locally embed, and optionally upload authoritative supplemental pages.

The manifest supplies explicit URL-first metadata. This script never uses
Pinecone inference and refuses the production index and benchmark namespace.
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
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_local_embeddings import MODEL_NAME, EXPECTED_DIMENSION, chunk_text
from scripts.migrate_html_registry_local_bge import extract_page_content

PRODUCTION_INDEX = "lbrce-index"
BENCHMARK_NAMESPACE = "lbrce_local_bge_v1"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_full_v1"
EXPECTED_PAGES = 21
USER_AGENT = "LBRCE-Supplemental-Local-BGE/1.0"

# When repairing the two curated pages, retain only the strongest content
# chunk for each topic. This prevents navigation-only or unrelated sections
# from competing with the authoritative fact under the same metadata filter.
TARGET_TOPIC_KEYWORDS = {
    "college_location": {
        "mylavaram": 8,
        "krishna district": 8,
        "how to reach": 7,
        "nearest bus station": 5,
        "nearest railway station": 5,
        "nearest airport": 5,
        "google map": 3,
    },
    "central_library": {
        "about central library": 8,
        "library @ a glance": 8,
        "working hours": 8,
        "65429": 8,
        "digital library": 6,
        "reading capacity": 5,
        "e-journals": 5,
        "e-books": 5,
    },
    "transportation": {
        "bus routes": 8,
        "bus fares": 8,
        "bus fee": 8,
        "college transport": 7,
        "bus facility": 6,
        "academic year 2026": 5,
        "31 buses": 5,
        "boarding": 3,
    },
}


def canonical_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, parsed.query, ""))


def stable_id(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:20]


def metadata_for(chunk: dict) -> dict:
    return {key: value for key, value in chunk.items() if key != "id" and value is not None and isinstance(value, (str, int, float, bool))}


def select_targeted_chunks(chunks: list[dict], topics: set[str]) -> list[dict]:
    """Select one content-bearing chunk for each requested curated topic."""
    selected: list[dict] = []
    for topic in sorted(topics):
        candidates = [chunk for chunk in chunks if chunk.get("topic") == topic]
        if not candidates:
            raise SystemExit(f"REFUSED: no supplemental chunks found for topic={topic!r}")
        weights = TARGET_TOPIC_KEYWORDS.get(topic, {})
        ranked = sorted(
            candidates,
            key=lambda chunk: (
                sum(weight for term, weight in weights.items() if term in chunk.get("text", "").lower()),
                len(chunk.get("text", "")),
            ),
            reverse=True,
        )
        best = ranked[0]
        score = sum(
            weight for term, weight in weights.items()
            if term in best.get("text", "").lower()
        )
        if weights and score <= 0:
            raise SystemExit(
                f"REFUSED: selected page for topic={topic!r} has no authoritative keyword match"
            )
        selected.append(best)
        print(
            f"Selected topic={topic} chunk_index={best.get('chunk_index')} "
            f"keyword_score={score} id={best.get('id')}"
        )
    return selected


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages") if isinstance(data, dict) else data
    if not isinstance(pages, list) or len(pages) != EXPECTED_PAGES:
        raise SystemExit(f"REFUSED: expected exactly {EXPECTED_PAGES} supplemental pages")
    urls = [canonical_url(page.get("url", "")) for page in pages]
    if any(not url.startswith("https://lbrce.ac.in") for url in urls):
        raise SystemExit("REFUSED: supplemental manifest contains a non-LBRCE URL")
    if len(urls) != len(set(urls)):
        raise SystemExit("REFUSED: duplicate supplemental canonical URLs")
    required = {"id", "url", "title", "page_category", "topic"}
    for page in pages:
        missing = required - set(page)
        if missing:
            raise SystemExit(f"REFUSED: supplemental page missing fields: {sorted(missing)}")
    return pages


def fetch_pages(
    pages: list[dict],
    output_dir: Path,
    timeout: float,
    delay: float,
    selected_topics: set[str] | None = None,
) -> list[dict]:
    session = requests.Session()
    prepared = []
    failures = []
    for number, page in enumerate(pages, start=1):
        url = page["url"]
        print(f"[{number}/{len(pages)}] Fetching {url}")
        try:
            response = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            title, text, headings = extract_page_content(response.text)
            if not text.strip():
                raise ValueError("empty extracted content")
            document_id = f"supp-html-bge-{stable_id(url)}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            base = {
                "document_id": document_id,
                "source_url": url,
                "canonical_url": canonical_url(url),
                "resolved_url": response.url,
                "title": page["title"],
                "page_category": page["page_category"],
                "topic": page["topic"],
                "department": page.get("department"),
                "academic_year": None,
                "source_type": "html",
                "resource_type": "html_page",
                "approved_source": True,
                "content_hash": content_hash,
            }
            for index, value in enumerate(chunk_text(text)):
                prepared.append({
                    "id": f"{document_id}-{index:05d}",
                    "text": value,
                    "chunk_index": index,
                    **base,
                    "manifest_id": page["id"],
                    "migration_embedding_model": MODEL_NAME,
                    "migration_version": "local_bge_supplement_v1",
                })
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
        if delay:
            import time
            time.sleep(delay)
    if failures:
        raise SystemExit(f"REFUSED: supplemental failures: {json.dumps(failures, indent=2)}")
    ids = [chunk["id"] for chunk in prepared]
    if len(ids) != len(set(ids)):
        raise SystemExit("REFUSED: duplicate supplemental chunk IDs")
    if selected_topics:
        prepared = select_targeted_chunks(prepared, selected_topics)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "supplemental_chunks.jsonl").write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in prepared), encoding="utf-8"
    )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="scripts/supplemental_authoritative_pages.json")
    parser.add_argument("--output-dir", default="migration_artifacts/full_registry_local_bge")
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--topics",
        nargs="+",
        choices=sorted(TARGET_TOPIC_KEYWORDS),
        help="Regenerate only one selected authoritative chunk per requested topic.",
    )
    parser.add_argument(
        "--replace-selected-topics",
        action="store_true",
        help="When used with --topics and confirmation, delete only the selected topics' existing vectors before upsert.",
    )
    parser.add_argument("--confirm-supplemental-migration", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: cannot target production lbrce-index")
    if args.namespace in {"", "default", BENCHMARK_NAMESPACE} or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: invalid supplemental namespace")

    pages = load_manifest(Path(args.manifest))
    selected_topics = set(args.topics or [])
    if selected_topics:
        pages = [page for page in pages if page.get("topic") in selected_topics]
        if len(pages) != len(selected_topics):
            found = {page.get("topic") for page in pages}
            missing = sorted(selected_topics - found)
            raise SystemExit(f"REFUSED: requested topic pages not found: {missing}")
    chunks = fetch_pages(
        pages,
        Path(args.output_dir),
        args.timeout,
        args.delay,
        selected_topics=selected_topics or None,
    )
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(chunks), EXPECTED_DIMENSION):
        raise SystemExit(f"REFUSED: invalid supplemental vector shape: {vectors.shape}")
    vector_path = Path(args.output_dir) / "supplemental_vectors.npy"
    np.save(vector_path, vectors)
    print(json.dumps({
        "supplemental_pages": len(pages),
        "supplemental_chunks": len(chunks),
        "vector_shape": list(vectors.shape),
        "model": MODEL_NAME,
        "dimension": EXPECTED_DIMENSION,
        "pinecone_inference_used": False,
        "target_index": args.index_name,
        "target_namespace": args.namespace,
    }, indent=2))

    if not args.confirm_supplemental_migration:
        print("DRY RUN: supplemental vectors created locally; no Pinecone connection or upsert performed")
        return 0
    if args.replace_selected_topics and not selected_topics:
        raise SystemExit("REFUSED: --replace-selected-topics requires --topics")

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
    if args.replace_selected_topics:
        document_ids = sorted({chunk["document_id"] for chunk in chunks})
        if len(document_ids) != len(selected_topics):
            raise SystemExit(
                "REFUSED: selected replacement did not resolve exactly one document per topic"
            )
        for document_id in document_ids:
            index.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=args.namespace,
            )
            print(f"Deleted existing vectors for document_id={document_id}")
    uploaded = 0
    for start in range(0, len(chunks), args.batch_size):
        payload = [
            {"id": chunk["id"], "values": vector.astype(float).tolist(), "metadata": metadata_for(chunk)}
            for chunk, vector in zip(chunks[start:start + args.batch_size], vectors[start:start + args.batch_size])
        ]
        index.upsert(vectors=payload, namespace=args.namespace)
        uploaded += len(payload)
        print(f"Uploaded {uploaded}/{len(chunks)} supplemental vectors")
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
