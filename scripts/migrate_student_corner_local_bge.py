"""Targeted local-BGE migration for missing LBRCE Student Corner pages.

The script fetches only the pages in student_corner_manifest.json, applies the
same malformed-HTML fallback and chunker used by the corrected HTML migration,
embeds locally with BAAI/bge-large-en-v1.5, and performs no Pinecone write unless
--confirm-student-corner-migration is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # preparation-only mode does not require the model package
    SentenceTransformer = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_local_embeddings import MODEL_NAME, EXPECTED_DIMENSION, chunk_text
from scripts.migrate_html_registry_local_bge import extract_page_content

PRODUCTION_INDEX = "lbrce-index"
BENCHMARK_NAMESPACE = "lbrce_local_bge_v1"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_full_v1"
USER_AGENT = "LBRCE-Student-Corner-Local-BGE/1.0"


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


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages") if isinstance(data, dict) else data
    if not isinstance(pages, list) or not pages:
        raise SystemExit("REFUSED: Student Corner manifest has no pages")
    urls = [canonical_url(str(page.get("url", ""))) for page in pages]
    if any(not url.startswith("https://lbrce.ac.in") for url in urls):
        raise SystemExit("REFUSED: manifest contains a non-LBRCE URL")
    if len(urls) != len(set(urls)):
        raise SystemExit("REFUSED: manifest contains duplicate canonical URLs")
    required = {"id", "url", "title", "page_category", "topic"}
    for page in pages:
        missing = required - set(page)
        if missing:
            raise SystemExit(f"REFUSED: page {page.get('id')} missing {sorted(missing)}")
    return pages


def prepare_pages(pages: list[dict], output_dir: Path, timeout: float, delay: float) -> list[dict]:
    session = requests.Session()
    prepared: list[dict] = []
    failures: list[dict] = []
    for number, page in enumerate(pages, start=1):
        url = page["url"]
        print(f"[{number}/{len(pages)}] Fetching {url}")
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            title, text, headings = extract_page_content(response.text)
            if not text.strip():
                raise ValueError("empty extracted content")
            chunks = chunk_text(text)
            if not chunks:
                raise ValueError("zero chunks")
            document_id = f"student-corner-bge-{stable_id(url)}"
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            for index, value in enumerate(chunks):
                prepared.append(
                    {
                        "id": f"{document_id}-{index:05d}",
                        "text": value,
                        "chunk_index": index,
                        "document_id": document_id,
                        "source_url": url,
                        "canonical_url": canonical_url(url),
                        "resolved_url": response.url,
                        "title": page["title"],
                        "page_category": page["page_category"],
                        "topic": page["topic"],
                        "resource_type": "student_corner_html",
                        "source_type": "html",
                        "approved_source": True,
                        "content_hash": content_hash,
                        "headings": " | ".join(headings[:20]),
                        "migration_embedding_model": MODEL_NAME,
                        "migration_version": "student_corner_local_bge_v1",
                    }
                )
            if delay:
                import time
                time.sleep(delay)
        except Exception as exc:
            failures.append({"url": url, "error": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "page_count": len(pages),
        "chunk_count": len(prepared),
        "failure_count": len(failures),
        "failures": failures,
        "pinecone_written": False,
        "ready_for_embedding": not failures and bool(prepared),
    }
    (output_dir / "student_corner_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"REFUSED: Student Corner extraction failures: {json.dumps(failures, indent=2)}")
    ids = [chunk["id"] for chunk in prepared]
    if len(ids) != len(set(ids)):
        raise SystemExit("REFUSED: duplicate Student Corner vector IDs")
    (output_dir / "student_corner_chunks.jsonl").write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in prepared),
        encoding="utf-8",
    )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="scripts/student_corner_manifest.json")
    parser.add_argument("--output-dir", default="migration_artifacts/student_corner_local_bge")
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prepare-only", action="store_true", help="Fetch, extract, chunk, and audit without loading the embedding model")
    parser.add_argument("--confirm-student-corner-migration", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: cannot target production lbrce-index")
    if args.namespace in {"", "default", BENCHMARK_NAMESPACE} or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: invalid Student Corner namespace")

    pages = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    chunks = prepare_pages(pages, output_dir, args.timeout, args.delay)
    if args.prepare_only:
        print(json.dumps({
            "student_corner_pages": len(pages),
            "student_corner_chunks": len(chunks),
            "pinecone_inference_used": False,
            "pinecone_written": False,
            "mode": "prepare_only",
        }, indent=2))
        return 0
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers is required for embedding; install it or use --prepare-only")
    print(json.dumps({
        "student_corner_pages": len(pages),
        "student_corner_chunks": len(chunks),
        "model": MODEL_NAME,
        "dimension": EXPECTED_DIMENSION,
        "pinecone_inference_used": False,
        "target_index": args.index_name,
        "target_namespace": args.namespace,
    }, indent=2))

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
        raise SystemExit(f"REFUSED: invalid Student Corner vector shape {vectors.shape}")
    np.save(output_dir / "student_corner_vectors.npy", vectors)

    if not args.confirm_student_corner_migration:
        print("DRY RUN: Student Corner vectors created locally; no Pinecone connection or upsert performed")
        return 0

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is missing")
    pc = Pinecone(api_key=api_key)
    names = {item["name"] if isinstance(item, dict) else item.name for item in pc.list_indexes()}
    if args.index_name not in names:
        raise SystemExit(f"REFUSED: target index {args.index_name} does not exist")
    index = pc.Index(args.index_name)
    for start in range(0, len(chunks), args.batch_size):
        batch_chunks = chunks[start:start + args.batch_size]
        batch_vectors = vectors[start:start + args.batch_size]
        payload = []
        for chunk, vector in zip(batch_chunks, batch_vectors):
            metadata = {key: value for key, value in chunk.items() if key not in {"id", "text"} and value is not None}
            metadata["text"] = chunk["text"]
            payload.append({"id": chunk["id"], "values": vector.tolist(), "metadata": metadata})
        index.upsert(vectors=payload, namespace=args.namespace)
        print(f"Upserted Student Corner batch {start // args.batch_size + 1} ({len(payload)} vectors)")
    print(json.dumps({
        "student_corner_pages": len(pages),
        "student_corner_chunks": len(chunks),
        "uploaded": len(chunks),
        "index": args.index_name,
        "namespace": args.namespace,
        "pinecone_inference_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
