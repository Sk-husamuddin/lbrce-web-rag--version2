"""Fetch, chunk, and upsert only approved selected HTML pages."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import settings
from backend.embedding import get_configured_embedding_generator
from backend.indexing.pinecone_indexer import PineconeIndexer
from backend.ingestion.chunker import chunk_parsed_document
from backend.ingestion.html_parser import HTMLParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("selected-html-ingestion")

DEFAULT_MANIFEST = Path(__file__).with_name("selected_placement_pages.json")
DEFAULT_REGISTRY = Path(__file__).with_name("selected_html_ingestion_registry.json")
ALLOWED_HOSTS = {"lbrce.ac.in", "www.lbrce.ac.in"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Manifest must contain a non-empty pages array")
    for index, page in enumerate(pages):
        url = str(page.get("url", ""))
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"Page {index} is outside the approved HTTPS LBRCE domain: {url}")
    return pages


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def save_registry(path: Path, registry: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest selected LBRCE HTML pages only")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--force", action="store_true", help="Reprocess already successful pages")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse without embedding or upserting")
    parser.add_argument(
        "--replace-resource-type",
        action="store_true",
        help=(
            "Delete existing Pinecone vectors whose resource_type matches the manifest "
            "before re-ingesting. Use only with --force and an approved selected manifest."
        ),
    )
    parser.add_argument("--delay", type=float, default=0.5)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    pages = load_manifest(args.manifest)
    registry = load_registry(args.registry)
    parser = HTMLParser(base_url=(settings.LBRCE_BASE_URL if settings else "https://www.lbrce.ac.in"), use_cache=False)
    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_resource_type = str(manifest_data.get("resource_type") or "selected_html")

    if args.replace_resource_type and args.dry_run:
        raise ValueError("--replace-resource-type cannot be combined with --dry-run")
    if args.replace_resource_type and manifest_resource_type != "student_list_html":
        raise ValueError(
            "--replace-resource-type is restricted to the student_list_html manifest"
        )

    embedder = None
    indexer = None
    if not args.dry_run:
        if settings is None:
            raise RuntimeError("Backend settings are unavailable; configure the project .env first")
        embedder = get_configured_embedding_generator()
        indexer = PineconeIndexer(
            api_key=settings.PINECONE_API_KEY,
            environment="",
            index_name=settings.PINECONE_INDEX_NAME,
            dimension=int(getattr(settings, "EMBEDDING_DIMENSION", 1024)),
            metric="cosine",
            namespace=getattr(settings, "PINECONE_NAMESPACE", ""),
        )
        if args.replace_resource_type:
            logger.warning(
                "Deleting existing Pinecone vectors with resource_type=%s before replacement.",
                manifest_resource_type,
            )
            indexer.index.delete(filter={"resource_type": manifest_resource_type})
            logger.info(
                "Existing %s vectors deleted; the selected manifest will be re-ingested.",
                manifest_resource_type,
            )

    total_vectors = 0
    try:
        for number, record in enumerate(pages, start=1):
            url = str(record["url"])
            if registry.get(url, {}).get("status") == "success" and not args.force:
                logger.info("[%d/%d] Skipping checkpointed page: %s", number, len(pages), url)
                continue
            try:
                document = await parser.fetch_and_parse(url)
                if document is None or not document.content.strip():
                    raise RuntimeError("page returned no parsed text")
                document.metadata = {
                    **(document.metadata or {}),
                    "selected_resource": "true",
                    "resource_type": manifest_resource_type,
                    "manifest_title": record.get("title", ""),
                    **{
                        key: value
                        for key, value in record.items()
                        if key in {"department", "academic_year", "term", "semester", "section"}
                        and value not in (None, "")
                    },
                    "ingested_at": now_utc(),
                }
                chunks = chunk_parsed_document(document)
                if not chunks:
                    raise RuntimeError("page produced no chunks")
                if args.dry_run:
                    logger.info("[%d/%d] dry-run parsed %d chunks: %s", number, len(pages), len(chunks), url)
                    status = "dry_run"
                else:
                    vectors = embedder.embed_chunks(chunks)
                    indexer.upsert_chunks(chunks, vectors)
                    total_vectors += len(chunks)
                    logger.info("[%d/%d] upserted %d vectors: %s", number, len(pages), len(chunks), url)
                    status = "success"
                registry[url] = {"status": status, "title": record.get("title", ""), "chunk_count": len(chunks), "updated_at": now_utc()}
            except Exception as exc:
                logger.exception("[%d/%d] failed for %s", number, len(pages), url)
                registry[url] = {"status": "failed", "title": record.get("title", ""), "error": str(exc), "updated_at": now_utc()}
            finally:
                save_registry(args.registry, registry)
                if args.delay > 0:
                    time.sleep(args.delay)
    finally:
        client = getattr(parser, "_client", None)
        if client is not None:
            await client.aclose()
    logger.info("Selected HTML ingestion complete. New vectors upserted: %d", total_vectors)


if __name__ == "__main__":
    asyncio.run(main())
