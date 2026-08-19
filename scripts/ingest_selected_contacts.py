"""Selective ingestion of approved LBRCE department contact pages.

This script intentionally processes only the URLs in
selected_department_contacts.json. It fetches and chunks the HTML text, embeds
it with the configured Pinecone inference model, and upserts the chunks into the
existing Pinecone index. It does not process PDFs, images, or any other pages.

Successful URLs are checkpointed in scripts/department_contact_ingestion_registry.json,
so rerunning the command does not re-fetch or re-embed completed pages unless
--force is supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("selected-contact-ingestion")

DEFAULT_MANIFEST = Path(__file__).with_name("selected_department_contacts.json")
REGISTRY_FILE = Path(__file__).with_name("department_contact_ingestion_registry.json")
ALLOWED_HOSTS = {"lbrce.ac.in", "www.lbrce.ac.in"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Manifest must contain a non-empty 'pages' array")
    for index, page in enumerate(pages):
        if not isinstance(page, dict) or not page.get("url"):
            raise ValueError(f"Manifest page {index} must contain a URL")
        parsed = urlparse(str(page["url"]))
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"Manifest page {index} is outside the approved HTTPS LBRCE domain")
    return data


def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Could not load contact registry: %s", exc)
        return {}


def save_registry(registry: dict) -> None:
    temporary = REGISTRY_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(REGISTRY_FILE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest approved LBRCE department contact pages only")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", nargs="*", default=None, help="Optional department keys, e.g. ece cse csm")
    parser.add_argument("--force", action="store_true", help="Reprocess pages marked successful")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse only; do not embed or upsert")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between pages in seconds")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    pages = manifest["pages"]
    if args.only:
        requested = {value.lower() for value in args.only}
        pages = [page for page in pages if str(page.get("department", "")).lower() in requested]
        if not pages:
            raise SystemExit("No manifest pages matched --only")

    registry = load_registry()
    parser = HTMLParser(
        base_url=(settings.LBRCE_BASE_URL if settings else "https://www.lbrce.ac.in"),
        use_cache=True,
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

    total_vectors = 0
    for number, record in enumerate(pages, start=1):
        url = str(record["url"])
        previous = registry.get(url, {})
        if previous.get("status") == "success" and not args.force:
            logger.info("[%d/%d] Skipping checkpointed contact page: %s", number, len(pages), url)
            continue

        logger.info("[%d/%d] Fetching contact page: %s", number, len(pages), url)
        try:
            document = await parser.fetch_and_parse(url)
            if document is None or not document.content.strip():
                raise RuntimeError("page returned no parsed text")

            # Add explicit manifest metadata without replacing parser metadata.
            document.department = record.get("department") or document.department
            document.metadata = {
                **(document.metadata or {}),
                "selected_resource": "true",
                "resource_type": "department_contact_page",
                "department_key": record.get("department", ""),
                "manifest_title": record.get("title", ""),
                "ingested_at": now_utc(),
            }
            chunks = chunk_parsed_document(document)
            if not chunks:
                raise RuntimeError("page produced no chunks")

            if args.dry_run:
                logger.info("[%d/%d] dry-run parsed %d chunks: %s", number, len(pages), len(chunks), url)
            else:
                vectors = embedder.embed_chunks(chunks)
                indexer.upsert_chunks(chunks, vectors)
                total_vectors += len(chunks)
                logger.info("[%d/%d] upserted %d vectors: %s", number, len(pages), len(chunks), url)

            registry[url] = {
                "status": "dry_run" if args.dry_run else "success",
                "department": record.get("department"),
                "title": record.get("title"),
                "chunk_count": len(chunks),
                "updated_at": now_utc(),
            }
        except Exception as exc:
            logger.exception("[%d/%d] contact-page ingestion failed for %s", number, len(pages), url)
            registry[url] = {
                "status": "failed",
                "department": record.get("department"),
                "title": record.get("title"),
                "error": str(exc),
                "updated_at": now_utc(),
            }
        finally:
            save_registry(registry)
            if args.delay > 0:
                time.sleep(args.delay)

    # HTMLParser exposes its async client through the context-manager lifecycle;
    # close the lazily-created client explicitly after the sequential run.
    client = getattr(parser, "_client", None)
    if client is not None:
        await client.aclose()
    logger.info("Contact-page ingestion complete. New vectors upserted: %d", total_vectors)


if __name__ == "__main__":
    asyncio.run(main())
