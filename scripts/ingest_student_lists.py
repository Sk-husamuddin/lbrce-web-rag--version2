"""Ingest official LBRCE student-list tables without flattening HTML structure.

Use --dry-run first. Use --replace-resource-type only when replacing the old
student_list_html corpus; it deletes only that metadata class before upserting
the newly parsed roster chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.settings import settings
from backend.embedding import get_configured_embedding_generator
from backend.indexing.pinecone_indexer import PineconeIndexer
from backend.ingestion.student_list_parser import fetch_student_list_html, parse_student_list_html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("student-list-ingestion")

DEFAULT_MANIFEST = PROJECT_ROOT / "scripts" / "selected_student_list_pages.json"
RESOURCE_TYPE = "student_list_html"
EMBED_BATCH = 16
EMBED_DELAY = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest official LBRCE student-list tables")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="Re-fetch and reprocess every approved page")
    parser.add_argument("--department", type=str, default=None, help="Optional department key, for example cse")
    parser.add_argument("--dry-run", action="store_true", help="Parse and count chunks without embedding/upserting")
    parser.add_argument(
        "--replace-resource-type",
        action="store_true",
        help="Delete existing resource_type=student_list_html vectors before upserting",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between page fetches")
    return parser.parse_args()


def load_pages(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("resource_type") != RESOURCE_TYPE:
        raise ValueError(f"Manifest resource_type must be {RESOURCE_TYPE!r}")
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Manifest must contain a non-empty pages array")
    return pages


def embed_and_upsert(chunks, embedder, indexer) -> int:
    total = 0
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vectors = embedder.embed_chunks(batch)
        indexer.upsert_chunks(batch, vectors)
        total += len(batch)
        logger.info("Upserted roster batch %d (%d vectors)", start // EMBED_BATCH + 1, len(batch))
        if start + EMBED_BATCH < len(chunks):
            time.sleep(EMBED_DELAY)
    return total


async def main() -> None:
    args = parse_args()
    pages = load_pages(args.manifest)
    if args.department:
        department = args.department.strip().lower()
        pages = [page for page in pages if str(page.get("department") or "").lower() == department]
        if not pages:
            raise ValueError(f"No manifest page found for department {args.department!r}")
    if args.replace_resource_type and args.dry_run:
        raise ValueError("--replace-resource-type cannot be combined with --dry-run")
    if args.replace_resource_type and not args.force:
        raise ValueError("--replace-resource_type requires --force so every page is rebuilt")

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
            if args.department:
                logger.warning(
                    "Deleting existing %s vectors for department=%s before replacement",
                    RESOURCE_TYPE,
                    args.department,
                )
                indexer.index.delete(
                    filter={
                        "$and": [
                            {"resource_type": RESOURCE_TYPE},
                            {"department": args.department.strip().lower()},
                        ]
                    }
                )
            else:
                logger.warning("Deleting existing %s vectors before replacement", RESOURCE_TYPE)
                indexer.index.delete(filter={"resource_type": RESOURCE_TYPE})

    total_chunks = 0
    total_upserted = 0
    for number, page in enumerate(pages, start=1):
        url = str(page["url"])
        title = str(page.get("title") or "LBRCE Student List")
        department = str(page.get("department") or "")
        html = await fetch_student_list_html(url)
        chunks = parse_student_list_html(
            html,
            url=url,
            title=title,
            department=department,
        )
        total_chunks += len(chunks)
        logger.info("[%d/%d] %s -> %d row-safe chunks", number, len(pages), url, len(chunks))
        if not args.dry_run:
            total_upserted += embed_and_upsert(chunks, embedder, indexer)
        if args.delay:
            await asyncio.sleep(args.delay)

    logger.info("Student-list ingestion complete: %d chunks, %d vectors upserted", total_chunks, total_upserted)


if __name__ == "__main__":
    asyncio.run(main())
