"""
LBRCE Safe Ingestion Script
============================
Design:
  1. Crawl HTML pages (max_pages=50, already implemented).
  2. Immediately chunk + embed + upsert HTML chunks → fast baseline in Pinecone.
  3. Filter discovered PDFs: skip image/xlsx/doc/etc links, keep only .pdf URLs.
  4. Process PDFs in small batches (PDF_BATCH=20), downloading, parsing, chunking,
     embedding, and upserting each batch before moving to the next.
  5. Log running vector count after every batch.

This means partial progress is preserved even if killed mid-run.

CHANGELOG (this revision):
  - Added an optional department filter for PDFs (--dept), e.g. `--dept cse`,
    so a run can ingest only one department's PDFs instead of everything.
  - Added a persisted "already-chunked HTML pages" registry so that HTML
    pages that were already chunked + embedded + upserted in a previous run
    are skipped entirely on subsequent runs (no re-fetch, no re-chunk, no
    re-embed) — avoiding wasted time re-processing ~400+ unchanged pages.
  - NEW: --max-pages / --max-pdfs / --max-images let a single run override
    MAX_HTML_PAGES / MAX_PDFS / MAX_IMAGES without editing constants, so a
    scoped test run (e.g. 400 pages / 200 CSE PDFs / 200 CSE images) can be
    driven entirely from the CLI.
  - NEW: --dept now also scopes Phase 4 (images), not just Phase 3 (PDFs).
    Previously there was no way to restrict timetable images to a single
    department -- Phase 4 only filtered by TIMETABLE_KEYWORDS across every
    department's discovered images.
"""

import sys
import os
import logging
import asyncio
import time
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.config.settings import settings
from backend.ingestion.crawler import LBRCECrawler
from backend.ingestion.pdf_parser import parse_pdf_from_url
from backend.ingestion.image_parser import ImageParser
from backend.ingestion.chunker import chunk_parsed_document, chunk_pdf_document, chunk_image_document
from backend.embedding import get_configured_embedding_generator
from backend.indexing.pinecone_indexer import PineconeIndexer
from backend.ingestion.selected_resources import (
    load_selected_manifest,
    timetable_chunks,
    timetable_pdf_chunks,
    pdf_url_chunks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Tunables (defaults -- override per-run via --max-pages/--max-pdfs/--max-images) ──
MAX_HTML_PAGES = 300       # how many HTML pages to crawl (was 50 — too shallow to
                            # reach department/timetable/academics subpages)
MAX_PDFS       = 400       # cap on PDFs to process (most-recently discovered first)
PDF_BATCH      = 20        # PDFs to process per embed+upsert round
MAX_IMAGES     = 100       # safety ceiling AFTER keyword filtering below —
                            # with filtering in place this should rarely bind
IMAGE_BATCH    = 10        # images per embed+upsert round -- smaller than PDF_BATCH
                            # since each vision-LLM call is slower/costlier than
                            # a local PDF text extraction

# Only images whose URL contains one of these (case-insensitive) get processed.
# The site's discovered images are dominated by NPTEL certificates, achievement
# badges, and event photos (see crawl log) -- none of that is useful for a
# student-facing RAG chatbot, and running all ~950 of them through a vision
# LLM would waste time/rate-limit budget for zero value.
#
# Two independent signals, since either alone could miss a real timetable:
#   - "timetable" matches the folder path (e.g. .../cse_timetables/...)
#   - "tt_ay" matches the filename convention actually used on-site
#     (e.g. V_SEM_F_SEC_TT_AY_2026-27.jpg) -- catches cases where a
#     timetable image sits in a folder that doesn't literally say
#     "timetable" in its name.
TIMETABLE_KEYWORDS = ["timetable", "time_table", "time-table", "ttable", "tt_ay"]

# ── Department filter ────────────────────────────────────────────────────
# Keyword sets per department, used to restrict PDF ingestion (and, as of
# this revision, image ingestion) to a single department via --dept <name>.
# Matches on URL path segment (e.g. "/cse/"), subdomain (e.g.
# cse.lbrce.ac.in), and filename prefixes department docs commonly use
# (e.g. "CSE_LESSON_PLAN...", "R23_CSE_Syllabus.pdf").
#
# NOTE: CSE, CSM (CSE-AI&ML) and AI (CSE-AI&DS) are separate departments on
# this site with overlapping naming ("CSE(AI&ML)" mentions "CSE" in the
# filename). EXCLUDE_KEYWORDS strips those false positives back out so
# --dept cse doesn't pull in CSM/AI docs just because "cse" appears in the
# combined program name.
DEPARTMENT_KEYWORDS = {
    "cse":   {
        "include": ["/cse/", "cse.lbrce.ac.in", "csedept", "cse_"],
        "exclude": ["csm", "cse(ai&ml)", "cse_ai&ml", "ai&ds", "cse-ai&ml"],
    },
    "csm":   {"include": ["/csm/", "csm_"], "exclude": []},
    "ai":    {"include": ["/ai/", "ai_"], "exclude": ["mail", "said", "train"]},
    "civil": {"include": ["/civil/", "civil_"], "exclude": []},
    "eee":   {"include": ["/eee/", "eee_"], "exclude": []},
    "ece":   {"include": ["/ece/", "ece_"], "exclude": []},
    "it":    {"include": ["/it/", "it_"], "exclude": []},
    "mech":  {"include": ["/mech/", "mech_"], "exclude": []},
    "mba":   {"include": ["/mba/", "mba_"], "exclude": []},
    "ase":   {"include": ["/ase/", "ase_"], "exclude": []},
}


def is_department_url(url: str, dept: str) -> bool:
    """True if the URL looks like it belongs to the given department."""
    cfg = DEPARTMENT_KEYWORDS.get(dept.lower())
    if cfg is None:
        raise ValueError(
            f"Unknown --dept '{dept}'. Known departments: "
            f"{', '.join(sorted(DEPARTMENT_KEYWORDS))}"
        )
    lowered = url.lower()
    if any(bad in lowered for bad in cfg["exclude"]):
        return False
    return any(good in lowered for good in cfg["include"])


# Discovered via lbrce.ac.in/academic_pages/timetables.php -- the site's own
# "Department Wise Timetables" hub page. Seeding the crawler here directly,
# in addition to the homepage, guarantees every department's timetable
# subpage is reached within a couple of hops, instead of depending on
# whether the general BFS crawl happens to reach it before its page budget
# runs out (CSE's section is far more link-rich than e.g. EEE/ECE, so a
# homepage-only crawl reliably finds CSE timetables but can miss others).
TIMETABLE_HUB_URLS = [
    "https://www.lbrce.ac.in/ase/asetimetables.php",
    "https://www.lbrce.ac.in/ai/aitimetables.php",
    "https://www.lbrce.ac.in/civil/civiltimetables.php",
    "https://www.lbrce.ac.in/cse/csetimetables.php",
    "https://www.lbrce.ac.in/csm/csmtimetables.php",
    "https://www.lbrce.ac.in/eee/eeetimetables.php",
    "https://www.lbrce.ac.in/ece/ecetimetables.php",
    "https://www.lbrce.ac.in/it/ittimetables.php",
    "https://www.lbrce.ac.in/mech/mechtimetables.php",
    "https://www.lbrce.ac.in/mba/mbatimetables.php",
]
EMBED_BATCH    = 16        # conservative texts per Pinecone inference call
EMBED_MAX_RETRIES   = 3     # retries on 429 RESOURCE_EXHAUSTED before leaving a group pending
EMBED_RETRY_BASE_WAIT = 60  # seconds; allow the per-minute token window to clear
EMBED_BATCH_DELAY  = 30     # seconds between calls; actual chunk sizes vary widely

# Crawl results cache — lets subsequent runs skip the crawl phase entirely
# (--from-cache) and/or process only one content type (--images-only,
# --pdfs-only) without re-discovering URLs from scratch every time.
CACHE_FILE = Path(__file__).parent / "ingestion_cache.json"

# Registry of HTML page URLs that have ALREADY been chunked + embedded +
# upserted in a previous run. On each run, any parsed page whose URL is
# already in this file is skipped in Phase 2 — it is never re-chunked,
# re-embedded, or re-upserted. This is what makes repeat runs fast: once
# ~400+ stable pages (nav, static department pages, etc.) have been chunked
# once, they're never touched again unless you pass --force-rechunk.
CHUNKED_HTML_REGISTRY_FILE = Path(__file__).parent / "chunked_html_pages.json"
# Image extraction and indexing checkpoint. This stores OCR text locally so a
# later run can reuse it without calling the vision model again.
IMAGE_REGISTRY_FILE = Path(__file__).parent / "image_ingestion_registry.json"
# PDF processing checkpoint. Successful and permanently failed URLs are kept
# out of future runs unless explicitly retried or force-reprocessed.
PDF_REGISTRY_FILE = Path(__file__).parent / "pdf_ingestion_registry.json"
# Selected timetable metadata checkpoint. The image pixels are never sent to
# Gemma; only the small metadata chunks are embedded.
TIMETABLE_REGISTRY_FILE = Path(__file__).parent / "timetable_ingestion_registry.json"
# ──────────────────────────────────────────────────────────────────────────────


def is_pdf_url(url: str) -> bool:
    """Only process URLs that end in .pdf (case-insensitive)."""
    return url.lower().split("?")[0].endswith(".pdf")


def is_timetable_url(url: str) -> bool:
    """True if the URL looks like a timetable image based on filename/path."""
    lowered = url.lower()
    return any(keyword in lowered for keyword in TIMETABLE_KEYWORDS)


def save_crawl_cache(parsed_pages, raw_pdf_urls, raw_image_urls) -> None:
    """
    Persist crawl results to disk so future runs can skip Phase 1 entirely
    via --from-cache, or process only one content type via --images-only /
    --pdfs-only without re-crawling. HTML page *content* isn't cached (only
    URLs) since HTML chunks are already embedded+upserted by the time this
    is called — only PDF/image URLs are needed for a later selective run.
    """
    data = {
        "html_page_count": len(parsed_pages),
        "pdf_urls": list(raw_pdf_urls),
        "image_urls": list(raw_image_urls),
    }
    CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Saved crawl cache to {CACHE_FILE} "
                f"({len(data['pdf_urls'])} PDF URLs, {len(data['image_urls'])} image URLs).")


def load_crawl_cache() -> dict:
    """Load a previously saved crawl cache. Raises if the file doesn't exist."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"No cache file at {CACHE_FILE} — run once without --from-cache first "
            f"to generate it."
        )
    data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    logger.info(f"Loaded crawl cache: {len(data['pdf_urls'])} PDF URLs, "
                f"{len(data['image_urls'])} image URLs "
                f"(from a crawl of {data['html_page_count']} HTML pages).")
    return data


def load_chunked_html_registry() -> set:
    """
    Load the set of HTML page URLs already chunked+embedded+upserted in a
    previous run. Returns an empty set if no registry exists yet (first run).
    """
    if not CHUNKED_HTML_REGISTRY_FILE.exists():
        return set()
    data = json.loads(CHUNKED_HTML_REGISTRY_FILE.read_text(encoding="utf-8"))
    urls = set(data.get("chunked_urls", []))
    logger.info(f"Loaded chunked-HTML registry: {len(urls)} pages already "
                f"chunked in previous runs (will be skipped).")
    return urls


def save_chunked_html_registry(urls: set) -> None:
    """Persist the updated set of chunked HTML page URLs."""
    data = {"chunked_urls": sorted(urls)}
    CHUNKED_HTML_REGISTRY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Saved chunked-HTML registry: {len(urls)} pages total.")


def load_image_registry() -> dict:
    """Load image OCR/indexing checkpoints, returning an empty registry initially."""
    if not IMAGE_REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(IMAGE_REGISTRY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("registry root must be an object")
        return data
    except Exception as exc:
        logger.warning("Could not load image registry %s: %s", IMAGE_REGISTRY_FILE, exc)
        return {}


def save_image_registry(registry: dict) -> None:
    """Atomically persist image OCR text and indexing status."""
    temp_file = IMAGE_REGISTRY_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(IMAGE_REGISTRY_FILE)


def image_hash(image_data: bytes) -> str:
    """Return a stable content hash for a downloaded image."""
    return hashlib.sha256(image_data).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pdf_registry() -> dict:
    """Load PDF processing checkpoints, returning an empty registry initially."""
    if not PDF_REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(PDF_REGISTRY_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("registry root must be an object")
        return data
    except Exception as exc:
        logger.warning("Could not load PDF registry %s: %s", PDF_REGISTRY_FILE, exc)
        return {}


def save_pdf_registry(registry: dict) -> None:
    """Atomically persist PDF processing checkpoints."""
    temp_file = PDF_REGISTRY_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(PDF_REGISTRY_FILE)


def load_timetable_registry() -> dict:
    """Load selected timetable metadata checkpoints."""
    if not TIMETABLE_REGISTRY_FILE.exists():
        return {}
    try:
        data = json.loads(TIMETABLE_REGISTRY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Could not load timetable registry %s: %s", TIMETABLE_REGISTRY_FILE, exc)
        return {}


def save_timetable_registry(registry: dict) -> None:
    """Atomically persist selected timetable metadata checkpoints."""
    temp_file = TIMETABLE_REGISTRY_FILE.with_suffix(".tmp")
    temp_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_file.replace(TIMETABLE_REGISTRY_FILE)


def parse_args():
    parser = argparse.ArgumentParser(description="LBRCE ingestion pipeline")
    parser.add_argument(
        "--from-cache", action="store_true",
        help="Skip Phase 1 crawl entirely; load PDF/image URLs from a previous run's cache."
    )
    parser.add_argument(
        "--selected-resources", type=Path, default=None,
        help="Process only the timetable and PDF records in a selected JSON manifest."
    )
    parser.add_argument(
        "--timetables-only", action="store_true",
        help="With --selected-resources, ingest only selected timetable metadata chunks."
    )
    parser.add_argument(
        "--force-reprocess-timetables", action="store_true",
        help=(
            "With --selected-resources, ignore successful timetable checkpoints "
            "and re-embed/upsert all selected timetable metadata records."
        ),
    )
    parser.add_argument(
        "--images-only", action="store_true",
        help="Only run Phase 4 (images). Implies --from-cache. Skips HTML and PDF phases."
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Run HTML/PDF ingestion but skip Phase 4 entirely; no vision API calls are made."
    )
    parser.add_argument(
        "--pdfs-only", action="store_true",
        help="Only run Phase 3 (PDFs). Implies --from-cache. Skips HTML and image phases."
    )
    parser.add_argument(
        "--skip-html", action="store_true",
        help="Skip HTML crawl + embedding (still crawls to discover PDF/image URLs unless --from-cache is also set)."
    )
    parser.add_argument(
        "--dept", type=str, default=None,
        help=(
            "Restrict PDF ingestion (Phase 3) AND image ingestion (Phase 4) "
            "to a single department, e.g. --dept cse. Known departments: "
            + ", ".join(sorted(DEPARTMENT_KEYWORDS))
        ),
    )
    parser.add_argument(
        "--force-rechunk", action="store_true",
        help=(
            "Ignore the chunked-HTML registry and re-chunk/re-embed/re-upsert "
            "every HTML page discovered this run, even ones already processed "
            "in a previous run."
        ),
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help=f"Override the HTML page crawl cap for this run (default {MAX_HTML_PAGES})."
    )
    parser.add_argument(
        "--max-pdfs", type=int, default=None,
        help=f"Override the PDF processing cap for this run (default {MAX_PDFS})."
    )
    parser.add_argument(
        "--retry-failed-pdfs", action="store_true",
        help="Retry PDF URLs previously recorded as failed or empty."
    )
    parser.add_argument(
        "--force-reprocess-pdfs", action="store_true",
        help="Ignore the PDF registry and reprocess every selected PDF URL."
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help=f"Override the image processing cap for this run (default {MAX_IMAGES})."
    )
    parser.add_argument(
        "--recheck-images", action="store_true",
        help=(
            "Re-download already-successful images and compare their content hash; "
            "unchanged images still skip the vision call."
        ),
    )
    parser.add_argument(
        "--force-reprocess-images", action="store_true",
        help=(
            "Ignore successful image checkpoints and call the vision model again. "
            "Use only after changing the vision model or extraction prompt."
        ),
    )
    return parser.parse_args()


def embed_and_upsert(chunks, embedder, indexer, label: str) -> int:
    """Embed and immediately upsert conservative sub-batches.

    A successful sub-batch is written before the next embedding call begins.
    If a later sub-batch fails, earlier vectors remain safely indexed, while a
    zero return tells the caller not to mark the enclosing page as complete.
    """
    if not chunks:
        logger.info(f"[{label}] No chunks to embed.")
        return 0

    total_upserted = 0
    had_failure = False
    num_sub_batches = (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH

    for i in range(0, len(chunks), EMBED_BATCH):
        sub = chunks[i : i + EMBED_BATCH]
        sub_batch_num = i // EMBED_BATCH + 1
        vecs = None

        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                vecs = embedder.embed_chunks(sub)
                break
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if is_rate_limit and attempt < EMBED_MAX_RETRIES:
                    wait_s = EMBED_RETRY_BASE_WAIT * attempt
                    logger.warning(
                        f"[{label}] Sub-batch {sub_batch_num}/{num_sub_batches} rate-limited "
                        f"(attempt {attempt}/{EMBED_MAX_RETRIES}) — retrying in {wait_s}s"
                    )
                    time.sleep(wait_s)
                    continue
                logger.error(
                    f"[{label}] Sub-batch {sub_batch_num}/{num_sub_batches} failed "
                    f"(attempt {attempt}/{EMBED_MAX_RETRIES}): {exc}"
                )
                had_failure = True
                break

        if vecs is None:
            continue

        try:
            indexer.upsert_chunks(sub, vecs)
            total_upserted += len(sub)
            logger.info(
                f"[{label}] Upserted sub-batch {sub_batch_num}/{num_sub_batches} "
                f"({len(sub)} vectors)."
            )
        except Exception as exc:
            logger.error(
                f"[{label}] Upsert failed for sub-batch "
                f"{sub_batch_num}/{num_sub_batches}: {exc}"
            )
            had_failure = True
            break

        if i + EMBED_BATCH < len(chunks):
            time.sleep(EMBED_BATCH_DELAY)

    if had_failure:
        logger.error(
            f"[{label}] Incomplete group: {total_upserted} vectors were saved, "
            "but the enclosing page/batch will remain pending for retry."
        )
        return 0

    return total_upserted


async def main():
    args = parse_args()
    selected_manifest = None
    if args.selected_resources:
        selected_manifest = load_selected_manifest(args.selected_resources)
        if args.from_cache or args.images_only or args.pdfs_only or args.skip_html:
            raise SystemExit(
                "--selected-resources cannot be combined with --from-cache, "
                "--images-only, --pdfs-only, or --skip-html."
            )
        if args.timetables_only and not selected_manifest.get("timetables"):
            raise SystemExit("The selected manifest contains no timetable records.")

    if args.images_only and args.skip_images:
        raise SystemExit("--images-only and --skip-images cannot be used together.")

    if args.dept and args.dept.lower() not in DEPARTMENT_KEYWORDS:
        raise SystemExit(
            f"Unknown --dept '{args.dept}'. Known departments: "
            f"{', '.join(sorted(DEPARTMENT_KEYWORDS))}"
        )

    # Per-run overrides of the module-level tunables (defaults preserved).
    max_html_pages = args.max_pages if args.max_pages is not None else MAX_HTML_PAGES
    max_pdfs = args.max_pdfs if args.max_pdfs is not None else MAX_PDFS
    max_images = args.max_images if args.max_images is not None else MAX_IMAGES

    # --images-only / --pdfs-only imply --from-cache: there is no reason to
    # process only one content type but still redo the crawl to get its URLs.
    use_cache = args.from_cache or args.images_only or args.pdfs_only or bool(selected_manifest)
    run_html  = not (args.skip_html or args.images_only or args.pdfs_only or selected_manifest)
    run_pdfs  = not args.images_only and not args.timetables_only
    run_images = not (args.pdfs_only or args.skip_images or selected_manifest)

    embedder = get_configured_embedding_generator()
    indexer  = PineconeIndexer(
        api_key=settings.PINECONE_API_KEY,
        environment="",
        index_name=settings.PINECONE_INDEX_NAME,
        dimension=int(getattr(settings, "EMBEDDING_DIMENSION", 1024)),
        namespace=getattr(settings, "PINECONE_NAMESPACE", ""),
    )
    image_parser = None
    if run_images:
        # Initialize only when the image phase is enabled. This keeps
        # --pdfs-only runs independent from optional vision/image dependencies.
        image_parser = ImageParser()  # picks up OPENROUTER_API_KEY(_2) from env
    total_vectors = 0

    if use_cache:
        # ── Skip Phase 1 entirely — reuse a previous run's discovered URLs ───
        if selected_manifest:
            url_only_resource_types = {
                "timetable_pdf",
                "regulation_pdf",
                "syllabus_pdf",
                "academic_syllabus_pdf",
                "exam_results_pdf",
            }
            selected_pdf_records = selected_manifest.get("pdfs", [])
            raw_pdf_urls = [
                record["url"]
                for record in selected_pdf_records
                if str(record.get("resource_type") or "").strip().lower()
                not in url_only_resource_types
                and not record.get("url_only")
            ]
            raw_image_urls = []
            logger.info(
                "Selected-resource mode: %d PDFs and %d timetable records loaded from %s.",
                len(raw_pdf_urls),
                len(selected_manifest.get("timetables", [])),
                args.selected_resources,
            )
        else:
            cache = load_crawl_cache()
            raw_pdf_urls = cache["pdf_urls"]
            raw_image_urls = cache["image_urls"]
    else:
        # ── Phase 1: Crawl HTML ───────────────────────────────────────────
        logger.info(f"Phase 1 — Crawling HTML pages (cap {max_html_pages})…")
        crawler = LBRCECrawler(
            seed_urls=[settings.LBRCE_BASE_URL] + TIMETABLE_HUB_URLS,
            max_pages=max_html_pages,
            request_delay=0.5,
        )
        parsed_pages, raw_pdf_urls, raw_image_urls = await crawler.crawl()
        logger.info(
            f"Crawled {len(parsed_pages)} HTML pages, discovered {len(raw_pdf_urls)} raw PDF URLs, "
            f"{len(raw_image_urls)} raw image URLs."
        )
        save_crawl_cache(parsed_pages, raw_pdf_urls, raw_image_urls)

        if run_html:
            # ── Phase 2: Chunk + embed + upsert HTML immediately ───────────
            # Skip pages already chunked in a previous run (unless
            # --force-rechunk is passed). This is what avoids re-processing
            # the 400+ pages that don't change run-to-run.
            logger.info("Phase 2 — Chunking and upserting HTML pages…")

            already_chunked = set() if args.force_rechunk else load_chunked_html_registry()

            def page_url(page) -> str:
                # LBRCECrawler.crawl() returns a List[ParsedDocument] (see
                # backend/ingestion/html_parser.py); the field holding the
                # page's URL on that object is `.source_url`.
                return page.source_url

            new_pages = [p for p in parsed_pages if page_url(p) not in already_chunked]
            skipped_count = len(parsed_pages) - len(new_pages)
            if skipped_count:
                logger.info(
                    f"Skipping {skipped_count} HTML pages already chunked in a "
                    f"previous run ({len(new_pages)} new/changed pages remain)."
                )

            total_html_chunks = 0
            logger.info(
                f"HTML pages to process: {len(new_pages)}; "
                f"checkpointing one page at a time."
            )

            # Process and checkpoint each page independently. A rate limit or
            # interruption can therefore redo at most the current page rather
            # than the entire 3,000+ chunk corpus.
            for page_num, page in enumerate(new_pages, start=1):
                url = page_url(page)
                page_chunks = chunk_parsed_document(page)
                total_html_chunks += len(page_chunks)
                logger.info(
                    f"HTML page {page_num}/{len(new_pages)}: "
                    f"{len(page_chunks)} chunks — {url}"
                )

                if not page_chunks:
                    already_chunked.add(url)
                    save_chunked_html_registry(already_chunked)
                    continue

                count = embed_and_upsert(page_chunks, embedder, indexer, f"HTML-page-{page_num}")
                if count == 0:
                    logger.error(
                        "Stopping HTML phase after an incomplete page; "
                        "successful earlier pages are already checkpointed."
                    )
                    break

                total_vectors += count
                already_chunked.add(url)
                save_chunked_html_registry(already_chunked)
                logger.info(
                    f"Checkpointed HTML page {page_num}/{len(new_pages)}; "
                    f"running vector total: {total_vectors}"
                )

            logger.info(
                f"HTML chunks visited this run: {total_html_chunks}; "
                f"HTML registry now contains {len(already_chunked)} pages."
            )

    if selected_manifest:
        # ── Selective timetable metadata phase ─────────────────────────────
        # Only the structured metadata is embedded. Original image URLs are
        # stored in Pinecone metadata for later visual-resource responses.
        timetable_registry = load_timetable_registry()
        selected_chunks = (
            timetable_chunks(selected_manifest)
            + timetable_pdf_chunks(selected_manifest)
            + pdf_url_chunks(selected_manifest)
        )
        pending_chunks = [
            chunk for chunk in selected_chunks
            if args.force_reprocess_timetables
            or timetable_registry.get(chunk.document_id, {}).get("status") != "success"
        ]
        logger.info(
            "Selected URL/resource metadata: %d total, %d pending.",
            len(selected_chunks),
            len(pending_chunks),
        )
        if pending_chunks:
            count = embed_and_upsert(
                pending_chunks, embedder, indexer, "Selected-timetable-metadata"
            )
            if count == 0:
                raise SystemExit(
                    "Selected timetable metadata ingestion was incomplete; "
                    "completed records remain checkpointed for retry."
                )
            for chunk in pending_chunks:
                timetable_registry[chunk.document_id] = {
                    "status": "success",
                    "source_url": chunk.source_url,
                    "image_url": chunk.metadata.get("image_url"),
                    "pdf_url": chunk.metadata.get("pdf_url"),
                    "resource_type": chunk.metadata.get("resource_type"),
                    "vector_count": 1,
                    "updated_at": now_utc(),
                }
            save_timetable_registry(timetable_registry)
            total_vectors += count
            logger.info("Checkpointed %d selected timetable metadata records.", len(pending_chunks))
        else:
            logger.info("All selected URL/resource metadata records are already indexed.")

    if run_pdfs:
        # ── Phase 3: Filter + process PDFs with persistent checkpoints ─────
        pdf_urls = [u for u in raw_pdf_urls if is_pdf_url(u)]
        logger.info(f"Filtered to {len(pdf_urls)} .pdf URLs (dropped non-PDF links).")

        if args.dept:
            before = len(pdf_urls)
            pdf_urls = [u for u in pdf_urls if is_department_url(u, args.dept)]
            logger.info(
                f"Filtered down to {len(pdf_urls)} '{args.dept}' department PDF "
                f"URLs (from {before})."
            )

        pdf_registry = load_pdf_registry()
        selected_pdf_metadata = {
            record["url"]: record
            for record in (selected_manifest or {}).get("pdfs", [])
        }
        terminal_statuses = {"success", "failed", "empty"}
        if args.force_reprocess_pdfs:
            pending_pdf_urls = pdf_urls
        elif args.retry_failed_pdfs:
            pending_pdf_urls = [
                url for url in pdf_urls
                if pdf_registry.get(url, {}).get("status") != "success"
            ]
        else:
            pending_pdf_urls = [
                url for url in pdf_urls
                if pdf_registry.get(url, {}).get("status") not in terminal_statuses
            ]

        skipped_pdfs = len(pdf_urls) - len(pending_pdf_urls)
        if skipped_pdfs:
            logger.info(
                "Skipping %s PDFs already recorded in the PDF registry.", skipped_pdfs
            )

        if len(pending_pdf_urls) > max_pdfs:
            logger.info(f"Capping pending PDFs at {max_pdfs} for this run.")
            pending_pdf_urls = pending_pdf_urls[:max_pdfs]

        logger.info(
            f"Phase 3 — Processing {len(pending_pdf_urls)} pending PDFs "
            "with per-document checkpoints…"
        )
        for pdf_num, url in enumerate(pending_pdf_urls, start=1):
            logger.info(f"PDF {pdf_num}/{len(pending_pdf_urls)}: {url}")
            pdf_chunks = []
            try:
                docs = parse_pdf_from_url(url, timeout=20.0)
                if docs:
                    for doc in docs:
                        pdf_chunks.extend(chunk_pdf_document(doc))

                if selected_pdf_metadata.get(url):
                    manifest_record = selected_pdf_metadata[url]
                    for chunk in pdf_chunks:
                        chunk.metadata.update({
                            "document_type": manifest_record.get("document_type"),
                            "regulation": manifest_record.get("regulation"),
                            "priority": manifest_record.get("priority"),
                            "selected_resource": "true",
                            "page_category": manifest_record.get("page_category"),
                            "topic": manifest_record.get("topic"),
                        })

                if not pdf_chunks:
                    pdf_registry[url] = {
                        "source_url": url,
                        "status": "empty",
                        "vector_count": 0,
                        "updated_at": now_utc(),
                    }
                    save_pdf_registry(pdf_registry)
                    logger.warning("No indexable text found in PDF: %s", url)
                    continue

                count = embed_and_upsert(pdf_chunks, embedder, indexer, f"PDF-{pdf_num}")
                if count == 0:
                    # A partial embedding/upsert is left pending so a later
                    # run can retry it; stop to avoid hammering a rate limit.
                    logger.error(
                        "Stopping PDF phase after an incomplete document; "
                        "successfully completed PDFs remain checkpointed."
                    )
                    break

                pdf_registry[url] = {
                    "source_url": url,
                    "status": "success",
                    "vector_count": count,
                    "updated_at": now_utc(),
                }
                save_pdf_registry(pdf_registry)
                total_vectors += count
                logger.info(
                    f"Checkpointed PDF {pdf_num}/{len(pending_pdf_urls)} "
                    f"({count} vectors); running vector total: {total_vectors}"
                )
            except Exception as exc:
                pdf_registry[url] = {
                    "source_url": url,
                    "status": "failed",
                    "vector_count": 0,
                    "error": str(exc),
                    "updated_at": now_utc(),
                }
                save_pdf_registry(pdf_registry)
                logger.warning("Skipping PDF %s: %s", url, exc)
    else:
        logger.info("Skipping PDF phase (--images-only).")

    if run_images:
        # ── Phase 4: Filter + process images (timetables etc.) in batches ──
        image_urls = [u for u in raw_image_urls if is_timetable_url(u)]
        logger.info(
            f"Filtered {len(raw_image_urls)} discovered images down to "
            f"{len(image_urls)} timetable-matching images (keywords: {TIMETABLE_KEYWORDS})."
        )

        if args.dept:
            before = len(image_urls)
            image_urls = [u for u in image_urls if is_department_url(u, args.dept)]
            logger.info(
                f"Filtered down to {len(image_urls)} '{args.dept}' department "
                f"timetable images (from {before})."
            )

        image_registry = load_image_registry()
        current_model = ImageParser.VISION_MODEL
        current_prompt = ImageParser.PROMPT_VERSION

        # Apply the cap after removing successful checkpoints. This means a
        # later --max-images 50 run processes up to 50 pending images rather
        # than spending the cap on images that were already completed.
        if args.force_reprocess_images or args.recheck_images:
            pending_urls = image_urls
        else:
            pending_urls = [
                url for url in image_urls
                if image_registry.get(url, {}).get("status") != "success"
            ]
            skipped_success = len(image_urls) - len(pending_urls)
            if skipped_success:
                logger.info(
                    "Skipping %s images already marked successful in the image registry.",
                    skipped_success,
                )

        if len(pending_urls) > max_images:
            logger.info(f"Capping pending images at {max_images} for this run.")
            pending_urls = pending_urls[:max_images]

        logger.info(
            f"Phase 4 — Processing {len(pending_urls)} pending images in batches of {IMAGE_BATCH}…"
        )
        for batch_start in range(0, len(pending_urls), IMAGE_BATCH):
            batch = pending_urls[batch_start : batch_start + IMAGE_BATCH]
            batch_num = batch_start // IMAGE_BATCH + 1
            logger.info(
                f"Image batch {batch_num}: {len(batch)} images "
                f"({batch_start+1}–{batch_start+len(batch)} of {len(pending_urls)})"
            )

            image_chunks = []
            ready_images = []
            for url in batch:
                entry = image_registry.get(url, {})
                try:
                    import httpx
                    response = httpx.get(url, timeout=30.0, follow_redirects=True)
                    response.raise_for_status()
                    raw_image = response.content
                    digest = image_hash(raw_image)

                    cached_text = entry.get("extracted_text")
                    cache_matches = (
                        bool(cached_text)
                        and entry.get("sha256") == digest
                        and entry.get("model") == current_model
                        and entry.get("prompt_version") == current_prompt
                    )

                    if (
                        not args.force_reprocess_images
                        and entry.get("status") == "success"
                        and cache_matches
                    ):
                        logger.info("Skipping unchanged successful image: %s", url)
                        continue

                    if not args.force_reprocess_images and cache_matches:
                        logger.info("Reusing cached OCR text without a vision call: %s", url)
                        doc = ImageParser.document_from_text(
                            cached_text, url, extraction_method="cached_vision_llm"
                        )
                    else:
                        doc = image_parser.parse(raw_image, url)
                        if doc:
                            image_registry[url] = {
                                "status": "extracted",
                                "source_url": url,
                                "sha256": digest,
                                "extracted_text": doc.content,
                                "model": current_model,
                                "prompt_version": current_prompt,
                                "updated_at": now_utc(),
                            }
                            save_image_registry(image_registry)

                    if doc:
                        chunks = chunk_image_document(doc)
                        if chunks:
                            image_chunks.extend(chunks)
                            ready_images.append((url, digest, len(chunks)))
                        else:
                            logger.warning("No chunks produced from image: %s", url)
                    else:
                        image_registry[url] = {
                            "status": "failed",
                            "source_url": url,
                            "sha256": digest,
                            "model": current_model,
                            "prompt_version": current_prompt,
                            "error": "No content extracted",
                            "updated_at": now_utc(),
                        }
                        save_image_registry(image_registry)
                        logger.warning("No content extracted from image: %s", url)
                except Exception as exc:
                    # Preserve extracted OCR text if indexing/download fails so
                    # a later run can retry Pinecone without calling Gemma.
                    existing = image_registry.get(url, {})
                    existing.update({
                        "source_url": url,
                        "status": existing.get("status", "failed"),
                        "error": str(exc),
                        "updated_at": now_utc(),
                    })
                    image_registry[url] = existing
                    save_image_registry(image_registry)
                    logger.warning("Skipping image %s: %s", url, exc)

            count = embed_and_upsert(image_chunks, embedder, indexer, f"Image-batch-{batch_num}")
            total_vectors += count

            # Only mark entries successful after the whole batch has been
            # embedded and upserted. If embedding is partially rate-limited,
            # entries remain extracted and their OCR text is reused next time.
            if count > 0 and len(ready_images) > 0:
                for url, digest, vector_count in ready_images:
                    image_registry[url] = {
                        **image_registry.get(url, {}),
                        "status": "success",
                        "sha256": digest,
                        "vector_count": vector_count,
                        "updated_at": now_utc(),
                    }
                save_image_registry(image_registry)
                logger.info("Checkpointed %s successfully indexed images.", len(ready_images))

            logger.info(f"Running vector total: {total_vectors}")
    else:
        logger.info("Skipping image phase (--pdfs-only).")

    logger.info(f"Ingestion complete. Total vectors upserted this run: {total_vectors}")


if __name__ == "__main__":
    asyncio.run(main())