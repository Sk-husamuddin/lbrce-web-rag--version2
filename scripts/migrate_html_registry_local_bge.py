"""Registry-driven full LBRCE HTML migration using local BGE embeddings.

The supplied ``chunked_html_pages.json`` is the only page source. The script:

1. Validates exactly 488 registry entries.
2. Canonicalizes lbrce.ac.in/www.lbrce.ac.in aliases without silently losing them.
3. Fetches every unique canonical page and records failures.
4. Extracts leaf-safe, page-specific content and deterministic metadata.
5. Writes a complete dry-run manifest before any embedding or Pinecone write.
6. Optionally generates local BGE vectors and upserts only to the dedicated
   full-migration namespace with per-page checkpoints.

It refuses the production ``lbrce-index`` and never calls Pinecone inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # HTML preparation and parser imports do not require embeddings
    SentenceTransformer = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_local_embeddings import (
    MODEL_NAME,
    EXPECTED_DIMENSION,
    classify_category,
    classify_department,
    classify_topic,
    extract_academic_year,
    extract_text,
    chunk_text,
)

logger = logging.getLogger("registry_local_bge_migration")

PRODUCTION_INDEX = "lbrce-index"
BENCHMARK_NAMESPACE = "lbrce_local_bge_v1"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_full_v1"
DEFAULT_REGISTRY_ENTRIES = 488
USER_AGENT = "LBRCE-Full-Registry-Migration/1.0"

# Common navigation labels that can survive extraction even when the page body
# was missed because of malformed legacy HTML. These are used only for audit
# and quarantine decisions, never as retrieval metadata.
_NAVIGATION_LABELS = {
    "home", "administration", "lbrce trust", "founder chairman",
    "honorary chairman", "chairman", "vice-chairman", "president",
    "principal", "vice-principal", "governing body", "academic council",
    "board of studies (bos)", "finance committee", "strategic plan",
    "organisation structure", "dean academics", "academic regulations",
    "academic calendars", "lesson plans", "course structure & syllabus",
    "students list", "timetables", "faculty", "service rules", "rti act",
    "careers", "college brochure", "contact", "quick links",
    "mandatory disclosure", "aicte approvals", "ugc recognitions",
    "jntu affiliations", "nba", "nirf", "financial audit statements",
}


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, parsed.query, ""))


def load_registry(path: Path, expected_entries: int, excluded_urls: set[str] | None = None) -> tuple[list[str], dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    urls = data.get("chunked_urls") if isinstance(data, dict) else data
    if not isinstance(urls, list):
        raise SystemExit("REFUSED: registry must contain a chunked_urls list")
    excluded_urls = excluded_urls or set()
    original_count = len(urls)
    urls = [url for url in urls if canonical_url(url) not in excluded_urls]
    if len(urls) != expected_entries:
        raise SystemExit(
            f"REFUSED: expected exactly {expected_entries} registry entries after exclusions, found {len(urls)} "
            f"(original registry count: {original_count})"
        )
    if any(not isinstance(url, str) or not url.startswith(("http://", "https://")) for url in urls):
        raise SystemExit("REFUSED: registry contains a non-HTTP URL")
    groups: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        groups[canonical_url(url)].append(url)
    return urls, dict(groups)


def stable_id(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def fetch_html(session: requests.Session, url: str, timeout: float) -> tuple[str, str, int]:
    response = session.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "html" not in content_type and "text/plain" not in content_type:
        raise ValueError(f"unexpected content type: {content_type}")
    return response.text, response.url, response.status_code


def extract_page_content(html: str) -> tuple[str, str, list[str]]:
    """Use the project extractor, then a conservative visible-text fallback."""
    title, text, headings = extract_text(html)
    if text.strip():
        return title, text, headings

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    headings = [node.get_text(" ", strip=True) for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]) if node.get_text(" ", strip=True)]
    body = soup.body or soup
    leaf_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "pre"]
    leaves = []
    for node in body.find_all(leaf_tags):
        if node.find(leaf_tags):
            continue
        value = " ".join(node.get_text(" ", strip=True).split())
        if len(value) > 2:
            leaves.append(value)
    if not leaves:
        value = " ".join(body.get_text(" ", strip=True).split())
        if len(value) > 2:
            leaves.append(value)
    deduped = []
    for value in leaves:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    return title, "\n\n".join(deduped), headings


def build_page_record(
    original_urls: list[str],
    resolved_url: str,
    title: str,
    text: str,
    headings: list[str],
) -> dict:
    canonical = canonical_url(original_urls[0])
    category = classify_category(original_urls[0], title, text)
    department = classify_department(original_urls[0], title, text)
    topic = classify_topic(original_urls[0], category)
    academic_year = extract_academic_year(text, category, original_urls[0])
    return {
        "document_id": f"html-bge-{stable_id(canonical)}",
        "source_url": original_urls[0],
        "source_aliases": original_urls,
        "resolved_url": resolved_url,
        "canonical_url": canonical,
        "title": title,
        "page_category": category,
        "topic": topic,
        "department": department,
        "academic_year": academic_year,
        "source_type": "html",
        "resource_type": "html_page",
        "approved_source": True,
        "heading_count": len(headings),
        "content_chars": len(text),
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def assess_page_quality(url: str, title: str, text: str, chunks: list[dict]) -> str | None:
    """Return a quarantine reason for empty or navigation-only extraction.

    The full-document fallback normally recovers malformed pages. This final
    guard prevents a page that still contains only shared navigation labels
    from entering the vector corpus as if it were authoritative content.
    """
    if not chunks:
        return "zero_chunks"
    if len(text.strip()) < 80:
        return "too_short"

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    non_navigation = [line for line in lines if line not in _NAVIGATION_LABELS]
    has_substantive_line = any(len(line) >= 120 for line in non_navigation)
    if len(text) < 2000 and len(non_navigation) < 4 and not has_substantive_line:
        return "navigation_only"
    return None


def build_chunks(page_record: dict, text: str) -> list[dict]:
    chunks = []
    for chunk_index, chunk in enumerate(chunk_text(text)):
        chunks.append({
            "id": f"{page_record['document_id']}-{chunk_index:05d}",
            "text": chunk,
            "source_url": page_record["source_url"],
            "source_aliases": json.dumps(page_record["source_aliases"], ensure_ascii=False),
            "resolved_url": page_record["resolved_url"],
            "canonical_url": page_record["canonical_url"],
            "title": page_record["title"],
            "source_type": "html",
            "resource_type": "html_page",
            "page_category": page_record["page_category"],
            "topic": page_record["topic"],
            "department": page_record["department"],
            "academic_year": page_record["academic_year"],
            "document_id": page_record["document_id"],
            "chunk_index": chunk_index,
            "content_hash": page_record["content_hash"],
            "approved_source": True,
            "migration_embedding_model": MODEL_NAME,
            "migration_version": "local_bge_registry_v1",
        })
    return chunks


def validate_chunk_metadata(chunks: list[dict]) -> None:
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise SystemExit("REFUSED: duplicate stable chunk IDs detected")
    required = [
        "id", "text", "source_url", "canonical_url", "resource_type",
        "page_category", "topic", "document_id", "chunk_index",
    ]
    for chunk in chunks:
        missing = [key for key in required if chunk.get(key) is None or chunk.get(key) == ""]
        if missing:
            raise SystemExit(f"REFUSED: {chunk['id']} missing metadata: {missing}")
        if chunk["resource_type"] != "html_page":
            raise SystemExit(f"REFUSED: unexpected resource type: {chunk['resource_type']}")


def fetch_and_prepare(
    registry_path: Path,
    output_dir: Path,
    timeout: float,
    delay: float,
    from_cache: bool,
    retry_failed: bool = False,
    expected_entries: int = DEFAULT_REGISTRY_ENTRIES,
    excluded_urls: set[str] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    urls, groups = load_registry(registry_path, expected_entries, excluded_urls)
    cache_path = output_dir / "fetched_pages.jsonl"
    page_records_path = output_dir / "page_metadata.json"
    chunk_path = output_dir / "chunks.jsonl"
    audit_path = output_dir / "registry_audit.json"

    cached_pages = load_jsonl(cache_path) if cache_path.exists() else []
    cached_records = json.loads(page_records_path.read_text(encoding="utf-8")) if page_records_path.exists() else []
    cached_chunks = load_jsonl(chunk_path) if chunk_path.exists() else []
    previous_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    if from_cache and previous_audit.get("ready_for_embedding"):
        return cached_pages, cached_chunks, previous_audit

    failures = []
    short_pages = list(previous_audit.get("short_pages", []))
    quarantined_pages = list(previous_audit.get("quarantined_pages", []))
    reuse_partial = from_cache or retry_failed
    pages = list(cached_pages) if reuse_partial else []
    page_records = list(cached_records) if reuse_partial else []
    chunks = list(cached_chunks) if reuse_partial else []
    successful = {page.get("canonical_url") for page in pages}
    retry_set = {item.get("canonical_url") for item in previous_audit.get("failed_pages", [])} if retry_failed else None
    session = requests.Session()

    targets = []
    for canonical, aliases in sorted(groups.items()):
        if retry_set is not None and canonical not in retry_set:
            continue
        if canonical in successful:
            continue
        targets.append((canonical, aliases))

    for index, (canonical, aliases) in enumerate(targets, start=1):
        selected_url = aliases[0]
        print(f"[{index}/{len(groups)}] Fetching {selected_url}")
        try:
            html, resolved_url, status_code = fetch_html(session, selected_url, timeout)
            title, text, headings = extract_page_content(html)
            if not text.strip():
                raise ValueError("empty extracted page content after fallback")
            if len(text) < 80:
                short_pages.append({"url": selected_url, "chars": len(text)})
            record = build_page_record(aliases, resolved_url, title, text, headings)
            page_chunks = build_chunks(record, text)
            quality_reason = assess_page_quality(selected_url, title, text, page_chunks)
            if quality_reason:
                quarantined_pages.append({
                    "canonical_url": canonical,
                    "source_urls": aliases,
                    "reason": quality_reason,
                    "chars": len(text),
                    "chunk_count": len(page_chunks),
                })
                raise ValueError(
                    f"page failed content-quality audit: {quality_reason}"
                )
            page_records.append(record)
            pages.append({
                "source_url": selected_url,
                "source_aliases": aliases,
                "canonical_url": canonical,
                "title": title,
                "content": text,
                "headings": headings,
                "status_code": status_code,
            })
            chunks.extend(page_chunks)
        except Exception as exc:
            logger.warning("Failed %s: %s", selected_url, exc)
            failures.append({"canonical_url": canonical, "source_urls": aliases, "error": str(exc)})
        if delay > 0:
            time.sleep(delay)

    if failures:
        audit = {
            "registry_entries": len(urls),
            "unique_canonical_pages": len(groups),
            "fetched_pages": len(pages),
            "failed_pages": failures,
            "short_pages": short_pages,
            "quarantined_pages": quarantined_pages,
            "duplicate_alias_groups": {key: value for key, value in groups.items() if len(value) > 1},
            "ready_for_embedding": False,
            "pinecone_written": False,
        }
        atomic_write_jsonl(cache_path, pages)
        atomic_write_json(page_records_path, page_records)
        atomic_write_jsonl(chunk_path, chunks)
        atomic_write_json(audit_path, audit)
        raise SystemExit(
            f"REFUSED: {len(failures)} page fetch/extraction failures. Review {audit_path} before retrying."
        )

    validate_chunk_metadata(chunks)
    audit = {
        "registry_entries": len(urls),
        "unique_canonical_pages": len(groups),
        "fetched_pages": len(pages),
        "duplicate_alias_groups": {key: value for key, value in groups.items() if len(value) > 1},
        "short_pages": short_pages,
        "quarantined_pages": quarantined_pages,
        "failure_count": 0,
        "chunk_count": len(chunks),
        "category_counts": dict(Counter(record["page_category"] for record in page_records)),
        "topic_counts": dict(Counter(record["topic"] for record in page_records)),
        "department_counts": dict(Counter(record["department"] or "none" for record in page_records)),
        "ready_for_embedding": True,
        "pinecone_written": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_jsonl(cache_path, pages)
    atomic_write_json(page_records_path, page_records)
    atomic_write_jsonl(chunk_path, chunks)
    atomic_write_json(audit_path, audit)
    return pages, chunks, audit


def pinecone_metadata(chunk: dict) -> dict:
    metadata = {}
    for key, value in chunk.items():
        if key == "id" or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def upsert_chunks(
    pages: list[dict],
    chunks: list[dict],
    output_dir: Path,
    index_name: str,
    namespace: str,
    batch_size: int,
    page_batch_size: int,
) -> None:
    if index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: cannot target production lbrce-index")
    if namespace in {"", "default", BENCHMARK_NAMESPACE} or "prod" in namespace.lower():
        raise SystemExit("REFUSED: invalid full-migration namespace")

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is missing")
    pc = Pinecone(api_key=api_key)
    names = {item["name"] if isinstance(item, dict) else item.name for item in pc.list_indexes()}
    if index_name not in names:
        raise SystemExit(f"REFUSED: target index {index_name} does not exist")
    description = pc.describe_index(index_name)
    if int(description.dimension) != EXPECTED_DIMENSION or description.metric != "cosine":
        raise SystemExit("REFUSED: target index is not 1024-dimensional cosine")

    index = pc.Index(index_name)
    checkpoint_path = output_dir / "upsert_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {"completed_pages": []}
    completed = set(checkpoint.get("completed_pages", []))
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers is required for full-registry embedding")
    model = SentenceTransformer(MODEL_NAME)
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_doc[chunk["document_id"]].append(chunk)
    pages_by_doc = {stable_id(page["canonical_url"]): page for page in pages}
    uploaded = 0

    for page_index, (document_id_hash, page) in enumerate(sorted(pages_by_doc.items()), start=1):
        canonical = page["canonical_url"]
        if canonical in completed:
            continue
        page_chunks = chunks_by_doc.get(f"html-bge-{document_id_hash}", [])
        if not page_chunks:
            raise SystemExit(f"REFUSED: no chunks found for {canonical}")
        model_texts = [chunk["text"] for chunk in page_chunks]
        vectors = model.encode(
            model_texts,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(page_chunks), EXPECTED_DIMENSION):
            raise SystemExit(f"REFUSED: vector shape for {canonical}: {vectors.shape}")
        payload = [
            {"id": chunk["id"], "values": vector.astype(float).tolist(), "metadata": pinecone_metadata(chunk)}
            for chunk, vector in zip(page_chunks, vectors)
        ]
        for start in range(0, len(payload), batch_size):
            index.upsert(vectors=payload[start:start + batch_size], namespace=namespace)
        uploaded += len(payload)
        completed.add(canonical)
        atomic_write_json(checkpoint_path, {
            "index": index_name,
            "namespace": namespace,
            "model": MODEL_NAME,
            "dimension": EXPECTED_DIMENSION,
            "completed_pages": sorted(completed),
            "uploaded_chunks_this_run": uploaded,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"Uploaded {page_index}/{len(pages)} pages: {canonical} ({len(payload)} chunks)")

    stats = index.describe_index_stats()
    print(json.dumps({
        "index": index_name,
        "namespace": namespace,
        "uploaded_chunks_this_run": uploaded,
        "pinecone_inference_used": False,
        "stats": stats.to_dict() if hasattr(stats, "to_dict") else str(stats),
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="scripts/chunked_html_pages.json")
    parser.add_argument("--expected-registry-entries", type=int, default=DEFAULT_REGISTRY_ENTRIES)
    parser.add_argument("--exclude-url", action="append", default=[], help="Canonical URL to exclude; repeatable")
    parser.add_argument("--output-dir", default="migration_artifacts/full_registry_local_bge")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed pages from the previous audit")
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--upsert-batch-size", type=int, default=100)
    parser.add_argument("--confirm-full-migration", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: target index is production lbrce-index")
    if args.namespace in {"", "default", BENCHMARK_NAMESPACE} or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: invalid full-migration namespace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    excluded_urls = {canonical_url(url) for url in args.exclude_url}
    pages, chunks, audit = fetch_and_prepare(
        Path(args.registry),
        output_dir,
        args.timeout,
        args.delay,
        args.from_cache,
        args.retry_failed,
        args.expected_registry_entries,
        excluded_urls,
    )
    print(json.dumps(audit, indent=2))
    print(f"Dry-run artifacts written to: {output_dir.resolve()}")
    print("PASS: exact registry validated, pages extracted, metadata assigned, and chunks checked")
    print("Pinecone writes: 0 during preparation")

    if not args.confirm_full_migration:
        print("DRY RUN: local embedding and Pinecone upsert not performed")
        return 0

    upsert_chunks(
        pages,
        chunks,
        output_dir,
        args.index_name,
        args.namespace,
        args.upsert_batch_size,
        1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
