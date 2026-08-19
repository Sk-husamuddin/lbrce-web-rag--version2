"""Targeted local-BGE migration for R23 syllabi and exam-results resources.

The script processes only resources listed in academic_resources_manifest.json.
It supports preparation-only extraction, local BGE embedding, and an explicit
migration-index upsert. It never targets the production index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pinecone import Pinecone

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_local_embeddings import MODEL_NAME, EXPECTED_DIMENSION, chunk_text
from scripts.migrate_html_registry_local_bge import extract_page_content
from backend.ingestion.pdf_parser import parse_pdf_from_url
from backend.ingestion.chunker import chunk_pdf_document

PRODUCTION_INDEX = "lbrce-index"
BENCHMARK_NAMESPACE = "lbrce_local_bge_v1"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_full_v1"
USER_AGENT = "LBRCE-Academic-Resources-Local-BGE/1.0"


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


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("REFUSED: academic manifest must be a JSON object")
    resources = data.get("resources", [])
    if not isinstance(resources, list) or not resources:
        raise SystemExit("REFUSED: academic manifest has no resources")
    all_resources = list(resources)
    exam_page = data.get("exam_results_page")
    if exam_page:
        all_resources.insert(0, exam_page)
    allow_url_metadata_fallback = bool(data.get("allow_url_metadata_fallback", False))
    allow_pdf_url_metadata_fallback = bool(data.get("allow_pdf_url_metadata_fallback", False))
    if data.get("include_r23_result_pdfs") and exam_page:
        try:
            response = requests.get(exam_page["url"], timeout=45, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            existing_urls = {canonical_url(str(item["url"])) for item in all_resources}
            for anchor in soup.find_all("a", href=True):
                href = str(anchor.get("href") or "").strip()
                label = " ".join(anchor.get_text(" ", strip=True).split())
                candidate = urljoin(response.url, href)
                lowered = f"{candidate} {label}".lower()
                if ".pdf" not in lowered or "r23" not in lowered:
                    continue
                candidate = canonical_url(candidate)
                if candidate in existing_urls:
                    continue
                resource_id = f"r23-result-{stable_id(candidate)}"
                all_resources.append({
                    "id": resource_id,
                    "kind": "pdf",
                    "url": candidate,
                    "title": label or "LBRCE R23 examination result PDF",
                    "page_category": "examination",
                    "topic": "exam_results",
                    "regulation": "R23",
                    "document_type": "exam_results_pdf",
                    "allow_url_metadata_fallback": allow_pdf_url_metadata_fallback,
                })
                existing_urls.add(candidate)
        except Exception as exc:
            raise SystemExit(f"REFUSED: could not discover R23 result PDFs from exam-results page: {exc}")
    seen = set()
    for resource in all_resources:
        required = {"id", "kind", "url", "title", "page_category", "topic"}
        missing = required - set(resource)
        if missing:
            raise SystemExit(f"REFUSED: resource {resource.get('id')} missing {sorted(missing)}")
        url = canonical_url(str(resource["url"]))
        if not url.startswith("https://lbrce.ac.in"):
            raise SystemExit(f"REFUSED: non-LBRCE URL: {url}")
        if url in seen:
            raise SystemExit(f"REFUSED: duplicate URL: {url}")
        seen.add(url)
        if resource["kind"] not in {"html", "pdf"}:
            raise SystemExit(f"REFUSED: unsupported resource kind: {resource['kind']}")
        resource.setdefault("allow_url_metadata_fallback", allow_url_metadata_fallback or (allow_pdf_url_metadata_fallback and resource["kind"] == "pdf"))
    return {"resources": all_resources}


def pdf_record(resource: dict, url: str, chunk, index: int) -> dict:
    page_number = getattr(chunk, "page_number", None)
    base_id = f"academic-pdf-bge-{stable_id(url)}"
    chunk_id = f"{base_id}-{page_number or 0:04d}-{index:04d}"
    metadata = dict(getattr(chunk, "metadata", {}) or {})
    metadata.update({
        "resource_type": "academic_pdf",
        "source_type": "pdf",
        "page_category": resource["page_category"],
        "topic": resource["topic"],
        "document_type": resource.get("document_type", "academic_pdf"),
        "regulation": resource.get("regulation"),
        "department": resource.get("department"),
        "source_url": url,
        "pdf_url": url,
        "approved_source": True,
    })
    return {
        "id": chunk_id,
        "text": str(getattr(chunk, "text", "") or ""),
        "chunk_index": index,
        "document_id": base_id,
        "source_url": url,
        "canonical_url": canonical_url(url),
        "title": resource["title"],
        "page_category": resource["page_category"],
        "topic": resource["topic"],
        "resource_type": "academic_pdf",
        "source_type": "pdf",
        "document_type": resource.get("document_type", "academic_pdf"),
        "regulation": resource.get("regulation"),
        "department": resource.get("department"),
        "page_number": page_number,
        "metadata": metadata,
        "migration_embedding_model": MODEL_NAME,
        "migration_version": "academic_resources_local_bge_v2",
    }


def html_url_metadata_record(resource: dict, url: str, *, reason: str, content_type: str = "") -> dict:
    base_id = f"academic-html-url-bge-{stable_id(url)}"
    text = (
        f"Official LBRCE examination resource: {resource['title']}. "
        f"Open the official directory here: {url}. "
        f"The live HTML body was not embedded in this run because the source server "
        f"returned {content_type or 'no usable HTML content'} ({reason})."
    )
    metadata = {
        "resource_type": "academic_html_url",
        "source_type": "html_url_metadata",
        "page_category": resource["page_category"],
        "topic": resource["topic"],
        "document_type": resource.get("document_type", "directory_html"),
        "source_url": url,
        "approved_source": True,
        "url_metadata_only": True,
        "availability_note": reason,
        "source_content_type": content_type,
    }
    return {
        "id": base_id,
        "text": text,
        "chunk_index": 0,
        "document_id": base_id,
        "source_url": url,
        "canonical_url": canonical_url(url),
        "title": resource["title"],
        "page_category": resource["page_category"],
        "topic": resource["topic"],
        "resource_type": "academic_html_url",
        "source_type": "html_url_metadata",
        "document_type": resource.get("document_type", "directory_html"),
        "url_metadata_only": True,
        "availability_note": reason,
        "source_content_type": content_type,
        "metadata": metadata,
        "migration_embedding_model": MODEL_NAME,
        "migration_version": "academic_resources_local_bge_v2",
    }


def pdf_url_metadata_record(resource: dict, url: str, *, reason: str, content_type: str = "") -> dict:
    base_id = f"academic-pdf-url-bge-{stable_id(url)}"
    url_resource_type = str(resource.get("resource_type") or "academic_pdf_url")
    department = resource.get("department") or "all applicable departments"
    text = (
        f"Official LBRCE academic resource: {resource['title']}. "
        f"This is the official PDF URL for {department}. "
        f"Open the source document here: {url}. "
        f"The PDF body was not embedded in this migration because {reason}. "
        "This record is a source-link directory entry, not the PDF text."
    )
    metadata = {
        "resource_type": url_resource_type,
        "source_type": "pdf_url_metadata",
        "page_category": resource["page_category"],
        "topic": resource["topic"],
        "document_type": resource.get("document_type", "academic_pdf"),
        "regulation": resource.get("regulation"),
        "department": department,
        "source_url": url,
        "pdf_url": url,
        "approved_source": True,
        "url_metadata_only": True,
        "url_first": True,
        "availability_note": reason,
        "source_content_type": content_type,
    }
    return {
        "id": base_id,
        "text": text,
        "chunk_index": 0,
        "document_id": base_id,
        "source_url": url,
        "canonical_url": canonical_url(url),
        "title": resource["title"],
        "page_category": resource["page_category"],
        "topic": resource["topic"],
        "resource_type": url_resource_type,
        "source_type": "pdf_url_metadata",
        "document_type": resource.get("document_type", "academic_pdf"),
        "regulation": resource.get("regulation"),
        "department": department,
        "url_metadata_only": True,
        "url_first": True,
        "availability_note": reason,
        "source_content_type": content_type,
        "metadata": metadata,
        "migration_embedding_model": MODEL_NAME,
        "migration_version": "academic_resources_local_bge_v3",
    }


def prepare_resources(resources: list[dict], output_dir: Path, timeout: float) -> list[dict]:
    session = requests.Session()
    prepared: list[dict] = []
    failures: list[dict] = []
    for number, resource in enumerate(resources, start=1):
        url = resource["url"]
        print(f"[{number}/{len(resources)}] Fetching {url}")
        try:
            if resource["kind"] == "html":
                response = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
                response.raise_for_status()
                title, text, headings = extract_page_content(response.text)
                lowered_html = response.text.lower()
                anti_bot = any(marker in lowered_html for marker in ("one moment, please", "please wait while your request is being verified", "cf-chl-", "cloudflare"))
                chunks = chunk_text(text)
                if anti_bot:
                    if resource.get("allow_url_metadata_fallback"):
                        prepared.append(html_url_metadata_record(resource, url, reason="anti-bot verification interstitial", content_type=response.headers.get("content-type", "")))
                        continue
                    raise ValueError("anti-bot verification interstitial")
                if not chunks:
                    if resource.get("allow_url_metadata_fallback"):
                        prepared.append(html_url_metadata_record(resource, url, reason="zero extracted HTML chunks", content_type=response.headers.get("content-type", "")))
                        continue
                    raise ValueError("zero HTML chunks")
                doc_id = f"academic-html-bge-{stable_id(url)}"
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                for index, value in enumerate(chunks):
                    prepared.append({
                        "id": f"{doc_id}-{index:05d}",
                        "text": value,
                        "chunk_index": index,
                        "document_id": doc_id,
                        "source_url": url,
                        "canonical_url": canonical_url(url),
                        "resolved_url": response.url,
                        "title": resource["title"],
                        "page_category": resource["page_category"],
                        "topic": resource["topic"],
                        "resource_type": "academic_html",
                        "source_type": "html",
                        "document_type": resource.get("document_type", "directory_html"),
                        "approved_source": True,
                        "content_hash": content_hash,
                        "headings": " | ".join(headings[:20]),
                        "migration_embedding_model": MODEL_NAME,
                        "migration_version": "academic_resources_local_bge_v2",
                    })
            else:
                if resource.get("url_only"):
                    prepared.append(
                        pdf_url_metadata_record(
                            resource,
                            url,
                            reason="the manifest specifies URL-first policy; PDF body intentionally not embedded",
                            content_type="application/pdf (not fetched)",
                        )
                    )
                    continue
                probe = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
                content_type = (probe.headers.get("content-type") or "").lower()
                is_pdf = "application/pdf" in content_type or probe.content[:4] == b"%PDF"
                if not is_pdf:
                    reason = "anti-bot interstitial or non-PDF response"
                    if resource.get("allow_url_metadata_fallback"):
                        prepared.append(pdf_url_metadata_record(resource, url, reason=reason, content_type=content_type))
                        continue
                    raise ValueError(f"non-PDF response: {content_type or 'missing content type'}")
                docs = parse_pdf_from_url(url, timeout=timeout) or []
                index = 0
                for doc in docs:
                    for chunk in chunk_pdf_document(doc):
                        if str(getattr(chunk, "text", "") or "").strip():
                            prepared.append(pdf_record(resource, url, chunk, index))
                            index += 1
                if index == 0:
                    if resource.get("allow_url_metadata_fallback"):
                        prepared.append(pdf_url_metadata_record(resource, url, reason="PDF served but no extractable text", content_type=content_type))
                    else:
                        raise ValueError("zero PDF text chunks")
        except Exception as exc:
            failures.append({"id": resource.get("id"), "url": url, "error": str(exc)})

    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [item["id"] for item in prepared]
    if len(ids) != len(set(ids)):
        raise SystemExit("REFUSED: duplicate academic resource vector IDs")
    audit = {
        "resource_count": len(resources),
        "chunk_count": len(prepared),
        "failure_count": len(failures),
        "failures": failures,
        "pinecone_written": False,
        "pinecone_inference_used": False,
        "ready_for_embedding": not failures and bool(prepared),
        "url_metadata_fallback_count": sum(1 for item in prepared if item.get("url_metadata_only")),
    }
    (output_dir / "academic_resources_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output_dir / "academic_resources_chunks.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in prepared), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"REFUSED: academic resource extraction failures: {json.dumps(failures, indent=2)}")
    return prepared


def pinecone_safe_metadata(item: dict) -> dict:
    """Return metadata accepted by Pinecone's scalar/list metadata schema.

    Prepared records retain a nested ``metadata`` object for local artifacts,
    but Pinecone metadata values cannot be dictionaries. Copy the nested fields
    into the top-level payload when they are scalar or string-list values and
    omit the nested dictionary itself.
    """
    safe: dict = {}

    def add_value(key: str, value) -> None:
        if value is None or isinstance(value, dict):
            return
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
            return
        if isinstance(value, list) and all(isinstance(entry, str) for entry in value):
            safe[key] = value
            return
        safe[key] = str(value)

    for key, value in item.items():
        if key in {"id", "text", "metadata"}:
            continue
        add_value(key, value)

    nested = item.get("metadata")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if key not in safe:
                add_value(key, value)

    safe["text"] = str(item.get("text") or "")
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="scripts/academic_resources_manifest.json")
    parser.add_argument("--output-dir", default="migration_artifacts/academic_resources_local_bge")
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-academic-migration", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: cannot target production lbrce-index")
    if args.namespace in {"", "default", BENCHMARK_NAMESPACE} or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: invalid academic-resource namespace")

    manifest = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    chunks = prepare_resources(manifest["resources"], output_dir, args.timeout)
    print(json.dumps({
        "resources": len(manifest["resources"]),
        "chunks": len(chunks),
        "pinecone_inference_used": False,
        "pinecone_written": False,
        "mode": "prepare_only" if args.prepare_only else "embedding_pending",
    }, indent=2))
    if args.prepare_only:
        return 0
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers is required for embedding; use --prepare-only first")

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [item["text"] for item in chunks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(chunks), EXPECTED_DIMENSION):
        raise SystemExit(f"REFUSED: invalid vector shape {vectors.shape}")
    np.save(output_dir / "academic_resources_vectors.npy", vectors)
    if not args.confirm_academic_migration:
        print("DRY RUN: local vectors created; no Pinecone connection or upsert performed")
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
        payload = []
        for item, vector in zip(chunks[start:start + args.batch_size], vectors[start:start + args.batch_size]):
            metadata = pinecone_safe_metadata(item)
            payload.append({"id": item["id"], "values": vector.tolist(), "metadata": metadata})
        index.upsert(vectors=payload, namespace=args.namespace)
        print(f"Upserted academic-resource batch {start // args.batch_size + 1} ({len(payload)} vectors)")
    print(json.dumps({
        "resources": len(manifest["resources"]),
        "chunks": len(chunks),
        "uploaded": len(chunks),
        "index": args.index_name,
        "namespace": args.namespace,
        "pinecone_inference_used": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
