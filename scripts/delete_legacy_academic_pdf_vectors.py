"""Delete legacy full-PDF vectors for approved academic URLs only.

This is intentionally separate from URL-first ingestion. It never deletes by a
broad resource_type because older regulation records may have been mislabeled as
`timetable_pdf`. Deletion is exact by source URL and requires an explicit flag.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

PRODUCTION_INDEX = "lbrce-index"
DEFAULT_INDEX = "lbrce-local-bge-index"
DEFAULT_NAMESPACE = "lbrce_local_bge_full_v1"


def load_urls(manifest_paths: list[str]) -> list[str]:
    urls: list[str] = []
    for raw_path in manifest_paths:
        data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        records = data.get("pdfs") or data.get("resources") or []
        for record in records:
            if not isinstance(record, dict):
                continue
            url = str(record.get("url") or "").strip()
            resource_type = str(record.get("resource_type") or "").lower()
            if url and (record.get("url_only") or resource_type in {
                "regulation_pdf", "syllabus_pdf", "academic_syllabus_pdf", "exam_results_pdf"
            }):
                if url not in urls:
                    urls.append(url)
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Manifest containing URL-first PDF records; repeat for multiple manifests.",
    )
    parser.add_argument("--index-name", default=DEFAULT_INDEX)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--confirm-delete", action="store_true")
    args = parser.parse_args()

    if args.index_name == PRODUCTION_INDEX:
        raise SystemExit("REFUSED: production index deletion is not allowed")
    if not args.namespace or "prod" in args.namespace.lower():
        raise SystemExit("REFUSED: invalid or production namespace")

    urls = load_urls(args.manifest)
    if not urls:
        raise SystemExit("REFUSED: no URL-first academic PDF records found")

    print(json.dumps({
        "index": args.index_name,
        "namespace": args.namespace,
        "urls": urls,
        "url_count": len(urls),
        "delete_requested": args.confirm_delete,
    }, indent=2))

    if not args.confirm_delete:
        print("DRY RUN: no Pinecone vectors deleted. Add --confirm-delete after reviewing the URLs.")
        return 0

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is missing")

    index = Pinecone(api_key=api_key).Index(args.index_name)
    deleted_filters = 0
    for url in urls:
        # Delete by both fields because legacy ingestion paths used either
        # source_url or pdf_url as the direct document URL.
        index.delete(
            filter={"source_url": {"$eq": url}},
            namespace=args.namespace,
        )
        index.delete(
            filter={"pdf_url": {"$eq": url}},
            namespace=args.namespace,
        )
        deleted_filters += 2
        print(f"Submitted exact delete filters for {url}")

    print(json.dumps({"deleted_filter_operations": deleted_filters}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
