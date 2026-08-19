"""Audit existing local-BGE HTML migration artifacts without re-crawling or embedding.

The audit groups chunks by document_id, joins page metadata with the full
page content stored in fetched_pages.jsonl, and flags pages that have zero
chunks, very short content, or mostly shared navigation labels. It never
connects to Pinecone.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


NAVIGATION_LABELS = {
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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def page_issue(record: dict, chunks: list[dict]) -> str | None:
    content = str(record.get("content") or "")
    if not chunks:
        return "zero_chunks"
    if len(content.strip()) < 80:
        return "too_short"
    lines = [line.strip().lower() for line in content.splitlines() if line.strip()]
    non_navigation = [line for line in lines if line not in NAVIGATION_LABELS]
    has_substantive_line = any(len(line) >= 120 for line in non_navigation)
    if len(content) < 2000 and len(non_navigation) < 4 and not has_substantive_line:
        return "navigation_only"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("migration_artifacts/full_registry_local_bge"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; defaults to <artifact-dir>/content_quality_audit.json",
    )
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    page_path = artifact_dir / "fetched_pages.jsonl"
    metadata_path = artifact_dir / "page_metadata.json"
    chunks_path = artifact_dir / "chunks.jsonl"
    audit_path = artifact_dir / "registry_audit.json"
    if not metadata_path.exists() or not chunks_path.exists() or not page_path.exists():
        raise SystemExit(
            f"Missing required artifacts: {metadata_path}, {page_path}, and/or {chunks_path}"
        )

    metadata_pages = json.loads(metadata_path.read_text(encoding="utf-8"))
    fetched_pages = load_jsonl(page_path)
    fetched_by_canonical = {
        str(page.get("canonical_url") or ""): page
        for page in fetched_pages
        if page.get("canonical_url")
    }
    pages = []
    for metadata in metadata_pages:
        fetched = fetched_by_canonical.get(str(metadata.get("canonical_url") or ""), {})
        pages.append({**fetched, **metadata, "content": fetched.get("content", "")})

    chunks = load_jsonl(chunks_path)
    chunks_by_document: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[str(chunk.get("document_id") or "")].append(chunk)

    issues = []
    page_rows = []
    for page in pages:
        document_id = str(page.get("document_id") or "")
        page_chunks = chunks_by_document.get(document_id, [])
        reason = page_issue(page, page_chunks)
        row = {
            "document_id": document_id,
            "canonical_url": page.get("canonical_url"),
            "source_url": page.get("source_url"),
            "title": page.get("title"),
            "page_category": page.get("page_category"),
            "topic": page.get("topic"),
            "content_chars": len(str(page.get("content") or "")),
            "chunk_count": len(page_chunks),
            "issue": reason,
        }
        page_rows.append(row)
        if reason:
            issues.append(row)

    result = {
        "artifact_dir": str(artifact_dir),
        "page_metadata_exists": metadata_path.exists(),
        "fetched_pages_exists": page_path.exists(),
        "content_joined_from_fetched_pages": True,
        "registry_audit_exists": audit_path.exists(),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "issue_count": len(issues),
        "issue_counts": {
            reason: sum(1 for row in issues if row["issue"] == reason)
            for reason in sorted({row["issue"] for row in issues})
        },
        "issues": issues,
        "pages": page_rows,
        "pinecone_written": False,
    }
    output = args.output or artifact_dir / "content_quality_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "issue_count": len(issues),
        "issue_counts": result["issue_counts"],
        "output": str(output),
        "pinecone_written": False,
    }, indent=2))
    if args.fail_on_issues and issues:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
