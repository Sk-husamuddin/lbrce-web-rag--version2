"""Offline 20-page benchmark for the local LBRCE metadata migration.

This script intentionally does not import Pinecone and does not upsert vectors.
It fetches representative official HTML pages, assigns deterministic metadata,
creates paragraph-safe chunks, generates local BGE embeddings, and writes local
artifacts for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # preparation and metadata-only tests do not need the model
    SentenceTransformer = None

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EXPECTED_DIMENSION = 1024
USER_AGENT = "LBRCE-Local-Metadata-Benchmark/1.0"

DEPARTMENT_RULES = {
    "cse_ai_ml": ["/csm/", "cse-ai", "cse_ai", "ai&ml"],
    "ai_ds": ["/ai/", "ai&ds", "ai_ds"],
    "cse": ["/cse/", "computer-science"],
    "ece": ["/ece/", "electronics"],
    "eee": ["/eee/", "electrical"],
    "ce": ["/civil/", "/ce/", "civil"],
    "ase": ["/ase/", "aerospace"],
    "it": ["/it/", "information-technology"],
    "me": ["/mech/", "/me/", "mechanical"],
    "mba": ["/mba/", "business-administration"],
}

CATEGORY_RULES = [
    ("student_list", ["studentslist", "student-list", "student list"]),
    ("timetable_directory", ["timetable", "time_table", "timetables"]),
    ("placement", ["placement", "stat25", "stat26"]),
    ("admission", ["admission", "eapcet", "ecet", "icet", "pgecet"]),
    ("department_contact", ["contact", "hod", "department-contact"]),
    ("regulation_directory", ["regulation", "acadregulation"]),
    ("academic_programs", ["course_structure", "syllabus", "academics"]),
    ("facility", ["library", "hostel", "transport", "facility", "studentcorner"]),
]


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return f"https://{parsed.netloc.lower()}{path}"


def _path(url: str) -> str:
    return urlparse(url).path.lower().rstrip("/") or "/"


def classify_category(url: str, title: str, text: str) -> str:
    """Classify by canonical URL first; page navigation text is only a fallback."""
    path = _path(url)
    if "studentslist" in path or "student-list" in path:
        return "student_list"
    if "timetable" in path or "timetables" in path or "time_table" in path:
        return "timetable_directory"
    if "/placements/" in path or "placement" in path:
        return "placement"
    if "/admission" in path or "admission" in path:
        return "admission"
    if "acadregulation" in path or "/regulation" in path:
        return "regulation_directory"
    if "course_structure" in path or "/academics/" in path:
        return "academic_programs"
    if "/studentcorner_pages/" in path or any(signal in path for signal in ("library", "hostel", "transport")):
        return "facility"
    if path.endswith("contact.php") or "contact" in path or "hod" in path:
        return "department_contact"
    if path in {"/", "/overview.php"}:
        return "college_profile"
    if re.search(r"/(cse|ece|eee|it|mech|ase|mba|ai|csm|civil)$", path):
        return "department_home"

    haystack = f"{title} {text[:3000]}".lower()
    for category, signals in CATEGORY_RULES:
        if any(signal in haystack for signal in signals):
            return category
    return "general_html"


def classify_department(url: str, title: str, text: str) -> str | None:
    """Prefer exact URL path segments so navigation cannot mislabel pages."""
    path = _path(url)
    segments = {segment for segment in path.split("/") if segment}
    segment_rules = {
        "csm": "cse_ai_ml",
        "ai": "ai_ds",
        "cse": "cse",
        "ece": "ece",
        "eee": "eee",
        "civil": "ce",
        "ase": "ase",
        "it": "it",
        "mech": "me",
        "mba": "mba",
    }
    for segment, department in segment_rules.items():
        if segment in segments:
            return department
    return None


def classify_topic(url: str, category: str) -> str:
    """Add a narrower topic for categories that contain multiple page types."""
    path = _path(url)
    if "central_library" in path or "library" in path:
        return "library"
    if "hostel" in path:
        return "hostel"
    if category == "placement":
        return "placement_statistics"
    if category == "admission":
        return "admission_procedure"
    if category == "department_contact":
        return "department_contact"
    if category == "student_list":
        return "student_list"
    if category == "timetable_directory":
        return "timetable_directory"
    if category == "regulation_directory":
        return "regulation_directory"
    if category == "academic_programs":
        return "academic_programs"
    return category


def extract_academic_years(text: str) -> list[str]:
    matches = re.findall(r"\b(20\d{2}\s*[-–]\s*\d{2})\b", text)
    years = []
    for value in matches:
        normalized = value.replace("–", "-").replace(" ", "")
        if normalized not in years:
            years.append(normalized)
    return years


def extract_academic_year(text: str, category: str, url: str) -> str | None:
    """Set a single year only when the URL/category points to a year-specific page."""
    years = extract_academic_years(text)
    path = _path(url)
    if category in {"placement", "student_list", "timetable_directory"} and years:
        return years[0]
    if re.search(r"20\d{2}[-–]\d{2}", path):
        return years[0] if years else None
    return None


def extract_text(html: str) -> tuple[str, str, list[str]]:
    """Extract leaf-most content blocks without parent/child duplication.

    Some LBRCE pages contain meaningful sections after the closing body/html
    elements because of legacy markup. We first prefer the normal main/article/
    body root. If that root yields substantially less text than the complete
    parsed document, we retry against the full document so authoritative
    sections such as location and library details are not silently discarded.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]
    headings = [
        re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        for node in soup.find_all(heading_tags)
        if node.get_text(strip=True)
    ]
    content_tags = set(heading_tags + ["p", "div", "section", "article", "li", "td", "th", "blockquote", "pre"])

    def leaf_blocks(root) -> list[str]:
        blocks: list[str] = []
        seen: set[str] = set()
        for node in root.find_all(list(content_tags)):
            # Keep only the deepest matching block so a parent container cannot
            # duplicate all of its nested paragraph/table text.
            if any(child.name in content_tags for child in node.find_all(list(content_tags))):
                continue
            value = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            if len(value) < 3 or value in seen:
                continue
            seen.add(value)
            blocks.append(value)
        return blocks

    selected_root = soup.find("main") or soup.find("article") or soup.find("body") or soup
    blocks = leaf_blocks(selected_root)
    selected_text = "\n".join(blocks)

    # Legacy pages may place the real content outside <body>. Only use the
    # full-document fallback when it materially contains more text, preventing
    # ordinary pages from receiving duplicated navigation content.
    full_blocks = leaf_blocks(soup)
    full_text = "\n".join(full_blocks)
    if len(full_text) > max(len(selected_text) + 300, int(len(selected_text) * 1.5)):
        blocks = full_blocks
        selected_text = full_text

    if not blocks:
        fallback = re.sub(r"\s+", " ", selected_root.get_text(" ", strip=True)).strip()
        if fallback:
            blocks = [fallback]
    return title, "\n".join(blocks), headings


def chunk_text(text: str, max_chars: int = 2400) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        if current and current_length + len(paragraph) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(paragraph)
        current_length += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def fetch_page(url: str, timeout: float) -> str:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="scripts/local_benchmark_pages.json")
    parser.add_argument("--output-dir", default="migration_artifacts/local_benchmark")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pages = manifest.get("pages", [])
    if len(pages) != 20:
        raise ValueError(f"Expected exactly 20 benchmark pages, found {len(pages)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    page_results: list[dict] = []

    for index, entry in enumerate(pages):
        url = entry["url"]
        print(f"[{index + 1:02d}/20] Fetching {url}")
        html = fetch_page(url, args.timeout)
        title, text, headings = extract_text(html)
        category = classify_category(url, title, text)
        department = classify_department(url, title, text)
        topic = classify_topic(url, category)
        academic_year = extract_academic_year(text, category, url)
        page_id = hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:16]
        page_chunks = chunk_text(text)
        expected_category = entry.get("expected_category")

        page_results.append({
            "url": url,
            "canonical_url": canonical_url(url),
            "title": title,
            "category": category,
            "topic": topic,
            "expected_category": expected_category,
            "category_match": category == expected_category,
            "department": department,
            "academic_year": academic_year,
            "heading_count": len(headings),
            "text_chars": len(text),
            "chunk_count": len(page_chunks),
        })

        for chunk_index, chunk in enumerate(page_chunks):
            chunks.append({
                "id": f"html-{page_id}-{chunk_index:04d}",
                "text": chunk,
                "source_url": url,
                "canonical_url": canonical_url(url),
                "source_type": "html",
                "resource_type": "html_page",
                "page_category": category,
                "topic": topic,
                "department": department,
                "academic_year": academic_year,
                "document_id": page_id,
                "chunk_index": chunk_index,
                "approved_source": True,
            })
        time.sleep(args.delay)

    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers is required to run the embedding benchmark")
    model = SentenceTransformer(MODEL_NAME)
    passage_texts = [record["text"] for record in chunks]

    vectors = model.encode(
        passage_texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != EXPECTED_DIMENSION:
        raise RuntimeError(f"Expected (*, {EXPECTED_DIMENSION}), got {vectors.shape}")

    (output_dir / "benchmark_pages.json").write_text(json.dumps(page_results, indent=2), encoding="utf-8")
    with (output_dir / "benchmark_chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in chunks:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    np.save(output_dir / "benchmark_vectors.npy", vectors)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "dimension": int(vectors.shape[1]),
        "page_count": len(page_results),
        "chunk_count": len(chunks),
        "category_matches": sum(1 for item in page_results if item["category_match"]),
        "category_mismatches": [item for item in page_results if not item["category_match"]],
        "vector_file": "benchmark_vectors.npy",
        "pinecone_written": False,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to: {output_dir.resolve()}")
    print("PASS: local benchmark completed without Pinecone writes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
