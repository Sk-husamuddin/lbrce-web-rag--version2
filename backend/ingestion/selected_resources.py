"""Helpers for selective PDF and timetable-resource ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from backend.ingestion.chunker import Chunk


def load_selected_manifest(path: str | Path) -> Dict[str, Any]:
    """Load and validate a selective resource manifest."""
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Selected resource manifest must contain a JSON object")

    timetables = data.get("timetables", [])
    pdfs = data.get("pdfs", [])
    if not isinstance(timetables, list) or not isinstance(pdfs, list):
        raise ValueError("Manifest fields 'timetables' and 'pdfs' must be arrays")

    for index, record in enumerate(timetables):
        if not isinstance(record, dict) or not record.get("url"):
            raise ValueError(f"Timetable record {index} must contain a URL")
    for index, record in enumerate(pdfs):
        if not isinstance(record, dict) or not record.get("url"):
            raise ValueError(f"PDF record {index} must contain a URL")

    return data


def _stable_id(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _timetable_text(
    title: str,
    department: str,
    course: str,
    academic_year: str,
    term: str,
    semester: str,
    section: str,
    resource_label: str,
) -> str:
    """Build compact searchable text without extracting timetable pixels."""
    section_label = section or "all regular sections / not separately specified"
    return (
        f"{title}. Department: {department}. Course: {course}. "
        f"Academic year: {academic_year}. Term: {term}. Semester: {semester}. "
        f"Section: {section_label}. "
        f"The original timetable {resource_label} is available from the official LBRCE timetable page."
    )


def _record_scope(manifest: Dict[str, Any], record: Dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Resolve shared timetable fields from a record and manifest defaults."""
    scope = manifest.get("scope", {})
    return (
        str(record.get("title") or "Selected timetable"),
        str(record.get("department") or scope.get("department") or ""),
        str(record.get("course") or scope.get("course") or "B.Tech"),
        str(record.get("academic_year") or scope.get("academic_year") or ""),
        str(record.get("term") or scope.get("term") or ""),
        str(record.get("semester") or ""),
    )


def timetable_chunks(manifest: Dict[str, Any]) -> List[Chunk]:
    """Create one searchable metadata chunk per selected timetable image."""
    default_source_url = manifest.get("source_page", "")
    chunks: List[Chunk] = []

    for record in manifest.get("timetables", []):
        image_url = str(record["url"])
        title, department, course, academic_year, term, semester = _record_scope(manifest, record)
        section = str(record.get("section") or "")
        source_url = str(record.get("source_url") or default_source_url)
        text = _timetable_text(
            title, department, course, academic_year, term, semester, section, "image"
        )

        document_id = f"timetable-{_stable_id(image_url)}"
        metadata: Dict[str, Any] = {
            "resource_type": "timetable_image",
            "image_url": image_url,
            "source_url": source_url,
            "department": department,
            "academic_year": academic_year,
            "term": term,
            "semester": semester,
            "section": section,
            "regulation": record.get("regulation"),
            "class_room": record.get("class_room"),
            "effective_from": record.get("effective_from"),
        }
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}-0",
                text=text,
                source_url=source_url,
                title=title,
                source_type="timetable_image",
                department=department,
                page_number=None,
                document_id=document_id,
                metadata=metadata,
            )
        )

    return chunks


def timetable_pdf_chunks(manifest: Dict[str, Any]) -> List[Chunk]:
    """Create one URL-based metadata chunk per selected timetable PDF.

    Some approved timetable PDFs are image-only and therefore produce no text
    through the normal PDF parser. They still need a searchable Pinecone
    record so the assistant can return the official PDF URL. No PDF pixels or
    OCR output are used here.
    """
    chunks: List[Chunk] = []

    for record in manifest.get("pdfs", []):
        if record.get("resource_type") not in (None, "timetable_pdf"):
            continue
        pdf_url = str(record["url"])
        title, department, course, academic_year, term, semester = _record_scope(manifest, record)
        section = str(record.get("section") or "")
        source_page = str(record.get("source_url") or manifest.get("source_page", ""))
        text = _timetable_text(
            title, department, course, academic_year, term, semester, section, "PDF"
        )

        document_id = f"timetable-pdf-{_stable_id(pdf_url)}"
        metadata: Dict[str, Any] = {
            "resource_type": "timetable_pdf",
            "timetable": "true",
            "pdf_url": pdf_url,
            "source_page": source_page,
            "department": department,
            "academic_year": academic_year,
            "term": term,
            "semester": semester,
            "section": section,
            "variant": record.get("variant"),
            "regulation": record.get("regulation"),
        }
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}-0",
                text=text,
                # Keep the direct PDF URL in the base metadata so the existing
                # frontend PDF renderer can use it without a frontend change.
                source_url=pdf_url,
                title=title,
                source_type="pdf",
                department=department,
                page_number=None,
                document_id=document_id,
                metadata=metadata,
            )
        )

    return chunks



def pdf_url_chunks(manifest: Dict[str, Any]) -> List[Chunk]:
    """Create one URL-only metadata chunk per approved academic PDF.

    Regulation, syllabus, and examination PDFs are intentionally not parsed or
    embedded as full documents. The searchable record stores the exact official
    PDF URL so the assistant can return it to the user and direct them to the
    original document for detailed reading.
    """
    chunks: List[Chunk] = []
    url_resource_types = {
        "regulation_pdf",
        "syllabus_pdf",
        "academic_syllabus_pdf",
        "exam_results_pdf",
    }

    for record in manifest.get("pdfs", []):
        resource_type = str(record.get("resource_type") or "").strip().lower()
        if resource_type not in url_resource_types and not record.get("url_only"):
            continue

        pdf_url = str(record["url"])
        source_page = str(record.get("source_url") or manifest.get("source_page", ""))
        title = str(record.get("title") or "Official LBRCE PDF")
        regulation = str(record.get("regulation") or "")
        document_type = str(record.get("document_type") or resource_type or "academic_pdf")
        department = str(record.get("department") or "")
        text = (
            f"{title}. Official LBRCE PDF URL: {pdf_url}. "
            f"Regulation: {regulation or 'not specified'}. "
            f"Department: {department or 'not specified'}. "
            "This is the exact official PDF requested by the user. "
            "Open the PDF URL to check the detailed information."
        )
        document_id = f"academic-pdf-url-{_stable_id(pdf_url, resource_type or document_type)}"
        metadata: Dict[str, Any] = {
            "resource_type": resource_type or "academic_pdf_url",
            "source_type": "pdf",
            "pdf_url": pdf_url,
            "source_url": pdf_url,
            "directory_url": source_page,
            "document_type": document_type,
            "regulation": regulation,
            "department": department,
            "approved_source": bool(record.get("approved", True)),
            "url_metadata_only": True,
            "url_first": True,
        }
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}-0",
                text=text,
                source_url=pdf_url,
                title=title,
                source_type="pdf",
                department=department,
                page_number=None,
                document_id=document_id,
                metadata=metadata,
            )
        )

    return chunks
