"""
Unified document chunking for LBRCE AI Assistant.

Accepts ParsedDocument, PDFDocument, or CleanedContent objects and
produces metadata-preserving chunks for RAG ingestion.

Following Component F spec:
  - chunk_size ≈ 500 tokens (configurable)
  - overlap ≈ 50 tokens (configurable)
  - Every chunk must retain full metadata
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with full metadata for source attribution."""
    chunk_id: str
    text: str
    source_url: str
    title: str
    source_type: str
    department: str
    page_number: Optional[int]
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _detect_source_type(document: Dict[str, Any]) -> str:
    """Detect source type from document dict keys."""
    if document.get("source_type"):
        return document["source_type"]
    # Infer from available keys
    if document.get("pdf_metadata") is not None:
        return "pdf"
    if document.get("headings") is not None:
        return "html"
    return "unknown"


def _safe_get(document: Dict[str, Any], key: str, default: Any = "") -> Any:
    """Safely get a value from a document dict."""
    return document.get(key, default)


def _generate_chunk_id(document_id: str, source_url: str, page_number: int, chunk_index: int) -> str:
    """Generate a unique chunk ID."""
    return f"{document_id}_{source_url}_{page_number}_{chunk_index}"


def _token_estimate(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def chunk_document(
    document: Dict[str, Any],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk a document dict into retrieval units with preserved metadata.

    Accepts document shapes from html_parser.py (ParsedDocument) or
    pdf_parser.py (PDFDocument) or cleaner.py (CleanedContent).

    Args:
        document: Document dict with fields like document_id, title,
                  source_url, content, department, page_number, etc.
        chunk_size: Target tokens per chunk (default: 500, per Component F spec).
        overlap: Overlap tokens between consecutive chunks (default: 50).

    Returns:
        List of Chunk objects with full metadata.
    """
    # Validate configurations
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    # Normalize input: ensure we have content. Some parser adapters provide
    # content as a list of blocks, so normalize that before calling string
    # methods.
    content = _safe_get(document, "content", "")
    if isinstance(content, list):
        content = "\n".join(str(c) for c in content if c is not None)
    content = str(content or "")
    if not content.strip():
        return []

    document_id = _safe_get(document, "document_id", "")
    title = _safe_get(document, "title", "")
    source_url = _safe_get(document, "source_url", "")
    department = _safe_get(document, "department", "")
    raw_page_number = _safe_get(document, "page_number", 0)
    try:
        page_number = int(raw_page_number) if raw_page_number not in (None, "") else 0
    except (TypeError, ValueError):
        page_number = 0

    source_type = _detect_source_type(document)

    # Token-based chunking using character approximation
    # ~4 chars per token is the rough heuristic
    char_per_token = 4
    chunk_char_size = chunk_size * char_per_token  # ~2000 chars for 500 tokens
    chunk_char_overlap = overlap * char_per_token  # ~200 chars for 50 tokens

    # Preserve meaningful paragraph, list, and timetable boundaries while
    # normalizing only horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", content)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        if not text[start:].strip():
            break

        # Determine candidate end
        end = min(start + chunk_char_size, len(text))

        # Adjust boundary so we don't cut mid-word, if we didn't hit the end of the text
        if end < len(text):
            window = text[start:end]
            search_start = max(0, len(window) - 50)
            boundary = max(
                window.rfind(" ", search_start),
                window.rfind("\n", search_start),
            )
            if boundary != -1:
                end = start + boundary

        chunk_text = text[start:end].strip()

        # Generate chunk ID
        chunk_id = _generate_chunk_id(
            document_id, source_url, page_number, chunk_index
        )

        if chunk_text:
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_url=source_url,
                title=title,
                source_type=source_type,
                department=department,
                page_number=page_number,
                document_id=document_id,
                metadata=dict(document.get("metadata") or {}),
            )
            chunks.append(chunk)
            chunk_index += 1

        # If we reached the end, break
        if end >= len(text):
            break

        # Calculate next start with overlap and prevent infinite loop
        next_start = end - chunk_char_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

        # Skip very small trailing chunks (less than 20 chars)
        if len(text[start:].strip()) < 20:
            break

    return chunks


def chunk_parsed_document(
    parsed: Any,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk a ParsedDocument instance.

    Accepts html_parser.py ParsedDocument with fields:
      - document_id, title, content, source_url, source_type="html",
        department, headings, links, pdf_links, metadata
    """
    # Convert ParsedDocument to dict shape
    document_dict = {
        "document_id": parsed.document_id,
        "title": parsed.title,
        "source_url": parsed.source_url,
        "content": parsed.content,
        "department": parsed.department,
        "page_number": parsed.metadata.get("page_number", 0) if parsed.metadata else 0,
        "source_type": parsed.source_type,
        "metadata": dict(parsed.metadata or {}),
    }
    if str(document_dict["metadata"].get("resource_type", "")).lower() == "student_list_html":
        return chunk_student_list_document(parsed, chunk_size=chunk_size)
    return chunk_document(document_dict, chunk_size=chunk_size, overlap=overlap)


def chunk_student_list_document(
    parsed: Any,
    chunk_size: int = 500,
) -> List[Chunk]:
    """Chunk a student-list HTML page at complete roster-row boundaries.

    The generic character-overlap chunker can split a row and repeat rows at
    chunk boundaries. Student roster pages are structured as cohort/section
    headings followed by pipe-delimited table rows, so this specialized path
    keeps each row intact, carries the active headings into every chunk, and
    uses no overlap. It is intended only for ``student_list_html`` resources.
    """
    content = str(getattr(parsed, "content", "") or "")
    if not content.strip():
        return []

    chunk_char_size = chunk_size * 4
    blocks = [
        re.sub(r"[ \t]+", " ", block).strip()
        for block in re.split(r"\n\s*\n", content)
        if block and block.strip()
    ]
    if not blocks:
        return []

    cohort_pattern = re.compile(
        r"\b(?:(?:19|20)\d{2}\s+Batch\s*-\s*)?"
        r"(II|III|IV)\s+Year(?:\s+Students\s+List)?\b",
        re.IGNORECASE,
    )
    section_pattern = re.compile(
        r"(?:/\s*)?sec\.?\b|\bsection\b|\b(?:II|III|IV|V|VI|VII)\s+Sem\.?\b",
        re.IGNORECASE,
    )
    row_pattern = re.compile(r"^\s*\d+\s*\|", re.IGNORECASE)
    column_pattern = re.compile(
        r"\b(?:s\.?\s*no|regd?\.?\s*(?:num|no)?|roll(?:\s+number|\s+no\.?)?|"
        r"admn\.?\s*no|student\s+name)\b",
        re.IGNORECASE,
    )

    document_id = getattr(parsed, "document_id", "")
    title = getattr(parsed, "title", "")
    source_url = getattr(parsed, "source_url", "")
    department = getattr(parsed, "department", "")
    source_type = getattr(parsed, "source_type", "html")
    original_metadata = dict(getattr(parsed, "metadata", {}) or {})
    page_number = int(original_metadata.get("page_number", 0) or 0)

    chunks: List[Chunk] = []
    chunk_index = 0
    active_cohort = ""
    active_section = ""
    active_column_header = ""
    row_buffer: List[str] = []
    loose_buffer: List[str] = []

    def context_blocks() -> List[str]:
        return [
            value
            for value in (active_cohort, active_section, active_column_header)
            if value
        ]

    def emit(text_blocks: List[str]) -> None:
        nonlocal chunk_index
        text = "\n\n".join(block for block in text_blocks if block).strip()
        if not text:
            return
        metadata = dict(original_metadata)
        if active_cohort:
            metadata["student_list_cohort"] = active_cohort
        if active_section:
            metadata["student_list_section"] = active_section
        chunks.append(
            Chunk(
                chunk_id=_generate_chunk_id(document_id, source_url, page_number, chunk_index),
                text=text,
                source_url=source_url,
                title=title,
                source_type=source_type,
                department=department,
                page_number=page_number,
                document_id=document_id,
                metadata=metadata,
            )
        )
        chunk_index += 1

    def flush_rows() -> None:
        nonlocal row_buffer
        if row_buffer:
            emit(context_blocks() + row_buffer)
            row_buffer = []

    for block in blocks:
        cohort_match = cohort_pattern.search(block)
        if cohort_match and not row_pattern.match(block):
            flush_rows()
            active_cohort = block
            active_section = ""
            active_column_header = ""
            loose_buffer = []
            continue

        if section_pattern.search(block) and not row_pattern.match(block):
            flush_rows()
            active_section = block
            active_column_header = ""
            loose_buffer = []
            continue

        if column_pattern.search(block) and not row_pattern.match(block):
            active_column_header = block
            continue

        if row_pattern.match(block):
            candidate = context_blocks() + row_buffer + [block]
            candidate_text = "\n\n".join(candidate)
            if row_buffer and len(candidate_text) > chunk_char_size:
                flush_rows()
            row_buffer.append(block)
            continue

        # Preserve non-table text. Flush rows before changing the loose
        # context so no explanatory text is silently discarded.
        flush_rows()
        if block not in loose_buffer:
            loose_buffer.append(block)
            emit(context_blocks() + loose_buffer)
            loose_buffer = []

    flush_rows()
    if not chunks:
        return chunk_document(parsed.__dict__, chunk_size=chunk_size, overlap=0)
    return chunks


def chunk_pdf_document(
    pdf: Any,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk a PDFDocument instance or list of them.

    Accepts pdf_parser.py PDFDocument with fields:
      - document_id, title, content, source_url, source_type="pdf",
        department, pdf_metadata, tables_detected, images_count, metadata
    """
    if isinstance(pdf, list):
        chunks = []
        for doc in pdf:
            chunks.extend(chunk_pdf_document(doc, chunk_size=chunk_size, overlap=overlap))
        return chunks

    # Convert PDFDocument to dict shape
    document_dict = {
        "document_id": pdf.document_id,
        "title": pdf.title,
        "source_url": pdf.source_url,
        "content": pdf.content,
        "department": pdf.department,
        "page_number": pdf.metadata.get("page_number", 0) if pdf.metadata else 0,
        "source_type": pdf.source_type,
        "metadata": dict(pdf.metadata or {}),
    }
    return chunk_document(document_dict, chunk_size=chunk_size, overlap=overlap)


def chunk_image_document(
    image: Any,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk an ImageDocument instance (from image_parser.py).

    Accepts image_parser.py ImageDocument with fields:
      - document_id, title, content, source_url, source_type="image",
        department, metadata

    Timetable/document image extractions are usually short (< 500 tokens
    total), so this typically produces a single chunk — but chunk_document()
    handles the general case correctly either way.
    """
    document_dict = {
        "document_id": image.document_id,
        "title": image.title,
        "source_url": image.source_url,
        "content": image.content,
        "department": image.department,
        "page_number": 0,
        "source_type": image.source_type,
        "metadata": dict(image.metadata or {}),
    }
    return chunk_document(document_dict, chunk_size=chunk_size, overlap=overlap)


def chunk_cleaned_content(
    cleaned: Any,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Chunk a CleanedContent instance.

    Accepts cleaner.py CleanedContent with fields:
      - document_id, source_url, cleaned_text: str, original_length,
        cleaned_length, word_count, etc.
    """
    content = getattr(cleaned, "cleaned_text", "")
    if not content.strip():
        # Fallback: treat cleaned as raw content
        content = getattr(cleaned, "cleaned_content", "") or getattr(cleaned, "content", "")
    if not content.strip():
        return []

    document_id = getattr(cleaned, "document_id", "")
    source_url = getattr(cleaned, "source_url", "")
    title = getattr(cleaned, "title", "")
    department = getattr(cleaned, "department", "")
    page_number = int(getattr(cleaned, "page_number", 0))

    document_dict = {
        "document_id": document_id,
        "title": title,
        "source_url": source_url,
        "content": content,
        "department": department,
        "page_number": page_number,
        "source_type": "cleaned",
        "metadata": dict(getattr(cleaned, "metadata", {}) or {}),
    }
    return chunk_document(document_dict, chunk_size=chunk_size, overlap=overlap)


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test with parsed document shape (html_parser)
        from backend.ingestion.html_parser import ParsedDocument, HTMLParser

        parser = HTMLParser()
        doc = parser.parse(
            "<html><body><h1>Test Page</h1><p>This is test content for chunking.</p></body></html>",
            "https://example.com/test",
        )

        chunks = chunk_parsed_document(doc, chunk_size=500, overlap=50)
        print(f"ParsedDocument chunks: {len(chunks)}")
        if chunks:
            c = chunks[0]
            print(f"  First chunk ID: {c.chunk_id}")
            print(f"  First chunk text: {c.text[:80]}...")
            print(f"  Metadata - source_type: {c.source_type}, department: {c.department}, page: {c.page_number}")

        # Test with PDF document shape
        from backend.ingestion.pdf_parser import PDFDocument, PDFParser

        pdf_parser = PDFParser()
        pdf_doc = PDFDocument(
            document_id="test_pdf",
            title="Test PDF",
            content="Lakireddy Bali Reddy College of Engineering\nEngineering and Science\nComputer Science\nProfessor & HOD, Dept of CSE\nVeeraiah\nContent goes here with more text to fill chunks properly.",
            source_url="https://example.com/test.pdf",
            source_type="pdf",
            department="CSE",
        )
        # Set metadata with page_number
        pdf_doc.metadata = {"page_number": 1}

        pdf_chunks = chunk_pdf_document(pdf_doc, chunk_size=500, overlap=50)
        print(f"\nPDFDocument chunks: {len(pdf_chunks)}")
        if pdf_chunks:
            c = pdf_chunks[0]
            print(f"  First chunk ID: {c.chunk_id}")
            print(f"  First chunk text: {c.text[:80]}...")
            print(f"  Metadata - source_type: {c.source_type}, department: {c.department}, page: {c.page_number}")

        # Test with CleanedContent
        from backend.ingestion.cleaner import CleanedContent, clean_content

        cleaned = await clean_content(
            "Hello World! This is some test content that should be chunked properly with metadata preserved.",
            "https://example.com",
            "test_doc_001",
        )
        cleaned_chunks = chunk_cleaned_content(cleaned, chunk_size=500, overlap=50)
        print(f"\nCleanedContent chunks: {len(cleaned_chunks)}")
        if cleaned_chunks:
            c = cleaned_chunks[0]
            print(f"  First chunk ID: {c.chunk_id}")
            print(f"  First chunk text: {c.text[:80]}...")
            print(f"  Metadata - source_type: {c.source_type}, department: {c.department}, page: {c.page_number}")

    asyncio.run(test())
