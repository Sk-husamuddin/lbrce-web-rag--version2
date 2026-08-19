"""
PDF Parser using PyMuPDF (fitz) for LBRCE document extraction.

This module handles PDF content extraction from LBRCE website.
It extracts text, metadata, and identifies structural elements
from PDF documents found on the LBRCE website.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urlunparse

import fitz  # PyMuPDF
import re  # for text cleaning

logger = logging.getLogger(__name__)


@dataclass
class PDFDocument:
    """Represents a parsed PDF document with extracted content and metadata."""
    document_id: str
    title: str
    content: str
    source_url: str
    source_type: str = "pdf"
    department: Optional[str] = None
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)
    tables_detected: bool = False
    images_count: int = 0
    metadata: dict = field(default_factory=dict)


class PDFParser:
    """
    Parses PDF documents using PyMuPDF (fitz).

    Extracts:
    - Full text content
    - Document title/metadata
    - Structural elements (headings via font analysis)
    - Tables and image count
    """

    # Maximum PDF size to process (in bytes) - prevents OOM on large files
    MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def parse(self, pdf_data: bytes, url: str) -> Optional[List[PDFDocument]]:
        """
        Parse PDF bytes and extract structured content.

        Args:
            pdf_data: Raw PDF file bytes.
            url: Source URL for metadata.

        Returns:
            PDFDocument with extracted content, or None if parsing failed.
        """
        try:
            # Check PDF size
            if len(pdf_data) > self.MAX_PDF_SIZE:
                logger.warning(f"PDF too large ({len(pdf_data)} bytes): {url}")
                return None

            # Open PDF document using bytes stream
            doc = fitz.open(stream=pdf_data, filetype="pdf")

            if doc.page_count == 0:
                logger.warning(f"Empty PDF: {url}")
                return None

            # Store page count before closing
            page_count = doc.page_count

            # Extract metadata before closing
            metadata = dict(doc.metadata) if doc.metadata else {}
            title = metadata.get("title", "").strip() or self._extract_title_from_pages(doc)

            # Generate document ID
            document_id = self._generate_document_id(url)

            # Infer department from URL
            department = self._infer_department(url)

            # Extract text from all pages
            tables_detected = False
            images_count = 0

            # First pass: compute aggregated stats
            for page_num in range(page_count):
                page = doc[page_num]
                try:
                    if page.get_tables():
                        tables_detected = True
                except AttributeError:
                    pass
                images_count += len(page.get_images())

            documents = []
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text("text")
                cleaned_content = self._clean_pdf_text(text)

                page_meta = {
                    "page_count": page_count,
                    "file_size": len(pdf_data),
                    "title_from_meta": bool(metadata.get("title")),
                    "page_number": page_num + 1,
                }

                documents.append(
                    PDFDocument(
                        document_id=document_id,
                        title=title,
                        content=cleaned_content,
                        source_url=url,
                        source_type="pdf",
                        department=department,
                        pdf_metadata=metadata,
                        tables_detected=tables_detected,
                        images_count=images_count,
                        metadata=page_meta,
                    )
                )

            # Close document
            doc.close()

            return documents

        except Exception as e:
            logger.error(f"Failed to parse PDF from {url}: {e}")
            return None

    def _extract_title_from_pages(self, doc: fitz.Document) -> str:
        """Extract title from first page if no metadata title."""
        try:
            first_page = doc[0]
            text = first_page.get_text("text")[:500]
            # Look for a large/bold text that might be a title
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                return lines[0][:200]
        except Exception:
            pass
        return "Untitled PDF"

    def _clean_pdf_text(self, text: str) -> str:
        """Clean and normalize PDF-extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        # Remove duplicate spaces
        text = re.sub(r"[ \t]+\n", "\n", text)
        # Remove form feed characters
        text = text.replace("\f", "")
        return text.strip()

    def _generate_document_id(self, url: str) -> str:
        """Generate a stable document ID from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return "pdf"
        doc_id = re.sub(r"[^a-zA-Z0-9/_-]", "_", path)
        doc_id = doc_id.replace("/", "_")
        return doc_id[:100]

    def _infer_department(self, url: str) -> Optional[str]:
        """Infer department from URL path."""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        dept_keywords = {
            "cse": "Computer Science Engineering",
            "cs": "Computer Science Engineering",
            "it": "Information Technology",
            "ece": "Electronics and Communication Engineering",
            "eee": "Electrical and Electronics Engineering",
            "mech": "Mechanical Engineering",
            "civil": "Civil Engineering",
            "ai": "Artificial Intelligence",
            "ml": "Machine Learning",
            "ds": "Data Science",
            "mba": "Master of Business Administration",
            "mca": "Master of Computer Applications",
            "h&s": "Humanities and Sciences",
            "physics": "Physics",
            "chemistry": "Chemistry",
            "mathematics": "Mathematics",
            "english": "English",
            "library": "Library",
            "exam": "Examination",
            "result": "Results",
            "notification": "Notifications",
        }

        for part in path_parts:
            part_lower = part.lower()
            if part_lower in dept_keywords:
                return dept_keywords[part_lower]

        return None


def _candidate_pdf_urls(url: str) -> List[str]:
    """Return safe URL candidates for known legacy LBRCE PDF hosts.

    Some older CSE links use the retired ``cse.lbrce.ac.in`` hostname and
    plain HTTP. We only generate alternatives inside the approved LBRCE
    domain; arbitrary hosts are never rewritten.
    """
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    candidates: List[str] = []

    def add_candidate(scheme: str, host: str) -> None:
        candidate = urlunparse(
            (scheme, host, parsed.path, parsed.params, parsed.query, "")
        )
        if candidate not in candidates:
            candidates.append(candidate)

    # Try the original URL first so valid links preserve their normal path.
    add_candidate(parsed.scheme or "https", parsed.netloc)

    # Upgrade legacy HTTP links to HTTPS on the same approved host.
    if hostname in {"lbrce.ac.in", "www.lbrce.ac.in", "cse.lbrce.ac.in"}:
        add_candidate("https", hostname)

    # The retired CSE subdomain was historically mirrored under the main
    # website. Try the main host as a final, bounded fallback. A 404 is
    # treated as an unavailable legacy link and is not retried indefinitely.
    if hostname == "cse.lbrce.ac.in":
        add_candidate("https", "www.lbrce.ac.in")

    return candidates


def parse_pdf_from_url(url: str, timeout: float = 30.0) -> Optional[List[PDFDocument]]:
    """Fetch and parse a PDF using bounded, approved-domain URL candidates.

    The function handles stale HTTP links and the retired CSE subdomain while
    preserving the URL that actually produced the PDF in the returned source
    metadata. Downloads are streamed and capped before parsing to avoid
    loading arbitrarily large responses into memory.
    """
    import httpx

    parser = PDFParser(timeout=timeout)
    last_error: Optional[str] = None

    for candidate_url in _candidate_pdf_urls(url):
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "LBRCE-AI-Assistant/1.0"},
            ) as client:
                with client.stream("GET", candidate_url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    path_without_query = urlparse(candidate_url).path.lower()
                    if "application/pdf" not in content_type and not path_without_query.endswith(".pdf"):
                        last_error = f"non-PDF content type: {content_type or 'unknown'}"
                        continue

                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > parser.MAX_PDF_SIZE:
                                last_error = f"PDF exceeds {parser.MAX_PDF_SIZE} byte limit"
                                continue
                        except ValueError:
                            pass

                    chunks: List[bytes] = []
                    total_size = 0
                    for chunk in response.iter_bytes():
                        total_size += len(chunk)
                        if total_size > parser.MAX_PDF_SIZE:
                            last_error = f"PDF exceeds {parser.MAX_PDF_SIZE} byte limit"
                            break
                        chunks.append(chunk)
                    else:
                        parsed_documents = parser.parse(b"".join(chunks), candidate_url)
                        if parsed_documents:
                            if candidate_url != url:
                                logger.info(
                                    "Parsed legacy PDF through fallback URL: %s -> %s",
                                    url,
                                    candidate_url,
                                )
                            return parsed_documents
                        last_error = "PDF parsing returned no documents"

        except httpx.HTTPStatusError as exc:
            last_error = f"HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)

        logger.warning("Skipping PDF candidate %s (%s)", candidate_url, last_error)

    logger.error("Failed to fetch/parse PDF from %s (%s)", url, last_error or "no candidates")
    return None


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test with a known LBRCE PDF
        pdf_url = "https://www.lbrce.ac.in/csedept/activity_docs/LBRCE_CSE_ATAL_FDP_2022-23.pdf"
        result = parse_pdf_from_url(pdf_url)
        if result:
            print(f"Parsed {len(result)} pages.")
            if len(result) > 0:
                first = result[0]
                print(f"Title: {first.title}")
                print(f"Document ID: {first.document_id}")
                print(f"Pages: {first.metadata.get('page_count')}")
                print(f"Tables detected: {first.tables_detected}")
                print(f"Images: {first.images_count}")
                print(f"First Page Content length: {len(first.content)}")
                print(f"First Page Content preview: {first.content[:300]}...")
                print(f"First Page Metadata: {first.metadata}")
        else:
            print(f"Failed to parse PDF: {pdf_url}")

    asyncio.run(test())