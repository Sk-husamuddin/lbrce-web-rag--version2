"""
Ingestion package for LBRCE AI Assistant.

Handles web crawling, HTML/PDF parsing, content cleaning, chunking, and indexing.
"""

from backend.ingestion.html_parser import ParsedDocument, HTMLParser, parse_single_page
from backend.ingestion.pdf_parser import PDFDocument, PDFParser, parse_pdf_from_url
from backend.ingestion.crawler import LBRCECrawler
from backend.ingestion.chunker import (
    Chunk,
    chunk_document,
    chunk_cleaned_content,
    chunk_parsed_document,
    chunk_pdf_document,
)

__all__ = [
    "ParsedDocument",
    "PDFDocument",
    "HTMLParser",
    "PDFParser",
    "LBRCECrawler",
    "parse_single_page",
    "parse_pdf_from_url",
    "Chunk",
    "chunk_document",
    "chunk_cleaned_content",
    "chunk_parsed_document",
    "chunk_pdf_document",
]