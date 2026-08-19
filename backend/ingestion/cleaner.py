"""
Content cleaning and transformation for LBRCE AI Assistant.

Handles text normalization, HTML entities conversion, unwanted elements removal, and content transformation into cleaned chunks.
"""

import logging
import re
from dataclasses import dataclass, field
from html import unescape
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CleanedContent:
    """Represents cleaned content with chunks and metadata."""
    document_id: str
    source_url: str
    cleaned_text: str
    original_length: int = 0
    cleaned_length: int = 0
    word_count: int = 0
    punctuation_count: int = 0
    whitespace_ratio: float = 0.0
    content_quality_score: float = 0.0


class ContentCleaner:
    """
    Cleans and transforms raw content from HTML/PDF parsing for ingestion into RAG systems.

    Performs:
    - Text cleaning and normalization
    - HTML entity conversion
    - Unwanted content removal
    - Chunking for efficient RAG retrieval
    """

    # Chunking configuration
    CHUNK_SIZE = 1000
    OVERLAP_SIZE = 100

    def __init__(self):
        """Initialize content cleaner."""
        pass

    def clean(self, raw_content: str, source_url: str, document_id: str) -> CleanedContent:
        """
        Clean raw content and chunk it for RAG storage.

        Args:
            raw_content: Raw text content from HTML/PDF extraction.
            source_url: Source URL for metadata.
            document_id: Document ID for reference.

        Returns:
            CleanedContent with chunks and metadata.
        """
        if not raw_content.strip():
            return CleanedContent(
                document_id=document_id,
                source_url=source_url,
                cleaned_text="",
                original_length=0,
                cleaned_length=0,
                word_count=0,
                punctuation_count=0,
                whitespace_ratio=0.0,
                content_quality_score=0.0,
            )

        # Convert HTML entities and clean raw content
        cleaned = self._convert_html_entities(raw_content)
        cleaned = self._remove_noise_elements(cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._remove_redundant_punctuation(cleaned)

        # Analyze cleaned content
        original_length = len(raw_content)
        cleaned_length = len(cleaned)
        word_count = len(cleaned.split())
        punctuation_count = self._count_punctuation(cleaned)
        whitespace_ratio = self._calculate_whitespace_ratio(cleaned)
        content_quality_score = self._calculate_quality_score(raw_content, cleaned)

        return CleanedContent(
            document_id=document_id,
            source_url=source_url,
            cleaned_text=cleaned,
            original_length=original_length,
            cleaned_length=cleaned_length,
            word_count=word_count,
            punctuation_count=punctuation_count,
            whitespace_ratio=whitespace_ratio,
            content_quality_score=content_quality_score,
        )

    def _convert_html_entities(self, text: str) -> str:
        """Convert HTML entities to plain text."""
        try:
            return unescape(text)
        except Exception:
            # Fallback implementation
            import re
            text = re.sub(r"&lt;", "<", text)
            text = re.sub(r"&gt;", ">", text)
            text = re.sub(r"&amp;", "&", text)
            text = re.sub(r"&quot;", '"', text)
            text = re.sub(r"&apos;", "'", text)
            return text

    def _remove_noise_elements(self, text: str) -> str:
        """Remove unwanted elements from content."""
        # Remove common noise patterns
        import re
        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)
        # Remove phone numbers (simple patterns)
        text = re.sub(r"\+?\d{10,}\s*\d{10}\s*\d{4}", "", text)
        # Remove URLs (simple patterns)
        text = re.sub(r"https?://\S+", "", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        import re
        # Convert tabs to spaces
        text = re.sub(r"\t", " ", text)
        # Normalize newlines (keep paragraph structure)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        # Remove excessive spaces
        text = re.sub(r"  +", " ", text)
        return text.strip()

    def _remove_redundant_punctuation(self, text: str) -> str:
        """Remove redundant punctuation while preserving meaning."""
        import re
        # Remove redundant exclamation marks
        text = re.sub(r"!{2,}", "!", text)
        # Remove redundant question marks
        text = re.sub(r"\?{2,}", "?", text)
        # Remove redundant periods at end of lines
        text = re.sub(r"\.{2,}\s*$", ".", text)
        return text

    def _count_punctuation(self, text: str) -> int:
        """Count punctuation marks in text."""
        import re
        punctuation_marks = re.findall(r"[.!?;,:]*", text)
        total = 0
        for mark in punctuation_marks:
            total += len(mark)
        return total

    def _calculate_whitespace_ratio(self, text: str) -> float:
        """Calculate ratio of whitespace characters in text."""
        if not text:
            return 0.0
        whitespace_count = len(re.findall(r"\s", text))
        total_chars = len(text)
        return whitespace_count / total_chars if total_chars > 0 else 0.0

    def _calculate_quality_score(self, original: str, cleaned: str) -> float:
        """Calculate content quality score based on cleaning ratio."""
        if not original:
            return 0.0

        original_density = len(original.split()) / len(original)
        cleaned_density = len(cleaned.split()) / len(cleaned)

        # Penalize too much content loss
        loss_ratio = (len(original) - len(cleaned)) / len(original)
        score = (cleaned_density / original_density) * (1.0 - loss_ratio)

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

async def clean_content(raw_content: str, source_url: str, document_id: str) -> CleanedContent:
    """
    Convenience function to clean content in one call.

    Args:
        raw_content: Raw content to clean.
        source_url: Source URL for metadata.
        document_id: Document ID for reference.

    Returns:
        CleanedContent with chunks and metadata.
    """
    cleaner = ContentCleaner()
    return cleaner.clean(raw_content, source_url, document_id)


if __name__ == "__main__":
    import asyncio

    async def test():
# Test with HTML content
        raw_html = "<html><body><p>Hello World!</p><p>This is <b>bold</b> text.</p><p>Phone: 1234567890</p></body></html>"
        result = await clean_content(raw_html, "https://example.com", "test_html_doc")
        print("Cleaned Text:", result.cleaned_text)
        print("Original length:", result.original_length)
        print("Cleaned length:", result.cleaned_length)
        print("Word count:", result.word_count)
        print("Quality score:", "{:.2f}".format(result.content_quality_score))

        # Test with PDF content
        raw_pdf = "Lakireddy Bali Reddy College of Engineering (A)\nEngineering and Science\nComputer Science\nProfessor & HOD, Dept of CSE\nVeeraiah\nDr. P. Associate Professor, Dept..."
        pdf_result = await clean_content(raw_pdf, "https://example.com/pdf.pdf", "test_pdf_doc")
        print("PDF Cleaned Text:", pdf_result.cleaned_text[:100] + "...")
        print("PDF Quality score:", "{:.2f}".format(pdf_result.content_quality_score))

    asyncio.run(test())