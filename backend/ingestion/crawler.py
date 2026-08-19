"""
Web Crawler for LBRCE AI Assistant.

Crawls the LBRCE website starting from seed URLs, discovering HTML pages and PDF files.
Features rate limiting, visited URL tracking, domain restriction, and logging.
"""

import asyncio
import logging
import time
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from backend.ingestion.html_parser import HTMLParser, ParsedDocument
from backend.ingestion.pdf_parser import PDFParser, PDFDocument
from backend.ingestion.image_parser import ImageParser, ImageDocument

logger = logging.getLogger(__name__)

# File extensions that are never worth queueing as a "page to crawl": they're
# not HTML, and they're also not one of the two content types the ingestion
# pipeline actually processes (.pdf and timetable images). Filtering these
# out BEFORE queueing -- rather than after fetching, via the content-type
# check in HTMLParser.fetch() -- avoids downloading megabyte-sized .pptx/.ppt
# course-material decks (and .xlsx student lists, .docx, archives, media
# files, etc.) that were never going to be parsed anyway. Each one currently
# costs a full round-trip (1-3s) for nothing -- see the crawl log full of
# "Non-HTML content type" warnings for .pptx/.ppt/.xlsx links.
#
# .pdf and image extensions are intentionally NOT in this list -- those are
# handled by the separate pdf_links / image_links discovery in html_parser
# and are still meant to be fetched, just not queued as HTML pages.
NON_CRAWLABLE_EXTENSIONS = (
    # Presentations
    ".ppt", ".pptx",
    # Word docs
    ".doc", ".docx", ".rtf",
    # Spreadsheets (the ingestion pipeline doesn't process these -- only .pdf)
    ".xls", ".xlsx", ".xlsm", ".csv",
    # Archives
    ".zip", ".rar", ".7z", ".tar", ".gz",
    # Media
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".wav",
    # Misc binary/non-HTML
    ".exe", ".apk", ".iso",
)

# Handled separately (still fetched, just via pdf_links/image_links
# discovery rather than being queued as a crawl-target HTML page).
PDF_EXTENSION = ".pdf"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp")


def is_crawlable_html_link(url: str) -> bool:
    """
    True if this URL is worth queueing as an HTML page to crawl.

    False for PDFs, images, and any known non-HTML document/media/archive
    extension -- all of which would just be downloaded and immediately
    discarded by HTMLParser.fetch()'s content-type check. Checked against
    the URL path only (query string stripped) so a link like
    `.../file.pptx?download=1` is still correctly excluded.
    """
    path = urlparse(url).path.lower()
    if path.endswith(PDF_EXTENSION) or path.endswith(IMAGE_EXTENSIONS):
        return False
    if path.endswith(NON_CRAWLABLE_EXTENSIONS):
        return False
    return True


def is_valid_lbrce_url(url: str, allowed_domains: Set[str]) -> bool:
    """Check if the URL belongs to allowed LBRCE domains."""
    try:
        parsed = urlparse(url)
        # Check domain suffix/matches
        domain = parsed.netloc.lower()
        # Remove port if exists
        if ":" in domain:
            domain = domain.split(":")[0]
        
        # Check if domain matches any of the allowed domains
        for allowed in allowed_domains:
            if domain == allowed or domain.endswith("." + allowed):
                return True
        return False
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """Normalize URL by stripping fragments and trailing slashes."""
    try:
        parsed = urlparse(url)
        # Strip fragment
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path
        if path.endswith("/"):
            path = path.rstrip("/")
        query = parsed.query
        
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized
    except Exception:
        return url


class LBRCECrawler:
    """
    BFS-style web crawler for the LBRCE website.
    Discovers HTML content and PDFs, respecting domain constraints and polite delays.
    """

    def __init__(
        self,
        seed_urls: List[str],
        allowed_domains: Optional[List[str]] = None,
        max_pages: int = 50,
        request_delay: float = 1.0,
        timeout: float = 10.0,
    ):
        """Initialize crawler configuration.

        Args:
            seed_urls: Start URLs for crawl.
            allowed_domains: Domains allowed to crawl (defaults to lbrce.ac.in).
            max_pages: Limit on total pages to fetch to prevent infinite crawling.
            request_delay: Polite delay between requests in seconds.
            timeout: Timeout for HTTP requests.
        """
        self.seed_urls = [normalize_url(url) for url in seed_urls]
        self.allowed_domains = set(allowed_domains) if allowed_domains else {"lbrce.ac.in", "lbrce.com"}
        self.max_pages = max_pages
        self.request_delay = request_delay
        self.timeout = timeout
        
        self.visited_urls: Set[str] = set()
        self.discovered_pdfs: Set[str] = set()
        self.discovered_images: Set[str] = set()
        self.parsed_pages: List[ParsedDocument] = []
        self.failed_urls: Dict[str, str] = {}
        self.skipped_non_html_count = 0  # links filtered out before fetch (pptx/xlsx/docx/etc.)
        
        self.html_parser = HTMLParser(timeout=timeout)

    async def crawl(self) -> Tuple[List[ParsedDocument], Set[str], Set[str]]:
        """Run BFS crawl starting from seed URLs.

        Returns:
            Tuple of (parsed HTML documents, discovered PDF URLs, discovered image URLs).
        """
        queue: List[str] = list(self.seed_urls)
        pages_crawled = 0

        async with self.html_parser as parser:
            while queue and pages_crawled < self.max_pages:
                url = queue.pop(0)
                
                if url in self.visited_urls:
                    continue
                
                self.visited_urls.add(url)
                logger.info(f"Crawling URL ({pages_crawled + 1}/{self.max_pages}): {url}")
                
                # Fetch page content
                html_content = await parser.fetch(url)
                if not html_content:
                    self.failed_urls[url] = "Failed to fetch content or non-HTML page"
                    continue
                
                # Parse HTML content
                parsed_doc = parser.parse(html_content, url)
                if not parsed_doc:
                    self.failed_urls[url] = "Failed to parse HTML"
                    continue
                
                self.parsed_pages.append(parsed_doc)
                pages_crawled += 1
                
                # Record discovered PDFs
                for pdf in parsed_doc.pdf_links:
                    norm_pdf = normalize_url(pdf)
                    if is_valid_lbrce_url(norm_pdf, self.allowed_domains):
                        self.discovered_pdfs.add(norm_pdf)

                # Record discovered images (timetables etc.)
                for img in parsed_doc.image_links:
                    norm_img = normalize_url(img)
                    if is_valid_lbrce_url(norm_img, self.allowed_domains):
                        self.discovered_images.add(norm_img)
                
                # Queue new links -- skip PDFs/images (handled separately
                # above via pdf_links/image_links) and skip any known
                # non-HTML document/media/archive extension (.pptx, .xlsx,
                # .docx, .zip, etc.) so those are never fetched at all.
                for link in parsed_doc.links:
                    norm_link = normalize_url(link)
                    if norm_link in self.visited_urls or norm_link in queue:
                        continue
                    if not is_valid_lbrce_url(norm_link, self.allowed_domains):
                        continue
                    if not is_crawlable_html_link(norm_link):
                        self.skipped_non_html_count += 1
                        continue
                    queue.append(norm_link)
                
                # Polite delay
                if queue and self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                    
        logger.info(
            f"Crawl completed. Crawled {pages_crawled} HTML pages, "
            f"discovered {len(self.discovered_pdfs)} PDFs, "
            f"{len(self.discovered_images)} images, "
            f"skipped {self.skipped_non_html_count} non-HTML links (pptx/xlsx/docx/zip/etc.) before fetching."
        )
        return self.parsed_pages, self.discovered_pdfs, self.discovered_images


if __name__ == "__main__":
    # Test crawler local setup
    async def test():
        logging.basicConfig(level=logging.INFO)
        crawler = LBRCECrawler(
            seed_urls=["https://www.lbrce.ac.in/"],
            max_pages=2,
            request_delay=0.5
        )
        pages, pdfs, images = await crawler.crawl()
        print(f"Parsed {len(pages)} HTML pages:")
        for p in pages:
            print(f"  - Title: {p.title} | URL: {p.source_url}")
        print(f"Discovered {len(pdfs)} PDFs:")
        for pdf in list(pdfs)[:5]:
            print(f"  - PDF: {pdf}")
        print(f"Discovered {len(images)} images:")
        for img in list(images)[:5]:
            print(f"  - Image: {img}")

    asyncio.run(test())