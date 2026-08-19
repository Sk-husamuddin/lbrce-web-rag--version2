"""
HTML Parser using BeautifulSoup for LBRCE website content extraction.

This module handles fetching and parsing HTML pages from the LBRCE website,
extracting structured content while removing noise like scripts, styles,
navigation, and footer elements.

Includes a lightweight on-disk page cache keyed by URL, using HTTP
conditional requests (ETag / Last-Modified) so unchanged pages are not
re-downloaded or re-parsed on subsequent runs.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Location of the on-disk cache file (kept next to the project root, two
# levels up from this module, mirroring the original snippet's layout).
PAGE_CACHE_FILE = Path(__file__).parent.parent.parent / "page_cache.json"


def load_page_cache() -> dict:
    """Load the page cache from disk. Returns an empty dict if missing/corrupt."""
    if not PAGE_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(PAGE_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load page cache, starting fresh: {e}")
        return {}


def save_page_cache(cache: dict) -> None:
    """Persist the page cache to disk."""
    try:
        PAGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PAGE_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except OSError as e:
        logger.warning(f"Failed to save page cache: {e}")


@dataclass
class ParsedDocument:
    """Represents a parsed HTML document with extracted content and metadata."""
    document_id: str
    title: str
    content: str
    source_url: str
    source_type: str = "html"
    department: Optional[str] = None
    headings: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    pdf_links: List[str] = field(default_factory=list)
    image_links: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_cache_dict(self) -> dict:
        """Serialize the fields needed to reconstruct this document from cache."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "department": self.department,
            "headings": self.headings,
            "links": self.links,
            "pdf_links": self.pdf_links,
            "image_links": self.image_links,
            "metadata": self.metadata,
        }

    @classmethod
    def from_cache_dict(cls, data: dict) -> "ParsedDocument":
        """Reconstruct a ParsedDocument from a cached dict."""
        return cls(
            document_id=data["document_id"],
            title=data["title"],
            content=data["content"],
            source_url=data["source_url"],
            source_type=data.get("source_type", "html"),
            department=data.get("department"),
            headings=data.get("headings", []),
            links=data.get("links", []),
            pdf_links=data.get("pdf_links", []),
            image_links=data.get("image_links", []),
            metadata=data.get("metadata", {}),
        )


class HTMLParser:
    """
    Parses HTML content from LBRCE website using BeautifulSoup.

    Extracts:
    - Title
    - Headings (h1-h6)
    - Main text content (paragraphs, lists)
    - Links (internal and external)
    - PDF links for further processing

    Also supports an on-disk cache keyed by URL. When a previously fetched
    page returns HTTP 304 Not Modified (via ETag / Last-Modified conditional
    headers), the cached ParsedDocument is returned directly without
    re-parsing.
    """

    # HTML tags to remove completely (noise)
    NOISE_TAGS = {
        "script", "style", "noscript", "iframe", "svg", "canvas",
        "header", "footer", "nav", "aside", "form", "button",
        "input", "select", "textarea", "label"
    }

    # CSS selectors for common noise elements
    NOISE_SELECTORS = [
        "[class*='cookie']", "[class*='banner']", "[class*='popup']",
        "[class*='modal']", "[class*='overlay']", "[class*='sidebar']",
        "[class*='widget']", "[class*='social']", "[class*='share']",
        "[id*='cookie']", "[id*='banner']", "[id*='popup']",
        ".navbar", ".footer", ".header", ".sidebar", ".menu",
        ".navigation", ".breadcrumbs", ".pagination"
    ]

    # Content-bearing tags to preserve
    CONTENT_TAGS = {
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "div", "span", "li", "td", "th",
        "blockquote", "pre", "code", "article", "section", "main"
    }

    def __init__(
        self,
        base_url: str = "https://www.lbrce.ac.in",
        timeout: float = 30.0,
        user_agent: str = "LBRCE-AI-Assistant/1.0 (+https://github.com/lbrce-ai)",
        use_cache: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self._client: Optional[httpx.AsyncClient] = None

        # Page cache (URL -> {etag, last_modified, fetched_at, document...})
        self.use_cache = use_cache
        self._page_cache: dict = load_page_cache() if use_cache else {}

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True
            )
        return self._client

    async def fetch(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from a URL and return it as text.

        Kept as the original public contract (returns Optional[str]) since
        callers such as LBRCECrawler.crawl() call `fetch()` then `parse()`
        directly, without going through the cache-aware fetch_and_parse().

        Args:
            url: The URL to fetch.

        Returns:
            HTML content as string, or None if fetch failed / non-HTML /
            not modified (304 is treated as "nothing new to return" here --
            callers that want 304 handling should use fetch_and_parse()).
        """
        response = await self._fetch_response(url)
        if response is None or response.status_code == 304:
            return None
        return response.text

    async def _fetch_response(
        self, url: str, conditional_headers: Optional[dict] = None
    ) -> Optional[httpx.Response]:
        """
        Internal helper: fetch a URL and return the raw httpx.Response.

        Used by fetch_and_parse() to support conditional (ETag /
        If-Modified-Since) requests for the on-disk page cache. Not intended
        to be called directly by external code -- use fetch() (text) or
        fetch_and_parse() (cache-aware ParsedDocument) instead.

        Args:
            url: The URL to fetch.
            conditional_headers: Optional extra headers (e.g. If-None-Match,
                If-Modified-Since) used for cache-aware conditional requests.

        Returns:
            The httpx.Response object, or None if the request failed outright.
            Note: a 304 Not Modified response is returned as-is (not None) so
            callers can distinguish "not modified" from "request failed".
        """
        client = self._get_client()
        try:
            response = await client.get(url, headers=conditional_headers or {})
            # 304 has no body and isn't a normal "success" status, but it's
            # not an error either -- let the caller decide what to do.
            if response.status_code == 304:
                return response

            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                logger.warning(f"Non-HTML content type: {content_type} for {url}")
                return None
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            return None

    def parse(self, html: str, url: str) -> Optional[ParsedDocument]:
        """
        Parse HTML content and extract structured information.

        Args:
            html: Raw HTML content.
            url: Source URL for metadata.

        Returns:
            ParsedDocument with extracted content, or None if parsing failed.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            logger.error(f"Failed to parse HTML from {url}: {e}")
            return None

        # Remove noise elements
        self._remove_noise(soup)

        # Extract metadata
        title = self._extract_title(soup)
        headings = self._extract_headings(soup)
        content = self._extract_content(soup)
        links = self._extract_links(soup, url)
        pdf_links = self._extract_pdf_links(links)
        # Also scan <img src> tags directly -- timetable images are often
        # embedded via <img>, not linked via <a href>, so pdf_links-style
        # <a> scanning alone would miss most of them.
        image_links = self._extract_image_links(soup, url) + self._extract_image_link_hrefs(links)

        # Generate document ID from URL
        document_id = self._generate_document_id(url)

        # Determine department from URL path
        department = self._infer_department(url)

        return ParsedDocument(
            document_id=document_id,
            title=title,
            content=content,
            source_url=url,
            source_type="html",
            department=department,
            headings=headings,
            links=links,
            pdf_links=pdf_links,
            image_links=image_links,
            metadata={
                "url_path": urlparse(url).path,
                "heading_count": len(headings),
                "link_count": len(links),
                "pdf_link_count": len(pdf_links),
                "image_link_count": len(image_links),
                "content_length": len(content)
            }
        )

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """Remove noise elements from the parsed HTML."""
        # Remove noise tags
        for tag_name in self.NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove elements matching noise selectors
        for selector in self.NOISE_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

        # Remove comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try <title> tag first
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try meta og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        return "Untitled"

    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract all headings (h1-h6) in document order."""
        headings = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            text = tag.get_text(strip=True)
            if text:
                headings.append(text)
        return headings

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main text content from the page.

        Prioritizes content in <main>, <article>, or <section> tags.
        Falls back to body if no semantic containers found.
        """
        # Try to find main content container
        # Match semantic content containers only when the complete id/class
        # token represents a content/main/body container. A loose substring
        # match incorrectly selects layout elements such as ``headerContent``
        # and can hide tables that are present elsewhere in the document.
        content_container_re = re.compile(
            r"^(?:content|main|body)(?:[-_][a-z0-9]+)*$",
            re.I,
        )
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("section", role="main") or
            soup.find("div", id=content_container_re) or
            soup.find("div", class_=content_container_re)
        )

        if not main_content:
            main_content = soup.find("body")

        # Some legacy LBRCE templates close </body> before their main
        # sections. In that case BeautifulSoup places the useful sections
        # outside body, while body contains only the header/navigation shell.
        # Use the parsed document as the fallback root when the selected
        # container has no meaningful text, after noise has been removed.
        if not main_content or len(main_content.get_text(" ", strip=True)) < 100:
            if soup.get_text(" ", strip=True):
                main_content = soup

        if not main_content:
            return ""

        # Extract text from content-bearing tags. Table rows are handled as
        # single structured fragments so that all numeric cells are preserved
        # together and individual td/th elements are not duplicated below.
        texts = []
        content_tags = set(self.CONTENT_TAGS) | {"tr"}
        for tag in main_content.find_all(content_tags):
            # Skip if tag is inside a noise element we missed.
            if tag.find_parent(self.NOISE_TAGS):
                continue

            if tag.name == "tr":
                cells = tag.find_all(["th", "td"], recursive=False)
                row_values = [cell.get_text(" ", strip=True) for cell in cells]
                text = " | ".join(value for value in row_values if value)
                if text and len(text) > 2:
                    texts.append(text)
                continue

            # Table cells have already been captured through their row.
            if tag.find_parent("table"):
                continue

            # Capture only the deepest matching content elements. A parent
            # container's get_text() recursively includes matching children,
            # so visiting both would duplicate the same paragraph at each
            # nesting level.
            if tag.find(self.CONTENT_TAGS):
                continue
            text = tag.get_text(strip=True)
            if text and len(text) > 2:  # Filter very short fragments
                texts.append(text)

        # Join with newlines, preserving paragraph structure
        content = "\n\n".join(texts)

        # Clean up whitespace
        content = self._clean_text(content)

        return content

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page, normalized to absolute URLs."""
        links = []
        seen: Set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Normalize to absolute URL
            absolute_url = urljoin(base_url, href)

            # Only include LBRCE domain links
            if self._is_lbrce_domain(absolute_url):
                if absolute_url not in seen:
                    seen.add(absolute_url)
                    links.append(absolute_url)

        return links

    def _extract_pdf_links(self, links: List[str]) -> List[str]:
        """Filter PDF links from a list of URLs."""
        pdf_links = []
        for link in links:
            if link.lower().endswith(".pdf") or ".pdf?" in link.lower():
                pdf_links.append(link)
        return pdf_links

    def _extract_image_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract <img src> URLs, filtered to timetable-relevant image types.

        Timetable images are usually embedded via <img>, not linked via
        <a href>, so this must scan <img> tags directly rather than relying
        on the <a>-based links list alone.
        """
        image_exts = (".jpg", ".jpeg", ".png")
        images: List[str] = []
        seen: Set[str] = set()
        for img_tag in soup.find_all("img", src=True):
            src = img_tag["src"].strip()
            if not src or src.startswith("data:"):
                continue
            absolute_url = urljoin(base_url, src)
            if absolute_url.lower().endswith(image_exts) and self._is_lbrce_domain(absolute_url):
                if absolute_url not in seen:
                    seen.add(absolute_url)
                    images.append(absolute_url)
        return images

    def _extract_image_link_hrefs(self, links: List[str]) -> List[str]:
        """Filter <a href> links that point directly to image files
        (covers cases where a timetable image is linked, not just embedded).
        """
        image_exts = (".jpg", ".jpeg", ".png")
        return [link for link in links if link.lower().endswith(image_exts)]

    def _is_lbrce_domain(self, url: str) -> bool:
        """Check if URL belongs to the exact LBRCE host or a subdomain."""
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        return hostname == "lbrce.ac.in" or hostname.endswith(".lbrce.ac.in")

    def _generate_document_id(self, url: str) -> str:
        """Generate a stable document ID from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return "home"
        # Replace special characters with underscores
        doc_id = re.sub(r"[^a-zA-Z0-9/_-]", "_", path)
        doc_id = doc_id.replace("/", "_")
        return doc_id[:100]  # Limit length

    def _infer_department(self, url: str) -> Optional[str]:
        """Infer department from URL path."""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]

        # Common department keywords in LBRCE URLs
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
            "hs": "Humanities and Sciences",
            "physics": "Physics",
            "chemistry": "Chemistry",
            "mathematics": "Mathematics",
            "english": "English",
            "library": "Library",
            "placement": "Training and Placement",
            "training": "Training and Placement",
            "research": "Research and Development",
            "rd": "Research and Development",
            "admin": "Administration",
            "admissions": "Admissions",
            "exam": "Examination",
            "examination": "Examination",
        }

        for part in path_parts:
            part_lower = part.lower()
            if part_lower in dept_keywords:
                return dept_keywords[part_lower]

        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Remove trailing spaces on lines
        text = re.sub(r"[ \t]+\n", "\n", text)
        # Remove leading/trailing whitespace
        return text.strip()

    async def fetch_and_parse(self, url: str) -> Optional[ParsedDocument]:
        """
        Fetch and parse a URL in one call, using the on-disk cache when
        possible.

        Flow:
        1. Look up any cached ETag / Last-Modified for this URL.
        2. Issue a GET with those as conditional headers.
        3. If the server replies 304 Not Modified, reconstruct and return
           the previously cached ParsedDocument (no re-parsing needed).
        4. Otherwise parse the fresh HTML, update the cache with the new
           ETag/Last-Modified plus the parsed document, and return it.

        Args:
            url: The URL to fetch and parse.

        Returns:
            ParsedDocument or None if failed.
        """
        cached_entry = self._page_cache.get(url) if self.use_cache else None

        conditional_headers = {}
        if cached_entry:
            if cached_entry.get("etag"):
                conditional_headers["If-None-Match"] = cached_entry["etag"]
            if cached_entry.get("last_modified"):
                conditional_headers["If-Modified-Since"] = cached_entry["last_modified"]

        response = await self._fetch_response(url, conditional_headers=conditional_headers or None)
        if response is None:
            return None

        # Not modified since last fetch -- reuse the cached parsed document.
        if response.status_code == 304 and cached_entry and "document" in cached_entry:
            logger.info(f"Cache hit (304 Not Modified) for {url}")
            return ParsedDocument.from_cache_dict(cached_entry["document"])

        html = response.text
        doc = self.parse(html, url)
        if doc is None:
            return None

        if self.use_cache:
            self._page_cache[url] = {
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
                "fetched_at": datetime.now().isoformat(),
                "document": doc.to_cache_dict(),
            }
            save_page_cache(self._page_cache)

        return doc


async def parse_single_page(url: str, base_url: str = "https://www.lbrce.ac.in") -> Optional[ParsedDocument]:
    """
    Convenience function to parse a single page.

    Args:
        url: URL to parse.
        base_url: Base URL for the site.

    Returns:
        ParsedDocument or None.
    """
    async with HTMLParser(base_url=base_url) as parser:
        return await parser.fetch_and_parse(url)


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test with LBRCE homepage
        result = await parse_single_page("https://www.lbrce.ac.in")
        if result:
            print(f"Title: {result.title}")
            print(f"Document ID: {result.document_id}")
            print(f"Department: {result.department}")
            print(f"Headings ({len(result.headings)}): {result.headings[:5]}")
            print(f"Links ({len(result.links)}): {result.links[:5]}")
            print(f"PDF Links ({len(result.pdf_links)}): {result.pdf_links[:5]}")
            print(f"Content length: {len(result.content)}")
            print(f"Content preview: {result.content[:500]}...")
        else:
            print("Failed to parse")

    asyncio.run(test())